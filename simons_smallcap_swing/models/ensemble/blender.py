"""
models/ensemble/blender.py - Institutional forecast combination layer.

This module combines base-model predictions into a single blended score using
explicit constraints, fallback logic, and audit-ready diagnostics.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_id_with_prefix(prefix: str) -> str:
    from datetime import datetime, timezone

    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if np.isnan(v) or np.isinf(v) else v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat() if pd.notna(value) else None
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def config_hash(cfg: Mapping[str, Any], n: int = 24) -> str:
    blob = json.dumps(json_safe(dict(cfg)), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:n]


def _to_df(obj: pd.DataFrame | str | Path | None) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, (str, Path)):
        p = Path(obj)
        if not p.exists():
            return pd.DataFrame()
        if p.suffix.lower() == ".parquet":
            return pd.read_parquet(p)
        if p.suffix.lower() in {".json", ".jsonl"}:
            return pd.read_json(p)
        return pd.read_csv(p)
    return obj.copy()


def _write_parquet_safe(df: pd.DataFrame, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(p, index=False)
        return str(p)
    except Exception:
        alt = p.with_suffix(".csv")
        df.to_csv(alt, index=False)
        return str(alt)


def _write_json_safe(payload: Mapping[str, Any], path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(json_safe(dict(payload)), sort_keys=True, indent=2, default=str),
        encoding="utf-8",
    )
    return str(p)


@dataclass(frozen=True)
class BlenderConfig:
    blend_mode: str = "weighted_oof_optimized"  # equal_weight | weighted_static | weighted_oof_optimized
    update_mode: str = "frozen"  # frozen | scheduled_refit | event_driven_refit
    primary_metric: str = "ic"  # ic | mse

    # Eligibility
    min_models_eligible: int = 2
    min_model_coverage: float = 0.6
    min_model_quality: float = -1.0
    redundant_corr_threshold: float = 0.98

    # Weight constraints
    allow_short_weights: bool = False
    max_weight_per_model: float = 0.65
    min_weight_effective: float = 1e-4

    # Objective regularization (for weighted_oof_optimized)
    lambda_concentration: float = 0.05
    lambda_diversity: float = 0.10
    lambda_turnover: float = 0.00
    n_opt_steps: int = 400
    opt_lr: float = 0.05

    # Optional post-regularization
    shrink_to_equal: float = 0.10

    # Static mode
    static_weights: Mapping[str, float] = field(default_factory=dict)

    # Missing-model policy
    renormalize_min_mass: float = 0.50
    fallback_order: tuple[str, ...] = (
        "renormalize_available",
        "fallback_to_equal_weight_on_available",
        "fallback_to_best_base",
        "drop_row",
    )

    # Acceptance policy vs best base
    accept_delta_primary: float = 0.0
    equivalent_epsilon: float = 5e-4
    require_effective_n_min: float = 1.25
    fallback_to_best_base_if_not_accepted: bool = True

    # Run controls
    split_role: str = "oof"
    random_seed: int = 42
    run_id: str = ""


def _normalize_predictions(predictions: pd.DataFrame | str | Path) -> pd.DataFrame:
    df = _to_df(predictions)
    if len(df) == 0:
        raise ValueError("predictions are empty")

    out = df.copy()

    rename = {}
    if "model" in out.columns and "model_id" not in out.columns:
        rename["model"] = "model_id"
    if "prediction" in out.columns and "y_pred" not in out.columns:
        rename["prediction"] = "y_pred"
    if "score" in out.columns and "y_pred" not in out.columns:
        rename["score"] = "y_pred"
    if "asset" in out.columns and "symbol" not in out.columns:
        rename["asset"] = "symbol"

    out = out.rename(columns=rename)

    required = {"date", "symbol", "model_id", "y_pred"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"predictions missing required columns: {missing}")

    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)
    out["model_id"] = out["model_id"].astype(str)
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")

    if "split_role" not in out.columns:
        out["split_role"] = "oof"
    out["split_role"] = out["split_role"].fillna("oof").astype(str).str.lower()

    out = out.drop_duplicates(subset=["date", "symbol", "model_id"], keep="last")
    out = out.sort_values(["date", "symbol", "model_id"]).reset_index(drop=True)
    return out


def _normalize_labels(labels: pd.DataFrame | str | Path | None) -> pd.DataFrame:
    df = _to_df(labels)
    if len(df) == 0:
        return pd.DataFrame(columns=["date", "symbol", "y_true"])

    out = df.copy()
    rename = {}
    if "target" in out.columns and "y_true" not in out.columns:
        rename["target"] = "y_true"
    if "label" in out.columns and "y_true" not in out.columns:
        rename["label"] = "y_true"
    if "asset" in out.columns and "symbol" not in out.columns:
        rename["asset"] = "symbol"

    out = out.rename(columns=rename)
    required = {"date", "symbol", "y_true"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"labels missing required columns: {missing}")

    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    out = out[["date", "symbol", "y_true"]].drop_duplicates(subset=["date", "symbol"], keep="last")
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_blend_panel(
    predictions: pd.DataFrame | str | Path,
    labels: pd.DataFrame | str | Path | None = None,
    *,
    split_role: str = "oof",
) -> pd.DataFrame:
    pred = _normalize_predictions(predictions)
    pred = pred[pred["split_role"] == split_role.lower()].copy()
    if len(pred) == 0:
        raise ValueError(f"no predictions with split_role={split_role}")

    wide = pred.pivot_table(index=["date", "symbol"], columns="model_id", values="y_pred", aggfunc="last")
    wide = wide.sort_index().reset_index()

    if labels is None:
        return wide

    lab = _normalize_labels(labels)
    if len(lab) == 0:
        return wide

    out = wide.merge(lab, on=["date", "symbol"], how="left")
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan")
    corr, _ = spearmanr(x[mask], y[mask])
    return float(corr)


def _metric_primary(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 5:
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    if metric.lower() == "mse":
        return float(-np.mean((yt - yp) ** 2))
    return _safe_spearman(yt, yp)


def _basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 5:
        return {
            "n_obs": int(mask.sum()),
            "ic": np.nan,
            "mse": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "hit_ratio": np.nan,
            "long_short_spread_20": np.nan,
        }

    yt = y_true[mask]
    yp = y_pred[mask]
    mse = float(np.mean((yt - yp) ** 2))

    n = len(yt)
    idx = np.argsort(yp)
    k = max(1, int(0.2 * n))
    ls20 = float(yt[idx[-k:]].mean() - yt[idx[:k]].mean())

    return {
        "n_obs": int(n),
        "ic": _safe_spearman(yt, yp),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(yt - yp))),
        "hit_ratio": float((np.sign(yt) == np.sign(yp)).mean()),
        "long_short_spread_20": ls20,
    }


def _project_simplex_with_cap(w: np.ndarray, cap: float, max_iter: int = 200) -> np.ndarray:
    if len(w) == 0:
        return w

    n = len(w)
    cap_eff = float(max(cap, 1.0 / n + 1e-9))

    z = np.clip(np.asarray(w, dtype=float), 0.0, cap_eff)
    if z.sum() <= 0:
        z = np.full(n, 1.0 / n)

    for _ in range(max_iter):
        z = np.clip(z, 0.0, cap_eff)
        s = z.sum()
        if s <= 0:
            z = np.full(n, 1.0 / n)
            break
        z = z / s
        over = z > cap_eff + 1e-12
        if not over.any():
            break
        excess = float((z[over] - cap_eff).sum())
        z[over] = cap_eff
        free = ~over
        if free.any():
            z[free] += excess / free.sum()
        else:
            z = np.full(n, 1.0 / n)
            break

    z = np.clip(z, 0.0, cap_eff)
    z = z / max(z.sum(), 1e-12)
    return z


def _model_quality_and_coverage(panel: pd.DataFrame, models: Sequence[str], primary_metric: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y = panel["y_true"].to_numpy(dtype=float) if "y_true" in panel.columns else np.full(len(panel), np.nan)

    for m in models:
        x = pd.to_numeric(panel[m], errors="coerce").to_numpy(dtype=float)
        coverage = float(np.isfinite(x).mean())
        quality = _metric_primary(y, x, primary_metric) if "y_true" in panel.columns else np.nan
        rows.append({"model_id": m, "coverage": coverage, "quality": quality})

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["quality", "coverage"], ascending=[False, False]).reset_index(drop=True)
    return out


def select_candidate_models(panel: pd.DataFrame, cfg: BlenderConfig) -> tuple[list[str], pd.DataFrame]:
    base_cols = {"date", "symbol", "y_true"}
    models = [c for c in panel.columns if c not in base_cols]
    if not models:
        raise ValueError("no model columns found in blend panel")

    stats = _model_quality_and_coverage(panel, models, cfg.primary_metric)
    stats["eligible_quality"] = stats["quality"].fillna(0.0) >= cfg.min_model_quality
    stats["eligible_coverage"] = stats["coverage"] >= cfg.min_model_coverage
    stats["eligible_initial"] = stats["eligible_quality"] & stats["eligible_coverage"]

    eligible = stats.loc[stats["eligible_initial"], "model_id"].tolist()

    # Redundancy pruning: for highly correlated pairs, keep only better-quality model.
    dropped_redundant: set[str] = set()
    if len(eligible) >= 2:
        for i, a in enumerate(eligible):
            for b in eligible[i + 1 :]:
                xa = pd.to_numeric(panel[a], errors="coerce").to_numpy(dtype=float)
                xb = pd.to_numeric(panel[b], errors="coerce").to_numpy(dtype=float)
                corr = _safe_spearman(xa, xb)
                if np.isfinite(corr) and abs(corr) >= cfg.redundant_corr_threshold:
                    qa = float(stats.loc[stats["model_id"] == a, "quality"].iloc[0])
                    qb = float(stats.loc[stats["model_id"] == b, "quality"].iloc[0])
                    if np.isnan(qa) and np.isnan(qb):
                        # Keep the one with higher coverage.
                        ca = float(stats.loc[stats["model_id"] == a, "coverage"].iloc[0])
                        cb = float(stats.loc[stats["model_id"] == b, "coverage"].iloc[0])
                        drop = b if ca >= cb else a
                    else:
                        drop = b if qa >= qb else a
                    dropped_redundant.add(drop)

    eligible = [m for m in eligible if m not in dropped_redundant]
    stats["dropped_redundant"] = stats["model_id"].isin(dropped_redundant)
    stats["eligible_final"] = stats["model_id"].isin(eligible)

    return eligible, stats


def _estimate_weights_optimized(
    X: np.ndarray,
    y: np.ndarray,
    cfg: BlenderConfig,
    *,
    previous_weights: np.ndarray | None = None,
) -> np.ndarray:
    if cfg.allow_short_weights:
        raise ValueError("allow_short_weights=True is not supported in this implementation")

    n, p = X.shape
    if p == 0:
        return np.array([], dtype=float)

    # Impute missing predictions per-model with median, preserving PIT semantics.
    Xf = np.asarray(X, dtype=float).copy()
    for j in range(p):
        col = Xf[:, j]
        med = np.nanmedian(col) if np.isfinite(col).any() else 0.0
        col[~np.isfinite(col)] = med
        Xf[:, j] = col

    valid = np.isfinite(y)
    if valid.sum() < max(20, p * 4):
        return np.full(p, 1.0 / p)

    Xv = Xf[valid]
    yv = y[valid]

    # Similarity matrix for diversity penalty.
    C = np.corrcoef(Xv, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)

    w_prev = previous_weights if previous_weights is not None else np.full(p, 1.0 / p)
    w = np.full(p, 1.0 / p)

    for _ in range(cfg.n_opt_steps):
        pred = Xv @ w
        grad_mse = -(Xv.T @ (yv - pred)) / max(len(yv), 1)
        grad_conc = 2.0 * cfg.lambda_concentration * w
        grad_div = 2.0 * cfg.lambda_diversity * (C @ w)
        grad_turn = 2.0 * cfg.lambda_turnover * (w - w_prev)
        grad = grad_mse + grad_conc + grad_div + grad_turn

        w_new = w - cfg.opt_lr * grad
        w_new = _project_simplex_with_cap(w_new, cfg.max_weight_per_model)

        if np.linalg.norm(w_new - w) < 1e-10:
            w = w_new
            break
        w = w_new

    return w


def estimate_blend_weights(
    panel: pd.DataFrame,
    models: Sequence[str],
    cfg: BlenderConfig,
) -> np.ndarray:
    if len(models) == 0:
        return np.array([], dtype=float)

    if cfg.blend_mode == "equal_weight":
        return np.full(len(models), 1.0 / len(models))

    if cfg.blend_mode == "weighted_static":
        raw = np.array([float(cfg.static_weights.get(m, 0.0)) for m in models], dtype=float)
        if np.isfinite(raw).sum() == 0 or raw.sum() <= 0:
            raw = np.full(len(models), 1.0 / len(models))
        return _project_simplex_with_cap(raw, cfg.max_weight_per_model)

    if cfg.blend_mode != "weighted_oof_optimized":
        raise ValueError(f"unsupported blend_mode: {cfg.blend_mode}")

    if "y_true" not in panel.columns:
        return np.full(len(models), 1.0 / len(models))

    X = panel.loc[:, list(models)].to_numpy(dtype=float)
    y = panel["y_true"].to_numpy(dtype=float)
    return _estimate_weights_optimized(X, y, cfg)


def _apply_shrink_and_floor(weights: np.ndarray, cfg: BlenderConfig) -> np.ndarray:
    if len(weights) == 0:
        return weights

    w = np.asarray(weights, dtype=float)
    ew = np.full_like(w, 1.0 / len(w))
    lam = float(np.clip(cfg.shrink_to_equal, 0.0, 1.0))
    w = (1.0 - lam) * w + lam * ew

    w[w < cfg.min_weight_effective] = 0.0
    if w.sum() <= 0:
        w = ew

    return _project_simplex_with_cap(w, cfg.max_weight_per_model)


def apply_blend(
    panel: pd.DataFrame,
    models: Sequence[str],
    weights: np.ndarray,
    cfg: BlenderConfig,
    *,
    best_base_model: str | None = None,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    if len(models) == 0:
        return pd.Series([], dtype=float), pd.Series([], dtype=str), {
            "pct_rows_renormalized": 0.0,
            "pct_rows_fallback": 1.0,
        }

    X = panel.loc[:, list(models)].to_numpy(dtype=float)
    w = np.asarray(weights, dtype=float)

    scores = np.full(len(panel), np.nan, dtype=float)
    status = np.full(len(panel), "drop_row", dtype=object)

    n_renorm = 0
    n_fallback = 0

    model_to_idx = {m: i for i, m in enumerate(models)}
    best_idx = model_to_idx.get(best_base_model) if best_base_model is not None else None

    for i in range(len(panel)):
        row = X[i]
        finite = np.isfinite(row)
        mass = float(np.sum(w[finite]))

        if mass > 0 and mass >= cfg.renormalize_min_mass:
            w_eff = np.zeros_like(w)
            w_eff[finite] = w[finite] / mass
            scores[i] = float(np.nansum(row * w_eff))
            if mass < 0.999999:
                status[i] = "renormalize_available"
                n_renorm += 1
            else:
                status[i] = "full"
            continue

        # Fallback chain.
        handled = False
        for step in cfg.fallback_order:
            if step == "renormalize_available" and mass > 0:
                w_eff = np.zeros_like(w)
                w_eff[finite] = w[finite] / mass
                scores[i] = float(np.nansum(row * w_eff))
                status[i] = "renormalize_available"
                n_renorm += 1
                handled = True
                break

            if step == "fallback_to_equal_weight_on_available" and finite.any():
                scores[i] = float(np.nanmean(row[finite]))
                status[i] = "fallback_equal_available"
                n_fallback += 1
                handled = True
                break

            if step == "fallback_to_best_base" and best_idx is not None and np.isfinite(row[best_idx]):
                scores[i] = float(row[best_idx])
                status[i] = "fallback_best_base"
                n_fallback += 1
                handled = True
                break

            if step == "drop_row":
                scores[i] = np.nan
                status[i] = "drop_row"
                n_fallback += 1
                handled = True
                break

        if not handled:
            scores[i] = np.nan
            status[i] = "drop_row"
            n_fallback += 1

    diag = {
        "pct_rows_renormalized": float(n_renorm / max(len(panel), 1)),
        "pct_rows_fallback": float(n_fallback / max(len(panel), 1)),
    }
    return pd.Series(scores, index=panel.index), pd.Series(status, index=panel.index), diag


def _weight_concentration_stats(weights: np.ndarray) -> dict[str, float]:
    w = np.asarray(weights, dtype=float)
    hhi = float(np.sum(w**2)) if len(w) else np.nan
    eff_n = float(1.0 / hhi) if np.isfinite(hhi) and hhi > 0 else np.nan
    return {"weight_concentration_hhi": hhi, "effective_n_models": eff_n}


def run_blend_sensitivity_suite(
    panel: pd.DataFrame,
    models: Sequence[str],
    weights: np.ndarray,
    cfg: BlenderConfig,
    *,
    best_base_model: str | None = None,
) -> dict[str, Any]:
    y = panel["y_true"].to_numpy(dtype=float) if "y_true" in panel.columns else None
    if y is None or len(models) == 0:
        return {
            "leave_one_model_out": [],
            "weight_perturbation": {},
        }

    loo_rows: list[dict[str, Any]] = []
    for m in models:
        keep_models = [x for x in models if x != m]
        if not keep_models:
            continue
        keep_idx = [models.index(x) for x in keep_models]
        w_keep = weights[keep_idx]
        w_keep = _project_simplex_with_cap(w_keep, cfg.max_weight_per_model)

        score, _, _ = apply_blend(panel, keep_models, w_keep, cfg, best_base_model=best_base_model)
        met = _metric_primary(y, score.to_numpy(dtype=float), cfg.primary_metric)
        loo_rows.append({"dropped_model": m, "metric_after_drop": met})

    rng = np.random.RandomState(cfg.random_seed)
    pert_metrics = []
    for _ in range(32):
        noise = rng.normal(0.0, 0.02, size=len(weights))
        w_pert = _project_simplex_with_cap(weights + noise, cfg.max_weight_per_model)
        s, _, _ = apply_blend(panel, models, w_pert, cfg, best_base_model=best_base_model)
        pert_metrics.append(_metric_primary(y, s.to_numpy(dtype=float), cfg.primary_metric))

    pert_arr = np.asarray(pert_metrics, dtype=float)
    return {
        "leave_one_model_out": loo_rows,
        "weight_perturbation": {
            "n_samples": int(len(pert_arr)),
            "metric_mean": float(np.nanmean(pert_arr)),
            "metric_std": float(np.nanstd(pert_arr)),
            "metric_p05": float(np.nanpercentile(pert_arr, 5)),
            "metric_p95": float(np.nanpercentile(pert_arr, 95)),
        },
    }


def _accept_blend(
    blend_primary: float,
    best_base_primary: float,
    effective_n_models: float,
    cfg: BlenderConfig,
) -> tuple[bool, str]:
    if not np.isfinite(blend_primary) or not np.isfinite(best_base_primary):
        return False, "primary_metric_not_finite"

    better = blend_primary >= best_base_primary + cfg.accept_delta_primary
    equivalent = abs(blend_primary - best_base_primary) <= cfg.equivalent_epsilon
    diversified = np.isfinite(effective_n_models) and effective_n_models >= cfg.require_effective_n_min

    if better:
        return True, "primary_improvement"
    if equivalent and diversified:
        return True, "equivalent_primary_but_more_diversified"
    return False, "below_acceptance_policy"


def persist_blend_artifacts(
    blended_scores: pd.DataFrame,
    blend_weights: pd.DataFrame,
    blend_metrics: Mapping[str, Any],
    blend_metadata: Mapping[str, Any],
    sensitivity_report: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    return {
        "blended_scores": _write_parquet_safe(blended_scores, out / "blended_scores.parquet"),
        "blend_weights": _write_parquet_safe(blend_weights, out / "blend_weights.parquet"),
        "blend_metrics": _write_json_safe(dict(blend_metrics), out / "blend_metrics.json"),
        "blend_metadata": _write_json_safe(dict(blend_metadata), out / "blend_metadata.json"),
        "sensitivity_report": _write_json_safe(dict(sensitivity_report), out / "sensitivity_report.json"),
    }


def run_blender(
    predictions: pd.DataFrame | str | Path,
    labels: pd.DataFrame | str | Path | None = None,
    *,
    config: BlenderConfig | None = None,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    data_snapshot_id: str = "unknown",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    cfg = config or BlenderConfig()
    rid = run_id or cfg.run_id or run_id_with_prefix("blend")

    panel = build_blend_panel(predictions, labels, split_role=cfg.split_role)
    if len(panel) == 0:
        raise ValueError("blend panel is empty")

    eligible_models, eligibility_table = select_candidate_models(panel, cfg)
    if len(eligible_models) < cfg.min_models_eligible:
        # Degrade to top-coverage models if eligibility too strict.
        fallback_candidates = eligibility_table.sort_values("coverage", ascending=False)["model_id"].head(cfg.min_models_eligible)
        eligible_models = fallback_candidates.tolist()

    if len(eligible_models) == 0:
        raise ValueError("no eligible models after filtering")

    weights = estimate_blend_weights(panel, eligible_models, cfg)
    weights = _apply_shrink_and_floor(weights, cfg)

    best_base_model = None
    best_base_primary = np.nan
    if "y_true" in panel.columns:
        qual = eligibility_table.copy()
        qual = qual[qual["model_id"].isin(eligible_models)]
        if len(qual):
            qual = qual.sort_values("quality", ascending=False)
            best_base_model = str(qual.iloc[0]["model_id"])
            best_base_primary = float(qual.iloc[0]["quality"])

    blended_raw, row_status, row_diag = apply_blend(
        panel,
        eligible_models,
        weights,
        cfg,
        best_base_model=best_base_model,
    )

    blended_final = blended_raw.copy()
    blend_status = "active"
    acceptance_reason = "no_labels"

    metrics = {
        "primary_metric": cfg.primary_metric,
        "blend_primary": np.nan,
        "best_base_primary": best_base_primary,
        "accepted": False,
    }

    if "y_true" in panel.columns:
        y_true = panel["y_true"].to_numpy(dtype=float)
        blend_primary = _metric_primary(y_true, blended_raw.to_numpy(dtype=float), cfg.primary_metric)
        weight_stats = _weight_concentration_stats(weights)

        accepted, reason = _accept_blend(
            blend_primary,
            best_base_primary,
            weight_stats["effective_n_models"],
            cfg,
        )

        if not accepted and cfg.fallback_to_best_base_if_not_accepted and best_base_model is not None:
            blended_final = pd.to_numeric(panel[best_base_model], errors="coerce")
            blend_status = "fallback_to_best_base"
        elif accepted:
            blend_status = "active"
        else:
            blend_status = "active_not_accepted"

        acceptance_reason = reason
        metrics.update(
            {
                "blend_primary": float(blend_primary),
                "accepted": bool(accepted),
                "acceptance_reason": reason,
                **weight_stats,
                **_basic_metrics(y_true, blended_raw.to_numpy(dtype=float)),
            }
        )
    else:
        metrics.update(_weight_concentration_stats(weights))

    blended_scores = panel[["date", "symbol"]].copy()
    blended_scores["score_blend_raw"] = blended_raw.values
    blended_scores["score_blend"] = blended_final.values
    blended_scores["row_status"] = row_status.values
    blended_scores["blend_status"] = blend_status
    blended_scores["run_id"] = rid

    if best_base_model is not None:
        blended_scores["score_best_base"] = pd.to_numeric(panel[best_base_model], errors="coerce")
        blended_scores["best_base_model"] = best_base_model

    blend_weights = pd.DataFrame(
        {
            "model_id": eligible_models,
            "weight": weights,
        }
    )
    blend_weights = blend_weights.merge(
        eligibility_table[["model_id", "coverage", "quality", "eligible_final"]],
        on="model_id",
        how="left",
    )
    blend_weights["run_id"] = rid

    sensitivity = run_blend_sensitivity_suite(
        panel,
        eligible_models,
        weights,
        cfg,
        best_base_model=best_base_model,
    )

    runtime_sec = float(time.perf_counter() - t0)
    metadata = {
        "run_id": rid,
        "timestamp": utc_now_iso(),
        "config_hash": config_hash(asdict(cfg), n=24),
        "data_snapshot_id": data_snapshot_id,
        "blend_mode": cfg.blend_mode,
        "update_mode": cfg.update_mode,
        "n_models_candidates": int(len([c for c in panel.columns if c not in {"date", "symbol", "y_true"}])),
        "n_models_eligible": int(eligibility_table["eligible_final"].sum()) if "eligible_final" in eligibility_table.columns else int(len(eligible_models)),
        "n_models_blended": int(len(eligible_models)),
        "pct_rows_renormalized": row_diag["pct_rows_renormalized"],
        "pct_rows_fallback": row_diag["pct_rows_fallback"],
        "fallback_status": blend_status,
        "acceptance_reason": acceptance_reason,
        "runtime_sec": runtime_sec,
    }

    metrics.update(
        {
            "best_base_model": best_base_model,
            "runtime_sec": runtime_sec,
            "pct_rows_renormalized": row_diag["pct_rows_renormalized"],
            "pct_rows_fallback": row_diag["pct_rows_fallback"],
        }
    )

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_blend_artifacts(
            blended_scores,
            blend_weights,
            metrics,
            metadata,
            sensitivity,
            output_dir=output_dir,
        )

    return {
        "blended_scores": blended_scores,
        "blend_weights": blend_weights,
        "blend_metrics": metrics,
        "blend_metadata": metadata,
        "sensitivity_report": sensitivity,
        "eligibility_table": eligibility_table,
        "artifacts": artifacts,
    }


__all__ = [
    "BlenderConfig",
    "build_blend_panel",
    "select_candidate_models",
    "estimate_blend_weights",
    "apply_blend",
    "run_blend_sensitivity_suite",
    "persist_blend_artifacts",
    "run_blender",
]
