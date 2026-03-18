"""
models/ensemble/diversity_metrics.py - Diversity governance for ensemble candidates.

This module evaluates pairwise model complementarity on aligned OOF panels and
produces candidate model subsets for downstream blender/stacking.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DiversityConfig:
    # Core metrics
    top_k_frac: float = 0.2
    disagreement_tau: float = 0.0
    min_intersection: int = 200
    weak_coverage_ratio: float = 0.4

    # Score weights: diversity
    w_pred: float = 0.25
    w_err: float = 0.35
    w_topk: float = 0.20
    w_disagree: float = 0.20

    # Score weights: redundancy
    u_pred: float = 0.30
    u_err: float = 0.40
    u_topk: float = 0.20
    u_disagree: float = 0.10

    # Thresholds / flags
    pred_corr_high: float = 0.98
    err_corr_high: float = 0.95
    topk_overlap_high: float = 0.85
    low_power_n: int = 300
    unstable_pair_std_threshold: float = 0.15

    # Candidate-set selection objective
    alpha_quality: float = 0.70
    beta_diversity: float = 0.25
    gamma_complexity: float = 0.05
    max_set_size: int = 8
    min_delta_score: float = 1e-4

    # Selection metric
    quality_metric: str = "ic"  # ic | mse

    # Run controls
    run_id: str = ""


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
        csv_path = p.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return str(csv_path)


def _write_json_safe(payload: Mapping[str, Any], path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(json_safe(dict(payload)), sort_keys=True, indent=2, default=str),
        encoding="utf-8",
    )
    return str(p)


def _write_config_snapshot(cfg: DiversityConfig, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json_safe(asdict(cfg))
    try:
        import yaml  # type: ignore

        p.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
        return str(p)
    except Exception:
        alt = p.with_suffix(".json")
        alt.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        return str(alt)


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


def _normalize_labels(labels: pd.DataFrame | str | Path) -> pd.DataFrame:
    df = _to_df(labels)
    if len(df) == 0:
        raise ValueError("labels are empty")

    out = df.copy()

    rename = {}
    if "asset" in out.columns and "symbol" not in out.columns:
        rename["asset"] = "symbol"
    if "target" in out.columns and "y_true" not in out.columns:
        rename["target"] = "y_true"
    if "label" in out.columns and "y_true" not in out.columns:
        rename["label"] = "y_true"

    out = out.rename(columns=rename)

    required = {"date", "symbol", "y_true"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"labels missing required columns: {missing}")

    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")

    keep_cols = ["date", "symbol", "y_true"]
    if "regime_id" in out.columns:
        keep_cols.append("regime_id")

    out = out[keep_cols].drop_duplicates(subset=["date", "symbol"], keep="last")
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)

    return out


def build_aligned_panel(
    predictions: pd.DataFrame | str | Path,
    labels: pd.DataFrame | str | Path,
    *,
    split_role: str = "oof",
) -> pd.DataFrame:
    pred = _normalize_predictions(predictions)
    lab = _normalize_labels(labels)

    pred = pred[pred["split_role"] == split_role.lower()].copy()
    if len(pred) == 0:
        raise ValueError(f"no predictions with split_role={split_role}")

    panel = pred.merge(lab, on=["date", "symbol"], how="inner", validate="many_to_one")
    panel = panel.dropna(subset=["y_pred", "y_true"]).sort_values(["date", "symbol", "model_id"]).reset_index(drop=True)

    if panel.duplicated(subset=["date", "symbol", "model_id"]).any():
        raise ValueError("aligned panel has duplicated (date,symbol,model_id)")

    if len(panel) == 0:
        raise ValueError("aligned panel is empty")

    return panel


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan")
    corr, _ = spearmanr(x[mask], y[mask])
    return float(corr)


def _safe_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return _safe_spearman(y_true, y_pred)


def _quality_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    m = str(metric).lower()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 5:
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    if m == "mse":
        return float(-np.mean((yt - yp) ** 2))
    return _safe_ic(yt, yp)


def _topk_overlap_for_pair(pair: pd.DataFrame, top_k_frac: float) -> float:
    if len(pair) == 0:
        return float("nan")

    fracs = []
    for _, g in pair.groupby("date", sort=True):
        n = len(g)
        k = max(1, int(np.floor(n * top_k_frac)))
        a_idx = set(g.sort_values("y_pred_a", ascending=False).head(k)["symbol"].tolist())
        b_idx = set(g.sort_values("y_pred_b", ascending=False).head(k)["symbol"].tolist())
        frac = len(a_idx.intersection(b_idx)) / max(k, 1)
        fracs.append(frac)

    return float(np.median(fracs)) if fracs else float("nan")


def _disagreement(pair: pd.DataFrame, tau: float) -> float:
    if len(pair) == 0:
        return float("nan")

    def sgn(v: np.ndarray, t: float) -> np.ndarray:
        out = np.zeros_like(v, dtype=int)
        out[v >= t] = 1
        out[v <= -t] = -1
        return out

    a = pair["y_pred_a"].to_numpy(dtype=float)
    b = pair["y_pred_b"].to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return float("nan")
    sa = sgn(a[mask], tau)
    sb = sgn(b[mask], tau)
    return float(np.mean(sa != sb))


def _error_corr(pair: pd.DataFrame) -> float:
    if len(pair) == 0:
        return float("nan")
    ea = pair["y_true"].to_numpy(dtype=float) - pair["y_pred_a"].to_numpy(dtype=float)
    eb = pair["y_true"].to_numpy(dtype=float) - pair["y_pred_b"].to_numpy(dtype=float)
    return _safe_spearman(ea, eb)


def _incremental_gain_pair(pair: pd.DataFrame, metric: str) -> float:
    if len(pair) == 0:
        return float("nan")

    y = pair["y_true"].to_numpy(dtype=float)
    pa = pair["y_pred_a"].to_numpy(dtype=float)
    pb = pair["y_pred_b"].to_numpy(dtype=float)
    blend = 0.5 * pa + 0.5 * pb

    qa = _quality_metric(y, pa, metric)
    qb = _quality_metric(y, pb, metric)
    qpair = _quality_metric(y, blend, metric)

    if not np.isfinite(qa) or not np.isfinite(qb) or not np.isfinite(qpair):
        return float("nan")

    return float(qpair - max(qa, qb))


def _pair_stability_std(pair: pd.DataFrame, n_chunks: int = 4) -> float:
    if len(pair) < n_chunks * 20:
        return float("nan")

    dates = np.array(sorted(pair["date"].unique()))
    if len(dates) < n_chunks:
        return float("nan")

    chunks = np.array_split(dates, n_chunks)
    vals = []
    for d in chunks:
        g = pair[pair["date"].isin(d)]
        v = _error_corr(g)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 2:
        return float("nan")
    return float(np.std(vals))


def _diversity_score(
    pred_corr: float,
    err_corr: float,
    topk_overlap: float,
    disagreement: float,
    cfg: DiversityConfig,
) -> float:
    return float(
        cfg.w_pred * (1.0 - abs(pred_corr if np.isfinite(pred_corr) else 1.0))
        + cfg.w_err * (1.0 - abs(err_corr if np.isfinite(err_corr) else 1.0))
        + cfg.w_topk * (1.0 - (topk_overlap if np.isfinite(topk_overlap) else 1.0))
        + cfg.w_disagree * (disagreement if np.isfinite(disagreement) else 0.0)
    )


def _redundancy_score(
    pred_corr: float,
    err_corr: float,
    topk_overlap: float,
    disagreement: float,
    cfg: DiversityConfig,
) -> float:
    return float(
        cfg.u_pred * abs(pred_corr if np.isfinite(pred_corr) else 1.0)
        + cfg.u_err * abs(err_corr if np.isfinite(err_corr) else 1.0)
        + cfg.u_topk * (topk_overlap if np.isfinite(topk_overlap) else 1.0)
        - cfg.u_disagree * (disagreement if np.isfinite(disagreement) else 0.0)
    )


def _model_quality_table(panel: pd.DataFrame, cfg: DiversityConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id, g in panel.groupby("model_id", sort=True):
        y = g["y_true"].to_numpy(dtype=float)
        p = g["y_pred"].to_numpy(dtype=float)
        q = _quality_metric(y, p, cfg.quality_metric)
        rows.append(
            {
                "model_id": str(model_id),
                "n_obs": int(len(g)),
                "quality_raw": q,
                "date_first": pd.Timestamp(g["date"].min()),
                "date_last": pd.Timestamp(g["date"].max()),
            }
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    q = out["quality_raw"].to_numpy(dtype=float)
    if np.isfinite(q).sum() == 0:
        out["quality_norm"] = 0.5
    else:
        qmin = np.nanmin(q)
        qmax = np.nanmax(q)
        if np.isfinite(qmax - qmin) and (qmax - qmin) > 1e-12:
            out["quality_norm"] = (q - qmin) / (qmax - qmin)
        else:
            out["quality_norm"] = 0.5

    return out.sort_values("quality_raw", ascending=False).reset_index(drop=True)


def _regime_collapse_flag(pair: pd.DataFrame, cfg: DiversityConfig, overall_diversity: float) -> bool:
    if "regime_id" not in pair.columns:
        return False

    regime_vals: list[float] = []
    for _, g in pair.groupby("regime_id", sort=True):
        if len(g) < max(30, cfg.min_intersection // 4):
            continue
        pred_corr = _safe_spearman(
            g["y_pred_a"].to_numpy(dtype=float), g["y_pred_b"].to_numpy(dtype=float)
        )
        err_corr = _error_corr(g)
        topk = _topk_overlap_for_pair(g, cfg.top_k_frac)
        dis = _disagreement(g, cfg.disagreement_tau)
        regime_vals.append(_diversity_score(pred_corr, err_corr, topk, dis, cfg))

    if len(regime_vals) < 2:
        return False

    return bool(np.nanmin(regime_vals) + 0.15 < np.nanmedian(regime_vals) and np.isfinite(overall_diversity))


def compute_pairwise_metrics(panel: pd.DataFrame, cfg: DiversityConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = sorted(panel["model_id"].astype(str).unique().tolist())
    if len(models) < 2:
        raise ValueError("need at least two models for diversity analysis")

    model_quality = _model_quality_table(panel, cfg)
    model_frames = {
        m: panel.loc[panel["model_id"] == m, ["date", "symbol", "y_pred", "y_true", *(["regime_id"] if "regime_id" in panel.columns else [])]].copy()
        for m in models
    }

    pair_rows: list[dict[str, Any]] = []
    for i, model_a in enumerate(models):
        for model_b in models[i + 1 :]:
            a = model_frames[model_a].rename(columns={"y_pred": "y_pred_a"})
            b = model_frames[model_b].rename(columns={"y_pred": "y_pred_b", "y_true": "y_true_b"})
            keep_cols = ["date", "symbol", "y_pred_b", "y_true_b"]
            if "regime_id" in b.columns:
                keep_cols.append("regime_id")
            pair = a.merge(
                b[keep_cols],
                on=["date", "symbol"],
                how="inner",
                suffixes=("", "_r"),
            )
            if len(pair) == 0:
                continue

            if "y_true_b" in pair.columns:
                pair["y_true"] = pair["y_true"].fillna(pair["y_true_b"])

            n_intersection = int(len(pair))
            n_a = int(len(a))
            n_b = int(len(b))
            coverage_ratio = float(n_intersection / max(min(n_a, n_b), 1))

            pred_corr = _safe_spearman(
                pair["y_pred_a"].to_numpy(dtype=float),
                pair["y_pred_b"].to_numpy(dtype=float),
            )
            err_corr = _error_corr(pair)
            topk_overlap = _topk_overlap_for_pair(pair, cfg.top_k_frac)
            disagree = _disagreement(pair, cfg.disagreement_tau)
            incremental_gain = _incremental_gain_pair(pair, cfg.quality_metric)
            pair_stability_std = _pair_stability_std(pair)

            diversity_score = _diversity_score(pred_corr, err_corr, topk_overlap, disagree, cfg)
            redundancy_score = _redundancy_score(pred_corr, err_corr, topk_overlap, disagree, cfg)

            high_pred_corr_flag = bool(np.isfinite(pred_corr) and abs(pred_corr) >= cfg.pred_corr_high)
            high_error_corr_flag = bool(np.isfinite(err_corr) and abs(err_corr) >= cfg.err_corr_high)
            high_topk_overlap_flag = bool(np.isfinite(topk_overlap) and topk_overlap >= cfg.topk_overlap_high)
            near_duplicate_models_flag = bool(
                high_pred_corr_flag and high_error_corr_flag and high_topk_overlap_flag
            )
            low_power_flag = bool(n_intersection < cfg.low_power_n)
            weak_comparability_flag = bool(
                n_intersection < cfg.min_intersection or coverage_ratio < cfg.weak_coverage_ratio
            )
            unstable_pair_flag = bool(
                np.isfinite(pair_stability_std)
                and pair_stability_std > cfg.unstable_pair_std_threshold
            )
            regime_collapse_flag = _regime_collapse_flag(pair, cfg, diversity_score)

            pair_rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_intersection": n_intersection,
                    "pct_relative_coverage": coverage_ratio,
                    "date_range_intersection": f"{pair['date'].min().date()}::{pair['date'].max().date()}",
                    "pred_corr": pred_corr,
                    "error_corr": err_corr,
                    "topk_overlap": topk_overlap,
                    "disagreement": disagree,
                    "incremental_gain": incremental_gain,
                    "pair_stability_std": pair_stability_std,
                    "diversity_score": diversity_score,
                    "redundancy_score": redundancy_score,
                    "high_pred_corr_flag": high_pred_corr_flag,
                    "high_error_corr_flag": high_error_corr_flag,
                    "high_topk_overlap_flag": high_topk_overlap_flag,
                    "near_duplicate_models_flag": near_duplicate_models_flag,
                    "low_power_flag": low_power_flag,
                    "unstable_pair_flag": unstable_pair_flag,
                    "regime_collapse_flag": regime_collapse_flag,
                    "weak_comparability_flag": weak_comparability_flag,
                }
            )

    pairwise = pd.DataFrame(pair_rows)
    if len(pairwise) == 0:
        raise ValueError("unable to compute pairwise metrics: no valid pair intersections")

    pairwise = pairwise.sort_values(
        ["redundancy_score", "n_intersection"], ascending=[False, False]
    ).reset_index(drop=True)

    redundancy_flags = pairwise[
        [
            "model_a",
            "model_b",
            "high_pred_corr_flag",
            "high_error_corr_flag",
            "high_topk_overlap_flag",
            "near_duplicate_models_flag",
            "low_power_flag",
            "unstable_pair_flag",
            "regime_collapse_flag",
            "weak_comparability_flag",
        ]
    ].copy()

    flag_cols = [
        "high_pred_corr_flag",
        "high_error_corr_flag",
        "high_topk_overlap_flag",
        "near_duplicate_models_flag",
        "low_power_flag",
        "unstable_pair_flag",
        "regime_collapse_flag",
        "weak_comparability_flag",
    ]
    redundancy_flags["flag_count"] = redundancy_flags[flag_cols].sum(axis=1)
    redundancy_flags["critical_flag"] = redundancy_flags[
        ["near_duplicate_models_flag", "low_power_flag", "unstable_pair_flag", "regime_collapse_flag"]
    ].any(axis=1)

    return pairwise, model_quality, redundancy_flags


def _pair_lookup(pairwise: pd.DataFrame, value_col: str) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for _, r in pairwise.iterrows():
        a = str(r["model_a"])
        b = str(r["model_b"])
        v = float(r[value_col]) if pd.notna(r[value_col]) else float("nan")
        out[(a, b)] = v
        out[(b, a)] = v
    return out


def _flag_lookup(pairwise: pd.DataFrame, flag_col: str) -> dict[tuple[str, str], bool]:
    out: dict[tuple[str, str], bool] = {}
    for _, r in pairwise.iterrows():
        a = str(r["model_a"])
        b = str(r["model_b"])
        v = bool(r.get(flag_col, False))
        out[(a, b)] = v
        out[(b, a)] = v
    return out


def generate_candidate_sets(
    model_quality: pd.DataFrame,
    pairwise: pd.DataFrame,
    cfg: DiversityConfig,
) -> pd.DataFrame:
    if len(model_quality) == 0:
        return pd.DataFrame(
            columns=[
                "step",
                "selected_model",
                "set_size",
                "model_set",
                "quality_component",
                "diversity_component",
                "complexity_penalty",
                "objective_score",
                "delta_score",
                "stopped",
                "stop_reason",
            ]
        )

    quality_norm = {
        str(r["model_id"]): float(r["quality_norm"])
        for _, r in model_quality.iterrows()
        if pd.notna(r["quality_norm"])
    }

    if not quality_norm:
        return pd.DataFrame()

    pair_div = _pair_lookup(pairwise, "diversity_score")
    near_dup = _flag_lookup(pairwise, "near_duplicate_models_flag")

    ordered_models = [str(m) for m in model_quality.sort_values("quality_raw", ascending=False)["model_id"].tolist()]
    current_set = [ordered_models[0]]

    def components(models: Sequence[str]) -> dict[str, float]:
        q = float(np.mean([quality_norm.get(m, 0.0) for m in models])) if models else 0.0
        if len(models) < 2:
            d = 0.0
        else:
            vals = []
            for i, a in enumerate(models):
                for b in models[i + 1 :]:
                    v = pair_div.get((a, b), np.nan)
                    if np.isfinite(v):
                        vals.append(v)
            d = float(np.mean(vals)) if vals else 0.0

        c = float(len(models))
        score = cfg.alpha_quality * q + cfg.beta_diversity * d - cfg.gamma_complexity * c
        return {
            "quality_component": q,
            "diversity_component": d,
            "complexity_penalty": c,
            "objective_score": float(score),
        }

    rows: list[dict[str, Any]] = []
    base = components(current_set)
    rows.append(
        {
            "step": 0,
            "selected_model": current_set[0],
            "set_size": 1,
            "model_set": ",".join(current_set),
            "quality_component": base["quality_component"],
            "diversity_component": base["diversity_component"],
            "complexity_penalty": base["complexity_penalty"],
            "objective_score": base["objective_score"],
            "delta_score": np.nan,
            "stopped": False,
            "stop_reason": "",
        }
    )

    while len(current_set) < cfg.max_set_size:
        current = components(current_set)
        best_model = None
        best_delta = -np.inf
        best_components = None

        for candidate in ordered_models:
            if candidate in current_set:
                continue

            if any(near_dup.get((candidate, m), False) for m in current_set):
                continue

            test_set = current_set + [candidate]
            test_comp = components(test_set)
            delta = test_comp["objective_score"] - current["objective_score"]
            if delta > best_delta:
                best_delta = float(delta)
                best_model = candidate
                best_components = test_comp

        if best_model is None or best_components is None:
            rows.append(
                {
                    "step": len(rows),
                    "selected_model": "",
                    "set_size": len(current_set),
                    "model_set": ",".join(current_set),
                    "quality_component": current["quality_component"],
                    "diversity_component": current["diversity_component"],
                    "complexity_penalty": current["complexity_penalty"],
                    "objective_score": current["objective_score"],
                    "delta_score": 0.0,
                    "stopped": True,
                    "stop_reason": "no_viable_candidate",
                }
            )
            break

        if best_delta <= cfg.min_delta_score:
            rows.append(
                {
                    "step": len(rows),
                    "selected_model": "",
                    "set_size": len(current_set),
                    "model_set": ",".join(current_set),
                    "quality_component": current["quality_component"],
                    "diversity_component": current["diversity_component"],
                    "complexity_penalty": current["complexity_penalty"],
                    "objective_score": current["objective_score"],
                    "delta_score": float(best_delta),
                    "stopped": True,
                    "stop_reason": "min_delta_not_met",
                }
            )
            break

        current_set.append(best_model)
        rows.append(
            {
                "step": len(rows),
                "selected_model": best_model,
                "set_size": len(current_set),
                "model_set": ",".join(current_set),
                "quality_component": best_components["quality_component"],
                "diversity_component": best_components["diversity_component"],
                "complexity_penalty": best_components["complexity_penalty"],
                "objective_score": best_components["objective_score"],
                "delta_score": float(best_delta),
                "stopped": False,
                "stop_reason": "",
            }
        )

    return pd.DataFrame(rows)


def _summary_payload(
    panel: pd.DataFrame,
    pairwise: pd.DataFrame,
    model_quality: pd.DataFrame,
    candidate_sets: pd.DataFrame,
    redundancy_flags: pd.DataFrame,
    runtime_sec: float,
    cfg: DiversityConfig,
    run_id: str,
) -> dict[str, Any]:
    n_models = int(panel["model_id"].nunique())
    n_pairs = int(len(pairwise))
    high_pred_pct = float(pairwise["high_pred_corr_flag"].mean()) if n_pairs else np.nan
    high_err_pct = float(pairwise["high_error_corr_flag"].mean()) if n_pairs else np.nan
    low_cov_pct = float(pairwise["weak_comparability_flag"].mean()) if n_pairs else np.nan

    final_set = ""
    if len(candidate_sets):
        last = candidate_sets.iloc[-1]
        final_set = str(last.get("model_set", ""))

    return {
        "run_id": run_id,
        "timestamp": utc_now_iso(),
        "quality_metric": cfg.quality_metric,
        "n_rows_panel": int(len(panel)),
        "n_models": n_models,
        "n_pairs": n_pairs,
        "n_candidate_steps": int(len(candidate_sets)),
        "final_candidate_set": final_set,
        "avg_diversity_score": float(pd.to_numeric(pairwise["diversity_score"], errors="coerce").mean()),
        "avg_redundancy_score": float(pd.to_numeric(pairwise["redundancy_score"], errors="coerce").mean()),
        "pct_pairs_high_pred_corr": high_pred_pct,
        "pct_pairs_high_error_corr": high_err_pct,
        "pct_pairs_low_coverage": low_cov_pct,
        "pct_pairs_unstable": float(pairwise["unstable_pair_flag"].mean()) if n_pairs else np.nan,
        "pct_pairs_low_power": float(pairwise["low_power_flag"].mean()) if n_pairs else np.nan,
        "runtime_sec": float(runtime_sec),
        "top_model_by_quality": model_quality["model_id"].iloc[0] if len(model_quality) else None,
        "n_redundancy_critical": int(redundancy_flags["critical_flag"].sum()) if len(redundancy_flags) else 0,
    }


def persist_diversity_outputs(
    pairwise: pd.DataFrame,
    summary: Mapping[str, Any],
    candidate_sets: pd.DataFrame,
    redundancy_flags: pd.DataFrame,
    model_quality: pd.DataFrame,
    cfg: DiversityConfig,
    metadata: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "pairwise_diversity_metrics": _write_parquet_safe(pairwise, out / "pairwise_diversity_metrics.parquet"),
        "pairwise_diversity_summary": _write_json_safe(dict(summary), out / "pairwise_diversity_summary.json"),
        "candidate_model_sets": _write_parquet_safe(candidate_sets, out / "candidate_model_sets.parquet"),
        "redundancy_flags": _write_parquet_safe(redundancy_flags, out / "redundancy_flags.parquet"),
        "model_quality": _write_parquet_safe(model_quality, out / "model_quality.parquet"),
        "diversity_config_snapshot": _write_config_snapshot(cfg, out / "diversity_config_snapshot.yaml"),
        "diversity_metadata": _write_json_safe(dict(metadata), out / "diversity_metadata.json"),
    }
    return artifacts


def run_diversity_metrics(
    predictions: pd.DataFrame | str | Path,
    labels: pd.DataFrame | str | Path,
    *,
    config: DiversityConfig | None = None,
    split_role: str = "oof",
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    data_snapshot_id: str = "unknown",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    cfg = config or DiversityConfig()

    rid = run_id or cfg.run_id or run_id_with_prefix("diversity")
    panel = build_aligned_panel(predictions, labels, split_role=split_role)

    pairwise, model_quality, redundancy_flags = compute_pairwise_metrics(panel, cfg)
    candidate_sets = generate_candidate_sets(model_quality, pairwise, cfg)

    runtime_sec = float(time.perf_counter() - t0)
    summary = _summary_payload(
        panel,
        pairwise,
        model_quality,
        candidate_sets,
        redundancy_flags,
        runtime_sec,
        cfg,
        rid,
    )

    metadata = {
        "run_id": rid,
        "timestamp": utc_now_iso(),
        "data_snapshot_id": data_snapshot_id,
        "split_role": split_role,
        "config_hash": config_hash(asdict(cfg), n=24),
        "n_models": int(panel["model_id"].nunique()),
        "n_pairs": int(len(pairwise)),
        "panel_date_first": str(panel["date"].min().date()),
        "panel_date_last": str(panel["date"].max().date()),
        "quality_metric": cfg.quality_metric,
    }

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_diversity_outputs(
            pairwise,
            summary,
            candidate_sets,
            redundancy_flags,
            model_quality,
            cfg,
            metadata,
            output_dir=output_dir,
        )

    return {
        "pairwise_diversity_metrics": pairwise,
        "pairwise_diversity_summary": summary,
        "candidate_model_sets": candidate_sets,
        "redundancy_flags": redundancy_flags,
        "model_quality": model_quality,
        "diversity_metadata": metadata,
        "artifacts": artifacts,
    }


__all__ = [
    "DiversityConfig",
    "build_aligned_panel",
    "compute_pairwise_metrics",
    "generate_candidate_sets",
    "persist_diversity_outputs",
    "run_diversity_metrics",
]
