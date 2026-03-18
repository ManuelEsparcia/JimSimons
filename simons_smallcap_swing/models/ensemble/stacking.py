"""
models/ensemble/stacking.py - Institutional level-1 meta-learning.

This module trains a regularized stacking layer on OOF base-model predictions,
with purged temporal CV, one-standard-error hyperparameter selection,
acceptance gates, and deterministic artifacts.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass, asdict
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


def _write_pickle_safe(obj: Any, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(obj, f)
    return str(p)


@dataclass(frozen=True)
class StackingConfig:
    meta_model_type: str = "ridge"  # ridge | elasticnet (future)
    primary_metric: str = "ic"  # ic | mse
    alpha_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

    # Temporal CV
    n_folds: int = 5
    min_train_dates: int = 126
    gap_days: int = 5
    embargo_days: int = 0

    # Matrix hygiene
    max_feature_corr: float = 0.999
    min_feature_variance: float = 1e-10

    # Acceptance gates
    min_samples: int = 800
    min_samples_per_feature_ratio: float = 20.0
    performance_delta_min: float = 0.0
    equivalence_epsilon: float = 5e-4
    stability_std_max: float = 0.10
    max_df_eff: float = 50.0

    fallback_to_blender: bool = True
    split_role_train: str = "oof"
    split_role_oos: str = "oos"
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
    return out.sort_values(["date", "symbol", "model_id"]).reset_index(drop=True)


def _normalize_labels(labels: pd.DataFrame | str | Path) -> pd.DataFrame:
    df = _to_df(labels)
    if len(df) == 0:
        raise ValueError("labels are empty")

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


def build_meta_feature_matrix(
    predictions: pd.DataFrame | str | Path,
    labels: pd.DataFrame | str | Path,
    *,
    split_role: str = "oof",
    model_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    pred = _normalize_predictions(predictions)
    pred = pred[pred["split_role"] == split_role.lower()].copy()
    if len(pred) == 0:
        raise ValueError(f"no predictions with split_role={split_role}")

    if model_ids is not None:
        use = set(str(m) for m in model_ids)
        pred = pred[pred["model_id"].isin(use)].copy()

    lab = _normalize_labels(labels)

    wide = pred.pivot_table(index=["date", "symbol"], columns="model_id", values="y_pred", aggfunc="last")
    wide = wide.sort_index().reset_index()

    out = wide.merge(lab, on=["date", "symbol"], how="inner")
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)

    if out.duplicated(subset=["date", "symbol"]).any():
        raise ValueError("meta matrix has duplicated (date,symbol)")
    if len(out) == 0:
        raise ValueError("meta matrix is empty after alignment")

    return out


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan")
    corr, _ = spearmanr(x[mask], y[mask])
    return float(corr)


def _metric_primary(y_true: np.ndarray, y_pred: np.ndarray, primary: str) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 5:
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    if primary.lower() == "mse":
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
        }

    yt = y_true[mask]
    yp = y_pred[mask]
    mse = float(np.mean((yt - yp) ** 2))
    return {
        "n_obs": int(len(yt)),
        "ic": _safe_spearman(yt, yp),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(yt - yp))),
        "hit_ratio": float((np.sign(yt) == np.sign(yp)).mean()),
    }

def validate_meta_matrix(meta: pd.DataFrame) -> dict[str, Any]:
    feature_cols = [c for c in meta.columns if c not in {"date", "symbol", "y_true"}]
    if len(feature_cols) == 0:
        raise ValueError("meta matrix has no feature columns")

    issues: list[str] = []
    if meta["date"].isna().any():
        issues.append("null_dates")
    if meta["symbol"].isna().any():
        issues.append("null_symbols")
    if meta["y_true"].isna().mean() > 0.50:
        issues.append("high_missing_labels")

    return {
        "n_rows": int(len(meta)),
        "n_features_raw": int(len(feature_cols)),
        "issues": issues,
    }


def prune_meta_features(meta: pd.DataFrame, cfg: StackingConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_cols = [c for c in meta.columns if c not in {"date", "symbol", "y_true"}]
    work = meta.copy()

    dropped_low_var = []
    for c in feature_cols:
        v = float(np.nanvar(pd.to_numeric(work[c], errors="coerce")))
        if not np.isfinite(v) or v <= cfg.min_feature_variance:
            dropped_low_var.append(c)

    keep = [c for c in feature_cols if c not in dropped_low_var]

    dropped_high_corr: list[str] = []
    if len(keep) >= 2:
        corr = work[keep].corr(method="spearman").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        # deterministic drop: lower-variance column in each high-corr pair.
        for c in upper.columns:
            high = upper.index[(upper[c] > cfg.max_feature_corr) & upper[c].notna()].tolist()
            for r in high:
                if c in dropped_high_corr or r in dropped_high_corr:
                    continue
                var_c = float(np.nanvar(pd.to_numeric(work[c], errors="coerce")))
                var_r = float(np.nanvar(pd.to_numeric(work[r], errors="coerce")))
                drop = c if var_c <= var_r else r
                dropped_high_corr.append(drop)

    keep = [c for c in keep if c not in set(dropped_high_corr)]
    if not keep:
        raise ValueError("all meta-features removed during pruning")

    out = work[["date", "symbol", "y_true", *keep]].copy()

    vif_max = np.nan
    if len(keep) >= 2:
        X = out[keep].to_numpy(dtype=float)
        for j in range(X.shape[1]):
            col = X[:, j]
            med = np.nanmedian(col) if np.isfinite(col).any() else 0.0
            col[~np.isfinite(col)] = med
            X[:, j] = col
        C = np.corrcoef(X, rowvar=False)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            invC = np.linalg.pinv(C)
            vif = np.diag(invC)
            vif_max = float(np.nanmax(vif))
        except Exception:
            vif_max = np.nan

    report = {
        "n_features_before": int(len(feature_cols)),
        "n_features_after": int(len(keep)),
        "dropped_low_variance": sorted(dropped_low_var),
        "dropped_high_correlation": sorted(set(dropped_high_corr)),
        "vif_max": vif_max,
    }
    return out, report


def build_purged_cv_folds(
    dates: pd.Series,
    cfg: StackingConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    d = pd.to_datetime(dates).to_numpy()
    unique_dates = np.sort(np.unique(d))

    if len(unique_dates) <= cfg.min_train_dates + cfg.n_folds:
        raise ValueError("not enough unique dates for temporal CV")

    available = len(unique_dates) - cfg.min_train_dates
    block = max(1, available // cfg.n_folds)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(cfg.n_folds):
        test_start = cfg.min_train_dates + i * block
        test_end = min(test_start + block, len(unique_dates))
        if test_start >= len(unique_dates):
            break

        test_dates = unique_dates[test_start:test_end]
        train_end = max(0, test_start - cfg.gap_days - cfg.embargo_days)
        train_dates = unique_dates[:train_end]

        if len(train_dates) < max(40, cfg.min_train_dates // 2):
            continue

        train_mask = np.isin(d, train_dates)
        test_mask = np.isin(d, test_dates)

        if train_mask.sum() < 50 or test_mask.sum() < 20:
            continue

        folds.append((train_mask, test_mask))

    if len(folds) < 2:
        raise ValueError("insufficient valid temporal folds after purging")

    return folds


def _fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    Xf = np.asarray(X, dtype=float)
    yv = np.asarray(y, dtype=float)

    x_mu = np.nanmean(Xf, axis=0)
    x_sd = np.nanstd(Xf, axis=0)
    x_sd[x_sd < 1e-12] = 1.0

    Xz = (Xf - x_mu) / x_sd
    Xz = np.nan_to_num(Xz, nan=0.0, posinf=0.0, neginf=0.0)

    y_mu = float(np.nanmean(yv))
    yc = yv - y_mu
    yc = np.nan_to_num(yc, nan=0.0, posinf=0.0, neginf=0.0)

    p = Xz.shape[1]
    A = Xz.T @ Xz + float(alpha) * np.eye(p)
    b = Xz.T @ yc
    coef_z = np.linalg.pinv(A) @ b

    coef = coef_z / x_sd
    intercept = y_mu - float(np.dot(x_mu, coef))
    return coef.astype(float), float(intercept)


def _fit_predict_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=float(alpha), fit_intercept=True)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        coef = model.coef_.astype(float)
        intercept = float(model.intercept_)
        return pred.astype(float), coef, intercept
    except Exception:
        coef, intercept = _fit_ridge_closed_form(X_train, y_train, alpha)
        pred = X_test @ coef + intercept
        return pred.astype(float), coef, intercept


def _alpha_cv_scores(
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    alpha_grid: Sequence[float],
    primary_metric: str,
) -> dict[float, list[float]]:
    scores: dict[float, list[float]] = {float(a): [] for a in alpha_grid}

    for tr_mask, va_mask in folds:
        Xtr = X[tr_mask]
        ytr = y[tr_mask]
        Xva = X[va_mask]
        yva = y[va_mask]

        # Fold-level scaling (train-only).
        mu = np.nanmean(Xtr, axis=0)
        sd = np.nanstd(Xtr, axis=0)
        sd[sd < 1e-12] = 1.0
        Xtr_s = np.nan_to_num((Xtr - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
        Xva_s = np.nan_to_num((Xva - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)

        valid_tr = np.isfinite(ytr)
        valid_va = np.isfinite(yva)
        if valid_tr.sum() < max(30, Xtr_s.shape[1] * 2) or valid_va.sum() < 20:
            continue

        Xtr_use = Xtr_s[valid_tr]
        ytr_use = ytr[valid_tr]
        Xva_use = Xva_s[valid_va]
        yva_use = yva[valid_va]

        for alpha in alpha_grid:
            pred, _, _ = _fit_predict_ridge(Xtr_use, ytr_use, Xva_use, float(alpha))
            score = _metric_primary(yva_use, pred, primary_metric)
            if np.isfinite(score):
                scores[float(alpha)].append(float(score))

    return scores


def _select_alpha_one_se(scores_by_alpha: Mapping[float, Sequence[float]]) -> tuple[float, pd.DataFrame]:
    rows = []
    for alpha, vals in scores_by_alpha.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            rows.append({"alpha": float(alpha), "mean_score": np.nan, "std_score": np.nan, "n": 0, "se": np.nan})
        else:
            rows.append(
                {
                    "alpha": float(alpha),
                    "mean_score": float(np.mean(arr)),
                    "std_score": float(np.std(arr, ddof=0)),
                    "n": int(len(arr)),
                    "se": float(np.std(arr, ddof=0) / np.sqrt(max(len(arr), 1))),
                }
            )

    table = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)
    valid = table["mean_score"].notna()
    if not valid.any():
        best_alpha = float(table["alpha"].median())
        return best_alpha, table

    best_idx = int(table.loc[valid, "mean_score"].idxmax())
    best_mean = float(table.loc[best_idx, "mean_score"])
    best_se = float(table.loc[best_idx, "se"]) if np.isfinite(table.loc[best_idx, "se"]) else 0.0

    threshold = best_mean - best_se
    candidates = table[(table["mean_score"] >= threshold) & table["mean_score"].notna()]
    if len(candidates) == 0:
        return float(table.loc[best_idx, "alpha"]), table

    # One-SE rule for ridge: choose the largest alpha among statistically equivalent options.
    alpha = float(candidates["alpha"].max())
    return alpha, table


def _degrees_of_freedom_ridge(X: np.ndarray, alpha: float) -> float:
    Xf = np.asarray(X, dtype=float)
    Xf = np.nan_to_num(Xf, nan=0.0, posinf=0.0, neginf=0.0)
    XtX = Xf.T @ Xf
    p = XtX.shape[0]
    A = XtX + float(alpha) * np.eye(p)
    try:
        H_part = np.linalg.pinv(A) @ XtX
        return float(np.trace(H_part))
    except Exception:
        return float("nan")

def train_stacking_ridge(
    meta: pd.DataFrame,
    cfg: StackingConfig,
) -> dict[str, Any]:
    feature_cols = [c for c in meta.columns if c not in {"date", "symbol", "y_true"}]
    X = meta[feature_cols].to_numpy(dtype=float)
    y = meta["y_true"].to_numpy(dtype=float)

    folds = build_purged_cv_folds(meta["date"], cfg)
    alpha_scores = _alpha_cv_scores(X, y, folds, cfg.alpha_grid, cfg.primary_metric)
    alpha_selected, alpha_table = _select_alpha_one_se(alpha_scores)

    oof = np.full(len(meta), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []

    for fold_id, (tr_mask, va_mask) in enumerate(folds):
        Xtr = X[tr_mask]
        ytr = y[tr_mask]
        Xva = X[va_mask]
        yva = y[va_mask]

        mu = np.nanmean(Xtr, axis=0)
        sd = np.nanstd(Xtr, axis=0)
        sd[sd < 1e-12] = 1.0
        Xtr_s = np.nan_to_num((Xtr - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
        Xva_s = np.nan_to_num((Xva - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)

        valid_tr = np.isfinite(ytr)
        valid_va = np.isfinite(yva)
        if valid_tr.sum() < max(30, Xtr_s.shape[1] * 2) or valid_va.sum() < 20:
            continue

        pred_va, coef, intercept = _fit_predict_ridge(
            Xtr_s[valid_tr],
            ytr[valid_tr],
            Xva_s[valid_va],
            alpha_selected,
        )

        va_idx = np.where(va_mask)[0]
        va_idx_valid = va_idx[valid_va]
        oof[va_idx_valid] = pred_va

        metric_va = _metric_primary(yva[valid_va], pred_va, cfg.primary_metric)
        fold_rows.append(
            {
                "fold": int(fold_id),
                "n_train": int(valid_tr.sum()),
                "n_valid": int(valid_va.sum()),
                "alpha": float(alpha_selected),
                "metric_valid": metric_va,
            }
        )

        for c, b in zip(feature_cols, coef):
            coef_rows.append({"fold": int(fold_id), "feature": c, "coef": float(b), "intercept": intercept})

    # Refit final model on all valid rows.
    valid_all = np.isfinite(y)
    Xall = X[valid_all]
    yall = y[valid_all]

    mu_all = np.nanmean(Xall, axis=0)
    sd_all = np.nanstd(Xall, axis=0)
    sd_all[sd_all < 1e-12] = 1.0
    Xall_s = np.nan_to_num((Xall - mu_all) / sd_all, nan=0.0, posinf=0.0, neginf=0.0)

    # Final fit in scaled space; convert back to original-space coefficients.
    coef_s, intercept_s = _fit_ridge_closed_form(Xall_s, yall, alpha_selected)
    coef_orig = coef_s / sd_all
    intercept_orig = float(intercept_s - np.dot(mu_all / sd_all, coef_s))

    Xfull = np.nan_to_num(X.copy(), nan=np.nan)
    for j in range(Xfull.shape[1]):
        col = Xfull[:, j]
        med = np.nanmedian(col) if np.isfinite(col).any() else 0.0
        col[~np.isfinite(col)] = med
        Xfull[:, j] = col
    full_pred = Xfull @ coef_orig + intercept_orig

    df_eff = _degrees_of_freedom_ridge(Xall_s, alpha_selected)

    coef_df = pd.DataFrame(coef_rows)
    coef_summary = pd.DataFrame(columns=["feature", "coef_mean", "coef_std", "abs_coef_mean", "sign_flip_rate"])
    if len(coef_df):
        rows = []
        for feat, g in coef_df.groupby("feature", sort=True):
            arr = g["coef"].to_numpy(dtype=float)
            rows.append(
                {
                    "feature": feat,
                    "coef_mean": float(np.mean(arr)),
                    "coef_std": float(np.std(arr, ddof=0)),
                    "abs_coef_mean": float(np.mean(np.abs(arr))),
                    "sign_flip_rate": float((np.sign(arr) != np.sign(np.nanmedian(arr))).mean()),
                }
            )
        coef_summary = pd.DataFrame(rows).sort_values("abs_coef_mean", ascending=False).reset_index(drop=True)

    return {
        "feature_cols": feature_cols,
        "oof_pred": oof,
        "full_pred": full_pred,
        "alpha_selected": float(alpha_selected),
        "alpha_cv_table": alpha_table,
        "fold_metrics": pd.DataFrame(fold_rows),
        "coef_by_fold": coef_df,
        "coef_summary": coef_summary,
        "final_coef": coef_orig,
        "final_intercept": intercept_orig,
        "df_eff": float(df_eff),
    }


def _best_base_and_blender_baselines(meta: pd.DataFrame, cfg: StackingConfig) -> dict[str, Any]:
    feature_cols = [c for c in meta.columns if c not in {"date", "symbol", "y_true"}]
    y = meta["y_true"].to_numpy(dtype=float)

    best_model = None
    best_metric = -np.inf
    by_model: dict[str, float] = {}

    for c in feature_cols:
        p = pd.to_numeric(meta[c], errors="coerce").to_numpy(dtype=float)
        m = _metric_primary(y, p, cfg.primary_metric)
        by_model[c] = m
        if np.isfinite(m) and m > best_metric:
            best_metric = float(m)
            best_model = c

    # Blender baseline: equal-weight over available base predictions.
    X = meta[feature_cols].to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        blender_score = np.nanmean(X, axis=1)
    blender_metric = _metric_primary(y, blender_score, cfg.primary_metric)

    return {
        "best_base_model": best_model,
        "best_base_metric": float(best_metric) if np.isfinite(best_metric) else np.nan,
        "best_base_by_model": by_model,
        "blender_equal_weight_metric": float(blender_metric),
        "blender_equal_weight_score": blender_score,
    }


def run_stack_acceptance_gates(
    meta: pd.DataFrame,
    train_out: Mapping[str, Any],
    baselines: Mapping[str, Any],
    cfg: StackingConfig,
) -> dict[str, Any]:
    y = meta["y_true"].to_numpy(dtype=float)
    oof = np.asarray(train_out["oof_pred"], dtype=float)
    valid = np.isfinite(y) & np.isfinite(oof)

    fold_metrics = train_out["fold_metrics"]
    fold_std = float(pd.to_numeric(fold_metrics["metric_valid"], errors="coerce").std()) if len(fold_metrics) else np.nan

    stack_metric = _metric_primary(y[valid], oof[valid], cfg.primary_metric) if valid.sum() else np.nan
    best_base = float(baselines.get("best_base_metric", np.nan))
    blender = float(baselines.get("blender_equal_weight_metric", np.nan))

    n_rows = int(len(meta))
    n_feat = int(len(train_out.get("feature_cols", [])))

    gate_1_hygiene = bool(valid.sum() >= max(50, n_rows // 2))
    gate_2_sufficiency = bool(
        n_rows >= cfg.min_samples and n_feat > 0 and (n_rows / max(n_feat, 1)) >= cfg.min_samples_per_feature_ratio
    )

    target = np.nanmax([best_base, blender])
    perf_improve = bool(np.isfinite(stack_metric) and np.isfinite(target) and stack_metric >= target + cfg.performance_delta_min)
    perf_equiv = bool(np.isfinite(stack_metric) and np.isfinite(target) and abs(stack_metric - target) <= cfg.equivalence_epsilon)
    gate_3_performance = bool(perf_improve or perf_equiv)

    gate_4_stability = bool(np.isfinite(fold_std) and fold_std <= cfg.stability_std_max)
    gate_5_complexity = bool(np.isfinite(train_out["df_eff"]) and train_out["df_eff"] <= cfg.max_df_eff)
    gate_6_operability = bool(n_feat > 0 and np.isfinite(train_out["alpha_selected"]))

    accepted = bool(
        gate_1_hygiene
        and gate_2_sufficiency
        and gate_3_performance
        and gate_4_stability
        and gate_5_complexity
        and gate_6_operability
    )

    return {
        "accepted": accepted,
        "stack_metric": float(stack_metric) if np.isfinite(stack_metric) else np.nan,
        "best_base_metric": best_base,
        "blender_metric": blender,
        "fold_stability_std": fold_std,
        "df_eff": float(train_out["df_eff"]),
        "gates": {
            "hygiene": gate_1_hygiene,
            "sufficiency": gate_2_sufficiency,
            "performance": gate_3_performance,
            "stability": gate_4_stability,
            "complexity": gate_5_complexity,
            "operability": gate_6_operability,
        },
    }

def persist_stack_artifacts(
    meta_model_obj: Mapping[str, Any],
    stack_oof: pd.DataFrame,
    stack_oos: pd.DataFrame,
    schema: Mapping[str, Any],
    metrics_summary: Mapping[str, Any],
    gates_report: Mapping[str, Any],
    feature_importance: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    return {
        "meta_model": _write_pickle_safe(meta_model_obj, out / "meta_model.pkl"),
        "stack_oof": _write_parquet_safe(stack_oof, out / "stack_oof.parquet"),
        "stack_oos": _write_parquet_safe(stack_oos, out / "stack_oos.parquet"),
        "meta_feature_matrix_schema": _write_json_safe(dict(schema), out / "meta_feature_matrix_schema.json"),
        "stack_metrics_summary": _write_json_safe(dict(metrics_summary), out / "stack_metrics_summary.json"),
        "stack_gates_report": _write_json_safe(dict(gates_report), out / "stack_gates_report.json"),
        "meta_feature_importance": _write_parquet_safe(feature_importance, out / "meta_feature_importance.parquet"),
        "manifest": _write_json_safe(dict(manifest), out / "manifest.json"),
    }


def run_stacking(
    predictions: pd.DataFrame | str | Path,
    labels: pd.DataFrame | str | Path,
    *,
    candidate_models: Sequence[str] | None = None,
    config: StackingConfig | None = None,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    data_snapshot_id: str = "unknown",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    cfg = config or StackingConfig()
    rid = run_id or cfg.run_id or run_id_with_prefix("stack")

    meta_raw = build_meta_feature_matrix(
        predictions,
        labels,
        split_role=cfg.split_role_train,
        model_ids=candidate_models,
    )
    validation = validate_meta_matrix(meta_raw)
    meta, feature_pruning = prune_meta_features(meta_raw, cfg)

    train_out = train_stacking_ridge(meta, cfg)
    baselines = _best_base_and_blender_baselines(meta, cfg)
    gates = run_stack_acceptance_gates(meta, train_out, baselines, cfg)

    y_true = meta["y_true"].to_numpy(dtype=float)
    stack_raw = train_out["oof_pred"]
    blender_score = np.asarray(baselines["blender_equal_weight_score"], dtype=float)
    best_base_model = baselines.get("best_base_model")
    best_base_score = pd.to_numeric(meta[best_base_model], errors="coerce").to_numpy(dtype=float) if best_base_model else np.full(len(meta), np.nan)

    if gates["accepted"]:
        stack_final = stack_raw.copy()
        stack_status = "active"
    else:
        if cfg.fallback_to_blender:
            stack_final = blender_score.copy()
            stack_status = "fallback_blender"
        else:
            stack_final = stack_raw.copy()
            stack_status = "active_not_accepted"

    stack_oof = meta[["date", "symbol"]].copy()
    stack_oof["stack_score_raw"] = stack_raw
    stack_oof["stack_score"] = stack_final
    stack_oof["blender_score_eqw"] = blender_score
    stack_oof["best_base_score"] = best_base_score
    stack_oof["stack_status"] = stack_status
    stack_oof["run_id"] = rid

    # OOS scoring (if split_role_oos predictions exist).
    stack_oos = pd.DataFrame(columns=["date", "symbol", "stack_score_raw", "stack_score", "run_id"])
    try:
        meta_oos = build_meta_feature_matrix(
            predictions,
            labels,
            split_role=cfg.split_role_oos,
            model_ids=train_out["feature_cols"],
        )
        fcols = train_out["feature_cols"]
        Xo = meta_oos[fcols].to_numpy(dtype=float)
        for j in range(Xo.shape[1]):
            col = Xo[:, j]
            med = np.nanmedian(col) if np.isfinite(col).any() else 0.0
            col[~np.isfinite(col)] = med
            Xo[:, j] = col

        coef = np.asarray(train_out["final_coef"], dtype=float)
        inter = float(train_out["final_intercept"])
        pred_oos = Xo @ coef + inter

        stack_oos = meta_oos[["date", "symbol"]].copy()
        stack_oos["stack_score_raw"] = pred_oos
        if gates["accepted"]:
            stack_oos["stack_score"] = pred_oos
        else:
            stack_oos["stack_score"] = np.nanmean(Xo, axis=1) if cfg.fallback_to_blender else pred_oos
        stack_oos["run_id"] = rid
    except Exception:
        # No OOS block is valid for the current inputs; keep empty table.
        pass

    metrics_summary = {
        "run_id": rid,
        "timestamp": utc_now_iso(),
        "primary_metric": cfg.primary_metric,
        "stack_status": stack_status,
        "n_rows_meta_train": int(len(meta)),
        "n_meta_features": int(len(train_out["feature_cols"])),
        "alpha_selected": float(train_out["alpha_selected"]),
        "df_eff": float(train_out["df_eff"]),
        "improvement_vs_best_base": (
            float(gates["stack_metric"] - gates["best_base_metric"])
            if np.isfinite(gates["stack_metric"]) and np.isfinite(gates["best_base_metric"])
            else np.nan
        ),
        "improvement_vs_blender": (
            float(gates["stack_metric"] - gates["blender_metric"])
            if np.isfinite(gates["stack_metric"]) and np.isfinite(gates["blender_metric"])
            else np.nan
        ),
        "fold_stability_std": gates["fold_stability_std"],
        "accepted": bool(gates["accepted"]),
        **_basic_metrics(y_true, stack_raw),
    }

    schema = {
        "feature_cols": list(train_out["feature_cols"]),
        "n_features": int(len(train_out["feature_cols"])),
        "meta_model_type": cfg.meta_model_type,
        "primary_metric": cfg.primary_metric,
        "split_role_train": cfg.split_role_train,
        "split_role_oos": cfg.split_role_oos,
    }

    meta_model_obj = {
        "model_type": "ridge",
        "alpha": float(train_out["alpha_selected"]),
        "feature_cols": list(train_out["feature_cols"]),
        "coef": np.asarray(train_out["final_coef"], dtype=float),
        "intercept": float(train_out["final_intercept"]),
        "run_id": rid,
    }

    runtime_sec = float(time.perf_counter() - t0)
    manifest = {
        "run_id": rid,
        "timestamp": utc_now_iso(),
        "config_hash": config_hash(asdict(cfg), n=24),
        "data_snapshot_id": data_snapshot_id,
        "stack_status": stack_status,
        "accepted": bool(gates["accepted"]),
        "n_rows_meta_train": int(len(meta)),
        "n_rows_meta_oos": int(len(stack_oos)),
        "feature_pruning": feature_pruning,
        "validation": validation,
        "runtime_sec": runtime_sec,
    }

    gates_report = {
        "run_id": rid,
        "accepted": bool(gates["accepted"]),
        "stack_status": stack_status,
        "gates": gates["gates"],
        "stack_metric": gates["stack_metric"],
        "best_base_metric": gates["best_base_metric"],
        "blender_metric": gates["blender_metric"],
        "fold_stability_std": gates["fold_stability_std"],
        "df_eff": gates["df_eff"],
    }

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_stack_artifacts(
            meta_model_obj,
            stack_oof,
            stack_oos,
            schema,
            metrics_summary,
            gates_report,
            train_out["coef_summary"],
            manifest,
            output_dir=output_dir,
        )

    return {
        "stack_oof": stack_oof,
        "stack_oos": stack_oos,
        "meta_model": meta_model_obj,
        "meta_feature_importance": train_out["coef_summary"],
        "alpha_cv_table": train_out["alpha_cv_table"],
        "fold_metrics": train_out["fold_metrics"],
        "stack_metrics_summary": metrics_summary,
        "stack_gates_report": gates_report,
        "meta_feature_matrix_schema": schema,
        "manifest": manifest,
        "artifacts": artifacts,
    }


__all__ = [
    "StackingConfig",
    "build_meta_feature_matrix",
    "validate_meta_matrix",
    "prune_meta_features",
    "build_purged_cv_folds",
    "train_stacking_ridge",
    "run_stack_acceptance_gates",
    "persist_stack_artifacts",
    "run_stacking",
]
