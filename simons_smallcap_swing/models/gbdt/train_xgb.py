"""
models/gbdt/train_xgb.py — XGBoost training with temporal discipline.

Same contract as train_lgbm: per-fold, early stopping on valid, OOF/OOS,
per-date IC, promotion gates. Uses XGBoost DMatrix API.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from .train_lgbm import _spearman_ic, _compute_metrics


@dataclass
class XGBTrainConfig:
    objective: str = "reg:squarederror"
    primary_metric: str = "ic"
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 2000
    early_stopping_rounds: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.7
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    min_child_weight: int = 5
    gamma: float = 0.0
    tree_method: str = "hist"
    seed: int = 42
    metric_threshold: float = 0.02
    overfit_gap_max: float = 0.10
    coverage_min: float = 0.80
    extra_params: dict = field(default_factory=dict)


def train_xgb(
    features: np.ndarray,
    labels: np.ndarray,
    splits: list[dict[str, np.ndarray]],
    *,
    dates: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    config: XGBTrainConfig | None = None,
) -> dict[str, Any]:
    """Train XGBoost across temporal folds."""
    cfg = config or XGBTrainConfig()
    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=float).ravel()
    N, P = X.shape

    try:
        import xgboost as xgb
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False

    oof_preds = np.full(N, np.nan)
    test_preds_accum = np.zeros(N)
    test_counts = np.zeros(N, dtype=int)
    fold_metrics = []
    models = []
    importances = np.zeros(P)

    params = {
        "objective": cfg.objective,
        "max_depth": cfg.max_depth,
        "eta": cfg.learning_rate,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "alpha": cfg.reg_alpha,
        "lambda": cfg.reg_lambda,
        "min_child_weight": cfg.min_child_weight,
        "gamma": cfg.gamma,
        "tree_method": cfg.tree_method,
        "seed": cfg.seed,
        "verbosity": 0,
        **cfg.extra_params,
    }

    for k, fold in enumerate(splits):
        tr_idx, va_idx = fold["train_idx"], fold["valid_idx"]
        te_idx = fold.get("test_idx", np.array([], dtype=int))

        if HAS_XGB:
            dtrain = xgb.DMatrix(X[tr_idx], label=y[tr_idx])
            dvalid = xgb.DMatrix(X[va_idx], label=y[va_idx])
            evals = [(dvalid, "valid")]
            model = xgb.train(params, dtrain, num_boost_round=cfg.n_estimators,
                              evals=evals, early_stopping_rounds=cfg.early_stopping_rounds,
                              verbose_eval=False)
            pred_va = model.predict(dvalid)
            pred_te = model.predict(xgb.DMatrix(X[te_idx])) if len(te_idx) > 0 else np.array([])
            imp = np.zeros(P)
            scores = model.get_score(importance_type="gain")
            for fname, val in scores.items():
                idx = int(fname.replace("f", "")) if fname.startswith("f") else 0
                if 0 <= idx < P:
                    imp[idx] = val
            best_iter = model.best_iteration
        else:
            from numpy.linalg import lstsq
            w, _, _, _ = lstsq(X[tr_idx], y[tr_idx], rcond=None)
            model = {"weights": w, "engine": "ridge_fallback"}
            pred_va = X[va_idx] @ w
            pred_te = X[te_idx] @ w if len(te_idx) > 0 else np.array([])
            imp = np.abs(w)
            best_iter = 1

        oof_preds[va_idx] = pred_va
        if len(te_idx) > 0:
            test_preds_accum[te_idx] += pred_te
            test_counts[te_idx] += 1
        models.append(model)
        importances += imp / max(imp.sum(), 1e-12)

        m_valid = _compute_metrics(y[va_idx], pred_va,
                                   dates=dates[va_idx] if dates is not None else None,
                                   primary=cfg.primary_metric)
        m_test = _compute_metrics(y[te_idx], pred_te,
                                  dates=dates[te_idx] if dates is not None and len(te_idx) > 0 else None,
                                  primary=cfg.primary_metric) if len(te_idx) > 0 else {}
        fold_metrics.append({
            "fold": k, "best_iteration": best_iter,
            "valid_primary": m_valid.get("primary", np.nan),
            "test_primary": m_test.get("primary", np.nan),
            "valid_metrics": m_valid, "test_metrics": m_test,
        })

    # Aggregate
    test_mask = test_counts > 0
    test_preds = np.full(N, np.nan)
    test_preds[test_mask] = test_preds_accum[test_mask] / test_counts[test_mask]

    oof_valid = np.isfinite(oof_preds)
    oof_metrics = _compute_metrics(y[oof_valid], oof_preds[oof_valid],
                                   dates=dates[oof_valid] if dates is not None else None,
                                   primary=cfg.primary_metric)

    valid_p = [f["valid_primary"] for f in fold_metrics if np.isfinite(f["valid_primary"])]
    test_p = [f["test_primary"] for f in fold_metrics if np.isfinite(f.get("test_primary", np.nan))]

    summary = {
        "engine": "xgboost" if HAS_XGB else "ridge_fallback",
        "objective": cfg.objective,
        "primary_metric": cfg.primary_metric,
        "primary_valid_mean": float(np.mean(valid_p)) if valid_p else np.nan,
        "primary_test_mean": float(np.mean(test_p)) if test_p else np.nan,
        "oof_primary": oof_metrics.get("primary", np.nan),
        "oof_median_ic": oof_metrics.get("median_ic", np.nan),
        "coverage": float(oof_valid.mean()),
        "overfit_gap": round(float(np.mean(valid_p) - np.mean(test_p)), 4) if valid_p and test_p else np.nan,
        "n_folds": len(splits),
        "seed": cfg.seed,
    }

    accepted = True; reasons = []
    if np.isfinite(summary.get("oof_primary", np.nan)) and summary["oof_primary"] < cfg.metric_threshold:
        accepted = False; reasons.append("metric_below_threshold")
    if np.isfinite(summary.get("overfit_gap", np.nan)) and abs(summary["overfit_gap"]) > cfg.overfit_gap_max:
        accepted = False; reasons.append("overfit_gap_exceeded")
    if summary["coverage"] < cfg.coverage_min:
        accepted = False; reasons.append("coverage_low")
    summary["accepted"] = accepted
    summary["rejection_reasons"] = reasons

    imp_norm = importances / max(importances.sum(), 1e-12)
    importance = dict(zip(feature_names or [f"f{i}" for i in range(P)], imp_norm.tolist()))

    return {
        "oof_preds": oof_preds, "test_preds": test_preds,
        "metrics_by_fold": fold_metrics, "summary": summary,
        "models": models, "feature_importance": importance,
    }
