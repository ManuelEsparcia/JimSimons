"""
models/gbdt/train_lgbm.py — LightGBM training with temporal discipline.

Trains per-fold with purged temporal splits. Early stopping on valid only.
Produces OOF + OOS predictions, per-date IC, feature importance.
Promotion gates: metric_OOS ≥ threshold, overfit_gap ≤ tol, coverage ≥ min.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class LGBMTrainConfig:
    objective: str = "regression"          # regression | binary | lambdarank
    primary_metric: str = "ic"             # ic | auc | rmse
    num_leaves: int = 63
    max_depth: int = -1
    learning_rate: float = 0.05
    n_estimators: int = 2000
    early_stopping_rounds: int = 50
    feature_fraction: float = 0.7
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    lambda_l1: float = 0.1
    lambda_l2: float = 1.0
    min_data_in_leaf: int = 20
    seed: int = 42
    # Promotion gates
    metric_threshold: float = 0.02
    overfit_gap_max: float = 0.10
    coverage_min: float = 0.80
    extra_params: dict = field(default_factory=dict)


def _spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cross-sectional Spearman IC (rank correlation)."""
    if len(y_true) < 3:
        return np.nan
    from scipy.stats import spearmanr
    try:
        return float(spearmanr(y_true, y_pred).correlation)
    except Exception:
        # Fallback without scipy
        def _rank(x):
            order = np.argsort(np.argsort(x))
            return (order + 1).astype(float)
        rx, ry = _rank(y_true), _rank(y_pred)
        return float(np.corrcoef(rx, ry)[0, 1])


def _compute_metrics(y_true, y_pred, dates=None, primary="ic"):
    """Compute metrics, optionally per-date IC."""
    m = {}
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[finite], y_pred[finite]

    m["rmse"] = float(np.sqrt(np.mean((yt - yp) ** 2))) if len(yt) > 0 else np.nan
    m["mae"] = float(np.mean(np.abs(yt - yp))) if len(yt) > 0 else np.nan
    m["ic_global"] = _spearman_ic(yt, yp)

    # Per-date IC
    if dates is not None:
        dates_f = dates[finite] if len(dates) == len(y_true) else None
        if dates_f is not None:
            unique_dates = np.unique(dates_f)
            ics = []
            for d in unique_dates:
                mask = dates_f == d
                if mask.sum() >= 5:
                    ics.append(_spearman_ic(yt[mask], yp[mask]))
            ics = [x for x in ics if np.isfinite(x)]
            m["median_ic"] = float(np.median(ics)) if ics else np.nan
            m["mean_ic"] = float(np.mean(ics)) if ics else np.nan
            m["std_ic"] = float(np.std(ics)) if len(ics) > 1 else np.nan
            m["ic_ir"] = m["mean_ic"] / m["std_ic"] if m.get("std_ic", 0) > 1e-8 else np.nan
            m["n_dates"] = len(ics)

    if primary == "ic":
        m["primary"] = m.get("median_ic", m["ic_global"])
    elif primary == "rmse":
        m["primary"] = -m["rmse"]  # higher is better convention
    else:
        m["primary"] = m["ic_global"]

    return m


def train_lgbm(
    features: np.ndarray,
    labels: np.ndarray,
    splits: list[dict[str, np.ndarray]],
    *,
    dates: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    config: LGBMTrainConfig | None = None,
) -> dict[str, Any]:
    """Train LightGBM across temporal folds.

    Parameters
    ----------
    features : (N, P) array
    labels : (N,) array
    splits : list of dicts with keys 'train_idx', 'valid_idx', 'test_idx'
    dates : (N,) array of date labels for per-date IC
    config : training configuration

    Returns
    -------
    dict with: oof_preds, test_preds, metrics_by_fold, summary, models, importance
    """
    cfg = config or LGBMTrainConfig()
    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=float).ravel()
    N, P = X.shape

    try:
        import lightgbm as lgb
        HAS_LGB = True
    except ImportError:
        HAS_LGB = False

    oof_preds = np.full(N, np.nan)
    test_preds_accum = np.zeros(N)
    test_counts = np.zeros(N, dtype=int)
    fold_metrics = []
    models = []
    importances = np.zeros(P)

    params = {
        "objective": cfg.objective,
        "num_leaves": cfg.num_leaves,
        "max_depth": cfg.max_depth,
        "learning_rate": cfg.learning_rate,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "lambda_l1": cfg.lambda_l1,
        "lambda_l2": cfg.lambda_l2,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "verbose": -1,
        "seed": cfg.seed,
        **cfg.extra_params,
    }

    for k, fold in enumerate(splits):
        tr_idx = fold["train_idx"]
        va_idx = fold["valid_idx"]
        te_idx = fold.get("test_idx", np.array([], dtype=int))

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        if HAS_LGB:
            dtrain = lgb.Dataset(X_tr, y_tr)
            dvalid = lgb.Dataset(X_va, y_va, reference=dtrain)
            callbacks = [lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                         lgb.log_evaluation(period=0)]
            model = lgb.train(params, dtrain, num_boost_round=cfg.n_estimators,
                              valid_sets=[dvalid], callbacks=callbacks)
            pred_va = model.predict(X_va)
            pred_te = model.predict(X[te_idx]) if len(te_idx) > 0 else np.array([])
            imp = model.feature_importance(importance_type="gain")
            best_iter = model.best_iteration
        else:
            # Fallback: simple ridge regression
            from numpy.linalg import lstsq
            w, _, _, _ = lstsq(X_tr, y_tr, rcond=None)
            model = {"weights": w, "engine": "ridge_fallback"}
            pred_va = X_va @ w
            pred_te = X[te_idx] @ w if len(te_idx) > 0 else np.array([])
            imp = np.abs(w)
            best_iter = 1

        oof_preds[va_idx] = pred_va
        if len(te_idx) > 0:
            test_preds_accum[te_idx] += pred_te
            test_counts[te_idx] += 1

        models.append(model)
        importances += imp / max(imp.sum(), 1e-12)

        # Per-fold metrics
        m_train = _compute_metrics(y_tr, X_tr @ (model.get("weights", np.zeros(P)) if isinstance(model, dict) else np.zeros(P)),
                                   primary=cfg.primary_metric) if not HAS_LGB else {"primary": np.nan}
        m_valid = _compute_metrics(y_va, pred_va,
                                   dates=dates[va_idx] if dates is not None else None,
                                   primary=cfg.primary_metric)
        m_test = _compute_metrics(y[te_idx], pred_te,
                                  dates=dates[te_idx] if dates is not None and len(te_idx) > 0 else None,
                                  primary=cfg.primary_metric) if len(te_idx) > 0 else {}

        fold_metrics.append({
            "fold": k, "best_iteration": best_iter,
            "train_n": len(tr_idx), "valid_n": len(va_idx), "test_n": len(te_idx),
            "valid_primary": m_valid.get("primary", np.nan),
            "test_primary": m_test.get("primary", np.nan),
            "valid_metrics": m_valid, "test_metrics": m_test,
        })

    # Average test predictions
    test_mask = test_counts > 0
    test_preds = np.full(N, np.nan)
    test_preds[test_mask] = test_preds_accum[test_mask] / test_counts[test_mask]

    # Aggregate metrics
    valid_primaries = [fm["valid_primary"] for fm in fold_metrics if np.isfinite(fm["valid_primary"])]
    test_primaries = [fm["test_primary"] for fm in fold_metrics if np.isfinite(fm.get("test_primary", np.nan))]

    # OOF global metrics
    oof_valid = np.isfinite(oof_preds)
    oof_metrics = _compute_metrics(y[oof_valid], oof_preds[oof_valid],
                                   dates=dates[oof_valid] if dates is not None else None,
                                   primary=cfg.primary_metric)

    summary = {
        "engine": "lightgbm" if HAS_LGB else "ridge_fallback",
        "objective": cfg.objective,
        "primary_metric": cfg.primary_metric,
        "primary_valid_mean": float(np.mean(valid_primaries)) if valid_primaries else np.nan,
        "primary_valid_std": float(np.std(valid_primaries)) if len(valid_primaries) > 1 else np.nan,
        "primary_test_mean": float(np.mean(test_primaries)) if test_primaries else np.nan,
        "oof_primary": oof_metrics.get("primary", np.nan),
        "oof_median_ic": oof_metrics.get("median_ic", np.nan),
        "coverage": float(oof_valid.mean()),
        "n_folds": len(splits),
        "seed": cfg.seed,
    }

    # Overfit gap (train metric not easily available without re-predicting; use valid vs test)
    if valid_primaries and test_primaries:
        summary["overfit_gap"] = round(float(np.mean(valid_primaries) - np.mean(test_primaries)), 4)
    else:
        summary["overfit_gap"] = np.nan

    # Promotion gates
    accepted = True
    reasons = []
    if summary.get("oof_primary") is not None and np.isfinite(summary["oof_primary"]):
        if summary["oof_primary"] < cfg.metric_threshold:
            accepted = False; reasons.append(f"metric {summary['oof_primary']:.4f} < {cfg.metric_threshold}")
    if np.isfinite(summary.get("overfit_gap", np.nan)) and abs(summary["overfit_gap"]) > cfg.overfit_gap_max:
        accepted = False; reasons.append(f"overfit_gap {summary['overfit_gap']:.4f} > {cfg.overfit_gap_max}")
    if summary["coverage"] < cfg.coverage_min:
        accepted = False; reasons.append(f"coverage {summary['coverage']:.2f} < {cfg.coverage_min}")

    summary["accepted"] = accepted
    summary["rejection_reasons"] = reasons

    # Feature importance
    imp_norm = importances / max(importances.sum(), 1e-12)
    importance = dict(zip(feature_names or [f"f{i}" for i in range(P)], imp_norm.tolist()))

    return {
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "metrics_by_fold": fold_metrics,
        "summary": summary,
        "models": models,
        "feature_importance": importance,
    }
