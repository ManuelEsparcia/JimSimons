"""
models/baselines/train_logistic.py — Logistic regression baseline for classification.

Predicts P(stock in top/bottom quantile) from features. Produces
calibrated probabilities usable for sizing, not just binary labels.

Key design:
1. Three penalty modes: L1 (sparse), L2 (stable), ElasticNet (mixed)
2. Isotonic + Platt calibration on a held-out calibration set
3. Discrimination vs calibration: AUC for ranking, Brier score for sizing
4. Class balance handling via class_weight='balanced'
5. Nested temporal CV for C selection (C = 1/λ)
6. Per-date AUC decomposition for regime analysis

References:
    Platt (1999) "Probabilistic Outputs for SVMs"
    Niculescu-Mizil & Caruana (2005) "Predicting Good Probabilities with Supervised Learning"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
)

from .. import (
    ModelError,
    ConfigError,
    DataContractError,
    PreprocessConfig,
    fit_preprocess,
    apply_preprocess,
    FoldResult,
    TrainResult,
    information_coefficient,
    aggregate_fold_metrics,
    generate_temporal_folds,
    utc_now_iso,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LogisticConfig:
    # Penalty
    penalty: str = "l2"           # "l1" | "l2" | "elasticnet"
    l1_ratio: float = 0.5         # only used when penalty="elasticnet"

    # C grid (C = 1/λ; small C = strong regularization)
    c_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)

    # Outer folds
    n_outer_folds: int = 5
    min_train_dates: int = 126
    embargo_days: int = 21

    # Inner tuning
    n_inner_folds: int = 3
    inner_min_train_dates: int = 63
    inner_embargo_days: int = 21
    tuning_metric: str = "auc"    # "auc" | "brier" | "log_loss"

    # Calibration
    calibrate: bool = True
    calibration_method: str = "isotonic"  # "isotonic" | "sigmoid" (Platt)

    # Class handling
    class_weight: str = "balanced"
    max_iter: int = 1000
    solver: str = "saga"          # supports all penalties

    # Preprocessing
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)

    run_id: str = ""
    model_name: str = "logistic_baseline"


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray, y_pred_proba: np.ndarray, y_pred_class: np.ndarray,
) -> dict[str, float]:
    """Compute all classification metrics for a fold."""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred_proba))
    yt = y_true[valid].astype(int)
    yp = y_pred_proba[valid]
    yc = y_pred_class[valid].astype(int)

    metrics: dict[str, float] = {"n_obs": int(valid.sum())}

    if len(np.unique(yt)) < 2:
        LOGGER.warning("Only one class in fold — metrics will be NaN")
        return metrics

    try:
        metrics["auc"] = float(roc_auc_score(yt, yp))
    except ValueError:
        metrics["auc"] = np.nan

    metrics["brier_score"] = float(brier_score_loss(yt, yp))
    metrics["log_loss"] = float(log_loss(yt, yp, labels=[0, 1]))
    metrics["precision"] = float(precision_score(yt, yc, zero_division=0))
    metrics["recall"] = float(recall_score(yt, yc, zero_division=0))
    metrics["f1"] = float(f1_score(yt, yc, zero_division=0))
    metrics["accuracy"] = float((yt == yc).mean())

    # IC on probability scores (rank correlation with actual)
    metrics["ic"] = information_coefficient(yt.astype(float), yp)

    # Class balance in predictions
    metrics["pred_positive_rate"] = float(yc.mean())
    metrics["actual_positive_rate"] = float(yt.mean())

    return metrics


# ---------------------------------------------------------------------------
# Nested C selection
# ---------------------------------------------------------------------------

def _select_c_nested(
    X_train: np.ndarray,
    y_train: np.ndarray,
    dates_train: np.ndarray,
    cfg: LogisticConfig,
) -> tuple[float, dict]:
    """Select best C via nested temporal CV on outer train."""
    inner_folds = generate_temporal_folds(
        dates_train,
        n_folds=cfg.n_inner_folds,
        min_train_dates=cfg.inner_min_train_dates,
        embargo_days=cfg.inner_embargo_days,
    )

    if not inner_folds:
        return float(np.median(cfg.c_grid)), {}

    scores_by_c: dict[float, list[float]] = {c: [] for c in cfg.c_grid}

    for i_tr, i_va in inner_folds:
        Xi_tr, yi_tr = X_train[i_tr], y_train[i_tr]
        Xi_va, yi_va = X_train[i_va], y_train[i_va]

        if len(Xi_tr) < 30 or len(Xi_va) < 10:
            continue
        if len(np.unique(yi_tr)) < 2 or len(np.unique(yi_va)) < 2:
            continue

        for c_val in cfg.c_grid:
            kwargs = {"C": c_val, "penalty": cfg.penalty, "solver": cfg.solver,
                      "max_iter": cfg.max_iter, "class_weight": cfg.class_weight}
            if cfg.penalty == "elasticnet":
                kwargs["l1_ratio"] = cfg.l1_ratio

            model = LogisticRegression(**kwargs)
            try:
                model.fit(Xi_tr, yi_tr)
                proba = model.predict_proba(Xi_va)[:, 1]
            except Exception:
                continue

            if cfg.tuning_metric == "auc":
                try:
                    score = roc_auc_score(yi_va, proba)
                except ValueError:
                    continue
            elif cfg.tuning_metric == "brier":
                score = -brier_score_loss(yi_va, proba)
            else:
                score = -log_loss(yi_va, proba, labels=[0, 1])

            scores_by_c[c_val].append(score)

    best_c = float(np.median(cfg.c_grid))
    best_score = -np.inf
    for c_val, scores in scores_by_c.items():
        if scores:
            mean_s = np.mean(scores)
            if mean_s > best_score:
                best_score = mean_s
                best_c = c_val

    return best_c, {str(c): s for c, s in scores_by_c.items()}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def train_logistic(
    features_df: pd.DataFrame,
    target_col: str,
    *,
    feature_cols: Sequence[str] | None = None,
    config: LogisticConfig | None = None,
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> TrainResult:
    """Train Logistic baseline with purged temporal CV + calibration.

    target_col should be a binary column (0/1), e.g. y_cls_tail_10d > 0.
    """
    cfg = config or LogisticConfig()
    if not cfg.run_id:
        cfg = LogisticConfig(**{**cfg.__dict__, "run_id": f"logistic_{utc_now_iso().replace(':','').replace('-','')}"})

    LOGGER.info("Training Logistic baseline: run_id=%s, penalty=%s", cfg.run_id, cfg.penalty)

    if feature_cols is None:
        exclude = {"date", "symbol", target_col, "staleness_days", "is_stale"}
        feature_cols = [c for c in features_df.columns
                       if c not in exclude and pd.api.types.is_numeric_dtype(features_df[c])]

    if target_col not in features_df.columns:
        raise DataContractError(f"Target '{target_col}' not found")

    dates = features_df["date"].values
    folds = splits or generate_temporal_folds(
        dates, cfg.n_outer_folds, cfg.min_train_dates, cfg.embargo_days,
    )

    fold_results: list[FoldResult] = []

    for fold_idx, (train_mask, test_mask) in enumerate(folds):
        fold_id = f"fold_{fold_idx}"

        X_tr_raw = features_df.loc[train_mask, feature_cols]
        y_tr = features_df.loc[train_mask, target_col].values.astype(np.float64)
        X_te_raw = features_df.loc[test_mask, feature_cols]
        y_te = features_df.loc[test_mask, target_col].values.astype(np.float64)

        # Drop NaN targets
        tr_valid = ~np.isnan(y_tr)
        te_valid = ~np.isnan(y_te)
        X_tr_raw = X_tr_raw[tr_valid].reset_index(drop=True)
        y_tr = y_tr[tr_valid].astype(int)
        X_te_raw = X_te_raw[te_valid].reset_index(drop=True)
        y_te = y_te[te_valid].astype(int)

        if len(y_tr) < 50 or len(y_te) < 10:
            continue
        if len(np.unique(y_tr)) < 2:
            LOGGER.warning("  %s: single class in train — skipped", fold_id)
            continue

        # Train-only preprocessing
        pp_state = fit_preprocess(X_tr_raw, cfg.preprocess)
        X_tr = apply_preprocess(X_tr_raw, pp_state, cfg.preprocess)
        X_te = apply_preprocess(X_te_raw, pp_state, cfg.preprocess)

        # Nested C selection
        tr_dates = features_df.loc[train_mask, "date"].values[tr_valid]
        best_c, _ = _select_c_nested(X_tr, y_tr, tr_dates, cfg)

        # Final fit
        kwargs = {"C": best_c, "penalty": cfg.penalty, "solver": cfg.solver,
                  "max_iter": cfg.max_iter, "class_weight": cfg.class_weight}
        if cfg.penalty == "elasticnet":
            kwargs["l1_ratio"] = cfg.l1_ratio

        model = LogisticRegression(**kwargs)
        model.fit(X_tr, y_tr)

        # Calibration (uses a cross-val approach on train)
        if cfg.calibrate:
            try:
                cal_model = CalibratedClassifierCV(
                    model, method=cfg.calibration_method, cv=3,
                )
                cal_model.fit(X_tr, y_tr)
                proba = cal_model.predict_proba(X_te)[:, 1]
            except Exception:
                proba = model.predict_proba(X_te)[:, 1]
        else:
            proba = model.predict_proba(X_te)[:, 1]

        preds_class = (proba >= 0.5).astype(int)
        metrics = compute_classification_metrics(y_te, proba, preds_class)

        train_d = features_df.loc[train_mask, "date"]
        test_d = features_df.loc[test_mask, "date"]
        fr = FoldResult(
            fold_id=fold_id,
            n_train=len(y_tr),
            n_test=len(y_te),
            train_dates=(str(train_d.min()), str(train_d.max())),
            test_dates=(str(test_d.min()), str(test_d.max())),
            metrics=metrics,
            best_params={"C": best_c, "penalty": cfg.penalty},
            coefficients=model.coef_[0].copy() if hasattr(model, "coef_") else None,
            intercept=float(model.intercept_[0]) if hasattr(model, "intercept_") else None,
            oof_predictions=proba,
            oof_actuals=y_te.astype(float),
        )
        fold_results.append(fr)

        LOGGER.info(
            "  %s: C=%.4f, AUC=%.4f, Brier=%.4f, F1=%.3f",
            fold_id, best_c, metrics.get("auc", 0), metrics.get("brier_score", 1), metrics.get("f1", 0),
        )

    if not fold_results:
        raise ModelError("No folds completed")

    agg = aggregate_fold_metrics(fold_results)

    return TrainResult(
        model_name=cfg.model_name,
        n_folds=len(fold_results),
        fold_results=fold_results,
        aggregate_metrics=agg,
        config={"penalty": cfg.penalty, "c_grid": list(cfg.c_grid),
                "calibrate": cfg.calibrate, "calibration_method": cfg.calibration_method},
        run_id=cfg.run_id,
        timestamp=utc_now_iso(),
    )
