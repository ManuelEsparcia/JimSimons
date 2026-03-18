"""
models/baselines/train_ridge.py — Institutional Ridge regression baseline.

The baseline that every other model must beat. If Ridge with proper
preprocessing and temporal CV achieves IC 0.03, then LightGBM achieving
IC 0.04 only justifies its complexity cost if that improvement is
STABLE across folds (high IC IR), not a lucky draw.

Key design decisions:
1. Nested temporal CV: outer folds for evaluation, inner for alpha tuning.
   Alpha never sees the outer test fold.
2. Train-only preprocessing: impute → winsor → standardize, all fitted
   on train partition only. Applied identically to test.
3. IC/IR as primary metrics, not MSE. We're ranking stocks, not
   minimising squared error.
4. Coefficient stability analysis: std(β) across folds. Unstable
   coefficients signal overfitting even when IC looks good.
5. Per-date IC decomposition: IC_t for each test date, not just
   aggregate. Reveals regime dependence.

References:
    Hoerl & Kennard (1970) "Ridge Regression: Biased Estimation"
    Hastie, Tibshirani, Friedman (2009) "Elements of Statistical Learning" Ch.3
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .. import (
    ModelError,
    ConfigError,
    DataContractError,
    PreprocessConfig,
    PreprocessState,
    fit_preprocess,
    apply_preprocess,
    FoldResult,
    TrainResult,
    compute_all_metrics,
    aggregate_fold_metrics,
    information_coefficient,
    generate_temporal_folds,
    utc_now_iso,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RidgeConfig:
    """Configuration for Ridge baseline training."""
    # Alpha grid (log scale)
    alpha_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0)

    # Outer evaluation
    n_outer_folds: int = 5
    min_train_dates: int = 126    # ~6 months
    embargo_days: int = 21        # ~1 month

    # Inner tuning (nested within outer train)
    n_inner_folds: int = 3
    inner_min_train_dates: int = 63
    inner_embargo_days: int = 21
    tuning_metric: str = "ic"     # "ic" | "mse" | "long_short_spread_20"

    # Preprocessing
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)

    # Model
    fit_intercept: bool = True

    # Run metadata
    run_id: str = ""
    model_name: str = "ridge_baseline"


# ---------------------------------------------------------------------------
# Inner CV: alpha selection (sees ONLY outer train data)
# ---------------------------------------------------------------------------

def _select_alpha_nested(
    X_train: np.ndarray,
    y_train: np.ndarray,
    dates_train: np.ndarray,
    alpha_grid: Sequence[float],
    cfg: RidgeConfig,
) -> tuple[float, dict[str, list[float]]]:
    """Select best alpha using nested temporal CV on the outer train set.

    CRITICAL: This function NEVER sees the outer test fold. It splits
    the outer train into inner train/valid using temporal walk-forward.

    Returns (best_alpha, {alpha: [scores_per_inner_fold]}).
    """
    inner_folds = generate_temporal_folds(
        dates_train,
        n_folds=cfg.n_inner_folds,
        min_train_dates=cfg.inner_min_train_dates,
        embargo_days=cfg.inner_embargo_days,
    )

    if not inner_folds:
        LOGGER.warning("No valid inner folds — using median alpha")
        return float(np.median(alpha_grid)), {}

    scores_by_alpha: dict[float, list[float]] = {a: [] for a in alpha_grid}

    for inner_train_mask, inner_valid_mask in inner_folds:
        Xi_tr = X_train[inner_train_mask]
        yi_tr = y_train[inner_train_mask]
        Xi_va = X_train[inner_valid_mask]
        yi_va = y_train[inner_valid_mask]

        if len(Xi_tr) < 20 or len(Xi_va) < 10:
            continue

        for alpha in alpha_grid:
            model = Ridge(alpha=alpha, fit_intercept=cfg.fit_intercept)
            model.fit(Xi_tr, yi_tr)
            preds = model.predict(Xi_va)

            if cfg.tuning_metric == "ic":
                score = information_coefficient(yi_va, preds)
            elif cfg.tuning_metric == "mse":
                score = -float(np.mean((yi_va - preds) ** 2))  # negative MSE
            else:
                score = information_coefficient(yi_va, preds)

            if not np.isnan(score):
                scores_by_alpha[alpha].append(score)

    # Select alpha with best mean score
    best_alpha = float(np.median(alpha_grid))
    best_score = -np.inf

    for alpha, scores in scores_by_alpha.items():
        if scores:
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha

    return best_alpha, {str(a): s for a, s in scores_by_alpha.items()}


# ---------------------------------------------------------------------------
# Per-date IC decomposition
# ---------------------------------------------------------------------------

def compute_per_date_ic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: np.ndarray,
) -> pd.Series:
    """Compute IC for each unique date in the test set.

    Returns a Series indexed by date with IC values.
    This reveals regime dependence: does the model work uniformly
    or only in certain market conditions?
    """
    df = pd.DataFrame({"y": y_true, "pred": y_pred, "date": dates})
    ic_by_date = df.groupby("date").apply(
        lambda g: information_coefficient(g["y"].values, g["pred"].values)
        if len(g) >= 10 else np.nan
    )
    return ic_by_date


# ---------------------------------------------------------------------------
# Coefficient stability analysis
# ---------------------------------------------------------------------------

def coefficient_stability(
    fold_results: list[FoldResult],
) -> dict[str, float]:
    """Analyse stability of coefficients across folds.

    Unstable coefficients (high std relative to mean) signal overfitting
    even when IC looks good. A truly robust signal has consistent β
    across temporal folds.
    """
    coefs = [fr.coefficients for fr in fold_results if fr.coefficients is not None]
    if len(coefs) < 2:
        return {}

    coef_matrix = np.array(coefs)  # (n_folds, n_features)
    mean_abs = np.abs(coef_matrix).mean(axis=0)
    std_coef = coef_matrix.std(axis=0)

    # Coefficient variation: std / mean_abs (high = unstable)
    cv = std_coef / (mean_abs + 1e-10)

    return {
        "coef_mean_abs": float(mean_abs.mean()),
        "coef_std_mean": float(std_coef.mean()),
        "coef_cv_mean": float(cv.mean()),
        "coef_cv_max": float(cv.max()),
        "n_unstable_coefs": int((cv > 2.0).sum()),
        "pct_sign_stable": float((np.sign(coef_matrix).std(axis=0) < 0.5).mean()),
    }


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def train_ridge(
    features_df: pd.DataFrame,
    target_col: str,
    *,
    feature_cols: Sequence[str] | None = None,
    config: RidgeConfig | None = None,
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> TrainResult:
    """Train Ridge baseline with purged temporal CV.

    Parameters
    ----------
    features_df : DataFrame
        Must have 'date', 'symbol', feature columns, and target_col.
    target_col : str
        Column name of the prediction target (e.g. 'y_fwd_ret_net_10d').
    feature_cols : list of str, optional
        If None, uses all numeric columns except date/symbol/target.
    config : RidgeConfig, optional
    splits : list of (train_mask, test_mask) tuples, optional
        If provided, uses these instead of generating walk-forward folds.
        These should come from labels/purged_splits for consistency.

    Returns
    -------
    TrainResult with fold-level and aggregate results.
    """
    cfg = config or RidgeConfig()
    if not cfg.run_id:
        cfg = RidgeConfig(**{**cfg.__dict__, "run_id": f"ridge_{utc_now_iso().replace(':','').replace('-','')}"})

    LOGGER.info("Training Ridge baseline: run_id=%s", cfg.run_id)

    # Resolve feature columns
    if feature_cols is None:
        exclude = {"date", "symbol", target_col, "staleness_days", "is_stale"}
        feature_cols = [c for c in features_df.columns
                       if c not in exclude and pd.api.types.is_numeric_dtype(features_df[c])]

    if not feature_cols:
        raise ConfigError("No feature columns found")

    # Validate target
    if target_col not in features_df.columns:
        raise DataContractError(f"Target column '{target_col}' not found")

    # Get dates for fold generation
    dates = features_df["date"].values

    # Generate or use provided splits
    if splits is None:
        folds = generate_temporal_folds(
            dates,
            n_folds=cfg.n_outer_folds,
            min_train_dates=cfg.min_train_dates,
            embargo_days=cfg.embargo_days,
        )
    else:
        folds = splits

    if not folds:
        raise ModelError("No valid temporal folds generated")

    # --- Train across folds ---
    fold_results: list[FoldResult] = []
    all_oof_preds = []
    all_oof_actuals = []
    all_oof_dates = []
    all_oof_symbols = []

    for fold_idx, (train_mask, test_mask) in enumerate(folds):
        fold_id = f"fold_{fold_idx}"
        LOGGER.info("  %s: train=%d, test=%d", fold_id, train_mask.sum(), test_mask.sum())

        # Extract data
        X_train_raw = features_df.loc[train_mask, feature_cols]
        y_train = features_df.loc[train_mask, target_col].values.astype(np.float64)
        X_test_raw = features_df.loc[test_mask, feature_cols]
        y_test = features_df.loc[test_mask, target_col].values.astype(np.float64)

        test_dates = features_df.loc[test_mask, "date"].values
        test_symbols = features_df.loc[test_mask, "symbol"].values

        # Drop rows where target is NaN
        train_valid = ~np.isnan(y_train)
        test_valid = ~np.isnan(y_test)
        X_train_raw = X_train_raw[train_valid].reset_index(drop=True)
        y_train = y_train[train_valid]
        X_test_raw = X_test_raw[test_valid].reset_index(drop=True)
        y_test = y_test[test_valid]
        test_dates = test_dates[test_valid]
        test_symbols = test_symbols[test_valid]

        if len(y_train) < 50 or len(y_test) < 10:
            LOGGER.warning("  %s: skipped (too few observations)", fold_id)
            continue

        # FIT preprocessing on TRAIN ONLY
        pp_state = fit_preprocess(X_train_raw, cfg.preprocess)
        X_train = apply_preprocess(X_train_raw, pp_state, cfg.preprocess)
        X_test = apply_preprocess(X_test_raw, pp_state, cfg.preprocess)

        # Nested alpha selection (uses ONLY train data)
        train_dates_for_tuning = features_df.loc[train_mask, "date"].values[train_valid]
        best_alpha, alpha_scores = _select_alpha_nested(
            X_train, y_train, train_dates_for_tuning, cfg.alpha_grid, cfg,
        )

        # Final fit with best alpha on FULL outer train
        model = Ridge(alpha=best_alpha, fit_intercept=cfg.fit_intercept)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Metrics
        metrics = compute_all_metrics(y_test, preds)

        # Per-date IC
        per_date_ic = compute_per_date_ic(y_test, preds, test_dates)
        metrics["ic_per_date_mean"] = float(per_date_ic.mean())
        metrics["ic_per_date_std"] = float(per_date_ic.std())
        metrics["ic_per_date_positive_frac"] = float((per_date_ic > 0).mean())

        # Store fold result
        train_d = features_df.loc[train_mask, "date"]
        test_d = features_df.loc[test_mask, "date"]
        fr = FoldResult(
            fold_id=fold_id,
            n_train=len(y_train),
            n_test=len(y_test),
            train_dates=(str(train_d.min()), str(train_d.max())),
            test_dates=(str(test_d.min()), str(test_d.max())),
            metrics=metrics,
            best_params={"alpha": best_alpha},
            coefficients=model.coef_.copy(),
            intercept=float(model.intercept_),
            oof_predictions=preds,
            oof_actuals=y_test,
            oof_dates=test_dates,
            oof_symbols=test_symbols,
            preprocess_state=pp_state,
        )
        fold_results.append(fr)

        all_oof_preds.append(preds)
        all_oof_actuals.append(y_test)
        all_oof_dates.append(test_dates)
        all_oof_symbols.append(test_symbols)

        LOGGER.info(
            "  %s: alpha=%.4f, IC=%.4f, hit=%.3f, LS20=%.4f",
            fold_id, best_alpha, metrics.get("ic", 0), metrics.get("hit_ratio", 0),
            metrics.get("long_short_spread_20", 0),
        )

    if not fold_results:
        raise ModelError("No folds completed successfully")

    # --- Aggregate results ---
    agg = aggregate_fold_metrics(fold_results)

    # Coefficient stability
    coef_stability = coefficient_stability(fold_results)
    agg.update(coef_stability)

    # Overall OOF IC
    if all_oof_preds:
        oof_p = np.concatenate(all_oof_preds)
        oof_a = np.concatenate(all_oof_actuals)
        agg["oof_ic_overall"] = information_coefficient(oof_a, oof_p)

    result = TrainResult(
        model_name=cfg.model_name,
        n_folds=len(fold_results),
        fold_results=fold_results,
        aggregate_metrics=agg,
        config={"alpha_grid": list(cfg.alpha_grid), "model_name": cfg.model_name,
                "n_outer_folds": cfg.n_outer_folds, "embargo_days": cfg.embargo_days,
                "tuning_metric": cfg.tuning_metric},
        run_id=cfg.run_id,
        timestamp=utc_now_iso(),
    )

    LOGGER.info(
        "Ridge baseline complete: %d folds, IC_mean=%.4f, IC_IR=%.2f, coef_cv=%.2f",
        len(fold_results),
        agg.get("ic_mean", 0), agg.get("ic_ir", 0), agg.get("coef_cv_mean", 0),
    )

    return result
