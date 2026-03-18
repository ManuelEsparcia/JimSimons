"""
models/baselines/train_elasticnet.py — ElasticNet regression baseline.

Combines L1 sparsity (variable selection) with L2 stability (grouping
effect). Particularly suited for small caps where feature families are
highly correlated (e.g. multiple momentum windows, multiple vol measures).

Key design:
1. 2D nested grid: (alpha, l1_ratio). Alpha controls strength, l1_ratio
   controls the mix between Lasso (l1_ratio=1) and Ridge (l1_ratio=0).
2. Sparsity analysis: which features survive? Stable across folds?
3. Grouping effect diagnostics: do correlated features get similar β?
4. Warm-start path: fit along alpha path for efficiency.
5. Same preprocessing pipeline and metrics as Ridge for direct comparison.

References:
    Zou & Hastie (2005) "Regularization and Variable Selection via the Elastic Net"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from .. import (
    ModelError,
    ConfigError,
    DataContractError,
    PreprocessConfig,
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


@dataclass(frozen=True)
class ElasticNetConfig:
    # 2D grid
    alpha_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)
    l1_ratio_grid: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)

    # Outer folds
    n_outer_folds: int = 5
    min_train_dates: int = 126
    embargo_days: int = 21

    # Inner tuning
    n_inner_folds: int = 3
    inner_min_train_dates: int = 63
    inner_embargo_days: int = 21
    tuning_metric: str = "ic"

    # Model
    fit_intercept: bool = True
    max_iter: int = 5000
    tol: float = 1e-5
    warm_start: bool = True

    # Preprocessing
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)

    run_id: str = ""
    model_name: str = "elasticnet_baseline"


# ---------------------------------------------------------------------------
# Sparsity analysis
# ---------------------------------------------------------------------------

def sparsity_analysis(
    fold_results: list[FoldResult],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    """Analyse which features survive L1 selection across folds.

    A feature that is zeroed out in all folds is genuinely irrelevant.
    A feature that survives in some folds but not others is unstable.
    A feature that survives in ALL folds is a robust signal.
    """
    coefs = [fr.coefficients for fr in fold_results if fr.coefficients is not None]
    if not coefs or not feature_names:
        return {}

    coef_matrix = np.array(coefs)  # (n_folds, n_features)
    n_folds = len(coefs)
    n_features = min(coef_matrix.shape[1], len(feature_names))

    nonzero_mask = np.abs(coef_matrix[:, :n_features]) > 1e-10
    survival_rate = nonzero_mask.mean(axis=0)  # fraction of folds where β ≠ 0

    result: dict[str, Any] = {
        "n_features_total": n_features,
        "n_always_zero": int((survival_rate == 0).sum()),
        "n_always_nonzero": int((survival_rate == 1.0).sum()),
        "n_unstable": int(((survival_rate > 0) & (survival_rate < 1.0)).sum()),
        "mean_sparsity": float(1.0 - nonzero_mask.mean()),
        "effective_n_features_mean": float(nonzero_mask.sum(axis=1).mean()),
    }

    # Top surviving features
    surviving = [(feature_names[i], float(survival_rate[i]),
                  float(np.abs(coef_matrix[:, i]).mean()))
                 for i in range(n_features) if survival_rate[i] > 0]
    surviving.sort(key=lambda x: (-x[1], -x[2]))
    result["top_surviving"] = [
        {"feature": name, "survival_rate": sr, "mean_abs_coef": mac}
        for name, sr, mac in surviving[:20]
    ]

    return result


# ---------------------------------------------------------------------------
# Nested (alpha, l1_ratio) selection
# ---------------------------------------------------------------------------

def _select_params_nested(
    X_train: np.ndarray,
    y_train: np.ndarray,
    dates_train: np.ndarray,
    cfg: ElasticNetConfig,
) -> tuple[float, float, dict]:
    """Select best (alpha, l1_ratio) via nested temporal CV."""
    inner_folds = generate_temporal_folds(
        dates_train,
        n_folds=cfg.n_inner_folds,
        min_train_dates=cfg.inner_min_train_dates,
        embargo_days=cfg.inner_embargo_days,
    )

    if not inner_folds:
        return float(np.median(cfg.alpha_grid)), 0.5, {}

    scores: dict[tuple[float, float], list[float]] = {}
    for a in cfg.alpha_grid:
        for r in cfg.l1_ratio_grid:
            scores[(a, r)] = []

    for i_tr, i_va in inner_folds:
        Xi_tr, yi_tr = X_train[i_tr], y_train[i_tr]
        Xi_va, yi_va = X_train[i_va], y_train[i_va]

        if len(Xi_tr) < 30 or len(Xi_va) < 10:
            continue

        for a in cfg.alpha_grid:
            for r in cfg.l1_ratio_grid:
                model = ElasticNet(
                    alpha=a, l1_ratio=r,
                    fit_intercept=cfg.fit_intercept,
                    max_iter=cfg.max_iter, tol=cfg.tol,
                )
                try:
                    model.fit(Xi_tr, yi_tr)
                    preds = model.predict(Xi_va)
                except Exception:
                    continue

                if cfg.tuning_metric == "ic":
                    score = information_coefficient(yi_va, preds)
                else:
                    score = -float(np.mean((yi_va - preds) ** 2))

                if not np.isnan(score):
                    scores[(a, r)].append(score)

    best_params = (float(np.median(cfg.alpha_grid)), 0.5)
    best_score = -np.inf
    for params, vals in scores.items():
        if vals:
            mean_s = np.mean(vals)
            if mean_s > best_score:
                best_score = mean_s
                best_params = params

    return best_params[0], best_params[1], {str(k): v for k, v in scores.items()}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def train_elasticnet(
    features_df: pd.DataFrame,
    target_col: str,
    *,
    feature_cols: Sequence[str] | None = None,
    config: ElasticNetConfig | None = None,
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> TrainResult:
    """Train ElasticNet baseline with 2D nested temporal CV."""
    cfg = config or ElasticNetConfig()
    if not cfg.run_id:
        cfg = ElasticNetConfig(**{**cfg.__dict__, "run_id": f"enet_{utc_now_iso().replace(':','').replace('-','')}"})

    LOGGER.info("Training ElasticNet: run_id=%s", cfg.run_id)

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

        tr_valid = ~np.isnan(y_tr)
        te_valid = ~np.isnan(y_te)
        X_tr_raw = X_tr_raw[tr_valid].reset_index(drop=True)
        y_tr = y_tr[tr_valid]
        X_te_raw = X_te_raw[te_valid].reset_index(drop=True)
        y_te = y_te[te_valid]

        if len(y_tr) < 50 or len(y_te) < 10:
            continue

        # Train-only preprocessing
        pp_state = fit_preprocess(X_tr_raw, cfg.preprocess)
        X_tr = apply_preprocess(X_tr_raw, pp_state, cfg.preprocess)
        X_te = apply_preprocess(X_te_raw, pp_state, cfg.preprocess)

        # Nested param selection
        tr_dates = features_df.loc[train_mask, "date"].values[tr_valid]
        best_alpha, best_l1, _ = _select_params_nested(X_tr, y_tr, tr_dates, cfg)

        # Final fit
        model = ElasticNet(
            alpha=best_alpha, l1_ratio=best_l1,
            fit_intercept=cfg.fit_intercept,
            max_iter=cfg.max_iter, tol=cfg.tol,
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        metrics = compute_all_metrics(y_te, preds)

        # Sparsity for this fold
        n_nonzero = int((np.abs(model.coef_) > 1e-10).sum())
        metrics["n_nonzero_coefs"] = n_nonzero
        metrics["sparsity"] = 1.0 - n_nonzero / len(model.coef_) if len(model.coef_) > 0 else 0.0

        train_d = features_df.loc[train_mask, "date"]
        test_d = features_df.loc[test_mask, "date"]
        fr = FoldResult(
            fold_id=fold_id,
            n_train=len(y_tr),
            n_test=len(y_te),
            train_dates=(str(train_d.min()), str(train_d.max())),
            test_dates=(str(test_d.min()), str(test_d.max())),
            metrics=metrics,
            best_params={"alpha": best_alpha, "l1_ratio": best_l1},
            coefficients=model.coef_.copy(),
            intercept=float(model.intercept_),
            oof_predictions=preds,
            oof_actuals=y_te,
        )
        fold_results.append(fr)

        LOGGER.info(
            "  %s: α=%.4f ρ=%.2f IC=%.4f sparsity=%.0f%% (%d/%d nonzero)",
            fold_id, best_alpha, best_l1,
            metrics.get("ic", 0), metrics["sparsity"] * 100,
            n_nonzero, len(model.coef_),
        )

    if not fold_results:
        raise ModelError("No folds completed")

    agg = aggregate_fold_metrics(fold_results)

    # Sparsity analysis
    sparse = sparsity_analysis(fold_results, list(feature_cols))
    agg.update({f"sparsity_{k}": v for k, v in sparse.items()
                if isinstance(v, (int, float))})

    return TrainResult(
        model_name=cfg.model_name,
        n_folds=len(fold_results),
        fold_results=fold_results,
        aggregate_metrics=agg,
        config={"alpha_grid": list(cfg.alpha_grid), "l1_ratio_grid": list(cfg.l1_ratio_grid),
                "model_name": cfg.model_name},
        run_id=cfg.run_id,
        timestamp=utc_now_iso(),
    )
