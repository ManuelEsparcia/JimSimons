"""
models — Model training, inference, and evaluation infrastructure.

Subpackages: baselines/, gbdt/, ensemble/, regimes/, inference/

Shared infrastructure lives here: train-only preprocessing pipeline,
purged fold management, standardised metrics, artifact persistence.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

class ModelError(RuntimeError):
    """Base error for the models package."""

class ConfigError(ModelError, ValueError):
    """Invalid model configuration."""

class DataContractError(ModelError):
    """Input data violates contract."""

class LeakageError(ModelError):
    """Temporal leakage detected in preprocessing or tuning."""


# ---------------------------------------------------------------------------
# Standardised metrics (what a PhD quant actually measures)
# ---------------------------------------------------------------------------

def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """IC = Spearman rank correlation between predictions and actuals.

    This is THE metric for cross-sectional equity models. Not MSE.
    Not R². IC tells you how well the model ranks stocks, which is
    what matters for portfolio construction.
    """
    from scipy.stats import spearmanr
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 10:
        return np.nan
    corr, _ = spearmanr(y_true[valid], y_pred[valid])
    return float(corr)


def information_ratio_of_ic(ic_series: np.ndarray) -> float:
    """IR = mean(IC) / std(IC).

    Measures stability of the signal. An IC of 0.05 with IR of 1.0 is
    much more valuable than IC of 0.10 with IR of 0.3.
    """
    valid = ic_series[~np.isnan(ic_series)]
    if len(valid) < 5:
        return np.nan
    mu = valid.mean()
    sigma = valid.std()
    return float(mu / sigma) if sigma > 1e-10 else np.nan


def hit_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction where sign(prediction) == sign(actual)."""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 10:
        return np.nan
    correct = np.sign(y_true[valid]) == np.sign(y_pred[valid])
    return float(correct.mean())


def long_short_spread(
    y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.2,
) -> float:
    """Mean return of top quantile minus bottom quantile by prediction.

    This is the backtest-free alpha proxy: if the model can separate
    the top 20% from the bottom 20%, it has economic value.
    """
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 20:
        return np.nan
    yt, yp = y_true[valid], y_pred[valid]
    n = len(yt)
    sorted_idx = np.argsort(yp)
    k = max(1, int(n * quantile))
    long_ret = yt[sorted_idx[-k:]].mean()
    short_ret = yt[sorted_idx[:k]].mean()
    return float(long_ret - short_ret)


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute all standard metrics for a fold."""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[valid] if valid.any() else y_true
    yp = y_pred[valid] if valid.any() else y_pred
    n = int(valid.sum())

    metrics = {
        "n_obs": n,
        "ic": information_coefficient(y_true, y_pred),
        "hit_ratio": hit_ratio(y_true, y_pred),
        "long_short_spread_20": long_short_spread(y_true, y_pred, 0.2),
        "long_short_spread_10": long_short_spread(y_true, y_pred, 0.1),
    }

    if n > 1:
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        metrics["mse"] = float(mean_squared_error(yt, yp))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics["mae"] = float(mean_absolute_error(yt, yp))

        ss_res = ((yt - yp) ** 2).sum()
        ss_tot = ((yt - yt.mean()) ** 2).sum()
        metrics["r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return metrics


# ---------------------------------------------------------------------------
# Train-only preprocessing pipeline (CRITICAL: no test leakage)
# ---------------------------------------------------------------------------

@dataclass
class PreprocessState:
    """Fitted preprocessing state from training data only.

    Stores all statistics learned from train so they can be applied
    to test without leakage.
    """
    impute_values: dict[str, float] = field(default_factory=dict)
    clip_lower: dict[str, float] = field(default_factory=dict)
    clip_upper: dict[str, float] = field(default_factory=dict)
    scale_mean: dict[str, float] = field(default_factory=dict)
    scale_std: dict[str, float] = field(default_factory=dict)
    kept_columns: list[str] = field(default_factory=list)
    n_train_rows: int = 0


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for the preprocessing pipeline."""
    impute_strategy: str = "median"     # "median" | "zero" | "drop"
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    standardize: bool = True
    drop_high_null_cols: bool = True
    max_null_fraction: float = 0.50
    scale_epsilon: float = 1e-8


def fit_preprocess(
    X_train: pd.DataFrame,
    config: PreprocessConfig | None = None,
) -> PreprocessState:
    """Fit preprocessing on TRAINING data only. No test data touches this."""
    cfg = config or PreprocessConfig()
    state = PreprocessState()
    state.n_train_rows = len(X_train)

    cols = list(X_train.columns)

    # Drop columns with excessive nulls
    if cfg.drop_high_null_cols:
        null_frac = X_train.isna().mean()
        cols = [c for c in cols if null_frac.get(c, 0) <= cfg.max_null_fraction]

    # Imputation values (train-only)
    for col in cols:
        if cfg.impute_strategy == "median":
            state.impute_values[col] = float(X_train[col].median())
        elif cfg.impute_strategy == "zero":
            state.impute_values[col] = 0.0
        else:
            state.impute_values[col] = float(X_train[col].median())

    # Winsorization bounds (train-only)
    for col in cols:
        valid = X_train[col].dropna()
        if len(valid) > 10:
            state.clip_lower[col] = float(valid.quantile(cfg.winsor_lower))
            state.clip_upper[col] = float(valid.quantile(cfg.winsor_upper))

    # Standardization params (train-only)
    if cfg.standardize:
        for col in cols:
            valid = X_train[col].dropna()
            state.scale_mean[col] = float(valid.mean()) if len(valid) > 0 else 0.0
            state.scale_std[col] = float(valid.std()) if len(valid) > 1 else 1.0

    state.kept_columns = cols
    return state


def apply_preprocess(
    X: pd.DataFrame,
    state: PreprocessState,
    config: PreprocessConfig | None = None,
) -> np.ndarray:
    """Apply fitted preprocessing to ANY data (train or test).

    Returns a numpy array with shape (n_rows, n_kept_columns).
    """
    cfg = config or PreprocessConfig()
    out = X[state.kept_columns].copy()

    # Impute
    for col in state.kept_columns:
        if col in state.impute_values:
            out[col] = out[col].fillna(state.impute_values[col])

    # Winsorize
    for col in state.kept_columns:
        if col in state.clip_lower and col in state.clip_upper:
            out[col] = out[col].clip(
                lower=state.clip_lower[col],
                upper=state.clip_upper[col],
            )

    # Standardize
    if cfg.standardize:
        for col in state.kept_columns:
            mu = state.scale_mean.get(col, 0.0)
            sigma = state.scale_std.get(col, 1.0)
            out[col] = (out[col] - mu) / (sigma + cfg.scale_epsilon)

    # Final NaN/Inf cleanup
    arr = out.values.astype(np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


# ---------------------------------------------------------------------------
# Fold results
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Results for a single temporal fold."""
    fold_id: str
    n_train: int
    n_test: int
    train_dates: tuple[str, str]  # (first, last)
    test_dates: tuple[str, str]
    metrics: dict[str, float]
    best_params: dict[str, Any]
    coefficients: np.ndarray | None = None
    intercept: float | None = None
    oof_predictions: np.ndarray | None = None
    oof_actuals: np.ndarray | None = None
    oof_dates: np.ndarray | None = None
    oof_symbols: np.ndarray | None = None
    preprocess_state: PreprocessState | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "fold_id": self.fold_id,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_dates": self.train_dates,
            "test_dates": self.test_dates,
            "metrics": self.metrics,
            "best_params": {k: _safe(v) for k, v in self.best_params.items()},
        }
        if self.intercept is not None:
            d["intercept"] = self.intercept
        return d


@dataclass
class TrainResult:
    """Aggregate result of training across all folds."""
    model_name: str
    n_folds: int
    fold_results: list[FoldResult]
    aggregate_metrics: dict[str, float]
    config: dict[str, Any]
    run_id: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "n_folds": self.n_folds,
            "aggregate_metrics": self.aggregate_metrics,
            "config": {k: _safe(v) for k, v in self.config.items()},
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "folds": [fr.to_dict() for fr in self.fold_results],
        }


def _safe(v):
    """JSON-safe conversion."""
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return None if np.isnan(v) else float(v)
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (tuple, list)): return [_safe(x) for x in v]
    return v


def aggregate_fold_metrics(folds: list[FoldResult]) -> dict[str, float]:
    """Aggregate metrics across folds."""
    if not folds:
        return {}
    all_keys = set()
    for f in folds:
        all_keys.update(f.metrics.keys())

    agg = {}
    for key in sorted(all_keys):
        vals = [f.metrics[key] for f in folds if key in f.metrics and not np.isnan(f.metrics[key])]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
            agg[f"{key}_median"] = float(np.median(vals))

    # IC Information Ratio
    ic_vals = np.array([f.metrics.get("ic", np.nan) for f in folds])
    agg["ic_ir"] = information_ratio_of_ic(ic_vals)

    return agg


# ---------------------------------------------------------------------------
# Temporal fold generation (simple walk-forward if splits not provided)
# ---------------------------------------------------------------------------

def generate_temporal_folds(
    dates: np.ndarray,
    n_folds: int = 5,
    min_train_dates: int = 126,
    embargo_days: int = 21,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate walk-forward temporal folds with embargo.

    Returns list of (train_date_mask, test_date_mask) boolean arrays.
    """
    unique_dates = np.sort(np.unique(dates))
    n_dates = len(unique_dates)
    test_size = max(1, (n_dates - min_train_dates) // n_folds)

    folds = []
    for i in range(n_folds):
        test_start_idx = min_train_dates + i * test_size
        test_end_idx = min(test_start_idx + test_size, n_dates)
        if test_start_idx >= n_dates:
            break

        test_dates_set = set(unique_dates[test_start_idx:test_end_idx])
        # Train: all dates before test_start - embargo
        train_end_idx = max(0, test_start_idx - embargo_days)
        train_dates_set = set(unique_dates[:train_end_idx])

        if len(train_dates_set) < min_train_dates // 2:
            continue

        train_mask = np.isin(dates, list(train_dates_set))
        test_mask = np.isin(dates, list(test_dates_set))
        folds.append((train_mask, test_mask))

    return folds


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
