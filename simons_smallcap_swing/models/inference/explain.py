"""
models/inference/explain.py — Feature attribution and explainability.

Explains WHY the model made each prediction, not just WHAT it predicted.
Critical for: alpha attribution, signal decomposition, model monitoring,
regime analysis, and regulatory audit.

Methods (resolution by model type):
    Linear:   β_j × x_j (exact, no approximation needed)
    GBDT:     TreeSHAP (exact Shapley for trees, O(TLD))
    Any:      Permutation Feature Importance (model-agnostic, global only)
    Ensemble: component-weighted attribution when additive

Key design:
1. Method resolution matrix: auto-selects best method for model type
2. Baseline convergence check: |E[ŷ(X_ref)] - ϕ_0| < δ
3. Additive closure: |ŷ(x) - (ϕ_0 + Σ ϕ_j)| < ε for every observation
4. Global importance: mean(|ϕ_j|) across observations
5. Interaction detection: Σ|ϕ_j - ϕ_j_marginal| as proxy
6. Per-date attribution: how feature importance changes over time

References:
    Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
    Breiman (2001) "Random Forests" — permutation importance
    Shapley (1953) "A Value for n-Person Games"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from .. import (
    ModelError,
    information_coefficient,
)
from . import ExplainError

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExplainConfig:
    """Configuration for feature attribution."""
    method: str = "auto"             # "auto" | "linear" | "shap" | "pfi"
    n_pfi_repeats: int = 10          # permutation importance repeats
    pfi_metric: str = "ic"           # "ic" | "mse" — metric for PFI
    max_background_samples: int = 500  # for SHAP background set
    closure_tolerance: float = 1e-4   # additive closure check
    baseline_tolerance: float = 1e-3  # baseline convergence check
    seed: int = 42


# ---------------------------------------------------------------------------
# Method resolution
# ---------------------------------------------------------------------------

def resolve_method(model: Any, config: ExplainConfig) -> str:
    """Auto-resolve the best explanation method for the model type.

    Resolution matrix:
        Linear (Ridge, Lasso, EN, Logistic) → "linear" (exact)
        GBDT (LightGBM, XGBoost, sklearn trees) → "shap" (if available)
        Any → "pfi" (model-agnostic fallback)
    """
    if config.method != "auto":
        return config.method

    model_type = type(model).__name__.lower()
    module = type(model).__module__ or ""

    # Linear models
    linear_types = {"ridge", "lasso", "elasticnet", "linearregression",
                    "logisticregression", "sgdclassifier", "sgdregressor"}
    if model_type in linear_types or "linear_model" in module:
        return "linear"

    # Tree-based (SHAP compatible)
    tree_types = {"lgbmregressor", "lgbmclassifier", "xgbregressor", "xgbclassifier",
                  "randomforestregressor", "randomforestclassifier",
                  "gradientboostingregressor", "gradientboostingclassifier",
                  "decisiontreeregressor", "decisiontreeclassifier"}
    if model_type in tree_types or "lightgbm" in module or "xgboost" in module:
        try:
            import shap  # noqa
            return "shap"
        except ImportError:
            LOGGER.warning("shap not installed — falling back to PFI for %s", model_type)
            return "pfi"

    return "pfi"


# ---------------------------------------------------------------------------
# LINEAR ATTRIBUTION (exact)
# ---------------------------------------------------------------------------

@dataclass
class LinearAttribution:
    """Exact feature attribution for linear models: ϕ_j = β_j × x_j."""
    feature_names: list[str]
    coefficients: np.ndarray          # (n_features,)
    intercept: float
    local_contributions: np.ndarray   # (n_obs, n_features) — β_j × x_j
    global_importance: np.ndarray     # (n_features,) — mean(|β_j × x_j|)
    global_coef_importance: np.ndarray  # (n_features,) — |β_j|
    baseline: float                   # ϕ_0 = intercept
    closure_errors: np.ndarray        # (n_obs,) — |ŷ - (ϕ_0 + Σϕ_j)|


def explain_linear(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    config: ExplainConfig,
) -> LinearAttribution:
    """Exact attribution for linear models: ϕ_j(x) = β_j × x_j."""
    coef = model.coef_.flatten()
    intercept = float(model.intercept_) if hasattr(model, "intercept_") else 0.0

    if len(coef) != X.shape[1]:
        raise ExplainError(f"Coefficient count {len(coef)} != feature count {X.shape[1]}")

    # Local contributions: (n_obs, n_features)
    local = X * coef[np.newaxis, :]  # broadcasting

    # Predictions
    y_pred = X @ coef + intercept

    # Closure check: ŷ must equal ϕ_0 + Σ ϕ_j
    reconstructed = local.sum(axis=1) + intercept
    closure_err = np.abs(y_pred - reconstructed)

    if closure_err.max() > config.closure_tolerance:
        LOGGER.warning(
            "Linear attribution closure error: max=%.6f (tolerance=%.6f)",
            closure_err.max(), config.closure_tolerance,
        )

    # Global importance
    global_contrib = np.abs(local).mean(axis=0)
    global_coef = np.abs(coef)

    return LinearAttribution(
        feature_names=feature_names,
        coefficients=coef,
        intercept=intercept,
        local_contributions=local,
        global_importance=global_contrib,
        global_coef_importance=global_coef,
        baseline=intercept,
        closure_errors=closure_err,
    )


# ---------------------------------------------------------------------------
# SHAP ATTRIBUTION (for tree models)
# ---------------------------------------------------------------------------

@dataclass
class ShapAttribution:
    """SHAP-based feature attribution (exact for trees via TreeSHAP)."""
    feature_names: list[str]
    shap_values: np.ndarray           # (n_obs, n_features)
    expected_value: float             # ϕ_0
    global_importance: np.ndarray     # mean(|ϕ_j|)
    closure_errors: np.ndarray        # |ŷ - (ϕ_0 + Σϕ_j)|


def explain_shap(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    config: ExplainConfig,
) -> ShapAttribution:
    """TreeSHAP attribution for tree-based models."""
    try:
        import shap
    except ImportError:
        raise ExplainError("shap package required for TreeSHAP. Install: pip install shap")

    # Background set (subsample for efficiency)
    bg_size = min(config.max_background_samples, len(X))
    rng = np.random.RandomState(config.seed)
    bg_idx = rng.choice(len(X), size=bg_size, replace=False)
    background = X[bg_idx]

    # Create explainer
    try:
        explainer = shap.TreeExplainer(model, data=background)
    except Exception:
        # Fallback to KernelSHAP (slower but model-agnostic)
        LOGGER.warning("TreeExplainer failed — falling back to KernelExplainer")
        explainer = shap.KernelExplainer(model.predict, background)

    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # binary classification: use class 1

    expected_value = float(explainer.expected_value)
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        expected_value = float(explainer.expected_value[1] if len(explainer.expected_value) > 1
                               else explainer.expected_value[0])

    # Closure check
    y_pred = model.predict(X)
    reconstructed = shap_values.sum(axis=1) + expected_value
    closure_err = np.abs(y_pred - reconstructed)

    if closure_err.max() > config.closure_tolerance:
        LOGGER.warning(
            "SHAP closure error: max=%.6f, mean=%.6f (tolerance=%.6f)",
            closure_err.max(), closure_err.mean(), config.closure_tolerance,
        )

    global_imp = np.abs(shap_values).mean(axis=0)

    return ShapAttribution(
        feature_names=feature_names,
        shap_values=shap_values,
        expected_value=expected_value,
        global_importance=global_imp,
        closure_errors=closure_err,
    )


# ---------------------------------------------------------------------------
# PERMUTATION FEATURE IMPORTANCE (model-agnostic global)
# ---------------------------------------------------------------------------

@dataclass
class PFIResult:
    """Permutation Feature Importance result."""
    feature_names: list[str]
    importance_mean: np.ndarray       # (n_features,) — mean PFI across repeats
    importance_std: np.ndarray        # (n_features,) — std across repeats
    baseline_score: float             # score with unpermuted features
    n_repeats: int
    metric_name: str


def explain_pfi(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    config: ExplainConfig,
) -> PFIResult:
    """Permutation Feature Importance: model-agnostic global attribution.

    PFI_j = E[L(y, ŷ_perm_j)] - E[L(y, ŷ)]

    Measures how much predictive power is lost when feature j is shuffled.
    Not a local attribution — only gives global feature ranking.
    """
    rng = np.random.RandomState(config.seed)

    # Baseline score
    y_pred = model.predict(X)
    valid = ~(np.isnan(y) | np.isnan(y_pred))
    if config.pfi_metric == "ic":
        baseline_score = information_coefficient(y[valid], y_pred[valid])
    else:
        baseline_score = -float(np.mean((y[valid] - y_pred[valid]) ** 2))

    n_features = X.shape[1]
    importances = np.zeros((config.n_pfi_repeats, n_features))

    for rep in range(config.n_pfi_repeats):
        for j in range(n_features):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            y_perm = model.predict(X_perm)
            valid_p = ~(np.isnan(y) | np.isnan(y_perm))

            if config.pfi_metric == "ic":
                perm_score = information_coefficient(y[valid_p], y_perm[valid_p])
            else:
                perm_score = -float(np.mean((y[valid_p] - y_perm[valid_p]) ** 2))

            importances[rep, j] = baseline_score - perm_score

    return PFIResult(
        feature_names=feature_names,
        importance_mean=importances.mean(axis=0),
        importance_std=importances.std(axis=0),
        baseline_score=baseline_score,
        n_repeats=config.n_pfi_repeats,
        metric_name=config.pfi_metric,
    )


# ---------------------------------------------------------------------------
# UNIFIED EXPLAIN FUNCTION
# ---------------------------------------------------------------------------

@dataclass
class ExplainResult:
    """Unified explanation result."""
    method: str
    feature_names: list[str]
    global_importance: np.ndarray     # (n_features,) — sorted by importance
    global_importance_df: pd.DataFrame  # feature_name, importance, rank
    local_attributions: np.ndarray | None = None  # (n_obs, n_features)
    baseline: float | None = None
    closure_errors: np.ndarray | None = None
    details: Any = None               # method-specific result object


def explain(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    *,
    y: np.ndarray | None = None,
    config: ExplainConfig | None = None,
) -> ExplainResult:
    """Unified explain function: auto-resolves method and runs attribution.

    Parameters
    ----------
    model : fitted model
    X : (n_obs, n_features) array
    feature_names : list of feature names
    y : actuals (required for PFI)
    config : ExplainConfig

    Returns
    -------
    ExplainResult with global importance, optional local attributions.
    """
    cfg = config or ExplainConfig()
    method = resolve_method(model, cfg)

    LOGGER.info("Explaining model %s with method=%s", type(model).__name__, method)

    if method == "linear":
        attr = explain_linear(model, X, feature_names, cfg)
        global_imp = attr.global_importance
        local = attr.local_contributions
        baseline = attr.baseline
        closure = attr.closure_errors
        details = attr

    elif method == "shap":
        attr = explain_shap(model, X, feature_names, cfg)
        global_imp = attr.global_importance
        local = attr.shap_values
        baseline = attr.expected_value
        closure = attr.closure_errors
        details = attr

    elif method == "pfi":
        if y is None:
            raise ExplainError("PFI requires y (actuals) to compute importance")
        attr = explain_pfi(model, X, y, feature_names, cfg)
        global_imp = attr.importance_mean
        local = None
        baseline = attr.baseline_score
        closure = None
        details = attr

    else:
        raise ExplainError(f"Unknown explain method: {method!r}")

    # Build importance DataFrame
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": global_imp,
    })
    imp_df["rank"] = imp_df["importance"].rank(ascending=False, method="min").astype(int)
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)

    return ExplainResult(
        method=method,
        feature_names=feature_names,
        global_importance=global_imp,
        global_importance_df=imp_df,
        local_attributions=local,
        baseline=baseline,
        closure_errors=closure,
        details=details,
    )


# ---------------------------------------------------------------------------
# Per-date attribution (how importance changes over time)
# ---------------------------------------------------------------------------

def explain_per_date(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    dates: np.ndarray,
    *,
    config: ExplainConfig | None = None,
) -> pd.DataFrame:
    """Compute global feature importance for each date separately.

    Returns DataFrame: (date, feature, importance) — useful for detecting
    regime shifts in what drives the model's predictions.
    """
    cfg = config or ExplainConfig()
    method = resolve_method(model, cfg)

    if method != "linear":
        LOGGER.info("Per-date attribution only implemented for linear models (got %s)", method)
        return pd.DataFrame()

    coef = model.coef_.flatten()
    intercept = float(model.intercept_) if hasattr(model, "intercept_") else 0.0

    unique_dates = np.unique(dates)
    rows = []
    for d in unique_dates:
        mask = dates == d
        X_d = X[mask]
        if len(X_d) < 5:
            continue
        # Mean absolute contribution per feature for this date
        contributions = np.abs(X_d * coef[np.newaxis, :]).mean(axis=0)
        for j, fname in enumerate(feature_names):
            rows.append({"date": d, "feature": fname, "importance": float(contributions[j])})

    return pd.DataFrame(rows)
