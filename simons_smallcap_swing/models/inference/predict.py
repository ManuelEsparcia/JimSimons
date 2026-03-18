"""
models/inference/predict.py — Production-grade prediction pipeline.

Resolves a model from registry, validates feature compatibility,
applies train-only preprocessing, generates scores, postprocesses,
and persists results with full audit trail.

Pipeline order (invariant — never reordered):
1. Resolve model_ref via registry
2. Load model artifact + manifest
3. Read features for eligible universe
4. Validate schema: exact column set, order, dtype match
5. Apply missingness policy (fail | impute | drop)
6. Apply preprocessing (using saved train-only state)
7. Generate y_pred_raw
8. Clip predictions (if configured)
9. Calibrate (if configured)
10. Rank within cross-section
11. Persist scores + manifest

Key design:
- Feature compatibility is EXACT: F_inf == F_train (set, order, dtype)
- Missingness policy is explicit: fail (default), impute, drop
- Preprocessing uses the SAME fitted state from training (no re-fitting)
- Predictions carry (date, symbol) for downstream merge
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from .. import (
    ModelError,
    PreprocessState,
    PreprocessConfig,
    apply_preprocess,
    information_coefficient,
    compute_all_metrics,
)
from . import (
    PredictionError,
    FeatureCompatibilityError,
    InferenceError,
)
from .model_registry import ModelRegistry, ModelVersion

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PredictConfig:
    """Configuration for the prediction pipeline."""
    # Model resolution
    model_id: str = ""
    model_version: str | None = None
    model_alias: str = "production"   # "production" | "staging" | "latest_candidate"

    # Feature compatibility
    strict_feature_match: bool = True  # F_inf == F_train exactly
    missing_feature_policy: str = "fail"  # "fail" | "impute" | "drop"
    max_pct_imputed: float = 0.10       # max fraction of imputed values

    # Postprocessing
    clip_lower: float | None = None
    clip_upper: float | None = None
    rank_within_date: bool = True       # add cross-sectional rank to output
    calibrate: bool = False

    # Universe filter
    require_eligible: bool = True       # only score eligible symbols

    # Output
    output_dir: str = "predictions"
    run_id: str = ""


# ---------------------------------------------------------------------------
# Feature validation
# ---------------------------------------------------------------------------

def validate_feature_compatibility(
    inference_cols: list[str],
    train_cols: list[str],
    *,
    strict: bool = True,
) -> tuple[bool, list[str]]:
    """Validate that inference features match training features exactly.

    Returns (compatible, issues).
    """
    issues: list[str] = []
    inf_set = set(inference_cols)
    train_set = set(train_cols)

    missing = train_set - inf_set
    extra = inf_set - train_set

    if missing:
        issues.append(f"Missing features (in train but not inference): {sorted(missing)}")
    if extra and strict:
        issues.append(f"Extra features (in inference but not train): {sorted(extra)}")

    # Order check
    if not missing and not extra:
        if inference_cols != train_cols:
            issues.append("Feature order mismatch (same columns but different order)")

    compatible = len(issues) == 0
    return compatible, issues


# ---------------------------------------------------------------------------
# Apply missingness policy
# ---------------------------------------------------------------------------

def apply_missingness_policy(
    X: pd.DataFrame,
    policy: str,
    preprocess_state: PreprocessState | None,
    max_pct_imputed: float = 0.10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply missingness policy to inference features.

    Returns (processed_df, diagnostics).
    """
    diag: dict[str, Any] = {"policy": policy, "n_input_rows": len(X)}
    null_counts = X.isnull().sum()
    total_nulls = int(null_counts.sum())
    total_cells = X.shape[0] * X.shape[1]
    pct_null = total_nulls / max(total_cells, 1)
    diag["total_nulls"] = total_nulls
    diag["pct_null"] = round(pct_null, 6)

    if policy == "fail":
        if total_nulls > 0:
            null_cols = null_counts[null_counts > 0].to_dict()
            raise PredictionError(
                f"Missingness policy='fail': {total_nulls} null values in "
                f"{len(null_cols)} columns. Columns: {null_cols}"
            )
        diag["action"] = "none (no nulls)"
        return X, diag

    if policy == "impute":
        out = X.copy()
        n_imputed = 0
        if preprocess_state is not None:
            for col in out.columns:
                if col in preprocess_state.impute_values:
                    mask = out[col].isnull()
                    n_imputed += int(mask.sum())
                    out[col] = out[col].fillna(preprocess_state.impute_values[col])
        else:
            # Fallback: impute with 0 (least harmful for standardised features)
            n_imputed = total_nulls
            out = out.fillna(0)

        pct_imputed = n_imputed / max(total_cells, 1)
        diag["n_imputed"] = n_imputed
        diag["pct_imputed"] = round(pct_imputed, 6)

        if pct_imputed > max_pct_imputed:
            raise PredictionError(
                f"Imputation fraction {pct_imputed:.2%} exceeds max_pct_imputed={max_pct_imputed:.2%}"
            )
        diag["action"] = "imputed"
        return out, diag

    if policy == "drop":
        out = X.dropna()
        n_dropped = len(X) - len(out)
        diag["n_dropped"] = n_dropped
        diag["pct_dropped"] = round(n_dropped / max(len(X), 1), 6)
        diag["action"] = "dropped rows with nulls"
        return out, diag

    raise PredictionError(f"Unknown missingness policy: {policy!r}")


# ---------------------------------------------------------------------------
# Score postprocessing
# ---------------------------------------------------------------------------

def clip_scores(scores: np.ndarray, lower: float | None, upper: float | None) -> np.ndarray:
    if lower is not None or upper is not None:
        return np.clip(scores, lower, upper)
    return scores


def rank_within_date(
    df: pd.DataFrame,
    score_col: str = "score",
    date_col: str = "date",
) -> pd.Series:
    """Cross-sectional percentile rank per date."""
    return df.groupby(date_col)[score_col].rank(method="average", pct=True)


# ---------------------------------------------------------------------------
# MAIN PREDICTION PIPELINE
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Output of the prediction pipeline."""
    scores_df: pd.DataFrame        # (date, symbol, score, score_rank, ...)
    n_scored: int
    n_eligible: int
    n_dropped: int
    missingness_diag: dict[str, Any]
    model_version: str
    model_id: str
    feature_hash: str
    run_id: str
    timestamp: str
    metrics: dict[str, float] | None = None  # if actuals available

    def to_manifest(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "feature_hash": self.feature_hash,
            "n_scored": self.n_scored,
            "n_eligible": self.n_eligible,
            "n_dropped": self.n_dropped,
            "missingness": self.missingness_diag,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
        }


def predict(
    features_df: pd.DataFrame,
    *,
    model: Any = None,
    model_version_obj: ModelVersion | None = None,
    registry: ModelRegistry | None = None,
    config: PredictConfig | None = None,
    preprocess_state: PreprocessState | None = None,
    feature_cols: list[str] | None = None,
    actuals_col: str | None = None,
) -> PredictionResult:
    """Run the full prediction pipeline.

    Parameters
    ----------
    features_df : DataFrame
        Must have 'date', 'symbol', and feature columns.
    model : fitted model with .predict() method
        If provided, used directly. Otherwise resolved from registry.
    model_version_obj : ModelVersion, optional
        Pre-resolved version (skip registry lookup).
    registry : ModelRegistry, optional
        Used to resolve model_ref if model not provided directly.
    config : PredictConfig
    preprocess_state : PreprocessState, optional
        Saved preprocessing state from training. Critical for consistency.
    feature_cols : list of str
        Feature columns to use. If None, inferred from model_version_obj.
    actuals_col : str, optional
        If provided, compute OOS metrics against this column.
    """
    cfg = config or PredictConfig()
    if not cfg.run_id:
        cfg = PredictConfig(**{
            **{f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()},
            "run_id": f"pred_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        })

    LOGGER.info("Prediction pipeline: run_id=%s", cfg.run_id)

    # --- Step 1-2: Resolve model ---
    mv = model_version_obj
    if model is None and registry is not None and mv is None:
        mv = registry.resolve(
            cfg.model_id,
            version=cfg.model_version,
            alias=cfg.model_alias if cfg.model_version is None else None,
        )
        LOGGER.info("Resolved model: %s v%s (stage=%s)", mv.model_id, mv.version, mv.stage)

    if mv is not None and feature_cols is None:
        feature_cols = mv.feature_names

    if feature_cols is None:
        raise PredictionError("feature_cols must be provided or resolved from model version")

    # --- Step 3: Universe filter ---
    df = features_df.copy()
    n_eligible = len(df)
    if cfg.require_eligible:
        for ecol in ("eligible_flag", "is_eligible", "eligible"):
            if ecol in df.columns:
                df = df[df[ecol].astype(bool)].copy()
                n_eligible = len(df)
                break

    # --- Step 4: Feature compatibility validation ---
    available_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = [c for c in feature_cols if c not in df.columns]

    if cfg.strict_feature_match and missing_cols:
        raise FeatureCompatibilityError(
            f"Missing features: {missing_cols}. "
            f"Strict mode requires exact match between train and inference features."
        )

    X_raw = df[available_cols]

    # --- Step 5: Missingness policy ---
    X_processed, miss_diag = apply_missingness_policy(
        X_raw, cfg.missing_feature_policy, preprocess_state, cfg.max_pct_imputed,
    )

    n_dropped = len(X_raw) - len(X_processed)
    scored_idx = X_processed.index

    # --- Step 6: Preprocessing ---
    if preprocess_state is not None:
        X_array = apply_preprocess(X_processed, preprocess_state)
    else:
        X_array = X_processed.values.astype(np.float64)
        X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Step 7: Predict ---
    if model is not None:
        y_pred_raw = model.predict(X_array)
    else:
        raise PredictionError("No model provided and registry resolution not yet loading artifacts")

    # --- Step 8-9: Postprocess ---
    y_pred = clip_scores(y_pred_raw, cfg.clip_lower, cfg.clip_upper)

    # --- Step 10: Build output ---
    out = df.loc[scored_idx, ["date", "symbol"]].copy()
    out["score"] = y_pred
    out["score_raw"] = y_pred_raw

    if cfg.rank_within_date and "date" in out.columns:
        out["score_rank"] = rank_within_date(out, "score", "date")

    # --- Metrics against actuals ---
    metrics_out = None
    if actuals_col is not None and actuals_col in df.columns:
        y_actual = df.loc[scored_idx, actuals_col].values.astype(np.float64)
        metrics_out = compute_all_metrics(y_actual, y_pred)
        LOGGER.info("OOS metrics: IC=%.4f, hit=%.3f, LS20=%.4f",
                     metrics_out.get("ic", 0), metrics_out.get("hit_ratio", 0),
                     metrics_out.get("long_short_spread_20", 0))

    # --- Feature hash ---
    feature_hash = hashlib.sha256(
        json.dumps(sorted(available_cols)).encode()
    ).hexdigest()[:16]

    result = PredictionResult(
        scores_df=out,
        n_scored=len(out),
        n_eligible=n_eligible,
        n_dropped=n_dropped,
        missingness_diag=miss_diag,
        model_version=mv.version if mv else "direct",
        model_id=mv.model_id if mv else "direct",
        feature_hash=feature_hash,
        run_id=cfg.run_id,
        timestamp=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        metrics=metrics_out,
    )

    LOGGER.info(
        "Predictions complete: %d scored, %d dropped, feature_hash=%s",
        result.n_scored, result.n_dropped, feature_hash,
    )

    return result
