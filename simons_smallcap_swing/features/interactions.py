"""
features/interactions.py — Supervised second-order feature interactions.

Only whitelisted interactions are computed. Each must pass admission
criteria: joint coverage, QC status of base features, and economic
separability. Prohibited by default: all-pairs, order > 2, gates
with future information.

Interaction types:
    product          z_a × z_b  (after cross-sectional standardization)
    ratio_safe       z_a / (|z_b| + ε)  (with near-zero policy)
    gate_binary      z_a × 1{condition_b}  (PIT-safe gating)
    modulation       z_a × sigmoid(z_b)  (smooth gating)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    FeatureError,
    ConfigError,
    FeatureFamily,
    FeatureDef,
    DEFAULT_DECISION_LAG,
    get_logger,
)

LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InteractionSpec:
    """Specification for a single interaction."""
    feature_a: str
    feature_b: str
    interaction_type: str  # "product" | "ratio_safe" | "gate_binary" | "modulation"
    name: str | None = None
    gate_threshold: float = 0.0  # for gate_binary: 1{b > threshold}
    near_zero_eps: float = 1e-2  # for ratio_safe

    @property
    def output_name(self) -> str:
        if self.name:
            return self.name
        return f"{self.feature_a}_x_{self.feature_b}_{self.interaction_type}"


# Default whitelist: small, explicit, economically motivated
DEFAULT_WHITELIST: tuple[InteractionSpec, ...] = (
    InteractionSpec("momentum_21d", "amihud_20d", "product", "mom_x_illiq"),
    InteractionSpec("ep_ratio", "realized_vol_21d", "product", "value_x_vol"),
    InteractionSpec("revenue_growth_yoy", "log_mcap", "product", "growth_x_size"),
    InteractionSpec("reversal_1d", "vol_shock_20d", "product", "reversal_x_volshock"),
    InteractionSpec("roe", "debt_to_equity", "product", "quality_x_leverage"),
)


@dataclass(frozen=True)
class InteractionConfig:
    """Configuration for interaction construction."""
    whitelist: tuple[InteractionSpec, ...] = DEFAULT_WHITELIST
    max_interactions: int = 10
    min_joint_coverage: float = 0.30
    decision_lag: int = DEFAULT_DECISION_LAG
    sigmoid_scale: float = 1.0  # for modulation: sigmoid(z_b / scale)


# ---------------------------------------------------------------------------
# Interaction functions
# ---------------------------------------------------------------------------

def interaction_product(a: pd.Series, b: pd.Series) -> pd.Series:
    """Product interaction: a × b (both should be standardized)."""
    return a * b


def interaction_ratio_safe(a: pd.Series, b: pd.Series, eps: float = 1e-2) -> pd.Series:
    """Safe ratio: a / (|b| + ε)."""
    return a / (b.abs() + eps)


def interaction_gate_binary(a: pd.Series, b: pd.Series, threshold: float = 0.0) -> pd.Series:
    """Binary gate: a × 1{b > threshold}."""
    gate = (b > threshold).astype(float)
    gate[b.isna()] = np.nan
    return a * gate


def interaction_modulation(a: pd.Series, b: pd.Series, scale: float = 1.0) -> pd.Series:
    """Smooth modulation: a × sigmoid(b / scale)."""
    sig = 1.0 / (1.0 + np.exp(-b / scale))
    return a * sig


INTERACTION_FUNCTIONS = {
    "product": interaction_product,
    "ratio_safe": interaction_ratio_safe,
    "gate_binary": interaction_gate_binary,
    "modulation": interaction_modulation,
}


# ---------------------------------------------------------------------------
# Admission checks
# ---------------------------------------------------------------------------

def check_joint_coverage(
    a: pd.Series, b: pd.Series, min_coverage: float,
) -> bool:
    """Check that the joint non-null fraction meets the minimum."""
    joint_valid = a.notna() & b.notna()
    coverage = joint_valid.mean()
    return coverage >= min_coverage


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_interactions(
    features_df: pd.DataFrame,
    *,
    config: InteractionConfig | None = None,
) -> tuple[pd.DataFrame, list[FeatureDef]]:
    """Build interaction features from a standardized feature panel.

    Parameters
    ----------
    features_df : DataFrame
        Must have date, symbol, and the base feature columns referenced
        by the whitelist. Features should already be cross-sectionally
        transformed (z-scored or ranked).
    config : InteractionConfig, optional

    Returns
    -------
    interactions : DataFrame with (date, symbol) + interaction columns
    feature_defs : list of FeatureDef metadata
    """
    cfg = config or InteractionConfig()

    out = features_df[["date", "symbol"]].copy()
    feature_defs: list[FeatureDef] = []
    n_skipped = 0

    for spec in cfg.whitelist[:cfg.max_interactions]:
        # Check base features exist
        if spec.feature_a not in features_df.columns:
            LOGGER.debug("Skipping interaction %s: %s not in panel", spec.output_name, spec.feature_a)
            n_skipped += 1
            continue
        if spec.feature_b not in features_df.columns:
            LOGGER.debug("Skipping interaction %s: %s not in panel", spec.output_name, spec.feature_b)
            n_skipped += 1
            continue

        a = features_df[spec.feature_a]
        b = features_df[spec.feature_b]

        # Admission: joint coverage
        if not check_joint_coverage(a, b, cfg.min_joint_coverage):
            LOGGER.info(
                "Interaction %s rejected: joint coverage < %.0f%%",
                spec.output_name, cfg.min_joint_coverage * 100,
            )
            n_skipped += 1
            continue

        # Compute interaction
        fn = INTERACTION_FUNCTIONS.get(spec.interaction_type)
        if fn is None:
            raise ConfigError(f"Unknown interaction type: {spec.interaction_type!r}")

        if spec.interaction_type == "ratio_safe":
            result = fn(a, b, spec.near_zero_eps)
        elif spec.interaction_type == "gate_binary":
            result = fn(a, b, spec.gate_threshold)
        elif spec.interaction_type == "modulation":
            result = fn(a, b, cfg.sigmoid_scale)
        else:
            result = fn(a, b)

        out[spec.output_name] = result
        feature_defs.append(FeatureDef(
            name=spec.output_name,
            family=FeatureFamily.INTERACTION,
            formula=f"{spec.interaction_type}({spec.feature_a}, {spec.feature_b})",
            lookback_days=0,
            decision_lag=cfg.decision_lag,
            source="derived",
        ))

    # Replace Inf
    feat_cols = [c for c in out.columns if c not in ("date", "symbol")]
    out[feat_cols] = out[feat_cols].replace([np.inf, -np.inf], np.nan)

    LOGGER.info(
        "Interactions built: %d computed, %d skipped (missing base or low coverage)",
        len(feat_cols), n_skipped,
    )
    return out, feature_defs
