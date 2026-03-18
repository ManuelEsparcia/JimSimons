"""
features/cross_sectional.py — Cross-sectional feature transformations.

Transforms raw features into statistically comparable representations
within each date's cross-section. All transforms are strictly per-date
(never leak information across dates).

Canonical pipeline order: raw → validate → winsor → rank/zscore → persist

Transforms:
    cs_rank           Percentile rank ∈ (0, 1), method="average" for ties
    cs_rank_gauss     Rank → probit (inverse normal CDF), ≈ Gaussian marginal
    cs_zscore_robust  (X - median) / (MAD × 1.4826 + ε), robust to tails
    cs_winsor         Clip at [Q_lower, Q_upper] percentiles
    cs_quantile_bucket  Assign to q equal-frequency buckets {0, ..., q-1}
    cs_demean_group   Subtract group mean (sector/industry neutralization)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from . import (
    FeatureError,
    CoverageError,
    MAD_NORMAL_CONSISTENCY,
    DEFAULT_WINSOR_LOWER,
    DEFAULT_WINSOR_UPPER,
    DEFAULT_MAD_FLOOR,
    DEFAULT_MIN_OBS_CS,
    Severity,
    get_logger,
)

LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CrossSectionalConfig:
    """Configuration for cross-sectional transforms."""
    winsor_lower: float = DEFAULT_WINSOR_LOWER
    winsor_upper: float = DEFAULT_WINSOR_UPPER
    mad_floor: float = DEFAULT_MAD_FLOOR
    min_obs: int = DEFAULT_MIN_OBS_CS
    mad_degenerate_policy: str = "fallback_to_rank"  # "return_zero" | "fallback_to_rank" | "return_null"
    n_quantile_buckets: int = 5
    rank_method: str = "average"  # pandas rank method for ties


# ---------------------------------------------------------------------------
# Per-date transform functions
# ---------------------------------------------------------------------------

def cs_rank(
    values: pd.Series,
    *,
    method: str = "average",
    min_obs: int = DEFAULT_MIN_OBS_CS,
) -> pd.Series:
    """Percentile rank ∈ (0, 1) within a cross-section.

    Uses `method="average"` for deterministic tie handling (critical for
    small caps where rounded returns produce many ties).
    """
    valid = values.dropna()
    if len(valid) < min_obs:
        return pd.Series(np.nan, index=values.index)
    ranked = valid.rank(method=method, pct=True)
    return ranked.reindex(values.index)


def cs_rank_gauss(
    values: pd.Series,
    *,
    method: str = "average",
    min_obs: int = DEFAULT_MIN_OBS_CS,
) -> pd.Series:
    """Rank → probit transform (inverse normal CDF).

    Produces approximately Gaussian marginal distribution. Clips extreme
    ranks to avoid ±∞ at boundaries.
    """
    ranked = cs_rank(values, method=method, min_obs=min_obs)
    if ranked.isna().all():
        return ranked
    # Clip to (ε, 1-ε) to avoid ±∞ from ppf
    eps = 0.5 / (ranked.notna().sum() + 1)
    clipped = ranked.clip(lower=eps, upper=1.0 - eps)
    return clipped.map(sp_stats.norm.ppf)


def cs_zscore_robust(
    values: pd.Series,
    *,
    mad_floor: float = DEFAULT_MAD_FLOOR,
    min_obs: int = DEFAULT_MIN_OBS_CS,
    degenerate_policy: str = "fallback_to_rank",
) -> pd.Series:
    """Robust z-score: (X - median) / (MAD × 1.4826 + ε).

    When MAD < mad_floor (degenerate cross-section), applies the
    configured degenerate policy:
        "return_zero"      → all zeros
        "fallback_to_rank" → cs_rank_gauss instead
        "return_null"      → all NaN
    """
    valid = values.dropna()
    if len(valid) < min_obs:
        return pd.Series(np.nan, index=values.index)

    med = valid.median()
    mad = (valid - med).abs().median()
    scale = mad * MAD_NORMAL_CONSISTENCY

    if scale < mad_floor:
        if degenerate_policy == "return_zero":
            out = pd.Series(0.0, index=values.index)
            out[values.isna()] = np.nan
            return out
        if degenerate_policy == "fallback_to_rank":
            return cs_rank_gauss(values, min_obs=min_obs)
        return pd.Series(np.nan, index=values.index)

    z = (values - med) / scale
    return z


def cs_winsor(
    values: pd.Series,
    *,
    lower: float = DEFAULT_WINSOR_LOWER,
    upper: float = DEFAULT_WINSOR_UPPER,
    min_obs: int = DEFAULT_MIN_OBS_CS,
) -> pd.Series:
    """Winsorize at [Q_lower, Q_upper] percentiles within cross-section."""
    valid = values.dropna()
    if len(valid) < min_obs:
        return values.copy()
    q_lo = valid.quantile(lower)
    q_hi = valid.quantile(upper)
    return values.clip(lower=q_lo, upper=q_hi)


def cs_quantile_bucket(
    values: pd.Series,
    *,
    n_buckets: int = 5,
    min_obs: int = DEFAULT_MIN_OBS_CS,
) -> pd.Series:
    """Assign to equal-frequency quantile buckets {0, ..., n_buckets-1}.

    Uses rank-based assignment to avoid quantile boundary ambiguity.
    """
    valid = values.dropna()
    if len(valid) < min_obs:
        return pd.Series(np.nan, index=values.index)
    ranked = valid.rank(method="average", pct=True)
    # Bucket: floor(rank * n_buckets), clipped to [0, n_buckets-1]
    buckets = np.floor(ranked * n_buckets).clip(upper=n_buckets - 1).astype(float)
    return buckets.reindex(values.index)


def cs_demean_group(
    values: pd.Series,
    group: pd.Series,
    *,
    min_group_size: int = 3,
) -> pd.Series:
    """Subtract group mean (e.g., sector neutralization).

    Groups with fewer than min_group_size observations get NaN.
    """
    combined = pd.DataFrame({"val": values, "grp": group})
    combined = combined.dropna(subset=["val"])
    group_stats = combined.groupby("grp")["val"].agg(["mean", "count"])
    valid_groups = group_stats[group_stats["count"] >= min_group_size].index
    group_means = group_stats.loc[valid_groups, "mean"]
    mapped_mean = group.map(group_means)
    return values - mapped_mean


# ---------------------------------------------------------------------------
# Panel-level transform: apply per-date
# ---------------------------------------------------------------------------

def transform_panel(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    transform: str = "cs_zscore_robust",
    config: CrossSectionalConfig | None = None,
    date_col: str = "date",
    group_col: str | None = None,
) -> pd.DataFrame:
    """Apply a cross-sectional transform to each feature column per date.

    Parameters
    ----------
    df : DataFrame with date_col + feature_cols (+ optional group_col)
    feature_cols : columns to transform
    transform : one of "cs_rank", "cs_rank_gauss", "cs_zscore_robust",
                "cs_winsor", "cs_quantile_bucket", "cs_demean_group"
    config : CrossSectionalConfig
    date_col : date column name
    group_col : group column for cs_demean_group

    Returns
    -------
    DataFrame with transformed feature columns (date + symbol preserved)
    """
    cfg = config or CrossSectionalConfig()
    out = df.copy()

    # Choose the transform function
    TRANSFORMS = {
        "cs_rank": lambda s: cs_rank(s, method=cfg.rank_method, min_obs=cfg.min_obs),
        "cs_rank_gauss": lambda s: cs_rank_gauss(s, method=cfg.rank_method, min_obs=cfg.min_obs),
        "cs_zscore_robust": lambda s: cs_zscore_robust(
            s, mad_floor=cfg.mad_floor, min_obs=cfg.min_obs,
            degenerate_policy=cfg.mad_degenerate_policy,
        ),
        "cs_winsor": lambda s: cs_winsor(s, lower=cfg.winsor_lower, upper=cfg.winsor_upper, min_obs=cfg.min_obs),
        "cs_quantile_bucket": lambda s: cs_quantile_bucket(s, n_buckets=cfg.n_quantile_buckets, min_obs=cfg.min_obs),
    }

    if transform == "cs_demean_group":
        if group_col is None:
            raise FeatureError("cs_demean_group requires group_col")
        for col in feature_cols:
            out[col] = out.groupby(date_col).apply(
                lambda g: cs_demean_group(g[col], g[group_col])
            ).droplevel(0)
        return out

    fn = TRANSFORMS.get(transform)
    if fn is None:
        raise FeatureError(f"Unknown transform: {transform!r}")

    for col in feature_cols:
        out[col] = out.groupby(date_col)[col].transform(fn)

    n_degenerate = 0
    if transform == "cs_zscore_robust":
        # Count dates where all values became NaN (degenerate MAD)
        for col in feature_cols:
            per_date = out.groupby(date_col)[col].apply(lambda s: s.notna().sum() == 0)
            n_degenerate += per_date.sum()

    LOGGER.info(
        "Cross-sectional transform '%s' applied to %d features (%d degenerate date×feature pairs)",
        transform, len(feature_cols), n_degenerate,
    )
    return out
