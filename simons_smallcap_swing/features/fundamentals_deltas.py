"""
features/fundamentals_deltas.py — Fundamental change signals.

Computes QoQ and YoY deltas (absolute + relative) and accelerations
from PIT fundamental levels. Ensures fiscal comparability: only compares
fiscally equivalent periods (Q3→Q3, not Q3→Q4).

Feature families:
    Delta absolute:  metric_t - metric_{t-lag}
    Delta relative:  (metric_t - metric_{t-lag}) / |metric_{t-lag}| + ε
    Acceleration:    delta_t - delta_{t-lag}
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    FeatureError,
    DataContractError,
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
class DeltaConfig:
    """Configuration for fundamental delta computation."""
    decision_lag: int = DEFAULT_DECISION_LAG
    near_zero_base_threshold: float = 1e-4
    max_relative_delta: float = 10.0  # cap relative deltas at ±1000%
    compute_qoq: bool = True
    compute_yoy: bool = True
    compute_acceleration: bool = True

    # Metrics to compute deltas for
    delta_metrics: tuple[str, ...] = (
        "revenue", "net_income", "gross_margin", "operating_margin",
        "roe", "roa", "debt_to_equity", "fcf_yield",
    )


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _fiscal_lag_key(fiscal_year: pd.Series, fiscal_period: pd.Series) -> pd.Series:
    """Create a sortable fiscal period key: YYYY_QN."""
    fy = fiscal_year.astype(str)
    fp = fiscal_period.astype(str).str.upper()
    return fy + "_" + fp


def _find_comparable_lag(
    df: pd.DataFrame,
    symbol_col: str,
    fiscal_year_col: str,
    fiscal_period_col: str,
    metric_col: str,
    lag_quarters: int,
) -> pd.Series:
    """Find the comparable fiscal period value with the given lag.

    For QoQ (lag=1): Q3 2024 → Q2 2024
    For YoY (lag=4): Q3 2024 → Q3 2023

    Returns NaN where no comparable period exists.
    """
    work = df[[symbol_col, fiscal_year_col, fiscal_period_col, metric_col, "date"]].copy()
    work["_fy"] = work[fiscal_year_col].astype(float)

    # Map fiscal period to quarter number
    fp_map = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 4,
              "1": 1, "2": 2, "3": 3, "4": 4}
    work["_fq"] = work[fiscal_period_col].astype(str).str.upper().map(fp_map)

    # Compute target quarter
    total_q = work["_fy"] * 4 + work["_fq"]
    target_q = total_q - lag_quarters
    work["_target_fy"] = (target_q - 1) // 4
    work["_target_fq"] = ((target_q - 1) % 4) + 1

    # Self-join to find the lag value
    lag_df = work[[symbol_col, "_fy", "_fq", metric_col]].copy()
    lag_df.columns = [symbol_col, "_target_fy", "_target_fq", f"{metric_col}_lag"]

    merged = work.merge(lag_df, on=[symbol_col, "_target_fy", "_target_fq"], how="left")
    return merged[f"{metric_col}_lag"]


def compute_delta_absolute(
    current: pd.Series,
    lagged: pd.Series,
) -> pd.Series:
    """Absolute delta: current - lagged."""
    return current - lagged


def compute_delta_relative(
    current: pd.Series,
    lagged: pd.Series,
    threshold: float = 1e-4,
    cap: float = 10.0,
) -> pd.Series:
    """Relative delta: (current - lagged) / (|lagged| + ε).

    Returns NaN where |lagged| < threshold (near-zero base).
    Clips result to [-cap, cap].
    """
    safe_base = lagged.abs().where(lagged.abs() > threshold)
    delta = (current - lagged) / safe_base
    return delta.clip(lower=-cap, upper=cap)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_fundamental_deltas(
    fundamentals: pd.DataFrame,
    *,
    config: DeltaConfig | None = None,
) -> tuple[pd.DataFrame, list[FeatureDef]]:
    """Build QoQ/YoY deltas and accelerations from PIT fundamentals.

    Parameters
    ----------
    fundamentals : DataFrame
        Must have: date, symbol, fiscal_year, fiscal_period, + metric columns.
    config : DeltaConfig, optional

    Returns
    -------
    features : DataFrame with (date, symbol) + delta columns
    feature_defs : list of FeatureDef metadata
    """
    cfg = config or DeltaConfig()
    df = fundamentals.copy()

    # Resolve columns
    sym_col = "symbol" if "symbol" in df.columns else "ticker"
    fy_col = None
    for c in ("fiscal_year", "fy", "fiscalyear"):
        if c in df.columns:
            fy_col = c
            break
    fp_col = None
    for c in ("fiscal_period", "fp", "fiscalperiod", "quarter"):
        if c in df.columns:
            fp_col = c
            break

    out = df[["date", sym_col]].copy()
    out.columns = ["date", "symbol"]
    feature_defs: list[FeatureDef] = []

    # If no fiscal period info, fall back to simple shift-based deltas
    use_fiscal = fy_col is not None and fp_col is not None

    for metric in cfg.delta_metrics:
        if metric not in df.columns:
            continue

        current = df[metric]

        if cfg.compute_qoq:
            if use_fiscal:
                lagged_q = _find_comparable_lag(df, sym_col, fy_col, fp_col, metric, lag_quarters=1)
            else:
                lagged_q = df.groupby(sym_col)[metric].shift(1)

            name_abs = f"{metric}_delta_qoq"
            name_rel = f"{metric}_growth_qoq"
            out[name_abs] = compute_delta_absolute(current, lagged_q)
            out[name_rel] = compute_delta_relative(current, lagged_q, cfg.near_zero_base_threshold, cfg.max_relative_delta)
            feature_defs.append(FeatureDef(name_abs, FeatureFamily.FUNDAMENTAL_DELTA, f"{metric}_t - {metric}_{{t-1Q}}", 0, cfg.decision_lag, "edgar_pit"))
            feature_defs.append(FeatureDef(name_rel, FeatureFamily.FUNDAMENTAL_DELTA, f"({metric}_t - {metric}_{{t-1Q}}) / |{metric}_{{t-1Q}}|", 0, cfg.decision_lag, "edgar_pit"))

        if cfg.compute_yoy:
            if use_fiscal:
                lagged_y = _find_comparable_lag(df, sym_col, fy_col, fp_col, metric, lag_quarters=4)
            else:
                lagged_y = df.groupby(sym_col)[metric].shift(4)

            name_abs = f"{metric}_delta_yoy"
            name_rel = f"{metric}_growth_yoy"
            out[name_abs] = compute_delta_absolute(current, lagged_y)
            out[name_rel] = compute_delta_relative(current, lagged_y, cfg.near_zero_base_threshold, cfg.max_relative_delta)
            feature_defs.append(FeatureDef(name_abs, FeatureFamily.FUNDAMENTAL_DELTA, f"{metric}_t - {metric}_{{t-4Q}}", 0, cfg.decision_lag, "edgar_pit"))
            feature_defs.append(FeatureDef(name_rel, FeatureFamily.FUNDAMENTAL_DELTA, f"({metric}_t - {metric}_{{t-4Q}}) / |{metric}_{{t-4Q}}|", 0, cfg.decision_lag, "edgar_pit"))

        if cfg.compute_acceleration and cfg.compute_qoq:
            # Acceleration: delta_qoq_t - delta_qoq_{t-1Q}
            delta_col = f"{metric}_growth_qoq"
            if delta_col in out.columns:
                if use_fiscal:
                    lagged_delta = _find_comparable_lag(
                        pd.concat([df[[sym_col, fy_col, fp_col, "date"]], out[[delta_col]]], axis=1),
                        sym_col, fy_col, fp_col, delta_col, lag_quarters=1,
                    )
                else:
                    lagged_delta = out.groupby("symbol")[delta_col].shift(1)

                accel_name = f"{metric}_accel_qoq"
                out[accel_name] = out[delta_col] - lagged_delta
                feature_defs.append(FeatureDef(accel_name, FeatureFamily.FUNDAMENTAL_DELTA, f"Δ²{metric}_QoQ", 0, cfg.decision_lag, "edgar_pit"))

    # Replace Inf
    feat_cols = [c for c in out.columns if c not in ("date", "symbol")]
    out[feat_cols] = out[feat_cols].replace([np.inf, -np.inf], np.nan)

    LOGGER.info("Fundamental deltas built: %d features, %d rows", len(feat_cols), len(out))
    return out, feature_defs
