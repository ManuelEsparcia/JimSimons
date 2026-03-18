"""
labels/build_labels.py — Canonical forward return labels.

Constructs PIT, multi-horizon, economically consistent targets for
supervised learning on small-cap cross-sections.

Primary target: y_fwd_ret_net_10d (10-day forward return net of costs).

Pipeline (spec §19):
    1.  Load config, prices, universe, calendar
    2.  Resolve entry/exit dates per horizon via trading calendar
    3.  Compute forward gross returns
    4.  Attach cost components → net returns
    5.  Derive ranking (percentile) and classification (tail) targets
    6.  Apply exclusion hierarchy (12 levels, spec §17)
    7.  Run inline QC
    8.  Persist labels + metadata

Temporal convention (spec §4):
    decision_time = close(t-1)
    execution_date = t
    t_in  = t + decision_lag
    t_out = t + decision_lag + h
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    LabelExclusionReason, EXCLUSION_PRIORITY, Severity,
    utc_now_iso, config_hash, json_safe,
    read_dataframe, write_parquet_safe, write_json_safe,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration (spec §5, §20)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LabelConfig:
    horizons: tuple[int, ...] = (5, 10, 20)
    primary_target: str = "y_fwd_ret_net_10d"
    return_mode: str = "close_to_close"    # close_to_close | open_to_close | ...
    decision_lag: int = 1                  # sessions between signal and entry
    net_of_costs: bool = True
    missing_cost_policy: str = "strict_invalidate"  # strict_invalidate | fallback_proxy | zero_imputation
    classification_default: str = "tail_20_20_on_net_10d"
    tail_top_pct: float = 0.20
    tail_bot_pct: float = 0.20
    max_forward_gap_days: int = 5
    min_valid_cross_section: int = 30
    delisting_policy: str = "use_if_available"


# ---------------------------------------------------------------------------
# Calendar-aware date resolution
# ---------------------------------------------------------------------------

def resolve_calendar_offset(
    base_dates: pd.Series,
    offset: int,
    calendar: np.ndarray,
) -> pd.Series:
    """Resolve t + offset trading days on the calendar.

    Returns NaT if the offset falls outside the calendar range.
    """
    cal_set = {pd.Timestamp(d): i for i, d in enumerate(calendar)}
    result = []
    for d in base_dates:
        d = pd.Timestamp(d)
        idx = cal_set.get(d)
        if idx is None:
            # Find nearest calendar date
            diffs = np.abs(calendar - np.datetime64(d))
            idx = int(np.argmin(diffs))
        target_idx = idx + offset
        if 0 <= target_idx < len(calendar):
            result.append(pd.Timestamp(calendar[target_idx]))
        else:
            result.append(pd.NaT)
    return pd.Series(result, index=base_dates.index)


# ---------------------------------------------------------------------------
# Forward return computation (spec §8)
# ---------------------------------------------------------------------------

def compute_forward_returns(
    prices: pd.DataFrame,
    calendar: np.ndarray,
    horizons: Sequence[int],
    decision_lag: int,
    return_mode: str,
) -> pd.DataFrame:
    """Compute forward gross returns for each (date, symbol, horizon).

    y_gross(i,t,h) = P_exit(i, t+lag+h) / P_entry(i, t+lag) - 1
    """
    df = prices.sort_values(["symbol", "date"]).copy()

    # Resolve price columns
    entry_col = "close"  # default for close_to_close
    exit_col = "close"
    if return_mode == "open_to_close":
        entry_col = "open"
    elif return_mode == "close_to_open":
        exit_col = "open"

    # Build price lookup: (symbol, date) → price
    price_lookup: dict[tuple[str, Any], dict[str, float]] = {}
    for _, row in df.iterrows():
        key = (row["symbol"], pd.Timestamp(row["date"]))
        price_lookup[key] = {
            "close": float(row.get("close", np.nan)),
            "open": float(row.get("open", row.get("close", np.nan))),
        }

    results = []
    for _, row in df.iterrows():
        sym = row["symbol"]
        t = pd.Timestamp(row["date"])
        base = {"date": t, "symbol": sym}

        # Entry date
        t_in_series = resolve_calendar_offset(pd.Series([t]), decision_lag, calendar)
        t_in = t_in_series.iloc[0]
        base["t_entry"] = t_in

        for h in horizons:
            t_out_series = resolve_calendar_offset(pd.Series([t]), decision_lag + h, calendar)
            t_out = t_out_series.iloc[0]

            entry_px = price_lookup.get((sym, t_in), {}).get(entry_col)
            exit_px = price_lookup.get((sym, t_out), {}).get(exit_col)

            if entry_px and exit_px and entry_px > 0 and not pd.isna(t_in) and not pd.isna(t_out):
                ret = exit_px / entry_px - 1.0
            else:
                ret = np.nan

            base[f"y_fwd_ret_gross_{h}d"] = ret
            base[f"t_exit_{h}d"] = t_out

        results.append(base)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Cost attachment (spec §9)
# ---------------------------------------------------------------------------

def attach_costs(
    labels: pd.DataFrame,
    costs: pd.DataFrame | None,
    horizons: Sequence[int],
    config: LabelConfig,
) -> pd.DataFrame:
    """Compute net returns: y_net = y_gross - C_entry - C_exit - C_carry."""
    df = labels.copy()

    for h in horizons:
        gross_col = f"y_fwd_ret_gross_{h}d"
        net_col = f"y_fwd_ret_net_{h}d"

        if not config.net_of_costs or costs is None:
            df[net_col] = df[gross_col]
            continue

        # Merge costs
        if "cost_entry_bps" in costs.columns and "cost_exit_bps" in costs.columns:
            merged = df.merge(costs[["date", "symbol", "cost_entry_bps", "cost_exit_bps"]],
                             on=["date", "symbol"], how="left")
            c_entry = merged["cost_entry_bps"].fillna(0) / 1e4
            c_exit = merged["cost_exit_bps"].fillna(0) / 1e4

            if config.missing_cost_policy == "strict_invalidate":
                missing_cost = merged["cost_entry_bps"].isna() | merged["cost_exit_bps"].isna()
                df[net_col] = np.where(missing_cost, np.nan, df[gross_col] - c_entry - c_exit)
            else:
                df[net_col] = df[gross_col] - c_entry - c_exit
        else:
            df[net_col] = df[gross_col]

    return df


# ---------------------------------------------------------------------------
# Derived targets: ranking and classification (spec §11)
# ---------------------------------------------------------------------------

def derive_ranking_targets(
    labels: pd.DataFrame,
    horizons: Sequence[int],
    base_prefix: str = "y_fwd_ret_net",
) -> pd.DataFrame:
    """Percentile rank cross-sectional: F_t,h(y) ∈ [0,1]."""
    df = labels.copy()
    for h in horizons:
        src = f"{base_prefix}_{h}d"
        dst = f"y_rank_{h}d"
        if src in df.columns:
            df[dst] = df.groupby("date")[src].rank(pct=True, method="average")
    return df


def derive_classification_targets(
    labels: pd.DataFrame,
    horizons: Sequence[int],
    base_prefix: str = "y_fwd_ret_net",
    top_pct: float = 0.20,
    bot_pct: float = 0.20,
) -> pd.DataFrame:
    """Tail classification: +1 (top), -1 (bottom), 0 (middle)."""
    df = labels.copy()
    for h in horizons:
        src = f"{base_prefix}_{h}d"
        dst = f"y_cls_{h}d"
        if src not in df.columns:
            continue

        def _classify_group(g):
            valid = g.dropna()
            if len(valid) < 10:
                return pd.Series(np.nan, index=g.index)
            q_top = valid.quantile(1 - top_pct)
            q_bot = valid.quantile(bot_pct)
            result = pd.Series(0, index=g.index, dtype=float)
            result[g >= q_top] = 1
            result[g <= q_bot] = -1
            result[g.isna()] = np.nan
            return result

        df[dst] = df.groupby("date")[src].transform(_classify_group)
    return df


# ---------------------------------------------------------------------------
# Exclusion hierarchy (spec §17)
# ---------------------------------------------------------------------------

def apply_exclusion_hierarchy(
    labels: pd.DataFrame,
    universe: pd.DataFrame | None,
    horizons: Sequence[int],
    calendar: np.ndarray,
    config: LabelConfig,
) -> pd.DataFrame:
    """Apply 12-level exclusion hierarchy, set label_valid_flag and reason."""
    df = labels.copy()
    n = len(df)

    # Initialize
    df["label_valid_flag"] = True
    df["label_exclusion_reason"] = None

    def _invalidate(mask, reason):
        update = mask & df["label_valid_flag"]
        df.loc[update, "label_valid_flag"] = False
        df.loc[update, "label_exclusion_reason"] = reason.value

    # 1. OUT_OF_SAMPLE_DATE (dates outside calendar)
    cal_set = set(pd.Timestamp(d) for d in calendar)
    out_of_cal = ~df["date"].isin(cal_set)
    _invalidate(out_of_cal, LabelExclusionReason.OUT_OF_SAMPLE_DATE)

    # 2. NOT_IN_UNIVERSE
    if universe is not None and "is_eligible" in universe.columns:
        merged = df.merge(
            universe[["date", "symbol", "is_eligible"]].drop_duplicates(["date", "symbol"]),
            on=["date", "symbol"], how="left",
        )
        not_eligible = merged["is_eligible"].fillna(0).astype(int) != 1
        _invalidate(not_eligible, LabelExclusionReason.NOT_IN_UNIVERSE)

    # 3-4. MISSING ENTRY/EXIT PRICE
    primary_h = config.horizons[len(config.horizons)//2] if config.horizons else 10
    gross_col = f"y_fwd_ret_gross_{primary_h}d"
    if gross_col in df.columns:
        missing_ret = df[gross_col].isna()
        _invalidate(missing_ret, LabelExclusionReason.MISSING_EXIT_PRICE)

    # 5. INCOMPLETE_FORWARD_WINDOW (last h days)
    if len(calendar) > 0:
        max_date = pd.Timestamp(calendar[-1])
        for h in horizons:
            last_valid = resolve_calendar_offset(
                pd.Series([max_date]), -(config.decision_lag + h), calendar
            ).iloc[0]
            if pd.notna(last_valid):
                incomplete = df["date"] > last_valid
                _invalidate(incomplete, LabelExclusionReason.INCOMPLETE_FORWARD_WINDOW)

    # 9. MISSING_COST_INPUT (only if net_of_costs and strict)
    if config.net_of_costs and config.missing_cost_policy == "strict_invalidate":
        for h in horizons:
            net_col = f"y_fwd_ret_net_{h}d"
            gross_col_h = f"y_fwd_ret_gross_{h}d"
            if net_col in df.columns and gross_col_h in df.columns:
                has_gross = df[gross_col_h].notna()
                no_net = df[net_col].isna()
                _invalidate(has_gross & no_net, LabelExclusionReason.MISSING_COST_INPUT)

    # Count valid cross-section per date
    df["n_valid_cross_section"] = df.groupby("date")["label_valid_flag"].transform("sum")

    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def build_labels(
    prices: pd.DataFrame | str | Path,
    universe: pd.DataFrame | str | Path | None = None,
    calendar: pd.DatetimeIndex | np.ndarray | None = None,
    *,
    costs: pd.DataFrame | str | Path | None = None,
    config: LabelConfig | None = None,
    output_dir: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build the canonical label store.

    Returns dict with: labels, coverage, manifest.
    """
    cfg = config or LabelConfig()
    if not run_id:
        run_id = f"labels_{utc_now_iso().replace(':','').replace('-','')}"

    # Load
    if isinstance(prices, (str, Path)):
        prices = read_dataframe(prices)
    if isinstance(universe, (str, Path)):
        universe = read_dataframe(universe)
    if isinstance(costs, (str, Path)):
        costs = read_dataframe(costs)

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")

    # Calendar
    if calendar is None:
        calendar = np.sort(prices["date"].dropna().unique())
    else:
        calendar = np.sort(np.array(pd.DatetimeIndex(calendar)))

    # Forward returns
    labels = compute_forward_returns(prices, calendar, cfg.horizons, cfg.decision_lag, cfg.return_mode)

    # Costs
    labels = attach_costs(labels, costs, cfg.horizons, cfg)

    # Derived targets
    base = "y_fwd_ret_net" if cfg.net_of_costs else "y_fwd_ret_gross"
    labels = derive_ranking_targets(labels, cfg.horizons, base)
    labels = derive_classification_targets(labels, cfg.horizons, base, cfg.tail_top_pct, cfg.tail_bot_pct)

    # Exclusion hierarchy
    labels = apply_exclusion_hierarchy(labels, universe, cfg.horizons, calendar, cfg)

    # Coverage
    coverage_rows = []
    for h in cfg.horizons:
        col = f"y_fwd_ret_net_{h}d" if cfg.net_of_costs else f"y_fwd_ret_gross_{h}d"
        if col in labels.columns:
            n_valid = labels[col].notna().sum()
            n_total = len(labels)
            coverage_rows.append({"horizon": h, "n_valid": int(n_valid), "n_total": n_total,
                                  "coverage_pct": round(n_valid / max(n_total, 1) * 100, 2)})
    coverage = pd.DataFrame(coverage_rows)

    # Add metadata columns
    labels["run_id"] = run_id

    manifest = {
        "run_id": run_id,
        "config_hash": config_hash(json_safe(cfg.__dict__)),
        "primary_target": cfg.primary_target,
        "horizons": list(cfg.horizons),
        "return_mode": cfg.return_mode,
        "decision_lag": cfg.decision_lag,
        "net_of_costs": cfg.net_of_costs,
        "n_rows": len(labels),
        "n_valid": int(labels["label_valid_flag"].sum()),
        "pct_valid": round(float(labels["label_valid_flag"].mean()) * 100, 2),
        "coverage": coverage.to_dict("records") if len(coverage) > 0 else [],
        "timestamp": utc_now_iso(),
    }

    if output_dir:
        out = Path(output_dir) / run_id
        out.mkdir(parents=True, exist_ok=True)
        write_parquet_safe(labels, out / "labels.parquet")
        write_parquet_safe(coverage, out / "label_coverage.parquet")
        write_json_safe(manifest, out / "label_manifest.json")

    LOGGER.info("Labels: %d rows, %d valid (%.1f%%), horizons=%s",
                len(labels), manifest["n_valid"], manifest["pct_valid"], cfg.horizons)

    return {"labels": labels, "coverage": coverage, "manifest": manifest}
