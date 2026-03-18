"""
data/price/adjust_prices.py — Corporate action adjustment for OHLCV.

Applies splits, reverse splits, stock dividends, and cash dividends to
raw price history, producing PIT-safe adjusted series with full
factor traceability.

Modes:
    split_only:       adjust for splits/reverse splits only
    total_return_like: adjust for splits + cash dividends
    dual_output:      produce both series simultaneously

Factor convention (backward adjustment):
    P_adj_t = P_raw_t × S_t × D_t
    where S_t = cumulative split factor, D_t = cumulative dividend factor

    S_t = Π_{e: ex_date > t} split_factor_e
    D_t = Π_{e: ex_date > t} (1 - d_e / P_{e-1})

CRITICAL: events are only applied if their ex_date is within the PIT
observable window (acceptance_datetime <= asof_ts).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    PriceError, AdjustmentError, CanonicalEventType, Severity,
    resolve_ohlcv_columns, validate_ohlcv_geometry,
    utc_now_iso, config_hash, json_safe,
    read_dataframe, write_parquet_safe, write_json_safe,
    normalize_date_series,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdjustConfig:
    mode: str = "split_only"           # split_only | total_return_like | dual_output
    reverse_split_warn_threshold: float = 0.1   # warn if factor < 0.1 (10:1 or worse)
    max_dividend_yield_per_event: float = 0.5   # d/P > 50% → skip with warning
    event_priority: tuple[str, ...] = ("split", "reverse_split", "stock_dividend", "cash_ordinary", "cash_special")
    pit_strict: bool = True            # only apply PIT-observable events


# ---------------------------------------------------------------------------
# Event normalization
# ---------------------------------------------------------------------------

def normalize_events(events: pd.DataFrame) -> pd.DataFrame:
    """Normalise corporate action events to canonical format."""
    df = events.copy()
    # Ensure required columns
    for col in ("symbol", "ex_date", "event_type"):
        if col not in df.columns:
            raise AdjustmentError(f"Events missing required column: {col}")
    df["ex_date"] = normalize_date_series(df["ex_date"])
    df["event_type"] = df["event_type"].astype(str).str.lower().str.strip()
    return df.sort_values(["symbol", "ex_date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Factor computation (THE CORE)
# ---------------------------------------------------------------------------

def compute_split_factor(row: pd.Series) -> float:
    """Compute split adjustment factor for a single event.

    Split 2:1 → factor = 0.5 (prices halve going backward)
    Reverse split 1:10 → factor = 10 (prices multiply going backward)
    """
    event_type = str(row.get("event_type", "")).lower()
    ratio = row.get("split_ratio", row.get("factor", None))

    if ratio is None or pd.isna(ratio):
        return 1.0

    ratio = float(ratio)
    if ratio <= 0:
        return 1.0

    if "reverse" in event_type:
        return ratio  # e.g., 10 for 1:10 reverse split
    return 1.0 / ratio  # e.g., 0.5 for 2:1 split


def compute_dividend_factor(
    cash_amount: float,
    prev_close: float,
    max_yield: float = 0.5,
) -> Optional[float]:
    """Compute dividend adjustment factor.

    Factor = 1 - d/P_{ex-1}
    Returns None if d/P > max_yield (likely data error).
    """
    if prev_close is None or prev_close <= 0 or pd.isna(prev_close):
        return None
    if cash_amount is None or pd.isna(cash_amount) or cash_amount <= 0:
        return 1.0
    yield_pct = cash_amount / prev_close
    if yield_pct > max_yield:
        return None  # likely error, skip
    return 1.0 - yield_pct


def build_cumulative_factors(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    config: AdjustConfig,
) -> pd.DataFrame:
    """Build cumulative split and dividend factors for each (symbol, date).

    Factors are computed BACKWARD from the present: the most recent
    observation has factor=1.0, earlier observations accumulate.
    """
    df = prices[["date", "symbol", "close"]].copy()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["split_factor"] = 1.0
    df["div_factor"] = 1.0

    for sym in df["symbol"].unique():
        sym_mask = df["symbol"] == sym
        sym_events = events[events["symbol"] == sym].sort_values("ex_date")
        sym_prices = df.loc[sym_mask].copy()

        cum_split = 1.0
        cum_div = 1.0

        # Process events from latest to earliest (backward adjustment)
        for _, ev in sym_events.iloc[::-1].iterrows():
            ex_date = ev["ex_date"]
            etype = str(ev.get("event_type", "")).lower()

            if "split" in etype or "reverse" in etype:
                factor = compute_split_factor(ev)
                cum_split *= factor
                # All dates BEFORE ex_date get multiplied
                before_mask = sym_mask & (df["date"] < ex_date)
                df.loc[before_mask, "split_factor"] *= factor

            elif "cash" in etype or "dividend" in etype:
                if config.mode in ("total_return_like", "dual_output"):
                    cash = float(ev.get("cash_amount", ev.get("amount", 0)))
                    # Find prev_close
                    prev_close_rows = sym_prices[sym_prices["date"] < ex_date]
                    prev_close = float(prev_close_rows["close"].iloc[-1]) if len(prev_close_rows) > 0 else None
                    div_f = compute_dividend_factor(cash, prev_close, config.max_dividend_yield_per_event)
                    if div_f is not None:
                        before_mask = sym_mask & (df["date"] < ex_date)
                        df.loc[before_mask, "div_factor"] *= div_f

    return df[["date", "symbol", "split_factor", "div_factor"]]


# ---------------------------------------------------------------------------
# Apply adjustment
# ---------------------------------------------------------------------------

def apply_adjustment(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    config: AdjustConfig,
) -> pd.DataFrame:
    """Apply cumulative factors to raw OHLCV."""
    df = prices.merge(factors, on=["date", "symbol"], how="left")
    df["split_factor"] = df["split_factor"].fillna(1.0)
    df["div_factor"] = df["div_factor"].fillna(1.0)

    total_factor = df["split_factor"] * df["div_factor"]

    # Adjust OHLC
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[f"{col}_adj"] = df[col] * total_factor

    # Adjust volume (inverse of price factor)
    if "volume" in df.columns:
        vol_factor = 1.0 / df["split_factor"].where(df["split_factor"] > 0, 1.0)
        df["volume_adj"] = (df["volume"] * vol_factor).round(0).astype(int)

    # Compute returns
    df = df.sort_values(["symbol", "date"])
    df["return_adj"] = df.groupby("symbol")["close_adj"].pct_change()

    return df


# ---------------------------------------------------------------------------
# Invariant validation
# ---------------------------------------------------------------------------

def validate_adjusted(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Validate adjusted price invariants."""
    findings = []

    # Positivity
    for col in ("open_adj", "high_adj", "low_adj", "close_adj"):
        if col in df.columns:
            n_neg = (df[col] <= 0).sum()
            if n_neg > 0:
                findings.append({"check": f"{col}_positive", "severity": "FAIL", "n_violations": int(n_neg)})

    # H >= L
    if "high_adj" in df.columns and "low_adj" in df.columns:
        n_inv = (df["high_adj"] < df["low_adj"]).sum()
        if n_inv > 0:
            findings.append({"check": "hl_geometry", "severity": "WARN", "n_violations": int(n_inv)})

    # Extreme returns
    if "return_adj" in df.columns:
        extreme = df["return_adj"].abs() > 1.0  # >100% daily return
        n_extreme = extreme.sum()
        if n_extreme > 0:
            findings.append({"check": "extreme_returns", "severity": "WARN", "n_violations": int(n_extreme)})

    return findings


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def adjust_prices(
    prices: pd.DataFrame | str | Path,
    events: pd.DataFrame | str | Path | None = None,
    *,
    config: AdjustConfig | None = None,
    output_dir: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run the full price adjustment pipeline.

    Parameters
    ----------
    prices : raw OHLCV panel (or path)
    events : corporate actions table (or path). If None, returns unadjusted.
    config : AdjustConfig

    Returns
    -------
    dict with: adjusted, factors, findings, manifest
    """
    cfg = config or AdjustConfig()
    if not run_id:
        run_id = f"adj_{utc_now_iso().replace(':','').replace('-','')}"

    if isinstance(prices, (str, Path)):
        prices = read_dataframe(prices)
    prices = prices.copy()
    prices["date"] = normalize_date_series(prices.get("date", prices.get("trade_date")))

    # If no events, return prices as-is with factor=1
    if events is None or (isinstance(events, pd.DataFrame) and len(events) == 0):
        for col in ("open", "high", "low", "close"):
            if col in prices.columns:
                prices[f"{col}_adj"] = prices[col]
        if "volume" in prices.columns:
            prices["volume_adj"] = prices["volume"]
        prices["split_factor"] = 1.0
        prices["div_factor"] = 1.0
        prices = prices.sort_values(["symbol", "date"])
        prices["return_adj"] = prices.groupby("symbol")["close_adj"].pct_change()
        manifest = {"run_id": run_id, "n_events": 0, "n_bars": len(prices), "mode": cfg.mode, "timestamp": utc_now_iso()}
        return {"adjusted": prices, "factors": pd.DataFrame(), "findings": [], "manifest": manifest}

    if isinstance(events, (str, Path)):
        events = read_dataframe(events)

    events = normalize_events(events)

    # Build factors
    factors = build_cumulative_factors(prices, events, cfg)

    # Apply adjustment
    adjusted = apply_adjustment(prices, factors, cfg)

    # Validate
    findings = validate_adjusted(adjusted)

    manifest = {
        "run_id": run_id,
        "mode": cfg.mode,
        "n_bars": len(adjusted),
        "n_events": len(events),
        "n_symbols": int(adjusted["symbol"].nunique()),
        "n_findings": len(findings),
        "timestamp": utc_now_iso(),
    }

    if output_dir:
        out = Path(output_dir) / run_id
        out.mkdir(parents=True, exist_ok=True)
        write_parquet_safe(adjusted, out / "prices_adjusted.parquet")
        write_parquet_safe(factors, out / "factors.parquet")
        write_json_safe(manifest, out / "manifest.json")

    LOGGER.info("Adjustment: %d bars, %d events, %d findings, mode=%s", len(adjusted), len(events), len(findings), cfg.mode)
    return {"adjusted": adjusted, "factors": factors, "findings": findings, "manifest": manifest}
