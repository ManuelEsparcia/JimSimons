"""
data/price/market_proxies.py — Aggregate market state proxies.

Compresses the daily cross-section of adjusted prices into a small set
of robust, PIT-safe market state variables. These are NOT features for
the model — they're context variables for regime gating, risk monitoring,
and capacity assessment.

Core proxies (always computed):
    market_ret_{1,5,20}d      Equal-weight market return (robust mean)
    market_vol_{5,20}d        Realised volatility of market return
    dispersion_{5,20}d        Cross-sectional std of individual returns
    breadth_pct_up            Fraction of universe with positive return
    advance_decline_ratio     #advancing / #declining
    market_turnover_proxy     Robust median of turnover ratios

Exploratory proxies (optional):
    cross_section_corr        Mean pairwise correlation (rolling window)
    spread_proxy_agg          Aggregate bid-ask spread proxy
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    PriceError, Severity, severity_max,
    resolve_ohlcv_columns, validate_ohlcv_geometry,
    winsorize_series, robust_mean, robust_std,
    utc_now_iso, config_hash, json_safe,
    read_dataframe, write_parquet_safe, write_json_safe,
    normalize_date_series,
    LOGGER as _PARENT_LOGGER,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProxyConfig:
    min_universe_size: int = 20          # minimum symbols for valid proxy
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    return_windows: tuple[int, ...] = (1, 5, 20)
    vol_windows: tuple[int, ...] = (5, 20)
    enable_exploratory: bool = False
    corr_window: int = 63


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Add daily return column (close-to-close)."""
    df = prices.sort_values(["symbol", "date"]).copy()
    close_col = "close_adj" if "close_adj" in df.columns else "close"
    df["ret_1d"] = df.groupby("symbol")[close_col].pct_change()
    return df


def compute_proxies_for_date(
    date_panel: pd.DataFrame,
    date: Any,
    cfg: ProxyConfig,
) -> dict[str, Any]:
    """Compute all proxies for a single date's cross-section."""
    ret = date_panel["ret_1d"].dropna()
    n = len(ret)

    row: dict[str, Any] = {"date": date, "n_universe": n}

    if n < cfg.min_universe_size:
        row["severity"] = "WARN"
        row["valid"] = False
        return row

    # Winsorise returns
    ret_w = winsorize_series(ret, cfg.winsor_lower, cfg.winsor_upper)

    # Market return
    row["market_ret_1d"] = robust_mean(ret_w)

    # Breadth
    row["breadth_pct_up"] = float((ret > 0).mean())
    n_up = (ret > 0).sum()
    n_down = (ret < 0).sum()
    row["advance_decline_ratio"] = float(n_up / max(n_down, 1))

    # Dispersion
    row["dispersion_1d"] = float(ret_w.std())

    # Turnover (if available)
    if "turnover" in date_panel.columns:
        to = date_panel["turnover"].dropna()
        row["market_turnover_proxy"] = float(to.median()) if len(to) > 5 else np.nan
    elif "volume" in date_panel.columns and "close" in date_panel.columns:
        dv = (date_panel["volume"] * date_panel["close"]).dropna()
        row["market_turnover_proxy"] = float(dv.median()) if len(dv) > 5 else np.nan

    # Spread proxy
    if "high" in date_panel.columns and "low" in date_panel.columns and "close" in date_panel.columns:
        spread = ((date_panel["high"] - date_panel["low"]) / date_panel["close"]).dropna()
        row["spread_proxy_agg"] = float(spread.median()) if len(spread) > 5 else np.nan

    row["severity"] = "PASS"
    row["valid"] = True
    return row


def add_rolling_metrics(proxies_df: pd.DataFrame, cfg: ProxyConfig) -> pd.DataFrame:
    """Add rolling market vol and multi-day returns."""
    df = proxies_df.sort_values("date").copy()

    for w in cfg.return_windows:
        if w > 1:
            df[f"market_ret_{w}d"] = df["market_ret_1d"].rolling(w, min_periods=max(2, w//2)).sum()

    for w in cfg.vol_windows:
        df[f"market_vol_{w}d"] = df["market_ret_1d"].rolling(w, min_periods=max(2, w//2)).std() * np.sqrt(252)
        df[f"dispersion_{w}d"] = df["dispersion_1d"].rolling(w, min_periods=max(2, w//2)).mean()

    return df


def run_market_proxies(
    prices: pd.DataFrame | str | Path,
    *,
    universe: pd.DataFrame | None = None,
    config: ProxyConfig | None = None,
    output_dir: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build aggregate market state proxies."""
    cfg = config or ProxyConfig()
    if not run_id:
        run_id = f"proxy_{utc_now_iso().replace(':','').replace('-','')}"

    if isinstance(prices, (str, Path)):
        prices = read_dataframe(prices)

    df = compute_daily_returns(prices)

    # Apply universe filter if provided
    if universe is not None and "eligible" in universe.columns:
        df = df.merge(universe[["date", "symbol", "eligible"]], on=["date", "symbol"], how="left")
        df = df[df["eligible"].fillna(True).astype(bool)]

    # Compute per-date
    daily_rows = []
    for date, panel in df.groupby("date"):
        row = compute_proxies_for_date(panel, date, cfg)
        daily_rows.append(row)

    proxies = pd.DataFrame(daily_rows)

    # Add rolling
    if len(proxies) > 0:
        proxies = add_rolling_metrics(proxies, cfg)

    # Summary
    n_valid = int(proxies["valid"].sum()) if "valid" in proxies.columns else 0
    n_total = len(proxies)

    manifest = {
        "run_id": run_id,
        "n_dates": n_total,
        "n_valid": n_valid,
        "n_invalid": n_total - n_valid,
        "proxies_computed": [c for c in proxies.columns if c not in ("date", "n_universe", "severity", "valid")],
        "timestamp": utc_now_iso(),
    }

    if output_dir:
        out = Path(output_dir) / run_id
        out.mkdir(parents=True, exist_ok=True)
        write_parquet_safe(proxies, out / "market_proxies.parquet")
        write_json_safe(manifest, out / "manifest.json")

    LOGGER.info("Market proxies: %d dates (%d valid), %d proxies", n_total, n_valid, len(manifest["proxies_computed"]))
    return {"proxies": proxies, "manifest": manifest}
