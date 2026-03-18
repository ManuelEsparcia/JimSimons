"""
data.price — OHLCV price pipeline: fetch → adjust → proxies → QC.

Pipeline: fetch_prices → adjust_prices → market_proxies → qc_prices

Shared infrastructure centralised here: OHLCV validation, canonical
event types, severity hierarchy, robust aggregation utilities.
"""
from __future__ import annotations

import enum
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

class PriceError(RuntimeError):
    """Base error for the price pipeline."""

class FetchError(PriceError):
    """Error fetching price data from provider."""

class AdjustmentError(PriceError):
    """Error adjusting prices for corporate actions."""

class QCError(PriceError):
    """Price QC error."""


# ---------------------------------------------------------------------------
# Severity / Gate (shared across all price submodules)
# ---------------------------------------------------------------------------

class Severity(str, enum.Enum):
    PASS = "PASS"
    INFO = "INFO"
    WARN = "WARN"
    FAIL = "FAIL"


class Gate(str, enum.Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


def severity_max(values) -> str:
    """Return the highest severity from a collection."""
    order = {"PASS": 0, "INFO": 1, "WARN": 2, "FAIL": 3}
    best = "PASS"
    for v in values:
        s = v.value if isinstance(v, Severity) else str(v).upper()
        if order.get(s, -1) > order.get(best, -1):
            best = s
    return best


# ---------------------------------------------------------------------------
# Canonical event types for corporate actions
# ---------------------------------------------------------------------------

class CanonicalEventType(str, enum.Enum):
    SPLIT = "split"
    REVERSE_SPLIT = "reverse_split"
    STOCK_DIVIDEND = "stock_dividend"
    CASH_ORDINARY = "cash_ordinary"
    CASH_SPECIAL = "cash_special"
    CASH_RETURN_OF_CAPITAL = "cash_return_of_capital"
    SPIN_OFF = "spin_off"
    RIGHTS_ISSUE = "rights_issue"
    MERGER = "merger"
    TENDER_OFFER = "tender_offer"
    DELISTING = "delisting"
    RELISTING = "relisting"
    TICKER_CHANGE = "ticker_change"
    SYMBOL_CHANGE = "symbol_change"
    EXCHANGE_CHANGE = "exchange_change"
    BANKRUPTCY = "bankruptcy"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# OHLCV column resolution and validation
# ---------------------------------------------------------------------------

OHLCV_CANDIDATES: dict[str, tuple[str, ...]] = {
    "date": ("date", "trade_date", "session_date", "Date"),
    "symbol": ("symbol", "ticker", "TICKER", "instrument_id"),
    "open": ("open", "adj_open", "Open", "px_open"),
    "high": ("high", "adj_high", "High", "px_high"),
    "low": ("low", "adj_low", "Low", "px_low"),
    "close": ("close", "adj_close", "Close", "px_close", "adjusted_close"),
    "volume": ("volume", "vol", "Volume", "trade_volume"),
}


def resolve_ohlcv_columns(df: pd.DataFrame) -> dict[str, str]:
    """Resolve canonical OHLCV column names from actual DataFrame columns."""
    lowered = {c.lower(): c for c in df.columns}
    resolved = {}
    for semantic, candidates in OHLCV_CANDIDATES.items():
        for c in candidates:
            if c.lower() in lowered:
                resolved[semantic] = lowered[c.lower()]
                break
    return resolved


def validate_ohlcv_geometry(df: pd.DataFrame, cols: dict[str, str]) -> pd.Series:
    """Return boolean mask: True where OHLCV row is geometrically valid.

    Invalid: H<L, any OHLC ≤ 0, volume < 0, any NaN.
    """
    o, h, l, c = (df[cols[k]] for k in ("open", "high", "low", "close"))
    v = df[cols["volume"]] if "volume" in cols else pd.Series(0, index=df.index)
    return (
        o.notna() & h.notna() & l.notna() & c.notna()
        & (o > 0) & (h > 0) & (l > 0) & (c > 0) & (v >= 0)
        & (h >= l)
    )


# ---------------------------------------------------------------------------
# Robust aggregation (used by market_proxies)
# ---------------------------------------------------------------------------

def winsorize_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Winsorise a series at the given quantiles."""
    valid = s.dropna()
    if len(valid) < 10:
        return s
    lo = valid.quantile(lower_q)
    hi = valid.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)


def robust_mean(s: pd.Series, lower_q: float = 0.05, upper_q: float = 0.95) -> float:
    """Trimmed mean: exclude values outside [Q_lo, Q_hi], then mean."""
    valid = s.dropna()
    if len(valid) < 5:
        return float(valid.mean()) if len(valid) > 0 else np.nan
    lo = valid.quantile(lower_q)
    hi = valid.quantile(upper_q)
    trimmed = valid[(valid >= lo) & (valid <= hi)]
    return float(trimmed.mean()) if len(trimmed) > 0 else float(valid.median())


def robust_std(s: pd.Series, lower_q: float = 0.05, upper_q: float = 0.95) -> float:
    """Trimmed std: exclude values outside [Q_lo, Q_hi], then std."""
    valid = s.dropna()
    if len(valid) < 5:
        return float(valid.std()) if len(valid) > 1 else np.nan
    lo = valid.quantile(lower_q)
    hi = valid.quantile(upper_q)
    trimmed = valid[(valid >= lo) & (valid <= hi)]
    return float(trimmed.std()) if len(trimmed) > 1 else float(valid.std())


# ---------------------------------------------------------------------------
# Shared IO and utilities
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def config_hash(cfg: Mapping[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, pd.Timestamp): return value.isoformat() if pd.notna(value) else None
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Mapping): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(x) for x in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)): return None
    return value


def read_dataframe(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet": return pd.read_parquet(p)
    if p.suffix == ".csv": return pd.read_csv(p)
    for ext in (".parquet", ".csv"):
        if p.with_suffix(ext).exists():
            return read_dataframe(p.with_suffix(ext))
    raise FileNotFoundError(f"No data file found: {p}")


def write_parquet_safe(df: pd.DataFrame, path: Path, *, allow_csv: bool = True, compression: str = "snappy") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False, compression=compression)
        return str(path)
    except Exception as e:
        if allow_csv:
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            return str(csv_path)
        raise


def write_json_safe(payload: Mapping[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, default=str), encoding="utf-8")
    return str(path)


def normalize_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def concat_findings(frames) -> pd.DataFrame:
    valid = [f for f in frames if isinstance(f, pd.DataFrame) and len(f) > 0]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True)
