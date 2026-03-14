from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

# Allow direct script execution: `python simons_smallcap_swing/data/price/qc_prices.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir, reference_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, validate_schema

SEVERITY_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}
ALLOWED_ADJUSTMENT_MODES = {"passthrough_mvp", "split_only"}

# QC v2 thresholds (explicitly fixed for reproducibility in Week 2)
SYMBOL_MISSING_WARN_THRESHOLD = 0.05
SESSION_WEAK_RATIO_THRESHOLD = 0.70
SESSION_ABRUPT_DROP_RATIO_THRESHOLD = 0.50
EXTREME_RETURN_ABS_THRESHOLD = 0.35
PRICE_RTOL = 1e-6
PRICE_ATOL = 1e-9
VOLUME_RTOL = 1e-6
VOLUME_ATOL = 1e-6

RAW_SCHEMA = DataSchema(
    name="price_raw_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("open", "float64", nullable=False),
        ColumnSpec("high", "float64", nullable=False),
        ColumnSpec("low", "float64", nullable=False),
        ColumnSpec("close", "float64", nullable=False),
        ColumnSpec("volume", "number", nullable=False),
    ),
    primary_key=("date", "instrument_id"),
    allow_extra_columns=True,
)

ADJUSTED_SCHEMA = DataSchema(
    name="price_adjusted_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("open_adj", "float64", nullable=False),
        ColumnSpec("high_adj", "float64", nullable=False),
        ColumnSpec("low_adj", "float64", nullable=False),
        ColumnSpec("close_adj", "float64", nullable=False),
        ColumnSpec("volume_adj", "number", nullable=False),
        ColumnSpec("adjustment_mode", "string", nullable=False),
    ),
    primary_key=("date", "instrument_id"),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class PriceQCResult:
    gate_status: str
    n_fail: int
    n_warn: int
    summary_path: Path
    row_level_path: Path
    symbol_level_path: Path
    session_level_path: Path
    corporate_actions_consistency_path: Path
    failures_path: Path
    manifest_path: Path


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains invalid dates.")
    return parsed.dt.normalize()


def _record_issue(
    issues: list[dict[str, Any]],
    *,
    check_name: str,
    severity: str,
    message: str,
    date: object = None,
    instrument_id: object = None,
    ticker: object = None,
    observed_value: object = None,
    threshold: object = None,
) -> None:
    issues.append(
        {
            "date": pd.Timestamp(date).normalize() if date is not None else pd.NaT,
            "instrument_id": instrument_id,
            "ticker": ticker,
            "check_name": check_name,
            "severity": severity,
            "observed_value": str(observed_value) if observed_value is not None else "",
            "threshold": str(threshold) if threshold is not None else "",
            "message": message,
        }
    )


def _max_severity(values: pd.Series) -> str:
    if values.empty:
        return "PASS"
    idx = int(values.map(SEVERITY_ORDER).max())
    for severity, order in SEVERITY_ORDER.items():
        if order == idx:
            return severity
    return "PASS"


def _severity_sort_value(severity: str) -> int:
    return int(SEVERITY_ORDER.get(str(severity), 0))


def _ensure_non_empty_frame(df: pd.DataFrame, placeholder: dict[str, Any]) -> pd.DataFrame:
    if not df.empty:
        return df
    return pd.DataFrame([placeholder])


def _resolve_corporate_actions_path(
    *,
    raw_source: Path,
    corporate_actions_path: str | Path | None,
) -> tuple[pd.DataFrame, Path | None, str]:
    if corporate_actions_path is not None:
        source = Path(corporate_actions_path).expanduser().resolve()
        return read_parquet(source), source, "provided"

    candidates = [
        raw_source.resolve().parents[1] / "universe" / "corporate_actions.parquet",
        data_dir() / "universe" / "corporate_actions.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return read_parquet(candidate), candidate, "default_found"

    return pd.DataFrame(), None, "missing"


def _extract_split_events_for_qc(
    corporate_actions: pd.DataFrame,
    *,
    raw_instruments: set[str],
    issues: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {
        "corporate_events_total": int(len(corporate_actions)),
        "ignored_non_split_events": 0,
        "ignored_unknown_instrument_events": 0,
        "ignored_invalid_split_factor": 0,
        "split_events_total": 0,
    }

    if corporate_actions.empty:
        return pd.DataFrame(columns=["instrument_id", "effective_date", "split_factor"]), stats

    required = {"instrument_id", "event_type", "effective_date"}
    missing = sorted(required - set(corporate_actions.columns))
    if missing:
        _record_issue(
            issues,
            check_name="corporate_actions_schema",
            severity="FAIL",
            message=f"corporate_actions missing required columns: {missing}",
        )
        return pd.DataFrame(columns=["instrument_id", "effective_date", "split_factor"]), stats

    frame = corporate_actions.copy()
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["event_type"] = frame["event_type"].astype(str).str.strip().str.lower()

    split_mask = frame["event_type"].isin({"split", "reverse_split"})
    stats["ignored_non_split_events"] = int((~split_mask).sum())
    split_events = frame.loc[split_mask].copy()
    if split_events.empty:
        return pd.DataFrame(columns=["instrument_id", "effective_date", "split_factor"]), stats

    if "split_factor" not in split_events.columns:
        _record_issue(
            issues,
            check_name="corporate_actions_schema",
            severity="FAIL",
            message="corporate_actions split events require split_factor column.",
        )
        return pd.DataFrame(columns=["instrument_id", "effective_date", "split_factor"]), stats

    split_events["effective_date"] = pd.to_datetime(
        split_events["effective_date"], errors="coerce"
    ).dt.normalize()
    invalid_date = split_events["effective_date"].isna()
    if invalid_date.any():
        _record_issue(
            issues,
            check_name="corporate_actions_split_effective_date",
            severity="FAIL",
            message="corporate_actions split events contain invalid effective_date.",
            observed_value=int(invalid_date.sum()),
        )
        split_events = split_events.loc[~invalid_date].copy()

    split_events["split_factor"] = pd.to_numeric(split_events["split_factor"], errors="coerce")
    invalid_factor = split_events["split_factor"].isna() | (split_events["split_factor"] <= 0)
    if invalid_factor.any():
        stats["ignored_invalid_split_factor"] = int(invalid_factor.sum())
        _record_issue(
            issues,
            check_name="corporate_actions_split_factor",
            severity="FAIL",
            message="corporate_actions split events require split_factor > 0.",
            observed_value=int(invalid_factor.sum()),
            threshold="> 0",
        )
        split_events = split_events.loc[~invalid_factor].copy()

    before_instr_filter = len(split_events)
    split_events = split_events[split_events["instrument_id"].isin(raw_instruments)].copy()
    stats["ignored_unknown_instrument_events"] = int(before_instr_filter - len(split_events))
    stats["split_events_total"] = int(len(split_events))

    split_events = split_events[["instrument_id", "effective_date", "split_factor"]]
    split_events = split_events.sort_values(["instrument_id", "effective_date"]).reset_index(drop=True)
    return split_events, stats


def _compute_expected_split_profile(raw_panel: pd.DataFrame, split_events: pd.DataFrame) -> pd.DataFrame:
    profile = raw_panel[["date", "instrument_id"]].copy()
    profile["expected_split_factor"] = 1.0
    profile["expected_split_events_count"] = 0

    if split_events.empty:
        return profile

    for instrument_id, idx in raw_panel.groupby("instrument_id", sort=False).groups.items():
        inst_events = split_events.loc[split_events["instrument_id"] == instrument_id]
        if inst_events.empty:
            continue

        day_factors = (
            inst_events.groupby("effective_date", as_index=False)["split_factor"]
            .prod()
            .sort_values("effective_date")
            .reset_index(drop=True)
        )

        event_dates = day_factors["effective_date"].to_numpy(dtype="datetime64[ns]")
        event_factors = day_factors["split_factor"].to_numpy(dtype="float64")
        suffix_factors = np.cumprod(event_factors[::-1])[::-1]

        row_index = list(idx)
        row_dates = raw_panel.loc[row_index, "date"].to_numpy(dtype="datetime64[ns]")
        insert_idx = np.searchsorted(event_dates, row_dates, side="right")

        factors = np.ones(len(row_dates), dtype="float64")
        valid = insert_idx < len(event_dates)
        if valid.any():
            factors[valid] = suffix_factors[insert_idx[valid]]

        counts = (len(event_dates) - insert_idx).astype("int64")
        profile.loc[row_index, "expected_split_factor"] = factors
        profile.loc[row_index, "expected_split_events_count"] = counts

    return profile


def _build_coverage_by_symbol(raw: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sessions_sorted = pd.DatetimeIndex(sorted(sessions))

    for instrument_id, sub in raw.groupby("instrument_id", sort=True):
        ticker = str(sub["ticker"].iloc[-1])
        obs_dates = pd.DatetimeIndex(sorted(pd.to_datetime(sub["date"]).dt.normalize().unique()))
        start = obs_dates.min()
        end = obs_dates.max()
        expected = sessions_sorted[(sessions_sorted >= start) & (sessions_sorted <= end)]
        n_obs = int(len(obs_dates))
        n_expected = int(len(expected))
        n_missing = max(0, n_expected - n_obs)
        pct_missing = 0.0 if n_expected == 0 else float(n_missing / n_expected)

        rows.append(
            {
                "instrument_id": str(instrument_id),
                "ticker": ticker,
                "n_rows": int(len(sub)),
                "n_sessions_observed": n_obs,
                "n_sessions_expected": n_expected,
                "n_missing_sessions": n_missing,
                "pct_missing_sessions": pct_missing,
            }
        )

    if not rows:
        return pd.DataFrame(
            [{
                "instrument_id": "__NONE__",
                "ticker": "__NONE__",
                "n_rows": 0,
                "n_sessions_observed": 0,
                "n_sessions_expected": 0,
                "n_missing_sessions": 0,
                "pct_missing_sessions": 0.0,
            }]
        )

    return pd.DataFrame(rows).sort_values("instrument_id").reset_index(drop=True)


def _build_coverage_by_session(raw: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    sessions_sorted = pd.DatetimeIndex(sorted(sessions))
    counts = (
        raw.groupby(raw["date"])["instrument_id"]
        .nunique()
        .rename("n_symbols")
        .reindex(sessions_sorted, fill_value=0)
    )
    median_non_zero = float(counts[counts > 0].median()) if (counts > 0).any() else 0.0
    ratio = pd.Series(0.0, index=counts.index) if median_non_zero <= 0 else counts.astype(float) / median_non_zero

    session_level = pd.DataFrame({
        "date": pd.DatetimeIndex(counts.index),
        "n_symbols": counts.to_numpy(dtype="int64"),
        "coverage_ratio_vs_median": ratio.to_numpy(dtype="float64"),
    })
    session_level["is_empty_session"] = session_level["n_symbols"] == 0
    session_level["is_weak_session"] = session_level["coverage_ratio_vs_median"] < SESSION_WEAK_RATIO_THRESHOLD
    prev_count = session_level["n_symbols"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        drop_ratio = session_level["n_symbols"] / prev_count
    session_level["is_abrupt_drop"] = (
        prev_count.notna() & (prev_count > 0) & (drop_ratio < SESSION_ABRUPT_DROP_RATIO_THRESHOLD)
    )
    return session_level

def run_price_qc(
    *,
    raw_prices_path: str | Path | None = None,
    adjusted_prices_path: str | Path | None = None,
    trading_calendar_path: str | Path | None = None,
    ticker_history_map_path: str | Path | None = None,
    corporate_actions_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    run_id: str = "price_qc_v2",
) -> PriceQCResult:
    logger = get_logger("data.price.qc_prices")
    price_base = data_dir() / "price"
    reference_base = reference_dir()

    raw_source = Path(raw_prices_path).expanduser().resolve() if raw_prices_path else (price_base / "raw_prices.parquet")
    adjusted_source = Path(adjusted_prices_path).expanduser().resolve() if adjusted_prices_path else (price_base / "adjusted_prices.parquet")
    calendar_source = Path(trading_calendar_path).expanduser().resolve() if trading_calendar_path else (reference_base / "trading_calendar.parquet")
    ticker_map_source = Path(ticker_history_map_path).expanduser().resolve() if ticker_history_map_path else (reference_base / "ticker_history_map.parquet")

    raw = read_parquet(raw_source)
    adjusted = read_parquet(adjusted_source)
    calendar = read_parquet(calendar_source)
    ticker_map = read_parquet(ticker_map_source)

    issues: list[dict[str, Any]] = []

    raw_schema_result = validate_schema(raw, RAW_SCHEMA)
    if not raw_schema_result.ok:
        for issue in raw_schema_result.issues:
            _record_issue(issues, check_name="schema_raw", severity="FAIL", message=issue.message, observed_value=issue.code)

    adjusted_schema_result = validate_schema(adjusted, ADJUSTED_SCHEMA)
    if not adjusted_schema_result.ok:
        for issue in adjusted_schema_result.issues:
            _record_issue(issues, check_name="schema_adjusted", severity="FAIL", message=issue.message, observed_value=issue.code)

    if raw.empty:
        _record_issue(issues, check_name="coverage", severity="FAIL", message="raw_prices is empty.")
    if adjusted.empty:
        _record_issue(issues, check_name="coverage", severity="FAIL", message="adjusted_prices is empty.")

    raw = raw.copy()
    adjusted = adjusted.copy()
    ticker_map = ticker_map.copy()
    calendar = calendar.copy()

    raw["date"] = _normalize_date(raw["date"], column="date")
    adjusted["date"] = _normalize_date(adjusted["date"], column="date")
    raw["instrument_id"] = raw["instrument_id"].astype(str)
    adjusted["instrument_id"] = adjusted["instrument_id"].astype(str)
    raw["ticker"] = raw["ticker"].astype(str).str.upper().str.strip()
    adjusted["ticker"] = adjusted["ticker"].astype(str).str.upper().str.strip()

    sessions = _normalize_date(calendar.loc[calendar["is_session"].astype(bool), "date"], column="date")
    sessions_in_scope = pd.DatetimeIndex([])
    if not raw.empty:
        sessions_in_scope = pd.DatetimeIndex(sorted(sessions[(sessions >= raw["date"].min()) & (sessions <= raw["date"].max())].unique()))
    valid_sessions = set(pd.DatetimeIndex(sessions).tolist())

    ticker_map["instrument_id"] = ticker_map["instrument_id"].astype(str)
    ticker_map["ticker"] = ticker_map["ticker"].astype(str).str.upper().str.strip()
    ticker_map["start_date"] = _normalize_date(ticker_map["start_date"], column="start_date")
    ticker_map["end_date"] = pd.to_datetime(ticker_map["end_date"], errors="coerce").dt.normalize()

    invalid_intervals = ticker_map["end_date"].notna() & (ticker_map["end_date"] < ticker_map["start_date"])
    if invalid_intervals.any():
        _record_issue(issues, check_name="ticker_history_interval", severity="FAIL", message="ticker_history_map has rows with end_date < start_date.", observed_value=int(invalid_intervals.sum()))

    for col in ("open", "high", "low", "close", "volume"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    for col in ("open_adj", "high_adj", "low_adj", "close_adj", "volume_adj"):
        adjusted[col] = pd.to_numeric(adjusted[col], errors="coerce")

    for col in ["date", "instrument_id", "ticker", "open", "high", "low", "close", "volume"]:
        null_count = int(raw[col].isna().sum())
        if null_count > 0:
            _record_issue(issues, check_name="critical_nulls_raw", severity="FAIL", message=f"raw column '{col}' has null values.", observed_value=null_count, threshold=0)

    for col in ["date", "instrument_id", "ticker", "open_adj", "high_adj", "low_adj", "close_adj", "volume_adj"]:
        null_count = int(adjusted[col].isna().sum())
        if null_count > 0:
            _record_issue(issues, check_name="critical_nulls_adjusted", severity="FAIL", message=f"adjusted column '{col}' has null values.", observed_value=null_count, threshold=0)

    dup_raw = raw.duplicated(["date", "instrument_id"], keep=False)
    if dup_raw.any():
        sample = raw.loc[dup_raw, ["date", "instrument_id", "ticker"]].head(300)
        for row in sample.itertuples(index=False):
            _record_issue(issues, check_name="primary_key_raw", severity="FAIL", message="Duplicate (date, instrument_id) row in raw_prices.", date=row.date, instrument_id=row.instrument_id, ticker=row.ticker)

    dup_adj = adjusted.duplicated(["date", "instrument_id"], keep=False)
    if dup_adj.any():
        sample = adjusted.loc[dup_adj, ["date", "instrument_id", "ticker"]].head(300)
        for row in sample.itertuples(index=False):
            _record_issue(issues, check_name="primary_key_adjusted", severity="FAIL", message="Duplicate (date, instrument_id) row in adjusted_prices.", date=row.date, instrument_id=row.instrument_id, ticker=row.ticker)

    raw_sorted = raw.sort_values(["instrument_id", "date"]).copy()
    bad_raw_order = raw.groupby("instrument_id", sort=True)["date"].apply(lambda s: not s.is_monotonic_increasing)
    for instrument_id in bad_raw_order[bad_raw_order].index.tolist():
        ticker = str(raw_sorted.loc[raw_sorted["instrument_id"] == instrument_id, "ticker"].iloc[-1])
        _record_issue(issues, check_name="temporal_order_raw", severity="FAIL", message="raw_prices dates are not monotonic increasing for instrument.", instrument_id=instrument_id, ticker=ticker)

    adjusted_sorted = adjusted.sort_values(["instrument_id", "date"]).copy()
    bad_adj_order = adjusted.groupby("instrument_id", sort=True)["date"].apply(lambda s: not s.is_monotonic_increasing)
    for instrument_id in bad_adj_order[bad_adj_order].index.tolist():
        ticker = str(adjusted_sorted.loc[adjusted_sorted["instrument_id"] == instrument_id, "ticker"].iloc[-1])
        _record_issue(issues, check_name="temporal_order_adjusted", severity="FAIL", message="adjusted_prices dates are not monotonic increasing for instrument.", instrument_id=instrument_id, ticker=ticker)

    bad_raw_ohlc = (raw["high"] < raw[["open", "close", "low"]].max(axis=1)) | (raw["low"] > raw[["open", "close", "high"]].min(axis=1))
    for row in raw.loc[bad_raw_ohlc, ["date", "instrument_id", "ticker", "open", "high", "low", "close"]].head(300).itertuples(index=False):
        _record_issue(issues, check_name="ohlc_geometry_raw", severity="FAIL", message="raw OHLC geometry invalid.", date=row.date, instrument_id=row.instrument_id, ticker=row.ticker, observed_value=f"o={row.open},h={row.high},l={row.low},c={row.close}")

    bad_adj_ohlc = (adjusted["high_adj"] < adjusted[["open_adj", "close_adj", "low_adj"]].max(axis=1)) | (adjusted["low_adj"] > adjusted[["open_adj", "close_adj", "high_adj"]].min(axis=1))
    for row in adjusted.loc[bad_adj_ohlc, ["date", "instrument_id", "ticker", "open_adj", "high_adj", "low_adj", "close_adj"]].head(300).itertuples(index=False):
        _record_issue(issues, check_name="ohlc_geometry_adjusted", severity="FAIL", message="adjusted OHLC geometry invalid.", date=row.date, instrument_id=row.instrument_id, ticker=row.ticker, observed_value=f"o={row.open_adj},h={row.high_adj},l={row.low_adj},c={row.close_adj}")

    raw_nonpositive = (raw[["open", "high", "low", "close"]] <= 0).any(axis=1)
    for row in raw.loc[raw_nonpositive, ["date", "instrument_id", "ticker"]].head(300).itertuples(index=False):
        _record_issue(issues, check_name="price_nonpositive_raw", severity="FAIL", message="raw OHLC has non-positive values.", date=row.date, instrument_id=row.instrument_id, ticker=row.ticker)

    adj_nonpositive = (adjusted[["open_adj", "high_adj", "low_adj", "close_adj"]] <= 0).any(axis=1)
    for row in adjusted.loc[adj_nonpositive, ["date", "instrument_id", "ticker"]].head(300).itertuples(index=False):
        _record_issue(issues, check_name="price_nonpositive_adjusted", severity="FAIL", message="adjusted OHLC has non-positive values.", date=row.date, instrument_id=row.instrument_id, ticker=row.ticker)

    for row in raw.loc[raw["volume"] < 0, ["date", "instrument_id", "ticker", "volume"]].head(300).itertuples(index=False):
        _record_issue(issues, check_name="volume_negative_raw", severity="FAIL", message="raw volume is negative.", date=row.date, instrument_id=row.instrument_id, ticker=row.ticker, observed_value=row.volume)

    for row in adjusted.loc[adjusted["volume_adj"] < 0, ["date", "instrument_id", "ticker", "volume_adj"]].head(300).itertuples(index=False):
        _record_issue(issues, check_name="volume_negative_adjusted", severity="FAIL", message="adjusted volume is negative.", date=row.date, instrument_id=row.instrument_id, ticker=row.ticker, observed_value=row.volume_adj)

    for row in raw.loc[~raw["date"].isin(valid_sessions), ["date", "instrument_id", "ticker"]].head(300).itertuples(index=False):
        _record_issue(issues, check_name="calendar_membership_raw", severity="FAIL", message="raw date is outside trading sessions.", date=row.date, instrument_id=row.instrument_id, ticker=row.ticker)

    for row in adjusted.loc[~adjusted["date"].isin(valid_sessions), ["date", "instrument_id", "ticker"]].head(300).itertuples(index=False):
        _record_issue(issues, check_name="calendar_membership_adjusted", severity="FAIL", message="adjusted date is outside trading sessions.", date=row.date, instrument_id=row.instrument_id, ticker=row.ticker)

    def _pit_check(frame: pd.DataFrame, check_name: str) -> None:
        temp = frame.reset_index(drop=True).copy()
        temp["__row_id"] = temp.index
        merged = temp.merge(ticker_map[["instrument_id", "ticker", "start_date", "end_date"]], on=["instrument_id", "ticker"], how="left")
        merged["pit_valid"] = merged["start_date"].notna() & (merged["date"] >= merged["start_date"]) & (merged["end_date"].isna() | (merged["date"] <= merged["end_date"]))
        valid_by_row = merged.groupby("__row_id", as_index=True)["pit_valid"].any()
        invalid = temp.loc[temp["__row_id"].isin(valid_by_row.index[~valid_by_row])]
        for row in invalid.head(300).itertuples(index=False):
            _record_issue(issues, check_name=check_name, severity="FAIL", message="ticker/date is not valid per ticker_history_map PIT intervals.", date=row.date, instrument_id=row.instrument_id, ticker=row.ticker)

    _pit_check(raw[["date", "instrument_id", "ticker"]], "ticker_pit_consistency_raw")
    _pit_check(adjusted[["date", "instrument_id", "ticker"]], "ticker_pit_consistency_adjusted")

    adjusted_merge_columns = ["date", "instrument_id", "ticker", "open_adj", "high_adj", "low_adj", "close_adj", "volume_adj", "adjustment_mode"]
    for optional_col in ("cumulative_split_factor", "applied_split_events_count"):
        if optional_col in adjusted.columns:
            adjusted_merge_columns.append(optional_col)

    raw_vs_adj = raw.merge(adjusted[adjusted_merge_columns], on=["date", "instrument_id", "ticker"], how="inner")
    if len(raw_vs_adj) != len(raw):
        _record_issue(issues, check_name="raw_adjusted_alignment", severity="FAIL", message="adjusted does not align one-to-one with raw on (date, instrument_id, ticker).", observed_value=f"matched={len(raw_vs_adj)}, raw={len(raw)}")

    observed_adjustment_modes = sorted(adjusted["adjustment_mode"].dropna().astype(str).unique().tolist())
    adjustment_mode_detected = observed_adjustment_modes[0] if len(observed_adjustment_modes) == 1 else "mixed"

    if len(observed_adjustment_modes) != 1:
        _record_issue(issues, check_name="adjustment_mode", severity="FAIL", message="adjusted_prices must contain exactly one adjustment_mode.", observed_value=observed_adjustment_modes, threshold="single mode")
    else:
        mode = observed_adjustment_modes[0]
        if mode not in ALLOWED_ADJUSTMENT_MODES:
            _record_issue(issues, check_name="adjustment_mode", severity="FAIL", message="Unsupported adjustment_mode in adjusted_prices.", observed_value=mode, threshold="passthrough_mvp|split_only")
        elif mode == "passthrough_mvp":
            mismatch = (
                ~np.isclose(raw_vs_adj["open"], raw_vs_adj["open_adj"], rtol=PRICE_RTOL, atol=PRICE_ATOL)
                | ~np.isclose(raw_vs_adj["high"], raw_vs_adj["high_adj"], rtol=PRICE_RTOL, atol=PRICE_ATOL)
                | ~np.isclose(raw_vs_adj["low"], raw_vs_adj["low_adj"], rtol=PRICE_RTOL, atol=PRICE_ATOL)
                | ~np.isclose(raw_vs_adj["close"], raw_vs_adj["close_adj"], rtol=PRICE_RTOL, atol=PRICE_ATOL)
                | ~np.isclose(raw_vs_adj["volume"].astype(float), raw_vs_adj["volume_adj"].astype(float), rtol=VOLUME_RTOL, atol=VOLUME_ATOL)
            )
            for row in raw_vs_adj.loc[mismatch, ["date", "instrument_id", "ticker"]].head(300).itertuples(index=False):
                _record_issue(issues, check_name="adjusted_passthrough_consistency", severity="FAIL", message="Adjusted row differs from raw under passthrough mode.", date=row.date, instrument_id=row.instrument_id, ticker=row.ticker)
        elif mode == "split_only":
            if "cumulative_split_factor" not in raw_vs_adj.columns:
                _record_issue(issues, check_name="split_factor_presence", severity="FAIL", message="split_only mode requires cumulative_split_factor column.")
            if "applied_split_events_count" not in raw_vs_adj.columns:
                _record_issue(issues, check_name="split_event_count_presence", severity="FAIL", message="split_only mode requires applied_split_events_count column.")

            if "cumulative_split_factor" in raw_vs_adj.columns:
                observed_factor = pd.to_numeric(raw_vs_adj["cumulative_split_factor"], errors="coerce")
                invalid_factor = observed_factor.isna() | (observed_factor <= 0)
                if invalid_factor.any():
                    _record_issue(issues, check_name="split_factor_validity", severity="FAIL", message="cumulative_split_factor must be > 0.", observed_value=int(invalid_factor.sum()), threshold="> 0")

                valid = ~invalid_factor
                if valid.any():
                    factor = observed_factor.loc[valid]
                    mismatch = (
                        ~np.isclose(raw_vs_adj.loc[valid, "open_adj"], raw_vs_adj.loc[valid, "open"] * factor, rtol=PRICE_RTOL, atol=PRICE_ATOL)
                        | ~np.isclose(raw_vs_adj.loc[valid, "high_adj"], raw_vs_adj.loc[valid, "high"] * factor, rtol=PRICE_RTOL, atol=PRICE_ATOL)
                        | ~np.isclose(raw_vs_adj.loc[valid, "low_adj"], raw_vs_adj.loc[valid, "low"] * factor, rtol=PRICE_RTOL, atol=PRICE_ATOL)
                        | ~np.isclose(raw_vs_adj.loc[valid, "close_adj"], raw_vs_adj.loc[valid, "close"] * factor, rtol=PRICE_RTOL, atol=PRICE_ATOL)
                        | ~np.isclose(raw_vs_adj.loc[valid, "volume_adj"], raw_vs_adj.loc[valid, "volume"] / factor, rtol=VOLUME_RTOL, atol=VOLUME_ATOL)
                    )
                    for row in raw_vs_adj.loc[raw_vs_adj.loc[valid].index[mismatch], ["date", "instrument_id", "ticker"]].head(300).itertuples(index=False):
                        _record_issue(issues, check_name="adjusted_split_only_consistency", severity="FAIL", message="Adjusted values are inconsistent with split_only factor formula.", date=row.date, instrument_id=row.instrument_id, ticker=row.ticker)

            if "applied_split_events_count" in raw_vs_adj.columns:
                split_count = pd.to_numeric(raw_vs_adj["applied_split_events_count"], errors="coerce")
                invalid_count = split_count.isna() | (split_count < 0)
                if invalid_count.any():
                    _record_issue(issues, check_name="split_event_count_validity", severity="FAIL", message="applied_split_events_count must be non-negative.", observed_value=int(invalid_count.sum()), threshold=">= 0")

    if not raw_vs_adj.empty:
        return_frame = raw_vs_adj[["date", "instrument_id", "ticker", "close", "close_adj"]].copy()
        if "applied_split_events_count" in raw_vs_adj.columns:
            return_frame["applied_split_events_count"] = pd.to_numeric(
                raw_vs_adj["applied_split_events_count"], errors="coerce"
            ).fillna(0)
        else:
            return_frame["applied_split_events_count"] = 0
        return_frame["close_for_return"] = return_frame["close_adj"].astype(float) if adjustment_mode_detected == "split_only" else return_frame["close"].astype(float)
        return_frame.sort_values(["instrument_id", "date"], inplace=True)
        return_frame["ret_1d"] = return_frame.groupby("instrument_id", sort=False)["close_for_return"].pct_change()
        return_frame["is_split_boundary"] = (
            return_frame.groupby("instrument_id", sort=False)["applied_split_events_count"]
            .diff()
            .fillna(0)
            != 0
        )

        extreme_mask = (
            (return_frame["ret_1d"].abs() > EXTREME_RETURN_ABS_THRESHOLD)
            & ~return_frame["is_split_boundary"].astype(bool)
        )
        for row in return_frame.loc[extreme_mask, ["date", "instrument_id", "ticker", "ret_1d"]].head(400).itertuples(index=False):
            _record_issue(issues, check_name="extreme_return", severity="WARN", message="Suspiciously large 1D return (manual review suggested).", date=row.date, instrument_id=row.instrument_id, ticker=row.ticker, observed_value=float(row.ret_1d), threshold=EXTREME_RETURN_ABS_THRESHOLD)

    symbol_coverage = _build_coverage_by_symbol(raw, sessions_in_scope)
    for row in symbol_coverage.loc[symbol_coverage["pct_missing_sessions"] > SYMBOL_MISSING_WARN_THRESHOLD].itertuples(index=False):
        _record_issue(issues, check_name="symbol_coverage_low", severity="WARN", message="Symbol has missing sessions above warning threshold.", instrument_id=row.instrument_id, ticker=row.ticker, observed_value=float(row.pct_missing_sessions), threshold=SYMBOL_MISSING_WARN_THRESHOLD)

    session_level = _build_coverage_by_session(raw, sessions_in_scope)
    for row in session_level.loc[session_level["is_empty_session"]].itertuples(index=False):
        _record_issue(issues, check_name="session_empty", severity="FAIL", message="Trading session has zero symbols in raw prices.", date=row.date, observed_value=int(row.n_symbols), threshold="> 0")
    for row in session_level.loc[(~session_level["is_empty_session"]) & session_level["is_weak_session"]].itertuples(index=False):
        _record_issue(issues, check_name="session_coverage_weak", severity="WARN", message="Session coverage is weak relative to median session breadth.", date=row.date, observed_value=float(row.coverage_ratio_vs_median), threshold=SESSION_WEAK_RATIO_THRESHOLD)
    for row in session_level.loc[session_level["is_abrupt_drop"]].itertuples(index=False):
        _record_issue(issues, check_name="session_coverage_abrupt_drop", severity="WARN", message="Session coverage shows abrupt drop vs previous session.", date=row.date, observed_value=int(row.n_symbols), threshold=SESSION_ABRUPT_DROP_RATIO_THRESHOLD)

    corporate_actions, corporate_source, corporate_source_status = _resolve_corporate_actions_path(raw_source=raw_source, corporate_actions_path=corporate_actions_path)
    split_events, corporate_stats = _extract_split_events_for_qc(corporate_actions, raw_instruments=set(raw["instrument_id"].astype(str).unique().tolist()), issues=issues)

    corporate_consistency = raw_vs_adj[["date", "instrument_id", "ticker"]].copy()
    corporate_consistency["expected_split_factor"] = 1.0
    corporate_consistency["expected_split_events_count"] = 0
    corporate_consistency["observed_split_factor"] = np.nan
    corporate_consistency["observed_split_events_count"] = np.nan
    corporate_consistency["is_consistent"] = True
    corporate_consistency["reason"] = "no_split_events"

    if adjustment_mode_detected == "split_only" and "cumulative_split_factor" in raw_vs_adj.columns:
        # Keep strict row-level 1:1 alignment with raw_vs_adj to remain robust even if
        # duplicates are present (duplicates are validated by dedicated PK checks).
        profile = _compute_expected_split_profile(
            raw_vs_adj[["date", "instrument_id"]], split_events
        ).reset_index(drop=True)
        corporate_consistency = corporate_consistency.reset_index(drop=True)
        corporate_consistency["expected_split_factor"] = pd.to_numeric(
            profile["expected_split_factor"], errors="coerce"
        ).fillna(1.0).to_numpy(dtype=float)
        corporate_consistency["expected_split_events_count"] = pd.to_numeric(
            profile["expected_split_events_count"], errors="coerce"
        ).fillna(0).to_numpy(dtype=int)

        corporate_consistency["observed_split_factor"] = pd.to_numeric(
            raw_vs_adj["cumulative_split_factor"], errors="coerce"
        ).to_numpy(dtype=float)
        if "applied_split_events_count" in raw_vs_adj.columns:
            corporate_consistency["observed_split_events_count"] = pd.to_numeric(
                raw_vs_adj["applied_split_events_count"], errors="coerce"
            ).to_numpy(dtype=float)

        factor_consistent = np.isclose(corporate_consistency["observed_split_factor"].astype(float), corporate_consistency["expected_split_factor"].astype(float), rtol=PRICE_RTOL, atol=PRICE_ATOL)
        count_consistent = np.ones(len(corporate_consistency), dtype=bool)
        if "applied_split_events_count" in raw_vs_adj.columns:
            count_consistent = np.isclose(corporate_consistency["observed_split_events_count"].astype(float), corporate_consistency["expected_split_events_count"].astype(float), rtol=0.0, atol=0.0)

        corporate_consistency["is_consistent"] = factor_consistent & count_consistent
        corporate_consistency["reason"] = np.where(corporate_consistency["is_consistent"], "ok", "split_factor_or_count_mismatch_vs_corporate_actions")

        for row in corporate_consistency.loc[~corporate_consistency["is_consistent"], ["date", "instrument_id", "ticker", "expected_split_factor", "observed_split_factor", "expected_split_events_count", "observed_split_events_count"]].head(400).itertuples(index=False):
            _record_issue(issues, check_name="corporate_actions_split_factor_consistency", severity="FAIL", message="Adjusted split factor/count is inconsistent with corporate_actions split events.", date=row.date, instrument_id=row.instrument_id, ticker=row.ticker, observed_value=f"obs_factor={row.observed_split_factor},exp_factor={row.expected_split_factor},obs_count={row.observed_split_events_count},exp_count={row.expected_split_events_count}")

    if corporate_source_status == "missing":
        corporate_consistency = _ensure_non_empty_frame(corporate_consistency, {
            "date": pd.NaT,
            "instrument_id": "__NONE__",
            "ticker": "__NONE__",
            "expected_split_factor": 1.0,
            "expected_split_events_count": 0,
            "observed_split_factor": 1.0,
            "observed_split_events_count": 0,
            "is_consistent": True,
            "reason": "corporate_actions_unavailable",
        })

    if not issues:
        _record_issue(issues, check_name="qc_summary", severity="PASS", message="No WARN/FAIL issues detected.")

    row_level = _ensure_non_empty_frame(pd.DataFrame(issues), {
        "date": pd.NaT,
        "instrument_id": "__NONE__",
        "ticker": "__NONE__",
        "check_name": "qc_summary",
        "severity": "PASS",
        "observed_value": "",
        "threshold": "",
        "message": "No issues.",
    })

    issue_symbol = row_level[row_level["instrument_id"].notna() & (row_level["instrument_id"] != "")]
    symbol_issue_counts = issue_symbol.groupby(["instrument_id", "severity"], as_index=False).size().pivot(index="instrument_id", columns="severity", values="size").fillna(0)
    if not symbol_issue_counts.empty:
        for sev in ("FAIL", "WARN"):
            if sev not in symbol_issue_counts.columns:
                symbol_issue_counts[sev] = 0
        symbol_issue_counts = symbol_issue_counts.reset_index()
    else:
        symbol_issue_counts = pd.DataFrame(columns=["instrument_id", "FAIL", "WARN"])

    extreme_per_symbol = row_level.loc[row_level["check_name"] == "extreme_return"].groupby("instrument_id", as_index=False).size().rename(columns={"size": "n_extreme_return_flags"})
    split_fail_per_symbol = row_level.loc[row_level["check_name"].isin(["adjusted_split_only_consistency", "corporate_actions_split_factor_consistency"])].groupby("instrument_id", as_index=False).size().rename(columns={"size": "n_split_consistency_failures"})

    symbol_level = symbol_coverage.merge(symbol_issue_counts, on="instrument_id", how="left").merge(extreme_per_symbol, on="instrument_id", how="left").merge(split_fail_per_symbol, on="instrument_id", how="left")
    symbol_level["FAIL"] = symbol_level.get("FAIL", 0).fillna(0).astype(int)
    symbol_level["WARN"] = symbol_level.get("WARN", 0).fillna(0).astype(int)
    symbol_level["n_extreme_return_flags"] = symbol_level["n_extreme_return_flags"].fillna(0).astype(int)
    symbol_level["n_split_consistency_failures"] = symbol_level["n_split_consistency_failures"].fillna(0).astype(int)
    symbol_level["severity_max"] = np.where(symbol_level["FAIL"] > 0, "FAIL", np.where(symbol_level["WARN"] > 0, "WARN", "PASS"))
    symbol_level = symbol_level.rename(columns={"FAIL": "n_fail_rows", "WARN": "n_warn_rows"})
    symbol_level = symbol_level[["instrument_id", "ticker", "n_rows", "n_sessions_observed", "n_sessions_expected", "n_missing_sessions", "pct_missing_sessions", "n_fail_rows", "n_warn_rows", "n_extreme_return_flags", "n_split_consistency_failures", "severity_max"]].sort_values("instrument_id").reset_index(drop=True)
    symbol_level = _ensure_non_empty_frame(symbol_level, {
        "instrument_id": "__NONE__",
        "ticker": "__NONE__",
        "n_rows": 0,
        "n_sessions_observed": 0,
        "n_sessions_expected": 0,
        "n_missing_sessions": 0,
        "pct_missing_sessions": 0.0,
        "n_fail_rows": 0,
        "n_warn_rows": 0,
        "n_extreme_return_flags": 0,
        "n_split_consistency_failures": 0,
        "severity_max": "PASS",
    })

    issue_session = row_level[row_level["date"].notna()].copy()
    if not issue_session.empty:
        issue_session["date"] = pd.to_datetime(issue_session["date"], errors="coerce").dt.normalize()
    session_counts = issue_session.groupby(["date", "severity"], as_index=False).size().pivot(index="date", columns="severity", values="size").fillna(0) if not issue_session.empty else pd.DataFrame()
    if not session_counts.empty:
        for sev in ("FAIL", "WARN"):
            if sev not in session_counts.columns:
                session_counts[sev] = 0
        session_counts = session_counts.reset_index()
    else:
        session_counts = pd.DataFrame(columns=["date", "FAIL", "WARN"])

    session_level = session_level.merge(session_counts, on="date", how="left")
    session_level["FAIL"] = session_level.get("FAIL", 0).fillna(0).astype(int)
    session_level["WARN"] = session_level.get("WARN", 0).fillna(0).astype(int)
    session_level["severity_max"] = np.where(session_level["FAIL"] > 0, "FAIL", np.where(session_level["WARN"] > 0, "WARN", "PASS"))
    session_level = session_level.rename(columns={"FAIL": "n_fail_rows", "WARN": "n_warn_rows"})
    session_level = session_level[["date", "n_symbols", "coverage_ratio_vs_median", "is_empty_session", "is_weak_session", "is_abrupt_drop", "n_fail_rows", "n_warn_rows", "severity_max"]].sort_values("date").reset_index(drop=True)
    session_level = _ensure_non_empty_frame(session_level, {
        "date": pd.NaT,
        "n_symbols": 0,
        "coverage_ratio_vs_median": 0.0,
        "is_empty_session": False,
        "is_weak_session": False,
        "is_abrupt_drop": False,
        "n_fail_rows": 0,
        "n_warn_rows": 0,
        "severity_max": "PASS",
    })

    failures = _ensure_non_empty_frame(row_level[row_level["severity"].isin(["FAIL", "WARN"])].copy(), {
        "date": pd.NaT,
        "instrument_id": "__NONE__",
        "ticker": "__NONE__",
        "check_name": "qc_summary",
        "severity": "PASS",
        "observed_value": "",
        "threshold": "",
        "message": "No issues.",
    })

    gate_status = _max_severity(row_level["severity"])
    severity_counts = Counter(row_level["severity"].tolist())
    n_fail = int(severity_counts.get("FAIL", 0))
    n_warn = int(severity_counts.get("WARN", 0))

    qc_root = Path(output_dir).expanduser().resolve() if output_dir else (data_dir() / "price" / "qc" / run_id)
    qc_root.mkdir(parents=True, exist_ok=True)

    row_level_path = write_parquet(row_level, qc_root / "price_qc_row_level.parquet", schema_name="price_qc_row_level_v2", run_id=run_id)
    symbol_level_path = write_parquet(symbol_level, qc_root / "price_qc_symbol_level.parquet", schema_name="price_qc_symbol_level_v2", run_id=run_id)
    session_level_path = write_parquet(session_level, qc_root / "price_qc_session_level.parquet", schema_name="price_qc_session_level_v2", run_id=run_id)
    corporate_consistency_path = write_parquet(_ensure_non_empty_frame(corporate_consistency, {
        "date": pd.NaT,
        "instrument_id": "__NONE__",
        "ticker": "__NONE__",
        "expected_split_factor": 1.0,
        "expected_split_events_count": 0,
        "observed_split_factor": 1.0,
        "observed_split_events_count": 0,
        "is_consistent": True,
        "reason": "not_available",
    }), qc_root / "price_qc_corporate_actions_consistency.parquet", schema_name="price_qc_corporate_actions_consistency_v2", run_id=run_id)
    failures_path = write_parquet(failures, qc_root / "price_qc_failures.parquet", schema_name="price_qc_failures_v2", run_id=run_id)

    source_modes_detected = sorted(raw["source_mode"].dropna().astype(str).unique().tolist()) if "source_mode" in raw.columns else []
    min_symbol_coverage = float(symbol_level["pct_missing_sessions"].max()) if not symbol_level.empty else 0.0
    median_symbol_coverage = float(symbol_level["pct_missing_sessions"].median()) if not symbol_level.empty else 0.0
    n_extreme_return_flags = int((row_level["check_name"] == "extreme_return").sum())
    n_split_consistency_failures = int(row_level["check_name"].isin(["adjusted_split_only_consistency", "corporate_actions_split_factor_consistency"]).sum())

    symbol_rank = symbol_level.copy()
    symbol_rank["severity_rank"] = symbol_rank["severity_max"].map(_severity_sort_value)
    symbol_rank = symbol_rank.sort_values(["severity_rank", "n_fail_rows", "n_warn_rows", "pct_missing_sessions"], ascending=[False, False, False, False])
    worst_symbol = str(symbol_rank.iloc[0]["instrument_id"]) if not symbol_rank.empty else "__NONE__"

    session_rank = session_level.copy()
    session_rank["severity_rank"] = session_rank["severity_max"].map(_severity_sort_value)
    session_rank = session_rank.sort_values(["severity_rank", "n_fail_rows", "n_warn_rows", "coverage_ratio_vs_median"], ascending=[False, False, False, True])
    worst_session = pd.Timestamp(session_rank.iloc[0]["date"]).strftime("%Y-%m-%d") if not session_rank.empty and pd.notna(session_rank.iloc[0]["date"]) else "__NONE__"

    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "gate_status": gate_status,
        "n_rows_raw": int(len(raw)),
        "n_rows_adjusted": int(len(adjusted)),
        "n_symbols": int(raw["instrument_id"].nunique()) if not raw.empty else 0,
        "n_sessions": int(len(sessions_in_scope)),
        "n_fail_rows": n_fail,
        "n_warn_rows": n_warn,
        "worst_symbol": worst_symbol,
        "worst_session": worst_session,
        "min_symbol_coverage": min_symbol_coverage,
        "median_symbol_coverage": median_symbol_coverage,
        "n_extreme_return_flags": n_extreme_return_flags,
        "n_split_consistency_failures": n_split_consistency_failures,
        "adjustment_mode_detected": adjustment_mode_detected,
        "source_modes_detected": source_modes_detected,
        "adjustment_modes_observed": observed_adjustment_modes,
        "pct_rows_ohlc_invalid": float((row_level["check_name"] == "ohlc_geometry_raw").mean()),
        "pct_symbols_bad_coverage": float((symbol_level["pct_missing_sessions"] > SYMBOL_MISSING_WARN_THRESHOLD).mean()),
        "corporate_actions_source_status": corporate_source_status,
        "corporate_actions_split_stats": corporate_stats,
        "input_paths": {
            "raw_prices": str(raw_source),
            "adjusted_prices": str(adjusted_source),
            "trading_calendar": str(calendar_source),
            "ticker_history_map": str(ticker_map_source),
            "corporate_actions": str(corporate_source) if corporate_source else "",
        },
    }
    summary_path = qc_root / "price_qc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "gate_status": gate_status,
        "summary_path": str(summary_path),
        "row_level_path": str(row_level_path),
        "symbol_level_path": str(symbol_level_path),
        "session_level_path": str(session_level_path),
        "corporate_actions_consistency_path": str(corporate_consistency_path),
        "failures_path": str(failures_path),
    }
    manifest_path = qc_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "price_qc_completed",
        run_id=run_id,
        gate_status=gate_status,
        n_fail=n_fail,
        n_warn=n_warn,
        n_symbols=int(raw["instrument_id"].nunique()) if not raw.empty else 0,
        n_sessions=int(len(sessions_in_scope)),
        output_dir=str(qc_root),
    )

    return PriceQCResult(
        gate_status=gate_status,
        n_fail=n_fail,
        n_warn=n_warn,
        summary_path=summary_path,
        row_level_path=row_level_path,
        symbol_level_path=symbol_level_path,
        session_level_path=session_level_path,
        corporate_actions_consistency_path=corporate_consistency_path,
        failures_path=failures_path,
        manifest_path=manifest_path,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run QC v2 checks for price pipeline artifacts.")
    parser.add_argument("--raw-prices-path", type=str, default=None)
    parser.add_argument("--adjusted-prices-path", type=str, default=None)
    parser.add_argument("--trading-calendar-path", type=str, default=None)
    parser.add_argument("--ticker-history-map-path", type=str, default=None)
    parser.add_argument("--corporate-actions-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default="price_qc_v2")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_price_qc(
        raw_prices_path=args.raw_prices_path,
        adjusted_prices_path=args.adjusted_prices_path,
        trading_calendar_path=args.trading_calendar_path,
        ticker_history_map_path=args.ticker_history_map_path,
        corporate_actions_path=args.corporate_actions_path,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print("Price QC completed:")
    print(f"- gate_status: {result.gate_status}")
    print(f"- summary: {result.summary_path}")
    print(f"- row level: {result.row_level_path}")
    print(f"- symbol level: {result.symbol_level_path}")
    print(f"- session level: {result.session_level_path}")
    print(f"- corporate consistency: {result.corporate_actions_consistency_path}")
    print(f"- failures: {result.failures_path}")
    print(f"- manifest: {result.manifest_path}")


if __name__ == "__main__":
    main()
