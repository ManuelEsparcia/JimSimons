from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.reference.build_reference import build_reference_data
from simons_core.calendar import load_market_calendar
from simons_core.io.parquet_store import read_parquet
from simons_core.io.paths import reference_dir
from simons_core.schemas import assert_schema


def _expected_artifact_paths() -> dict[str, Path]:
    base = reference_dir()
    return {
        "trading_calendar": base / "trading_calendar.parquet",
        "ticker_history_map": base / "ticker_history_map.parquet",
        "symbols_metadata": base / "symbols_metadata.parquet",
        "sector_industry_map": base / "sector_industry_map.parquet",
    }


def _build_reference() -> dict[str, Path]:
    result = build_reference_data()
    return {
        "trading_calendar": result.trading_calendar,
        "ticker_history_map": result.ticker_history_map,
        "symbols_metadata": result.symbols_metadata,
        "sector_industry_map": result.sector_industry_map,
    }


def test_reference_artifacts_exist_and_non_empty() -> None:
    built = _build_reference()
    expected = _expected_artifact_paths()

    for key, path in expected.items():
        assert built[key] == path
        assert path.exists(), f"{key} does not exist: {path}"
        assert path.stat().st_size > 0, f"{key} is empty: {path}"


def test_reference_artifacts_match_minimum_schemas() -> None:
    _build_reference()
    paths = _expected_artifact_paths()

    trading_calendar = read_parquet(paths["trading_calendar"])
    ticker_history_map = read_parquet(paths["ticker_history_map"])
    symbols_metadata = read_parquet(paths["symbols_metadata"])
    sector_industry_map = read_parquet(paths["sector_industry_map"])

    assert_schema(trading_calendar, "reference_trading_calendar")
    assert_schema(ticker_history_map, "reference_ticker_history_map")
    assert_schema(symbols_metadata, "reference_symbols_metadata")
    assert_schema(sector_industry_map, "reference_sector_industry_map")


def test_trading_calendar_is_consumable_by_calendar_module() -> None:
    _build_reference()
    cal = load_market_calendar()

    assert cal.is_session("2026-01-05")
    assert not cal.is_session("2026-01-10")
    assert cal.next_trading_day("2026-01-09").date() == pd.Timestamp("2026-01-12").date()


def test_ticker_history_intervals_are_valid_and_instrument_ids_present() -> None:
    _build_reference()
    df = read_parquet(_expected_artifact_paths()["ticker_history_map"])

    assert df["instrument_id"].notna().all()
    assert df["ticker"].notna().all()
    assert (df["start_date"].notna()).all()

    valid_interval_mask = df["end_date"].isna() | (df["end_date"] >= df["start_date"])
    assert valid_interval_mask.all(), "Found rows with end_date < start_date."


def test_instrument_id_presence_in_reference_tables() -> None:
    _build_reference()
    paths = _expected_artifact_paths()

    ticker_history_map = read_parquet(paths["ticker_history_map"])
    symbols_metadata = read_parquet(paths["symbols_metadata"])
    sector_industry_map = read_parquet(paths["sector_industry_map"])

    assert ticker_history_map["instrument_id"].notna().all()
    assert symbols_metadata["instrument_id"].notna().all()
    assert sector_industry_map["instrument_id"].notna().all()
