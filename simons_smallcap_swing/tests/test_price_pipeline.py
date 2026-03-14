from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.price.adjust_prices import adjust_prices
from data.price.fetch_prices import fetch_prices
from data.price.qc_prices import run_price_qc
from data.reference.build_reference import build_reference_data
from data.universe.build_universe import build_universe
from simons_core.io.parquet_store import read_parquet
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


RAW_MIN_SCHEMA = DataSchema(
    name="price_raw_min_test",
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

ADJUSTED_MIN_SCHEMA = DataSchema(
    name="price_adjusted_min_test",
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


def _build_price_pipeline(tmp_workspace: dict[str, Path]) -> tuple[Path, Path, Path]:
    reference_root = tmp_workspace["data"] / "reference"
    universe_root = tmp_workspace["data"] / "universe"
    price_root = tmp_workspace["data"] / "price"

    build_reference_data(output_dir=reference_root, run_id="test_reference_price_mvp")
    universe_result = build_universe(
        reference_root=reference_root,
        output_dir=universe_root,
        run_id="test_universe_price_mvp",
    )
    raw_result = fetch_prices(
        reference_root=reference_root,
        universe_history_path=universe_result.universe_history,
        output_dir=price_root,
        run_id="test_price_fetch_mvp",
    )
    adjusted_result = adjust_prices(
        raw_prices_path=raw_result.raw_prices_path,
        output_dir=price_root,
        run_id="test_price_adjust_mvp",
    )
    return raw_result.raw_prices_path, adjusted_result.adjusted_prices_path, reference_root


def test_price_raw_and_adjusted_artifacts_are_generated_and_non_empty(
    tmp_workspace: dict[str, Path],
) -> None:
    raw_path, adjusted_path, _ = _build_price_pipeline(tmp_workspace)

    assert raw_path.exists()
    assert adjusted_path.exists()
    assert raw_path.stat().st_size > 0
    assert adjusted_path.stat().st_size > 0

    raw = read_parquet(raw_path)
    adjusted = read_parquet(adjusted_path)
    assert len(raw) > 0
    assert len(adjusted) > 0


def test_price_schema_pk_calendar_and_ohlc_consistency(tmp_workspace: dict[str, Path]) -> None:
    raw_path, adjusted_path, reference_root = _build_price_pipeline(tmp_workspace)
    raw = read_parquet(raw_path)
    adjusted = read_parquet(adjusted_path)
    calendar = read_parquet(reference_root / "trading_calendar.parquet")

    assert_schema(raw, RAW_MIN_SCHEMA)
    assert_schema(adjusted, ADJUSTED_MIN_SCHEMA)

    assert not raw.duplicated(["date", "instrument_id"]).any()
    assert not adjusted.duplicated(["date", "instrument_id"]).any()

    sessions = set(
        pd.to_datetime(
            calendar.loc[calendar["is_session"], "date"],
            errors="coerce",
        )
        .dt.normalize()
        .tolist()
    )
    raw_dates = set(pd.to_datetime(raw["date"], errors="coerce").dt.normalize().tolist())
    assert raw_dates.issubset(sessions)

    assert (raw[["open", "high", "low", "close"]] > 0).all().all()
    assert (raw["volume"] >= 0).all()
    assert (raw["high"] >= raw[["open", "close", "low"]].max(axis=1)).all()
    assert (raw["low"] <= raw[["open", "close", "high"]].min(axis=1)).all()


def test_price_ticker_pit_and_split_only_default_no_corporate_actions_consistency(
    tmp_workspace: dict[str, Path],
) -> None:
    raw_path, adjusted_path, reference_root = _build_price_pipeline(tmp_workspace)
    raw = read_parquet(raw_path).copy()
    adjusted = read_parquet(adjusted_path).copy()
    ticker_map = read_parquet(reference_root / "ticker_history_map.parquet").copy()

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    ticker_map["start_date"] = pd.to_datetime(ticker_map["start_date"], errors="coerce").dt.normalize()
    ticker_map["end_date"] = pd.to_datetime(ticker_map["end_date"], errors="coerce").dt.normalize()

    raw["__row_id"] = range(len(raw))
    merged = raw.merge(
        ticker_map[["instrument_id", "ticker", "start_date", "end_date"]],
        on=["instrument_id", "ticker"],
        how="left",
    )
    merged["pit_valid"] = (
        merged["start_date"].notna()
        & (merged["date"] >= merged["start_date"])
        & (merged["end_date"].isna() | (merged["date"] <= merged["end_date"]))
    )
    row_valid = merged.groupby("__row_id", as_index=True)["pit_valid"].any()
    assert row_valid.all(), "Found raw price rows incompatible with ticker_history_map PIT intervals."

    joined = raw.merge(adjusted, on=["date", "instrument_id", "ticker"], how="inner")
    assert len(joined) == len(raw)
    assert (joined["adjustment_mode"] == "split_only").all()
    assert "cumulative_split_factor" in joined.columns
    assert "applied_split_events_count" in joined.columns
    assert np.isclose(joined["cumulative_split_factor"], 1.0).all()
    assert (joined["applied_split_events_count"].astype(int) == 0).all()
    assert np.isclose(joined["open_adj"], joined["open"]).all()
    assert np.isclose(joined["high_adj"], joined["high"]).all()
    assert np.isclose(joined["low_adj"], joined["low"]).all()
    assert np.isclose(joined["close_adj"], joined["close"]).all()
    assert np.isclose(joined["volume_adj"], joined["volume"].astype(float)).all()


def test_price_qc_smoke_produces_usable_gate_and_artifacts(
    tmp_workspace: dict[str, Path],
) -> None:
    raw_path, adjusted_path, reference_root = _build_price_pipeline(tmp_workspace)
    qc_root = tmp_workspace["artifacts"] / "price_qc"

    result = run_price_qc(
        raw_prices_path=raw_path,
        adjusted_prices_path=adjusted_path,
        trading_calendar_path=reference_root / "trading_calendar.parquet",
        ticker_history_map_path=reference_root / "ticker_history_map.parquet",
        output_dir=qc_root,
        run_id="test_price_qc_mvp",
    )

    assert result.gate_status == "PASS"
    assert result.summary_path.exists()
    assert result.row_level_path.exists()
    assert result.symbol_level_path.exists()
    assert result.failures_path.exists()
    assert result.manifest_path.exists()
