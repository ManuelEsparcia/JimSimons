from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from simons_core.calendar import load_market_calendar
from simons_core.io.parquet_store import read_parquet, validate_dataset, write_parquet
from simons_core.schemas import (
    PITTimestampError,
    SchemaValidationError,
    assert_pit_no_lookahead,
    assert_schema,
    validate_schema,
)


def test_schema_validation_smoke(
    pit_schema,
    pit_valid_df: pd.DataFrame,
) -> None:
    result = validate_schema(pit_valid_df, pit_schema)
    assert result.ok
    assert result.issues == ()


def test_schema_validation_fails_when_required_column_missing(
    pit_schema,
    pit_valid_df: pd.DataFrame,
) -> None:
    missing_col_df = pit_valid_df.drop(columns=["symbol"])
    with pytest.raises(SchemaValidationError):
        assert_schema(missing_col_df, pit_schema)


def test_pit_invalid_acceptance_ts_fails(pit_invalid_df: pd.DataFrame) -> None:
    with pytest.raises(PITTimestampError):
        assert_pit_no_lookahead(
            pit_invalid_df,
            decision_col="asof",
            available_col="acceptance_ts",
        )


def test_pit_valid_acceptance_ts_passes(pit_valid_df: pd.DataFrame) -> None:
    assert_pit_no_lookahead(
        pit_valid_df,
        decision_col="asof",
        available_col="acceptance_ts",
    )


def test_calendar_smoke(sample_calendar_path: Path) -> None:
    cal = load_market_calendar(sample_calendar_path)

    assert cal.is_session("2026-01-06")
    assert not cal.is_session("2026-01-10")  # Saturday

    assert cal.next_trading_day("2026-01-09").date() == date(2026, 1, 12)
    assert cal.prev_trading_day("2026-01-12").date() == date(2026, 1, 9)

    week_sessions = cal.sessions_between("2026-01-05", "2026-01-09", inclusive="both")
    assert len(week_sessions) == 5


def test_parquet_store_write_read_smoke(
    tmp_workspace: dict[str, Path],
    pit_valid_df: pd.DataFrame,
) -> None:
    destination = tmp_workspace["artifacts"] / "pit_observation.parquet"
    written = write_parquet(
        pit_valid_df,
        destination,
        schema_name="pit_observation_test",
        run_id="test_run_001",
    )

    loaded = read_parquet(
        written,
        required_columns=("symbol", "asof", "acceptance_ts", "value"),
    )
    validate_dataset(written, required_columns=("symbol", "asof", "acceptance_ts", "value"))

    assert len(loaded) == len(pit_valid_df)
    assert set(loaded.columns) == set(pit_valid_df.columns)
    assert written.with_suffix(written.suffix + ".manifest.json").exists()
