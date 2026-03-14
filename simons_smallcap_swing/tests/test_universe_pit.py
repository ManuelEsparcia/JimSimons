from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.reference.build_reference import build_reference_data
from data.universe.build_universe import build_universe
from data.universe.universe_qc import run_universe_qc
from simons_core.io.parquet_store import read_parquet
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


UNIVERSE_MIN_SCHEMA = DataSchema(
    name="universe_history_min_test",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("is_eligible", "bool", nullable=False),
        ColumnSpec("exchange", "string", nullable=False),
        ColumnSpec("asset_type", "string", nullable=False),
        ColumnSpec("currency", "string", nullable=False),
        ColumnSpec("sector", "string", nullable=True),
        ColumnSpec("industry", "string", nullable=True),
    ),
    primary_key=("date", "instrument_id"),
    allow_extra_columns=True,
)


def _build_universe_in_tmp(tmp_workspace: dict[str, Path]) -> tuple[Path, Path, Path]:
    reference_root = tmp_workspace["data"] / "reference"
    universe_root = tmp_workspace["data"] / "universe"
    build_reference_data(output_dir=reference_root, run_id="test_reference_universe_mvp")
    result = build_universe(
        reference_root=reference_root,
        output_dir=universe_root,
        run_id="test_universe_mvp",
        config_path=None,
    )
    return result.universe_history, result.universe_current, reference_root


def test_universe_history_and_current_generated_non_empty(tmp_workspace: dict[str, Path]) -> None:
    history_path, current_path, _ = _build_universe_in_tmp(tmp_workspace)

    assert history_path.exists()
    assert current_path.exists()
    assert history_path.stat().st_size > 0
    assert current_path.stat().st_size > 0

    history = read_parquet(history_path)
    current = read_parquet(current_path)
    assert len(history) > 0
    assert len(current) > 0


def test_universe_schema_pk_calendar_and_current_consistency(
    tmp_workspace: dict[str, Path],
) -> None:
    history_path, current_path, reference_root = _build_universe_in_tmp(tmp_workspace)
    history = read_parquet(history_path)
    current = read_parquet(current_path)
    trading_calendar = read_parquet(reference_root / "trading_calendar.parquet")

    assert_schema(history, UNIVERSE_MIN_SCHEMA)
    assert not history.duplicated(["date", "instrument_id"]).any()

    calendar_sessions = set(
        pd.to_datetime(
            trading_calendar.loc[trading_calendar["is_session"], "date"],
            errors="coerce",
        )
        .dt.normalize()
        .tolist()
    )
    history_dates = set(pd.to_datetime(history["date"], errors="coerce").dt.normalize().tolist())
    assert history_dates.issubset(calendar_sessions)

    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    current["date"] = pd.to_datetime(current["date"]).dt.normalize()
    last_date = history["date"].max()

    assert current["date"].nunique() == 1
    assert current["date"].iloc[0] == last_date

    expected_current = history[(history["date"] == last_date) & (history["is_eligible"])]
    expected_keys = set(expected_current[["instrument_id", "ticker"]].itertuples(index=False, name=None))
    observed_keys = set(current[["instrument_id", "ticker"]].itertuples(index=False, name=None))
    assert observed_keys == expected_keys


def test_universe_ticker_is_pit_compatible_with_history_map(
    tmp_workspace: dict[str, Path],
) -> None:
    history_path, _, reference_root = _build_universe_in_tmp(tmp_workspace)
    history = read_parquet(history_path).copy()
    ticker_map = read_parquet(reference_root / "ticker_history_map.parquet").copy()

    history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.normalize()
    ticker_map["start_date"] = pd.to_datetime(ticker_map["start_date"], errors="coerce").dt.normalize()
    ticker_map["end_date"] = pd.to_datetime(ticker_map["end_date"], errors="coerce").dt.normalize()

    history["__row_id"] = range(len(history))
    merged = history.merge(
        ticker_map[["instrument_id", "ticker", "start_date", "end_date"]],
        on=["instrument_id", "ticker"],
        how="left",
    )
    merged["valid_interval"] = (
        merged["start_date"].notna()
        & (merged["date"] >= merged["start_date"])
        & (merged["end_date"].isna() | (merged["date"] <= merged["end_date"]))
    )
    row_ok = merged.groupby("__row_id", as_index=True)["valid_interval"].any()
    assert row_ok.all(), "Found universe rows with ticker/date incompatible with ticker_history_map."


def test_universe_qc_smoke_passes_for_mvp_dataset(tmp_workspace: dict[str, Path]) -> None:
    history_path, current_path, reference_root = _build_universe_in_tmp(tmp_workspace)
    qc_output = tmp_workspace["artifacts"] / "universe_qc"

    qc_result = run_universe_qc(
        universe_history_path=history_path,
        universe_current_path=current_path,
        ticker_history_map_path=reference_root / "ticker_history_map.parquet",
        trading_calendar_path=reference_root / "trading_calendar.parquet",
        output_dir=qc_output,
        run_id="test_universe_qc_mvp",
    )

    assert qc_result.gate_status == "PASS"
    assert qc_result.summary_path.exists()
    assert qc_result.daily_path.exists()
    assert qc_result.failures_path.exists()
    assert qc_result.manifest_path.exists()
