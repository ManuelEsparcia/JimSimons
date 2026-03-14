from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.edgar.edgar_qc import run_edgar_qc
from data.edgar.point_in_time import build_fundamentals_pit
from data.edgar.ticker_cik import build_ticker_cik_map
from data.reference.build_reference import build_reference_data
from data.universe.build_universe import build_universe
from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


TICKER_CIK_MIN_SCHEMA = DataSchema(
    name="ticker_cik_min_test",
    version="1.0.0",
    columns=(
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("cik", "string", nullable=False),
        ColumnSpec("start_date", "datetime64", nullable=False),
        ColumnSpec("end_date", "datetime64", nullable=True),
        ColumnSpec("is_active", "bool", nullable=False),
    ),
    primary_key=("instrument_id", "ticker", "start_date"),
    allow_extra_columns=True,
)

FUNDAMENTALS_PIT_MIN_SCHEMA = DataSchema(
    name="fundamentals_pit_min_test",
    version="1.0.0",
    columns=(
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("cik", "string", nullable=False),
        ColumnSpec("asof_date", "datetime64[ns, UTC]", nullable=False),
        ColumnSpec("acceptance_ts", "datetime64[ns, UTC]", nullable=False),
        ColumnSpec("filing_date", "datetime64", nullable=False),
        ColumnSpec("fiscal_period_end", "datetime64", nullable=False),
        ColumnSpec("metric_name", "string", nullable=False),
        ColumnSpec("metric_value", "number", nullable=False),
        ColumnSpec("metric_unit", "string", nullable=False),
        ColumnSpec("source_type", "string", nullable=False),
        ColumnSpec("data_quality", "string", nullable=False),
    ),
    primary_key=("instrument_id", "asof_date", "metric_name"),
    allow_extra_columns=True,
)


def _build_edgar_pipeline(tmp_workspace: dict[str, Path]) -> tuple[Path, Path]:
    reference_root = tmp_workspace["data"] / "reference"
    universe_root = tmp_workspace["data"] / "universe"
    edgar_root = tmp_workspace["data"] / "edgar"

    build_reference_data(output_dir=reference_root, run_id="test_reference_edgar_mvp")
    universe_result = build_universe(
        reference_root=reference_root,
        output_dir=universe_root,
        run_id="test_universe_edgar_mvp",
    )
    ticker_cik_result = build_ticker_cik_map(
        reference_root=reference_root,
        universe_history_path=universe_result.universe_history,
        output_dir=edgar_root,
        run_id="test_ticker_cik_mvp",
    )
    fundamentals_result = build_fundamentals_pit(
        ticker_cik_map_path=ticker_cik_result.ticker_cik_map_path,
        universe_history_path=universe_result.universe_history,
        output_dir=edgar_root,
        run_id="test_fundamentals_pit_mvp",
    )
    return ticker_cik_result.ticker_cik_map_path, fundamentals_result.fundamentals_pit_path


def test_ticker_cik_and_fundamentals_pit_are_generated_and_non_empty(
    tmp_workspace: dict[str, Path],
) -> None:
    ticker_cik_path, fundamentals_pit_path = _build_edgar_pipeline(tmp_workspace)

    assert ticker_cik_path.exists()
    assert fundamentals_pit_path.exists()
    assert ticker_cik_path.stat().st_size > 0
    assert fundamentals_pit_path.stat().st_size > 0

    ticker_cik = read_parquet(ticker_cik_path)
    pit = read_parquet(fundamentals_pit_path)
    assert len(ticker_cik) > 0
    assert len(pit) > 0


def test_edgar_pit_schema_metrics_and_identity_consistency(
    tmp_workspace: dict[str, Path],
) -> None:
    ticker_cik_path, fundamentals_pit_path = _build_edgar_pipeline(tmp_workspace)
    ticker_cik = read_parquet(ticker_cik_path).copy()
    pit = read_parquet(fundamentals_pit_path).copy()

    assert_schema(ticker_cik, TICKER_CIK_MIN_SCHEMA)
    assert_schema(pit, FUNDAMENTALS_PIT_MIN_SCHEMA)

    required_metrics = {"revenue", "net_income", "total_assets", "shares_outstanding"}
    assert required_metrics.issubset(set(pit["metric_name"].unique().tolist()))

    assert (pit["acceptance_ts"] <= pit["asof_date"]).all()

    ticker_cik["start_date"] = pd.to_datetime(ticker_cik["start_date"], errors="coerce").dt.normalize()
    ticker_cik["end_date"] = pd.to_datetime(ticker_cik["end_date"], errors="coerce").dt.normalize()
    pit["asof_session_date"] = pd.to_datetime(pit["asof_date"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    pit["__row_id"] = range(len(pit))

    merged = pit.merge(
        ticker_cik[["instrument_id", "cik", "start_date", "end_date"]],
        on=["instrument_id", "cik"],
        how="left",
    )
    merged["valid_map"] = (
        merged["start_date"].notna()
        & (merged["asof_session_date"] >= merged["start_date"])
        & (merged["end_date"].isna() | (merged["asof_session_date"] <= merged["end_date"]))
    )
    row_valid = merged.groupby("__row_id", as_index=True)["valid_map"].any()
    assert row_valid.all(), "Found fundamentals_pit rows with invalid CIK interval mapping."


def test_acceptance_after_asof_triggers_qc_fail(tmp_workspace: dict[str, Path]) -> None:
    ticker_cik_path, fundamentals_pit_path = _build_edgar_pipeline(tmp_workspace)
    bad_pit = read_parquet(fundamentals_pit_path).copy()

    bad_pit.loc[0, "acceptance_ts"] = bad_pit.loc[0, "asof_date"] + pd.Timedelta(days=1)
    bad_path = tmp_workspace["data"] / "edgar" / "fundamentals_pit_bad.parquet"
    _ = write_parquet(bad_pit, bad_path, schema_name="fundamentals_pit_bad_test", run_id="bad_case")

    qc_result = run_edgar_qc(
        fundamentals_pit_path=bad_path,
        ticker_cik_map_path=ticker_cik_path,
        output_dir=tmp_workspace["artifacts"] / "edgar_qc_bad",
        run_id="test_edgar_qc_bad",
    )
    assert qc_result.gate_status == "FAIL"
    assert qc_result.n_fail > 0


def test_edgar_qc_smoke_passes_for_valid_dataset(tmp_workspace: dict[str, Path]) -> None:
    ticker_cik_path, fundamentals_pit_path = _build_edgar_pipeline(tmp_workspace)
    qc_result = run_edgar_qc(
        fundamentals_pit_path=fundamentals_pit_path,
        ticker_cik_map_path=ticker_cik_path,
        output_dir=tmp_workspace["artifacts"] / "edgar_qc_ok",
        run_id="test_edgar_qc_ok",
    )

    assert qc_result.gate_status == "PASS"
    assert qc_result.summary_path.exists()
    assert qc_result.row_level_path.exists()
    assert qc_result.failures_path.exists()
    assert qc_result.metrics_path.exists()
    assert qc_result.manifest_path.exists()
