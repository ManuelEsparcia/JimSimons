from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.price.adjust_prices import adjust_prices
from data.price.fetch_prices import fetch_prices
from data.reference.build_reference import build_reference_data
from data.universe.build_universe import build_universe
from labels.build_labels import build_labels
from simons_core.io.parquet_store import read_parquet
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


LABELS_MIN_SCHEMA = DataSchema(
    name="labels_forward_min_test",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("horizon_days", "int64", nullable=False),
        ColumnSpec("entry_date", "datetime64", nullable=False),
        ColumnSpec("exit_date", "datetime64", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("label_value", "float64", nullable=False),
        ColumnSpec("price_entry", "float64", nullable=False),
        ColumnSpec("price_exit", "float64", nullable=False),
        ColumnSpec("source_price_field", "string", nullable=False),
    ),
    primary_key=("date", "instrument_id", "horizon_days", "label_name"),
    allow_extra_columns=True,
)


def _build_labels_inputs(tmp_workspace: dict[str, Path]) -> tuple[Path, Path, Path, Path]:
    reference_root = tmp_workspace["data"] / "reference"
    universe_root = tmp_workspace["data"] / "universe"
    price_root = tmp_workspace["data"] / "price"
    labels_root = tmp_workspace["data"] / "labels"

    build_reference_data(output_dir=reference_root, run_id="test_reference_labels_mvp")
    universe_result = build_universe(
        reference_root=reference_root,
        output_dir=universe_root,
        run_id="test_universe_labels_mvp",
    )
    raw_result = fetch_prices(
        reference_root=reference_root,
        universe_history_path=universe_result.universe_history,
        output_dir=price_root,
        run_id="test_price_fetch_labels_mvp",
    )
    adjusted_result = adjust_prices(
        raw_prices_path=raw_result.raw_prices_path,
        output_dir=price_root,
        run_id="test_price_adjust_labels_mvp",
    )
    return (
        adjusted_result.adjusted_prices_path,
        universe_result.universe_history,
        reference_root / "trading_calendar.parquet",
        labels_root,
    )


def test_build_labels_generates_non_empty_schema_valid_output(
    tmp_workspace: dict[str, Path],
) -> None:
    adjusted_path, universe_path, calendar_path, labels_root = _build_labels_inputs(tmp_workspace)
    result = build_labels(
        adjusted_prices_path=adjusted_path,
        universe_history_path=universe_path,
        trading_calendar_path=calendar_path,
        output_dir=labels_root,
        horizons=(1, 5, 20),
        decision_lag=1,
        run_id="test_build_labels_mvp",
    )

    assert result.labels_path.exists()
    assert result.summary_path.exists()
    assert (result.labels_path.with_suffix(".parquet.manifest.json")).exists()
    assert result.row_count > 0
    assert result.n_instruments > 0
    assert tuple(result.horizons) == (1, 5, 20)

    labels = read_parquet(result.labels_path)
    assert_schema(labels, LABELS_MIN_SCHEMA)
    assert len(labels) > 0
    assert not labels.duplicated(["date", "instrument_id", "horizon_days", "label_name"]).any()

    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    labels["entry_date"] = pd.to_datetime(labels["entry_date"], errors="coerce").dt.normalize()
    labels["exit_date"] = pd.to_datetime(labels["exit_date"], errors="coerce").dt.normalize()
    assert (labels["entry_date"] > labels["date"]).all()
    assert (labels["exit_date"] >= labels["entry_date"]).all()

    # No labels with incomplete horizon at the end of the series.
    sessions = pd.DatetimeIndex(
        sorted(
            pd.to_datetime(read_parquet(calendar_path)["date"], errors="coerce")
            .dt.normalize()
            .tolist()
        )
    )
    for horizon in (1, 5, 20):
        max_decision_idx = len(sessions) - 1 - (1 + horizon)  # lag=1
        max_decision_date = sessions[max_decision_idx]
        observed_max = labels.loc[labels["horizon_days"] == horizon, "date"].max()
        assert observed_max <= max_decision_date

    # Labels only for PIT eligible universe rows.
    eligible = read_parquet(universe_path).copy()
    eligible["date"] = pd.to_datetime(eligible["date"], errors="coerce").dt.normalize()
    eligible["instrument_id"] = eligible["instrument_id"].astype(str)
    eligible_pairs = set(
        eligible.loc[eligible["is_eligible"].astype(bool), ["date", "instrument_id"]]
        .itertuples(index=False, name=None)
    )
    label_pairs = set(labels[["date", "instrument_id"]].itertuples(index=False, name=None))
    assert label_pairs.issubset(eligible_pairs)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["horizons_built"] == [1, 5, 20]
    assert summary["n_rows_output"] == len(labels)
    assert summary["source_price_field"] == "close_adj"


def test_build_labels_exact_forward_return_in_small_fabricated_case(
    tmp_workspace: dict[str, Path],
) -> None:
    base = tmp_workspace["data"] / "labels_unit_case"
    base.mkdir(parents=True, exist_ok=True)
    adjusted_path = base / "adjusted_prices.parquet"
    universe_path = base / "universe_history.parquet"
    calendar_path = base / "trading_calendar.parquet"
    output_dir = base / "output"

    sessions = pd.bdate_range("2026-01-05", periods=6, freq="B")
    calendar = pd.DataFrame({"date": sessions, "is_session": True})
    calendar.to_parquet(calendar_path, index=False)

    adjusted = pd.DataFrame(
        {
            "date": list(sessions) * 2,
            "instrument_id": ["SIMX"] * 6 + ["SIMY"] * 6,
            "ticker": ["SIMX"] * 6 + ["SIMY"] * 6,
            "close_adj": [100.0, 110.0, 121.0, 133.1, 146.41, 161.051] + [50.0] * 6,
            "open_adj": [100.0, 110.0, 121.0, 133.1, 146.41, 161.051] + [50.0] * 6,
            "high_adj": [100.0, 110.0, 121.0, 133.1, 146.41, 161.051] + [50.0] * 6,
            "low_adj": [100.0, 110.0, 121.0, 133.1, 146.41, 161.051] + [50.0] * 6,
            "volume_adj": [1000.0] * 12,
        }
    )
    adjusted.to_parquet(adjusted_path, index=False)

    universe = pd.DataFrame(
        {
            "date": list(sessions) * 2,
            "instrument_id": ["SIMX"] * 6 + ["SIMY"] * 6,
            "ticker": ["SIMX"] * 6 + ["SIMY"] * 6,
            "is_eligible": [True] * 6 + [False] * 6,
        }
    )
    universe.to_parquet(universe_path, index=False)

    result = build_labels(
        adjusted_prices_path=adjusted_path,
        universe_history_path=universe_path,
        trading_calendar_path=calendar_path,
        output_dir=output_dir,
        horizons=(2,),
        decision_lag=1,
        run_id="test_build_labels_exact_case",
    )
    labels = read_parquet(result.labels_path)

    labels = labels.sort_values(["date", "instrument_id"]).reset_index(drop=True)
    assert set(labels["instrument_id"].unique().tolist()) == {"SIMX"}
    assert len(labels) == 3  # 6 sessions, lag=1, horizon=2 -> valid decision rows = 3

    expected_forward = 133.1 / 110.0 - 1.0  # for first decision date
    assert np.isclose(labels.loc[0, "label_value"], expected_forward)
    assert np.isclose(labels["label_value"].values, expected_forward).all()

    assert (pd.to_datetime(labels["entry_date"]) > pd.to_datetime(labels["date"])).all()
    assert (pd.to_datetime(labels["exit_date"]) >= pd.to_datetime(labels["entry_date"])).all()
