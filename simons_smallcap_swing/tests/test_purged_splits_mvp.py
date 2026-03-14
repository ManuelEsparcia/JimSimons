from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.price.adjust_prices import adjust_prices
from data.price.fetch_prices import fetch_prices
from data.reference.build_reference import build_reference_data
from data.universe.build_universe import build_universe
from labels.build_labels import build_labels
from labels.purged_splits import build_purged_splits
from simons_core.io.parquet_store import read_parquet
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


PURGED_SPLITS_MIN_SCHEMA = DataSchema(
    name="purged_splits_min_test",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("horizon_days", "int64", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("split_name", "string", nullable=False),
        ColumnSpec("split_role", "string", nullable=False),
        ColumnSpec("entry_date", "datetime64", nullable=False),
        ColumnSpec("exit_date", "datetime64", nullable=False),
    ),
    primary_key=("date", "instrument_id", "horizon_days", "label_name"),
    allow_extra_columns=True,
)


def _build_labels_inputs(tmp_workspace: dict[str, Path]) -> tuple[Path, Path, Path, Path]:
    reference_root = tmp_workspace["data"] / "reference"
    universe_root = tmp_workspace["data"] / "universe"
    price_root = tmp_workspace["data"] / "price"
    labels_root = tmp_workspace["data"] / "labels"

    build_reference_data(output_dir=reference_root, run_id="test_reference_purged_splits")
    universe_result = build_universe(
        reference_root=reference_root,
        output_dir=universe_root,
        run_id="test_universe_purged_splits",
    )
    raw_result = fetch_prices(
        reference_root=reference_root,
        universe_history_path=universe_result.universe_history,
        output_dir=price_root,
        run_id="test_fetch_purged_splits",
    )
    adjusted_result = adjust_prices(
        raw_prices_path=raw_result.raw_prices_path,
        output_dir=price_root,
        run_id="test_adjust_purged_splits",
    )
    labels_result = build_labels(
        adjusted_prices_path=adjusted_result.adjusted_prices_path,
        universe_history_path=universe_result.universe_history,
        trading_calendar_path=reference_root / "trading_calendar.parquet",
        output_dir=labels_root,
        horizons=(1, 5, 20),
        decision_lag=1,
        run_id="test_labels_for_purged_splits",
    )
    return (
        labels_result.labels_path,
        reference_root / "trading_calendar.parquet",
        labels_root,
        universe_result.universe_history,
    )


def test_purged_splits_generates_artifacts_and_valid_roles(
    tmp_workspace: dict[str, Path],
) -> None:
    labels_path, calendar_path, labels_root, _universe_path = _build_labels_inputs(tmp_workspace)
    result = build_purged_splits(
        labels_path=labels_path,
        trading_calendar_path=calendar_path,
        output_dir=labels_root,
        valid_fraction=0.2,
        test_fraction=0.2,
        embargo_sessions=1,
        run_id="test_purged_splits_mvp",
    )

    assert result.splits_path.exists()
    assert result.summary_path.exists()
    assert (result.splits_path.with_suffix(".parquet.manifest.json")).exists()
    assert result.row_count > 0

    splits = read_parquet(result.splits_path)
    assert len(splits) > 0
    assert_schema(splits, PURGED_SPLITS_MIN_SCHEMA)
    assert not splits.duplicated(["date", "instrument_id", "horizon_days", "label_name"]).any()

    allowed_roles = {
        "train",
        "valid",
        "test",
        "dropped_by_purge",
        "dropped_by_embargo",
    }
    assert set(splits["split_role"].astype(str).unique().tolist()).issubset(allowed_roles)
    assert (pd.to_datetime(splits["entry_date"]) > pd.to_datetime(splits["date"])).all()
    assert (pd.to_datetime(splits["exit_date"]) >= pd.to_datetime(splits["entry_date"])).all()

    core = splits[splits["split_role"].isin(["train", "valid", "test"])].copy()
    train_max = pd.to_datetime(core.loc[core["split_role"] == "train", "date"]).max()
    valid_min = pd.to_datetime(core.loc[core["split_role"] == "valid", "date"]).min()
    valid_max = pd.to_datetime(core.loc[core["split_role"] == "valid", "date"]).max()
    test_min = pd.to_datetime(core.loc[core["split_role"] == "test", "date"]).min()
    assert train_max < valid_min <= valid_max < test_min

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    expected_keys = {
        "split_method",
        "train_start",
        "train_end",
        "valid_start",
        "valid_end",
        "test_start",
        "test_end",
        "embargo_sessions",
        "n_train",
        "n_valid",
        "n_test",
        "n_dropped_by_purge",
        "n_dropped_by_embargo",
        "horizons_present",
        "label_names_present",
    }
    assert expected_keys.issubset(summary.keys())
    assert summary["n_rows_total"] == len(splits)


def test_purged_splits_fabricated_overlap_and_embargo_behavior(
    tmp_workspace: dict[str, Path],
) -> None:
    base = tmp_workspace["data"] / "purged_splits_small_case"
    labels_path = base / "labels_forward.parquet"
    calendar_path = base / "trading_calendar.parquet"
    output_dir = base / "output"
    base.mkdir(parents=True, exist_ok=True)

    sessions = pd.bdate_range("2026-01-05", periods=13, freq="B")
    calendar = pd.DataFrame({"date": sessions, "is_session": True})
    calendar.to_parquet(calendar_path, index=False)

    # 11 decision dates with horizon 2 -> entry=t+1, exit=t+2.
    rows: list[dict[str, object]] = []
    for idx in range(11):
        rows.append(
            {
                "date": sessions[idx],
                "instrument_id": "SIMA",
                "ticker": "AAA",
                "horizon_days": 2,
                "entry_date": sessions[idx + 1],
                "exit_date": sessions[idx + 2],
                "label_name": "fwd_ret_2d",
                "label_value": 0.01,
                "price_entry": 100.0,
                "price_exit": 101.0,
                "source_price_field": "close_adj",
            }
        )
    labels = pd.DataFrame(rows)
    labels.to_parquet(labels_path, index=False)

    result = build_purged_splits(
        labels_path=labels_path,
        trading_calendar_path=calendar_path,
        output_dir=output_dir,
        embargo_sessions=3,
        valid_start=str(sessions[5].date()),
        valid_end=str(sessions[6].date()),
        test_start=str(sessions[8].date()),
        test_end=str(sessions[8].date()),
        run_id="test_purged_splits_small_case",
    )
    splits = read_parquet(result.splits_path)
    assert len(splits) == 11

    # Force checks requested: overlap-driven purge and embargo-driven drops.
    assert (splits["split_role"] == "dropped_by_purge").any()
    assert (splits["split_role"] == "dropped_by_embargo").any()
    assert (splits["split_role"] == "train").any()
    assert (splits["split_role"] == "valid").any()
    assert (splits["split_role"] == "test").any()

    # Embargo after test_end=idx8 with embargo=3 should include idx9, idx10, idx11.
    # idx10 should be embargoed without overlap purge in this fabricated setup.
    embargo_expected_date = sessions[10]
    row = splits.loc[pd.to_datetime(splits["date"]) == embargo_expected_date]
    assert not row.empty
    assert row.iloc[0]["split_role"] == "dropped_by_embargo"

    core = splits[splits["split_role"].isin(["train", "valid", "test"])].copy()
    train_max = pd.to_datetime(core.loc[core["split_role"] == "train", "date"]).max()
    valid_min = pd.to_datetime(core.loc[core["split_role"] == "valid", "date"]).min()
    valid_max = pd.to_datetime(core.loc[core["split_role"] == "valid", "date"]).max()
    test_min = pd.to_datetime(core.loc[core["split_role"] == "test", "date"]).min()
    assert train_max < valid_min <= valid_max < test_min

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["n_dropped_by_purge"] >= 1
    assert summary["n_dropped_by_embargo"] >= 1
    assert summary["embargo_sessions"] == 3
