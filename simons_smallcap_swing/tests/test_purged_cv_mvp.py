from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from labels.purged_cv import build_purged_cv
from simons_core.io.parquet_store import read_parquet
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


PURGED_CV_MIN_SCHEMA = DataSchema(
    name="purged_cv_min_test",
    version="1.0.0",
    columns=(
        ColumnSpec("fold_id", "int64", nullable=False),
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("horizon_days", "int64", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("split_role", "string", nullable=False),
        ColumnSpec("entry_date", "datetime64", nullable=False),
        ColumnSpec("exit_date", "datetime64", nullable=False),
    ),
    primary_key=("fold_id", "date", "instrument_id", "horizon_days", "label_name"),
    allow_extra_columns=True,
)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: list[tuple[int, int]] = []
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end + 1:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def _overlaps_any(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    for i_start, i_end in intervals:
        if i_start > end:
            return False
        if i_end < start:
            continue
        return True
    return False


def test_purged_cv_generates_multifold_artifacts_and_temporal_integrity(
    tmp_workspace: dict[str, Path],
) -> None:
    base = tmp_workspace["data"] / "purged_cv_case_1"
    labels_path = base / "labels_forward.parquet"
    calendar_path = base / "trading_calendar.parquet"
    output_dir = base / "output"
    base.mkdir(parents=True, exist_ok=True)

    sessions = pd.bdate_range("2026-01-05", periods=30, freq="B")
    calendar = pd.DataFrame({"date": sessions, "is_session": True})
    calendar.to_parquet(calendar_path, index=False)

    rows: list[dict[str, object]] = []
    instruments = [("SIMA", "AAA"), ("SIMB", "BBB")]
    for idx in range(24):
        for instrument_id, ticker in instruments:
            rows.append(
                {
                    "date": sessions[idx],
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "horizon_days": 2,
                    "entry_date": sessions[idx + 1],
                    "exit_date": sessions[idx + 2],
                    "label_name": "fwd_ret_2d",
                    "label_value": 0.01 if instrument_id == "SIMA" else -0.01,
                    "price_entry": 100.0,
                    "price_exit": 101.0,
                    "source_price_field": "close_adj",
                }
            )
    pd.DataFrame(rows).to_parquet(labels_path, index=False)

    result = build_purged_cv(
        labels_path=labels_path,
        trading_calendar_path=calendar_path,
        output_dir=output_dir,
        n_folds=4,
        embargo_sessions=2,
        run_id="test_purged_cv_case_1",
    )

    assert result.folds_path.exists()
    assert result.summary_path.exists()
    assert (result.folds_path.with_suffix(".parquet.manifest.json")).exists()
    assert result.row_count > 0
    assert result.n_folds == 4

    folds = read_parquet(result.folds_path)
    assert len(folds) > 0
    assert_schema(folds, PURGED_CV_MIN_SCHEMA)
    assert not folds.duplicated(["fold_id", "date", "instrument_id", "horizon_days", "label_name"]).any()

    roles = set(folds["split_role"].astype(str).unique().tolist())
    assert roles.issubset({"train", "valid", "dropped_by_purge", "dropped_by_embargo"})
    assert (pd.to_datetime(folds["entry_date"]) > pd.to_datetime(folds["date"])).all()
    assert (pd.to_datetime(folds["exit_date"]) >= pd.to_datetime(folds["entry_date"])).all()
    assert len(set(folds["fold_id"].astype(int).tolist())) == 4

    session_pos_map = {pd.Timestamp(dt): i for i, dt in enumerate(pd.to_datetime(sessions))}
    for fold_id in sorted(set(folds["fold_id"].astype(int).tolist())):
        fold = folds[folds["fold_id"].astype(int) == fold_id].copy()
        valid_rows = fold[fold["split_role"] == "valid"].copy()
        assert not valid_rows.empty

        valid_pos = sorted({session_pos_map[pd.Timestamp(dt)] for dt in pd.to_datetime(valid_rows["date"])})
        assert valid_pos == list(range(min(valid_pos), max(valid_pos) + 1))

        valid_intervals = _merge_intervals(
            list(
                zip(
                    [session_pos_map[pd.Timestamp(dt)] for dt in pd.to_datetime(valid_rows["entry_date"])],
                    [session_pos_map[pd.Timestamp(dt)] for dt in pd.to_datetime(valid_rows["exit_date"])],
                )
            )
        )
        train_rows = fold[fold["split_role"] == "train"].copy()
        valid_end_pos = max(valid_pos)

        for row in train_rows.itertuples(index=False):
            entry_pos = session_pos_map[pd.Timestamp(row.entry_date)]
            exit_pos = session_pos_map[pd.Timestamp(row.exit_date)]
            decision_pos = session_pos_map[pd.Timestamp(row.date)]
            assert not _overlaps_any(entry_pos, exit_pos, valid_intervals)
            assert not (valid_end_pos < decision_pos <= (valid_end_pos + 2))

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    expected_keys = {
        "n_folds",
        "cv_method",
        "embargo_sessions",
        "folds_time_ranges",
        "n_train_by_fold",
        "n_valid_by_fold",
        "n_dropped_by_purge_by_fold",
        "n_dropped_by_embargo_by_fold",
        "horizons_present",
        "label_names_present",
    }
    assert expected_keys.issubset(summary.keys())
    assert summary["n_rows_total"] == len(folds)


def test_purged_cv_detects_overlap_purge_and_embargo_effect(
    tmp_workspace: dict[str, Path],
) -> None:
    base = tmp_workspace["data"] / "purged_cv_case_2"
    labels_path = base / "labels_forward.parquet"
    calendar_path = base / "trading_calendar.parquet"
    output_dir = base / "output"
    base.mkdir(parents=True, exist_ok=True)

    sessions = pd.bdate_range("2026-01-05", periods=14, freq="B")
    calendar = pd.DataFrame({"date": sessions, "is_session": True})
    calendar.to_parquet(calendar_path, index=False)

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
    pd.DataFrame(rows).to_parquet(labels_path, index=False)

    result = build_purged_cv(
        labels_path=labels_path,
        trading_calendar_path=calendar_path,
        output_dir=output_dir,
        n_folds=3,
        embargo_sessions=2,
        run_id="test_purged_cv_case_2",
    )

    folds = read_parquet(result.folds_path)
    assert len(folds) > 0
    assert (folds["split_role"] == "dropped_by_purge").any()
    assert (folds["split_role"] == "dropped_by_embargo").any()
    assert (folds["split_role"] == "train").any()
    assert (folds["split_role"] == "valid").any()

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["n_folds"] == 3
    assert summary["embargo_sessions"] == 2
    assert sum(summary["n_dropped_by_purge_by_fold"].values()) >= 1
    assert sum(summary["n_dropped_by_embargo_by_fold"].values()) >= 1


def test_purged_cv_builds_valid_blocks_per_label_horizon_group(
    tmp_workspace: dict[str, Path],
) -> None:
    base = tmp_workspace["data"] / "purged_cv_case_grouped_horizons"
    labels_path = base / "labels_forward.parquet"
    calendar_path = base / "trading_calendar.parquet"
    output_dir = base / "output"
    base.mkdir(parents=True, exist_ok=True)

    sessions = pd.bdate_range("2026-01-05", periods=40, freq="B")
    calendar = pd.DataFrame({"date": sessions, "is_session": True})
    calendar.to_parquet(calendar_path, index=False)

    rows: list[dict[str, object]] = []
    instruments = [("SIMA", "AAA"), ("SIMB", "BBB")]

    # Long coverage short-horizon label.
    for idx in range(30):
        for instrument_id, ticker in instruments:
            rows.append(
                {
                    "date": sessions[idx],
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "horizon_days": 1,
                    "entry_date": sessions[idx + 1],
                    "exit_date": sessions[idx + 1],
                    "label_name": "fwd_ret_1d",
                    "label_value": 0.001,
                    "price_entry": 100.0,
                    "price_exit": 100.1,
                    "source_price_field": "close_adj",
                }
            )

    # Shorter coverage long-horizon label.
    for idx in range(15):
        for instrument_id, ticker in instruments:
            rows.append(
                {
                    "date": sessions[idx],
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "horizon_days": 20,
                    "entry_date": sessions[idx + 1],
                    "exit_date": sessions[idx + 20],
                    "label_name": "fwd_ret_20d",
                    "label_value": 0.01,
                    "price_entry": 100.0,
                    "price_exit": 101.0,
                    "source_price_field": "close_adj",
                }
            )

    pd.DataFrame(rows).to_parquet(labels_path, index=False)

    result = build_purged_cv(
        labels_path=labels_path,
        trading_calendar_path=calendar_path,
        output_dir=output_dir,
        n_folds=4,
        embargo_sessions=1,
        run_id="test_purged_cv_case_grouped_horizons",
    )
    folds = read_parquet(result.folds_path)

    grouped = (
        folds.groupby(["fold_id", "label_name", "horizon_days", "split_role"], dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
    )

    # Regression guard for the real bug:
    # each (fold_id, label_name, horizon_days) block must carry valid rows,
    # even when label/horizon groups have different date coverage.
    by_block = grouped[grouped["split_role"] == "valid"]
    expected_blocks = {
        (int(fold_id), str(label_name), int(horizon))
        for fold_id, label_name, horizon in folds[
            ["fold_id", "label_name", "horizon_days"]
        ].drop_duplicates().itertuples(index=False)
    }
    observed_blocks = {
        (int(fold_id), str(label_name), int(horizon))
        for fold_id, label_name, horizon in by_block[
            ["fold_id", "label_name", "horizon_days"]
        ].itertuples(index=False)
    }
    assert expected_blocks == observed_blocks
