from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

# Allow direct script execution: `python simons_smallcap_swing/labels/purged_splits.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "purged_splits_mvp_v1"
SPLIT_NAME = "holdout_temporal_purged"
VALID_ROLES: tuple[str, ...] = (
    "train",
    "valid",
    "test",
    "dropped_by_purge",
    "dropped_by_embargo",
)

LABELS_INPUT_SCHEMA = DataSchema(
    name="purged_splits_labels_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("horizon_days", "int64", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("entry_date", "datetime64", nullable=False),
        ColumnSpec("exit_date", "datetime64", nullable=False),
    ),
    primary_key=("date", "instrument_id", "horizon_days", "label_name"),
    allow_extra_columns=True,
)

CALENDAR_INPUT_SCHEMA = DataSchema(
    name="purged_splits_calendar_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("is_session", "bool", nullable=False),
    ),
    primary_key=("date",),
    allow_extra_columns=True,
)

PURGED_SPLITS_SCHEMA = DataSchema(
    name="purged_splits_output_mvp",
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


@dataclass(frozen=True)
class PurgedSplitsResult:
    splits_path: Path
    summary_path: Path
    row_count: int
    n_train: int
    n_valid: int
    n_test: int
    n_dropped_by_purge: int
    n_dropped_by_embargo: int
    config_hash: str


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains invalid dates.")
    return parsed.dt.normalize()


def _parse_date_argument(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid date argument: {value}")
    return pd.Timestamp(parsed).normalize()


def _config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_inputs(
    *,
    labels_path: str | Path | None,
    trading_calendar_path: str | Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    base_data = data_dir()
    labels_source = (
        Path(labels_path).expanduser().resolve()
        if labels_path
        else base_data / "labels" / "labels_forward.parquet"
    )
    calendar_source = (
        Path(trading_calendar_path).expanduser().resolve()
        if trading_calendar_path
        else base_data / "reference" / "trading_calendar.parquet"
    )
    labels = read_parquet(labels_source)
    calendar = read_parquet(calendar_source)
    return labels, calendar, labels_source, calendar_source


def _prepare_inputs(
    labels: pd.DataFrame,
    calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[pd.Timestamp, int]]:
    assert_schema(labels, LABELS_INPUT_SCHEMA)
    assert_schema(calendar, CALENDAR_INPUT_SCHEMA)

    labels = labels.copy()
    calendar = calendar.copy()

    labels["date"] = _normalize_date(labels["date"], column="date")
    labels["entry_date"] = _normalize_date(labels["entry_date"], column="entry_date")
    labels["exit_date"] = _normalize_date(labels["exit_date"], column="exit_date")
    labels["instrument_id"] = labels["instrument_id"].astype(str)
    labels["label_name"] = labels["label_name"].astype(str)
    labels["horizon_days"] = pd.to_numeric(labels["horizon_days"], errors="coerce").astype("int64")

    if labels.empty:
        raise ValueError("labels_forward is empty.")
    if labels.duplicated(["date", "instrument_id", "horizon_days", "label_name"]).any():
        raise ValueError(
            "labels_forward has duplicate (date, instrument_id, horizon_days, label_name) rows."
        )
    if not (labels["entry_date"] > labels["date"]).all():
        raise ValueError("labels_forward violates temporal rule: entry_date must be > date.")
    if not (labels["exit_date"] >= labels["entry_date"]).all():
        raise ValueError("labels_forward violates temporal rule: exit_date must be >= entry_date.")

    calendar["date"] = _normalize_date(calendar["date"], column="date")
    calendar["is_session"] = calendar["is_session"].astype(bool)
    if calendar.duplicated(["date"]).any():
        raise ValueError("trading_calendar has duplicate date rows.")

    sessions = pd.DatetimeIndex(sorted(calendar.loc[calendar["is_session"], "date"].unique()))
    if sessions.empty:
        raise ValueError("trading_calendar has no active sessions.")
    session_pos_map = {pd.Timestamp(d): int(i) for i, d in enumerate(sessions)}

    invalid_label_dates = labels.loc[
        ~labels["date"].isin(set(session_pos_map.keys())),
        "date",
    ]
    if not invalid_label_dates.empty:
        sample = sorted({str(pd.Timestamp(d).date()) for d in invalid_label_dates.head(10).tolist()})
        raise ValueError(f"labels_forward contains decision dates outside trading sessions. Sample: {sample}")

    invalid_entry_dates = labels.loc[
        ~labels["entry_date"].isin(set(session_pos_map.keys())),
        "entry_date",
    ]
    if not invalid_entry_dates.empty:
        sample = sorted({str(pd.Timestamp(d).date()) for d in invalid_entry_dates.head(10).tolist()})
        raise ValueError(f"labels_forward contains entry_date outside trading sessions. Sample: {sample}")

    invalid_exit_dates = labels.loc[
        ~labels["exit_date"].isin(set(session_pos_map.keys())),
        "exit_date",
    ]
    if not invalid_exit_dates.empty:
        sample = sorted({str(pd.Timestamp(d).date()) for d in invalid_exit_dates.head(10).tolist()})
        raise ValueError(f"labels_forward contains exit_date outside trading sessions. Sample: {sample}")

    labels = labels.sort_values(["date", "instrument_id", "horizon_days", "label_name"]).reset_index(drop=True)
    return labels, sessions, session_pos_map


def _resolve_holdout_blocks(
    *,
    decision_dates: pd.DatetimeIndex,
    valid_fraction: float,
    test_fraction: float,
    valid_start: pd.Timestamp | None,
    valid_end: pd.Timestamp | None,
    test_start: pd.Timestamp | None,
    test_end: pd.Timestamp | None,
) -> tuple[set[pd.Timestamp], set[pd.Timestamp], dict[str, str]]:
    if len(decision_dates) < 5:
        raise ValueError("Need at least 5 decision sessions to build purged holdout splits.")

    if (valid_start is None) != (valid_end is None):
        raise ValueError("valid_start and valid_end must be provided together.")
    if (test_start is None) != (test_end is None):
        raise ValueError("test_start and test_end must be provided together.")

    dates_list = [pd.Timestamp(d) for d in decision_dates.tolist()]
    valid_dates: set[pd.Timestamp]
    test_dates: set[pd.Timestamp]

    if valid_start is not None and test_start is not None:
        if valid_start > valid_end:
            raise ValueError("valid_start must be <= valid_end.")
        if test_start > test_end:
            raise ValueError("test_start must be <= test_end.")
        if not (valid_end < test_start):
            raise ValueError("Expected non-overlapping temporal order: valid_end < test_start.")

        valid_dates = {d for d in dates_list if valid_start <= d <= valid_end}
        test_dates = {d for d in dates_list if test_start <= d <= test_end}
        if not valid_dates:
            raise ValueError("Provided valid date range does not intersect labels decision dates.")
        if not test_dates:
            raise ValueError("Provided test date range does not intersect labels decision dates.")
    else:
        if not (0.0 < valid_fraction < 0.5):
            raise ValueError("valid_fraction must be in (0, 0.5).")
        if not (0.0 < test_fraction < 0.5):
            raise ValueError("test_fraction must be in (0, 0.5).")
        if valid_fraction + test_fraction >= 0.8:
            raise ValueError("valid_fraction + test_fraction is too large for stable train coverage.")

        n_sessions = len(dates_list)
        n_test = max(1, int(np.ceil(n_sessions * test_fraction)))
        n_valid = max(1, int(np.ceil(n_sessions * valid_fraction)))
        test_slice = dates_list[-n_test:]
        valid_slice = dates_list[-(n_test + n_valid) : -n_test]
        if not valid_slice:
            raise ValueError("Unable to allocate a non-empty valid block. Increase data range.")
        valid_dates = set(valid_slice)
        test_dates = set(test_slice)

    train_dates = [d for d in dates_list if d not in valid_dates and d not in test_dates]
    if not train_dates:
        raise ValueError("Split produced empty train candidate date set.")

    valid_min = min(valid_dates)
    valid_max = max(valid_dates)
    test_min = min(test_dates)
    test_max = max(test_dates)
    if not (valid_min <= valid_max < test_min <= test_max):
        raise ValueError("Holdout temporal blocks are not ordered as valid then test.")

    ranges = {
        "valid_start": str(valid_min.date()),
        "valid_end": str(valid_max.date()),
        "test_start": str(test_min.date()),
        "test_end": str(test_max.date()),
        "train_start_candidate": str(min(train_dates).date()),
        "train_end_candidate": str(max(train_dates).date()),
    }
    return valid_dates, test_dates, ranges


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals)
    merged: list[tuple[int, int]] = []
    cur_start, cur_end = sorted_intervals[0]
    for start, end in sorted_intervals[1:]:
        if start <= cur_end + 1:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def _interval_overlaps_any(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    for i_start, i_end in intervals:
        if i_start > end:
            return False
        if i_end < start:
            continue
        return True
    return False


def build_purged_splits(
    *,
    labels_path: str | Path | None = None,
    trading_calendar_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    valid_fraction: float = 0.20,
    test_fraction: float = 0.20,
    embargo_sessions: int = 1,
    valid_start: str | None = None,
    valid_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
    run_id: str = MODULE_VERSION,
) -> PurgedSplitsResult:
    logger = get_logger("labels.purged_splits")
    embargo = int(embargo_sessions)
    if embargo < 0:
        raise ValueError("embargo_sessions must be >= 0.")

    labels, calendar, labels_source, calendar_source = _load_inputs(
        labels_path=labels_path,
        trading_calendar_path=trading_calendar_path,
    )
    labels, sessions, session_pos_map = _prepare_inputs(labels, calendar)
    decision_dates = pd.DatetimeIndex(sorted(labels["date"].unique()))

    valid_dates, test_dates, block_ranges = _resolve_holdout_blocks(
        decision_dates=decision_dates,
        valid_fraction=float(valid_fraction),
        test_fraction=float(test_fraction),
        valid_start=_parse_date_argument(valid_start),
        valid_end=_parse_date_argument(valid_end),
        test_start=_parse_date_argument(test_start),
        test_end=_parse_date_argument(test_end),
    )

    split = labels.copy()
    split["split_name"] = SPLIT_NAME
    split["split_role"] = "train_candidate"
    split.loc[split["date"].isin(valid_dates), "split_role"] = "valid"
    split.loc[split["date"].isin(test_dates), "split_role"] = "test"

    split["decision_pos"] = split["date"].map(session_pos_map).astype(int)
    split["entry_pos"] = split["entry_date"].map(session_pos_map).astype(int)
    split["exit_pos"] = split["exit_date"].map(session_pos_map).astype(int)

    eval_rows = split[split["split_role"].isin({"valid", "test"})].copy()
    if eval_rows.empty:
        raise ValueError("No evaluation rows (valid/test) were allocated.")
    eval_intervals = _merge_intervals(
        list(zip(eval_rows["entry_pos"].astype(int).tolist(), eval_rows["exit_pos"].astype(int).tolist()))
    )
    if not eval_intervals:
        raise ValueError("Failed to build evaluation intervals for purge.")

    train_mask = split["split_role"].eq("train_candidate")
    purge_mask = split.loc[train_mask].apply(
        lambda row: _interval_overlaps_any(int(row["entry_pos"]), int(row["exit_pos"]), eval_intervals),
        axis=1,
    )
    purged_index = split.loc[train_mask].index[purge_mask.to_numpy()]
    split.loc[purged_index, "split_role"] = "dropped_by_purge"

    embargo_positions: set[int] = set()
    if embargo > 0:
        for role in ("valid", "test"):
            role_rows = split[split["split_role"].eq(role)]
            if role_rows.empty:
                continue
            role_end = int(role_rows["decision_pos"].max())
            embargo_positions.update(
                i
                for i in range(role_end + 1, role_end + embargo + 1)
                if 0 <= i < len(sessions)
            )

    embargo_dates = {pd.Timestamp(sessions[pos]) for pos in sorted(embargo_positions)}
    embargo_mask = split["split_role"].eq("train_candidate") & split["date"].isin(embargo_dates)
    split.loc[embargo_mask, "split_role"] = "dropped_by_embargo"

    split.loc[split["split_role"].eq("train_candidate"), "split_role"] = "train"

    if not split["split_role"].isin(set(VALID_ROLES)).all():
        invalid = sorted(set(split["split_role"]) - set(VALID_ROLES))
        raise ValueError(f"Invalid split roles found: {invalid}")

    # Structural anti-leakage validation: final train intervals cannot overlap valid/test windows.
    final_train = split[split["split_role"].eq("train")]
    if not final_train.empty:
        contamination = final_train.apply(
            lambda row: _interval_overlaps_any(int(row["entry_pos"]), int(row["exit_pos"]), eval_intervals),
            axis=1,
        )
        if contamination.any():
            raise ValueError("Purged split still contains train rows contaminating valid/test windows.")
        if embargo_dates:
            if final_train["date"].isin(embargo_dates).any():
                raise ValueError("Purged split still contains train rows inside embargo window.")

    # Temporal ordering sanity for core roles.
    if not final_train.empty:
        valid_rows = split[split["split_role"].eq("valid")]
        test_rows = split[split["split_role"].eq("test")]
        if valid_rows.empty or test_rows.empty:
            raise ValueError("Expected non-empty valid and test roles.")
        if not (
            pd.Timestamp(final_train["date"].max())
            < pd.Timestamp(valid_rows["date"].min())
            <= pd.Timestamp(valid_rows["date"].max())
            < pd.Timestamp(test_rows["date"].min())
        ):
            raise ValueError("Temporal ordering check failed: expected train < valid < test by decision date.")

    output = split[
        [
            "date",
            "instrument_id",
            "horizon_days",
            "label_name",
            "split_name",
            "split_role",
            "entry_date",
            "exit_date",
        ]
    ].copy()
    output = output.sort_values(["date", "instrument_id", "horizon_days", "label_name"]).reset_index(drop=True)
    assert_schema(output, PURGED_SPLITS_SCHEMA)
    if output.duplicated(["date", "instrument_id", "horizon_days", "label_name"]).any():
        raise ValueError("purged_splits output contains duplicate logical PK rows.")

    config_hash = _config_hash(
        {
            "version": MODULE_VERSION,
            "split_method": SPLIT_NAME,
            "valid_fraction": float(valid_fraction),
            "test_fraction": float(test_fraction),
            "embargo_sessions": embargo,
            "explicit_ranges": {
                "valid_start": valid_start,
                "valid_end": valid_end,
                "test_start": test_start,
                "test_end": test_end,
            },
            "inputs": {
                "labels_forward": str(labels_source),
                "trading_calendar": str(calendar_source),
            },
        }
    )
    built_ts_utc = datetime.now(UTC).isoformat()
    output["run_id"] = run_id
    output["config_hash"] = config_hash
    output["built_ts_utc"] = built_ts_utc

    counts = output["split_role"].value_counts().to_dict()
    n_train = int(counts.get("train", 0))
    n_valid = int(counts.get("valid", 0))
    n_test = int(counts.get("test", 0))
    n_dropped_by_purge = int(counts.get("dropped_by_purge", 0))
    n_dropped_by_embargo = int(counts.get("dropped_by_embargo", 0))

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (data_dir() / "labels")
    target_dir.mkdir(parents=True, exist_ok=True)
    splits_path = write_parquet(
        output,
        target_dir / "purged_splits.parquet",
        schema_name=PURGED_SPLITS_SCHEMA.name,
        run_id=run_id,
    )

    train_rows = output[output["split_role"].eq("train")]
    summary_payload = {
        "created_at_utc": built_ts_utc,
        "run_id": run_id,
        "config_hash": config_hash,
        "split_method": SPLIT_NAME,
        "train_start": str(pd.Timestamp(train_rows["date"].min()).date()) if not train_rows.empty else None,
        "train_end": str(pd.Timestamp(train_rows["date"].max()).date()) if not train_rows.empty else None,
        "valid_start": block_ranges["valid_start"],
        "valid_end": block_ranges["valid_end"],
        "test_start": block_ranges["test_start"],
        "test_end": block_ranges["test_end"],
        "embargo_sessions": embargo,
        "n_rows_total": int(len(output)),
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "n_dropped_by_purge": n_dropped_by_purge,
        "n_dropped_by_embargo": n_dropped_by_embargo,
        "horizons_present": sorted({int(x) for x in output["horizon_days"].unique().tolist()}),
        "label_names_present": sorted({str(x) for x in output["label_name"].unique().tolist()}),
        "input_paths": {
            "labels_forward": str(labels_source),
            "trading_calendar": str(calendar_source),
        },
        "output_path": str(splits_path),
    }
    summary_path = target_dir / "purged_splits.summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "purged_splits_built",
        run_id=run_id,
        row_count=int(len(output)),
        n_train=n_train,
        n_valid=n_valid,
        n_test=n_test,
        n_dropped_by_purge=n_dropped_by_purge,
        n_dropped_by_embargo=n_dropped_by_embargo,
        output_path=str(splits_path),
    )

    return PurgedSplitsResult(
        splits_path=splits_path,
        summary_path=summary_path,
        row_count=int(len(output)),
        n_train=n_train,
        n_valid=n_valid,
        n_test=n_test,
        n_dropped_by_purge=n_dropped_by_purge,
        n_dropped_by_embargo=n_dropped_by_embargo,
        config_hash=config_hash,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build MVP purged temporal train/valid/test splits.")
    parser.add_argument("--labels-path", type=str, default=None)
    parser.add_argument("--trading-calendar-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--valid-fraction", type=float, default=0.20)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--embargo-sessions", type=int, default=1)
    parser.add_argument("--valid-start", type=str, default=None)
    parser.add_argument("--valid-end", type=str, default=None)
    parser.add_argument("--test-start", type=str, default=None)
    parser.add_argument("--test-end", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = build_purged_splits(
        labels_path=args.labels_path,
        trading_calendar_path=args.trading_calendar_path,
        output_dir=args.output_dir,
        valid_fraction=args.valid_fraction,
        test_fraction=args.test_fraction,
        embargo_sessions=args.embargo_sessions,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        test_start=args.test_start,
        test_end=args.test_end,
        run_id=args.run_id,
    )
    print("Purged splits built:")
    print(f"- path: {result.splits_path}")
    print(f"- summary: {result.summary_path}")
    print(f"- rows: {result.row_count}")
    print(f"- n_train: {result.n_train}")
    print(f"- n_valid: {result.n_valid}")
    print(f"- n_test: {result.n_test}")
    print(f"- n_dropped_by_purge: {result.n_dropped_by_purge}")
    print(f"- n_dropped_by_embargo: {result.n_dropped_by_embargo}")


if __name__ == "__main__":
    main()
