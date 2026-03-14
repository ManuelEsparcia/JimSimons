from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

# Allow direct script execution: `python simons_smallcap_swing/labels/purged_cv.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "purged_cv_mvp_v2"
CV_METHOD = "purged_kfold_full_history_grouped_by_label_horizon"
VALID_SPLIT_ROLES: tuple[str, ...] = (
    "train",
    "valid",
    "dropped_by_purge",
    "dropped_by_embargo",
)

LABELS_INPUT_SCHEMA = DataSchema(
    name="purged_cv_labels_input_mvp",
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
    name="purged_cv_calendar_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("is_session", "bool", nullable=False),
    ),
    primary_key=("date",),
    allow_extra_columns=True,
)

PURGED_CV_OUTPUT_SCHEMA = DataSchema(
    name="purged_cv_folds_mvp",
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


@dataclass(frozen=True)
class PurgedCVResult:
    folds_path: Path
    summary_path: Path
    row_count: int
    n_folds: int
    config_hash: str


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains invalid dates.")
    return parsed.dt.normalize()


def _normalize_optional_text_list(values: Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    normalized = sorted({str(item).strip() for item in values if str(item).strip()})
    return tuple(normalized)


def _normalize_optional_int_list(values: Iterable[int] | None) -> tuple[int, ...]:
    if values is None:
        return ()
    normalized = sorted({int(item) for item in values})
    if any(item <= 0 for item in normalized):
        raise ValueError(f"horizon_days filters must be positive. Received: {normalized}")
    return tuple(normalized)


def _config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


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


def _load_inputs(
    *,
    labels_path: str | Path | None,
    trading_calendar_path: str | Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    base = data_dir()
    labels_source = (
        Path(labels_path).expanduser().resolve()
        if labels_path
        else base / "labels" / "labels_forward.parquet"
    )
    calendar_source = (
        Path(trading_calendar_path).expanduser().resolve()
        if trading_calendar_path
        else base / "reference" / "trading_calendar.parquet"
    )
    labels = read_parquet(labels_source)
    calendar = read_parquet(calendar_source)
    return labels, calendar, labels_source, calendar_source


def _prepare_inputs(
    *,
    labels: pd.DataFrame,
    calendar: pd.DataFrame,
    selected_label_names: tuple[str, ...],
    selected_horizons: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[pd.Timestamp, int]]:
    assert_schema(labels, LABELS_INPUT_SCHEMA)
    assert_schema(calendar, CALENDAR_INPUT_SCHEMA)

    labels = labels.copy()
    calendar = calendar.copy()

    labels["date"] = _normalize_date(labels["date"], column="date")
    labels["entry_date"] = _normalize_date(labels["entry_date"], column="entry_date")
    labels["exit_date"] = _normalize_date(labels["exit_date"], column="exit_date")
    labels["instrument_id"] = labels["instrument_id"].astype(str)
    labels["horizon_days"] = pd.to_numeric(labels["horizon_days"], errors="coerce").astype("int64")
    labels["label_name"] = labels["label_name"].astype(str)

    if labels.duplicated(["date", "instrument_id", "horizon_days", "label_name"]).any():
        raise ValueError(
            "labels_forward has duplicate (date, instrument_id, horizon_days, label_name) rows."
        )
    if not (labels["entry_date"] > labels["date"]).all():
        raise ValueError("labels_forward violates temporal rule: entry_date must be > date.")
    if not (labels["exit_date"] >= labels["entry_date"]).all():
        raise ValueError("labels_forward violates temporal rule: exit_date must be >= entry_date.")

    if selected_label_names:
        labels = labels[labels["label_name"].isin(set(selected_label_names))].copy()
    if selected_horizons:
        labels = labels[labels["horizon_days"].isin(set(selected_horizons))].copy()
    if labels.empty:
        raise ValueError(
            "No labels available after applying label_name/horizon_days filters. "
            f"label_names={selected_label_names or 'ALL'}, horizons={selected_horizons or 'ALL'}"
        )

    calendar["date"] = _normalize_date(calendar["date"], column="date")
    calendar["is_session"] = calendar["is_session"].astype(bool)
    if calendar.duplicated(["date"]).any():
        raise ValueError("trading_calendar has duplicate date rows.")

    sessions = pd.DatetimeIndex(sorted(calendar.loc[calendar["is_session"], "date"].unique()))
    if sessions.empty:
        raise ValueError("trading_calendar has no active sessions.")
    session_pos_map = {pd.Timestamp(d): int(i) for i, d in enumerate(sessions)}
    valid_session_set = set(session_pos_map.keys())

    for col in ("date", "entry_date", "exit_date"):
        invalid = labels.loc[~labels[col].isin(valid_session_set), col]
        if not invalid.empty:
            sample = sorted({str(pd.Timestamp(item).date()) for item in invalid.head(10).tolist()})
            raise ValueError(f"labels_forward contains {col} outside trading sessions. Sample: {sample}")

    labels = labels.sort_values(["date", "instrument_id", "horizon_days", "label_name"]).reset_index(drop=True)
    return labels, sessions, session_pos_map


def _build_fold_blocks(decision_dates: pd.DatetimeIndex, n_folds: int) -> list[pd.DatetimeIndex]:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2.")
    if len(decision_dates) < n_folds:
        raise ValueError(
            f"Not enough decision dates ({len(decision_dates)}) for n_folds={n_folds}."
        )
    blocks: list[pd.DatetimeIndex] = []
    split_blocks = np.array_split(decision_dates.to_numpy(), n_folds)
    for raw_block in split_blocks:
        block = pd.DatetimeIndex(pd.to_datetime(raw_block))
        if len(block) == 0:
            raise ValueError("Fold generation produced an empty valid block.")
        blocks.append(block)
    return blocks


def _validate_fold_integrity(
    fold_frame: pd.DataFrame,
    *,
    fold_id: int,
    embargo_sessions: int,
    session_pos_map: dict[pd.Timestamp, int],
) -> None:
    valid_rows = fold_frame[fold_frame["split_role"] == "valid"]
    if valid_rows.empty:
        raise ValueError(f"Fold {fold_id} has no valid rows.")

    valid_intervals = _merge_intervals(
        list(
            zip(
                valid_rows["entry_pos"].astype(int).tolist(),
                valid_rows["exit_pos"].astype(int).tolist(),
            )
        )
    )
    train_rows = fold_frame[fold_frame["split_role"] == "train"]

    for row in train_rows.itertuples(index=False):
        if _interval_overlaps_any(int(row.entry_pos), int(row.exit_pos), valid_intervals):
            raise ValueError(f"Fold {fold_id} leakage detected: train interval overlaps valid window.")

    valid_end_pos = int(valid_rows["decision_pos"].max())
    embargo_upper = valid_end_pos + int(embargo_sessions)
    if embargo_upper > valid_end_pos:
        for row in train_rows.itertuples(index=False):
            if valid_end_pos < int(row.decision_pos) <= embargo_upper:
                raise ValueError(f"Fold {fold_id} embargo violation detected in train rows.")

    if fold_frame.duplicated(["fold_id", "date", "instrument_id", "horizon_days", "label_name"]).any():
        raise ValueError(f"Fold {fold_id} contains duplicate PK rows.")


def build_purged_cv(
    *,
    labels_path: str | Path | None = None,
    trading_calendar_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    n_folds: int = 5,
    embargo_sessions: int = 1,
    label_names: Iterable[str] | None = None,
    horizon_days: Iterable[int] | None = None,
    run_id: str = MODULE_VERSION,
) -> PurgedCVResult:
    logger = get_logger("labels.purged_cv")
    n_folds_int = int(n_folds)
    embargo = int(embargo_sessions)
    if embargo < 0:
        raise ValueError("embargo_sessions must be >= 0.")

    selected_label_names = _normalize_optional_text_list(label_names)
    selected_horizons = _normalize_optional_int_list(horizon_days)

    labels, calendar, labels_source, calendar_source = _load_inputs(
        labels_path=labels_path,
        trading_calendar_path=trading_calendar_path,
    )
    labels, _sessions, session_pos_map = _prepare_inputs(
        labels=labels,
        calendar=calendar,
        selected_label_names=selected_label_names,
        selected_horizons=selected_horizons,
    )
    base = labels.copy()
    base["decision_pos"] = base["date"].map(session_pos_map).astype(int)
    base["entry_pos"] = base["entry_date"].map(session_pos_map).astype(int)
    base["exit_pos"] = base["exit_date"].map(session_pos_map).astype(int)

    fold_frames: list[pd.DataFrame] = []
    fold_ranges: list[dict[str, Any]] = []
    n_train_by_fold: dict[str, int] = {}
    n_valid_by_fold: dict[str, int] = {}
    n_purge_by_fold: dict[str, int] = {}
    n_embargo_by_fold: dict[str, int] = {}

    grouped = base.groupby(["label_name", "horizon_days"], sort=True, dropna=False)
    for (label_name_value, horizon_value), group_frame in grouped:
        decision_dates = pd.DatetimeIndex(sorted(group_frame["date"].unique()))
        fold_blocks = _build_fold_blocks(decision_dates, n_folds_int)

        for fold_idx, block in enumerate(fold_blocks, start=1):
            valid_dates = set(pd.DatetimeIndex(block).tolist())
            fold = group_frame.copy()
            fold["fold_id"] = int(fold_idx)
            fold["split_role"] = "train_candidate"
            fold.loc[fold["date"].isin(valid_dates), "split_role"] = "valid"

            valid_rows = fold[fold["split_role"] == "valid"].copy()
            if valid_rows.empty:
                raise ValueError(
                    f"Fold {fold_idx} has no valid rows after assignment "
                    f"for label='{label_name_value}' horizon={int(horizon_value)}."
                )

            valid_intervals = _merge_intervals(
                list(
                    zip(
                        valid_rows["entry_pos"].astype(int).tolist(),
                        valid_rows["exit_pos"].astype(int).tolist(),
                    )
                )
            )
            train_mask = fold["split_role"].eq("train_candidate")
            purge_mask = fold.loc[train_mask].apply(
                lambda row: _interval_overlaps_any(
                    int(row["entry_pos"]),
                    int(row["exit_pos"]),
                    valid_intervals,
                ),
                axis=1,
            )
            fold.loc[fold.loc[train_mask].index[purge_mask.to_numpy()], "split_role"] = (
                "dropped_by_purge"
            )

            valid_end_pos = int(valid_rows["decision_pos"].max())
            embargo_upper = valid_end_pos + embargo
            if embargo > 0:
                embargo_mask = (
                    fold["split_role"].eq("train_candidate")
                    & (fold["decision_pos"] > valid_end_pos)
                    & (fold["decision_pos"] <= embargo_upper)
                )
                fold.loc[embargo_mask, "split_role"] = "dropped_by_embargo"

            fold.loc[fold["split_role"].eq("train_candidate"), "split_role"] = "train"
            fold["split_role"] = fold["split_role"].astype(str)

            invalid_roles = sorted(set(fold["split_role"].tolist()) - set(VALID_SPLIT_ROLES))
            if invalid_roles:
                raise ValueError(f"Fold {fold_idx} produced invalid split_role values: {invalid_roles}")

            _validate_fold_integrity(
                fold,
                fold_id=fold_idx,
                embargo_sessions=embargo,
                session_pos_map=session_pos_map,
            )

            fold_ranges.append(
                {
                    "fold_id": int(fold_idx),
                    "label_name": str(label_name_value),
                    "horizon_days": int(horizon_value),
                    "valid_start": str(pd.Timestamp(min(valid_dates)).date()),
                    "valid_end": str(pd.Timestamp(max(valid_dates)).date()),
                    "n_valid_dates": int(len(valid_dates)),
                }
            )
            fold_key = str(fold_idx)
            n_train_by_fold[fold_key] = n_train_by_fold.get(fold_key, 0) + int(
                (fold["split_role"] == "train").sum()
            )
            n_valid_by_fold[fold_key] = n_valid_by_fold.get(fold_key, 0) + int(
                (fold["split_role"] == "valid").sum()
            )
            n_purge_by_fold[fold_key] = n_purge_by_fold.get(fold_key, 0) + int(
                (fold["split_role"] == "dropped_by_purge").sum()
            )
            n_embargo_by_fold[fold_key] = n_embargo_by_fold.get(fold_key, 0) + int(
                (fold["split_role"] == "dropped_by_embargo").sum()
            )

            fold_frames.append(fold)

    output = pd.concat(fold_frames, ignore_index=True)
    output = output[
        [
            "fold_id",
            "date",
            "instrument_id",
            "horizon_days",
            "label_name",
            "split_role",
            "entry_date",
            "exit_date",
        ]
    ].copy()
    output = output.sort_values(
        ["fold_id", "date", "instrument_id", "horizon_days", "label_name"]
    ).reset_index(drop=True)

    assert_schema(output, PURGED_CV_OUTPUT_SCHEMA)
    if output.empty:
        raise ValueError("purged_cv output is empty.")

    built_ts = datetime.now(UTC).isoformat()
    config_hash = _config_hash(
        {
            "version": MODULE_VERSION,
            "cv_method": CV_METHOD,
            "n_folds": n_folds_int,
            "embargo_sessions": embargo,
            "label_names": list(selected_label_names),
            "horizon_days": list(selected_horizons),
            "paths": {
                "labels_forward": str(labels_source),
                "trading_calendar": str(calendar_source),
            },
        }
    )

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (data_dir() / "labels")
    target_dir.mkdir(parents=True, exist_ok=True)

    folds_path = write_parquet(
        output.assign(run_id=run_id, config_hash=config_hash, built_ts_utc=built_ts),
        target_dir / "purged_cv_folds.parquet",
        schema_name=PURGED_CV_OUTPUT_SCHEMA.name,
        run_id=run_id,
    )

    summary_payload: dict[str, Any] = {
        "created_at_utc": built_ts,
        "run_id": run_id,
        "config_hash": config_hash,
        "cv_method": CV_METHOD,
        "n_folds": n_folds_int,
        "embargo_sessions": embargo,
        "folds_time_ranges": fold_ranges,
        "n_rows_total": int(len(output)),
        "n_train_by_fold": n_train_by_fold,
        "n_valid_by_fold": n_valid_by_fold,
        "n_dropped_by_purge_by_fold": n_purge_by_fold,
        "n_dropped_by_embargo_by_fold": n_embargo_by_fold,
        "horizons_present": sorted({int(item) for item in output["horizon_days"].tolist()}),
        "label_names_present": sorted({str(item) for item in output["label_name"].tolist()}),
        "paths": {
            "labels_forward": str(labels_source),
            "trading_calendar": str(calendar_source),
            "output_path": str(folds_path),
        },
    }
    summary_path = target_dir / "purged_cv_folds.summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "purged_cv_built",
        run_id=run_id,
        n_folds=n_folds_int,
        n_rows=int(len(output)),
        output_path=str(folds_path),
        summary_path=str(summary_path),
    )

    return PurgedCVResult(
        folds_path=folds_path,
        summary_path=summary_path,
        row_count=int(len(output)),
        n_folds=n_folds_int,
        config_hash=config_hash,
    )


def _parse_csv_text_list(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    items = [part.strip() for part in str(value).split(",") if part.strip()]
    return tuple(sorted(set(items)))


def _parse_csv_int_list(value: str | None) -> tuple[int, ...]:
    if value is None:
        return ()
    items = [part.strip() for part in str(value).split(",") if part.strip()]
    if not items:
        return ()
    return tuple(sorted({int(item) for item in items}))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build purged temporal multi-fold CV splits.")
    parser.add_argument("--labels-path", type=str, default=None)
    parser.add_argument("--trading-calendar-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo-sessions", type=int, default=1)
    parser.add_argument("--label-names", type=str, default=None, help="Comma-separated label names.")
    parser.add_argument("--horizon-days", type=str, default=None, help="Comma-separated horizon days.")
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = build_purged_cv(
        labels_path=args.labels_path,
        trading_calendar_path=args.trading_calendar_path,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        embargo_sessions=args.embargo_sessions,
        label_names=_parse_csv_text_list(args.label_names),
        horizon_days=_parse_csv_int_list(args.horizon_days),
        run_id=args.run_id,
    )
    print("Purged CV folds built:")
    print(f"- path: {result.folds_path}")
    print(f"- summary: {result.summary_path}")
    print(f"- rows: {result.row_count}")
    print(f"- n_folds: {result.n_folds}")


if __name__ == "__main__":
    main()
