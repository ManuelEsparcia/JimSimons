from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import pandas as pd
from pandas.api.types import is_numeric_dtype

# Allow direct script execution: `python simons_smallcap_swing/datasets/build_model_dataset.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "model_dataset_mvp_v1"
DEFAULT_LABEL_NAMES: tuple[str, ...] = ("fwd_ret_5d",)
DEFAULT_TARGET_TYPE = "continuous_forward_return"
VALID_SPLIT_ROLES: tuple[str, ...] = (
    "train",
    "valid",
    "test",
    "dropped_by_purge",
    "dropped_by_embargo",
)


FEATURES_INPUT_SCHEMA = DataSchema(
    name="model_dataset_features_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
    ),
    primary_key=("date", "instrument_id"),
    allow_extra_columns=True,
)

LABELS_INPUT_SCHEMA = DataSchema(
    name="model_dataset_labels_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("horizon_days", "int64", nullable=False),
        ColumnSpec("entry_date", "datetime64", nullable=False),
        ColumnSpec("exit_date", "datetime64", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("label_value", "number", nullable=False),
    ),
    primary_key=("date", "instrument_id", "horizon_days", "label_name"),
    allow_extra_columns=True,
)

SPLITS_INPUT_SCHEMA = DataSchema(
    name="model_dataset_splits_input_mvp",
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

MODEL_DATASET_SCHEMA = DataSchema(
    name="model_dataset_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("horizon_days", "int64", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("split_name", "string", nullable=False),
        ColumnSpec("split_role", "string", nullable=False),
        ColumnSpec("entry_date", "datetime64", nullable=False),
        ColumnSpec("exit_date", "datetime64", nullable=False),
        ColumnSpec("target_value", "float64", nullable=False),
        ColumnSpec("target_type", "string", nullable=False),
    ),
    primary_key=("date", "instrument_id", "horizon_days", "label_name"),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class BuildModelDatasetResult:
    dataset_path: Path
    summary_path: Path
    row_count: int
    n_features: int
    feature_names: tuple[str, ...]
    selected_label_names: tuple[str, ...]
    selected_horizons: tuple[int, ...]
    config_hash: str


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains invalid dates.")
    return parsed.dt.normalize()


def _normalize_label_names(label_names: Iterable[str] | None) -> tuple[str, ...]:
    if label_names is None:
        return DEFAULT_LABEL_NAMES
    cleaned = sorted({str(item).strip() for item in label_names if str(item).strip()})
    if not cleaned:
        raise ValueError("label_names must include at least one non-empty label name.")
    return tuple(cleaned)


def _normalize_horizons(horizons: Iterable[int] | None) -> tuple[int, ...]:
    if horizons is None:
        return ()
    normalized = sorted({int(item) for item in horizons})
    if any(item <= 0 for item in normalized):
        raise ValueError(f"horizon_days values must be positive. Received: {normalized}")
    return tuple(normalized)


def _config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_inputs(
    *,
    features_path: str | Path | None,
    labels_path: str | Path | None,
    purged_splits_path: str | Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    base = data_dir()
    features_source = (
        Path(features_path).expanduser().resolve()
        if features_path
        else base / "features" / "features_matrix.parquet"
    )
    labels_source = (
        Path(labels_path).expanduser().resolve()
        if labels_path
        else base / "labels" / "labels_forward.parquet"
    )
    splits_source = (
        Path(purged_splits_path).expanduser().resolve()
        if purged_splits_path
        else base / "labels" / "purged_splits.parquet"
    )

    features = read_parquet(features_source)
    labels = read_parquet(labels_source)
    splits = read_parquet(splits_source)
    return features, labels, splits, features_source, labels_source, splits_source


def _prepare_features(features: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    assert_schema(features, FEATURES_INPUT_SCHEMA)
    frame = features.copy()
    frame["date"] = _normalize_date(frame["date"], column="date")
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    if frame.duplicated(["date", "instrument_id"]).any():
        raise ValueError("features_matrix has duplicate (date, instrument_id) rows.")

    metadata_columns = {
        "date",
        "instrument_id",
        "ticker",
        "run_id",
        "config_hash",
        "built_ts_utc",
    }
    feature_cols = [
        col
        for col in frame.columns
        if col not in metadata_columns and is_numeric_dtype(frame[col])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns detected in features_matrix.")
    feature_cols = sorted(feature_cols)

    for col in feature_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    keep_cols = ["date", "instrument_id", "ticker", *feature_cols]
    return frame[keep_cols].sort_values(["date", "instrument_id"]).reset_index(drop=True), tuple(feature_cols)


def _prepare_labels(
    labels: pd.DataFrame,
    *,
    selected_label_names: tuple[str, ...],
    selected_horizons: tuple[int, ...],
) -> pd.DataFrame:
    assert_schema(labels, LABELS_INPUT_SCHEMA)
    frame = labels.copy()
    frame["date"] = _normalize_date(frame["date"], column="date")
    frame["entry_date"] = _normalize_date(frame["entry_date"], column="entry_date")
    frame["exit_date"] = _normalize_date(frame["exit_date"], column="exit_date")
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["horizon_days"] = pd.to_numeric(frame["horizon_days"], errors="coerce").astype("int64")
    frame["label_name"] = frame["label_name"].astype(str)
    frame["label_value"] = pd.to_numeric(frame["label_value"], errors="coerce")

    if frame["label_value"].isna().any():
        raise ValueError("labels_forward has null/non-numeric label_value values.")
    if not (frame["entry_date"] > frame["date"]).all():
        raise ValueError("labels_forward violates temporal rule: entry_date must be > date.")
    if not (frame["exit_date"] >= frame["entry_date"]).all():
        raise ValueError("labels_forward violates temporal rule: exit_date must be >= entry_date.")
    if frame.duplicated(["date", "instrument_id", "horizon_days", "label_name"]).any():
        raise ValueError(
            "labels_forward has duplicate (date, instrument_id, horizon_days, label_name) rows."
        )

    frame = frame[frame["label_name"].isin(set(selected_label_names))].copy()
    if selected_horizons:
        frame = frame[frame["horizon_days"].isin(set(selected_horizons))].copy()
    if frame.empty:
        raise ValueError(
            "No labels left after filtering label_name/horizon_days. "
            f"Selected label_names={selected_label_names}, selected_horizons={selected_horizons or 'ALL'}"
        )

    return frame.sort_values(["date", "instrument_id", "horizon_days", "label_name"]).reset_index(drop=True)


def _prepare_splits(
    splits: pd.DataFrame,
    *,
    selected_label_names: tuple[str, ...],
    selected_horizons: tuple[int, ...],
) -> pd.DataFrame:
    assert_schema(splits, SPLITS_INPUT_SCHEMA)
    frame = splits.copy()
    frame["date"] = _normalize_date(frame["date"], column="date")
    frame["entry_date"] = _normalize_date(frame["entry_date"], column="entry_date")
    frame["exit_date"] = _normalize_date(frame["exit_date"], column="exit_date")
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["horizon_days"] = pd.to_numeric(frame["horizon_days"], errors="coerce").astype("int64")
    frame["label_name"] = frame["label_name"].astype(str)
    frame["split_name"] = frame["split_name"].astype(str)
    frame["split_role"] = frame["split_role"].astype(str)

    if frame.duplicated(["date", "instrument_id", "horizon_days", "label_name"]).any():
        raise ValueError(
            "purged_splits has duplicate (date, instrument_id, horizon_days, label_name) rows."
        )
    invalid_roles = sorted(set(frame["split_role"].tolist()) - set(VALID_SPLIT_ROLES))
    if invalid_roles:
        raise ValueError(f"purged_splits contains invalid split_role values: {invalid_roles}")

    frame = frame[frame["label_name"].isin(set(selected_label_names))].copy()
    if selected_horizons:
        frame = frame[frame["horizon_days"].isin(set(selected_horizons))].copy()
    if frame.empty:
        raise ValueError(
            "No split rows left after filtering label_name/horizon_days. "
            f"Selected label_names={selected_label_names}, selected_horizons={selected_horizons or 'ALL'}"
        )
    return frame.sort_values(["date", "instrument_id", "horizon_days", "label_name"]).reset_index(drop=True)


def build_model_dataset(
    *,
    features_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    purged_splits_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    label_names: Iterable[str] | None = None,
    horizon_days: Iterable[int] | None = None,
    target_type: str = DEFAULT_TARGET_TYPE,
    run_id: str = MODULE_VERSION,
) -> BuildModelDatasetResult:
    logger = get_logger("datasets.build_model_dataset")
    selected_label_names = _normalize_label_names(label_names)
    selected_horizons = _normalize_horizons(horizon_days)
    target_type_clean = str(target_type).strip()
    if not target_type_clean:
        raise ValueError("target_type cannot be empty.")

    features, labels, splits, features_source, labels_source, splits_source = _load_inputs(
        features_path=features_path,
        labels_path=labels_path,
        purged_splits_path=purged_splits_path,
    )

    prepared_features, feature_cols = _prepare_features(features)
    prepared_labels = _prepare_labels(
        labels,
        selected_label_names=selected_label_names,
        selected_horizons=selected_horizons,
    )
    prepared_splits = _prepare_splits(
        splits,
        selected_label_names=selected_label_names,
        selected_horizons=selected_horizons,
    )

    join_key = ["date", "instrument_id", "horizon_days", "label_name"]
    labels_key = prepared_labels[join_key].copy()
    splits_key = prepared_splits[join_key].copy()

    labels_missing_split = labels_key.merge(splits_key, on=join_key, how="left", indicator=True)
    n_labels_without_split = int((labels_missing_split["_merge"] == "left_only").sum())
    splits_missing_label = splits_key.merge(labels_key, on=join_key, how="left", indicator=True)
    n_splits_without_labels = int((splits_missing_label["_merge"] == "left_only").sum())

    labels_with_splits = prepared_labels.merge(
        prepared_splits,
        on=join_key,
        how="inner",
        suffixes=("_label", "_split"),
    )
    if labels_with_splits.empty:
        raise ValueError("No rows left after joining labels_forward with purged_splits on label PK.")

    if not (labels_with_splits["entry_date_label"] == labels_with_splits["entry_date_split"]).all():
        raise ValueError("entry_date mismatch detected between labels_forward and purged_splits for same PK.")
    if not (labels_with_splits["exit_date_label"] == labels_with_splits["exit_date_split"]).all():
        raise ValueError("exit_date mismatch detected between labels_forward and purged_splits for same PK.")

    label_split_panel = labels_with_splits[
        [
            "date",
            "instrument_id",
            "ticker",
            "horizon_days",
            "label_name",
            "split_name",
            "split_role",
            "entry_date_label",
            "exit_date_label",
            "label_value",
        ]
    ].rename(
        columns={
            "entry_date_label": "entry_date",
            "exit_date_label": "exit_date",
            "label_value": "target_value",
        }
    )

    with_features = label_split_panel.merge(
        prepared_features,
        on=["date", "instrument_id"],
        how="left",
        suffixes=("_label", "_feat"),
        indicator=True,
    )
    n_label_split_without_features = int((with_features["_merge"] == "left_only").sum())
    with_features = with_features[with_features["_merge"] == "both"].drop(columns=["_merge"]).copy()
    if with_features.empty:
        raise ValueError("No rows left after joining label/split rows with features.")

    with_features["ticker"] = with_features["ticker_feat"].astype(str).str.upper().str.strip()
    ticker_conflict = (
        with_features["ticker_label"].astype(str).str.upper().str.strip()
        != with_features["ticker_feat"].astype(str).str.upper().str.strip()
    )
    if ticker_conflict.any():
        sample = with_features.loc[
            ticker_conflict,
            ["date", "instrument_id", "ticker_label", "ticker_feat"],
        ].head(10)
        raise ValueError(
            "Ticker mismatch between labels_forward and features_matrix for joined rows. "
            f"Sample:\n{sample.to_string(index=False)}"
        )

    with_features["target_value"] = pd.to_numeric(with_features["target_value"], errors="coerce")
    if with_features["target_value"].isna().any():
        raise ValueError("target_value is null/non-numeric after dataset join.")
    with_features["target_type"] = target_type_clean

    output = with_features[
        [
            "date",
            "instrument_id",
            "ticker",
            "horizon_days",
            "label_name",
            "split_name",
            "split_role",
            "entry_date",
            "exit_date",
            "target_value",
            "target_type",
            *feature_cols,
        ]
    ].copy()
    output = output.sort_values(["date", "instrument_id", "horizon_days", "label_name"]).reset_index(drop=True)

    if output.duplicated(join_key).any():
        raise ValueError("model_dataset contains duplicate logical PK rows.")
    invalid_roles = sorted(set(output["split_role"].tolist()) - set(VALID_SPLIT_ROLES))
    if invalid_roles:
        raise ValueError(f"model_dataset contains invalid split_role values: {invalid_roles}")

    assert_schema(output, MODEL_DATASET_SCHEMA)

    config_hash = _config_hash(
        {
            "version": MODULE_VERSION,
            "selected_label_names": list(selected_label_names),
            "selected_horizon_days": list(selected_horizons),
            "target_type": target_type_clean,
            "feature_columns": list(feature_cols),
            "paths": {
                "features_matrix": str(features_source),
                "labels_forward": str(labels_source),
                "purged_splits": str(splits_source),
            },
        }
    )
    built_ts_utc = datetime.now(UTC).isoformat()
    output["run_id"] = run_id
    output["config_hash"] = config_hash
    output["built_ts_utc"] = built_ts_utc

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (data_dir() / "datasets")
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = write_parquet(
        output,
        target_dir / "model_dataset.parquet",
        schema_name=MODEL_DATASET_SCHEMA.name,
        run_id=run_id,
    )

    role_counts = output["split_role"].value_counts().to_dict()
    join_drop_counts = {
        "labels_without_split": n_labels_without_split,
        "splits_without_labels": n_splits_without_labels,
        "label_split_without_features": n_label_split_without_features,
    }
    summary_payload = {
        "created_at_utc": built_ts_utc,
        "run_id": run_id,
        "config_hash": config_hash,
        "label_name_selected": list(selected_label_names),
        "horizon_days_present": sorted({int(v) for v in output["horizon_days"].unique().tolist()}),
        "n_rows_total": int(len(output)),
        "n_train": int(role_counts.get("train", 0)),
        "n_valid": int(role_counts.get("valid", 0)),
        "n_test": int(role_counts.get("test", 0)),
        "n_dropped_by_purge": int(role_counts.get("dropped_by_purge", 0)),
        "n_dropped_by_embargo": int(role_counts.get("dropped_by_embargo", 0)),
        "n_features": int(len(feature_cols)),
        "feature_names": list(feature_cols),
        "pct_missing_by_feature": {
            col: float(output[col].isna().mean()) for col in feature_cols
        },
        "target_missing_rate": float(output["target_value"].isna().mean()),
        "join_drop_counts": join_drop_counts,
        "input_rows": {
            "features": int(len(prepared_features)),
            "labels_selected": int(len(prepared_labels)),
            "splits_selected": int(len(prepared_splits)),
            "after_labels_splits_join": int(len(label_split_panel)),
            "after_final_join": int(len(output)),
        },
        "input_paths": {
            "features_matrix": str(features_source),
            "labels_forward": str(labels_source),
            "purged_splits": str(splits_source),
        },
        "output_path": str(dataset_path),
    }
    summary_path = target_dir / "model_dataset.summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "model_dataset_built",
        run_id=run_id,
        row_count=int(len(output)),
        n_features=int(len(feature_cols)),
        selected_label_names=list(selected_label_names),
        selected_horizons=list(selected_horizons) if selected_horizons else "ALL",
        output_path=str(dataset_path),
    )

    return BuildModelDatasetResult(
        dataset_path=dataset_path,
        summary_path=summary_path,
        row_count=int(len(output)),
        n_features=int(len(feature_cols)),
        feature_names=feature_cols,
        selected_label_names=selected_label_names,
        selected_horizons=selected_horizons,
        config_hash=config_hash,
    )


def _parse_csv_strings(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return values


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer.")
    try:
        return tuple(int(item) for item in values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build canonical model dataset from features, labels and splits.")
    parser.add_argument("--features-path", type=str, default=None)
    parser.add_argument("--labels-path", type=str, default=None)
    parser.add_argument("--purged-splits-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--label-names", type=_parse_csv_strings, default=DEFAULT_LABEL_NAMES)
    parser.add_argument("--horizon-days", type=_parse_csv_ints, default=())
    parser.add_argument("--target-type", type=str, default=DEFAULT_TARGET_TYPE)
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = build_model_dataset(
        features_path=args.features_path,
        labels_path=args.labels_path,
        purged_splits_path=args.purged_splits_path,
        output_dir=args.output_dir,
        label_names=args.label_names,
        horizon_days=args.horizon_days,
        target_type=args.target_type,
        run_id=args.run_id,
    )
    print("Model dataset built:")
    print(f"- path: {result.dataset_path}")
    print(f"- summary: {result.summary_path}")
    print(f"- rows: {result.row_count}")
    print(f"- n_features: {result.n_features}")
    print(f"- selected_label_names: {list(result.selected_label_names)}")
    print(
        "- selected_horizons: "
        + (str(list(result.selected_horizons)) if result.selected_horizons else "ALL")
    )


if __name__ == "__main__":
    main()
