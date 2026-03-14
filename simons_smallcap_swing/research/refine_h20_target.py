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
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import Ridge

# Allow direct script execution:
# `python simons_smallcap_swing/research/refine_h20_target.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "refine_h20_target_mvp_v1"
PRIMARY_METRIC = "mse"
DEFAULT_MODEL_DATASET_PATH = "datasets/regression_h20/model_dataset.parquet"
DEFAULT_FOLDS_PATH = "labels/purged_cv_folds.parquet"

VARIANT_RAW = "raw_target_baseline"
VARIANT_CLIPPED = "clipped_target"
VARIANT_VOL_SCALED = "vol_scaled_target"
VARIANT_MARKET_RELATIVE = "market_relative_target"
DEFAULT_VARIANTS: tuple[str, ...] = (
    VARIANT_RAW,
    VARIANT_CLIPPED,
    VARIANT_VOL_SCALED,
    VARIANT_MARKET_RELATIVE,
)

VALID_ROLES: tuple[str, ...] = (
    "train",
    "valid",
    "dropped_by_purge",
    "dropped_by_embargo",
)
TRAIN_ROLE = "train"
VALID_ROLE = "valid"

DATASET_REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "instrument_id",
    "ticker",
    "horizon_days",
    "label_name",
    "target_value",
    "target_type",
)
FOLDS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "fold_id",
    "date",
    "instrument_id",
    "horizon_days",
    "label_name",
    "split_role",
)
FEATURE_EXCLUDED_COLUMNS: set[str] = {
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
    "run_id",
    "config_hash",
    "built_ts_utc",
}

DEFAULT_VOL_SCALE_COLUMNS: tuple[str, ...] = ("vol_20d", "vol_5d")
DEFAULT_MARKET_REFERENCE_COLUMNS: tuple[str, ...] = (
    "mkt_equal_weight_return_lag1",
    "mkt_return_1d_lag1",
)

RESULTS_SCHEMA = DataSchema(
    name="refine_h20_target_results_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("target_variant", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("target_type", "string", nullable=False),
        ColumnSpec("horizon_days", "int64", nullable=False),
        ColumnSpec("primary_metric", "string", nullable=False),
        ColumnSpec("mean_valid_primary_metric", "float64", nullable=False),
        ColumnSpec("median_valid_primary_metric", "float64", nullable=False),
        ColumnSpec("std_valid_primary_metric", "float64", nullable=False),
        ColumnSpec("improvement_vs_baseline", "float64", nullable=False),
        ColumnSpec("improvement_vs_dummy", "float64", nullable=False),
        ColumnSpec("winner_vs_baseline", "string", nullable=False),
        ColumnSpec("winner_vs_dummy", "string", nullable=False),
    ),
    primary_key=(),
    allow_extra_columns=True,
)

FOLD_SCHEMA = DataSchema(
    name="refine_h20_target_fold_metrics_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("target_variant", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("target_type", "string", nullable=False),
        ColumnSpec("horizon_days", "int64", nullable=False),
        ColumnSpec("fold_id", "int64", nullable=False),
        ColumnSpec("primary_metric", "string", nullable=False),
        ColumnSpec("model_valid_primary_metric", "float64", nullable=False),
        ColumnSpec("dummy_valid_primary_metric", "float64", nullable=False),
        ColumnSpec("baseline_valid_primary_metric", "float64", nullable=False),
        ColumnSpec("improvement_vs_dummy", "float64", nullable=False),
        ColumnSpec("improvement_vs_baseline", "float64", nullable=False),
    ),
    primary_key=(),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class RefineH20TargetResult:
    results_path: Path
    summary_path: Path
    fold_metrics_path: Path
    n_rows: int
    config_hash: str


def _cfg_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' has invalid dates.")
    return parsed.dt.normalize()


def _parse_csv_floats(text: str | None) -> tuple[float, ...]:
    if not text:
        return ()
    values: list[float] = []
    for token in text.split(","):
        item = token.strip()
        if item:
            values.append(float(item))
    return tuple(values)


def _parse_csv_texts(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _normalize_float_grid(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    normalized = tuple(sorted({float(v) for v in values}))
    if not normalized:
        raise ValueError(f"{name} cannot be empty.")
    if any(v <= 0 for v in normalized):
        raise ValueError(f"{name} must contain positive values. Received: {normalized}")
    return normalized


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], *, name: str) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _winner(
    improvement: float,
    *,
    tie_tolerance: float,
    positive_label: str,
    negative_label: str,
) -> str:
    if abs(float(improvement)) <= float(tie_tolerance):
        return "tie"
    return positive_label if float(improvement) > 0 else negative_label


def _first_available_column(
    preferred: tuple[str, ...],
    *,
    available_columns: tuple[str, ...],
) -> str | None:
    available_set = set(available_columns)
    for name in preferred:
        if name in available_set:
            return str(name)
    return None


def _resolve_fold_column_with_train_coverage(
    *,
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    for column in candidates:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            return str(column)
    return None


def _load_joined_inputs(
    *,
    model_dataset_path: Path,
    purged_cv_folds_path: Path,
    label_name: str,
    target_type: str,
    horizon_days: int,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[int, ...]]:
    dataset = read_parquet(model_dataset_path).copy()
    folds = read_parquet(purged_cv_folds_path).copy()
    _require_columns(dataset, DATASET_REQUIRED_COLUMNS, name="model_dataset")
    _require_columns(folds, FOLDS_REQUIRED_COLUMNS, name="purged_cv_folds")

    dataset["date"] = _normalize_date(dataset["date"], column="date")
    dataset["instrument_id"] = dataset["instrument_id"].astype(str)
    dataset["horizon_days"] = pd.to_numeric(dataset["horizon_days"], errors="coerce").astype("Int64")
    dataset["label_name"] = dataset["label_name"].astype(str)
    dataset["target_type"] = dataset["target_type"].astype(str)
    dataset["target_value"] = pd.to_numeric(dataset["target_value"], errors="coerce")
    if dataset["horizon_days"].isna().any():
        raise ValueError("model_dataset has invalid horizon_days.")
    if dataset["target_value"].isna().any():
        raise ValueError("model_dataset has invalid target_value.")
    dataset["horizon_days"] = dataset["horizon_days"].astype("int64")
    if dataset.duplicated(["date", "instrument_id", "horizon_days", "label_name"]).any():
        raise ValueError("model_dataset has duplicate PK rows.")

    dataset = dataset[
        (dataset["label_name"] == str(label_name))
        & (dataset["target_type"] == str(target_type))
        & (dataset["horizon_days"] == int(horizon_days))
    ].copy()
    if dataset.empty:
        raise ValueError(
            "No model_dataset rows after filters. "
            f"label_name={label_name}, target_type={target_type}, horizon_days={horizon_days}"
        )

    folds["fold_id"] = pd.to_numeric(folds["fold_id"], errors="coerce").astype("Int64")
    folds["date"] = _normalize_date(folds["date"], column="date")
    folds["instrument_id"] = folds["instrument_id"].astype(str)
    folds["horizon_days"] = pd.to_numeric(folds["horizon_days"], errors="coerce").astype("Int64")
    folds["label_name"] = folds["label_name"].astype(str)
    folds["split_role"] = folds["split_role"].astype(str)
    if folds["fold_id"].isna().any() or folds["horizon_days"].isna().any():
        raise ValueError("purged_cv_folds has invalid fold_id/horizon_days.")
    folds["fold_id"] = folds["fold_id"].astype("int64")
    folds["horizon_days"] = folds["horizon_days"].astype("int64")
    if folds.duplicated(["fold_id", "date", "instrument_id", "horizon_days", "label_name"]).any():
        raise ValueError("purged_cv_folds has duplicate fold PK rows.")
    invalid_roles = sorted(set(folds["split_role"].tolist()) - set(VALID_ROLES))
    if invalid_roles:
        raise ValueError(f"purged_cv_folds has invalid roles: {invalid_roles}")

    folds = folds[
        (folds["label_name"] == str(label_name)) & (folds["horizon_days"] == int(horizon_days))
    ].copy()
    if folds.empty:
        raise ValueError(
            "No purged_cv_folds rows after filters. "
            f"label_name={label_name}, horizon_days={horizon_days}"
        )

    merged = folds.merge(
        dataset,
        on=["date", "instrument_id", "horizon_days", "label_name"],
        how="inner",
        suffixes=("_fold", ""),
    )
    if merged.empty:
        raise ValueError("No rows after joining model_dataset and purged_cv_folds on PK.")
    if "split_role_fold" in merged.columns:
        merged["split_role"] = merged["split_role_fold"].astype(str)
        merged = merged.drop(columns=["split_role_fold"])

    feature_cols = tuple(
        sorted(
            [
                col
                for col in merged.columns
                if col not in FEATURE_EXCLUDED_COLUMNS and is_numeric_dtype(merged[col])
            ]
        )
    )
    if not feature_cols:
        raise ValueError("No numeric features available for refinement.")

    fold_ids = tuple(sorted({int(v) for v in merged["fold_id"].tolist()}))
    return merged, feature_cols, fold_ids


def _prepare_fold_matrix(
    *,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame, pd.DataFrame]:
    x_train_df = train_df[list(feature_cols)].copy()
    x_valid_df = valid_df[list(feature_cols)].copy()
    y_train_raw = pd.to_numeric(train_df["target_value"], errors="coerce").to_numpy(dtype=float)
    y_valid_raw = pd.to_numeric(valid_df["target_value"], errors="coerce").to_numpy(dtype=float)

    med = x_train_df.median(axis=0, skipna=True)
    all_nan_cols = sorted([col for col in feature_cols if pd.isna(med[col])])
    kept = [col for col in feature_cols if col not in set(all_nan_cols)]
    if not kept:
        raise ValueError("All feature columns are all-NaN in train.")

    x_train_df = x_train_df[kept].fillna(med[kept])
    x_valid_df = x_valid_df[kept].fillna(med[kept])
    mean = x_train_df.mean(axis=0)
    std = x_train_df.std(axis=0, ddof=0).replace(0.0, 1.0)
    x_train_df = (x_train_df - mean) / std
    x_valid_df = (x_valid_df - mean) / std

    if not np.isfinite(x_train_df.to_numpy(dtype=float)).all() or not np.isfinite(
        x_valid_df.to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite values after train-only preprocessing.")

    return (
        x_train_df.to_numpy(dtype=float),
        x_valid_df.to_numpy(dtype=float),
        y_train_raw,
        y_valid_raw,
        kept,
        train_df.copy(),
        valid_df.copy(),
    )


def _resolve_numeric_series(
    frame: pd.DataFrame,
    *,
    column: str,
    train_fill_value: float,
) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    values = np.where(np.isfinite(values), values, float(train_fill_value))
    return values.astype(float)


def _fit_ridge_variant_fold(
    *,
    variant_name: str,
    x_train: np.ndarray,
    x_valid: np.ndarray,
    y_train_raw: np.ndarray,
    y_valid_raw: np.ndarray,
    train_raw_features: pd.DataFrame,
    valid_raw_features: pd.DataFrame,
    alpha_grid: tuple[float, ...],
    target_clip_abs: float,
    vol_scale_column: str | None,
    vol_scale_min: float,
    market_reference_column: str | None,
) -> tuple[float, float, float, dict[str, Any]]:
    notes: dict[str, Any] = {"variant": variant_name}

    if variant_name == VARIANT_RAW:
        y_train_model = y_train_raw.astype(float)

        def invert_valid(pred_model: np.ndarray) -> np.ndarray:
            return pred_model.astype(float)

        dummy_valid_pred_raw = np.full(
            shape=y_valid_raw.shape,
            fill_value=float(np.mean(y_train_model)),
            dtype=float,
        )
        notes["construction"] = "identity target."

    elif variant_name == VARIANT_CLIPPED:
        y_train_model = np.clip(y_train_raw.astype(float), -float(target_clip_abs), float(target_clip_abs))

        def invert_valid(pred_model: np.ndarray) -> np.ndarray:
            return np.clip(pred_model.astype(float), -float(target_clip_abs), float(target_clip_abs))

        dummy_center = float(np.mean(y_train_model))
        dummy_valid_pred_raw = np.full(
            shape=y_valid_raw.shape,
            fill_value=float(np.clip(dummy_center, -float(target_clip_abs), float(target_clip_abs))),
            dtype=float,
        )
        notes["construction"] = f"target clipped to [-{target_clip_abs}, {target_clip_abs}] for fit."

    elif variant_name == VARIANT_VOL_SCALED:
        if vol_scale_column is None:
            raise ValueError("vol_scaled_target requires an available volatility feature column.")

        train_vol_raw = pd.to_numeric(train_raw_features[vol_scale_column], errors="coerce")
        train_fill = float(train_vol_raw.median(skipna=True))
        if not np.isfinite(train_fill) or train_fill == 0.0:
            train_fill = 1.0
        train_vol = np.abs(
            _resolve_numeric_series(
                train_raw_features,
                column=vol_scale_column,
                train_fill_value=train_fill,
            )
        )
        valid_vol = np.abs(
            _resolve_numeric_series(
                valid_raw_features,
                column=vol_scale_column,
                train_fill_value=train_fill,
            )
        )
        train_scale = np.clip(train_vol, float(vol_scale_min), None)
        valid_scale = np.clip(valid_vol, float(vol_scale_min), None)

        y_train_model = y_train_raw.astype(float) / train_scale

        def invert_valid(pred_model: np.ndarray) -> np.ndarray:
            return pred_model.astype(float) * valid_scale

        dummy_center = float(np.mean(y_train_model))
        dummy_valid_pred_raw = (
            np.full(shape=y_valid_raw.shape, fill_value=dummy_center, dtype=float) * valid_scale
        )
        notes["construction"] = f"target divided by {vol_scale_column} in train, restored in valid."
        notes["vol_scale_column"] = vol_scale_column

    elif variant_name == VARIANT_MARKET_RELATIVE:
        if market_reference_column is None:
            raise ValueError("market_relative_target requires an available market reference feature column.")

        train_mkt_raw = pd.to_numeric(train_raw_features[market_reference_column], errors="coerce")
        train_fill = float(train_mkt_raw.median(skipna=True))
        if not np.isfinite(train_fill):
            train_fill = 0.0
        train_mkt = _resolve_numeric_series(
            train_raw_features,
            column=market_reference_column,
            train_fill_value=train_fill,
        )
        valid_mkt = _resolve_numeric_series(
            valid_raw_features,
            column=market_reference_column,
            train_fill_value=train_fill,
        )

        y_train_model = y_train_raw.astype(float) - train_mkt

        def invert_valid(pred_model: np.ndarray) -> np.ndarray:
            return pred_model.astype(float) + valid_mkt

        dummy_center = float(np.mean(y_train_model))
        dummy_valid_pred_raw = np.full(shape=y_valid_raw.shape, fill_value=dummy_center, dtype=float) + valid_mkt
        notes["construction"] = (
            f"target adjusted by subtracting {market_reference_column} in train, restored in valid."
        )
        notes["market_reference_column"] = market_reference_column

    else:
        raise ValueError(f"Unsupported variant '{variant_name}'.")

    best_alpha = float(alpha_grid[0])
    best_mse = float("inf")
    for alpha in alpha_grid:
        model = Ridge(alpha=float(alpha), fit_intercept=True)
        model.fit(x_train, y_train_model)
        pred_model = model.predict(x_valid)
        pred_raw = invert_valid(pred_model)
        mse = float(np.mean(np.square(y_valid_raw - pred_raw)))
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(alpha)

    dummy_mse = float(np.mean(np.square(y_valid_raw - dummy_valid_pred_raw)))
    return best_alpha, best_mse, dummy_mse, notes


def run_refine_h20_target(
    *,
    model_dataset_path: str | Path | None = None,
    purged_cv_folds_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    label_name: str = "fwd_ret_20d",
    target_type: str = "continuous_forward_return",
    horizon_days: int = 20,
    variants: Iterable[str] = DEFAULT_VARIANTS,
    alpha_grid: Iterable[float] = (0.1, 1.0, 10.0),
    target_clip_abs: float = 0.20,
    vol_scale_columns: Iterable[str] = DEFAULT_VOL_SCALE_COLUMNS,
    vol_scale_min: float = 1e-6,
    market_reference_columns: Iterable[str] = DEFAULT_MARKET_REFERENCE_COLUMNS,
    tie_tolerance: float = 1e-12,
    fail_on_invalid_fold: bool = False,
    run_id: str = MODULE_VERSION,
) -> RefineH20TargetResult:
    logger = get_logger("research.refine_h20_target")
    if int(horizon_days) <= 0:
        raise ValueError("horizon_days must be positive.")
    if float(target_clip_abs) <= 0:
        raise ValueError("target_clip_abs must be > 0.")
    if float(vol_scale_min) <= 0:
        raise ValueError("vol_scale_min must be > 0.")
    if float(tie_tolerance) < 0:
        raise ValueError("tie_tolerance must be >= 0.")

    selected_variants = tuple(dict.fromkeys(str(v).strip() for v in variants if str(v).strip()))
    if not selected_variants:
        raise ValueError("variants cannot be empty.")
    invalid = sorted(set(selected_variants) - set(DEFAULT_VARIANTS))
    if invalid:
        raise ValueError(f"Unsupported variants: {invalid}. Allowed: {DEFAULT_VARIANTS}")
    if VARIANT_RAW not in selected_variants:
        raise ValueError(f"{VARIANT_RAW} must be included in variants.")

    base = data_dir()
    dataset_source = (
        Path(model_dataset_path).expanduser().resolve()
        if model_dataset_path
        else (base / DEFAULT_MODEL_DATASET_PATH)
    )
    folds_source = (
        Path(purged_cv_folds_path).expanduser().resolve()
        if purged_cv_folds_path
        else (base / DEFAULT_FOLDS_PATH)
    )
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (base / "research")
    target_dir.mkdir(parents=True, exist_ok=True)
    alpha_values = _normalize_float_grid(alpha_grid, name="alpha_grid")
    vol_candidates = tuple(dict.fromkeys(str(item).strip() for item in vol_scale_columns if str(item).strip()))
    market_candidates = tuple(
        str(item).strip() for item in market_reference_columns if str(item).strip()
    )

    merged, feature_cols, fold_ids = _load_joined_inputs(
        model_dataset_path=dataset_source,
        purged_cv_folds_path=folds_source,
        label_name=label_name,
        target_type=target_type,
        horizon_days=int(horizon_days),
    )
    vol_candidates_resolved = tuple(vol_candidates)
    market_reference_column = _first_available_column(
        market_candidates,
        available_columns=feature_cols,
    )

    built_ts_utc = datetime.now(UTC).isoformat()
    config_hash = _cfg_hash(
        {
            "module_version": MODULE_VERSION,
            "model_dataset_path": str(dataset_source),
            "purged_cv_folds_path": str(folds_source),
            "label_name": label_name,
            "target_type": target_type,
            "horizon_days": int(horizon_days),
            "variants": list(selected_variants),
            "alpha_grid": list(alpha_values),
            "target_clip_abs": float(target_clip_abs),
            "vol_scale_columns": list(vol_candidates),
            "vol_scale_columns_resolved_order": list(vol_candidates_resolved),
            "vol_scale_min": float(vol_scale_min),
            "market_reference_columns": list(market_candidates),
            "market_reference_column_resolved": market_reference_column,
            "tie_tolerance": float(tie_tolerance),
            "fail_on_invalid_fold": bool(fail_on_invalid_fold),
            "run_id": run_id,
        }
    )

    by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in selected_variants}
    notes: list[str] = [
        "Target variants are fitted using train rows only per fold.",
        "Validation metric is always computed on raw target_value.",
        "split_role dropped_by_purge/dropped_by_embargo are excluded from fit/eval.",
    ]

    for variant in selected_variants:
        if variant == VARIANT_VOL_SCALED and not vol_candidates_resolved:
            notes.append(f"Skipped variant '{variant}': no volatility feature column found in dataset.")
            continue
        if variant == VARIANT_MARKET_RELATIVE and market_reference_column is None:
            notes.append(
                f"Skipped variant '{variant}': no market reference feature column found in dataset."
            )
            continue

        for fold_id in fold_ids:
            fold = merged[merged["fold_id"] == int(fold_id)].copy()
            train_df = fold[fold["split_role"] == TRAIN_ROLE].copy()
            valid_df = fold[fold["split_role"] == VALID_ROLE].copy()
            if train_df.empty or valid_df.empty:
                reason = (
                    f"fold={fold_id}: empty train/valid "
                    f"(n_train={len(train_df)}, n_valid={len(valid_df)})"
                )
                if fail_on_invalid_fold:
                    raise ValueError(reason)
                notes.append(f"Skipped {variant} {reason}")
                continue

            try:
                (
                    x_train,
                    x_valid,
                    y_train_raw,
                    y_valid_raw,
                    kept_features,
                    train_raw_features,
                    valid_raw_features,
                ) = _prepare_fold_matrix(
                    train_df=train_df,
                    valid_df=valid_df,
                    feature_cols=feature_cols,
                )

                fold_vol_scale_column = _resolve_fold_column_with_train_coverage(
                    frame=train_raw_features,
                    candidates=vol_candidates_resolved,
                )
                fold_market_reference_column = market_reference_column
                if (
                    variant == VARIANT_MARKET_RELATIVE
                    and fold_market_reference_column is not None
                    and fold_market_reference_column in train_raw_features.columns
                ):
                    mkt_vals = pd.to_numeric(
                        train_raw_features[fold_market_reference_column], errors="coerce"
                    )
                    if not mkt_vals.notna().any():
                        fold_market_reference_column = None

                alpha_selected, model_valid_mse, dummy_valid_mse, variant_notes = _fit_ridge_variant_fold(
                    variant_name=variant,
                    x_train=x_train,
                    x_valid=x_valid,
                    y_train_raw=y_train_raw,
                    y_valid_raw=y_valid_raw,
                    train_raw_features=train_raw_features,
                    valid_raw_features=valid_raw_features,
                    alpha_grid=alpha_values,
                    target_clip_abs=float(target_clip_abs),
                    vol_scale_column=fold_vol_scale_column,
                    vol_scale_min=float(vol_scale_min),
                    market_reference_column=fold_market_reference_column,
                )
            except Exception as exc:
                reason = f"fold={fold_id}: variant={variant} failed: {exc}"
                if fail_on_invalid_fold:
                    raise ValueError(reason) from exc
                notes.append(f"Skipped {reason}")
                continue

            by_variant[variant].append(
                {
                    "target_variant": variant,
                    "label_name": label_name,
                    "target_type": target_type,
                    "horizon_days": int(horizon_days),
                    "fold_id": int(fold_id),
                    "primary_metric": PRIMARY_METRIC,
                    "model_valid_primary_metric": float(model_valid_mse),
                    "dummy_valid_primary_metric": float(dummy_valid_mse),
                    "improvement_vs_dummy": float(dummy_valid_mse - model_valid_mse),
                    "n_train": int(len(train_df)),
                    "n_valid": int(len(valid_df)),
                    "n_features_used": int(len(kept_features)),
                    "alpha_selected": float(alpha_selected),
                    "variant_notes_json": json.dumps(variant_notes, sort_keys=True),
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "built_ts_utc": built_ts_utc,
                }
            )

    variant_frames: dict[str, pd.DataFrame] = {}
    for variant in selected_variants:
        rows = by_variant.get(variant, [])
        if rows:
            variant_frames[variant] = pd.DataFrame(rows).copy()

    if VARIANT_RAW not in variant_frames:
        raise ValueError("No completed folds for raw_target_baseline.")

    variant_fold_counts_before_common: dict[str, int] = {
        variant: int(frame["fold_id"].astype(int).nunique()) for variant, frame in variant_frames.items()
    }
    common_fold_ids: set[int] | None = None
    for frame in variant_frames.values():
        fold_set = {int(v) for v in frame["fold_id"].astype(int).tolist()}
        common_fold_ids = fold_set if common_fold_ids is None else (common_fold_ids & fold_set)
    common_fold_ids_sorted = sorted(common_fold_ids or [])
    if not common_fold_ids_sorted:
        raise ValueError(
            "No common completed folds across target variants; cannot compare variants cleanly."
        )
    notes.append(
        "Comparison policy: same_common_folds_only across evaluated target variants."
    )
    notes.append(f"Common fold ids used for all variants: {common_fold_ids_sorted}")
    notes.append(
        f"Variant fold counts before common-fold restriction: {variant_fold_counts_before_common}"
    )
    if len(common_fold_ids_sorted) < 2:
        notes.append(
            "Warning: common fold count < 2; target-variant comparison is fragile."
        )

    baseline_df = variant_frames[VARIANT_RAW].copy()
    baseline_df = baseline_df[
        baseline_df["fold_id"].astype(int).isin(set(common_fold_ids_sorted))
    ][["fold_id", "model_valid_primary_metric"]].rename(
        columns={"model_valid_primary_metric": "baseline_valid_primary_metric"}
    )

    fold_records: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    for variant in selected_variants:
        if variant not in variant_frames:
            continue
        variant_df = variant_frames[variant].copy()
        variant_df = variant_df[
            variant_df["fold_id"].astype(int).isin(set(common_fold_ids_sorted))
        ].copy()
        merged_baseline = variant_df.merge(baseline_df, on="fold_id", how="inner")
        if merged_baseline.empty:
            continue
        merged_baseline["improvement_vs_baseline"] = (
            merged_baseline["baseline_valid_primary_metric"] - merged_baseline["model_valid_primary_metric"]
        )
        mean_metric = float(merged_baseline["model_valid_primary_metric"].mean())
        mean_dummy = float(merged_baseline["dummy_valid_primary_metric"].mean())
        improvement_vs_baseline = float(merged_baseline["improvement_vs_baseline"].mean())
        improvement_vs_dummy = float(mean_dummy - mean_metric)
        result_records.append(
            {
                "target_variant": variant,
                "label_name": label_name,
                "target_type": target_type,
                "horizon_days": int(horizon_days),
                "primary_metric": PRIMARY_METRIC,
                "mean_valid_primary_metric": mean_metric,
                "median_valid_primary_metric": float(
                    merged_baseline["model_valid_primary_metric"].median()
                ),
                "std_valid_primary_metric": float(
                    merged_baseline["model_valid_primary_metric"].std(ddof=0)
                ),
                "improvement_vs_baseline": improvement_vs_baseline,
                "improvement_vs_dummy": improvement_vs_dummy,
                "winner_vs_baseline": _winner(
                    improvement_vs_baseline,
                    tie_tolerance=float(tie_tolerance),
                    positive_label="variant",
                    negative_label="baseline",
                ),
                "winner_vs_dummy": _winner(
                    improvement_vs_dummy,
                    tie_tolerance=float(tie_tolerance),
                    positive_label="variant",
                    negative_label="dummy",
                ),
                "n_folds_used_common": int(len(common_fold_ids_sorted)),
                "n_folds_completed_before_common": int(
                    variant_fold_counts_before_common.get(variant, 0)
                ),
                "run_id": run_id,
                "config_hash": config_hash,
                "built_ts_utc": built_ts_utc,
            }
        )
        for row in merged_baseline.to_dict(orient="records"):
            fold_records.append(
                {
                    **row,
                    "comparison_policy": "same_common_folds_only",
                    "n_common_folds_used": int(len(common_fold_ids_sorted)),
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "built_ts_utc": built_ts_utc,
                }
            )

    if not result_records:
        raise ValueError("No target variant produced comparable fold metrics against baseline.")

    results_df = pd.DataFrame(result_records).sort_values(
        ["mean_valid_primary_metric", "target_variant"]
    ).reset_index(drop=True)
    fold_df = pd.DataFrame(fold_records).sort_values(
        ["target_variant", "fold_id"]
    ).reset_index(drop=True)

    for col in (
        "target_variant",
        "label_name",
        "target_type",
        "primary_metric",
        "winner_vs_baseline",
        "winner_vs_dummy",
    ):
        results_df[col] = results_df[col].astype("string")
    for col in (
        "mean_valid_primary_metric",
        "median_valid_primary_metric",
        "std_valid_primary_metric",
        "improvement_vs_baseline",
        "improvement_vs_dummy",
    ):
        results_df[col] = pd.to_numeric(results_df[col], errors="coerce")
    results_df["horizon_days"] = pd.to_numeric(
        results_df["horizon_days"], errors="coerce"
    ).astype("int64")
    assert_schema(results_df, RESULTS_SCHEMA)

    for col in ("target_variant", "label_name", "target_type", "primary_metric"):
        fold_df[col] = fold_df[col].astype("string")
    for col in (
        "model_valid_primary_metric",
        "dummy_valid_primary_metric",
        "baseline_valid_primary_metric",
        "improvement_vs_dummy",
        "improvement_vs_baseline",
    ):
        fold_df[col] = pd.to_numeric(fold_df[col], errors="coerce")
    fold_df["horizon_days"] = pd.to_numeric(fold_df["horizon_days"], errors="coerce").astype("int64")
    fold_df["fold_id"] = pd.to_numeric(fold_df["fold_id"], errors="coerce").astype("int64")
    assert_schema(fold_df, FOLD_SCHEMA)

    results_path = write_parquet(
        results_df,
        target_dir / "refine_h20_target_results.parquet",
        schema_name=RESULTS_SCHEMA.name,
        run_id=run_id,
    )
    fold_metrics_path = write_parquet(
        fold_df,
        target_dir / "refine_h20_target_fold_metrics.parquet",
        schema_name=FOLD_SCHEMA.name,
        run_id=run_id,
    )

    best_row = results_df.sort_values(["mean_valid_primary_metric", "target_variant"]).iloc[0]
    variants_beating_baseline = sorted(
        results_df.loc[
            results_df["winner_vs_baseline"].astype(str) == "variant", "target_variant"
        ]
        .astype(str)
        .unique()
        .tolist()
    )
    variants_beating_dummy = sorted(
        results_df.loc[
            results_df["winner_vs_dummy"].astype(str) == "variant", "target_variant"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    if variants_beating_baseline and str(best_row["winner_vs_dummy"]) == "variant":
        recommendation = "adopt_best_variant_and_revalidate"
    elif variants_beating_baseline:
        recommendation = "recheck_variant_stability_before_adoption"
    elif variants_beating_dummy:
        recommendation = "keep_baseline_and_continue_target_refinement"
    else:
        recommendation = "no_target_variant_improvement_keep_baseline"

    summary_payload = {
        "module_version": MODULE_VERSION,
        "baseline_variant": VARIANT_RAW,
        "best_variant": str(best_row["target_variant"]),
        "variants_evaluated": results_df["target_variant"].astype(str).tolist(),
        "variants_beating_baseline": variants_beating_baseline,
        "variants_beating_dummy": variants_beating_dummy,
        "recommendation": recommendation,
        "label_name": label_name,
        "target_type": target_type,
        "horizon_days": int(horizon_days),
        "input_paths": {
            "model_dataset": str(dataset_source),
            "purged_cv_folds": str(folds_source),
        },
        "output_paths": {
            "results": str(results_path),
            "fold_metrics": str(fold_metrics_path),
        },
        "target_construction_policy": {
            "train_only_target_transform_fit": True,
            "validation_scored_on_raw_target": True,
            "vol_scale_columns_resolved_order": list(vol_candidates_resolved),
            "vol_scale_min": float(vol_scale_min),
            "market_reference_column_resolved": market_reference_column,
            "uses_only_decision_date_features_for_adjustments": True,
        },
        "comparison_policy": {
            "same_common_folds_only": True,
            "common_fold_ids": common_fold_ids_sorted,
            "n_common_folds": int(len(common_fold_ids_sorted)),
            "variant_fold_counts_before_common": variant_fold_counts_before_common,
        },
        "run_id": run_id,
        "config_hash": config_hash,
        "built_ts_utc": built_ts_utc,
        "notes": notes,
    }
    summary_path = target_dir / "refine_h20_target_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "refine_h20_target_completed",
        run_id=run_id,
        n_results_rows=int(len(results_df)),
        best_variant=str(best_row["target_variant"]),
        results_path=str(results_path),
    )
    return RefineH20TargetResult(
        results_path=results_path,
        summary_path=summary_path,
        fold_metrics_path=fold_metrics_path,
        n_rows=int(len(results_df)),
        config_hash=config_hash,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refine H20 regression target variants (ridge vs dummy, purged CV)."
    )
    parser.add_argument("--model-dataset-path", type=str, default=None)
    parser.add_argument("--purged-cv-folds-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--label-name", type=str, default="fwd_ret_20d")
    parser.add_argument("--target-type", type=str, default="continuous_forward_return")
    parser.add_argument("--horizon-days", type=int, default=20)
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated target variants to evaluate.",
    )
    parser.add_argument("--ridge-alphas", type=str, default="0.1,1.0,10.0")
    parser.add_argument("--target-clip-abs", type=float, default=0.20)
    parser.add_argument(
        "--vol-scale-columns",
        type=str,
        default=",".join(DEFAULT_VOL_SCALE_COLUMNS),
    )
    parser.add_argument("--vol-scale-min", type=float, default=1e-6)
    parser.add_argument(
        "--market-reference-columns",
        type=str,
        default=",".join(DEFAULT_MARKET_REFERENCE_COLUMNS),
    )
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    parser.add_argument("--fail-on-invalid-fold", action="store_true")
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    variant_list = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    result = run_refine_h20_target(
        model_dataset_path=args.model_dataset_path,
        purged_cv_folds_path=args.purged_cv_folds_path,
        output_dir=args.output_dir,
        label_name=args.label_name,
        target_type=args.target_type,
        horizon_days=args.horizon_days,
        variants=variant_list,
        alpha_grid=_parse_csv_floats(args.ridge_alphas) or (0.1, 1.0, 10.0),
        target_clip_abs=float(args.target_clip_abs),
        vol_scale_columns=_parse_csv_texts(args.vol_scale_columns) or DEFAULT_VOL_SCALE_COLUMNS,
        vol_scale_min=float(args.vol_scale_min),
        market_reference_columns=_parse_csv_texts(args.market_reference_columns)
        or DEFAULT_MARKET_REFERENCE_COLUMNS,
        tie_tolerance=float(args.tie_tolerance),
        fail_on_invalid_fold=bool(args.fail_on_invalid_fold),
        run_id=args.run_id,
    )
    print(
        json.dumps(
            {
                "results_path": str(result.results_path),
                "fold_metrics_path": str(result.fold_metrics_path),
                "summary_path": str(result.summary_path),
                "n_rows": result.n_rows,
                "config_hash": result.config_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
