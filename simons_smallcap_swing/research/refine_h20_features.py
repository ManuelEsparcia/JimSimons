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
# `python simons_smallcap_swing/research/refine_h20_features.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "refine_h20_features_mvp_v1"
PRIMARY_METRIC = "mse"
DEFAULT_MODEL_DATASET_PATH = "datasets/regression_h20/model_dataset.parquet"
DEFAULT_FOLDS_PATH = "labels/purged_cv_folds.parquet"

BASELINE_VARIANT = "baseline_all_features"
VARIANT_LOW_MISSINGNESS = "low_missingness_only"
VARIANT_LOW_COLLINEARITY = "low_collinearity_only"
VARIANT_STABLE_SIGN = "stable_sign_features_only"
VARIANT_LOW_MISS_PLUS_LOW_COL = "low_missingness_plus_low_collinearity"
VARIANT_STABLE_PLUS_LOW_COL = "stable_sign_plus_low_collinearity"
DEFAULT_VARIANTS: tuple[str, ...] = (
    BASELINE_VARIANT,
    VARIANT_LOW_MISSINGNESS,
    VARIANT_LOW_COLLINEARITY,
    VARIANT_STABLE_SIGN,
    VARIANT_LOW_MISS_PLUS_LOW_COL,
    VARIANT_STABLE_PLUS_LOW_COL,
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

RESULTS_SCHEMA = DataSchema(
    name="refine_h20_features_results_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("candidate_variant", "string", nullable=False),
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
        ColumnSpec("n_features_used", "int64", nullable=False),
    ),
    primary_key=(),
    allow_extra_columns=True,
)

FOLD_SCHEMA = DataSchema(
    name="refine_h20_features_fold_metrics_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("candidate_variant", "string", nullable=False),
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
        ColumnSpec("n_features_used", "int64", nullable=False),
    ),
    primary_key=(),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class RefineH20FeaturesResult:
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


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < 3:
        return 0.0
    x_vals = aligned.iloc[:, 0].astype(float)
    y_vals = aligned.iloc[:, 1].astype(float)
    if float(x_vals.std(ddof=0)) == 0.0 or float(y_vals.std(ddof=0)) == 0.0:
        return 0.0
    corr = float(x_vals.corr(y_vals))
    if pd.isna(corr):
        return 0.0
    return corr


def _compute_train_sign_maps(
    *,
    merged: pd.DataFrame,
    feature_cols: tuple[str, ...],
    fold_ids: tuple[int, ...],
    min_abs_train_corr: float,
) -> dict[int, dict[str, int]]:
    sign_map: dict[int, dict[str, int]] = {}
    for fold_id in fold_ids:
        fold = merged[(merged["fold_id"] == int(fold_id)) & (merged["split_role"] == TRAIN_ROLE)].copy()
        if fold.empty:
            sign_map[int(fold_id)] = {col: 0 for col in feature_cols}
            continue
        y = pd.to_numeric(fold["target_value"], errors="coerce")
        signs: dict[str, int] = {}
        for col in feature_cols:
            corr = _safe_corr(pd.to_numeric(fold[col], errors="coerce"), y)
            if abs(float(corr)) < float(min_abs_train_corr):
                signs[col] = 0
            else:
                signs[col] = 1 if corr > 0 else -1
        sign_map[int(fold_id)] = signs
    return sign_map


def _train_abs_corr_map(train_df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    y = pd.to_numeric(train_df["target_value"], errors="coerce")
    values: dict[str, float] = {}
    for col in cols:
        corr = _safe_corr(pd.to_numeric(train_df[col], errors="coerce"), y)
        values[col] = abs(float(corr))
    return values


def _prune_by_collinearity(
    *,
    train_df: pd.DataFrame,
    cols: list[str],
    threshold: float,
) -> tuple[list[str], list[str]]:
    if len(cols) <= 1:
        return cols.copy(), []

    x = train_df[cols].copy()
    med = x.median(axis=0, skipna=True)
    x = x.fillna(med)
    corr = x.corr().abs()

    corr_order = _train_abs_corr_map(train_df, cols)
    ordered = sorted(cols, key=lambda c: (-float(corr_order.get(c, 0.0)), c))
    kept: list[str] = []
    dropped: list[str] = []
    for col in ordered:
        keep_col = True
        for prev in kept:
            corr_val = corr.loc[col, prev]
            if pd.notna(corr_val) and float(corr_val) > float(threshold):
                keep_col = False
                break
        if keep_col:
            kept.append(col)
        else:
            dropped.append(col)
    return sorted(kept), sorted(dropped)


def _select_features_for_fold(
    *,
    variant_name: str,
    fold_id: int,
    fold_ids: tuple[int, ...],
    train_df: pd.DataFrame,
    all_feature_cols: tuple[str, ...],
    sign_map: dict[int, dict[str, int]],
    missingness_threshold: float,
    collinearity_threshold: float,
    stability_min_history_folds: int,
) -> tuple[list[str], dict[str, Any]]:
    selected = list(all_feature_cols)
    dropped_missing: list[str] = []
    dropped_collinearity: list[str] = []
    dropped_stability: list[str] = []

    apply_missing = variant_name in {VARIANT_LOW_MISSINGNESS, VARIANT_LOW_MISS_PLUS_LOW_COL}
    apply_col = variant_name in {
        VARIANT_LOW_COLLINEARITY,
        VARIANT_LOW_MISS_PLUS_LOW_COL,
        VARIANT_STABLE_PLUS_LOW_COL,
    }
    apply_stable = variant_name in {VARIANT_STABLE_SIGN, VARIANT_STABLE_PLUS_LOW_COL}

    if apply_missing:
        miss = train_df[selected].isna().mean()
        kept = [col for col in selected if float(miss.get(col, 1.0)) <= float(missingness_threshold)]
        dropped_missing = sorted(set(selected) - set(kept))
        selected = sorted(kept)

    if apply_stable:
        history = [fid for fid in fold_ids if int(fid) <= int(fold_id)]
        stable: list[str] = []
        for col in selected:
            signs = [
                int(sign_map.get(int(fid), {}).get(col, 0))
                for fid in history
                if int(sign_map.get(int(fid), {}).get(col, 0)) != 0
            ]
            if len(signs) < int(stability_min_history_folds):
                continue
            if all(sign == signs[0] for sign in signs):
                stable.append(col)
        dropped_stability = sorted(set(selected) - set(stable))
        selected = sorted(stable)

    if apply_col:
        selected, dropped_collinearity = _prune_by_collinearity(
            train_df=train_df,
            cols=selected,
            threshold=float(collinearity_threshold),
        )

    if not selected:
        fallback = list(all_feature_cols)
        corr_map = _train_abs_corr_map(train_df, fallback)
        selected = [max(fallback, key=lambda c: (float(corr_map.get(c, 0.0)), c))]

    details = {
        "dropped_due_missingness": dropped_missing,
        "dropped_due_collinearity": dropped_collinearity,
        "dropped_due_stability": dropped_stability,
        "selected_features": sorted(selected),
        "selection_policy_train_only": True,
    }
    return sorted(selected), details


def _prepare_fold_matrix(
    *,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    selected_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    x_train = train_df[selected_cols].copy()
    x_valid = valid_df[selected_cols].copy()
    y_train = pd.to_numeric(train_df["target_value"], errors="coerce").to_numpy(dtype=float)
    y_valid = pd.to_numeric(valid_df["target_value"], errors="coerce").to_numpy(dtype=float)

    med = x_train.median(axis=0, skipna=True)
    all_nan_cols = sorted([col for col in selected_cols if pd.isna(med[col])])
    kept = [col for col in selected_cols if col not in set(all_nan_cols)]
    if not kept:
        raise ValueError("All selected features are all-NaN in train.")

    x_train = x_train[kept].fillna(med[kept])
    x_valid = x_valid[kept].fillna(med[kept])
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=0).replace(0.0, 1.0)
    x_train = (x_train - mean) / std
    x_valid = (x_valid - mean) / std

    if not np.isfinite(x_train.to_numpy(dtype=float)).all() or not np.isfinite(
        x_valid.to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite values after train-only preprocessing.")

    return (
        x_train.to_numpy(dtype=float),
        x_valid.to_numpy(dtype=float),
        y_train,
        y_valid,
        kept,
    )


def _fit_ridge_valid_mse(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    alpha_grid: tuple[float, ...],
) -> tuple[float, float]:
    best_alpha = float(alpha_grid[0])
    best_mse = float("inf")
    for alpha in alpha_grid:
        model = Ridge(alpha=float(alpha), fit_intercept=True)
        model.fit(x_train, y_train)
        pred = model.predict(x_valid)
        mse = float(np.mean(np.square(y_valid - pred)))
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(alpha)
    return best_alpha, best_mse


def run_refine_h20_features(
    *,
    model_dataset_path: str | Path | None = None,
    purged_cv_folds_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    label_name: str = "fwd_ret_20d",
    target_type: str = "continuous_forward_return",
    horizon_days: int = 20,
    variants: Iterable[str] = DEFAULT_VARIANTS,
    alpha_grid: Iterable[float] = (0.1, 1.0, 10.0),
    missingness_threshold: float = 0.35,
    collinearity_threshold: float = 0.95,
    min_abs_train_corr: float = 0.02,
    stability_min_history_folds: int = 2,
    tie_tolerance: float = 1e-12,
    fail_on_invalid_fold: bool = False,
    run_id: str = MODULE_VERSION,
) -> RefineH20FeaturesResult:
    logger = get_logger("research.refine_h20_features")
    if int(horizon_days) <= 0:
        raise ValueError("horizon_days must be positive.")
    if not (0.0 <= float(missingness_threshold) <= 1.0):
        raise ValueError("missingness_threshold must be in [0,1].")
    if not (0.0 < float(collinearity_threshold) <= 1.0):
        raise ValueError("collinearity_threshold must be in (0,1].")
    if float(min_abs_train_corr) < 0.0:
        raise ValueError("min_abs_train_corr must be >= 0.")
    if int(stability_min_history_folds) <= 0:
        raise ValueError("stability_min_history_folds must be >= 1.")
    if float(tie_tolerance) < 0:
        raise ValueError("tie_tolerance must be >= 0.")

    selected_variants = tuple(dict.fromkeys(str(v).strip() for v in variants if str(v).strip()))
    if not selected_variants:
        raise ValueError("variants cannot be empty.")
    invalid = sorted(set(selected_variants) - set(DEFAULT_VARIANTS))
    if invalid:
        raise ValueError(f"Unsupported variants: {invalid}. Allowed: {DEFAULT_VARIANTS}")
    if BASELINE_VARIANT not in selected_variants:
        raise ValueError(f"{BASELINE_VARIANT} must be included in variants.")

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

    merged, feature_cols, fold_ids = _load_joined_inputs(
        model_dataset_path=dataset_source,
        purged_cv_folds_path=folds_source,
        label_name=label_name,
        target_type=target_type,
        horizon_days=int(horizon_days),
    )
    sign_map = _compute_train_sign_maps(
        merged=merged,
        feature_cols=feature_cols,
        fold_ids=fold_ids,
        min_abs_train_corr=float(min_abs_train_corr),
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
            "missingness_threshold": float(missingness_threshold),
            "collinearity_threshold": float(collinearity_threshold),
            "min_abs_train_corr": float(min_abs_train_corr),
            "stability_min_history_folds": int(stability_min_history_folds),
            "tie_tolerance": float(tie_tolerance),
            "fail_on_invalid_fold": bool(fail_on_invalid_fold),
            "run_id": run_id,
        }
    )

    by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in selected_variants}
    notes: list[str] = [
        "Feature selection is performed train-only per fold.",
        "split_role dropped_by_purge/dropped_by_embargo are excluded from fit/eval.",
    ]

    for variant in selected_variants:
        for fold_id in fold_ids:
            fold = merged[merged["fold_id"] == int(fold_id)].copy()
            train_df = fold[fold["split_role"] == TRAIN_ROLE].copy()
            valid_df = fold[fold["split_role"] == VALID_ROLE].copy()
            if train_df.empty or valid_df.empty:
                reason = f"fold={fold_id}: empty train/valid (n_train={len(train_df)}, n_valid={len(valid_df)})"
                if fail_on_invalid_fold:
                    raise ValueError(reason)
                notes.append(f"Skipped {variant} {reason}")
                continue

            try:
                selected_cols, details = _select_features_for_fold(
                    variant_name=variant,
                    fold_id=int(fold_id),
                    fold_ids=fold_ids,
                    train_df=train_df,
                    all_feature_cols=feature_cols,
                    sign_map=sign_map,
                    missingness_threshold=float(missingness_threshold),
                    collinearity_threshold=float(collinearity_threshold),
                    stability_min_history_folds=int(stability_min_history_folds),
                )
                x_train, x_valid, y_train, y_valid, kept_cols = _prepare_fold_matrix(
                    train_df=train_df,
                    valid_df=valid_df,
                    selected_cols=selected_cols,
                )
                alpha_selected, model_valid_mse = _fit_ridge_valid_mse(
                    x_train=x_train,
                    y_train=y_train,
                    x_valid=x_valid,
                    y_valid=y_valid,
                    alpha_grid=alpha_values,
                )
                dummy_pred = float(np.mean(y_train))
                dummy_valid_mse = float(np.mean(np.square(y_valid - dummy_pred)))
            except Exception as exc:
                reason = f"fold={fold_id}: variant={variant} failed: {exc}"
                if fail_on_invalid_fold:
                    raise ValueError(reason) from exc
                notes.append(f"Skipped {reason}")
                continue

            by_variant[variant].append(
                {
                    "candidate_variant": variant,
                    "label_name": label_name,
                    "target_type": target_type,
                    "horizon_days": int(horizon_days),
                    "fold_id": int(fold_id),
                    "primary_metric": PRIMARY_METRIC,
                    "model_valid_primary_metric": float(model_valid_mse),
                    "dummy_valid_primary_metric": float(dummy_valid_mse),
                    "improvement_vs_dummy": float(dummy_valid_mse - model_valid_mse),
                    "n_features_used": int(len(kept_cols)),
                    "selected_features_json": json.dumps(sorted(kept_cols), sort_keys=True),
                    "selection_details_json": json.dumps(details, sort_keys=True),
                    "alpha_selected": float(alpha_selected),
                    "n_train": int(len(train_df)),
                    "n_valid": int(len(valid_df)),
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "built_ts_utc": built_ts_utc,
                }
            )

    baseline_rows = by_variant.get(BASELINE_VARIANT, [])
    if not baseline_rows:
        raise ValueError("No completed folds for baseline_all_features.")
    baseline_df = pd.DataFrame(baseline_rows)[["fold_id", "model_valid_primary_metric"]].rename(
        columns={"model_valid_primary_metric": "baseline_valid_primary_metric"}
    )

    fold_records: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    for variant in selected_variants:
        rows = by_variant.get(variant, [])
        if not rows:
            continue
        variant_df = pd.DataFrame(rows)
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
                "candidate_variant": variant,
                "label_name": label_name,
                "target_type": target_type,
                "horizon_days": int(horizon_days),
                "primary_metric": PRIMARY_METRIC,
                "mean_valid_primary_metric": mean_metric,
                "median_valid_primary_metric": float(merged_baseline["model_valid_primary_metric"].median()),
                "std_valid_primary_metric": float(merged_baseline["model_valid_primary_metric"].std(ddof=0)),
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
                "n_features_used": int(round(float(merged_baseline["n_features_used"].median()))),
                "run_id": run_id,
                "config_hash": config_hash,
                "built_ts_utc": built_ts_utc,
            }
        )
        for row in merged_baseline.to_dict(orient="records"):
            fold_records.append(
                {
                    **row,
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "built_ts_utc": built_ts_utc,
                }
            )

    if not result_records:
        raise ValueError("No variant produced comparable fold metrics against baseline.")

    results_df = pd.DataFrame(result_records).sort_values(
        ["mean_valid_primary_metric", "candidate_variant"]
    ).reset_index(drop=True)
    fold_df = pd.DataFrame(fold_records).sort_values(
        ["candidate_variant", "fold_id"]
    ).reset_index(drop=True)

    for col in (
        "candidate_variant",
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
    results_df["n_features_used"] = pd.to_numeric(
        results_df["n_features_used"], errors="coerce"
    ).astype("int64")
    assert_schema(results_df, RESULTS_SCHEMA)

    for col in ("candidate_variant", "label_name", "target_type", "primary_metric"):
        fold_df[col] = fold_df[col].astype("string")
    for col in (
        "model_valid_primary_metric",
        "dummy_valid_primary_metric",
        "baseline_valid_primary_metric",
        "improvement_vs_dummy",
        "improvement_vs_baseline",
    ):
        fold_df[col] = pd.to_numeric(fold_df[col], errors="coerce")
    fold_df["horizon_days"] = pd.to_numeric(fold_df["horizon_days"], errors="coerce").astype(
        "int64"
    )
    fold_df["fold_id"] = pd.to_numeric(fold_df["fold_id"], errors="coerce").astype("int64")
    fold_df["n_features_used"] = pd.to_numeric(
        fold_df["n_features_used"], errors="coerce"
    ).astype("int64")
    assert_schema(fold_df, FOLD_SCHEMA)

    results_path = write_parquet(
        results_df,
        target_dir / "refine_h20_features_results.parquet",
        schema_name=RESULTS_SCHEMA.name,
        run_id=run_id,
    )
    fold_metrics_path = write_parquet(
        fold_df,
        target_dir / "refine_h20_features_fold_metrics.parquet",
        schema_name=FOLD_SCHEMA.name,
        run_id=run_id,
    )

    best_row = results_df.sort_values(["mean_valid_primary_metric", "candidate_variant"]).iloc[0]
    variants_beating_baseline = sorted(
        results_df.loc[
            results_df["winner_vs_baseline"].astype(str) == "variant", "candidate_variant"
        ]
        .astype(str)
        .unique()
        .tolist()
    )
    variants_beating_dummy = sorted(
        results_df.loc[
            results_df["winner_vs_dummy"].astype(str) == "variant", "candidate_variant"
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
        recommendation = "keep_baseline_and_continue_feature_label_improvement"
    else:
        recommendation = "no_variant_improvement_keep_baseline"

    summary_payload = {
        "module_version": MODULE_VERSION,
        "baseline_variant": BASELINE_VARIANT,
        "best_variant": str(best_row["candidate_variant"]),
        "variants_evaluated": results_df["candidate_variant"].astype(str).tolist(),
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
        "selection_policy": {
            "train_only": True,
            "missingness_threshold": float(missingness_threshold),
            "collinearity_threshold": float(collinearity_threshold),
            "min_abs_train_corr": float(min_abs_train_corr),
            "stability_min_history_folds": int(stability_min_history_folds),
        },
        "run_id": run_id,
        "config_hash": config_hash,
        "built_ts_utc": built_ts_utc,
        "notes": notes,
    }
    summary_path = target_dir / "refine_h20_features_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "refine_h20_features_completed",
        run_id=run_id,
        n_results_rows=int(len(results_df)),
        best_variant=str(best_row["candidate_variant"]),
        results_path=str(results_path),
    )
    return RefineH20FeaturesResult(
        results_path=results_path,
        summary_path=summary_path,
        fold_metrics_path=fold_metrics_path,
        n_rows=int(len(results_df)),
        config_hash=config_hash,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refine H20 feature set variants (train-only selection) for ridge vs dummy."
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
        help="Comma-separated variants to evaluate.",
    )
    parser.add_argument("--ridge-alphas", type=str, default="0.1,1.0,10.0")
    parser.add_argument("--missingness-threshold", type=float, default=0.35)
    parser.add_argument("--collinearity-threshold", type=float, default=0.95)
    parser.add_argument("--min-abs-train-corr", type=float, default=0.02)
    parser.add_argument("--stability-min-history-folds", type=int, default=2)
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    parser.add_argument("--fail-on-invalid-fold", action="store_true")
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    variant_list = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    result = run_refine_h20_features(
        model_dataset_path=args.model_dataset_path,
        purged_cv_folds_path=args.purged_cv_folds_path,
        output_dir=args.output_dir,
        label_name=args.label_name,
        target_type=args.target_type,
        horizon_days=args.horizon_days,
        variants=variant_list,
        alpha_grid=_parse_csv_floats(args.ridge_alphas) or (0.1, 1.0, 10.0),
        missingness_threshold=float(args.missingness_threshold),
        collinearity_threshold=float(args.collinearity_threshold),
        min_abs_train_corr=float(args.min_abs_train_corr),
        stability_min_history_folds=int(args.stability_min_history_folds),
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
