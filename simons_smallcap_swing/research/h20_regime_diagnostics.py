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
from sklearn.linear_model import Ridge

# Allow direct script execution:
# `python simons_smallcap_swing/research/h20_regime_diagnostics.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.refine_h20_features import (
    BASELINE_VARIANT,
    DEFAULT_VARIANTS as FEATURE_VARIANTS,
    VARIANT_STABLE_PLUS_LOW_COL,
    _compute_train_sign_maps,
    _fit_ridge_valid_mse,
    _load_joined_inputs,
    _prepare_fold_matrix,
    _select_features_for_fold,
)
from simons_core.io.parquet_store import write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "h20_regime_diagnostics_mvp_v1"
PRIMARY_METRIC = "mse"
DEFAULT_MODEL_DATASET_PATH = "datasets/regression_h20/model_dataset.parquet"
DEFAULT_FOLDS_PATH = "labels/purged_cv_folds.parquet"
DEFAULT_CANDIDATE_VARIANTS: tuple[str, ...] = (BASELINE_VARIANT, VARIANT_STABLE_PLUS_LOW_COL)

SEGMENT_FAMILY_TO_COLUMNS: dict[str, tuple[str, ...]] = {
    "high_vs_low_volatility": ("vol_20d", "vol_5d"),
    "high_vs_low_market_breadth": ("mkt_breadth_up_lag1", "mkt_coverage_ratio_lag1"),
    "high_vs_low_cross_sectional_dispersion": ("mkt_cross_sectional_vol_lag1",),
    "high_vs_low_liquidity": ("log_dollar_volume_lag1", "turnover_proxy_lag1", "log_volume_lag1"),
}
TIME_SPLIT_FAMILY = "time_split_early_vs_late"

VALID_REQUIRED_COLUMNS: tuple[str, ...] = (
    "fold_id",
    "date",
    "instrument_id",
    "target_value",
    "model_prediction",
    "dummy_prediction",
    "candidate_variant",
)

RESULTS_SCHEMA = DataSchema(
    name="h20_regime_diagnostics_results_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("candidate_variant", "string", nullable=False),
        ColumnSpec("segment_family", "string", nullable=False),
        ColumnSpec("segment_name", "string", nullable=False),
        ColumnSpec("n_obs", "int64", nullable=False),
        ColumnSpec("model_mse", "float64", nullable=False),
        ColumnSpec("dummy_mse", "float64", nullable=False),
        ColumnSpec("improvement_vs_dummy", "float64", nullable=False),
        ColumnSpec("winner_vs_dummy", "string", nullable=False),
    ),
    primary_key=(),
    allow_extra_columns=True,
)

FOLD_SCHEMA = DataSchema(
    name="h20_regime_diagnostics_fold_metrics_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("candidate_variant", "string", nullable=False),
        ColumnSpec("segment_family", "string", nullable=False),
        ColumnSpec("segment_name", "string", nullable=False),
        ColumnSpec("fold_id", "int64", nullable=False),
        ColumnSpec("n_obs", "int64", nullable=False),
        ColumnSpec("model_mse", "float64", nullable=False),
        ColumnSpec("dummy_mse", "float64", nullable=False),
        ColumnSpec("improvement_vs_dummy", "float64", nullable=False),
        ColumnSpec("winner_vs_dummy", "string", nullable=False),
    ),
    primary_key=(),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class H20RegimeDiagnosticsResult:
    results_path: Path
    summary_path: Path
    fold_metrics_path: Path
    n_rows: int
    config_hash: str


def _cfg_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


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
    return tuple(item.strip() for item in text.split(",") if item.strip())


def _normalize_float_grid(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    normalized = tuple(sorted({float(v) for v in values}))
    if not normalized:
        raise ValueError(f"{name} cannot be empty.")
    if any(v <= 0 for v in normalized):
        raise ValueError(f"{name} must contain positive values. Received: {normalized}")
    return normalized


def _normalize_variant_list(variants: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(str(v).strip() for v in variants if str(v).strip()))
    if not selected:
        raise ValueError("candidate_variants cannot be empty.")
    invalid = sorted(set(selected) - set(FEATURE_VARIANTS))
    if invalid:
        raise ValueError(f"Unsupported candidate_variants: {invalid}. Allowed: {FEATURE_VARIANTS}")
    if BASELINE_VARIANT not in selected:
        raise ValueError(f"{BASELINE_VARIANT} must be included in candidate_variants.")
    return selected


def _winner_vs_dummy(improvement: float, *, tie_tolerance: float) -> str:
    if abs(float(improvement)) <= float(tie_tolerance):
        return "tie"
    return "variant" if float(improvement) > 0 else "dummy"


def _resolve_segment_columns(feature_cols: tuple[str, ...]) -> tuple[dict[str, str], list[str]]:
    resolved: dict[str, str] = {}
    notes: list[str] = []
    feature_set = set(feature_cols)
    for family, candidates in SEGMENT_FAMILY_TO_COLUMNS.items():
        match = next((col for col in candidates if col in feature_set), None)
        if match is None:
            notes.append(f"Segment family '{family}' skipped: no proxy columns found in dataset.")
            continue
        resolved[family] = match
    return resolved, notes


def _collect_candidate_predictions(
    *,
    merged: pd.DataFrame,
    feature_cols: tuple[str, ...],
    fold_ids: tuple[int, ...],
    candidate_variants: tuple[str, ...],
    alpha_grid: tuple[float, ...],
    missingness_threshold: float,
    collinearity_threshold: float,
    min_abs_train_corr: float,
    stability_min_history_folds: int,
    fail_on_invalid_fold: bool,
    run_id: str,
    config_hash: str,
    built_ts_utc: str,
    logger: Any,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    sign_map = _compute_train_sign_maps(
        merged=merged,
        feature_cols=feature_cols,
        fold_ids=fold_ids,
        min_abs_train_corr=float(min_abs_train_corr),
    )
    notes: list[str] = []
    prediction_frames: list[pd.DataFrame] = []
    variant_fold_counts: dict[str, int] = {}

    for variant in candidate_variants:
        variant_fold_ids: set[int] = set()
        for fold_id in fold_ids:
            fold = merged[merged["fold_id"] == int(fold_id)].copy()
            train_df = fold[fold["split_role"] == "train"].copy()
            valid_df = fold[fold["split_role"] == "valid"].copy()
            if train_df.empty or valid_df.empty:
                reason = f"variant={variant} fold={fold_id}: empty train/valid (n_train={len(train_df)}, n_valid={len(valid_df)})"
                if fail_on_invalid_fold:
                    raise ValueError(reason)
                notes.append(f"Skipped {reason}")
                continue

            try:
                selected_cols, _details = _select_features_for_fold(
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
                best_alpha, _ = _fit_ridge_valid_mse(
                    x_train=x_train,
                    y_train=y_train,
                    x_valid=x_valid,
                    y_valid=y_valid,
                    alpha_grid=alpha_grid,
                )
            except Exception as exc:
                reason = f"variant={variant} fold={fold_id} failed: {exc}"
                if fail_on_invalid_fold:
                    raise ValueError(reason) from exc
                notes.append(f"Skipped {reason}")
                continue

            model = Ridge(alpha=float(best_alpha), fit_intercept=True)
            model.fit(x_train, y_train)
            model_pred = model.predict(x_valid)
            dummy_pred = np.full(shape=y_valid.shape, fill_value=float(np.mean(y_train)), dtype=float)

            base_cols = [
                "fold_id",
                "date",
                "instrument_id",
                "ticker",
                "horizon_days",
                "label_name",
                "target_type",
                "target_value",
            ]
            payload_feature_cols = [col for col in feature_cols if col not in set(base_cols)]
            valid_base = valid_df[base_cols + payload_feature_cols].copy()
            valid_base["candidate_variant"] = str(variant)
            valid_base["model_prediction"] = model_pred.astype(float)
            valid_base["dummy_prediction"] = dummy_pred.astype(float)
            valid_base["alpha_selected"] = float(best_alpha)
            valid_base["n_features_used"] = int(len(kept_cols))
            valid_base["run_id"] = run_id
            valid_base["config_hash"] = config_hash
            valid_base["built_ts_utc"] = built_ts_utc
            prediction_frames.append(valid_base)
            variant_fold_ids.add(int(fold_id))

        variant_fold_counts[str(variant)] = int(len(variant_fold_ids))

    if not prediction_frames:
        raise ValueError("No candidate variant produced valid predictions for regime diagnostics.")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    for col in VALID_REQUIRED_COLUMNS:
        if col not in predictions.columns:
            raise ValueError(f"Internal predictions table missing required column '{col}'.")
    return predictions, notes, variant_fold_counts


def _restrict_to_common_folds(
    predictions: pd.DataFrame,
    *,
    candidate_variants: tuple[str, ...],
) -> tuple[pd.DataFrame, list[int]]:
    fold_sets: list[set[int]] = []
    for variant in candidate_variants:
        variant_rows = predictions[predictions["candidate_variant"].astype(str) == str(variant)]
        if variant_rows.empty:
            continue
        fold_sets.append(set(pd.to_numeric(variant_rows["fold_id"], errors="coerce").dropna().astype(int)))
    if not fold_sets:
        raise ValueError("No completed folds found for any candidate variant.")
    common = set.intersection(*fold_sets)
    common_sorted = sorted(common)
    if not common_sorted:
        raise ValueError("No common fold_id intersection across candidate variants.")
    out = predictions[
        pd.to_numeric(predictions["fold_id"], errors="coerce").astype(int).isin(set(common_sorted))
    ].copy()
    return out, common_sorted


def _build_segment_assignments(
    *,
    merged: pd.DataFrame,
    common_fold_ids: list[int],
    resolved_segment_columns: dict[str, str],
    notes: list[str],
) -> pd.DataFrame:
    assignments: list[pd.DataFrame] = []

    for family, column in resolved_segment_columns.items():
        for fold_id in common_fold_ids:
            fold = merged[merged["fold_id"] == int(fold_id)].copy()
            train_vals = pd.to_numeric(
                fold.loc[fold["split_role"] == "train", column],
                errors="coerce",
            )
            if not train_vals.notna().any():
                notes.append(
                    f"Segment family '{family}' fold={fold_id} skipped: no train values for column '{column}'."
                )
                continue
            threshold = float(train_vals.median(skipna=True))
            valid_rows = fold[fold["split_role"] == "valid"].copy()
            valid_rows[column] = pd.to_numeric(valid_rows[column], errors="coerce")
            valid_rows = valid_rows[valid_rows[column].notna()].copy()
            if valid_rows.empty:
                notes.append(
                    f"Segment family '{family}' fold={fold_id} skipped: no valid values for column '{column}'."
                )
                continue
            valid_rows["segment_family"] = family
            valid_rows["segment_name"] = np.where(valid_rows[column] >= threshold, "high", "low")
            valid_rows["segment_column"] = column
            valid_rows["segment_threshold"] = threshold
            assignments.append(
                valid_rows[
                    [
                        "fold_id",
                        "date",
                        "instrument_id",
                        "segment_family",
                        "segment_name",
                        "segment_column",
                        "segment_threshold",
                    ]
                ].copy()
            )

    valid_common = merged[
        (merged["split_role"] == "valid")
        & (pd.to_numeric(merged["fold_id"], errors="coerce").astype(int).isin(set(common_fold_ids)))
    ].copy()
    if not valid_common.empty:
        dates = sorted(pd.to_datetime(valid_common["date"]).dt.normalize().unique().tolist())
        split_date = pd.Timestamp(dates[(len(dates) - 1) // 2])
        valid_common["segment_family"] = TIME_SPLIT_FAMILY
        valid_common["segment_name"] = np.where(
            pd.to_datetime(valid_common["date"]).dt.normalize() <= split_date,
            "early",
            "late",
        )
        valid_common["segment_column"] = "date"
        valid_common["segment_threshold"] = str(split_date.date())
        assignments.append(
            valid_common[
                [
                    "fold_id",
                    "date",
                    "instrument_id",
                    "segment_family",
                    "segment_name",
                    "segment_column",
                    "segment_threshold",
                ]
            ].copy()
        )

    if not assignments:
        raise ValueError("No segment assignments could be constructed from available regime proxies.")
    out = pd.concat(assignments, ignore_index=True)
    out = out.drop_duplicates(["fold_id", "date", "instrument_id", "segment_family"], keep="first")
    return out


def _segment_metrics_frame(
    *,
    segmented: pd.DataFrame,
    tie_tolerance: float,
    run_id: str,
    config_hash: str,
    built_ts_utc: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    group_cols = ["candidate_variant", "segment_family", "segment_name"]
    for keys, group in segmented.groupby(group_cols, dropna=False):
        candidate_variant, family, name = (str(keys[0]), str(keys[1]), str(keys[2]))
        model_err = np.square(
            pd.to_numeric(group["target_value"], errors="coerce")
            - pd.to_numeric(group["model_prediction"], errors="coerce")
        )
        dummy_err = np.square(
            pd.to_numeric(group["target_value"], errors="coerce")
            - pd.to_numeric(group["dummy_prediction"], errors="coerce")
        )
        model_mse = float(np.nanmean(model_err))
        dummy_mse = float(np.nanmean(dummy_err))
        improvement = float(dummy_mse - model_mse)
        result_rows.append(
            {
                "candidate_variant": candidate_variant,
                "segment_family": family,
                "segment_name": name,
                "n_obs": int(len(group)),
                "model_mse": model_mse,
                "dummy_mse": dummy_mse,
                "improvement_vs_dummy": improvement,
                "winner_vs_dummy": _winner_vs_dummy(improvement, tie_tolerance=float(tie_tolerance)),
                "run_id": run_id,
                "config_hash": config_hash,
                "built_ts_utc": built_ts_utc,
            }
        )

    fold_group_cols = ["candidate_variant", "segment_family", "segment_name", "fold_id"]
    for keys, group in segmented.groupby(fold_group_cols, dropna=False):
        candidate_variant, family, name, fold_id = (
            str(keys[0]),
            str(keys[1]),
            str(keys[2]),
            int(keys[3]),
        )
        model_err = np.square(
            pd.to_numeric(group["target_value"], errors="coerce")
            - pd.to_numeric(group["model_prediction"], errors="coerce")
        )
        dummy_err = np.square(
            pd.to_numeric(group["target_value"], errors="coerce")
            - pd.to_numeric(group["dummy_prediction"], errors="coerce")
        )
        model_mse = float(np.nanmean(model_err))
        dummy_mse = float(np.nanmean(dummy_err))
        improvement = float(dummy_mse - model_mse)
        fold_rows.append(
            {
                "candidate_variant": candidate_variant,
                "segment_family": family,
                "segment_name": name,
                "fold_id": fold_id,
                "n_obs": int(len(group)),
                "model_mse": model_mse,
                "dummy_mse": dummy_mse,
                "improvement_vs_dummy": improvement,
                "winner_vs_dummy": _winner_vs_dummy(improvement, tie_tolerance=float(tie_tolerance)),
                "run_id": run_id,
                "config_hash": config_hash,
                "built_ts_utc": built_ts_utc,
            }
        )

    if not result_rows:
        raise ValueError("No segment-level diagnostics rows were produced.")
    return pd.DataFrame(result_rows), pd.DataFrame(fold_rows)


def run_h20_regime_diagnostics(
    *,
    model_dataset_path: str | Path | None = None,
    purged_cv_folds_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    label_name: str = "fwd_ret_20d",
    target_type: str = "continuous_forward_return",
    horizon_days: int = 20,
    candidate_variants: Iterable[str] = DEFAULT_CANDIDATE_VARIANTS,
    alpha_grid: Iterable[float] = (0.1, 1.0, 10.0),
    missingness_threshold: float = 0.35,
    collinearity_threshold: float = 0.95,
    min_abs_train_corr: float = 0.02,
    stability_min_history_folds: int = 2,
    tie_tolerance: float = 1e-12,
    fail_on_invalid_fold: bool = False,
    run_id: str = MODULE_VERSION,
) -> H20RegimeDiagnosticsResult:
    logger = get_logger("research.h20_regime_diagnostics")
    if int(horizon_days) <= 0:
        raise ValueError("horizon_days must be positive.")
    if float(tie_tolerance) < 0:
        raise ValueError("tie_tolerance must be >= 0.")
    if not (0.0 <= float(missingness_threshold) <= 1.0):
        raise ValueError("missingness_threshold must be in [0, 1].")
    if not (0.0 < float(collinearity_threshold) <= 1.0):
        raise ValueError("collinearity_threshold must be in (0, 1].")
    if float(min_abs_train_corr) < 0:
        raise ValueError("min_abs_train_corr must be >= 0.")
    if int(stability_min_history_folds) <= 0:
        raise ValueError("stability_min_history_folds must be >= 1.")

    selected_variants = _normalize_variant_list(candidate_variants)
    alpha_values = _normalize_float_grid(alpha_grid, name="alpha_grid")

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

    merged, feature_cols, fold_ids = _load_joined_inputs(
        model_dataset_path=dataset_source,
        purged_cv_folds_path=folds_source,
        label_name=label_name,
        target_type=target_type,
        horizon_days=int(horizon_days),
    )
    resolved_segment_columns, notes = _resolve_segment_columns(feature_cols)
    built_ts_utc = datetime.now(UTC).isoformat()
    config_hash = _cfg_hash(
        {
            "module_version": MODULE_VERSION,
            "model_dataset_path": str(dataset_source),
            "purged_cv_folds_path": str(folds_source),
            "label_name": label_name,
            "target_type": target_type,
            "horizon_days": int(horizon_days),
            "candidate_variants": list(selected_variants),
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

    predictions, pred_notes, variant_fold_counts = _collect_candidate_predictions(
        merged=merged,
        feature_cols=feature_cols,
        fold_ids=fold_ids,
        candidate_variants=selected_variants,
        alpha_grid=alpha_values,
        missingness_threshold=float(missingness_threshold),
        collinearity_threshold=float(collinearity_threshold),
        min_abs_train_corr=float(min_abs_train_corr),
        stability_min_history_folds=int(stability_min_history_folds),
        fail_on_invalid_fold=bool(fail_on_invalid_fold),
        run_id=run_id,
        config_hash=config_hash,
        built_ts_utc=built_ts_utc,
        logger=logger,
    )
    notes.extend(pred_notes)

    predictions_common, common_folds = _restrict_to_common_folds(
        predictions,
        candidate_variants=selected_variants,
    )
    notes.append("Comparison policy: same_common_folds_only across candidate variants.")
    notes.append(f"Common fold ids used: {common_folds}")
    notes.append(f"Variant fold counts before common-fold restriction: {variant_fold_counts}")

    segment_assignments = _build_segment_assignments(
        merged=merged,
        common_fold_ids=common_folds,
        resolved_segment_columns=resolved_segment_columns,
        notes=notes,
    )
    segmented = predictions_common.merge(
        segment_assignments,
        on=["fold_id", "date", "instrument_id"],
        how="inner",
    )
    if segmented.empty:
        raise ValueError("Segment assignments produced no overlap with valid prediction rows.")

    results_df, fold_df = _segment_metrics_frame(
        segmented=segmented,
        tie_tolerance=float(tie_tolerance),
        run_id=run_id,
        config_hash=config_hash,
        built_ts_utc=built_ts_utc,
    )

    for col in ("candidate_variant", "segment_family", "segment_name", "winner_vs_dummy"):
        results_df[col] = results_df[col].astype("string")
    for col in ("model_mse", "dummy_mse", "improvement_vs_dummy"):
        results_df[col] = pd.to_numeric(results_df[col], errors="coerce")
    results_df["n_obs"] = pd.to_numeric(results_df["n_obs"], errors="coerce").astype("int64")
    assert_schema(results_df, RESULTS_SCHEMA)

    for col in ("candidate_variant", "segment_family", "segment_name", "winner_vs_dummy"):
        fold_df[col] = fold_df[col].astype("string")
    for col in ("model_mse", "dummy_mse", "improvement_vs_dummy"):
        fold_df[col] = pd.to_numeric(fold_df[col], errors="coerce")
    fold_df["n_obs"] = pd.to_numeric(fold_df["n_obs"], errors="coerce").astype("int64")
    fold_df["fold_id"] = pd.to_numeric(fold_df["fold_id"], errors="coerce").astype("int64")
    assert_schema(fold_df, FOLD_SCHEMA)

    results_path = write_parquet(
        results_df,
        target_dir / "h20_regime_diagnostics_results.parquet",
        schema_name=RESULTS_SCHEMA.name,
        run_id=run_id,
    )
    fold_metrics_path = write_parquet(
        fold_df,
        target_dir / "h20_regime_diagnostics_fold_metrics.parquet",
        schema_name=FOLD_SCHEMA.name,
        run_id=run_id,
    )

    beats_dummy: dict[str, list[str]] = {}
    loses_dummy: dict[str, list[str]] = {}
    best_segment_by_variant: dict[str, dict[str, Any]] = {}
    for variant, group in results_df.groupby("candidate_variant", dropna=False):
        variant_str = str(variant)
        beats = group[group["winner_vs_dummy"].astype(str) == "variant"]
        loses = group[group["winner_vs_dummy"].astype(str) == "dummy"]
        beats_dummy[variant_str] = sorted(
            [f"{r.segment_family}:{r.segment_name}" for r in beats.itertuples(index=False)]
        )
        loses_dummy[variant_str] = sorted(
            [f"{r.segment_family}:{r.segment_name}" for r in loses.itertuples(index=False)]
        )
        best_row = group.sort_values(["improvement_vs_dummy", "n_obs"], ascending=[False, False]).iloc[0]
        best_segment_by_variant[variant_str] = {
            "segment_family": str(best_row["segment_family"]),
            "segment_name": str(best_row["segment_name"]),
            "improvement_vs_dummy": float(best_row["improvement_vs_dummy"]),
            "winner_vs_dummy": str(best_row["winner_vs_dummy"]),
            "n_obs": int(best_row["n_obs"]),
        }

    total_beating_segments = int(
        (results_df["winner_vs_dummy"].astype(str) == "variant").sum()
    )
    if total_beating_segments == 0:
        recommendation = "no_regime_specific_edge_detected_continue_features_labels_refinement"
    elif total_beating_segments <= 2:
        recommendation = "weak_regime_specific_edge_revalidate_before_action"
    else:
        recommendation = "focus_next_refinement_on_segments_with_positive_edge"

    summary_payload = {
        "module_version": MODULE_VERSION,
        "candidate_variants_evaluated": sorted(results_df["candidate_variant"].astype(str).unique().tolist()),
        "segment_families_evaluated": sorted(results_df["segment_family"].astype(str).unique().tolist()),
        "segments_where_model_beats_dummy": beats_dummy,
        "segments_where_model_loses_to_dummy": loses_dummy,
        "best_segment_by_variant": best_segment_by_variant,
        "recommendation": recommendation,
        "label_name": label_name,
        "target_type": target_type,
        "horizon_days": int(horizon_days),
        "comparison_policy": {
            "same_common_folds_only": True,
            "common_fold_ids": common_folds,
            "n_common_folds": int(len(common_folds)),
            "variant_fold_counts_before_common": variant_fold_counts,
        },
        "segment_policy": {
            "train_only_thresholds_for_high_low_segments": True,
            "time_split_rule": "early_vs_late_by_median_valid_date_within_common_folds",
            "segment_column_by_family": resolved_segment_columns,
        },
        "input_paths": {
            "model_dataset": str(dataset_source),
            "purged_cv_folds": str(folds_source),
        },
        "output_paths": {
            "results": str(results_path),
            "fold_metrics": str(fold_metrics_path),
        },
        "run_id": run_id,
        "config_hash": config_hash,
        "built_ts_utc": built_ts_utc,
        "notes": notes,
    }
    summary_path = target_dir / "h20_regime_diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "h20_regime_diagnostics_completed",
        run_id=run_id,
        n_results_rows=int(len(results_df)),
        n_variants=int(results_df["candidate_variant"].nunique()),
        output_path=str(results_path),
    )

    return H20RegimeDiagnosticsResult(
        results_path=results_path,
        summary_path=summary_path,
        fold_metrics_path=fold_metrics_path,
        n_rows=int(len(results_df)),
        config_hash=config_hash,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run H20 regime/subuniverse diagnostics for selected feature variants."
    )
    parser.add_argument("--model-dataset-path", type=str, default=None)
    parser.add_argument("--purged-cv-folds-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--label-name", type=str, default="fwd_ret_20d")
    parser.add_argument("--target-type", type=str, default="continuous_forward_return")
    parser.add_argument("--horizon-days", type=int, default=20)
    parser.add_argument(
        "--candidate-variants",
        type=str,
        default=",".join(DEFAULT_CANDIDATE_VARIANTS),
        help="Comma-separated variants from refine_h20_features.py.",
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
    result = run_h20_regime_diagnostics(
        model_dataset_path=args.model_dataset_path,
        purged_cv_folds_path=args.purged_cv_folds_path,
        output_dir=args.output_dir,
        label_name=args.label_name,
        target_type=args.target_type,
        horizon_days=int(args.horizon_days),
        candidate_variants=_parse_csv_texts(args.candidate_variants) or DEFAULT_CANDIDATE_VARIANTS,
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
                "n_rows": int(result.n_rows),
                "config_hash": result.config_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
