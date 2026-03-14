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

# Allow direct script execution:
# `python simons_smallcap_swing/research/h20_regime_conditioned_refinement.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.h20_regime_diagnostics import (
    DEFAULT_FOLDS_PATH,
    DEFAULT_MODEL_DATASET_PATH,
    _build_segment_assignments,
    _collect_candidate_predictions,
    _normalize_float_grid,
    _parse_csv_floats,
    _parse_csv_texts,
    _resolve_segment_columns,
    _restrict_to_common_folds,
)
from research.refine_h20_features import (
    BASELINE_VARIANT,
    DEFAULT_VARIANTS as FEATURE_VARIANTS,
    VARIANT_STABLE_PLUS_LOW_COL,
    _load_joined_inputs,
)
from simons_core.io.parquet_store import write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "h20_regime_conditioned_refinement_mvp_v1"
PRIMARY_METRIC = "mse"
DEFAULT_CANDIDATE_VARIANTS: tuple[str, ...] = (BASELINE_VARIANT, VARIANT_STABLE_PLUS_LOW_COL)
DEFAULT_TARGET_REGIMES: tuple[str, ...] = (
    "time_split_early_vs_late:late",
    "high_vs_low_market_breadth:low",
    "high_vs_low_cross_sectional_dispersion:low",
    "high_vs_low_liquidity:low",
)

RESULTS_SCHEMA = DataSchema(
    name="h20_regime_conditioned_results_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("candidate_variant", "string", nullable=False),
        ColumnSpec("regime_family", "string", nullable=False),
        ColumnSpec("regime_name", "string", nullable=False),
        ColumnSpec("n_obs", "int64", nullable=False),
        ColumnSpec("n_folds_used", "int64", nullable=False),
        ColumnSpec("model_mse", "float64", nullable=False),
        ColumnSpec("dummy_mse", "float64", nullable=False),
        ColumnSpec("improvement_vs_dummy", "float64", nullable=False),
        ColumnSpec("winner_vs_dummy", "string", nullable=False),
        ColumnSpec("comparison_variant", "string", nullable=False),
        ColumnSpec("improvement_vs_comparison_variant", "float64", nullable=False),
        ColumnSpec("winner_vs_comparison_variant", "string", nullable=False),
    ),
    primary_key=(),
    allow_extra_columns=True,
)

FOLD_SCHEMA = DataSchema(
    name="h20_regime_conditioned_fold_metrics_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("candidate_variant", "string", nullable=False),
        ColumnSpec("regime_family", "string", nullable=False),
        ColumnSpec("regime_name", "string", nullable=False),
        ColumnSpec("fold_id", "int64", nullable=False),
        ColumnSpec("n_obs", "int64", nullable=False),
        ColumnSpec("model_mse", "float64", nullable=False),
        ColumnSpec("dummy_mse", "float64", nullable=False),
        ColumnSpec("improvement_vs_dummy", "float64", nullable=False),
        ColumnSpec("winner_vs_dummy", "string", nullable=False),
        ColumnSpec("comparison_variant", "string", nullable=False),
        ColumnSpec("improvement_vs_comparison_variant", "float64", nullable=False),
        ColumnSpec("winner_vs_comparison_variant", "string", nullable=False),
    ),
    primary_key=(),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class H20RegimeConditionedRefinementResult:
    results_path: Path
    summary_path: Path
    fold_metrics_path: Path
    n_rows: int
    config_hash: str


def _cfg_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _normalize_variant_list(variants: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(str(v).strip() for v in variants if str(v).strip()))
    if not selected:
        raise ValueError("candidate_variants cannot be empty.")
    invalid = sorted(set(selected) - set(FEATURE_VARIANTS))
    if invalid:
        raise ValueError(f"Unsupported candidate_variants: {invalid}. Allowed: {FEATURE_VARIANTS}")
    if BASELINE_VARIANT not in selected:
        raise ValueError(f"{BASELINE_VARIANT} must be included in candidate_variants.")
    if len(selected) < 2:
        raise ValueError("candidate_variants must contain at least 2 entries for comparison.")
    return selected


def _normalize_regime_targets(values: Iterable[str]) -> tuple[tuple[str, str], ...]:
    targets: list[tuple[str, str]] = []
    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(
                f"Invalid regime target '{text}'. Expected format '<regime_family>:<regime_name>'."
            )
        family, name = text.split(":", 1)
        family_norm = family.strip()
        name_norm = name.strip()
        if not family_norm or not name_norm:
            raise ValueError(f"Invalid regime target '{text}'. Family/name cannot be empty.")
        targets.append((family_norm, name_norm))
    deduped = tuple(dict.fromkeys(targets))
    if not deduped:
        raise ValueError("target_regimes cannot be empty.")
    return deduped


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


def _safe_mse(y_true: pd.Series, y_pred: pd.Series) -> float:
    err = pd.to_numeric(y_true, errors="coerce") - pd.to_numeric(y_pred, errors="coerce")
    return float(np.nanmean(np.square(err)))


def _comparison_partner_map(
    *,
    variants: tuple[str, ...],
    comparison_variant: str,
) -> dict[str, str]:
    if str(comparison_variant) not in set(variants):
        raise ValueError(
            f"comparison_variant='{comparison_variant}' must be included in candidate_variants."
        )
    if len(variants) == 2:
        a, b = variants
        return {a: b, b: a}

    first_alt = next((v for v in variants if v != comparison_variant), None)
    if first_alt is None:
        raise ValueError("Unable to resolve comparison variant mapping.")
    partner: dict[str, str] = {}
    for variant in variants:
        if variant == comparison_variant:
            partner[variant] = first_alt
        else:
            partner[variant] = comparison_variant
    return partner


def _evaluate_regime_block(
    *,
    regime_df: pd.DataFrame,
    regime_family: str,
    regime_name: str,
    candidate_variants: tuple[str, ...],
    comparison_partner_by_variant: dict[str, str],
    tie_tolerance: float,
    run_id: str,
    config_hash: str,
    built_ts_utc: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int], str | None]:
    fold_sets: list[set[int]] = []
    for variant in candidate_variants:
        variant_folds = set(
            pd.to_numeric(
                regime_df.loc[regime_df["candidate_variant"].astype(str) == str(variant), "fold_id"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .tolist()
        )
        if not variant_folds:
            return pd.DataFrame(), pd.DataFrame(), [], (
                f"Regime {regime_family}:{regime_name} skipped: variant={variant} has no rows."
            )
        fold_sets.append(variant_folds)

    common_folds = sorted(set.intersection(*fold_sets))
    if not common_folds:
        return pd.DataFrame(), pd.DataFrame(), [], (
            f"Regime {regime_family}:{regime_name} skipped: no common folds across variants."
        )

    block = regime_df[
        pd.to_numeric(regime_df["fold_id"], errors="coerce").astype(int).isin(set(common_folds))
    ].copy()
    if block.empty:
        return pd.DataFrame(), pd.DataFrame(), [], (
            f"Regime {regime_family}:{regime_name} skipped: empty block after common-fold filtering."
        )

    agg_metrics: dict[str, dict[str, float]] = {}
    for variant in candidate_variants:
        vg = block[block["candidate_variant"].astype(str) == str(variant)].copy()
        if vg.empty:
            return pd.DataFrame(), pd.DataFrame(), [], (
                f"Regime {regime_family}:{regime_name} skipped: variant={variant} empty after common folds."
            )
        agg_metrics[str(variant)] = {
            "n_obs": int(len(vg)),
            "model_mse": _safe_mse(vg["target_value"], vg["model_prediction"]),
            "dummy_mse": _safe_mse(vg["target_value"], vg["dummy_prediction"]),
            "n_folds_used": int(len(common_folds)),
        }

    rows: list[dict[str, Any]] = []
    for variant in candidate_variants:
        variant_str = str(variant)
        comp_variant = str(comparison_partner_by_variant[variant_str])
        model_mse = float(agg_metrics[variant_str]["model_mse"])
        dummy_mse = float(agg_metrics[variant_str]["dummy_mse"])
        improvement_dummy = float(dummy_mse - model_mse)
        other_mse = float(agg_metrics[comp_variant]["model_mse"])
        improvement_other = float(other_mse - model_mse)
        rows.append(
            {
                "candidate_variant": variant_str,
                "regime_family": regime_family,
                "regime_name": regime_name,
                "n_obs": int(agg_metrics[variant_str]["n_obs"]),
                "n_folds_used": int(agg_metrics[variant_str]["n_folds_used"]),
                "model_mse": model_mse,
                "dummy_mse": dummy_mse,
                "improvement_vs_dummy": improvement_dummy,
                "winner_vs_dummy": _winner(
                    improvement_dummy,
                    tie_tolerance=float(tie_tolerance),
                    positive_label="variant",
                    negative_label="dummy",
                ),
                "comparison_variant": comp_variant,
                "improvement_vs_comparison_variant": improvement_other,
                "winner_vs_comparison_variant": _winner(
                    improvement_other,
                    tie_tolerance=float(tie_tolerance),
                    positive_label="variant",
                    negative_label="other_variant",
                ),
                "run_id": run_id,
                "config_hash": config_hash,
                "built_ts_utc": built_ts_utc,
            }
        )

    fold_model_mse: dict[tuple[str, int], float] = {}
    fold_dummy_mse: dict[tuple[str, int], float] = {}
    fold_n_obs: dict[tuple[str, int], int] = {}
    for (variant, fold_id), group in block.groupby(["candidate_variant", "fold_id"], dropna=False):
        key = (str(variant), int(fold_id))
        fold_model_mse[key] = _safe_mse(group["target_value"], group["model_prediction"])
        fold_dummy_mse[key] = _safe_mse(group["target_value"], group["dummy_prediction"])
        fold_n_obs[key] = int(len(group))

    fold_rows: list[dict[str, Any]] = []
    for variant in candidate_variants:
        variant_str = str(variant)
        comp_variant = str(comparison_partner_by_variant[variant_str])
        for fold_id in common_folds:
            key = (variant_str, int(fold_id))
            if key not in fold_model_mse:
                continue
            model_mse = float(fold_model_mse[key])
            dummy_mse = float(fold_dummy_mse[key])
            improvement_dummy = float(dummy_mse - model_mse)
            comp_key = (comp_variant, int(fold_id))
            if comp_key in fold_model_mse:
                improvement_other = float(fold_model_mse[comp_key] - model_mse)
                winner_other = _winner(
                    improvement_other,
                    tie_tolerance=float(tie_tolerance),
                    positive_label="variant",
                    negative_label="other_variant",
                )
            else:
                improvement_other = float("nan")
                winner_other = "tie"

            fold_rows.append(
                {
                    "candidate_variant": variant_str,
                    "regime_family": regime_family,
                    "regime_name": regime_name,
                    "fold_id": int(fold_id),
                    "n_obs": int(fold_n_obs[key]),
                    "model_mse": model_mse,
                    "dummy_mse": dummy_mse,
                    "improvement_vs_dummy": improvement_dummy,
                    "winner_vs_dummy": _winner(
                        improvement_dummy,
                        tie_tolerance=float(tie_tolerance),
                        positive_label="variant",
                        negative_label="dummy",
                    ),
                    "comparison_variant": comp_variant,
                    "improvement_vs_comparison_variant": improvement_other,
                    "winner_vs_comparison_variant": winner_other,
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "built_ts_utc": built_ts_utc,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(fold_rows), common_folds, None


def run_h20_regime_conditioned_refinement(
    *,
    model_dataset_path: str | Path | None = None,
    purged_cv_folds_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    label_name: str = "fwd_ret_20d",
    target_type: str = "continuous_forward_return",
    horizon_days: int = 20,
    candidate_variants: Iterable[str] = DEFAULT_CANDIDATE_VARIANTS,
    target_regimes: Iterable[str] = DEFAULT_TARGET_REGIMES,
    comparison_variant: str = BASELINE_VARIANT,
    alpha_grid: Iterable[float] = (0.1, 1.0, 10.0),
    missingness_threshold: float = 0.35,
    collinearity_threshold: float = 0.95,
    min_abs_train_corr: float = 0.02,
    stability_min_history_folds: int = 2,
    tie_tolerance: float = 1e-12,
    fail_on_invalid_fold: bool = False,
    run_id: str = MODULE_VERSION,
) -> H20RegimeConditionedRefinementResult:
    logger = get_logger("research.h20_regime_conditioned_refinement")
    if str(target_type) != "continuous_forward_return":
        raise ValueError("target_type must be 'continuous_forward_return' for ridge H20 conditioning.")
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
    selected_regimes = _normalize_regime_targets(target_regimes)
    comparison_partner_by_variant = _comparison_partner_map(
        variants=selected_variants,
        comparison_variant=str(comparison_variant).strip(),
    )
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
            "target_regimes": [f"{f}:{n}" for f, n in selected_regimes],
            "comparison_variant": str(comparison_variant).strip(),
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

    results_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    per_regime_common_folds: dict[str, list[int]] = {}
    available_regimes = {
        (str(row.segment_family), str(row.segment_name))
        for row in segmented[["segment_family", "segment_name"]].drop_duplicates().itertuples(index=False)
    }

    for regime_family, regime_name in selected_regimes:
        regime_key = f"{regime_family}:{regime_name}"
        if (regime_family, regime_name) not in available_regimes:
            notes.append(f"Regime {regime_key} not materialized from current artifacts. Skipped.")
            continue
        regime_df = segmented[
            (segmented["segment_family"].astype(str) == str(regime_family))
            & (segmented["segment_name"].astype(str) == str(regime_name))
            & (segmented["candidate_variant"].astype(str).isin(set(selected_variants)))
        ].copy()
        if regime_df.empty:
            notes.append(f"Regime {regime_key} has no rows after candidate filtering. Skipped.")
            continue

        regime_results, regime_fold, regime_common, skip_reason = _evaluate_regime_block(
            regime_df=regime_df,
            regime_family=regime_family,
            regime_name=regime_name,
            candidate_variants=selected_variants,
            comparison_partner_by_variant=comparison_partner_by_variant,
            tie_tolerance=float(tie_tolerance),
            run_id=run_id,
            config_hash=config_hash,
            built_ts_utc=built_ts_utc,
        )
        if skip_reason:
            notes.append(skip_reason)
            continue
        if regime_results.empty:
            notes.append(f"Regime {regime_key} produced empty results after same-fold filtering. Skipped.")
            continue
        per_regime_common_folds[regime_key] = [int(fid) for fid in regime_common]
        results_frames.append(regime_results)
        fold_frames.append(regime_fold)

    if not results_frames:
        raise ValueError("No regime-conditioned results were produced for requested regimes.")

    results_df = pd.concat(results_frames, ignore_index=True)
    fold_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()

    for col in (
        "candidate_variant",
        "regime_family",
        "regime_name",
        "winner_vs_dummy",
        "comparison_variant",
        "winner_vs_comparison_variant",
    ):
        results_df[col] = results_df[col].astype("string")
    for col in (
        "model_mse",
        "dummy_mse",
        "improvement_vs_dummy",
        "improvement_vs_comparison_variant",
    ):
        results_df[col] = pd.to_numeric(results_df[col], errors="coerce")
    results_df["n_obs"] = pd.to_numeric(results_df["n_obs"], errors="coerce").astype("int64")
    results_df["n_folds_used"] = pd.to_numeric(results_df["n_folds_used"], errors="coerce").astype("int64")
    assert_schema(results_df, RESULTS_SCHEMA)

    if not fold_df.empty:
        for col in (
            "candidate_variant",
            "regime_family",
            "regime_name",
            "winner_vs_dummy",
            "comparison_variant",
            "winner_vs_comparison_variant",
        ):
            fold_df[col] = fold_df[col].astype("string")
        for col in (
            "model_mse",
            "dummy_mse",
            "improvement_vs_dummy",
            "improvement_vs_comparison_variant",
        ):
            fold_df[col] = pd.to_numeric(fold_df[col], errors="coerce")
        fold_df["fold_id"] = pd.to_numeric(fold_df["fold_id"], errors="coerce").astype("int64")
        fold_df["n_obs"] = pd.to_numeric(fold_df["n_obs"], errors="coerce").astype("int64")
        assert_schema(fold_df, FOLD_SCHEMA)

    results_path = write_parquet(
        results_df,
        target_dir / "h20_regime_conditioned_results.parquet",
        schema_name=RESULTS_SCHEMA.name,
        run_id=run_id,
    )
    fold_metrics_path = write_parquet(
        fold_df,
        target_dir / "h20_regime_conditioned_fold_metrics.parquet",
        schema_name=FOLD_SCHEMA.name,
        run_id=run_id,
    )

    beats_dummy: dict[str, list[str]] = {}
    loses_dummy: dict[str, list[str]] = {}
    best_regime_by_variant: dict[str, dict[str, Any]] = {}
    for variant, group in results_df.groupby("candidate_variant", dropna=False):
        variant_str = str(variant)
        beats = group[group["winner_vs_dummy"].astype(str) == "variant"]
        loses = group[group["winner_vs_dummy"].astype(str) == "dummy"]
        beats_dummy[variant_str] = sorted(
            [f"{r.regime_family}:{r.regime_name}" for r in beats.itertuples(index=False)]
        )
        loses_dummy[variant_str] = sorted(
            [f"{r.regime_family}:{r.regime_name}" for r in loses.itertuples(index=False)]
        )
        best_row = group.sort_values(
            ["improvement_vs_dummy", "improvement_vs_comparison_variant", "n_obs"],
            ascending=[False, False, False],
        ).iloc[0]
        best_regime_by_variant[variant_str] = {
            "regime_family": str(best_row["regime_family"]),
            "regime_name": str(best_row["regime_name"]),
            "improvement_vs_dummy": float(best_row["improvement_vs_dummy"]),
            "winner_vs_dummy": str(best_row["winner_vs_dummy"]),
            "winner_vs_comparison_variant": str(best_row["winner_vs_comparison_variant"]),
            "n_obs": int(best_row["n_obs"]),
            "n_folds_used": int(best_row["n_folds_used"]),
        }

    best_variant_by_regime: dict[str, dict[str, Any]] = {}
    for (family, name), group in results_df.groupby(["regime_family", "regime_name"], dropna=False):
        row = group.sort_values(
            ["improvement_vs_dummy", "improvement_vs_comparison_variant", "n_obs"],
            ascending=[False, False, False],
        ).iloc[0]
        key = f"{family}:{name}"
        best_variant_by_regime[key] = {
            "candidate_variant": str(row["candidate_variant"]),
            "improvement_vs_dummy": float(row["improvement_vs_dummy"]),
            "winner_vs_dummy": str(row["winner_vs_dummy"]),
            "winner_vs_comparison_variant": str(row["winner_vs_comparison_variant"]),
            "n_obs": int(row["n_obs"]),
            "n_folds_used": int(row["n_folds_used"]),
        }

    strong_rows = results_df[
        (results_df["winner_vs_dummy"].astype(str) == "variant")
        & (results_df["winner_vs_comparison_variant"].astype(str) == "variant")
        & (pd.to_numeric(results_df["n_folds_used"], errors="coerce").astype(int) >= 3)
    ]
    if strong_rows.empty:
        recommendation = "regime_conditioning_still_fragile_continue_improve_features_or_labels"
    elif len(strong_rows) == 1:
        recommendation = "narrow_regime_edge_retest_before_escalation"
    else:
        recommendation = "regime_niche_detected_focus_refinement_on_winning_regimes"

    summary_payload = {
        "module_version": MODULE_VERSION,
        "candidate_variants_evaluated": sorted(
            results_df["candidate_variant"].astype(str).unique().tolist()
        ),
        "regimes_evaluated": sorted(
            [f"{r.regime_family}:{r.regime_name}" for r in results_df.itertuples(index=False)]
        ),
        "regimes_requested": [f"{family}:{name}" for family, name in selected_regimes],
        "regimes_where_variant_beats_dummy": beats_dummy,
        "regimes_where_variant_loses_to_dummy": loses_dummy,
        "best_regime_by_variant": best_regime_by_variant,
        "best_variant_by_regime": best_variant_by_regime,
        "recommendation": recommendation,
        "label_name": label_name,
        "target_type": target_type,
        "horizon_days": int(horizon_days),
        "comparison_policy": {
            "same_common_folds_only": True,
            "global_common_fold_ids": [int(fid) for fid in common_folds],
            "per_regime_common_fold_ids": per_regime_common_folds,
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
    summary_path = target_dir / "h20_regime_conditioned_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "h20_regime_conditioned_refinement_completed",
        run_id=run_id,
        n_results_rows=int(len(results_df)),
        n_regimes=int(results_df[["regime_family", "regime_name"]].drop_duplicates().shape[0]),
        output_path=str(results_path),
    )

    return H20RegimeConditionedRefinementResult(
        results_path=results_path,
        summary_path=summary_path,
        fold_metrics_path=fold_metrics_path,
        n_rows=int(len(results_df)),
        config_hash=config_hash,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run H20 regime-conditioned refinement for selected variants."
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
    parser.add_argument(
        "--target-regimes",
        type=str,
        default=",".join(DEFAULT_TARGET_REGIMES),
        help="Comma-separated regime targets with format family:name.",
    )
    parser.add_argument("--comparison-variant", type=str, default=BASELINE_VARIANT)
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
    result = run_h20_regime_conditioned_refinement(
        model_dataset_path=args.model_dataset_path,
        purged_cv_folds_path=args.purged_cv_folds_path,
        output_dir=args.output_dir,
        label_name=args.label_name,
        target_type=args.target_type,
        horizon_days=int(args.horizon_days),
        candidate_variants=_parse_csv_texts(args.candidate_variants) or DEFAULT_CANDIDATE_VARIANTS,
        target_regimes=_parse_csv_texts(args.target_regimes) or DEFAULT_TARGET_REGIMES,
        comparison_variant=args.comparison_variant,
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
