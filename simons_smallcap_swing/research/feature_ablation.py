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

# Allow direct script execution:
# `python simons_smallcap_swing/research/feature_ablation.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.baselines.cross_validated_baselines import (
    MODE_DUMMY_CLASSIFIER_CV,
    MODE_DUMMY_REGRESSOR_CV,
    MODE_LOGISTIC_CV,
    MODE_RIDGE_CV,
    run_cross_validated_baseline,
)
from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "feature_ablation_mvp_v1"
PRIMARY_METRIC_BY_TASK = {"regression": "mse", "classification": "log_loss"}

TASK_CONFIG = {
    "regression": {
        "label_name": "fwd_ret_5d",
        "target_type": "continuous_forward_return",
        "model_mode": MODE_RIDGE_CV,
        "dummy_mode": MODE_DUMMY_REGRESSOR_CV,
        "model_name": "ridge_cv",
        "dummy_name": "dummy_regressor_cv",
    },
    "classification": {
        "label_name": "fwd_dir_up_5d",
        "target_type": "binary_direction",
        "model_mode": MODE_LOGISTIC_CV,
        "dummy_mode": MODE_DUMMY_CLASSIFIER_CV,
        "model_name": "logistic_cv",
        "dummy_name": "dummy_classifier_cv",
    },
}

MANDATORY_FAMILIES: tuple[str, ...] = (
    "price_momentum",
    "vol_liquidity",
    "market_context",
    "fundamentals",
    "all_features",
)
OPTIONAL_FAMILIES: tuple[str, ...] = ("price_plus_market", "price_plus_fundamentals")

DATASET_REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "instrument_id",
    "ticker",
    "horizon_days",
    "label_name",
    "target_value",
    "target_type",
)
EXCLUDED_FEATURE_COLUMNS: set[str] = {
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
    name="feature_ablation_results_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("task_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("target_type", "string", nullable=False),
        ColumnSpec("feature_family", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("primary_metric", "string", nullable=False),
        ColumnSpec("mean_valid_primary_metric", "float64", nullable=False),
        ColumnSpec("median_valid_primary_metric", "float64", nullable=False),
        ColumnSpec("std_valid_primary_metric", "float64", nullable=False),
        ColumnSpec("n_folds", "int64", nullable=False),
        ColumnSpec("n_features_used", "int64", nullable=False),
        ColumnSpec("improvement_vs_dummy", "float64", nullable=False),
        ColumnSpec("winner_vs_dummy", "string", nullable=False),
    ),
    primary_key=(),
    allow_extra_columns=True,
)

FOLD_RESULTS_SCHEMA = DataSchema(
    name="feature_ablation_fold_metrics_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("task_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("target_type", "string", nullable=False),
        ColumnSpec("feature_family", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("dummy_model_name", "string", nullable=False),
        ColumnSpec("primary_metric", "string", nullable=False),
        ColumnSpec("fold_id", "int64", nullable=False),
        ColumnSpec("model_valid_primary_metric", "float64", nullable=False),
        ColumnSpec("dummy_valid_primary_metric", "float64", nullable=False),
        ColumnSpec("improvement_vs_dummy", "float64", nullable=False),
        ColumnSpec("n_features_used", "int64", nullable=False),
    ),
    primary_key=(),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class FeatureAblationResult:
    results_path: Path
    summary_path: Path
    fold_metrics_path: Path
    n_rows: int
    config_hash: str


def _parse_csv_floats(text: str | None) -> tuple[float, ...]:
    if not text:
        return ()
    out = []
    for item in text.split(","):
        token = item.strip()
        if token:
            out.append(float(token))
    return tuple(out)


def _normalize_float_grid(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    norm = tuple(sorted({float(v) for v in values}))
    if not norm:
        raise ValueError(f"{name} cannot be empty.")
    if any(v <= 0 for v in norm):
        raise ValueError(f"{name} must be positive. Received: {norm}")
    return norm


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' has invalid dates.")
    return parsed.dt.normalize()


def _cfg_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _detect_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    cols = [
        col
        for col in frame.columns
        if col not in EXCLUDED_FEATURE_COLUMNS and is_numeric_dtype(frame[col])
    ]
    if not cols:
        raise ValueError("No numeric feature columns detected in model_dataset.")
    return tuple(sorted(cols))


def _feature_columns_from_features_matrix(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"features_matrix_path does not exist: {path}")
    frame = read_parquet(path)
    cols = [
        col
        for col in frame.columns
        if col not in {"date", "instrument_id", "ticker", "run_id", "config_hash", "built_ts_utc"}
        and is_numeric_dtype(frame[col])
    ]
    return set(cols)


def _build_feature_families(
    *,
    feature_columns: tuple[str, ...],
    include_optional_families: bool,
) -> tuple[dict[str, tuple[str, ...]], list[str]]:
    notes: list[str] = []
    cols = list(feature_columns)

    def _pick(predicate: Any) -> tuple[str, ...]:
        return tuple(sorted([col for col in cols if predicate(col)]))

    price_momentum = _pick(
        lambda c: c in {"ret_1d_lag1", "ret_5d_lag1", "ret_20d_lag1", "momentum_20d_excl_1d"}
        or c.startswith("ret_")
        or "momentum" in c
    )
    vol_liquidity = _pick(
        lambda c: c in {
            "vol_5d",
            "vol_20d",
            "abs_ret_1d_lag1",
            "log_volume_lag1",
            "turnover_proxy_lag1",
            "log_dollar_volume_lag1",
        }
        or c.startswith("vol_")
        or "volume" in c
        or "turnover" in c
        or c.startswith("abs_ret_")
    )
    market_context = _pick(lambda c: c.startswith("mkt_"))
    fundamentals = _pick(
        lambda c: c in {
            "log_total_assets",
            "shares_outstanding",
            "revenue_scale_proxy",
            "net_income_scale_proxy",
        }
        or "asset" in c
        or "share" in c
        or "revenue" in c
        or "income" in c
    )

    families: dict[str, tuple[str, ...]] = {
        "price_momentum": price_momentum,
        "vol_liquidity": vol_liquidity,
        "market_context": market_context,
        "fundamentals": fundamentals,
        "all_features": tuple(sorted(cols)),
    }

    missing_mandatory = [name for name in MANDATORY_FAMILIES if len(families[name]) == 0]
    if missing_mandatory:
        raise ValueError(
            "Mandatory feature families are empty. "
            f"missing={missing_mandatory}, detected_features={sorted(cols)}"
        )

    if include_optional_families:
        price_plus_market = tuple(sorted(set(price_momentum) | set(market_context)))
        if price_plus_market:
            families["price_plus_market"] = price_plus_market
        else:
            notes.append("Skipped optional family 'price_plus_market' because it is empty.")

        price_plus_fundamentals = tuple(sorted(set(price_momentum) | set(fundamentals)))
        if price_plus_fundamentals:
            families["price_plus_fundamentals"] = price_plus_fundamentals
        else:
            notes.append("Skipped optional family 'price_plus_fundamentals' because it is empty.")

    return families, notes


def _prepare_model_dataset(path: Path) -> pd.DataFrame:
    frame = read_parquet(path).copy()
    missing = [col for col in DATASET_REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"model_dataset missing required columns: {missing}")
    frame["date"] = _normalize_date(frame["date"], column="date")
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str)
    frame["horizon_days"] = pd.to_numeric(frame["horizon_days"], errors="coerce")
    frame["target_value"] = pd.to_numeric(frame["target_value"], errors="coerce")
    frame["label_name"] = frame["label_name"].astype(str)
    frame["target_type"] = frame["target_type"].astype(str)
    if frame["horizon_days"].isna().any():
        raise ValueError("model_dataset has invalid horizon_days.")
    if frame["target_value"].isna().any():
        raise ValueError("model_dataset has invalid target_value.")
    frame["horizon_days"] = frame["horizon_days"].astype("int64")
    if frame.duplicated(["date", "instrument_id", "horizon_days", "label_name"]).any():
        raise ValueError("model_dataset has duplicate PK rows.")
    return frame


def _subset_dataset_for_family(task_df: pd.DataFrame, feature_cols: tuple[str, ...]) -> pd.DataFrame:
    keep = [
        "date",
        "instrument_id",
        "ticker",
        "horizon_days",
        "label_name",
        "target_value",
        "target_type",
    ]
    for extra_col in ("split_name",):
        if extra_col in task_df.columns:
            keep.append(extra_col)
    keep.extend(feature_cols)
    out = task_df[keep].copy()
    if out.empty:
        raise ValueError("Family-specific dataset is empty.")
    return out


def _completed_fold_metrics(path: Path) -> pd.DataFrame:
    frame = read_parquet(path).copy()
    if "status" not in frame.columns or "valid_primary_metric" not in frame.columns:
        raise ValueError(f"Invalid fold metrics schema at {path}")
    frame = frame[frame["status"].astype(str) == "completed"].copy()
    if frame.empty:
        raise ValueError(f"No completed folds at {path}")
    frame["fold_id"] = pd.to_numeric(frame["fold_id"], errors="coerce").astype("int64")
    frame["valid_primary_metric"] = pd.to_numeric(frame["valid_primary_metric"], errors="coerce")
    frame["n_features_used"] = pd.to_numeric(frame.get("n_features_used", 0), errors="coerce").fillna(0).astype("int64")
    frame = frame[frame["valid_primary_metric"].notna()].copy()
    if frame.empty:
        raise ValueError(f"No completed folds with valid metric at {path}")
    return frame


def _winner(improvement_vs_dummy: float, tie_tolerance: float) -> str:
    if abs(float(improvement_vs_dummy)) <= float(tie_tolerance):
        return "tie"
    return "model" if float(improvement_vs_dummy) > 0 else "dummy"


def run_feature_ablation(
    *,
    model_dataset_path: str | Path | None = None,
    purged_cv_folds_path: str | Path | None = None,
    features_matrix_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    regression_label_name: str = "fwd_ret_5d",
    classification_label_name: str = "fwd_dir_up_5d",
    horizon_days: int | None = 5,
    include_optional_families: bool = True,
    alpha_grid: Iterable[float] = (0.1, 1.0, 10.0),
    c_grid: Iterable[float] = (0.1, 1.0, 10.0),
    tie_tolerance: float = 1e-12,
    fail_on_invalid_fold: bool = False,
    run_id: str = MODULE_VERSION,
) -> FeatureAblationResult:
    logger = get_logger("research.feature_ablation")
    base = data_dir()
    dataset_source = (
        Path(model_dataset_path).expanduser().resolve()
        if model_dataset_path
        else (base / "datasets" / "model_dataset.parquet")
    )
    folds_source = (
        Path(purged_cv_folds_path).expanduser().resolve()
        if purged_cv_folds_path
        else (base / "labels" / "purged_cv_folds.parquet")
    )
    features_source = (
        Path(features_matrix_path).expanduser().resolve()
        if features_matrix_path
        else (base / "features" / "features_matrix.parquet")
    )
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (base / "research")
    target_dir.mkdir(parents=True, exist_ok=True)

    if not folds_source.exists():
        raise FileNotFoundError(f"purged_cv_folds not found: {folds_source}")

    if horizon_days is not None and int(horizon_days) <= 0:
        raise ValueError("horizon_days must be positive when provided.")
    selected_horizon = None if horizon_days is None else int(horizon_days)
    alpha_values = _normalize_float_grid(alpha_grid, name="alpha_grid")
    c_values = _normalize_float_grid(c_grid, name="c_grid")

    dataset = _prepare_model_dataset(dataset_source)
    model_feature_cols = _detect_feature_columns(dataset)
    matrix_feature_cols = _feature_columns_from_features_matrix(
        features_source if features_source.exists() else None
    )
    if matrix_feature_cols:
        intersect = tuple(sorted(set(model_feature_cols) & set(matrix_feature_cols)))
        if not intersect:
            raise ValueError(
                "No overlap between model_dataset numeric features and features_matrix features."
            )
        feature_cols = intersect
    else:
        feature_cols = model_feature_cols

    families, notes = _build_feature_families(
        feature_columns=feature_cols,
        include_optional_families=bool(include_optional_families),
    )

    task_overrides = {
        "regression": regression_label_name,
        "classification": classification_label_name,
    }
    built_ts_utc = datetime.now(UTC).isoformat()
    cfg_hash = _cfg_hash(
        {
            "module_version": MODULE_VERSION,
            "model_dataset_path": str(dataset_source),
            "purged_cv_folds_path": str(folds_source),
            "features_matrix_path": str(features_source) if features_source.exists() else None,
            "regression_label_name": regression_label_name,
            "classification_label_name": classification_label_name,
            "horizon_days": selected_horizon,
            "families": {k: list(v) for k, v in families.items()},
            "include_optional_families": bool(include_optional_families),
            "alpha_grid": list(alpha_values),
            "c_grid": list(c_values),
            "tie_tolerance": float(tie_tolerance),
            "fail_on_invalid_fold": bool(fail_on_invalid_fold),
        }
    )

    results_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    dummy_mean_by_task: dict[str, float] = {}
    evaluated_tasks: list[str] = []

    tmp_dir = target_dir / f"_feature_ablation_tmp_{run_id}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for task_name, cfg in TASK_CONFIG.items():
        label_name = str(task_overrides[task_name]).strip()
        if not label_name:
            notes.append(f"Skipped {task_name}: empty label_name.")
            continue

        task_df = dataset[
            (dataset["label_name"] == label_name)
            & (dataset["target_type"] == cfg["target_type"])
        ].copy()
        if selected_horizon is not None:
            task_df = task_df[task_df["horizon_days"] == selected_horizon].copy()
        if task_df.empty:
            notes.append(
                f"Skipped {task_name}: no rows for label={label_name}, "
                f"target_type={cfg['target_type']}, horizon={selected_horizon}."
            )
            continue

        evaluated_tasks.append(task_name)

        # Dummy baseline once per task (feature-agnostic by design).
        task_all_path = tmp_dir / f"{task_name}_all_features.parquet"
        _subset_dataset_for_family(task_df, tuple(feature_cols)).to_parquet(task_all_path, index=False)
        dummy_output_dir = tmp_dir / f"{task_name}_dummy"
        dummy_result = run_cross_validated_baseline(
            mode=cfg["dummy_mode"],
            model_dataset_path=task_all_path,
            purged_cv_folds_path=folds_source,
            output_dir=dummy_output_dir,
            label_name=label_name,
            horizon_days=selected_horizon,
            alpha_grid=alpha_values,
            c_grid=c_values,
            fail_on_invalid_fold=bool(fail_on_invalid_fold),
            write_predictions=False,
            run_id=f"{run_id}_{task_name}_{cfg['dummy_name']}",
        )
        dummy_completed = _completed_fold_metrics(dummy_result.fold_metrics_path)[
            ["fold_id", "valid_primary_metric"]
        ].rename(columns={"valid_primary_metric": "dummy_valid_primary_metric"})

        for family_name, family_features in families.items():
            family_dataset = _subset_dataset_for_family(task_df, family_features)
            family_dataset_path = tmp_dir / f"{task_name}_{family_name}.parquet"
            family_dataset.to_parquet(family_dataset_path, index=False)

            model_output_dir = tmp_dir / f"{task_name}_{family_name}_{cfg['model_name']}"
            model_result = run_cross_validated_baseline(
                mode=cfg["model_mode"],
                model_dataset_path=family_dataset_path,
                purged_cv_folds_path=folds_source,
                output_dir=model_output_dir,
                label_name=label_name,
                horizon_days=selected_horizon,
                alpha_grid=alpha_values,
                c_grid=c_values,
                fail_on_invalid_fold=bool(fail_on_invalid_fold),
                write_predictions=False,
                run_id=f"{run_id}_{task_name}_{family_name}_{cfg['model_name']}",
            )
            model_completed = _completed_fold_metrics(model_result.fold_metrics_path)[
                ["fold_id", "valid_primary_metric", "n_features_used"]
            ].rename(columns={"valid_primary_metric": "model_valid_primary_metric"})

            merged = model_completed.merge(dummy_completed, on="fold_id", how="inner")
            if merged.empty:
                raise ValueError(
                    f"No common completed folds for task={task_name}, family={family_name}."
                )

            merged["improvement_vs_dummy"] = (
                merged["dummy_valid_primary_metric"] - merged["model_valid_primary_metric"]
            )
            model_mean = float(merged["model_valid_primary_metric"].mean())
            model_median = float(merged["model_valid_primary_metric"].median())
            model_std = float(merged["model_valid_primary_metric"].std(ddof=0))
            dummy_mean = float(merged["dummy_valid_primary_metric"].mean())
            improvement = float(dummy_mean - model_mean)
            winner = _winner(improvement, float(tie_tolerance))
            dummy_mean_by_task[task_name] = dummy_mean

            n_features_used = int(round(float(merged["n_features_used"].median())))
            results_rows.append(
                {
                    "task_name": task_name,
                    "label_name": label_name,
                    "target_type": cfg["target_type"],
                    "feature_family": family_name,
                    "model_name": cfg["model_name"],
                    "primary_metric": PRIMARY_METRIC_BY_TASK[task_name],
                    "mean_valid_primary_metric": model_mean,
                    "median_valid_primary_metric": model_median,
                    "std_valid_primary_metric": model_std,
                    "n_folds": int(len(merged)),
                    "n_features_used": int(n_features_used),
                    "improvement_vs_dummy": improvement,
                    "winner_vs_dummy": winner,
                    "run_id": run_id,
                    "config_hash": cfg_hash,
                    "built_ts_utc": built_ts_utc,
                }
            )

            for row in merged.itertuples(index=False):
                fold_rows.append(
                    {
                        "task_name": task_name,
                        "label_name": label_name,
                        "target_type": cfg["target_type"],
                        "feature_family": family_name,
                        "model_name": cfg["model_name"],
                        "dummy_model_name": cfg["dummy_name"],
                        "primary_metric": PRIMARY_METRIC_BY_TASK[task_name],
                        "fold_id": int(row.fold_id),
                        "model_valid_primary_metric": float(row.model_valid_primary_metric),
                        "dummy_valid_primary_metric": float(row.dummy_valid_primary_metric),
                        "improvement_vs_dummy": float(row.improvement_vs_dummy),
                        "n_features_used": int(row.n_features_used),
                        "run_id": run_id,
                        "config_hash": cfg_hash,
                        "built_ts_utc": built_ts_utc,
                    }
                )

    if not results_rows:
        raise ValueError("No feature ablation results were produced.")

    results_df = pd.DataFrame(results_rows).sort_values(
        ["task_name", "mean_valid_primary_metric", "feature_family"]
    ).reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows).sort_values(
        ["task_name", "feature_family", "fold_id"]
    ).reset_index(drop=True)

    for col in ("task_name", "label_name", "target_type", "feature_family", "model_name", "primary_metric", "winner_vs_dummy"):
        results_df[col] = results_df[col].astype("string")
    for col in (
        "mean_valid_primary_metric",
        "median_valid_primary_metric",
        "std_valid_primary_metric",
        "improvement_vs_dummy",
    ):
        results_df[col] = pd.to_numeric(results_df[col], errors="coerce")
    results_df["n_folds"] = pd.to_numeric(results_df["n_folds"], errors="coerce").astype("int64")
    results_df["n_features_used"] = pd.to_numeric(results_df["n_features_used"], errors="coerce").astype("int64")
    assert_schema(results_df, RESULTS_SCHEMA)

    for col in ("task_name", "label_name", "target_type", "feature_family", "model_name", "dummy_model_name", "primary_metric"):
        fold_df[col] = fold_df[col].astype("string")
    for col in ("model_valid_primary_metric", "dummy_valid_primary_metric", "improvement_vs_dummy"):
        fold_df[col] = pd.to_numeric(fold_df[col], errors="coerce")
    fold_df["fold_id"] = pd.to_numeric(fold_df["fold_id"], errors="coerce").astype("int64")
    fold_df["n_features_used"] = pd.to_numeric(fold_df["n_features_used"], errors="coerce").astype("int64")
    assert_schema(fold_df, FOLD_RESULTS_SCHEMA)

    results_path = write_parquet(
        results_df,
        target_dir / "feature_ablation_results.parquet",
        schema_name=RESULTS_SCHEMA.name,
        run_id=run_id,
    )
    fold_metrics_path = write_parquet(
        fold_df,
        target_dir / "feature_ablation_fold_metrics.parquet",
        schema_name=FOLD_RESULTS_SCHEMA.name,
        run_id=run_id,
    )

    best_family_by_task: dict[str, str] = {}
    best_model_by_task: dict[str, str] = {}
    families_beating_dummy: dict[str, list[str]] = {}
    families_not_beating_dummy: dict[str, list[str]] = {}
    for task_name, group in results_df.groupby("task_name", sort=True):
        best = group.sort_values(["mean_valid_primary_metric", "feature_family"]).iloc[0]
        best_family_by_task[str(task_name)] = str(best["feature_family"])

        beating = sorted(group.loc[group["winner_vs_dummy"] == "model", "feature_family"].astype(str).unique().tolist())
        not_beating = sorted(
            group.loc[group["winner_vs_dummy"] != "model", "feature_family"].astype(str).unique().tolist()
        )
        families_beating_dummy[str(task_name)] = beating
        families_not_beating_dummy[str(task_name)] = not_beating

        cfg = TASK_CONFIG[str(task_name)]
        dummy_mean = float(dummy_mean_by_task[str(task_name)])
        best_metric = float(best["mean_valid_primary_metric"])
        if abs(dummy_mean - best_metric) <= float(tie_tolerance):
            best_model_by_task[str(task_name)] = "tie"
        elif best_metric < dummy_mean:
            best_model_by_task[str(task_name)] = cfg["model_name"]
        else:
            best_model_by_task[str(task_name)] = cfg["dummy_name"]

    summary_payload = {
        "module_version": MODULE_VERSION,
        "built_ts_utc": built_ts_utc,
        "run_id": run_id,
        "config_hash": cfg_hash,
        "tasks_evaluated": sorted(set(evaluated_tasks)),
        "feature_families_evaluated": sorted(results_df["feature_family"].astype(str).unique().tolist()),
        "best_family_by_task": best_family_by_task,
        "best_model_by_task": best_model_by_task,
        "families_beating_dummy": families_beating_dummy,
        "families_not_beating_dummy": families_not_beating_dummy,
        "n_results_rows": int(len(results_df)),
        "n_fold_rows": int(len(fold_df)),
        "input_paths": {
            "model_dataset": str(dataset_source),
            "purged_cv_folds": str(folds_source),
            "features_matrix": str(features_source) if features_source.exists() else None,
        },
        "output_paths": {
            "feature_ablation_results": str(results_path),
            "feature_ablation_fold_metrics": str(fold_metrics_path),
        },
        "notes": notes,
    }
    summary_path = target_dir / "feature_ablation_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "feature_ablation_completed",
        run_id=run_id,
        n_results_rows=int(len(results_df)),
        n_fold_rows=int(len(fold_df)),
        results_path=str(results_path),
        summary_path=str(summary_path),
    )
    return FeatureAblationResult(
        results_path=results_path,
        summary_path=summary_path,
        fold_metrics_path=fold_metrics_path,
        n_rows=int(len(results_df)),
        config_hash=cfg_hash,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MVP feature-family ablation against dummy baselines using purged CV."
    )
    parser.add_argument("--model-dataset-path", type=str, default=None)
    parser.add_argument("--purged-cv-folds-path", type=str, default=None)
    parser.add_argument("--features-matrix-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--regression-label-name", type=str, default="fwd_ret_5d")
    parser.add_argument("--classification-label-name", type=str, default="fwd_dir_up_5d")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--ridge-alphas", type=str, default="0.1,1.0,10.0")
    parser.add_argument("--logistic-cs", type=str, default="0.1,1.0,10.0")
    parser.add_argument("--no-optional-families", action="store_true")
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    parser.add_argument("--fail-on-invalid-fold", action="store_true")
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_feature_ablation(
        model_dataset_path=args.model_dataset_path,
        purged_cv_folds_path=args.purged_cv_folds_path,
        features_matrix_path=args.features_matrix_path,
        output_dir=args.output_dir,
        regression_label_name=args.regression_label_name,
        classification_label_name=args.classification_label_name,
        horizon_days=args.horizon_days,
        include_optional_families=not bool(args.no_optional_families),
        alpha_grid=_parse_csv_floats(args.ridge_alphas) or (0.1, 1.0, 10.0),
        c_grid=_parse_csv_floats(args.logistic_cs) or (0.1, 1.0, 10.0),
        tie_tolerance=args.tie_tolerance,
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
