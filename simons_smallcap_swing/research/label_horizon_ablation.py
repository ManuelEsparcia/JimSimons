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


MODULE_VERSION = "label_horizon_ablation_mvp_v1"
DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 20)
FEATURE_FAMILY_BEST_FROM_ABLATION = "best_from_feature_ablation"

TASK_CONFIG: dict[str, dict[str, str]] = {
    "regression": {
        "target_type": "continuous_forward_return",
        "label_prefix": "fwd_ret",
        "model_mode": MODE_RIDGE_CV,
        "dummy_mode": MODE_DUMMY_REGRESSOR_CV,
        "model_name": "ridge_cv",
        "dummy_name": "dummy_regressor_cv",
        "primary_metric": "mse",
    },
    "classification": {
        "target_type": "binary_direction",
        "label_prefix": "fwd_dir_up",
        "model_mode": MODE_LOGISTIC_CV,
        "dummy_mode": MODE_DUMMY_CLASSIFIER_CV,
        "model_name": "logistic_cv",
        "dummy_name": "dummy_classifier_cv",
        "primary_metric": "log_loss",
    },
}

SUPPORTED_FEATURE_FAMILIES: tuple[str, ...] = (
    "price_momentum",
    "vol_liquidity",
    "market_context",
    "fundamentals",
    "all_features",
    "price_plus_market",
    "price_plus_fundamentals",
    FEATURE_FAMILY_BEST_FROM_ABLATION,
)

RESULTS_SCHEMA = DataSchema(
    name="label_horizon_ablation_results_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("task_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("target_type", "string", nullable=False),
        ColumnSpec("horizon_days", "int64", nullable=False),
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

FOLD_SCHEMA = DataSchema(
    name="label_horizon_ablation_fold_metrics_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("task_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("target_type", "string", nullable=False),
        ColumnSpec("horizon_days", "int64", nullable=False),
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

FEATURES_EXCLUDED_COLUMNS: set[str] = {
    "date",
    "instrument_id",
    "ticker",
    "run_id",
    "config_hash",
    "built_ts_utc",
}


@dataclass(frozen=True)
class LabelHorizonAblationResult:
    results_path: Path
    summary_path: Path
    fold_metrics_path: Path
    n_rows: int
    config_hash: str


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' has invalid dates.")
    return parsed.dt.normalize()


def _parse_csv_ints(text: str | None) -> tuple[int, ...]:
    if not text:
        return ()
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _parse_csv_floats(text: str | None) -> tuple[float, ...]:
    if not text:
        return ()
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def _normalize_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    values = tuple(sorted({int(h) for h in horizons}))
    if not values:
        raise ValueError("horizons cannot be empty.")
    if any(h <= 0 for h in values):
        raise ValueError(f"horizons must be positive. Received: {values}")
    return values


def _normalize_float_grid(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    values = tuple(sorted({float(v) for v in values}))
    if not values:
        raise ValueError(f"{name} cannot be empty.")
    if any(v <= 0 for v in values):
        raise ValueError(f"{name} must contain positive values. Received: {values}")
    return values


def _cfg_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _label_name_for(task_name: str, horizon_days: int) -> str:
    return f"{TASK_CONFIG[task_name]['label_prefix']}_{int(horizon_days)}d"


def _winner(improvement: float, tolerance: float) -> str:
    if improvement > tolerance:
        return "model"
    if improvement < -tolerance:
        return "dummy"
    return "tie"


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], *, name: str) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _load_labels(path: Path) -> pd.DataFrame:
    frame = read_parquet(path).copy()
    _require_columns(
        frame,
        ("date", "instrument_id", "ticker", "horizon_days", "label_name", "label_value"),
        name="labels_forward",
    )
    frame["date"] = _normalize_date(frame["date"], column="date")
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str)
    frame["horizon_days"] = pd.to_numeric(frame["horizon_days"], errors="coerce").astype("Int64")
    frame["label_name"] = frame["label_name"].astype(str)
    frame["label_value"] = pd.to_numeric(frame["label_value"], errors="coerce")
    if frame["horizon_days"].isna().any():
        raise ValueError("labels_forward has invalid horizon_days.")
    if frame["label_value"].isna().any():
        raise ValueError("labels_forward has invalid label_value.")
    frame["horizon_days"] = frame["horizon_days"].astype("int64")
    if frame.duplicated(["date", "instrument_id", "horizon_days", "label_name"]).any():
        raise ValueError("labels_forward has duplicate (date, instrument_id, horizon_days, label_name).")
    return frame


def _load_features(path: Path) -> tuple[pd.DataFrame, tuple[str, ...]]:
    frame = read_parquet(path).copy()
    _require_columns(frame, ("date", "instrument_id", "ticker"), name="features_matrix")
    frame["date"] = _normalize_date(frame["date"], column="date")
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str)
    if frame.duplicated(["date", "instrument_id"]).any():
        raise ValueError("features_matrix has duplicate (date, instrument_id) rows.")
    feature_cols = tuple(
        sorted(
            [
                col
                for col in frame.columns
                if col not in FEATURES_EXCLUDED_COLUMNS and is_numeric_dtype(frame[col])
            ]
        )
    )
    if not feature_cols:
        raise ValueError("No numeric features found in features_matrix.")
    return frame, feature_cols


def _load_folds(path: Path) -> pd.DataFrame:
    frame = read_parquet(path).copy()
    _require_columns(
        frame,
        ("fold_id", "date", "instrument_id", "horizon_days", "label_name", "split_role"),
        name="purged_cv_folds",
    )
    frame["date"] = _normalize_date(frame["date"], column="date")
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["horizon_days"] = pd.to_numeric(frame["horizon_days"], errors="coerce").astype("Int64")
    frame["label_name"] = frame["label_name"].astype(str)
    frame["split_role"] = frame["split_role"].astype(str)
    if frame["horizon_days"].isna().any():
        raise ValueError("purged_cv_folds has invalid horizon_days.")
    frame["horizon_days"] = frame["horizon_days"].astype("int64")
    if frame.duplicated(["fold_id", "date", "instrument_id", "horizon_days", "label_name"]).any():
        raise ValueError("purged_cv_folds has duplicate PK rows.")
    return frame


def _build_feature_families(feature_cols: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    cols = list(feature_cols)

    def _pick(predicate: Any) -> tuple[str, ...]:
        return tuple(sorted([c for c in cols if predicate(c)]))

    price_momentum = _pick(lambda c: c.startswith("ret_") or "momentum" in c)
    vol_liquidity = _pick(
        lambda c: c.startswith("vol_") or "volume" in c or "turnover" in c or c.startswith("abs_ret_")
    )
    market_context = _pick(lambda c: c.startswith("mkt_"))
    fundamentals = _pick(lambda c: ("asset" in c) or ("share" in c) or ("revenue" in c) or ("income" in c))
    return {
        "price_momentum": price_momentum,
        "vol_liquidity": vol_liquidity,
        "market_context": market_context,
        "fundamentals": fundamentals,
        "all_features": tuple(sorted(cols)),
        "price_plus_market": tuple(sorted(set(price_momentum) | set(market_context))),
        "price_plus_fundamentals": tuple(sorted(set(price_momentum) | set(fundamentals))),
    }


def _resolve_family_by_task(
    *,
    feature_family: str,
    families: dict[str, tuple[str, ...]],
    feature_ablation_summary_path: Path | None,
) -> dict[str, str]:
    selected = str(feature_family).strip()
    if selected not in SUPPORTED_FEATURE_FAMILIES:
        raise ValueError(f"Unsupported feature_family '{selected}'. Allowed: {SUPPORTED_FEATURE_FAMILIES}")
    if selected != FEATURE_FAMILY_BEST_FROM_ABLATION:
        if selected not in families or not families[selected]:
            raise ValueError(f"Feature family '{selected}' is unavailable or empty.")
        return {"regression": selected, "classification": selected}

    if feature_ablation_summary_path is None or not feature_ablation_summary_path.exists():
        raise FileNotFoundError("feature_ablation_summary.json is required for best_from_feature_ablation mode.")
    payload = json.loads(feature_ablation_summary_path.read_text(encoding="utf-8"))
    best_by_task = payload.get("best_family_by_task", {})
    if not isinstance(best_by_task, dict):
        raise ValueError("feature_ablation_summary.json missing best_family_by_task.")
    out: dict[str, str] = {}
    for task_name in TASK_CONFIG:
        family = str(best_by_task.get(task_name, "")).strip()
        if family not in families or not families[family]:
            raise ValueError(f"Resolved family '{family}' for task '{task_name}' is unavailable/empty.")
        out[task_name] = family
    return out


def _build_model_dataset(
    *,
    labels_subset: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: tuple[str, ...],
    target_type: str,
) -> pd.DataFrame:
    joined = labels_subset.merge(
        features[["date", "instrument_id", "ticker", *feature_cols]],
        on=["date", "instrument_id"],
        how="inner",
        suffixes=("_label", ""),
    )
    if joined.empty:
        raise ValueError("No rows after labels/features join for selected combination.")
    if joined.duplicated(["date", "instrument_id", "horizon_days", "label_name"]).any():
        raise ValueError("Joined dataset has duplicate PK rows.")
    joined["target_value"] = pd.to_numeric(joined["label_value"], errors="coerce")
    if joined["target_value"].isna().any():
        raise ValueError("Joined dataset has invalid target_value.")
    joined["target_type"] = str(target_type)
    joined["ticker"] = joined["ticker_label"].astype(str)
    return joined[
        ["date", "instrument_id", "ticker", "horizon_days", "label_name", "target_value", "target_type", *feature_cols]
    ].copy()


def _completed_fold_metrics(path: Path) -> pd.DataFrame:
    frame = read_parquet(path).copy()
    _require_columns(frame, ("fold_id", "status", "valid_primary_metric"), name="cv_fold_metrics")
    out = frame[frame["status"].astype(str) == "completed"].copy()
    if out.empty:
        raise ValueError(f"No completed folds in {path}")
    out["fold_id"] = pd.to_numeric(out["fold_id"], errors="coerce").astype("int64")
    out["valid_primary_metric"] = pd.to_numeric(out["valid_primary_metric"], errors="coerce")
    out["n_features_used"] = pd.to_numeric(out.get("n_features_used", 0), errors="coerce").fillna(0).astype("int64")
    return out


def run_label_horizon_ablation(
    *,
    labels_forward_path: str | Path | None = None,
    features_matrix_path: str | Path | None = None,
    purged_cv_folds_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    feature_family: str = "all_features",
    feature_ablation_summary_path: str | Path | None = None,
    alpha_grid: Iterable[float] = (0.1, 1.0, 10.0),
    c_grid: Iterable[float] = (0.1, 1.0, 10.0),
    tie_tolerance: float = 1e-12,
    fail_on_invalid_fold: bool = False,
    run_id: str = MODULE_VERSION,
) -> LabelHorizonAblationResult:
    logger = get_logger("research.label_horizon_ablation")
    if float(tie_tolerance) < 0:
        raise ValueError("tie_tolerance must be >= 0.")

    base = data_dir()
    labels_source = Path(labels_forward_path).expanduser().resolve() if labels_forward_path else (base / "labels" / "labels_forward.parquet")
    features_source = Path(features_matrix_path).expanduser().resolve() if features_matrix_path else (base / "features" / "features_matrix.parquet")
    folds_source = Path(purged_cv_folds_path).expanduser().resolve() if purged_cv_folds_path else (base / "labels" / "purged_cv_folds.parquet")
    summary_source = Path(feature_ablation_summary_path).expanduser().resolve() if feature_ablation_summary_path else (base / "research" / "feature_ablation_summary.json")
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (base / "research")
    target_dir.mkdir(parents=True, exist_ok=True)

    selected_horizons = _normalize_horizons(horizons)
    alpha_values = _normalize_float_grid(alpha_grid, name="alpha_grid")
    c_values = _normalize_float_grid(c_grid, name="c_grid")

    labels = _load_labels(labels_source)
    features, feature_cols = _load_features(features_source)
    _ = _load_folds(folds_source)  # loaded for schema-level validation
    families = _build_feature_families(feature_cols)
    family_by_task = _resolve_family_by_task(
        feature_family=feature_family,
        families=families,
        feature_ablation_summary_path=summary_source if feature_family == FEATURE_FAMILY_BEST_FROM_ABLATION else None,
    )

    built_ts = datetime.now(UTC).isoformat()
    config_hash = _cfg_hash(
        {
            "module_version": MODULE_VERSION,
            "horizons": list(selected_horizons),
            "feature_family_request": feature_family,
            "feature_family_used_by_task": family_by_task,
            "labels_forward_path": str(labels_source),
            "features_matrix_path": str(features_source),
            "purged_cv_folds_path": str(folds_source),
            "feature_ablation_summary_path": str(summary_source) if feature_family == FEATURE_FAMILY_BEST_FROM_ABLATION else None,
            "alpha_grid": list(alpha_values),
            "c_grid": list(c_values),
            "tie_tolerance": float(tie_tolerance),
            "fail_on_invalid_fold": bool(fail_on_invalid_fold),
            "run_id": run_id,
        }
    )

    tmp_dir = target_dir / f"_label_horizon_ablation_tmp_{run_id}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    notes: list[str] = []

    for horizon in selected_horizons:
        for task_name, task_cfg in TASK_CONFIG.items():
            label_name = _label_name_for(task_name, horizon)
            task_target_type = task_cfg["target_type"]
            family_name = family_by_task[task_name]
            family_cols = families[family_name]

            label_subset = labels[(labels["label_name"] == label_name) & (labels["horizon_days"] == int(horizon))].copy()
            if label_subset.empty:
                missing.append(
                    {
                        "task_name": task_name,
                        "label_name": label_name,
                        "target_type": task_target_type,
                        "horizon_days": int(horizon),
                        "reason": "missing_in_labels_forward",
                    }
                )
                notes.append(f"WARN: missing label/horizon in labels_forward for {label_name}.")
                continue

            dataset = _build_model_dataset(
                labels_subset=label_subset,
                features=features,
                feature_cols=family_cols,
                target_type=task_target_type,
            )
            dataset_path = tmp_dir / f"{task_name}_{label_name}_{family_name}_dataset.parquet"
            dataset.to_parquet(dataset_path, index=False)

            dummy_run = run_cross_validated_baseline(
                mode=task_cfg["dummy_mode"],
                model_dataset_path=dataset_path,
                purged_cv_folds_path=folds_source,
                output_dir=tmp_dir / f"{task_name}_{label_name}_{family_name}_dummy",
                label_name=label_name,
                horizon_days=int(horizon),
                alpha_grid=alpha_values,
                c_grid=c_values,
                fail_on_invalid_fold=bool(fail_on_invalid_fold),
                write_predictions=False,
                run_id=f"{run_id}_{task_name}_{label_name}_{task_cfg['dummy_name']}",
            )
            dummy = _completed_fold_metrics(dummy_run.fold_metrics_path)[["fold_id", "valid_primary_metric"]].rename(
                columns={"valid_primary_metric": "dummy_valid_primary_metric"}
            )

            model_run = run_cross_validated_baseline(
                mode=task_cfg["model_mode"],
                model_dataset_path=dataset_path,
                purged_cv_folds_path=folds_source,
                output_dir=tmp_dir / f"{task_name}_{label_name}_{family_name}_{task_cfg['model_name']}",
                label_name=label_name,
                horizon_days=int(horizon),
                alpha_grid=alpha_values,
                c_grid=c_values,
                fail_on_invalid_fold=bool(fail_on_invalid_fold),
                write_predictions=False,
                run_id=f"{run_id}_{task_name}_{label_name}_{family_name}_{task_cfg['model_name']}",
            )
            model = _completed_fold_metrics(model_run.fold_metrics_path)[["fold_id", "valid_primary_metric", "n_features_used"]].rename(
                columns={"valid_primary_metric": "model_valid_primary_metric"}
            )

            merged = model.merge(dummy, on="fold_id", how="inner")
            if merged.empty:
                missing.append(
                    {
                        "task_name": task_name,
                        "label_name": label_name,
                        "target_type": task_target_type,
                        "horizon_days": int(horizon),
                        "reason": "no_common_completed_folds",
                    }
                )
                notes.append(f"WARN: no common completed folds for {label_name}.")
                continue

            merged["improvement_vs_dummy"] = merged["dummy_valid_primary_metric"] - merged["model_valid_primary_metric"]
            improvement = float(merged["improvement_vs_dummy"].mean())
            result_rows.append(
                {
                    "task_name": task_name,
                    "label_name": label_name,
                    "target_type": task_target_type,
                    "horizon_days": int(horizon),
                    "feature_family": family_name,
                    "model_name": task_cfg["model_name"],
                    "primary_metric": task_cfg["primary_metric"],
                    "mean_valid_primary_metric": float(merged["model_valid_primary_metric"].mean()),
                    "median_valid_primary_metric": float(merged["model_valid_primary_metric"].median()),
                    "std_valid_primary_metric": float(merged["model_valid_primary_metric"].std(ddof=0)),
                    "n_folds": int(len(merged)),
                    "n_features_used": int(round(float(merged["n_features_used"].median()))),
                    "improvement_vs_dummy": improvement,
                    "winner_vs_dummy": _winner(improvement, float(tie_tolerance)),
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "built_ts_utc": built_ts,
                }
            )

            for row in merged.itertuples(index=False):
                fold_rows.append(
                    {
                        "task_name": task_name,
                        "label_name": label_name,
                        "target_type": task_target_type,
                        "horizon_days": int(horizon),
                        "feature_family": family_name,
                        "model_name": task_cfg["model_name"],
                        "dummy_model_name": task_cfg["dummy_name"],
                        "primary_metric": task_cfg["primary_metric"],
                        "fold_id": int(row.fold_id),
                        "model_valid_primary_metric": float(row.model_valid_primary_metric),
                        "dummy_valid_primary_metric": float(row.dummy_valid_primary_metric),
                        "improvement_vs_dummy": float(row.improvement_vs_dummy),
                        "n_features_used": int(row.n_features_used),
                        "run_id": run_id,
                        "config_hash": config_hash,
                        "built_ts_utc": built_ts,
                    }
                )

    if not result_rows:
        raise ValueError(f"No ablation rows produced. missing_label_combinations={missing}")

    results = pd.DataFrame(result_rows).sort_values(
        ["task_name", "horizon_days", "mean_valid_primary_metric", "label_name"]
    ).reset_index(drop=True)
    folds = pd.DataFrame(fold_rows).sort_values(
        ["task_name", "horizon_days", "label_name", "fold_id"]
    ).reset_index(drop=True)

    for col in ("task_name", "label_name", "target_type", "feature_family", "model_name", "primary_metric", "winner_vs_dummy"):
        results[col] = results[col].astype("string")
    for col in ("mean_valid_primary_metric", "median_valid_primary_metric", "std_valid_primary_metric", "improvement_vs_dummy"):
        results[col] = pd.to_numeric(results[col], errors="coerce")
    results["horizon_days"] = pd.to_numeric(results["horizon_days"], errors="coerce").astype("int64")
    results["n_folds"] = pd.to_numeric(results["n_folds"], errors="coerce").astype("int64")
    results["n_features_used"] = pd.to_numeric(results["n_features_used"], errors="coerce").astype("int64")
    assert_schema(results, RESULTS_SCHEMA)

    for col in ("task_name", "label_name", "target_type", "feature_family", "model_name", "dummy_model_name", "primary_metric"):
        folds[col] = folds[col].astype("string")
    for col in ("model_valid_primary_metric", "dummy_valid_primary_metric", "improvement_vs_dummy"):
        folds[col] = pd.to_numeric(folds[col], errors="coerce")
    folds["horizon_days"] = pd.to_numeric(folds["horizon_days"], errors="coerce").astype("int64")
    folds["fold_id"] = pd.to_numeric(folds["fold_id"], errors="coerce").astype("int64")
    folds["n_features_used"] = pd.to_numeric(folds["n_features_used"], errors="coerce").astype("int64")
    assert_schema(folds, FOLD_SCHEMA)

    results_path = write_parquet(
        results,
        target_dir / "label_horizon_ablation_results.parquet",
        schema_name=RESULTS_SCHEMA.name,
        run_id=run_id,
    )
    folds_path = write_parquet(
        folds,
        target_dir / "label_horizon_ablation_fold_metrics.parquet",
        schema_name=FOLD_SCHEMA.name,
        run_id=run_id,
    )

    best_label_by_task: dict[str, str] = {}
    best_horizon_by_task: dict[str, int] = {}
    beating: dict[str, list[str]] = {}
    not_beating: dict[str, list[str]] = {}
    for task_name, group in results.groupby("task_name", sort=True):
        best = group.sort_values(["mean_valid_primary_metric", "horizon_days", "label_name"]).iloc[0]
        best_label_by_task[str(task_name)] = str(best["label_name"])
        best_horizon_by_task[str(task_name)] = int(best["horizon_days"])
        beating[str(task_name)] = sorted(
            [
                f"{label}@{int(h)}"
                for label, h in group.loc[group["winner_vs_dummy"] == "model", ["label_name", "horizon_days"]].itertuples(index=False)
            ]
        )
        not_beating[str(task_name)] = sorted(
            [
                f"{label}@{int(h)}"
                for label, h in group.loc[group["winner_vs_dummy"] != "model", ["label_name", "horizon_days"]].itertuples(index=False)
            ]
        )

    summary = {
        "module_version": MODULE_VERSION,
        "run_id": run_id,
        "config_hash": config_hash,
        "built_ts_utc": built_ts,
        "horizons_evaluated": [int(h) for h in selected_horizons],
        "tasks_evaluated": sorted(results["task_name"].astype(str).unique().tolist()),
        "feature_family_request": feature_family,
        "feature_family_used": family_by_task,
        "best_label_by_task": best_label_by_task,
        "best_horizon_by_task": best_horizon_by_task,
        "combinations_beating_dummy": beating,
        "combinations_not_beating_dummy": not_beating,
        "missing_label_combinations": missing,
        "n_results_rows": int(len(results)),
        "n_fold_rows": int(len(folds)),
        "input_paths": {
            "labels_forward": str(labels_source),
            "features_matrix": str(features_source),
            "purged_cv_folds": str(folds_source),
            "feature_ablation_summary": str(summary_source) if feature_family == FEATURE_FAMILY_BEST_FROM_ABLATION else None,
        },
        "output_paths": {
            "label_horizon_ablation_results": str(results_path),
            "label_horizon_ablation_fold_metrics": str(folds_path),
        },
        "notes": notes,
    }
    summary_path = target_dir / "label_horizon_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "label_horizon_ablation_completed",
        run_id=run_id,
        n_results_rows=int(len(results)),
        n_fold_rows=int(len(folds)),
        summary_path=str(summary_path),
    )
    return LabelHorizonAblationResult(
        results_path=results_path,
        summary_path=summary_path,
        fold_metrics_path=folds_path,
        n_rows=int(len(results)),
        config_hash=config_hash,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MVP label/horizon ablation against dummy baselines.")
    parser.add_argument("--labels-forward-path", default=None)
    parser.add_argument("--features-matrix-path", default=None)
    parser.add_argument("--purged-cv-folds-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--horizons", default="1,5,20")
    parser.add_argument("--feature-family", default="all_features")
    parser.add_argument("--feature-ablation-summary-path", default=None)
    parser.add_argument("--ridge-alphas", default="0.1,1.0,10.0")
    parser.add_argument("--logistic-cs", default="0.1,1.0,10.0")
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    parser.add_argument("--fail-on-invalid-fold", action="store_true")
    parser.add_argument("--run-id", default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_label_horizon_ablation(
        labels_forward_path=args.labels_forward_path,
        features_matrix_path=args.features_matrix_path,
        purged_cv_folds_path=args.purged_cv_folds_path,
        output_dir=args.output_dir,
        horizons=_parse_csv_ints(args.horizons) or DEFAULT_HORIZONS,
        feature_family=args.feature_family,
        feature_ablation_summary_path=args.feature_ablation_summary_path,
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
