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

# Allow direct script execution:
# `python simons_smallcap_swing/research/improve_best_candidate.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.baselines.cross_validated_baselines import (
    MODE_DUMMY_REGRESSOR_CV,
    MODE_RIDGE_CV,
    run_cross_validated_baseline,
)
from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "improve_best_candidate_mvp_v1"
PRIMARY_METRIC = "mse"
BASELINE_VARIANT = "baseline_all_features"

REQUIRED_DATASET_COLUMNS: tuple[str, ...] = (
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

VARIANT_BASELINE = BASELINE_VARIANT
VARIANT_NO_FUNDAMENTALS = "no_fundamentals"
VARIANT_NO_MARKET_CONTEXT = "no_market_context"
VARIANT_BEST_FAMILY_ONLY = "best_feature_family_only"
VARIANT_TARGET_CLIP = "clipped_target_if_enabled"
VARIANT_FEATURE_WINSOR = "winsorized_features_if_enabled"

RESULTS_SCHEMA = DataSchema(
    name="improve_best_candidate_results_mvp",
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
    name="improve_best_candidate_fold_metrics_mvp",
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
class ImproveBestCandidateResult:
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


def _normalize_float_grid(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    normalized = tuple(sorted({float(v) for v in values}))
    if not normalized:
        raise ValueError(f"{name} cannot be empty.")
    if any(v <= 0 for v in normalized):
        raise ValueError(f"{name} must contain positive values. Received: {normalized}")
    return normalized


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' has invalid dates.")
    return parsed.dt.normalize()


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


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], *, name: str) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _load_model_dataset(
    *,
    path: Path,
    label_name: str,
    target_type: str,
    horizon_days: int,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    frame = read_parquet(path).copy()
    _require_columns(frame, REQUIRED_DATASET_COLUMNS, name="model_dataset")

    frame["date"] = _normalize_date(frame["date"], column="date")
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str)
    frame["horizon_days"] = pd.to_numeric(frame["horizon_days"], errors="coerce").astype("Int64")
    frame["label_name"] = frame["label_name"].astype(str)
    frame["target_type"] = frame["target_type"].astype(str)
    frame["target_value"] = pd.to_numeric(frame["target_value"], errors="coerce")
    if frame["horizon_days"].isna().any():
        raise ValueError("model_dataset has invalid horizon_days values.")
    if frame["target_value"].isna().any():
        raise ValueError("model_dataset has invalid target_value values.")
    frame["horizon_days"] = frame["horizon_days"].astype("int64")

    if frame.duplicated(["date", "instrument_id", "horizon_days", "label_name"]).any():
        raise ValueError("model_dataset has duplicate PK rows.")

    filtered = frame[
        (frame["label_name"] == str(label_name))
        & (frame["target_type"] == str(target_type))
        & (frame["horizon_days"] == int(horizon_days))
    ].copy()
    if filtered.empty:
        raise ValueError(
            "No rows found for selected candidate focus: "
            f"label_name={label_name}, target_type={target_type}, horizon_days={horizon_days}."
        )

    feature_cols = tuple(
        sorted(
            [
                col
                for col in filtered.columns
                if col not in EXCLUDED_FEATURE_COLUMNS and is_numeric_dtype(filtered[col])
            ]
        )
    )
    if not feature_cols:
        raise ValueError("No numeric features detected for selected candidate focus.")

    return filtered, feature_cols


def _build_feature_families(feature_cols: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    cols = list(feature_cols)

    def _pick(predicate: Any) -> tuple[str, ...]:
        return tuple(sorted([col for col in cols if predicate(col)]))

    return {
        "price_momentum": _pick(lambda c: c.startswith("ret_") or "momentum" in c),
        "vol_liquidity": _pick(
            lambda c: c.startswith("vol_")
            or "volume" in c
            or "turnover" in c
            or c.startswith("abs_ret_")
        ),
        "market_context": _pick(lambda c: c.startswith("mkt_")),
        "fundamentals": _pick(
            lambda c: "asset" in c or "share" in c or "revenue" in c or "income" in c
        ),
        "all_features": tuple(sorted(cols)),
    }


def _resolve_best_family(
    *,
    families: dict[str, tuple[str, ...]],
    best_feature_family: str | None,
    feature_ablation_summary_path: Path,
    notes: list[str],
) -> str:
    if best_feature_family:
        selected = str(best_feature_family).strip()
        if selected not in families or len(families[selected]) == 0:
            raise ValueError(
                f"best_feature_family='{selected}' is unavailable or empty for current dataset."
            )
        return selected

    if not feature_ablation_summary_path.exists():
        notes.append(
            "feature_ablation_summary.json not found; fallback to all_features for best_feature_family_only."
        )
        return "all_features"

    payload = json.loads(feature_ablation_summary_path.read_text(encoding="utf-8"))
    best_by_task = payload.get("best_family_by_task", {})
    if not isinstance(best_by_task, dict):
        notes.append(
            "feature_ablation_summary.json missing best_family_by_task; fallback to all_features."
        )
        return "all_features"
    selected = str(best_by_task.get("regression", "all_features")).strip()
    if selected not in families or len(families[selected]) == 0:
        notes.append(
            f"best_family_by_task.regression='{selected}' unavailable; fallback to all_features."
        )
        return "all_features"
    return selected


def _variant_order(
    *,
    enable_target_clipping: bool,
    enable_feature_winsorization: bool,
) -> tuple[str, ...]:
    variants: list[str] = [
        VARIANT_BASELINE,
        VARIANT_NO_FUNDAMENTALS,
        VARIANT_NO_MARKET_CONTEXT,
        VARIANT_BEST_FAMILY_ONLY,
    ]
    if enable_target_clipping:
        variants.append(VARIANT_TARGET_CLIP)
    if enable_feature_winsorization:
        variants.append(VARIANT_FEATURE_WINSOR)
    return tuple(variants)


def _select_base_columns(frame: pd.DataFrame, feature_cols: tuple[str, ...]) -> pd.DataFrame:
    keep = list(REQUIRED_DATASET_COLUMNS)
    if "split_name" in frame.columns:
        keep.append("split_name")
    keep.extend(feature_cols)
    return frame[keep].copy()


def _materialize_variant(
    *,
    base_frame: pd.DataFrame,
    all_feature_cols: tuple[str, ...],
    families: dict[str, tuple[str, ...]],
    variant_name: str,
    best_family_name: str,
    target_clip_abs: float,
    feature_winsor_abs: float,
) -> tuple[pd.DataFrame, tuple[str, ...], str]:
    if variant_name == VARIANT_BASELINE:
        selected_features = all_feature_cols
        variant_df = _select_base_columns(base_frame, selected_features)
        return variant_df, selected_features, "all numeric features."

    if variant_name == VARIANT_NO_FUNDAMENTALS:
        selected_features = tuple(
            sorted(set(all_feature_cols) - set(families.get("fundamentals", ())))
        )
        if not selected_features:
            raise ValueError("no_fundamentals produced an empty feature set.")
        variant_df = _select_base_columns(base_frame, selected_features)
        return variant_df, selected_features, "removed fundamentals family."

    if variant_name == VARIANT_NO_MARKET_CONTEXT:
        selected_features = tuple(
            sorted(set(all_feature_cols) - set(families.get("market_context", ())))
        )
        if not selected_features:
            raise ValueError("no_market_context produced an empty feature set.")
        variant_df = _select_base_columns(base_frame, selected_features)
        return variant_df, selected_features, "removed market_context family."

    if variant_name == VARIANT_BEST_FAMILY_ONLY:
        selected_features = tuple(families[best_family_name])
        if not selected_features:
            raise ValueError("best_feature_family_only produced an empty feature set.")
        variant_df = _select_base_columns(base_frame, selected_features)
        return (
            variant_df,
            selected_features,
            f"only feature family '{best_family_name}'.",
        )

    if variant_name == VARIANT_TARGET_CLIP:
        selected_features = all_feature_cols
        variant_df = _select_base_columns(base_frame, selected_features)
        variant_df["target_value"] = variant_df["target_value"].clip(
            lower=-float(target_clip_abs),
            upper=float(target_clip_abs),
        )
        return (
            variant_df,
            selected_features,
            f"target_value clipped to [-{target_clip_abs}, {target_clip_abs}].",
        )

    if variant_name == VARIANT_FEATURE_WINSOR:
        selected_features = all_feature_cols
        variant_df = _select_base_columns(base_frame, selected_features)
        for col in selected_features:
            variant_df[col] = pd.to_numeric(variant_df[col], errors="coerce").clip(
                lower=-float(feature_winsor_abs),
                upper=float(feature_winsor_abs),
            )
        if variant_df[list(selected_features)].isna().all(axis=None):
            raise ValueError("winsorized_features_if_enabled produced all-NaN feature matrix.")
        return (
            variant_df,
            selected_features,
            f"all features clipped to [-{feature_winsor_abs}, {feature_winsor_abs}].",
        )

    raise ValueError(f"Unsupported variant_name '{variant_name}'.")


def _completed_fold_metrics(path: Path) -> pd.DataFrame:
    frame = read_parquet(path).copy()
    _require_columns(
        frame,
        ("fold_id", "status", "valid_primary_metric", "n_features_used"),
        name="cv_baseline_fold_metrics",
    )
    out = frame[frame["status"].astype(str) == "completed"].copy()
    if out.empty:
        raise ValueError(f"No completed folds at {path}")
    out["fold_id"] = pd.to_numeric(out["fold_id"], errors="coerce").astype("int64")
    out["valid_primary_metric"] = pd.to_numeric(out["valid_primary_metric"], errors="coerce")
    out["n_features_used"] = (
        pd.to_numeric(out["n_features_used"], errors="coerce").fillna(0).astype("int64")
    )
    out = out[out["valid_primary_metric"].notna()].copy()
    if out.empty:
        raise ValueError(f"No completed folds with valid metric at {path}")
    return out


def run_improve_best_candidate(
    *,
    model_dataset_path: str | Path | None = None,
    purged_cv_folds_path: str | Path | None = None,
    feature_ablation_summary_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    label_name: str = "fwd_ret_20d",
    target_type: str = "continuous_forward_return",
    horizon_days: int = 20,
    best_feature_family: str | None = None,
    alpha_grid: Iterable[float] = (0.1, 1.0, 10.0),
    tie_tolerance: float = 1e-12,
    enable_target_clipping: bool = False,
    target_clip_abs: float = 0.20,
    enable_feature_winsorization: bool = False,
    feature_winsor_abs: float = 5.0,
    fail_on_invalid_fold: bool = False,
    run_id: str = MODULE_VERSION,
) -> ImproveBestCandidateResult:
    logger = get_logger("research.improve_best_candidate")
    if int(horizon_days) <= 0:
        raise ValueError("horizon_days must be positive.")
    if float(tie_tolerance) < 0:
        raise ValueError("tie_tolerance must be >= 0.")
    if float(target_clip_abs) <= 0:
        raise ValueError("target_clip_abs must be > 0.")
    if float(feature_winsor_abs) <= 0:
        raise ValueError("feature_winsor_abs must be > 0.")

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
    feature_ablation_summary_source = (
        Path(feature_ablation_summary_path).expanduser().resolve()
        if feature_ablation_summary_path
        else (base / "research" / "feature_ablation_summary.json")
    )
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (base / "research")
    target_dir.mkdir(parents=True, exist_ok=True)

    alpha_values = _normalize_float_grid(alpha_grid, name="alpha_grid")
    candidate_df, all_feature_cols = _load_model_dataset(
        path=dataset_source,
        label_name=label_name,
        target_type=target_type,
        horizon_days=int(horizon_days),
    )
    families = _build_feature_families(all_feature_cols)
    notes: list[str] = []
    best_family_name = _resolve_best_family(
        families=families,
        best_feature_family=best_feature_family,
        feature_ablation_summary_path=feature_ablation_summary_source,
        notes=notes,
    )
    variants = _variant_order(
        enable_target_clipping=bool(enable_target_clipping),
        enable_feature_winsorization=bool(enable_feature_winsorization),
    )
    built_ts_utc = datetime.now(UTC).isoformat()
    config_hash = _cfg_hash(
        {
            "module_version": MODULE_VERSION,
            "model_dataset_path": str(dataset_source),
            "purged_cv_folds_path": str(folds_source),
            "feature_ablation_summary_path": str(feature_ablation_summary_source),
            "label_name": label_name,
            "target_type": target_type,
            "horizon_days": int(horizon_days),
            "best_feature_family_resolved": best_family_name,
            "variants": list(variants),
            "alpha_grid": list(alpha_values),
            "tie_tolerance": float(tie_tolerance),
            "enable_target_clipping": bool(enable_target_clipping),
            "target_clip_abs": float(target_clip_abs),
            "enable_feature_winsorization": bool(enable_feature_winsorization),
            "feature_winsor_abs": float(feature_winsor_abs),
            "fail_on_invalid_fold": bool(fail_on_invalid_fold),
            "run_id": run_id,
        }
    )

    tmp_dir = (
        target_dir
        / f"_improve_best_candidate_tmp_{run_id}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
    )
    tmp_dir.mkdir(parents=True, exist_ok=True)

    evals: dict[str, dict[str, Any]] = {}
    for variant_name in variants:
        try:
            variant_df, variant_features, variant_note = _materialize_variant(
                base_frame=candidate_df,
                all_feature_cols=all_feature_cols,
                families=families,
                variant_name=variant_name,
                best_family_name=best_family_name,
                target_clip_abs=float(target_clip_abs),
                feature_winsor_abs=float(feature_winsor_abs),
            )
        except Exception as exc:
            notes.append(f"Skipped variant '{variant_name}': {exc}")
            continue

        dataset_path = tmp_dir / f"{variant_name}_dataset.parquet"
        variant_df.to_parquet(dataset_path, index=False)

        model_run = run_cross_validated_baseline(
            mode=MODE_RIDGE_CV,
            model_dataset_path=dataset_path,
            purged_cv_folds_path=folds_source,
            output_dir=tmp_dir / f"{variant_name}_ridge_cv",
            label_name=label_name,
            horizon_days=int(horizon_days),
            alpha_grid=alpha_values,
            c_grid=(0.1, 1.0, 10.0),
            fail_on_invalid_fold=bool(fail_on_invalid_fold),
            write_predictions=False,
            run_id=f"{run_id}_{variant_name}_ridge",
        )
        dummy_run = run_cross_validated_baseline(
            mode=MODE_DUMMY_REGRESSOR_CV,
            model_dataset_path=dataset_path,
            purged_cv_folds_path=folds_source,
            output_dir=tmp_dir / f"{variant_name}_dummy_regressor_cv",
            label_name=label_name,
            horizon_days=int(horizon_days),
            alpha_grid=alpha_values,
            c_grid=(0.1, 1.0, 10.0),
            fail_on_invalid_fold=bool(fail_on_invalid_fold),
            write_predictions=False,
            run_id=f"{run_id}_{variant_name}_dummy",
        )

        model_fold = _completed_fold_metrics(model_run.fold_metrics_path)[
            ["fold_id", "valid_primary_metric", "n_features_used"]
        ].rename(columns={"valid_primary_metric": "model_valid_primary_metric"})
        dummy_fold = _completed_fold_metrics(dummy_run.fold_metrics_path)[
            ["fold_id", "valid_primary_metric"]
        ].rename(columns={"valid_primary_metric": "dummy_valid_primary_metric"})
        merged = model_fold.merge(dummy_fold, on="fold_id", how="inner")
        if merged.empty:
            notes.append(f"Skipped variant '{variant_name}': no common completed folds vs dummy.")
            continue
        merged["improvement_vs_dummy"] = (
            merged["dummy_valid_primary_metric"] - merged["model_valid_primary_metric"]
        )

        evals[variant_name] = {
            "variant_name": variant_name,
            "variant_note": variant_note,
            "features_used": tuple(variant_features),
            "n_features_used": int(round(float(merged["n_features_used"].median()))),
            "n_folds": int(len(merged)),
            "model_mean": float(merged["model_valid_primary_metric"].mean()),
            "model_median": float(merged["model_valid_primary_metric"].median()),
            "model_std": float(merged["model_valid_primary_metric"].std(ddof=0)),
            "dummy_mean": float(merged["dummy_valid_primary_metric"].mean()),
            "fold_metrics": merged[
                [
                    "fold_id",
                    "model_valid_primary_metric",
                    "dummy_valid_primary_metric",
                    "improvement_vs_dummy",
                    "n_features_used",
                ]
            ].copy(),
        }

    if VARIANT_BASELINE not in evals:
        raise ValueError(
            "Baseline variant 'baseline_all_features' could not be evaluated. "
            "Cannot compute improvements."
        )

    baseline_fold = evals[VARIANT_BASELINE]["fold_metrics"][
        ["fold_id", "model_valid_primary_metric"]
    ].rename(columns={"model_valid_primary_metric": "baseline_valid_primary_metric"})

    results_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for variant_name in variants:
        if variant_name not in evals:
            continue
        item = evals[variant_name]
        merged_vs_baseline = item["fold_metrics"].merge(baseline_fold, on="fold_id", how="inner")
        if merged_vs_baseline.empty:
            notes.append(
                f"Variant '{variant_name}' has no common completed folds vs baseline; skipped."
            )
            continue
        merged_vs_baseline["improvement_vs_baseline"] = (
            merged_vs_baseline["baseline_valid_primary_metric"]
            - merged_vs_baseline["model_valid_primary_metric"]
        )

        improvement_vs_baseline = float(merged_vs_baseline["improvement_vs_baseline"].mean())
        improvement_vs_dummy = float(item["dummy_mean"] - item["model_mean"])
        winner_vs_baseline = _winner(
            improvement_vs_baseline,
            tie_tolerance=float(tie_tolerance),
            positive_label="variant",
            negative_label="baseline",
        )
        winner_vs_dummy = _winner(
            improvement_vs_dummy,
            tie_tolerance=float(tie_tolerance),
            positive_label="variant",
            negative_label="dummy",
        )

        results_rows.append(
            {
                "candidate_variant": variant_name,
                "label_name": label_name,
                "target_type": target_type,
                "horizon_days": int(horizon_days),
                "primary_metric": PRIMARY_METRIC,
                "mean_valid_primary_metric": float(item["model_mean"]),
                "median_valid_primary_metric": float(item["model_median"]),
                "std_valid_primary_metric": float(item["model_std"]),
                "improvement_vs_baseline": improvement_vs_baseline,
                "improvement_vs_dummy": improvement_vs_dummy,
                "winner_vs_baseline": winner_vs_baseline,
                "winner_vs_dummy": winner_vs_dummy,
                "n_features_used": int(item["n_features_used"]),
                "run_id": run_id,
                "config_hash": config_hash,
                "built_ts_utc": built_ts_utc,
            }
        )

        for row in merged_vs_baseline.itertuples(index=False):
            fold_rows.append(
                {
                    "candidate_variant": variant_name,
                    "label_name": label_name,
                    "target_type": target_type,
                    "horizon_days": int(horizon_days),
                    "fold_id": int(row.fold_id),
                    "primary_metric": PRIMARY_METRIC,
                    "model_valid_primary_metric": float(row.model_valid_primary_metric),
                    "dummy_valid_primary_metric": float(row.dummy_valid_primary_metric),
                    "baseline_valid_primary_metric": float(row.baseline_valid_primary_metric),
                    "improvement_vs_dummy": float(row.improvement_vs_dummy),
                    "improvement_vs_baseline": float(row.improvement_vs_baseline),
                    "n_features_used": int(row.n_features_used),
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "built_ts_utc": built_ts_utc,
                }
            )

    if not results_rows:
        raise ValueError("No comparable variant results were produced.")

    results_df = pd.DataFrame(results_rows).sort_values(
        ["mean_valid_primary_metric", "candidate_variant"]
    ).reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows).sort_values(["candidate_variant", "fold_id"]).reset_index(
        drop=True
    )

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
        target_dir / "improve_best_candidate_results.parquet",
        schema_name=RESULTS_SCHEMA.name,
        run_id=run_id,
    )
    fold_metrics_path = write_parquet(
        fold_df,
        target_dir / "improve_best_candidate_fold_metrics.parquet",
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

    if (
        str(best_row["winner_vs_baseline"]) == "variant"
        and str(best_row["winner_vs_dummy"]) == "variant"
    ):
        recommendation = "adopt_best_variant_and_revalidate"
    elif variants_beating_baseline:
        recommendation = "recheck_variant_stability_before_adoption"
    elif variants_beating_dummy:
        recommendation = "keep_baseline_and_continue_feature_label_improvement"
    else:
        recommendation = "no_variant_improvement_keep_baseline"

    summary_payload = {
        "module_version": MODULE_VERSION,
        "built_ts_utc": built_ts_utc,
        "run_id": run_id,
        "config_hash": config_hash,
        "baseline_variant": VARIANT_BASELINE,
        "best_variant": str(best_row["candidate_variant"]),
        "variants_evaluated": results_df["candidate_variant"].astype(str).tolist(),
        "variants_beating_baseline": variants_beating_baseline,
        "variants_beating_dummy": variants_beating_dummy,
        "recommendation": recommendation,
        "label_name": label_name,
        "target_type": target_type,
        "horizon_days": int(horizon_days),
        "best_feature_family_resolved": best_family_name,
        "best_variant_mean_valid_primary_metric": float(best_row["mean_valid_primary_metric"]),
        "baseline_mean_valid_primary_metric": float(
            results_df.loc[
                results_df["candidate_variant"].astype(str) == VARIANT_BASELINE,
                "mean_valid_primary_metric",
            ].iloc[0]
        ),
        "variant_notes": {key: str(value["variant_note"]) for key, value in evals.items()},
        "variant_n_features": {key: int(value["n_features_used"]) for key, value in evals.items()},
        "input_paths": {
            "model_dataset": str(dataset_source),
            "purged_cv_folds": str(folds_source),
            "feature_ablation_summary": str(feature_ablation_summary_source),
        },
        "output_paths": {
            "improve_best_candidate_results": str(results_path),
            "improve_best_candidate_fold_metrics": str(fold_metrics_path),
        },
        "notes": notes,
    }
    summary_path = target_dir / "improve_best_candidate_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "improve_best_candidate_completed",
        run_id=run_id,
        n_results_rows=int(len(results_df)),
        best_variant=str(best_row["candidate_variant"]),
        results_path=str(results_path),
        summary_path=str(summary_path),
    )
    return ImproveBestCandidateResult(
        results_path=results_path,
        summary_path=summary_path,
        fold_metrics_path=fold_metrics_path,
        n_rows=int(len(results_df)),
        config_hash=config_hash,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run directed improvement variants for the current best regression candidate."
    )
    parser.add_argument("--model-dataset-path", type=str, default=None)
    parser.add_argument("--purged-cv-folds-path", type=str, default=None)
    parser.add_argument("--feature-ablation-summary-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--label-name", type=str, default="fwd_ret_20d")
    parser.add_argument("--target-type", type=str, default="continuous_forward_return")
    parser.add_argument("--horizon-days", type=int, default=20)
    parser.add_argument("--best-feature-family", type=str, default=None)
    parser.add_argument("--ridge-alphas", type=str, default="0.1,1.0,10.0")
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    parser.add_argument("--enable-target-clipping", action="store_true")
    parser.add_argument("--target-clip-abs", type=float, default=0.20)
    parser.add_argument("--enable-feature-winsorization", action="store_true")
    parser.add_argument("--feature-winsor-abs", type=float, default=5.0)
    parser.add_argument("--fail-on-invalid-fold", action="store_true")
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_improve_best_candidate(
        model_dataset_path=args.model_dataset_path,
        purged_cv_folds_path=args.purged_cv_folds_path,
        feature_ablation_summary_path=args.feature_ablation_summary_path,
        output_dir=args.output_dir,
        label_name=args.label_name,
        target_type=args.target_type,
        horizon_days=args.horizon_days,
        best_feature_family=args.best_feature_family,
        alpha_grid=_parse_csv_floats(args.ridge_alphas) or (0.1, 1.0, 10.0),
        tie_tolerance=args.tie_tolerance,
        enable_target_clipping=bool(args.enable_target_clipping),
        target_clip_abs=float(args.target_clip_abs),
        enable_feature_winsorization=bool(args.enable_feature_winsorization),
        feature_winsor_abs=float(args.feature_winsor_abs),
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
