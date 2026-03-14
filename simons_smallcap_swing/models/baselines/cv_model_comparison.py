from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

# Allow direct script execution:
# `python simons_smallcap_swing/models/baselines/cv_model_comparison.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.logging import get_logger


MODULE_VERSION = "cv_model_comparison_mvp_v1"
REGRESSION_TASK = "regression_cv_baselines"
CLASSIFICATION_TASK = "classification_cv_baselines"
TASK_ORDER: tuple[str, ...] = (REGRESSION_TASK, CLASSIFICATION_TASK)

COMPARABLE_STATUS = "comparable"
NON_COMPARABLE_STATUS = "non_comparable"
MISSING_STATUS = "missing"
TIE = "tie"

TASK_CONFIG: dict[str, dict[str, Any]] = {
    REGRESSION_TASK: {
        "model_a_name": "ridge_cv",
        "model_b_name": "dummy_regressor_cv",
        "expected_target_type": "continuous_forward_return",
        "primary_metric": "mse",
        "direction": "min",
    },
    CLASSIFICATION_TASK: {
        "model_a_name": "logistic_cv",
        "model_b_name": "dummy_classifier_cv",
        "expected_target_type": "binary_direction",
        "primary_metric": "log_loss",
        "direction": "min",
    },
}


@dataclass(frozen=True)
class ModelCVInput:
    fold_metrics_path: Path
    summary_path: Path | None
    expected_model_name: str


@dataclass(frozen=True)
class TaskComparisonOutcome:
    task_name: str
    model_a_name: str
    model_b_name: str
    label_name: str | None
    horizon_days: int | None
    target_type: str | None
    cv_method: str | None
    split_name: str | None
    primary_metric: str | None
    comparability_status: str
    mean_delta: float | None
    median_delta: float | None
    std_delta: float | None
    model_a_fold_wins: int
    model_b_fold_wins: int
    ties: int
    winner_global: str
    notes: list[str]
    model_a_fold_metrics_path: str
    model_b_fold_metrics_path: str
    fold_level: pd.DataFrame


@dataclass(frozen=True)
class CVModelComparisonResult:
    summary_path: Path
    table_path: Path
    fold_level_path: Path | None
    regression_status: str
    classification_status: str


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _resolve_fold_metrics_path(
    *,
    provided_path: str | Path | None,
    artifacts_dir: Path,
    model_name: str,
) -> Path:
    if provided_path:
        return Path(provided_path).expanduser().resolve()
    return artifacts_dir / model_name / "cv_baseline_fold_metrics.parquet"


def _resolve_summary_path(
    *,
    provided_path: str | Path | None,
    artifacts_dir: Path,
    model_name: str,
) -> Path:
    if provided_path:
        return Path(provided_path).expanduser().resolve()
    return artifacts_dir / model_name / "cv_baseline_summary.json"


def _load_summary_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON summary: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Summary file {path} does not contain a JSON object.")
    return payload


def _load_fold_metrics(input_cfg: ModelCVInput) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if not input_cfg.fold_metrics_path.exists():
        raise FileNotFoundError(f"Fold metrics file not found: {input_cfg.fold_metrics_path}")

    frame = read_parquet(input_cfg.fold_metrics_path)
    required_cols = {
        "model_name",
        "fold_id",
        "label_name",
        "horizon_days",
        "target_type",
        "primary_metric",
        "valid_primary_metric",
    }
    missing_cols = sorted(required_cols - set(frame.columns))
    if missing_cols:
        raise ValueError(
            f"Fold metrics {input_cfg.fold_metrics_path} missing required columns: {missing_cols}"
        )

    frame = frame.copy()
    frame["model_name"] = frame["model_name"].astype(str)
    frame["fold_id"] = pd.to_numeric(frame["fold_id"], errors="coerce").astype("Int64")
    frame["label_name"] = frame["label_name"].astype(str)
    frame["horizon_days"] = pd.to_numeric(frame["horizon_days"], errors="coerce").astype("Int64")
    frame["target_type"] = frame["target_type"].astype(str)
    frame["primary_metric"] = frame["primary_metric"].astype(str)
    frame["valid_primary_metric"] = pd.to_numeric(frame["valid_primary_metric"], errors="coerce")
    if "status" in frame.columns:
        frame = frame[frame["status"].astype(str) == "completed"].copy()

    if frame.empty:
        raise ValueError(f"No completed folds found in {input_cfg.fold_metrics_path}")
    if frame["fold_id"].isna().any():
        raise ValueError(f"Fold metrics {input_cfg.fold_metrics_path} has invalid fold_id values.")

    expected_name = input_cfg.expected_model_name
    unique_models = sorted(frame["model_name"].unique().tolist())
    if expected_name in unique_models:
        frame = frame[frame["model_name"] == expected_name].copy()
    elif len(unique_models) == 1:
        # Keep file usable even when aliasing differs.
        pass
    else:
        raise ValueError(
            f"Fold metrics {input_cfg.fold_metrics_path} has multiple model_name values "
            f"{unique_models} and expected model '{expected_name}' is not present."
        )

    if frame.empty:
        raise ValueError(
            f"Fold metrics {input_cfg.fold_metrics_path} has no rows for model '{expected_name}'."
        )

    summary = _load_summary_if_exists(input_cfg.summary_path)
    return frame.sort_values(["fold_id"]).reset_index(drop=True), summary


def _single_or_none(values: pd.Series, field: str, reasons: list[str]) -> Any | None:
    unique_vals = sorted(pd.Series(values).dropna().unique().tolist())
    if len(unique_vals) == 1:
        return unique_vals[0]
    reasons.append(f"{field} not unique: {unique_vals}")
    return None


def _fold_winner(delta: float, tie_tolerance: float) -> str:
    if pd.isna(delta):
        return NON_COMPARABLE_STATUS
    if abs(float(delta)) <= tie_tolerance:
        return TIE
    return "model_a" if float(delta) < 0 else "model_b"


def _global_winner(
    *,
    wins_a: int,
    wins_b: int,
    mean_delta: float | None,
    tie_tolerance: float,
) -> str:
    if wins_a > wins_b:
        return "model_a"
    if wins_b > wins_a:
        return "model_b"
    if mean_delta is None:
        return TIE
    if abs(mean_delta) <= tie_tolerance:
        return TIE
    return "model_a" if mean_delta < 0 else "model_b"


def _compare_task(
    *,
    task_name: str,
    model_a_input: ModelCVInput,
    model_b_input: ModelCVInput,
    tie_tolerance: float,
    n_valid_tolerance_ratio: float,
    run_id: str,
) -> TaskComparisonOutcome:
    config = TASK_CONFIG[task_name]
    expected_target = str(config["expected_target_type"])
    expected_primary_metric = str(config["primary_metric"])
    expected_model_a_name = str(config["model_a_name"])
    expected_model_b_name = str(config["model_b_name"])
    reasons: list[str] = []

    frame_a, summary_a = _load_fold_metrics(model_a_input)
    frame_b, summary_b = _load_fold_metrics(model_b_input)

    label_a = _single_or_none(frame_a["label_name"], "model_a label_name", reasons)
    label_b = _single_or_none(frame_b["label_name"], "model_b label_name", reasons)
    horizon_a = _single_or_none(frame_a["horizon_days"], "model_a horizon_days", reasons)
    horizon_b = _single_or_none(frame_b["horizon_days"], "model_b horizon_days", reasons)
    target_a = _single_or_none(frame_a["target_type"], "model_a target_type", reasons)
    target_b = _single_or_none(frame_b["target_type"], "model_b target_type", reasons)
    primary_metric_a = _single_or_none(frame_a["primary_metric"], "model_a primary_metric", reasons)
    primary_metric_b = _single_or_none(frame_b["primary_metric"], "model_b primary_metric", reasons)

    model_name_a = str(_single_or_none(frame_a["model_name"], "model_a model_name", reasons) or expected_model_a_name)
    model_name_b = str(_single_or_none(frame_b["model_name"], "model_b model_name", reasons) or expected_model_b_name)

    if label_a != label_b:
        reasons.append(f"label_name mismatch: model_a={label_a} model_b={label_b}")
    if horizon_a != horizon_b:
        reasons.append(f"horizon_days mismatch: model_a={horizon_a} model_b={horizon_b}")
    if target_a != target_b:
        reasons.append(f"target_type mismatch: model_a={target_a} model_b={target_b}")
    if target_a != expected_target:
        reasons.append(f"unexpected target_type for task '{task_name}': expected={expected_target} observed={target_a}")
    if primary_metric_a != primary_metric_b:
        reasons.append(f"primary_metric mismatch: model_a={primary_metric_a} model_b={primary_metric_b}")
    if primary_metric_a != expected_primary_metric:
        reasons.append(
            f"unexpected primary_metric for task '{task_name}': expected={expected_primary_metric} observed={primary_metric_a}"
        )

    fold_ids_a = sorted(frame_a["fold_id"].astype(int).unique().tolist())
    fold_ids_b = sorted(frame_b["fold_id"].astype(int).unique().tolist())
    if fold_ids_a != fold_ids_b:
        reasons.append(f"fold_id mismatch: model_a={fold_ids_a} model_b={fold_ids_b}")

    cv_method_a = summary_a.get("cv_method") if summary_a else None
    cv_method_b = summary_b.get("cv_method") if summary_b else None
    split_name_a = summary_a.get("split_name") if summary_a else None
    split_name_b = summary_b.get("split_name") if summary_b else None
    if cv_method_a is not None and cv_method_b is not None and str(cv_method_a) != str(cv_method_b):
        reasons.append(f"cv_method mismatch: model_a={cv_method_a} model_b={cv_method_b}")
    if split_name_a is not None and split_name_b is not None and str(split_name_a) != str(split_name_b):
        reasons.append(f"split_name mismatch: model_a={split_name_a} model_b={split_name_b}")

    merge_cols = ["fold_id", "valid_primary_metric"]
    if "n_valid" in frame_a.columns and "n_valid" in frame_b.columns:
        merge_cols.append("n_valid")

    merge_a = frame_a[merge_cols].rename(
        columns={"valid_primary_metric": "model_a_primary", "n_valid": "model_a_n_valid"}
    )
    merge_b = frame_b[merge_cols].rename(
        columns={"valid_primary_metric": "model_b_primary", "n_valid": "model_b_n_valid"}
    )
    merged = merge_a.merge(merge_b, on="fold_id", how="inner")
    if len(merged) != len(fold_ids_a) or len(merged) != len(fold_ids_b):
        reasons.append("fold merge cardinality mismatch between model_a and model_b")

    if "model_a_n_valid" in merged.columns and "model_b_n_valid" in merged.columns:
        for row in merged.itertuples(index=False):
            left = int(row.model_a_n_valid)
            right = int(row.model_b_n_valid)
            tolerance = max(1, int(round(max(left, right) * float(n_valid_tolerance_ratio))))
            if abs(left - right) > tolerance:
                reasons.append(
                    f"n_valid mismatch on fold_id={int(row.fold_id)}: "
                    f"model_a={left} model_b={right} tolerance={tolerance}"
                )

    if merged["model_a_primary"].isna().any() or merged["model_b_primary"].isna().any():
        reasons.append("null valid_primary_metric found on comparable folds")

    comparability_status = COMPARABLE_STATUS if not reasons else NON_COMPARABLE_STATUS
    fold_level = pd.DataFrame()
    if comparability_status == COMPARABLE_STATUS:
        fold_level = merged[["fold_id", "model_a_primary", "model_b_primary"]].copy()
        fold_level["task_name"] = task_name
        fold_level["delta_primary"] = fold_level["model_a_primary"] - fold_level["model_b_primary"]
        fold_level["fold_winner"] = fold_level["delta_primary"].apply(
            lambda value: _fold_winner(float(value), tie_tolerance)
        )
        fold_level["run_id"] = run_id
        fold_level = fold_level[
            ["task_name", "fold_id", "model_a_primary", "model_b_primary", "delta_primary", "fold_winner", "run_id"]
        ].sort_values(["fold_id"])

        wins_a = int((fold_level["fold_winner"] == "model_a").sum())
        wins_b = int((fold_level["fold_winner"] == "model_b").sum())
        ties = int((fold_level["fold_winner"] == TIE).sum())
        mean_delta = _to_float_or_none(fold_level["delta_primary"].mean(skipna=True))
        median_delta = _to_float_or_none(fold_level["delta_primary"].median(skipna=True))
        std_delta = _to_float_or_none(fold_level["delta_primary"].std(skipna=True, ddof=0))
        winner_global = _global_winner(
            wins_a=wins_a,
            wins_b=wins_b,
            mean_delta=mean_delta,
            tie_tolerance=tie_tolerance,
        )
    else:
        wins_a = 0
        wins_b = 0
        ties = 0
        mean_delta = None
        median_delta = None
        std_delta = None
        winner_global = NON_COMPARABLE_STATUS

    return TaskComparisonOutcome(
        task_name=task_name,
        model_a_name=model_name_a,
        model_b_name=model_name_b,
        label_name=str(label_a) if label_a is not None else None,
        horizon_days=int(horizon_a) if horizon_a is not None and not pd.isna(horizon_a) else None,
        target_type=str(target_a) if target_a is not None else None,
        cv_method=str(cv_method_a) if cv_method_a is not None else (str(cv_method_b) if cv_method_b is not None else None),
        split_name=str(split_name_a) if split_name_a is not None else (str(split_name_b) if split_name_b is not None else None),
        primary_metric=expected_primary_metric,
        comparability_status=comparability_status,
        mean_delta=mean_delta,
        median_delta=median_delta,
        std_delta=std_delta,
        model_a_fold_wins=wins_a,
        model_b_fold_wins=wins_b,
        ties=ties,
        winner_global=winner_global,
        notes=reasons,
        model_a_fold_metrics_path=str(model_a_input.fold_metrics_path),
        model_b_fold_metrics_path=str(model_b_input.fold_metrics_path),
        fold_level=fold_level,
    )


def _missing_task_outcome(
    *,
    task_name: str,
    model_a_path: Path,
    model_b_path: Path,
    notes: list[str],
) -> TaskComparisonOutcome:
    config = TASK_CONFIG[task_name]
    return TaskComparisonOutcome(
        task_name=task_name,
        model_a_name=str(config["model_a_name"]),
        model_b_name=str(config["model_b_name"]),
        label_name=None,
        horizon_days=None,
        target_type=str(config["expected_target_type"]),
        cv_method=None,
        split_name=None,
        primary_metric=str(config["primary_metric"]),
        comparability_status=MISSING_STATUS,
        mean_delta=None,
        median_delta=None,
        std_delta=None,
        model_a_fold_wins=0,
        model_b_fold_wins=0,
        ties=0,
        winner_global=NON_COMPARABLE_STATUS,
        notes=notes,
        model_a_fold_metrics_path=str(model_a_path),
        model_b_fold_metrics_path=str(model_b_path),
        fold_level=pd.DataFrame(),
    )


def run_cv_model_comparison(
    *,
    artifacts_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    ridge_fold_metrics_path: str | Path | None = None,
    ridge_summary_path: str | Path | None = None,
    dummy_regressor_fold_metrics_path: str | Path | None = None,
    dummy_regressor_summary_path: str | Path | None = None,
    logistic_fold_metrics_path: str | Path | None = None,
    logistic_summary_path: str | Path | None = None,
    dummy_classifier_fold_metrics_path: str | Path | None = None,
    dummy_classifier_summary_path: str | Path | None = None,
    strict_missing: bool = False,
    tie_tolerance: float = 1e-12,
    n_valid_tolerance_ratio: float = 0.05,
    run_id: str = MODULE_VERSION,
) -> CVModelComparisonResult:
    logger = get_logger("models.baselines.cv_model_comparison")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be >= 0.")
    if n_valid_tolerance_ratio < 0:
        raise ValueError("n_valid_tolerance_ratio must be >= 0.")

    base_artifacts = Path(__file__).resolve().parents[1] / "artifacts"
    artifacts_root = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else base_artifacts
    output_root = Path(output_dir).expanduser().resolve() if output_dir else artifacts_root
    output_root.mkdir(parents=True, exist_ok=True)

    task_inputs: dict[str, tuple[ModelCVInput, ModelCVInput]] = {
        REGRESSION_TASK: (
            ModelCVInput(
                fold_metrics_path=_resolve_fold_metrics_path(
                    provided_path=ridge_fold_metrics_path,
                    artifacts_dir=artifacts_root,
                    model_name="ridge_cv",
                ),
                summary_path=_resolve_summary_path(
                    provided_path=ridge_summary_path,
                    artifacts_dir=artifacts_root,
                    model_name="ridge_cv",
                ),
                expected_model_name="ridge_cv",
            ),
            ModelCVInput(
                fold_metrics_path=_resolve_fold_metrics_path(
                    provided_path=dummy_regressor_fold_metrics_path,
                    artifacts_dir=artifacts_root,
                    model_name="dummy_regressor_cv",
                ),
                summary_path=_resolve_summary_path(
                    provided_path=dummy_regressor_summary_path,
                    artifacts_dir=artifacts_root,
                    model_name="dummy_regressor_cv",
                ),
                expected_model_name="dummy_regressor_cv",
            ),
        ),
        CLASSIFICATION_TASK: (
            ModelCVInput(
                fold_metrics_path=_resolve_fold_metrics_path(
                    provided_path=logistic_fold_metrics_path,
                    artifacts_dir=artifacts_root,
                    model_name="logistic_cv",
                ),
                summary_path=_resolve_summary_path(
                    provided_path=logistic_summary_path,
                    artifacts_dir=artifacts_root,
                    model_name="logistic_cv",
                ),
                expected_model_name="logistic_cv",
            ),
            ModelCVInput(
                fold_metrics_path=_resolve_fold_metrics_path(
                    provided_path=dummy_classifier_fold_metrics_path,
                    artifacts_dir=artifacts_root,
                    model_name="dummy_classifier_cv",
                ),
                summary_path=_resolve_summary_path(
                    provided_path=dummy_classifier_summary_path,
                    artifacts_dir=artifacts_root,
                    model_name="dummy_classifier_cv",
                ),
                expected_model_name="dummy_classifier_cv",
            ),
        ),
    }

    outcomes: dict[str, TaskComparisonOutcome] = {}
    global_notes: list[str] = []

    for task_name in TASK_ORDER:
        model_a_input, model_b_input = task_inputs[task_name]
        missing_notes: list[str] = []
        if not model_a_input.fold_metrics_path.exists():
            missing_notes.append(f"missing fold metrics artifact: {model_a_input.fold_metrics_path}")
        if not model_b_input.fold_metrics_path.exists():
            missing_notes.append(f"missing fold metrics artifact: {model_b_input.fold_metrics_path}")

        if missing_notes:
            if strict_missing:
                raise FileNotFoundError("; ".join(missing_notes))
            outcome = _missing_task_outcome(
                task_name=task_name,
                model_a_path=model_a_input.fold_metrics_path,
                model_b_path=model_b_input.fold_metrics_path,
                notes=missing_notes,
            )
            outcomes[task_name] = outcome
            global_notes.extend([f"{task_name}: {note}" for note in missing_notes])
            continue

        try:
            outcome = _compare_task(
                task_name=task_name,
                model_a_input=model_a_input,
                model_b_input=model_b_input,
                tie_tolerance=tie_tolerance,
                n_valid_tolerance_ratio=n_valid_tolerance_ratio,
                run_id=run_id,
            )
        except Exception as exc:
            if strict_missing:
                raise
            note = f"comparison failed: {exc}"
            global_notes.append(f"{task_name}: {note}")
            outcome = _missing_task_outcome(
                task_name=task_name,
                model_a_path=model_a_input.fold_metrics_path,
                model_b_path=model_b_input.fold_metrics_path,
                notes=[note],
            )
        outcomes[task_name] = outcome

    built_ts = datetime.now(UTC).isoformat()
    table_rows: list[dict[str, Any]] = []
    fold_level_rows: list[pd.DataFrame] = []
    for task_name in TASK_ORDER:
        outcome = outcomes[task_name]
        table_rows.append(
            {
                "task_name": outcome.task_name,
                "model_a_name": outcome.model_a_name,
                "model_b_name": outcome.model_b_name,
                "label_name": outcome.label_name,
                "horizon_days": outcome.horizon_days,
                "target_type": outcome.target_type,
                "cv_method": outcome.cv_method,
                "split_name": outcome.split_name,
                "primary_metric": outcome.primary_metric,
                "comparability_status": outcome.comparability_status,
                "mean_delta": outcome.mean_delta,
                "median_delta": outcome.median_delta,
                "std_delta": outcome.std_delta,
                "model_a_fold_wins": outcome.model_a_fold_wins,
                "model_b_fold_wins": outcome.model_b_fold_wins,
                "ties": outcome.ties,
                "winner_global": outcome.winner_global,
                "model_a_fold_metrics_path": outcome.model_a_fold_metrics_path,
                "model_b_fold_metrics_path": outcome.model_b_fold_metrics_path,
                "notes_json": json.dumps(outcome.notes, sort_keys=True),
                "run_id": run_id,
                "built_ts_utc": built_ts,
            }
        )
        if not outcome.fold_level.empty:
            fold_level_rows.append(outcome.fold_level.copy())

    table_frame = pd.DataFrame(table_rows)
    table_path = write_parquet(
        table_frame,
        output_root / "cv_model_comparison_table.parquet",
        schema_name="cv_model_comparison_table_mvp",
        run_id=run_id,
    )

    fold_level_path: Path | None = None
    if fold_level_rows:
        fold_level_frame = pd.concat(fold_level_rows, ignore_index=True)
        fold_level_path = write_parquet(
            fold_level_frame,
            output_root / "cv_model_comparison_fold_level.parquet",
            schema_name="cv_model_comparison_fold_level_mvp",
            run_id=run_id,
        )

    regression = outcomes[REGRESSION_TASK]
    classification = outcomes[CLASSIFICATION_TASK]
    summary = {
        "module_version": MODULE_VERSION,
        "benchmark_run_id": run_id,
        "tasks_compared": list(TASK_ORDER),
        "regression": {
            "task_name": regression.task_name,
            "model_a_name": regression.model_a_name,
            "model_b_name": regression.model_b_name,
            "label_name": regression.label_name,
            "horizon_days": regression.horizon_days,
            "target_type": regression.target_type,
            "cv_method": regression.cv_method,
            "split_name": regression.split_name,
            "comparability_status": regression.comparability_status,
            "primary_metric": regression.primary_metric,
            "mean_delta": regression.mean_delta,
            "median_delta": regression.median_delta,
            "std_delta": regression.std_delta,
            "model_a_fold_wins": regression.model_a_fold_wins,
            "model_b_fold_wins": regression.model_b_fold_wins,
            "ties": regression.ties,
            "winner_global": regression.winner_global,
            "notes": regression.notes,
        },
        "classification": {
            "task_name": classification.task_name,
            "model_a_name": classification.model_a_name,
            "model_b_name": classification.model_b_name,
            "label_name": classification.label_name,
            "horizon_days": classification.horizon_days,
            "target_type": classification.target_type,
            "cv_method": classification.cv_method,
            "split_name": classification.split_name,
            "comparability_status": classification.comparability_status,
            "primary_metric": classification.primary_metric,
            "mean_delta": classification.mean_delta,
            "median_delta": classification.median_delta,
            "std_delta": classification.std_delta,
            "model_a_fold_wins": classification.model_a_fold_wins,
            "model_b_fold_wins": classification.model_b_fold_wins,
            "ties": classification.ties,
            "winner_global": classification.winner_global,
            "notes": classification.notes,
        },
        "output_paths": {
            "table_path": str(table_path),
            "fold_level_path": str(fold_level_path) if fold_level_path else None,
        },
        "notes": global_notes,
        "built_ts_utc": built_ts,
    }
    summary_path = output_root / "cv_model_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "cv_model_comparison_completed",
        summary_path=str(summary_path),
        table_path=str(table_path),
        fold_level_path=str(fold_level_path) if fold_level_path else None,
        run_id=run_id,
    )
    return CVModelComparisonResult(
        summary_path=summary_path,
        table_path=table_path,
        fold_level_path=fold_level_path,
        regression_status=regression.comparability_status,
        classification_status=classification.comparability_status,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare baseline CV artifacts by task.")
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--output-dir", default=None)

    parser.add_argument("--ridge-fold-metrics-path", default=None)
    parser.add_argument("--ridge-summary-path", default=None)
    parser.add_argument("--dummy-regressor-fold-metrics-path", default=None)
    parser.add_argument("--dummy-regressor-summary-path", default=None)
    parser.add_argument("--logistic-fold-metrics-path", default=None)
    parser.add_argument("--logistic-summary-path", default=None)
    parser.add_argument("--dummy-classifier-fold-metrics-path", default=None)
    parser.add_argument("--dummy-classifier-summary-path", default=None)

    parser.add_argument("--strict-missing", action="store_true")
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    parser.add_argument("--n-valid-tolerance-ratio", type=float, default=0.05)
    parser.add_argument("--run-id", default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_cv_model_comparison(
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
        ridge_fold_metrics_path=args.ridge_fold_metrics_path,
        ridge_summary_path=args.ridge_summary_path,
        dummy_regressor_fold_metrics_path=args.dummy_regressor_fold_metrics_path,
        dummy_regressor_summary_path=args.dummy_regressor_summary_path,
        logistic_fold_metrics_path=args.logistic_fold_metrics_path,
        logistic_summary_path=args.logistic_summary_path,
        dummy_classifier_fold_metrics_path=args.dummy_classifier_fold_metrics_path,
        dummy_classifier_summary_path=args.dummy_classifier_summary_path,
        strict_missing=bool(args.strict_missing),
        tie_tolerance=float(args.tie_tolerance),
        n_valid_tolerance_ratio=float(args.n_valid_tolerance_ratio),
        run_id=args.run_id,
    )
    print(
        json.dumps(
            {
                "summary_path": str(result.summary_path),
                "table_path": str(result.table_path),
                "fold_level_path": str(result.fold_level_path) if result.fold_level_path else None,
                "regression_status": result.regression_status,
                "classification_status": result.classification_status,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
