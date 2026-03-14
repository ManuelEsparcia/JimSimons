from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

# Allow direct script execution: `python simons_smallcap_swing/models/baselines/run_baseline_benchmarks.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.baselines.model_comparison import compare_baseline_metrics
from simons_core.io.parquet_store import write_parquet
from simons_core.logging import get_logger


MODULE_VERSION = "baseline_benchmark_suite_mvp_v1"

REGRESSION_TASK = "regression_baselines"
CLASSIFICATION_TASK = "classification_baselines"
TASK_ORDER: tuple[str, ...] = (REGRESSION_TASK, CLASSIFICATION_TASK)

TASK_CONFIG: dict[str, dict[str, str]] = {
    REGRESSION_TASK: {
        "model_a_alias": "ridge",
        "model_b_alias": "dummy_regressor",
        "model_a_default_name": "ridge_baseline",
        "model_b_default_name": "dummy_regressor_baseline",
        "default_model_a_metrics": "ridge_baseline_metrics.json",
        "default_model_b_metrics": "dummy_regressor_metrics.json",
        "expected_target_type": "continuous_forward_return",
    },
    CLASSIFICATION_TASK: {
        "model_a_alias": "logistic",
        "model_b_alias": "dummy_classifier",
        "model_a_default_name": "logistic_baseline",
        "model_b_default_name": "dummy_classifier_baseline",
        "default_model_a_metrics": "logistic_baseline_metrics.json",
        "default_model_b_metrics": "dummy_classifier_metrics.json",
        "expected_target_type": "binary_direction",
    },
}

MISSING_STATUS = "missing"
ERROR_STATUS = "error"
NON_COMPARABLE_STATUS = "non_comparable"


@dataclass(frozen=True)
class TaskBenchmarkOutcome:
    task_name: str
    model_a_name: str
    model_b_name: str
    label_name: str | None
    target_type: str | None
    split_name: str | None
    comparability_status: str
    primary_metric: str | None
    model_a_valid_primary: float | None
    model_b_valid_primary: float | None
    model_a_test_primary: float | None
    model_b_test_primary: float | None
    winner_valid: str
    winner_test: str
    notes: list[str]
    model_a_metrics_path: str
    model_b_metrics_path: str
    task_summary_path: str | None
    task_table_path: str | None


@dataclass(frozen=True)
class BaselineBenchmarkResult:
    summary_path: Path
    table_path: Path
    manifest_path: Path
    regression_status: str
    classification_status: str


def _resolve_metrics_path(
    *,
    provided_path: str | Path | None,
    artifacts_dir: Path,
    default_filename: str,
) -> Path:
    if provided_path:
        return Path(provided_path).expanduser().resolve()
    return artifacts_dir / default_filename


def _load_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path).expanduser().resolve()
    return json.loads(file_path.read_text(encoding="utf-8"))


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


def _missing_outcome(
    *,
    task_name: str,
    model_a_metrics_path: Path,
    model_b_metrics_path: Path,
    missing_notes: list[str],
) -> TaskBenchmarkOutcome:
    config = TASK_CONFIG[task_name]
    return TaskBenchmarkOutcome(
        task_name=task_name,
        model_a_name=config["model_a_default_name"],
        model_b_name=config["model_b_default_name"],
        label_name=None,
        target_type=None,
        split_name=None,
        comparability_status=MISSING_STATUS,
        primary_metric=None,
        model_a_valid_primary=None,
        model_b_valid_primary=None,
        model_a_test_primary=None,
        model_b_test_primary=None,
        winner_valid=NON_COMPARABLE_STATUS,
        winner_test=NON_COMPARABLE_STATUS,
        notes=missing_notes,
        model_a_metrics_path=str(model_a_metrics_path),
        model_b_metrics_path=str(model_b_metrics_path),
        task_summary_path=None,
        task_table_path=None,
    )


def _error_outcome(
    *,
    task_name: str,
    model_a_metrics_path: Path,
    model_b_metrics_path: Path,
    error_text: str,
) -> TaskBenchmarkOutcome:
    config = TASK_CONFIG[task_name]
    return TaskBenchmarkOutcome(
        task_name=task_name,
        model_a_name=config["model_a_default_name"],
        model_b_name=config["model_b_default_name"],
        label_name=None,
        target_type=None,
        split_name=None,
        comparability_status=ERROR_STATUS,
        primary_metric=None,
        model_a_valid_primary=None,
        model_b_valid_primary=None,
        model_a_test_primary=None,
        model_b_test_primary=None,
        winner_valid=NON_COMPARABLE_STATUS,
        winner_test=NON_COMPARABLE_STATUS,
        notes=[error_text],
        model_a_metrics_path=str(model_a_metrics_path),
        model_b_metrics_path=str(model_b_metrics_path),
        task_summary_path=None,
        task_table_path=None,
    )


def _run_task_comparison(
    *,
    task_name: str,
    model_a_metrics_path: Path,
    model_b_metrics_path: Path,
    output_dir: Path,
    run_id: str,
    tie_tolerance: float,
) -> TaskBenchmarkOutcome:
    config = TASK_CONFIG[task_name]
    expected_target_type = config["expected_target_type"]

    comparison_result = compare_baseline_metrics(
        model_a_metrics_path=model_a_metrics_path,
        model_b_metrics_path=model_b_metrics_path,
        output_dir=output_dir,
        run_id=run_id,
        tie_tolerance=tie_tolerance,
    )
    comparison_summary = _load_json(comparison_result.summary_path)

    model_a_name = str(comparison_summary["model_names"]["model_a"])
    model_b_name = str(comparison_summary["model_names"]["model_b"])
    label_name_a = str(comparison_summary["label_names"]["model_a"])
    label_name_b = str(comparison_summary["label_names"]["model_b"])
    target_type_a = str(comparison_summary["target_types"]["model_a"])
    target_type_b = str(comparison_summary["target_types"]["model_b"])
    split_name_a = str(comparison_summary["split_names"]["model_a"])
    split_name_b = str(comparison_summary["split_names"]["model_b"])

    notes: list[str] = list(comparison_summary.get("notes", []))
    comparability_status = str(comparison_summary.get("comparability_status", NON_COMPARABLE_STATUS))
    winner_valid = str(comparison_summary.get("winner_valid", NON_COMPARABLE_STATUS))
    winner_test = str(comparison_summary.get("winner_test", NON_COMPARABLE_STATUS))

    # Guardrail: never allow cross-task mixture in benchmark suite.
    if target_type_a != expected_target_type or target_type_b != expected_target_type:
        comparability_status = NON_COMPARABLE_STATUS
        winner_valid = NON_COMPARABLE_STATUS
        winner_test = NON_COMPARABLE_STATUS
        notes.append(
            f"task target_type mismatch: expected '{expected_target_type}', "
            f"observed model_a='{target_type_a}', model_b='{target_type_b}'"
        )

    primary_metric = comparison_summary.get("primary_metric_used")
    primary_values = comparison_summary.get("primary_metric_values", {})
    valid_values = primary_values.get("valid", {})
    test_values = primary_values.get("test", {})

    return TaskBenchmarkOutcome(
        task_name=task_name,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        label_name=label_name_a if label_name_a == label_name_b else None,
        target_type=target_type_a if target_type_a == target_type_b else None,
        split_name=split_name_a if split_name_a == split_name_b else None,
        comparability_status=comparability_status,
        primary_metric=str(primary_metric) if primary_metric is not None else None,
        model_a_valid_primary=_to_float_or_none(valid_values.get("model_a")),
        model_b_valid_primary=_to_float_or_none(valid_values.get("model_b")),
        model_a_test_primary=_to_float_or_none(test_values.get("model_a")),
        model_b_test_primary=_to_float_or_none(test_values.get("model_b")),
        winner_valid=winner_valid,
        winner_test=winner_test,
        notes=notes,
        model_a_metrics_path=str(model_a_metrics_path),
        model_b_metrics_path=str(model_b_metrics_path),
        task_summary_path=str(comparison_result.summary_path),
        task_table_path=str(comparison_result.table_path),
    )


def run_baseline_benchmarks(
    *,
    artifacts_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    ridge_metrics_path: str | Path | None = None,
    dummy_regressor_metrics_path: str | Path | None = None,
    logistic_metrics_path: str | Path | None = None,
    dummy_classifier_metrics_path: str | Path | None = None,
    strict_missing: bool = False,
    tie_tolerance: float = 1e-12,
    run_id: str = MODULE_VERSION,
) -> BaselineBenchmarkResult:
    logger = get_logger("models.baselines.run_baseline_benchmarks")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be >= 0.")

    default_artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts"
    artifacts_root = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir
    output_root = Path(output_dir).expanduser().resolve() if output_dir else artifacts_root
    output_root.mkdir(parents=True, exist_ok=True)
    task_root = output_root / "_baseline_task_comparisons"
    task_root.mkdir(parents=True, exist_ok=True)

    regression_model_a = _resolve_metrics_path(
        provided_path=ridge_metrics_path,
        artifacts_dir=artifacts_root,
        default_filename=TASK_CONFIG[REGRESSION_TASK]["default_model_a_metrics"],
    )
    regression_model_b = _resolve_metrics_path(
        provided_path=dummy_regressor_metrics_path,
        artifacts_dir=artifacts_root,
        default_filename=TASK_CONFIG[REGRESSION_TASK]["default_model_b_metrics"],
    )
    classification_model_a = _resolve_metrics_path(
        provided_path=logistic_metrics_path,
        artifacts_dir=artifacts_root,
        default_filename=TASK_CONFIG[CLASSIFICATION_TASK]["default_model_a_metrics"],
    )
    classification_model_b = _resolve_metrics_path(
        provided_path=dummy_classifier_metrics_path,
        artifacts_dir=artifacts_root,
        default_filename=TASK_CONFIG[CLASSIFICATION_TASK]["default_model_b_metrics"],
    )

    task_inputs = {
        REGRESSION_TASK: (regression_model_a, regression_model_b),
        CLASSIFICATION_TASK: (classification_model_a, classification_model_b),
    }

    outcomes: dict[str, TaskBenchmarkOutcome] = {}
    global_notes: list[str] = []

    for task_name in TASK_ORDER:
        model_a_path, model_b_path = task_inputs[task_name]
        missing_messages: list[str] = []
        if not model_a_path.exists():
            missing_messages.append(f"missing metrics artifact: {model_a_path}")
        if not model_b_path.exists():
            missing_messages.append(f"missing metrics artifact: {model_b_path}")

        if missing_messages:
            if strict_missing:
                raise FileNotFoundError("; ".join(missing_messages))
            outcome = _missing_outcome(
                task_name=task_name,
                model_a_metrics_path=model_a_path,
                model_b_metrics_path=model_b_path,
                missing_notes=missing_messages,
            )
            outcomes[task_name] = outcome
            global_notes.extend([f"{task_name}: {msg}" for msg in missing_messages])
            continue

        try:
            outcome = _run_task_comparison(
                task_name=task_name,
                model_a_metrics_path=model_a_path,
                model_b_metrics_path=model_b_path,
                output_dir=task_root / task_name,
                run_id=f"{run_id}_{task_name}",
                tie_tolerance=tie_tolerance,
            )
            outcomes[task_name] = outcome
            if outcome.comparability_status != "comparable":
                global_notes.append(
                    f"{task_name}: comparability_status={outcome.comparability_status}"
                )
        except Exception as exc:
            outcome = _error_outcome(
                task_name=task_name,
                model_a_metrics_path=model_a_path,
                model_b_metrics_path=model_b_path,
                error_text=f"comparison error: {exc}",
            )
            outcomes[task_name] = outcome
            global_notes.append(f"{task_name}: comparison error: {exc}")

    built_ts = datetime.now(UTC).isoformat()

    table_rows: list[dict[str, Any]] = []
    for task_name in TASK_ORDER:
        out = outcomes[task_name]
        table_rows.append(
            {
                "task_name": out.task_name,
                "model_a_name": out.model_a_name,
                "model_b_name": out.model_b_name,
                "label_name": out.label_name,
                "target_type": out.target_type,
                "split_name": out.split_name,
                "comparability_status": out.comparability_status,
                "primary_metric": out.primary_metric,
                "model_a_valid_primary": out.model_a_valid_primary,
                "model_b_valid_primary": out.model_b_valid_primary,
                "model_a_test_primary": out.model_a_test_primary,
                "model_b_test_primary": out.model_b_test_primary,
                "winner_valid": out.winner_valid,
                "winner_test": out.winner_test,
                "run_id": run_id,
                "built_ts_utc": built_ts,
            }
        )
    table_frame = pd.DataFrame(table_rows)
    table_path = write_parquet(
        table_frame,
        output_root / "baseline_benchmark_table.parquet",
        schema_name="baseline_benchmark_table_mvp",
        run_id=run_id,
    )

    def _task_summary(out: TaskBenchmarkOutcome) -> dict[str, Any]:
        return {
            "label_name": out.label_name,
            "split_name": out.split_name,
            "comparability_status": out.comparability_status,
            "primary_metric": out.primary_metric,
            "winner_valid": out.winner_valid,
            "winner_test": out.winner_test,
            "model_a_name": out.model_a_name,
            "model_b_name": out.model_b_name,
            "model_a_valid_primary": out.model_a_valid_primary,
            "model_b_valid_primary": out.model_b_valid_primary,
            "model_a_test_primary": out.model_a_test_primary,
            "model_b_test_primary": out.model_b_test_primary,
            "target_type": out.target_type,
            "notes": out.notes,
            "model_a_metrics_path": out.model_a_metrics_path,
            "model_b_metrics_path": out.model_b_metrics_path,
            "task_summary_path": out.task_summary_path,
            "task_table_path": out.task_table_path,
        }

    summary_payload = {
        "benchmark_run_id": run_id,
        "built_ts_utc": built_ts,
        "tasks_compared": list(TASK_ORDER),
        "regression": _task_summary(outcomes[REGRESSION_TASK]),
        "classification": _task_summary(outcomes[CLASSIFICATION_TASK]),
        "notes": global_notes,
        "artifacts": {
            "baseline_benchmark_table_path": str(table_path),
        },
    }
    summary_path = output_root / "baseline_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    manifest_payload = {
        "benchmark_run_id": run_id,
        "built_ts_utc": built_ts,
        "module_version": MODULE_VERSION,
        "artifacts": {
            "baseline_benchmark_summary_path": str(summary_path),
            "baseline_benchmark_table_path": str(table_path),
            "task_comparison_root": str(task_root),
        },
        "task_status": {
            task_name: outcomes[task_name].comparability_status for task_name in TASK_ORDER
        },
    }
    manifest_path = output_root / "baseline_benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "baseline_benchmark_suite_completed",
        run_id=run_id,
        regression_status=outcomes[REGRESSION_TASK].comparability_status,
        classification_status=outcomes[CLASSIFICATION_TASK].comparability_status,
        summary_path=str(summary_path),
        table_path=str(table_path),
    )

    return BaselineBenchmarkResult(
        summary_path=summary_path,
        table_path=Path(table_path),
        manifest_path=manifest_path,
        regression_status=outcomes[REGRESSION_TASK].comparability_status,
        classification_status=outcomes[CLASSIFICATION_TASK].comparability_status,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run homogeneous baseline benchmark suite (regression/classification).")
    parser.add_argument("--artifacts-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--ridge-metrics-path", type=str, default=None)
    parser.add_argument("--dummy-regressor-metrics-path", type=str, default=None)
    parser.add_argument("--logistic-metrics-path", type=str, default=None)
    parser.add_argument("--dummy-classifier-metrics-path", type=str, default=None)
    parser.add_argument("--strict-missing", action="store_true")
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_baseline_benchmarks(
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
        ridge_metrics_path=args.ridge_metrics_path,
        dummy_regressor_metrics_path=args.dummy_regressor_metrics_path,
        logistic_metrics_path=args.logistic_metrics_path,
        dummy_classifier_metrics_path=args.dummy_classifier_metrics_path,
        strict_missing=bool(args.strict_missing),
        tie_tolerance=float(args.tie_tolerance),
        run_id=args.run_id,
    )
    print("Baseline benchmark suite completed:")
    print(f"- summary: {result.summary_path}")
    print(f"- table: {result.table_path}")
    print(f"- manifest: {result.manifest_path}")
    print(f"- regression_status: {result.regression_status}")
    print(f"- classification_status: {result.classification_status}")


if __name__ == "__main__":
    main()
