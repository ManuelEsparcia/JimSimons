from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time
from typing import Callable

# Allow direct script execution: `python simons_smallcap_swing/run_week4_cv_validation.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from labels.purged_cv import build_purged_cv
from models.baselines.cross_validated_baselines import run_cross_validated_baseline
from models.baselines.cv_model_comparison import run_cv_model_comparison
from simons_core.io.paths import data_dir


@dataclass(frozen=True)
class Week4RunResult:
    run_prefix: str
    data_root: Path
    manifest_path: Path
    artifacts: dict[str, Path]
    statuses: dict[str, str]


def _run_id(prefix: str, step: str) -> str:
    return f"{prefix}_{step}"


def _run_step(
    idx: int,
    total: int,
    label: str,
    func: Callable[..., object],
    **kwargs: object,
) -> object:
    t0 = time.perf_counter()
    print(f"[{idx}/{total}] {label} ...")
    try:
        out = func(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"Step failed [{idx}/{total}] {label}: {exc}") from exc
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    print(f"[{idx}/{total}] {label} done ({elapsed_ms} ms)")
    return out


def _ensure_week4_prerequisites(base_data: Path, model_dataset_path: Path) -> dict[str, Path]:
    expected = {
        "trading_calendar": base_data / "reference" / "trading_calendar.parquet",
        "labels_forward": base_data / "labels" / "labels_forward.parquet",
        "model_dataset": model_dataset_path,
    }
    missing = [name for name, path in expected.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Week 4 runner requires Week 3 artifacts. "
            f"Missing: {missing}. Run the Week 3 flow first for data_root: {base_data}"
        )
    return expected


def run_week4_cv_validation(
    *,
    run_prefix: str = "week4_cv_validation",
    data_root: str | Path | None = None,
    model_dataset_path: str | Path | None = None,
    n_folds: int = 5,
    embargo_sessions: int = 1,
    regression_label_name: str = "fwd_ret_5d",
    classification_label_name: str = "fwd_dir_up_5d",
    horizon_days: int = 5,
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0),
    logistic_cs: tuple[float, ...] = (0.1, 1.0, 10.0),
    dummy_regressor_strategy: str = "mean",
    dummy_classifier_strategy: str = "prior",
    fail_on_invalid_fold: bool = False,
    comparison_strict_missing: bool = False,
    comparison_tie_tolerance: float = 1e-12,
    comparison_n_valid_tolerance_ratio: float = 0.05,
) -> Week4RunResult:
    base_data = Path(data_root).expanduser().resolve() if data_root else data_dir()
    labels_root = base_data / "labels"
    reference_root = base_data / "reference"
    model_artifacts_root = base_data / "models" / "artifacts"
    labels_root.mkdir(parents=True, exist_ok=True)
    reference_root.mkdir(parents=True, exist_ok=True)
    model_artifacts_root.mkdir(parents=True, exist_ok=True)

    dataset_path = (
        Path(model_dataset_path).expanduser().resolve()
        if model_dataset_path
        else (base_data / "datasets" / "model_dataset.parquet")
    )
    prereq = _ensure_week4_prerequisites(base_data, dataset_path)

    total_steps = 6
    step = 1
    statuses: dict[str, str] = {}

    purged_cv = _run_step(
        step,
        total_steps,
        "build purged CV folds",
        build_purged_cv,
        labels_path=prereq["labels_forward"],
        trading_calendar_path=prereq["trading_calendar"],
        output_dir=labels_root,
        n_folds=int(n_folds),
        embargo_sessions=int(embargo_sessions),
        run_id=_run_id(run_prefix, "purged_cv"),
    )
    statuses["build_purged_cv"] = "DONE"
    step += 1

    ridge_output_dir = model_artifacts_root / "ridge_cv"
    ridge_cv = _run_step(
        step,
        total_steps,
        "cross-validated baseline ridge_cv",
        run_cross_validated_baseline,
        mode="ridge_cv",
        model_dataset_path=dataset_path,
        purged_cv_folds_path=purged_cv.folds_path,
        output_dir=ridge_output_dir,
        label_name=regression_label_name,
        horizon_days=int(horizon_days),
        alpha_grid=ridge_alphas,
        fail_on_invalid_fold=bool(fail_on_invalid_fold),
        run_id=_run_id(run_prefix, "ridge_cv"),
    )
    statuses["ridge_cv"] = "DONE"
    step += 1

    logistic_output_dir = model_artifacts_root / "logistic_cv"
    logistic_cv = _run_step(
        step,
        total_steps,
        "cross-validated baseline logistic_cv",
        run_cross_validated_baseline,
        mode="logistic_cv",
        model_dataset_path=dataset_path,
        purged_cv_folds_path=purged_cv.folds_path,
        output_dir=logistic_output_dir,
        label_name=classification_label_name,
        horizon_days=int(horizon_days),
        c_grid=logistic_cs,
        fail_on_invalid_fold=bool(fail_on_invalid_fold),
        run_id=_run_id(run_prefix, "logistic_cv"),
    )
    statuses["logistic_cv"] = "DONE"
    step += 1

    dummy_reg_output_dir = model_artifacts_root / "dummy_regressor_cv"
    dummy_regressor_cv = _run_step(
        step,
        total_steps,
        "cross-validated baseline dummy_regressor_cv",
        run_cross_validated_baseline,
        mode="dummy_regressor_cv",
        model_dataset_path=dataset_path,
        purged_cv_folds_path=purged_cv.folds_path,
        output_dir=dummy_reg_output_dir,
        label_name=regression_label_name,
        horizon_days=int(horizon_days),
        dummy_strategy=dummy_regressor_strategy,
        fail_on_invalid_fold=bool(fail_on_invalid_fold),
        run_id=_run_id(run_prefix, "dummy_regressor_cv"),
    )
    statuses["dummy_regressor_cv"] = "DONE"
    step += 1

    dummy_cls_output_dir = model_artifacts_root / "dummy_classifier_cv"
    dummy_classifier_cv = _run_step(
        step,
        total_steps,
        "cross-validated baseline dummy_classifier_cv",
        run_cross_validated_baseline,
        mode="dummy_classifier_cv",
        model_dataset_path=dataset_path,
        purged_cv_folds_path=purged_cv.folds_path,
        output_dir=dummy_cls_output_dir,
        label_name=classification_label_name,
        horizon_days=int(horizon_days),
        dummy_strategy=dummy_classifier_strategy,
        fail_on_invalid_fold=bool(fail_on_invalid_fold),
        run_id=_run_id(run_prefix, "dummy_classifier_cv"),
    )
    statuses["dummy_classifier_cv"] = "DONE"
    step += 1

    comparison = _run_step(
        step,
        total_steps,
        "compare CV models by task",
        run_cv_model_comparison,
        artifacts_dir=model_artifacts_root,
        output_dir=model_artifacts_root,
        ridge_fold_metrics_path=ridge_cv.fold_metrics_path,
        ridge_summary_path=ridge_cv.summary_path,
        dummy_regressor_fold_metrics_path=dummy_regressor_cv.fold_metrics_path,
        dummy_regressor_summary_path=dummy_regressor_cv.summary_path,
        logistic_fold_metrics_path=logistic_cv.fold_metrics_path,
        logistic_summary_path=logistic_cv.summary_path,
        dummy_classifier_fold_metrics_path=dummy_classifier_cv.fold_metrics_path,
        dummy_classifier_summary_path=dummy_classifier_cv.summary_path,
        strict_missing=bool(comparison_strict_missing),
        tie_tolerance=float(comparison_tie_tolerance),
        n_valid_tolerance_ratio=float(comparison_n_valid_tolerance_ratio),
        run_id=_run_id(run_prefix, "cv_model_comparison"),
    )
    statuses["cv_model_comparison"] = "DONE"

    artifacts: dict[str, Path] = {
        "purged_cv_folds": purged_cv.folds_path,
        "purged_cv_summary": purged_cv.summary_path,
        "ridge_cv_fold_metrics": ridge_cv.fold_metrics_path,
        "ridge_cv_summary": ridge_cv.summary_path,
        "ridge_cv_predictions": ridge_cv.predictions_path if ridge_cv.predictions_path else ridge_output_dir / "cv_baseline_predictions.parquet",
        "logistic_cv_fold_metrics": logistic_cv.fold_metrics_path,
        "logistic_cv_summary": logistic_cv.summary_path,
        "logistic_cv_predictions": logistic_cv.predictions_path if logistic_cv.predictions_path else logistic_output_dir / "cv_baseline_predictions.parquet",
        "dummy_regressor_cv_fold_metrics": dummy_regressor_cv.fold_metrics_path,
        "dummy_regressor_cv_summary": dummy_regressor_cv.summary_path,
        "dummy_regressor_cv_predictions": dummy_regressor_cv.predictions_path if dummy_regressor_cv.predictions_path else dummy_reg_output_dir / "cv_baseline_predictions.parquet",
        "dummy_classifier_cv_fold_metrics": dummy_classifier_cv.fold_metrics_path,
        "dummy_classifier_cv_summary": dummy_classifier_cv.summary_path,
        "dummy_classifier_cv_predictions": dummy_classifier_cv.predictions_path if dummy_classifier_cv.predictions_path else dummy_cls_output_dir / "cv_baseline_predictions.parquet",
        "cv_model_comparison_summary": comparison.summary_path,
        "cv_model_comparison_table": comparison.table_path,
        "cv_model_comparison_fold_level": comparison.fold_level_path if comparison.fold_level_path else model_artifacts_root / "cv_model_comparison_fold_level.parquet",
    }

    manifest_payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_prefix": run_prefix,
        "data_root": str(base_data),
        "steps_total": total_steps,
        "flags": {
            "n_folds": int(n_folds),
            "embargo_sessions": int(embargo_sessions),
            "regression_label_name": regression_label_name,
            "classification_label_name": classification_label_name,
            "horizon_days": int(horizon_days),
            "ridge_alphas": [float(x) for x in ridge_alphas],
            "logistic_cs": [float(x) for x in logistic_cs],
            "dummy_regressor_strategy": dummy_regressor_strategy,
            "dummy_classifier_strategy": dummy_classifier_strategy,
            "fail_on_invalid_fold": bool(fail_on_invalid_fold),
            "comparison_strict_missing": bool(comparison_strict_missing),
            "comparison_tie_tolerance": float(comparison_tie_tolerance),
            "comparison_n_valid_tolerance_ratio": float(comparison_n_valid_tolerance_ratio),
        },
        "statuses": statuses,
        "comparison": {
            "regression_status": comparison.regression_status,
            "classification_status": comparison.classification_status,
        },
        "prerequisites": {key: str(path) for key, path in prereq.items()},
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }
    manifest_path = base_data / f"week4_cv_validation_manifest_{run_prefix}.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] Week 4 CV validation pipeline completed. Manifest: {manifest_path}")

    return Week4RunResult(
        run_prefix=run_prefix,
        data_root=base_data,
        manifest_path=manifest_path,
        artifacts=artifacts,
        statuses=statuses,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Week 4 purged multi-fold CV baseline pipeline.")
    parser.add_argument("--run-prefix", type=str, default="week4_cv_validation")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--model-dataset-path", type=str, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo-sessions", type=int, default=1)
    parser.add_argument("--regression-label-name", type=str, default="fwd_ret_5d")
    parser.add_argument("--classification-label-name", type=str, default="fwd_dir_up_5d")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--ridge-alphas", type=str, default="0.1,1.0,10.0")
    parser.add_argument("--logistic-cs", type=str, default="0.1,1.0,10.0")
    parser.add_argument("--dummy-regressor-strategy", type=str, default="mean")
    parser.add_argument("--dummy-classifier-strategy", type=str, default="prior")
    parser.add_argument("--fail-on-invalid-fold", action="store_true")
    parser.add_argument("--comparison-strict-missing", action="store_true")
    parser.add_argument("--comparison-tie-tolerance", type=float, default=1e-12)
    parser.add_argument("--comparison-n-valid-tolerance-ratio", type=float, default=0.05)
    return parser


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    items = [part.strip() for part in str(value).split(",") if part.strip()]
    if not items:
        raise ValueError("CSV float list cannot be empty.")
    return tuple(float(item) for item in items)


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_week4_cv_validation(
        run_prefix=args.run_prefix,
        data_root=args.data_root,
        model_dataset_path=args.model_dataset_path,
        n_folds=args.n_folds,
        embargo_sessions=args.embargo_sessions,
        regression_label_name=args.regression_label_name,
        classification_label_name=args.classification_label_name,
        horizon_days=args.horizon_days,
        ridge_alphas=_parse_csv_floats(args.ridge_alphas),
        logistic_cs=_parse_csv_floats(args.logistic_cs),
        dummy_regressor_strategy=args.dummy_regressor_strategy,
        dummy_classifier_strategy=args.dummy_classifier_strategy,
        fail_on_invalid_fold=args.fail_on_invalid_fold,
        comparison_strict_missing=args.comparison_strict_missing,
        comparison_tie_tolerance=args.comparison_tie_tolerance,
        comparison_n_valid_tolerance_ratio=args.comparison_n_valid_tolerance_ratio,
    )
    print("Week 4 statuses:")
    for key, value in result.statuses.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
