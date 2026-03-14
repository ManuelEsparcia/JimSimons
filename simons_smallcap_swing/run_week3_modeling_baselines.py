from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time
from typing import Callable

# Allow direct script execution: `python simons_smallcap_swing/run_week3_modeling_baselines.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets.build_model_dataset import build_model_dataset
from features.build_features import build_features
from labels.build_labels import build_labels
from labels.purged_splits import build_purged_splits
from models.baselines.run_baseline_benchmarks import run_baseline_benchmarks
from models.baselines.train_dummy_baselines import train_dummy_baseline
from models.baselines.train_logistic import train_logistic_baseline
from models.baselines.train_ridge import train_ridge_baseline
from simons_core.io.paths import data_dir


@dataclass(frozen=True)
class Week3RunResult:
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


def _build_dirs(base_data: Path) -> dict[str, Path]:
    paths = {
        "reference": base_data / "reference",
        "universe": base_data / "universe",
        "price": base_data / "price",
        "edgar": base_data / "edgar",
        "labels": base_data / "labels",
        "features": base_data / "features",
        "datasets": base_data / "datasets",
        "datasets_regression": base_data / "datasets" / "regression",
        "datasets_classification": base_data / "datasets" / "classification",
        "model_artifacts": base_data / "models" / "artifacts",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _ensure_week3_prerequisites(base_data: Path) -> dict[str, Path]:
    expected = {
        "trading_calendar": base_data / "reference" / "trading_calendar.parquet",
        "universe_history": base_data / "universe" / "universe_history.parquet",
        "adjusted_prices": base_data / "price" / "adjusted_prices.parquet",
        "market_proxies": base_data / "price" / "market_proxies.parquet",
        "fundamentals_pit": base_data / "edgar" / "fundamentals_pit.parquet",
    }
    missing = [name for name, path in expected.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Week 3 runner requires Week 1/2 artifacts. "
            f"Missing: {missing}. Populate data_root first: {base_data}"
        )
    return expected


def run_week3_modeling_baselines(
    *,
    run_prefix: str = "week3_modeling_baselines",
    data_root: str | Path | None = None,
    include_binary_direction_labels: bool = True,
    allow_missing_binary_label: bool = False,
    benchmark_strict_missing: bool = False,
) -> Week3RunResult:
    base_data = Path(data_root).expanduser().resolve() if data_root else data_dir()
    paths = _build_dirs(base_data)
    prereq = _ensure_week3_prerequisites(base_data)

    total_steps = 8
    step = 1
    statuses: dict[str, str] = {}

    labels = _run_step(
        step,
        total_steps,
        "build labels",
        build_labels,
        adjusted_prices_path=prereq["adjusted_prices"],
        universe_history_path=prereq["universe_history"],
        trading_calendar_path=prereq["trading_calendar"],
        output_dir=paths["labels"],
        include_binary_direction=bool(include_binary_direction_labels),
        run_id=_run_id(run_prefix, "labels"),
    )
    statuses["labels"] = "DONE"
    step += 1

    features = _run_step(
        step,
        total_steps,
        "build features",
        build_features,
        adjusted_prices_path=prereq["adjusted_prices"],
        universe_history_path=prereq["universe_history"],
        market_proxies_path=prereq["market_proxies"],
        fundamentals_pit_path=prereq["fundamentals_pit"],
        trading_calendar_path=prereq["trading_calendar"],
        output_dir=paths["features"],
        run_id=_run_id(run_prefix, "features"),
    )
    statuses["features"] = "DONE"
    step += 1

    purged = _run_step(
        step,
        total_steps,
        "build purged splits",
        build_purged_splits,
        labels_path=labels.labels_path,
        trading_calendar_path=prereq["trading_calendar"],
        output_dir=paths["labels"],
        run_id=_run_id(run_prefix, "purged_splits"),
    )
    statuses["purged_splits"] = "DONE"
    step += 1

    datasets_regression = _run_step(
        step,
        total_steps,
        "build model dataset (regression)",
        build_model_dataset,
        features_path=features.features_path,
        labels_path=labels.labels_path,
        purged_splits_path=purged.splits_path,
        output_dir=paths["datasets_regression"],
        label_names=("fwd_ret_5d",),
        target_type="continuous_forward_return",
        run_id=_run_id(run_prefix, "model_dataset_regression"),
    )
    statuses["model_dataset_regression"] = "DONE"

    dataset_classification_path = paths["datasets_classification"] / "model_dataset.parquet"
    dataset_classification_summary_path = paths["datasets_classification"] / "model_dataset.summary.json"
    classification_dataset_built = False
    try:
        datasets_classification = build_model_dataset(
            features_path=features.features_path,
            labels_path=labels.labels_path,
            purged_splits_path=purged.splits_path,
            output_dir=paths["datasets_classification"],
            label_names=("fwd_dir_up_5d",),
            target_type="binary_direction",
            run_id=_run_id(run_prefix, "model_dataset_classification"),
        )
        classification_dataset_built = True
        statuses["model_dataset_classification"] = "DONE"
        dataset_classification_path = datasets_classification.dataset_path
        dataset_classification_summary_path = datasets_classification.summary_path
    except Exception as exc:
        if allow_missing_binary_label:
            statuses["model_dataset_classification"] = f"SKIPPED: {exc}"
            print(f"[4/{total_steps}] build model dataset (classification) skipped: {exc}")
        else:
            raise RuntimeError(
                "Step failed [4/8] build model dataset (classification): "
                f"{exc}"
            ) from exc
    step += 1

    ridge = _run_step(
        step,
        total_steps,
        "train ridge baseline",
        train_ridge_baseline,
        model_dataset_path=datasets_regression.dataset_path,
        output_dir=paths["model_artifacts"],
        label_name="fwd_ret_5d",
        split_name="holdout_temporal_purged",
        run_id=_run_id(run_prefix, "ridge"),
    )
    statuses["train_ridge"] = "DONE"
    step += 1

    logistic_metrics_path = paths["model_artifacts"] / "logistic_baseline_metrics.json"
    logistic_predictions_path = paths["model_artifacts"] / "logistic_baseline_predictions.parquet"
    logistic_status = "SKIPPED"
    dummy_classifier_metrics_path = paths["model_artifacts"] / "dummy_classifier_metrics.json"
    dummy_classifier_predictions_path = paths["model_artifacts"] / "dummy_classifier_predictions.parquet"
    dummy_classifier_status = "SKIPPED"

    if classification_dataset_built:
        try:
            logistic = _run_step(
                step,
                total_steps,
                "train logistic baseline",
                train_logistic_baseline,
                model_dataset_path=dataset_classification_path,
                output_dir=paths["model_artifacts"],
                label_name="fwd_dir_up_5d",
                split_name="holdout_temporal_purged",
                run_id=_run_id(run_prefix, "logistic"),
            )
            logistic_metrics_path = logistic.metrics_path
            logistic_predictions_path = logistic.predictions_path
            logistic_status = "DONE"
        except Exception as exc:
            if allow_missing_binary_label:
                logistic_status = f"SKIPPED: {exc}"
                print(f"[{step}/{total_steps}] train logistic baseline skipped: {exc}")
            else:
                raise
    else:
        print(f"[{step}/{total_steps}] train logistic baseline skipped: classification dataset unavailable")
    statuses["train_logistic"] = logistic_status
    step += 1

    dummy_regressor = _run_step(
        step,
        total_steps,
        "train dummy baselines",
        train_dummy_baseline,
        mode="dummy_regressor",
        model_dataset_path=datasets_regression.dataset_path,
        output_dir=paths["model_artifacts"],
        label_name="fwd_ret_5d",
        split_name="holdout_temporal_purged",
        run_id=_run_id(run_prefix, "dummy_regressor"),
    )
    dummy_classifier_metrics = paths["model_artifacts"] / "dummy_classifier_metrics.json"
    dummy_classifier_predictions = paths["model_artifacts"] / "dummy_classifier_predictions.parquet"
    if classification_dataset_built:
        try:
            dummy_classifier = train_dummy_baseline(
                mode="dummy_classifier",
                model_dataset_path=dataset_classification_path,
                output_dir=paths["model_artifacts"],
                label_name="fwd_dir_up_5d",
                split_name="holdout_temporal_purged",
                run_id=_run_id(run_prefix, "dummy_classifier"),
            )
            dummy_classifier_metrics = dummy_classifier.metrics_path
            dummy_classifier_predictions = dummy_classifier.predictions_path
            dummy_classifier_status = "DONE"
        except Exception as exc:
            if allow_missing_binary_label:
                dummy_classifier_status = f"SKIPPED: {exc}"
                print(f"[{step}/{total_steps}] train dummy_classifier skipped: {exc}")
            else:
                raise
    else:
        dummy_classifier_status = "SKIPPED: classification dataset unavailable"
        print(f"[{step}/{total_steps}] train dummy_classifier skipped: classification dataset unavailable")
    statuses["train_dummy_regressor"] = "DONE"
    statuses["train_dummy_classifier"] = dummy_classifier_status
    dummy_classifier_metrics_path = dummy_classifier_metrics
    dummy_classifier_predictions_path = dummy_classifier_predictions
    step += 1

    benchmark = _run_step(
        step,
        total_steps,
        "run baseline benchmarks",
        run_baseline_benchmarks,
        output_dir=paths["model_artifacts"],
        ridge_metrics_path=ridge.metrics_path,
        dummy_regressor_metrics_path=dummy_regressor.metrics_path,
        logistic_metrics_path=logistic_metrics_path,
        dummy_classifier_metrics_path=dummy_classifier_metrics_path,
        strict_missing=benchmark_strict_missing,
        run_id=_run_id(run_prefix, "baseline_benchmark"),
    )
    statuses["baseline_benchmark"] = "DONE"

    artifacts: dict[str, Path] = {
        "labels_forward": labels.labels_path,
        "labels_summary": labels.summary_path,
        "features_matrix": features.features_path,
        "features_summary": features.summary_path,
        "purged_splits": purged.splits_path,
        "purged_splits_summary": purged.summary_path,
        "model_dataset_regression": datasets_regression.dataset_path,
        "model_dataset_regression_summary": datasets_regression.summary_path,
        "model_dataset_classification": dataset_classification_path,
        "model_dataset_classification_summary": dataset_classification_summary_path,
        "ridge_metrics": ridge.metrics_path,
        "ridge_predictions": ridge.predictions_path,
        "dummy_regressor_metrics": dummy_regressor.metrics_path,
        "dummy_regressor_predictions": dummy_regressor.predictions_path,
        "logistic_metrics": logistic_metrics_path,
        "logistic_predictions": logistic_predictions_path,
        "dummy_classifier_metrics": dummy_classifier_metrics_path,
        "dummy_classifier_predictions": dummy_classifier_predictions_path,
        "baseline_benchmark_summary": benchmark.summary_path,
        "baseline_benchmark_table": benchmark.table_path,
        "baseline_benchmark_manifest": benchmark.manifest_path,
    }

    manifest_payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_prefix": run_prefix,
        "data_root": str(base_data),
        "steps_total": total_steps,
        "flags": {
            "include_binary_direction_labels": bool(include_binary_direction_labels),
            "allow_missing_binary_label": bool(allow_missing_binary_label),
            "benchmark_strict_missing": bool(benchmark_strict_missing),
        },
        "statuses": statuses,
        "benchmarks": {
            "regression_status": benchmark.regression_status,
            "classification_status": benchmark.classification_status,
        },
        "prerequisites": {key: str(path) for key, path in prereq.items()},
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }
    manifest_path = base_data / f"week3_modeling_baselines_manifest_{run_prefix}.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] Week 3 modeling baseline pipeline completed. Manifest: {manifest_path}")

    return Week3RunResult(
        run_prefix=run_prefix,
        data_root=base_data,
        manifest_path=manifest_path,
        artifacts=artifacts,
        statuses=statuses,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Week 3 baseline modeling pipeline end-to-end.")
    parser.add_argument("--run-prefix", type=str, default="week3_modeling_baselines")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument(
        "--include-binary-direction-labels",
        action="store_true",
        help="Build binary direction labels (fwd_dir_up_*). Default: enabled in runner logic.",
    )
    parser.add_argument(
        "--disable-binary-direction-labels",
        action="store_true",
        help="Disable binary direction labels generation.",
    )
    parser.add_argument(
        "--allow-missing-binary-label",
        action="store_true",
        help="Skip classification baseline steps if binary label/dataset is unavailable.",
    )
    parser.add_argument(
        "--benchmark-strict-missing",
        action="store_true",
        help="Fail benchmark suite when any required metrics artifact is missing.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    include_binary = True
    if args.disable_binary_direction_labels:
        include_binary = False
    elif args.include_binary_direction_labels:
        include_binary = True

    result = run_week3_modeling_baselines(
        run_prefix=args.run_prefix,
        data_root=args.data_root,
        include_binary_direction_labels=include_binary,
        allow_missing_binary_label=args.allow_missing_binary_label,
        benchmark_strict_missing=args.benchmark_strict_missing,
    )
    print("Week 3 statuses:")
    for key, value in result.statuses.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
