from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from models.baselines.run_baseline_benchmarks import run_baseline_benchmarks
from simons_core.io.parquet_store import read_parquet


def _regression_metrics_block(*, n: int, mse: float, mae: float, r2: float, pearson_ic: float, spearman_ic: float) -> dict[str, float]:
    return {
        "n": n,
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "pearson_ic": pearson_ic,
        "spearman_ic": spearman_ic,
    }


def _classification_metrics_block(
    *,
    n: int,
    log_loss: float,
    accuracy: float,
    balanced_accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    roc_auc: float,
    average_precision: float,
) -> dict[str, float]:
    return {
        "n": n,
        "log_loss": log_loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
    }


def _write_metrics(
    *,
    path: Path,
    model_name: str,
    label_name: str,
    split_name: str,
    target_type: str,
    horizon_days: int,
    train_metrics: dict[str, float],
    valid_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> Path:
    payload = {
        "model_name": model_name,
        "label_name": label_name,
        "split_name": split_name,
        "target_type": target_type,
        "horizon_days": horizon_days,
        "split_counts_modelable_roles": {
            "train": int(train_metrics["n"]),
            "valid": int(valid_metrics["n"]),
            "test": int(test_metrics["n"]),
        },
        "metrics": {
            "train": train_metrics,
            "valid": valid_metrics,
            "test": test_metrics,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _build_complete_metrics_bundle(base: Path) -> dict[str, Path]:
    paths = {
        "ridge": base / "ridge_baseline_metrics.json",
        "dummy_regressor": base / "dummy_regressor_metrics.json",
        "logistic": base / "logistic_baseline_metrics.json",
        "dummy_classifier": base / "dummy_classifier_metrics.json",
    }

    _write_metrics(
        path=paths["ridge"],
        model_name="ridge_baseline",
        label_name="fwd_ret_5d",
        split_name="holdout_temporal_purged",
        target_type="continuous_forward_return",
        horizon_days=5,
        train_metrics=_regression_metrics_block(n=30, mse=0.8, mae=0.7, r2=0.11, pearson_ic=0.1, spearman_ic=0.09),
        valid_metrics=_regression_metrics_block(n=10, mse=0.9, mae=0.8, r2=0.06, pearson_ic=0.05, spearman_ic=0.04),
        test_metrics=_regression_metrics_block(n=10, mse=1.0, mae=0.9, r2=0.02, pearson_ic=0.02, spearman_ic=0.01),
    )
    _write_metrics(
        path=paths["dummy_regressor"],
        model_name="dummy_regressor_baseline",
        label_name="fwd_ret_5d",
        split_name="holdout_temporal_purged",
        target_type="continuous_forward_return",
        horizon_days=5,
        train_metrics=_regression_metrics_block(n=30, mse=1.2, mae=1.0, r2=-0.1, pearson_ic=0.0, spearman_ic=0.0),
        valid_metrics=_regression_metrics_block(n=10, mse=1.3, mae=1.1, r2=-0.2, pearson_ic=0.0, spearman_ic=0.0),
        test_metrics=_regression_metrics_block(n=10, mse=1.4, mae=1.2, r2=-0.3, pearson_ic=0.0, spearman_ic=0.0),
    )
    _write_metrics(
        path=paths["logistic"],
        model_name="logistic_baseline",
        label_name="fwd_dir_up_5d",
        split_name="holdout_temporal_purged",
        target_type="binary_direction",
        horizon_days=5,
        train_metrics=_classification_metrics_block(
            n=30, log_loss=0.61, accuracy=0.60, balanced_accuracy=0.60, precision=0.60, recall=0.60, f1=0.60, roc_auc=0.63, average_precision=0.62
        ),
        valid_metrics=_classification_metrics_block(
            n=10, log_loss=0.64, accuracy=0.58, balanced_accuracy=0.58, precision=0.58, recall=0.58, f1=0.58, roc_auc=0.61, average_precision=0.60
        ),
        test_metrics=_classification_metrics_block(
            n=10, log_loss=0.66, accuracy=0.57, balanced_accuracy=0.57, precision=0.57, recall=0.57, f1=0.57, roc_auc=0.60, average_precision=0.59
        ),
    )
    _write_metrics(
        path=paths["dummy_classifier"],
        model_name="dummy_classifier_baseline",
        label_name="fwd_dir_up_5d",
        split_name="holdout_temporal_purged",
        target_type="binary_direction",
        horizon_days=5,
        train_metrics=_classification_metrics_block(
            n=30, log_loss=0.69, accuracy=0.50, balanced_accuracy=0.50, precision=0.50, recall=0.50, f1=0.50, roc_auc=0.50, average_precision=0.50
        ),
        valid_metrics=_classification_metrics_block(
            n=10, log_loss=0.70, accuracy=0.50, balanced_accuracy=0.50, precision=0.50, recall=0.50, f1=0.50, roc_auc=0.50, average_precision=0.50
        ),
        test_metrics=_classification_metrics_block(
            n=10, log_loss=0.71, accuracy=0.50, balanced_accuracy=0.50, precision=0.50, recall=0.50, f1=0.50, roc_auc=0.50, average_precision=0.50
        ),
    )
    return paths


def test_run_baseline_benchmarks_mvp_generates_summary_and_table(
    tmp_workspace: dict[str, Path],
) -> None:
    artifacts_dir = tmp_workspace["data"] / "baseline_artifacts_case_1"
    output_dir = tmp_workspace["artifacts"] / "baseline_benchmark_case_1"
    _ = _build_complete_metrics_bundle(artifacts_dir)

    result = run_baseline_benchmarks(
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        run_id="test_baseline_benchmark_case_1",
    )

    assert result.summary_path.exists()
    assert result.table_path.exists()
    assert result.manifest_path.exists()

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["benchmark_run_id"] == "test_baseline_benchmark_case_1"
    assert set(summary["tasks_compared"]) == {"regression_baselines", "classification_baselines"}

    regression = summary["regression"]
    classification = summary["classification"]
    assert regression["comparability_status"] == "comparable"
    assert classification["comparability_status"] == "comparable"
    assert regression["winner_valid"] == "model_a"
    assert regression["winner_test"] == "model_a"
    assert classification["winner_valid"] == "model_a"
    assert classification["winner_test"] == "model_a"

    table = read_parquet(result.table_path)
    assert len(table) == 2
    assert set(table["task_name"].astype(str).tolist()) == {"regression_baselines", "classification_baselines"}


def test_run_baseline_benchmarks_mvp_marks_missing_artifact_clearly(
    tmp_workspace: dict[str, Path],
) -> None:
    artifacts_dir = tmp_workspace["data"] / "baseline_artifacts_case_2"
    output_dir = tmp_workspace["artifacts"] / "baseline_benchmark_case_2"
    paths = _build_complete_metrics_bundle(artifacts_dir)
    paths["dummy_classifier"].unlink()

    result = run_baseline_benchmarks(
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        run_id="test_baseline_benchmark_case_2",
        strict_missing=False,
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["regression"]["comparability_status"] == "comparable"
    assert summary["classification"]["comparability_status"] == "missing"
    assert summary["classification"]["winner_valid"] == "non_comparable"
    assert any("missing metrics artifact" in note for note in summary["classification"]["notes"])

    table = read_parquet(result.table_path)
    classification_row = table.loc[table["task_name"].astype(str) == "classification_baselines"].iloc[0]
    assert str(classification_row["comparability_status"]) == "missing"


def test_run_baseline_benchmarks_mvp_does_not_mix_incompatible_tasks(
    tmp_workspace: dict[str, Path],
) -> None:
    artifacts_dir = tmp_workspace["data"] / "baseline_artifacts_case_3"
    output_dir = tmp_workspace["artifacts"] / "baseline_benchmark_case_3"
    paths = _build_complete_metrics_bundle(artifacts_dir)

    # Force a cross-task mismatch in classification by injecting regression target_type.
    broken_logistic = json.loads(paths["logistic"].read_text(encoding="utf-8"))
    broken_logistic["target_type"] = "continuous_forward_return"
    paths["logistic"].write_text(json.dumps(broken_logistic, indent=2, sort_keys=True), encoding="utf-8")

    result = run_baseline_benchmarks(
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        run_id="test_baseline_benchmark_case_3",
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["classification"]["comparability_status"] == "non_comparable"
    assert summary["classification"]["winner_valid"] == "non_comparable"
    assert summary["classification"]["winner_test"] == "non_comparable"
    assert any("task target_type mismatch" in note for note in summary["classification"]["notes"])

