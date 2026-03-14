from __future__ import annotations

import json
from pathlib import Path

from models.baselines.model_comparison import compare_baseline_metrics
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


def _write_metrics_payload(
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
        "split_counts_modelable_roles": {"train": int(train_metrics["n"]), "valid": int(valid_metrics["n"]), "test": int(test_metrics["n"])},
        "metrics": {
            "train": train_metrics,
            "valid": valid_metrics,
            "test": test_metrics,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_model_comparison_mvp_comparable_regression_with_split_specific_winners(
    tmp_workspace: dict[str, Path],
) -> None:
    base = tmp_workspace["data"] / "model_comparison_case_1"
    output_dir = tmp_workspace["artifacts"] / "model_comparison_case_1"
    model_a = _write_metrics_payload(
        path=base / "ridge_a_metrics.json",
        model_name="ridge_baseline_a",
        label_name="fwd_ret_5d",
        split_name="holdout_temporal_purged",
        target_type="continuous_forward_return",
        horizon_days=5,
        train_metrics=_regression_metrics_block(n=30, mse=0.80, mae=0.70, r2=0.11, pearson_ic=0.09, spearman_ic=0.08),
        valid_metrics=_regression_metrics_block(n=10, mse=0.90, mae=0.75, r2=0.07, pearson_ic=0.06, spearman_ic=0.05),
        test_metrics=_regression_metrics_block(n=10, mse=1.20, mae=0.88, r2=0.01, pearson_ic=0.03, spearman_ic=0.02),
    )
    model_b = _write_metrics_payload(
        path=base / "ridge_b_metrics.json",
        model_name="ridge_baseline_b",
        label_name="fwd_ret_5d",
        split_name="holdout_temporal_purged",
        target_type="continuous_forward_return",
        horizon_days=5,
        train_metrics=_regression_metrics_block(n=30, mse=0.82, mae=0.71, r2=0.10, pearson_ic=0.08, spearman_ic=0.07),
        valid_metrics=_regression_metrics_block(n=10, mse=1.00, mae=0.80, r2=0.06, pearson_ic=0.05, spearman_ic=0.04),
        test_metrics=_regression_metrics_block(n=10, mse=1.10, mae=0.83, r2=0.02, pearson_ic=0.04, spearman_ic=0.03),
    )

    result = compare_baseline_metrics(
        model_a_metrics_path=model_a,
        model_b_metrics_path=model_b,
        output_dir=output_dir,
        run_id="test_model_comparison_mvp_regression",
    )

    assert result.summary_path.exists()
    assert result.table_path.exists()
    assert result.comparability_status == "comparable"
    assert result.primary_metric_used == "mse"
    assert result.winner_valid == "model_a"
    assert result.winner_test == "model_b"

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["comparability_status"] == "comparable"
    assert summary["winner_valid"] == "model_a"
    assert summary["winner_test"] == "model_b"
    assert summary["winner_valid_model_name"] == "ridge_baseline_a"
    assert summary["winner_test_model_name"] == "ridge_baseline_b"
    assert summary["primary_metric_used"] == "mse"

    table = read_parquet(result.table_path)
    assert len(table) > 0
    assert set(table["model_slot"].astype(str).unique().tolist()) == {"model_a", "model_b"}
    assert "mse" in set(table["metric_name"].astype(str).tolist())


def test_model_comparison_mvp_detects_non_comparable_label_name(
    tmp_workspace: dict[str, Path],
) -> None:
    base = tmp_workspace["data"] / "model_comparison_case_2"
    output_dir = tmp_workspace["artifacts"] / "model_comparison_case_2"
    model_a = _write_metrics_payload(
        path=base / "a_metrics.json",
        model_name="ridge_baseline",
        label_name="fwd_ret_5d",
        split_name="holdout_temporal_purged",
        target_type="continuous_forward_return",
        horizon_days=5,
        train_metrics=_regression_metrics_block(n=25, mse=0.9, mae=0.8, r2=0.1, pearson_ic=0.1, spearman_ic=0.1),
        valid_metrics=_regression_metrics_block(n=8, mse=1.0, mae=0.9, r2=0.0, pearson_ic=0.0, spearman_ic=0.0),
        test_metrics=_regression_metrics_block(n=8, mse=1.1, mae=1.0, r2=-0.1, pearson_ic=-0.1, spearman_ic=-0.1),
    )
    model_b = _write_metrics_payload(
        path=base / "b_metrics.json",
        model_name="ridge_baseline_alt",
        label_name="fwd_ret_20d",
        split_name="holdout_temporal_purged",
        target_type="continuous_forward_return",
        horizon_days=5,
        train_metrics=_regression_metrics_block(n=25, mse=0.8, mae=0.7, r2=0.2, pearson_ic=0.2, spearman_ic=0.2),
        valid_metrics=_regression_metrics_block(n=8, mse=0.9, mae=0.8, r2=0.1, pearson_ic=0.1, spearman_ic=0.1),
        test_metrics=_regression_metrics_block(n=8, mse=1.0, mae=0.9, r2=0.0, pearson_ic=0.0, spearman_ic=0.0),
    )

    result = compare_baseline_metrics(
        model_a_metrics_path=model_a,
        model_b_metrics_path=model_b,
        output_dir=output_dir,
        run_id="test_model_comparison_mvp_label_mismatch",
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["comparability_status"] == "non_comparable"
    assert summary["winner_valid"] == "non_comparable"
    assert summary["winner_test"] == "non_comparable"
    assert any("label_name mismatch" in note for note in summary["notes"])


def test_model_comparison_mvp_detects_non_comparable_split_name(
    tmp_workspace: dict[str, Path],
) -> None:
    base = tmp_workspace["data"] / "model_comparison_case_3"
    output_dir = tmp_workspace["artifacts"] / "model_comparison_case_3"
    model_a = _write_metrics_payload(
        path=base / "a_metrics.json",
        model_name="logistic_baseline_a",
        label_name="fwd_dir_up_5d",
        split_name="holdout_temporal_purged",
        target_type="binary_direction",
        horizon_days=5,
        train_metrics=_classification_metrics_block(
            n=40, log_loss=0.64, accuracy=0.58, balanced_accuracy=0.58, precision=0.57, recall=0.60, f1=0.58, roc_auc=0.61, average_precision=0.60
        ),
        valid_metrics=_classification_metrics_block(
            n=12, log_loss=0.67, accuracy=0.56, balanced_accuracy=0.56, precision=0.55, recall=0.58, f1=0.56, roc_auc=0.59, average_precision=0.58
        ),
        test_metrics=_classification_metrics_block(
            n=12, log_loss=0.69, accuracy=0.55, balanced_accuracy=0.55, precision=0.54, recall=0.57, f1=0.55, roc_auc=0.58, average_precision=0.57
        ),
    )
    model_b = _write_metrics_payload(
        path=base / "b_metrics.json",
        model_name="logistic_baseline_b",
        label_name="fwd_dir_up_5d",
        split_name="rolling_purged_v1",
        target_type="binary_direction",
        horizon_days=5,
        train_metrics=_classification_metrics_block(
            n=40, log_loss=0.63, accuracy=0.59, balanced_accuracy=0.59, precision=0.58, recall=0.61, f1=0.59, roc_auc=0.62, average_precision=0.61
        ),
        valid_metrics=_classification_metrics_block(
            n=12, log_loss=0.66, accuracy=0.57, balanced_accuracy=0.57, precision=0.56, recall=0.59, f1=0.57, roc_auc=0.60, average_precision=0.59
        ),
        test_metrics=_classification_metrics_block(
            n=12, log_loss=0.68, accuracy=0.56, balanced_accuracy=0.56, precision=0.55, recall=0.58, f1=0.56, roc_auc=0.59, average_precision=0.58
        ),
    )

    result = compare_baseline_metrics(
        model_a_metrics_path=model_a,
        model_b_metrics_path=model_b,
        output_dir=output_dir,
        run_id="test_model_comparison_mvp_split_mismatch",
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["comparability_status"] == "non_comparable"
    assert summary["winner_valid"] == "non_comparable"
    assert summary["winner_test"] == "non_comparable"
    assert any("split_name mismatch" in note for note in summary["notes"])


def test_model_comparison_mvp_handles_tie_policy(
    tmp_workspace: dict[str, Path],
) -> None:
    base = tmp_workspace["data"] / "model_comparison_case_4"
    output_dir = tmp_workspace["artifacts"] / "model_comparison_case_4"
    common_train = _classification_metrics_block(
        n=35, log_loss=0.62, accuracy=0.60, balanced_accuracy=0.60, precision=0.60, recall=0.60, f1=0.60, roc_auc=0.63, average_precision=0.62
    )
    common_valid = _classification_metrics_block(
        n=10, log_loss=0.70, accuracy=0.51, balanced_accuracy=0.51, precision=0.50, recall=0.52, f1=0.51, roc_auc=0.52, average_precision=0.51
    )
    common_test = _classification_metrics_block(
        n=10, log_loss=0.71, accuracy=0.50, balanced_accuracy=0.50, precision=0.49, recall=0.51, f1=0.50, roc_auc=0.51, average_precision=0.50
    )
    model_a = _write_metrics_payload(
        path=base / "a_metrics.json",
        model_name="logistic_a",
        label_name="fwd_dir_up_5d",
        split_name="holdout_temporal_purged",
        target_type="binary_direction",
        horizon_days=5,
        train_metrics=common_train,
        valid_metrics=common_valid,
        test_metrics=common_test,
    )
    model_b = _write_metrics_payload(
        path=base / "b_metrics.json",
        model_name="logistic_b",
        label_name="fwd_dir_up_5d",
        split_name="holdout_temporal_purged",
        target_type="binary_direction",
        horizon_days=5,
        train_metrics=common_train,
        valid_metrics=common_valid,
        test_metrics=common_test,
    )

    result = compare_baseline_metrics(
        model_a_metrics_path=model_a,
        model_b_metrics_path=model_b,
        output_dir=output_dir,
        run_id="test_model_comparison_mvp_tie",
        tie_tolerance=0.0,
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["comparability_status"] == "comparable"
    assert summary["primary_metric_used"] == "log_loss"
    assert summary["winner_valid"] == "tie"
    assert summary["winner_test"] == "tie"
    assert summary["winner_valid_model_name"] is None
    assert summary["winner_test_model_name"] is None

