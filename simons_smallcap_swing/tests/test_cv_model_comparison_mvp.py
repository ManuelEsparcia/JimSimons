from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from models.baselines.cv_model_comparison import run_cv_model_comparison
from simons_core.io.parquet_store import read_parquet


def _write_fold_metrics(
    *,
    path: Path,
    model_name: str,
    label_name: str,
    horizon_days: int,
    target_type: str,
    primary_metric: str,
    fold_to_metric: dict[int, float],
    n_valid: int = 10,
) -> Path:
    rows: list[dict[str, object]] = []
    for fold_id, value in sorted(fold_to_metric.items()):
        rows.append(
            {
                "model_name": model_name,
                "fold_id": int(fold_id),
                "label_name": label_name,
                "horizon_days": int(horizon_days),
                "target_type": target_type,
                "status": "completed",
                "primary_metric": primary_metric,
                "valid_primary_metric": float(value),
                "n_valid": int(n_valid),
                "n_train": 20,
            }
        )
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _write_summary(
    *,
    path: Path,
    model_name: str,
    label_name: str,
    horizon_days: int,
    target_type: str,
    primary_metric: str,
    cv_method: str = "purged_kfold_full_history",
    split_name: str = "holdout_temporal_purged",
) -> Path:
    payload = {
        "model_name": model_name,
        "label_name": label_name,
        "horizon_days": horizon_days,
        "target_type": target_type,
        "primary_metric": primary_metric,
        "cv_method": cv_method,
        "split_name": split_name,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _build_valid_bundle(base: Path) -> dict[str, Path]:
    paths = {
        "ridge_fold": base / "ridge_cv" / "cv_baseline_fold_metrics.parquet",
        "ridge_summary": base / "ridge_cv" / "cv_baseline_summary.json",
        "dummy_reg_fold": base / "dummy_regressor_cv" / "cv_baseline_fold_metrics.parquet",
        "dummy_reg_summary": base / "dummy_regressor_cv" / "cv_baseline_summary.json",
        "log_fold": base / "logistic_cv" / "cv_baseline_fold_metrics.parquet",
        "log_summary": base / "logistic_cv" / "cv_baseline_summary.json",
        "dummy_cls_fold": base / "dummy_classifier_cv" / "cv_baseline_fold_metrics.parquet",
        "dummy_cls_summary": base / "dummy_classifier_cv" / "cv_baseline_summary.json",
    }

    _write_fold_metrics(
        path=paths["ridge_fold"],
        model_name="ridge_cv",
        label_name="fwd_ret_5d",
        horizon_days=5,
        target_type="continuous_forward_return",
        primary_metric="mse",
        fold_to_metric={1: 0.90, 2: 1.00, 3: 0.95},
    )
    _write_summary(
        path=paths["ridge_summary"],
        model_name="ridge_cv",
        label_name="fwd_ret_5d",
        horizon_days=5,
        target_type="continuous_forward_return",
        primary_metric="mse",
    )
    _write_fold_metrics(
        path=paths["dummy_reg_fold"],
        model_name="dummy_regressor_cv",
        label_name="fwd_ret_5d",
        horizon_days=5,
        target_type="continuous_forward_return",
        primary_metric="mse",
        fold_to_metric={1: 1.20, 2: 1.30, 3: 1.10},
    )
    _write_summary(
        path=paths["dummy_reg_summary"],
        model_name="dummy_regressor_cv",
        label_name="fwd_ret_5d",
        horizon_days=5,
        target_type="continuous_forward_return",
        primary_metric="mse",
    )
    _write_fold_metrics(
        path=paths["log_fold"],
        model_name="logistic_cv",
        label_name="fwd_dir_up_5d",
        horizon_days=5,
        target_type="binary_direction",
        primary_metric="log_loss",
        fold_to_metric={1: 0.62, 2: 0.63, 3: 0.64},
    )
    _write_summary(
        path=paths["log_summary"],
        model_name="logistic_cv",
        label_name="fwd_dir_up_5d",
        horizon_days=5,
        target_type="binary_direction",
        primary_metric="log_loss",
    )
    _write_fold_metrics(
        path=paths["dummy_cls_fold"],
        model_name="dummy_classifier_cv",
        label_name="fwd_dir_up_5d",
        horizon_days=5,
        target_type="binary_direction",
        primary_metric="log_loss",
        fold_to_metric={1: 0.70, 2: 0.71, 3: 0.72},
    )
    _write_summary(
        path=paths["dummy_cls_summary"],
        model_name="dummy_classifier_cv",
        label_name="fwd_dir_up_5d",
        horizon_days=5,
        target_type="binary_direction",
        primary_metric="log_loss",
    )
    return paths


def test_cv_model_comparison_mvp_valid_comparison_outputs_summary_table_and_fold_level(
    tmp_workspace: dict[str, Path],
) -> None:
    artifacts = tmp_workspace["data"] / "cv_compare_case_valid"
    output_dir = tmp_workspace["artifacts"] / "cv_compare_valid_output"
    paths = _build_valid_bundle(artifacts)

    result = run_cv_model_comparison(
        output_dir=output_dir,
        ridge_fold_metrics_path=paths["ridge_fold"],
        ridge_summary_path=paths["ridge_summary"],
        dummy_regressor_fold_metrics_path=paths["dummy_reg_fold"],
        dummy_regressor_summary_path=paths["dummy_reg_summary"],
        logistic_fold_metrics_path=paths["log_fold"],
        logistic_summary_path=paths["log_summary"],
        dummy_classifier_fold_metrics_path=paths["dummy_cls_fold"],
        dummy_classifier_summary_path=paths["dummy_cls_summary"],
        run_id="test_cv_model_comparison_valid",
    )

    assert result.summary_path.exists()
    assert result.table_path.exists()
    assert result.fold_level_path is not None and result.fold_level_path.exists()

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["regression"]["comparability_status"] == "comparable"
    assert summary["classification"]["comparability_status"] == "comparable"
    assert summary["regression"]["winner_global"] == "model_a"
    assert summary["classification"]["winner_global"] == "model_a"

    table = read_parquet(result.table_path)
    assert len(table) == 2
    assert set(table["task_name"].astype(str).tolist()) == {
        "regression_cv_baselines",
        "classification_cv_baselines",
    }

    fold_level = read_parquet(result.fold_level_path)
    assert len(fold_level) == 6
    assert set(fold_level["fold_winner"].astype(str).tolist()) == {"model_a"}


def test_cv_model_comparison_mvp_detects_non_comparable_label_mismatch(
    tmp_workspace: dict[str, Path],
) -> None:
    artifacts = tmp_workspace["data"] / "cv_compare_case_label_mismatch"
    output_dir = tmp_workspace["artifacts"] / "cv_compare_label_mismatch_output"
    paths = _build_valid_bundle(artifacts)

    # Break regression comparability by label mismatch.
    broken_dummy = pd.read_parquet(paths["dummy_reg_fold"]).copy()
    broken_dummy["label_name"] = "fwd_ret_10d"
    broken_dummy.to_parquet(paths["dummy_reg_fold"], index=False)

    result = run_cv_model_comparison(
        output_dir=output_dir,
        ridge_fold_metrics_path=paths["ridge_fold"],
        dummy_regressor_fold_metrics_path=paths["dummy_reg_fold"],
        logistic_fold_metrics_path=paths["log_fold"],
        dummy_classifier_fold_metrics_path=paths["dummy_cls_fold"],
        run_id="test_cv_model_comparison_label_mismatch",
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["regression"]["comparability_status"] == "non_comparable"
    assert summary["regression"]["winner_global"] == "non_comparable"
    assert any("label_name mismatch" in note for note in summary["regression"]["notes"])
    assert summary["classification"]["comparability_status"] == "comparable"


def test_cv_model_comparison_mvp_detects_non_comparable_fold_mismatch(
    tmp_workspace: dict[str, Path],
) -> None:
    artifacts = tmp_workspace["data"] / "cv_compare_case_fold_mismatch"
    output_dir = tmp_workspace["artifacts"] / "cv_compare_fold_mismatch_output"
    paths = _build_valid_bundle(artifacts)

    # Drop fold_id=3 in dummy regressor.
    broken_dummy = pd.read_parquet(paths["dummy_reg_fold"]).copy()
    broken_dummy = broken_dummy[broken_dummy["fold_id"].astype(int) != 3].copy()
    broken_dummy.to_parquet(paths["dummy_reg_fold"], index=False)

    result = run_cv_model_comparison(
        output_dir=output_dir,
        ridge_fold_metrics_path=paths["ridge_fold"],
        dummy_regressor_fold_metrics_path=paths["dummy_reg_fold"],
        logistic_fold_metrics_path=paths["log_fold"],
        dummy_classifier_fold_metrics_path=paths["dummy_cls_fold"],
        run_id="test_cv_model_comparison_fold_mismatch",
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["regression"]["comparability_status"] == "non_comparable"
    assert any("fold_id mismatch" in note for note in summary["regression"]["notes"])
    assert summary["classification"]["comparability_status"] == "comparable"


def test_cv_model_comparison_mvp_handles_tie_with_tolerance(
    tmp_workspace: dict[str, Path],
) -> None:
    artifacts = tmp_workspace["data"] / "cv_compare_case_tie"
    output_dir = tmp_workspace["artifacts"] / "cv_compare_tie_output"
    paths = _build_valid_bundle(artifacts)

    # Force regression tie by fold wins and mean delta.
    ridge = pd.read_parquet(paths["ridge_fold"]).copy()
    dummy = pd.read_parquet(paths["dummy_reg_fold"]).copy()
    ridge["valid_primary_metric"] = [1.00, 1.02, 1.00]
    dummy["valid_primary_metric"] = [1.02, 1.00, 1.00]
    ridge.to_parquet(paths["ridge_fold"], index=False)
    dummy.to_parquet(paths["dummy_reg_fold"], index=False)

    result = run_cv_model_comparison(
        output_dir=output_dir,
        ridge_fold_metrics_path=paths["ridge_fold"],
        dummy_regressor_fold_metrics_path=paths["dummy_reg_fold"],
        logistic_fold_metrics_path=paths["log_fold"],
        dummy_classifier_fold_metrics_path=paths["dummy_cls_fold"],
        tie_tolerance=1e-9,
        run_id="test_cv_model_comparison_tie",
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["regression"]["comparability_status"] == "comparable"
    assert summary["regression"]["winner_global"] == "tie"
    assert int(summary["regression"]["model_a_fold_wins"]) == 1
    assert int(summary["regression"]["model_b_fold_wins"]) == 1
    assert int(summary["regression"]["ties"]) == 1
