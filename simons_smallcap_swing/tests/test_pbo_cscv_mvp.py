from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from simons_core.io.parquet_store import read_parquet
from validation.pbo_cscv import run_pbo_cscv


def _write_fold_metrics(
    *,
    path: Path,
    model_name: str,
    target_type: str,
    label_name: str,
    primary_metric: str,
    values_by_fold: list[float],
) -> None:
    rows = []
    for idx, value in enumerate(values_by_fold, start=1):
        rows.append(
            {
                "model_name": model_name,
                "fold_id": idx,
                "label_name": label_name,
                "horizon_days": 5,
                "target_type": target_type,
                "primary_metric": primary_metric,
                "valid_primary_metric": float(value),
                "status": "completed",
            }
        )
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _seed_case(
    tmp_workspace: dict[str, Path],
    *,
    include_classification: bool = True,
) -> dict[str, Path]:
    root = tmp_workspace["data"] / "pbo_case"
    artifacts = root / "models" / "artifacts"
    output_dir = root / "validation"
    for directory in (artifacts, output_dir):
        directory.mkdir(parents=True, exist_ok=True)

    ridge_path = artifacts / "ridge_cv" / "cv_baseline_fold_metrics.parquet"
    dummy_reg_path = artifacts / "dummy_regressor_cv" / "cv_baseline_fold_metrics.parquet"
    logistic_path = artifacts / "logistic_cv" / "cv_baseline_fold_metrics.parquet"
    dummy_cls_path = artifacts / "dummy_classifier_cv" / "cv_baseline_fold_metrics.parquet"

    # Crafted so IS winner frequently degrades OOS.
    _write_fold_metrics(
        path=ridge_path,
        model_name="ridge_cv",
        target_type="continuous_forward_return",
        label_name="fwd_ret_5d",
        primary_metric="mse",
        values_by_fold=[0.01, 0.01, 0.25, 0.25],
    )
    _write_fold_metrics(
        path=dummy_reg_path,
        model_name="dummy_regressor_cv",
        target_type="continuous_forward_return",
        label_name="fwd_ret_5d",
        primary_metric="mse",
        values_by_fold=[0.09, 0.09, 0.05, 0.05],
    )

    if include_classification:
        _write_fold_metrics(
            path=logistic_path,
            model_name="logistic_cv",
            target_type="binary_direction",
            label_name="fwd_dir_up_5d",
            primary_metric="log_loss",
            values_by_fold=[0.2, 0.2, 0.9, 0.9],
        )
        _write_fold_metrics(
            path=dummy_cls_path,
            model_name="dummy_classifier_cv",
            target_type="binary_direction",
            label_name="fwd_dir_up_5d",
            primary_metric="log_loss",
            values_by_fold=[0.7, 0.7, 0.3, 0.3],
        )

    return {
        "ridge_path": ridge_path,
        "dummy_reg_path": dummy_reg_path,
        "logistic_path": logistic_path,
        "dummy_cls_path": dummy_cls_path,
        "output_dir": output_dir,
    }


def test_pbo_cscv_generates_artifacts(tmp_workspace: dict[str, Path]) -> None:
    paths = _seed_case(tmp_workspace)
    result = run_pbo_cscv(
        ridge_cv_path=paths["ridge_path"],
        dummy_regressor_cv_path=paths["dummy_reg_path"],
        logistic_cv_path=paths["logistic_path"],
        dummy_classifier_cv_path=paths["dummy_cls_path"],
        output_dir=paths["output_dir"],
        max_partitions=16,
        seed=7,
        run_id="test_pbo_artifacts",
    )

    assert result.results_path.exists()
    assert result.partitions_path.exists()
    assert result.summary_path.exists()
    assert result.results_path.with_suffix(".parquet.manifest.json").exists()
    assert result.partitions_path.with_suffix(".parquet.manifest.json").exists()

    results = read_parquet(result.results_path)
    assert len(results) > 0
    assert {"task_name", "partition_id", "candidate_name", "oos_percentile"}.issubset(results.columns)


def test_pbo_cscv_keeps_tasks_separate(tmp_workspace: dict[str, Path]) -> None:
    paths = _seed_case(tmp_workspace)
    result = run_pbo_cscv(
        ridge_cv_path=paths["ridge_path"],
        dummy_regressor_cv_path=paths["dummy_reg_path"],
        logistic_cv_path=paths["logistic_path"],
        dummy_classifier_cv_path=paths["dummy_cls_path"],
        output_dir=paths["output_dir"],
        max_partitions=16,
        seed=11,
        run_id="test_pbo_tasks",
    )
    results = read_parquet(result.results_path)
    tasks = sorted(results["task_name"].astype(str).unique().tolist())
    assert tasks == ["classification_candidates", "regression_candidates"]


def test_pbo_cscv_flags_overfit_in_fabricated_case(tmp_workspace: dict[str, Path]) -> None:
    paths = _seed_case(tmp_workspace)
    result = run_pbo_cscv(
        ridge_cv_path=paths["ridge_path"],
        dummy_regressor_cv_path=paths["dummy_reg_path"],
        logistic_cv_path=paths["logistic_path"],
        dummy_classifier_cv_path=paths["dummy_cls_path"],
        output_dir=paths["output_dir"],
        max_partitions=16,
        seed=3,
        run_id="test_pbo_overfit_flag",
    )
    results = read_parquet(result.results_path)
    winners = results[results["is_in_sample_winner"]]
    assert len(winners) > 0
    assert winners["overfit_flag"].fillna(0).astype(int).sum() >= 1


def test_pbo_cscv_summary_has_consumable_estimates(tmp_workspace: dict[str, Path]) -> None:
    paths = _seed_case(tmp_workspace)
    result = run_pbo_cscv(
        ridge_cv_path=paths["ridge_path"],
        dummy_regressor_cv_path=paths["dummy_reg_path"],
        logistic_cv_path=paths["logistic_path"],
        dummy_classifier_cv_path=paths["dummy_cls_path"],
        output_dir=paths["output_dir"],
        max_partitions=16,
        seed=5,
        run_id="test_pbo_summary",
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert "pbo_estimate_by_task" in summary
    for task, value in summary["pbo_estimate_by_task"].items():
        if value is not None:
            assert 0.0 <= float(value) <= 1.0, f"{task} pbo_estimate out of range"


def test_pbo_cscv_handles_missing_inputs_explicitly(tmp_workspace: dict[str, Path]) -> None:
    paths = _seed_case(tmp_workspace, include_classification=False)
    result = run_pbo_cscv(
        ridge_cv_path=paths["ridge_path"],
        dummy_regressor_cv_path=paths["dummy_reg_path"],
        logistic_cv_path=paths["logistic_path"],  # missing on purpose
        dummy_classifier_cv_path=paths["dummy_cls_path"],  # missing on purpose
        output_dir=paths["output_dir"],
        max_partitions=16,
        seed=13,
        run_id="test_pbo_missing_inputs",
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert "notes" in summary and len(summary["notes"]) > 0
    assert summary["task_status_by_task"]["classification_candidates"] == "WARN"
    # regression still should run and keep outputs consumable
    assert summary["task_status_by_task"]["regression_candidates"] in {"PASS", "WARN", "FAIL"}
