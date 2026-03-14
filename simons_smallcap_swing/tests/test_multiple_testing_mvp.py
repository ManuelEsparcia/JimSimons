from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from simons_core.io.parquet_store import read_parquet
from validation.multiple_testing import run_multiple_testing


def _write_candidate_tests(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_multiple_testing_generates_artifacts(tmp_workspace: dict[str, Path]) -> None:
    root = tmp_workspace["data"] / "multiple_testing_case"
    input_path = root / "candidate_tests.parquet"
    output_dir = root / "validation"
    rows = [
        {
            "task_name": "regression_candidates",
            "candidate_name": "ridge_cv",
            "metric_name": "mse",
            "raw_score": 0.10,
            "raw_pvalue": 0.01,
        },
        {
            "task_name": "regression_candidates",
            "candidate_name": "dummy_regressor_cv",
            "metric_name": "mse",
            "raw_score": 0.20,
            "raw_pvalue": 0.20,
        },
    ]
    _write_candidate_tests(input_path, rows)

    result = run_multiple_testing(
        candidate_tests_path=input_path,
        output_dir=output_dir,
        run_id="test_multiple_testing_artifacts",
    )

    assert result.results_path.exists()
    assert result.metrics_path.exists()
    assert result.summary_path.exists()
    assert result.results_path.with_suffix(".parquet.manifest.json").exists()
    assert result.metrics_path.with_suffix(".parquet.manifest.json").exists()

    results = read_parquet(result.results_path)
    assert len(results) == 2
    assert {"task_name", "candidate_name", "adjusted_pvalue_bonferroni", "adjusted_pvalue_bh"}.issubset(
        results.columns
    )


def test_multiple_testing_bonferroni_formula(tmp_workspace: dict[str, Path]) -> None:
    root = tmp_workspace["data"] / "multiple_testing_bonf"
    input_path = root / "candidate_tests.parquet"
    output_dir = root / "validation"
    rows = [
        {"task_name": "regression_candidates", "candidate_name": "a", "metric_name": "mse", "raw_score": 0.10, "raw_pvalue": 0.01},
        {"task_name": "regression_candidates", "candidate_name": "b", "metric_name": "mse", "raw_score": 0.20, "raw_pvalue": 0.03},
        {"task_name": "regression_candidates", "candidate_name": "c", "metric_name": "mse", "raw_score": 0.30, "raw_pvalue": 0.20},
    ]
    _write_candidate_tests(input_path, rows)

    result = run_multiple_testing(
        candidate_tests_path=input_path,
        output_dir=output_dir,
        run_id="test_multiple_testing_bonf",
    )
    out = read_parquet(result.results_path).set_index("candidate_name")
    assert np.isclose(float(out.loc["a", "adjusted_pvalue_bonferroni"]), 0.03, atol=1e-12)
    assert np.isclose(float(out.loc["b", "adjusted_pvalue_bonferroni"]), 0.09, atol=1e-12)
    assert np.isclose(float(out.loc["c", "adjusted_pvalue_bonferroni"]), 0.60, atol=1e-12)


def test_multiple_testing_bh_formula(tmp_workspace: dict[str, Path]) -> None:
    root = tmp_workspace["data"] / "multiple_testing_bh"
    input_path = root / "candidate_tests.parquet"
    output_dir = root / "validation"
    rows = [
        {"task_name": "regression_candidates", "candidate_name": "a", "metric_name": "mse", "raw_score": 0.10, "raw_pvalue": 0.01},
        {"task_name": "regression_candidates", "candidate_name": "b", "metric_name": "mse", "raw_score": 0.20, "raw_pvalue": 0.03},
        {"task_name": "regression_candidates", "candidate_name": "c", "metric_name": "mse", "raw_score": 0.30, "raw_pvalue": 0.20},
    ]
    _write_candidate_tests(input_path, rows)

    result = run_multiple_testing(
        candidate_tests_path=input_path,
        output_dir=output_dir,
        run_id="test_multiple_testing_bh",
    )
    out = read_parquet(result.results_path).set_index("candidate_name")
    assert np.isclose(float(out.loc["a", "adjusted_pvalue_bh"]), 0.03, atol=1e-12)
    assert np.isclose(float(out.loc["b", "adjusted_pvalue_bh"]), 0.045, atol=1e-12)
    assert np.isclose(float(out.loc["c", "adjusted_pvalue_bh"]), 0.20, atol=1e-12)


def test_multiple_testing_without_pvalues_is_warn(tmp_workspace: dict[str, Path]) -> None:
    root = tmp_workspace["data"] / "multiple_testing_no_p"
    input_path = root / "candidate_tests.parquet"
    output_dir = root / "validation"
    rows = [
        {
            "task_name": "classification_candidates",
            "candidate_name": "logistic_cv",
            "metric_name": "log_loss",
            "raw_score": 0.65,
            "raw_pvalue": None,
        },
        {
            "task_name": "classification_candidates",
            "candidate_name": "dummy_classifier_cv",
            "metric_name": "log_loss",
            "raw_score": 0.70,
            "raw_pvalue": None,
        },
    ]
    _write_candidate_tests(input_path, rows)

    result = run_multiple_testing(
        candidate_tests_path=input_path,
        output_dir=output_dir,
        run_id="test_multiple_testing_no_p",
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["overall_status"] == "WARN"
    assert "classification_candidates" in summary["missing_pvalue_blocks"]

    out = read_parquet(result.results_path)
    assert set(out["testing_status"].astype(str).unique().tolist()) == {"heuristic_only_no_pvalue"}


def test_multiple_testing_overall_status_coherent(tmp_workspace: dict[str, Path]) -> None:
    root = tmp_workspace["data"] / "multiple_testing_status"
    input_path = root / "candidate_tests.parquet"
    output_dir = root / "validation"
    rows = [
        {"task_name": "regression_candidates", "candidate_name": "ridge_cv", "metric_name": "mse", "raw_score": 0.10, "raw_pvalue": 0.001},
        {"task_name": "regression_candidates", "candidate_name": "dummy_regressor_cv", "metric_name": "mse", "raw_score": 0.20, "raw_pvalue": 0.20},
        {"task_name": "classification_candidates", "candidate_name": "logistic_cv", "metric_name": "log_loss", "raw_score": 0.65, "raw_pvalue": None},
        {"task_name": "classification_candidates", "candidate_name": "dummy_classifier_cv", "metric_name": "log_loss", "raw_score": 0.70, "raw_pvalue": None},
    ]
    _write_candidate_tests(input_path, rows)

    result = run_multiple_testing(
        candidate_tests_path=input_path,
        output_dir=output_dir,
        run_id="test_multiple_testing_status",
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["task_status_by_task"]["regression_candidates"] in {"PASS", "WARN"}
    assert summary["task_status_by_task"]["classification_candidates"] == "WARN"
    assert summary["overall_status"] == "WARN"
