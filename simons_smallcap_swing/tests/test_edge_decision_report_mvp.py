from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.edge_decision_report import run_edge_decision_report
from simons_core.io.parquet_store import read_parquet


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _seed_edge_decision_inputs(
    tmp_workspace: dict[str, Path],
    *,
    validation_overall: str = "PASS",
    leakage_status: str = "PASS",
    include_optional: bool = True,
) -> dict[str, Path]:
    root = tmp_workspace["data"] / "edge_decision_case"
    research_dir = root / "research"
    validation_dir = root / "validation"
    backtest_dir = root / "backtest"
    models_dir = root / "models" / "artifacts"
    out_dir = root / "research_out"
    for path in (research_dir, validation_dir, backtest_dir, models_dir, out_dir):
        path.mkdir(parents=True, exist_ok=True)

    feature_results_path = research_dir / "feature_ablation_results.parquet"
    label_results_path = research_dir / "label_horizon_ablation_results.parquet"
    validation_summary_path = validation_dir / "validation_suite_summary.json"
    feature_summary_path = research_dir / "feature_ablation_summary.json"
    label_summary_path = research_dir / "label_horizon_ablation_summary.json"
    pbo_summary_path = validation_dir / "pbo_cscv_summary.json"
    multiple_summary_path = validation_dir / "multiple_testing_summary.json"
    backtest_summary_path = backtest_dir / "backtest_diagnostics_summary.json"
    cv_summary_path = models_dir / "cv_model_comparison_summary.json"

    feature_rows = [
        {
            "task_name": "regression",
            "label_name": "fwd_ret_5d",
            "target_type": "continuous_forward_return",
            "feature_family": "all_features",
            "model_name": "ridge_cv",
            "primary_metric": "mse",
            "mean_valid_primary_metric": 0.08,
            "median_valid_primary_metric": 0.08,
            "std_valid_primary_metric": 0.01,
            "n_folds": 2,
            "n_features_used": 8,
            "improvement_vs_dummy": 0.02,
            "winner_vs_dummy": "model",
        },
        {
            "task_name": "classification",
            "label_name": "fwd_dir_up_5d",
            "target_type": "binary_direction",
            "feature_family": "all_features",
            "model_name": "logistic_cv",
            "primary_metric": "log_loss",
            "mean_valid_primary_metric": 0.66,
            "median_valid_primary_metric": 0.66,
            "std_valid_primary_metric": 0.01,
            "n_folds": 2,
            "n_features_used": 8,
            "improvement_vs_dummy": -0.01,
            "winner_vs_dummy": "dummy",
        },
    ]
    pd.DataFrame(feature_rows).to_parquet(feature_results_path, index=False)

    label_rows = [
        {
            "task_name": "regression",
            "label_name": "fwd_ret_5d",
            "target_type": "continuous_forward_return",
            "horizon_days": 5,
            "feature_family": "all_features",
            "model_name": "ridge_cv",
            "primary_metric": "mse",
            "mean_valid_primary_metric": 0.08,
            "median_valid_primary_metric": 0.08,
            "std_valid_primary_metric": 0.01,
            "n_folds": 2,
            "n_features_used": 8,
            "improvement_vs_dummy": 0.02,
            "winner_vs_dummy": "model",
        },
        {
            "task_name": "classification",
            "label_name": "fwd_dir_up_5d",
            "target_type": "binary_direction",
            "horizon_days": 5,
            "feature_family": "all_features",
            "model_name": "logistic_cv",
            "primary_metric": "log_loss",
            "mean_valid_primary_metric": 0.66,
            "median_valid_primary_metric": 0.66,
            "std_valid_primary_metric": 0.01,
            "n_folds": 2,
            "n_features_used": 8,
            "improvement_vs_dummy": -0.01,
            "winner_vs_dummy": "dummy",
        },
    ]
    pd.DataFrame(label_rows).to_parquet(label_results_path, index=False)

    _write_json(
        validation_summary_path,
        {
            "overall_status": validation_overall,
            "leakage_status": leakage_status,
            "cv_comparison_status": "PASS",
            "signal_quality_status": "PASS",
            "portfolio_backtest_status": "PASS",
        },
    )

    _write_json(
        feature_summary_path,
        {
            "best_family_by_task": {
                "regression": "all_features",
                "classification": "all_features",
            }
        },
    )
    _write_json(
        label_summary_path,
        {
            "best_label_by_task": {"regression": "fwd_ret_5d", "classification": "fwd_dir_up_5d"},
            "best_horizon_by_task": {"regression": 5, "classification": 5},
        },
    )

    if include_optional:
        _write_json(
            pbo_summary_path,
            {
                "overall_status": "WARN",
                "pbo_estimate_by_task": {
                    "regression_candidates": 0.20,
                    "classification_candidates": 0.60,
                },
            },
        )
        _write_json(
            multiple_summary_path,
            {
                "overall_status": "WARN",
                "task_status_by_task": {
                    "regression_candidates": "PASS",
                    "classification_candidates": "WARN",
                },
            },
        )
        _write_json(
            backtest_summary_path,
            {
                "best_mode_by_cumulative_net_return": "long_short_top_bottom",
                "max_drawdown_net_all_modes": -0.12,
                "mean_cost_drag": 0.0012,
            },
        )
        _write_json(
            cv_summary_path,
            {
                "regression": {"comparability_status": "comparable", "winner_global": "model_a"},
                "classification": {"comparability_status": "comparable", "winner_global": "model_b"},
            },
        )

    return {
        "feature_results_path": feature_results_path,
        "feature_summary_path": feature_summary_path,
        "label_results_path": label_results_path,
        "label_summary_path": label_summary_path,
        "validation_summary_path": validation_summary_path,
        "pbo_summary_path": pbo_summary_path,
        "multiple_summary_path": multiple_summary_path,
        "backtest_summary_path": backtest_summary_path,
        "cv_summary_path": cv_summary_path,
        "output_dir": out_dir,
    }


def test_edge_decision_report_promotes_strong_candidate(tmp_workspace: dict[str, Path]) -> None:
    paths = _seed_edge_decision_inputs(tmp_workspace, validation_overall="PASS", leakage_status="PASS", include_optional=True)
    result = run_edge_decision_report(
        feature_ablation_results_path=paths["feature_results_path"],
        feature_ablation_summary_path=paths["feature_summary_path"],
        label_horizon_ablation_results_path=paths["label_results_path"],
        label_horizon_ablation_summary_path=paths["label_summary_path"],
        validation_suite_summary_path=paths["validation_summary_path"],
        pbo_cscv_summary_path=paths["pbo_summary_path"],
        multiple_testing_summary_path=paths["multiple_summary_path"],
        backtest_diagnostics_summary_path=paths["backtest_summary_path"],
        cv_model_comparison_summary_path=paths["cv_summary_path"],
        output_dir=paths["output_dir"],
        run_id="test_edge_decision_promote",
    )

    assert result.candidates_path.exists()
    assert result.report_path.exists()
    assert result.summary_md_path.exists()
    assert result.candidates_path.with_suffix(".parquet.manifest.json").exists()

    candidates = read_parquet(result.candidates_path)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))

    assert len(candidates) > 0
    assert bool(candidates["promoted_flag"].any())
    assert report["recommendation_next_step"] == "try_slightly_richer_model"
    assert report["best_candidate_overall"]["winner_vs_dummy"] == "model"


def test_edge_decision_report_degrades_on_validation_fail(tmp_workspace: dict[str, Path]) -> None:
    paths = _seed_edge_decision_inputs(tmp_workspace, validation_overall="FAIL", leakage_status="FAIL", include_optional=True)
    result = run_edge_decision_report(
        feature_ablation_results_path=paths["feature_results_path"],
        feature_ablation_summary_path=paths["feature_summary_path"],
        label_horizon_ablation_results_path=paths["label_results_path"],
        label_horizon_ablation_summary_path=paths["label_summary_path"],
        validation_suite_summary_path=paths["validation_summary_path"],
        pbo_cscv_summary_path=paths["pbo_summary_path"],
        multiple_testing_summary_path=paths["multiple_summary_path"],
        backtest_diagnostics_summary_path=paths["backtest_summary_path"],
        cv_model_comparison_summary_path=paths["cv_summary_path"],
        output_dir=paths["output_dir"],
        run_id="test_edge_decision_validation_fail",
    )

    candidates = read_parquet(result.candidates_path)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))

    assert not bool(candidates["promoted_flag"].any())
    assert report["overall_research_status"] == "FAIL"
    assert report["recommendation_next_step"] == "pause_and_rethink"
    assert len(report["candidates_failing_validation"]) >= 1


def test_edge_decision_report_missing_optional_inputs_warns_without_crash(
    tmp_workspace: dict[str, Path],
) -> None:
    paths = _seed_edge_decision_inputs(tmp_workspace, validation_overall="PASS", leakage_status="PASS", include_optional=False)
    result = run_edge_decision_report(
        feature_ablation_results_path=paths["feature_results_path"],
        feature_ablation_summary_path=paths["feature_summary_path"],
        label_horizon_ablation_results_path=paths["label_results_path"],
        label_horizon_ablation_summary_path=paths["label_summary_path"],
        validation_suite_summary_path=paths["validation_summary_path"],
        pbo_cscv_summary_path=paths["pbo_summary_path"],  # intentionally missing
        multiple_testing_summary_path=paths["multiple_summary_path"],  # intentionally missing
        backtest_diagnostics_summary_path=paths["backtest_summary_path"],  # intentionally missing
        cv_model_comparison_summary_path=paths["cv_summary_path"],  # intentionally missing
        output_dir=paths["output_dir"],
        run_id="test_edge_decision_missing_optional",
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    candidates = read_parquet(result.candidates_path)

    assert len(candidates) > 0
    assert len(report["missing_inputs"]) >= 3
    assert report["overall_research_status"] == "WARN"
    assert report["recommendation_next_step"] in {
        "try_slightly_richer_model",
        "improve_features_or_labels",
        "pause_and_rethink",
    }
