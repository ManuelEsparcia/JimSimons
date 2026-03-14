from __future__ import annotations

import json
from pathlib import Path

from simons_core.io.parquet_store import read_parquet
from validation.validation_suite import run_validation_suite


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _seed_validation_inputs(
    tmp_workspace: dict[str, Path],
    *,
    leakage_status: str = "PASS",
    include_cv: bool = True,
    include_decile: bool = True,
    include_paper: bool = True,
    include_backtest: bool = True,
    positive_spread_rate: float = 0.62,
) -> dict[str, Path]:
    data_root = tmp_workspace["data"] / "validation_suite_case"
    validation_dir = data_root / "validation"
    backtest_dir = data_root / "backtest"
    models_dir = data_root / "models" / "artifacts"
    signals_dir = data_root / "signals"
    out_dir = data_root / "validation_outputs"
    for path in (validation_dir, backtest_dir, models_dir, signals_dir, out_dir):
        path.mkdir(parents=True, exist_ok=True)

    leakage_path = validation_dir / "leakage_audit_summary.json"
    cv_path = models_dir / "cv_model_comparison_summary.json"
    decile_path = signals_dir / "decile_analysis_summary.json"
    paper_path = signals_dir / "paper_portfolio_summary.json"
    backtest_path = backtest_dir / "backtest_diagnostics_summary.json"

    _write_json(
        leakage_path,
        {
            "overall_status": leakage_status,
            "n_checks_run": 10,
            "n_fail": 0 if leakage_status != "FAIL" else 1,
            "n_warn": 0,
            "n_pass": 9,
        },
    )

    if include_cv:
        _write_json(
            cv_path,
            {
                "regression": {
                    "comparability_status": "comparable",
                    "winner_global": "model_a",
                    "primary_metric": "mse",
                },
                "classification": {
                    "comparability_status": "comparable",
                    "winner_global": "model_a",
                    "primary_metric": "log_loss",
                },
            },
        )

    if include_decile:
        _write_json(
            decile_path,
            {
                "mean_top_minus_bottom_spread": 0.0032,
                "positive_spread_rate": positive_spread_rate,
                "monotonicity_score": 0.08,
                "n_dates": 22,
            },
        )

    if include_paper:
        _write_json(
            paper_path,
            {
                "mode_summaries": [
                    {"portfolio_mode": "long_only_top", "positive_return_rate": 0.56, "mean_daily_gross_return": 0.0011},
                    {"portfolio_mode": "long_short_top_bottom", "positive_return_rate": 0.63, "mean_daily_gross_return": 0.0017},
                ]
            },
        )

    if include_backtest:
        _write_json(
            backtest_path,
            {
                "max_drawdown_net_all_modes": -0.12,
                "mean_cost_drag": 0.0012,
                "total_cost_paid_all_modes": 0.0045,
            },
        )

    return {
        "leakage_path": leakage_path,
        "cv_path": cv_path,
        "decile_path": decile_path,
        "paper_path": paper_path,
        "backtest_path": backtest_path,
        "output_dir": out_dir,
    }


def test_validation_suite_generates_artifacts(tmp_workspace: dict[str, Path]) -> None:
    paths = _seed_validation_inputs(tmp_workspace)
    result = run_validation_suite(
        leakage_summary_path=paths["leakage_path"],
        backtest_diagnostics_summary_path=paths["backtest_path"],
        cv_model_comparison_summary_path=paths["cv_path"],
        decile_analysis_summary_path=paths["decile_path"],
        paper_portfolio_summary_path=paths["paper_path"],
        output_dir=paths["output_dir"],
        run_id="test_validation_suite_artifacts",
    )

    assert result.findings_path.exists()
    assert result.metrics_path.exists()
    assert result.summary_path.exists()
    assert result.findings_path.with_suffix(".parquet.manifest.json").exists()
    assert result.metrics_path.with_suffix(".parquet.manifest.json").exists()

    findings = read_parquet(result.findings_path)
    assert len(findings) > 0
    assert {"validation_block", "metric_name", "severity", "status"}.issubset(findings.columns)


def test_validation_suite_leakage_fail_forces_overall_fail(tmp_workspace: dict[str, Path]) -> None:
    paths = _seed_validation_inputs(tmp_workspace, leakage_status="FAIL")
    result = run_validation_suite(
        leakage_summary_path=paths["leakage_path"],
        backtest_diagnostics_summary_path=paths["backtest_path"],
        cv_model_comparison_summary_path=paths["cv_path"],
        decile_analysis_summary_path=paths["decile_path"],
        paper_portfolio_summary_path=paths["paper_path"],
        output_dir=paths["output_dir"],
        run_id="test_validation_suite_leakage_fail",
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["overall_status"] == "FAIL"
    assert summary["leakage_status"] == "FAIL"
    assert "leakage_integrity" in summary["failed_blocks"]


def test_validation_suite_handles_missing_inputs_explicitly(tmp_workspace: dict[str, Path]) -> None:
    paths = _seed_validation_inputs(
        tmp_workspace,
        include_cv=False,
        include_decile=False,
        include_paper=False,
        include_backtest=False,
    )
    result = run_validation_suite(
        leakage_summary_path=paths["leakage_path"],
        backtest_diagnostics_summary_path=paths["backtest_path"],
        cv_model_comparison_summary_path=paths["cv_path"],
        decile_analysis_summary_path=paths["decile_path"],
        paper_portfolio_summary_path=paths["paper_path"],
        output_dir=paths["output_dir"],
        run_id="test_validation_suite_missing_inputs",
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["overall_status"] == "WARN"
    assert "cv_comparison_robustness" in summary["warning_blocks"]
    assert "signal_quality" in summary["warning_blocks"]
    assert summary["input_artifacts"]["cv_model_comparison_summary"]["exists"] is False


def test_validation_suite_clean_case_is_pass(tmp_workspace: dict[str, Path]) -> None:
    paths = _seed_validation_inputs(tmp_workspace)
    result = run_validation_suite(
        leakage_summary_path=paths["leakage_path"],
        backtest_diagnostics_summary_path=paths["backtest_path"],
        cv_model_comparison_summary_path=paths["cv_path"],
        decile_analysis_summary_path=paths["decile_path"],
        paper_portfolio_summary_path=paths["paper_path"],
        output_dir=paths["output_dir"],
        run_id="test_validation_suite_clean_pass",
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["overall_status"] == "PASS"
    assert summary["failed_blocks"] == []


def test_validation_suite_overall_status_matches_worst_severity(tmp_workspace: dict[str, Path]) -> None:
    paths = _seed_validation_inputs(tmp_workspace, positive_spread_rate=0.10)
    result = run_validation_suite(
        leakage_summary_path=paths["leakage_path"],
        backtest_diagnostics_summary_path=paths["backtest_path"],
        cv_model_comparison_summary_path=paths["cv_path"],
        decile_analysis_summary_path=paths["decile_path"],
        paper_portfolio_summary_path=paths["paper_path"],
        output_dir=paths["output_dir"],
        run_id="test_validation_suite_signal_fail",
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["signal_quality_status"] == "FAIL"
    assert summary["overall_status"] == "FAIL"
