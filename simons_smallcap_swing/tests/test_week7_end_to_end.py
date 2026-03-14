from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_week7_validation import run_week7_validation
from simons_core.io.parquet_store import read_parquet


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
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _seed_week7_prerequisites(data_root: Path) -> dict[str, Path]:
    labels_root = data_root / "labels"
    features_root = data_root / "features"
    datasets_root = data_root / "datasets"
    reference_root = data_root / "reference"
    edgar_root = data_root / "edgar"
    models_root = data_root / "models" / "artifacts"
    universe_root = data_root / "universe"
    price_root = data_root / "price"
    signals_root = data_root / "signals"
    portfolio_root = data_root / "portfolio"
    execution_root = data_root / "execution"
    backtest_root = data_root / "backtest"
    for directory in (
        labels_root,
        features_root,
        datasets_root,
        reference_root,
        edgar_root,
        models_root,
        universe_root,
        price_root,
        signals_root,
        portfolio_root,
        execution_root,
        backtest_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    sessions = pd.bdate_range("2026-10-05", periods=10, freq="B")
    decision_dates = sessions[:6]
    instruments = [("SIMA", "AAA"), ("SIMB", "BBB")]
    split_roles = {
        0: "train",
        1: "dropped_by_purge",
        2: "valid",
        3: "test",
        4: "dropped_by_embargo",
        5: "dropped_by_embargo",
    }
    cv_roles = {
        0: "train",
        1: "dropped_by_purge",
        2: "valid",
        3: "dropped_by_embargo",
        4: "dropped_by_embargo",
        5: "dropped_by_embargo",
    }

    trading_calendar_path = reference_root / "trading_calendar.parquet"
    pd.DataFrame({"date": sessions, "is_session": True}).to_parquet(trading_calendar_path, index=False)

    labels_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []
    splits_rows: list[dict[str, object]] = []
    cv_rows: list[dict[str, object]] = []

    for idx, date in enumerate(decision_dates):
        entry_date = sessions[idx + 1]
        exit_date = sessions[idx + 2]
        split_role = split_roles[idx]
        cv_role = cv_roles[idx]
        for instrument_id, ticker in instruments:
            sign = 1.0 if instrument_id == "SIMA" else -1.0
            label_value = sign * 0.01 * float(idx + 1)
            ret_1d_lag1 = sign * 0.001 * float(idx + 1)
            ret_5d_lag1 = sign * 0.002 * float(idx + 1)
            mkt_breadth = 0.45 + 0.01 * float(idx)
            log_assets = 10.0 + (0.2 if instrument_id == "SIMB" else 0.0)

            labels_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "horizon_days": 1,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "label_name": "fwd_ret_1d",
                    "label_value": label_value,
                    "price_entry": 100.0 + idx,
                    "price_exit": 101.0 + idx,
                    "source_price_field": "close_adj",
                }
            )
            feature_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "ret_1d_lag1": ret_1d_lag1,
                    "ret_5d_lag1": ret_5d_lag1,
                    "mkt_breadth_up_lag1": mkt_breadth,
                    "log_total_assets": log_assets,
                }
            )
            dataset_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "horizon_days": 1,
                    "label_name": "fwd_ret_1d",
                    "split_name": "holdout_temporal_purged",
                    "split_role": split_role,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "target_value": label_value,
                    "target_type": "continuous_forward_return",
                    "ret_1d_lag1": ret_1d_lag1,
                    "ret_5d_lag1": ret_5d_lag1,
                    "mkt_breadth_up_lag1": mkt_breadth,
                    "log_total_assets": log_assets,
                }
            )
            splits_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "horizon_days": 1,
                    "label_name": "fwd_ret_1d",
                    "split_name": "holdout_temporal_purged",
                    "split_role": split_role,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                }
            )
            cv_rows.append(
                {
                    "fold_id": 1,
                    "date": date,
                    "instrument_id": instrument_id,
                    "horizon_days": 1,
                    "label_name": "fwd_ret_1d",
                    "split_role": cv_role,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                }
            )

    labels_path = labels_root / "labels_forward.parquet"
    features_path = features_root / "features_matrix.parquet"
    dataset_path = datasets_root / "model_dataset.parquet"
    splits_path = labels_root / "purged_splits.parquet"
    cv_path = labels_root / "purged_cv_folds.parquet"
    pd.DataFrame(labels_rows).to_parquet(labels_path, index=False)
    pd.DataFrame(feature_rows).to_parquet(features_path, index=False)
    pd.DataFrame(dataset_rows).to_parquet(dataset_path, index=False)
    pd.DataFrame(splits_rows).to_parquet(splits_path, index=False)
    pd.DataFrame(cv_rows).to_parquet(cv_path, index=False)

    (labels_root / "purged_splits.summary.json").write_text(
        json.dumps({"embargo_sessions": 1}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (labels_root / "purged_cv_folds.summary.json").write_text(
        json.dumps({"embargo_sessions": 1}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    fundamentals_path = edgar_root / "fundamentals_pit.parquet"
    fundamentals_rows = []
    for instrument_id, _ticker in instruments:
        for asof in (sessions[2], sessions[5]):
            fundamentals_rows.append(
                {
                    "instrument_id": instrument_id,
                    "asof_date": pd.Timestamp(asof).tz_localize("UTC") + pd.Timedelta(hours=23),
                    "acceptance_ts": pd.Timestamp(asof).tz_localize("UTC") + pd.Timedelta(hours=10),
                    "metric_name": "Revenues",
                    "metric_value": 100_000_000.0,
                }
            )
    pd.DataFrame(fundamentals_rows).to_parquet(fundamentals_path, index=False)

    # PBO/CSCV inputs.
    ridge_cv_path = models_root / "ridge_cv" / "cv_baseline_fold_metrics.parquet"
    dummy_reg_path = models_root / "dummy_regressor_cv" / "cv_baseline_fold_metrics.parquet"
    logistic_cv_path = models_root / "logistic_cv" / "cv_baseline_fold_metrics.parquet"
    dummy_cls_path = models_root / "dummy_classifier_cv" / "cv_baseline_fold_metrics.parquet"
    _write_fold_metrics(
        path=ridge_cv_path,
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
    _write_fold_metrics(
        path=logistic_cv_path,
        model_name="logistic_cv",
        target_type="binary_direction",
        label_name="fwd_dir_up_5d",
        primary_metric="log_loss",
        values_by_fold=[0.20, 0.20, 0.90, 0.90],
    )
    _write_fold_metrics(
        path=dummy_cls_path,
        model_name="dummy_classifier_cv",
        target_type="binary_direction",
        label_name="fwd_dir_up_5d",
        primary_metric="log_loss",
        values_by_fold=[0.70, 0.70, 0.30, 0.30],
    )

    # Compatibility placeholders from Week 1-6.
    universe_history_path = universe_root / "universe_history.parquet"
    adjusted_prices_path = price_root / "adjusted_prices.parquet"
    signals_daily_path = signals_root / "signals_daily.parquet"
    portfolio_holdings_path = portfolio_root / "portfolio_holdings.parquet"
    costs_daily_path = execution_root / "costs_daily.parquet"
    backtest_daily_path = backtest_root / "backtest_daily.parquet"
    pd.DataFrame(
        [{"date": decision_dates[0], "instrument_id": "SIMA", "ticker": "AAA", "is_eligible": True}]
    ).to_parquet(universe_history_path, index=False)
    pd.DataFrame(
        [{"date": decision_dates[0], "instrument_id": "SIMA", "ticker": "AAA", "close_adj": 100.0}]
    ).to_parquet(adjusted_prices_path, index=False)
    pd.DataFrame(
        [{"date": decision_dates[0], "instrument_id": "SIMA", "ticker": "AAA", "raw_score": 0.5}]
    ).to_parquet(signals_daily_path, index=False)
    pd.DataFrame(
        [{"date": decision_dates[0], "instrument_id": "SIMA", "ticker": "AAA", "target_weight": 1.0}]
    ).to_parquet(portfolio_holdings_path, index=False)
    pd.DataFrame(
        [{"date": decision_dates[0], "portfolio_mode": "long_only_top_n", "total_cost": 0.0005}]
    ).to_parquet(costs_daily_path, index=False)
    pd.DataFrame(
        [{"date": decision_dates[0], "portfolio_mode": "long_only_top_n", "gross_return": 0.01, "net_return": 0.0095}]
    ).to_parquet(backtest_daily_path, index=False)

    return {
        "labels_forward": labels_path,
        "features_matrix": features_path,
        "model_dataset": dataset_path,
        "trading_calendar": trading_calendar_path,
        "purged_splits": splits_path,
        "purged_cv_folds": cv_path,
        "fundamentals_pit": fundamentals_path,
        "ridge_cv": ridge_cv_path,
        "dummy_regressor_cv": dummy_reg_path,
        "logistic_cv": logistic_cv_path,
        "dummy_classifier_cv": dummy_cls_path,
        "universe_history": universe_history_path,
        "adjusted_prices": adjusted_prices_path,
        "signals_daily": signals_daily_path,
        "portfolio_holdings": portfolio_holdings_path,
        "costs_daily": costs_daily_path,
        "backtest_daily": backtest_daily_path,
    }


def test_week7_runner_end_to_end_smoke(tmp_workspace: dict[str, Path]) -> None:
    prereq = _seed_week7_prerequisites(tmp_workspace["data"])
    compatibility_counts_before = {
        key: int(len(read_parquet(path)))
        for key, path in prereq.items()
        if key
        in {
            "universe_history",
            "adjusted_prices",
            "signals_daily",
            "portfolio_holdings",
            "costs_daily",
            "backtest_daily",
        }
    }

    result = run_week7_validation(
        run_prefix="test_week7_e2e",
        data_root=tmp_workspace["data"],
        max_partitions=16,
        seed=11,
        alpha=0.05,
    )

    expected_artifacts = {
        "leakage_findings": result.artifacts["leakage_audit_findings"],
        "leakage_summary": result.artifacts["leakage_audit_summary"],
        "suite_findings": result.artifacts["validation_suite_findings"],
        "suite_summary": result.artifacts["validation_suite_summary"],
        "pbo_results": result.artifacts["pbo_cscv_results"],
        "pbo_summary": result.artifacts["pbo_cscv_summary"],
        "multiple_results": result.artifacts["multiple_testing_results"],
        "multiple_summary": result.artifacts["multiple_testing_summary"],
    }
    for path in expected_artifacts.values():
        assert path.exists()
        assert path.stat().st_size > 0

    leakage_findings = read_parquet(expected_artifacts["leakage_findings"])
    suite_findings = read_parquet(expected_artifacts["suite_findings"])
    pbo_results = read_parquet(expected_artifacts["pbo_results"])
    multiple_results = read_parquet(expected_artifacts["multiple_results"])
    assert len(leakage_findings) > 0
    assert len(suite_findings) > 0
    assert len(pbo_results) > 0
    assert len(multiple_results) > 0

    leakage_summary = json.loads(expected_artifacts["leakage_summary"].read_text(encoding="utf-8"))
    suite_summary = json.loads(expected_artifacts["suite_summary"].read_text(encoding="utf-8"))
    pbo_summary = json.loads(expected_artifacts["pbo_summary"].read_text(encoding="utf-8"))
    multiple_summary = json.loads(expected_artifacts["multiple_summary"].read_text(encoding="utf-8"))

    assert leakage_summary["overall_status"] in {"PASS", "WARN", "FAIL"}
    assert suite_summary["overall_status"] in {"PASS", "WARN", "FAIL"}
    assert pbo_summary["overall_status"] in {"PASS", "WARN", "FAIL"}
    assert multiple_summary["overall_status"] in {"PASS", "WARN", "FAIL"}

    # Optional missing summaries should degrade to WARN in validation_suite, not crash.
    assert suite_summary["overall_status"] == "WARN"
    assert "cv_comparison_robustness" in suite_summary["warning_blocks"]
    assert "signal_quality" in suite_summary["warning_blocks"]
    assert suite_summary["input_artifacts"]["cv_model_comparison_summary"]["exists"] is False
    assert suite_summary["input_artifacts"]["decile_analysis_summary"]["exists"] is False
    assert suite_summary["input_artifacts"]["paper_portfolio_summary"]["exists"] is False

    assert set(pbo_summary["tasks_evaluated"]) == {"regression_candidates", "classification_candidates"}
    assert set(multiple_results["task_name"].astype(str).unique().tolist()) == {
        "classification_candidates",
        "regression_candidates",
    }

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_prefix"] == "test_week7_e2e"
    assert manifest["statuses"]["leakage_audit"] == "DONE"
    assert manifest["statuses"]["validation_suite"] == "DONE"
    assert manifest["statuses"]["pbo_cscv"] == "DONE"
    assert manifest["statuses"]["multiple_testing"] == "DONE"

    compatibility_counts_after = {
        key: int(len(read_parquet(path)))
        for key, path in prereq.items()
        if key in compatibility_counts_before
    }
    assert compatibility_counts_after == compatibility_counts_before
