from __future__ import annotations

import json
from pathlib import Path
from datetime import UTC, datetime

import pandas as pd

from run_week8_research_refinement import run_week8_research_refinement
from simons_core.io.parquet_store import read_parquet


def _seed_week8_prerequisites(data_root: Path) -> dict[str, Path]:
    datasets_root = data_root / "datasets"
    labels_root = data_root / "labels"
    features_root = data_root / "features"
    validation_root = data_root / "validation"
    research_root = data_root / "research"
    for directory in (datasets_root, labels_root, features_root, validation_root, research_root):
        directory.mkdir(parents=True, exist_ok=True)
    (datasets_root / "regression_h20").mkdir(parents=True, exist_ok=True)

    sessions = pd.bdate_range("2026-01-05", periods=40, freq="B")
    decision_dates = sessions[:12]
    instruments = [("SIMA", "AAA", 1.0), ("SIMB", "BBB", -1.0), ("SIMC", "CCC", 0.6)]
    horizons = (1, 5, 20)

    feature_rows: list[dict[str, object]] = []
    labels_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []
    h20_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for idx, date in enumerate(decision_dates):
        for inst_idx, (instrument_id, ticker, sign) in enumerate(instruments):
            ret_1d_lag1 = sign * (0.002 + (0.0005 * idx))
            ret_5d_lag1 = sign * (0.004 + (0.0007 * idx))
            vol_20d = 0.18 + (0.01 * (idx % 4)) + (0.005 * inst_idx)
            mkt_breadth_up_lag1 = 0.45 + (0.02 if idx >= 6 else -0.01) + (0.01 * (inst_idx - 1))
            mkt_cross_sectional_vol_lag1 = 0.22 + (0.03 if inst_idx == 2 else 0.0) + (0.005 * (idx % 3))
            log_dollar_volume_lag1 = 10.1 + (0.2 * inst_idx) + (0.03 * idx)
            log_total_assets = 8.5 + (0.3 * inst_idx)

            feature_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "ret_1d_lag1": ret_1d_lag1,
                    "ret_5d_lag1": ret_5d_lag1,
                    "vol_20d": vol_20d,
                    "mkt_breadth_up_lag1": mkt_breadth_up_lag1,
                    "mkt_cross_sectional_vol_lag1": mkt_cross_sectional_vol_lag1,
                    "log_dollar_volume_lag1": log_dollar_volume_lag1,
                    "log_total_assets": log_total_assets,
                }
            )

            for horizon in horizons:
                entry_date = sessions[idx + 1]
                exit_date = sessions[idx + horizon]
                base_ret = sign * 0.0025 * float(horizon) + (0.0004 * idx) + (0.0002 * (inst_idx - 1))
                label_ret = base_ret
                label_dir = 1 if label_ret > 0 else 0

                labels_rows.append(
                    {
                        "date": date,
                        "instrument_id": instrument_id,
                        "ticker": ticker,
                        "horizon_days": int(horizon),
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "label_name": f"fwd_ret_{horizon}d",
                        "label_value": float(label_ret),
                    }
                )
                labels_rows.append(
                    {
                        "date": date,
                        "instrument_id": instrument_id,
                        "ticker": ticker,
                        "horizon_days": int(horizon),
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "label_name": f"fwd_dir_up_{horizon}d",
                        "label_value": int(label_dir),
                    }
                )

            # Canonical model_dataset for 5d tasks (feature_ablation baseline inputs).
            dataset_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "horizon_days": 5,
                    "label_name": "fwd_ret_5d",
                    "target_type": "continuous_forward_return",
                    "target_value": float(sign * 0.012 + (0.0003 * idx)),
                    "ret_1d_lag1": ret_1d_lag1,
                    "ret_5d_lag1": ret_5d_lag1,
                    "vol_20d": vol_20d,
                    "mkt_breadth_up_lag1": mkt_breadth_up_lag1,
                    "mkt_cross_sectional_vol_lag1": mkt_cross_sectional_vol_lag1,
                    "log_dollar_volume_lag1": log_dollar_volume_lag1,
                    "log_total_assets": log_total_assets,
                }
            )
            dataset_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "horizon_days": 5,
                    "label_name": "fwd_dir_up_5d",
                    "target_type": "binary_direction",
                    "target_value": int(1 if sign > 0 else 0),
                    "ret_1d_lag1": ret_1d_lag1,
                    "ret_5d_lag1": ret_5d_lag1,
                    "vol_20d": vol_20d,
                    "mkt_breadth_up_lag1": mkt_breadth_up_lag1,
                    "mkt_cross_sectional_vol_lag1": mkt_cross_sectional_vol_lag1,
                    "log_dollar_volume_lag1": log_dollar_volume_lag1,
                    "log_total_assets": log_total_assets,
                }
            )

            # H20 canonical dataset (regression).
            h20_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "horizon_days": 20,
                    "label_name": "fwd_ret_20d",
                    "target_type": "continuous_forward_return",
                    "target_value": float(sign * 0.04 + (0.0005 * idx)),
                    "ret_1d_lag1": ret_1d_lag1,
                    "ret_5d_lag1": ret_5d_lag1,
                    "vol_20d": vol_20d,
                    "mkt_breadth_up_lag1": mkt_breadth_up_lag1,
                    "mkt_cross_sectional_vol_lag1": mkt_cross_sectional_vol_lag1,
                    "log_dollar_volume_lag1": log_dollar_volume_lag1,
                    "log_total_assets": log_total_assets,
                }
            )

    labels_df = pd.DataFrame(labels_rows)
    features_df = pd.DataFrame(feature_rows)

    # Purged CV folds across all labels/horizons used in Week 8 modules.
    for fold_id in (1, 2, 3, 4):
        for idx, date in enumerate(decision_dates):
            if fold_id == 1:
                role = "train" if idx <= 3 else "valid" if idx <= 5 else "dropped_by_purge"
            elif fold_id == 2:
                if idx <= 5:
                    role = "train"
                elif idx <= 7:
                    role = "valid"
                elif idx <= 9:
                    role = "dropped_by_purge"
                else:
                    role = "dropped_by_embargo"
            elif fold_id == 3:
                if idx <= 7:
                    role = "train"
                elif idx <= 9:
                    role = "valid"
                else:
                    role = "dropped_by_embargo"
            else:
                role = "train" if idx <= 8 else "valid" if idx <= 10 else "dropped_by_embargo"

            subset = labels_df[labels_df["date"] == date]
            for row in subset.itertuples(index=False):
                fold_rows.append(
                    {
                        "fold_id": int(fold_id),
                        "date": row.date,
                        "instrument_id": row.instrument_id,
                        "horizon_days": int(row.horizon_days),
                        "label_name": str(row.label_name),
                        "split_role": role,
                        "entry_date": row.entry_date,
                        "exit_date": row.exit_date,
                    }
                )

    model_dataset_path = datasets_root / "model_dataset.parquet"
    h20_dataset_path = datasets_root / "regression_h20" / "model_dataset.parquet"
    labels_path = labels_root / "labels_forward.parquet"
    features_path = features_root / "features_matrix.parquet"
    folds_path = labels_root / "purged_cv_folds.parquet"

    pd.DataFrame(dataset_rows).to_parquet(model_dataset_path, index=False)
    pd.DataFrame(h20_rows).to_parquet(h20_dataset_path, index=False)
    labels_df.to_parquet(labels_path, index=False)
    features_df.to_parquet(features_path, index=False)
    pd.DataFrame(fold_rows).to_parquet(folds_path, index=False)

    validation_summary_path = validation_root / "validation_suite_summary.json"
    validation_summary_path.write_text(
        json.dumps(
            {
                "overall_status": "WARN",
                "leakage_status": "WARN",
                "cv_comparison_status": "WARN",
                "signal_quality_status": "WARN",
                "portfolio_backtest_status": "WARN",
                "key_findings": ["synthetic_smoke_fixture"],
                "built_ts_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return {
        "model_dataset": model_dataset_path,
        "h20_model_dataset": h20_dataset_path,
        "labels_forward": labels_path,
        "features_matrix": features_path,
        "purged_cv_folds": folds_path,
        "validation_suite_summary": validation_summary_path,
    }


def test_week8_runner_end_to_end_smoke(tmp_workspace: dict[str, Path]) -> None:
    prereq = _seed_week8_prerequisites(tmp_workspace["data"])
    compatibility_counts_before = {
        "labels_forward": int(len(read_parquet(prereq["labels_forward"]))),
        "features_matrix": int(len(read_parquet(prereq["features_matrix"]))),
        "purged_cv_folds": int(len(read_parquet(prereq["purged_cv_folds"]))),
    }

    result = run_week8_research_refinement(
        run_prefix="test_week8_e2e",
        data_root=tmp_workspace["data"],
        model_dataset_path=prereq["model_dataset"],
        h20_model_dataset_path=prereq["h20_model_dataset"],
        labels_forward_path=prereq["labels_forward"],
        features_matrix_path=prereq["features_matrix"],
        purged_cv_folds_path=prereq["purged_cv_folds"],
        validation_suite_summary_path=prereq["validation_suite_summary"],
    )

    expected_keys = {
        "feature_ablation_results",
        "label_horizon_ablation_results",
        "edge_decision_report",
        "improve_best_candidate_results",
        "refine_h20_features_results",
        "refine_h20_target_results",
        "h20_regime_diagnostics_results",
        "h20_regime_conditioned_results",
    }
    assert expected_keys.issubset(set(result.artifacts.keys()))
    for key in expected_keys:
        path = result.artifacts[key]
        assert path.exists()
        assert path.stat().st_size > 0

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_prefix"] == "test_week8_e2e"
    assert manifest["step_statuses"]["feature_ablation"] == "DONE"
    assert manifest["step_statuses"]["label_horizon_ablation"] == "DONE"
    assert manifest["step_statuses"]["edge_decision_report"] == "DONE"
    assert manifest["step_statuses"]["improve_best_candidate"] == "DONE"
    assert manifest["step_statuses"]["refine_h20_features"] == "DONE"
    assert manifest["step_statuses"]["refine_h20_target"] == "DONE"
    assert manifest["step_statuses"]["h20_regime_diagnostics"] == "DONE"
    assert manifest["step_statuses"]["h20_regime_conditioned_refinement"] == "DONE"

    assert "final_recommendation" in manifest
    assert "week8_final_recommendation" in manifest
    assert "global_h20_status" in manifest
    assert "regime_conditioned_h20_status" in manifest
    assert manifest["should_try_richer_model_now"] is False
    assert manifest["week8_final_recommendation"] != "try_slightly_richer_model"

    # Week 1-7 compatibility: key base inputs should remain unchanged in row count.
    compatibility_counts_after = {
        "labels_forward": int(len(read_parquet(prereq["labels_forward"]))),
        "features_matrix": int(len(read_parquet(prereq["features_matrix"]))),
        "purged_cv_folds": int(len(read_parquet(prereq["purged_cv_folds"]))),
    }
    assert compatibility_counts_after == compatibility_counts_before
