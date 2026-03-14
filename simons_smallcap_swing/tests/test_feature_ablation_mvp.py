from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.feature_ablation import run_feature_ablation
from simons_core.io.parquet_store import read_parquet


def _seed_feature_ablation_inputs(tmp_workspace: dict[str, Path]) -> tuple[Path, Path, Path]:
    base = tmp_workspace["data"] / "feature_ablation_case"
    base.mkdir(parents=True, exist_ok=True)
    dataset_path = base / "model_dataset.parquet"
    folds_path = base / "purged_cv_folds.parquet"
    features_matrix_path = base / "features_matrix.parquet"

    sessions = pd.bdate_range("2026-01-05", periods=6, freq="B")
    instruments = [("SIMA", "AAA"), ("SIMB", "BBB")]
    labels = [
        ("fwd_ret_5d", "continuous_forward_return"),
        ("fwd_dir_up_5d", "binary_direction"),
    ]

    dataset_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    folds_rows: list[dict[str, object]] = []

    for idx, date in enumerate(sessions):
        for inst_pos, (instrument_id, ticker) in enumerate(instruments):
            sign = 1.0 if inst_pos == 0 else -1.0
            price_signal = sign * float(idx + 1) / 10.0

            row_features = {
                "ret_1d_lag1": price_signal,
                "ret_5d_lag1": 0.5 * price_signal,
                "ret_20d_lag1": 0.2 * price_signal,
                "momentum_20d_excl_1d": -0.8 * price_signal,
                "vol_5d": 0.2 + 0.01 * idx,
                "vol_20d": 0.3 + 0.01 * idx,
                "abs_ret_1d_lag1": abs(price_signal),
                "log_volume_lag1": 10.0 + 0.02 * idx,
                "turnover_proxy_lag1": 1.0 + 0.01 * idx,
                "log_dollar_volume_lag1": 12.0 + 0.02 * idx,
                "mkt_breadth_up_lag1": 0.45 + 0.01 * idx,
                "mkt_equal_weight_return_lag1": 0.001 * idx,
                "mkt_cross_sectional_vol_lag1": 0.02 + 0.001 * idx,
                "mkt_coverage_ratio_lag1": 0.85 - 0.01 * idx,
                "log_total_assets": 10.0 + 0.005 * idx,
                "shares_outstanding": 1000.0 + float(idx),
                "revenue_scale_proxy": 5.0 + 0.02 * idx,
                "net_income_scale_proxy": 1.0 + 0.01 * idx,
            }
            feature_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    **row_features,
                }
            )

            entry_date = sessions[min(idx + 1, len(sessions) - 1)]
            exit_date = sessions[min(idx + 2, len(sessions) - 1)]
            for label_name, target_type in labels:
                if target_type == "continuous_forward_return":
                    target_value = 0.3 * price_signal
                else:
                    target_value = 1.0 if price_signal > 0 else 0.0

                dataset_rows.append(
                    {
                        "date": date,
                        "instrument_id": instrument_id,
                        "ticker": ticker,
                        "horizon_days": 5,
                        "label_name": label_name,
                        "split_name": "holdout_temporal_purged",
                        "split_role": "train",
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "target_value": target_value,
                        "target_type": target_type,
                        **row_features,
                    }
                )

                # Fold 1
                if idx <= 2:
                    role_f1 = "train"
                elif idx in {3, 4}:
                    role_f1 = "valid"
                else:
                    role_f1 = "dropped_by_embargo"
                folds_rows.append(
                    {
                        "fold_id": 1,
                        "date": date,
                        "instrument_id": instrument_id,
                        "horizon_days": 5,
                        "label_name": label_name,
                        "split_role": role_f1,
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                    }
                )

                # Fold 2
                if idx in {4, 5}:
                    role_f2 = "valid"
                elif idx == 2:
                    role_f2 = "dropped_by_purge"
                else:
                    role_f2 = "train"
                folds_rows.append(
                    {
                        "fold_id": 2,
                        "date": date,
                        "instrument_id": instrument_id,
                        "horizon_days": 5,
                        "label_name": label_name,
                        "split_role": role_f2,
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                    }
                )

    pd.DataFrame(dataset_rows).to_parquet(dataset_path, index=False)
    pd.DataFrame(folds_rows).to_parquet(folds_path, index=False)
    pd.DataFrame(feature_rows).to_parquet(features_matrix_path, index=False)
    return dataset_path, folds_path, features_matrix_path


def test_feature_ablation_mvp_runs_and_identifies_better_family(tmp_workspace: dict[str, Path]) -> None:
    dataset_path, folds_path, features_matrix_path = _seed_feature_ablation_inputs(tmp_workspace)
    output_dir = tmp_workspace["data"] / "research"
    result = run_feature_ablation(
        model_dataset_path=dataset_path,
        purged_cv_folds_path=folds_path,
        features_matrix_path=features_matrix_path,
        output_dir=output_dir,
        regression_label_name="fwd_ret_5d",
        classification_label_name="fwd_dir_up_5d",
        horizon_days=5,
        run_id="test_feature_ablation_mvp",
    )

    assert result.results_path.exists()
    assert result.summary_path.exists()
    assert result.fold_metrics_path.exists()
    assert result.results_path.with_suffix(".parquet.manifest.json").exists()
    assert result.fold_metrics_path.with_suffix(".parquet.manifest.json").exists()

    results = read_parquet(result.results_path)
    fold_metrics = read_parquet(result.fold_metrics_path)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert len(results) > 0
    assert len(fold_metrics) > 0

    mandatory_families = {"price_momentum", "vol_liquidity", "market_context", "fundamentals", "all_features"}
    observed_families = set(results["feature_family"].astype(str).unique().tolist())
    assert mandatory_families.issubset(observed_families)

    # No task/target leakage mixing.
    regression_rows = results[results["task_name"].astype(str) == "regression"]
    classification_rows = results[results["task_name"].astype(str) == "classification"]
    assert set(regression_rows["target_type"].astype(str).unique().tolist()) == {"continuous_forward_return"}
    assert set(classification_rows["target_type"].astype(str).unique().tolist()) == {"binary_direction"}
    assert set(regression_rows["label_name"].astype(str).unique().tolist()) == {"fwd_ret_5d"}
    assert set(classification_rows["label_name"].astype(str).unique().tolist()) == {"fwd_dir_up_5d"}

    # Fabricated case: price_momentum should beat fundamentals in regression.
    reg_price = regression_rows[regression_rows["feature_family"].astype(str) == "price_momentum"].iloc[0]
    reg_fund = regression_rows[regression_rows["feature_family"].astype(str) == "fundamentals"].iloc[0]
    assert float(reg_price["mean_valid_primary_metric"]) < float(reg_fund["mean_valid_primary_metric"])
    assert str(reg_price["winner_vs_dummy"]) in {"model", "tie"}

    # improvement_vs_dummy is consistent with fold-level metrics.
    reg_price_folds = fold_metrics[
        (fold_metrics["task_name"].astype(str) == "regression")
        & (fold_metrics["feature_family"].astype(str) == "price_momentum")
    ].copy()
    calc_improvement = float(
        (
            pd.to_numeric(reg_price_folds["dummy_valid_primary_metric"], errors="coerce")
            - pd.to_numeric(reg_price_folds["model_valid_primary_metric"], errors="coerce")
        ).mean()
    )
    assert np.isclose(calc_improvement, float(reg_price["improvement_vs_dummy"]), atol=1e-12)

    # Summary consumable and aligned with outputs.
    assert set(summary["tasks_evaluated"]) == {"classification", "regression"}
    assert "best_family_by_task" in summary and "regression" in summary["best_family_by_task"]
    assert "families_beating_dummy" in summary and "regression" in summary["families_beating_dummy"]
    assert set(summary["feature_families_evaluated"]).issuperset(mandatory_families)
