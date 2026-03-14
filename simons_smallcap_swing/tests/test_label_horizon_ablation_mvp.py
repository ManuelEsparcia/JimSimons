from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.label_horizon_ablation import run_label_horizon_ablation
from simons_core.io.parquet_store import read_parquet


def _seed_label_horizon_ablation_inputs(tmp_workspace: dict[str, Path]) -> tuple[Path, Path, Path]:
    base = tmp_workspace["data"] / "label_horizon_ablation_case"
    base.mkdir(parents=True, exist_ok=True)
    labels_path = base / "labels_forward.parquet"
    features_path = base / "features_matrix.parquet"
    folds_path = base / "purged_cv_folds.parquet"

    sessions = pd.bdate_range("2026-01-05", periods=6, freq="B")
    instruments = [("SIMA", "AAA", 1.0), ("SIMB", "BBB", -1.0), ("SIMC", "CCC", 1.0)]
    horizons = [1, 5]
    labels_meta = [("fwd_ret", "continuous_forward_return"), ("fwd_dir_up", "binary_direction")]

    labels_rows: list[dict[str, object]] = []
    features_rows: list[dict[str, object]] = []
    folds_rows: list[dict[str, object]] = []

    for idx, date in enumerate(sessions):
        for instrument_id, ticker, sign in instruments:
            base_signal = sign * float(idx + 1) / 10.0
            features_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "ret_1d_lag1": base_signal,
                    "ret_5d_lag1": 0.5 * base_signal,
                    "ret_20d_lag1": 0.2 * base_signal,
                    "momentum_20d_excl_1d": -0.7 * base_signal,
                    "vol_5d": 0.2 + 0.01 * idx,
                    "vol_20d": 0.3 + 0.01 * idx,
                    "abs_ret_1d_lag1": abs(base_signal),
                    "log_volume_lag1": 10.0 + 0.02 * idx,
                    "turnover_proxy_lag1": 1.0 + 0.01 * idx,
                    "log_dollar_volume_lag1": 12.0 + 0.03 * idx,
                    "mkt_breadth_up_lag1": 0.45 + 0.01 * idx,
                    "mkt_equal_weight_return_lag1": 0.001 * idx,
                    "mkt_cross_sectional_vol_lag1": 0.02 + 0.001 * idx,
                    "mkt_coverage_ratio_lag1": 0.9 - 0.01 * idx,
                    "log_total_assets": 10.0 + 0.005 * idx,
                    "shares_outstanding": 1000.0 + float(idx),
                    "revenue_scale_proxy": 5.0 + 0.03 * idx,
                    "net_income_scale_proxy": 1.0 + 0.02 * idx,
                }
            )

            entry_date = sessions[min(idx + 1, len(sessions) - 1)]
            exit_date = sessions[min(idx + 2, len(sessions) - 1)]
            for horizon in horizons:
                for prefix, target_type in labels_meta:
                    label_name = f"{prefix}_{horizon}d"
                    if target_type == "continuous_forward_return":
                        label_value = 0.35 * base_signal + (0.01 if horizon == 1 else 0.0)
                    else:
                        label_value = 1.0 if base_signal > 0 else 0.0

                    labels_rows.append(
                        {
                            "date": date,
                            "instrument_id": instrument_id,
                            "ticker": ticker,
                            "horizon_days": horizon,
                            "entry_date": entry_date,
                            "exit_date": exit_date,
                            "label_name": label_name,
                            "label_value": label_value,
                        }
                    )

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
                            "horizon_days": horizon,
                            "label_name": label_name,
                            "split_role": role_f1,
                            "entry_date": entry_date,
                            "exit_date": exit_date,
                        }
                    )

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
                            "horizon_days": horizon,
                            "label_name": label_name,
                            "split_role": role_f2,
                            "entry_date": entry_date,
                            "exit_date": exit_date,
                        }
                    )

    pd.DataFrame(labels_rows).to_parquet(labels_path, index=False)
    pd.DataFrame(features_rows).to_parquet(features_path, index=False)
    pd.DataFrame(folds_rows).to_parquet(folds_path, index=False)
    return labels_path, features_path, folds_path


def test_label_horizon_ablation_mvp_runs_and_handles_missing_combos(
    tmp_workspace: dict[str, Path],
) -> None:
    labels_path, features_path, folds_path = _seed_label_horizon_ablation_inputs(tmp_workspace)
    output_dir = tmp_workspace["data"] / "research"

    result = run_label_horizon_ablation(
        labels_forward_path=labels_path,
        features_matrix_path=features_path,
        purged_cv_folds_path=folds_path,
        output_dir=output_dir,
        horizons=(1, 5, 20),
        feature_family="all_features",
        run_id="test_label_horizon_ablation_mvp",
    )

    assert result.results_path.exists()
    assert result.fold_metrics_path.exists()
    assert result.summary_path.exists()
    assert result.results_path.with_suffix(".parquet.manifest.json").exists()
    assert result.fold_metrics_path.with_suffix(".parquet.manifest.json").exists()

    results = read_parquet(result.results_path)
    folds = read_parquet(result.fold_metrics_path)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert len(results) > 0
    assert len(folds) > 0
    assert set(results["task_name"].astype(str).unique().tolist()) == {"regression", "classification"}

    regression_rows = results[results["task_name"].astype(str) == "regression"].copy()
    classification_rows = results[results["task_name"].astype(str) == "classification"].copy()
    assert set(regression_rows["target_type"].astype(str).unique().tolist()) == {"continuous_forward_return"}
    assert set(classification_rows["target_type"].astype(str).unique().tolist()) == {"binary_direction"}
    assert all(label.startswith("fwd_ret_") for label in regression_rows["label_name"].astype(str).tolist())
    assert all(label.startswith("fwd_dir_up_") for label in classification_rows["label_name"].astype(str).tolist())

    # improvement_vs_dummy must be consistent with fold-level metrics.
    sample = results.iloc[0]
    sample_folds = folds[
        (folds["task_name"].astype(str) == str(sample["task_name"]))
        & (folds["label_name"].astype(str) == str(sample["label_name"]))
        & (pd.to_numeric(folds["horizon_days"], errors="coerce") == int(sample["horizon_days"]))
        & (folds["feature_family"].astype(str) == str(sample["feature_family"]))
    ].copy()
    calc_improvement = float(
        (
            pd.to_numeric(sample_folds["dummy_valid_primary_metric"], errors="coerce")
            - pd.to_numeric(sample_folds["model_valid_primary_metric"], errors="coerce")
        ).mean()
    )
    assert np.isclose(calc_improvement, float(sample["improvement_vs_dummy"]), atol=1e-12)

    # At least one combination should beat dummy in the fabricated case.
    assert (results["winner_vs_dummy"].astype(str) == "model").any()

    # Missing horizon (20d) should be explicitly tracked as WARN-style note.
    missing = summary.get("missing_label_combinations", [])
    assert len(missing) >= 2
    assert {int(item["horizon_days"]) for item in missing} == {20}
    assert any("WARN:" in note for note in summary.get("notes", []))

    assert set(summary["tasks_evaluated"]) == {"classification", "regression"}
    assert "best_label_by_task" in summary and "regression" in summary["best_label_by_task"]
