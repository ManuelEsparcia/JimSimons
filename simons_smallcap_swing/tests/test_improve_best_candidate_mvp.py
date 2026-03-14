from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.improve_best_candidate import run_improve_best_candidate
from simons_core.io.parquet_store import read_parquet


def _seed_improve_best_candidate_inputs(
    tmp_workspace: dict[str, Path],
) -> tuple[Path, Path, Path]:
    base = tmp_workspace["data"] / "improve_best_candidate_case"
    base.mkdir(parents=True, exist_ok=True)
    dataset_path = base / "model_dataset.parquet"
    folds_path = base / "purged_cv_folds.parquet"
    feature_ablation_summary_path = base / "feature_ablation_summary.json"

    sessions = pd.bdate_range("2026-01-05", periods=6, freq="B")
    instruments = [("SIMA", "AAA", 1.0), ("SIMB", "BBB", -1.0)]

    dataset_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for idx, date in enumerate(sessions):
        for instrument_id, ticker, sign in instruments:
            signal = sign * float(idx + 1) / 10.0
            target_value = 0.3 * signal
            # Force one strong outlier so clipped-target variant can improve deterministically.
            if idx == 4 and instrument_id == "SIMB":
                target_value = 3.0

            entry_date = sessions[min(idx + 1, len(sessions) - 1)]
            exit_date = sessions[min(idx + 2, len(sessions) - 1)]

            dataset_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "horizon_days": 20,
                    "label_name": "fwd_ret_20d",
                    "split_name": "cv_purged",
                    "target_value": target_value,
                    "target_type": "continuous_forward_return",
                    "ret_1d_lag1": signal,
                    "ret_5d_lag1": 0.5 * signal,
                    "ret_20d_lag1": signal,
                    "momentum_20d_excl_1d": -0.2 * signal,
                    "vol_20d": 0.20 + 0.01 * idx,
                    "log_volume_lag1": 10.0 + 0.03 * idx,
                    "mkt_breadth_up_lag1": 0.50 + 0.01 * idx,
                    "log_total_assets": 11.0 + (8.0 * signal if idx <= 2 else -8.0 * signal),
                    "revenue_scale_proxy": 4.0 + (5.0 * signal if idx <= 2 else -5.0 * signal),
                }
            )

            if idx <= 2:
                role_fold_1 = "train"
            elif idx in {3, 4}:
                role_fold_1 = "valid"
            else:
                role_fold_1 = "dropped_by_embargo"
            fold_rows.append(
                {
                    "fold_id": 1,
                    "date": date,
                    "instrument_id": instrument_id,
                    "horizon_days": 20,
                    "label_name": "fwd_ret_20d",
                    "split_role": role_fold_1,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                }
            )

            if idx in {4, 5}:
                role_fold_2 = "valid"
            elif idx == 2:
                role_fold_2 = "dropped_by_purge"
            else:
                role_fold_2 = "train"
            fold_rows.append(
                {
                    "fold_id": 2,
                    "date": date,
                    "instrument_id": instrument_id,
                    "horizon_days": 20,
                    "label_name": "fwd_ret_20d",
                    "split_role": role_fold_2,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                }
            )

    pd.DataFrame(dataset_rows).to_parquet(dataset_path, index=False)
    pd.DataFrame(fold_rows).to_parquet(folds_path, index=False)
    feature_ablation_summary_path.write_text(
        json.dumps(
            {
                "best_family_by_task": {
                    "regression": "price_momentum",
                    "classification": "market_context",
                }
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return dataset_path, folds_path, feature_ablation_summary_path


def test_improve_best_candidate_mvp_runs_and_compares_against_baseline_and_dummy(
    tmp_workspace: dict[str, Path],
) -> None:
    dataset_path, folds_path, summary_path = _seed_improve_best_candidate_inputs(tmp_workspace)
    output_dir = tmp_workspace["data"] / "research"

    result = run_improve_best_candidate(
        model_dataset_path=dataset_path,
        purged_cv_folds_path=folds_path,
        feature_ablation_summary_path=summary_path,
        output_dir=output_dir,
        label_name="fwd_ret_20d",
        target_type="continuous_forward_return",
        horizon_days=20,
        enable_target_clipping=True,
        target_clip_abs=0.5,
        run_id="test_improve_best_candidate_mvp",
    )

    assert result.results_path.exists()
    assert result.summary_path.exists()
    assert result.fold_metrics_path.exists()
    assert result.results_path.with_suffix(".parquet.manifest.json").exists()
    assert result.fold_metrics_path.with_suffix(".parquet.manifest.json").exists()

    results = read_parquet(result.results_path)
    folds = read_parquet(result.fold_metrics_path)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert len(results) > 0
    assert len(folds) > 0

    required_cols = {
        "candidate_variant",
        "label_name",
        "target_type",
        "horizon_days",
        "primary_metric",
        "mean_valid_primary_metric",
        "improvement_vs_baseline",
        "improvement_vs_dummy",
        "winner_vs_baseline",
        "winner_vs_dummy",
        "n_features_used",
    }
    assert required_cols.issubset(set(results.columns))

    # No task/target mixing.
    assert set(results["target_type"].astype(str).unique().tolist()) == {"continuous_forward_return"}
    assert set(results["label_name"].astype(str).unique().tolist()) == {"fwd_ret_20d"}
    assert set(pd.to_numeric(results["horizon_days"], errors="coerce").astype(int).tolist()) == {20}

    # Formula consistency: improvement_vs_baseline = baseline_metric - variant_metric.
    baseline_metric = float(
        results.loc[
            results["candidate_variant"].astype(str) == "baseline_all_features",
            "mean_valid_primary_metric",
        ].iloc[0]
    )
    for row in results.itertuples(index=False):
        expected = baseline_metric - float(row.mean_valid_primary_metric)
        assert np.isclose(float(row.improvement_vs_baseline), expected, atol=1e-12)

    # Fabricated case: clipped target variant should improve over baseline.
    clipped = results[
        results["candidate_variant"].astype(str) == "clipped_target_if_enabled"
    ].iloc[0]
    assert float(clipped["improvement_vs_baseline"]) > 0.0
    assert str(clipped["winner_vs_baseline"]) == "variant"

    assert "best_variant" in summary
    assert "variants_evaluated" in summary
    assert "recommendation" in summary
