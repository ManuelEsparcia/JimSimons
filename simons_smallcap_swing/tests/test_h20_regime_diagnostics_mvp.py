from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.h20_regime_diagnostics import run_h20_regime_diagnostics
from simons_core.io.parquet_store import read_parquet


def _seed_h20_regime_case(tmp_workspace: dict[str, Path]) -> tuple[Path, Path]:
    base = tmp_workspace["data"] / "h20_regime_diagnostics_case"
    base.mkdir(parents=True, exist_ok=True)
    dataset_path = base / "model_dataset.parquet"
    folds_path = base / "purged_cv_folds.parquet"

    sessions = pd.bdate_range("2026-01-05", periods=12, freq="B")
    instruments = [
        ("SIMA", "AAA", 1.0, 0.45),
        ("SIMB", "BBB", -1.0, 0.60),
        ("SIMC", "CCC", 0.7, 0.75),
    ]

    dataset_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for idx, date in enumerate(sessions):
        for inst_idx, (instrument_id, ticker, sign, vol_level) in enumerate(instruments):
            signal = sign * (0.25 + (idx / 20.0))
            high_regime = 1.0 if vol_level >= 0.60 else 0.0
            noise = ((idx + inst_idx) % 5 - 2) * 0.01
            target = (0.85 if high_regime else 0.20) * signal + noise

            dataset_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "horizon_days": 20,
                    "label_name": "fwd_ret_20d",
                    "target_type": "continuous_forward_return",
                    "target_value": target,
                    "ret_1d_lag1": signal,
                    "ret_5d_lag1": 0.8 * signal + (0.02 * inst_idx),
                    "vol_20d": vol_level,
                    "mkt_breadth_up_lag1": 0.10 + (0.03 * inst_idx) - (0.01 * (idx % 2)),
                    "mkt_cross_sectional_vol_lag1": 0.15 + (0.02 * inst_idx) + (0.005 * (idx % 3)),
                    "log_dollar_volume_lag1": 10.0 + (0.20 * inst_idx) + (0.03 * idx),
                }
            )

            for fold_id in (1, 2, 3):
                if fold_id == 1:
                    if idx <= 4:
                        role = "train"
                    elif idx <= 6:
                        role = "valid"
                    elif idx <= 8:
                        role = "dropped_by_purge"
                    else:
                        role = "dropped_by_embargo"
                elif fold_id == 2:
                    if idx <= 6:
                        role = "train"
                    elif idx <= 8:
                        role = "valid"
                    elif idx == 9:
                        role = "dropped_by_purge"
                    else:
                        role = "dropped_by_embargo"
                else:
                    if idx <= 8:
                        role = "train"
                    elif idx <= 10:
                        role = "valid"
                    else:
                        role = "dropped_by_purge"

                fold_rows.append(
                    {
                        "fold_id": fold_id,
                        "date": date,
                        "instrument_id": instrument_id,
                        "horizon_days": 20,
                        "label_name": "fwd_ret_20d",
                        "split_role": role,
                    }
                )

    pd.DataFrame(dataset_rows).to_parquet(dataset_path, index=False)
    pd.DataFrame(fold_rows).to_parquet(folds_path, index=False)
    return dataset_path, folds_path


def test_h20_regime_diagnostics_mvp_outputs_and_segment_comparison(
    tmp_workspace: dict[str, Path],
) -> None:
    dataset_path, folds_path = _seed_h20_regime_case(tmp_workspace)
    output_dir = tmp_workspace["data"] / "research"

    result = run_h20_regime_diagnostics(
        model_dataset_path=dataset_path,
        purged_cv_folds_path=folds_path,
        output_dir=output_dir,
        label_name="fwd_ret_20d",
        target_type="continuous_forward_return",
        horizon_days=20,
        candidate_variants=("baseline_all_features", "stable_sign_plus_low_collinearity"),
        run_id="test_h20_regime_diagnostics_mvp",
    )

    assert result.results_path.exists()
    assert result.fold_metrics_path.exists()
    assert result.summary_path.exists()
    assert result.results_path.with_suffix(".parquet.manifest.json").exists()
    assert result.fold_metrics_path.with_suffix(".parquet.manifest.json").exists()

    results = read_parquet(result.results_path)
    fold_metrics = read_parquet(result.fold_metrics_path)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert len(results) > 0
    assert len(fold_metrics) > 0

    required_result_cols = {
        "candidate_variant",
        "segment_family",
        "segment_name",
        "n_obs",
        "model_mse",
        "dummy_mse",
        "improvement_vs_dummy",
        "winner_vs_dummy",
    }
    assert required_result_cols.issubset(set(results.columns))

    # Formula check: improvement_vs_dummy = dummy_mse - model_mse.
    for row in results.itertuples(index=False):
        expected = float(row.dummy_mse) - float(row.model_mse)
        assert np.isclose(float(row.improvement_vs_dummy), expected, atol=1e-12)

    # Fabricated case: at least one segment should beat dummy.
    assert (results["winner_vs_dummy"].astype(str) == "variant").any()

    # Time split family should always be present.
    assert "time_split_early_vs_late" in set(results["segment_family"].astype(str).tolist())

    # No obvious leakage policy regressions in summary.
    segment_policy = summary.get("segment_policy", {})
    assert bool(segment_policy.get("train_only_thresholds_for_high_low_segments")) is True

    comparison_policy = summary.get("comparison_policy", {})
    assert bool(comparison_policy.get("same_common_folds_only")) is True
    common_folds = sorted(int(v) for v in comparison_policy.get("common_fold_ids", []))
    assert len(common_folds) > 0

    assert "candidate_variants_evaluated" in summary
    assert "segments_where_model_beats_dummy" in summary
    assert "best_segment_by_variant" in summary
    assert "recommendation" in summary
