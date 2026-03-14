from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.h20_regime_conditioned_refinement import run_h20_regime_conditioned_refinement
from simons_core.io.parquet_store import read_parquet


def _seed_regime_conditioned_case(tmp_workspace: dict[str, Path]) -> tuple[Path, Path]:
    base = tmp_workspace["data"] / "h20_regime_conditioned_case"
    base.mkdir(parents=True, exist_ok=True)
    dataset_path = base / "model_dataset.parquet"
    folds_path = base / "purged_cv_folds.parquet"

    sessions = pd.bdate_range("2026-01-05", periods=12, freq="B")
    instruments = [
        ("SIMA", "AAA", 1.0, 10.00),
        ("SIMB", "BBB", -1.0, 10.10),
        ("SIMC", "CCC", 0.7, 11.00),
        ("SIMD", "DDD", -0.7, 11.10),
    ]

    dataset_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for idx, date in enumerate(sessions):
        for inst_idx, (instrument_id, ticker, sign, log_dv) in enumerate(instruments):
            good = sign * (0.25 + (idx / 18.0))
            bad_flip = 4.0 * good if idx < 6 else -6.0 * good
            noise = 0.01 * (((idx + inst_idx) % 5) - 2)
            low_liq = float(log_dv <= 10.10)
            coef = 1.20 if idx >= 6 and low_liq > 0 else 0.60 if low_liq > 0 else 0.40
            target = coef * good + noise

            dataset_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "horizon_days": 20,
                    "label_name": "fwd_ret_20d",
                    "target_type": "continuous_forward_return",
                    "target_value": target,
                    "ret_1d_lag1": good,
                    "ret_5d_lag1": 0.8 * good + (0.01 * inst_idx),
                    "z_bad_flip": bad_flip,
                    "mkt_breadth_up_lag1": (0.20 if idx >= 6 else 0.80) + (0.01 * (inst_idx - 1)),
                    "mkt_cross_sectional_vol_lag1": (0.20 if low_liq > 0 else 0.80) + (0.005 * (idx % 2)),
                    "log_dollar_volume_lag1": log_dv + (0.01 * (idx % 3)),
                }
            )

            for fold_id in (1, 2, 3, 4):
                if fold_id == 1:
                    if idx <= 3:
                        role = "train"
                    elif idx <= 5:
                        role = "valid"
                    elif idx <= 8:
                        role = "dropped_by_purge"
                    else:
                        role = "dropped_by_embargo"
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
                    if idx <= 9:
                        role = "train"
                    elif idx <= 11:
                        role = "valid"
                    else:
                        role = "dropped_by_embargo"

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


def test_h20_regime_conditioned_refinement_mvp_outputs(tmp_workspace: dict[str, Path]) -> None:
    dataset_path, folds_path = _seed_regime_conditioned_case(tmp_workspace)
    output_dir = tmp_workspace["data"] / "research"

    result = run_h20_regime_conditioned_refinement(
        model_dataset_path=dataset_path,
        purged_cv_folds_path=folds_path,
        output_dir=output_dir,
        label_name="fwd_ret_20d",
        target_type="continuous_forward_return",
        horizon_days=20,
        candidate_variants=("baseline_all_features", "stable_sign_plus_low_collinearity"),
        target_regimes=(
            "time_split_early_vs_late:late",
            "high_vs_low_market_breadth:low",
            "high_vs_low_cross_sectional_dispersion:low",
            "high_vs_low_liquidity:low",
        ),
        run_id="test_h20_regime_conditioned_refinement_mvp",
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

    required_cols = {
        "candidate_variant",
        "regime_family",
        "regime_name",
        "n_obs",
        "n_folds_used",
        "model_mse",
        "dummy_mse",
        "improvement_vs_dummy",
        "winner_vs_dummy",
        "comparison_variant",
        "improvement_vs_comparison_variant",
        "winner_vs_comparison_variant",
    }
    assert required_cols.issubset(set(results.columns))

    for row in results.itertuples(index=False):
        expected_dummy = float(row.dummy_mse) - float(row.model_mse)
        assert np.isclose(float(row.improvement_vs_dummy), expected_dummy, atol=1e-12)

    # Within each regime, check pairwise comparison formula and same folds.
    for (family, name), group in results.groupby(["regime_family", "regime_name"]):
        group = group.copy()
        mse_map = {
            str(r.candidate_variant): float(r.model_mse)
            for r in group.itertuples(index=False)
        }
        for row in group.itertuples(index=False):
            other = str(row.comparison_variant)
            expected_other = mse_map[other] - float(row.model_mse)
            assert np.isclose(float(row.improvement_vs_comparison_variant), expected_other, atol=1e-12)

        fold_sets = []
        for variant in group["candidate_variant"].astype(str).tolist():
            variant_folds = sorted(
                set(
                    pd.to_numeric(
                        fold_metrics.loc[
                            (fold_metrics["regime_family"].astype(str) == str(family))
                            & (fold_metrics["regime_name"].astype(str) == str(name))
                            & (fold_metrics["candidate_variant"].astype(str) == variant),
                            "fold_id",
                        ],
                        errors="coerce",
                    )
                    .dropna()
                    .astype(int)
                    .tolist()
                )
            )
            fold_sets.append(tuple(variant_folds))
        assert len(set(fold_sets)) == 1

    # Synthetic case should find at least one regime where model beats dummy.
    assert (results["winner_vs_dummy"].astype(str) == "variant").any()

    # Synthetic case should find at least one regime where stable-sign beats baseline.
    stable = results[
        (results["candidate_variant"].astype(str) == "stable_sign_plus_low_collinearity")
        & (results["winner_vs_comparison_variant"].astype(str) == "variant")
    ]
    assert len(stable) > 0

    # Segment policy should be train-only for thresholds.
    segment_policy = summary.get("segment_policy", {})
    assert bool(segment_policy.get("train_only_thresholds_for_high_low_segments")) is True
    comparison_policy = summary.get("comparison_policy", {})
    assert bool(comparison_policy.get("same_common_folds_only")) is True
    assert "best_variant_by_regime" in summary
    assert "recommendation" in summary
