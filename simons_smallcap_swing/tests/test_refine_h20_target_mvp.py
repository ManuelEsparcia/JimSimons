from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.refine_h20_target import run_refine_h20_target
from simons_core.io.parquet_store import read_parquet


def _seed_refine_h20_target_inputs(tmp_workspace: dict[str, Path]) -> tuple[Path, Path]:
    base = tmp_workspace["data"] / "refine_h20_target_case"
    base.mkdir(parents=True, exist_ok=True)
    dataset_path = base / "model_dataset.parquet"
    folds_path = base / "purged_cv_folds.parquet"

    sessions = pd.bdate_range("2026-01-05", periods=8, freq="B")
    instruments = [("SIMA", "AAA", 1.0), ("SIMB", "BBB", -1.0)]

    dataset_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for idx, date in enumerate(sessions):
        for inst_pos, (instrument_id, ticker, sign) in enumerate(instruments):
            signal = sign * float(idx + 1) / 5.0
            # Force fallback from vol_20d to vol_5d in train folds.
            vol_20d = np.nan if idx <= 4 else (0.6 + (0.25 * idx) + (0.15 * inst_pos))
            vol_5d = 0.4 + (0.20 * idx) + (0.10 * inst_pos)
            mkt_ref = 0.01 * ((idx % 3) - 1)
            target_value = signal * vol_5d

            dataset_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "horizon_days": 20,
                    "label_name": "fwd_ret_20d",
                    "target_type": "continuous_forward_return",
                    "target_value": target_value,
                    "ret_1d_lag1": signal,
                    "ret_5d_lag1": 0.7 * signal,
                    "vol_20d": vol_20d,
                    "vol_5d": vol_5d,
                    "mkt_equal_weight_return_lag1": mkt_ref,
                }
            )

            if idx <= 3:
                role_f1 = "train"
            elif idx in {4, 5}:
                role_f1 = "valid"
            elif idx == 6:
                role_f1 = "dropped_by_purge"
            else:
                role_f1 = "dropped_by_embargo"
            fold_rows.append(
                {
                    "fold_id": 1,
                    "date": date,
                    "instrument_id": instrument_id,
                    "horizon_days": 20,
                    "label_name": "fwd_ret_20d",
                    "split_role": role_f1,
                }
            )

            if idx <= 4:
                role_f2 = "train"
            elif idx in {5, 6}:
                role_f2 = "valid"
            else:
                role_f2 = "dropped_by_purge"
            fold_rows.append(
                {
                    "fold_id": 2,
                    "date": date,
                    "instrument_id": instrument_id,
                    "horizon_days": 20,
                    "label_name": "fwd_ret_20d",
                    "split_role": role_f2,
                }
            )

    pd.DataFrame(dataset_rows).to_parquet(dataset_path, index=False)
    pd.DataFrame(fold_rows).to_parquet(folds_path, index=False)
    return dataset_path, folds_path


def test_refine_h20_target_mvp_outputs_and_variant_comparison(tmp_workspace: dict[str, Path]) -> None:
    dataset_path, folds_path = _seed_refine_h20_target_inputs(tmp_workspace)
    output_dir = tmp_workspace["data"] / "research"

    result = run_refine_h20_target(
        model_dataset_path=dataset_path,
        purged_cv_folds_path=folds_path,
        output_dir=output_dir,
        label_name="fwd_ret_20d",
        target_type="continuous_forward_return",
        horizon_days=20,
        variants=(
            "raw_target_baseline",
            "clipped_target",
            "vol_scaled_target",
            "market_relative_target",
        ),
        target_clip_abs=0.25,
        run_id="test_refine_h20_target_mvp",
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
        "target_variant",
        "label_name",
        "target_type",
        "horizon_days",
        "primary_metric",
        "mean_valid_primary_metric",
        "improvement_vs_baseline",
        "improvement_vs_dummy",
        "winner_vs_baseline",
        "winner_vs_dummy",
    }
    assert required_cols.issubset(set(results.columns))

    # No task mixing.
    assert set(results["target_type"].astype(str).unique().tolist()) == {"continuous_forward_return"}
    assert set(results["label_name"].astype(str).unique().tolist()) == {"fwd_ret_20d"}
    assert set(pd.to_numeric(results["horizon_days"], errors="coerce").astype(int).tolist()) == {20}

    # Formula consistency: improvement_vs_baseline = baseline_metric - variant_metric.
    baseline_metric = float(
        results.loc[
            results["target_variant"].astype(str) == "raw_target_baseline",
            "mean_valid_primary_metric",
        ].iloc[0]
    )
    for row in results.itertuples(index=False):
        expected = baseline_metric - float(row.mean_valid_primary_metric)
        assert np.isclose(float(row.improvement_vs_baseline), expected, atol=1e-12)

    # Fabricated case: vol-scaled variant should improve vs baseline.
    vol_scaled = results[results["target_variant"].astype(str) == "vol_scaled_target"].iloc[0]
    assert float(vol_scaled["improvement_vs_baseline"]) > 0.0
    assert str(vol_scaled["winner_vs_baseline"]) == "variant"

    # Vol-scaled should not fail when fallback volatility column exists.
    vol_scaled_folds = folds[folds["target_variant"].astype(str) == "vol_scaled_target"].copy()
    assert len(vol_scaled_folds) > 0
    used_vol_cols = {
        json.loads(str(item)).get("vol_scale_column")
        for item in vol_scaled_folds["variant_notes_json"].astype(str).tolist()
    }
    assert "vol_5d" in used_vol_cols

    # Comparability policy: all variants must be evaluated on the same common folds.
    comparison_policy = summary.get("comparison_policy", {})
    assert bool(comparison_policy.get("same_common_folds_only")) is True
    common_fold_ids = sorted(int(x) for x in comparison_policy.get("common_fold_ids", []))
    assert len(common_fold_ids) > 0

    per_variant_fold_sets = {
        variant: sorted(
            set(
                pd.to_numeric(
                    folds.loc[folds["target_variant"].astype(str) == variant, "fold_id"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .tolist()
            )
        )
        for variant in results["target_variant"].astype(str).unique().tolist()
    }
    for fold_set in per_variant_fold_sets.values():
        assert fold_set == common_fold_ids

    used_common_counts = pd.to_numeric(
        results["n_folds_used_common"], errors="coerce"
    ).dropna().astype(int).unique().tolist()
    assert used_common_counts == [len(common_fold_ids)]

    # No obvious leakage in target construction policy.
    policy = summary.get("target_construction_policy", {})
    assert bool(policy.get("train_only_target_transform_fit")) is True
    assert bool(policy.get("validation_scored_on_raw_target")) is True
    assert bool(policy.get("uses_only_decision_date_features_for_adjustments")) is True

    assert "best_variant" in summary
    assert "variants_evaluated" in summary
    assert "recommendation" in summary
