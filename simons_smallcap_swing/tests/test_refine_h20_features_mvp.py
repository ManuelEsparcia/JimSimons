from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.refine_h20_features import run_refine_h20_features
from simons_core.io.parquet_store import read_parquet


def _seed_refine_h20_inputs(tmp_workspace: dict[str, Path]) -> tuple[Path, Path]:
    base = tmp_workspace["data"] / "refine_h20_features_case"
    base.mkdir(parents=True, exist_ok=True)
    dataset_path = base / "model_dataset.parquet"
    folds_path = base / "purged_cv_folds.parquet"

    sessions = pd.bdate_range("2026-01-05", periods=10, freq="B")
    instruments = [("SIMA", "AAA", 1.0), ("SIMB", "BBB", -1.0), ("SIMC", "CCC", 0.5)]

    dataset_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for idx, date in enumerate(sessions):
        for instrument_id, ticker, sign in instruments:
            good = sign * float(idx + 1) / 10.0
            # Highly collinear with good in train windows, flips sign in late dates.
            bad_flip = good if idx < 6 else -1.5 * good
            target = 0.35 * good

            dataset_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "horizon_days": 20,
                    "label_name": "fwd_ret_20d",
                    "target_type": "continuous_forward_return",
                    "target_value": target,
                    "a_good_signal": good,
                    "z_bad_flip": bad_flip,
                    "mkt_noise": 0.01 * ((idx % 3) - 1),
                    # Train-only missingness check: non-missing in fold3 train, all-NaN in fold3 valid.
                    "valid_only_missing_feature": good if idx <= 7 else np.nan,
                }
            )

            # Fold 1: train 0-3, valid 4-5, dropped 6-9
            if idx <= 3:
                role1 = "train"
            elif idx <= 5:
                role1 = "valid"
            elif idx <= 7:
                role1 = "dropped_by_purge"
            else:
                role1 = "dropped_by_embargo"
            fold_rows.append(
                {
                    "fold_id": 1,
                    "date": date,
                    "instrument_id": instrument_id,
                    "horizon_days": 20,
                    "label_name": "fwd_ret_20d",
                    "split_role": role1,
                }
            )

            # Fold 2: train 0-5, valid 6-7, dropped 8-9
            if idx <= 5:
                role2 = "train"
            elif idx <= 7:
                role2 = "valid"
            elif idx == 8:
                role2 = "dropped_by_purge"
            else:
                role2 = "dropped_by_embargo"
            fold_rows.append(
                {
                    "fold_id": 2,
                    "date": date,
                    "instrument_id": instrument_id,
                    "horizon_days": 20,
                    "label_name": "fwd_ret_20d",
                    "split_role": role2,
                }
            )

            # Fold 3: train 0-7, valid 8-9
            role3 = "train" if idx <= 7 else "valid"
            fold_rows.append(
                {
                    "fold_id": 3,
                    "date": date,
                    "instrument_id": instrument_id,
                    "horizon_days": 20,
                    "label_name": "fwd_ret_20d",
                    "split_role": role3,
                }
            )

    pd.DataFrame(dataset_rows).to_parquet(dataset_path, index=False)
    pd.DataFrame(fold_rows).to_parquet(folds_path, index=False)
    return dataset_path, folds_path


def test_refine_h20_features_mvp_outputs_and_train_only_selection(
    tmp_workspace: dict[str, Path],
) -> None:
    dataset_path, folds_path = _seed_refine_h20_inputs(tmp_workspace)
    output_dir = tmp_workspace["data"] / "research"

    result = run_refine_h20_features(
        model_dataset_path=dataset_path,
        purged_cv_folds_path=folds_path,
        output_dir=output_dir,
        label_name="fwd_ret_20d",
        target_type="continuous_forward_return",
        horizon_days=20,
        missingness_threshold=0.05,
        collinearity_threshold=0.95,
        min_abs_train_corr=0.01,
        stability_min_history_folds=2,
        run_id="test_refine_h20_features_mvp",
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
    assert set(results["target_type"].astype(str).unique().tolist()) == {"continuous_forward_return"}
    assert set(results["label_name"].astype(str).unique().tolist()) == {"fwd_ret_20d"}
    assert set(pd.to_numeric(results["horizon_days"], errors="coerce").astype(int).tolist()) == {20}

    required_cols = {
        "candidate_variant",
        "mean_valid_primary_metric",
        "improvement_vs_baseline",
        "improvement_vs_dummy",
        "winner_vs_baseline",
        "winner_vs_dummy",
        "n_features_used",
    }
    assert required_cols.issubset(set(results.columns))

    # Formula check: improvement_vs_baseline = baseline_metric - variant_metric.
    baseline_metric = float(
        results.loc[
            results["candidate_variant"].astype(str) == "baseline_all_features",
            "mean_valid_primary_metric",
        ].iloc[0]
    )
    for row in results.itertuples(index=False):
        expected = baseline_metric - float(row.mean_valid_primary_metric)
        assert np.isclose(float(row.improvement_vs_baseline), expected, atol=1e-12)

    # Fabricated case: low collinearity variant should beat baseline.
    low_col = results[
        results["candidate_variant"].astype(str) == "low_collinearity_only"
    ].iloc[0]
    assert float(low_col["improvement_vs_baseline"]) > 0.0
    assert str(low_col["winner_vs_baseline"]) == "variant"

    # Train-only missingness check:
    # In fold 3, `valid_only_missing_feature` is NaN only in valid; train is complete.
    # It must remain selected for low_missingness_only if selection uses train only.
    low_missing_fold3 = folds[
        (folds["candidate_variant"].astype(str) == "low_missingness_only")
        & (pd.to_numeric(folds["fold_id"], errors="coerce").astype(int) == 3)
    ].iloc[0]
    selected = json.loads(str(low_missing_fold3["selected_features_json"]))
    assert "valid_only_missing_feature" in selected

    # Summary consumable.
    assert "best_variant" in summary
    assert "variants_evaluated" in summary
    assert "recommendation" in summary
