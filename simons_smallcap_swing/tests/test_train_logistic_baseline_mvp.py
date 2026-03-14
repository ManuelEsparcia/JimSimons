from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datasets.build_model_dataset import build_model_dataset
from models.baselines.train_logistic import train_logistic_baseline
from simons_core.io.parquet_store import read_parquet
from test_build_model_dataset_mvp import _build_model_dataset_inputs


def test_train_logistic_baseline_runs_end_to_end_on_binary_dataset(
    tmp_workspace: dict[str, Path],
) -> None:
    paths = _build_model_dataset_inputs(tmp_workspace)
    dataset_result = build_model_dataset(
        features_path=paths["features"],
        labels_path=paths["labels"],
        purged_splits_path=paths["splits"],
        output_dir=paths["datasets_root"],
        label_names=("fwd_ret_5d",),
        run_id="test_model_dataset_for_logistic",
    )

    # Convert the continuous dataset into a binary-direction variant for this baseline test.
    binary_dataset_path = tmp_workspace["data"] / "datasets" / "model_dataset_binary.parquet"
    model_df = read_parquet(dataset_result.dataset_path).copy()
    model_df["label_name"] = "fwd_dir_up_5d"
    model_df["target_type"] = "binary_direction"
    model_df["target_value"] = (pd.to_numeric(model_df["target_value"], errors="coerce") > 0).astype(int)
    for role in ("train", "valid", "test"):
        role_idx = model_df.index[model_df["split_role"].astype(str) == role].tolist()
        for pos, idx in enumerate(role_idx):
            model_df.loc[idx, "target_value"] = pos % 2
    model_df.to_parquet(binary_dataset_path, index=False)

    output_dir = tmp_workspace["artifacts"] / "logistic_baseline"
    result = train_logistic_baseline(
        model_dataset_path=binary_dataset_path,
        output_dir=output_dir,
        label_name="fwd_dir_up_5d",
        c_grid=(0.1, 1.0, 10.0),
        run_id="test_train_logistic_mvp",
    )

    assert result.metrics_path.exists()
    assert result.predictions_path.exists()
    assert result.model_path.exists()
    assert result.feature_stats_path.exists()
    assert result.n_features_used > 0
    assert result.n_train > 0 and result.n_valid > 0 and result.n_test > 0
    assert float(result.c_selected) in {0.1, 1.0, 10.0}

    predictions = read_parquet(result.predictions_path)
    assert len(predictions) > 0
    assert set(predictions["split_role"].astype(str).unique().tolist()).issubset(
        {"train", "valid", "test"}
    )
    assert "valid" in set(predictions["split_role"].astype(str).tolist())
    assert "test" in set(predictions["split_role"].astype(str).tolist())
    assert predictions["pred_proba"].between(0.0, 1.0).all()
    assert set(predictions["pred_class"].astype(int).unique().tolist()).issubset({0, 1})

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["model_name"] == "logistic_baseline"
    assert metrics["label_name"] == "fwd_dir_up_5d"
    assert metrics["metrics"]["valid"]["n"] == result.n_valid
    assert metrics["metrics"]["test"]["n"] == result.n_test
    for split in ("valid", "test"):
        assert metrics["metrics"][split]["log_loss"] >= 0.0
        for metric_name in ("accuracy", "balanced_accuracy", "precision", "recall", "f1"):
            value = metrics["metrics"][split][metric_name]
            assert 0.0 <= value <= 1.0


def test_train_logistic_baseline_uses_train_only_preprocessing_and_excludes_dropped_roles(
    tmp_workspace: dict[str, Path],
) -> None:
    base = tmp_workspace["data"] / "logistic_small_case"
    dataset_path = base / "model_dataset.parquet"
    output_dir = base / "artifacts"
    base.mkdir(parents=True, exist_ok=True)

    rows = [
        # train rows (both classes present)
        {
            "date": "2026-01-05",
            "instrument_id": "SIMA",
            "ticker": "AAA",
            "horizon_days": 5,
            "label_name": "fwd_dir_up_5d",
            "split_name": "holdout_temporal_purged",
            "split_role": "train",
            "entry_date": "2026-01-06",
            "exit_date": "2026-01-13",
            "target_value": 0,
            "target_type": "binary_direction",
            "f1": 1.0,
            "f2": 1.0,
            "f3": np.nan,
            "sector": "tech",
        },
        {
            "date": "2026-01-06",
            "instrument_id": "SIMA",
            "ticker": "AAA",
            "horizon_days": 5,
            "label_name": "fwd_dir_up_5d",
            "split_name": "holdout_temporal_purged",
            "split_role": "train",
            "entry_date": "2026-01-07",
            "exit_date": "2026-01-14",
            "target_value": 1,
            "target_type": "binary_direction",
            "f1": 2.0,
            "f2": np.nan,
            "f3": np.nan,
            "sector": "tech",
        },
        {
            "date": "2026-01-07",
            "instrument_id": "SIMA",
            "ticker": "AAA",
            "horizon_days": 5,
            "label_name": "fwd_dir_up_5d",
            "split_name": "holdout_temporal_purged",
            "split_role": "train",
            "entry_date": "2026-01-08",
            "exit_date": "2026-01-15",
            "target_value": 0,
            "target_type": "binary_direction",
            "f1": 3.0,
            "f2": 3.0,
            "f3": np.nan,
            "sector": "tech",
        },
        {
            "date": "2026-01-08",
            "instrument_id": "SIMA",
            "ticker": "AAA",
            "horizon_days": 5,
            "label_name": "fwd_dir_up_5d",
            "split_name": "holdout_temporal_purged",
            "split_role": "valid",
            "entry_date": "2026-01-09",
            "exit_date": "2026-01-16",
            "target_value": 1,
            "target_type": "binary_direction",
            "f1": 1000.0,
            "f2": 1000.0,
            "f3": np.nan,
            "sector": "tech",
        },
        {
            "date": "2026-01-09",
            "instrument_id": "SIMA",
            "ticker": "AAA",
            "horizon_days": 5,
            "label_name": "fwd_dir_up_5d",
            "split_name": "holdout_temporal_purged",
            "split_role": "test",
            "entry_date": "2026-01-12",
            "exit_date": "2026-01-19",
            "target_value": 1,
            "target_type": "binary_direction",
            "f1": 2000.0,
            "f2": 2000.0,
            "f3": np.nan,
            "sector": "tech",
        },
        {
            "date": "2026-01-10",
            "instrument_id": "SIMA",
            "ticker": "AAA",
            "horizon_days": 5,
            "label_name": "fwd_dir_up_5d",
            "split_name": "holdout_temporal_purged",
            "split_role": "dropped_by_purge",
            "entry_date": "2026-01-13",
            "exit_date": "2026-01-20",
            "target_value": 0,
            "target_type": "binary_direction",
            "f1": 9999.0,
            "f2": 9999.0,
            "f3": np.nan,
            "sector": "tech",
        },
    ]
    frame = pd.DataFrame(rows)
    for col in ("date", "entry_date", "exit_date"):
        frame[col] = pd.to_datetime(frame[col])
    frame.to_parquet(dataset_path, index=False)

    result = train_logistic_baseline(
        model_dataset_path=dataset_path,
        output_dir=output_dir,
        label_name="fwd_dir_up_5d",
        c_grid=(0.1, 1.0, 10.0),
        run_id="test_train_logistic_small_case",
    )

    predictions = read_parquet(result.predictions_path)
    assert set(predictions["split_role"].astype(str).unique().tolist()) == {"train", "valid", "test"}
    assert "dropped_by_purge" not in set(predictions["split_role"].astype(str).tolist())
    assert len(predictions) == 5  # 3 train + 1 valid + 1 test

    feature_stats = json.loads(result.feature_stats_path.read_text(encoding="utf-8"))
    assert "f3" in feature_stats["features_dropped_all_nan_train"]
    assert "sector" not in feature_stats["features_used"]  # non numeric
    assert np.isclose(feature_stats["imputer_median_train_only"]["f2"], 2.0)
    assert feature_stats["imputer_median_train_only"]["f2"] != 1000.0

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["split_counts_modelable_roles"]["train"] == 3
    assert metrics["split_counts_modelable_roles"]["valid"] == 1
    assert metrics["split_counts_modelable_roles"]["test"] == 1
    assert metrics["split_counts_all_roles"]["dropped_by_purge"] == 1
    assert metrics["n_features_used"] == 2  # f1, f2


def test_train_logistic_baseline_fails_clearly_when_default_binary_label_missing(
    tmp_workspace: dict[str, Path],
) -> None:
    base = tmp_workspace["data"] / "logistic_missing_label_case"
    dataset_path = base / "model_dataset.parquet"
    base.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        [
            {
                "date": "2026-01-05",
                "instrument_id": "SIMA",
                "ticker": "AAA",
                "horizon_days": 5,
                "label_name": "fwd_ret_5d",
                "split_name": "holdout_temporal_purged",
                "split_role": "train",
                "entry_date": "2026-01-06",
                "exit_date": "2026-01-13",
                "target_value": 0.01,
                "target_type": "continuous_forward_return",
                "f1": 1.0,
            },
            {
                "date": "2026-01-06",
                "instrument_id": "SIMA",
                "ticker": "AAA",
                "horizon_days": 5,
                "label_name": "fwd_ret_5d",
                "split_name": "holdout_temporal_purged",
                "split_role": "valid",
                "entry_date": "2026-01-07",
                "exit_date": "2026-01-14",
                "target_value": -0.01,
                "target_type": "continuous_forward_return",
                "f1": 2.0,
            },
            {
                "date": "2026-01-07",
                "instrument_id": "SIMA",
                "ticker": "AAA",
                "horizon_days": 5,
                "label_name": "fwd_ret_5d",
                "split_name": "holdout_temporal_purged",
                "split_role": "test",
                "entry_date": "2026-01-08",
                "exit_date": "2026-01-15",
                "target_value": 0.02,
                "target_type": "continuous_forward_return",
                "f1": 3.0,
            },
        ]
    )
    for col in ("date", "entry_date", "exit_date"):
        frame[col] = pd.to_datetime(frame[col])
    frame.to_parquet(dataset_path, index=False)

    with pytest.raises(ValueError, match="Regenerate labels with binary direction enabled"):
        _ = train_logistic_baseline(
            model_dataset_path=dataset_path,
            # default label_name = fwd_dir_up_5d
            run_id="test_missing_binary_label",
        )

