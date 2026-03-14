from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.baselines.train_dummy_baselines import train_dummy_baseline
from simons_core.io.parquet_store import read_parquet


def _write_model_dataset(path: Path, rows: list[dict[str, object]]) -> Path:
    frame = pd.DataFrame(rows)
    for col in ("date", "entry_date", "exit_date"):
        frame[col] = pd.to_datetime(frame[col])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def test_train_dummy_regressor_mvp_generates_artifacts_and_uses_train_mean_only(
    tmp_workspace: dict[str, Path],
) -> None:
    base = tmp_workspace["data"] / "dummy_regressor_case"
    dataset_path = base / "model_dataset.parquet"
    output_dir = tmp_workspace["artifacts"] / "dummy_regressor_case"

    _write_model_dataset(
        dataset_path,
        rows=[
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
                "target_value": 1.0,
                "target_type": "continuous_forward_return",
            },
            {
                "date": "2026-01-06",
                "instrument_id": "SIMA",
                "ticker": "AAA",
                "horizon_days": 5,
                "label_name": "fwd_ret_5d",
                "split_name": "holdout_temporal_purged",
                "split_role": "train",
                "entry_date": "2026-01-07",
                "exit_date": "2026-01-14",
                "target_value": 2.0,
                "target_type": "continuous_forward_return",
            },
            {
                "date": "2026-01-07",
                "instrument_id": "SIMA",
                "ticker": "AAA",
                "horizon_days": 5,
                "label_name": "fwd_ret_5d",
                "split_name": "holdout_temporal_purged",
                "split_role": "train",
                "entry_date": "2026-01-08",
                "exit_date": "2026-01-15",
                "target_value": 3.0,
                "target_type": "continuous_forward_return",
            },
            {
                "date": "2026-01-08",
                "instrument_id": "SIMA",
                "ticker": "AAA",
                "horizon_days": 5,
                "label_name": "fwd_ret_5d",
                "split_name": "holdout_temporal_purged",
                "split_role": "valid",
                "entry_date": "2026-01-09",
                "exit_date": "2026-01-16",
                "target_value": 100.0,
                "target_type": "continuous_forward_return",
            },
            {
                "date": "2026-01-09",
                "instrument_id": "SIMA",
                "ticker": "AAA",
                "horizon_days": 5,
                "label_name": "fwd_ret_5d",
                "split_name": "holdout_temporal_purged",
                "split_role": "test",
                "entry_date": "2026-01-12",
                "exit_date": "2026-01-19",
                "target_value": 200.0,
                "target_type": "continuous_forward_return",
            },
            {
                "date": "2026-01-10",
                "instrument_id": "SIMA",
                "ticker": "AAA",
                "horizon_days": 5,
                "label_name": "fwd_ret_5d",
                "split_name": "holdout_temporal_purged",
                "split_role": "dropped_by_purge",
                "entry_date": "2026-01-13",
                "exit_date": "2026-01-20",
                "target_value": 9999.0,
                "target_type": "continuous_forward_return",
            },
        ],
    )

    result = train_dummy_baseline(
        mode="dummy_regressor",
        model_dataset_path=dataset_path,
        output_dir=output_dir,
        label_name="fwd_ret_5d",
        dummy_strategy="mean",
        run_id="test_dummy_regressor_mvp",
    )

    assert result.metrics_path.exists()
    assert result.predictions_path.exists()
    assert np.isclose(result.learned_statistic, 2.0)
    assert result.n_train == 3
    assert result.n_valid == 1
    assert result.n_test == 1

    predictions = read_parquet(result.predictions_path)
    assert len(predictions) == 5
    assert set(predictions["split_role"].astype(str).unique().tolist()) == {"train", "valid", "test"}
    assert np.allclose(predictions["prediction"].to_numpy(dtype=float), 2.0)
    assert "dropped_by_purge" not in set(predictions["split_role"].astype(str).tolist())

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["mode"] == "dummy_regressor"
    assert metrics["dummy_strategy"] == "mean"
    assert metrics["target_type"] == "continuous_forward_return"
    assert metrics["train_target_statistic_name"] == "mean"
    assert np.isclose(metrics["train_target_statistic_value"], 2.0)
    assert metrics["split_counts_all_roles"]["dropped_by_purge"] == 1
    assert metrics["split_counts_modelable_roles"]["train"] == 3
    assert metrics["metrics"]["valid"]["n"] == 1
    assert metrics["metrics"]["test"]["n"] == 1


def test_train_dummy_classifier_mvp_generates_artifacts_and_uses_train_positive_rate_only(
    tmp_workspace: dict[str, Path],
) -> None:
    base = tmp_workspace["data"] / "dummy_classifier_case"
    dataset_path = base / "model_dataset.parquet"
    output_dir = tmp_workspace["artifacts"] / "dummy_classifier_case"

    _write_model_dataset(
        dataset_path,
        rows=[
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
                "target_value": 1,
                "target_type": "binary_direction",
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
            },
            {
                "date": "2026-01-08",
                "instrument_id": "SIMA",
                "ticker": "AAA",
                "horizon_days": 5,
                "label_name": "fwd_dir_up_5d",
                "split_name": "holdout_temporal_purged",
                "split_role": "train",
                "entry_date": "2026-01-09",
                "exit_date": "2026-01-16",
                "target_value": 1,
                "target_type": "binary_direction",
            },
            {
                "date": "2026-01-09",
                "instrument_id": "SIMA",
                "ticker": "AAA",
                "horizon_days": 5,
                "label_name": "fwd_dir_up_5d",
                "split_name": "holdout_temporal_purged",
                "split_role": "valid",
                "entry_date": "2026-01-12",
                "exit_date": "2026-01-19",
                "target_value": 0,
                "target_type": "binary_direction",
            },
            {
                "date": "2026-01-12",
                "instrument_id": "SIMA",
                "ticker": "AAA",
                "horizon_days": 5,
                "label_name": "fwd_dir_up_5d",
                "split_name": "holdout_temporal_purged",
                "split_role": "test",
                "entry_date": "2026-01-13",
                "exit_date": "2026-01-20",
                "target_value": 0,
                "target_type": "binary_direction",
            },
            {
                "date": "2026-01-13",
                "instrument_id": "SIMA",
                "ticker": "AAA",
                "horizon_days": 5,
                "label_name": "fwd_dir_up_5d",
                "split_name": "holdout_temporal_purged",
                "split_role": "dropped_by_embargo",
                "entry_date": "2026-01-14",
                "exit_date": "2026-01-21",
                "target_value": 0,
                "target_type": "binary_direction",
            },
        ],
    )

    result = train_dummy_baseline(
        mode="dummy_classifier",
        model_dataset_path=dataset_path,
        output_dir=output_dir,
        label_name="fwd_dir_up_5d",
        dummy_strategy="prior",
        run_id="test_dummy_classifier_mvp",
    )

    assert result.metrics_path.exists()
    assert result.predictions_path.exists()
    assert np.isclose(result.learned_statistic, 0.75)
    assert result.n_train == 4
    assert result.n_valid == 1
    assert result.n_test == 1

    predictions = read_parquet(result.predictions_path)
    assert len(predictions) == 6
    assert set(predictions["split_role"].astype(str).unique().tolist()) == {"train", "valid", "test"}
    assert np.allclose(predictions["pred_proba"].to_numpy(dtype=float), 0.75)
    assert set(predictions["pred_class"].astype(int).unique().tolist()) == {1}
    assert "dropped_by_embargo" not in set(predictions["split_role"].astype(str).tolist())

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["mode"] == "dummy_classifier"
    assert metrics["dummy_strategy"] == "prior"
    assert metrics["target_type"] == "binary_direction"
    assert metrics["train_target_statistic_name"] == "positive_rate_train"
    assert np.isclose(metrics["train_target_statistic_value"], 0.75)
    assert metrics["split_counts_all_roles"]["dropped_by_embargo"] == 1
    assert metrics["class_balance"]["train"]["positive_rate"] == pytest.approx(0.75)
    assert metrics["metrics"]["valid"]["n"] == 1
    assert metrics["metrics"]["test"]["n"] == 1


def test_train_dummy_classifier_mvp_fails_on_non_binary_target(
    tmp_workspace: dict[str, Path],
) -> None:
    base = tmp_workspace["data"] / "dummy_classifier_non_binary_case"
    dataset_path = base / "model_dataset.parquet"
    output_dir = tmp_workspace["artifacts"] / "dummy_classifier_non_binary_case"

    _write_model_dataset(
        dataset_path,
        rows=[
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
                "target_value": 2,
                "target_type": "binary_direction",
            },
            {
                "date": "2026-01-07",
                "instrument_id": "SIMA",
                "ticker": "AAA",
                "horizon_days": 5,
                "label_name": "fwd_dir_up_5d",
                "split_name": "holdout_temporal_purged",
                "split_role": "valid",
                "entry_date": "2026-01-08",
                "exit_date": "2026-01-15",
                "target_value": 1,
                "target_type": "binary_direction",
            },
            {
                "date": "2026-01-08",
                "instrument_id": "SIMA",
                "ticker": "AAA",
                "horizon_days": 5,
                "label_name": "fwd_dir_up_5d",
                "split_name": "holdout_temporal_purged",
                "split_role": "test",
                "entry_date": "2026-01-09",
                "exit_date": "2026-01-16",
                "target_value": 0,
                "target_type": "binary_direction",
            },
        ],
    )

    with pytest.raises(ValueError, match="Expected binary target values"):
        _ = train_dummy_baseline(
            mode="dummy_classifier",
            model_dataset_path=dataset_path,
            output_dir=output_dir,
            label_name="fwd_dir_up_5d",
            run_id="test_dummy_classifier_non_binary",
        )

