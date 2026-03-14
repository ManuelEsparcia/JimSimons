from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.baselines.cross_validated_baselines import run_cross_validated_baseline
from simons_core.io.parquet_store import read_parquet


def _build_cv_inputs(
    *,
    tmp_workspace: dict[str, Path],
    label_name: str,
    target_type: str,
    make_fold2_invalid: bool = False,
) -> tuple[Path, Path]:
    base = tmp_workspace["data"] / "cv_baseline_case"
    base.mkdir(parents=True, exist_ok=True)
    dataset_path = base / "model_dataset.parquet"
    folds_path = base / "purged_cv_folds.parquet"

    sessions = pd.bdate_range("2026-01-05", periods=6, freq="B")
    instruments = [("SIMA", "AAA"), ("SIMB", "BBB")]

    dataset_rows: list[dict[str, object]] = []
    for idx, session in enumerate(sessions):
        for inst_pos, (instrument_id, ticker) in enumerate(instruments):
            target_value: float
            if target_type == "binary_direction":
                target_value = float((idx + inst_pos) % 2)
            else:
                target_value = float((idx + 1) * (0.01 if inst_pos == 0 else -0.01))

            if idx <= 2:
                f2_value = 1.0 if idx == 0 else (np.nan if idx == 1 else 3.0)
            else:
                f2_value = 9999.0

            dataset_rows.append(
                {
                    "date": session,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "horizon_days": 5,
                    "label_name": label_name,
                    "split_name": "holdout_temporal_purged",
                    "split_role": "train",
                    "entry_date": sessions[min(idx + 1, len(sessions) - 1)],
                    "exit_date": sessions[min(idx + 2, len(sessions) - 1)],
                    "target_value": target_value,
                    "target_type": target_type,
                    "f1": float(idx + 1 + inst_pos),
                    "f2": f2_value,
                    "f3": np.nan,
                }
            )

    fold_rows: list[dict[str, object]] = []
    for idx, session in enumerate(sessions):
        for instrument_id, _ticker in instruments:
            # Fold 1: valid at idx 3-4, embargo at idx 5.
            if idx in {3, 4}:
                role_fold1 = "valid"
            elif idx == 5:
                role_fold1 = "dropped_by_embargo"
            else:
                role_fold1 = "train"
            fold_rows.append(
                {
                    "fold_id": 1,
                    "date": session,
                    "instrument_id": instrument_id,
                    "horizon_days": 5,
                    "label_name": label_name,
                    "split_role": role_fold1,
                    "entry_date": sessions[min(idx + 1, len(sessions) - 1)],
                    "exit_date": sessions[min(idx + 2, len(sessions) - 1)],
                }
            )

            # Fold 2: valid at idx 4-5, purge at idx 2.
            if make_fold2_invalid:
                role_fold2 = "train" if idx <= 3 else "dropped_by_purge"
            else:
                if idx in {4, 5}:
                    role_fold2 = "valid"
                elif idx == 2:
                    role_fold2 = "dropped_by_purge"
                else:
                    role_fold2 = "train"
            fold_rows.append(
                {
                    "fold_id": 2,
                    "date": session,
                    "instrument_id": instrument_id,
                    "horizon_days": 5,
                    "label_name": label_name,
                    "split_role": role_fold2,
                    "entry_date": sessions[min(idx + 1, len(sessions) - 1)],
                    "exit_date": sessions[min(idx + 2, len(sessions) - 1)],
                }
            )

    pd.DataFrame(dataset_rows).to_parquet(dataset_path, index=False)
    pd.DataFrame(fold_rows).to_parquet(folds_path, index=False)
    return dataset_path, folds_path


def test_cross_validated_ridge_mvp_generates_artifacts_and_respects_train_only_preprocessing(
    tmp_workspace: dict[str, Path],
) -> None:
    dataset_path, folds_path = _build_cv_inputs(
        tmp_workspace=tmp_workspace,
        label_name="fwd_ret_5d",
        target_type="continuous_forward_return",
    )
    output_dir = tmp_workspace["artifacts"] / "cv_ridge"
    result = run_cross_validated_baseline(
        mode="ridge_cv",
        model_dataset_path=dataset_path,
        purged_cv_folds_path=folds_path,
        output_dir=output_dir,
        label_name="fwd_ret_5d",
        horizon_days=5,
        alpha_grid=(0.1, 1.0, 10.0),
        run_id="test_cv_ridge_mvp",
    )

    assert result.fold_metrics_path.exists()
    assert result.summary_path.exists()
    assert result.predictions_path is not None and result.predictions_path.exists()

    fold_metrics = read_parquet(result.fold_metrics_path)
    assert len(fold_metrics) == 2
    assert set(fold_metrics["status"].astype(str).tolist()) == {"completed"}
    assert "valid_mse" in fold_metrics.columns
    assert "train_mse" in fold_metrics.columns
    assert "alpha_selected" in fold_metrics.columns

    fold1 = fold_metrics.loc[fold_metrics["fold_id"].astype(int) == 1].iloc[0]
    medians = json.loads(str(fold1["imputer_median_train_only_json"]))
    dropped = json.loads(str(fold1["features_dropped_all_nan_train_json"]))
    assert medians["f2"] != 9999.0
    assert "f3" in dropped

    predictions = read_parquet(result.predictions_path)
    assert len(predictions) > 0
    assert set(predictions["split_role"].astype(str).unique().tolist()) == {"train", "valid"}
    assert "dropped_by_purge" not in set(predictions["split_role"].astype(str).tolist())
    assert "dropped_by_embargo" not in set(predictions["split_role"].astype(str).tolist())

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["model_name"] == "ridge_cv"
    assert summary["primary_metric"] == "mse"
    assert summary["n_folds"] == 2
    assert summary["folds_completed"] == 2


def test_cross_validated_logistic_mvp_generates_fold_and_summary_artifacts(
    tmp_workspace: dict[str, Path],
) -> None:
    dataset_path, folds_path = _build_cv_inputs(
        tmp_workspace=tmp_workspace,
        label_name="fwd_dir_up_5d",
        target_type="binary_direction",
    )
    output_dir = tmp_workspace["artifacts"] / "cv_logistic"
    result = run_cross_validated_baseline(
        mode="logistic_cv",
        model_dataset_path=dataset_path,
        purged_cv_folds_path=folds_path,
        output_dir=output_dir,
        label_name="fwd_dir_up_5d",
        horizon_days=5,
        c_grid=(0.1, 1.0, 10.0),
        run_id="test_cv_logistic_mvp",
    )

    assert result.fold_metrics_path.exists()
    assert result.summary_path.exists()
    assert result.predictions_path is not None and result.predictions_path.exists()

    fold_metrics = read_parquet(result.fold_metrics_path)
    assert len(fold_metrics) == 2
    assert set(fold_metrics["status"].astype(str).tolist()) == {"completed"}
    assert "valid_log_loss" in fold_metrics.columns
    assert "c_selected" in fold_metrics.columns

    predictions = read_parquet(result.predictions_path)
    assert len(predictions) > 0
    assert predictions["pred_proba"].between(0.0, 1.0).all()
    assert set(predictions["pred_class"].astype(int).unique().tolist()).issubset({0, 1})
    assert set(predictions["split_role"].astype(str).unique().tolist()) == {"train", "valid"}

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["model_name"] == "logistic_cv"
    assert summary["primary_metric"] == "log_loss"
    assert summary["n_folds"] == 2
    assert summary["folds_completed"] == 2


def test_cross_validated_baselines_handle_invalid_fold_with_skip_or_fail(
    tmp_workspace: dict[str, Path],
) -> None:
    dataset_path, folds_path = _build_cv_inputs(
        tmp_workspace=tmp_workspace,
        label_name="fwd_ret_5d",
        target_type="continuous_forward_return",
        make_fold2_invalid=True,
    )

    skip_result = run_cross_validated_baseline(
        mode="ridge_cv",
        model_dataset_path=dataset_path,
        purged_cv_folds_path=folds_path,
        output_dir=tmp_workspace["artifacts"] / "cv_invalid_skip",
        label_name="fwd_ret_5d",
        horizon_days=5,
        fail_on_invalid_fold=False,
        run_id="test_cv_invalid_skip",
    )
    skip_metrics = read_parquet(skip_result.fold_metrics_path)
    statuses = set(skip_metrics["status"].astype(str).tolist())
    assert statuses == {"completed", "skipped"}
    assert (skip_metrics["fold_id"].astype(int) == 2).any()

    with pytest.raises(ValueError, match="missing train/valid rows"):
        run_cross_validated_baseline(
            mode="ridge_cv",
            model_dataset_path=dataset_path,
            purged_cv_folds_path=folds_path,
            output_dir=tmp_workspace["artifacts"] / "cv_invalid_fail",
            label_name="fwd_ret_5d",
            horizon_days=5,
            fail_on_invalid_fold=True,
            run_id="test_cv_invalid_fail",
        )


def test_cross_validated_dummy_regressor_cv_generates_artifacts_and_train_stat_only(
    tmp_workspace: dict[str, Path],
) -> None:
    dataset_path, folds_path = _build_cv_inputs(
        tmp_workspace=tmp_workspace,
        label_name="fwd_ret_5d",
        target_type="continuous_forward_return",
    )

    # Make fold-1 train target mean deterministic (1.0) and valid very different (100.0)
    df = pd.read_parquet(dataset_path)
    train_dates_fold1 = sorted(pd.to_datetime(df["date"]).unique().tolist())[:3]
    valid_dates_fold1 = sorted(pd.to_datetime(df["date"]).unique().tolist())[3:5]
    df.loc[pd.to_datetime(df["date"]).isin(train_dates_fold1), "target_value"] = 1.0
    df.loc[pd.to_datetime(df["date"]).isin(valid_dates_fold1), "target_value"] = 100.0
    df.to_parquet(dataset_path, index=False)

    output_dir = tmp_workspace["artifacts"] / "cv_dummy_regressor"
    result = run_cross_validated_baseline(
        mode="dummy_regressor_cv",
        model_dataset_path=dataset_path,
        purged_cv_folds_path=folds_path,
        output_dir=output_dir,
        label_name="fwd_ret_5d",
        horizon_days=5,
        dummy_strategy="mean",
        run_id="test_cv_dummy_regressor",
    )

    assert result.fold_metrics_path.exists()
    assert result.summary_path.exists()
    assert result.predictions_path is not None and result.predictions_path.exists()

    fold_metrics = read_parquet(result.fold_metrics_path)
    assert len(fold_metrics) == 2
    assert set(fold_metrics["status"].astype(str).tolist()) == {"completed"}
    assert set(fold_metrics["dummy_strategy"].astype(str).tolist()) == {"mean"}
    fold1 = fold_metrics.loc[fold_metrics["fold_id"].astype(int) == 1].iloc[0]
    assert np.isclose(float(fold1["dummy_stat_train"]), 1.0)
    assert float(fold1["n_features_used"]) == 0.0

    predictions = read_parquet(result.predictions_path)
    assert set(predictions["split_role"].astype(str).unique().tolist()) == {"train", "valid"}
    fold1_valid = predictions[
        (predictions["fold_id"].astype(int) == 1)
        & (predictions["split_role"].astype(str) == "valid")
    ]
    assert len(fold1_valid) > 0
    assert np.isclose(fold1_valid["prediction"].astype(float), 1.0).all()

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["model_name"] == "dummy_regressor_cv"
    assert summary["dummy_strategy"] == "mean"
    assert summary["primary_metric"] == "mse"
    assert summary["folds_completed"] == 2


def test_cross_validated_dummy_classifier_cv_generates_artifacts_and_train_prior_only(
    tmp_workspace: dict[str, Path],
) -> None:
    dataset_path, folds_path = _build_cv_inputs(
        tmp_workspace=tmp_workspace,
        label_name="fwd_dir_up_5d",
        target_type="binary_direction",
    )

    # Fold-1 train all zeros, valid all ones to check prior learned only on train.
    df = pd.read_parquet(dataset_path)
    train_dates_fold1 = sorted(pd.to_datetime(df["date"]).unique().tolist())[:3]
    valid_dates_fold1 = sorted(pd.to_datetime(df["date"]).unique().tolist())[3:5]
    df.loc[pd.to_datetime(df["date"]).isin(train_dates_fold1), "target_value"] = 0
    df.loc[pd.to_datetime(df["date"]).isin(valid_dates_fold1), "target_value"] = 1
    df.to_parquet(dataset_path, index=False)

    output_dir = tmp_workspace["artifacts"] / "cv_dummy_classifier"
    result = run_cross_validated_baseline(
        mode="dummy_classifier_cv",
        model_dataset_path=dataset_path,
        purged_cv_folds_path=folds_path,
        output_dir=output_dir,
        label_name="fwd_dir_up_5d",
        horizon_days=5,
        dummy_strategy="prior",
        run_id="test_cv_dummy_classifier",
    )

    assert result.fold_metrics_path.exists()
    assert result.summary_path.exists()
    assert result.predictions_path is not None and result.predictions_path.exists()

    fold_metrics = read_parquet(result.fold_metrics_path)
    assert len(fold_metrics) == 2
    assert set(fold_metrics["status"].astype(str).tolist()) == {"completed"}
    assert set(fold_metrics["dummy_strategy"].astype(str).tolist()) == {"prior"}
    fold1 = fold_metrics.loc[fold_metrics["fold_id"].astype(int) == 1].iloc[0]
    assert np.isclose(float(fold1["dummy_stat_train"]), 0.0)
    assert float(fold1["n_features_used"]) == 0.0

    predictions = read_parquet(result.predictions_path)
    assert set(predictions["split_role"].astype(str).unique().tolist()) == {"train", "valid"}
    fold1_valid = predictions[
        (predictions["fold_id"].astype(int) == 1)
        & (predictions["split_role"].astype(str) == "valid")
    ]
    assert len(fold1_valid) > 0
    assert set(fold1_valid["pred_class"].astype(int).unique().tolist()) == {0}
    assert (fold1_valid["pred_proba"].astype(float) <= 1e-3).all()

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["model_name"] == "dummy_classifier_cv"
    assert summary["dummy_strategy"] == "prior"
    assert summary["primary_metric"] == "log_loss"
    assert summary["folds_completed"] == 2
