from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_week4_cv_validation import run_week4_cv_validation
from simons_core.io.parquet_store import read_parquet


def _seed_week4_prerequisites(data_root: Path) -> dict[str, Path]:
    reference_root = data_root / "reference"
    labels_root = data_root / "labels"
    datasets_root = data_root / "datasets"
    universe_root = data_root / "universe"
    price_root = data_root / "price"
    edgar_root = data_root / "edgar"
    features_root = data_root / "features"
    for directory in (
        reference_root,
        labels_root,
        datasets_root,
        universe_root,
        price_root,
        edgar_root,
        features_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    sessions = pd.bdate_range("2026-01-05", periods=15, freq="B")
    trading_calendar = pd.DataFrame({"date": sessions, "is_session": True})
    trading_calendar_path = reference_root / "trading_calendar.parquet"
    trading_calendar.to_parquet(trading_calendar_path, index=False)

    instruments = [("SIMA", "AAA"), ("SIMB", "BBB")]
    label_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    for i, dt in enumerate(sessions[:-5]):
        entry_date = sessions[i + 1]
        exit_date = sessions[i + 5]
        for j, (instrument_id, ticker) in enumerate(instruments):
            ret_value = 0.01 * (i + 1) * (1 if j == 0 else -1)
            dir_value = int((i + j) % 2 == 0)

            label_rows.append(
                {
                    "date": dt,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "horizon_days": 5,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "label_name": "fwd_ret_5d",
                    "label_value": float(ret_value),
                    "price_entry": 100.0 + i,
                    "price_exit": 101.0 + i,
                    "source_price_field": "close_adj",
                }
            )
            label_rows.append(
                {
                    "date": dt,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "horizon_days": 5,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "label_name": "fwd_dir_up_5d",
                    "label_value": int(dir_value),
                    "price_entry": 100.0 + i,
                    "price_exit": 101.0 + i,
                    "source_price_field": "close_adj",
                }
            )

            common_fields = {
                "date": dt,
                "instrument_id": instrument_id,
                "ticker": ticker,
                "horizon_days": 5,
                "split_name": "holdout_temporal_purged",
                "split_role": "train",
                "entry_date": entry_date,
                "exit_date": exit_date,
                "f1": float(i + 1 + j),
                "f2": float((i % 3) - 1),
                "f3": float(np.nan if i % 4 == 0 else i + j),
            }
            model_rows.append(
                {
                    **common_fields,
                    "label_name": "fwd_ret_5d",
                    "target_type": "continuous_forward_return",
                    "target_value": float(ret_value),
                }
            )
            model_rows.append(
                {
                    **common_fields,
                    "label_name": "fwd_dir_up_5d",
                    "target_type": "binary_direction",
                    "target_value": int(dir_value),
                }
            )

    labels_forward_path = labels_root / "labels_forward.parquet"
    model_dataset_path = datasets_root / "model_dataset.parquet"
    pd.DataFrame(label_rows).to_parquet(labels_forward_path, index=False)
    pd.DataFrame(model_rows).to_parquet(model_dataset_path, index=False)

    # Compatibility placeholders from Week 1/2/3.
    week1_2_3_paths = {
        "universe_history": universe_root / "universe_history.parquet",
        "adjusted_prices": price_root / "adjusted_prices.parquet",
        "fundamentals_pit": edgar_root / "fundamentals_pit.parquet",
        "features_matrix": features_root / "features_matrix.parquet",
    }
    pd.DataFrame(
        {
            "date": [sessions[0]],
            "instrument_id": ["SIMA"],
            "ticker": ["AAA"],
            "is_eligible": [True],
        }
    ).to_parquet(week1_2_3_paths["universe_history"], index=False)
    pd.DataFrame(
        {
            "date": [sessions[0]],
            "instrument_id": ["SIMA"],
            "ticker": ["AAA"],
            "close_adj": [100.0],
            "volume_adj": [1_000_000.0],
        }
    ).to_parquet(week1_2_3_paths["adjusted_prices"], index=False)
    pd.DataFrame(
        {
            "instrument_id": ["SIMA"],
            "asof_date": [sessions[0]],
            "metric_name": ["Revenues"],
            "metric_value": [100_000_000.0],
        }
    ).to_parquet(week1_2_3_paths["fundamentals_pit"], index=False)
    pd.DataFrame(
        {
            "date": [sessions[0]],
            "instrument_id": ["SIMA"],
            "ticker": ["AAA"],
            "ret_1d_lag1": [0.01],
        }
    ).to_parquet(week1_2_3_paths["features_matrix"], index=False)

    return {
        "trading_calendar": trading_calendar_path,
        "labels_forward": labels_forward_path,
        "model_dataset": model_dataset_path,
        **week1_2_3_paths,
    }


def test_week4_runner_end_to_end_smoke(tmp_workspace: dict[str, Path]) -> None:
    prereq = _seed_week4_prerequisites(tmp_workspace["data"])
    compatibility_counts_before = {
        key: int(len(read_parquet(path)))
        for key, path in prereq.items()
        if key in {"universe_history", "adjusted_prices", "fundamentals_pit", "features_matrix"}
    }

    result = run_week4_cv_validation(
        run_prefix="test_week4_e2e",
        data_root=tmp_workspace["data"],
        n_folds=3,
        embargo_sessions=1,
        horizon_days=5,
    )

    expected_artifacts = {
        "purged_cv_folds": result.artifacts["purged_cv_folds"],
        "ridge_fold_metrics": result.artifacts["ridge_cv_fold_metrics"],
        "ridge_summary": result.artifacts["ridge_cv_summary"],
        "logistic_fold_metrics": result.artifacts["logistic_cv_fold_metrics"],
        "logistic_summary": result.artifacts["logistic_cv_summary"],
        "dummy_reg_fold_metrics": result.artifacts["dummy_regressor_cv_fold_metrics"],
        "dummy_reg_summary": result.artifacts["dummy_regressor_cv_summary"],
        "dummy_cls_fold_metrics": result.artifacts["dummy_classifier_cv_fold_metrics"],
        "dummy_cls_summary": result.artifacts["dummy_classifier_cv_summary"],
        "cv_comparison_summary": result.artifacts["cv_model_comparison_summary"],
        "cv_comparison_table": result.artifacts["cv_model_comparison_table"],
    }
    for path in expected_artifacts.values():
        assert path.exists()
        assert path.stat().st_size > 0

    purged_cv_folds = read_parquet(expected_artifacts["purged_cv_folds"])
    assert len(purged_cv_folds) > 0
    assert set(purged_cv_folds["split_role"].astype(str).unique().tolist()).issubset(
        {"train", "valid", "dropped_by_purge", "dropped_by_embargo"}
    )

    for key in (
        "ridge_fold_metrics",
        "logistic_fold_metrics",
        "dummy_reg_fold_metrics",
        "dummy_cls_fold_metrics",
    ):
        frame = read_parquet(expected_artifacts[key])
        assert len(frame) > 0
        assert "valid_primary_metric" in frame.columns

    for pred_key in (
        "ridge_cv_predictions",
        "logistic_cv_predictions",
        "dummy_regressor_cv_predictions",
        "dummy_classifier_cv_predictions",
    ):
        pred_frame = read_parquet(result.artifacts[pred_key])
        roles = set(pred_frame["split_role"].astype(str).unique().tolist())
        assert roles.issubset({"train", "valid"})
        assert "dropped_by_purge" not in roles
        assert "dropped_by_embargo" not in roles

    comparison_summary = json.loads(expected_artifacts["cv_comparison_summary"].read_text(encoding="utf-8"))
    assert set(comparison_summary["tasks_compared"]) == {
        "regression_cv_baselines",
        "classification_cv_baselines",
    }
    assert comparison_summary["regression"]["comparability_status"] in {
        "comparable",
        "non_comparable",
        "missing",
    }
    assert comparison_summary["classification"]["comparability_status"] in {
        "comparable",
        "non_comparable",
        "missing",
    }

    comparison_table = read_parquet(expected_artifacts["cv_comparison_table"])
    assert len(comparison_table) == 2
    assert set(comparison_table["task_name"].astype(str).tolist()) == {
        "regression_cv_baselines",
        "classification_cv_baselines",
    }

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_prefix"] == "test_week4_e2e"
    assert manifest["statuses"]["build_purged_cv"] == "DONE"
    assert manifest["statuses"]["ridge_cv"] == "DONE"
    assert manifest["statuses"]["logistic_cv"] == "DONE"
    assert manifest["statuses"]["dummy_regressor_cv"] == "DONE"
    assert manifest["statuses"]["dummy_classifier_cv"] == "DONE"
    assert manifest["statuses"]["cv_model_comparison"] == "DONE"

    # Compatibility with Week 1/2/3 placeholders remains intact.
    compatibility_counts_after = {
        key: int(len(read_parquet(path)))
        for key, path in prereq.items()
        if key in compatibility_counts_before
    }
    assert compatibility_counts_after == compatibility_counts_before
