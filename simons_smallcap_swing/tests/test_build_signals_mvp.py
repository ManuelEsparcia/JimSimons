from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from signals.build_signals import build_signals
from simons_core.io.parquet_store import read_parquet


def _seed_predictions_regression_case(tmp_workspace: dict[str, Path]) -> tuple[Path, Path]:
    base = tmp_workspace["data"] / "signals_regression_case"
    base.mkdir(parents=True, exist_ok=True)
    predictions_path = base / "ridge_baseline_predictions.parquet"
    universe_path = base / "universe_history.parquet"

    d1 = pd.Timestamp("2026-02-02")
    d2 = pd.Timestamp("2026-02-03")
    rows = [
        {"date": d1, "instrument_id": "SIMA", "ticker": "AAA", "split_name": "holdout_temporal_purged", "split_role": "valid", "horizon_days": 5, "label_name": "fwd_ret_5d", "prediction": 0.90},
        {"date": d1, "instrument_id": "SIMB", "ticker": "BBB", "split_name": "holdout_temporal_purged", "split_role": "valid", "horizon_days": 5, "label_name": "fwd_ret_5d", "prediction": 0.50},
        {"date": d1, "instrument_id": "SIMC", "ticker": "CCC", "split_name": "holdout_temporal_purged", "split_role": "valid", "horizon_days": 5, "label_name": "fwd_ret_5d", "prediction": 0.10},
        {"date": d1, "instrument_id": "SIMD", "ticker": "DDD", "split_name": "holdout_temporal_purged", "split_role": "valid", "horizon_days": 5, "label_name": "fwd_ret_5d", "prediction": -0.20},
        {"date": d1, "instrument_id": "SIME", "ticker": "EEE", "split_name": "holdout_temporal_purged", "split_role": "valid", "horizon_days": 5, "label_name": "fwd_ret_5d", "prediction": None},
        {"date": d1, "instrument_id": "SIMF", "ticker": "FFF", "split_name": "holdout_temporal_purged", "split_role": "train", "horizon_days": 5, "label_name": "fwd_ret_5d", "prediction": 0.80},
        {"date": d2, "instrument_id": "SIMA", "ticker": "AAA", "split_name": "holdout_temporal_purged", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "prediction": 0.30},
        {"date": d2, "instrument_id": "SIMB", "ticker": "BBB", "split_name": "holdout_temporal_purged", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "prediction": 0.20},
        {"date": d2, "instrument_id": "SIMC", "ticker": "CCC", "split_name": "holdout_temporal_purged", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "prediction": -0.10},
        {"date": d2, "instrument_id": "SIMD", "ticker": "DDD", "split_name": "holdout_temporal_purged", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "prediction": -0.40},
        {"date": d2, "instrument_id": "SIME", "ticker": "EEE", "split_name": "holdout_temporal_purged", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "prediction": 0.00},
    ]
    pd.DataFrame(rows).to_parquet(predictions_path, index=False)

    universe_rows = []
    for instrument_id in ("SIMA", "SIMB", "SIMC", "SIMD", "SIME", "SIMF"):
        universe_rows.append(
            {
                "date": d1,
                "instrument_id": instrument_id,
                "ticker": instrument_id[-3:],
                "is_eligible": True,
            }
        )
    for instrument_id in ("SIMA", "SIMB", "SIMC", "SIME"):
        universe_rows.append(
            {
                "date": d2,
                "instrument_id": instrument_id,
                "ticker": instrument_id[-3:],
                "is_eligible": True,
            }
        )
    universe_rows.append(
        {
            "date": d2,
            "instrument_id": "SIMD",
            "ticker": "DDD",
            "is_eligible": False,
        }
    )
    pd.DataFrame(universe_rows).to_parquet(universe_path, index=False)
    return predictions_path, universe_path


def _seed_predictions_classification_case(tmp_workspace: dict[str, Path]) -> tuple[Path, Path]:
    base = tmp_workspace["data"] / "signals_classification_case"
    base.mkdir(parents=True, exist_ok=True)
    predictions_path = base / "logistic_baseline_predictions.parquet"
    universe_path = base / "universe_history.parquet"

    d1 = pd.Timestamp("2026-02-04")
    d2 = pd.Timestamp("2026-02-05")
    rows = [
        {"date": d1, "instrument_id": "SIMA", "ticker": "AAA", "split_name": "holdout_temporal_purged", "split_role": "valid", "horizon_days": 5, "label_name": "fwd_dir_up_5d", "pred_proba": 0.90},
        {"date": d1, "instrument_id": "SIMB", "ticker": "BBB", "split_name": "holdout_temporal_purged", "split_role": "valid", "horizon_days": 5, "label_name": "fwd_dir_up_5d", "pred_proba": 0.40},
        {"date": d1, "instrument_id": "SIMC", "ticker": "CCC", "split_name": "holdout_temporal_purged", "split_role": "train", "horizon_days": 5, "label_name": "fwd_dir_up_5d", "pred_proba": 0.70},
        {"date": d2, "instrument_id": "SIMA", "ticker": "AAA", "split_name": "holdout_temporal_purged", "split_role": "test", "horizon_days": 5, "label_name": "fwd_dir_up_5d", "pred_proba": 0.60},
        {"date": d2, "instrument_id": "SIMB", "ticker": "BBB", "split_name": "holdout_temporal_purged", "split_role": "test", "horizon_days": 5, "label_name": "fwd_dir_up_5d", "pred_proba": 0.20},
        {"date": d2, "instrument_id": "SIMC", "ticker": "CCC", "split_name": "holdout_temporal_purged", "split_role": "test", "horizon_days": 5, "label_name": "fwd_dir_up_5d", "pred_proba": 0.80},
    ]
    pd.DataFrame(rows).to_parquet(predictions_path, index=False)

    universe_rows = []
    for date in (d1, d2):
        for instrument_id, ticker in (("SIMA", "AAA"), ("SIMB", "BBB"), ("SIMC", "CCC")):
            universe_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "is_eligible": True,
                }
            )
    pd.DataFrame(universe_rows).to_parquet(universe_path, index=False)
    return predictions_path, universe_path


def test_build_signals_mvp_regression_ranking_buckets_and_nan_handling(
    tmp_workspace: dict[str, Path],
) -> None:
    predictions_path, universe_path = _seed_predictions_regression_case(tmp_workspace)
    output_dir = tmp_workspace["artifacts"] / "signals_regression"

    result = build_signals(
        predictions_path=predictions_path,
        output_dir=output_dir,
        model_name="ridge_baseline",
        label_name="fwd_ret_5d",
        split_roles=("valid", "test"),
        n_buckets=4,
        top_buckets=1,
        bottom_buckets=1,
        universe_history_path=universe_path,
        use_universe_filter=True,
        run_id="test_signals_regression_mvp",
    )

    assert result.signals_path.exists()
    assert result.summary_path.exists()
    assert (result.signals_path.with_suffix(".parquet.manifest.json")).exists()
    assert result.row_count > 0

    signals = read_parquet(result.signals_path)
    assert len(signals) == 9  # train excluded + one ineligible row excluded

    required_cols = {
        "date",
        "instrument_id",
        "ticker",
        "split_role",
        "label_name",
        "model_name",
        "signal_source_type",
        "raw_score",
        "rank_pct",
        "bucket",
        "signal_side",
        "is_top",
        "is_bottom",
    }
    assert required_cols.issubset(set(signals.columns))
    assert set(signals["split_role"].astype(str).unique().tolist()).issubset({"valid", "test"})
    assert "train" not in set(signals["split_role"].astype(str).tolist())

    ranked = signals[signals["rank_pct"].notna()].copy()
    assert ranked["rank_pct"].between(0.0, 1.0).all()

    d1 = pd.Timestamp("2026-02-02")
    day1 = signals[signals["date"] == d1].copy()
    assert int(day1["n_names_ranked"].max()) == 4

    top_day1 = day1.loc[day1["instrument_id"] == "SIMA"].iloc[0]
    bottom_day1 = day1.loc[day1["instrument_id"] == "SIMD"].iloc[0]
    nan_day1 = day1.loc[day1["instrument_id"] == "SIME"].iloc[0]

    assert int(top_day1["bucket"]) == 4
    assert bool(top_day1["is_top"]) is True
    assert int(top_day1["signal_side"]) == 1

    assert int(bottom_day1["bucket"]) == 1
    assert bool(bottom_day1["is_bottom"]) is True
    assert int(bottom_day1["signal_side"]) == -1

    assert pd.isna(nan_day1["rank_pct"])
    assert pd.isna(nan_day1["bucket"])
    assert bool(nan_day1["is_top"]) is False
    assert bool(nan_day1["is_bottom"]) is False
    assert int(nan_day1["signal_side"]) == 0

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["model_name"] == "ridge_baseline"
    assert summary["signal_source_type"] == "regression_prediction"
    assert summary["n_rows_nan_score"] == 1
    assert summary["filter_stats"]["n_rows_excluded_by_split_role"] == 1
    assert summary["filter_stats"]["n_rows_excluded_by_universe_filter"] == 1


def test_build_signals_mvp_classification_uses_pred_proba_and_test_only_filter(
    tmp_workspace: dict[str, Path],
) -> None:
    predictions_path, universe_path = _seed_predictions_classification_case(tmp_workspace)
    output_dir = tmp_workspace["artifacts"] / "signals_classification"

    result = build_signals(
        predictions_path=predictions_path,
        output_dir=output_dir,
        model_name="logistic_baseline",
        label_name="fwd_dir_up_5d",
        split_roles=("test",),
        n_buckets=3,
        top_buckets=1,
        bottom_buckets=1,
        universe_history_path=universe_path,
        use_universe_filter=True,
        run_id="test_signals_classification_mvp",
    )

    assert result.signal_source_type == "classification_probability"
    signals = read_parquet(result.signals_path).sort_values(["date", "instrument_id"]).reset_index(drop=True)
    assert len(signals) == 3
    assert set(signals["split_role"].astype(str).unique().tolist()) == {"test"}
    assert signals["raw_score"].between(0.0, 1.0).all()
    assert signals["rank_pct"].between(0.0, 1.0).all()
    assert set(signals["signal_side"].astype(int).unique().tolist()).issubset({-1, 0, 1})

    top_row = signals.loc[signals["instrument_id"] == "SIMC"].iloc[0]
    bottom_row = signals.loc[signals["instrument_id"] == "SIMB"].iloc[0]
    assert int(top_row["bucket"]) == 3
    assert bool(top_row["is_top"]) is True
    assert int(top_row["signal_side"]) == 1

    assert int(bottom_row["bucket"]) == 1
    assert bool(bottom_row["is_bottom"]) is True
    assert int(bottom_row["signal_side"]) == -1

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["n_rows_output"] == 3
    assert summary["split_roles_included"] == ["test"]
    assert summary["n_rows_nan_score"] == 0
