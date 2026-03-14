from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signals.decile_analysis import run_decile_analysis
from simons_core.io.parquet_store import read_parquet


def _seed_decile_case(tmp_workspace: dict[str, Path]) -> tuple[Path, Path]:
    base = tmp_workspace["data"] / "signals_decile_case"
    base.mkdir(parents=True, exist_ok=True)
    signals_path = base / "signals_daily.parquet"
    labels_path = base / "labels_forward.parquet"

    d1 = pd.Timestamp("2026-03-02")
    d2 = pd.Timestamp("2026-03-03")
    signals_rows = [
        {"date": d1, "instrument_id": "SIMA", "ticker": "AAA", "split_role": "test", "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "signal_source_type": "regression_prediction", "raw_score": 0.90, "rank_pct": 1.0, "bucket": 4, "signal_side": 1, "is_top": True, "is_bottom": False, "n_names_ranked": 4, "bucket_scheme": "quantile_4_top1_bottom1"},
        {"date": d1, "instrument_id": "SIMB", "ticker": "BBB", "split_role": "test", "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "signal_source_type": "regression_prediction", "raw_score": 0.60, "rank_pct": 0.75, "bucket": 3, "signal_side": 0, "is_top": False, "is_bottom": False, "n_names_ranked": 4, "bucket_scheme": "quantile_4_top1_bottom1"},
        {"date": d1, "instrument_id": "SIMC", "ticker": "CCC", "split_role": "test", "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "signal_source_type": "regression_prediction", "raw_score": 0.20, "rank_pct": 0.50, "bucket": 2, "signal_side": 0, "is_top": False, "is_bottom": False, "n_names_ranked": 4, "bucket_scheme": "quantile_4_top1_bottom1"},
        {"date": d1, "instrument_id": "SIMD", "ticker": "DDD", "split_role": "test", "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "signal_source_type": "regression_prediction", "raw_score": -0.10, "rank_pct": 0.25, "bucket": 1, "signal_side": -1, "is_top": False, "is_bottom": True, "n_names_ranked": 4, "bucket_scheme": "quantile_4_top1_bottom1"},
        {"date": d2, "instrument_id": "SIMA", "ticker": "AAA", "split_role": "test", "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "signal_source_type": "regression_prediction", "raw_score": 0.80, "rank_pct": 1.0, "bucket": 4, "signal_side": 1, "is_top": True, "is_bottom": False, "n_names_ranked": 4, "bucket_scheme": "quantile_4_top1_bottom1"},
        {"date": d2, "instrument_id": "SIMB", "ticker": "BBB", "split_role": "test", "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "signal_source_type": "regression_prediction", "raw_score": 0.50, "rank_pct": 0.75, "bucket": 3, "signal_side": 0, "is_top": False, "is_bottom": False, "n_names_ranked": 4, "bucket_scheme": "quantile_4_top1_bottom1"},
        {"date": d2, "instrument_id": "SIMC", "ticker": "CCC", "split_role": "test", "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "signal_source_type": "regression_prediction", "raw_score": 0.10, "rank_pct": 0.50, "bucket": 2, "signal_side": 0, "is_top": False, "is_bottom": False, "n_names_ranked": 4, "bucket_scheme": "quantile_4_top1_bottom1"},
        {"date": d2, "instrument_id": "SIMD", "ticker": "DDD", "split_role": "test", "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "signal_source_type": "regression_prediction", "raw_score": -0.20, "rank_pct": 0.25, "bucket": 1, "signal_side": -1, "is_top": False, "is_bottom": True, "n_names_ranked": 4, "bucket_scheme": "quantile_4_top1_bottom1"},
    ]
    pd.DataFrame(signals_rows).to_parquet(signals_path, index=False)

    labels_rows = [
        {"date": d1, "instrument_id": "SIMA", "ticker": "AAA", "horizon_days": 5, "entry_date": d1 + pd.Timedelta(days=1), "exit_date": d1 + pd.Timedelta(days=6), "label_name": "fwd_ret_5d", "label_value": 0.04, "price_entry": 100.0, "price_exit": 104.0, "source_price_field": "close_adj"},
        {"date": d1, "instrument_id": "SIMB", "ticker": "BBB", "horizon_days": 5, "entry_date": d1 + pd.Timedelta(days=1), "exit_date": d1 + pd.Timedelta(days=6), "label_name": "fwd_ret_5d", "label_value": 0.02, "price_entry": 100.0, "price_exit": 102.0, "source_price_field": "close_adj"},
        {"date": d1, "instrument_id": "SIMC", "ticker": "CCC", "horizon_days": 5, "entry_date": d1 + pd.Timedelta(days=1), "exit_date": d1 + pd.Timedelta(days=6), "label_name": "fwd_ret_5d", "label_value": -0.01, "price_entry": 100.0, "price_exit": 99.0, "source_price_field": "close_adj"},
        {"date": d1, "instrument_id": "SIMD", "ticker": "DDD", "horizon_days": 5, "entry_date": d1 + pd.Timedelta(days=1), "exit_date": d1 + pd.Timedelta(days=6), "label_name": "fwd_ret_5d", "label_value": -0.03, "price_entry": 100.0, "price_exit": 97.0, "source_price_field": "close_adj"},
        {"date": d2, "instrument_id": "SIMA", "ticker": "AAA", "horizon_days": 5, "entry_date": d2 + pd.Timedelta(days=1), "exit_date": d2 + pd.Timedelta(days=6), "label_name": "fwd_ret_5d", "label_value": 0.03, "price_entry": 100.0, "price_exit": 103.0, "source_price_field": "close_adj"},
        {"date": d2, "instrument_id": "SIMB", "ticker": "BBB", "horizon_days": 5, "entry_date": d2 + pd.Timedelta(days=1), "exit_date": d2 + pd.Timedelta(days=6), "label_name": "fwd_ret_5d", "label_value": 0.01, "price_entry": 100.0, "price_exit": 101.0, "source_price_field": "close_adj"},
        {"date": d2, "instrument_id": "SIMC", "ticker": "CCC", "horizon_days": 5, "entry_date": d2 + pd.Timedelta(days=1), "exit_date": d2 + pd.Timedelta(days=6), "label_name": "fwd_ret_5d", "label_value": -0.02, "price_entry": 100.0, "price_exit": 98.0, "source_price_field": "close_adj"},
        {"date": d2, "instrument_id": "SIMD", "ticker": "DDD", "horizon_days": 5, "entry_date": d2 + pd.Timedelta(days=1), "exit_date": d2 + pd.Timedelta(days=6), "label_name": "fwd_ret_5d", "label_value": -0.04, "price_entry": 100.0, "price_exit": 96.0, "source_price_field": "close_adj"},
    ]
    pd.DataFrame(labels_rows).to_parquet(labels_path, index=False)
    return signals_path, labels_path


def test_decile_analysis_mvp_generates_artifacts_and_expected_spread(
    tmp_workspace: dict[str, Path],
) -> None:
    signals_path, labels_path = _seed_decile_case(tmp_workspace)
    output_dir = tmp_workspace["artifacts"] / "decile_analysis"

    result = run_decile_analysis(
        signals_path=signals_path,
        labels_path=labels_path,
        output_dir=output_dir,
        model_name="ridge_baseline",
        label_name="fwd_ret_5d",
        horizon_days=5,
        split_roles=("test",),
        expected_n_buckets=4,
        run_id="test_decile_analysis_mvp",
    )

    assert result.decile_daily_path.exists()
    assert result.decile_summary_path.exists()
    assert result.summary_json_path.exists()
    assert result.row_count_daily > 0
    assert result.row_count_summary == 4

    daily = read_parquet(result.decile_daily_path)
    summary = read_parquet(result.decile_summary_path)
    summary_json = json.loads(result.summary_json_path.read_text(encoding="utf-8"))

    assert len(daily) == 8  # 2 dates x 4 buckets
    assert len(summary) == 4
    assert (daily["n_names_bucket"] == 1).all()
    assert daily["bucket"].between(1, 4).all()
    assert np.isclose(float(summary_json["mean_top_minus_bottom_spread"]), 0.07, atol=1e-12)
    assert np.isclose(float(summary_json["positive_spread_rate"]), 1.0, atol=1e-12)
    assert summary_json["monotonicity_score"] is not None
    assert float(summary_json["monotonicity_score"]) > 0.9
    assert summary_json["n_buckets"] == 4
    assert summary_json["n_dates"] == 2

    # Spread should be same across all rows of each date.
    spread_by_date = daily.groupby("date")["top_minus_bottom_spread"].first().to_dict()
    for value in spread_by_date.values():
        assert np.isclose(float(value), 0.07, atol=1e-12)


def test_decile_analysis_fails_on_ambiguous_model_or_label(
    tmp_workspace: dict[str, Path],
) -> None:
    signals_path, labels_path = _seed_decile_case(tmp_workspace)
    signals = read_parquet(signals_path).copy()
    signals.loc[signals.index[0], "model_name"] = "other_model"
    signals.to_parquet(signals_path, index=False)

    with pytest.raises(ValueError, match="Expected exactly one model_name"):
        _ = run_decile_analysis(
            signals_path=signals_path,
            labels_path=labels_path,
            output_dir=tmp_workspace["artifacts"] / "decile_analysis_ambiguous",
            # model_name intentionally omitted
            label_name="fwd_ret_5d",
            horizon_days=5,
            split_roles=("test",),
            run_id="test_decile_analysis_ambiguous",
        )
