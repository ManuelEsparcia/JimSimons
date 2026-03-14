from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signals.paper_portfolio import run_paper_portfolio
from simons_core.io.parquet_store import read_parquet


def _seed_paper_portfolio_case(tmp_workspace: dict[str, Path]) -> tuple[Path, Path]:
    base = tmp_workspace["data"] / "paper_portfolio_case"
    base.mkdir(parents=True, exist_ok=True)
    signals_path = base / "signals_daily.parquet"
    labels_path = base / "labels_forward.parquet"

    d1 = pd.Timestamp("2026-03-10")
    d2 = pd.Timestamp("2026-03-11")
    d3 = pd.Timestamp("2026-03-12")

    signals_rows = [
        # Date 1: top + bottom present (executable for both modes)
        {"date": d1, "instrument_id": "SIMA", "ticker": "AAA", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "signal_source_type": "regression_prediction", "raw_score": 0.9, "rank_pct": 1.0, "bucket": 4, "signal_side": 1, "is_top": True, "is_bottom": False, "n_names_ranked": 5, "bucket_scheme": "quantile_4_top1_bottom1"},
        {"date": d1, "instrument_id": "SIMB", "ticker": "BBB", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "signal_source_type": "regression_prediction", "raw_score": 0.8, "rank_pct": 0.8, "bucket": 4, "signal_side": 1, "is_top": True, "is_bottom": False, "n_names_ranked": 5, "bucket_scheme": "quantile_4_top1_bottom1"},
        {"date": d1, "instrument_id": "SIMC", "ticker": "CCC", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "signal_source_type": "regression_prediction", "raw_score": 0.2, "rank_pct": 0.6, "bucket": 2, "signal_side": 0, "is_top": False, "is_bottom": False, "n_names_ranked": 5, "bucket_scheme": "quantile_4_top1_bottom1"},
        {"date": d1, "instrument_id": "SIMD", "ticker": "DDD", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "signal_source_type": "regression_prediction", "raw_score": -0.2, "rank_pct": 0.4, "bucket": 1, "signal_side": -1, "is_top": False, "is_bottom": True, "n_names_ranked": 5, "bucket_scheme": "quantile_4_top1_bottom1"},
        {"date": d1, "instrument_id": "SIME", "ticker": "EEE", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "signal_source_type": "regression_prediction", "raw_score": -0.3, "rank_pct": 0.2, "bucket": 1, "signal_side": -1, "is_top": False, "is_bottom": True, "n_names_ranked": 5, "bucket_scheme": "quantile_4_top1_bottom1"},
        # Date 2: top present, bottom missing (long_only executable, long_short skipped)
        {"date": d2, "instrument_id": "SIMA", "ticker": "AAA", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "signal_source_type": "regression_prediction", "raw_score": 0.5, "rank_pct": 1.0, "bucket": 4, "signal_side": 1, "is_top": True, "is_bottom": False, "n_names_ranked": 2, "bucket_scheme": "quantile_4_top1_bottom1"},
        {"date": d2, "instrument_id": "SIMB", "ticker": "BBB", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "signal_source_type": "regression_prediction", "raw_score": 0.1, "rank_pct": 0.5, "bucket": 2, "signal_side": 0, "is_top": False, "is_bottom": False, "n_names_ranked": 2, "bucket_scheme": "quantile_4_top1_bottom1"},
        # Date 3: bottom present, top missing (both requested modes skipped)
        {"date": d3, "instrument_id": "SIMD", "ticker": "DDD", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "signal_source_type": "regression_prediction", "raw_score": -0.6, "rank_pct": 0.5, "bucket": 1, "signal_side": -1, "is_top": False, "is_bottom": True, "n_names_ranked": 1, "bucket_scheme": "quantile_4_top1_bottom1"},
    ]
    pd.DataFrame(signals_rows).to_parquet(signals_path, index=False)

    labels_rows = [
        {"date": d1, "instrument_id": "SIMA", "ticker": "AAA", "horizon_days": 5, "entry_date": d1 + pd.Timedelta(days=1), "exit_date": d1 + pd.Timedelta(days=6), "label_name": "fwd_ret_5d", "label_value": 0.04, "price_entry": 100.0, "price_exit": 104.0, "source_price_field": "close_adj"},
        {"date": d1, "instrument_id": "SIMB", "ticker": "BBB", "horizon_days": 5, "entry_date": d1 + pd.Timedelta(days=1), "exit_date": d1 + pd.Timedelta(days=6), "label_name": "fwd_ret_5d", "label_value": 0.02, "price_entry": 100.0, "price_exit": 102.0, "source_price_field": "close_adj"},
        {"date": d1, "instrument_id": "SIMC", "ticker": "CCC", "horizon_days": 5, "entry_date": d1 + pd.Timedelta(days=1), "exit_date": d1 + pd.Timedelta(days=6), "label_name": "fwd_ret_5d", "label_value": 0.00, "price_entry": 100.0, "price_exit": 100.0, "source_price_field": "close_adj"},
        {"date": d1, "instrument_id": "SIMD", "ticker": "DDD", "horizon_days": 5, "entry_date": d1 + pd.Timedelta(days=1), "exit_date": d1 + pd.Timedelta(days=6), "label_name": "fwd_ret_5d", "label_value": -0.03, "price_entry": 100.0, "price_exit": 97.0, "source_price_field": "close_adj"},
        {"date": d1, "instrument_id": "SIME", "ticker": "EEE", "horizon_days": 5, "entry_date": d1 + pd.Timedelta(days=1), "exit_date": d1 + pd.Timedelta(days=6), "label_name": "fwd_ret_5d", "label_value": -0.01, "price_entry": 100.0, "price_exit": 99.0, "source_price_field": "close_adj"},
        {"date": d2, "instrument_id": "SIMA", "ticker": "AAA", "horizon_days": 5, "entry_date": d2 + pd.Timedelta(days=1), "exit_date": d2 + pd.Timedelta(days=6), "label_name": "fwd_ret_5d", "label_value": 0.01, "price_entry": 100.0, "price_exit": 101.0, "source_price_field": "close_adj"},
        {"date": d2, "instrument_id": "SIMB", "ticker": "BBB", "horizon_days": 5, "entry_date": d2 + pd.Timedelta(days=1), "exit_date": d2 + pd.Timedelta(days=6), "label_name": "fwd_ret_5d", "label_value": -0.01, "price_entry": 100.0, "price_exit": 99.0, "source_price_field": "close_adj"},
        {"date": d3, "instrument_id": "SIMD", "ticker": "DDD", "horizon_days": 5, "entry_date": d3 + pd.Timedelta(days=1), "exit_date": d3 + pd.Timedelta(days=6), "label_name": "fwd_ret_5d", "label_value": -0.02, "price_entry": 100.0, "price_exit": 98.0, "source_price_field": "close_adj"},
    ]
    pd.DataFrame(labels_rows).to_parquet(labels_path, index=False)
    return signals_path, labels_path


def test_paper_portfolio_mvp_generates_artifacts_and_valid_weights_returns(
    tmp_workspace: dict[str, Path],
) -> None:
    signals_path, labels_path = _seed_paper_portfolio_case(tmp_workspace)
    output_dir = tmp_workspace["artifacts"] / "paper_portfolio"

    result = run_paper_portfolio(
        signals_path=signals_path,
        labels_path=labels_path,
        output_dir=output_dir,
        model_name="ridge_baseline",
        label_name="fwd_ret_5d",
        horizon_days=5,
        split_roles=("test",),
        portfolio_modes=("long_only_top", "long_short_top_bottom"),
        run_id="test_paper_portfolio_mvp",
    )

    assert result.daily_path.exists()
    assert result.positions_path.exists()
    assert result.summary_path.exists()
    assert result.row_count_daily > 0
    assert result.row_count_positions > 0

    daily = read_parquet(result.daily_path)
    positions = read_parquet(result.positions_path)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert set(daily["portfolio_mode"].astype(str).unique().tolist()) == {
        "long_only_top",
        "long_short_top_bottom",
    }
    assert set(positions["portfolio_mode"].astype(str).unique().tolist()) == {
        "long_only_top",
        "long_short_top_bottom",
    }

    d1 = pd.Timestamp("2026-03-10")
    d1_ls = daily[(daily["date"] == d1) & (daily["portfolio_mode"] == "long_short_top_bottom")].iloc[0]
    assert bool(d1_ls["is_executable"]) is True
    assert np.isclose(float(d1_ls["gross_portfolio_return"]), 0.05, atol=1e-12)

    d1_pos_ls = positions[(positions["date"] == d1) & (positions["portfolio_mode"] == "long_short_top_bottom")]
    long_weights = d1_pos_ls.loc[d1_pos_ls["side"] == "long", "weight"].to_numpy(dtype=float)
    short_weights = d1_pos_ls.loc[d1_pos_ls["side"] == "short", "weight"].to_numpy(dtype=float)
    assert np.allclose(long_weights, [0.5, 0.5], atol=1e-12)
    assert np.allclose(short_weights, [-0.5, -0.5], atol=1e-12)
    assert np.isclose(float(d1_pos_ls["contribution"].sum()), float(d1_ls["gross_portfolio_return"]), atol=1e-12)

    d1_lo = daily[(daily["date"] == d1) & (daily["portfolio_mode"] == "long_only_top")].iloc[0]
    assert np.isclose(float(d1_lo["gross_portfolio_return"]), 0.03, atol=1e-12)

    d2 = pd.Timestamp("2026-03-11")
    d2_ls = daily[(daily["date"] == d2) & (daily["portfolio_mode"] == "long_short_top_bottom")].iloc[0]
    assert bool(d2_ls["is_executable"]) is False
    assert str(d2_ls["skip_reason"]) == "missing_short"
    assert pd.isna(d2_ls["gross_portfolio_return"])

    d3 = pd.Timestamp("2026-03-12")
    d3_lo = daily[(daily["date"] == d3) & (daily["portfolio_mode"] == "long_only_top")].iloc[0]
    assert bool(d3_lo["is_executable"]) is False
    assert str(d3_lo["skip_reason"]) == "missing_long"
    assert pd.isna(d3_lo["gross_portfolio_return"])

    mode_names = {item["portfolio_mode"] for item in summary["mode_summaries"]}
    assert mode_names == {"long_only_top", "long_short_top_bottom"}


def test_paper_portfolio_fails_on_ambiguous_model_name_when_not_provided(
    tmp_workspace: dict[str, Path],
) -> None:
    signals_path, labels_path = _seed_paper_portfolio_case(tmp_workspace)
    signals = read_parquet(signals_path).copy()
    signals.loc[signals.index[0], "model_name"] = "other_model"
    signals.to_parquet(signals_path, index=False)

    with pytest.raises(ValueError, match="Expected exactly one model_name"):
        _ = run_paper_portfolio(
            signals_path=signals_path,
            labels_path=labels_path,
            output_dir=tmp_workspace["artifacts"] / "paper_portfolio_ambiguous",
            # model_name intentionally omitted
            label_name="fwd_ret_5d",
            horizon_days=5,
            split_roles=("test",),
            portfolio_modes=("long_only_top", "long_short_top_bottom"),
            run_id="test_paper_portfolio_ambiguous",
        )
