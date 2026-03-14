from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio.construct_portfolio import run_construct_portfolio
from simons_core.io.parquet_store import read_parquet


def _seed_construct_portfolio_case(tmp_workspace: dict[str, Path]) -> tuple[Path, Path]:
    base = tmp_workspace["data"] / "construct_portfolio_case"
    base.mkdir(parents=True, exist_ok=True)
    signals_path = base / "signals_daily.parquet"
    universe_path = base / "universe_history.parquet"

    d1 = pd.Timestamp("2026-04-06")
    d2 = pd.Timestamp("2026-04-07")

    signals_rows = [
        # Date 1
        {"date": d1, "instrument_id": "SIMA", "ticker": "AAA", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "raw_score": 0.90, "rank_pct": 1.0},
        {"date": d1, "instrument_id": "SIMB", "ticker": "BBB", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "raw_score": 0.70, "rank_pct": 0.8},
        {"date": d1, "instrument_id": "SIMC", "ticker": "CCC", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "raw_score": 0.20, "rank_pct": 0.6},
        {"date": d1, "instrument_id": "SIMD", "ticker": "DDD", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "raw_score": -0.10, "rank_pct": 0.4},
        {"date": d1, "instrument_id": "SIME", "ticker": "EEE", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "raw_score": -0.30, "rank_pct": 0.2},
        {"date": d1, "instrument_id": "SIMX", "ticker": "XXX", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "raw_score": 1.50, "rank_pct": 1.0},
        # Date 2
        {"date": d2, "instrument_id": "SIMA", "ticker": "AAA", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "raw_score": 0.80, "rank_pct": 1.0},
        {"date": d2, "instrument_id": "SIMC", "ticker": "CCC", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "raw_score": 0.60, "rank_pct": 0.8},
        {"date": d2, "instrument_id": "SIMB", "ticker": "BBB", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "raw_score": 0.10, "rank_pct": 0.6},
        {"date": d2, "instrument_id": "SIME", "ticker": "EEE", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "raw_score": -0.20, "rank_pct": 0.4},
        {"date": d2, "instrument_id": "SIMD", "ticker": "DDD", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "raw_score": -0.50, "rank_pct": 0.2},
        {"date": d2, "instrument_id": "SIMX", "ticker": "XXX", "split_role": "test", "horizon_days": 5, "label_name": "fwd_ret_5d", "model_name": "ridge_baseline", "raw_score": 1.60, "rank_pct": 1.0},
    ]
    pd.DataFrame(signals_rows).to_parquet(signals_path, index=False)

    universe_rows = []
    for date in (d1, d2):
        for instrument_id in ("SIMA", "SIMB", "SIMC", "SIMD", "SIME"):
            universe_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "is_eligible": True,
                }
            )
        universe_rows.append(
            {
                "date": date,
                "instrument_id": "SIMX",
                "is_eligible": False,
            }
        )
    pd.DataFrame(universe_rows).to_parquet(universe_path, index=False)
    return signals_path, universe_path


def test_construct_portfolio_mvp_generates_holdings_rebalance_and_summary(
    tmp_workspace: dict[str, Path],
) -> None:
    signals_path, universe_path = _seed_construct_portfolio_case(tmp_workspace)
    output_dir = tmp_workspace["artifacts"] / "portfolio_mvp"

    result = run_construct_portfolio(
        signals_path=signals_path,
        universe_history_path=universe_path,
        output_dir=output_dir,
        model_name="ridge_baseline",
        label_name="fwd_ret_5d",
        horizon_days=5,
        split_roles=("test",),
        portfolio_modes=("long_only_top_n", "long_short_top_bottom_n"),
        top_n=2,
        bottom_n=2,
        run_id="test_construct_portfolio_mvp",
    )

    assert result.holdings_path.exists()
    assert result.rebalance_path.exists()
    assert result.summary_path.exists()
    assert result.row_count_holdings > 0

    holdings = read_parquet(result.holdings_path)
    rebalance = read_parquet(result.rebalance_path)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert "SIMX" not in set(holdings["instrument_id"].astype(str).tolist())

    expected_holdings_cols = {
        "date",
        "instrument_id",
        "ticker",
        "model_name",
        "label_name",
        "portfolio_mode",
        "side",
        "raw_score",
        "rank_pct",
        "target_weight",
        "gross_exposure_side",
        "net_exposure",
    }
    expected_rebalance_cols = {
        "date",
        "instrument_id",
        "ticker",
        "prev_weight",
        "target_weight",
        "weight_change",
        "abs_weight_change",
        "entered_flag",
        "exited_flag",
        "turnover_contribution",
        "portfolio_mode",
        "model_name",
        "label_name",
    }
    assert expected_holdings_cols.issubset(set(holdings.columns))
    assert expected_rebalance_cols.issubset(set(rebalance.columns))

    assert not holdings.duplicated(
        ["date", "instrument_id", "portfolio_mode", "model_name", "label_name"], keep=False
    ).any()

    d1 = pd.Timestamp("2026-04-06")
    d2 = pd.Timestamp("2026-04-07")

    d1_long_only = holdings[
        (holdings["date"] == d1) & (holdings["portfolio_mode"] == "long_only_top_n")
    ].copy()
    assert set(d1_long_only["instrument_id"].tolist()) == {"SIMA", "SIMB"}
    assert np.allclose(d1_long_only["target_weight"].to_numpy(dtype=float), [0.5, 0.5], atol=1e-12)

    d2_long_only = holdings[
        (holdings["date"] == d2) & (holdings["portfolio_mode"] == "long_only_top_n")
    ].copy()
    assert set(d2_long_only["instrument_id"].tolist()) == {"SIMA", "SIMC"}

    d1_ls = holdings[
        (holdings["date"] == d1) & (holdings["portfolio_mode"] == "long_short_top_bottom_n")
    ].copy()
    assert set(d1_ls.loc[d1_ls["side"] == "long", "instrument_id"].tolist()) == {"SIMA", "SIMB"}
    assert set(d1_ls.loc[d1_ls["side"] == "short", "instrument_id"].tolist()) == {"SIMD", "SIME"}
    assert np.isclose(float(d1_ls.loc[d1_ls["side"] == "long", "target_weight"].sum()), 1.0, atol=1e-12)
    assert np.isclose(float(d1_ls.loc[d1_ls["side"] == "short", "target_weight"].sum()), -1.0, atol=1e-12)

    # Entries/exits + turnover in long_only mode when B is replaced by C on date 2.
    d2_long_only_reb = rebalance[
        (rebalance["date"] == d2) & (rebalance["portfolio_mode"] == "long_only_top_n")
    ].copy()
    b_row = d2_long_only_reb[d2_long_only_reb["instrument_id"] == "SIMB"].iloc[0]
    c_row = d2_long_only_reb[d2_long_only_reb["instrument_id"] == "SIMC"].iloc[0]
    assert bool(b_row["exited_flag"]) is True
    assert bool(c_row["entered_flag"]) is True
    assert np.isclose(float(b_row["turnover_contribution"]), 0.25, atol=1e-12)
    assert np.isclose(float(c_row["turnover_contribution"]), 0.25, atol=1e-12)

    turnover_d2_long_only = float(d2_long_only_reb["turnover_contribution"].sum())
    assert np.isclose(turnover_d2_long_only, 0.5, atol=1e-12)

    # Turnover consistency: 0.5 * sum(abs_weight_change) by date/mode.
    grouped = rebalance.groupby(["date", "portfolio_mode"], as_index=False).agg(
        turnover=("turnover_contribution", "sum"),
        abs_sum=("abs_weight_change", "sum"),
    )
    assert np.allclose(
        grouped["turnover"].to_numpy(dtype=float),
        0.5 * grouped["abs_sum"].to_numpy(dtype=float),
        atol=1e-12,
    )

    mode_names = {item["portfolio_mode"] for item in summary["mode_summaries"]}
    assert mode_names == {"long_only_top_n", "long_short_top_bottom_n"}
