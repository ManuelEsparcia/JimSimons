from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from execution.cost_model import run_cost_model
from simons_core.io.parquet_store import read_parquet


def _seed_cost_model_case(tmp_workspace: dict[str, Path]) -> tuple[Path, Path]:
    base = tmp_workspace["data"] / "cost_model_case"
    base.mkdir(parents=True, exist_ok=True)
    holdings_path = base / "portfolio_holdings.parquet"
    rebalance_path = base / "portfolio_rebalance.parquet"

    d1 = pd.Timestamp("2026-05-04")
    d2 = pd.Timestamp("2026-05-05")

    holdings_rows = [
        {"date": d1, "instrument_id": "SIMA", "ticker": "AAA", "model_name": "ridge_baseline", "label_name": "fwd_ret_5d", "portfolio_mode": "long_short_top_bottom_n", "side": "long", "raw_score": 0.9, "rank_pct": 1.0, "target_weight": 0.5, "gross_exposure_side": 1.0, "net_exposure": 0.0},
        {"date": d1, "instrument_id": "SIMB", "ticker": "BBB", "model_name": "ridge_baseline", "label_name": "fwd_ret_5d", "portfolio_mode": "long_short_top_bottom_n", "side": "short", "raw_score": -0.8, "rank_pct": 0.0, "target_weight": -0.5, "gross_exposure_side": 1.0, "net_exposure": 0.0},
        {"date": d2, "instrument_id": "SIMA", "ticker": "AAA", "model_name": "ridge_baseline", "label_name": "fwd_ret_5d", "portfolio_mode": "long_short_top_bottom_n", "side": "long", "raw_score": 0.8, "rank_pct": 1.0, "target_weight": 0.3, "gross_exposure_side": 1.0, "net_exposure": 0.0},
        {"date": d2, "instrument_id": "SIMC", "ticker": "CCC", "model_name": "ridge_baseline", "label_name": "fwd_ret_5d", "portfolio_mode": "long_short_top_bottom_n", "side": "short", "raw_score": -0.7, "rank_pct": 0.0, "target_weight": -0.3, "gross_exposure_side": 1.0, "net_exposure": 0.0},
    ]
    pd.DataFrame(holdings_rows).to_parquet(holdings_path, index=False)

    rebalance_rows = [
        # Date 1 entries
        {"date": d1, "instrument_id": "SIMA", "ticker": "AAA", "prev_weight": 0.0, "target_weight": 0.5, "weight_change": 0.5, "abs_weight_change": 0.5, "entered_flag": True, "exited_flag": False, "turnover_contribution": 0.25, "portfolio_mode": "long_short_top_bottom_n", "model_name": "ridge_baseline", "label_name": "fwd_ret_5d"},
        {"date": d1, "instrument_id": "SIMB", "ticker": "BBB", "prev_weight": 0.0, "target_weight": -0.5, "weight_change": -0.5, "abs_weight_change": 0.5, "entered_flag": True, "exited_flag": False, "turnover_contribution": 0.25, "portfolio_mode": "long_short_top_bottom_n", "model_name": "ridge_baseline", "label_name": "fwd_ret_5d"},
        # Date 2 rebalance with one exit and one entry
        {"date": d2, "instrument_id": "SIMA", "ticker": "AAA", "prev_weight": 0.5, "target_weight": 0.3, "weight_change": -0.2, "abs_weight_change": 0.2, "entered_flag": False, "exited_flag": False, "turnover_contribution": 0.1, "portfolio_mode": "long_short_top_bottom_n", "model_name": "ridge_baseline", "label_name": "fwd_ret_5d"},
        {"date": d2, "instrument_id": "SIMB", "ticker": "BBB", "prev_weight": -0.5, "target_weight": 0.0, "weight_change": 0.5, "abs_weight_change": 0.5, "entered_flag": False, "exited_flag": True, "turnover_contribution": 0.25, "portfolio_mode": "long_short_top_bottom_n", "model_name": "ridge_baseline", "label_name": "fwd_ret_5d"},
        {"date": d2, "instrument_id": "SIMC", "ticker": "CCC", "prev_weight": 0.0, "target_weight": -0.3, "weight_change": -0.3, "abs_weight_change": 0.3, "entered_flag": True, "exited_flag": False, "turnover_contribution": 0.15, "portfolio_mode": "long_short_top_bottom_n", "model_name": "ridge_baseline", "label_name": "fwd_ret_5d"},
    ]
    pd.DataFrame(rebalance_rows).to_parquet(rebalance_path, index=False)
    return holdings_path, rebalance_path


def test_cost_model_mvp_generates_outputs_and_expected_costs(tmp_workspace: dict[str, Path]) -> None:
    holdings_path, rebalance_path = _seed_cost_model_case(tmp_workspace)
    output_dir = tmp_workspace["artifacts"] / "cost_model"

    result = run_cost_model(
        holdings_path=holdings_path,
        rebalance_path=rebalance_path,
        output_dir=output_dir,
        model_name="ridge_baseline",
        label_name="fwd_ret_5d",
        portfolio_modes=("long_short_top_bottom_n",),
        cost_bps_per_turnover=10.0,
        entry_bps=5.0,
        exit_bps=7.0,
        run_id="test_cost_model_mvp",
    )

    assert result.costs_positions_path.exists()
    assert result.costs_daily_path.exists()
    assert result.costs_summary_path.exists()
    assert result.row_count_positions > 0
    assert result.row_count_daily > 0

    costs_positions = read_parquet(result.costs_positions_path).sort_values(
        ["date", "instrument_id"]
    ).reset_index(drop=True)
    costs_daily = read_parquet(result.costs_daily_path).sort_values(["date"]).reset_index(drop=True)
    summary = json.loads(result.costs_summary_path.read_text(encoding="utf-8"))

    required_positions = {
        "turnover_cost",
        "entry_cost",
        "exit_cost",
        "total_cost",
        "turnover_contribution",
    }
    required_daily = {
        "gross_turnover",
        "total_turnover_cost",
        "total_entry_cost",
        "total_exit_cost",
        "total_cost",
    }
    assert required_positions.issubset(set(costs_positions.columns))
    assert required_daily.issubset(set(costs_daily.columns))
    assert (costs_positions["total_cost"] >= 0.0).all()

    # Spot check per-position formulas.
    d1_a = costs_positions[
        (costs_positions["date"] == pd.Timestamp("2026-05-04"))
        & (costs_positions["instrument_id"] == "SIMA")
    ].iloc[0]
    assert np.isclose(float(d1_a["turnover_cost"]), 0.0005, atol=1e-12)
    assert np.isclose(float(d1_a["entry_cost"]), 0.00025, atol=1e-12)
    assert np.isclose(float(d1_a["exit_cost"]), 0.0, atol=1e-12)
    assert np.isclose(float(d1_a["total_cost"]), 0.00075, atol=1e-12)

    d2_b = costs_positions[
        (costs_positions["date"] == pd.Timestamp("2026-05-05"))
        & (costs_positions["instrument_id"] == "SIMB")
    ].iloc[0]
    assert np.isclose(float(d2_b["turnover_cost"]), 0.0005, atol=1e-12)
    assert np.isclose(float(d2_b["entry_cost"]), 0.0, atol=1e-12)
    assert np.isclose(float(d2_b["exit_cost"]), 0.00035, atol=1e-12)
    assert np.isclose(float(d2_b["total_cost"]), 0.00085, atol=1e-12)

    # Daily aggregation reconciliation.
    positions_daily = (
        costs_positions.groupby("date", as_index=False)
        .agg(
            total_cost=("total_cost", "sum"),
            total_turnover_cost=("turnover_cost", "sum"),
            total_entry_cost=("entry_cost", "sum"),
            total_exit_cost=("exit_cost", "sum"),
            gross_turnover=("abs_weight_change", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    merged = costs_daily.merge(positions_daily, on="date", suffixes=("_daily", "_positions"))
    assert np.allclose(
        merged["total_cost_daily"].to_numpy(dtype=float),
        merged["total_cost_positions"].to_numpy(dtype=float),
        atol=1e-12,
    )
    assert np.allclose(
        merged["gross_turnover_daily"].to_numpy(dtype=float),
        merged["gross_turnover_positions"].to_numpy(dtype=float),
        atol=1e-12,
    )

    assert summary["model_name"] == "ridge_baseline"
    assert summary["label_name"] == "fwd_ret_5d"
    assert summary["cost_bps_per_turnover"] == 10.0
    assert summary["entry_bps"] == 5.0
    assert summary["exit_bps"] == 7.0
    assert summary["n_dates"] == 2
