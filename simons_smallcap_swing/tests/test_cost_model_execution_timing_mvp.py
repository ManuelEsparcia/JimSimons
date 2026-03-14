from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from execution.cost_model import run_cost_model
from simons_core.io.parquet_store import read_parquet


def _seed_execution_rebalance_case(tmp_workspace: dict[str, Path]) -> Path:
    base = tmp_workspace["data"] / "cost_model_exec_timing_case"
    base.mkdir(parents=True, exist_ok=True)
    rebalance_path = base / "execution_rebalance.parquet"

    d1 = pd.Timestamp("2026-07-01")
    d2 = pd.Timestamp("2026-07-02")
    d3 = pd.Timestamp("2026-07-03")
    d4 = pd.Timestamp("2026-07-06")

    rows = [
        {
            "signal_date": d1,
            "execution_date": d2,
            "cost_date": d2,
            "instrument_id": "SIMA",
            "ticker": "AAA",
            "prev_weight": 0.0,
            "target_weight": 1.0,
            "execution_weight": 1.0,
            "weight_change_signal": 1.0,
            "weight_change_execution": 1.0,
            "entered_flag": True,
            "exited_flag": False,
            "is_executable": True,
            "skip_reason": None,
            "portfolio_mode": "long_only_top_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
        },
        {
            "signal_date": d2,
            "execution_date": d3,
            "cost_date": d3,
            "instrument_id": "SIMA",
            "ticker": "AAA",
            "prev_weight": 1.0,
            "target_weight": 0.5,
            "execution_weight": 0.5,
            "weight_change_signal": -0.5,
            "weight_change_execution": -0.5,
            "entered_flag": False,
            "exited_flag": False,
            "is_executable": True,
            "skip_reason": None,
            "portfolio_mode": "long_only_top_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
        },
        {
            "signal_date": d3,
            "execution_date": d4,
            "cost_date": pd.NaT,
            "instrument_id": "SIMA",
            "ticker": "AAA",
            "prev_weight": 0.5,
            "target_weight": 0.2,
            "execution_weight": 0.5,
            "weight_change_signal": -0.3,
            "weight_change_execution": 0.0,
            "entered_flag": False,
            "exited_flag": False,
            "is_executable": False,
            "skip_reason": "no_execution_session",
            "portfolio_mode": "long_only_top_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
        },
    ]
    pd.DataFrame(rows).to_parquet(rebalance_path, index=False)
    return rebalance_path


def test_cost_model_uses_cost_date_and_execution_fields(tmp_workspace: dict[str, Path]) -> None:
    rebalance_path = _seed_execution_rebalance_case(tmp_workspace)
    output_dir = tmp_workspace["artifacts"] / "cost_model_exec_timing"

    result = run_cost_model(
        rebalance_path=rebalance_path,
        output_dir=output_dir,
        model_name="ridge_baseline",
        label_name="fwd_ret_5d",
        portfolio_modes=("long_only_top_n",),
        cost_bps_per_turnover=10.0,
        entry_bps=5.0,
        exit_bps=7.0,
        run_id="test_cost_model_execution_timing_mvp",
    )

    assert result.costs_positions_path.exists()
    assert result.costs_daily_path.exists()
    assert result.costs_summary_path.exists()

    positions = read_parquet(result.costs_positions_path).sort_values("signal_date").reset_index(drop=True)
    daily = read_parquet(result.costs_daily_path).sort_values("cost_date").reset_index(drop=True)
    summary = json.loads(result.costs_summary_path.read_text(encoding="utf-8"))

    required_positions = {
        "signal_date",
        "execution_date",
        "cost_date",
        "execution_weight",
        "weight_change_execution",
        "is_executable",
        "total_cost",
    }
    assert required_positions.issubset(set(positions.columns))
    assert {"cost_date", "total_cost", "gross_turnover"}.issubset(set(daily.columns))

    # Non executable row should not create effective cost.
    non_exec = positions[positions["is_executable"] == False].iloc[0]  # noqa: E712
    assert np.isclose(float(non_exec["total_cost"]), 0.0, atol=1e-12)
    assert pd.isna(non_exec["cost_date"])

    # Daily costs are keyed by cost_date (d2 and d3 only).
    assert list(pd.to_datetime(daily["cost_date"]).dt.normalize()) == [
        pd.Timestamp("2026-07-02"),
        pd.Timestamp("2026-07-03"),
    ]

    # Row1: abs change 1.0 => turnover_cost 0.001 + entry 0.0005
    d2 = daily[daily["cost_date"] == pd.Timestamp("2026-07-02")].iloc[0]
    assert np.isclose(float(d2["total_turnover_cost"]), 0.001, atol=1e-12)
    assert np.isclose(float(d2["total_entry_cost"]), 0.0005, atol=1e-12)
    assert np.isclose(float(d2["total_exit_cost"]), 0.0, atol=1e-12)
    assert np.isclose(float(d2["total_cost"]), 0.0015, atol=1e-12)

    # Row2: abs change 0.5 => turnover_cost 0.0005 and no entry/exit
    d3 = daily[daily["cost_date"] == pd.Timestamp("2026-07-03")].iloc[0]
    assert np.isclose(float(d3["total_turnover_cost"]), 0.0005, atol=1e-12)
    assert np.isclose(float(d3["total_entry_cost"]), 0.0, atol=1e-12)
    assert np.isclose(float(d3["total_exit_cost"]), 0.0, atol=1e-12)
    assert np.isclose(float(d3["total_cost"]), 0.0005, atol=1e-12)

    assert summary["cost_aggregation_date"] == "cost_date"
    assert summary["rebalance_input_mode"] == "execution_rebalance"
    assert summary["n_non_executable_rows"] == 1

