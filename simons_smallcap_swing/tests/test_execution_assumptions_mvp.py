from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from execution.assumptions import run_execution_assumptions
from simons_core.io.parquet_store import read_parquet


def _seed_execution_assumptions_case(tmp_workspace: dict[str, Path]) -> tuple[Path, Path, Path, Path]:
    base = tmp_workspace["data"] / "execution_assumptions_case"
    base.mkdir(parents=True, exist_ok=True)

    holdings_path = base / "portfolio_holdings.parquet"
    rebalance_path = base / "portfolio_rebalance.parquet"
    calendar_path = base / "trading_calendar.parquet"
    prices_path = base / "adjusted_prices.parquet"

    d1 = pd.Timestamp("2026-06-05")  # Friday
    d2 = pd.Timestamp("2026-06-08")  # Monday
    d3 = pd.Timestamp("2026-06-09")  # Tuesday

    holdings_rows = [
        {
            "date": d1,
            "instrument_id": "SIMA",
            "ticker": "AAA",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
            "portfolio_mode": "long_only_top_n",
            "target_weight": 1.0,
        },
        {
            "date": d2,
            "instrument_id": "SIMA",
            "ticker": "AAA",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
            "portfolio_mode": "long_only_top_n",
            "target_weight": 0.5,
        },
        {
            "date": d3,
            "instrument_id": "SIMA",
            "ticker": "AAA",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
            "portfolio_mode": "long_only_top_n",
            "target_weight": 0.2,
        },
    ]
    pd.DataFrame(holdings_rows).to_parquet(holdings_path, index=False)

    rebalance_rows = [
        {
            "date": d1,
            "instrument_id": "SIMA",
            "ticker": "AAA",
            "prev_weight": 0.0,
            "target_weight": 1.0,
            "weight_change": 1.0,
            "abs_weight_change": 1.0,
            "entered_flag": True,
            "exited_flag": False,
            "portfolio_mode": "long_only_top_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
        },
        {
            "date": d2,
            "instrument_id": "SIMA",
            "ticker": "AAA",
            "prev_weight": 1.0,
            "target_weight": 0.5,
            "weight_change": -0.5,
            "abs_weight_change": 0.5,
            "entered_flag": False,
            "exited_flag": False,
            "portfolio_mode": "long_only_top_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
        },
        {
            "date": d3,
            "instrument_id": "SIMA",
            "ticker": "AAA",
            "prev_weight": 0.5,
            "target_weight": 0.2,
            "weight_change": -0.3,
            "abs_weight_change": 0.3,
            "entered_flag": False,
            "exited_flag": False,
            "portfolio_mode": "long_only_top_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
        },
    ]
    pd.DataFrame(rebalance_rows).to_parquet(rebalance_path, index=False)

    # Calendar intentionally skips weekend to enforce session-based delay.
    calendar_rows = [
        {"date": pd.Timestamp("2026-06-05"), "is_session": True},
        {"date": pd.Timestamp("2026-06-06"), "is_session": False},
        {"date": pd.Timestamp("2026-06-07"), "is_session": False},
        {"date": pd.Timestamp("2026-06-08"), "is_session": True},
        {"date": pd.Timestamp("2026-06-09"), "is_session": True},
    ]
    pd.DataFrame(calendar_rows).to_parquet(calendar_path, index=False)

    prices_rows = [
        {"date": d1, "instrument_id": "SIMA", "close_adj": 100.0},
        {"date": d2, "instrument_id": "SIMA", "close_adj": 101.0},
        # Missing d3 on purpose for skip_if_missing_next_session test:
        # execution_date for signal d2 is d3 -> not executable under that mode.
    ]
    pd.DataFrame(prices_rows).to_parquet(prices_path, index=False)

    return holdings_path, rebalance_path, calendar_path, prices_path


def test_execution_assumptions_full_fill_delay_and_skips(tmp_workspace: dict[str, Path]) -> None:
    holdings_path, rebalance_path, calendar_path, _ = _seed_execution_assumptions_case(tmp_workspace)
    output_dir = tmp_workspace["artifacts"] / "execution_assumptions_full_fill"

    result = run_execution_assumptions(
        holdings_path=holdings_path,
        rebalance_path=rebalance_path,
        trading_calendar_path=calendar_path,
        output_dir=output_dir,
        model_name="ridge_baseline",
        label_name="fwd_ret_5d",
        portfolio_modes=("long_only_top_n",),
        execution_delay_sessions=1,
        fill_assumption="full_fill",
        cost_timing="apply_on_execution_date",
        run_id="test_execution_assumptions_full_fill",
    )

    assert result.execution_holdings_path.exists()
    assert result.execution_rebalance_path.exists()
    assert result.execution_assumptions_summary_path.exists()

    execution_holdings = read_parquet(result.execution_holdings_path).sort_values("signal_date").reset_index(drop=True)
    execution_rebalance = read_parquet(result.execution_rebalance_path).sort_values("signal_date").reset_index(drop=True)
    summary = json.loads(result.execution_assumptions_summary_path.read_text(encoding="utf-8"))

    required_holdings = {
        "signal_date",
        "execution_date",
        "execution_weight",
        "is_executable",
        "skip_reason",
        "execution_delay_sessions",
        "fill_assumption",
    }
    required_rebalance = {
        "signal_date",
        "execution_date",
        "cost_date",
        "execution_weight",
        "weight_change_signal",
        "weight_change_execution",
        "is_executable",
        "skip_reason",
    }
    assert required_holdings.issubset(set(execution_holdings.columns))
    assert required_rebalance.issubset(set(execution_rebalance.columns))

    # Friday signal must execute Monday (calendar sessions, not civil day +1).
    friday_row = execution_holdings[execution_holdings["signal_date"] == pd.Timestamp("2026-06-05")].iloc[0]
    assert pd.Timestamp(friday_row["execution_date"]).normalize() == pd.Timestamp("2026-06-08")
    assert bool(friday_row["is_executable"]) is True
    assert np.isclose(float(friday_row["execution_weight"]), 1.0, atol=1e-12)

    # Last signal has no next session available and must be marked as skipped.
    last_row = execution_holdings[execution_holdings["signal_date"] == pd.Timestamp("2026-06-09")].iloc[0]
    assert bool(last_row["is_executable"]) is False
    assert str(last_row["skip_reason"]) == "no_execution_session"
    assert np.isclose(float(last_row["execution_weight"]), 0.0, atol=1e-12)

    # Rebalance row skipped -> execution_weight equals prev_weight and no execution weight change.
    last_reb = execution_rebalance[execution_rebalance["signal_date"] == pd.Timestamp("2026-06-09")].iloc[0]
    assert bool(last_reb["is_executable"]) is False
    assert np.isclose(float(last_reb["execution_weight"]), float(last_reb["prev_weight"]), atol=1e-12)
    assert np.isclose(float(last_reb["weight_change_execution"]), 0.0, atol=1e-12)

    assert summary["execution_delay_sessions"] == 1
    assert summary["fill_assumption"] == "full_fill"
    assert summary["cost_timing"] == "apply_on_execution_date"
    assert summary["n_signal_rows"] == 3
    assert summary["n_skipped_rows"] == 1
    assert "no_execution_session" in summary["skip_reasons"]


def test_execution_assumptions_skip_if_missing_price(tmp_workspace: dict[str, Path]) -> None:
    holdings_path, rebalance_path, calendar_path, prices_path = _seed_execution_assumptions_case(tmp_workspace)
    output_dir = tmp_workspace["artifacts"] / "execution_assumptions_skip_missing"

    result = run_execution_assumptions(
        holdings_path=holdings_path,
        rebalance_path=rebalance_path,
        trading_calendar_path=calendar_path,
        adjusted_prices_path=prices_path,
        output_dir=output_dir,
        model_name="ridge_baseline",
        label_name="fwd_ret_5d",
        portfolio_modes=("long_only_top_n",),
        execution_delay_sessions=1,
        fill_assumption="skip_if_missing_next_session",
        cost_timing="apply_on_signal_date",
        run_id="test_execution_assumptions_skip_missing",
    )

    execution_holdings = read_parquet(result.execution_holdings_path).sort_values("signal_date").reset_index(drop=True)

    monday_signal = execution_holdings[execution_holdings["signal_date"] == pd.Timestamp("2026-06-08")].iloc[0]
    # signal_date 2026-06-08 -> execution_date 2026-06-09, but price on 2026-06-09 is missing
    assert bool(monday_signal["is_executable"]) is False
    assert str(monday_signal["skip_reason"]) == "missing_price_on_execution_date"

