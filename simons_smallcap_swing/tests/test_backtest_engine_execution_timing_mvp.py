from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import run_backtest_engine
from simons_core.io.parquet_store import read_parquet


def _seed_backtest_execution_case(tmp_workspace: dict[str, Path]) -> tuple[Path, Path, Path, Path]:
    base = tmp_workspace["data"] / "backtest_exec_timing_case"
    base.mkdir(parents=True, exist_ok=True)

    holdings_path = base / "execution_holdings.parquet"
    costs_path = base / "costs_daily.parquet"
    prices_path = base / "adjusted_prices.parquet"
    calendar_path = base / "trading_calendar.parquet"

    d1 = pd.Timestamp("2026-08-03")
    d2 = pd.Timestamp("2026-08-04")
    d3 = pd.Timestamp("2026-08-05")
    d4 = pd.Timestamp("2026-08-06")

    holdings_rows = [
        # Executable and captures d2 -> d3
        {
            "signal_date": d1,
            "execution_date": d2,
            "instrument_id": "SIMA",
            "ticker": "AAA",
            "portfolio_mode": "long_only_top_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
            "target_weight": 1.0,
            "execution_weight": 1.0,
            "fill_assumption": "full_fill",
            "execution_delay_sessions": 1,
            "is_executable": True,
            "skip_reason": None,
        },
        # Executable and captures d3 -> d4
        {
            "signal_date": d2,
            "execution_date": d3,
            "instrument_id": "SIMA",
            "ticker": "AAA",
            "portfolio_mode": "long_only_top_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
            "target_weight": 1.0,
            "execution_weight": 1.0,
            "fill_assumption": "full_fill",
            "execution_delay_sessions": 1,
            "is_executable": True,
            "skip_reason": None,
        },
        # Executable but no t+1 return available (d4 -> missing), should drop.
        {
            "signal_date": d3,
            "execution_date": d4,
            "instrument_id": "SIMA",
            "ticker": "AAA",
            "portfolio_mode": "long_only_top_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
            "target_weight": 1.0,
            "execution_weight": 1.0,
            "fill_assumption": "full_fill",
            "execution_delay_sessions": 1,
            "is_executable": True,
            "skip_reason": None,
        },
        # Non executable should not generate exposure.
        {
            "signal_date": d1,
            "execution_date": d2,
            "instrument_id": "SIMB",
            "ticker": "BBB",
            "portfolio_mode": "long_only_top_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
            "target_weight": 1.0,
            "execution_weight": 0.0,
            "fill_assumption": "full_fill",
            "execution_delay_sessions": 1,
            "is_executable": False,
            "skip_reason": "missing_price_on_execution_date",
        },
    ]
    pd.DataFrame(holdings_rows).to_parquet(holdings_path, index=False)

    # Keep mismatched legacy `date` to ensure engine uses explicit `cost_date`.
    costs_rows = [
        {
            "cost_date": d2,
            "date": d1,
            "portfolio_mode": "long_only_top_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
            "total_cost": 0.01,
        },
        {
            "cost_date": d3,
            "date": d2,
            "portfolio_mode": "long_only_top_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
            "total_cost": 0.02,
        },
    ]
    pd.DataFrame(costs_rows).to_parquet(costs_path, index=False)

    prices_rows = [
        {"date": d2, "instrument_id": "SIMA", "ticker": "AAA", "close_adj": 100.0},
        {"date": d3, "instrument_id": "SIMA", "ticker": "AAA", "close_adj": 110.0},
        {"date": d4, "instrument_id": "SIMA", "ticker": "AAA", "close_adj": 99.0},
    ]
    pd.DataFrame(prices_rows).to_parquet(prices_path, index=False)

    calendar_rows = [
        {"date": d1, "is_session": True},
        {"date": d2, "is_session": True},
        {"date": d3, "is_session": True},
        {"date": d4, "is_session": True},
    ]
    pd.DataFrame(calendar_rows).to_parquet(calendar_path, index=False)

    return holdings_path, costs_path, prices_path, calendar_path


def test_backtest_engine_uses_execution_date_and_cost_date(tmp_workspace: dict[str, Path]) -> None:
    holdings_path, costs_path, prices_path, calendar_path = _seed_backtest_execution_case(tmp_workspace)
    output_dir = tmp_workspace["artifacts"] / "backtest_exec_timing"

    result = run_backtest_engine(
        holdings_path=holdings_path,
        costs_daily_path=costs_path,
        adjusted_prices_path=prices_path,
        trading_calendar_path=calendar_path,
        output_dir=output_dir,
        model_name="ridge_baseline",
        label_name="fwd_ret_5d",
        portfolio_modes=("long_only_top_n",),
        run_id="test_backtest_engine_execution_timing_mvp",
    )

    assert result.backtest_daily_path.exists()
    assert result.backtest_contributions_path.exists()
    assert result.backtest_summary_path.exists()

    daily = read_parquet(result.backtest_daily_path).sort_values("return_start_date").reset_index(drop=True)
    contrib = read_parquet(result.backtest_contributions_path).sort_values("return_start_date").reset_index(drop=True)
    summary = json.loads(result.backtest_summary_path.read_text(encoding="utf-8"))

    assert {"return_start_date", "return_end_date", "gross_return", "net_return"}.issubset(set(daily.columns))
    assert {"signal_date", "execution_date", "execution_weight", "realized_return"}.issubset(set(contrib.columns))

    # Should keep only d2 and d3 starts (d4 has no t+1 return).
    assert list(pd.to_datetime(daily["return_start_date"]).dt.normalize()) == [pd.Timestamp("2026-08-04"), pd.Timestamp("2026-08-05")]
    assert len(contrib) == 2

    # Realized returns: d2->d3 = +10%, d3->d4 = -10%.
    assert np.allclose(daily["gross_return"].to_numpy(dtype=float), [0.1, -0.1], atol=1e-12)

    # Costs by cost_date: 0.01 on d2, 0.02 on d3.
    assert np.allclose(daily["total_cost"].to_numpy(dtype=float), [0.01, 0.02], atol=1e-12)
    assert np.allclose(daily["net_return"].to_numpy(dtype=float), [0.09, -0.12], atol=1e-12)

    # Accounting identities.
    assert np.allclose(
        (daily["gross_return"] - daily["total_cost"]).to_numpy(dtype=float),
        daily["net_return"].to_numpy(dtype=float),
        atol=1e-12,
    )
    contrib_daily = contrib.groupby("date", as_index=False)["contribution"].sum().sort_values("date").reset_index(drop=True)
    assert np.allclose(
        contrib_daily["contribution"].to_numpy(dtype=float),
        daily["gross_return"].to_numpy(dtype=float),
        atol=1e-12,
    )

    # Explicit timing diagnostics.
    assert summary["rows_dropped_non_executable"] == 1
    assert summary["rows_dropped_missing_tplus1_return"] == 1
    assert summary["holdings_input_mode"] == "execution_holdings"

