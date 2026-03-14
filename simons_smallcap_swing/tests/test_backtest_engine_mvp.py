from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import run_backtest_engine
from simons_core.io.parquet_store import read_parquet


def _seed_backtest_case(tmp_workspace: dict[str, Path]) -> tuple[Path, Path, Path, Path]:
    base = tmp_workspace["data"] / "backtest_engine_case"
    base.mkdir(parents=True, exist_ok=True)

    holdings_path = base / "portfolio_holdings.parquet"
    costs_path = base / "costs_daily.parquet"
    prices_path = base / "adjusted_prices.parquet"
    calendar_path = base / "trading_calendar.parquet"

    d1 = pd.Timestamp("2026-06-01")
    d2 = pd.Timestamp("2026-06-02")
    d3 = pd.Timestamp("2026-06-03")

    holdings_rows = []
    for date in (d1, d2, d3):
        holdings_rows.extend(
            [
                {
                    "date": date,
                    "instrument_id": "SIMA",
                    "ticker": "AAA",
                    "model_name": "ridge_baseline",
                    "label_name": "fwd_ret_5d",
                    "portfolio_mode": "long_only_top_n",
                    "target_weight": 0.5,
                },
                {
                    "date": date,
                    "instrument_id": "SIMB",
                    "ticker": "BBB",
                    "model_name": "ridge_baseline",
                    "label_name": "fwd_ret_5d",
                    "portfolio_mode": "long_only_top_n",
                    "target_weight": 0.5,
                },
            ]
        )
    pd.DataFrame(holdings_rows).to_parquet(holdings_path, index=False)

    costs_rows = [
        {
            "date": d1,
            "portfolio_mode": "long_only_top_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
            "total_cost": 0.001,
        },
        {
            "date": d2,
            "portfolio_mode": "long_only_top_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
            "total_cost": 0.002,
        },
        {
            "date": d3,
            "portfolio_mode": "long_only_top_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
            "total_cost": 0.003,
        },
    ]
    pd.DataFrame(costs_rows).to_parquet(costs_path, index=False)

    prices_rows = [
        {"date": d1, "instrument_id": "SIMA", "ticker": "AAA", "close_adj": 100.0},
        {"date": d2, "instrument_id": "SIMA", "ticker": "AAA", "close_adj": 110.0},
        {"date": d3, "instrument_id": "SIMA", "ticker": "AAA", "close_adj": 99.0},
        {"date": d1, "instrument_id": "SIMB", "ticker": "BBB", "close_adj": 200.0},
        {"date": d2, "instrument_id": "SIMB", "ticker": "BBB", "close_adj": 180.0},
        {"date": d3, "instrument_id": "SIMB", "ticker": "BBB", "close_adj": 198.0},
    ]
    pd.DataFrame(prices_rows).to_parquet(prices_path, index=False)

    calendar_rows = [
        {"date": d1, "is_session": True},
        {"date": d2, "is_session": True},
        {"date": d3, "is_session": True},
    ]
    pd.DataFrame(calendar_rows).to_parquet(calendar_path, index=False)

    return holdings_path, costs_path, prices_path, calendar_path


def test_backtest_engine_mvp_outputs_and_accounting(tmp_workspace: dict[str, Path]) -> None:
    holdings_path, costs_path, prices_path, calendar_path = _seed_backtest_case(tmp_workspace)
    output_dir = tmp_workspace["artifacts"] / "backtest_engine"

    result = run_backtest_engine(
        holdings_path=holdings_path,
        costs_daily_path=costs_path,
        adjusted_prices_path=prices_path,
        trading_calendar_path=calendar_path,
        output_dir=output_dir,
        model_name="ridge_baseline",
        label_name="fwd_ret_5d",
        portfolio_modes=("long_only_top_n",),
        run_id="test_backtest_engine_mvp",
    )

    assert result.backtest_daily_path.exists()
    assert result.backtest_contributions_path.exists()
    assert result.backtest_summary_path.exists()
    assert result.row_count_daily > 0
    assert result.row_count_contributions > 0

    daily = read_parquet(result.backtest_daily_path).sort_values("date").reset_index(drop=True)
    contributions = (
        read_parquet(result.backtest_contributions_path)
        .sort_values(["date", "instrument_id"])
        .reset_index(drop=True)
    )
    summary = json.loads(result.backtest_summary_path.read_text(encoding="utf-8"))

    assert set(["gross_return", "total_cost", "net_return", "gross_equity", "net_equity", "drawdown_net"]).issubset(
        set(daily.columns)
    )
    assert set(["target_weight", "realized_return", "contribution"]).issubset(set(contributions.columns))
    assert len(daily) == 2  # d3 has no t+1 return and must be excluded

    # d1: 0.5*0.10 + 0.5*(-0.10) = 0.0
    # d2: 0.5*(-0.10) + 0.5*(+0.10) = 0.0
    assert np.allclose(daily["gross_return"].to_numpy(dtype=float), [0.0, 0.0], atol=1e-12)
    assert np.allclose(daily["total_cost"].to_numpy(dtype=float), [0.001, 0.002], atol=1e-12)
    assert np.allclose(daily["net_return"].to_numpy(dtype=float), [-0.001, -0.002], atol=1e-12)

    # Contribution aggregation -> gross_return
    contrib_agg = (
        contributions.groupby("date", as_index=False)["contribution"].sum()
        .sort_values("date")
        .reset_index(drop=True)
    )
    assert np.allclose(
        contrib_agg["contribution"].to_numpy(dtype=float),
        daily["gross_return"].to_numpy(dtype=float),
        atol=1e-12,
    )

    # Net accounting identity.
    assert np.allclose(
        (daily["gross_return"] - daily["total_cost"]).to_numpy(dtype=float),
        daily["net_return"].to_numpy(dtype=float),
        atol=1e-12,
    )

    # Equity and drawdown checks.
    expected_gross_equity = np.array([1.0, 1.0], dtype=float)
    expected_net_equity = np.array([0.999, 0.999 * 0.998], dtype=float)
    assert np.allclose(daily["gross_equity"].to_numpy(dtype=float), expected_gross_equity, atol=1e-12)
    assert np.allclose(daily["net_equity"].to_numpy(dtype=float), expected_net_equity, atol=1e-12)
    assert np.isclose(float(daily.loc[0, "drawdown_net"]), 0.0, atol=1e-12)
    assert float(daily.loc[1, "drawdown_net"]) < 0.0
    assert (daily["drawdown_net"].to_numpy(dtype=float) <= 1e-12).all()

    # Last date without t+1 return is handled explicitly and tracked.
    assert int(summary["rows_dropped_missing_tplus1_return"]) == 2
    assert summary["return_convention"] in {
        "holdings_t_apply_to_close_adj_t_to_tplus1",
        "execution_weight_at_execution_date_apply_to_close_adj_execution_date_to_next_session",
    }
