from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.diagnostics import run_backtest_diagnostics
from simons_core.io.parquet_store import read_parquet


def _seed_backtest_diagnostics_case(tmp_workspace: dict[str, Path]) -> tuple[Path, Path, Path, Path]:
    base = tmp_workspace["data"] / "backtest_diagnostics_case"
    base.mkdir(parents=True, exist_ok=True)

    backtest_daily_path = base / "backtest_daily.parquet"
    backtest_summary_path = base / "backtest_summary.json"
    costs_daily_path = base / "costs_daily.parquet"
    execution_summary_path = base / "execution_assumptions_summary.json"

    d1 = pd.Timestamp("2026-10-05")
    d2 = pd.Timestamp("2026-10-06")

    backtest_daily_rows = [
        {
            "date": d1,
            "return_start_date": d1,
            "return_end_date": d2,
            "portfolio_mode": "long_only_top_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
            "gross_return": 0.05,
            "total_cost": 0.01,
            "net_return": 0.04,
            "gross_equity": 1.05,
            "net_equity": 1.04,
            "drawdown_net": 0.0,
            "n_positions": 10,
        },
        {
            "date": d2,
            "return_start_date": d2,
            "return_end_date": d2 + pd.Timedelta(days=1),
            "portfolio_mode": "long_only_top_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
            "gross_return": 0.00,
            "total_cost": 0.01,
            "net_return": -0.01,
            "gross_equity": 1.05,
            "net_equity": 1.0296,
            "drawdown_net": -0.01,
            "n_positions": 11,
        },
        {
            "date": d1,
            "return_start_date": d1,
            "return_end_date": d2,
            "portfolio_mode": "long_short_top_bottom_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
            "gross_return": 0.03,
            "total_cost": 0.02,
            "net_return": 0.01,
            "gross_equity": 1.03,
            "net_equity": 1.01,
            "drawdown_net": 0.0,
            "n_positions": 20,
        },
        {
            "date": d2,
            "return_start_date": d2,
            "return_end_date": d2 + pd.Timedelta(days=1),
            "portfolio_mode": "long_short_top_bottom_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
            "gross_return": 0.04,
            "total_cost": 0.02,
            "net_return": 0.02,
            "gross_equity": 1.0712,
            "net_equity": 1.0302,
            "drawdown_net": 0.0,
            "n_positions": 19,
        },
    ]
    pd.DataFrame(backtest_daily_rows).to_parquet(backtest_daily_path, index=False)

    costs_daily_rows = [
        {
            "cost_date": d1,
            "date": d1,
            "portfolio_mode": "long_only_top_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
            "n_positions": 10,
            "gross_turnover": 0.50,
            "total_turnover_cost": 0.007,
            "total_entry_cost": 0.002,
            "total_exit_cost": 0.001,
            "total_cost": 0.01,
        },
        {
            "cost_date": d2,
            "date": d2,
            "portfolio_mode": "long_only_top_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
            "n_positions": 11,
            "gross_turnover": 0.55,
            "total_turnover_cost": 0.006,
            "total_entry_cost": 0.002,
            "total_exit_cost": 0.002,
            "total_cost": 0.01,
        },
        {
            "cost_date": d1,
            "date": d1,
            "portfolio_mode": "long_short_top_bottom_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
            "n_positions": 20,
            "gross_turnover": 0.80,
            "total_turnover_cost": 0.012,
            "total_entry_cost": 0.004,
            "total_exit_cost": 0.004,
            "total_cost": 0.02,
        },
        {
            "cost_date": d2,
            "date": d2,
            "portfolio_mode": "long_short_top_bottom_n",
            "model_name": "ridge_baseline",
            "label_name": "fwd_ret_5d",
            "n_positions": 19,
            "gross_turnover": 0.75,
            "total_turnover_cost": 0.011,
            "total_entry_cost": 0.005,
            "total_exit_cost": 0.004,
            "total_cost": 0.02,
        },
    ]
    pd.DataFrame(costs_daily_rows).to_parquet(costs_daily_path, index=False)

    backtest_summary = {
        "model_name": "ridge_baseline",
        "label_name": "fwd_ret_5d",
        "rows_dropped_non_executable": 3,
        "rows_dropped_non_session_execution_date": 0,
        "rows_dropped_missing_tplus1_return": 1,
    }
    backtest_summary_path.write_text(json.dumps(backtest_summary, indent=2), encoding="utf-8")

    execution_summary = {
        "execution_delay_sessions": 1,
        "fill_assumption": "full_fill",
        "cost_timing": "apply_on_execution_date",
        "n_skipped_rows": 2,
    }
    execution_summary_path.write_text(json.dumps(execution_summary, indent=2), encoding="utf-8")

    return backtest_daily_path, backtest_summary_path, costs_daily_path, execution_summary_path


def test_backtest_diagnostics_mvp_outputs_and_best_mode(tmp_workspace: dict[str, Path]) -> None:
    backtest_daily_path, backtest_summary_path, costs_daily_path, execution_summary_path = (
        _seed_backtest_diagnostics_case(tmp_workspace)
    )
    output_dir = tmp_workspace["artifacts"] / "backtest_diagnostics"

    result = run_backtest_diagnostics(
        backtest_daily_path=backtest_daily_path,
        backtest_summary_path=backtest_summary_path,
        costs_daily_path=costs_daily_path,
        execution_assumptions_summary_path=execution_summary_path,
        output_dir=output_dir,
        model_name="ridge_baseline",
        label_name="fwd_ret_5d",
        run_id="test_backtest_diagnostics_mvp",
    )

    assert result.diagnostics_daily_path.exists()
    assert result.diagnostics_by_mode_path.exists()
    assert result.diagnostics_summary_path.exists()
    assert result.row_count_daily == 4
    assert result.row_count_by_mode == 2

    diagnostics_daily = read_parquet(result.diagnostics_daily_path).sort_values(
        ["portfolio_mode", "date"]
    )
    diagnostics_by_mode = read_parquet(result.diagnostics_by_mode_path).sort_values("portfolio_mode")
    summary = json.loads(result.diagnostics_summary_path.read_text(encoding="utf-8"))

    assert not diagnostics_daily.empty
    assert not diagnostics_by_mode.empty
    assert {"daily_cost_drag", "gross_return", "net_return", "total_cost"}.issubset(
        diagnostics_daily.columns
    )
    assert {"cumulative_net_return", "total_cost_paid", "avg_turnover_if_available"}.issubset(
        diagnostics_by_mode.columns
    )

    assert np.allclose(
        diagnostics_daily["daily_cost_drag"].to_numpy(dtype=float),
        (diagnostics_daily["gross_return"] - diagnostics_daily["net_return"]).to_numpy(dtype=float),
        atol=1e-12,
    )
    assert np.allclose(
        diagnostics_daily["daily_cost_drag"].to_numpy(dtype=float),
        diagnostics_daily["total_cost"].to_numpy(dtype=float),
        atol=1e-12,
    )

    long_only = diagnostics_by_mode[diagnostics_by_mode["portfolio_mode"] == "long_only_top_n"].iloc[0]
    long_short = diagnostics_by_mode[
        diagnostics_by_mode["portfolio_mode"] == "long_short_top_bottom_n"
    ].iloc[0]
    assert np.isclose(float(long_only["cumulative_net_return"]), 0.0296, atol=1e-12)
    assert np.isclose(float(long_short["cumulative_net_return"]), 0.0302, atol=1e-12)

    assert summary["best_mode_by_cumulative_net_return"] == "long_short_top_bottom_n"
    assert summary["best_mode_by_sharpe_like_if_available"] == "long_short_top_bottom_n"
    assert np.isclose(float(summary["total_cost_paid_all_modes"]), 0.06, atol=1e-12)
    assert "execution_delay_sessions=1" in str(summary["notes_on_execution_assumptions"])
    assert isinstance(summary["notes_on_dropped_rows_if_available"], dict)

