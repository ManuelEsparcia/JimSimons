from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_week6_execution_backtest import run_week6_execution_backtest
from simons_core.io.parquet_store import read_parquet


def _seed_week6_prerequisites(data_root: Path) -> dict[str, Path]:
    signals_root = data_root / "signals"
    universe_root = data_root / "universe"
    reference_root = data_root / "reference"
    price_root = data_root / "price"
    labels_root = data_root / "labels"
    features_root = data_root / "features"
    edgar_root = data_root / "edgar"
    for directory in (
        signals_root,
        universe_root,
        reference_root,
        price_root,
        labels_root,
        features_root,
        edgar_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    d1 = pd.Timestamp("2026-09-01")
    d2 = pd.Timestamp("2026-09-02")
    d3 = pd.Timestamp("2026-09-03")

    signals_rows = []
    for date, a_score, b_score in (
        (d1, 0.9, -0.2),
        (d2, 0.8, -0.1),
        (d3, 0.7, -0.3),
    ):
        signals_rows.append(
            {
                "date": date,
                "instrument_id": "SIMA",
                "ticker": "AAA",
                "split_role": "test",
                "split_name": "holdout_temporal_purged",
                "horizon_days": 5,
                "label_name": "fwd_ret_5d",
                "model_name": "ridge_baseline",
                "raw_score": float(a_score),
                "rank_pct": 1.0,
            }
        )
        signals_rows.append(
            {
                "date": date,
                "instrument_id": "SIMB",
                "ticker": "BBB",
                "split_role": "test",
                "split_name": "holdout_temporal_purged",
                "horizon_days": 5,
                "label_name": "fwd_ret_5d",
                "model_name": "ridge_baseline",
                "raw_score": float(b_score),
                "rank_pct": 0.5,
            }
        )
    signals_path = signals_root / "signals_daily.parquet"
    pd.DataFrame(signals_rows).to_parquet(signals_path, index=False)

    universe_rows = []
    for date in (d1, d2, d3):
        universe_rows.extend(
            [
                {"date": date, "instrument_id": "SIMA", "ticker": "AAA", "is_eligible": True},
                {"date": date, "instrument_id": "SIMB", "ticker": "BBB", "is_eligible": True},
            ]
        )
    universe_path = universe_root / "universe_history.parquet"
    pd.DataFrame(universe_rows).to_parquet(universe_path, index=False)

    trading_calendar_path = reference_root / "trading_calendar.parquet"
    pd.DataFrame(
        {
            "date": [d1, d2, d3],
            "is_session": [True, True, True],
        }
    ).to_parquet(trading_calendar_path, index=False)

    prices_path = price_root / "adjusted_prices.parquet"
    pd.DataFrame(
        [
            {"date": d2, "instrument_id": "SIMA", "ticker": "AAA", "close_adj": 100.0, "volume_adj": 1_000_000.0},
            {"date": d3, "instrument_id": "SIMA", "ticker": "AAA", "close_adj": 110.0, "volume_adj": 1_100_000.0},
            {"date": d2, "instrument_id": "SIMB", "ticker": "BBB", "close_adj": 200.0, "volume_adj": 800_000.0},
            {"date": d3, "instrument_id": "SIMB", "ticker": "BBB", "close_adj": 180.0, "volume_adj": 850_000.0},
        ]
    ).to_parquet(prices_path, index=False)

    # Compatibility placeholders from Week 1-5 (must remain unchanged).
    labels_forward = labels_root / "labels_forward.parquet"
    pd.DataFrame(
        [
            {
                "date": d1,
                "instrument_id": "SIMA",
                "ticker": "AAA",
                "horizon_days": 5,
                "entry_date": d2,
                "exit_date": d3,
                "label_name": "fwd_ret_5d",
                "label_value": 0.01,
            }
        ]
    ).to_parquet(labels_forward, index=False)

    features_matrix = features_root / "features_matrix.parquet"
    pd.DataFrame(
        [{"date": d1, "instrument_id": "SIMA", "ticker": "AAA", "ret_1d_lag1": 0.01}]
    ).to_parquet(features_matrix, index=False)

    purged_cv_folds = labels_root / "purged_cv_folds.parquet"
    pd.DataFrame(
        [
            {
                "fold_id": 0,
                "date": d1,
                "instrument_id": "SIMA",
                "horizon_days": 5,
                "label_name": "fwd_ret_5d",
                "split_role": "train",
                "entry_date": d2,
                "exit_date": d3,
            }
        ]
    ).to_parquet(purged_cv_folds, index=False)

    fundamentals_pit = edgar_root / "fundamentals_pit.parquet"
    pd.DataFrame(
        [{"instrument_id": "SIMA", "asof_date": d1, "metric_name": "Revenues", "metric_value": 100_000_000.0}]
    ).to_parquet(fundamentals_pit, index=False)

    decile_daily = signals_root / "decile_daily.parquet"
    pd.DataFrame(
        [
            {
                "date": d1,
                "bucket": 1,
                "n_names_bucket": 2,
                "bucket_mean_target": 0.01,
                "bucket_median_target": 0.01,
                "n_names_ranked": 2,
                "model_name": "ridge_baseline",
                "label_name": "fwd_ret_5d",
            }
        ]
    ).to_parquet(decile_daily, index=False)

    return {
        "signals_daily": signals_path,
        "universe_history": universe_path,
        "trading_calendar": trading_calendar_path,
        "adjusted_prices": prices_path,
        "labels_forward": labels_forward,
        "features_matrix": features_matrix,
        "purged_cv_folds": purged_cv_folds,
        "fundamentals_pit": fundamentals_pit,
        "decile_daily": decile_daily,
    }


def test_week6_runner_end_to_end_smoke(tmp_workspace: dict[str, Path]) -> None:
    prereq = _seed_week6_prerequisites(tmp_workspace["data"])
    compatibility_counts_before = {
        key: int(len(read_parquet(path)))
        for key, path in prereq.items()
        if key in {"labels_forward", "features_matrix", "purged_cv_folds", "fundamentals_pit", "decile_daily"}
    }

    result = run_week6_execution_backtest(
        run_prefix="test_week6_e2e",
        data_root=tmp_workspace["data"],
        model_name="ridge_baseline",
        label_name="fwd_ret_5d",
        split_name="holdout_temporal_purged",
        horizon_days=5,
        split_roles=("test",),
        portfolio_modes=("long_only_top_n", "long_short_top_bottom_n"),
        top_n=1,
        bottom_n=1,
        execution_delay_sessions=1,
        fill_assumption="full_fill",
        cost_timing="apply_on_execution_date",
        cost_bps_per_turnover=10.0,
        entry_bps=2.0,
        exit_bps=2.0,
    )

    expected_artifacts = {
        "portfolio_holdings": result.artifacts["portfolio_holdings"],
        "portfolio_rebalance": result.artifacts["portfolio_rebalance"],
        "execution_holdings": result.artifacts["execution_holdings"],
        "execution_rebalance": result.artifacts["execution_rebalance"],
        "costs_daily": result.artifacts["costs_daily"],
        "costs_positions": result.artifacts["costs_positions"],
        "backtest_daily": result.artifacts["backtest_daily"],
        "backtest_contributions": result.artifacts["backtest_contributions"],
        "backtest_summary": result.artifacts["backtest_summary"],
        "backtest_diagnostics_daily": result.artifacts["backtest_diagnostics_daily"],
        "backtest_diagnostics_by_mode": result.artifacts["backtest_diagnostics_by_mode"],
        "backtest_diagnostics_summary": result.artifacts["backtest_diagnostics_summary"],
    }
    for path in expected_artifacts.values():
        assert path.exists()
        assert path.stat().st_size > 0

    portfolio_holdings = read_parquet(expected_artifacts["portfolio_holdings"])
    portfolio_rebalance = read_parquet(expected_artifacts["portfolio_rebalance"])
    execution_holdings = read_parquet(expected_artifacts["execution_holdings"])
    execution_rebalance = read_parquet(expected_artifacts["execution_rebalance"])
    costs_daily = read_parquet(expected_artifacts["costs_daily"])
    costs_positions = read_parquet(expected_artifacts["costs_positions"])
    backtest_daily = read_parquet(expected_artifacts["backtest_daily"])
    backtest_contributions = read_parquet(expected_artifacts["backtest_contributions"])
    backtest_summary = json.loads(expected_artifacts["backtest_summary"].read_text(encoding="utf-8"))
    diagnostics_daily = read_parquet(expected_artifacts["backtest_diagnostics_daily"])
    diagnostics_by_mode = read_parquet(expected_artifacts["backtest_diagnostics_by_mode"])
    diagnostics_summary = json.loads(expected_artifacts["backtest_diagnostics_summary"].read_text(encoding="utf-8"))

    assert len(portfolio_holdings) > 0
    assert len(portfolio_rebalance) > 0
    assert len(execution_holdings) > 0
    assert len(execution_rebalance) > 0
    assert len(costs_daily) > 0
    assert len(costs_positions) > 0
    assert len(backtest_daily) > 0
    assert len(backtest_contributions) > 0
    assert len(diagnostics_daily) > 0
    assert len(diagnostics_by_mode) > 0

    non_exec_holdings = execution_holdings[execution_holdings["is_executable"] == False]  # noqa: E712
    assert len(non_exec_holdings) > 0

    # Cost model may keep non-executable rows with zero cost or drop zero-change rows;
    # in both cases, non-executable signals must not create positive costs.
    non_exec_positions = costs_positions[costs_positions["is_executable"] == False]  # noqa: E712
    if len(non_exec_positions) > 0:
        assert np.allclose(non_exec_positions["total_cost"].to_numpy(dtype=float), 0.0, atol=1e-12)
    non_exec_cost_total = float(
        costs_positions.merge(
            non_exec_holdings[["signal_date", "instrument_id"]].drop_duplicates(),
            on=["signal_date", "instrument_id"],
            how="inner",
        )["total_cost"].sum()
    )
    assert np.isclose(non_exec_cost_total, 0.0, atol=1e-12)

    # Non-executable signal rows should not generate contributions/exposure in backtest.
    non_exec_keys = non_exec_holdings[["signal_date", "instrument_id"]].copy()
    merged_non_exec = non_exec_keys.merge(
        backtest_contributions[["signal_date", "instrument_id"]],
        on=["signal_date", "instrument_id"],
        how="inner",
    )
    assert merged_non_exec.empty

    assert np.allclose(
        (backtest_daily["gross_return"] - backtest_daily["total_cost"]).to_numpy(dtype=float),
        backtest_daily["net_return"].to_numpy(dtype=float),
        atol=1e-12,
    )

    assert backtest_summary["rows_dropped_non_executable"] > 0
    assert diagnostics_summary["best_mode_by_cumulative_net_return"] in {
        "long_only_top_n",
        "long_short_top_bottom_n",
    }

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_prefix"] == "test_week6_e2e"
    assert manifest["statuses"]["construct_portfolio"] == "DONE"
    assert manifest["statuses"]["execution_assumptions"] == "DONE"
    assert manifest["statuses"]["cost_model"] == "DONE"
    assert manifest["statuses"]["backtest_engine"] == "DONE"
    assert manifest["statuses"]["backtest_diagnostics"] == "DONE"

    compatibility_counts_after = {
        key: int(len(read_parquet(path)))
        for key, path in prereq.items()
        if key in compatibility_counts_before
    }
    assert compatibility_counts_after == compatibility_counts_before
