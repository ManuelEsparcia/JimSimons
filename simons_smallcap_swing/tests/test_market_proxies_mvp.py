from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.price.adjust_prices import adjust_prices
from data.price.fetch_prices import fetch_prices
from data.price.market_proxies import build_market_proxies
from data.reference.build_reference import build_reference_data
from data.universe.build_universe import build_universe
from simons_core.io.parquet_store import read_parquet


def _build_week2_inputs(tmp_workspace: dict[str, Path]) -> dict[str, Path]:
    reference_root = tmp_workspace["data"] / "reference"
    universe_root = tmp_workspace["data"] / "universe"
    price_root = tmp_workspace["data"] / "price"

    build_reference_data(output_dir=reference_root, run_id="test_reference_market_proxies")
    universe_result = build_universe(
        reference_root=reference_root,
        output_dir=universe_root,
        run_id="test_universe_market_proxies",
    )
    raw_result = fetch_prices(
        reference_root=reference_root,
        universe_history_path=universe_result.universe_history,
        output_dir=price_root,
        run_id="test_fetch_market_proxies",
    )
    adjusted_result = adjust_prices(
        raw_prices_path=raw_result.raw_prices_path,
        output_dir=price_root,
        run_id="test_adjust_market_proxies",
    )
    return {
        "adjusted": adjusted_result.adjusted_prices_path,
        "universe": universe_result.universe_history,
        "calendar": reference_root / "trading_calendar.parquet",
    }


def test_market_proxies_mvp_generates_artifacts_and_summary(
    tmp_workspace: dict[str, Path],
) -> None:
    paths = _build_week2_inputs(tmp_workspace)
    output_dir = tmp_workspace["artifacts"] / "market_proxies"

    result = build_market_proxies(
        adjusted_prices_path=paths["adjusted"],
        universe_history_path=paths["universe"],
        trading_calendar_path=paths["calendar"],
        output_dir=output_dir,
        run_id="test_market_proxies_mvp",
    )

    assert result.market_proxies_path.exists()
    assert result.summary_path.exists()
    assert result.row_count > 0

    frame = read_parquet(result.market_proxies_path)
    assert len(frame) > 0

    required_columns = {
        "date",
        "n_names",
        "n_names_with_prices",
        "coverage_ratio",
        "equal_weight_return",
        "median_return",
        "breadth_up",
        "breadth_down",
        "cross_sectional_vol",
        "turnover_proxy",
    }
    assert required_columns.issubset(set(frame.columns))

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    expected_summary_keys = {
        "start_date",
        "end_date",
        "n_sessions",
        "avg_coverage_ratio",
        "min_coverage_ratio",
        "avg_n_names",
        "worst_session_by_coverage",
        "pct_sessions_low_coverage",
    }
    assert expected_summary_keys.issubset(summary.keys())


def test_market_proxies_mvp_pk_and_range_constraints(
    tmp_workspace: dict[str, Path],
) -> None:
    paths = _build_week2_inputs(tmp_workspace)
    result = build_market_proxies(
        adjusted_prices_path=paths["adjusted"],
        universe_history_path=paths["universe"],
        trading_calendar_path=paths["calendar"],
        output_dir=tmp_workspace["artifacts"] / "market_proxies_ranges",
        run_id="test_market_proxies_ranges",
    )
    frame = read_parquet(result.market_proxies_path)

    assert not frame.duplicated(["date"]).any()
    assert frame["coverage_ratio"].between(0.0, 1.0).all()
    assert frame["breadth_up"].between(0.0, 1.0).all()
    assert frame["breadth_down"].between(0.0, 1.0).all()
    assert (frame["n_names_with_prices"] <= frame["n_names"]).all()
    assert (frame["cross_sectional_vol"] >= 0).all()


def test_market_proxies_small_case_calculation_and_pit_membership(
    tmp_workspace: dict[str, Path],
) -> None:
    base = tmp_workspace["data"]
    adjusted_path = base / "price" / "adjusted_small.parquet"
    universe_path = base / "universe" / "universe_small.parquet"
    calendar_path = base / "reference" / "trading_calendar_small.parquet"
    adjusted_path.parent.mkdir(parents=True, exist_ok=True)
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    calendar_path.parent.mkdir(parents=True, exist_ok=True)

    dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])

    # Adjusted has both names on all dates.
    adjusted = pd.DataFrame(
        [
            {"date": dates[0], "instrument_id": "SIMA", "ticker": "AAA", "close_adj": 100.0, "volume_adj": 1000.0},
            {"date": dates[1], "instrument_id": "SIMA", "ticker": "AAA", "close_adj": 110.0, "volume_adj": 1000.0},
            {"date": dates[2], "instrument_id": "SIMA", "ticker": "AAA", "close_adj": 121.0, "volume_adj": 1000.0},
            {"date": dates[0], "instrument_id": "SIMB", "ticker": "BBB", "close_adj": 100.0, "volume_adj": 1000.0},
            {"date": dates[1], "instrument_id": "SIMB", "ticker": "BBB", "close_adj": 90.0, "volume_adj": 1000.0},
            {"date": dates[2], "instrument_id": "SIMB", "ticker": "BBB", "close_adj": 81.0, "volume_adj": 1000.0},
        ]
    )
    adjusted.to_parquet(adjusted_path, index=False)

    # PIT universe: SIMB enters only from day 2.
    universe = pd.DataFrame(
        [
            {"date": dates[0], "instrument_id": "SIMA", "ticker": "AAA", "is_eligible": True},
            {"date": dates[1], "instrument_id": "SIMA", "ticker": "AAA", "is_eligible": True},
            {"date": dates[2], "instrument_id": "SIMA", "ticker": "AAA", "is_eligible": True},
            {"date": dates[1], "instrument_id": "SIMB", "ticker": "BBB", "is_eligible": True},
            {"date": dates[2], "instrument_id": "SIMB", "ticker": "BBB", "is_eligible": True},
        ]
    )
    universe.to_parquet(universe_path, index=False)

    calendar = pd.DataFrame({"date": dates, "is_session": True})
    calendar.to_parquet(calendar_path, index=False)

    result = build_market_proxies(
        adjusted_prices_path=adjusted_path,
        universe_history_path=universe_path,
        trading_calendar_path=calendar_path,
        output_dir=tmp_workspace["artifacts"] / "market_proxies_small_case",
        run_id="test_market_proxies_small_case",
    )
    frame = read_parquet(result.market_proxies_path).sort_values("date").reset_index(drop=True)

    # PIT-safe check: day 1 must include only SIMA despite SIMB existing in adjusted_prices.
    day1 = frame.loc[frame["date"] == dates[0]].iloc[0]
    assert int(day1["n_names"]) == 1

    day2 = frame.loc[frame["date"] == dates[1]].iloc[0]
    assert int(day2["n_names"]) == 2
    assert int(day2["n_names_with_prices"]) == 2
    assert np.isclose(float(day2["coverage_ratio"]), 1.0)
    assert np.isclose(float(day2["equal_weight_return"]), 0.0, atol=1e-12)
    assert np.isclose(float(day2["median_return"]), 0.0, atol=1e-12)
    assert np.isclose(float(day2["breadth_up"]), 0.5, atol=1e-12)
    assert np.isclose(float(day2["breadth_down"]), 0.5, atol=1e-12)
    assert np.isclose(float(day2["cross_sectional_vol"]), np.sqrt(0.02), atol=1e-12)
    assert np.isclose(float(day2["turnover_proxy"]), 1.0, atol=1e-12)
