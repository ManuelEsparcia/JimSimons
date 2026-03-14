from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.edgar.point_in_time import build_fundamentals_pit
from data.edgar.ticker_cik import build_ticker_cik_map
from data.price.adjust_prices import adjust_prices
from data.price.fetch_prices import fetch_prices
from data.price.market_proxies import build_market_proxies
from data.reference.build_reference import build_reference_data
from data.universe.build_universe import build_universe
from data.universe.corporate_actions import build_corporate_actions
from features.build_features import build_features
from simons_core.io.parquet_store import read_parquet
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


FEATURES_MIN_SCHEMA = DataSchema(
    name="features_matrix_min_test",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("ret_1d_lag1", "float64", nullable=True),
        ColumnSpec("ret_5d_lag1", "float64", nullable=True),
        ColumnSpec("ret_20d_lag1", "float64", nullable=True),
        ColumnSpec("momentum_20d_excl_1d", "float64", nullable=True),
        ColumnSpec("vol_5d", "float64", nullable=True),
        ColumnSpec("vol_20d", "float64", nullable=True),
        ColumnSpec("abs_ret_1d_lag1", "float64", nullable=True),
        ColumnSpec("log_volume_lag1", "float64", nullable=True),
        ColumnSpec("turnover_proxy_lag1", "float64", nullable=True),
        ColumnSpec("log_dollar_volume_lag1", "float64", nullable=True),
        ColumnSpec("mkt_breadth_up_lag1", "float64", nullable=True),
        ColumnSpec("mkt_equal_weight_return_lag1", "float64", nullable=True),
        ColumnSpec("mkt_cross_sectional_vol_lag1", "float64", nullable=True),
        ColumnSpec("mkt_coverage_ratio_lag1", "float64", nullable=True),
        ColumnSpec("log_total_assets", "float64", nullable=True),
        ColumnSpec("shares_outstanding", "float64", nullable=True),
        ColumnSpec("revenue_scale_proxy", "float64", nullable=True),
        ColumnSpec("net_income_scale_proxy", "float64", nullable=True),
    ),
    primary_key=("date", "instrument_id"),
    allow_extra_columns=True,
)


def _build_features_inputs(tmp_workspace: dict[str, Path]) -> dict[str, Path]:
    reference_root = tmp_workspace["data"] / "reference"
    universe_root = tmp_workspace["data"] / "universe"
    price_root = tmp_workspace["data"] / "price"
    edgar_root = tmp_workspace["data"] / "edgar"
    features_root = tmp_workspace["data"] / "features"

    build_reference_data(output_dir=reference_root, run_id="test_reference_features_mvp")
    universe_result = build_universe(
        reference_root=reference_root,
        output_dir=universe_root,
        run_id="test_universe_features_mvp",
    )
    corporate_actions_result = build_corporate_actions(
        reference_root=reference_root,
        universe_history_path=universe_result.universe_history,
        output_dir=universe_root,
        run_id="test_corporate_actions_features_mvp",
    )
    raw_result = fetch_prices(
        reference_root=reference_root,
        universe_history_path=universe_result.universe_history,
        output_dir=price_root,
        run_id="test_fetch_features_mvp",
    )
    adjusted_result = adjust_prices(
        raw_prices_path=raw_result.raw_prices_path,
        corporate_actions_path=corporate_actions_result.corporate_actions_path,
        output_dir=price_root,
        run_id="test_adjust_features_mvp",
    )
    market_result = build_market_proxies(
        adjusted_prices_path=adjusted_result.adjusted_prices_path,
        universe_history_path=universe_result.universe_history,
        trading_calendar_path=reference_root / "trading_calendar.parquet",
        output_dir=price_root,
        run_id="test_market_features_mvp",
    )
    ticker_cik_result = build_ticker_cik_map(
        reference_root=reference_root,
        universe_history_path=universe_result.universe_history,
        output_dir=edgar_root,
        run_id="test_ticker_cik_features_mvp",
    )
    fundamentals_result = build_fundamentals_pit(
        ticker_cik_map_path=ticker_cik_result.ticker_cik_map_path,
        universe_history_path=universe_result.universe_history,
        submissions_raw_path=edgar_root / "submissions_raw.parquet",
        companyfacts_raw_path=edgar_root / "companyfacts_raw.parquet",
        output_dir=edgar_root,
        run_id="test_fundamentals_features_mvp",
    )

    return {
        "adjusted": adjusted_result.adjusted_prices_path,
        "universe": universe_result.universe_history,
        "market": market_result.market_proxies_path,
        "fundamentals": fundamentals_result.fundamentals_pit_path,
        "calendar": reference_root / "trading_calendar.parquet",
        "features_root": features_root,
    }


def test_build_features_generates_non_empty_schema_valid_output(
    tmp_workspace: dict[str, Path],
) -> None:
    paths = _build_features_inputs(tmp_workspace)
    result = build_features(
        adjusted_prices_path=paths["adjusted"],
        universe_history_path=paths["universe"],
        market_proxies_path=paths["market"],
        fundamentals_pit_path=paths["fundamentals"],
        trading_calendar_path=paths["calendar"],
        output_dir=paths["features_root"],
        run_id="test_build_features_mvp",
    )

    assert result.features_path.exists()
    assert result.summary_path.exists()
    assert (result.features_path.with_suffix(".parquet.manifest.json")).exists()
    assert result.row_count > 0
    assert result.n_instruments > 0
    assert result.n_features >= 18

    features = read_parquet(result.features_path)
    assert_schema(features, FEATURES_MIN_SCHEMA)
    assert len(features) > 0
    assert not features.duplicated(["date", "instrument_id"]).any()

    features["date"] = pd.to_datetime(features["date"], errors="coerce").dt.normalize()
    features["instrument_id"] = features["instrument_id"].astype(str)
    eligible = read_parquet(paths["universe"]).copy()
    eligible["date"] = pd.to_datetime(eligible["date"], errors="coerce").dt.normalize()
    eligible["instrument_id"] = eligible["instrument_id"].astype(str)
    eligible_pairs = set(
        eligible.loc[eligible["is_eligible"].astype(bool), ["date", "instrument_id"]]
        .itertuples(index=False, name=None)
    )
    feature_pairs = set(features[["date", "instrument_id"]].itertuples(index=False, name=None))
    assert feature_pairs.issubset(eligible_pairs)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["n_rows"] == len(features)
    assert summary["n_instruments"] == int(features["instrument_id"].nunique())
    assert summary["decision_lag_days"] == 1
    assert summary["n_features"] >= 18
    assert len(summary["feature_names"]) >= 18
    assert "pct_missing_by_feature" in summary


def test_build_features_small_case_lagged_market_and_fundamentals_pit(
    tmp_workspace: dict[str, Path],
) -> None:
    base = tmp_workspace["data"] / "features_small_case"
    adjusted_path = base / "price" / "adjusted_prices.parquet"
    universe_path = base / "universe" / "universe_history.parquet"
    market_path = base / "price" / "market_proxies.parquet"
    fundamentals_path = base / "edgar" / "fundamentals_pit.parquet"
    calendar_path = base / "reference" / "trading_calendar.parquet"
    output_dir = base / "features"

    for folder in (adjusted_path.parent, universe_path.parent, market_path.parent, fundamentals_path.parent, calendar_path.parent):
        folder.mkdir(parents=True, exist_ok=True)

    sessions = pd.bdate_range("2026-01-05", periods=10, freq="B")
    calendar = pd.DataFrame({"date": sessions, "is_session": True})
    calendar.to_parquet(calendar_path, index=False)

    close_a = [100.0]
    for _ in range(9):
        close_a.append(close_a[-1] * 1.1)
    close_b = [50.0 + float(i) for i in range(10)]

    adjusted = pd.DataFrame(
        {
            "date": list(sessions) * 2,
            "instrument_id": ["SIMA"] * 10 + ["SIMB"] * 10,
            "ticker": ["AAA"] * 10 + ["BBB"] * 10,
            "open_adj": close_a + close_b,
            "high_adj": close_a + close_b,
            "low_adj": close_a + close_b,
            "close_adj": close_a + close_b,
            "volume_adj": [1000.0] * 10 + [500.0] * 10,
        }
    )
    adjusted.to_parquet(adjusted_path, index=False)

    universe = pd.DataFrame(
        {
            "date": list(sessions) * 2,
            "instrument_id": ["SIMA"] * 10 + ["SIMB"] * 10,
            "ticker": ["AAA"] * 10 + ["BBB"] * 10,
            "is_eligible": [True] * 10 + [False] * 10,
        }
    )
    universe.to_parquet(universe_path, index=False)

    market = pd.DataFrame(
        {
            "date": sessions,
            "breadth_up": [0.40 + 0.01 * i for i in range(10)],
            "equal_weight_return": [0.01] * 10,
            "cross_sectional_vol": [0.02 + 0.001 * i for i in range(10)],
            "coverage_ratio": [0.95] * 10,
        }
    )
    market.to_parquet(market_path, index=False)

    asof_dates = [sessions[2], sessions[5]]  # 2026-01-07 and 2026-01-12
    facts_rows: list[dict[str, object]] = []
    metric_values = {
        "revenue": [100.0, 200.0],
        "net_income": [10.0, 20.0],
        "total_assets": [1000.0, 2000.0],
        "shares_outstanding": [1_000_000.0, 1_200_000.0],
    }
    for metric_name, values in metric_values.items():
        for idx, asof in enumerate(asof_dates):
            facts_rows.append(
                {
                    "instrument_id": "SIMA",
                    "ticker": "AAA",
                    "cik": "0000000001",
                    "asof_date": pd.Timestamp(asof).tz_localize("UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59),
                    "acceptance_ts": pd.Timestamp(asof).tz_localize("UTC") + pd.Timedelta(hours=12),
                    "metric_name": metric_name,
                    "metric_value": values[idx],
                }
            )
    fundamentals = pd.DataFrame(facts_rows)
    fundamentals.to_parquet(fundamentals_path, index=False)

    result = build_features(
        adjusted_prices_path=adjusted_path,
        universe_history_path=universe_path,
        market_proxies_path=market_path,
        fundamentals_pit_path=fundamentals_path,
        trading_calendar_path=calendar_path,
        output_dir=output_dir,
        run_id="test_build_features_small_case",
    )

    features = read_parquet(result.features_path).sort_values(["date", "instrument_id"]).reset_index(drop=True)
    features["date"] = pd.to_datetime(features["date"], errors="coerce").dt.normalize()
    assert set(features["instrument_id"].unique().tolist()) == {"SIMA"}
    assert not features.duplicated(["date", "instrument_id"]).any()

    row_jan06 = features.loc[features["date"] == sessions[1]].iloc[0]
    assert np.isclose(float(row_jan06["mkt_breadth_up_lag1"]), 0.40, atol=1e-12)

    row_jan13 = features.loc[features["date"] == sessions[6]].iloc[0]
    expected_ret_1d = 0.1
    expected_ret_5d = (1.1**5) - 1.0
    assert np.isclose(float(row_jan13["ret_1d_lag1"]), expected_ret_1d, atol=1e-12)
    assert np.isclose(float(row_jan13["ret_5d_lag1"]), expected_ret_5d, atol=1e-12)
    assert np.isclose(float(row_jan13["vol_5d"]), 0.0, atol=1e-12)

    # PIT check: second fundamentals snapshot is visible only from decision_ref_date = 2026-01-12 onward.
    row_jan12 = features.loc[features["date"] == sessions[5]].iloc[0]
    row_jan13 = features.loc[features["date"] == sessions[6]].iloc[0]
    assert np.isclose(float(row_jan12["revenue_scale_proxy"]), np.log1p(100.0), atol=1e-12)
    assert np.isclose(float(row_jan13["revenue_scale_proxy"]), np.log1p(200.0), atol=1e-12)
    assert np.isclose(float(row_jan12["log_total_assets"]), np.log(1000.0), atol=1e-12)
    assert np.isclose(float(row_jan13["log_total_assets"]), np.log(2000.0), atol=1e-12)

