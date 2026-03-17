from __future__ import annotations

import copy
import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import pytest


REQUIRED_OUTPUT_COLUMNS = {
    "symbol",
    "date",
    "borrow_fee_annual",
    "borrow_fee_daily",
    "borrow_availability_score",
    "htb_flag",
    "borrow_tier",
    "proxy_quality",
    "proxy_source",
    "stress_score",
    "jump_flag",
    "stale_input_flag",
    "fallback_flag",
    "config_version",
    "asof_timestamp",
}

TIER_ORDER = {"easy": 0, "medium": 1, "hard": 2, "blocked": 3}
QUALITY_ORDER = {"high": 2, "medium": 1, "low": 0}


# -----------------------------------------------------------------------------
# Import adapter
# -----------------------------------------------------------------------------


def _load_borrow_cost_proxy_module():
    module_names = [
        "simons_smallcap_swing.data.borrow.borrow_cost_proxy",
        "data.borrow.borrow_cost_proxy",
        "borrow_cost_proxy",
    ]
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue

    here = Path(__file__).resolve()
    candidate_paths = [
        here.parents[3] / "simons_smallcap_swing" / "data" / "borrow" / "borrow_cost_proxy.py",
        here.parents[2] / "data" / "borrow" / "borrow_cost_proxy.py",
        here.parents[1] / "borrow_cost_proxy.py",
    ]
    for path in candidate_paths:
        if path.exists():
            spec = importlib.util.spec_from_file_location("borrow_cost_proxy", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module

    raise ModuleNotFoundError(
        "Could not import borrow_cost_proxy.py. Expected it at "
        "simons_smallcap_swing.data.borrow.borrow_cost_proxy or in a nearby repo path."
    )


@pytest.fixture(scope="session")
def borrow_module():
    return _load_borrow_cost_proxy_module()


# -----------------------------------------------------------------------------
# Synthetic data builders
# -----------------------------------------------------------------------------


def _business_days(n: int = 35, start: str = "2025-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)



def _make_price_panel(symbol_params: Mapping[str, Mapping[str, float]], dates: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    dates = _business_days() if dates is None else dates
    rows = []
    for symbol, params in symbol_params.items():
        base_price = float(params.get("base_price", 25.0))
        price_slope = float(params.get("price_slope", 0.15))
        base_volume = float(params.get("base_volume", 1_000_000.0))
        volume_slope = float(params.get("volume_slope", 0.0))
        market_cap = float(params.get("market_cap", 1_000_000_000.0))
        float_shares = float(params.get("float_shares", 50_000_000.0))
        vol_amp = float(params.get("vol_amp", 0.015))
        size_bucket = params.get("size_bucket", "mid")
        for i, dt in enumerate(dates):
            cyc = ((i % 5) - 2) / 2.0
            close = max(1.0, base_price + price_slope * i + base_price * vol_amp * cyc)
            volume = max(10_000.0, base_volume + volume_slope * i + base_volume * 0.02 * cyc)
            rows.append(
                {
                    "symbol": symbol,
                    "instrument_id": symbol,
                    "date": pd.Timestamp(dt),
                    "close": close,
                    "adj_close": close,
                    "volume": volume,
                    "dollar_volume": close * volume,
                    "market_cap": market_cap,
                    "float_shares": float_shares,
                    "size_bucket": size_bucket,
                    "shares_outstanding": market_cap / close,
                    "turnover": volume / max(float_shares, 1.0),
                }
            )
    return pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)



def _make_universe(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel[["symbol", "instrument_id", "date", "size_bucket"]].copy()
    out["membership_state"] = "member"
    out["is_member"] = 1
    out["short_subuniverse_flag"] = 1
    return out



def _make_short_interest(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    base_rows = []
    for row in rows:
        rec = dict(row)
        rec.setdefault("date", rec.get("reference_date", rec.get("asof_date")))
        rec.setdefault("effective_date", rec.get("date"))
        rec.setdefault("asof_date", rec.get("date"))
        rec.setdefault("short_interest_ratio", 0.05)
        rec.setdefault("days_to_cover", 1.0)
        rec.setdefault("utilization", 0.20)
        base_rows.append(rec)
    return pd.DataFrame(base_rows)



def _make_direct_feed(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    base_rows = []
    for row in rows:
        rec = dict(row)
        rec.setdefault("date", rec.get("asof_date"))
        rec.setdefault("asof_date", rec.get("date"))
        rec.setdefault("borrow_fee_annual", 0.02)
        rec.setdefault("availability_score", rec.get("borrow_availability_score", 0.95))
        rec.setdefault("htb_flag", 0)
        base_rows.append(rec)
    return pd.DataFrame(base_rows)



def _make_htb_flags(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    base_rows = []
    for row in rows:
        rec = dict(row)
        rec.setdefault("date", rec.get("asof_date"))
        rec.setdefault("asof_date", rec.get("date"))
        rec.setdefault("explicit_htb_flag", 1)
        base_rows.append(rec)
    return pd.DataFrame(base_rows)



def _empty_df(columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


# -----------------------------------------------------------------------------
# Invocation adapter
# -----------------------------------------------------------------------------


def _extract_primary_dataframe(result: Any) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, dict):
        preferred_keys = [
            "daily",
            "borrow_proxy",
            "proxy",
            "output",
            "outputs",
            "result",
            "data",
        ]
        for key in preferred_keys:
            value = result.get(key)
            if isinstance(value, pd.DataFrame):
                return value
        for value in result.values():
            if isinstance(value, pd.DataFrame) and REQUIRED_OUTPUT_COLUMNS.issubset(set(value.columns)):
                return value
        for value in result.values():
            if isinstance(value, pd.DataFrame):
                return value
    if isinstance(result, (list, tuple)):
        for value in result:
            if isinstance(value, pd.DataFrame) and REQUIRED_OUTPUT_COLUMNS.issubset(set(value.columns)):
                return value
        for value in result:
            if isinstance(value, pd.DataFrame):
                return value
    raise TypeError("Could not extract the main borrow proxy DataFrame from module return value.")



def _find_runner(module: Any):
    candidate_names = [
        "run_borrow_cost_proxy",
        "build_borrow_cost_proxy",
        "build_borrow_proxy",
        "compute_borrow_cost_proxy",
        "estimate_borrow_cost_proxy",
        "construct_borrow_cost_proxy",
        "run",
        "build",
        "compute",
    ]
    for name in candidate_names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn

    for name, obj in vars(module).items():
        if callable(obj) and "borrow" in name.lower() and "proxy" in name.lower():
            return obj

    raise AttributeError(
        "No obvious runner function found in borrow_cost_proxy.py. "
        "Expected something like build_borrow_cost_proxy(...) or run_borrow_cost_proxy(...)."
    )



def _default_config(module: Any):
    cfg = getattr(module, "DEFAULT_CONFIG", None)
    if cfg is None:
        return None
    return copy.deepcopy(cfg)



def _coerce_date_like(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["date", "asof_date", "effective_date", "reference_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out



def _run_proxy(
    module: Any,
    *,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    short_interest: Optional[pd.DataFrame] = None,
    direct_feed: Optional[pd.DataFrame] = None,
    htb_flags: Optional[pd.DataFrame] = None,
    previous_state: Optional[pd.DataFrame] = None,
    asof_timestamp: str = "2025-02-28T21:00:00Z",
    run_id: str = "pytest_borrow_proxy",
):
    runner = _find_runner(module)
    sig = inspect.signature(runner)

    cfg = _default_config(module)
    inputs: Dict[str, Any] = {
        "prices": _coerce_date_like(prices),
        "prices_df": _coerce_date_like(prices),
        "market_df": _coerce_date_like(prices),
        "market_data": _coerce_date_like(prices),
        "universe": _coerce_date_like(universe),
        "universe_df": _coerce_date_like(universe),
        "short_interest": _coerce_date_like(short_interest if short_interest is not None else _empty_df(["symbol", "date", "asof_date", "short_interest_ratio", "days_to_cover", "utilization"])),
        "short_interest_df": _coerce_date_like(short_interest if short_interest is not None else _empty_df(["symbol", "date", "asof_date", "short_interest_ratio", "days_to_cover", "utilization"])),
        "si_df": _coerce_date_like(short_interest if short_interest is not None else _empty_df(["symbol", "date", "asof_date", "short_interest_ratio", "days_to_cover", "utilization"])),
        "direct_lending_feed": _coerce_date_like(direct_feed if direct_feed is not None else _empty_df(["symbol", "date", "asof_date", "borrow_fee_annual", "availability_score", "htb_flag"])),
        "direct_feed": _coerce_date_like(direct_feed if direct_feed is not None else _empty_df(["symbol", "date", "asof_date", "borrow_fee_annual", "availability_score", "htb_flag"])),
        "direct_feed_df": _coerce_date_like(direct_feed if direct_feed is not None else _empty_df(["symbol", "date", "asof_date", "borrow_fee_annual", "availability_score", "htb_flag"])),
        "htb_flags": _coerce_date_like(htb_flags if htb_flags is not None else _empty_df(["symbol", "date", "asof_date", "explicit_htb_flag"])),
        "htb_df": _coerce_date_like(htb_flags if htb_flags is not None else _empty_df(["symbol", "date", "asof_date", "explicit_htb_flag"])),
        "explicit_htb_df": _coerce_date_like(htb_flags if htb_flags is not None else _empty_df(["symbol", "date", "asof_date", "explicit_htb_flag"])),
        "previous_state": _coerce_date_like(previous_state if previous_state is not None else _empty_df(["symbol", "date", "borrow_tier", "borrow_fee_annual", "borrow_fee_daily", "borrow_availability_score"])),
        "prev_state": _coerce_date_like(previous_state if previous_state is not None else _empty_df(["symbol", "date", "borrow_tier", "borrow_fee_annual", "borrow_fee_daily", "borrow_availability_score"])),
        "previous_proxy": _coerce_date_like(previous_state if previous_state is not None else _empty_df(["symbol", "date", "borrow_tier", "borrow_fee_annual", "borrow_fee_daily", "borrow_availability_score"])),
        "config": cfg,
        "cfg": cfg,
        "asof_timestamp": asof_timestamp,
        "asof_ts": asof_timestamp,
        "asof": asof_timestamp,
        "run_id": run_id,
    }

    kwargs: Dict[str, Any] = {}
    missing_required = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if name in inputs:
            kwargs[name] = inputs[name]
            continue
        if param.default is inspect._empty and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            missing_required.append(name)

    if missing_required:
        raise TypeError(
            "Could not adapt pytest inputs to borrow_cost_proxy runner. "
            f"Missing required parameters: {missing_required}. "
            "Please align the adapter in tests/data/borrow/test_borrow_cost_proxy_unit.py."
        )

    result = runner(**kwargs)
    out = _extract_primary_dataframe(result).copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.sort_values([c for c in ["symbol", "date"] if c in out.columns]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Assertion helpers
# -----------------------------------------------------------------------------


def _latest_row(df: pd.DataFrame, symbol: str) -> pd.Series:
    sub = df.loc[df["symbol"] == symbol].sort_values("date")
    assert not sub.empty, f"No rows found for symbol={symbol!r}"
    return sub.iloc[-1]



def _tier_value(row: pd.Series) -> int:
    return TIER_ORDER[str(row["borrow_tier"]).lower()]



def _assert_required_contract(df: pd.DataFrame):
    missing = REQUIRED_OUTPUT_COLUMNS - set(df.columns)
    assert not missing, f"Missing required borrow proxy output columns: {sorted(missing)}"
    assert not df.empty, "Borrow proxy output is unexpectedly empty."
    assert df[["symbol", "date"]].drop_duplicates().shape[0] == len(df), "Output must be unique on (symbol, date)."
    assert (pd.to_numeric(df["borrow_fee_annual"], errors="coerce") >= 0).all(), "borrow_fee_annual must be non-negative."
    assert (pd.to_numeric(df["borrow_fee_daily"], errors="coerce") >= 0).all(), "borrow_fee_daily must be non-negative."
    alpha = pd.to_numeric(df["borrow_availability_score"], errors="coerce")
    assert ((alpha >= 0) & (alpha <= 1)).all(), "borrow_availability_score must stay in [0,1]."
    assert set(df["borrow_tier"].astype(str).str.lower().unique()).issubset(set(TIER_ORDER)), "Unexpected borrow_tier values found."
    assert set(df["proxy_quality"].astype(str).str.lower().unique()).issubset(set(QUALITY_ORDER)), "Unexpected proxy_quality values found."


# -----------------------------------------------------------------------------
# Core fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def base_inputs():
    dates = _business_days(35)
    prices = _make_price_panel(
        {
            "AAA": {"base_price": 30.0, "base_volume": 2_500_000.0, "market_cap": 4_000_000_000.0, "float_shares": 80_000_000.0, "vol_amp": 0.010, "size_bucket": "large"},
            "BBB": {"base_price": 18.0, "base_volume": 900_000.0, "market_cap": 900_000_000.0, "float_shares": 45_000_000.0, "vol_amp": 0.020, "size_bucket": "mid"},
        },
        dates=dates,
    )
    universe = _make_universe(prices)
    return {
        "dates": dates,
        "prices": prices,
        "universe": universe,
    }


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_output_contract_basic_columns_and_ranges(borrow_module, base_inputs):
    out = _run_proxy(
        borrow_module,
        prices=base_inputs["prices"],
        universe=base_inputs["universe"],
    )
    _assert_required_contract(out)



def test_short_interest_with_future_asof_is_ignored(borrow_module, base_inputs):
    last_date = base_inputs["dates"][-1]
    future_si = _make_short_interest(
        [
            {
                "symbol": "AAA",
                "date": last_date,
                "asof_date": last_date + pd.Timedelta(days=15),
                "short_interest_ratio": 0.85,
                "days_to_cover": 20.0,
                "utilization": 0.98,
            }
        ]
    )

    out_with_future = _run_proxy(
        borrow_module,
        prices=base_inputs["prices"],
        universe=base_inputs["universe"],
        short_interest=future_si,
        run_id="pytest_future_si",
    )
    out_without_si = _run_proxy(
        borrow_module,
        prices=base_inputs["prices"],
        universe=base_inputs["universe"],
        short_interest=_empty_df(future_si.columns),
        run_id="pytest_no_si",
    )

    row_future = _latest_row(out_with_future, "AAA")
    row_none = _latest_row(out_without_si, "AAA")

    assert row_future["borrow_fee_annual"] == pytest.approx(row_none["borrow_fee_annual"], rel=1e-10, abs=1e-12)
    assert row_future["borrow_availability_score"] == pytest.approx(row_none["borrow_availability_score"], rel=1e-10, abs=1e-12)
    assert str(row_future["borrow_tier"]).lower() == str(row_none["borrow_tier"]).lower()



def test_direct_lending_feed_dominates_other_sources(borrow_module, base_inputs):
    last_date = base_inputs["dates"][-1]
    benign_si = _make_short_interest(
        [
            {
                "symbol": "AAA",
                "date": last_date,
                "asof_date": last_date,
                "short_interest_ratio": 0.02,
                "days_to_cover": 0.5,
                "utilization": 0.05,
            }
        ]
    )
    severe_direct = _make_direct_feed(
        [
            {
                "symbol": "AAA",
                "date": last_date,
                "asof_date": last_date,
                "borrow_fee_annual": 1.20,
                "availability_score": 0.02,
                "htb_flag": 1,
            }
        ]
    )

    out_without_direct = _run_proxy(
        borrow_module,
        prices=base_inputs["prices"],
        universe=base_inputs["universe"],
        short_interest=benign_si,
        run_id="pytest_no_direct",
    )
    out_with_direct = _run_proxy(
        borrow_module,
        prices=base_inputs["prices"],
        universe=base_inputs["universe"],
        short_interest=benign_si,
        direct_feed=severe_direct,
        run_id="pytest_with_direct",
    )

    row_no_direct = _latest_row(out_without_direct, "AAA")
    row_with_direct = _latest_row(out_with_direct, "AAA")

    assert row_with_direct["borrow_fee_annual"] >= row_no_direct["borrow_fee_annual"]
    assert row_with_direct["borrow_availability_score"] <= row_no_direct["borrow_availability_score"]
    assert _tier_value(row_with_direct) >= _tier_value(row_no_direct)
    assert row_with_direct["borrow_fee_annual"] >= 0.90  # direct severe fee should not be washed out materially
    assert int(row_with_direct["htb_flag"]) == 1
    assert "direct" in str(row_with_direct["proxy_source"]).lower()



def test_higher_days_to_cover_increases_borrow_fee(borrow_module):
    dates = _business_days(35)
    prices = _make_price_panel(
        {
            "LOWDTC": {"base_price": 20.0, "base_volume": 1_500_000.0, "market_cap": 1_000_000_000.0, "float_shares": 50_000_000.0, "vol_amp": 0.015},
            "HIGHDTC": {"base_price": 20.0, "base_volume": 1_500_000.0, "market_cap": 1_000_000_000.0, "float_shares": 50_000_000.0, "vol_amp": 0.015},
        },
        dates=dates,
    )
    universe = _make_universe(prices)
    si = _make_short_interest(
        [
            {
                "symbol": "LOWDTC",
                "date": dates[-1],
                "asof_date": dates[-1],
                "short_interest_ratio": 0.12,
                "days_to_cover": 1.0,
                "utilization": 0.35,
            },
            {
                "symbol": "HIGHDTC",
                "date": dates[-1],
                "asof_date": dates[-1],
                "short_interest_ratio": 0.12,
                "days_to_cover": 12.0,
                "utilization": 0.35,
            },
        ]
    )

    out = _run_proxy(
        borrow_module,
        prices=prices,
        universe=universe,
        short_interest=si,
        run_id="pytest_dtc_monotonic",
    )

    low = _latest_row(out, "LOWDTC")
    high = _latest_row(out, "HIGHDTC")
    assert high["borrow_fee_annual"] >= low["borrow_fee_annual"]
    assert high["borrow_availability_score"] <= low["borrow_availability_score"]
    assert _tier_value(high) >= _tier_value(low)



def test_lower_adv20_increases_borrow_fee(borrow_module):
    dates = _business_days(35)
    prices = _make_price_panel(
        {
            "LIQUID": {"base_price": 25.0, "base_volume": 4_000_000.0, "market_cap": 1_500_000_000.0, "float_shares": 60_000_000.0, "vol_amp": 0.015},
            "ILLIQ": {"base_price": 25.0, "base_volume": 150_000.0, "market_cap": 1_500_000_000.0, "float_shares": 60_000_000.0, "vol_amp": 0.015},
        },
        dates=dates,
    )
    universe = _make_universe(prices)

    out = _run_proxy(
        borrow_module,
        prices=prices,
        universe=universe,
        run_id="pytest_adv_monotonic",
    )

    liquid = _latest_row(out, "LIQUID")
    illiq = _latest_row(out, "ILLIQ")
    assert illiq["borrow_fee_annual"] >= liquid["borrow_fee_annual"]
    assert illiq["borrow_availability_score"] <= liquid["borrow_availability_score"]
    assert _tier_value(illiq) >= _tier_value(liquid)



def test_higher_volatility_increases_borrow_fee(borrow_module):
    dates = _business_days(35)
    prices = _make_price_panel(
        {
            "LOWVOL": {"base_price": 22.0, "base_volume": 1_000_000.0, "market_cap": 800_000_000.0, "float_shares": 40_000_000.0, "vol_amp": 0.005},
            "HIGHVOL": {"base_price": 22.0, "base_volume": 1_000_000.0, "market_cap": 800_000_000.0, "float_shares": 40_000_000.0, "vol_amp": 0.080},
        },
        dates=dates,
    )
    universe = _make_universe(prices)

    out = _run_proxy(
        borrow_module,
        prices=prices,
        universe=universe,
        run_id="pytest_vol_monotonic",
    )

    low = _latest_row(out, "LOWVOL")
    high = _latest_row(out, "HIGHVOL")
    assert high["borrow_fee_annual"] >= low["borrow_fee_annual"]
    assert high["borrow_availability_score"] <= low["borrow_availability_score"]
    assert _tier_value(high) >= _tier_value(low)



def test_explicit_htb_flag_dominates_benign_market_profile(borrow_module, base_inputs):
    last_date = base_inputs["dates"][-1]
    htb = _make_htb_flags(
        [
            {
                "symbol": "AAA",
                "date": last_date,
                "asof_date": last_date,
                "explicit_htb_flag": 1,
            }
        ]
    )

    out = _run_proxy(
        borrow_module,
        prices=base_inputs["prices"],
        universe=base_inputs["universe"],
        htb_flags=htb,
        run_id="pytest_explicit_htb",
    )
    row = _latest_row(out, "AAA")

    assert int(row["htb_flag"]) == 1
    assert _tier_value(row) >= TIER_ORDER["hard"]
    assert row["borrow_fee_annual"] > 0
    assert "htb" in str(row["proxy_source"]).lower() or _tier_value(row) == TIER_ORDER["blocked"]



def test_blocked_signal_is_not_smoothed_away(borrow_module):
    dates = _business_days(35)
    prices = _make_price_panel(
        {
            "CRISIS": {"base_price": 12.0, "base_volume": 600_000.0, "market_cap": 350_000_000.0, "float_shares": 30_000_000.0, "vol_amp": 0.020},
        },
        dates=dates,
    )
    universe = _make_universe(prices)
    direct_feed = _make_direct_feed(
        [
            {
                "symbol": "CRISIS",
                "date": dates[-1],
                "asof_date": dates[-1],
                "borrow_fee_annual": 3.50,
                "availability_score": 0.00,
                "htb_flag": 1,
            }
        ]
    )

    out = _run_proxy(
        borrow_module,
        prices=prices,
        universe=universe,
        direct_feed=direct_feed,
        run_id="pytest_blocked_not_smoothed_away",
    )
    row = _latest_row(out, "CRISIS")

    assert int(row["htb_flag"]) == 1
    assert _tier_value(row) == TIER_ORDER["blocked"] or row["borrow_fee_annual"] >= 1.50
    assert row["borrow_availability_score"] <= 0.10



def test_partial_inputs_trigger_conservative_fallback_and_non_high_quality(borrow_module):
    dates = _business_days(8)
    prices = _make_price_panel(
        {
            "MICRO": {"base_price": 4.0, "base_volume": 40_000.0, "market_cap": 40_000_000.0, "float_shares": 10_000_000.0, "vol_amp": 0.060, "size_bucket": "micro"},
        },
        dates=dates,
    )
    universe = _make_universe(prices)

    out = _run_proxy(
        borrow_module,
        prices=prices,
        universe=universe,
        short_interest=None,
        direct_feed=None,
        htb_flags=None,
        run_id="pytest_conservative_fallback",
    )
    row = _latest_row(out, "MICRO")

    assert int(row["fallback_flag"]) == 1
    assert str(row["proxy_quality"]).lower() in {"medium", "low"}
    assert _tier_value(row) >= TIER_ORDER["medium"]

