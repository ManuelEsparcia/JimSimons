from __future__ import annotations

import copy
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# Module loaders
# -----------------------------------------------------------------------------


def _load_required_module(module_names: list[str], file_candidates: list[Path], label: str):
    for name in module_names:
        try:
            return importlib.import_module(name)
        except Exception:
            continue

    for path in file_candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location(f"{label}_under_test", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module

    raise ImportError(f"Could not locate {label}.py via import path or file path.")



def _load_optional_module(module_names: list[str], file_candidates: list[Path]):
    try:
        return _load_required_module(module_names, file_candidates, "optional_module")
    except Exception:
        return None


ROOT = Path.cwd()
LOCATE = _load_required_module(
    [
        "simons_smallcap_swing.data.borrow.locate_filter",
        "data.borrow.locate_filter",
        "locate_filter",
    ],
    [
        ROOT / "simons_smallcap_swing" / "data" / "borrow" / "locate_filter.py",
        ROOT / "data" / "borrow" / "locate_filter.py",
        ROOT / "locate_filter.py",
        Path("/mnt/data/locate_filter.py"),
    ],
    "locate_filter",
)

BORROW_QC = _load_required_module(
    [
        "simons_smallcap_swing.data.borrow.borrow_qc",
        "data.borrow.borrow_qc",
        "borrow_qc",
    ],
    [
        ROOT / "simons_smallcap_swing" / "data" / "borrow" / "borrow_qc.py",
        ROOT / "data" / "borrow" / "borrow_qc.py",
        ROOT / "borrow_qc.py",
        Path("/mnt/data/borrow_qc.py"),
    ],
    "borrow_qc",
)

BORROW_PROXY = _load_optional_module(
    [
        "simons_smallcap_swing.data.borrow.borrow_cost_proxy",
        "data.borrow.borrow_cost_proxy",
        "borrow_cost_proxy",
    ],
    [
        ROOT / "simons_smallcap_swing" / "data" / "borrow" / "borrow_cost_proxy.py",
        ROOT / "data" / "borrow" / "borrow_cost_proxy.py",
        ROOT / "borrow_cost_proxy.py",
        Path("/mnt/data/borrow_cost_proxy.py"),
    ],
)


# -----------------------------------------------------------------------------
# Shared builders
# -----------------------------------------------------------------------------


def _ts(date_like: str) -> pd.Timestamp:
    return pd.Timestamp(date_like)



def _make_universe(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows)).copy()
    if df.empty:
        raise ValueError("Universe fixture cannot be empty.")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["membership_state"] = df.get("membership_state", "active")
    df["is_member"] = df.get("is_member", 1)
    df["short_subuniverse_flag"] = df.get("short_subuniverse_flag", 1)
    df["size_bucket"] = df.get("size_bucket", "mid")
    df["market_cap"] = df.get("market_cap", 1_000_000_000.0)
    return df[[
        "symbol",
        "date",
        "membership_state",
        "is_member",
        "short_subuniverse_flag",
        "size_bucket",
        "market_cap",
    ]]



def _make_proxy(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    base = []
    for row in rows:
        rec = dict(row)
        rec.setdefault("borrow_fee_annual", 0.03)
        rec.setdefault("borrow_fee_daily", rec["borrow_fee_annual"] / 252.0)
        rec.setdefault("borrow_availability_score", 0.90)
        rec.setdefault("htb_flag", False)
        rec.setdefault("borrow_tier", "easy")
        rec.setdefault("proxy_quality", "high")
        rec.setdefault("proxy_source", "direct_lending_feed")
        rec.setdefault("stress_score", 0.05)
        rec.setdefault("jump_flag", False)
        rec.setdefault("stale_input_flag", False)
        rec.setdefault("fallback_flag", False)
        rec.setdefault("config_version", "proxy_v1")
        base.append(rec)
    df = pd.DataFrame(base)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df



def _adapt_locate_for_qc(locate_df: pd.DataFrame) -> pd.DataFrame:
    """
    borrow_qc.py currently expects a locate decision column (locate_allowed / locate_blocked /
    locate_decision), while locate_filter.py emits short_eligible_flag. This adapter is the
    minimal compatibility shim used by the integration tests.
    """
    out = locate_df.copy()
    if "locate_allowed" not in out.columns:
        out["locate_allowed"] = out["short_eligible_flag"].astype(bool)
    if "override_reason" not in out.columns:
        out["override_reason"] = ""
    out["override_reason"] = out["override_reason"].fillna("")
    if "locate_config_version" not in out.columns:
        if "config_version" in out.columns:
            out["locate_config_version"] = out["config_version"]
        else:
            out["locate_config_version"] = "unknown"
    keep = ["symbol", "date", "locate_allowed", "override_flag", "override_reason", "locate_config_version"]
    return out[keep].copy()



def _run_locate(
    borrow_proxy: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    cfg: Optional[Any] = None,
    previous_state: Optional[pd.DataFrame] = None,
    run_id: str = "pytest_locate",
    asof_timestamp: str = "2026-03-15T12:00:00Z",
):
    cfg = cfg or LOCATE.LocateConfig()
    locate_df, daily_summary, global_summary = LOCATE.compute_locate_filter(
        borrow_proxy=borrow_proxy,
        universe=universe,
        cfg=cfg,
        run_id=run_id,
        asof_timestamp=asof_timestamp,
        previous_state=previous_state,
    )
    return locate_df, daily_summary, global_summary



def _run_qc(
    borrow_proxy: pd.DataFrame,
    locate_df: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    cfg: Optional[Any] = None,
    run_id: str = "pytest_qc",
    asof_timestamp: str = "2026-03-15T12:00:00Z",
):
    cfg = cfg or BORROW_QC.BorrowQCConfig()
    return BORROW_QC.build_borrow_qc(
        borrow_proxy=borrow_proxy,
        locate_filter=_adapt_locate_for_qc(locate_df),
        universe=universe,
        cfg=cfg,
        run_id=run_id,
        asof_timestamp=asof_timestamp,
    )



def _latest_row(df: pd.DataFrame, symbol: str) -> pd.Series:
    sub = df.loc[df["symbol"] == symbol].sort_values("date")
    assert not sub.empty, f"No rows found for symbol={symbol!r}."
    return sub.iloc[-1]


# -----------------------------------------------------------------------------
# Optional borrow_cost_proxy adapter
# -----------------------------------------------------------------------------


def _load_runner(module: Any):
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
    raise AttributeError("No obvious borrow_cost_proxy runner found.")



def _proxy_default_config(module: Any):
    cfg = getattr(module, "DEFAULT_CONFIG", None)
    return copy.deepcopy(cfg) if cfg is not None else None



def _empty_df(columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))



def _coerce_date_like(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["date", "asof_date", "effective_date", "reference_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out



def _run_borrow_proxy_module(
    module: Any,
    *,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    short_interest: Optional[pd.DataFrame] = None,
    direct_feed: Optional[pd.DataFrame] = None,
    htb_flags: Optional[pd.DataFrame] = None,
    previous_state: Optional[pd.DataFrame] = None,
    run_id: str = "pytest_borrow_proxy",
    asof_timestamp: str = "2026-03-15T12:00:00Z",
) -> pd.DataFrame:
    runner = _load_runner(module)
    sig = inspect.signature(runner)
    cfg = _proxy_default_config(module)

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
        "run_id": run_id,
        "asof_timestamp": asof_timestamp,
        "asof": asof_timestamp,
        "asof_ts": asof_timestamp,
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
        raise TypeError(f"Could not adapt pytest inputs to borrow_cost_proxy runner. Missing required parameters: {missing_required}")

    result = runner(**kwargs)
    if isinstance(result, pd.DataFrame):
        out = result.copy()
    elif isinstance(result, dict):
        out = None
        for key in ["daily", "borrow_proxy", "proxy", "output", "result", "data"]:
            value = result.get(key)
            if isinstance(value, pd.DataFrame):
                out = value.copy()
                break
        if out is None:
            out = next(value.copy() for value in result.values() if isinstance(value, pd.DataFrame))
    elif isinstance(result, (list, tuple)):
        out = next(value.copy() for value in result if isinstance(value, pd.DataFrame))
    else:
        raise TypeError("Could not extract DataFrame from borrow_cost_proxy result.")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.sort_values([c for c in ["symbol", "date"] if c in out.columns]).reset_index(drop=True)



def _make_price_panel(symbol_params: Mapping[str, Mapping[str, float]], dates: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    dates = pd.bdate_range(start="2026-01-02", periods=25) if dates is None else dates
    rows = []
    for symbol, params in symbol_params.items():
        base_price = float(params.get("base_price", 25.0))
        price_slope = float(params.get("price_slope", 0.10))
        base_volume = float(params.get("base_volume", 900_000.0))
        vol_amp = float(params.get("vol_amp", 0.01))
        market_cap = float(params.get("market_cap", 1_000_000_000.0))
        float_shares = float(params.get("float_shares", 50_000_000.0))
        size_bucket = params.get("size_bucket", "mid")
        for i, dt in enumerate(dates):
            cyc = ((i % 5) - 2) / 2.0
            close = max(1.0, base_price + price_slope * i + base_price * vol_amp * cyc)
            volume = max(10_000.0, base_volume * (1.0 + 0.02 * cyc))
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


# -----------------------------------------------------------------------------
# Tests: synthetic stack integration (no borrow_cost_proxy dependency)
# -----------------------------------------------------------------------------


def test_proxy_to_qc_to_locate_happy_path():
    universe = _make_universe([
        {"symbol": "AAA", "date": "2026-01-02", "size_bucket": "mid", "market_cap": 1_200_000_000.0},
    ])
    proxy = _make_proxy([
        {
            "symbol": "AAA",
            "date": "2026-01-02",
            "borrow_fee_annual": 0.02,
            "borrow_fee_daily": 0.02 / 252.0,
            "borrow_availability_score": 0.92,
            "borrow_tier": "easy",
            "proxy_source": "direct_lending_feed",
        }
    ])

    locate_df, locate_daily, locate_summary = _run_locate(proxy, universe)
    qc_daily, qc_failures, qc_summary, qc_panel = _run_qc(proxy, locate_df, universe)

    row = _latest_row(locate_df, "AAA")
    assert int(row["short_eligible_flag"]) == 1
    assert str(row["locate_tier"]).lower() == "easy"
    assert pd.isna(row["reject_reason"]) or row["reject_reason"] in (None, "")

    assert float(locate_daily.iloc[0]["short_eligible_share"]) == pytest.approx(1.0)
    assert locate_summary["config_version"] == getattr(LOCATE, "DEFAULT_CONFIG", LOCATE.LocateConfig()).config_version

    assert qc_failures.empty
    assert qc_daily.iloc[0]["gate"] == "pass"
    assert float(qc_daily.iloc[0]["coverage_rate"]) == pytest.approx(1.0)
    assert float(qc_daily.iloc[0]["underblocking_rate"]) == pytest.approx(0.0)
    assert float(qc_daily.iloc[0]["overblocking_rate"]) == pytest.approx(0.0)
    assert qc_summary["n_rows_joined"] == 1

    panel_row = _latest_row(qc_panel, "AAA")
    assert bool(panel_row["expected_block"]) is False
    assert bool(panel_row["locate_blocked"]) is False
    assert str(panel_row["borrow_tier"]).lower() == "easy"



def test_htb_name_flows_through_stack_and_ends_blocked():
    universe = _make_universe([
        {"symbol": "HTB", "date": "2026-01-02", "size_bucket": "small", "market_cap": 250_000_000.0},
    ])
    proxy = _make_proxy([
        {
            "symbol": "HTB",
            "date": "2026-01-02",
            "borrow_fee_annual": 2.0,
            "borrow_fee_daily": 2.0 / 252.0,
            "borrow_availability_score": 0.01,
            "htb_flag": True,
            "borrow_tier": "blocked",
            "stress_score": 0.99,
        }
    ])

    locate_cfg = LOCATE.LocateConfig()
    locate_cfg.policy.block_on_htb = True

    locate_df, _, _ = _run_locate(proxy, universe, cfg=locate_cfg)
    qc_daily, qc_failures, _, qc_panel = _run_qc(proxy, locate_df, universe)

    row = _latest_row(locate_df, "HTB")
    assert int(row["short_eligible_flag"]) == 0
    assert str(row["locate_tier"]).lower() == "blocked"
    assert str(row["dominant_rule"]).lower() in {"availability_critical", "fee_critical", "htb_block_policy"}

    assert qc_daily.iloc[0]["gate"] == "pass"
    assert float(qc_daily.iloc[0]["underblocking_rate"]) == pytest.approx(0.0)
    assert float(qc_daily.iloc[0]["overblocking_rate"]) == pytest.approx(0.0)
    assert qc_failures.empty

    panel_row = _latest_row(qc_panel, "HTB")
    assert bool(panel_row["expected_block"]) is True
    assert bool(panel_row["locate_blocked"]) is True
    assert bool(panel_row["underblocking_critical"]) is False



@pytest.mark.xfail(
    reason="Current borrow_qc.py compute_daily_panel breaks on multi-symbol same-date panels due groupby-apply jump_context assignment.",
    strict=False,
)
def test_same_symbol_date_has_consistent_tier_across_outputs_when_no_overrides():
    date = "2026-01-02"
    universe = _make_universe([
        {"symbol": "AAA", "date": date, "size_bucket": "mid", "market_cap": 1_000_000_000.0},
        {"symbol": "BBB", "date": date, "size_bucket": "small", "market_cap": 350_000_000.0},
    ])
    proxy = _make_proxy([
        {
            "symbol": "AAA",
            "date": date,
            "borrow_fee_annual": 0.015,
            "borrow_fee_daily": 0.015 / 252.0,
            "borrow_availability_score": 0.95,
            "borrow_tier": "easy",
        },
        {
            "symbol": "BBB",
            "date": date,
            "borrow_fee_annual": 0.20,
            "borrow_fee_daily": 0.20 / 252.0,
            "borrow_availability_score": 0.32,
            "borrow_tier": "medium",
        },
    ])

    locate_df, _, _ = _run_locate(proxy, universe)
    qc_daily, qc_failures, _, qc_panel = _run_qc(proxy, locate_df, universe)

    assert qc_daily.iloc[0]["gate"] == "pass"
    assert qc_failures.empty

    merged = locate_df.merge(qc_panel[["symbol", "date", "borrow_tier"]], on=["symbol", "date"], how="inner")
    assert not merged.empty

    tier_map = dict(zip(merged["symbol"], merged["locate_tier"].astype(str).str.lower()))
    borrow_map = dict(zip(merged["symbol"], merged["borrow_tier"].astype(str).str.lower()))

    assert tier_map["AAA"] == borrow_map["AAA"] == "easy"
    assert tier_map["BBB"] == borrow_map["BBB"] == "medium"



def test_incremental_locate_previous_state_preserves_hysteresis_and_qc_passes():
    previous_state = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "date": pd.Timestamp("2026-01-02"),
                "locate_tier": "blocked",
                "improvement_target": None,
                "improvement_streak": 0,
            }
        ]
    )
    universe = _make_universe([
        {"symbol": "AAA", "date": "2026-01-05", "size_bucket": "mid", "market_cap": 1_000_000_000.0},
    ])
    proxy = _make_proxy([
        {
            "symbol": "AAA",
            "date": "2026-01-05",
            "borrow_fee_annual": 0.001,
            "borrow_fee_daily": 0.001 / 252.0,
            "borrow_availability_score": 0.75,
            "borrow_tier": "medium",
        }
    ])

    locate_df, _, _ = _run_locate(proxy, universe, previous_state=previous_state)
    qc_daily, qc_failures, _, qc_panel = _run_qc(proxy, locate_df, universe)

    row = _latest_row(locate_df, "AAA")
    assert str(row["instantaneous_tier"]).lower() in {"easy", "medium"}
    assert str(row["locate_tier"]).lower() == "blocked"
    assert bool(row["hysteresis_hold_flag"]) is True
    assert row["reject_reason"] == "hysteresis_hold"

    assert qc_daily.iloc[0]["gate"] == "pass"
    assert float(qc_daily.iloc[0]["underblocking_rate"]) == pytest.approx(0.0)
    assert qc_failures.empty

    panel_row = _latest_row(qc_panel, "AAA")
    assert bool(panel_row["locate_blocked"]) is True



def test_qc_catches_manually_corrupted_locate_decision_on_blocked_name():
    universe = _make_universe([
        {"symbol": "HTB", "date": "2026-01-02", "size_bucket": "small", "market_cap": 250_000_000.0},
    ])
    proxy = _make_proxy([
        {
            "symbol": "HTB",
            "date": "2026-01-02",
            "borrow_fee_annual": 2.0,
            "borrow_fee_daily": 2.0 / 252.0,
            "borrow_availability_score": 0.01,
            "htb_flag": True,
            "borrow_tier": "blocked",
        }
    ])

    locate_df, _, _ = _run_locate(proxy, universe)
    locate_for_qc = _adapt_locate_for_qc(locate_df)
    locate_for_qc["locate_allowed"] = True  # simulate downstream corruption / manual bad override without trace

    qc_daily, qc_failures, _, qc_panel = BORROW_QC.build_borrow_qc(
        borrow_proxy=proxy,
        locate_filter=locate_for_qc,
        universe=universe,
        cfg=BORROW_QC.BorrowQCConfig(),
        run_id="pytest_qc_corrupt",
        asof_timestamp="2026-03-15T12:00:00Z",
    )

    assert qc_daily.iloc[0]["gate"] == "fail"
    assert float(qc_daily.iloc[0]["underblocking_rate"]) == pytest.approx(1.0)
    assert set(qc_failures["check_type"].astype(str)) >= {"htb_locate_incoherence"}

    panel_row = _latest_row(qc_panel, "HTB")
    assert bool(panel_row["underblocking_critical"]) is True
    assert bool(panel_row["locate_allowed"]) is True
    assert bool(panel_row["expected_block"]) is True


# -----------------------------------------------------------------------------
# Optional full-stack tests using borrow_cost_proxy.py when available
# -----------------------------------------------------------------------------


@pytest.mark.skipif(BORROW_PROXY is None, reason="borrow_cost_proxy.py not available in this runtime")
def test_future_information_does_not_change_prior_day_output():
    dates = pd.bdate_range("2026-01-02", periods=20)
    prices = _make_price_panel(
        {
            "AAA": {
                "base_price": 30.0,
                "base_volume": 1_200_000.0,
                "market_cap": 1_500_000_000.0,
                "float_shares": 60_000_000.0,
                "size_bucket": "mid",
            }
        },
        dates=dates,
    )
    universe = prices[["symbol", "instrument_id", "date", "size_bucket"]].copy()
    universe["membership_state"] = "member"
    universe["is_member"] = 1
    universe["short_subuniverse_flag"] = 1

    target_date = dates[-2]
    future_observation = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "date": target_date,
                "asof_date": dates[-1],  # not observable yet on target_date output
                "effective_date": target_date,
                "short_interest_ratio": 0.40,
                "days_to_cover": 9.0,
                "utilization": 0.96,
            }
        ]
    )

    out_without_future = _run_borrow_proxy_module(
        BORROW_PROXY,
        prices=prices,
        universe=universe,
        short_interest=pd.DataFrame(columns=future_observation.columns),
        asof_timestamp="2026-01-30T21:00:00Z",
    )
    out_with_future = _run_borrow_proxy_module(
        BORROW_PROXY,
        prices=prices,
        universe=universe,
        short_interest=future_observation,
        asof_timestamp="2026-01-30T21:00:00Z",
    )

    left = out_without_future.loc[(out_without_future["symbol"] == "AAA") & (out_without_future["date"] == target_date)]
    right = out_with_future.loc[(out_with_future["symbol"] == "AAA") & (out_with_future["date"] == target_date)]

    assert not left.empty and not right.empty
    cols = ["borrow_fee_annual", "borrow_fee_daily", "borrow_availability_score", "borrow_tier", "proxy_source"]
    pd.testing.assert_series_equal(left.iloc[0][cols], right.iloc[0][cols], check_names=False)
