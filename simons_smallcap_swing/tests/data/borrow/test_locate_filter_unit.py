from __future__ import annotations

import copy
import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import pytest


REQUIRED_OUTPUT_COLUMNS = {
    "date",
    "symbol",
    "run_id",
    "short_eligible_flag",
    "locate_tier",
    "locate_tier_code",
    "reject_reason",
    "dominant_rule",
    "all_triggered_rules",
    "borrow_fee_daily",
    "borrow_fee_annual",
    "availability_score",
    "htb_flag",
    "proxy_quality",
    "fallback_flag",
    "override_flag",
    "data_quality_flag",
    "locate_config_version",
    "upstream_proxy_version",
    "asof_timestamp",
    "instantaneous_tier",
    "tier_changed_flag",
    "hysteresis_hold_flag",
}

TIER_ORDER = {"easy": 0, "medium": 1, "hard": 2, "blocked": 3}


# -----------------------------------------------------------------------------
# Import adapter
# -----------------------------------------------------------------------------


def _load_locate_filter_module():
    module_names = [
        "simons_smallcap_swing.data.borrow.locate_filter",
        "data.borrow.locate_filter",
        "locate_filter",
    ]
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue

    here = Path(__file__).resolve()
    candidate_paths = [
        here.parents[3] / "simons_smallcap_swing" / "data" / "borrow" / "locate_filter.py",
        here.parents[2] / "data" / "borrow" / "locate_filter.py",
        here.parents[1] / "locate_filter.py",
    ]
    for path in candidate_paths:
        if path.exists():
            spec = importlib.util.spec_from_file_location("locate_filter", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module

    raise ModuleNotFoundError(
        "Could not import locate_filter.py. Expected it at "
        "simons_smallcap_swing.data.borrow.locate_filter or in a nearby repo path."
    )


@pytest.fixture(scope="session")
def locate_module():
    return _load_locate_filter_module()


# -----------------------------------------------------------------------------
# Synthetic builders
# -----------------------------------------------------------------------------


def _business_days(n: int = 3, start: str = "2025-03-03") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)



def _make_universe(symbols: list[str], dates: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    dates = _business_days() if dates is None else dates
    rows = []
    for dt in dates:
        for i, symbol in enumerate(symbols):
            rows.append(
                {
                    "date": pd.Timestamp(dt),
                    "symbol": symbol,
                    "membership_state": "member",
                    "is_member": 1,
                    "size_bucket": "large" if i == 0 else "mid",
                    "market_cap": 5_000_000_000.0 if i == 0 else 1_000_000_000.0,
                    "short_subuniverse_flag": 1,
                }
            )
    return pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)



def _borrow_row(
    symbol: str,
    date: Any,
    *,
    borrow_fee_annual: float = 0.03,
    borrow_fee_daily: float = 0.00012,
    borrow_availability_score: float = 0.95,
    htb_flag: Any = False,
    borrow_tier: str = "easy",
    proxy_quality: str = "high",
    proxy_source: str = "direct_lending_feed",
    stress_score: float = 0.10,
    fallback_flag: Any = False,
    stale_input_flag: Any = False,
    config_version: str = "borrow_proxy_v1",
) -> Dict[str, Any]:
    return {
        "date": pd.Timestamp(date),
        "symbol": symbol,
        "borrow_fee_annual": borrow_fee_annual,
        "borrow_fee_daily": borrow_fee_daily,
        "borrow_availability_score": borrow_availability_score,
        "htb_flag": htb_flag,
        "borrow_tier": borrow_tier,
        "proxy_quality": proxy_quality,
        "proxy_source": proxy_source,
        "stress_score": stress_score,
        "fallback_flag": fallback_flag,
        "stale_input_flag": stale_input_flag,
        "config_version": config_version,
        "asof_timestamp": "2025-03-07T21:00:00Z",
    }



def _make_borrow_proxy(rows: list[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([dict(r) for r in rows]).sort_values(["date", "symbol"]).reset_index(drop=True)



def _make_previous_state(rows: list[Mapping[str, Any]]) -> pd.DataFrame:
    out = pd.DataFrame([dict(r) for r in rows])
    if out.empty:
        return pd.DataFrame(columns=["symbol", "date", "locate_tier", "improvement_target", "improvement_streak"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Invocation adapter
# -----------------------------------------------------------------------------


def _default_config(module: Any):
    cfg = getattr(module, "DEFAULT_CONFIG", None)
    if cfg is not None:
        return copy.deepcopy(cfg)
    cfg_type = getattr(module, "LocateConfig", None)
    if cfg_type is not None:
        return cfg_type()
    return None



def _coerce_date_like(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    out = df.copy()
    for col in ["date", "asof_timestamp"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out



def _find_runner(module: Any):
    candidate_names = [
        "compute_locate_filter",
        "run_locate_filter",
        "build_locate_filter",
        "construct_locate_filter",
        "run",
        "build",
        "compute",
    ]
    for name in candidate_names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    for name, obj in vars(module).items():
        if callable(obj) and "locate" in name.lower() and "filter" in name.lower():
            return obj
    raise AttributeError("No obvious locate_filter runner found.")



def _extract_primary_dataframe(result: Any) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, dict):
        for key in ["locate", "daily", "output", "outputs", "result", "data"]:
            value = result.get(key)
            if isinstance(value, pd.DataFrame):
                return value
        for value in result.values():
            if isinstance(value, pd.DataFrame) and REQUIRED_OUTPUT_COLUMNS.issubset(set(value.columns)):
                return value
    if isinstance(result, (tuple, list)):
        for value in result:
            if isinstance(value, pd.DataFrame) and REQUIRED_OUTPUT_COLUMNS.issubset(set(value.columns)):
                return value
        for value in result:
            if isinstance(value, pd.DataFrame):
                return value
    raise TypeError("Could not extract locate DataFrame from locate_filter return value.")



def _run_locate(
    module: Any,
    *,
    borrow_proxy: pd.DataFrame,
    universe: pd.DataFrame,
    previous_state: Optional[pd.DataFrame] = None,
    config: Any = None,
    run_id: str = "pytest_locate_filter",
    asof_timestamp: str = "2025-03-07T21:00:00Z",
) -> pd.DataFrame:
    runner = _find_runner(module)
    sig = inspect.signature(runner)
    cfg = _default_config(module) if config is None else config

    inputs = {
        "borrow_proxy": _coerce_date_like(borrow_proxy),
        "borrow_proxy_df": _coerce_date_like(borrow_proxy),
        "proxy": _coerce_date_like(borrow_proxy),
        "proxy_df": _coerce_date_like(borrow_proxy),
        "universe": _coerce_date_like(universe),
        "universe_df": _coerce_date_like(universe),
        "previous_state": _coerce_date_like(previous_state if previous_state is not None else _make_previous_state([])),
        "prev_state": _coerce_date_like(previous_state if previous_state is not None else _make_previous_state([])),
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
        raise TypeError(
            "Could not adapt pytest inputs to locate_filter runner. "
            f"Missing required parameters: {missing_required}."
        )

    result = runner(**kwargs)
    out = _extract_primary_dataframe(result).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Assertion helpers
# -----------------------------------------------------------------------------


def _latest_row(df: pd.DataFrame, symbol: str) -> pd.Series:
    sub = df.loc[df["symbol"] == symbol].sort_values("date")
    assert not sub.empty, f"No rows found for symbol={symbol!r}"
    return sub.iloc[-1]



def _assert_contract(df: pd.DataFrame):
    missing = REQUIRED_OUTPUT_COLUMNS - set(df.columns)
    assert not missing, f"Missing required locate output columns: {sorted(missing)}"
    assert not df.empty, "Locate output is unexpectedly empty."
    assert df[["date", "symbol"]].drop_duplicates().shape[0] == len(df), "Output must be unique on (date, symbol)."
    tiers = df["locate_tier"].astype(str).str.lower()
    assert set(tiers.unique()).issubset(set(TIER_ORDER)), "Unexpected locate_tier values found."
    codes = pd.to_numeric(df["locate_tier_code"], errors="coerce")
    mapped_codes = tiers.map(TIER_ORDER)
    assert (codes == mapped_codes).all(), "locate_tier_code must match locate_tier."
    blocked = df[tiers.eq("blocked")]
    if not blocked.empty:
        assert (blocked["short_eligible_flag"] == 0).all(), "Blocked names cannot be short-eligible."
        assert blocked["reject_reason"].notna().all(), "Blocked names must carry reject_reason."


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def base_dates() -> pd.DatetimeIndex:
    return _business_days(3)


@pytest.fixture
def base_universe(base_dates: pd.DatetimeIndex) -> pd.DataFrame:
    return _make_universe(["AAA", "BBB"], dates=base_dates)


@pytest.fixture
def base_borrow_proxy(base_dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dt in base_dates:
        rows.append(_borrow_row("AAA", dt, borrow_fee_daily=0.00010, borrow_availability_score=0.97, proxy_quality="high"))
        rows.append(_borrow_row("BBB", dt, borrow_fee_daily=0.00045, borrow_availability_score=0.55, borrow_tier="medium", proxy_quality="high", proxy_source="short_interest_derived"))
    return _make_borrow_proxy(rows)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_output_contract_basic_columns_and_consistency(locate_module, base_borrow_proxy, base_universe):
    out = _run_locate(
        locate_module,
        borrow_proxy=base_borrow_proxy,
        universe=base_universe,
        run_id="pytest_locate_contract",
    )
    _assert_contract(out)
    assert set(out["short_eligible_flag"].unique()).issubset({0, 1})
    assert out["asof_timestamp"].astype(str).str.len().gt(0).all()



def test_explicit_blocked_name_is_not_short_eligible_via_manual_blocklist(locate_module, base_dates, base_universe):
    borrow = _make_borrow_proxy(
        [_borrow_row("AAA", dt, borrow_fee_daily=0.00005, borrow_availability_score=0.99) for dt in base_dates]
    )
    cfg = _default_config(locate_module)
    cfg.manual_blocklist = ["AAA"]

    out = _run_locate(
        locate_module,
        borrow_proxy=borrow,
        universe=base_universe.loc[base_universe["symbol"].eq("AAA")],
        config=cfg,
        run_id="pytest_manual_block",
    )
    row = _latest_row(out, "AAA")

    assert row["short_eligible_flag"] == 0
    assert str(row["locate_tier"]).lower() == "blocked"
    assert row["reject_reason"] == "explicit_block_override"
    assert row["dominant_rule"] == "explicit_block_override"
    assert row["override_flag"] is True or bool(row["override_flag"]) is True



def test_availability_critical_forces_rejection_even_with_allow_override(locate_module, base_dates, base_universe):
    borrow = _make_borrow_proxy(
        [_borrow_row("AAA", dt, borrow_fee_daily=0.00001, borrow_availability_score=0.03, proxy_quality="high") for dt in base_dates]
    )
    cfg = _default_config(locate_module)
    cfg.allow_override_symbols = ["AAA"]

    out = _run_locate(
        locate_module,
        borrow_proxy=borrow,
        universe=base_universe.loc[base_universe["symbol"].eq("AAA")],
        config=cfg,
        run_id="pytest_availability_critical",
    )
    row = _latest_row(out, "AAA")

    assert row["short_eligible_flag"] == 0
    assert str(row["locate_tier"]).lower() == "blocked"
    assert row["dominant_rule"] == "availability_critical"
    assert row["reject_reason"] == "availability_critical"
    assert bool(row["override_flag"]) is True  # attempted override remains visible, but cannot unblock critical availability



def test_allow_override_can_unblock_fee_critical_when_not_structural_or_availability(locate_module, base_dates, base_universe):
    borrow = _make_borrow_proxy(
        [_borrow_row("AAA", dt, borrow_fee_daily=0.02000, borrow_fee_annual=4.0, borrow_availability_score=0.60, proxy_quality="high") for dt in base_dates]
    )
    cfg = _default_config(locate_module)
    cfg.allow_override_symbols = ["AAA"]

    out = _run_locate(
        locate_module,
        borrow_proxy=borrow,
        universe=base_universe.loc[base_universe["symbol"].eq("AAA")],
        config=cfg,
        run_id="pytest_allow_override",
    )
    row = _latest_row(out, "AAA")

    assert row["short_eligible_flag"] == 1
    assert str(row["locate_tier"]).lower() in {"medium", "hard", "easy"}
    assert row["override_flag"] is True or bool(row["override_flag"]) is True
    assert row["reject_reason"] in {None, "tier_mapping_soft"}
    assert "fee_critical" not in str(row["all_triggered_rules"])



def test_improvement_is_delayed_by_hysteresis_hold(locate_module, base_dates, base_universe):
    borrow = _make_borrow_proxy(
        [_borrow_row("AAA", dt, borrow_fee_daily=0.00120, borrow_fee_annual=0.35, borrow_availability_score=0.25, htb_flag=True, borrow_tier="hard") for dt in base_dates]
    )
    cfg = _default_config(locate_module)
    prev = _make_previous_state(
        [
            {
                "symbol": "AAA",
                "date": base_dates[-2],
                "locate_tier": "blocked",
                "improvement_target": "hard",
                "improvement_streak": 1,
            }
        ]
    )

    out = _run_locate(
        locate_module,
        borrow_proxy=borrow,
        universe=base_universe.loc[base_universe["symbol"].eq("AAA")],
        previous_state=prev,
        config=cfg,
        run_id="pytest_hysteresis_hold",
    )
    first_row = out.loc[out["symbol"].eq("AAA")].sort_values("date").iloc[0]

    assert first_row["instantaneous_tier"] == "hard"
    assert first_row["locate_tier"] == "blocked"
    assert first_row["short_eligible_flag"] == 0
    assert first_row["hysteresis_hold_flag"] is True or bool(first_row["hysteresis_hold_flag"]) is True
    assert first_row["reject_reason"] == "hysteresis_hold"
    assert first_row["improvement_target"] == "hard"
    assert int(first_row["improvement_streak"]) == 2



def test_improvement_is_released_once_persistence_requirement_is_met(locate_module, base_dates, base_universe):
    borrow = _make_borrow_proxy(
        [_borrow_row("AAA", dt, borrow_fee_daily=0.00120, borrow_fee_annual=0.35, borrow_availability_score=0.25, htb_flag=True, borrow_tier="hard") for dt in base_dates]
    )
    cfg = _default_config(locate_module)
    prev = _make_previous_state(
        [
            {
                "symbol": "AAA",
                "date": base_dates[-2],
                "locate_tier": "blocked",
                "improvement_target": "hard",
                "improvement_streak": 2,
            }
        ]
    )

    out = _run_locate(
        locate_module,
        borrow_proxy=borrow,
        universe=base_universe.loc[base_universe["symbol"].eq("AAA")],
        previous_state=prev,
        config=cfg,
        run_id="pytest_hysteresis_release",
    )
    row = _latest_row(out, "AAA")

    assert row["instantaneous_tier"] == "hard"
    assert row["locate_tier"] == "hard"
    assert row["short_eligible_flag"] == 1
    assert bool(row["hysteresis_hold_flag"]) is False
    assert row["reject_reason"] == "tier_mapping_soft"
    assert pd.isna(row["improvement_target"]) or row["improvement_target"] is None
    assert int(row["improvement_streak"]) == 0



def test_deterioration_is_applied_immediately_without_hysteresis_hold(locate_module, base_dates, base_universe):
    borrow = _make_borrow_proxy(
        [_borrow_row("AAA", dt, borrow_fee_daily=0.00800, borrow_fee_annual=1.8, borrow_availability_score=0.04, proxy_quality="high") for dt in base_dates]
    )
    prev = _make_previous_state(
        [{"symbol": "AAA", "date": base_dates[-2], "locate_tier": "easy", "improvement_target": None, "improvement_streak": 0}]
    )

    out = _run_locate(
        locate_module,
        borrow_proxy=borrow,
        universe=base_universe.loc[base_universe["symbol"].eq("AAA")],
        previous_state=prev,
        run_id="pytest_deterioration_immediate",
    )
    first_row = out.loc[out["symbol"].eq("AAA")].sort_values("date").iloc[0]

    assert first_row["instantaneous_tier"] == "blocked"
    assert first_row["locate_tier"] == "blocked"
    assert first_row["short_eligible_flag"] == 0
    assert bool(first_row["hysteresis_hold_flag"]) is False
    assert first_row["dominant_rule"] == "availability_critical"
    assert bool(first_row["tier_changed_flag"]) is True



def test_hard_shorts_can_be_disabled_by_policy(locate_module, base_dates, base_universe):
    borrow = _make_borrow_proxy(
        [_borrow_row("AAA", dt, borrow_fee_daily=0.00130, borrow_fee_annual=0.30, borrow_availability_score=0.22, htb_flag=True, borrow_tier="hard") for dt in base_dates]
    )
    cfg = _default_config(locate_module)
    cfg.policy.allow_hard_shorts = False

    out = _run_locate(
        locate_module,
        borrow_proxy=borrow,
        universe=base_universe.loc[base_universe["symbol"].eq("AAA")],
        config=cfg,
        run_id="pytest_hard_disabled",
    )
    row = _latest_row(out, "AAA")

    assert row["instantaneous_tier"] == "blocked"
    assert row["locate_tier"] == "blocked"
    assert row["short_eligible_flag"] == 0
    assert row["reject_reason"] == "blocked_unspecified" or row["reject_reason"] == "tier_mapping_soft"
    assert "tier_mapping_soft" in str(row["all_triggered_rules"])



def test_structural_missing_borrow_join_blocks_name(locate_module, base_dates, base_universe):
    borrow = _make_borrow_proxy([
        _borrow_row("AAA", base_dates[-1], borrow_fee_daily=0.00008, borrow_availability_score=0.98),
    ])

    out = _run_locate(
        locate_module,
        borrow_proxy=borrow,
        universe=base_universe.loc[base_universe["symbol"].eq("AAA")],
        run_id="pytest_structural_missing",
    )

    early_rows = out.sort_values("date").iloc[:-1]
    assert not early_rows.empty
    assert (early_rows["locate_tier"] == "blocked").all()
    assert (early_rows["short_eligible_flag"] == 0).all()
    assert (early_rows["dominant_rule"] == "structural_missing").all()
    assert early_rows["data_quality_flag"].astype(str).str.contains("missing_borrow_join").all()



def test_helper_max_short_weight_decreases_monotonically_and_blocked_is_zero(locate_module):
    cfg = _default_config(locate_module)
    easy = locate_module.max_short_weight_for_tier("easy", cfg)
    medium = locate_module.max_short_weight_for_tier("medium", cfg)
    hard = locate_module.max_short_weight_for_tier("hard", cfg)
    blocked = locate_module.max_short_weight_for_tier("blocked", cfg)

    assert easy > medium > hard >= blocked
    assert blocked == pytest.approx(0.0, abs=1e-15)



def test_apply_locate_sizing_respects_tier_caps(locate_module):
    cfg = _default_config(locate_module)
    df = pd.DataFrame({"locate_tier": ["easy", "medium", "hard", "blocked"]})
    out = locate_module.apply_locate_sizing(df, cfg=cfg)

    assert "max_short_weight" in out.columns
    weights = out["max_short_weight"].tolist()
    assert weights[0] > weights[1] > weights[2] >= weights[3]
    assert weights[3] == pytest.approx(0.0, abs=1e-15)



def test_compute_net_short_return_subtracts_fee_only_when_position_is_short(locate_module):
    raw = pd.Series([0.010, 0.010, -0.020, 0.005], dtype=float)
    fee = pd.Series([0.001, 0.001, 0.002, np.nan], dtype=float)
    flags = pd.Series([1, 0, True, False])

    net = locate_module.compute_net_short_return(raw, fee, flags)

    assert net.iloc[0] == pytest.approx(0.009, abs=1e-12)
    assert net.iloc[1] == pytest.approx(0.010, abs=1e-12)
    assert net.iloc[2] == pytest.approx(-0.022, abs=1e-12)
    assert net.iloc[3] == pytest.approx(0.005, abs=1e-12)
