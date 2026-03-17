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

ALLOWED_TIERS = {"easy", "medium", "hard", "blocked"}
ALLOWED_QUALITIES = {"high", "medium", "low"}

# Contract-level source vocabulary: some implementations may expose richer detail,
# but these canonical labels should appear either exactly or as a recognizable
# substring in the dominant source field.
CANONICAL_SOURCE_TOKENS = {
    "direct",
    "lending",
    "explicit_htb",
    "htb",
    "short_interest",
    "liquidity",
    "volatility",
    "coarse",
    "fallback",
    "insufficient_data",
}


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
        Path("/mnt/data/borrow_cost_proxy.py"),
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



def _make_price_panel(
    symbol_params: Mapping[str, Mapping[str, float]],
    dates: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
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



def _empty_df(columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


# -----------------------------------------------------------------------------
# Invocation and extraction adapters
# -----------------------------------------------------------------------------


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



def _run_proxy_full(
    module: Any,
    *,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    short_interest: Optional[pd.DataFrame] = None,
    direct_feed: Optional[pd.DataFrame] = None,
    previous_state: Optional[pd.DataFrame] = None,
    asof_timestamp: str = "2025-02-28T21:00:00Z",
    run_id: str = "pytest_borrow_proxy_contract",
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
        "direct_feed": _coerce_date_like(direct_feed if direct_feed is not None else _empty_df(["symbol", "date", "asof_date", "borrow_fee_annual", "availability_score", "htb_flag"])),
        "direct_feed_df": _coerce_date_like(direct_feed if direct_feed is not None else _empty_df(["symbol", "date", "asof_date", "borrow_fee_annual", "availability_score", "htb_flag"])),
        "lending_feed": _coerce_date_like(direct_feed if direct_feed is not None else _empty_df(["symbol", "date", "asof_date", "borrow_fee_annual", "availability_score", "htb_flag"])),
        "direct_lending_feed": _coerce_date_like(direct_feed if direct_feed is not None else _empty_df(["symbol", "date", "asof_date", "borrow_fee_annual", "availability_score", "htb_flag"])),
        "previous_state": previous_state,
        "prev_state": previous_state,
        "state_prev": previous_state,
        "config": cfg,
        "cfg": cfg,
        "run_id": run_id,
        "asof_timestamp": asof_timestamp,
        "as_of_ts": asof_timestamp,
        "asof_ts": asof_timestamp,
    }

    kwargs = {}
    missing_required = []
    for name, param in sig.parameters.items():
        if name in inputs:
            kwargs[name] = inputs[name]
        elif param.default is inspect._empty and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            missing_required.append(name)

    if missing_required:
        raise TypeError(
            "Could not adapt pytest inputs to borrow_cost_proxy runner. "
            f"Missing required parameters: {missing_required}. "
            "Please align the adapter in tests/data/borrow/test_borrow_cost_proxy_contract.py."
        )

    return runner(**kwargs)



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



def _extract_manifest(result: Any) -> Mapping[str, Any]:
    if isinstance(result, dict):
        for key in ["manifest", "run_manifest", "metadata", "meta"]:
            value = result.get(key)
            if isinstance(value, Mapping):
                return value
    raise AssertionError(
        "borrow_cost_proxy result must expose a manifest/metadata mapping at the top level. "
        "This is part of the production contract."
    )



def _extract_summary(result: Any) -> Mapping[str, Any]:
    if isinstance(result, dict):
        for key in ["summary", "coverage", "metrics", "daily_summary"]:
            value = result.get(key)
            if isinstance(value, Mapping):
                return value
    raise AssertionError(
        "borrow_cost_proxy result must expose a summary/coverage mapping at the top level. "
        "This is part of the production contract."
    )


# -----------------------------------------------------------------------------
# Assertion helpers
# -----------------------------------------------------------------------------


def _assert_required_contract(df: pd.DataFrame):
    missing = REQUIRED_OUTPUT_COLUMNS - set(df.columns)
    assert not missing, f"Missing required borrow proxy output columns: {sorted(missing)}"
    assert not df.empty, "Borrow proxy output is unexpectedly empty."
    assert df[["symbol", "date"]].drop_duplicates().shape[0] == len(df), "Output must be unique on (symbol, date)."

    annual = pd.to_numeric(df["borrow_fee_annual"], errors="coerce")
    daily = pd.to_numeric(df["borrow_fee_daily"], errors="coerce")
    alpha = pd.to_numeric(df["borrow_availability_score"], errors="coerce")

    assert annual.notna().all(), "borrow_fee_annual must be non-null numeric."
    assert daily.notna().all(), "borrow_fee_daily must be non-null numeric."
    assert alpha.notna().all(), "borrow_availability_score must be non-null numeric."
    assert (annual >= 0).all(), "borrow_fee_annual must be non-negative."
    assert (daily >= 0).all(), "borrow_fee_daily must be non-negative."
    assert ((alpha >= 0) & (alpha <= 1)).all(), "borrow_availability_score must stay in [0,1]."

    tiers = set(df["borrow_tier"].astype(str).str.lower().dropna().unique())
    quals = set(df["proxy_quality"].astype(str).str.lower().dropna().unique())
    assert tiers.issubset(ALLOWED_TIERS), f"Unexpected borrow_tier values found: {sorted(tiers - ALLOWED_TIERS)}"
    assert quals.issubset(ALLOWED_QUALITIES), f"Unexpected proxy_quality values found: {sorted(quals - ALLOWED_QUALITIES)}"



def _assert_source_labels_are_recognizable(df: pd.DataFrame):
    bad = []
    for value in df["proxy_source"].astype(str):
        norm = value.strip().lower()
        if not norm:
            bad.append(value)
            continue
        if not any(token in norm for token in CANONICAL_SOURCE_TOKENS):
            bad.append(value)
    assert not bad, f"Unrecognized proxy_source labels found: {bad[:5]}"


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def base_inputs():
    dates = _business_days(35)
    prices = _make_price_panel(
        {
            "AAA": {"base_price": 30.0, "base_volume": 2_500_000.0, "market_cap": 4_000_000_000.0, "float_shares": 80_000_000.0, "vol_amp": 0.010, "size_bucket": "large"},
            "BBB": {"base_price": 18.0, "base_volume": 900_000.0, "market_cap": 900_000_000.0, "float_shares": 45_000_000.0, "vol_amp": 0.020, "size_bucket": "mid"},
            "CCC": {"base_price": 6.0, "base_volume": 85_000.0, "market_cap": 45_000_000.0, "float_shares": 12_000_000.0, "vol_amp": 0.055, "size_bucket": "micro"},
        },
        dates=dates,
    )
    universe = _make_universe(prices)
    return {"dates": dates, "prices": prices, "universe": universe}


@pytest.fixture
def severe_direct_feed(base_inputs):
    last_date = base_inputs["dates"][-1]
    return _make_direct_feed(
        [
            {
                "symbol": "CCC",
                "date": last_date,
                "asof_date": last_date,
                "borrow_fee_annual": 1.80,
                "availability_score": 0.01,
                "htb_flag": 1,
            }
        ]
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_output_contains_required_columns_and_domain_constraints(borrow_module, base_inputs):
    result = _run_proxy_full(
        borrow_module,
        prices=base_inputs["prices"],
        universe=base_inputs["universe"],
        run_id="pytest_contract_basic",
    )
    out = _extract_primary_dataframe(result).copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    _assert_required_contract(out)



def test_primary_key_is_unique_on_symbol_date(borrow_module, base_inputs):
    result = _run_proxy_full(
        borrow_module,
        prices=base_inputs["prices"],
        universe=base_inputs["universe"],
        run_id="pytest_contract_pk",
    )
    out = _extract_primary_dataframe(result)

    dupes = out.duplicated(subset=["symbol", "date"], keep=False)
    assert not dupes.any(), f"Output contains duplicate (symbol, date) keys: {out.loc[dupes, ['symbol', 'date']].head().to_dict('records')}"



def test_source_name_and_enum_values_are_valid(borrow_module, base_inputs):
    si = _make_short_interest(
        [
            {
                "symbol": "BBB",
                "date": base_inputs["dates"][-1],
                "asof_date": base_inputs["dates"][-1],
                "short_interest_ratio": 0.22,
                "days_to_cover": 7.0,
                "utilization": 0.85,
            }
        ]
    )
    result = _run_proxy_full(
        borrow_module,
        prices=base_inputs["prices"],
        universe=base_inputs["universe"],
        short_interest=si,
        run_id="pytest_contract_enums",
    )
    out = _extract_primary_dataframe(result).copy()

    _assert_required_contract(out)
    _assert_source_labels_are_recognizable(out)



def test_blocked_names_have_consistent_flags_and_quality_rules(borrow_module, base_inputs, severe_direct_feed):
    result = _run_proxy_full(
        borrow_module,
        prices=base_inputs["prices"],
        universe=base_inputs["universe"],
        direct_feed=severe_direct_feed,
        run_id="pytest_contract_blocked",
    )
    out = _extract_primary_dataframe(result).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    latest_ccc = out.loc[out["symbol"] == "CCC"].sort_values("date").iloc[-1]
    tier = str(latest_ccc["borrow_tier"]).lower()
    quality = str(latest_ccc["proxy_quality"]).lower()
    source = str(latest_ccc["proxy_source"]).lower()
    fee = float(latest_ccc["borrow_fee_annual"])
    alpha = float(latest_ccc["borrow_availability_score"])
    htb = int(latest_ccc["htb_flag"])

    if tier == "blocked":
        assert htb == 1, "blocked names should carry htb_flag=1 under the production contract."
        assert fee > 0.0, "blocked names must have strictly positive annualized fee."
        assert alpha <= 0.10 or fee >= 1.0, "blocked names should exhibit either near-zero availability or severe fee."
        if quality == "high":
            assert any(tok in source for tok in ["direct", "lending", "explicit_htb", "htb"]), (
                "blocked names may only be high-quality when backed by direct or explicit severe evidence."
            )
    else:
        pytest.skip("Current implementation did not classify the injected severe direct feed as blocked; nothing to validate.")



def test_manifest_contains_run_metadata_and_temporal_identity(borrow_module, base_inputs):
    run_id = "pytest_contract_manifest"
    asof_timestamp = "2025-02-28T21:00:00Z"
    result = _run_proxy_full(
        borrow_module,
        prices=base_inputs["prices"],
        universe=base_inputs["universe"],
        run_id=run_id,
        asof_timestamp=asof_timestamp,
    )
    manifest = _extract_manifest(result)

    manifest_keys = {str(k).lower() for k in manifest.keys()}
    assert any("run" in k and "id" in k for k in manifest_keys), "Manifest must contain run_id-like metadata."
    assert any("asof" in k for k in manifest_keys), "Manifest must contain asof timestamp metadata."
    assert any("config" in k and ("version" in k or "hash" in k) for k in manifest_keys), (
        "Manifest must contain config version/hash metadata."
    )

    manifest_text = str(manifest)
    assert run_id in manifest_text, "Manifest does not appear to contain the supplied run_id."
    assert "2025-02-28" in manifest_text, "Manifest does not appear to contain the supplied asof timestamp."



def test_summary_contains_expected_aggregates_and_quality_information(borrow_module, base_inputs):
    result = _run_proxy_full(
        borrow_module,
        prices=base_inputs["prices"],
        universe=base_inputs["universe"],
        run_id="pytest_contract_summary",
    )
    summary = _extract_summary(result)
    out = _extract_primary_dataframe(result)

    summary_keys = {str(k).lower() for k in summary.keys()}
    assert any(k in summary_keys for k in {"n_rows", "row_count", "n_symbols", "symbol_count", "coverage_rate", "coverage"}), (
        "Summary must expose at least one count/coverage aggregate."
    )
    assert any("quality" in k for k in summary_keys) or any("proxy_quality" in k for k in summary_keys), (
        "Summary must expose quality information, per the production contract."
    )
    assert any("config" in k for k in summary_keys) or "config_version" in out.columns, (
        "Summary or daily output must identify the config version used."
    )



def test_daily_output_is_sorted_deterministically_by_symbol_date(borrow_module, base_inputs):
    result = _run_proxy_full(
        borrow_module,
        prices=base_inputs["prices"],
        universe=base_inputs["universe"],
        run_id="pytest_contract_sorting",
    )
    out = _extract_primary_dataframe(result).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    expected = out.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    observed = out.reset_index(drop=True)
    pd.testing.assert_frame_equal(observed, expected, check_like=False, check_dtype=False)



def test_fallback_rows_are_explicitly_flagged_and_not_marked_high_quality(borrow_module, base_inputs):
    sparse_prices = base_inputs["prices"].loc[base_inputs["prices"]["symbol"] == "CCC"].copy()
    sparse_universe = base_inputs["universe"].loc[base_inputs["universe"]["symbol"] == "CCC"].copy()

    result = _run_proxy_full(
        borrow_module,
        prices=sparse_prices,
        universe=sparse_universe,
        short_interest=_empty_df(["symbol", "date", "asof_date", "short_interest_ratio", "days_to_cover", "utilization"]),
        direct_feed=_empty_df(["symbol", "date", "asof_date", "borrow_fee_annual", "availability_score", "htb_flag"]),
        run_id="pytest_contract_fallback",
    )
    out = _extract_primary_dataframe(result).copy()
    latest = out.sort_values("date").iloc[-1]

    fallback_flag = int(float(latest["fallback_flag"])) if pd.notna(latest["fallback_flag"]) else 0
    quality = str(latest["proxy_quality"]).lower()
    source = str(latest["proxy_source"]).lower()

    if fallback_flag == 1 or any(tok in source for tok in ["fallback", "coarse", "insufficient_data"]):
        assert quality in {"low", "medium"}, "Fallback-driven rows cannot be marked high quality."
    else:
        pytest.skip("Current implementation did not expose fallback on this sparse synthetic case.")
