from __future__ import annotations

import copy
import importlib
import importlib.util
import inspect
import sys
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

ALLOWED_TIERS = {"easy", "medium", "hard", "blocked"}
TIER_TO_CODE = {"easy": 0, "medium": 1, "hard": 2, "blocked": 3}

EXPECTED_DAILY_SUMMARY_COLUMNS = {
    "date",
    "universe_size",
    "short_eligible_share",
    "tier_easy_share",
    "tier_medium_share",
    "tier_hard_share",
    "tier_blocked_share",
    "avg_borrow_fee_daily",
    "avg_availability_score",
    "locate_config_version",
}

EXPECTED_SUMMARY_KEYS = {
    "n_rows",
    "n_dates",
    "n_symbols",
    "mean_short_eligible_share",
    "mean_blocked_share",
    "mean_hard_share",
    "blocked_reason_distribution",
    "tier_distribution",
    "override_share",
    "low_quality_share",
    "fallback_share",
    "hysteresis_hold_share",
    "config_version",
}


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
        Path("/mnt/data/locate_filter.py"),
    ]
    for path in candidate_paths:
        if path.exists():
            spec = importlib.util.spec_from_file_location("locate_filter", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
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
# Invocation adapters
# -----------------------------------------------------------------------------


def _default_config(module: Any):
    cfg = getattr(module, "DEFAULT_CONFIG", None)
    if cfg is not None:
        return copy.deepcopy(cfg)
    cfg_type = getattr(module, "LocateConfig", None)
    if cfg_type is not None:
        return cfg_type()
    return None



def _find_compute_runner(module: Any):
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



def _coerce_date_like(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    out = df.copy()
    for col in ["date", "asof_timestamp"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out



def _run_locate_full(
    module: Any,
    *,
    borrow_proxy: pd.DataFrame,
    universe: pd.DataFrame,
    previous_state: Optional[pd.DataFrame] = None,
    config: Any = None,
    run_id: str = "pytest_locate_contract",
    asof_timestamp: str = "2025-03-07T21:00:00Z",
):
    runner = _find_compute_runner(module)
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
        "as_of_ts": asof_timestamp,
        "asof_ts": asof_timestamp,
    }

    kwargs = {}
    for name, param in sig.parameters.items():
        if name in inputs:
            kwargs[name] = inputs[name]
        elif param.default is inspect._empty:
            raise TypeError(f"Runner requires unsupported parameter: {name}")

    result = runner(**kwargs)

    if isinstance(result, tuple):
        # Preferred contract: (locate, daily_summary, summary) or (locate, daily_summary, summary, manifest)
        if len(result) >= 3 and isinstance(result[0], pd.DataFrame) and isinstance(result[1], pd.DataFrame) and isinstance(result[2], dict):
            locate = result[0]
            daily_summary = result[1]
            summary = result[2]
            manifest = result[3] if len(result) >= 4 and isinstance(result[3], dict) else None
            return locate, daily_summary, summary, manifest, cfg

    if isinstance(result, dict):
        locate = None
        daily_summary = None
        summary = None
        manifest = None
        for key in ["locate", "daily", "output", "outputs", "result", "data"]:
            if isinstance(result.get(key), pd.DataFrame):
                locate = result[key]
                break
        for key in ["daily_summary", "summary_daily"]:
            if isinstance(result.get(key), pd.DataFrame):
                daily_summary = result[key]
                break
        for key in ["summary", "aggregate_summary", "global_summary"]:
            if isinstance(result.get(key), dict):
                summary = result[key]
                break
        if isinstance(result.get("manifest"), dict):
            manifest = result["manifest"]
        if locate is not None and daily_summary is not None and summary is not None:
            return locate, daily_summary, summary, manifest, cfg

    raise TypeError("Could not extract locate / daily_summary / summary from locate_filter return value.")



def _build_manifest(module: Any, *, locate: pd.DataFrame, daily_summary: pd.DataFrame, summary: Mapping[str, Any], cfg: Any):
    manifest_fn = getattr(module, "manifest_payload", None)
    if callable(manifest_fn):
        return manifest_fn(
            borrow_proxy_path=Path("/tmp/borrow_proxy.parquet"),
            universe_path=Path("/tmp/universe.parquet"),
            previous_state_path=None,
            config_path=None,
            run_id="pytest_locate_contract",
            asof_timestamp="2025-03-07T21:00:00Z",
            cfg=cfg,
            locate=locate,
            daily_summary=daily_summary,
            summary=summary,
        )
    return None


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture()
def locate_contract_bundle(locate_module):
    dates = _business_days(3)
    universe = _make_universe(["AAA", "BBB", "CCC"], dates=dates)
    borrow_proxy = _make_borrow_proxy(
        [
            _borrow_row("AAA", dates[0], borrow_fee_annual=0.020, borrow_fee_daily=0.00008, borrow_availability_score=0.96, borrow_tier="easy"),
            _borrow_row("BBB", dates[0], borrow_fee_annual=0.120, borrow_fee_daily=0.00050, borrow_availability_score=0.55, borrow_tier="medium", proxy_quality="medium"),
            _borrow_row("CCC", dates[0], borrow_fee_annual=2.000, borrow_fee_daily=0.00800, borrow_availability_score=0.03, borrow_tier="blocked", htb_flag=True, fallback_flag=True, proxy_quality="low", proxy_source="coarse_fallback"),
            _borrow_row("AAA", dates[1], borrow_fee_annual=0.022, borrow_fee_daily=0.00009, borrow_availability_score=0.95, borrow_tier="easy"),
            _borrow_row("BBB", dates[1], borrow_fee_annual=0.135, borrow_fee_daily=0.00055, borrow_availability_score=0.52, borrow_tier="medium", proxy_quality="medium"),
            _borrow_row("CCC", dates[1], borrow_fee_annual=2.000, borrow_fee_daily=0.00850, borrow_availability_score=0.03, borrow_tier="blocked", htb_flag=True, fallback_flag=True, proxy_quality="low", proxy_source="coarse_fallback"),
            _borrow_row("AAA", dates[2], borrow_fee_annual=0.021, borrow_fee_daily=0.00008, borrow_availability_score=0.97, borrow_tier="easy"),
            _borrow_row("BBB", dates[2], borrow_fee_annual=0.140, borrow_fee_daily=0.00057, borrow_availability_score=0.50, borrow_tier="medium", proxy_quality="medium"),
            _borrow_row("CCC", dates[2], borrow_fee_annual=2.000, borrow_fee_daily=0.00900, borrow_availability_score=0.02, borrow_tier="blocked", htb_flag=True, fallback_flag=True, proxy_quality="low", proxy_source="coarse_fallback"),
        ]
    )
    locate, daily_summary, summary, manifest, cfg = _run_locate_full(
        locate_module,
        borrow_proxy=borrow_proxy,
        universe=universe,
        previous_state=_make_previous_state([]),
    )
    if manifest is None:
        manifest = _build_manifest(locate_module, locate=locate, daily_summary=daily_summary, summary=summary, cfg=cfg)
    return {
        "locate": locate,
        "daily_summary": daily_summary,
        "summary": summary,
        "manifest": manifest,
        "cfg": cfg,
    }


# -----------------------------------------------------------------------------
# Contract tests
# -----------------------------------------------------------------------------


def test_locate_output_contains_required_columns(locate_contract_bundle):
    locate = locate_contract_bundle["locate"]
    assert REQUIRED_OUTPUT_COLUMNS.issubset(set(locate.columns))
    assert not bool(locate[list(REQUIRED_OUTPUT_COLUMNS)].isna().all(axis=None))



def test_reject_reason_is_null_when_name_is_easy_and_eligible(locate_contract_bundle):
    locate = locate_contract_bundle["locate"]
    easy = locate[(locate["symbol"] == "AAA") & (locate["locate_tier"] == "easy")]
    assert not easy.empty
    assert (easy["short_eligible_flag"] == 1).all()
    assert easy["reject_reason"].isna().all()



def test_blocked_names_are_ineligible_and_have_consistent_flags(locate_contract_bundle):
    locate = locate_contract_bundle["locate"]
    blocked = locate[locate["symbol"] == "CCC"]
    assert not blocked.empty
    assert (blocked["locate_tier"] == "blocked").all()
    assert (blocked["short_eligible_flag"] == 0).all()
    assert blocked["reject_reason"].notna().all()
    assert blocked["reject_reason"].isin({"availability_critical", "explicit_block_override", "hysteresis_hold", "fee_critical", "blocked_unspecified"}).all()



def test_tier_code_matches_tier_label(locate_contract_bundle):
    locate = locate_contract_bundle["locate"]
    expected_codes = locate["locate_tier"].map(TIER_TO_CODE)
    pd.testing.assert_series_equal(
        locate["locate_tier_code"].reset_index(drop=True),
        expected_codes.reset_index(drop=True),
        check_names=False,
    )



def test_contract_domains_and_cross_field_coherence(locate_contract_bundle):
    locate = locate_contract_bundle["locate"]

    assert set(locate["locate_tier"].dropna().unique()).issubset(ALLOWED_TIERS)
    assert locate[["date", "symbol"]].duplicated().sum() == 0
    assert locate["short_eligible_flag"].dropna().isin([0, 1, False, True]).all()

    medium_or_hard = locate[locate["locate_tier"].isin(["medium", "hard"])]
    if not medium_or_hard.empty:
        assert (medium_or_hard["short_eligible_flag"] == 1).all()
        assert (medium_or_hard["reject_reason"] == "tier_mapping_soft").all()

    blocked = locate[locate["locate_tier"] == "blocked"]
    if not blocked.empty:
        assert (blocked["short_eligible_flag"] == 0).all()
        assert blocked["reject_reason"].notna().all()



def test_daily_summary_contains_expected_columns_and_is_sorted(locate_contract_bundle):
    daily = locate_contract_bundle["daily_summary"]
    assert EXPECTED_DAILY_SUMMARY_COLUMNS.issubset(set(daily.columns))
    assert daily["date"].is_monotonic_increasing
    assert (daily["universe_size"] > 0).all()

    share_cols = [c for c in daily.columns if c.endswith("_share") and not c.startswith("delta_")]
    for col in share_cols:
        s = pd.to_numeric(daily[col], errors="coerce")
        assert ((s >= 0.0) & (s <= 1.0) | s.isna()).all(), f"share column out of range: {col}"

    tier_sum = (
        pd.to_numeric(daily["tier_easy_share"], errors="coerce").fillna(0.0)
        + pd.to_numeric(daily["tier_medium_share"], errors="coerce").fillna(0.0)
        + pd.to_numeric(daily["tier_hard_share"], errors="coerce").fillna(0.0)
        + pd.to_numeric(daily["tier_blocked_share"], errors="coerce").fillna(0.0)
    )
    assert np.allclose(tier_sum.to_numpy(dtype=float), 1.0, atol=1e-9)



def test_summary_contains_expected_keys_and_valid_ranges(locate_contract_bundle):
    summary = locate_contract_bundle["summary"]
    assert EXPECTED_SUMMARY_KEYS.issubset(set(summary.keys()))
    assert int(summary["n_rows"]) > 0
    assert int(summary["n_dates"]) > 0
    assert int(summary["n_symbols"]) > 0
    assert 0.0 <= float(summary["mean_short_eligible_share"]) <= 1.0
    assert 0.0 <= float(summary["mean_blocked_share"]) <= 1.0
    assert 0.0 <= float(summary["mean_hard_share"]) <= 1.0
    assert isinstance(summary["blocked_reason_distribution"], dict)
    assert isinstance(summary["tier_distribution"], dict)

    tier_dist = summary["tier_distribution"]
    if tier_dist:
        assert set(tier_dist).issubset(ALLOWED_TIERS)
        assert abs(sum(float(v) for v in tier_dist.values()) - 1.0) < 1e-9



def test_manifest_contains_metadata_and_embeds_config_summary(locate_contract_bundle):
    manifest = locate_contract_bundle["manifest"]
    if manifest is None:
        pytest.skip("locate_filter implementation does not expose manifest payload builder")

    assert manifest.get("module") == "data.borrow.locate_filter"
    assert manifest.get("run_id") == "pytest_locate_contract"
    assert manifest.get("asof_timestamp") == "2025-03-07T21:00:00Z"
    assert isinstance(manifest.get("inputs"), dict)
    assert isinstance(manifest.get("outputs"), dict)
    assert isinstance(manifest.get("summary"), dict)
    assert isinstance(manifest.get("config"), dict)

    outputs = manifest["outputs"]
    assert outputs.get("required_columns_present") is True
    assert int(outputs.get("locate_daily_rows", 0)) > 0
    assert int(outputs.get("daily_summary_rows", 0)) > 0
    assert int(outputs.get("n_dates", 0)) > 0
    assert int(outputs.get("n_symbols", 0)) > 0

    cfg = manifest["config"]
    assert cfg.get("config_version") == manifest.get("config_version")
    assert isinstance(cfg.get("thresholds"), dict)
    assert isinstance(cfg.get("policy"), dict)
    assert isinstance(cfg.get("hysteresis"), dict)
    assert isinstance(cfg.get("sizing"), dict)



def test_output_is_sorted_deterministically_by_date_symbol(locate_contract_bundle):
    locate = locate_contract_bundle["locate"]
    expected = locate.sort_values(["date", "symbol"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(locate.reset_index(drop=True), expected, check_like=False)
