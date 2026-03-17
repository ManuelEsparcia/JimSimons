from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest



def _load_borrow_qc_module():
    module_names = [
        "simons_smallcap_swing.data.borrow.borrow_qc",
        "data.borrow.borrow_qc",
        "borrow_qc",
    ]
    for name in module_names:
        try:
            return importlib.import_module(name)
        except Exception:
            pass

    candidate_paths = [
        Path.cwd() / "simons_smallcap_swing" / "data" / "borrow" / "borrow_qc.py",
        Path.cwd() / "data" / "borrow" / "borrow_qc.py",
        Path.cwd() / "borrow_qc.py",
        Path("/mnt/data/borrow_qc.py"),
    ]
    for path in candidate_paths:
        if path.exists():
            spec = importlib.util.spec_from_file_location("borrow_qc_under_test", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module

    raise ImportError("Could not locate borrow_qc.py via import path or file path.")


borrow_qc = _load_borrow_qc_module()


@pytest.fixture()
def cfg():
    return borrow_qc.BorrowQCConfig()



def _borrow_row(
    *,
    symbol: str,
    date: str,
    fee_annual: float = 0.05,
    fee_daily: float = 0.05 / 252.0,
    alpha: float = 0.80,
    htb: bool = False,
    tier: str = "easy",
    quality: str = "high",
    source: str = "direct_lending_feed",
    stress: float = 0.10,
    jump: bool = False,
    stale: bool = False,
    fallback: bool = False,
    config_version: str = "proxy_v1",
):
    return {
        "symbol": symbol,
        "date": pd.Timestamp(date),
        "borrow_fee_annual": fee_annual,
        "borrow_fee_daily": fee_daily,
        "borrow_availability_score": alpha,
        "htb_flag": htb,
        "borrow_tier": tier,
        "proxy_quality": quality,
        "proxy_source": source,
        "stress_score": stress,
        "jump_flag": jump,
        "stale_input_flag": stale,
        "fallback_flag": fallback,
        "config_version": config_version,
    }



def _locate_row(
    *,
    symbol: str,
    date: str,
    allowed: bool = True,
    override_flag: bool = False,
    override_reason: str = "",
    config_version: str = "locate_v1",
):
    return {
        "symbol": symbol,
        "date": pd.Timestamp(date),
        "locate_allowed": allowed,
        "override_flag": override_flag,
        "override_reason": override_reason,
        "config_version": config_version,
    }



def _universe_row(
    *,
    symbol: str,
    date: str,
    size_bucket: str = "small",
    liq_bucket: str = "medium",
    short_flag: bool = True,
    membership_state: str = "active",
):
    return {
        "symbol": symbol,
        "date": pd.Timestamp(date),
        "size_bucket": size_bucket,
        "liq_bucket": liq_bucket,
        "short_subuniverse_flag": short_flag,
        "membership_state": membership_state,
    }



def _run_qc(borrow_rows, locate_rows, universe_rows, cfg=None):
    borrow_df = pd.DataFrame(list(borrow_rows))
    locate_df = pd.DataFrame(list(locate_rows))
    universe_df = pd.DataFrame(list(universe_rows))
    return borrow_qc.build_borrow_qc(
        borrow_proxy=borrow_df,
        locate_filter=locate_df,
        universe=universe_df,
        cfg=cfg or borrow_qc.BorrowQCConfig(),
        run_id="pytest_run",
        asof_timestamp="2026-03-15T12:00:00Z",
    )



def _failure_check_types(failures: pd.DataFrame) -> set[str]:
    if failures.empty:
        return set()
    return set(failures["check_type"].astype(str))



def test_clean_input_produces_pass_gate(cfg):
    daily, failures, summary, panel = _run_qc(
        borrow_rows=[_borrow_row(symbol="AAA", date="2026-01-02")],
        locate_rows=[_locate_row(symbol="AAA", date="2026-01-02", allowed=True)],
        universe_rows=[_universe_row(symbol="AAA", date="2026-01-02")],
        cfg=cfg,
    )

    assert not daily.empty
    assert daily.iloc[0]["gate"] == "pass"
    assert summary["n_rows_joined"] == 1
    assert bool(panel.iloc[0]["hard_numeric_fail"]) is False
    assert _failure_check_types(failures) == set()



def test_negative_borrow_fee_triggers_fail_numeric(cfg):
    daily, failures, summary, panel = _run_qc(
        borrow_rows=[_borrow_row(symbol="AAA", date="2026-01-02", fee_annual=-0.01, fee_daily=-0.01 / 252.0)],
        locate_rows=[_locate_row(symbol="AAA", date="2026-01-02", allowed=True)],
        universe_rows=[_universe_row(symbol="AAA", date="2026-01-02")],
        cfg=cfg,
    )

    assert daily.iloc[0]["gate"] == "fail"
    assert "negative_fee" in _failure_check_types(failures)
    assert failures.loc[failures["check_type"] == "negative_fee", "severity"].iloc[0] == "FAIL_NUMERIC"
    assert bool(panel.iloc[0]["hard_numeric_fail"]) is True
    assert summary["validation"]["numeric_failures"] >= 1



def test_availability_out_of_range_triggers_fail_numeric(cfg):
    daily, failures, _, panel = _run_qc(
        borrow_rows=[_borrow_row(symbol="AAA", date="2026-01-02", alpha=1.25)],
        locate_rows=[_locate_row(symbol="AAA", date="2026-01-02", allowed=True)],
        universe_rows=[_universe_row(symbol="AAA", date="2026-01-02")],
        cfg=cfg,
    )

    assert daily.iloc[0]["gate"] == "fail"
    assert "availability_out_of_range" in _failure_check_types(failures)
    assert bool(panel.iloc[0]["hard_numeric_fail"]) is True


@pytest.mark.xfail(
    reason="Current borrow_qc.py crashes in _json_default when invalid_borrow_tier tries to serialize a list threshold.",
    strict=False,
)
def test_invalid_borrow_tier_should_surface_catalog_failure_not_exception(cfg):
    daily, failures, _, panel = _run_qc(
        borrow_rows=[_borrow_row(symbol="AAA", date="2026-01-02", tier="very_hard")],
        locate_rows=[_locate_row(symbol="AAA", date="2026-01-02", allowed=True)],
        universe_rows=[_universe_row(symbol="AAA", date="2026-01-02")],
        cfg=cfg,
    )

    assert daily.iloc[0]["gate"] == "fail"
    assert "invalid_borrow_tier" in _failure_check_types(failures)
    assert bool(panel.iloc[0]["hard_numeric_fail"]) is True



def test_missing_locate_join_is_detected_as_missing_critical(cfg):
    borrow_df = pd.DataFrame([_borrow_row(symbol="AAA", date="2026-01-02")])
    locate_df = pd.DataFrame(columns=["symbol", "date", "locate_allowed", "override_flag", "override_reason", "config_version"])
    universe_df = pd.DataFrame([_universe_row(symbol="AAA", date="2026-01-02")])

    daily, failures, _, panel = borrow_qc.build_borrow_qc(
        borrow_proxy=borrow_df,
        locate_filter=locate_df,
        universe=universe_df,
        cfg=cfg,
        run_id="pytest_run",
        asof_timestamp="2026-03-15T12:00:00Z",
    )

    assert daily.iloc[0]["gate"] == "fail"
    assert "missing_critical_field" in _failure_check_types(failures)
    assert bool(panel.iloc[0]["missing_join_locate"]) is True
    assert bool(panel.iloc[0]["hard_numeric_fail"]) is True



def test_htb_underblocking_is_detected_and_fails_day(cfg):
    daily, failures, _, panel = _run_qc(
        borrow_rows=[_borrow_row(symbol="AAA", date="2026-01-02", htb=True, tier="blocked", alpha=0.02, fee_annual=1.25)],
        locate_rows=[_locate_row(symbol="AAA", date="2026-01-02", allowed=True, override_flag=False, override_reason="")],
        universe_rows=[_universe_row(symbol="AAA", date="2026-01-02")],
        cfg=cfg,
    )

    assert daily.iloc[0]["gate"] == "fail"
    assert pytest.approx(float(daily.iloc[0]["underblocking_rate"]), rel=0, abs=1e-12) == 1.0
    assert bool(panel.iloc[0]["underblocking_critical"]) is True
    assert "htb_locate_incoherence" in _failure_check_types(failures)



def test_stale_derived_flag_sets_stale_failure_and_freshness_penalty(cfg):
    dates = pd.bdate_range("2026-01-02", periods=8)
    borrow_rows = [_borrow_row(symbol="AAA", date=str(d.date()), fee_annual=0.15, fee_daily=0.15 / 252.0, alpha=0.40) for d in dates]
    locate_rows = [_locate_row(symbol="AAA", date=str(d.date()), allowed=True) for d in dates]
    universe_rows = [_universe_row(symbol="AAA", date=str(d.date())) for d in dates]

    daily, failures, _, panel = _run_qc(borrow_rows, locate_rows, universe_rows, cfg=cfg)

    last_panel = panel.sort_values(["symbol", "date"]).iloc[-1]
    last_daily = daily.sort_values("date").iloc[-1]

    assert bool(last_panel["stale_derived_flag"]) is True
    assert bool(last_panel["stale_effective_flag"]) is True
    assert float(last_daily["stale_share"]) == pytest.approx(1.0, rel=0, abs=1e-12)
    assert float(last_daily["score_freshness"]) == pytest.approx(0.0, rel=0, abs=1e-12)
    assert "stale_share" in _failure_check_types(failures)



def test_duplicate_borrow_key_triggers_structural_failure_record(cfg):
    borrow_rows = [
        _borrow_row(symbol="AAA", date="2026-01-02"),
        _borrow_row(symbol="AAA", date="2026-01-02", fee_annual=0.06),
    ]
    locate_rows = [_locate_row(symbol="AAA", date="2026-01-02", allowed=True)]
    universe_rows = [_universe_row(symbol="AAA", date="2026-01-02")]

    daily, failures, summary, _ = _run_qc(borrow_rows, locate_rows, universe_rows, cfg=cfg)

    assert not daily.empty
    assert "duplicate_key_borrow_proxy" in _failure_check_types(failures)
    assert summary["validation"]["structural_failures"] >= 1
    assert failures.loc[failures["check_type"] == "duplicate_key_borrow_proxy", "severity"].iloc[0] == "FAIL_STRUCTURAL"



def test_new_symbol_without_history_does_not_inflate_jump_share(cfg):
    borrow_rows = [
        _borrow_row(symbol="AAA", date="2026-01-02", fee_annual=0.10),
        _borrow_row(symbol="AAA", date="2026-01-05", fee_annual=0.10),
        _borrow_row(symbol="BBB", date="2026-01-05", fee_annual=0.10),
    ]
    locate_rows = [
        _locate_row(symbol="AAA", date="2026-01-02", allowed=True),
        _locate_row(symbol="AAA", date="2026-01-05", allowed=True),
        _locate_row(symbol="BBB", date="2026-01-05", allowed=True),
    ]
    universe_rows = [
        _universe_row(symbol="AAA", date="2026-01-02"),
        _universe_row(symbol="AAA", date="2026-01-05"),
        _universe_row(symbol="BBB", date="2026-01-05"),
    ]

    daily, failures, _, panel = _run_qc(borrow_rows, locate_rows, universe_rows, cfg=cfg)

    second_day = daily.sort_values("date").iloc[-1]
    second_day_panel = panel.loc[panel["date"] == pd.Timestamp("2026-01-05")].sort_values("symbol")

    assert float(second_day["jump_share"]) == pytest.approx(0.0, rel=0, abs=1e-12)
    assert bool(second_day_panel.loc[second_day_panel["symbol"] == "BBB", "has_history"].iloc[0]) is False
    assert "jump_share" not in _failure_check_types(failures)
