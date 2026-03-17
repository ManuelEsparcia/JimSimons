from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# Import adapter
# -----------------------------------------------------------------------------


def _load_adjust_prices_module():
    module_names = [
        "simons_smallcap_swing.data.price.adjust_prices",
        "data.price.adjust_prices",
        "adjust_prices",
    ]
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue

    here = Path(__file__).resolve()
    candidate_paths = [
        here.parents[3] / "simons_smallcap_swing" / "data" / "price" / "adjust_prices.py",
        here.parents[2] / "data" / "price" / "adjust_prices.py",
        here.parents[1] / "adjust_prices.py",
        Path("/mnt/data/adjust_prices.py"),
    ]
    for path in candidate_paths:
        if path.exists():
            module_name = f"_adjust_prices_test_{abs(hash(str(path)))}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module

    raise ModuleNotFoundError(
        "Could not import adjust_prices.py. Expected it at "
        "simons_smallcap_swing.data.price.adjust_prices or in a nearby repo path."
    )


@pytest.fixture(scope="session")
def adjust_prices_module():
    return _load_adjust_prices_module()


# -----------------------------------------------------------------------------
# Helpers / fixtures
# -----------------------------------------------------------------------------


def _cfg(module, **adjustment_overrides: Any):
    adj_defaults = {
        "mode": module.AdjustmentMode.DUAL_OUTPUT.value,
        "include_special_cash_dividends": False,
        "allow_empty_corporate_actions": True,
        "unsupported_event_severity": module.Severity.WARN.value,
        "missing_prev_close_severity": module.Severity.WARN.value,
        "invalid_dividend_severity": module.Severity.WARN.value,
        "abort_on_fail_symbol": False,
    }
    adj_defaults.update(adjustment_overrides)
    return module.AdjustPricesConfig(
        reverse_split=module.ReverseSplitPolicy(extreme_threshold=20.0),
        conflict=module.ConflictPolicy(),
        adjustment=module.AdjustmentPolicy(**adj_defaults),
        output=module.OutputPolicy(),
    )


@pytest.fixture()
def base_prices() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "AAA", "instrument_id": "101", "date": "2026-01-01", "open": 100.0, "high": 110.0, "low": 90.0, "close": 100.0, "volume": 1000.0},
            {"symbol": "AAA", "instrument_id": "101", "date": "2026-01-02", "open": 50.0, "high": 55.0, "low": 45.0, "close": 50.0, "volume": 2000.0},
            {"symbol": "AAA", "instrument_id": "101", "date": "2026-01-03", "open": 48.0, "high": 49.0, "low": 47.0, "close": 48.0, "volume": 2200.0},
        ]
    )


@pytest.fixture()
def run_adjust(adjust_prices_module):
    module = adjust_prices_module

    def _run(
        prices: pd.DataFrame,
        events: pd.DataFrame,
        *,
        asof_ts_utc: str = "2026-01-04T00:00:00Z",
        start_date: str = "2026-01-01",
        end_date: str = "2026-01-03",
        config: Any | None = None,
        run_id: str = "unit_test_run",
    ):
        cfg = config or _cfg(module)
        return module.adjust_prices(
            prices_raw=prices,
            corporate_actions=events,
            run_id=run_id,
            asof_ts_utc=asof_ts_utc,
            start_date=start_date,
            end_date=end_date,
            config=cfg,
        )

    return _run


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_split_2_for_1_adjusts_ohlc_volume_and_returns_correctly(adjust_prices_module, base_prices, run_adjust):
    module = adjust_prices_module
    events = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "instrument_id": "101",
                "event_type": module.CanonicalEventType.SPLIT.value,
                "ex_date": "2026-01-02",
                "split_ratio": 0.5,
                "source_provider": "primary_provider",
            }
        ]
    )

    artifacts = run_adjust(base_prices.iloc[:2].copy(), events, end_date="2026-01-02")
    adj = artifacts.adjusted
    ev = artifacts.events_applied

    row_pre = adj.loc[adj["date"] == pd.Timestamp("2026-01-01")].iloc[0]
    row_event = adj.loc[adj["date"] == pd.Timestamp("2026-01-02")].iloc[0]

    assert row_pre["split_factor_cum"] == pytest.approx(0.5)
    assert row_pre["close_adj_split"] == pytest.approx(50.0)
    assert row_pre["open_adj_split"] == pytest.approx(50.0)
    assert row_pre["high_adj_split"] == pytest.approx(55.0)
    assert row_pre["low_adj_split"] == pytest.approx(45.0)
    assert row_pre["volume_adj"] == pytest.approx(2000.0)
    assert row_event["split_factor_cum"] == pytest.approx(1.0)
    assert row_event["close_adj_split"] == pytest.approx(50.0)
    assert row_event["ret_1d_adj_split"] == pytest.approx(0.0)

    assert len(ev) == 1
    assert ev.iloc[0]["outcome"] == module.EventOutcome.APPLIED.value
    assert ev.iloc[0]["split_factor_elementary"] == pytest.approx(0.5)



def test_extreme_reverse_split_warns_and_scales_history(adjust_prices_module, base_prices, run_adjust):
    module = adjust_prices_module
    events = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "instrument_id": "101",
                "event_type": module.CanonicalEventType.REVERSE_SPLIT.value,
                "ex_date": "2026-01-03",
                "split_ratio": 25.0,
                "source_provider": "primary_provider",
            }
        ]
    )

    artifacts = run_adjust(base_prices.copy(), events)
    adj = artifacts.adjusted
    ev = artifacts.events_applied.iloc[0]

    jan1 = adj.loc[adj["date"] == pd.Timestamp("2026-01-01")].iloc[0]
    jan2 = adj.loc[adj["date"] == pd.Timestamp("2026-01-02")].iloc[0]
    jan3 = adj.loc[adj["date"] == pd.Timestamp("2026-01-03")].iloc[0]

    assert ev["severity"] == module.Severity.WARN.value
    assert ev["outcome"] == module.EventOutcome.APPLIED.value
    assert jan1["close_adj_split"] == pytest.approx(2500.0)
    assert jan2["close_adj_split"] == pytest.approx(1250.0)
    assert jan3["close_adj_split"] == pytest.approx(48.0)
    assert jan1["volume_adj"] == pytest.approx(40.0)



def test_cash_dividend_computes_exact_total_return_factor(adjust_prices_module, base_prices, run_adjust):
    module = adjust_prices_module
    cfg = _cfg(module, mode=module.AdjustmentMode.TOTAL_RETURN_LIKE.value)
    events = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "instrument_id": "101",
                "event_type": module.CanonicalEventType.CASH_DIVIDEND.value,
                "ex_date": "2026-01-03",
                "cash_amount": 5.0,
                "source_provider": "primary_provider",
            }
        ]
    )

    artifacts = run_adjust(base_prices.copy(), config=cfg, events=events)
    adj = artifacts.adjusted
    ev = artifacts.events_applied.iloc[0]

    phi = 1.0 - 5.0 / 50.0
    jan2 = adj.loc[adj["date"] == pd.Timestamp("2026-01-02")].iloc[0]
    jan3 = adj.loc[adj["date"] == pd.Timestamp("2026-01-03")].iloc[0]

    assert ev["prev_close_for_dividend"] == pytest.approx(50.0)
    assert ev["dividend_factor_elementary"] == pytest.approx(phi)
    assert jan2["dividend_factor_cum"] == pytest.approx(phi)
    assert jan2["close_adj_total"] == pytest.approx(50.0 * phi)
    assert jan3["close_adj_total"] == pytest.approx(48.0)
    assert jan2["volume_adj"] == pytest.approx(2000.0)



def test_dividend_greater_or_equal_prev_close_is_skipped_and_flagged(adjust_prices_module, base_prices, run_adjust):
    module = adjust_prices_module
    events = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "instrument_id": "101",
                "event_type": module.CanonicalEventType.CASH_DIVIDEND.value,
                "ex_date": "2026-01-03",
                "cash_amount": 50.0,
                "source_provider": "primary_provider",
            }
        ]
    )

    artifacts = run_adjust(base_prices.copy(), events)
    adj = artifacts.adjusted
    ev = artifacts.events_applied.iloc[0]

    assert ev["outcome"] == module.EventOutcome.SKIPPED_INVALID_DIVIDEND_FACTOR.value
    assert ev["failure_code"] == module.FailureCode.INVALID_DIVIDEND_FACTOR.value
    assert bool(ev["include_in_dividend_chain"]) is False
    assert adj["dividend_factor_cum"].eq(1.0).all()
    assert adj["close_adj_total"].eq(adj["close_raw"]).all()



def test_missing_prev_close_skips_dividend_and_flags(adjust_prices_module, base_prices, run_adjust):
    module = adjust_prices_module
    events = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "instrument_id": "101",
                "event_type": module.CanonicalEventType.CASH_DIVIDEND.value,
                "ex_date": "2026-01-01",
                "cash_amount": 1.0,
                "source_provider": "primary_provider",
            }
        ]
    )

    artifacts = run_adjust(base_prices.copy(), events)
    ev = artifacts.events_applied.iloc[0]
    adj = artifacts.adjusted

    assert ev["outcome"] == module.EventOutcome.SKIPPED_MISSING_PREV_CLOSE.value
    assert ev["failure_code"] == module.FailureCode.MISSING_PREV_CLOSE_FOR_DIVIDEND.value
    assert pd.isna(ev["prev_close_for_dividend"])
    assert adj["dividend_factor_cum"].eq(1.0).all()



def test_multiple_same_day_events_follow_stable_precedence(adjust_prices_module, base_prices, run_adjust):
    module = adjust_prices_module
    events = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "instrument_id": "101",
                "event_type": module.CanonicalEventType.CASH_DIVIDEND.value,
                "ex_date": "2026-01-03",
                "cash_amount": 4.0,
                "source_provider": "primary_provider",
            },
            {
                "symbol": "AAA",
                "instrument_id": "101",
                "event_type": module.CanonicalEventType.SPLIT.value,
                "ex_date": "2026-01-03",
                "split_ratio": 0.5,
                "source_provider": "primary_provider",
            },
        ]
    )

    artifacts = run_adjust(base_prices.copy(), events)
    ev = artifacts.events_applied.reset_index(drop=True)
    adj = artifacts.adjusted

    assert ev["event_type"].tolist() == [
        module.CanonicalEventType.SPLIT.value,
        module.CanonicalEventType.CASH_DIVIDEND.value,
    ]
    assert ev["same_day_precedence_rank"].tolist() == [1, 3]

    jan2 = adj.loc[adj["date"] == pd.Timestamp("2026-01-02")].iloc[0]
    expected_phi = 1.0 - 4.0 / 50.0
    assert jan2["split_factor_cum"] == pytest.approx(0.5)
    assert jan2["dividend_factor_cum"] == pytest.approx(expected_phi)
    assert jan2["close_adj_split"] == pytest.approx(25.0)
    assert jan2["close_adj_total"] == pytest.approx(50.0 * 0.5 * expected_phi)
    assert jan2["volume_adj"] == pytest.approx(4000.0)



def test_event_outside_requested_range_is_skipped_without_affecting_series(adjust_prices_module, base_prices, run_adjust):
    module = adjust_prices_module
    events = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "instrument_id": "101",
                "event_type": module.CanonicalEventType.SPLIT.value,
                "ex_date": "2026-01-03",
                "split_ratio": 0.5,
                "source_provider": "primary_provider",
            }
        ]
    )

    artifacts = run_adjust(base_prices.iloc[:2].copy(), events, end_date="2026-01-02")
    ev = artifacts.events_applied.iloc[0]
    adj = artifacts.adjusted

    assert ev["outcome"] == module.EventOutcome.SKIPPED_OUTSIDE_RANGE.value
    assert adj["split_factor_cum"].eq(1.0).all()
    assert adj["close_adj_split"].eq(adj["close_raw"]).all()



def test_event_not_visible_by_asof_is_not_applied(adjust_prices_module, base_prices, run_adjust):
    module = adjust_prices_module
    events = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "instrument_id": "101",
                "event_type": module.CanonicalEventType.SPLIT.value,
                "ex_date": "2026-01-02",
                "split_ratio": 0.5,
                "announcement_ts": "2026-01-03T12:00:00Z",
                "source_provider": "primary_provider",
            }
        ]
    )

    artifacts = run_adjust(
        base_prices.iloc[:2].copy(),
        events,
        asof_ts_utc="2026-01-02T00:00:00Z",
        end_date="2026-01-02",
    )
    adj = artifacts.adjusted

    assert artifacts.events_applied.empty
    assert adj["split_factor_cum"].eq(1.0).all()
    assert adj["close_adj_split"].eq(adj["close_raw"]).all()



def test_symbol_without_events_matches_raw_algebraically(adjust_prices_module, base_prices, run_adjust):
    module = adjust_prices_module
    cfg = _cfg(module, mode=module.AdjustmentMode.DUAL_OUTPUT.value)
    artifacts = run_adjust(base_prices.copy(), pd.DataFrame(), config=cfg)
    adj = artifacts.adjusted

    assert adj["split_factor_cum"].eq(1.0).all()
    assert adj["dividend_factor_cum"].eq(1.0).all()
    assert adj["open_adj_split"].eq(adj["open_raw"]).all()
    assert adj["close_adj_split"].eq(adj["close_raw"]).all()
    assert adj["open_adj_total"].eq(adj["open_raw"]).all()
    assert adj["close_adj_total"].eq(adj["close_raw"]).all()
    assert adj["volume_adj"].eq(adj["volume_raw"]).all()



def test_unsupported_event_is_logged_without_silent_adjustment(adjust_prices_module, base_prices, run_adjust):
    module = adjust_prices_module
    events = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "instrument_id": "101",
                "event_type": module.CanonicalEventType.MERGER.value,
                "effective_date": "2026-01-02",
                "source_provider": "primary_provider",
            }
        ]
    )

    artifacts = run_adjust(base_prices.copy(), events)
    ev = artifacts.events_applied.iloc[0]
    adj = artifacts.adjusted

    assert ev["outcome"] == module.EventOutcome.SKIPPED_UNSUPPORTED.value
    assert ev["failure_code"] == module.FailureCode.UNSUPPORTED_EVENT_TYPE.value
    assert ev["severity"] == module.Severity.WARN.value
    assert adj["split_factor_cum"].eq(1.0).all()
    assert adj["has_unsupported_events"].any()



def test_special_cash_dividend_excluded_by_default_policy(adjust_prices_module, base_prices, run_adjust):
    module = adjust_prices_module
    events = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "instrument_id": "101",
                "event_type": module.CanonicalEventType.SPECIAL_CASH_DIVIDEND.value,
                "ex_date": "2026-01-03",
                "cash_amount": 1.0,
                "source_provider": "primary_provider",
            }
        ]
    )

    artifacts = run_adjust(base_prices.copy(), events)
    ev = artifacts.events_applied.iloc[0]
    adj = artifacts.adjusted

    assert ev["outcome"] == module.EventOutcome.SKIPPED_SPECIAL_DIVIDEND_EXCLUDED.value
    assert bool(ev["include_in_dividend_chain"]) is False
    assert adj["dividend_factor_cum"].eq(1.0).all()



def test_volume_is_adjusted_only_by_split_not_dividend(adjust_prices_module, base_prices, run_adjust):
    module = adjust_prices_module
    events = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "instrument_id": "101",
                "event_type": module.CanonicalEventType.CASH_DIVIDEND.value,
                "ex_date": "2026-01-03",
                "cash_amount": 5.0,
                "source_provider": "primary_provider",
            }
        ]
    )

    artifacts = run_adjust(base_prices.copy(), events)
    adj = artifacts.adjusted

    assert adj["volume_adj"].tolist() == pytest.approx(adj["volume_raw"].tolist())
    assert (adj["dividend_factor_cum"] < 1.0).any()
