from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path('/mnt/data/universe_qc.py')


def _load_module():
    name = 'universe_qc_under_test'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


uq = _load_module()


def _default_cfg():
    cfg, _payload, _hash = uq.load_config(None)
    return cfg


def _base_universe_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'date': '2020-02-01',
                'instrument_id': 'I1',
                'issuer_id': 'ISS1',
                'symbol': 'AAA',
                'is_eligible': 1,
                'membership_state': 'eligible',
                'primary_exclusion_reason': pd.NA,
                'all_failed_reasons': pd.NA,
                'market_cap_usd': 500_000_000,
                'adv20_usd': 5_000_000,
                'price_ref': 25.0,
                'listing_exchange': 'NYSE',
                'security_type': 'COMMON_STOCK',
                'trading_status': 'active',
                'list_date': '2020-01-01',
                'delist_date': pd.NaT,
            },
            {
                'date': '2020-02-02',
                'instrument_id': 'I1',
                'issuer_id': 'ISS1',
                'symbol': 'AAA',
                'is_eligible': 1,
                'membership_state': 'eligible',
                'primary_exclusion_reason': pd.NA,
                'all_failed_reasons': pd.NA,
                'market_cap_usd': 510_000_000,
                'adv20_usd': 5_100_000,
                'price_ref': 26.0,
                'listing_exchange': 'NYSE',
                'security_type': 'COMMON_STOCK',
                'trading_status': 'active',
                'list_date': '2020-01-01',
                'delist_date': pd.NaT,
            },
        ]
    )


def _base_listings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'instrument_id': 'I1',
                'issuer_id': 'ISS1',
                'symbol': 'AAA',
                'list_date': '2020-01-01',
                'delist_date': pd.NaT,
                'ticker_start_date': '2020-01-01',
                'ticker_end_date': pd.NaT,
                'status': 'active',
                'listing_exchange': 'NYSE',
                'security_type': 'COMMON_STOCK',
            }
        ]
    )


def _base_calendar() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(['2020-02-01', '2020-02-02']))


def _prep_loaded_df(df: pd.DataFrame) -> pd.DataFrame:
    loaded = df.copy()
    loaded['date'] = pd.to_datetime(loaded['date']).dt.normalize()
    for col in ['list_date', 'delist_date']:
        if col in loaded.columns:
            loaded[col] = pd.to_datetime(loaded[col], errors='coerce').dt.normalize()
    return loaded


def test_structural_duplicate_key_and_invalid_is_eligible_are_detected():
    df = _prep_loaded_df(_base_universe_rows())
    bad = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    bad.loc[0, 'is_eligible'] = 2
    failures = pd.DataFrame(uq.structural_fail_fast_checks(bad))
    assert uq.CheckType.STRUCTURAL_DUPLICATE_LOGICAL_KEY.value in set(failures['check_type'])
    assert uq.CheckType.STRUCTURAL_INVALID_IS_ELIGIBLE.value in set(failures['check_type'])
    assert (failures['severity'] == uq.Severity.FAIL.value).any()


def test_run_pit_checks_flags_pre_listing_post_death_and_eligible_after_termination():
    listings = uq.load_listings_master(_write_csv_temp(_base_listings()))
    lifecycle = uq.build_listing_lifecycle(listings, pd.DataFrame(columns=sorted(uq.OPTIONAL_CORPORATE_ACTION_COLUMNS)))
    symbol_windows = uq.build_symbol_windows(listings)

    df = pd.DataFrame(
        [
            {
                'date': '2019-12-31', 'instrument_id': 'I1', 'symbol': 'AAA', 'is_eligible': 0,
                'membership_state': 'ineligible_rule', 'primary_exclusion_reason': uq.ExclusionReason.NOT_YET_LISTED.value,
            },
            {
                'date': '2020-02-03', 'instrument_id': 'I1', 'symbol': 'AAA', 'is_eligible': 1,
                'membership_state': 'eligible', 'primary_exclusion_reason': pd.NA,
            },
        ]
    )
    df = _prep_loaded_df(df)
    lifecycle = lifecycle.copy()
    lifecycle.loc[0, 'death_date'] = pd.Timestamp('2020-02-02')
    lifecycle.loc[0, 'terminal_ca_id'] = 'CA_TERM_1'
    attached = uq.attach_lifecycle(df, lifecycle)

    failures = pd.DataFrame(uq.run_pit_checks(attached, symbol_windows, {}, _default_cfg()))
    types = set(failures['check_type'])
    assert uq.CheckType.PIT_OUTSIDE_LIFECYCLE_WINDOW.value in types
    assert uq.CheckType.PIT_ELIGIBLE_AFTER_TERMINATION.value in types


def test_run_semantic_checks_detects_reason_order_and_membership_mismatch():
    cfg = _default_cfg()
    df = pd.DataFrame(
        [
            {
                'date': '2020-02-01', 'instrument_id': 'I1', 'symbol': 'AAA', 'is_eligible': 1,
                'membership_state': uq.MembershipState.ELIGIBLE.value,
                'primary_exclusion_reason': uq.ExclusionReason.PRICE_BELOW_MIN.value,
                'all_failed_reasons': f"{uq.ExclusionReason.ADV20_BELOW_MIN.value}|{uq.ExclusionReason.PRICE_BELOW_MIN.value}",
                'listing_exchange': 'NYSE', 'security_type': 'COMMON_STOCK', 'list_date': '2020-01-01',
                'price_ref': 25.0, 'adv20_usd': 5_000_000, 'market_cap_usd': 500_000_000,
            }
        ]
    )
    df = _prep_loaded_df(df)
    failures = pd.DataFrame(uq.run_semantic_checks(df, cfg))
    types = set(failures['check_type'])
    assert uq.CheckType.SEMANTIC_ELIGIBILITY_REASON_MISMATCH.value in types
    assert uq.CheckType.SEMANTIC_PRIMARY_REASON_ORDER_MISMATCH.value in types
    assert uq.CheckType.SEMANTIC_MEMBERSHIP_STATE_MISMATCH.value in types


def test_run_semantic_checks_revalidates_critical_rules_for_eligible_rows():
    cfg = _default_cfg()
    cfg = uq.UniverseQCConfig(
        critical_rules=cfg.critical_rules,
        thresholds=cfg.thresholds,
        validation=uq.ValidationConfig(strict_mode=True, revalidate_sample_frac=1.0, revalidate_max_rows=100_000, random_seed=17),
        output=cfg.output,
        missing_critical_meta_cols=cfg.missing_critical_meta_cols,
    )
    df = pd.DataFrame(
        [
            {
                'date': '2020-02-01', 'instrument_id': 'I1', 'symbol': 'AAA', 'is_eligible': 1,
                'membership_state': uq.MembershipState.ELIGIBLE.value,
                'primary_exclusion_reason': pd.NA, 'all_failed_reasons': pd.NA,
                'listing_exchange': 'NYSE', 'security_type': 'COMMON_STOCK', 'list_date': '2020-01-25',
                'price_ref': 1.0, 'adv20_usd': 100.0, 'market_cap_usd': 10_000.0,
            }
        ]
    )
    df = _prep_loaded_df(df)
    failures = pd.DataFrame(uq.run_semantic_checks(df, cfg))
    assert uq.CheckType.SEMANTIC_CRITICAL_RULE_REVALIDATION.value in set(failures['check_type'])


def test_run_calendar_temporal_coverage_checks_detects_missing_session_and_turnover():
    cfg = _default_cfg()
    df = pd.DataFrame(
        [
            {'date': '2020-02-01', 'instrument_id': 'I1', 'symbol': 'AAA', 'is_eligible': 1, 'primary_exclusion_reason': pd.NA,
             'market_cap_usd': 500_000_000, 'adv20_usd': 5_000_000, 'price_ref': 25.0, 'listing_exchange': 'NYSE',
             'security_type': 'COMMON_STOCK', 'membership_state': 'eligible', 'trading_status': 'active', 'list_date': '2020-01-01', 'birth_date': '2020-01-01', 'death_date': pd.NaT, 'terminal_ca_id': pd.NA},
            {'date': '2020-02-03', 'instrument_id': 'I2', 'symbol': 'BBB', 'is_eligible': 1, 'primary_exclusion_reason': pd.NA,
             'market_cap_usd': 500_000_000, 'adv20_usd': 5_000_000, 'price_ref': 25.0, 'listing_exchange': 'NYSE',
             'security_type': 'COMMON_STOCK', 'membership_state': 'eligible', 'trading_status': 'active', 'list_date': '2020-01-01', 'birth_date': '2020-01-01', 'death_date': pd.NaT, 'terminal_ca_id': pd.NA},
        ]
    )
    df = _prep_loaded_df(df)
    lifecycle = pd.DataFrame([
        {'instrument_id': 'I1', 'birth_date': pd.Timestamp('2020-01-01'), 'death_date': pd.NaT, 'is_dead_observed': False},
        {'instrument_id': 'I2', 'birth_date': pd.Timestamp('2020-01-01'), 'death_date': pd.NaT, 'is_dead_observed': False},
    ])
    calendar = pd.DatetimeIndex(pd.to_datetime(['2020-02-01', '2020-02-02', '2020-02-03']))
    daily, failures = uq.run_calendar_temporal_coverage_checks(df, lifecycle, calendar, {}, cfg)
    failures = pd.DataFrame(failures)
    assert uq.CheckType.CALENDAR_MISSING_SESSION.value in set(failures['check_type'])
    assert uq.CheckType.TEMPORAL_EXTREME_TURNOVER.value in set(failures['check_type'])
    assert len(daily) == 2


def test_build_gates_fails_on_critical_and_warns_on_missing_meta():
    cfg = _default_cfg()
    daily = pd.DataFrame([
        {
            'date': pd.Timestamp('2020-02-01'),
            'n_rows': 10,
            'turnover': 0.0,
            'delisted_coverage_ratio': 1.0,
            'pct_missing_critical_meta': cfg.thresholds.pct_missing_critical_meta_warn + 0.01,
        }
    ])
    failures = pd.DataFrame([
        uq.make_failure(
            date='2020-02-01', instrument_id='I1', symbol='AAA',
            check_type=uq.CheckType.PIT_OUTSIDE_LIFECYCLE_WINDOW,
            severity=uq.Severity.FAIL,
            flag_reason='boom',
            evidence_summary='critical pit issue',
        )
    ])
    gates = uq.build_gates(daily, failures, cfg)
    gate_map = {g.name: g.severity.value for g in gates}
    assert gate_map['gate1_integrity_pit_critical'] == uq.Severity.FAIL.value
    assert gate_map['gate5_missing_critical_meta'] == uq.Severity.WARN.value


@pytest.fixture()
def clean_input_files(tmp_path: Path):
    universe = _base_universe_rows()
    listings = _base_listings()
    calendar = pd.DataFrame({'date': ['2020-02-01', '2020-02-02']})

    u_path = tmp_path / 'universe.csv'
    l_path = tmp_path / 'listings.csv'
    c_path = tmp_path / 'calendar.csv'
    universe.to_csv(u_path, index=False)
    listings.to_csv(l_path, index=False)
    calendar.to_csv(c_path, index=False)
    return u_path, l_path, c_path


def _write_csv_temp(df: pd.DataFrame) -> Path:
    import tempfile
    path = Path(tempfile.mkstemp(suffix='.csv')[1])
    df.to_csv(path, index=False)
    return path


def test_audit_universe_clean_case_passes(clean_input_files):
    u_path, l_path, c_path = clean_input_files
    cfg, payload, cfg_hash = uq.load_config(None)
    artifacts = uq.audit_universe(
        universe_history_path=u_path,
        listings_master_path=l_path,
        calendar_path=c_path,
        corporate_actions_path=None,
        run_id='qc_clean',
        config=cfg,
        config_hash=cfg_hash,
        config_payload=payload,
    )
    assert artifacts.summary['gate_status'] == uq.Severity.PASS.value
    assert artifacts.failures.empty
    assert set(artifacts.daily['n_eligible']) == {1}


def test_audit_universe_detects_eligible_after_death_and_fails(tmp_path: Path):
    universe = _base_universe_rows()
    universe = pd.concat([
        universe,
        pd.DataFrame([{
            'date': '2020-02-03', 'instrument_id': 'I1', 'issuer_id': 'ISS1', 'symbol': 'AAA',
            'is_eligible': 1, 'membership_state': 'eligible', 'primary_exclusion_reason': pd.NA,
            'all_failed_reasons': pd.NA, 'market_cap_usd': 520_000_000, 'adv20_usd': 5_200_000,
            'price_ref': 27.0, 'listing_exchange': 'NYSE', 'security_type': 'COMMON_STOCK',
            'trading_status': 'active', 'list_date': '2020-01-01', 'delist_date': '2020-02-02',
        }])
    ], ignore_index=True)
    listings = _base_listings()
    listings.loc[0, 'delist_date'] = '2020-02-02'
    calendar = pd.DataFrame({'date': ['2020-02-01', '2020-02-02', '2020-02-03']})
    u_path = tmp_path / 'universe_bad.csv'
    l_path = tmp_path / 'listings_bad.csv'
    c_path = tmp_path / 'calendar_bad.csv'
    universe.to_csv(u_path, index=False)
    listings.to_csv(l_path, index=False)
    calendar.to_csv(c_path, index=False)

    cfg, payload, cfg_hash = uq.load_config(None)
    artifacts = uq.audit_universe(
        universe_history_path=u_path,
        listings_master_path=l_path,
        calendar_path=c_path,
        corporate_actions_path=None,
        run_id='qc_bad',
        config=cfg,
        config_hash=cfg_hash,
        config_payload=payload,
    )
    assert artifacts.summary['gate_status'] == uq.Severity.FAIL.value
    assert uq.CheckType.PIT_OUTSIDE_LIFECYCLE_WINDOW.value in set(artifacts.failures['check_type'])
    assert uq.CheckType.PIT_ELIGIBLE_AFTER_TERMINATION.value in set(artifacts.failures['check_type'])


def test_audit_universe_is_reproducible_on_same_inputs(clean_input_files):
    u_path, l_path, c_path = clean_input_files
    cfg, payload, cfg_hash = uq.load_config(None)
    a1 = uq.audit_universe(
        universe_history_path=u_path,
        listings_master_path=l_path,
        calendar_path=c_path,
        corporate_actions_path=None,
        run_id='qc_r1',
        config=cfg,
        config_hash=cfg_hash,
        config_payload=payload,
    )
    a2 = uq.audit_universe(
        universe_history_path=u_path,
        listings_master_path=l_path,
        calendar_path=c_path,
        corporate_actions_path=None,
        run_id='qc_r2',
        config=cfg,
        config_hash=cfg_hash,
        config_payload=payload,
    )
    pd.testing.assert_frame_equal(a1.daily, a2.daily)
    pd.testing.assert_frame_equal(a1.failures, a2.failures)
    assert a1.summary['gate_status'] == a2.summary['gate_status']
    assert a1.summary['n_fail'] == a2.summary['n_fail']
    assert a1.summary['n_warn'] == a2.summary['n_warn']

