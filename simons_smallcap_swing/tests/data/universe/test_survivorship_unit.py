from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pandas.testing as pdt
import pytest


MODULE_PATH = Path('/mnt/data/survivorship.py')


@pytest.fixture(scope='module')
def survivorship_mod():
    spec = importlib.util.spec_from_file_location('survivorship_mod', MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load module from {MODULE_PATH}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def cfg(survivorship_mod):
    mod = survivorship_mod
    return mod.AuditConfig(
        baseline=mod.BaselineConfig(variant='current_eligible', tradable_statuses=('ACTIVE', 'TRADABLE')),
        lifecycle=mod.LifecycleConfig(
            required_security_types=tuple(),
            allowed_exchanges=tuple(),
            excluded_share_classes=tuple(),
            include_statuses=tuple(),
            relisting_policy='new_instrument_unless_proven_continuous',
            low_confidence_when_missing_lifecycle=True,
        ),
        thresholds=mod.ThresholdConfig(
            low_delisted_coverage_threshold=0.95,
            mean_net_gap_rate_threshold=0.01,
            missing_dead_threshold=0,
            cagr_diff_threshold=0.50,
        ),
        risk=mod.RiskWeightsConfig(
            r1_weight=0.30,
            r2_weight=0.20,
            r3_weight=0.25,
            r4_weight=0.15,
            r5_weight=0.10,
            tau_gap=0.01,
            tau_missing=1,
            tau_cagr=0.50,
        ),
        output=mod.OutputConfig(output_dir='unused', compression='snappy', include_problem_cases_all=True),
    )


@pytest.fixture(autouse=True)
def deterministic_runtime(monkeypatch: pytest.MonkeyPatch, survivorship_mod):
    monkeypatch.setattr(survivorship_mod, 'utc_now_iso', lambda: '2026-03-16T10:00:00Z')
    monkeypatch.setattr(survivorship_mod, 'maybe_git_code_version', lambda: 'git:test')


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _universe_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df['instrument_id'] = df['instrument_id'].astype(str)
        df['symbol'] = df['symbol'].astype(str)
        if 'issuer_id' not in df.columns:
            df['issuer_id'] = pd.NA
        else:
            df['issuer_id'] = df['issuer_id'].astype(str)
        if 'all_failed_reasons' not in df.columns:
            df['all_failed_reasons'] = None
        if 'membership_state' not in df.columns:
            df['membership_state'] = 'eligible'
        if 'primary_exclusion_reason' not in df.columns:
            df['primary_exclusion_reason'] = None
    return df.sort_values(['date', 'instrument_id']).reset_index(drop=True)



def _lifecycle_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        for col in ['list_date', 'delist_date', 'ticker_start_date', 'ticker_end_date']:
            if col not in df.columns:
                df[col] = pd.NaT
            df[col] = pd.to_datetime(df[col])
        df['instrument_id'] = df['instrument_id'].astype(str)
        df['symbol'] = df['symbol'].astype(str)
        if 'issuer_id' not in df.columns:
            df['issuer_id'] = pd.NA
        else:
            df['issuer_id'] = df['issuer_id'].astype(str)
        if 'canonical_instrument_id' not in df.columns:
            df['canonical_instrument_id'] = df['instrument_id']
    return df.reset_index(drop=True)



def _ca_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[
            'event_type', 'effective_date', 'instrument_id', 'from_instrument_id', 'to_instrument_id',
            'old_instrument_id', 'new_instrument_id', 'predecessor_instrument_id', 'successor_instrument_id',
            'linked_corporate_action_id'
        ])
    df = pd.DataFrame(rows)
    df['event_type'] = df['event_type'].astype(str).str.lower()
    df['effective_date'] = pd.to_datetime(df['effective_date'])
    for col in [
        'instrument_id', 'from_instrument_id', 'to_instrument_id', 'old_instrument_id',
        'new_instrument_id', 'predecessor_instrument_id', 'successor_instrument_id',
        'linked_corporate_action_id'
    ]:
        if col not in df.columns:
            df[col] = None
        else:
            df[col] = df[col].astype(str)
    return df.reset_index(drop=True)



def _prices_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df['instrument_id'] = df['instrument_id'].astype(str)
        if 'price' not in df.columns:
            raise AssertionError('price rows must include price')
    return df.sort_values(['instrument_id', 'date']).reset_index(drop=True)



def _prepare_context(mod, universe_rows, lifecycle_rows, ca_rows=None, config=None):
    config = config or mod.AuditConfig()
    universe = _universe_df(universe_rows)
    lifecycle = _lifecycle_df(lifecycle_rows)
    ca = _ca_df(ca_rows or [])
    return mod.prepare_context(universe_history=universe, lifecycle=lifecycle, ca=ca, config=config)



def _run_audit(mod, cfg, universe_rows, lifecycle_rows, ca_rows=None, price_rows=None, run_id='pytest_surv'):
    universe = _universe_df(universe_rows)
    lifecycle = _lifecycle_df(lifecycle_rows)
    ca = _ca_df(ca_rows or [])
    prices = _prices_df(price_rows or []) if price_rows is not None else None
    return mod.run_survivorship_audit(
        universe_history=universe,
        lifecycle_master=lifecycle,
        corporate_actions=ca,
        prices=prices,
        config=cfg,
        config_hash='cfg:test',
        run_id=run_id,
        asof_ts_utc='2026-03-16T10:00:00Z',
        input_paths={
            'universe_history_path': 'mem://universe',
            'lifecycle_master_path': 'mem://lifecycle',
            'corporate_actions_path': 'mem://ca',
            'prices_path': 'mem://prices',
        },
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_ticker_change_is_classified_as_identity_continuity(survivorship_mod, cfg):
    mod = survivorship_mod
    ctx = _prepare_context(
        mod,
        universe_rows=[
            {'date': '2024-01-01', 'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'is_eligible': 1},
            {'date': '2024-01-02', 'instrument_id': 'B', 'issuer_id': 'I1', 'symbol': 'BBB', 'is_eligible': 1},
        ],
        lifecycle_rows=[
            {'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'list_date': '2024-01-01', 'canonical_instrument_id': 'CAN1'},
            {'instrument_id': 'B', 'issuer_id': 'I1', 'symbol': 'BBB', 'list_date': '2024-01-02', 'canonical_instrument_id': 'CAN1'},
        ],
        ca_rows=[
            {'event_type': 'ticker_change', 'effective_date': '2024-01-02', 'from_instrument_id': 'A', 'to_instrument_id': 'B'}
        ],
        config=cfg,
    )

    classification, detail, linked_id, reason = mod.classify_absence(pd.Timestamp('2024-01-02'), 'A', ctx, cfg)
    assert classification == mod.AbsenceClassification.IDENTITY_CONTINUITY
    assert linked_id == 'B'
    assert 'active instrument B' in detail
    assert reason is None



def test_merger_terminal_is_classified_as_economic_termination(survivorship_mod, cfg):
    mod = survivorship_mod
    ctx = _prepare_context(
        mod,
        universe_rows=[
            {'date': '2024-01-01', 'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'is_eligible': 1},
        ],
        lifecycle_rows=[
            {'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'list_date': '2024-01-01'},
        ],
        ca_rows=[
            {'event_type': 'merger', 'effective_date': '2024-01-02', 'instrument_id': 'A'}
        ],
        config=cfg,
    )

    classification, detail, linked_id, reason = mod.classify_absence(pd.Timestamp('2024-01-03'), 'A', ctx, cfg)
    assert classification == mod.AbsenceClassification.ECONOMIC_TERMINATION
    assert 'after economic death date 2024-01-02' in detail
    assert linked_id is None
    assert reason is None



def test_legitimate_rule_exclusion_is_recognized_from_ineligible_row(survivorship_mod, cfg):
    mod = survivorship_mod
    ctx = _prepare_context(
        mod,
        universe_rows=[
            {
                'date': '2024-01-02', 'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'is_eligible': 0,
                'membership_state': 'excluded', 'primary_exclusion_reason': 'BAD_EXCHANGE',
                'all_failed_reasons': '["BAD_EXCHANGE", "LOW_LIQUIDITY"]',
            },
        ],
        lifecycle_rows=[
            {'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'list_date': '2024-01-01'},
        ],
        config=cfg,
    )

    classification, detail, linked_id, reason = mod.classify_absence(pd.Timestamp('2024-01-02'), 'A', ctx, cfg)
    assert classification == mod.AbsenceClassification.LEGITIMATE_RULE_EXCLUSION
    assert reason == 'BAD_EXCHANGE'
    assert 'reason BAD_EXCHANGE' in detail
    assert linked_id is None



def test_low_confidence_is_used_when_lifecycle_evidence_is_missing(survivorship_mod, cfg):
    mod = survivorship_mod
    ctx = _prepare_context(
        mod,
        universe_rows=[
            {'date': '2024-01-01', 'instrument_id': 'B', 'issuer_id': 'I2', 'symbol': 'BBB', 'is_eligible': 1},
        ],
        lifecycle_rows=[
            {'instrument_id': 'B', 'issuer_id': 'I2', 'symbol': 'BBB', 'list_date': '2024-01-01'},
        ],
        config=cfg,
    )

    classification, detail, linked_id, reason = mod.classify_absence(pd.Timestamp('2024-01-01'), 'X', ctx, cfg)
    assert classification == mod.AbsenceClassification.LOW_CONFIDENCE
    assert 'lifecycle evidence missing' in detail
    assert linked_id is None
    assert reason is None



def test_structural_missing_when_expected_but_unexplained(survivorship_mod, cfg):
    mod = survivorship_mod
    ctx = _prepare_context(
        mod,
        universe_rows=[
            {'date': '2024-01-01', 'instrument_id': 'B', 'issuer_id': 'I2', 'symbol': 'BBB', 'is_eligible': 1},
            {'date': '2024-01-02', 'instrument_id': 'B', 'issuer_id': 'I2', 'symbol': 'BBB', 'is_eligible': 1},
        ],
        lifecycle_rows=[
            {'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'list_date': '2024-01-01'},
            {'instrument_id': 'B', 'issuer_id': 'I2', 'symbol': 'BBB', 'list_date': '2024-01-01'},
        ],
        config=cfg,
    )

    classification, detail, linked_id, reason = mod.classify_absence(pd.Timestamp('2024-01-02'), 'A', ctx, cfg)
    assert classification == mod.AbsenceClassification.STRUCTURAL_MISSING
    assert 'missing from PIT eligible set without valid explanation' in detail
    assert linked_id is None
    assert reason is None



def test_delisted_coverage_ratio_and_missing_dead_are_computed_correctly(survivorship_mod, cfg):
    mod = survivorship_mod
    artifacts = _run_audit(
        mod,
        cfg,
        universe_rows=[
            {'date': '2024-01-01', 'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'is_eligible': 1},
            {'date': '2024-01-01', 'instrument_id': 'B', 'issuer_id': 'I2', 'symbol': 'BBB', 'is_eligible': 1},
            {
                'date': '2024-01-02', 'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'is_eligible': 0,
                'membership_state': 'excluded', 'primary_exclusion_reason': 'BAD_EXCHANGE',
                'all_failed_reasons': 'BAD_EXCHANGE',
            },
            # B intentionally absent on 2024-01-02 -> structural missing dead name
        ],
        lifecycle_rows=[
            {'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'list_date': '2024-01-01', 'delist_date': '2024-01-02'},
            {'instrument_id': 'B', 'issuer_id': 'I2', 'symbol': 'BBB', 'list_date': '2024-01-01', 'delist_date': '2024-01-02'},
        ],
        ca_rows=[],
    )

    daily = artifacts.daily.set_index('date')
    row = daily.loc[pd.Timestamp('2024-01-02')]
    assert row['n_dead_expected'] == 2
    assert row['n_missing_dead_instruments'] == 1
    assert row['delisted_coverage_ratio'] == pytest.approx(0.5)
    assert row['n_legitimate_rule_exclusion'] == 1
    assert row['n_structural_missing'] == 1

    missing_dead = artifacts.missing_dead
    assert set(missing_dead['instrument_id']) == {'B'}



def test_apply_gates_blocks_when_coverage_and_gap_are_bad(survivorship_mod, cfg):
    mod = survivorship_mod
    summary_metrics = {
        'delisted_coverage_ratio_mean': 0.50,
        'pct_days_low_delisted_coverage': 0.50,
        'mean_net_gap_rate': 0.20,
        'n_missing_dead_instruments': 3,
        'CAGR_diff_pit_vs_naive': 0.80,
    }
    gates = mod.apply_gates(summary_metrics, cfg, has_prices=True)
    by_name = {g.name: g for g in gates}
    assert by_name['delisted_coverage_ratio_mean'].severity == mod.Severity.FAIL
    assert by_name['pct_days_low_delisted_coverage'].severity == mod.Severity.FAIL
    assert by_name['mean_net_gap_rate'].severity == mod.Severity.FAIL
    assert by_name['n_missing_dead_instruments'].severity == mod.Severity.FAIL
    assert by_name['abs_CAGR_diff_pit_vs_naive'].severity == mod.Severity.FAIL
    assert mod.max_severity(g.severity for g in gates) == mod.Severity.FAIL



def test_economic_comparison_and_summary_metrics_are_produced_when_prices_exist(survivorship_mod, cfg):
    mod = survivorship_mod
    artifacts = _run_audit(
        mod,
        cfg,
        universe_rows=[
            {'date': '2024-01-01', 'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'is_eligible': 1},
            {'date': '2024-01-02', 'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'is_eligible': 1},
            {'date': '2024-01-03', 'instrument_id': 'B', 'issuer_id': 'I2', 'symbol': 'BBB', 'is_eligible': 1},
        ],
        lifecycle_rows=[
            {'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'list_date': '2024-01-01'},
            {'instrument_id': 'B', 'issuer_id': 'I2', 'symbol': 'BBB', 'list_date': '2024-01-01'},
        ],
        price_rows=[
            {'date': '2024-01-01', 'instrument_id': 'A', 'price': 100.0},
            {'date': '2024-01-02', 'instrument_id': 'A', 'price': 120.0},
            {'date': '2024-01-03', 'instrument_id': 'A', 'price': 144.0},
            {'date': '2024-01-01', 'instrument_id': 'B', 'price': 100.0},
            {'date': '2024-01-02', 'instrument_id': 'B', 'price': 90.0},
            {'date': '2024-01-03', 'instrument_id': 'B', 'price': 81.0},
        ],
    )

    comp = artifacts.comparison
    assert comp is not None and not comp.empty
    assert list(comp.columns) == [
        'date', 'pit_return', 'naive_return', 'delta_return_naive_minus_pit',
        'pit_constituents_with_return', 'naive_constituents_with_return', 'pit_nav', 'naive_nav'
    ]
    jan2 = comp.set_index('date').loc[pd.Timestamp('2024-01-02')]
    assert jan2['pit_return'] == pytest.approx(0.20)
    assert jan2['naive_return'] == pytest.approx(-0.10)
    assert pd.notna(artifacts.summary['CAGR_pit'])
    assert pd.notna(artifacts.summary['CAGR_naive'])
    assert pd.notna(artifacts.summary['CAGR_diff_pit_vs_naive'])



def test_run_survivorship_audit_is_reproducible_under_same_inputs(survivorship_mod, cfg):
    mod = survivorship_mod
    universe_rows = [
        {'date': '2024-01-01', 'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'is_eligible': 1},
        {'date': '2024-01-02', 'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'is_eligible': 0,
         'membership_state': 'excluded', 'primary_exclusion_reason': 'BAD_EXCHANGE', 'all_failed_reasons': 'BAD_EXCHANGE'},
        {'date': '2024-01-01', 'instrument_id': 'B', 'issuer_id': 'I2', 'symbol': 'BBB', 'is_eligible': 1},
    ]
    lifecycle_rows = [
        {'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'list_date': '2024-01-01', 'delist_date': '2024-01-02'},
        {'instrument_id': 'B', 'issuer_id': 'I2', 'symbol': 'BBB', 'list_date': '2024-01-01', 'delist_date': '2024-01-02'},
    ]

    first = _run_audit(mod, cfg, universe_rows=universe_rows, lifecycle_rows=lifecycle_rows, run_id='same_run')
    second = _run_audit(mod, cfg, universe_rows=copy.deepcopy(universe_rows), lifecycle_rows=copy.deepcopy(lifecycle_rows), run_id='same_run')

    pdt.assert_frame_equal(first.daily, second.daily)
    pdt.assert_frame_equal(first.problem_cases, second.problem_cases)
    pdt.assert_frame_equal(first.missing_dead, second.missing_dead)
    assert first.comparison is None and second.comparison is None
    assert first.summary == second.summary
    assert first.manifest == second.manifest
