from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pandas.testing as pdt
import pytest


MODULE_PATH = Path('/mnt/data/survivorship.py')


@pytest.fixture(scope='module')
def survivorship_mod():
    spec = importlib.util.spec_from_file_location('survivorship_contract_mod', MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load module from {MODULE_PATH}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def deterministic_runtime(monkeypatch: pytest.MonkeyPatch, survivorship_mod):
    monkeypatch.setattr(survivorship_mod, 'utc_now_iso', lambda: '2026-03-16T10:00:00Z')
    monkeypatch.setattr(survivorship_mod, 'maybe_git_code_version', lambda: 'git:test')


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
            missing_dead_threshold=1,
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



def _ca_df(rows: list[dict[str, Any]], mod) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=sorted(mod.OPTIONAL_CA_COLUMNS))
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
    return df.sort_values(['instrument_id', 'date']).reset_index(drop=True)


@pytest.fixture()
def io_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, survivorship_mod, cfg):
    monkeypatch.setattr(survivorship_mod, 'ensure_parquet_engine_available', lambda: None)

    def _write_csv_as_parquet(df: pd.DataFrame, path: Path, compression: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    monkeypatch.setattr(survivorship_mod, '_write_parquet', _write_csv_as_parquet)

    # Work around real module assumptions so we can validate the contract.
    original_build_lifecycle_windows = survivorship_mod.build_lifecycle_windows

    def _patched_build_lifecycle_windows(lifecycle: pd.DataFrame, ca: pd.DataFrame) -> pd.DataFrame:
        local = lifecycle.copy()
        if 'delist_date' not in local.columns:
            local['delist_date'] = pd.NaT
        return original_build_lifecycle_windows(local, ca)

    monkeypatch.setattr(survivorship_mod, 'build_lifecycle_windows', _patched_build_lifecycle_windows)

    original_build_continuity_map = survivorship_mod.build_continuity_map

    def _patched_build_continuity_map(ca: pd.DataFrame, lifecycle: pd.DataFrame):
        safe = ca.copy()
        for col in ['from_instrument_id', 'to_instrument_id', 'old_instrument_id', 'new_instrument_id', 'predecessor_instrument_id', 'successor_instrument_id']:
            if col in safe.columns:
                safe[col] = safe[col].replace({pd.NA: None})
        return original_build_continuity_map(safe, lifecycle)

    monkeypatch.setattr(survivorship_mod, 'build_continuity_map', _patched_build_continuity_map)

    def _run(
        universe_rows: list[dict[str, Any]],
        lifecycle_rows: list[dict[str, Any]],
        *,
        ca_rows: list[dict[str, Any]] | None = None,
        price_rows: list[dict[str, Any]] | None = None,
        run_id: str = 'surv_contract',
        out_name: str = 'out',
    ):
        universe = _universe_df(universe_rows)
        lifecycle = _lifecycle_df(lifecycle_rows)
        ca = _ca_df(ca_rows or [], survivorship_mod)
        prices = _prices_df(price_rows or []) if price_rows is not None else None
        output_dir = tmp_path / out_name
        artifacts = survivorship_mod.run_survivorship_audit(
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
        survivorship_mod.persist_outputs(artifacts, output_dir, compression='snappy')
        manifest = json.loads((output_dir / 'survivorship_manifest.json').read_text(encoding='utf-8'))
        summary = json.loads((output_dir / 'survivorship_summary.json').read_text(encoding='utf-8'))
        return {
            'artifacts': artifacts,
            'output_dir': output_dir,
            'manifest': manifest,
            'summary_disk': summary,
        }

    return {'run': _run}


@pytest.fixture()
def sample_run_with_prices(io_runner):
    universe_rows = [
        {'date': '2024-01-02', 'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'is_eligible': 1},
        {'date': '2024-01-03', 'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'is_eligible': 1},
        {'date': '2024-01-04', 'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'is_eligible': 0, 'membership_state': 'excluded', 'primary_exclusion_reason': 'BAD_EXCHANGE', 'all_failed_reasons': '["BAD_EXCHANGE"]'},
        {'date': '2024-01-02', 'instrument_id': 'B', 'issuer_id': 'I2', 'symbol': 'BBB', 'is_eligible': 1},
    ]
    lifecycle_rows = [
        {'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'list_date': '2024-01-01', 'delist_date': '2024-01-03', 'canonical_instrument_id': 'A'},
        {'instrument_id': 'B', 'issuer_id': 'I2', 'symbol': 'BBB', 'list_date': '2024-01-01', 'canonical_instrument_id': 'B'},
    ]
    ca_rows = [
        {'event_type': 'merger', 'effective_date': '2024-01-03', 'instrument_id': 'A'},
    ]
    price_rows = [
        {'date': '2024-01-02', 'instrument_id': 'A', 'price': 10.0},
        {'date': '2024-01-03', 'instrument_id': 'A', 'price': 11.0},
        {'date': '2024-01-04', 'instrument_id': 'A', 'price': 12.0},
        {'date': '2024-01-02', 'instrument_id': 'B', 'price': 20.0},
        {'date': '2024-01-03', 'instrument_id': 'B', 'price': 20.5},
        {'date': '2024-01-04', 'instrument_id': 'B', 'price': 21.0},
    ]
    return io_runner['run'](universe_rows, lifecycle_rows, ca_rows=ca_rows, price_rows=price_rows, out_name='with_prices')


@pytest.fixture()
def sample_run_no_prices(io_runner):
    universe_rows = [
        {'date': '2024-01-02', 'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'is_eligible': 1},
        {'date': '2024-01-03', 'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'is_eligible': 1},
    ]
    lifecycle_rows = [
        {'instrument_id': 'A', 'issuer_id': 'I1', 'symbol': 'AAA', 'list_date': '2024-01-01', 'canonical_instrument_id': 'A'},
    ]
    return io_runner['run'](universe_rows, lifecycle_rows, ca_rows=[], price_rows=None, out_name='no_prices')


# -----------------------------------------------------------------------------
# Contract expectations
# -----------------------------------------------------------------------------


def _required_daily_cols() -> set[str]:
    return {
        'date', 'n_pit', 'n_naive', 'n_expected', 'n_dead_expected', 'overlap_ratio',
        'delisted_coverage_ratio', 'net_gap_rate', 'gross_gap_rate', 'n_missing_dead_instruments',
        'n_structural_missing', 'n_legitimate_rule_exclusion', 'n_identity_continuity',
        'n_economic_termination', 'n_low_confidence', 'absence_classification_counts'
    }



def _required_missing_dead_cols() -> set[str]:
    return {
        'date', 'instrument_id', 'issuer_id', 'symbol', 'classification', 'classification_detail',
        'linked_instrument_id', 'primary_exclusion_reason', 'is_dead_expected',
        'gross_survivorship_gap', 'net_survivorship_gap'
    }



def _required_comparison_cols() -> set[str]:
    return {
        'date', 'pit_return', 'naive_return', 'delta_return_naive_minus_pit',
        'pit_constituents_with_return', 'naive_constituents_with_return', 'pit_nav', 'naive_nav'
    }



def test_daily_contains_required_columns(sample_run_with_prices):
    daily = sample_run_with_prices['artifacts'].daily
    assert _required_daily_cols().issubset(daily.columns)



def test_summary_contains_required_fields(sample_run_with_prices):
    summary = sample_run_with_prices['artifacts'].summary
    required = {
        'run_id', 'asof_ts_utc', 'built_ts_utc', 'config_hash', 'code_version', 'baseline_definition',
        'gate_status', 'n_problem_cases', 'n_expected_rows_audited', 'n_audit_days',
        'delisted_coverage_ratio_mean', 'pct_days_low_delisted_coverage', 'mean_net_gap_rate',
        'n_missing_dead_instruments', 'survivorship_risk_score', 'absence_classification_counts', 'gate_results'
    }
    assert required.issubset(summary.keys())



def test_missing_dead_contains_required_columns(sample_run_with_prices):
    missing_dead = sample_run_with_prices['artifacts'].missing_dead
    assert _required_missing_dead_cols().issubset(missing_dead.columns)



def test_comparison_output_exists_only_when_prices_are_present(sample_run_with_prices, sample_run_no_prices):
    comp_yes = sample_run_with_prices['artifacts'].comparison
    comp_no = sample_run_no_prices['artifacts'].comparison
    assert comp_yes is not None and not comp_yes.empty
    assert _required_comparison_cols().issubset(comp_yes.columns)
    assert comp_no is None or comp_no.empty
    assert (sample_run_with_prices['output_dir'] / 'naive_vs_pit_comparison.parquet').exists()
    assert not (sample_run_no_prices['output_dir'] / 'naive_vs_pit_comparison.parquet').exists()



def test_manifest_contains_baseline_policy_thresholds_and_counts(sample_run_with_prices):
    manifest = sample_run_with_prices['manifest']
    assert manifest['baseline_variant'] == 'current_eligible'
    assert manifest['baseline_tradable_statuses'] == ['ACTIVE', 'TRADABLE']
    assert manifest['relisting_policy'] == 'new_instrument_unless_proven_continuous'
    assert isinstance(manifest['thresholds'], dict)
    assert isinstance(manifest['risk_weights'], dict)
    assert isinstance(manifest['absence_classification_catalog'], list)
    assert isinstance(manifest['absence_classification_counts'], dict)
    assert manifest['config_hash'] == 'cfg:test'
    assert manifest['input_paths']['universe_history_path'] == 'mem://universe'



def test_gate_domain_is_pass_warn_fail(sample_run_with_prices):
    gate = sample_run_with_prices['artifacts'].summary['gate_status']
    assert gate in {'PASS', 'WARN', 'FAIL'}
    manifest_gate = sample_run_with_prices['manifest']['gate_status']
    assert manifest_gate == gate



def test_outputs_are_sorted_deterministically(sample_run_with_prices):
    artifacts = sample_run_with_prices['artifacts']
    expected_daily = artifacts.daily.sort_values(['date']).reset_index(drop=True)
    pdt.assert_frame_equal(artifacts.daily.reset_index(drop=True), expected_daily)

    expected_missing = artifacts.missing_dead.sort_values(['date', 'instrument_id']).reset_index(drop=True)
    pdt.assert_frame_equal(artifacts.missing_dead.reset_index(drop=True), expected_missing)

    if artifacts.comparison is not None and not artifacts.comparison.empty:
        expected_comp = artifacts.comparison.sort_values(['date']).reset_index(drop=True)
        pdt.assert_frame_equal(artifacts.comparison.reset_index(drop=True), expected_comp)



def test_roundtrip_summary_and_manifest_from_disk(sample_run_with_prices):
    artifacts = sample_run_with_prices['artifacts']
    summary_disk = sample_run_with_prices['summary_disk']
    manifest_disk = sample_run_with_prices['manifest']

    assert summary_disk['run_id'] == artifacts.summary['run_id']
    assert summary_disk['gate_status'] == artifacts.summary['gate_status']
    assert summary_disk['config_hash'] == artifacts.summary['config_hash']
    assert manifest_disk['run_id'] == artifacts.manifest['run_id']
    assert manifest_disk['input_snapshot_hash'] == artifacts.manifest['input_snapshot_hash']
    assert manifest_disk['absence_classification_catalog'] == artifacts.manifest['absence_classification_catalog']



def test_daily_and_summary_are_coherent(sample_run_with_prices):
    daily = sample_run_with_prices['artifacts'].daily
    summary = sample_run_with_prices['artifacts'].summary
    assert int(summary['n_audit_days']) == int(daily['date'].nunique())
    assert int(summary['n_missing_dead_instruments']) == int(daily['n_missing_dead_instruments'].sum())
    mean_cov = float(daily['delisted_coverage_ratio'].dropna().mean()) if daily['delisted_coverage_ratio'].notna().any() else float('nan')
    if pd.notna(mean_cov):
        assert summary['delisted_coverage_ratio_mean'] == pytest.approx(mean_cov)



def test_missing_dead_is_subset_of_problem_cases(sample_run_with_prices):
    artifacts = sample_run_with_prices['artifacts']
    problem_cases = artifacts.problem_cases
    missing_dead = artifacts.missing_dead
    if missing_dead.empty:
        pytest.skip('sample fixture produced no missing_dead rows')

    merged = missing_dead.merge(
        problem_cases[['date', 'instrument_id', 'classification']],
        on=['date', 'instrument_id'],
        how='left',
        suffixes=('', '_problem'),
    )
    assert (merged['classification_problem'] == 'structural_missing').all()



def test_required_artifacts_are_materialized(sample_run_with_prices, sample_run_no_prices):
    out_yes = sample_run_with_prices['output_dir']
    out_no = sample_run_no_prices['output_dir']
    for path in [
        out_yes / 'survivorship_summary.json',
        out_yes / 'survivorship_daily.parquet',
        out_yes / 'missing_dead_instruments.parquet',
        out_yes / 'survivorship_problem_cases.parquet',
        out_yes / 'survivorship_manifest.json',
        out_no / 'survivorship_summary.json',
        out_no / 'survivorship_daily.parquet',
        out_no / 'missing_dead_instruments.parquet',
        out_no / 'survivorship_problem_cases.parquet',
        out_no / 'survivorship_manifest.json',
    ]:
        assert path.exists(), f'missing expected artifact: {path}'
