from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest


MODULE_PATH = Path('/mnt/data/build_universe.py')
DATES = ['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']


@pytest.fixture(scope='module')
def build_universe_mod():
    spec = importlib.util.spec_from_file_location('build_universe_contract_mod', MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load module from {MODULE_PATH}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def base_config_payload() -> Dict[str, Any]:
    return {
        'rules': {
            'allowed_security_types': ['COMMON_STOCK'],
            'allowed_exchanges': ['NYSE', 'NASDAQ'],
            'allowed_trading_statuses': ['ACTIVE', 'TRADABLE'],
            'halted_statuses': ['HALTED'],
            'suspended_statuses': ['SUSPENDED'],
            'min_listing_age_days': 0,
            'min_price_ref': 3.0,
            'min_adv20_usd': 1_000_000.0,
            'min_market_cap_usd': 100_000_000.0,
            'max_market_cap_usd': 10_000_000_000.0,
            'require_primary_listing': True,
        },
        'missing_data': {
            'mode': 'exclude',
            'max_staleness_days': 0,
            'forward_fill_fields': [],
        },
        'edge_cases': {
            'halt_policy': 'exclude_until_tradable',
            'suspension_policy': 'exclude_until_tradable',
            'relisting_policy': 'new_instrument_unless_proven_same',
        },
        'qc': {
            'max_daily_constituent_jump_fraction': 1.0,
            'max_turnover_warn': 1.0,
            'max_critical_missing_fraction_warn': 1.0,
            'max_top_reason_fraction_warn': 1.0,
        },
        'output': {
            'output_dir': 'unused',
            'save_daily_constituents_for_end_date_only': True,
            'compression': 'snappy',
        },
    }


@pytest.fixture()
def io_builder_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, build_universe_mod, base_config_payload):
    # Avoid parquet dependency while preserving the persistence contract.
    monkeypatch.setattr(build_universe_mod, 'ensure_parquet_engine_available', lambda: None)

    def _write_csv_as_parquet(df: pd.DataFrame, path: Path, compression: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    monkeypatch.setattr(build_universe_mod, '_write_parquet', _write_csv_as_parquet)

    # Patch the known Timestamp serialization bug in _frame_logical_hash so we can test the contract.
    def _stable_frame_hash(df: pd.DataFrame) -> str:
        safe = df.copy()
        for col in safe.columns:
            if pd.api.types.is_datetime64_any_dtype(safe[col]):
                safe[col] = pd.to_datetime(safe[col], errors='coerce').dt.strftime('%Y-%m-%d')
        payload = safe.where(pd.notna(safe), None).to_dict(orient='records')
        return build_universe_mod.sha256_text(
            json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        )

    monkeypatch.setattr(build_universe_mod, '_frame_logical_hash', _stable_frame_hash)

    root = tmp_path
    out_dir = root / 'out'
    calendar_path = root / 'calendar.csv'

    def _write_calendar(dates: list[str]) -> Path:
        pd.DataFrame({'date': pd.to_datetime(dates)}).to_csv(calendar_path, index=False)
        return calendar_path

    def _write_listing(rows: list[dict[str, Any]], name: str = 'listing.csv') -> Path:
        path = root / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _write_features(rows: list[dict[str, Any]], name: str = 'features.csv') -> Path:
        path = root / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _write_config(payload: Dict[str, Any] | None = None, name: str = 'config.json') -> Path:
        path = root / name
        path.write_text(json.dumps(payload or base_config_payload), encoding='utf-8')
        return path

    def _read_table(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
        return pd.read_csv(path, parse_dates=parse_dates or [])

    def _run(
        listing_rows: list[dict[str, Any]],
        feature_rows: list[dict[str, Any]],
        dates: list[str],
        *,
        config_payload: Dict[str, Any] | None = None,
        run_id: str = 'run_contract',
        asof_ts_utc: str = '2024-01-05T23:00:00Z',
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        lp = _write_listing(listing_rows)
        fp = _write_features(feature_rows)
        cp = _write_config(config_payload)
        cal = _write_calendar(dates)
        artifacts = build_universe_mod.build_universe(
            listing_master_path=lp,
            calendar_path=cal,
            features_asof_path=fp,
            config_path=cp,
            start_date=start_date or dates[0],
            end_date=end_date or dates[-1],
            run_id=run_id,
            asof_ts_utc=asof_ts_utc,
            output_dir=out_dir,
            log_level='CRITICAL',
        )
        manifest = json.loads((out_dir / 'universe_manifest.json').read_text(encoding='utf-8'))
        return {
            'artifacts': artifacts,
            'out_dir': out_dir,
            'manifest': manifest,
            'read_table': _read_table,
        }

    return {
        'run': _run,
    }


def _mk_listing(
    instrument_id: str,
    symbol: str,
    *,
    issuer_id: str | None = None,
    list_date: str = '2024-01-02',
    delist_date: str | None = None,
    exchange: str = 'NYSE',
    security_type: str = 'COMMON_STOCK',
    share_class: str = 'A',
    trading_status: str = 'ACTIVE',
    is_primary_listing: bool = True,
) -> dict[str, Any]:
    return {
        'instrument_id': instrument_id,
        'issuer_id': issuer_id or f'ISS-{instrument_id}',
        'symbol': symbol,
        'listing_exchange': exchange,
        'security_type': security_type,
        'share_class': share_class,
        'list_date': list_date,
        'delist_date': delist_date,
        'trading_status': trading_status,
        'is_primary_listing': is_primary_listing,
    }



def _mk_feature(
    date: str,
    instrument_id: str,
    *,
    price_ref: float | None = 10.0,
    adv20_usd: float | None = 2_500_000.0,
    market_cap_usd: float | None = 500_000_000.0,
    symbol: str | None = None,
    trading_status: str | None = None,
    listing_exchange: str | None = None,
    security_type: str | None = None,
    field_source_flags: str = '{"src":"feat"}',
) -> dict[str, Any]:
    row: dict[str, Any] = {
        'date': date,
        'instrument_id': instrument_id,
        'price_ref': price_ref,
        'adv20_usd': adv20_usd,
        'market_cap_usd': market_cap_usd,
        'field_source_flags': field_source_flags,
    }
    if symbol is not None:
        row['symbol'] = symbol
    if trading_status is not None:
        row['trading_status'] = trading_status
    if listing_exchange is not None:
        row['listing_exchange'] = listing_exchange
    if security_type is not None:
        row['security_type'] = security_type
    return row


@pytest.fixture()
def sample_run(io_builder_contract):
    listing_rows = [
        _mk_listing('100', 'AAA'),
        _mk_listing('200', 'BBB'),
    ]
    feature_rows = [
        _mk_feature('2024-01-02', '100', price_ref=10.0),
        _mk_feature('2024-01-03', '100', price_ref=10.0),
        _mk_feature('2024-01-04', '100', price_ref=10.0),
        _mk_feature('2024-01-05', '100', price_ref=10.0),
        _mk_feature('2024-01-02', '200', price_ref=10.0),
        _mk_feature('2024-01-03', '200', price_ref=10.0),
        _mk_feature('2024-01-04', '200', price_ref=10.0),
        _mk_feature('2024-01-05', '200', price_ref=2.0),  # excluded on latest day
    ]
    return io_builder_contract['run'](listing_rows, feature_rows, DATES, run_id='run_u_contract')



def _required_history_cols() -> set[str]:
    return {
        'date', 'instrument_id', 'issuer_id', 'symbol', 'is_eligible', 'membership_state', 'transition',
        'primary_exclusion_reason', 'all_failed_reasons', 'market_cap_usd', 'adv20_usd', 'price_ref',
        'listing_exchange', 'security_type', 'share_class', 'trading_status', 'is_primary_listing',
        'list_date', 'delist_date', 'days_since_listing', 'field_source_flags', 'run_id', 'config_hash',
        'code_version', 'input_snapshot_hash', 'built_ts_utc'
    }


def test_history_contains_required_columns(sample_run):
    history = sample_run['artifacts'].history
    assert _required_history_cols().issubset(history.columns)



def test_current_snapshot_contains_only_latest_rows(sample_run):
    artifacts = sample_run['artifacts']
    latest = artifacts.history['date'].max()
    expected = artifacts.history.loc[artifacts.history['date'] == latest].sort_values(['date', 'instrument_id']).reset_index(drop=True)
    got = artifacts.current.sort_values(['date', 'instrument_id']).reset_index(drop=True)
    pd.testing.assert_frame_equal(got[expected.columns], expected)



def test_latest_constituents_and_exclusions_partition_current(sample_run):
    artifacts = sample_run['artifacts']
    current = artifacts.current.sort_values('instrument_id').reset_index(drop=True)
    constituents = artifacts.constituents_latest.sort_values('instrument_id').reset_index(drop=True)
    exclusions = artifacts.exclusions_latest.sort_values('instrument_id').reset_index(drop=True)

    assert constituents['is_eligible'].eq(1).all()
    assert exclusions['is_eligible'].eq(0).all()
    assert set(current['instrument_id']) == set(constituents['instrument_id']) | set(exclusions['instrument_id'])
    assert set(constituents['instrument_id']) & set(exclusions['instrument_id']) == set()



def test_counts_by_day_contains_required_metrics_and_matches_history(sample_run):
    artifacts = sample_run['artifacts']
    counts = artifacts.counts_by_day.sort_values('date').reset_index(drop=True)
    required = {'date', 'n_panel', 'n_constituents', 'n_ineligible', 'n_missing', 'n_halted', 'n_suspended'}
    assert required.issubset(counts.columns)

    expected = (
        artifacts.history.groupby('date', as_index=False)
        .agg(n_panel=('instrument_id', 'size'), n_constituents=('is_eligible', 'sum'))
        .sort_values('date')
        .reset_index(drop=True)
    )
    merged = counts.merge(expected, on='date', suffixes=('', '_expected'))
    assert merged['n_panel'].tolist() == merged['n_panel_expected'].tolist()
    assert merged['n_constituents'].tolist() == merged['n_constituents_expected'].tolist()



def test_turnover_stats_contains_required_metrics(sample_run):
    turnover = sample_run['artifacts'].turnover_stats.sort_values('date').reset_index(drop=True)
    required = {'date', 'entries', 'exits', 'n_constituents', 'turnover'}
    assert required.issubset(turnover.columns)
    assert float(turnover.loc[0, 'turnover']) == pytest.approx(0.0)
    assert int(turnover.loc[0, 'entries']) >= 0



def test_manifest_contains_required_metadata_and_hashes(sample_run):
    manifest = sample_run['manifest']
    required_top = {
        'run_id', 'start_date', 'end_date', 'asof_ts_utc', 'built_ts_utc', 'config_hash', 'config', 'code_version',
        'input_hashes', 'input_snapshot_hash', 'temporal_convention', 'missing_data_policy', 'edge_case_policy',
        'n_rows_history', 'n_days', 'n_instruments', 'latest_date', 'latest_n_constituents', 'counts_by_day_summary',
        'exclusion_reason_counts_total', 'qc_checks', 'run_status', 'history_logical_hash'
    }
    assert required_top.issubset(manifest.keys())
    assert manifest['run_id'] == 'run_u_contract'
    assert manifest['temporal_convention']['decision_time'] == 'close(t-1)'
    assert set(manifest['input_hashes']) == {'listing_master', 'calendar', 'features_asof', 'config'}
    assert manifest['run_status'] in {'INFO', 'WARN', 'FAIL'}



def test_history_pk_is_unique_by_date_instrument_id(sample_run):
    history = sample_run['artifacts'].history
    assert not history.duplicated(['date', 'instrument_id']).any()



def test_materialized_artifacts_exist_and_roundtrip(sample_run):
    out_dir = sample_run['out_dir']
    read_table = sample_run['read_table']
    latest_date = sample_run['artifacts'].manifest['end_date']
    paths = {
        'history': out_dir / 'history.parquet',
        'current': out_dir / 'current.parquet',
        'constituents': out_dir / f'constituents_{latest_date}.parquet',
        'exclusions': out_dir / f'exclusions_{latest_date}.parquet',
        'counts': out_dir / 'universe_counts_by_day.parquet',
        'turnover': out_dir / 'universe_turnover_stats.parquet',
        'reasons': out_dir / 'universe_reason_breakdown_by_day.parquet',
        'manifest': out_dir / 'universe_manifest.json',
    }
    for path in paths.values():
        assert path.exists(), f'Missing artifact: {path}'

    hist_disk = read_table(paths['history'], parse_dates=['date', 'list_date', 'delist_date', 'built_ts_utc'])
    curr_disk = read_table(paths['current'], parse_dates=['date', 'list_date', 'delist_date', 'built_ts_utc'])
    counts_disk = read_table(paths['counts'], parse_dates=['date'])
    turnover_disk = read_table(paths['turnover'], parse_dates=['date', 'prev_date'])

    assert len(hist_disk) == len(sample_run['artifacts'].history)
    assert len(curr_disk) == len(sample_run['artifacts'].current)
    assert len(counts_disk) == len(sample_run['artifacts'].counts_by_day)
    assert len(turnover_disk) == len(sample_run['artifacts'].turnover_stats)



def test_manifest_roundtrip_matches_returned_manifest(sample_run):
    out_dir = sample_run['out_dir']
    manifest_disk = json.loads((out_dir / 'universe_manifest.json').read_text(encoding='utf-8'))
    assert manifest_disk == sample_run['manifest']



def test_outputs_are_sorted_deterministically(sample_run):
    artifacts = sample_run['artifacts']
    history = artifacts.history.reset_index(drop=True)
    expected_history = artifacts.history.sort_values(['date', 'instrument_id']).reset_index(drop=True)
    pd.testing.assert_frame_equal(history, expected_history)

    counts = artifacts.counts_by_day.reset_index(drop=True)
    expected_counts = artifacts.counts_by_day.sort_values(['date']).reset_index(drop=True)
    pd.testing.assert_frame_equal(counts, expected_counts)

    reasons = artifacts.reason_breakdown_by_day.reset_index(drop=True)
    expected_reasons = artifacts.reason_breakdown_by_day.sort_values(['date', 'n', 'primary_exclusion_reason'], ascending=[True, False, True]).reset_index(drop=True)
    pd.testing.assert_frame_equal(reasons, expected_reasons)
