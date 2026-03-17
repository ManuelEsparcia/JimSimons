from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest


MODULE_PATH = Path('/mnt/data/build_universe.py')


@pytest.fixture(scope='module')
def build_universe_mod():
    spec = importlib.util.spec_from_file_location('build_universe_mod', MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load module from {MODULE_PATH}')
    mod = importlib.util.module_from_spec(spec)
    import sys
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
def io_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, build_universe_mod, base_config_payload):
    monkeypatch.setattr(build_universe_mod, 'persist_outputs', lambda *args, **kwargs: None)

    def _stable_frame_hash(df: pd.DataFrame) -> str:
        safe = df.copy()
        for col in safe.columns:
            if pd.api.types.is_datetime64_any_dtype(safe[col]):
                safe[col] = pd.to_datetime(safe[col], errors='coerce').dt.strftime('%Y-%m-%d')
        return build_universe_mod.sha256_text(
            json.dumps(safe.fillna('__NA__').to_dict(orient='records'), sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        )

    monkeypatch.setattr(build_universe_mod, '_frame_logical_hash', _stable_frame_hash)

    root = tmp_path
    calendar_path = root / 'calendar.csv'
    config_path = root / 'config.json'

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

    def _run(
        listing_rows: list[dict[str, Any]],
        feature_rows: list[dict[str, Any]],
        dates: list[str],
        *,
        config_payload: Dict[str, Any] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        run_id: str = 'run_test',
        asof_ts_utc: str = '2024-01-10T23:00:00Z',
        listing_name: str = 'listing.csv',
        feature_name: str = 'features.csv',
        config_name: str = 'config.json',
    ):
        lp = _write_listing(listing_rows, listing_name)
        fp = _write_features(feature_rows, feature_name)
        cp = _write_config(config_payload, config_name)
        cal = _write_calendar(dates)
        return build_universe_mod.build_universe(
            listing_master_path=lp,
            calendar_path=cal,
            features_asof_path=fp,
            config_path=cp,
            start_date=start_date or dates[0],
            end_date=end_date or dates[-1],
            run_id=run_id,
            asof_ts_utc=asof_ts_utc,
            output_dir=root / 'out',
            log_level='CRITICAL',
        )

    return {
        'root': root,
        'run': _run,
        'write_config': _write_config,
        'write_calendar': _write_calendar,
        'write_listing': _write_listing,
        'write_features': _write_features,
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
    effective_from: str | None = None,
    effective_to: str | None = None,
) -> dict[str, Any]:
    row = {
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
    if effective_from is not None or effective_to is not None:
        row['effective_from'] = effective_from
        row['effective_to'] = effective_to
    return row



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


DATES_5 = ['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-08']


def test_builds_complete_panel_over_calendar_and_instruments(io_builder):
    listing_rows = [
        _mk_listing('100', 'AAA', list_date='2024-01-02'),
        _mk_listing('200', 'BBB', list_date='2024-01-04'),
    ]
    feature_rows = [
        _mk_feature(d, '100')
        for d in DATES_5
    ] + [
        _mk_feature(d, '200')
        for d in DATES_5[2:]
    ]

    artifacts = io_builder['run'](listing_rows, feature_rows, DATES_5)
    history = artifacts.history

    assert len(history) == 8
    assert history['instrument_id'].nunique() == 2
    assert history['date'].min() == pd.Timestamp('2024-01-02')
    assert history['date'].max() == pd.Timestamp('2024-01-08')
    assert not history.duplicated(['date', 'instrument_id']).any()


def test_no_rows_before_list_date_and_no_rows_after_delist_date(io_builder):
    listing_rows = [
        _mk_listing('100', 'AAA', list_date='2024-01-04', delist_date='2024-01-05'),
    ]
    feature_rows = [
        _mk_feature('2024-01-04', '100'),
        _mk_feature('2024-01-05', '100'),
    ]
    artifacts = io_builder['run'](listing_rows, feature_rows, DATES_5)
    history = artifacts.history.sort_values('date').reset_index(drop=True)

    assert history['date'].tolist() == [pd.Timestamp('2024-01-04'), pd.Timestamp('2024-01-05')]
    assert history['is_eligible'].tolist() == [1, 1]


def test_primary_exclusion_reason_is_null_iff_eligible(io_builder):
    listing_rows = [
        _mk_listing('100', 'AAA'),
        _mk_listing('200', 'BBB'),
    ]
    feature_rows = [
        _mk_feature('2024-01-02', '100', price_ref=10.0),
        _mk_feature('2024-01-03', '100', price_ref=10.0),
        _mk_feature('2024-01-02', '200', price_ref=2.0),
        _mk_feature('2024-01-03', '200', price_ref=2.0),
    ]
    artifacts = io_builder['run'](
        listing_rows,
        feature_rows,
        ['2024-01-02', '2024-01-03'],
        start_date='2024-01-02',
        end_date='2024-01-03',
    )
    history = artifacts.history.sort_values(['instrument_id', 'date']).reset_index(drop=True)

    eligible = history['is_eligible'] == 1
    assert history.loc[eligible, 'primary_exclusion_reason'].isna().all()
    assert history.loc[~eligible, 'primary_exclusion_reason'].notna().all()


def test_all_failed_reasons_contains_all_triggers_in_filter_order(io_builder):
    listing_rows = [
        _mk_listing(
            '100',
            'AAA',
            exchange='OTC',
            trading_status='HALTED',
            share_class='B',
            is_primary_listing=False,
        )
    ]
    config = io_builder['write_config']  # just to satisfy type checker when editing
    custom_config = {
        'rules': {
            'allowed_security_types': ['COMMON_STOCK'],
            'allowed_exchanges': ['NYSE', 'NASDAQ'],
            'allowed_trading_statuses': ['ACTIVE'],
            'halted_statuses': ['HALTED'],
            'suspended_statuses': ['SUSPENDED'],
            'min_listing_age_days': 10,
            'min_price_ref': 3.0,
            'min_adv20_usd': 1_000_000.0,
            'min_market_cap_usd': 100_000_000.0,
            'max_market_cap_usd': 5_000_000_000.0,
            'require_primary_listing': True,
            'allowed_share_classes': [],
            'excluded_share_classes': ['B'],
            'excluded_instrument_ids': [],
            'excluded_symbols': [],
        },
        'missing_data': {'mode': 'exclude', 'max_staleness_days': 0, 'forward_fill_fields': []},
        'edge_cases': {'halt_policy': 'exclude_until_tradable', 'suspension_policy': 'exclude_until_tradable', 'relisting_policy': 'new_instrument_unless_proven_same'},
        'qc': {'max_daily_constituent_jump_fraction': 1.0, 'max_turnover_warn': 1.0, 'max_critical_missing_fraction_warn': 1.0, 'max_top_reason_fraction_warn': 1.0},
        'output': {'output_dir': 'unused', 'save_daily_constituents_for_end_date_only': True, 'compression': 'snappy'},
    }
    feature_rows = [
        _mk_feature('2024-01-02', '100', price_ref=2.5, adv20_usd=100_000.0, market_cap_usd=50_000_000.0),
    ]

    artifacts = io_builder['run'](
        listing_rows,
        feature_rows,
        ['2024-01-02'],
        start_date='2024-01-02',
        end_date='2024-01-02',
        config_payload=custom_config,
    )
    row = artifacts.history.iloc[0]
    reasons = row['all_failed_reasons'].split('|')

    assert reasons[0] == 'BAD_EXCHANGE'
    assert reasons == [
        'BAD_EXCHANGE',
        'HALTED',
        'TOO_YOUNG_SINCE_LISTING',
        'PRICE_BELOW_MIN',
        'ADV20_BELOW_MIN',
        'MCAP_OUT_OF_BAND',
        'SPECIAL_RULE_EXCLUSION',
    ]
    assert row['primary_exclusion_reason'] == 'BAD_EXCHANGE'
    assert row['membership_state'] == 'ineligible_rule'


def test_membership_state_maps_missing_halted_and_suspended_correctly(io_builder):
    listing_rows = [
        _mk_listing('100', 'AAA', trading_status='ACTIVE'),
        _mk_listing('200', 'BBB', trading_status='HALTED'),
        _mk_listing('300', 'CCC', trading_status='SUSPENDED'),
    ]
    feature_rows = [
        _mk_feature('2024-01-02', '100', price_ref=None),
        _mk_feature('2024-01-02', '200', price_ref=10.0),
        _mk_feature('2024-01-02', '300', price_ref=10.0),
    ]
    artifacts = io_builder['run'](
        listing_rows,
        feature_rows,
        ['2024-01-02'],
        start_date='2024-01-02',
        end_date='2024-01-02',
    )
    history = artifacts.history.set_index('instrument_id')

    assert history.loc['100', 'membership_state'] == 'ineligible_missing'
    assert history.loc['200', 'membership_state'] == 'ineligible_halted'
    assert history.loc['300', 'membership_state'] == 'ineligible_suspended'


def test_transition_enter_stay_in_exit_and_state_change_outside_universe(io_builder):
    listing_rows = [
        _mk_listing('100', 'AAA'),
        _mk_listing('200', 'BBB', trading_status='HALTED'),
    ]
    feature_rows = [
        _mk_feature('2024-01-02', '100', price_ref=10.0),
        _mk_feature('2024-01-03', '100', price_ref=10.0),
        _mk_feature('2024-01-04', '100', price_ref=2.0),
        _mk_feature('2024-01-02', '200', price_ref=10.0),
        _mk_feature('2024-01-03', '200', price_ref=10.0, trading_status='SUSPENDED'),
    ]
    artifacts = io_builder['run'](
        listing_rows,
        feature_rows,
        ['2024-01-02', '2024-01-03', '2024-01-04'],
        start_date='2024-01-02',
        end_date='2024-01-04',
    )
    h = artifacts.history.sort_values(['instrument_id', 'date']).reset_index(drop=True)

    aaa = h[h['instrument_id'] == '100']
    assert aaa['transition'].tolist() == ['ENTER', 'STAY_IN', 'EXIT']

    bbb = h[h['instrument_id'] == '200']
    assert bbb['transition'].tolist() == ['STAY_OUT', 'STATE_CHANGE_OUTSIDE_UNIVERSE', 'STATE_CHANGE_OUTSIDE_UNIVERSE']


def test_turnover_is_computed_at_instrument_id_level_not_ticker_level(io_builder):
    listing_rows = [
        _mk_listing('100', 'AAA', effective_from='2024-01-02', effective_to='2024-01-03'),
        _mk_listing('100', 'AAB', effective_from='2024-01-04', effective_to='2024-01-08'),
    ]
    feature_rows = [
        _mk_feature('2024-01-02', '100'),
        _mk_feature('2024-01-03', '100'),
        _mk_feature('2024-01-04', '100', symbol='AAB'),
        _mk_feature('2024-01-05', '100', symbol='AAB'),
        _mk_feature('2024-01-08', '100', symbol='AAB'),
    ]
    artifacts = io_builder['run'](listing_rows, feature_rows, DATES_5)

    history = artifacts.history.sort_values('date')
    assert history['symbol'].tolist() == ['AAA', 'AAA', 'AAB', 'AAB', 'AAB']

    turnover = artifacts.turnover_stats.sort_values('date').reset_index(drop=True)
    row = turnover.loc[turnover['date'] == pd.Timestamp('2024-01-04')].iloc[0]
    assert row['entries'] == 0
    assert row['exits'] == 0
    assert row['turnover'] == pytest.approx(0.0)


def test_counts_by_day_reflect_entries_and_exits(io_builder):
    listing_rows = [
        _mk_listing('100', 'AAA', list_date='2024-01-02'),
        _mk_listing('200', 'BBB', list_date='2024-01-03'),
    ]
    feature_rows = [
        _mk_feature('2024-01-02', '100'),
        _mk_feature('2024-01-03', '100'),
        _mk_feature('2024-01-04', '100', price_ref=2.0),
        _mk_feature('2024-01-03', '200'),
        _mk_feature('2024-01-04', '200'),
    ]
    artifacts = io_builder['run'](
        listing_rows,
        feature_rows,
        ['2024-01-02', '2024-01-03', '2024-01-04'],
        start_date='2024-01-02',
        end_date='2024-01-04',
    )
    turnover = artifacts.turnover_stats.sort_values('date').reset_index(drop=True)

    assert turnover.loc[0, 'entries'] == 1
    assert turnover.loc[1, 'entries'] == 1
    assert turnover.loc[2, 'exits'] == 1
    assert turnover.loc[2, 'n_constituents'] == 1


def test_reproducible_history_and_manifest_hash_under_same_inputs(io_builder):
    listing_rows = [_mk_listing('100', 'AAA')]
    feature_rows = [_mk_feature(d, '100') for d in ['2024-01-02', '2024-01-03']]

    a1 = io_builder['run'](
        listing_rows,
        feature_rows,
        ['2024-01-02', '2024-01-03'],
        start_date='2024-01-02',
        end_date='2024-01-03',
        run_id='run_A',
        asof_ts_utc='2024-01-03T22:00:00Z',
        listing_name='listing_a.csv',
        feature_name='features_a.csv',
        config_name='config_a.json',
    )
    a2 = io_builder['run'](
        listing_rows,
        feature_rows,
        ['2024-01-02', '2024-01-03'],
        start_date='2024-01-02',
        end_date='2024-01-03',
        run_id='run_B',
        asof_ts_utc='2024-01-03T22:00:00Z',
        listing_name='listing_b.csv',
        feature_name='features_b.csv',
        config_name='config_b.json',
    )

    cols_to_compare = [c for c in a1.history.columns if c not in {'run_id', 'built_ts_utc'}]
    pd.testing.assert_frame_equal(
        a1.history[cols_to_compare].reset_index(drop=True),
        a2.history[cols_to_compare].reset_index(drop=True),
        check_like=False,
    )
    assert a1.manifest['config_hash'] == a2.manifest['config_hash']
    assert a1.manifest['input_snapshot_hash'] == a2.manifest['input_snapshot_hash']
    assert a1.manifest['history_logical_hash'] != a2.manifest['history_logical_hash']
