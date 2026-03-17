from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest


MODULE_PATH = Path('/mnt/data/universe_qc.py')

DAILY_REQUIRED_COLUMNS = {
    'date',
    'n_rows',
    'n_eligible',
    'n_ineligible',
    'turnover',
    'n_enter',
    'n_exit',
    'size_jump_abs',
    'size_jump_rel',
    'pct_missing_critical_meta',
    'n_expected_dead',
    'n_dead_appearing',
    'delisted_coverage_ratio',
}

FAILURES_REQUIRED_COLUMNS = {
    'date',
    'instrument_id',
    'symbol',
    'check_type',
    'severity',
    'flag_reason',
    'evidence_summary',
    'linked_corporate_action_id',
}

SUMMARY_REQUIRED_KEYS = {
    'gate_status',
    'n_rows_processed',
    'n_distinct_instruments',
    'n_fail',
    'n_warn',
    'delisted_coverage_ratio_mean',
    'pct_days_high_turnover',
    'pct_days_extreme_turnover',
    'pct_missing_critical_meta',
    'top_failure_types',
    'run_id',
    'config_hash',
    'code_version',
    'gates',
}

MANIFEST_REQUIRED_KEYS = {
    'run_id',
    'generated_at_utc',
    'code_version',
    'python_version',
    'config_hash',
    'config',
    'inputs',
    'outputs',
    'gate_status',
    'n_fail',
    'n_warn',
}

OUTPUT_KEYS = {'summary', 'daily', 'failures', 'manifest'}
GATE_DOMAIN = {'PASS', 'WARN', 'FAIL'}
SEVERITY_DOMAIN = {'PASS', 'WARN', 'FAIL'}


@pytest.fixture(scope='session')
def universe_qc_mod():
    spec = importlib.util.spec_from_file_location('universe_qc_contract_mod', MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load module from {MODULE_PATH}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def contract_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, universe_qc_mod):
    mod = universe_qc_mod

    monkeypatch.setattr(mod, 'ensure_parquet_engine_available', lambda: None)

    original_to_parquet = pd.DataFrame.to_parquet

    def _to_parquet_csv(self: pd.DataFrame, path: str | Path, *args: Any, **kwargs: Any) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_csv(path, index=kwargs.get('index', False))

    monkeypatch.setattr(pd.DataFrame, 'to_parquet', _to_parquet_csv, raising=True)

    cfg, payload, cfg_hash = mod.load_config(None)

    root = tmp_path
    out_root = root / 'out'

    def _base_rows() -> list[dict[str, Any]]:
        return [
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
                'terminal_ca_id': pd.NA,
                'death_reason': pd.NA,
                'config_hash': cfg_hash,
                'run_id': 'persisted_run',
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
                'terminal_ca_id': pd.NA,
                'death_reason': pd.NA,
                'config_hash': cfg_hash,
                'run_id': 'persisted_run',
            },
        ]

    def _write_inputs(rows: list[dict[str, Any]], tag: str) -> Dict[str, Path]:
        u = root / f'universe_{tag}.csv'
        l = root / f'listings_{tag}.csv'
        c = root / f'calendar_{tag}.csv'
        ca = root / f'ca_{tag}.csv'
        pd.DataFrame(rows).to_csv(u, index=False)
        pd.DataFrame(
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
        ).to_csv(l, index=False)
        pd.DataFrame({'date': pd.to_datetime(['2020-02-01', '2020-02-02'])}).to_csv(c, index=False)
        pd.DataFrame(columns=sorted(mod.OPTIONAL_CORPORATE_ACTION_COLUMNS)).to_csv(ca, index=False)
        return {'universe': u, 'listings': l, 'calendar': c, 'ca': ca}

    def _run(*, dirty: bool, run_id: str):
        rows = _base_rows()
        tag = 'dirty' if dirty else 'clean'
        if dirty:
            dup = dict(rows[0])
            dup['is_eligible'] = 2
            dup['membership_state'] = 'bad_state'
            rows.append(dup)
        paths = _write_inputs(rows, tag)
        artifacts = mod.audit_universe(
            universe_history_path=paths['universe'],
            listings_master_path=paths['listings'],
            calendar_path=paths['calendar'],
            corporate_actions_path=paths['ca'],
            run_id=run_id,
            config=cfg,
            config_hash=cfg_hash,
            config_payload=payload,
        )
        out_dir = out_root / run_id
        mod.persist_outputs(artifacts, out_dir, cfg.output.compression)
        return {
            'artifacts': artifacts,
            'out_dir': out_dir,
            'inputs': paths,
            'config': cfg,
            'config_hash': cfg_hash,
            'payload': payload,
        }

    def _read_table(path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        df = pd.read_csv(path)
        for col in ('date',):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df

    def _semantic_sort(df: pd.DataFrame, kind: str) -> pd.DataFrame:
        if df is None or df.empty:
            return df.copy()
        if kind == 'daily':
            cols = [c for c in ['date', 'n_rows', 'n_eligible'] if c in df.columns]
        else:
            cols = [c for c in ['date', 'severity', 'check_type', 'instrument_id', 'symbol'] if c in df.columns]
        return df.sort_values(cols, na_position='last').reset_index(drop=True)

    clean = _run(dirty=False, run_id='uq_contract_clean')
    dirty = _run(dirty=True, run_id='uq_contract_dirty')

    return {
        'mod': mod,
        'clean': clean,
        'dirty': dirty,
        'read_table': _read_table,
        'semantic_sort': _semantic_sort,
        'original_to_parquet': original_to_parquet,
    }


def test_daily_contains_required_columns(contract_case):
    daily = contract_case['clean']['artifacts'].daily
    assert DAILY_REQUIRED_COLUMNS.issubset(set(daily.columns))
    reason_cols = [c for c in daily.columns if c.startswith('n_reason_')]
    assert reason_cols, 'Expected at least one daily breakdown column by primary reason.'



def test_failures_contains_required_columns_and_valid_severity(contract_case):
    failures = contract_case['dirty']['artifacts'].failures
    assert not failures.empty
    assert FAILURES_REQUIRED_COLUMNS.issubset(set(failures.columns))
    assert set(failures['severity'].dropna().unique()).issubset(SEVERITY_DOMAIN)



def test_summary_contains_required_keys_and_gate_domain(contract_case):
    summary = contract_case['clean']['artifacts'].summary
    assert SUMMARY_REQUIRED_KEYS.issubset(set(summary.keys()))
    assert summary['gate_status'] in GATE_DOMAIN
    assert isinstance(summary['gates'], list) and summary['gates']



def test_manifest_contains_required_keys_and_outputs(contract_case):
    manifest = contract_case['clean']['artifacts'].manifest
    assert MANIFEST_REQUIRED_KEYS.issubset(set(manifest.keys()))
    assert OUTPUT_KEYS.issubset(set(manifest['outputs'].keys()))
    assert manifest['gate_status'] in GATE_DOMAIN



def test_summary_counts_are_coherent_with_tables(contract_case):
    dirty = contract_case['dirty']['artifacts']
    summary = dirty.summary
    failures = dirty.failures
    daily = dirty.daily
    assert summary['n_rows_processed'] == len(pd.read_csv(contract_case['dirty']['inputs']['universe']))
    assert summary['n_distinct_instruments'] == 1
    assert summary['n_fail'] == int((failures['severity'] == 'FAIL').sum())
    assert summary['n_warn'] == int((failures['severity'] == 'WARN').sum())
    assert len(daily) == 2



def test_manifest_output_paths_are_materialized(contract_case):
    clean = contract_case['clean']
    manifest = clean['artifacts'].manifest
    out_dir = clean['out_dir']
    for rel in manifest['outputs'].values():
        path = out_dir.parent / rel
        assert path.exists(), f'Missing materialized artifact: {path}'



def test_persisted_tables_match_returned_tables_semantically(contract_case):
    clean = contract_case['clean']
    read_table = contract_case['read_table']
    sort_df = contract_case['semantic_sort']
    persisted_daily = read_table(clean['out_dir'] / 'universe_qc_daily.parquet')
    persisted_failures = read_table(clean['out_dir'] / 'universe_qc_failures.parquet')
    returned_daily = clean['artifacts'].daily
    returned_failures = clean['artifacts'].failures
    pd.testing.assert_frame_equal(sort_df(persisted_daily, 'daily'), sort_df(returned_daily, 'daily'), check_dtype=False)
    pd.testing.assert_frame_equal(sort_df(persisted_failures, 'failures'), sort_df(returned_failures, 'failures'), check_dtype=False)



def _json_semantic_normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_semantic_normalize(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_json_semantic_normalize(v) for v in obj]
    if isinstance(obj, list):
        return [_json_semantic_normalize(v) for v in obj]
    return obj



def test_roundtrip_manifest_and_summary_json(contract_case):
    clean = contract_case['clean']
    persisted_manifest = json.loads((clean['out_dir'] / 'manifest.json').read_text(encoding='utf-8'))
    persisted_summary = json.loads((clean['out_dir'] / 'universe_qc_summary.json').read_text(encoding='utf-8'))
    assert persisted_manifest == _json_semantic_normalize(clean['artifacts'].manifest)
    assert persisted_summary == _json_semantic_normalize(clean['artifacts'].summary)



def test_daily_and_failures_are_sorted_deterministically(contract_case):
    dirty = contract_case['dirty']['artifacts']
    daily = dirty.daily
    failures = dirty.failures
    expected_daily = daily.sort_values(['date']).reset_index(drop=True)
    expected_failures = failures.sort_values(['date', 'severity', 'check_type', 'instrument_id'], na_position='last').reset_index(drop=True)
    pd.testing.assert_frame_equal(daily.reset_index(drop=True), expected_daily, check_dtype=False)
    pd.testing.assert_frame_equal(failures.reset_index(drop=True), expected_failures, check_dtype=False)



def test_failure_and_metric_counts_are_coherent(contract_case):
    dirty = contract_case['dirty']['artifacts']
    daily = dirty.daily
    failures = dirty.failures
    summary = dirty.summary
    assert summary['top_failure_types']
    total_counted = sum(int(x['size']) for x in summary['top_failure_types'])
    assert total_counted <= len(failures)
    assert (daily['n_rows'] >= daily['n_eligible']).all()
    assert (daily['n_rows'] >= daily['n_ineligible']).all()



def test_clean_case_is_pass_and_dirty_case_is_fail(contract_case):
    assert contract_case['clean']['artifacts'].summary['gate_status'] == 'PASS'
    assert contract_case['dirty']['artifacts'].summary['gate_status'] == 'FAIL'
