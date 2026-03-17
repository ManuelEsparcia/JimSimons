
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


REPO_DATA = Path('/mnt/data')


def _load_module(name: str, filename: str):
    path = REPO_DATA / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _touch(path: Path, payload: str = 'ok') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding='utf-8')


@pytest.fixture(scope='module')
def build_mod():
    return _load_module('build_universe_cli_smoke', 'build_universe.py')


@pytest.fixture(scope='module')
def survivorship_mod():
    return _load_module('survivorship_cli_smoke', 'survivorship.py')


@pytest.fixture(scope='module')
def qc_mod():
    return _load_module('universe_qc_cli_smoke', 'universe_qc.py')


@pytest.fixture(scope='module')
def ca_mod():
    return _load_module('corporate_actions_cli_smoke', 'corporate_actions.py')


def test_build_universe_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, build_mod):
    listing = tmp_path / 'listing.csv'
    calendar = tmp_path / 'calendar.csv'
    features = tmp_path / 'features.csv'
    config = tmp_path / 'config.json'
    outdir = tmp_path / 'out'
    for p in [listing, calendar, features]:
        _touch(p, 'x')
    _write_json(config, {'dummy': True})

    calls = {}

    def fake_build_universe(**kwargs):
        calls.update(kwargs)
        out = Path(kwargs['output_dir'])
        out.mkdir(parents=True, exist_ok=True)
        _touch(out / 'history.parquet')
        _touch(out / 'current.parquet')
        _touch(out / 'universe_manifest.json', '{}')
        return SimpleNamespace()

    monkeypatch.setattr(build_mod, 'build_universe', fake_build_universe)

    rc = build_mod.main([
        '--listing-master-path', str(listing),
        '--calendar-path', str(calendar),
        '--features-asof-path', str(features),
        '--config-path', str(config),
        '--start-date', '2026-01-01',
        '--end-date', '2026-01-03',
        '--as-of-ts-utc', '2026-03-15T10:00:00Z',
        '--run-id', 'run_build',
        '--output-dir', str(outdir),
    ])

    assert rc == 0
    assert calls['run_id'] == 'run_build'
    assert calls['asof_ts_utc'] == '2026-03-15T10:00:00Z'
    assert calls['output_dir'] == str(outdir)
    assert (outdir / 'history.parquet').exists()
    assert (outdir / 'current.parquet').exists()
    assert (outdir / 'universe_manifest.json').exists()


def test_survivorship_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, survivorship_mod):
    universe_history = tmp_path / 'universe_history.csv'
    lifecycle = tmp_path / 'lifecycle.csv'
    config = tmp_path / 'config.json'
    outdir = tmp_path / 'out_surv'
    for p in [universe_history, lifecycle]:
        _touch(p, 'x')
    _write_json(config, {'dummy': True})

    monkeypatch.setattr(
        survivorship_mod,
        'load_config',
        lambda path: (SimpleNamespace(output=SimpleNamespace(output_dir=str(outdir), compression='snappy')), {'dummy': True}, 'cfg123'),
    )
    monkeypatch.setattr(survivorship_mod, 'load_universe_history', lambda p: pd.DataFrame({'x': [1]}))
    monkeypatch.setattr(survivorship_mod, 'load_lifecycle_master', lambda p: pd.DataFrame({'x': [1]}))
    monkeypatch.setattr(survivorship_mod, 'load_corporate_actions', lambda p: pd.DataFrame())
    monkeypatch.setattr(survivorship_mod, 'load_prices', lambda p: None)

    def fake_run_survivorship_audit(**kwargs):
        return SimpleNamespace(
            summary={'gate_status': 'PASS', 'run_id': kwargs['run_id']},
            daily=pd.DataFrame({'date': ['2026-01-02'], 'gate_status': ['PASS']}),
            missing_dead=pd.DataFrame({'instrument_id': []}),
            problem_cases=pd.DataFrame({'instrument_id': []}),
            comparison=None,
            manifest={'run_id': kwargs['run_id'], 'asof_ts_utc': kwargs['asof_ts_utc']},
        )

    def fake_persist(artifacts, output_dir, compression):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / 'survivorship_summary.json', artifacts.summary)
        _touch(out / 'survivorship_daily.parquet')
        _touch(out / 'missing_dead_instruments.parquet')
        _touch(out / 'survivorship_problem_cases.parquet')
        _write_json(out / 'survivorship_manifest.json', artifacts.manifest)

    monkeypatch.setattr(survivorship_mod, 'run_survivorship_audit', fake_run_survivorship_audit)
    monkeypatch.setattr(survivorship_mod, 'persist_outputs', fake_persist)

    rc = survivorship_mod.main([
        '--universe-history-path', str(universe_history),
        '--lifecycle-master-path', str(lifecycle),
        '--config-path', str(config),
        '--run-id', 'run_surv',
        '--asof-ts-utc', '2026-03-15T10:00:00Z',
        '--output-dir', str(outdir),
    ])

    assert rc == 0
    assert (outdir / 'survivorship_summary.json').exists()
    assert (outdir / 'survivorship_daily.parquet').exists()
    assert (outdir / 'survivorship_manifest.json').exists()
    summary = json.loads((outdir / 'survivorship_summary.json').read_text(encoding='utf-8'))
    assert summary['gate_status'] == 'PASS'


def test_universe_qc_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qc_mod):
    history = tmp_path / 'history.csv'
    listings = tmp_path / 'listings.csv'
    calendar = tmp_path / 'calendar.csv'
    config = tmp_path / 'config.json'
    outdir = tmp_path / 'out_qc'
    for p in [history, listings, calendar]:
        _touch(p, 'x')
    _write_json(config, {'dummy': True})

    monkeypatch.setattr(
        qc_mod,
        'load_config',
        lambda path: (SimpleNamespace(output=SimpleNamespace(output_dir=str(outdir), compression='snappy')), {'dummy': True}, 'cfg456'),
    )

    def fake_audit(**kwargs):
        return SimpleNamespace(
            daily=pd.DataFrame({'date': ['2026-01-02'], 'gate_status': ['PASS']}),
            failures=pd.DataFrame(columns=['failure_code', 'severity']),
            summary={'gate_status': 'PASS', 'n_fail': 0, 'n_warn': 0},
            manifest={'run_id': kwargs['run_id'], 'inputs': {'universe_history_path': kwargs['universe_history_path']}},
        )

    def fake_persist(artifacts, output_dir, compression):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _touch(out / 'universe_qc_daily.parquet')
        _touch(out / 'universe_qc_failures.parquet')
        _write_json(out / 'universe_qc_summary.json', artifacts.summary)
        _write_json(out / 'manifest.json', artifacts.manifest)

    monkeypatch.setattr(qc_mod, 'audit_universe', fake_audit)
    monkeypatch.setattr(qc_mod, 'persist_outputs', fake_persist)

    rc = qc_mod.main([
        '--universe-history-path', str(history),
        '--listings-master-path', str(listings),
        '--calendar-path', str(calendar),
        '--config-path', str(config),
        '--run-id', 'run_qc',
        '--output-dir', str(outdir),
    ])

    assert rc == 0
    run_dir = outdir / 'run_qc'
    assert (run_dir / 'universe_qc_daily.parquet').exists()
    assert (run_dir / 'universe_qc_summary.json').exists()
    summary = json.loads((run_dir / 'universe_qc_summary.json').read_text(encoding='utf-8'))
    assert summary['gate_status'] == 'PASS'


def test_corporate_actions_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ca_mod):
    raw = tmp_path / 'raw.csv'
    identity = tmp_path / 'identity.csv'
    config = tmp_path / 'config.json'
    outdir = tmp_path / 'out_ca'
    for p in [raw, identity]:
        _touch(p, 'x')
    _write_json(config, {'dummy': True})

    monkeypatch.setattr(
        ca_mod,
        'load_config',
        lambda path: (SimpleNamespace(output=SimpleNamespace(output_dir=str(outdir), compression='snappy')), {'dummy': True}, 'cfg789'),
    )

    def fake_canonicalize(**kwargs):
        return SimpleNamespace(
            history=pd.DataFrame({'event_id': ['e1'], 'instrument_id': ['i1']}),
            current=pd.DataFrame({'event_id': ['e1'], 'known_asof': ['2026-03-15T10:00:00Z']}),
            failures=pd.DataFrame(columns=['failure_class']),
            summary={'gate_status': 'PASS', 'canonical_event_count': 1, 'failure_count': 0},
            manifest={'run_id': kwargs['run_id'], 'asof_ts_utc': kwargs['asof_ts_utc']},
        )

    def fake_persist(artifacts, output_dir, run_id, compression):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _touch(out / 'history.parquet')
        _touch(out / 'current.parquet')
        _touch(out / f'failures_{run_id}.parquet')
        _write_json(out / f'summary_{run_id}.json', artifacts.summary)
        _write_json(out / f'manifest_{run_id}.json', artifacts.manifest)

    monkeypatch.setattr(ca_mod, 'canonicalize_corporate_actions', fake_canonicalize)
    monkeypatch.setattr(ca_mod, 'persist_outputs', fake_persist)

    rc = ca_mod.main([
        '--raw-paths', str(raw),
        '--identity-master-path', str(identity),
        '--config-path', str(config),
        '--run-id', 'run_ca',
        '--asof-ts-utc', '2026-03-15T10:00:00Z',
        '--output-dir', str(outdir),
    ])

    assert rc == 0
    assert (outdir / 'history.parquet').exists()
    assert (outdir / 'current.parquet').exists()
    assert (outdir / 'summary_run_ca.json').exists()
    summary = json.loads((outdir / 'summary_run_ca.json').read_text(encoding='utf-8'))
    assert summary['canonical_event_count'] == 1
