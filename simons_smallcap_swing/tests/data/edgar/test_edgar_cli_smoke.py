from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# Import helpers
# -----------------------------------------------------------------------------


def _load_module(module_basename: str):
    module_names = [
        f"simons_smallcap_swing.data.edgar.{module_basename}",
        f"data.edgar.{module_basename}",
        module_basename,
    ]
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue

    here = Path(__file__).resolve()
    candidate_paths = [
        here.parents[3] / "simons_smallcap_swing" / "data" / "edgar" / f"{module_basename}.py",
        here.parents[2] / "data" / "edgar" / f"{module_basename}.py",
        here.parents[1] / f"{module_basename}.py",
        Path(f"/mnt/data/{module_basename}.py"),
    ]
    for path in candidate_paths:
        if path.exists():
            spec = importlib.util.spec_from_file_location(f"{module_basename}_cli_smoke_under_test", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module

    raise ModuleNotFoundError(f"Could not import {module_basename}.py from expected repo paths.")


@pytest.fixture(scope="session")
def ticker_cik_module():
    return _load_module("ticker_cik")


@pytest.fixture(scope="session")
def fetch_companyfacts_module():
    return _load_module("fetch_companyfacts")


@pytest.fixture(scope="session")
def fetch_submissions_module():
    return _load_module("fetch_submissions")


@pytest.fixture(scope="session")
def parse_xbrl_module():
    return _load_module("parse_xbrl")


@pytest.fixture(scope="session")
def filings_flags_module():
    return _load_module("filings_flags")


@pytest.fixture(scope="session")
def point_in_time_module():
    return _load_module("point_in_time")


@pytest.fixture(scope="session")
def edgar_qc_module():
    return _load_module("edgar_qc")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))



def _write_csv(rows: Iterable[Mapping[str, Any]], path: Path) -> Path:
    df = pd.DataFrame([dict(r) for r in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path



def _patch_edgar_qc_parquet(monkeypatch: pytest.MonkeyPatch, edgar_qc_module: Any) -> None:
    def _writer(path: Path, df: pd.DataFrame, *, index: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=index)

    monkeypatch.setattr(edgar_qc_module, "write_parquet", _writer)



def _sec_mapping_rows() -> list[dict[str, Any]]:
    return [
        {
            "ticker": "AAA",
            "cik_str": "1",
            "company_name": "AAA INC",
            "primary_exchange": "NASDAQ",
            "security_class": "COMMON",
            "effective_from": "2020-01-01",
            "effective_to": None,
            "ingest_ts_utc": "2026-03-01T00:00:00Z",
        }
    ]



def _submission_payload(cik: str = "0000000001", symbol: str = "AAA") -> dict[str, Any]:
    return {
        "cik": cik,
        "name": "AAA INC",
        "tickers": [symbol],
        "exchanges": ["NASDAQ"],
        "filings": {
            "recent": {
                "accessionNumber": ["0000000001-26-000001"],
                "form": ["10-K"],
                "filingDate": ["2026-02-01"],
                "acceptanceDateTime": ["2026-02-01T14:30:00Z"],
            }
        },
    }



def _companyfacts_payload(cik: str = "0000000001") -> dict[str, Any]:
    return {
        "cik": cik,
        "entityName": "AAA INC",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "label": "Revenue",
                    "description": "Revenue",
                    "units": {
                        "USD": [
                            {
                                "val": 250.0,
                                "start": "2025-01-01",
                                "end": "2025-12-31",
                                "filed": "2026-02-01",
                                "accepted": "2026-02-01T14:30:00Z",
                                "fy": 2025,
                                "fp": "FY",
                                "frame": "CY2025",
                                "form": "10-K",
                                "accn": "0000000001-26-000001",
                            }
                        ]
                    },
                },
                "Assets": {
                    "label": "Assets",
                    "description": "Assets",
                    "units": {
                        "USD": [
                            {
                                "val": 500.0,
                                "end": "2025-12-31",
                                "filed": "2026-02-01",
                                "accepted": "2026-02-01T14:30:00Z",
                                "fy": 2025,
                                "fp": "FY",
                                "form": "10-K",
                                "accn": "0000000001-26-000001",
                            }
                        ]
                    },
                },
            }
        },
    }



def _build_filings_inputs(root: Path) -> tuple[Path, Path, Path]:
    submissions = pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "symbol": "AAA",
                "accession_number": "0000000001-26-000001",
                "form_type": "10-K",
                "filing_date": "2026-02-01",
                "acceptance_datetime": "2026-02-01T14:30:00Z",
                "period_end": "2025-12-31",
                "document_text": "Annual report",
                "filer_status": "large_accelerated_filer",
            }
        ]
    )
    facts = pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "metric_name": "revenue",
                "concept": "revenue",
                "value": 250.0,
                "unit": "USD",
                "period_end": "2025-12-31",
                "filed_date": "2026-02-01",
                "acceptance_datetime": "2026-02-01T14:30:00Z",
                "accession_number": "0000000001-26-000001",
                "source_accession_number": "0000000001-26-000001",
                "form_type": "10-K",
                "source_form_type": "10-K",
            },
            {
                "cik": "0000000001",
                "metric_name": "total_assets",
                "concept": "assets",
                "value": 500.0,
                "unit": "USD",
                "period_end": "2025-12-31",
                "filed_date": "2026-02-01",
                "acceptance_datetime": "2026-02-01T14:30:00Z",
                "accession_number": "0000000001-26-000001",
                "source_accession_number": "0000000001-26-000001",
                "form_type": "10-K",
                "source_form_type": "10-K",
            },
        ]
    )
    submissions_path = root / "submissions.csv"
    facts_path = root / "facts.csv"
    submissions.to_csv(submissions_path, index=False)
    facts.to_csv(facts_path, index=False)
    flags_cfg = {
        "output_dir": str(root / "flags_out"),
        "strict_parquet": False,
        "persistence": {"prefer_parquet": False},
    }
    flags_cfg_path = root / "filings_flags_config.json"
    flags_cfg_path.write_text(json.dumps(flags_cfg, indent=2), encoding="utf-8")
    return submissions_path, facts_path, flags_cfg_path



def _build_point_in_time_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    parsed_facts = pd.DataFrame(
        [
            {
                "cik": "1",
                "metric_name": "Revenue",
                "value": 250.0,
                "period_end": "2025-12-31",
                "filed_date": "2026-02-01",
                "acceptance_datetime": "2026-02-01T14:30:00Z",
                "source_accession_number": "0001-26-000001",
                "source_form_type": "10-K",
                "fact_quality_score": 0.9,
                "amendment_type": "original",
            },
            {
                "cik": "1",
                "metric_name": "TotalAssets",
                "value": 500.0,
                "period_end": "2025-12-31",
                "filed_date": "2026-02-01",
                "acceptance_datetime": "2026-02-01T14:30:00Z",
                "source_accession_number": "0001-26-000001",
                "source_form_type": "10-K",
                "fact_quality_score": 0.95,
                "amendment_type": "original",
            },
        ]
    )
    mapping = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "cik": "1",
                "effective_from": "2020-01-01",
                "effective_to": None,
                "mapping_version": "map_v1",
                "resolution_status": "exact",
                "confidence_score": 1.0,
            }
        ]
    )
    flags = pd.DataFrame(
        [
            {
                "cik": "1",
                "accession_number": "0001-26-000001",
                "pit_action": "allow",
                "quality_score": 1.0,
                "severity": "info",
            }
        ]
    )
    parsed_path = root / "parsed_facts.csv"
    mapping_path = root / "ticker_cik_history.csv"
    flags_path = root / "filings_flags.csv"
    parsed_facts.to_csv(parsed_path, index=False)
    mapping.to_csv(mapping_path, index=False)
    flags.to_csv(flags_path, index=False)

    pit_cfg = {
        "pit_config_version": "pytest_cli_pit_v1",
        "storage": {
            "output_root": str(root / "pit_out"),
            "allow_csv_fallback": True,
            "compression": "snappy",
        },
        "calendar": {"freq": "D", "asof_time_of_day": "23:59:59", "timezone": "UTC"},
        "metrics": {"include": ["Revenue", "TotalAssets"], "exclude": []},
        "flags": {"enabled": True},
        "validation": {
            "abort_on_input_contract_failure": True,
            "coverage_warn_threshold": 0.0,
            "coverage_fail_threshold": 0.0,
            "abort_on_leakage": True,
            "abort_on_duplicate_output": True,
            "abort_on_identity_ambiguity": False,
        },
    }
    pit_cfg_path = root / "pit_config.json"
    pit_cfg_path.write_text(json.dumps(pit_cfg, indent=2), encoding="utf-8")
    return parsed_path, mapping_path, flags_path, pit_cfg_path



def _build_edgar_qc_inputs(root: Path) -> dict[str, Path]:
    mapping = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "cik": "0000000001",
                "effective_from": "2025-01-01",
                "effective_to": "2100-01-01",
                "source_priority": 1,
                "source_type": "official_sec_source",
                "sector": "Tech",
                "size_bucket": "mid",
                "liquidity_decile": 5,
                "run_id": "map_run",
            }
        ]
    )
    submissions = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "cik": "0000000001",
                "source_filing_id": "0001-26-000001",
                "form_type": "10-K",
                "acceptance_ts": "2026-02-10T14:00:00Z",
                "filing_date": "2026-02-10",
                "status_code": 200,
                "raw_artifact_id": "sub_raw_1",
                "run_id": "sub_run",
            }
        ]
    )
    companyfacts = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "cik": "0000000001",
                "metric": "Revenue",
                "unit": "USD",
                "value": 100.0,
                "period_end": "2025-12-31",
                "acceptance_ts": "2026-02-10T14:00:00Z",
                "source_filing_id": "0001-26-000001",
                "run_id": "cf_run",
            }
        ]
    )
    parsed = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "cik": "0000000001",
                "metric": "Revenue",
                "value": 100.0,
                "unit": "USD",
                "period_end": "2025-12-31",
                "acceptance_ts": "2026-02-10T14:00:00Z",
                "source_filing_id": "0001-26-000001",
                "form_type": "10-K",
                "amendment_flag": False,
                "raw_artifact_id": "cf_raw_1",
                "run_id": "parse_run",
            }
        ]
    )
    pit = pd.DataFrame(
        [
            {
                "asof": "2026-03-15T23:59:59Z",
                "symbol": "AAA",
                "cik": "0000000001",
                "metric": "Revenue",
                "value": 100.0,
                "unit": "USD",
                "period_end": "2025-12-31",
                "acceptance_ts": "2026-02-10T14:00:00Z",
                "source_filing_id": "0001-26-000001",
                "form_type": "10-K",
                "run_id": "pit_run",
            }
        ]
    )

    paths = {
        "ticker_cik_outputs": root / "ticker_cik_for_qc.csv",
        "submissions_raw": root / "submissions_for_qc.csv",
        "companyfacts_raw": root / "companyfacts_for_qc.csv",
        "parsed_facts": root / "parsed_for_qc.csv",
        "pit_store": root / "pit_for_qc.csv",
    }
    mapping.to_csv(paths["ticker_cik_outputs"], index=False)
    submissions.to_csv(paths["submissions_raw"], index=False)
    companyfacts.to_csv(paths["companyfacts_raw"], index=False)
    parsed.to_csv(paths["parsed_facts"], index=False)
    pit.to_csv(paths["pit_store"], index=False)
    return paths


# -----------------------------------------------------------------------------
# Smoke tests
# -----------------------------------------------------------------------------


def test_ticker_cik_cli_smoke(tmp_path: Path, ticker_cik_module: Any, capsys: pytest.CaptureFixture[str]) -> None:
    sec_path = _write_csv(_sec_mapping_rows(), tmp_path / "sec_source.csv")
    cfg = json.loads(json.dumps(ticker_cik_module.DEFAULT_CONFIG))
    cfg["storage"]["output_root"] = str(tmp_path / "map_out")
    cfg["storage"]["allow_csv_fallback"] = True
    cfg["identity"]["strict_abort_on_unresolved_conflict"] = False
    cfg_path = tmp_path / "ticker_cik_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    ticker_cik_module.main(
        [
            "--source-path",
            str(sec_path),
            "--config-path",
            str(cfg_path),
            "--run-id",
            "pytest_cli_map",
            "--asof",
            "2026-03-15",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "pytest_cli_map"
    assert Path(payload["artifacts"]["history_path"]).exists()
    assert Path(payload["artifacts"]["current_path"]).exists()
    assert Path(payload["artifacts"]["manifest_path"]).exists()



def test_fetch_companyfacts_cli_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fetch_companyfacts_module: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cik_path = _write_csv([{"cik": "1"}], tmp_path / "ciks.csv")
    outdir = tmp_path / "companyfacts_out"
    manifest_path = outdir / "companyfacts_manifest_pytest_cf_cli.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"ok": True}), encoding="utf-8")

    def _fake_run_ingestion(*, cik_list_path: str, config: Any, run_id: str, asof: str):
        assert Path(cik_list_path).exists()
        assert run_id == "pytest_cf_cli"
        assert asof == "2026-03-15T10:00:00Z"
        return {
            "manifest": {
                "gate": "pass",
                "gate_reason": "ok",
                "requested_cik_count": 1,
                "success_cik_count": 1,
                "coverage_ratio": 1.0,
                "facts_total_rows": 1,
            },
            "manifest_path": str(manifest_path),
        }

    monkeypatch.setattr(fetch_companyfacts_module, "run_ingestion", _fake_run_ingestion)

    cfg = {
        "output_dir": str(outdir),
        "table_format": "csv",
        "allow_csv_fallback": True,
        "max_workers": 1,
    }
    cfg_path = tmp_path / "fetch_companyfacts_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    rc = fetch_companyfacts_module.main(
        [
            "--cik-list-path",
            str(cik_path),
            "--config-path",
            str(cfg_path),
            "--run-id",
            "pytest_cf_cli",
            "--asof",
            "2026-03-15T10:00:00Z",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["status"] == "pass"
    assert Path(payload["manifest_path"]).exists()



def test_fetch_submissions_cli_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fetch_submissions_module: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cik_path = _write_csv([{"cik": "1"}], tmp_path / "ciks.csv")
    outdir = tmp_path / "submissions_out"
    manifest_path = outdir / "raw" / "manifest_pytest_sub_cli.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"ok": True}), encoding="utf-8")

    def _fake_run_fetch_submissions(**kwargs):
        assert Path(kwargs["cik_list_path"]).exists()
        assert kwargs["run_id"] == "pytest_sub_cli"
        return {
            "run_id": "pytest_sub_cli",
            "gate": "pass",
            "artifacts": {
                "events_table": {"path": str(outdir / "events.csv"), "rows": 1},
                "payloads_table": {"path": str(outdir / "payloads.csv"), "rows": 1},
                "manifest": {"path": str(manifest_path), "rows": 1},
            },
        }

    monkeypatch.setattr(fetch_submissions_module, "run_fetch_submissions", _fake_run_fetch_submissions)

    cfg = {
        "output_dir": str(outdir),
        "table_format": "csv",
        "allow_csv_fallback": True,
        "max_workers": 1,
        "include_history_files": False,
    }
    cfg_path = tmp_path / "fetch_submissions_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    (outdir / "events.csv").parent.mkdir(parents=True, exist_ok=True)
    (outdir / "events.csv").write_text("", encoding="utf-8")
    (outdir / "payloads.csv").write_text("", encoding="utf-8")

    argv = [
        "prog",
        "--cik-list-path",
        str(cik_path),
        "--config-path",
        str(cfg_path),
        "--run-id",
        "pytest_sub_cli",
        "--asof",
        "2026-03-15T10:00:00Z",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    fetch_submissions_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "pytest_sub_cli"
    assert payload["gate"] == "pass"
    assert Path(payload["artifacts"]["manifest"]["path"]).exists()
    assert Path(payload["artifacts"]["events_table"]["path"]).exists()
    assert Path(payload["artifacts"]["payloads_table"]["path"]).exists()



def test_parse_xbrl_cli_smoke(
    tmp_path: Path,
    parse_xbrl_module: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submissions_raw = tmp_path / "submissions_raw"
    companyfacts_raw = tmp_path / "companyfacts_raw"
    submissions_raw.mkdir(parents=True, exist_ok=True)
    companyfacts_raw.mkdir(parents=True, exist_ok=True)
    cik = "0000000001"
    (submissions_raw / f"CIK{cik}.json").write_text(json.dumps(_submission_payload(cik=cik), indent=2), encoding="utf-8")
    (companyfacts_raw / f"CIK{cik}.json").write_text(json.dumps(_companyfacts_payload(cik=cik), indent=2), encoding="utf-8")

    cfg = {
        "output_dir": str(tmp_path / "parse_out"),
        "table_format": "csv",
        "allow_csv_fallback": True,
        "warn_reject_ratio": 0.95,
        "systemic_mapping_failure_ratio": 0.95,
        "unknown_taxonomy_policy": "reject",
        "unknown_unit_policy": "reject",
        "conflict_policy": "reject",
        "unresolved_tie_policy": "reject",
    }
    cfg_path = tmp_path / "parse_xbrl_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    rc = parse_xbrl_module.main(
        [
            "--submissions-path",
            str(submissions_raw),
            "--companyfacts-path",
            str(companyfacts_raw),
            "--config-path",
            str(cfg_path),
            "--run-id",
            "pytest_parse_cli",
            "--asof",
            "2026-03-15",
        ]
    )

    manifest = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert manifest["run_id"] == "pytest_parse_cli"
    assert Path(manifest["outputs"]["facts_canonical"]).exists()
    assert Path(manifest["outputs"]["facts_rejected"]).exists()
    assert (tmp_path / "parse_out" / "manifest_pytest_parse_cli.json").exists()



def test_filings_flags_cli_smoke(
    tmp_path: Path,
    filings_flags_module: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submissions_path, facts_path, cfg_path = _build_filings_inputs(tmp_path)

    rc = filings_flags_module.main(
        [
            "--submissions-path",
            str(submissions_path),
            "--facts-path",
            str(facts_path),
            "--config-path",
            str(cfg_path),
            "--run-id",
            "pytest_flags_cli",
            "--asof",
            "2026-03-15T23:59:59Z",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["status"] == "ok"
    assert Path(payload["files"]["filings_flags"]).exists()
    assert Path(payload["files"]["manifest"]).exists()
    assert Path(payload["files"]["filings_flags_summary"]).exists()



def test_point_in_time_cli_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    point_in_time_module: Any,
) -> None:
    parsed_path, mapping_path, flags_path, cfg_path = _build_point_in_time_inputs(tmp_path)

    argv = [
        "prog",
        "--parsed-facts-path",
        str(parsed_path),
        "--ticker-cik-mapping-path",
        str(mapping_path),
        "--pit-config-path",
        str(cfg_path),
        "--filings-flags-path",
        str(flags_path),
        "--run-id",
        "pytest_pit_cli",
        "--start-date",
        "2026-03-15",
        "--end-date",
        "2026-03-15",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    point_in_time_module.main()

    out_root = tmp_path / "pit_out"
    part_files = list(out_root.rglob("part-pytest_pit_cli.*"))
    assert part_files, "Expected a materialized PIT partition file."
    assert (out_root / "pit_coverage_pytest_pit_cli.csv").exists() or (out_root / "pit_coverage_pytest_pit_cli.parquet").exists()
    assert (out_root / "manifest_pytest_pit_cli.json").exists()



def test_edgar_qc_cli_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    edgar_qc_module: Any,
) -> None:
    _patch_edgar_qc_parquet(monkeypatch, edgar_qc_module)
    paths = _build_edgar_qc_inputs(tmp_path)

    rc = edgar_qc_module.main(
        [
            "--ticker-cik-outputs",
            str(paths["ticker_cik_outputs"]),
            "--submissions-raw",
            str(paths["submissions_raw"]),
            "--companyfacts-raw",
            str(paths["companyfacts_raw"]),
            "--parsed-facts",
            str(paths["parsed_facts"]),
            "--pit-store",
            str(paths["pit_store"]),
            "--output-dir",
            str(tmp_path / "qc_out"),
            "--run-id",
            "pytest_qc_cli",
        ]
    )

    outdir = tmp_path / "qc_out"
    assert rc == 0
    assert (outdir / "edgar_qc_summary.json").exists()
    assert (outdir / "edgar_qc_checks.parquet").exists()
    assert (outdir / "edgar_qc_metrics.parquet").exists()
    assert (outdir / "manifest.json").exists()
