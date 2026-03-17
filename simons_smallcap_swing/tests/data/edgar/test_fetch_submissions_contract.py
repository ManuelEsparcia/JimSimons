from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd
import pytest


EVENTS_REQUIRED_COLUMNS = {
    "cik",
    "accession_number",
    "form_type",
    "filing_date",
    "acceptance_datetime",
    "primary_doc",
    "source_payload_url",
    "payload_kind",
    "run_id",
    "asof",
    "payload_sha256",
    "event_fingerprint",
}

PAYLOADS_REQUIRED_COLUMNS = {
    "cik",
    "payload_kind",
    "source_url",
    "http_status",
    "fetched_at_utc",
    "attempts",
    "payload_sha256",
    "payload_json",
    "raw_json_path",
    "run_id",
    "asof",
}

FAILURES_REQUIRED_COLUMNS = {
    "cik",
    "failure_class",
    "stage",
    "reason",
    "run_id",
    "asof",
}

METRICS_REQUIRED_COLUMNS = {
    "cik",
    "status",
    "attempts",
    "latency_seconds",
}

RECONCILIATION_REQUIRED_COLUMNS = {
    "cik",
    "accession_number",
    "reconciliation_class",
    "previous_run_id",
    "previous_asof",
    "current_run_id",
    "current_asof",
    "current_fingerprint",
    "previous_fingerprint",
    "filing_date",
    "late_threshold_asof",
}

MANIFEST_REQUIRED_KEYS = {
    "module",
    "schema_version",
    "run_id",
    "asof",
    "config_hash",
    "target_universe_size",
    "successful_ciks",
    "failures_by_class",
    "total_submissions_extracted",
    "late_arrivals",
    "changed_records",
    "gate",
    "gate_reason",
    "aggregate_metrics",
    "artifacts",
    "started_at_utc",
    "finished_at_utc",
}

ARTIFACT_KEYS = {
    "events_table",
    "payloads_table",
    "failures_table",
    "reconciliation_table",
    "metrics_table",
    "manifest",
}


# -----------------------------------------------------------------------------
# Import adapter
# -----------------------------------------------------------------------------


def _load_fetch_submissions_module():
    module_names = [
        "simons_smallcap_swing.data.edgar.fetch_submissions",
        "data.edgar.fetch_submissions",
        "fetch_submissions",
    ]
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue

    here = Path(__file__).resolve()
    candidate_paths = [
        here.parents[3] / "simons_smallcap_swing" / "data" / "edgar" / "fetch_submissions.py",
        here.parents[2] / "data" / "edgar" / "fetch_submissions.py",
        here.parents[1] / "fetch_submissions.py",
        Path("/mnt/data/fetch_submissions.py"),
    ]
    for path in candidate_paths:
        if path.exists():
            spec = importlib.util.spec_from_file_location("fetch_submissions_contract_under_test", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module

    raise ModuleNotFoundError(
        "Could not import fetch_submissions.py. Expected it at "
        "simons_smallcap_swing.data.edgar.fetch_submissions or in a nearby repo path."
    )


@pytest.fixture(scope="session")
def fetch_submissions_module():
    return _load_fetch_submissions_module()


# -----------------------------------------------------------------------------
# Synthetic builders
# -----------------------------------------------------------------------------


def _root_payload(cik: str) -> Dict[str, Any]:
    return {
        "cik": cik,
        "name": f"ISSUER {cik}",
        "tickers": ["AAPL" if cik.endswith("193") else "MSFT"],
        "exchanges": ["Nasdaq"],
        "sic": "3571",
        "sicDescription": "Electronic Computers",
        "stateOfIncorporation": "CA",
        "fiscalYearEnd": "1231",
        "filings": {
            "recent": {
                "accessionNumber": [f"{cik}-26-000001"],
                "filingDate": ["2026-03-01"],
                "reportDate": ["2025-12-31"],
                "acceptanceDateTime": ["2026-03-01T14:30:00Z"],
                "act": ["34"],
                "form": ["10-K"],
                "fileNumber": ["001-00001"],
                "filmNumber": ["26543210"],
                "items": [""],
                "size": [123456],
                "isXBRL": [1],
                "isInlineXBRL": [1],
                "primaryDocument": ["annual.htm"],
                "primaryDocDescription": ["Annual report"],
            },
            "files": [
                {
                    "name": f"CIK{cik}-submissions-001.json",
                    "filingCount": 25,
                    "filingFrom": "2025-01-01",
                    "filingTo": "2025-12-31",
                }
            ],
        },
    }


def _event_row(module: Any, *, cik: str, accession_number: str, filing_date: str, acceptance_datetime: str, run_id: str, asof: str, payload_sha256: str, form_type: str = "10-K", primary_doc: str = "annual.htm") -> Dict[str, Any]:
    row = {
        "cik": cik,
        "accession_number": accession_number,
        "form_type": form_type,
        "filing_date": filing_date,
        "report_date": "2025-12-31",
        "acceptance_datetime": acceptance_datetime,
        "primary_doc": primary_doc,
        "primary_doc_description": "Annual report",
        "submission_document_url": module.build_submission_source_url(cik, accession_number, primary_doc),
        "source_payload_url": module.SEC_SUBMISSIONS_URL.format(cik=cik),
        "source_scope": "filings.recent",
        "source_file_name": None,
        "payload_kind": "root_submissions",
        "act": "34",
        "file_number": "001-00001",
        "film_number": "26543210",
        "items": "",
        "size": 123456,
        "is_xbrl": True,
        "is_inline_xbrl": True,
        "entity_name": f"ISSUER {cik}",
        "sic": "3571",
        "sic_description": "Electronic Computers",
        "state_of_incorporation": "CA",
        "fiscal_year_end": "1231",
        "run_id": run_id,
        "asof": asof,
        "payload_sha256": payload_sha256,
    }
    row["event_fingerprint"] = module.event_fingerprint_from_row(row)
    return row


def _write_cik_list(path: Path, values: Iterable[str]) -> Path:
    path.write_text(json.dumps(list(values)), encoding="utf-8")
    return path


def _write_config(path: Path, *, output_dir: Path) -> Path:
    config = {
        "output_dir": str(output_dir),
        "table_format": "csv",
        "allow_csv_fallback": True,
        "max_workers": 1,
        "skip_existing_success": False,
        "force_refresh": False,
        "include_history_files": True,
        "fail_on_gate_fail": False,
        "rate_limit_per_second": 1000.0,
        "backoff_base_seconds": 0.001,
        "backoff_max_seconds": 0.001,
        "backoff_jitter_seconds": 0.0,
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return path



def _make_success_result(module: Any, *, cik: str, run_id: str, asof: str, output_dir: Path) -> Any:
    root_payload = _root_payload(cik)
    history_payload = {
        "accessionNumber": [f"{cik}-25-000007"],
        "filingDate": ["2025-08-01"],
        "reportDate": ["2025-06-30"],
        "acceptanceDateTime": ["2025-08-01T13:00:00Z"],
        "act": ["34"],
        "form": ["10-Q"],
        "fileNumber": ["001-00001"],
        "filmNumber": ["25555555"],
        "items": [""],
        "size": [65432],
        "isXBRL": [1],
        "isInlineXBRL": [1],
        "primaryDocument": ["q2.htm"],
        "primaryDocDescription": ["Quarterly report"],
    }
    fetched_at_utc = "2026-03-15T10:00:00Z"
    root_url = module.SEC_SUBMISSIONS_URL.format(cik=cik)
    root_sha = module.sha256_text(module.stable_json_dumps(root_payload))
    history_sha = module.sha256_text(module.stable_json_dumps(history_payload))

    raw_path = module.raw_payload_json_path(output_dir, asof, run_id, cik)
    envelope = {
        "cik": cik,
        "requested_url": root_url,
        "http_status": 200,
        "fetched_at_utc": fetched_at_utc,
        "run_id": run_id,
        "asof": asof,
        "envelope_sha256": root_sha,
        "root_payload": root_payload,
        "history_payloads": [
            {
                "filename": f"CIK{cik}-submissions-001.json",
                "source_url": module.SEC_SUBMISSIONS_HISTORY_URL.format(filename=f"CIK{cik}-submissions-001.json"),
                "payload_sha256": history_sha,
                "payload": history_payload,
            }
        ],
    }
    module.persist_raw_envelope(raw_path, envelope)

    payload_rows = [
        module.build_payload_index_row(
            cik=cik,
            payload_kind="root_submissions",
            source_url=root_url,
            file_name=None,
            http_status=200,
            fetched_at_utc=fetched_at_utc,
            attempts=1,
            payload_sha256=root_sha,
            payload_json=root_payload,
            raw_json_path=str(raw_path),
            run_id=run_id,
            asof=asof,
        ),
        module.build_payload_index_row(
            cik=cik,
            payload_kind="history_file",
            source_url=module.SEC_SUBMISSIONS_HISTORY_URL.format(filename=f"CIK{cik}-submissions-001.json"),
            file_name=f"CIK{cik}-submissions-001.json",
            http_status=200,
            fetched_at_utc=fetched_at_utc,
            attempts=1,
            payload_sha256=history_sha,
            payload_json=history_payload,
            raw_json_path=str(raw_path),
            run_id=run_id,
            asof=asof,
        ),
    ]

    # Deliberately unsorted; runner should sort deterministically.
    events = [
        _event_row(
            module,
            cik=cik,
            accession_number=f"{cik}-26-000001",
            filing_date="2026-03-01",
            acceptance_datetime="2026-03-01T14:30:00Z",
            run_id=run_id,
            asof=asof,
            payload_sha256=root_sha,
            form_type="10-K",
            primary_doc="annual.htm",
        ),
        _event_row(
            module,
            cik=cik,
            accession_number=f"{cik}-25-000007",
            filing_date="2025-08-01",
            acceptance_datetime="2025-08-01T13:00:00Z",
            run_id=run_id,
            asof=asof,
            payload_sha256=history_sha,
            form_type="10-Q",
            primary_doc="q2.htm",
        ),
    ]

    metrics = {
        "cik": cik,
        "status": "success",
        "http_status": 200,
        "attempts": 1,
        "latency_seconds": 0.0123,
        "fetched_at_utc": fetched_at_utc,
        "payload_path": str(raw_path),
        "source_mode": "network_fetch",
        "history_files_used": 1,
        "history_fetch_failures": 0,
        "root_recent_rows": 1,
        "history_rows": 1,
        "total_rows": 2,
        "payload_rows": 2,
        "exact_duplicates_removed": 0,
        "row_level_failures": 0,
    }

    return module.FetchResult(
        cik=cik,
        url=root_url,
        status="success",
        http_status=200,
        attempts=1,
        fetched_at_utc=fetched_at_utc,
        payload_path=str(raw_path),
        payload_sha256=root_sha,
        payload_rows=payload_rows,
        events=events,
        failures=[],
        metrics=metrics,
    )



def _make_failure_result(module: Any, *, cik: str, run_id: str, asof: str) -> Any:
    url = module.SEC_SUBMISSIONS_URL.format(cik=cik)
    failure = {
        "cik": cik,
        "failure_class": "schema_failure",
        "stage": "payload_validation",
        "reason": "synthetic_schema_failure",
        "source_url": url,
        "run_id": run_id,
        "asof": asof,
    }
    metrics = {
        "cik": cik,
        "status": "schema_failure",
        "http_status": 200,
        "attempts": 1,
        "latency_seconds": 0.004,
        "fetched_at_utc": "2026-03-15T10:00:05Z",
        "payload_path": None,
        "source_mode": "network_fetch",
    }
    return module.FetchResult(
        cik=cik,
        url=url,
        status="schema_failure",
        http_status=200,
        attempts=1,
        fetched_at_utc="2026-03-15T10:00:05Z",
        payload_path=None,
        payload_sha256=None,
        payload_rows=[],
        events=[],
        failures=[failure],
        metrics=metrics,
    )



def _run_pipeline(module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, with_previous_state: bool = False):
    output_dir = tmp_path / "edgar_raw"
    cik_list_path = _write_cik_list(tmp_path / "ciks.json", ["0000320193", "0000789019"])
    config_path = _write_config(tmp_path / "config.json", output_dir=output_dir)
    run_id = "pytest_fetch_submissions_contract"
    asof = "2026-03-15T10:00:00Z"

    def fake_process_single_cik(*, cik: str, client: Any, config: Any, output_dir: Path, asof: str, run_id: str, start_date: Optional[str], end_date: Optional[str]):
        if cik == "0000320193":
            return _make_success_result(module, cik=cik, run_id=run_id, asof=asof, output_dir=output_dir)
        return _make_failure_result(module, cik=cik, run_id=run_id, asof=asof)

    monkeypatch.setattr(module, "process_single_cik", fake_process_single_cik)

    if with_previous_state:
        previous = pd.DataFrame(
            [
                {
                    "cik": "0000320193",
                    "accession_number": "0000320193-25-000007",
                    "event_fingerprint": "different-fingerprint",
                    "run_id": "older_run",
                    "asof": "2026-03-10T10:00:00Z",
                    "filing_date": "2025-08-01",
                },
                {
                    "cik": "0000320193",
                    "accession_number": "0000320193-25-000099",
                    "event_fingerprint": "older-fingerprint",
                    "run_id": "older_run",
                    "asof": "2026-03-10T10:00:00Z",
                    "filing_date": "2025-01-15",
                },
            ]
        )
        monkeypatch.setattr(module, "load_previous_event_state", lambda base_dir, current_asof, current_run_id: previous)
    else:
        monkeypatch.setattr(module, "load_previous_event_state", lambda base_dir, current_asof, current_run_id: pd.DataFrame(columns=["cik", "accession_number", "event_fingerprint", "asof", "run_id", "filing_date"]))

    manifest = module.run_fetch_submissions(
        cik_list_path=str(cik_list_path),
        config_path=str(config_path),
        run_id=run_id,
        asof=asof,
    )
    return manifest, output_dir, run_id, asof



def _read_artifact(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise AssertionError(f"Unsupported artifact extension: {path}")


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_events_output_contains_required_columns_and_unique_identity(fetch_submissions_module, tmp_path, monkeypatch):
    module = fetch_submissions_module
    manifest, _, _, _ = _run_pipeline(module, tmp_path, monkeypatch)

    events = _read_artifact(manifest["artifacts"]["events_table"]["path"])

    assert EVENTS_REQUIRED_COLUMNS.issubset(events.columns)
    assert not events.duplicated(subset=["cik", "accession_number", "event_fingerprint"]).any()



def test_payloads_output_contains_required_columns_and_links_to_raw(fetch_submissions_module, tmp_path, monkeypatch):
    module = fetch_submissions_module
    manifest, _, _, _ = _run_pipeline(module, tmp_path, monkeypatch)

    payloads = _read_artifact(manifest["artifacts"]["payloads_table"]["path"])

    assert PAYLOADS_REQUIRED_COLUMNS.issubset(payloads.columns)
    assert set(payloads["payload_kind"]) == {"root_submissions", "history_file"}
    assert payloads["raw_json_path"].notna().all()
    assert payloads["raw_json_path"].map(lambda p: Path(p).exists()).all()



def test_reconciliation_output_contains_required_columns_and_allowed_classes(fetch_submissions_module, tmp_path, monkeypatch):
    module = fetch_submissions_module
    manifest, _, _, _ = _run_pipeline(module, tmp_path, monkeypatch, with_previous_state=True)

    reconciliation = _read_artifact(manifest["artifacts"]["reconciliation_table"]["path"])

    assert RECONCILIATION_REQUIRED_COLUMNS.issubset(reconciliation.columns)
    assert set(reconciliation["reconciliation_class"].dropna().unique()).issubset({"seen", "novel", "late_arrival", "changed_record"})
    assert "changed_record" in set(reconciliation["reconciliation_class"])



def test_failures_output_contains_required_columns_and_schema_failure(fetch_submissions_module, tmp_path, monkeypatch):
    module = fetch_submissions_module
    manifest, _, _, _ = _run_pipeline(module, tmp_path, monkeypatch)

    failures = _read_artifact(manifest["artifacts"]["failures_table"]["path"])

    assert FAILURES_REQUIRED_COLUMNS.issubset(failures.columns)
    assert "schema_failure" in set(failures["failure_class"].astype(str))



def test_metrics_output_contains_global_row_and_required_columns(fetch_submissions_module, tmp_path, monkeypatch):
    module = fetch_submissions_module
    manifest, _, _, _ = _run_pipeline(module, tmp_path, monkeypatch)

    metrics = _read_artifact(manifest["artifacts"]["metrics_table"]["path"])

    assert METRICS_REQUIRED_COLUMNS.issubset(metrics.columns)
    assert "__GLOBAL__" in set(metrics["cik"].astype(str))
    assert {"success", "schema_failure", "pass", "warn", "fail"}.intersection(set(metrics["status"].astype(str)))



def test_manifest_contains_required_keys_and_artifacts(fetch_submissions_module, tmp_path, monkeypatch):
    module = fetch_submissions_module
    manifest, _, run_id, asof = _run_pipeline(module, tmp_path, monkeypatch)

    assert MANIFEST_REQUIRED_KEYS.issubset(manifest.keys())
    assert ARTIFACT_KEYS.issubset(manifest["artifacts"].keys())
    assert manifest["run_id"] == run_id
    assert manifest["asof"] == asof
    assert manifest["module"] == "data.edgar.fetch_submissions"

    for key in ARTIFACT_KEYS:
        path = Path(manifest["artifacts"][key]["path"])
        assert path.exists(), f"missing artifact for {key}: {path}"



def test_raw_json_envelope_is_persisted_for_successful_cik(fetch_submissions_module, tmp_path, monkeypatch):
    module = fetch_submissions_module
    manifest, output_dir, run_id, asof = _run_pipeline(module, tmp_path, monkeypatch)

    raw_path = module.raw_payload_json_path(output_dir, asof, run_id, "0000320193")
    assert raw_path.exists()

    envelope = json.loads(raw_path.read_text(encoding="utf-8"))
    assert envelope["cik"] == "0000320193"
    assert envelope["run_id"] == run_id
    assert envelope["asof"] == asof



def test_manifest_counts_match_materialized_tables(fetch_submissions_module, tmp_path, monkeypatch):
    module = fetch_submissions_module
    manifest, _, _, _ = _run_pipeline(module, tmp_path, monkeypatch)

    events = _read_artifact(manifest["artifacts"]["events_table"]["path"])
    reconciliation = _read_artifact(manifest["artifacts"]["reconciliation_table"]["path"])

    assert manifest["total_submissions_extracted"] == len(events)
    assert manifest["late_arrivals"] == int((reconciliation["reconciliation_class"] == "late_arrival").sum())
    assert manifest["changed_records"] == int((reconciliation["reconciliation_class"] == "changed_record").sum())



def test_events_are_sorted_deterministically_by_cik_filing_acceptance_accession(fetch_submissions_module, tmp_path, monkeypatch):
    module = fetch_submissions_module
    manifest, _, _, _ = _run_pipeline(module, tmp_path, monkeypatch)

    events = _read_artifact(manifest["artifacts"]["events_table"]["path"])
    expected = events.sort_values(["cik", "filing_date", "acceptance_datetime", "accession_number"], na_position="last").reset_index(drop=True)
    actual = events.reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)



def test_manifest_json_roundtrip_matches_returned_manifest_except_runtime_manifest_pointer(fetch_submissions_module, tmp_path, monkeypatch):
    module = fetch_submissions_module
    manifest, _, _, _ = _run_pipeline(module, tmp_path, monkeypatch)

    manifest_path = Path(manifest["artifacts"]["manifest"]["path"])
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected = json.loads(json.dumps(manifest))
    expected["artifacts"] = dict(expected["artifacts"])
    expected["artifacts"].pop("manifest", None)

    assert loaded == expected
