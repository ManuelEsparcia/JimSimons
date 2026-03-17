from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest


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
            spec = importlib.util.spec_from_file_location("fetch_submissions", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
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


def _recent_node(
    *,
    accession: str = "0000320193-26-000001",
    filing_date: str = "2026-03-01",
    acceptance: str = "2026-03-01T14:30:00Z",
    form: str = "10-K",
    report_date: str = "2025-12-31",
    primary_doc: str = "a10-k.htm",
) -> Dict[str, Any]:
    return {
        "accessionNumber": [accession],
        "filingDate": [filing_date],
        "reportDate": [report_date],
        "acceptanceDateTime": [acceptance],
        "act": ["34"],
        "form": [form],
        "fileNumber": ["001-36743"],
        "filmNumber": ["26654321"],
        "items": [""],
        "size": [123456],
        "isXBRL": [1],
        "isInlineXBRL": [1],
        "primaryDocument": [primary_doc],
        "primaryDocDescription": ["Annual report"],
    }



def _root_payload(cik: str = "0000320193") -> Dict[str, Any]:
    return {
        "cik": cik,
        "name": "APPLE INC",
        "tickers": ["AAPL"],
        "exchanges": ["Nasdaq"],
        "sic": "3571",
        "sicDescription": "Electronic Computers",
        "stateOfIncorporation": "CA",
        "fiscalYearEnd": "0927",
        "filings": {
            "recent": _recent_node(),
            "files": [
                {
                    "name": "CIK0000320193-submissions-001.json",
                    "filingCount": 50,
                    "filingFrom": "2024-01-01",
                    "filingTo": "2025-12-31",
                }
            ],
        },
    }



def _history_payload(
    *,
    accession: str = "0000320193-24-000123",
    filing_date: str = "2024-11-01",
    acceptance: str = "2024-11-01T16:30:00Z",
) -> Dict[str, Any]:
    return _recent_node(
        accession=accession,
        filing_date=filing_date,
        acceptance=acceptance,
        form="10-K",
        report_date="2024-09-28",
        primary_doc="old10-k.htm",
    )


class _DummyResponse:
    def __init__(self, status_code: int, payload: Optional[dict[str, Any]] = None, *, json_exc: Optional[Exception] = None):
        self.status_code = status_code
        self._payload = payload
        self._json_exc = json_exc

    def json(self):
        if self._json_exc is not None:
            raise self._json_exc
        return self._payload


class _DummyClient:
    def __init__(self, result_tuple):
        self._result_tuple = result_tuple
        self.calls: List[str] = []

    def fetch_json(self, url: str):
        self.calls.append(url)
        return self._result_tuple


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_normalize_cik_zero_pads_and_strips_noise(fetch_submissions_module):
    module = fetch_submissions_module

    assert module.normalize_cik(" 320193 ") == "0000320193"
    assert module.normalize_cik("CIK 320193") == "0000320193"
    assert module.normalize_cik("0000320193") == "0000320193"



def test_sec_client_retries_only_for_transient_http_errors_and_then_succeeds(fetch_submissions_module, monkeypatch):
    module = fetch_submissions_module
    cfg = module.IngestConfig(max_retries=3, backoff_base_seconds=0.001, backoff_max_seconds=0.001)
    cfg.validate()
    client = module.SecSubmissionsClient(cfg, module.RateLimiter(1000.0))

    responses = [
        _DummyResponse(429),
        _DummyResponse(503),
        _DummyResponse(200, payload=_root_payload("0000320193")),
    ]
    calls: List[str] = []
    sleeps: List[int] = []

    def fake_get(*args, **kwargs):
        calls.append(args[0])
        return responses.pop(0)

    monkeypatch.setattr(module.requests, "get", fake_get)
    monkeypatch.setattr(client.rate_limiter, "acquire", lambda: None)
    monkeypatch.setattr(client, "_sleep_backoff", lambda attempts: sleeps.append(attempts))

    status, payload, http_status, attempts, reason = client.fetch_json("https://data.sec.gov/submissions/CIK0000320193.json")

    assert status == "success"
    assert payload is not None and payload["cik"] == "0000320193"
    assert http_status == 200
    assert attempts == 3
    assert reason is None
    assert len(calls) == 3
    assert sleeps == [1, 2]



def test_sec_client_does_not_retry_permanent_http_errors(fetch_submissions_module, monkeypatch):
    module = fetch_submissions_module
    cfg = module.IngestConfig(max_retries=5)
    cfg.validate()
    client = module.SecSubmissionsClient(cfg, module.RateLimiter(1000.0))

    calls: List[str] = []
    monkeypatch.setattr(client.rate_limiter, "acquire", lambda: None)
    monkeypatch.setattr(client, "_sleep_backoff", lambda attempts: (_ for _ in ()).throw(AssertionError("should not backoff")))

    def fake_get(*args, **kwargs):
        calls.append(args[0])
        return _DummyResponse(404)

    monkeypatch.setattr(module.requests, "get", fake_get)

    status, payload, http_status, attempts, reason = client.fetch_json("https://data.sec.gov/submissions/CIK0000320193.json")

    assert status == "permanent_failure"
    assert payload is None
    assert http_status == 404
    assert attempts == 1
    assert "permanent_http_404" in str(reason)
    assert len(calls) == 1



def test_validate_submissions_payload_rejects_payload_cik_mismatch(fetch_submissions_module):
    module = fetch_submissions_module
    payload = _root_payload("0000320193")
    payload["cik"] = "0000789019"

    with pytest.raises(module.PayloadSchemaError, match="payload cik mismatch"):
        module.validate_submissions_payload(payload, "0000320193")



def test_validate_recent_node_rejects_misaligned_critical_arrays(fetch_submissions_module):
    module = fetch_submissions_module
    recent = _recent_node()
    recent["filingDate"] = ["2026-03-01", "2026-03-02"]

    with pytest.raises(module.PayloadSchemaError, match="critical arrays not aligned"):
        module.validate_recent_node(recent, node_name="filings.recent")



def test_extract_events_emits_temporal_validation_failure_but_keeps_row(fetch_submissions_module):
    module = fetch_submissions_module
    node = _recent_node(filing_date="2026-03-03", acceptance="2026-03-01T14:30:00Z")

    rows, failures, metrics = module.extract_events_from_node(
        cik="0000320193",
        node=node,
        source_payload_url="https://data.sec.gov/submissions/CIK0000320193.json",
        source_scope="filings.recent",
        source_file_name=None,
        run_id="pytest_run",
        asof="2026-03-15T10:00:00Z",
        payload_sha256="abc123",
        payload_kind="root_submissions",
        entity_meta={"entity_name": "APPLE INC", "sic": "3571", "sic_description": "Electronic Computers", "state_of_incorporation": "CA", "fiscal_year_end": "0927"},
    )

    assert len(rows) == 1
    assert rows[0]["acceptance_datetime"] == "2026-03-01T14:30:00Z"
    assert rows[0]["filing_date"] == "2026-03-03"
    assert any(f["reason"] == "filing_date_gt_acceptance_datetime" for f in failures)
    assert metrics["temporal_inconsistencies"] == 1
    assert metrics["rows_valid_post_dedup"] == 1



def test_reconcile_current_vs_previous_classifies_seen_novel_late_and_changed(fetch_submissions_module):
    module = fetch_submissions_module

    previous = pd.DataFrame(
        [
            {
                "cik": "0000320193",
                "accession_number": "0000320193-24-000123",
                "event_fingerprint": "fp_seen",
                "run_id": "prev_run",
                "asof": "2026-03-10T00:00:00Z",
                "filing_date": "2024-11-01",
            },
            {
                "cik": "0000320193",
                "accession_number": "0000320193-25-000777",
                "event_fingerprint": "fp_old",
                "run_id": "prev_run",
                "asof": "2026-03-10T00:00:00Z",
                "filing_date": "2025-02-15",
            },
        ]
    )
    current = pd.DataFrame(
        [
            {
                "cik": "0000320193",
                "accession_number": "0000320193-24-000123",
                "event_fingerprint": "fp_seen",
                "run_id": "cur_run",
                "asof": "2026-03-15T00:00:00Z",
                "filing_date": "2024-11-01",
            },
            {
                "cik": "0000320193",
                "accession_number": "0000320193-26-000001",
                "event_fingerprint": "fp_novel",
                "run_id": "cur_run",
                "asof": "2026-03-15T00:00:00Z",
                "filing_date": "2026-03-12",
            },
            {
                "cik": "0000320193",
                "accession_number": "0000320193-23-000005",
                "event_fingerprint": "fp_late",
                "run_id": "cur_run",
                "asof": "2026-03-15T00:00:00Z",
                "filing_date": "2023-12-20",
            },
            {
                "cik": "0000320193",
                "accession_number": "0000320193-25-000777",
                "event_fingerprint": "fp_new_changed",
                "run_id": "cur_run",
                "asof": "2026-03-15T00:00:00Z",
                "filing_date": "2025-02-15",
            },
        ]
    )

    rec = module.reconcile_current_vs_previous(current, previous)
    classes = dict(zip(rec["accession_number"], rec["reconciliation_class"]))

    assert classes["0000320193-24-000123"] == "seen"
    assert classes["0000320193-26-000001"] == "novel"
    assert classes["0000320193-23-000005"] == "late_arrival"
    assert classes["0000320193-25-000777"] == "changed_record"



def test_load_cik_list_json_normalizes_deduplicates_and_sorts(fetch_submissions_module, tmp_path):
    module = fetch_submissions_module
    path = tmp_path / "ciks.json"
    path.write_text(json.dumps({"ciks": ["320193", "0000789019", " 320193 "]}), encoding="utf-8")

    ciks = module.load_cik_list(str(path))

    assert ciks == ["0000320193", "0000789019"]



def test_process_single_cik_reuses_existing_raw_without_network(fetch_submissions_module, tmp_path):
    module = fetch_submissions_module
    cfg = module.IngestConfig(output_dir=str(tmp_path), table_format="csv", skip_existing_success=True, force_refresh=False)
    cfg.validate()

    cik = "0000320193"
    run_id = "pytest_run"
    asof = "2026-03-15T10:00:00Z"
    raw_path = module.raw_payload_json_path(tmp_path, asof, run_id, cik)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    root_payload = _root_payload(cik)
    history_payloads = [
        {
            "name": "CIK0000320193-submissions-001.json",
            "url": module.SEC_SUBMISSIONS_HISTORY_URL.format(filename="CIK0000320193-submissions-001.json"),
            "http_status": 200,
            "attempts": 1,
            "payload_sha256": module.sha256_text(module.stable_json_dumps(_history_payload())),
            "payload": _history_payload(),
            "filingCount": 50,
            "filingFrom": "2024-01-01",
            "filingTo": "2025-12-31",
        }
    ]
    envelope = {
        "schema_version": "1.0",
        "module": "data.edgar.fetch_submissions",
        "cik": cik,
        "requested_url": module.SEC_SUBMISSIONS_URL.format(cik=cik),
        "http_status": 200,
        "fetched_at_utc": "2026-03-15T10:00:01Z",
        "run_id": run_id,
        "asof": asof,
        "entity_meta": {
            "entity_name": "APPLE INC",
            "sic": "3571",
            "sic_description": "Electronic Computers",
            "state_of_incorporation": "CA",
            "fiscal_year_end": "0927",
        },
        "main_payload_sha256": module.sha256_text(module.stable_json_dumps(root_payload)),
        "main_payload": root_payload,
        "history_payloads": history_payloads,
        "history_fetch_failures": [],
        "envelope_sha256": "manual-test-envelope",
    }
    raw_path.write_text(json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

    client = _DummyClient(("retryable_failure", None, 503, 9, "should_not_be_called"))
    result = module.process_single_cik(
        cik=cik,
        client=client,
        config=cfg,
        output_dir=Path(cfg.output_dir),
        asof=asof,
        run_id=run_id,
        start_date=None,
        end_date=None,
    )

    assert result.status == "success"
    assert result.metrics["source_mode"] == "reuse_existing_raw"
    assert result.attempts == 0
    assert client.calls == []
    assert result.payload_path == str(raw_path)
    assert len(result.events) >= 2  # root + history event
    assert {row["payload_kind"] for row in result.payload_rows} == {"root_submissions", "history_submissions"}



def test_classify_run_gate_distinguishes_pass_warn_and_fail(fetch_submissions_module):
    module = fetch_submissions_module
    cfg = module.IngestConfig(
        min_coverage_ratio=0.80,
        systemic_source_failure_ratio=0.50,
        payload_schema_failure_ratio_warn=0.10,
    )
    cfg.validate()

    assert module.classify_run_gate(
        total_ciks=10,
        success_ciks=9,
        retryable_failures=0,
        permanent_failures=0,
        schema_failures=0,
        local_failures=0,
        config=cfg,
    ) == ("pass", "ok")

    assert module.classify_run_gate(
        total_ciks=10,
        success_ciks=9,
        retryable_failures=0,
        permanent_failures=0,
        schema_failures=1,
        local_failures=0,
        config=cfg,
    ) == ("warn", "payload_schema_failure")

    assert module.classify_run_gate(
        total_ciks=10,
        success_ciks=4,
        retryable_failures=6,
        permanent_failures=0,
        schema_failures=0,
        local_failures=0,
        config=cfg,
    ) == ("fail", "systemic_source_failure")
