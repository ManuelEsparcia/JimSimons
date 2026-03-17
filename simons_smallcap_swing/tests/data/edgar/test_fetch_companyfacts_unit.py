from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd
import pandas.testing as pdt
import pytest


# -----------------------------------------------------------------------------
# Import adapter
# -----------------------------------------------------------------------------


def _load_fetch_companyfacts_module():
    module_names = [
        "simons_smallcap_swing.data.edgar.fetch_companyfacts",
        "data.edgar.fetch_companyfacts",
        "fetch_companyfacts",
    ]
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue

    here = Path(__file__).resolve()
    candidate_paths = [
        here.parents[3] / "simons_smallcap_swing" / "data" / "edgar" / "fetch_companyfacts.py",
        here.parents[2] / "data" / "edgar" / "fetch_companyfacts.py",
        here.parents[1] / "fetch_companyfacts.py",
        Path("/mnt/data/fetch_companyfacts.py"),
    ]
    for path in candidate_paths:
        if path.exists():
            spec = importlib.util.spec_from_file_location("fetch_companyfacts", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module

    raise ModuleNotFoundError(
        "Could not import fetch_companyfacts.py. Expected it at "
        "simons_smallcap_swing.data.edgar.fetch_companyfacts or in a nearby repo path."
    )


@pytest.fixture(scope="session")
def fetch_companyfacts_module():
    return _load_fetch_companyfacts_module()


# -----------------------------------------------------------------------------
# Synthetic builders
# -----------------------------------------------------------------------------


def _valid_payload(cik: str = "0000320193") -> Dict[str, Any]:
    return {
        "cik": cik,
        "entityName": "APPLE INC",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "label": "Assets",
                    "description": "Total assets",
                    "units": {
                        "USD": [
                            {
                                "end": "2024-09-28",
                                "filed": "2024-11-01",
                                "val": 352583000000,
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "accn": "0000320193-24-000123",
                            }
                        ]
                    },
                }
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "label": "Shares Outstanding",
                    "description": "Common shares outstanding",
                    "units": {
                        "shares": [
                            {
                                "end": "2024-09-28",
                                "filed": "2024-11-01",
                                "val": 15204137000,
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "accn": "0000320193-24-000123",
                                "frame": "CY2024Q4I",
                            }
                        ]
                    },
                }
            },
            "ifrs-full": {
                "Revenue": {
                    "label": "Revenue",
                    "description": "Revenue",
                    "units": {
                        "USD": [
                            {
                                "end": "2024-09-28",
                                "filed": "2024-11-01",
                                "val": 1,
                                "form": "10-K",
                                "accn": "0000320193-24-000123",
                            }
                        ]
                    },
                }
            },
        },
    }


class _DummyResponse:
    def __init__(self, status_code: int, payload: Optional[dict[str, Any]] = None, *, json_exc: Optional[Exception] = None):
        self.status_code = status_code
        self._payload = payload
        self._json_exc = json_exc

    def json(self):
        if self._json_exc is not None:
            raise self._json_exc
        return self._payload


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_normalize_cik_zero_pads_and_strips_noise(fetch_companyfacts_module):
    module = fetch_companyfacts_module

    assert module.normalize_cik(" 320193 ") == "0000320193"
    assert module.normalize_cik("CIK 320193") == "0000320193"
    assert module.normalize_cik("0000320193") == "0000320193"



def test_sec_client_retries_only_for_transient_http_errors_and_then_succeeds(fetch_companyfacts_module, monkeypatch):
    module = fetch_companyfacts_module
    cfg = module.IngestConfig(max_retries=3, backoff_base_seconds=0.001, backoff_max_seconds=0.001)
    cfg.validate()
    rate_limiter = module.RateLimiter(1000.0)
    client = module.SecCompanyFactsClient(cfg, rate_limiter)

    responses = [
        _DummyResponse(429),
        _DummyResponse(503),
        _DummyResponse(200, payload=_valid_payload("0000320193")),
    ]
    calls: List[str] = []
    sleeps: List[int] = []

    def fake_get(*args, **kwargs):
        calls.append(args[0])
        return responses.pop(0)

    monkeypatch.setattr(module.requests, "get", fake_get)
    monkeypatch.setattr(client.rate_limiter, "acquire", lambda: None)
    monkeypatch.setattr(client, "_sleep_backoff", lambda attempts: sleeps.append(attempts))

    status, payload, http_status, attempts, reason = client.fetch_json("0000320193")

    assert status == "success"
    assert payload is not None and payload["cik"] == "0000320193"
    assert http_status == 200
    assert attempts == 3
    assert reason is None
    assert len(calls) == 3
    assert sleeps == [1, 2]



def test_sec_client_does_not_retry_permanent_http_errors(fetch_companyfacts_module, monkeypatch):
    module = fetch_companyfacts_module
    cfg = module.IngestConfig(max_retries=5)
    cfg.validate()
    client = module.SecCompanyFactsClient(cfg, module.RateLimiter(1000.0))

    calls: List[str] = []
    monkeypatch.setattr(client.rate_limiter, "acquire", lambda: None)
    monkeypatch.setattr(client, "_sleep_backoff", lambda attempts: (_ for _ in ()).throw(AssertionError("should not backoff")))

    def fake_get(*args, **kwargs):
        calls.append(args[0])
        return _DummyResponse(404)

    monkeypatch.setattr(module.requests, "get", fake_get)

    status, payload, http_status, attempts, reason = client.fetch_json("0000320193")

    assert status == "permanent_failure"
    assert payload is None
    assert http_status == 404
    assert attempts == 1
    assert "permanent_http_404" in str(reason)
    assert len(calls) == 1



def test_validate_payload_structure_rejects_payload_cik_mismatch(fetch_companyfacts_module):
    module = fetch_companyfacts_module
    payload = _valid_payload("0000320193")
    payload["cik"] = "0000789019"

    with pytest.raises(module.PayloadSchemaError, match="payload cik mismatch"):
        module.validate_payload_structure(payload, "0000320193")



def test_extract_minimal_facts_filters_taxonomy_and_deduplicates_exact_duplicates(fetch_companyfacts_module):
    module = fetch_companyfacts_module
    payload = _valid_payload("0000320193")

    # add an exact duplicate inside us-gaap/Assets/USD
    payload["facts"]["us-gaap"]["Assets"]["units"]["USD"].append(
        dict(payload["facts"]["us-gaap"]["Assets"]["units"]["USD"][0])
    )

    rows, failures, metrics = module.extract_minimal_facts(
        cik="0000320193",
        payload=payload,
        taxonomy_filter={"us-gaap", "dei"},
        run_id="pytest_run",
        asof="2026-03-15T10:00:00Z",
        source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        payload_sha256="abc123",
    )

    df = pd.DataFrame(rows).sort_values(["taxonomy", "tag"], kind="mergesort").reset_index(drop=True)

    assert failures == []
    assert set(df["taxonomy"]) == {"us-gaap", "dei"}
    assert "ifrs-full" not in set(df["taxonomy"])
    assert len(df) == 2
    assert metrics["facts_extracted_pre_dedup"] == 3
    assert metrics["facts_extracted_post_dedup"] == 2
    assert metrics["exact_duplicates_removed"] == 1
    assert metrics["discarded_taxonomies"] == ["ifrs-full"]
    assert set(df.columns) >= set(module.REQUIRED_FACT_FIELDS)
    assert df.loc[df["taxonomy"] == "us-gaap", "payload_sha256"].iloc[0] == "abc123"



def test_extract_minimal_facts_emits_row_level_failure_for_missing_required_fields(fetch_companyfacts_module):
    module = fetch_companyfacts_module
    payload = {
        "cik": "0000320193",
        "entityName": "APPLE INC",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "label": "Assets",
                    "description": "Total assets",
                    "units": {
                        "USD": [
                            {
                                "end": "2024-09-28",
                                # filed intentionally missing -> required field failure
                                "val": 352583000000,
                                "form": "10-K",
                                "accn": "0000320193-24-000123",
                            }
                        ]
                    },
                }
            }
        },
    }

    rows, failures, metrics = module.extract_minimal_facts(
        cik="0000320193",
        payload=payload,
        taxonomy_filter={"us-gaap"},
        run_id="pytest_run",
        asof="2026-03-15T10:00:00Z",
        source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        payload_sha256="abc123",
    )

    assert rows == []
    assert len(failures) == 1
    failure = failures[0]
    assert failure["failure_class"] == "row_level_parse_issue"
    assert failure["stage"] == "tabular_extraction"
    assert "filed_date" in failure["reason"]
    assert metrics["row_level_failures"] == 1
    assert metrics["facts_extracted_post_dedup"] == 0



def test_load_cik_list_json_normalizes_deduplicates_and_sorts(fetch_companyfacts_module, tmp_path):
    module = fetch_companyfacts_module
    cik_path = tmp_path / "ciks.json"
    cik_path.write_text(json.dumps({"ciks": ["320193", "0000320193", "789019", "320193"]}), encoding="utf-8")

    out = module.load_cik_list(str(cik_path))

    assert out == ["0000320193", "0000789019"]



def test_classify_run_gate_captures_systemic_failure_warn_and_pass(fetch_companyfacts_module):
    module = fetch_companyfacts_module
    cfg = module.IngestConfig(
        min_coverage_ratio=0.80,
        systemic_api_failure_ratio=0.50,
        payload_anomaly_ratio_warn=0.20,
    )
    cfg.validate()

    gate, reason = module.classify_run_gate(
        total_ciks=10,
        success_ciks=4,
        retryable_failures=4,
        permanent_failures=1,
        schema_failures=0,
        input_failures=0,
        config=cfg,
    )
    assert (gate, reason) == ("fail", "systemic_api_failure")

    gate, reason = module.classify_run_gate(
        total_ciks=10,
        success_ciks=9,
        retryable_failures=0,
        permanent_failures=0,
        schema_failures=2,
        input_failures=0,
        config=cfg,
    )
    assert (gate, reason) == ("warn", "payload_anomaly")

    gate, reason = module.classify_run_gate(
        total_ciks=10,
        success_ciks=9,
        retryable_failures=0,
        permanent_failures=0,
        schema_failures=1,
        input_failures=0,
        config=cfg,
    )
    assert (gate, reason) == ("pass", "ok")



def test_process_single_cik_reuses_existing_raw_without_network(fetch_companyfacts_module, tmp_path, monkeypatch):
    module = fetch_companyfacts_module
    cfg = module.IngestConfig(
        output_dir=str(tmp_path / "out"),
        skip_existing_success=True,
        force_refresh=False,
        table_format="csv",
    )
    cfg.validate()

    client = module.SecCompanyFactsClient(cfg, module.RateLimiter(1000.0))
    output_dir = Path(cfg.output_dir)
    asof = "2026-03-15T10:00:00Z"
    run_id = "pytest_reuse"
    cik = "0000320193"
    payload = _valid_payload(cik)

    payload_file = module.raw_payload_path(output_dir, asof, run_id, cik)
    envelope = {
        "cik": cik,
        "url": module.SEC_COMPANYFACTS_URL.format(cik=cik),
        "http_status": 200,
        "fetched_at_utc": "2026-03-15T10:00:00Z",
        "payload_sha256": "reuse_sha",
        "payload": payload,
    }
    module.persist_raw_envelope(payload_file, envelope)

    monkeypatch.setattr(client, "fetch_json", lambda cik: (_ for _ in ()).throw(AssertionError("network should not be hit")))

    result = module.process_single_cik(
        cik=cik,
        client=client,
        config=cfg,
        output_dir=output_dir,
        asof=asof,
        run_id=run_id,
        taxonomy_filter={"us-gaap", "dei"},
    )

    assert result.status == "success"
    assert result.http_status == 200
    assert result.payload_path is not None and result.payload_path.endswith("CIK0000320193.json")
    assert result.payload_sha256 == "reuse_sha"
    assert result.metrics["source_mode"] == "reuse_existing_raw"
    assert result.metrics["facts_extracted_post_dedup"] == 2
    assert len(result.facts) == 2
    assert result.failures == []
