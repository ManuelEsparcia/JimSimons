from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd
import pytest


FACTS_REQUIRED_COLUMNS = {
    "cik",
    "entity_name",
    "taxonomy",
    "tag",
    "unit",
    "end_date",
    "filed_date",
    "value",
    "form",
    "accession_number",
    "source_url",
    "payload_sha256",
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
    "http_status",
    "attempts",
    "latency_seconds",
    "fetched_at_utc",
    "payload_path",
    "source_mode",
}

PAYLOAD_INDEX_REQUIRED_COLUMNS = {
    "cik",
    "status",
    "http_status",
    "attempts",
    "fetched_at_utc",
    "payload_path",
    "payload_sha256",
    "run_id",
    "asof",
}

MANIFEST_REQUIRED_KEYS = {
    "module",
    "run_id",
    "asof",
    "config_hash",
    "config",
    "started_at_utc",
    "ended_at_utc",
    "requested_cik_count",
    "success_cik_count",
    "retryable_failure_cik_count",
    "permanent_failure_cik_count",
    "schema_failure_cik_count",
    "coverage_ratio",
    "facts_total_rows",
    "failures_total_rows",
    "metrics_total_rows",
    "payload_index_total_rows",
    "gate",
    "gate_reason",
    "taxonomy_filter",
    "taxonomies_seen",
    "discarded_taxonomies_seen",
    "artifacts",
}

ARTIFACT_KEYS = {"facts", "failures", "metrics", "payload_index", "raw_payload_dir"}


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
            spec = importlib.util.spec_from_file_location("fetch_companyfacts_contract_under_test", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
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
        },
    }


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _write_cik_list(path: Path, values: Iterable[str]) -> Path:
    payload = list(values)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path



def _make_success_result(module: Any, *, cik: str, run_id: str, asof: str, output_dir: Path) -> Any:
    payload = _valid_payload(cik)
    payload_sha256 = module.sha256_text(module.stable_json_dumps(payload))
    url = module.SEC_COMPANYFACTS_URL.format(cik=cik)
    fetched_at_utc = "2026-03-15T10:00:00Z"
    raw_path = module.raw_payload_path(output_dir, asof, run_id, cik)
    envelope = {
        "cik": cik,
        "requested_url": url,
        "http_status": 200,
        "fetched_at_utc": fetched_at_utc,
        "run_id": run_id,
        "asof": asof,
        "payload_sha256": payload_sha256,
        "payload": payload,
    }
    module.persist_raw_envelope(raw_path, envelope)
    facts, failures, metrics = module.extract_minimal_facts(
        cik=cik,
        payload=payload,
        taxonomy_filter={"us-gaap", "dei"},
        run_id=run_id,
        asof=asof,
        source_url=url,
        payload_sha256=payload_sha256,
    )
    metrics = {
        "cik": cik,
        "status": "success",
        "http_status": 200,
        "attempts": 1,
        "latency_seconds": 0.012345,
        "fetched_at_utc": fetched_at_utc,
        "payload_path": str(raw_path),
        "payload_sha256": payload_sha256,
        "source_mode": "network_fetch",
        **metrics,
    }
    return module.FetchResult(
        cik=cik,
        url=url,
        status="success",
        http_status=200,
        attempts=1,
        fetched_at_utc=fetched_at_utc,
        payload_path=str(raw_path),
        payload_sha256=payload_sha256,
        facts=facts,
        failures=failures,
        metrics=metrics,
    )



def _make_failure_result(module: Any, *, cik: str, run_id: str, asof: str) -> Any:
    url = module.SEC_COMPANYFACTS_URL.format(cik=cik)
    failure = {
        "cik": cik,
        "failure_class": "schema_failure",
        "stage": "payload_validation",
        "http_status": 200,
        "attempts": 1,
        "reason": "payload.facts missing or empty",
        "source_url": url,
        "run_id": run_id,
        "asof": asof,
    }
    metrics = {
        "cik": cik,
        "status": "schema_failure",
        "http_status": 200,
        "attempts": 1,
        "latency_seconds": 0.009876,
        "fetched_at_utc": "2026-03-15T10:00:01Z",
        "payload_path": None,
        "source_mode": "network_fetch",
    }
    return module.FetchResult(
        cik=cik,
        url=url,
        status="schema_failure",
        http_status=200,
        attempts=1,
        fetched_at_utc="2026-03-15T10:00:01Z",
        payload_path=None,
        payload_sha256=None,
        facts=[],
        failures=[failure],
        metrics=metrics,
    )



def _run_pipeline(module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output_dir = tmp_path / "edgar_out"
    ciks = ["0000320193", "0000789019"]
    cik_list_path = _write_cik_list(tmp_path / "cik_list.json", ciks)
    run_id = "pytest_fetch_companyfacts_contract"
    asof = "2026-03-15T10:00:00Z"

    cfg = module.IngestConfig(
        output_dir=str(output_dir),
        table_format="csv",
        allow_csv_fallback=True,
        skip_existing_success=False,
        force_refresh=False,
        max_workers=1,
    )
    cfg.validate()

    def fake_process_single_cik(*, cik, client, config, output_dir, asof, run_id, taxonomy_filter):
        if cik == "0000320193":
            return _make_success_result(module, cik=cik, run_id=run_id, asof=asof, output_dir=output_dir)
        return _make_failure_result(module, cik=cik, run_id=run_id, asof=asof)

    monkeypatch.setattr(module, "process_single_cik", fake_process_single_cik)

    result = module.run_ingestion(
        cik_list_path=str(cik_list_path),
        config=cfg,
        run_id=run_id,
        asof=asof,
    )
    return result, cfg, output_dir, run_id, asof


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_minimal_facts_output_contains_required_columns(fetch_companyfacts_module, tmp_path, monkeypatch):
    module = fetch_companyfacts_module
    result, _, _, _, _ = _run_pipeline(module, tmp_path, monkeypatch)

    facts_df = result["facts_df"].copy()
    assert not facts_df.empty
    assert FACTS_REQUIRED_COLUMNS.issubset(facts_df.columns)
    assert set(facts_df["taxonomy"]) <= {"us-gaap", "dei"}
    assert facts_df["run_id"].nunique() == 1
    assert facts_df["asof"].nunique() == 1



def test_failures_output_contains_failure_class_and_cik(fetch_companyfacts_module, tmp_path, monkeypatch):
    module = fetch_companyfacts_module
    result, _, _, _, _ = _run_pipeline(module, tmp_path, monkeypatch)

    failures_df = result["failures_df"].copy()
    assert not failures_df.empty
    assert FAILURES_REQUIRED_COLUMNS.issubset(failures_df.columns)
    assert "schema_failure" in set(failures_df["failure_class"])
    assert "0000789019" in set(failures_df["cik"])
    assert failures_df["run_id"].nunique() == 1



def test_metrics_output_contains_required_columns(fetch_companyfacts_module, tmp_path, monkeypatch):
    module = fetch_companyfacts_module
    result, _, _, _, _ = _run_pipeline(module, tmp_path, monkeypatch)

    metrics_df = result["metrics_df"].copy()
    assert len(metrics_df) == 2
    assert METRICS_REQUIRED_COLUMNS.issubset(metrics_df.columns)
    assert set(metrics_df["status"]) == {"success", "schema_failure"}
    assert set(metrics_df["source_mode"]) == {"network_fetch"}



def test_payload_index_output_contains_required_columns_and_run_context(fetch_companyfacts_module, tmp_path, monkeypatch):
    module = fetch_companyfacts_module
    result, _, _, run_id, asof = _run_pipeline(module, tmp_path, monkeypatch)

    payload_index_df = result["payload_index_df"].sort_values("cik", kind="mergesort").reset_index(drop=True)
    assert len(payload_index_df) == 2
    assert PAYLOAD_INDEX_REQUIRED_COLUMNS.issubset(payload_index_df.columns)
    assert payload_index_df["run_id"].eq(run_id).all()
    assert payload_index_df["asof"].eq(asof).all()
    assert payload_index_df["cik"].tolist() == sorted(payload_index_df["cik"].tolist())



def test_manifest_contains_required_metadata_and_artifact_paths(fetch_companyfacts_module, tmp_path, monkeypatch):
    module = fetch_companyfacts_module
    result, cfg, output_dir, run_id, asof = _run_pipeline(module, tmp_path, monkeypatch)

    manifest = result["manifest"]
    manifest_path = Path(result["manifest_path"])

    assert MANIFEST_REQUIRED_KEYS.issubset(manifest.keys())
    assert manifest["module"] == "data.edgar.fetch_companyfacts"
    assert manifest["run_id"] == run_id
    assert manifest["asof"] == asof
    assert manifest["config_hash"] == module.config_hash(cfg)
    assert manifest["requested_cik_count"] == 2
    assert manifest["success_cik_count"] == 1
    assert manifest["schema_failure_cik_count"] == 1
    assert manifest["coverage_ratio"] == pytest.approx(0.5)
    assert set(manifest["artifacts"].keys()) == ARTIFACT_KEYS
    assert manifest_path.exists()
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk == manifest
    for key in ["facts", "failures", "metrics", "payload_index"]:
        assert Path(manifest["artifacts"][key]).exists(), key
    assert Path(manifest["artifacts"]["raw_payload_dir"]).parent.exists()
    assert str(output_dir) in manifest["artifacts"]["facts"]



def test_dual_persistence_raw_and_tabular_artifacts_are_present(fetch_companyfacts_module, tmp_path, monkeypatch):
    module = fetch_companyfacts_module
    result, _, _, _, _ = _run_pipeline(module, tmp_path, monkeypatch)

    manifest = result["manifest"]
    raw_dir = Path(manifest["artifacts"]["raw_payload_dir"])
    raw_files = list(raw_dir.glob("CIK*.json"))

    assert raw_dir.exists()
    assert len(raw_files) == 1
    assert raw_files[0].name == "CIK0000320193.json"

    envelope = json.loads(raw_files[0].read_text(encoding="utf-8"))
    assert envelope["cik"] == "0000320193"
    assert envelope["run_id"] == manifest["run_id"]
    assert envelope["asof"] == manifest["asof"]
    assert "payload" in envelope and isinstance(envelope["payload"], dict)



def test_rows_link_back_to_payload_and_run_context(fetch_companyfacts_module, tmp_path, monkeypatch):
    module = fetch_companyfacts_module
    result, _, _, run_id, asof = _run_pipeline(module, tmp_path, monkeypatch)

    facts_df = result["facts_df"].copy()
    payload_index_df = result["payload_index_df"].copy()

    success_index = payload_index_df.loc[payload_index_df["status"] == "success", ["cik", "payload_path", "payload_sha256"]]
    joined = facts_df.merge(success_index, on="cik", how="left", suffixes=("", "_idx"))

    assert joined["payload_path"].notna().all()
    assert joined["payload_sha256"].notna().all()
    assert joined["run_id"].eq(run_id).all()
    assert joined["asof"].eq(asof).all()
    assert joined["payload_sha256"].eq(joined["payload_sha256_idx"]).all()



def test_manifest_counts_are_consistent_with_output_tables(fetch_companyfacts_module, tmp_path, monkeypatch):
    module = fetch_companyfacts_module
    result, _, _, _, _ = _run_pipeline(module, tmp_path, monkeypatch)

    manifest = result["manifest"]
    facts_df = result["facts_df"]
    failures_df = result["failures_df"]
    metrics_df = result["metrics_df"]
    payload_index_df = result["payload_index_df"]

    assert manifest["facts_total_rows"] == len(facts_df)
    assert manifest["failures_total_rows"] == len(failures_df)
    assert manifest["metrics_total_rows"] == len(metrics_df)
    assert manifest["payload_index_total_rows"] == len(payload_index_df)
    assert manifest["taxonomies_seen"] == ["dei", "us-gaap"]
    assert manifest["discarded_taxonomies_seen"] == []
    assert manifest["gate"] == "fail"
    assert manifest["gate_reason"] == "coverage_collapse"



def test_artifact_files_roundtrip_to_same_row_counts(fetch_companyfacts_module, tmp_path, monkeypatch):
    module = fetch_companyfacts_module
    result, _, _, _, _ = _run_pipeline(module, tmp_path, monkeypatch)
    manifest = result["manifest"]

    facts_disk = pd.read_csv(manifest["artifacts"]["facts"])
    failures_disk = pd.read_csv(manifest["artifacts"]["failures"])
    metrics_disk = pd.read_csv(manifest["artifacts"]["metrics"])
    payload_index_disk = pd.read_csv(manifest["artifacts"]["payload_index"])

    assert len(facts_disk) == len(result["facts_df"])
    assert len(failures_disk) == len(result["failures_df"])
    assert len(metrics_disk) == len(result["metrics_df"])
    assert len(payload_index_disk) == len(result["payload_index_df"])
    assert set(facts_disk.columns) >= FACTS_REQUIRED_COLUMNS
    assert set(failures_disk.columns) >= FAILURES_REQUIRED_COLUMNS
