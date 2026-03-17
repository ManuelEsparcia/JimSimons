from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# Import adapter
# -----------------------------------------------------------------------------


def _load_parse_xbrl_module():
    module_names = [
        "simons_smallcap_swing.data.edgar.parse_xbrl",
        "data.edgar.parse_xbrl",
        "parse_xbrl",
    ]
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue

    here = Path(__file__).resolve()
    candidate_paths = [
        here.parents[3] / "simons_smallcap_swing" / "data" / "edgar" / "parse_xbrl.py",
        here.parents[2] / "data" / "edgar" / "parse_xbrl.py",
        here.parents[1] / "parse_xbrl.py",
        Path("/mnt/data/parse_xbrl.py"),
    ]
    for path in candidate_paths:
        if path.exists():
            spec = importlib.util.spec_from_file_location("parse_xbrl", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module

    raise ModuleNotFoundError(
        "Could not import parse_xbrl.py. Expected it at "
        "simons_smallcap_swing.data.edgar.parse_xbrl or in a nearby repo path."
    )


@pytest.fixture(scope="session")
def parse_xbrl_module():
    return _load_parse_xbrl_module()


# -----------------------------------------------------------------------------
# Helpers / fixtures
# -----------------------------------------------------------------------------


REQUIRED_CANONICAL_COLS = {
    "cik",
    "symbol",
    "issuer_name",
    "metric_name",
    "metric_family",
    "value",
    "unit_base",
    "period_end",
    "period_type",
    "filed_date",
    "acceptance_datetime",
    "source_taxonomy",
    "source_tag",
    "source_unit",
    "canonical_source_unit",
    "accession_number",
    "form_type",
    "mapping_version",
    "mapping_status",
    "unit_scale_factor",
    "canonical_priority_rank",
    "selection_reason",
    "quality_score",
    "source_record_hash",
    "run_id",
    "asof",
}

REQUIRED_REJECTED_COLS = {
    "cik",
    "taxonomy",
    "tag",
    "unit",
    "period_end",
    "filed_date",
    "acceptance_datetime",
    "rejection_class",
    "rejection_detail",
    "accession_number",
    "form_type",
    "value_raw",
    "run_id",
}

REQUIRED_METRICS_COLS = {"metric_group", "metric_name", "metric_value"}


@pytest.fixture()
def parsed_case(tmp_path: Path, parse_xbrl_module):
    module = parse_xbrl_module
    submissions_dir = tmp_path / "submissions_raw"
    companyfacts_dir = tmp_path / "companyfacts_raw"
    output_dir = tmp_path / "parsed_out"
    submissions_dir.mkdir(parents=True)
    companyfacts_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    cik = "0000320193"

    submissions_payload = {
        "cik": cik,
        "name": "APPLE INC",
        "tickers": ["AAPL"],
        "exchanges": ["NASDAQ"],
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000320193-26-000001",
                    "0000320193-26-000002",
                ],
                "form": ["10-K", "10-Q"],
                "filingDate": ["2026-02-01", "2026-05-01"],
                "acceptanceDateTime": [
                    "2026-02-01T14:30:00Z",
                    "2026-05-01T14:30:00Z",
                ],
            }
        },
    }
    (submissions_dir / f"CIK{cik}.json").write_text(json.dumps(submissions_payload), encoding="utf-8")

    companyfacts_payload = {
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
                                "val": 1000,
                                "end": "2025-12-31",
                                "filed": "2026-02-01",
                                "accepted": "2026-02-01T14:30:00Z",
                                "fy": 2025,
                                "fp": "FY",
                                "form": "10-K",
                                "accn": "0000320193-26-000001",
                            },
                            {
                                "val": 1000,
                                "end": "2025-12-31",
                                "filed": "2026-02-02",
                                "accepted": "2026-02-02T15:00:00Z",
                                "fy": 2025,
                                "fp": "FY",
                                "form": "10-K/A",
                                "accn": "0000320193-26-000001A",
                            },
                        ]
                    },
                },
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "label": "Revenue",
                    "description": "Revenue alias",
                    "units": {
                        "USD": [
                            {
                                "val": 250,
                                "start": "2025-10-01",
                                "end": "2025-12-31",
                                "filed": "2026-02-01",
                                "accepted": "2026-02-01T14:30:00Z",
                                "fy": 2025,
                                "fp": "Q4",
                                "frame": "CY2025Q4I",
                                "form": "10-K",
                                "accn": "0000320193-26-000001",
                            }
                        ]
                    },
                },
                "DefinitelyUnknownTag": {
                    "label": "Unknown",
                    "description": "Unmapped",
                    "units": {
                        "USD": [
                            {
                                "val": 999,
                                "end": "2025-12-31",
                                "filed": "2026-02-01",
                                "accepted": "2026-02-01T14:30:00Z",
                                "fy": 2025,
                                "fp": "FY",
                                "form": "10-K",
                                "accn": "0000320193-26-000001",
                            }
                        ]
                    },
                },
            }
        },
    }
    (companyfacts_dir / f"CIK{cik}.json").write_text(json.dumps(companyfacts_payload), encoding="utf-8")

    config = {
        "output_dir": str(output_dir),
        "table_format": "csv",
        "allow_csv_fallback": True,
        "warn_reject_ratio": 0.95,
        "systemic_mapping_failure_ratio": 0.95,
        "unknown_taxonomy_policy": "reject",
        "unknown_unit_policy": "reject",
        "conflict_policy": "reject",
        "unresolved_tie_policy": "reject",
    }
    config_path = tmp_path / "parse_xbrl_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    run_id = "pytest_parse_xbrl_contract"
    manifest = module.parse_xbrl_run(
        submissions_raw_path=str(submissions_dir),
        companyfacts_raw_path=str(companyfacts_dir),
        config_path=str(config_path),
        run_id=run_id,
        asof="2026-03-15",
    )

    canonical = pd.read_csv(manifest["outputs"]["facts_canonical"])
    rejected = pd.read_csv(manifest["outputs"]["facts_rejected"])
    metrics = pd.read_csv(manifest["outputs"]["facts_metrics"])
    manifest_path = output_dir / f"manifest_{run_id}.json"
    manifest_on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))

    return {
        "module": module,
        "manifest": manifest,
        "manifest_on_disk": manifest_on_disk,
        "manifest_path": manifest_path,
        "canonical": canonical,
        "rejected": rejected,
        "metrics": metrics,
        "config_path": config_path,
        "submissions_dir": submissions_dir,
        "companyfacts_dir": companyfacts_dir,
        "output_dir": output_dir,
        "run_id": run_id,
    }


# -----------------------------------------------------------------------------
# Contract tests
# -----------------------------------------------------------------------------


def test_canonical_output_contains_required_columns(parsed_case):
    canonical = parsed_case["canonical"]
    assert REQUIRED_CANONICAL_COLS.issubset(set(canonical.columns))
    assert not canonical.empty



def test_rejected_output_contains_required_columns_and_reason_context(parsed_case):
    rejected = parsed_case["rejected"]
    assert REQUIRED_REJECTED_COLS.issubset(set(rejected.columns))
    assert not rejected.empty
    assert rejected["rejection_class"].notna().all()
    assert rejected["run_id"].eq(parsed_case["run_id"]).all()



def test_metrics_output_contains_required_columns_and_core_pipeline_rows(parsed_case):
    metrics = parsed_case["metrics"]
    assert REQUIRED_METRICS_COLS.issubset(set(metrics.columns))
    keys = set(zip(metrics["metric_group"], metrics["metric_name"]))
    assert ("pipeline", "candidates_processed") in keys
    assert ("pipeline", "accepted_canonical") in keys
    assert ("pipeline", "rejected_total") in keys



def test_manifest_contains_run_metadata_and_existing_artifacts(parsed_case):
    manifest = parsed_case["manifest"]
    assert manifest["module"] == "data.edgar.parse_xbrl"
    assert manifest["run_id"] == parsed_case["run_id"]
    assert manifest["asof"] == "2026-03-15"
    assert manifest["config_path"] == str(parsed_case["config_path"].resolve())
    assert isinstance(manifest["mapping_version"], str)
    assert len(manifest["mapping_version"]) >= 8
    assert set(manifest["outputs"].keys()) == {"facts_canonical", "facts_rejected", "facts_metrics"}
    for path in manifest["outputs"].values():
        assert Path(path).exists()
    assert parsed_case["manifest_path"].exists()



def test_manifest_json_roundtrip_matches_runtime_manifest(parsed_case):
    manifest = parsed_case["manifest"]
    manifest_on_disk = parsed_case["manifest_on_disk"]
    assert manifest_on_disk == manifest



def test_canonical_key_is_unique_by_economic_key(parsed_case):
    canonical = parsed_case["canonical"]
    dupes = canonical.duplicated(subset=["cik", "metric_name", "period_end", "period_type"], keep=False)
    assert not dupes.any(), canonical.loc[dupes].to_dict(orient="records")



def test_manifest_counts_match_materialized_tables(parsed_case):
    manifest = parsed_case["manifest"]
    canonical = parsed_case["canonical"]
    rejected = parsed_case["rejected"]

    assert manifest["counts"]["accepted"] == len(canonical)
    assert manifest["counts"]["rejected"] == len(rejected)

    rejected_by_class = rejected["rejection_class"].value_counts(dropna=False).to_dict()
    assert manifest["counts"]["rejected_by_class"] == rejected_by_class

    accepted_by_taxonomy = canonical["source_taxonomy"].value_counts(dropna=False).to_dict()
    assert manifest["counts"]["accepted_by_taxonomy"] == accepted_by_taxonomy



def test_metrics_pipeline_counts_match_tables(parsed_case):
    canonical = parsed_case["canonical"]
    rejected = parsed_case["rejected"]
    metrics = parsed_case["metrics"]

    pipeline = metrics.set_index(["metric_group", "metric_name"])["metric_value"].to_dict()
    assert int(pipeline[("pipeline", "accepted_canonical")]) == len(canonical)
    assert int(pipeline[("pipeline", "rejected_total")]) == len(rejected)
    assert int(pipeline[("pipeline", "candidates_processed")]) >= len(canonical)



def test_rejected_classes_include_unmapped_or_duplicate_context(parsed_case):
    rejected = parsed_case["rejected"]
    classes = set(rejected["rejection_class"].dropna().astype(str))
    assert "unmapped_tag" in classes or "duplicate_dominated" in classes



def test_canonical_traceability_fields_are_non_null(parsed_case):
    canonical = parsed_case["canonical"]
    required_non_null = [
        "metric_name",
        "source_taxonomy",
        "source_tag",
        "accession_number",
        "source_record_hash",
        "mapping_version",
        "selection_reason",
        "run_id",
    ]
    for col in required_non_null:
        assert canonical[col].notna().all(), f"Column {col} contains nulls"



def test_same_input_same_config_produces_same_semantic_outputs(tmp_path: Path, parse_xbrl_module):
    module = parse_xbrl_module
    base = tmp_path / "rerun_case"
    submissions_dir = base / "submissions_raw"
    companyfacts_dir = base / "companyfacts_raw"
    out1 = base / "out1"
    out2 = base / "out2"
    for p in [submissions_dir, companyfacts_dir, out1, out2]:
        p.mkdir(parents=True, exist_ok=True)

    cik = "0000320193"
    (submissions_dir / f"CIK{cik}.json").write_text(
        json.dumps(
            {
                "cik": cik,
                "name": "APPLE INC",
                "tickers": ["AAPL"],
                "exchanges": ["NASDAQ"],
                "filings": {
                    "recent": {
                        "accessionNumber": ["0000320193-26-000001"],
                        "form": ["10-K"],
                        "filingDate": ["2026-02-01"],
                        "acceptanceDateTime": ["2026-02-01T14:30:00Z"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (companyfacts_dir / f"CIK{cik}.json").write_text(
        json.dumps(
            {
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
                                        "val": 1000,
                                        "end": "2025-12-31",
                                        "filed": "2026-02-01",
                                        "accepted": "2026-02-01T14:30:00Z",
                                        "fy": 2025,
                                        "fp": "FY",
                                        "form": "10-K",
                                        "accn": "0000320193-26-000001",
                                    }
                                ]
                            },
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def _run_once(output_dir: Path, run_id: str) -> pd.DataFrame:
        cfg = {
            "output_dir": str(output_dir),
            "table_format": "csv",
            "allow_csv_fallback": True,
            "warn_reject_ratio": 0.95,
            "systemic_mapping_failure_ratio": 0.95,
        }
        cfg_path = output_dir.parent / f"cfg_{run_id}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        manifest = module.parse_xbrl_run(
            submissions_raw_path=str(submissions_dir),
            companyfacts_raw_path=str(companyfacts_dir),
            config_path=str(cfg_path),
            run_id=run_id,
            asof="2026-03-15",
        )
        df = pd.read_csv(manifest["outputs"]["facts_canonical"])
        drop_cols = [c for c in ["run_id", "mapping_version"] if c in df.columns]
        return df.drop(columns=drop_cols).sort_values(["cik", "metric_name", "period_end", "period_type"], kind="stable").reset_index(drop=True)

    left = _run_once(out1, "run_left")
    right = _run_once(out2, "run_right")
    pd.testing.assert_frame_equal(left, right, check_dtype=False)
