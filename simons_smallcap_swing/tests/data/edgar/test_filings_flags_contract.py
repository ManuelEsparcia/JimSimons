from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


REQUIRED_FILINGS_COLS = {
    "cik",
    "accession_number",
    "form_type",
    "filing_date",
    "acceptance_datetime",
    "period_end",
    "form_family",
    "deadline_date",
    "deadline_policy_used",
    "days_late",
    "flag_late_filer",
    "flag_persistent_late_filer",
    "flag_amended",
    "amendment_class",
    "flag_restatement_hard",
    "flag_restatement_suspected",
    "flag_missing_core_fields",
    "core_coverage_ratio",
    "flag_sparse_fundamental_coverage",
    "flag_accounting_anomaly",
    "flag_balance_sheet_inconsistency",
    "flag_reporting_gap",
    "flag_irregular_filing_sequence",
    "quality_score",
    "severity",
    "pit_action",
    "dominant_quality_reason",
    "run_id",
    "config_version",
}

REQUIRED_FAILURES_COLS = {"stage", "severity", "code", "message", "row_count"}
REQUIRED_METRICS_COLS = {"group", "metric", "value"}
REQUIRED_SUMMARY_KEYS = {
    "run_id",
    "asof",
    "generated_at_utc",
    "config_version",
    "n_filings",
    "n_unique_cik",
    "quality_score_mean",
    "quality_score_median",
    "pit_action_distribution",
    "severity_distribution",
    "failure_count",
    "failures_by_severity",
}
REQUIRED_MANIFEST_KEYS = {
    "module",
    "run_id",
    "generated_at_utc",
    "asof",
    "inputs",
    "config_version",
    "row_counts",
    "linkage_methods",
    "files",
    "summary",
}
REQUIRED_ARTIFACT_KEYS = {"filings_flags", "filings_flags_metrics", "filings_flags_summary", "manifest"}


# -----------------------------------------------------------------------------
# Import adapter
# -----------------------------------------------------------------------------


def _load_filings_flags_module():
    module_names = [
        "simons_smallcap_swing.data.edgar.filings_flags",
        "data.edgar.filings_flags",
        "filings_flags",
    ]
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue

    here = Path(__file__).resolve()
    candidate_paths = [
        here.parents[3] / "simons_smallcap_swing" / "data" / "edgar" / "filings_flags.py",
        here.parents[2] / "data" / "edgar" / "filings_flags.py",
        here.parents[1] / "filings_flags.py",
        Path("/mnt/data/filings_flags.py"),
    ]
    for path in candidate_paths:
        if path.exists():
            spec = importlib.util.spec_from_file_location("filings_flags_contract_under_test", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module

    raise ModuleNotFoundError(
        "Could not import filings_flags.py. Expected it at "
        "simons_smallcap_swing.data.edgar.filings_flags or in a nearby repo path."
    )


@pytest.fixture(scope="session")
def filings_flags_module():
    return _load_filings_flags_module()


# -----------------------------------------------------------------------------
# Synthetic end-to-end fixture
# -----------------------------------------------------------------------------


@pytest.fixture()
def filings_flags_case(tmp_path: Path, filings_flags_module):
    module = filings_flags_module

    submissions = pd.DataFrame(
        [
            {
                "cik": "1",
                "accession_number": "0001-25-000001",
                "form_type": "10-Q",
                "filing_date": "2025-05-08",
                "acceptance_datetime": "2025-05-08T12:00:00Z",
                "period_end": "2025-03-31",
                "filer_status": "large_accelerated_filer",
                "document_text": "Quarterly report for Q1.",
                "ticker": "ABC",
                "sector": "Technology",
                "market_cap": 1_500_000_000,
                "liquidity_bucket": "mid",
            },
            {
                "cik": "1",
                "accession_number": "0001-25-000002",
                "form_type": "10-Q/A",
                "filing_date": "2025-05-20",
                "acceptance_datetime": "2025-05-20T12:00:00Z",
                "period_end": "2025-03-31",
                "filer_status": "large_accelerated_filer",
                "document_text": "Amendment including restatement of previously issued financial statements.",
                "ticker": "ABC",
                "sector": "Technology",
                "market_cap": 1_500_000_000,
                "liquidity_bucket": "mid",
            },
        ]
    )

    facts = pd.DataFrame(
        [
            # Original filing: complete and coherent.
            {"cik": "1", "accession_number": "0001-25-000001", "period_end": "2025-03-31", "concept": "Revenue", "value": 100.0, "observed_date": "2025-05-08T12:00:00Z", "acceptance_datetime": "2025-05-08T12:00:00Z", "form_type": "10-Q"},
            {"cik": "1", "accession_number": "0001-25-000001", "period_end": "2025-03-31", "concept": "NetIncomeLoss", "value": 10.0, "observed_date": "2025-05-08T12:00:00Z", "acceptance_datetime": "2025-05-08T12:00:00Z", "form_type": "10-Q"},
            {"cik": "1", "accession_number": "0001-25-000001", "period_end": "2025-03-31", "concept": "Assets", "value": 200.0, "observed_date": "2025-05-08T12:00:00Z", "acceptance_datetime": "2025-05-08T12:00:00Z", "form_type": "10-Q"},
            {"cik": "1", "accession_number": "0001-25-000001", "period_end": "2025-03-31", "concept": "Liabilities", "value": 120.0, "observed_date": "2025-05-08T12:00:00Z", "acceptance_datetime": "2025-05-08T12:00:00Z", "form_type": "10-Q"},
            {"cik": "1", "accession_number": "0001-25-000001", "period_end": "2025-03-31", "concept": "StockholdersEquity", "value": 80.0, "observed_date": "2025-05-08T12:00:00Z", "acceptance_datetime": "2025-05-08T12:00:00Z", "form_type": "10-Q"},
            # Amendment: material changes and balance-sheet inconsistency -> should become severe/exclude.
            {"cik": "1", "accession_number": "0001-25-000002", "period_end": "2025-03-31", "concept": "Revenue", "value": 90.0, "observed_date": "2025-05-20T12:00:00Z", "acceptance_datetime": "2025-05-20T12:00:00Z", "form_type": "10-Q/A"},
            {"cik": "1", "accession_number": "0001-25-000002", "period_end": "2025-03-31", "concept": "NetIncomeLoss", "value": 8.0, "observed_date": "2025-05-20T12:00:00Z", "acceptance_datetime": "2025-05-20T12:00:00Z", "form_type": "10-Q/A"},
            {"cik": "1", "accession_number": "0001-25-000002", "period_end": "2025-03-31", "concept": "Assets", "value": 210.0, "observed_date": "2025-05-20T12:00:00Z", "acceptance_datetime": "2025-05-20T12:00:00Z", "form_type": "10-Q/A"},
            {"cik": "1", "accession_number": "0001-25-000002", "period_end": "2025-03-31", "concept": "Liabilities", "value": 150.0, "observed_date": "2025-05-20T12:00:00Z", "acceptance_datetime": "2025-05-20T12:00:00Z", "form_type": "10-Q/A"},
            {"cik": "1", "accession_number": "0001-25-000002", "period_end": "2025-03-31", "concept": "StockholdersEquity", "value": 30.0, "observed_date": "2025-05-20T12:00:00Z", "acceptance_datetime": "2025-05-20T12:00:00Z", "form_type": "10-Q/A"},
            # One unlinked fact to force a non-empty failures artifact.
            {"cik": "1", "accession_number": "", "period_end": "2025-06-30", "concept": "Revenue", "value": 999.0, "observed_date": "2025-05-25T12:00:00Z", "acceptance_datetime": "2025-05-25T12:00:00Z", "form_type": "10-Q"},
        ]
    )

    submissions_path = tmp_path / "submissions.csv"
    facts_path = tmp_path / "facts.csv"
    output_dir = tmp_path / "flags_out"
    submissions.to_csv(submissions_path, index=False)
    facts.to_csv(facts_path, index=False)

    config = {
        "output_dir": str(output_dir),
        "strict_parquet": False,
        "persistence": {"prefer_parquet": False},
    }
    config_path = tmp_path / "filings_flags_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    run_id = "pytest_filings_flags_contract"
    result = module.run_filings_flags(
        submissions_path=submissions_path,
        facts_path=facts_path,
        config_path=config_path,
        run_id=run_id,
        asof="2025-06-01T00:00:00Z",
    )

    filings = result["filings_flags"].copy()
    metrics = result["metrics"].copy()
    failures = result["failures"].copy()
    summary = result["summary"].copy()
    manifest = result["manifest"].copy()

    filings_disk = pd.read_csv(result["files"]["filings_flags"])
    metrics_disk = pd.read_csv(result["files"]["filings_flags_metrics"])
    summary_disk = json.loads(Path(result["files"]["filings_flags_summary"]).read_text(encoding="utf-8"))
    manifest_path = output_dir / f"manifest_{run_id}.json"
    manifest_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures_disk = None
    if "filings_flags_failures" in result["files"]:
        failures_disk = pd.read_csv(result["files"]["filings_flags_failures"])

    return {
        "module": module,
        "result": result,
        "filings": filings,
        "metrics": metrics,
        "failures": failures,
        "summary": summary,
        "manifest": manifest,
        "filings_disk": filings_disk,
        "metrics_disk": metrics_disk,
        "failures_disk": failures_disk,
        "summary_disk": summary_disk,
        "manifest_disk": manifest_disk,
        "manifest_path": manifest_path,
        "output_dir": output_dir,
    }


# -----------------------------------------------------------------------------
# Contract tests
# -----------------------------------------------------------------------------


def test_main_flags_table_contains_required_columns(filings_flags_case):
    filings = filings_flags_case["filings"]
    assert REQUIRED_FILINGS_COLS.issubset(set(filings.columns))
    assert len(filings) == 2


def test_summary_contains_required_keys_and_valid_domains(filings_flags_case):
    summary = filings_flags_case["summary"]
    assert REQUIRED_SUMMARY_KEYS.issubset(set(summary.keys()))
    assert summary["n_filings"] == 2
    assert set(summary["pit_action_distribution"].keys()).issubset({"allow", "penalize", "exclude"})
    assert set(summary["severity_distribution"].keys()).issubset({"info", "warn", "critical"})
    assert 0.0 <= float(summary["quality_score_mean"]) <= 1.0
    assert 0.0 <= float(summary["quality_score_median"]) <= 1.0


def test_metrics_contains_required_columns_and_expected_aggregates(filings_flags_case):
    metrics = filings_flags_case["metrics"]
    assert REQUIRED_METRICS_COLS.issubset(set(metrics.columns))
    all_metrics = set(metrics.loc[metrics["group"] == "all", "metric"])
    assert "n_filings" in all_metrics
    assert "quality_score_mean" in all_metrics
    assert "quality_score_median" in all_metrics
    assert "rate::pit_action::allow" in all_metrics
    assert "rate::pit_action::exclude" in all_metrics
    assert any(str(g).startswith("form_family::") for g in metrics["group"].astype(str).unique())


def test_failures_artifact_is_present_and_well_formed_when_linkage_warns(filings_flags_case):
    failures = filings_flags_case["failures"]
    failures_disk = filings_flags_case["failures_disk"]
    assert not failures.empty
    assert REQUIRED_FAILURES_COLS.issubset(set(failures.columns))
    assert failures_disk is not None
    assert REQUIRED_FAILURES_COLS.issubset(set(failures_disk.columns))
    assert "UNLINKED_FACTS" in set(failures["code"].astype(str))
    assert set(failures["severity"].astype(str)).issubset({"warn", "critical"})


def test_manifest_contains_required_metadata_and_artifact_paths(filings_flags_case):
    manifest = filings_flags_case["manifest"]
    assert REQUIRED_MANIFEST_KEYS.issubset(set(manifest.keys()))
    assert manifest["module"] == "data.edgar.filings_flags"
    assert manifest["run_id"] == "pytest_filings_flags_contract"
    assert manifest["inputs"]["config_path"] is not None
    assert REQUIRED_ARTIFACT_KEYS.issubset(set(manifest["files"].keys()))
    for _, path in manifest["files"].items():
        assert Path(path).exists()


def test_manifest_row_counts_match_materialized_outputs(filings_flags_case):
    manifest = filings_flags_case["manifest"]
    filings = filings_flags_case["filings"]
    metrics = filings_flags_case["metrics"]
    failures = filings_flags_case["failures"]

    assert manifest["row_counts"]["filings_flags"] == len(filings)
    assert manifest["row_counts"]["metrics"] == len(metrics)
    assert manifest["row_counts"]["failures"] == len(failures)
    assert manifest["row_counts"]["submissions_after_pit"] == 2
    assert manifest["row_counts"]["facts_after_pit"] == 11


def test_manifest_on_disk_roundtrip_and_known_manifest_artifact_detail(filings_flags_case):
    manifest = filings_flags_case["manifest"]
    manifest_disk = filings_flags_case["manifest_disk"]
    # Known implementation detail: the in-memory manifest includes files["manifest"]
    # after writing the JSON, but the on-disk manifest does not contain that self-reference.
    disk_files = dict(manifest_disk["files"])
    mem_files = dict(manifest["files"])
    assert "manifest" in mem_files
    assert "manifest" not in disk_files
    comparable_mem = dict(manifest)
    comparable_mem["files"] = {k: v for k, v in mem_files.items() if k != "manifest"}
    assert comparable_mem == manifest_disk


def test_severity_and_pit_action_domains_are_valid_and_critical_rows_exclude(filings_flags_case):
    filings = filings_flags_case["filings"].copy()
    assert set(filings["severity"].astype(str)).issubset({"info", "warn", "critical"})
    assert set(filings["pit_action"].astype(str)).issubset({"allow", "penalize", "exclude"})

    critical_rows = filings[filings["severity"] == "critical"]
    assert not critical_rows.empty
    assert set(critical_rows["pit_action"].astype(str)) == {"exclude"}


def test_filings_table_is_sorted_deterministically_by_core_temporal_key(filings_flags_case):
    filings = filings_flags_case["filings"]
    sorted_filings = filings.sort_values(
        ["cik", "filing_date", "acceptance_datetime", "accession_number"],
        kind="mergesort",
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(filings.reset_index(drop=True), sorted_filings)


def test_summary_and_metrics_are_consistent_with_filings_table(filings_flags_case):
    filings = filings_flags_case["filings"]
    summary = filings_flags_case["summary"]
    metrics = filings_flags_case["metrics"]

    assert summary["pit_action_distribution"] == filings["pit_action"].value_counts(dropna=False).to_dict()
    assert summary["severity_distribution"] == filings["severity"].value_counts(dropna=False).to_dict()

    metrics_map = {
        (str(r["group"]), str(r["metric"])): float(r["value"])
        for _, r in metrics.iterrows()
    }
    assert metrics_map[("all", "n_filings")] == float(len(filings))
    assert metrics_map[("all", "rate::pit_action::allow")] == pytest.approx(float((filings["pit_action"] == "allow").mean()))
    assert metrics_map[("all", "rate::pit_action::exclude")] == pytest.approx(float((filings["pit_action"] == "exclude").mean()))
    assert metrics_map[("all", "quality_score_mean")] == pytest.approx(float(filings["quality_score"].mean()))


def test_roundtrip_disk_outputs_match_in_memory_results(filings_flags_case):
    filings_mem = filings_flags_case["filings"].copy().reset_index(drop=True)
    filings_disk = filings_flags_case["filings_disk"].copy().reset_index(drop=True)

    # CSV roundtrip can coerce identity and datetime-ish columns; compare semantically.
    filings_disk["cik"] = filings_disk["cik"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(10)
    for col in ["filing_date", "acceptance_datetime", "period_end", "deadline_date"]:
        filings_mem[col] = pd.to_datetime(filings_mem[col], errors="coerce", utc=True)
        filings_disk[col] = pd.to_datetime(filings_disk[col], errors="coerce", utc=True)

    comparable_cols = [
        "cik",
        "accession_number",
        "form_type",
        "filing_date",
        "acceptance_datetime",
        "period_end",
        "deadline_date",
        "flag_late_filer",
        "flag_amended",
        "flag_restatement_hard",
        "flag_balance_sheet_inconsistency",
        "quality_score",
        "severity",
        "pit_action",
        "run_id",
        "config_version",
    ]
    pd.testing.assert_frame_equal(
        filings_mem[comparable_cols],
        filings_disk[comparable_cols],
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        filings_flags_case["metrics"].reset_index(drop=True),
        filings_flags_case["metrics_disk"].reset_index(drop=True),
        check_dtype=False,
    )
    assert filings_flags_case["summary"] == filings_flags_case["summary_disk"]
