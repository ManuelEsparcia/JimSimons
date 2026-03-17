from __future__ import annotations

import copy
import importlib
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


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
            spec = importlib.util.spec_from_file_location("filings_flags", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module

    raise ModuleNotFoundError(
        "Could not import filings_flags.py. Expected it at "
        "simons_smallcap_swing.data.edgar.filings_flags or in a nearby repo path."
    )


@pytest.fixture(scope="session")
def filings_flags_module():
    return _load_filings_flags_module()


@pytest.fixture()
def cfg(filings_flags_module):
    return copy.deepcopy(filings_flags_module.DEFAULT_CONFIG)


@pytest.fixture()
def reverse_form_map(filings_flags_module, cfg):
    return filings_flags_module._reverse_form_family_map(cfg)


# -----------------------------------------------------------------------------
# Tests: form family / deadline helpers
# -----------------------------------------------------------------------------


def test_classify_form_family_and_infer_deadline_date(filings_flags_module, cfg, reverse_form_map):
    module = filings_flags_module

    assert module._classify_form_family("10-Q/A", reverse_form_map) == "10-Q"
    assert module._classify_form_family("8-K", reverse_form_map) == "8-K"
    assert module._classify_form_family("S-1/A", reverse_form_map) == "S-1"

    row = pd.Series(
        {
            "form_family": "10-Q",
            "period_end": pd.Timestamp("2025-03-31"),
            "filer_status": "large_accelerated_filer",
        }
    )
    deadline, policy = module._infer_deadline_date(row, cfg)
    assert deadline == pd.Timestamp("2025-05-10")
    assert policy == "matched"

    fallback_row = pd.Series(
        {
            "form_family": "10-Q",
            "period_end": pd.Timestamp("2025-03-31"),
            "filer_status": "something_unmapped",
        }
    )
    fallback_deadline, fallback_policy = module._infer_deadline_date(fallback_row, cfg)
    assert fallback_deadline == pd.Timestamp("2025-05-15")
    assert fallback_policy == "fallback_unknown"


# -----------------------------------------------------------------------------
# Tests: linkage
# -----------------------------------------------------------------------------


def test_link_facts_to_filings_prefers_exact_accession_then_fallback_period(filings_flags_module):
    module = filings_flags_module
    failures = []

    submissions = pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "accession_number": "0001-25-000001",
                "form_family": "10-Q",
                "filing_date": pd.Timestamp("2025-05-01", tz="UTC"),
                "period_end": pd.Timestamp("2025-03-31"),
            },
            {
                "cik": "0000000001",
                "accession_number": "0001-25-000002",
                "form_family": "10-Q",
                "filing_date": pd.Timestamp("2025-05-20", tz="UTC"),
                "period_end": pd.Timestamp("2025-03-31"),
            },
        ]
    )
    facts = pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "accession_number": "0001-25-000001",
                "period_end": pd.Timestamp("2025-03-31"),
                "concept": "Revenue",
                "value": 100.0,
                "observed_date": pd.Timestamp("2025-05-02", tz="UTC"),
                "acceptance_datetime": pd.Timestamp("2025-05-02T12:00:00Z"),
                "form_type": "10-Q",
            },
            {
                "cik": "0000000001",
                "accession_number": "",
                "period_end": pd.Timestamp("2025-03-31"),
                "concept": "Assets",
                "value": 250.0,
                "observed_date": pd.Timestamp("2025-05-25", tz="UTC"),
                "acceptance_datetime": pd.Timestamp("2025-05-25T12:00:00Z"),
                "form_type": "10-Q",
            },
        ]
    )

    linked = module._link_facts_to_filings(submissions, facts, failures)
    linked = linked.sort_values(["concept", "value"]).reset_index(drop=True)

    exact_row = linked.loc[linked["concept"] == "Revenue"].iloc[0]
    assert exact_row["link_method"] == "exact_accession"
    assert exact_row["linked_accession_number"] == "0001-25-000001"

    fallback_row = linked.loc[linked["concept"] == "Assets"].iloc[0]
    assert fallback_row["link_method"] == "fallback_period"
    assert fallback_row["linked_accession_number"] == "0001-25-000002"
    assert failures == []


# -----------------------------------------------------------------------------
# Tests: timeliness / amendments / completeness / structure
# -----------------------------------------------------------------------------


def test_compute_timeliness_flags_marks_late_and_persistent(filings_flags_module, cfg):
    module = filings_flags_module

    filings = pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "accession_number": "a1",
                "form_family": "10-Q",
                "filer_status": "large_accelerated_filer",
                "period_end": pd.Timestamp("2025-03-31"),
                "filing_date": pd.Timestamp("2025-05-10"),
                "acceptance_datetime": pd.Timestamp("2025-05-10T10:00:00Z"),
            },
            {
                "cik": "0000000001",
                "accession_number": "a2",
                "form_family": "10-Q",
                "filer_status": "large_accelerated_filer",
                "period_end": pd.Timestamp("2025-06-30"),
                "filing_date": pd.Timestamp("2025-08-15"),
                "acceptance_datetime": pd.Timestamp("2025-08-15T10:00:00Z"),
            },
            {
                "cik": "0000000001",
                "accession_number": "a3",
                "form_family": "10-Q",
                "filer_status": "large_accelerated_filer",
                "period_end": pd.Timestamp("2025-09-30"),
                "filing_date": pd.Timestamp("2025-11-12"),
                "acceptance_datetime": pd.Timestamp("2025-11-12T10:00:00Z"),
            },
            {
                "cik": "0000000001",
                "accession_number": "a4",
                "form_family": "10-Q",
                "filer_status": "large_accelerated_filer",
                "period_end": pd.Timestamp("2025-12-31"),
                "filing_date": pd.Timestamp("2026-02-09"),
                "acceptance_datetime": pd.Timestamp("2026-02-09T10:00:00Z"),
            },
        ]
    )

    out = module._compute_timeliness_flags(filings, cfg).sort_values("accession_number").reset_index(drop=True)
    assert out["flag_late_filer"].tolist() == [0, 1, 1, 0]
    assert out.loc[out["accession_number"] == "a2", "days_late"].iloc[0] == 6
    assert out.loc[out["accession_number"] == "a3", "flag_persistent_late_filer"].iloc[0] == 1
    assert out.loc[out["accession_number"] == "a4", "flag_persistent_late_filer"].iloc[0] == 1



def test_compute_amendment_flags_distinguishes_substantive_and_technical(filings_flags_module, cfg):
    module = filings_flags_module
    df = pd.DataFrame(
        [
            {
                "form_type": "10-K/A",
                "form_family": "10-K",
                "document_text": "This amendment includes a restatement of previously issued financial statements.",
            },
            {
                "form_type": "8-K/A",
                "form_family": "8-K",
                "document_text": "Updated exhibit references and cover page metadata only.",
            },
            {
                "form_type": "10-Q/A",
                "form_family": "10-Q",
                "document_text": "Corrected and amended filing after revision to note disclosure.",
            },
        ]
    )

    out = module._compute_amendment_flags(df, cfg)
    assert out["flag_amended"].tolist() == [1, 1, 1]
    assert out.loc[0, "amendment_class"] == "substantive"
    assert out.loc[1, "amendment_class"] == "technical"
    assert out.loc[2, "evidence_suspected_term"] == 1
    assert out.loc[2, "evidence_amended_sensitive_form"] == 1



def test_compute_completeness_flags_marks_missing_core_and_sparse_coverage(filings_flags_module, cfg):
    module = filings_flags_module
    filings = pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "accession_number": "a1",
                "form_family": "10-Q",
            }
        ]
    )
    linked_facts = pd.DataFrame(
        [
            {"cik": "0000000001", "linked_accession_number": "a1", "concept": "Revenue"},
            {"cik": "0000000001", "linked_accession_number": "a1", "concept": "Assets"},
        ]
    )

    out = module._compute_completeness_flags(filings, linked_facts, cfg)
    row = out.iloc[0]
    assert row["num_missing_core_fields"] == 3
    assert row["flag_missing_core_fields"] == 1
    assert row["flag_sparse_fundamental_coverage"] == 1
    assert row["core_coverage_ratio"] == pytest.approx(2 / 5)
    assert set(str(row["missing_core_fields_list"]).split("|")) == {"NetIncomeLoss", "Liabilities", "StockholdersEquity"}



def test_compute_reporting_structure_flags_detects_gap_and_irregular_sequence(filings_flags_module, cfg):
    module = filings_flags_module
    df = pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "accession_number": "q1",
                "form_family": "10-Q",
                "period_end": pd.Timestamp("2025-03-31"),
                "filing_date": pd.Timestamp("2025-05-10"),
            },
            {
                "cik": "0000000001",
                "accession_number": "q2",
                "form_family": "10-Q",
                "period_end": pd.Timestamp("2025-09-30"),
                "filing_date": pd.Timestamp("2025-11-10"),
            },
            {
                "cik": "0000000001",
                "accession_number": "q3",
                "form_family": "10-Q",
                "period_end": pd.Timestamp("2025-10-15"),
                "filing_date": pd.Timestamp("2025-11-25"),
            },
        ]
    )

    out = module._compute_reporting_structure_flags(df, cfg).sort_values("accession_number").reset_index(drop=True)
    assert out.loc[out["accession_number"] == "q2", "flag_reporting_gap"].iloc[0] == 1
    assert out.loc[out["accession_number"] == "q3", "flag_irregular_filing_sequence"].iloc[0] == 1


# -----------------------------------------------------------------------------
# Tests: anomalies / restatements / aggregation
# -----------------------------------------------------------------------------


def test_compute_period_anomalies_flags_revenue_outlier_after_history(filings_flags_module, cfg):
    module = filings_flags_module
    period_matrix = pd.DataFrame(
        [
            {"cik": "0000000001", "period_end": pd.Timestamp("2024-03-31"), "Revenue": 100.0},
            {"cik": "0000000001", "period_end": pd.Timestamp("2024-06-30"), "Revenue": 102.0},
            {"cik": "0000000001", "period_end": pd.Timestamp("2024-09-30"), "Revenue": 98.0},
            {"cik": "0000000001", "period_end": pd.Timestamp("2024-12-31"), "Revenue": 101.0},
            {"cik": "0000000001", "period_end": pd.Timestamp("2025-03-31"), "Revenue": 500.0},
        ]
    )

    out = module._compute_period_anomalies(period_matrix, cfg).sort_values("period_end").reset_index(drop=True)
    last = out.iloc[-1]
    assert last["flag_accounting_anomaly"] == 1
    assert "Revenue" in str(last["anomalous_concepts"])
    assert float(last["anomaly_magnitude"]) > 6.0



def test_compute_period_anomalies_flags_balance_sheet_inconsistency(filings_flags_module, cfg):
    module = filings_flags_module
    period_matrix = pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "period_end": pd.Timestamp("2025-03-31"),
                "Assets": 100.0,
                "Liabilities": 30.0,
                "StockholdersEquity": 50.0,
            }
        ]
    )

    out = module._compute_period_anomalies(period_matrix, cfg)
    row = out.iloc[0]
    assert row["flag_balance_sheet_inconsistency"] == 1
    assert row["flag_accounting_anomaly"] == 1
    assert row["balance_sheet_residual_ratio"] > cfg["anomaly"]["bs_epsilon"]



def test_compute_restatement_flags_sets_hard_and_suspected_correctly(filings_flags_module, cfg):
    module = filings_flags_module
    df = pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "accession_number": "a1",
                "flag_accounting_anomaly": 0,
                "flag_balance_sheet_inconsistency": 0,
                "evidence_explicit_hard_term": 0,
                "evidence_suspected_term": 0,
                "evidence_amended_sensitive_form": 1,
            },
            {
                "cik": "0000000002",
                "accession_number": "b1",
                "flag_accounting_anomaly": 1,
                "flag_balance_sheet_inconsistency": 1,
                "evidence_explicit_hard_term": 0,
                "evidence_suspected_term": 1,
                "evidence_amended_sensitive_form": 0,
            },
        ]
    )
    linked_facts = pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "period_end": pd.Timestamp("2025-03-31"),
                "concept": "Revenue",
                "value": 100.0,
                "observed_date": pd.Timestamp("2025-05-01", tz="UTC"),
                "linked_accession_number": "old",
            },
            {
                "cik": "0000000001",
                "period_end": pd.Timestamp("2025-03-31"),
                "concept": "Revenue",
                "value": 130.0,
                "observed_date": pd.Timestamp("2025-05-20", tz="UTC"),
                "linked_accession_number": "a1",
            },
        ]
    )

    out = module._compute_restatement_flags(df, linked_facts, cfg).sort_values("accession_number").reset_index(drop=True)

    hard_row = out.loc[out["accession_number"] == "a1"].iloc[0]
    assert hard_row["flag_restatement_hard"] == 1
    assert hard_row["flag_restatement_suspected"] == 0
    assert "material_fact_change" in str(hard_row["restatement_evidence_type"])

    suspected_row = out.loc[out["accession_number"] == "b1"].iloc[0]
    assert suspected_row["flag_restatement_hard"] == 0
    assert suspected_row["flag_restatement_suspected"] == 1
    assert "suspected_term" in str(suspected_row["restatement_evidence_type"])



def test_aggregate_severity_and_action_excludes_critical_and_penalizes_warn(filings_flags_module, cfg):
    module = filings_flags_module
    df = pd.DataFrame(
        [
            {
                "accession_number": "clean",
                "flag_late_filer": 0,
                "flag_persistent_late_filer": 0,
                "flag_amended": 0,
                "flag_restatement_hard": 0,
                "flag_restatement_suspected": 0,
                "flag_missing_core_fields": 0,
                "flag_sparse_fundamental_coverage": 0,
                "flag_accounting_anomaly": 0,
                "flag_balance_sheet_inconsistency": 0,
                "flag_reporting_gap": 0,
                "flag_irregular_filing_sequence": 0,
            },
            {
                "accession_number": "warn",
                "flag_late_filer": 0,
                "flag_persistent_late_filer": 0,
                "flag_amended": 0,
                "flag_restatement_hard": 0,
                "flag_restatement_suspected": 0,
                "flag_missing_core_fields": 1,
                "flag_sparse_fundamental_coverage": 0,
                "flag_accounting_anomaly": 0,
                "flag_balance_sheet_inconsistency": 0,
                "flag_reporting_gap": 0,
                "flag_irregular_filing_sequence": 0,
            },
            {
                "accession_number": "critical",
                "flag_late_filer": 0,
                "flag_persistent_late_filer": 0,
                "flag_amended": 0,
                "flag_restatement_hard": 1,
                "flag_restatement_suspected": 0,
                "flag_missing_core_fields": 0,
                "flag_sparse_fundamental_coverage": 0,
                "flag_accounting_anomaly": 0,
                "flag_balance_sheet_inconsistency": 0,
                "flag_reporting_gap": 0,
                "flag_irregular_filing_sequence": 0,
            },
        ]
    )

    out = module._aggregate_severity_and_action(df, cfg).sort_values("accession_number").reset_index(drop=True)

    clean = out.loc[out["accession_number"] == "clean"].iloc[0]
    warn = out.loc[out["accession_number"] == "warn"].iloc[0]
    critical = out.loc[out["accession_number"] == "critical"].iloc[0]

    assert clean["severity"] == "info"
    assert clean["pit_action"] == "allow"
    assert 0.99 <= float(clean["quality_score"]) <= 1.0

    assert warn["severity"] == "warn"
    assert warn["pit_action"] == "penalize"
    assert float(warn["quality_score"]) < float(clean["quality_score"])

    assert critical["severity"] == "critical"
    assert critical["pit_action"] == "exclude"
    assert critical["dominant_quality_reason"] == "flag_restatement_hard"
    assert float(critical["quality_score"]) < float(warn["quality_score"])
