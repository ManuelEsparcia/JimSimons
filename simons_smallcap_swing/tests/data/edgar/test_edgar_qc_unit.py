from __future__ import annotations

import copy
import importlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# Import adapter
# -----------------------------------------------------------------------------


def _load_edgar_qc_module():
    module_names = [
        "simons_smallcap_swing.data.edgar.edgar_qc",
        "data.edgar.edgar_qc",
        "edgar_qc",
    ]
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue

    here = Path(__file__).resolve()
    candidate_paths = [
        here.parents[3] / "simons_smallcap_swing" / "data" / "edgar" / "edgar_qc.py",
        here.parents[2] / "data" / "edgar" / "edgar_qc.py",
        here.parents[1] / "edgar_qc.py",
        Path("/mnt/data/edgar_qc.py"),
    ]
    for path in candidate_paths:
        if path.exists():
            spec = importlib.util.spec_from_file_location("edgar_qc", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module

    raise ModuleNotFoundError(
        "Could not import edgar_qc.py. Expected it at "
        "simons_smallcap_swing.data.edgar.edgar_qc or in a nearby repo path."
    )


@pytest.fixture(scope="session")
def edgar_qc_module():
    return _load_edgar_qc_module()


# -----------------------------------------------------------------------------
# Helpers / fixtures
# -----------------------------------------------------------------------------


def _ts(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


@pytest.fixture()
def cfg(edgar_qc_module):
    return copy.deepcopy(edgar_qc_module.DEFAULT_CONFIG)


@pytest.fixture()
def minimal_tables():
    mapping = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "cik": "0000000001",
                "valid_from": _ts("2025-01-01"),
                "valid_to": _ts("2026-12-31"),
                "source_priority": 1,
                "source_type": "official_sec_source",
                "sector": "Tech",
                "size_bucket": "mid",
                "liquidity_decile": 5,
                "run_id": "map_run",
            },
            {
                "symbol": "BBB",
                "cik": "0000000002",
                "valid_from": _ts("2025-01-01"),
                "valid_to": _ts("2026-12-31"),
                "source_priority": 1,
                "source_type": "official_sec_source",
                "sector": "Health",
                "size_bucket": "small",
                "liquidity_decile": 3,
                "run_id": "map_run",
            },
        ]
    )

    submissions = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "cik": "0000000001",
                "source_filing_id": "0001-26-000001",
                "acceptance_ts": _ts("2026-02-10 14:00:00"),
                "filing_date": _ts("2026-02-10 00:00:00"),
                "status_code": 200,
                "raw_artifact_id": "sub_raw_1",
                "run_id": "sub_run",
            },
            {
                "symbol": "BBB",
                "cik": "0000000002",
                "source_filing_id": "0002-26-000001",
                "acceptance_ts": _ts("2026-02-11 15:00:00"),
                "filing_date": _ts("2026-02-11 00:00:00"),
                "status_code": 200,
                "raw_artifact_id": "sub_raw_2",
                "run_id": "sub_run",
            },
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
                "period_end": _ts("2025-12-31"),
                "acceptance_ts": _ts("2026-02-10 14:00:00"),
                "source_filing_id": "0001-26-000001",
                "run_id": "cf_run",
            },
            {
                "symbol": "BBB",
                "cik": "0000000002",
                "metric": "Assets",
                "unit": "USD",
                "value": 220.0,
                "period_end": _ts("2025-12-31"),
                "acceptance_ts": _ts("2026-02-11 15:00:00"),
                "source_filing_id": "0002-26-000001",
                "run_id": "cf_run",
            },
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
                "period_end": _ts("2025-12-31"),
                "acceptance_ts": _ts("2026-02-10 14:00:00"),
                "source_filing_id": "0001-26-000001",
                "form_type": "10-K",
                "amendment_flag": False,
                "raw_artifact_id": "cf_raw_1",
                "run_id": "parse_run",
            },
            {
                "symbol": "BBB",
                "cik": "0000000002",
                "metric": "Assets",
                "value": 220.0,
                "unit": "USD",
                "period_end": _ts("2025-12-31"),
                "acceptance_ts": _ts("2026-02-11 15:00:00"),
                "source_filing_id": "0002-26-000001",
                "form_type": "10-K",
                "amendment_flag": False,
                "raw_artifact_id": "cf_raw_2",
                "run_id": "parse_run",
            },
        ]
    )

    pit = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "cik": "0000000001",
                "metric": "Revenue",
                "value": 100.0,
                "unit": "USD",
                "period_end": _ts("2025-12-31"),
                "acceptance_ts": _ts("2026-02-10 14:00:00"),
                "asof": _ts("2026-03-15 23:59:59"),
                "source_filing_id": "0001-26-000001",
                "form_type": "10-K",
                "raw_artifact_id": "cf_raw_1",
                "amendment_flag": False,
                "run_id": "pit_run",
                "sector": "Tech",
                "size_bucket": "mid",
                "liquidity_decile": 5,
            },
            {
                "symbol": "BBB",
                "cik": "0000000002",
                "metric": "Assets",
                "value": 220.0,
                "unit": "USD",
                "period_end": _ts("2025-12-31"),
                "acceptance_ts": _ts("2026-02-11 15:00:00"),
                "asof": _ts("2026-03-15 23:59:59"),
                "source_filing_id": "0002-26-000001",
                "form_type": "10-K",
                "raw_artifact_id": "cf_raw_2",
                "amendment_flag": False,
                "run_id": "pit_run",
                "sector": "Health",
                "size_bucket": "small",
                "liquidity_decile": 3,
            },
        ]
    )

    return {
        "ticker_cik_outputs": mapping,
        "submissions_raw": submissions,
        "companyfacts_raw": companyfacts,
        "parsed_facts": parsed,
        "pit_store": pit,
    }


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_canonicalize_cik_zero_pads_and_strips_noise(edgar_qc_module):
    module = edgar_qc_module

    assert module.canonicalize_cik("  12345  ") == "0000012345"
    assert module.canonicalize_cik("CIK 12345") == "0000012345"
    assert module.canonicalize_cik("0000012345") == "0000012345"
    assert module.canonicalize_cik(None) is None



def test_make_check_rejects_invalid_severity(edgar_qc_module):
    module = edgar_qc_module

    with pytest.raises(module.EdgarQCError, match="Invalid severity"):
        module.make_check(stage="schema", check_name="bad", value=1, severity="fatal")



def test_run_schema_checks_fails_on_missing_required_column(edgar_qc_module, minimal_tables):
    module = edgar_qc_module
    tables = copy.deepcopy(minimal_tables)
    tables["pit_store"] = tables["pit_store"].drop(columns=["metric"])

    checks = []
    failures = []
    schema_ok, metrics = module.run_schema_checks(tables, checks, failures)

    assert schema_ok is False
    assert metrics["rows::pit_store"] == len(tables["pit_store"])
    check_df = pd.DataFrame(checks)
    failure_df = pd.DataFrame(failures)
    row = check_df.loc[check_df["check_name"] == "required_columns::pit_store"].iloc[0]
    assert row["severity"] == "critical"
    assert bool(row["blocking"]) is True
    assert bool(row["passed"]) is False
    assert "metric" in json.loads(row["dimensions"])["missing_columns"]
    assert "missing_required_column" in set(failure_df["issue"])



def test_run_identity_checks_detects_overlapping_active_symbol_cik_conflicts(edgar_qc_module, cfg, minimal_tables):
    module = edgar_qc_module
    tables = copy.deepcopy(minimal_tables)
    conflict_row = tables["ticker_cik_outputs"].iloc[[0]].copy()
    conflict_row["cik"] = "0000009999"
    conflict_row["valid_from"] = _ts("2025-06-01")
    conflict_row["valid_to"] = _ts("2026-06-30")
    tables["ticker_cik_outputs"] = pd.concat([tables["ticker_cik_outputs"], conflict_row], ignore_index=True)

    checks = []
    failures = []
    metrics = module.run_identity_checks(tables, cfg, checks, failures)

    assert metrics["identity_conflict_share"] == pytest.approx(0.5)
    check_df = pd.DataFrame(checks)
    row = check_df.loc[check_df["check_name"] == "active_symbol_cik_conflict_share"].iloc[0]
    assert row["severity"] == "critical"
    assert bool(row["blocking"]) is True
    assert bool(row["passed"]) is False
    failure_df = pd.DataFrame(failures)
    assert "overlapping_active_mapping" in set(failure_df["issue"])
    assert "AAA" in set(failure_df["symbol"])



def test_run_identity_checks_detects_pit_vs_mapping_cik_mismatch(edgar_qc_module, cfg, minimal_tables):
    module = edgar_qc_module
    tables = copy.deepcopy(minimal_tables)
    tables["pit_store"].loc[tables["pit_store"]["symbol"] == "AAA", "cik"] = "0000009999"

    checks = []
    failures = []
    metrics = module.run_identity_checks(tables, cfg, checks, failures)

    assert metrics["pit_mapping_mismatch_share"] == pytest.approx(0.5)
    check_df = pd.DataFrame(checks)
    row = check_df.loc[check_df["check_name"] == "pit_vs_mapping_cik_mismatch_share"].iloc[0]
    assert row["severity"] == "critical"
    assert bool(row["blocking"]) is True
    failure_df = pd.DataFrame(failures)
    mismatch = failure_df.loc[failure_df["issue"] == "pit_mapping_cik_mismatch"].iloc[0]
    assert mismatch["symbol"] == "AAA"
    assert json.loads(mismatch["details"])["expected_cik"] == "0000000001"



def test_run_temporal_checks_flags_acceptance_after_asof_as_blocking(edgar_qc_module, minimal_tables):
    module = edgar_qc_module
    tables = copy.deepcopy(minimal_tables)
    tables["pit_store"].loc[0, "acceptance_ts"] = _ts("2026-03-16 00:00:00")

    checks = []
    failures = []
    metrics = module.run_temporal_checks(tables, checks, failures)

    assert metrics["pit_leakage_count"] == 1
    check_df = pd.DataFrame(checks)
    row = check_df.loc[check_df["check_name"] == "pit_leakage_count"].iloc[0]
    assert row["severity"] == "critical"
    assert bool(row["blocking"]) is True
    assert bool(row["passed"]) is False
    failure_df = pd.DataFrame(failures)
    leak = failure_df.loc[failure_df["issue"] == "acceptance_after_asof"].iloc[0]
    assert leak["symbol"] == "AAA"
    assert leak["metric"] == "Revenue"



def test_run_temporal_checks_flags_duplicate_pit_candidates_as_critical(edgar_qc_module, minimal_tables):
    module = edgar_qc_module
    tables = copy.deepcopy(minimal_tables)
    dup_row = tables["pit_store"].iloc[[0]].copy()
    dup_row["source_filing_id"] = "0001-26-000777"
    dup_row["acceptance_ts"] = _ts("2026-02-12 12:00:00")
    tables["pit_store"] = pd.concat([tables["pit_store"], dup_row], ignore_index=True)

    checks = []
    failures = []
    metrics = module.run_temporal_checks(tables, checks, failures)

    assert metrics["pit_duplicate_candidate_count"] == 2
    check_df = pd.DataFrame(checks)
    row = check_df.loc[check_df["check_name"] == "non_deterministic_candidate_resolution"].iloc[0]
    assert row["severity"] == "critical"
    assert bool(row["blocking"]) is True
    failure_df = pd.DataFrame(failures)
    assert "duplicate_pit_candidate" in set(failure_df["issue"])



def test_run_lineage_checks_critical_metric_missing_lineage_is_blocking(edgar_qc_module, cfg, minimal_tables):
    module = edgar_qc_module
    tables = copy.deepcopy(minimal_tables)
    tables["pit_store"].loc[0, "raw_artifact_id"] = None

    checks = []
    failures = []
    metrics = module.run_lineage_checks(tables, cfg, checks, failures)

    assert metrics["traceability_rate"] == pytest.approx(0.5)
    assert metrics["critical_metric_lineage_missing_share"] == pytest.approx(0.5)
    check_df = pd.DataFrame(checks)
    trace_row = check_df.loc[check_df["check_name"] == "traceability_rate"].iloc[0]
    crit_row = check_df.loc[check_df["check_name"] == "critical_metric_lineage_missing_share"].iloc[0]
    assert trace_row["severity"] == "critical"
    assert bool(trace_row["blocking"]) is True
    assert crit_row["severity"] == "critical"
    assert bool(crit_row["blocking"]) is True
    failure_df = pd.DataFrame(failures)
    miss = failure_df.loc[failure_df["issue"] == "missing_lineage"].iloc[0]
    assert miss["symbol"] == "AAA"
    assert miss["metric"] == "revenue"
    assert "raw_artifact_id" in json.loads(miss["details"])["missing_columns"]



def test_staleness_thresholds_are_metric_specific(edgar_qc_module, cfg):
    module = edgar_qc_module

    revenue_warn, revenue_fail = module.staleness_thresholds_for_metric("Revenue", cfg)
    shares_warn, shares_fail = module.staleness_thresholds_for_metric("shares_outstanding", cfg)
    other_warn, other_fail = module.staleness_thresholds_for_metric("OperatingCashFlow", cfg)

    assert (revenue_warn, revenue_fail) == (120, 220)
    assert (shares_warn, shares_fail) == (75, 120)
    assert (other_warn, other_fail) == (130, 240)



def test_decide_gate_is_non_compensatory_when_blocking_check_fails(edgar_qc_module):
    module = edgar_qc_module
    checks_df = pd.DataFrame(
        [
            module.make_check(
                stage="temporal",
                check_name="pit_leakage_count",
                value=1,
                severity="critical",
                blocking=True,
                passed=False,
                message="acceptance after asof",
            ),
            module.make_check(
                stage="coverage",
                check_name="pit_coverage",
                value=0.99,
                severity="info",
                blocking=False,
                passed=True,
            ),
        ]
    )

    gate, reasons = module.decide_gate(checks_df, score=99.9, cfg=module.DEFAULT_CONFIG)

    assert gate == "fail"
    assert any(r.startswith("blocking::pit_leakage_count") for r in reasons)



def test_aggregate_score_is_stable_for_same_metrics(edgar_qc_module):
    module = edgar_qc_module
    metrics = {
        "coverage::submissions": 0.95,
        "coverage::facts": 0.90,
        "coverage::pit": 0.92,
        "identity_conflict_share": 0.0,
        "pit_mapping_mismatch_share": 0.0,
        "multi_ticker_per_cik_share": 0.0,
        "parse_success_ratio": 0.97,
        "submission_download_failure_rate": 0.0,
        "robust_outlier_share": 0.0,
        "sign_rule_violation_share": 0.0,
        "accounting_residual_share": 0.0,
        "staleness_warn_share": 0.01,
        "critical_metric_staleness_fail_share": 0.0,
        "traceability_rate": 0.99,
        "critical_metric_lineage_missing_share": 0.0,
    }

    score1, losses1 = module.aggregate_score(metrics, module.DEFAULT_CONFIG)
    score2, losses2 = module.aggregate_score(metrics, module.DEFAULT_CONFIG)

    assert score1 == pytest.approx(score2)
    assert losses1 == losses2
    assert 0.0 <= score1 <= 100.0
