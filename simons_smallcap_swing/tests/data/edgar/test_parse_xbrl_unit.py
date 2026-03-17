from __future__ import annotations

import importlib
import importlib.util
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
# Helpers
# -----------------------------------------------------------------------------


def _cfg(module, **overrides: Any):
    cfg = module.ParseConfig(**overrides)
    cfg.validate()
    return cfg


@pytest.fixture()
def parse_cfg(parse_xbrl_module):
    return _cfg(parse_xbrl_module)


@pytest.fixture()
def submissions_lookup():
    return pd.DataFrame(
        [
            {
                "cik": "0000320193",
                "accession_number": "0000320193-26-000001",
                "symbol": "AAPL",
                "issuer_name": "APPLE INC",
                "form_type": "10-K",
                "filed_date": pd.Timestamp("2026-02-01"),
                "acceptance_datetime": pd.Timestamp("2026-02-01T14:30:00Z"),
                "exchange": "NASDAQ",
            }
        ]
    )


# -----------------------------------------------------------------------------
# Tests: pure semantic helpers
# -----------------------------------------------------------------------------


def test_resolve_mapping_exact_and_unmapped(parse_xbrl_module, parse_cfg):
    module = parse_xbrl_module
    exact = module.resolve_mapping(parse_cfg, "us-gaap", "Assets")
    assert exact.metric_name == "total_assets"
    assert exact.mapping_status == "exact"
    assert exact.mapping_priority == 1
    assert exact.unit_base == "USD"
    assert "Instant" in exact.allowed_period_types

    unmapped = module.resolve_mapping(parse_cfg, "us-gaap", "NotARealTag")
    assert unmapped.metric_name is None
    assert unmapped.mapping_status == "unmapped"
    assert unmapped.mapping_priority >= 10_000



def test_taxonomy_priority_uses_metric_override_before_global(parse_xbrl_module):
    module = parse_xbrl_module
    cfg = _cfg(
        module,
        metric_taxonomy_overrides={"revenue": {"issuer-extension": 0, "us-gaap": 2}},
    )

    rank, source = module.taxonomy_priority(cfg, "issuer-extension", "revenue")
    assert rank == 0
    assert source == "metric_override"

    global_rank, global_source = module.taxonomy_priority(cfg, "dei", "revenue")
    assert global_rank == cfg.taxonomy_priority["dei"]
    assert global_source == "global"



def test_resolve_unit_accepts_alias_unit_and_preserves_value(parse_xbrl_module, parse_cfg):
    module = parse_xbrl_module
    decision = module.resolve_unit(
        parse_cfg,
        metric_name="total_assets",
        source_unit="usdollars",
        source_value=1234.5,
        requested_unit_base="USD",
    )
    assert decision.rejection_class is None
    assert decision.canonical_source_unit == "USD"
    assert decision.unit_base == "USD"
    assert decision.normalized_value == pytest.approx(1234.5)
    assert decision.unit_scale_factor == pytest.approx(1.0)



def test_resolve_unit_rejects_incompatible_unit_when_policy_is_reject(parse_xbrl_module, parse_cfg):
    module = parse_xbrl_module
    decision = module.resolve_unit(
        parse_cfg,
        metric_name="total_assets",
        source_unit="shares",
        source_value=100.0,
        requested_unit_base="USD",
    )
    assert decision.rejection_class == "incompatible_unit"
    assert decision.normalized_value is None
    assert "not compatible" in str(decision.rejection_detail)



def test_resolve_period_type_classifies_quarter_and_negative_duration(parse_xbrl_module, parse_cfg):
    module = parse_xbrl_module

    quarter = module.resolve_period_type(
        parse_cfg,
        period_start=pd.Timestamp("2025-10-01"),
        period_end=pd.Timestamp("2025-12-31"),
        frame="CY2025Q4I",
        fy=2025,
        fp="Q4",
    )
    assert quarter.period_type == "Q"
    assert quarter.duration_days == 91
    assert quarter.context_quality >= 0.70
    assert quarter.rejection_class is None

    invalid = module.resolve_period_type(
        parse_cfg,
        period_start=pd.Timestamp("2026-01-01"),
        period_end=pd.Timestamp("2025-12-31"),
        frame=None,
        fy=2025,
        fp="FY",
    )
    assert invalid.period_type is None
    assert invalid.rejection_class == "invalid_date"
    assert invalid.rejection_detail == "period_start_after_period_end"



def test_check_value_bounds_uses_mapping_specific_minimum(parse_xbrl_module):
    module = parse_xbrl_module
    cfg = _cfg(module)
    mapping = module.resolve_mapping(cfg, "dei", "EntityCommonStockSharesOutstanding")

    violation = module.check_value_bounds(cfg, "shares_outstanding", "shares", -1.0, mapping)
    assert violation is not None
    assert violation[0] == "invalid_value"
    assert "below_min" in violation[1]


# -----------------------------------------------------------------------------
# Tests: candidate construction
# -----------------------------------------------------------------------------


def test_build_candidate_rows_rejects_unmapped_tag(parse_xbrl_module, parse_cfg, submissions_lookup):
    module = parse_xbrl_module
    facts_df = pd.DataFrame(
        [
            {
                "cik": "0000320193",
                "taxonomy": "us-gaap",
                "tag": "DefinitelyUnknownTag",
                "source_unit": "USD",
                "value_raw": "1000",
                "period_start": pd.Timestamp("2025-10-01"),
                "period_end": pd.Timestamp("2025-12-31"),
                "filed_date": pd.Timestamp("2026-02-01"),
                "acceptance_datetime": pd.Timestamp("2026-02-01T14:30:00Z"),
                "accession_number": "0000320193-26-000001",
                "entity_name": "APPLE INC",
                "label": "Unknown",
                "description": "Unknown",
                "frame": "CY2025Q4I",
                "fy": 2025,
                "fp": "Q4",
                "source_payload_path": "/tmp/companyfacts.json",
            }
        ]
    )

    accepted, rejected = module.build_candidate_rows(
        facts_df=facts_df,
        submissions_lookup=submissions_lookup,
        config=parse_cfg,
        mapping_version="mapping_v1",
        run_id="pytest_run",
        asof="2026-03-15",
    )

    assert accepted.empty
    assert len(rejected) == 1
    assert rejected.iloc[0]["rejection_class"] == "unmapped_tag"



def test_build_candidate_rows_rejects_filed_before_period_end(parse_xbrl_module, parse_cfg, submissions_lookup):
    module = parse_xbrl_module
    facts_df = pd.DataFrame(
        [
            {
                "cik": "0000320193",
                "taxonomy": "us-gaap",
                "tag": "Assets",
                "source_unit": "USD",
                "value_raw": "1000",
                "period_start": None,
                "period_end": pd.Timestamp("2025-12-31"),
                "filed_date": pd.Timestamp("2025-12-15"),
                "acceptance_datetime": pd.Timestamp("2026-02-01T14:30:00Z"),
                "accession_number": "0000320193-26-000001",
                "entity_name": "APPLE INC",
                "label": "Assets",
                "description": "Assets",
                "frame": None,
                "fy": 2025,
                "fp": "FY",
                "source_payload_path": "/tmp/companyfacts.json",
            }
        ]
    )

    accepted, rejected = module.build_candidate_rows(
        facts_df=facts_df,
        submissions_lookup=submissions_lookup,
        config=parse_cfg,
        mapping_version="mapping_v1",
        run_id="pytest_run",
        asof="2026-03-15",
    )

    assert accepted.empty
    assert len(rejected) == 1
    assert rejected.iloc[0]["rejection_class"] == "invalid_date"
    assert rejected.iloc[0]["rejection_detail"] == "filed_date_before_period_end"



def test_build_candidate_rows_emits_normalized_candidate_for_valid_fact(parse_xbrl_module, parse_cfg, submissions_lookup):
    module = parse_xbrl_module
    facts_df = pd.DataFrame(
        [
            {
                "cik": "0000320193",
                "taxonomy": "us-gaap",
                "tag": "Assets",
                "source_unit": "usdollars",
                "value_raw": "123456",
                "period_start": None,
                "period_end": pd.Timestamp("2025-12-31"),
                "filed_date": pd.Timestamp("2026-02-01"),
                "acceptance_datetime": pd.Timestamp("2026-02-01T14:30:00Z"),
                "accession_number": "0000320193-26-000001",
                "entity_name": "APPLE INC",
                "label": "Assets",
                "description": "Assets",
                "frame": None,
                "fy": 2025,
                "fp": "FY",
                "source_payload_path": "/tmp/companyfacts.json",
            }
        ]
    )

    accepted, rejected = module.build_candidate_rows(
        facts_df=facts_df,
        submissions_lookup=submissions_lookup,
        config=parse_cfg,
        mapping_version="mapping_v1",
        run_id="pytest_run",
        asof="2026-03-15",
    )

    assert rejected.empty
    assert len(accepted) == 1
    row = accepted.iloc[0]
    assert row["metric_name"] == "total_assets"
    assert row["period_type"] == "Instant"
    assert row["unit_base"] == "USD"
    assert row["value"] == pytest.approx(123456.0)
    assert row["symbol"] == "AAPL"
    assert row["form_type"] == "10-K"
    assert row["mapping_version"] == "mapping_v1"


# -----------------------------------------------------------------------------
# Tests: canonical selection
# -----------------------------------------------------------------------------


def test_select_canonical_facts_prefers_better_rank_and_rejects_dominated(parse_xbrl_module, parse_cfg):
    module = parse_xbrl_module
    candidates = pd.DataFrame(
        [
            {
                "cik": "0000320193",
                "metric_name": "revenue",
                "period_end": pd.Timestamp("2025-12-31"),
                "period_type": "Y",
                "source_taxonomy": "us-gaap",
                "source_tag": "Revenues",
                "source_unit": "USD",
                "value": 100.0,
                "filed_date": pd.Timestamp("2026-02-01"),
                "acceptance_datetime": pd.Timestamp("2026-02-01T14:30:00Z"),
                "accession_number": "0001",
                "form_type": "10-K",
                "mapping_status": "exact",
                "mapping_priority": 1,
                "unit_compatibility": "exact",
                "context_quality": 0.95,
                "source_taxonomy_priority": 1,
                "source_record_hash": "aaa",
                "run_id": "pytest_run",
            },
            {
                "cik": "0000320193",
                "metric_name": "revenue",
                "period_end": pd.Timestamp("2025-12-31"),
                "period_type": "Y",
                "source_taxonomy": "us-gaap",
                "source_tag": "RevenueFromContractWithCustomerExcludingAssessedTax",
                "source_unit": "USD",
                "value": 100.0,
                "filed_date": pd.Timestamp("2026-02-01"),
                "acceptance_datetime": pd.Timestamp("2026-02-01T14:30:00Z"),
                "accession_number": "0001",
                "form_type": "10-K",
                "mapping_status": "alias",
                "mapping_priority": 2,
                "unit_compatibility": "exact",
                "context_quality": 0.95,
                "source_taxonomy_priority": 1,
                "source_record_hash": "bbb",
                "run_id": "pytest_run",
            },
        ]
    )

    canonical, rejected = module.select_canonical_facts(parse_cfg, candidates, run_id="pytest_run")

    assert len(canonical) == 1
    assert canonical.iloc[0]["source_tag"] == "Revenues"
    assert canonical.iloc[0]["selection_reason"] in {"dominant_candidate", "single_candidate", "tie_broken_by_source_hash"}
    assert canonical.iloc[0]["canonical_priority_rank"] == 1
    assert canonical.iloc[0]["quality_score"] > 0
    assert len(rejected) == 1
    assert rejected.iloc[0]["rejection_class"] == "duplicate_dominated"



def test_select_canonical_facts_rejects_unresolved_tie_when_policy_is_reject(parse_xbrl_module):
    module = parse_xbrl_module
    cfg = _cfg(module, unresolved_tie_policy="reject")
    candidates = pd.DataFrame(
        [
            {
                "cik": "0000320193",
                "metric_name": "revenue",
                "period_end": pd.Timestamp("2025-12-31"),
                "period_type": "Y",
                "source_taxonomy": "us-gaap",
                "source_tag": "Revenues",
                "source_unit": "USD",
                "value": 100.0,
                "filed_date": pd.Timestamp("2026-02-01"),
                "acceptance_datetime": pd.Timestamp("2026-02-01T14:30:00Z"),
                "accession_number": "0001",
                "form_type": "10-K",
                "mapping_status": "exact",
                "mapping_priority": 1,
                "unit_compatibility": "exact",
                "context_quality": 0.95,
                "source_taxonomy_priority": 1,
                "source_record_hash": "hash_a",
                "run_id": "pytest_run",
            },
            {
                "cik": "0000320193",
                "metric_name": "revenue",
                "period_end": pd.Timestamp("2025-12-31"),
                "period_type": "Y",
                "source_taxonomy": "us-gaap",
                "source_tag": "Revenues",
                "source_unit": "USD",
                "value": 101.0,
                "filed_date": pd.Timestamp("2026-02-01"),
                "acceptance_datetime": pd.Timestamp("2026-02-01T14:30:00Z"),
                "accession_number": "0002",
                "form_type": "10-K/A",
                "mapping_status": "exact",
                "mapping_priority": 1,
                "unit_compatibility": "exact",
                "context_quality": 0.95,
                "source_taxonomy_priority": 1,
                "source_record_hash": "hash_b",
                "run_id": "pytest_run",
            },
        ]
    )

    canonical, rejected = module.select_canonical_facts(cfg, candidates, run_id="pytest_run")

    assert canonical.empty
    assert len(rejected) == 2
    assert set(rejected["rejection_class"].tolist()) == {"conflict_unresolved"}



def test_select_canonical_facts_accepts_unresolved_tie_when_policy_is_accept(parse_xbrl_module):
    module = parse_xbrl_module
    cfg = _cfg(module, unresolved_tie_policy="accept")
    candidates = pd.DataFrame(
        [
            {
                "cik": "0000320193",
                "metric_name": "revenue",
                "period_end": pd.Timestamp("2025-12-31"),
                "period_type": "Y",
                "source_taxonomy": "us-gaap",
                "source_tag": "Revenues",
                "source_unit": "USD",
                "value": 100.0,
                "filed_date": pd.Timestamp("2026-02-01"),
                "acceptance_datetime": pd.Timestamp("2026-02-01T14:30:00Z"),
                "accession_number": "0001",
                "form_type": "10-K",
                "mapping_status": "exact",
                "mapping_priority": 1,
                "unit_compatibility": "exact",
                "context_quality": 0.95,
                "source_taxonomy_priority": 1,
                "source_record_hash": "hash_a",
                "run_id": "pytest_run",
            },
            {
                "cik": "0000320193",
                "metric_name": "revenue",
                "period_end": pd.Timestamp("2025-12-31"),
                "period_type": "Y",
                "source_taxonomy": "us-gaap",
                "source_tag": "Revenues",
                "source_unit": "USD",
                "value": 101.0,
                "filed_date": pd.Timestamp("2026-02-01"),
                "acceptance_datetime": pd.Timestamp("2026-02-01T14:30:00Z"),
                "accession_number": "0002",
                "form_type": "10-K/A",
                "mapping_status": "exact",
                "mapping_priority": 1,
                "unit_compatibility": "exact",
                "context_quality": 0.95,
                "source_taxonomy_priority": 1,
                "source_record_hash": "hash_b",
                "run_id": "pytest_run",
            },
        ]
    )

    canonical, rejected = module.select_canonical_facts(cfg, candidates, run_id="pytest_run")

    assert len(canonical) == 1
    assert canonical.iloc[0]["selection_reason"] == "tie_accepted_by_policy"
    assert canonical.iloc[0]["quality_score"] > 0
    assert len(rejected) == 1
    assert rejected.iloc[0]["rejection_class"] == "duplicate_dominated"
