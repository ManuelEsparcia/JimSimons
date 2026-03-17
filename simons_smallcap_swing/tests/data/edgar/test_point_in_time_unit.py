from __future__ import annotations

import copy
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# Import adapter
# -----------------------------------------------------------------------------


def _load_point_in_time_module():
    module_names = [
        "simons_smallcap_swing.data.edgar.point_in_time",
        "data.edgar.point_in_time",
        "point_in_time",
    ]
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue

    here = Path(__file__).resolve()
    candidate_paths = [
        here.parents[3] / "simons_smallcap_swing" / "data" / "edgar" / "point_in_time.py",
        here.parents[2] / "data" / "edgar" / "point_in_time.py",
        here.parents[1] / "point_in_time.py",
        Path("/mnt/data/point_in_time.py"),
    ]
    for path in candidate_paths:
        if path.exists():
            spec = importlib.util.spec_from_file_location("point_in_time", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module

    raise ModuleNotFoundError(
        "Could not import point_in_time.py. Expected it at "
        "simons_smallcap_swing.data.edgar.point_in_time or in a nearby repo path."
    )


@pytest.fixture(scope="session")
def point_in_time_module():
    return _load_point_in_time_module()


# -----------------------------------------------------------------------------
# Helpers / fixtures
# -----------------------------------------------------------------------------


def _cfg(module):
    cfg = copy.deepcopy(module.DEFAULT_CONFIG)
    cfg["run_id"] = "pytest_run"
    cfg["pit_config_version"] = "pytest_cfg"
    return cfg


@pytest.fixture()
def cfg(point_in_time_module):
    return _cfg(point_in_time_module)


@pytest.fixture()
def asof():
    return pd.Timestamp("2026-03-15 23:59:59")


@pytest.fixture()
def active_map(point_in_time_module, asof):
    module = point_in_time_module
    mapping = pd.DataFrame([
        {
            "symbol": "AAA",
            "cik": "1",
            "effective_from": pd.Timestamp("2020-01-01"),
            "mapping_version": "map_v1",
            "resolution_status": "exact",
            "confidence_score": 1.0,
        }
    ])
    prepared = module.prepare_mapping(mapping)
    active, issues = module.active_mapping_for_asof(prepared, asof)
    assert issues.empty
    return active


@pytest.fixture()
def base_facts(point_in_time_module):
    module = point_in_time_module
    raw = pd.DataFrame(
        [
            {
                "cik": "1",
                "metric_name": "Revenue",
                "value": 100.0,
                "period_end": pd.Timestamp("2025-12-31"),
                "filed_date": pd.Timestamp("2026-02-01"),
                "acceptance_datetime": pd.Timestamp("2026-02-01T14:00:00Z"),
                "source_accession_number": "0001-26-000001",
                "source_form_type": "10-K",
                "fact_quality_score": 0.70,
                "amendment_type": "original",
            },
            {
                "cik": "1",
                "metric_name": "Revenue",
                "value": 105.0,
                "period_end": pd.Timestamp("2025-12-31"),
                "filed_date": pd.Timestamp("2026-02-10"),
                "acceptance_datetime": pd.Timestamp("2026-02-10T14:00:00Z"),
                "source_accession_number": "0001-26-000002",
                "source_form_type": "10-K",
                "fact_quality_score": 0.80,
                "amendment_type": "original",
            },
            {
                "cik": "1",
                "metric_name": "TotalAssets",
                "value": 250.0,
                "period_end": pd.Timestamp("2025-12-31"),
                "filed_date": pd.Timestamp("2026-02-10"),
                "acceptance_datetime": pd.Timestamp("2026-02-10T14:00:00Z"),
                "source_accession_number": "0001-26-000002",
                "source_form_type": "10-K",
                "fact_quality_score": 0.90,
                "amendment_type": "original",
            },
        ]
    )
    return module.prepare_facts(raw)


# -----------------------------------------------------------------------------
# Tests: candidate selection / PIT causality
# -----------------------------------------------------------------------------


def test_acceptance_after_asof_is_absolutely_blocked(point_in_time_module, cfg, active_map, asof):
    module = point_in_time_module
    facts = module.prepare_facts(
        pd.DataFrame(
            [
                {
                    "cik": "1",
                    "metric_name": "Revenue",
                    "value": 999.0,
                    "period_end": pd.Timestamp("2025-12-31"),
                    "filed_date": pd.Timestamp("2026-03-16"),
                    "acceptance_datetime": pd.Timestamp("2026-03-16T00:00:00Z"),
                    "source_accession_number": "0001-26-999999",
                    "source_form_type": "10-K/A",
                }
            ]
        )
    )

    selected, issues = module.select_candidates_for_asof(
        facts=facts,
        active_map=active_map,
        flags=module.prepare_flags(None, cfg),
        asof=asof,
        cfg=cfg,
    )

    assert selected.empty
    assert issues.empty



def test_last_observable_fact_is_selected(point_in_time_module, cfg, active_map, asof, base_facts):
    module = point_in_time_module
    selected, issues = module.select_candidates_for_asof(
        facts=base_facts,
        active_map=active_map,
        flags=module.prepare_flags(None, cfg),
        asof=asof,
        cfg=cfg,
    )

    assert issues.empty
    revenue = selected.loc[selected["metric_name"] == "Revenue"].iloc[0]
    assert revenue["value"] == pytest.approx(105.0)
    assert revenue["source_accession_number"] == "0001-26-000002"
    assert revenue["selection_reason"] == "latest_observable_ffill"
    assert bool(revenue["ffill_applied"]) is True



def test_amendment_only_applies_after_its_acceptance(point_in_time_module, cfg, active_map):
    module = point_in_time_module
    facts = module.prepare_facts(
        pd.DataFrame(
            [
                {
                    "cik": "1",
                    "metric_name": "Revenue",
                    "value": 100.0,
                    "period_end": pd.Timestamp("2025-12-31"),
                    "filed_date": pd.Timestamp("2026-02-01"),
                    "acceptance_datetime": pd.Timestamp("2026-02-01T14:00:00Z"),
                    "source_accession_number": "0001-26-000001",
                    "source_form_type": "10-K",
                    "amendment_type": "original",
                },
                {
                    "cik": "1",
                    "metric_name": "Revenue",
                    "value": 110.0,
                    "period_end": pd.Timestamp("2025-12-31"),
                    "filed_date": pd.Timestamp("2026-03-20"),
                    "acceptance_datetime": pd.Timestamp("2026-03-20T14:00:00Z"),
                    "source_accession_number": "0001-26-000003",
                    "source_form_type": "10-K/A",
                    "amendment_type": "substantive",
                },
            ]
        )
    )
    flags = module.prepare_flags(None, cfg)

    before_amendment = pd.Timestamp("2026-03-15 23:59:59")
    selected_before, _ = module.select_candidates_for_asof(facts, active_map, flags, before_amendment, cfg)
    assert selected_before.iloc[0]["value"] == pytest.approx(100.0)
    assert selected_before.iloc[0]["source_accession_number"] == "0001-26-000001"

    after_amendment = pd.Timestamp("2026-03-25 23:59:59")
    selected_after, _ = module.select_candidates_for_asof(facts, active_map, flags, after_amendment, cfg)
    assert selected_after.iloc[0]["value"] == pytest.approx(110.0)
    assert selected_after.iloc[0]["source_accession_number"] == "0001-26-000003"
    assert selected_after.iloc[0]["amendment_type"] == "substantive"



def test_equally_ranked_conflict_is_exposed_and_removed(point_in_time_module, cfg, active_map, asof):
    module = point_in_time_module
    facts = module.prepare_facts(
        pd.DataFrame(
            [
                {
                    "cik": "1",
                    "metric_name": "Revenue",
                    "value": 100.0,
                    "period_end": pd.Timestamp("2025-12-31"),
                    "filed_date": pd.Timestamp("2026-02-01"),
                    "acceptance_datetime": pd.Timestamp("2026-02-01T14:00:00Z"),
                    "source_accession_number": "0001-26-000001",
                    "source_form_type": "10-K",
                    "fact_quality_score": 0.9,
                },
                {
                    "cik": "1",
                    "metric_name": "Revenue",
                    "value": 120.0,
                    "period_end": pd.Timestamp("2025-12-31"),
                    "filed_date": pd.Timestamp("2026-02-01"),
                    "acceptance_datetime": pd.Timestamp("2026-02-01T14:00:00Z"),
                    "source_accession_number": "0001-26-000001",
                    "source_form_type": "10-K",
                    "fact_quality_score": 0.9,
                },
            ]
        )
    )

    selected, issues = module.select_candidates_for_asof(
        facts=facts,
        active_map=active_map,
        flags=module.prepare_flags(None, cfg),
        asof=asof,
        cfg=cfg,
    )

    assert selected.empty
    assert len(issues) == 1
    assert issues.iloc[0]["issue_type"] == "selection_conflict"
    assert issues.iloc[0]["severity"] == "HIGH"


# -----------------------------------------------------------------------------
# Tests: quality / staleness policy
# -----------------------------------------------------------------------------


def test_penalize_flag_preserves_value_and_reduces_weight(point_in_time_module, cfg, active_map, asof, base_facts):
    module = point_in_time_module
    flags = module.prepare_flags(
        pd.DataFrame(
            [
                {
                    "cik": "1",
                    "source_accession_number": "0001-26-000002",
                    "flag_severity": "WARN",
                    "pit_action": "penalize",
                    "quality_weight": 0.4,
                }
            ]
        ),
        cfg,
    )

    selected, _ = module.select_candidates_for_asof(base_facts, active_map, flags, asof, cfg)
    out = module.apply_staleness_and_quality(selected, asof, cfg)
    revenue = out.loc[out["metric_name"] == "Revenue"].iloc[0]

    assert revenue["pit_action"] == "penalize"
    assert revenue["quality_weight"] == pytest.approx(0.4)
    assert revenue["metric_value_pit"] == pytest.approx(105.0)
    assert bool(revenue["is_stale"]) is False



def test_exclude_flag_sets_value_nan_and_zero_weight(point_in_time_module, cfg, active_map, asof, base_facts):
    module = point_in_time_module
    flags = module.prepare_flags(
        pd.DataFrame(
            [
                {
                    "cik": "1",
                    "source_accession_number": "0001-26-000002",
                    "flag_severity": "CRITICAL",
                    "pit_action": "exclude",
                    "quality_weight": 0.0,
                }
            ]
        ),
        cfg,
    )

    selected, _ = module.select_candidates_for_asof(base_facts, active_map, flags, asof, cfg)
    out = module.apply_staleness_and_quality(selected, asof, cfg)
    revenue = out.loc[out["metric_name"] == "Revenue"].iloc[0]

    assert revenue["pit_action"] == "exclude"
    assert revenue["quality_weight"] == pytest.approx(0.0)
    assert pd.isna(revenue["metric_value_pit"])



def test_staleness_is_metric_specific(point_in_time_module, cfg, active_map):
    module = point_in_time_module
    asof = pd.Timestamp("2026-09-25 23:59:59")
    facts = module.prepare_facts(
        pd.DataFrame(
            [
                {
                    "cik": "1",
                    "metric_name": "Revenue",
                    "value": 100.0,
                    "period_end": pd.Timestamp("2025-12-31"),
                    "filed_date": pd.Timestamp("2026-02-01"),
                    "acceptance_datetime": pd.Timestamp("2026-02-01T14:00:00Z"),
                    "source_accession_number": "0001-26-000001",
                    "source_form_type": "10-K",
                },
                {
                    "cik": "1",
                    "metric_name": "TotalAssets",
                    "value": 500.0,
                    "period_end": pd.Timestamp("2025-12-31"),
                    "filed_date": pd.Timestamp("2026-02-01"),
                    "acceptance_datetime": pd.Timestamp("2026-02-01T14:00:00Z"),
                    "source_accession_number": "0001-26-000001",
                    "source_form_type": "10-K",
                },
            ]
        )
    )

    selected, _ = module.select_candidates_for_asof(
        facts=facts,
        active_map=active_map,
        flags=module.prepare_flags(None, cfg),
        asof=asof,
        cfg=cfg,
    )
    out = module.apply_staleness_and_quality(selected, asof, cfg)

    revenue = out.loc[out["metric_name"] == "Revenue"].iloc[0]
    assets = out.loc[out["metric_name"] == "TotalAssets"].iloc[0]

    assert bool(revenue["is_stale"]) is True
    assert revenue["pit_action"] == "stale"
    assert revenue["quality_weight"] == pytest.approx(0.0)
    assert pd.isna(revenue["metric_value_pit"])
    assert revenue["max_staleness_days"] == pytest.approx(140.0)

    assert bool(assets["is_stale"]) is True
    assert assets["pit_action"] == "stale"
    assert assets["quality_weight"] == pytest.approx(0.5)
    assert assets["metric_value_pit"] == pytest.approx(500.0)
    assert assets["max_staleness_days"] == pytest.approx(220.0)


# -----------------------------------------------------------------------------
# Tests: identity windows / panel construction / validations
# -----------------------------------------------------------------------------


def test_symbol_cik_resolution_is_point_in_time_safe_across_mapping_windows(point_in_time_module, cfg):
    module = point_in_time_module
    mapping = module.prepare_mapping(
        pd.DataFrame(
            [
                {
                    "symbol": "OLD",
                    "cik": "1",
                    "effective_from": pd.Timestamp("2020-01-01"),
                    "effective_to": pd.Timestamp("2026-03-10"),
                    "mapping_version": "map_v1",
                },
                {
                    "symbol": "NEW",
                    "cik": "1",
                    "effective_from": pd.Timestamp("2026-03-10"),
                    "mapping_version": "map_v1",
                },
            ]
        )
    )
    facts = module.prepare_facts(
        pd.DataFrame(
            [
                {
                    "cik": "1",
                    "metric_name": "Revenue",
                    "value": 100.0,
                    "period_end": pd.Timestamp("2025-12-31"),
                    "filed_date": pd.Timestamp("2026-02-01"),
                    "acceptance_datetime": pd.Timestamp("2026-02-01T14:00:00Z"),
                    "source_accession_number": "0001-26-000001",
                    "source_form_type": "10-K",
                }
            ]
        )
    )
    flags = module.prepare_flags(None, cfg)

    before = pd.Timestamp("2026-03-09 23:59:59")
    active_before, issues_before = module.active_mapping_for_asof(mapping, before)
    selected_before, _ = module.select_candidates_for_asof(facts, active_before, flags, before, cfg)
    assert issues_before.empty
    assert selected_before.iloc[0]["symbol"] == "OLD"

    after = pd.Timestamp("2026-03-10 23:59:59")
    active_after, issues_after = module.active_mapping_for_asof(mapping, after)
    selected_after, _ = module.select_candidates_for_asof(facts, active_after, flags, after, cfg)
    assert issues_after.empty
    assert selected_after.iloc[0]["symbol"] == "NEW"



def test_build_full_panel_marks_missing_rows_as_excluded(point_in_time_module, cfg, active_map, asof):
    module = point_in_time_module
    empty_selected = pd.DataFrame(
        columns=[
            "asof",
            "symbol",
            "cik",
            "metric_name",
            "metric_value_pit",
            "period_end",
            "filed_date",
            "acceptance_datetime",
            "source_accession_number",
            "source_form_type",
            "selection_reason",
            "ffill_applied",
            "staleness_days",
            "pit_action",
            "quality_weight",
            "is_stale",
            "age_bucket",
            "flag_severity",
            "mapping_version",
        ]
    )

    panel = module.build_full_panel(
        asof=asof,
        active_map=active_map,
        metrics=["Revenue"],
        selected=empty_selected,
        issues=pd.DataFrame(),
        cfg=cfg,
    )

    assert len(panel) == 1
    row = panel.iloc[0]
    assert row["symbol"] == "AAA"
    assert row["metric_name"] == "Revenue"
    assert row["pit_action"] == "exclude"
    assert row["selection_reason"] == "no_observable_fact"
    assert row["quality_weight"] == pytest.approx(0.0)
    assert row["age_bucket"] == "missing"
    assert pd.isna(row["metric_value_pit"])



def test_run_validations_detects_leakage_and_duplicate_output(point_in_time_module, cfg, asof):
    module = point_in_time_module
    panel = pd.DataFrame(
        [
            {
                "asof": asof,
                "symbol": "AAA",
                "cik": "0000000001",
                "metric_name": "Revenue",
                "source_acceptance_ts": asof + pd.Timedelta(seconds=1),
                "staleness_days": 0.0,
            },
            {
                "asof": asof,
                "symbol": "AAA",
                "cik": "0000000001",
                "metric_name": "Revenue",
                "source_acceptance_ts": asof - pd.Timedelta(days=1),
                "staleness_days": 1.0,
            },
        ]
    )
    issues = pd.DataFrame(columns=["asof", "issue_type", "symbol", "cik", "metric_name", "details", "severity"])
    coverage = pd.DataFrame(
        [
            {
                "asof": asof,
                "metric_name": "__ALL__",
                "coverage_ratio": 1.0,
                "coverage_severity": "PASS",
            }
        ]
    )

    findings = module.run_validations(panel, issues, coverage, cfg)
    issue_types = set(findings["issue_type"].tolist())

    assert "temporal_leakage" in issue_types
    assert "duplicate_output_key" in issue_types
    assert module.final_gate(findings) == "FAIL"
