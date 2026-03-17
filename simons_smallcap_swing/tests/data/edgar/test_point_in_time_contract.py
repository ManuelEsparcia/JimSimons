from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


REQUIRED_PIT_COLS = {
    "symbol",
    "cik",
    "mapping_version",
    "metric_name",
    "asof",
    "metric_value_pit",
    "source_period_end",
    "source_filed_date",
    "source_acceptance_ts",
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
    "run_id",
    "pit_config_version",
}

REQUIRED_COVERAGE_COLS = {
    "asof",
    "metric_name",
    "n_symbols",
    "n_expected",
    "n_usable",
    "coverage_ratio",
    "n_no_candidate",
    "n_quality_excluded",
    "n_stale",
    "n_penalized",
    "n_ffill",
    "coverage_severity",
}

REQUIRED_ISSUES_COLS = {
    "asof",
    "issue_type",
    "symbol",
    "cik",
    "metric_name",
    "details",
    "severity",
}

REQUIRED_MANIFEST_KEYS = {
    "run_id",
    "pit_config_version",
    "config_hash",
    "started_at_utc",
    "finished_at_utc",
    "asof_start",
    "asof_end",
    "n_asofs",
    "n_symbols_max",
    "n_metrics",
    "n_rows_pit",
    "n_rows_issues",
    "gate",
    "coverage_ratio_all_mean",
    "pct_penalized",
    "pct_excluded",
    "pct_stale",
    "paths",
}


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
            spec = importlib.util.spec_from_file_location("point_in_time_contract_under_test", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module

    raise ModuleNotFoundError(
        "Could not import point_in_time.py. Expected it at "
        "simons_smallcap_swing.data.edgar.point_in_time or in a nearby repo path."
    )


@pytest.fixture(scope="session")
def point_in_time_module():
    return _load_point_in_time_module()


# -----------------------------------------------------------------------------
# Synthetic end-to-end fixture
# -----------------------------------------------------------------------------


@pytest.fixture()
def point_in_time_case(tmp_path: Path, point_in_time_module):
    module = point_in_time_module

    facts = pd.DataFrame(
        [
            {
                "cik": "1",
                "metric_name": "Revenue",
                "value": 105.0,
                "period_end": "2025-12-31",
                "filed_date": "2026-02-10",
                "acceptance_datetime": "2026-02-10T14:00:00Z",
                "source_accession_number": "0001-26-000002",
                "source_form_type": "10-K",
                "fact_quality_score": 0.80,
                "amendment_type": "original",
            },
            {
                "cik": "1",
                "metric_name": "TotalAssets",
                "value": 250.0,
                "period_end": "2025-12-31",
                "filed_date": "2026-02-10",
                "acceptance_datetime": "2026-02-10T14:00:00Z",
                "source_accession_number": "0001-26-000002",
                "source_form_type": "10-K",
                "fact_quality_score": 0.90,
                "amendment_type": "original",
            },
            {
                "cik": "2",
                "metric_name": "Revenue",
                "value": 88.0,
                "period_end": "2025-12-31",
                "filed_date": "2026-02-12",
                "acceptance_datetime": "2026-02-12T09:30:00Z",
                "source_accession_number": "0002-26-000010",
                "source_form_type": "10-K",
                "fact_quality_score": 0.75,
                "amendment_type": "original",
            },
        ]
    )
    mapping = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "cik": "1",
                "effective_from": "2020-01-01",
                "mapping_version": "map_v1",
                "resolution_status": "exact",
                "confidence_score": 1.0,
            },
            {
                "symbol": "BBB",
                "cik": "2",
                "effective_from": "2020-01-01",
                "mapping_version": "map_v1",
                "resolution_status": "exact",
                "confidence_score": 1.0,
            },
        ]
    )
    flags = pd.DataFrame(
        [
            {
                "cik": "1",
                "source_accession_number": "0001-26-000002",
                "flag_severity": "WARN",
                "pit_action": "penalize",
                "quality_weight": 0.40,
                "source_form_type": "10-K",
            }
        ]
    )

    facts_path = tmp_path / "facts.csv"
    mapping_path = tmp_path / "mapping.csv"
    flags_path = tmp_path / "flags.csv"
    output_root = tmp_path / "pit_out"

    facts.to_csv(facts_path, index=False)
    mapping.to_csv(mapping_path, index=False)
    flags.to_csv(flags_path, index=False)

    config = {
        "pit_config_version": "pytest_pit_contract_v1",
        "storage": {
            "output_root": str(output_root),
            "allow_csv_fallback": True,
            "compression": "snappy",
        },
        "calendar": {
            "freq": "D",
            "asof_time_of_day": "23:59:59",
            "timezone": "UTC",
        },
        "metrics": {"include": ["Revenue", "TotalAssets"], "exclude": []},
        "flags": {"enabled": True},
        "validation": {
            "abort_on_input_contract_failure": True,
            "coverage_warn_threshold": 0.60,
            "coverage_fail_threshold": 0.30,
            "abort_on_leakage": True,
            "abort_on_duplicate_output": True,
            "abort_on_identity_ambiguity": False,
        },
    }
    config_path = tmp_path / "pit_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    run_id = "pytest_point_in_time_contract"
    result = module.materialize_point_in_time(
        parsed_facts_path=str(facts_path),
        ticker_cik_mapping_path=str(mapping_path),
        pit_config_path=str(config_path),
        run_id=run_id,
        start_date="2026-03-15",
        end_date="2026-03-16",
        filings_flags_path=str(flags_path),
    )

    pit = result["pit"].copy()
    coverage = result["coverage"].copy()
    issues = result["issues"].copy()
    manifest = result["manifest"].copy()

    coverage_path = Path(manifest["paths"]["coverage"])
    issues_path = Path(manifest["paths"]["issues"])
    manifest_path = Path(result["manifest_path"])
    partitions = sorted(output_root.glob("date=*/part-*.csv"))

    pit_disk = pd.concat((pd.read_csv(p) for p in partitions), ignore_index=True) if partitions else pd.DataFrame()
    coverage_disk = pd.read_csv(coverage_path)
    issues_disk = pd.read_csv(issues_path)
    manifest_disk = json.loads(manifest_path.read_text(encoding="utf-8"))

    return {
        "module": module,
        "result": result,
        "pit": pit,
        "coverage": coverage,
        "issues": issues,
        "manifest": manifest,
        "pit_disk": pit_disk,
        "coverage_disk": coverage_disk,
        "issues_disk": issues_disk,
        "manifest_disk": manifest_disk,
        "manifest_path": manifest_path,
        "coverage_path": coverage_path,
        "issues_path": issues_path,
        "partitions": partitions,
        "output_root": output_root,
        "run_id": run_id,
    }


# -----------------------------------------------------------------------------
# Contract tests
# -----------------------------------------------------------------------------


def test_pit_panel_contains_required_columns(point_in_time_case):
    pit = point_in_time_case["pit"]
    assert REQUIRED_PIT_COLS.issubset(set(pit.columns))
    assert len(pit) == 8


def test_coverage_output_contains_required_metrics(point_in_time_case):
    coverage = point_in_time_case["coverage"]
    assert REQUIRED_COVERAGE_COLS.issubset(set(coverage.columns))
    assert len(coverage) == 6  # 2 asofs x (Revenue, TotalAssets, __ALL__)
    assert set(coverage["metric_name"]) == {"Revenue", "TotalAssets", "__ALL__"}
    assert coverage["coverage_ratio"].between(0.0, 1.0).all()


def test_issues_output_contains_required_columns_even_when_empty(point_in_time_case):
    issues = point_in_time_case["issues"]
    assert REQUIRED_ISSUES_COLS.issubset(set(issues.columns))
    assert issues.empty


def test_manifest_contains_required_metadata_and_paths(point_in_time_case):
    manifest = point_in_time_case["manifest"]
    assert REQUIRED_MANIFEST_KEYS.issubset(set(manifest.keys()))
    assert manifest["run_id"] == point_in_time_case["run_id"]
    assert manifest["pit_config_version"] == "pytest_pit_contract_v1"
    assert manifest["gate"] in {"PASS", "WARN", "FAIL"}
    assert set(manifest["paths"].keys()) == {"output_root", "coverage", "issues"}
    assert Path(manifest["paths"]["output_root"]).exists()
    assert Path(manifest["paths"]["coverage"]).exists()
    assert Path(manifest["paths"]["issues"]).exists()


def test_partition_files_exist_for_each_asof(point_in_time_case):
    partitions = point_in_time_case["partitions"]
    assert len(partitions) == 2
    observed_dates = {p.parent.name.replace("date=", "") for p in partitions}
    assert observed_dates == {"2026-03-15", "2026-03-16"}
    for path in partitions:
        assert path.exists()
        df = pd.read_csv(path)
        assert REQUIRED_PIT_COLS.issubset(set(df.columns))


def test_selected_rows_have_traceability_to_filing_and_acceptance(point_in_time_case):
    pit = point_in_time_case["pit"]
    usable = pit[pit["pit_action"].isin(["allow", "penalize", "stale"])]
    assert not usable.empty
    for col in ["source_accession_number", "source_form_type", "source_period_end", "source_filed_date", "source_acceptance_ts"]:
        assert usable[col].notna().all(), f"Expected populated traceability column: {col}"


def test_output_key_is_unique_by_asof_symbol_metric_name(point_in_time_case):
    pit = point_in_time_case["pit"]
    dupes = pit.duplicated(subset=["asof", "symbol", "metric_name"], keep=False)
    assert not dupes.any(), pit.loc[dupes, ["asof", "symbol", "metric_name"]].to_dict("records")


def test_pit_action_and_quality_domains_are_valid(point_in_time_case):
    pit = point_in_time_case["pit"]
    assert set(pit["pit_action"]).issubset({"allow", "penalize", "exclude", "stale"})
    assert pit["quality_weight"].between(0.0, 1.0).all()
    assert set(pit["flag_severity"].dropna().astype(str).str.upper()).issubset({"NONE", "INFO", "WARN", "MEDIUM", "HIGH", "CRITICAL", "FAIL"})


def test_manifest_counts_match_materialized_outputs(point_in_time_case):
    manifest = point_in_time_case["manifest"]
    pit = point_in_time_case["pit"]
    issues = point_in_time_case["issues"]
    coverage = point_in_time_case["coverage"]

    assert int(manifest["n_rows_pit"]) == len(pit)
    assert int(manifest["n_rows_issues"]) == len(issues)
    assert int(manifest["n_asofs"]) == pit["asof"].nunique()
    assert int(manifest["n_metrics"]) == pit["metric_name"].nunique()
    observed_all_mean = float(coverage.loc[coverage["metric_name"] == "__ALL__", "coverage_ratio"].mean())
    assert float(manifest["coverage_ratio_all_mean"]) == pytest.approx(observed_all_mean)


def test_output_is_sorted_deterministically(point_in_time_case):
    pit = point_in_time_case["pit"]
    sorted_pit = pit.sort_values(["asof", "symbol", "metric_name"], kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(pit.reset_index(drop=True), sorted_pit)

    coverage = point_in_time_case["coverage"].copy()
    cov_order = coverage["metric_name"].map(lambda x: (0, "") if x == "__ALL__" else (1, str(x)))
    sorted_cov = coverage.assign(_order=cov_order).sort_values(["asof", "_order"], kind="mergesort").drop(columns=["_order"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(coverage.reset_index(drop=True), sorted_cov)


def test_disk_roundtrip_is_semantically_consistent(point_in_time_case):
    pit = point_in_time_case["pit"].copy()
    pit_disk = point_in_time_case["pit_disk"].copy()

    # Normalize dtypes that roundtrip through CSV as strings.
    for col in ["asof", "source_period_end", "source_filed_date", "source_acceptance_ts"]:
        pit[col] = pd.to_datetime(pit[col], utc=False, errors="coerce")
        pit_disk[col] = pd.to_datetime(pit_disk[col], utc=False, errors="coerce")

    pit["cik"] = pit["cik"].astype(str).str.zfill(10)
    pit_disk["cik"] = pit_disk["cik"].astype(str).str.zfill(10)

    sort_cols = ["asof", "symbol", "metric_name"]
    pit = pit.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    pit_disk = pit_disk.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(pit, pit_disk, check_dtype=False)


def test_manifest_json_roundtrip_matches_return_semantically(point_in_time_case):
    manifest = point_in_time_case["manifest"]
    manifest_disk = point_in_time_case["manifest_disk"]

    assert set(manifest_disk.keys()) == set(manifest.keys())
    assert manifest_disk["run_id"] == manifest["run_id"]
    assert manifest_disk["pit_config_version"] == manifest["pit_config_version"]
    assert manifest_disk["config_hash"] == manifest["config_hash"]
    assert manifest_disk["gate"] == manifest["gate"]
    assert manifest_disk["paths"] == manifest["paths"]
    assert int(manifest_disk["n_rows_pit"]) == int(manifest["n_rows_pit"])
    assert int(manifest_disk["n_rows_issues"]) == int(manifest["n_rows_issues"])
