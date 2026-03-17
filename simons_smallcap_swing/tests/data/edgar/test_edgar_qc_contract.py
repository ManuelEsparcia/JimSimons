
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SUMMARY_REQUIRED_KEYS = {
    "run_id",
    "gate",
    "score",
    "decision_reasons",
    "config_version",
    "config_hash",
    "timestamp_utc",
    "metrics",
    "score_losses",
    "counts",
}

CHECKS_REQUIRED_COLUMNS = {
    "stage",
    "check_name",
    "value",
    "threshold_warn",
    "threshold_fail",
    "severity",
    "blocking",
    "passed",
    "message",
    "dimensions",
}

FAILURES_REQUIRED_COLUMNS = {
    "stage",
    "check_name",
    "severity",
    "blocking",
    "issue",
    "symbol",
    "cik",
    "metric",
    "asof",
    "row_index",
    "value",
    "threshold",
    "details",
}

METRICS_REQUIRED_COLUMNS = {
    "asof",
    "pit_rows",
    "leakage_count",
    "median_staleness_days",
    "traceability_rate",
}

MANIFEST_REQUIRED_KEYS = {
    "run_id",
    "module",
    "generated_at_utc",
    "gate",
    "score",
    "decision_reasons",
    "config_version",
    "config_hash",
    "git_revision",
    "python_version",
    "artifacts",
    "inputs",
}

MANIFEST_ARTIFACT_KEYS = {"summary", "metrics", "failures", "checks", "manifest"}
MANIFEST_INPUT_KEYS = {
    "ticker_cik_outputs",
    "submissions_raw",
    "companyfacts_raw",
    "parsed_facts",
    "pit_store",
}


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
            spec = importlib.util.spec_from_file_location("edgar_qc_contract_under_test", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module

    raise ModuleNotFoundError("Could not import edgar_qc.py.")


@pytest.fixture(scope="session")
def edgar_qc_module():
    return _load_edgar_qc_module()


def _ts(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _read_materialized_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


@pytest.fixture()
def patch_write_parquet_to_csv(monkeypatch: pytest.MonkeyPatch, edgar_qc_module):
    def _writer(path: Path, df: pd.DataFrame, *, index: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=index)

    monkeypatch.setattr(edgar_qc_module, "write_parquet", _writer)
    return _writer


def _build_clean_inputs():
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
                "filing_date": _ts("2026-02-10"),
                "status_code": 200,
                "raw_artifact_id": "sub_raw_1",
                "run_id": "sub_run",
            },
            {
                "symbol": "BBB",
                "cik": "0000000002",
                "source_filing_id": "0002-26-000001",
                "acceptance_ts": _ts("2026-02-11 15:00:00"),
                "filing_date": _ts("2026-02-11"),
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
    return mapping, submissions, companyfacts, parsed, pit


@pytest.fixture()
def clean_case(tmp_path: Path, edgar_qc_module, patch_write_parquet_to_csv):
    mapping, submissions, companyfacts, parsed, pit = _build_clean_inputs()
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs_clean"
    input_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ticker_cik_outputs": input_dir / "ticker_cik_outputs.csv",
        "submissions_raw": input_dir / "submissions_raw.csv",
        "companyfacts_raw": input_dir / "companyfacts_raw.csv",
        "parsed_facts": input_dir / "parsed_facts.csv",
        "pit_store": input_dir / "pit_store.csv",
    }
    mapping.to_csv(paths["ticker_cik_outputs"], index=False)
    submissions.to_csv(paths["submissions_raw"], index=False)
    companyfacts.to_csv(paths["companyfacts_raw"], index=False)
    parsed.to_csv(paths["parsed_facts"], index=False)
    pit.to_csv(paths["pit_store"], index=False)

    result = edgar_qc_module.run_edgar_qc(
        ticker_cik_outputs_path=paths["ticker_cik_outputs"],
        submissions_raw_path=paths["submissions_raw"],
        companyfacts_raw_path=paths["companyfacts_raw"],
        parsed_facts_path=paths["parsed_facts"],
        pit_store_path=paths["pit_store"],
        output_dir=output_dir,
        config_path=None,
        run_id="pytest_edgar_qc_contract_clean",
    )
    return {"result": result, "output_dir": output_dir, "input_paths": paths}


@pytest.fixture()
def failure_case(tmp_path: Path, edgar_qc_module, patch_write_parquet_to_csv):
    mapping, submissions, companyfacts, parsed, pit = _build_clean_inputs()
    output_dir = tmp_path / "outputs_fail"
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    pit = pit.iloc[[0]].copy()
    mapping = mapping.iloc[[0]].copy()
    submissions = submissions.iloc[[0]].copy()
    companyfacts = companyfacts.iloc[[0]].copy()
    parsed = parsed.iloc[[0]].copy()
    pit.loc[:, "acceptance_ts"] = _ts("2026-03-16 00:00:00")  # deliberate leakage

    paths = {
        "ticker_cik_outputs": input_dir / "ticker_cik_outputs.csv",
        "submissions_raw": input_dir / "submissions_raw.csv",
        "companyfacts_raw": input_dir / "companyfacts_raw.csv",
        "parsed_facts": input_dir / "parsed_facts.csv",
        "pit_store": input_dir / "pit_store.csv",
    }
    mapping.to_csv(paths["ticker_cik_outputs"], index=False)
    submissions.to_csv(paths["submissions_raw"], index=False)
    companyfacts.to_csv(paths["companyfacts_raw"], index=False)
    parsed.to_csv(paths["parsed_facts"], index=False)
    pit.to_csv(paths["pit_store"], index=False)

    result = edgar_qc_module.run_edgar_qc(
        ticker_cik_outputs_path=paths["ticker_cik_outputs"],
        submissions_raw_path=paths["submissions_raw"],
        companyfacts_raw_path=paths["companyfacts_raw"],
        parsed_facts_path=paths["parsed_facts"],
        pit_store_path=paths["pit_store"],
        output_dir=output_dir,
        config_path=None,
        run_id="pytest_edgar_qc_contract_fail",
    )
    return {"result": result, "output_dir": output_dir, "input_paths": paths}


def test_summary_contains_required_keys_and_valid_domains(clean_case):
    summary = clean_case["result"]["summary"]
    assert SUMMARY_REQUIRED_KEYS.issubset(summary.keys())
    assert summary["gate"] in {"pass", "warn", "fail"}
    assert isinstance(summary["decision_reasons"], list)
    assert 0.0 <= float(summary["score"]) <= 100.0
    assert isinstance(summary["metrics"], dict)
    assert isinstance(summary["score_losses"], dict)
    assert isinstance(summary["counts"], dict)


def test_checks_output_contains_required_columns_and_valid_domains(clean_case, edgar_qc_module):
    checks = clean_case["result"]["checks"]
    assert CHECKS_REQUIRED_COLUMNS.issubset(checks.columns)
    assert set(checks["severity"].dropna().unique()).issubset(edgar_qc_module.VALID_SEVERITIES)
    assert checks["blocking"].map(lambda x: isinstance(x, (bool, int))).all()
    assert checks["passed"].map(lambda x: isinstance(x, (bool, int))).all()


def test_failures_output_contains_required_columns_and_valid_domains(failure_case, edgar_qc_module):
    failures = failure_case["result"]["failures"]
    assert not failures.empty
    assert FAILURES_REQUIRED_COLUMNS.issubset(failures.columns)
    assert set(failures["severity"].dropna().unique()).issubset(edgar_qc_module.VALID_SEVERITIES)
    assert failures["issue"].astype(str).str.len().gt(0).all()


def test_metrics_output_contains_required_columns_and_is_sorted(clean_case):
    metrics = clean_case["result"]["metrics"]
    assert METRICS_REQUIRED_COLUMNS.issubset(metrics.columns)
    pit_symbol_cols = [c for c in metrics.columns if c.startswith("pit_symbol_count")]
    assert pit_symbol_cols, metrics.columns.tolist()
    assert metrics["asof"].astype(str).tolist() == sorted(metrics["asof"].astype(str).tolist())
    assert metrics["pit_rows"].ge(0).all()
    for col in pit_symbol_cols:
        assert metrics[col].fillna(0).ge(0).all()
    assert metrics["leakage_count"].ge(0).all()


def test_manifest_contains_required_keys_and_artifacts_exist(clean_case):
    manifest = clean_case["result"]["manifest"]
    assert MANIFEST_REQUIRED_KEYS.issubset(manifest.keys())
    assert MANIFEST_ARTIFACT_KEYS.issubset(manifest["artifacts"].keys())
    assert MANIFEST_INPUT_KEYS.issubset(manifest["inputs"].keys())
    for path_str in manifest["artifacts"].values():
        assert Path(path_str).exists(), path_str
    for name, info in manifest["inputs"].items():
        assert set(info.keys()) == {"path", "sha256"}
        assert Path(info["path"]).exists(), name
        assert isinstance(info["sha256"], str) and len(info["sha256"]) == 64


def test_summary_counts_match_materialized_outputs(clean_case):
    result = clean_case["result"]
    summary = result["summary"]
    checks = result["checks"]
    failures = result["failures"]
    critical_failures = 0
    if not failures.empty and "severity" in failures.columns:
        critical_failures = int((failures["severity"] == "critical").sum())
    assert int(summary["counts"]["checks_total"]) == len(checks)
    assert int(summary["counts"]["checks_failed"]) == int((~checks["passed"]).sum())
    assert int(summary["counts"]["failures_total"]) == len(failures)
    assert int(summary["counts"]["critical_failures"]) == critical_failures


def test_manifest_json_roundtrip_matches_returned_manifest(clean_case):
    output_dir = clean_case["output_dir"]
    returned = clean_case["result"]["manifest"]
    on_disk = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == returned


def test_disk_artifacts_roundtrip_semantically(clean_case):
    output_dir = clean_case["output_dir"]
    result = clean_case["result"]
    summary_disk = json.loads((output_dir / "edgar_qc_summary.json").read_text(encoding="utf-8"))
    checks_disk = _read_materialized_csv(output_dir / "edgar_qc_checks.parquet")
    failures_disk = _read_materialized_csv(output_dir / "edgar_qc_failures.parquet")
    metrics_disk = _read_materialized_csv(output_dir / "edgar_qc_metrics.parquet")
    assert summary_disk == result["summary"]
    assert len(checks_disk) == len(result["checks"])
    assert len(failures_disk) == len(result["failures"])
    assert len(metrics_disk) == len(result["metrics"])
    assert set(checks_disk.columns) == set(result["checks"].columns)
    assert set(failures_disk.columns) == set(result["failures"].columns)
    assert set(metrics_disk.columns) == set(result["metrics"].columns)


def test_input_hashes_in_manifest_match_actual_files(clean_case, edgar_qc_module):
    manifest = clean_case["result"]["manifest"]
    for name, info in manifest["inputs"].items():
        path = Path(info["path"])
        assert info["sha256"] == edgar_qc_module.file_sha256(path), name


def test_failure_case_gate_and_reasons_are_consistent(failure_case):
    result = failure_case["result"]
    summary = result["summary"]
    failures = result["failures"]
    checks = result["checks"]
    assert summary["gate"] == "fail"
    assert isinstance(summary["decision_reasons"], list) and summary["decision_reasons"]
    assert not failures.empty
    assert (~checks["passed"]).any()


def test_materialized_checks_and_failures_preserve_same_rows_as_returned(failure_case):
    output_dir = failure_case["output_dir"]
    checks_disk = _read_materialized_csv(output_dir / "edgar_qc_checks.parquet")
    failures_disk = _read_materialized_csv(output_dir / "edgar_qc_failures.parquet")
    checks = failure_case["result"]["checks"].copy()
    failures = failure_case["result"]["failures"].copy()

    for df in (checks_disk, failures_disk, checks, failures):
        for col in ("dimensions", "details", "cik", "symbol", "metric", "asof", "value", "threshold"):
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
        if "cik" in df.columns:
            df["cik"] = df["cik"].map(lambda x: "" if x == "" else x.split(".")[0].zfill(10))

    pd.testing.assert_frame_equal(
        checks_disk.sort_values(list(checks_disk.columns.astype(str)), kind="stable").reset_index(drop=True),
        checks.sort_values(list(checks.columns.astype(str)), kind="stable").reset_index(drop=True),
        check_like=True,
    )
    pd.testing.assert_frame_equal(
        failures_disk.sort_values(list(failures_disk.columns.astype(str)), kind="stable").reset_index(drop=True),
        failures.sort_values(list(failures.columns.astype(str)), kind="stable").reset_index(drop=True),
        check_like=True,
    )
