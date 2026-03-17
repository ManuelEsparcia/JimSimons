from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# Import adapter
# -----------------------------------------------------------------------------


def _load_qc_prices_module():
    module_names = [
        "simons_smallcap_swing.data.price.qc_prices",
        "data.price.qc_prices",
        "qc_prices",
    ]
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue

    here = Path(__file__).resolve()
    candidate_paths = [
        here.parents[3] / "simons_smallcap_swing" / "data" / "price" / "qc_prices.py",
        here.parents[2] / "data" / "price" / "qc_prices.py",
        here.parents[1] / "qc_prices.py",
        Path("/mnt/data/qc_prices.py"),
    ]
    for path in candidate_paths:
        if path.exists():
            module_name = f"_qc_prices_contract_test_{abs(hash(str(path)))}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module

    raise ModuleNotFoundError("Could not import qc_prices.py")


@pytest.fixture(scope="session")
def qc_prices_module():
    return _load_qc_prices_module()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _cfg(module, **overrides: Any):
    payload = dataclasses.asdict(module.QCConfig())
    payload.update(overrides)
    return module.QCConfig(**payload)



def _write_json(path: Path, obj: Any) -> Path:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path



def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path



def _read_persisted_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df



def _semantic_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy()
    cols = [c for c in ["severity", "symbol", "date", "check_name", "failure_code"] if c in df.columns]
    if not cols:
        cols = list(df.columns)
    return df.sort_values(cols, na_position="last").reset_index(drop=True)



def _required_summary_keys() -> set[str]:
    return {
        "run_id",
        "as_of_ts_utc",
        "n_rows",
        "n_symbols",
        "pct_rows_ohlc_invalid",
        "pct_extreme_returns",
        "pct_symbols_bad_coverage",
        "pct_raw_adjusted_inconsistent",
        "pct_fail_row",
        "fail_count",
        "warn_count",
        "info_count",
        "structural_fail_count",
        "gate",
    }



def _required_row_level_columns() -> set[str]:
    return {
        "symbol",
        "date",
        "check_name",
        "severity",
        "observed_value",
        "threshold",
        "message",
        "failure_code",
    }



def _required_symbol_level_columns() -> set[str]:
    return {
        "symbol",
        "n_rows",
        "pct_missing_sessions_symbol",
        "pct_extreme_returns_symbol",
        "n_ohlc_violations",
        "n_nonpositive_prices",
        "n_raw_adjusted_flags",
        "severity_max",
    }



def _required_manifest_keys() -> set[str]:
    return {
        "run_id",
        "as_of_ts_utc",
        "config_hash",
        "prices_raw_snapshot",
        "calendar_snapshot",
        "config_snapshot",
        "code_version",
        "processing_duration_sec",
        "start_date",
        "end_date",
        "n_rows",
        "n_symbols",
        "gate",
        "generated_at_utc",
        "outputs",
    }


@pytest.fixture()
def contract_case(tmp_path: Path, qc_prices_module, monkeypatch):
    module = qc_prices_module

    raw = pd.DataFrame(
        [
            {"symbol": "AAA", "date": "2024-01-02", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0, "source_provider": "demo"},
            {"symbol": "AAA", "date": "2024-01-03", "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.0, "volume": 1200.0, "source_provider": "demo"},
            # impossible OHLC row to guarantee material failure presence
            {"symbol": "BBB", "date": "2024-01-02", "open": 50.0, "high": 49.0, "low": 48.0, "close": 49.5, "volume": 900.0, "source_provider": "demo"},
            {"symbol": "BBB", "date": "2024-01-03", "open": 49.0, "high": 50.0, "low": 48.0, "close": 49.0, "volume": 950.0, "source_provider": "demo"},
        ]
    )
    raw["date"] = pd.to_datetime(raw["date"])

    calendar = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"])})

    adjusted = pd.DataFrame(
        [
            {"symbol": "AAA", "date": "2024-01-02", "close_adj_split": 100.0, "close_adj_total": 100.0, "ret_1d_adj_split": None, "ret_1d_adj_total": None},
            {"symbol": "AAA", "date": "2024-01-03", "close_adj_split": 101.0, "close_adj_total": 101.0, "ret_1d_adj_split": 0.01, "ret_1d_adj_total": 0.01},
            {"symbol": "BBB", "date": "2024-01-02", "close_adj_split": 49.5, "close_adj_total": 49.5, "ret_1d_adj_split": None, "ret_1d_adj_total": None},
            {"symbol": "BBB", "date": "2024-01-03", "close_adj_split": 49.0, "close_adj_total": 49.0, "ret_1d_adj_split": -0.0101010101, "ret_1d_adj_total": -0.0101010101},
        ]
    )
    adjusted["date"] = pd.to_datetime(adjusted["date"])

    raw_path = _write_csv(raw, tmp_path / "prices_raw.csv")
    adjusted_path = _write_csv(adjusted, tmp_path / "prices_adjusted.csv")
    calendar_path = _write_csv(calendar, tmp_path / "calendar.csv")
    config_path = _write_json(
        tmp_path / "qc_config.json",
        dataclasses.asdict(
            _cfg(
                module,
                require_adjusted_consistency_check=True,
                pct_fail_row_fail_threshold=0.001,
                pct_symbols_bad_coverage_fail_threshold=0.9,
                pct_raw_adjusted_inconsistent_fail_threshold=0.9,
            )
        ),
    )
    output_dir = tmp_path / "out"

    def _write_parquet_as_csv(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    monkeypatch.setattr(module, "write_parquet", _write_parquet_as_csv)

    artifacts = module.run_qc_prices(
        prices_raw_path=raw_path,
        prices_adjusted_path=adjusted_path,
        calendar_path=calendar_path,
        config_path=config_path,
        output_dir=output_dir,
        run_id="qc_prices_contract",
        as_of_ts_utc="2024-01-03T23:59:59Z",
    )
    module.persist_artifacts(artifacts, output_dir)

    return {
        "module": module,
        "raw": raw,
        "adjusted": adjusted,
        "calendar": calendar,
        "config_path": config_path,
        "output_dir": output_dir,
        "artifacts": artifacts,
    }


# -----------------------------------------------------------------------------
# Contract tests
# -----------------------------------------------------------------------------


def test_qc_summary_contains_required_fields(contract_case):
    summary = contract_case["artifacts"].summary
    assert _required_summary_keys().issubset(summary.keys())
    assert summary["gate"] in {"pass", "warn", "fail"}



def test_qc_symbol_level_contains_required_columns(contract_case):
    symbol_level = contract_case["artifacts"].symbol_level
    assert _required_symbol_level_columns().issubset(symbol_level.columns)



def test_qc_row_level_contains_required_columns(contract_case):
    row_level = contract_case["artifacts"].row_level
    assert _required_row_level_columns().issubset(row_level.columns)



def test_qc_failures_is_subset_of_material_severities(contract_case):
    artifacts = contract_case["artifacts"]
    failures = artifacts.failures
    row_level = artifacts.row_level

    assert not failures.empty
    assert set(failures["severity"].dropna().astype(str)).issubset({"FAIL_ROW", "FAIL_STRUCTURAL"})

    material = row_level.loc[row_level["severity"].isin(["FAIL_ROW", "FAIL_STRUCTURAL"])].copy()
    assert _semantic_sort(failures).equals(_semantic_sort(material))



def test_manifest_contains_required_snapshots_and_metadata(contract_case):
    manifest = contract_case["artifacts"].manifest
    assert _required_manifest_keys().issubset(manifest.keys())
    assert manifest["run_id"] == "qc_prices_contract"
    assert manifest["gate"] in {"pass", "warn", "fail"}
    assert isinstance(manifest["outputs"], dict)
    assert {"qc_summary", "qc_symbol_level", "qc_row_level", "qc_failures", "manifest"}.issubset(manifest["outputs"].keys())



def test_gate_domain_is_pass_warn_fail(contract_case):
    gate = contract_case["artifacts"].summary["gate"]
    assert gate in {"pass", "warn", "fail"}



def test_failure_warn_info_counts_are_coherent(contract_case):
    artifacts = contract_case["artifacts"]
    row_level = artifacts.row_level
    summary = artifacts.summary

    fail_row_count = int((row_level["severity"] == "FAIL_ROW").sum())
    warn_count = int((row_level["severity"] == "WARN_SYMBOL").sum())
    info_count = int((row_level["severity"] == "INFO").sum())

    assert summary["fail_count"] == fail_row_count + summary["structural_fail_count"]
    assert summary["warn_count"] == warn_count
    assert summary["info_count"] == info_count



def test_outputs_are_sorted_deterministically(contract_case):
    artifacts = contract_case["artifacts"]

    row_expected = artifacts.row_level.sort_values(
        ["severity", "symbol", "date", "check_name"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    assert artifacts.row_level.reset_index(drop=True).equals(row_expected)

    symbol_expected = artifacts.symbol_level.sort_values(
        ["severity_max", "symbol"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)
    assert artifacts.symbol_level.reset_index(drop=True).equals(symbol_expected)

    failures_expected = artifacts.failures.sort_values(
        ["severity", "symbol", "date", "check_name"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    assert artifacts.failures.reset_index(drop=True).equals(failures_expected)



def test_persisted_artifacts_exist_and_manifest_roundtrips(contract_case):
    output_dir = contract_case["output_dir"]
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "qc_summary.json"
    row_path = output_dir / "qc_row_level.parquet"
    symbol_path = output_dir / "qc_symbol_level.parquet"
    failures_path = output_dir / "qc_failures.parquet"

    for path in [manifest_path, summary_path, row_path, symbol_path, failures_path]:
        assert path.exists(), f"Expected persisted artifact: {path}"

    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted_manifest == contract_case["artifacts"].manifest

    persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted_summary == contract_case["artifacts"].summary



def test_persisted_tables_match_returned_tables_semantically(contract_case):
    output_dir = contract_case["output_dir"]
    artifacts = contract_case["artifacts"]

    persisted_row = _read_persisted_table(output_dir / "qc_row_level.parquet")
    persisted_symbol = _read_persisted_table(output_dir / "qc_symbol_level.parquet")
    persisted_failures = _read_persisted_table(output_dir / "qc_failures.parquet")

    assert _semantic_sort(persisted_row).equals(_semantic_sort(artifacts.row_level))
    assert _semantic_sort(persisted_symbol).equals(_semantic_sort(artifacts.symbol_level))
    assert _semantic_sort(persisted_failures).equals(_semantic_sort(artifacts.failures))



def test_manifest_output_paths_point_to_materialized_files(contract_case):
    manifest = contract_case["artifacts"].manifest
    for _, path_str in manifest["outputs"].items():
        assert Path(path_str).exists()
