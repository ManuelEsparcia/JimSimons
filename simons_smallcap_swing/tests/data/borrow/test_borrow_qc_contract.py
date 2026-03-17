from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd
import pytest


EXPECTED_DAILY_COLUMNS = {
    "date",
    "coverage_rate",
    "coverage_smallcap",
    "jump_share",
    "stale_share",
    "underblocking_rate",
    "overblocking_rate",
    "avg_borrow_fee_daily",
    "p95_borrow_fee_daily",
    "blocked_share",
    "score_coverage",
    "score_stability",
    "score_coherence",
    "score_freshness",
    "score_total",
    "gate",
    "config_version",
}

EXPECTED_FAILURE_COLUMNS = {
    "date",
    "symbol",
    "check_type",
    "severity",
    "observed_value",
    "threshold_violated",
    "classification",
    "config_version",
    "message",
}

EXPECTED_SUMMARY_KEYS = {
    "module",
    "run_id",
    "config_version",
    "window_start",
    "window_end",
    "n_dates",
    "n_rows_joined",
    "pass_pct",
    "warn_pct",
    "fail_pct",
    "hard_fail_count",
    "avg_coverage_rate",
    "avg_jump_share",
    "avg_stale_share",
    "avg_underblocking_rate",
    "avg_overblocking_rate",
    "avg_score_total",
    "avg_score_coverage",
    "avg_score_stability",
    "avg_score_coherence",
    "avg_score_freshness",
    "borrow_proxy_versions",
    "locate_versions",
    "column_contract_hash",
    "market_calendar_name",
    "asof_timestamp",
    "validation",
}

ALLOWED_SEVERITIES = {"FAIL_STRUCTURAL", "FAIL_NUMERIC", "WARN", "INFO"}
ALLOWED_GATES = {"pass", "warn", "fail"}


# -----------------------------------------------------------------------------
# Import adapter
# -----------------------------------------------------------------------------


def _load_borrow_qc_module():
    module_names = [
        "simons_smallcap_swing.data.borrow.borrow_qc",
        "data.borrow.borrow_qc",
        "borrow_qc",
    ]
    for name in module_names:
        try:
            return importlib.import_module(name)
        except Exception:
            pass

    here = Path(__file__).resolve()
    candidate_paths = [
        here.parents[3] / "simons_smallcap_swing" / "data" / "borrow" / "borrow_qc.py",
        here.parents[2] / "data" / "borrow" / "borrow_qc.py",
        here.parents[1] / "borrow_qc.py",
        Path("/mnt/data/borrow_qc.py"),
    ]
    for path in candidate_paths:
        if path.exists():
            spec = importlib.util.spec_from_file_location("borrow_qc_under_test_contract", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module

    raise ModuleNotFoundError(
        "Could not import borrow_qc.py. Expected it at "
        "simons_smallcap_swing.data.borrow.borrow_qc or in a nearby repo path."
    )


@pytest.fixture(scope="session")
def borrow_qc_module():
    return _load_borrow_qc_module()


# -----------------------------------------------------------------------------
# Synthetic builders
# -----------------------------------------------------------------------------


def _borrow_row(
    *,
    symbol: str,
    date: str,
    fee_annual: float = 0.05,
    fee_daily: float = 0.05 / 252.0,
    alpha: float = 0.80,
    htb: bool = False,
    tier: str = "easy",
    quality: str = "high",
    source: str = "direct_lending_feed",
    stress: float = 0.10,
    jump: bool = False,
    stale: bool = False,
    fallback: bool = False,
    config_version: str = "proxy_v1",
) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "date": pd.Timestamp(date),
        "borrow_fee_annual": fee_annual,
        "borrow_fee_daily": fee_daily,
        "borrow_availability_score": alpha,
        "htb_flag": htb,
        "borrow_tier": tier,
        "proxy_quality": quality,
        "proxy_source": source,
        "stress_score": stress,
        "jump_flag": jump,
        "stale_input_flag": stale,
        "fallback_flag": fallback,
        "config_version": config_version,
    }



def _locate_row(
    *,
    symbol: str,
    date: str,
    allowed: bool = True,
    blocked: bool | None = None,
    decision: str | None = None,
    override_flag: bool = False,
    override_reason: str = "",
    config_version: str = "locate_v1",
) -> Dict[str, Any]:
    if blocked is None:
        blocked = not allowed
    if decision is None:
        decision = "allow" if allowed else "reject"
    return {
        "symbol": symbol,
        "date": pd.Timestamp(date),
        "locate_allowed": allowed,
        "locate_blocked": blocked,
        "locate_decision": decision,
        "override_flag": override_flag,
        "override_reason": override_reason,
        "config_version": config_version,
    }



def _universe_row(
    *,
    symbol: str,
    date: str,
    size_bucket: str = "small",
    liq_bucket: str = "medium",
    short_flag: bool = True,
    membership_state: str = "active",
    is_member: bool = True,
) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "date": pd.Timestamp(date),
        "size_bucket": size_bucket,
        "liq_bucket": liq_bucket,
        "short_subuniverse_flag": short_flag,
        "membership_state": membership_state,
        "is_member": is_member,
    }


@pytest.fixture()
def clean_payloads() -> Dict[str, pd.DataFrame]:
    borrow = pd.DataFrame(
        [
            _borrow_row(symbol="AAA", date="2026-01-02", fee_annual=0.03, fee_daily=0.03 / 252.0, alpha=0.92),
            _borrow_row(symbol="AAA", date="2026-01-05", fee_annual=0.05, fee_daily=0.05 / 252.0, alpha=0.82, tier="medium", quality="medium"),
        ]
    )
    locate = pd.DataFrame(
        [
            _locate_row(symbol="AAA", date="2026-01-02", allowed=True),
            _locate_row(symbol="AAA", date="2026-01-05", allowed=True),
        ]
    )
    universe = pd.DataFrame(
        [
            _universe_row(symbol="AAA", date="2026-01-02", size_bucket="small"),
            _universe_row(symbol="AAA", date="2026-01-05", size_bucket="small"),
        ]
    )
    return {"borrow": borrow, "locate": locate, "universe": universe}


@pytest.fixture()
def failing_payloads() -> Dict[str, pd.DataFrame]:
    borrow = pd.DataFrame(
        [
            _borrow_row(symbol="AAA", date="2026-01-02", fee_annual=-0.01, fee_daily=-0.01 / 252.0, alpha=0.92),
            _borrow_row(symbol="AAA", date="2026-01-05", fee_annual=1.50, fee_daily=1.50 / 252.0, alpha=0.02, htb=True, tier="blocked"),
        ]
    )
    locate = pd.DataFrame(
        [
            _locate_row(symbol="AAA", date="2026-01-02", allowed=True),
            _locate_row(symbol="AAA", date="2026-01-05", allowed=True, blocked=False, decision="allow"),
        ]
    )
    universe = pd.DataFrame(
        [
            _universe_row(symbol="AAA", date="2026-01-02", size_bucket="small"),
            _universe_row(symbol="AAA", date="2026-01-05", size_bucket="small"),
        ]
    )
    return {"borrow": borrow, "locate": locate, "universe": universe}


# -----------------------------------------------------------------------------
# Invocation helpers
# -----------------------------------------------------------------------------


def _run_qc(module: Any, payloads: Mapping[str, pd.DataFrame], run_id: str = "pytest_borrow_qc_contract"):
    return module.build_borrow_qc(
        borrow_proxy=payloads["borrow"],
        locate_filter=payloads["locate"],
        universe=payloads["universe"],
        cfg=module.BorrowQCConfig(),
        run_id=run_id,
        asof_timestamp="2026-03-15T12:00:00Z",
    )


# -----------------------------------------------------------------------------
# Contract tests
# -----------------------------------------------------------------------------


def test_qc_daily_contains_required_columns_and_gate_domain(borrow_qc_module, clean_payloads):
    daily, failures, summary, panel = _run_qc(borrow_qc_module, clean_payloads)

    assert not daily.empty
    assert EXPECTED_DAILY_COLUMNS.issubset(set(daily.columns))
    assert set(daily["gate"].dropna().astype(str).unique()).issubset(ALLOWED_GATES)
    assert pd.api.types.is_datetime64_any_dtype(daily["date"])
    assert daily["config_version"].astype(str).eq(summary["config_version"]).all()

    for score_col in ["score_coverage", "score_stability", "score_coherence", "score_freshness", "score_total"]:
        vals = pd.to_numeric(daily[score_col], errors="coerce")
        assert vals.notna().all(), f"{score_col} should be populated"
        assert ((vals >= 0.0) & (vals <= 100.0)).all(), f"{score_col} must live in [0, 100]"

    assert isinstance(failures, pd.DataFrame)
    assert isinstance(panel, pd.DataFrame)



def test_qc_failures_contains_required_columns_and_valid_severities(borrow_qc_module, failing_payloads):
    daily, failures, summary, panel = _run_qc(borrow_qc_module, failing_payloads)

    assert not failures.empty
    assert EXPECTED_FAILURE_COLUMNS.issubset(set(failures.columns))
    assert set(failures["severity"].dropna().astype(str).unique()).issubset(ALLOWED_SEVERITIES)
    assert failures["config_version"].astype(str).eq(summary["config_version"]).all()
    assert "negative_fee" in set(failures["check_type"].astype(str))
    assert daily.iloc[0]["gate"] == "fail"



def test_qc_summary_contains_expected_keys_and_consistent_rates(borrow_qc_module, failing_payloads):
    daily, failures, summary, panel = _run_qc(borrow_qc_module, failing_payloads, run_id="pytest_summary_contract")

    assert EXPECTED_SUMMARY_KEYS.issubset(set(summary.keys()))
    assert summary["module"] == "data/borrow/borrow_qc.py"
    assert summary["run_id"] == "pytest_summary_contract"
    assert summary["n_dates"] == int(daily["date"].nunique())
    assert summary["n_rows_joined"] == int(panel.shape[0])
    assert summary["borrow_proxy_versions"] == ["proxy_v1"]
    assert summary["locate_versions"] == ["locate_v1"]
    assert isinstance(summary["validation"], dict)
    assert {"structural_failures", "numeric_failures", "warns", "infos"}.issubset(summary["validation"].keys())

    total_rate = float(summary["pass_pct"] + summary["warn_pct"] + summary["fail_pct"])
    assert total_rate == pytest.approx(1.0, abs=1e-12)

    expected_hash = borrow_qc_module.contract_hash(
        borrow_qc_module.BORROW_REQUIRED_COLUMNS.union({"locate_allowed", "symbol", "date"})
    )
    assert summary["column_contract_hash"] == expected_hash



def test_qc_daily_contract_hash_and_gate_consistency(borrow_qc_module, clean_payloads):
    daily, failures, summary, panel = _run_qc(borrow_qc_module, clean_payloads, run_id="pytest_gate_contract")

    gate = str(daily.iloc[0]["gate"])
    hard_fail_count = int(summary["hard_fail_count"])
    if gate == "pass":
        assert hard_fail_count == 0
    elif gate == "fail":
        assert hard_fail_count >= 1 or not failures.empty

    assert summary["market_calendar_name"] == borrow_qc_module.BorrowQCConfig().market_calendar_name
    assert summary["column_contract_hash"] == borrow_qc_module.contract_hash(
        borrow_qc_module.BORROW_REQUIRED_COLUMNS.union({"locate_allowed", "symbol", "date"})
    )



def test_qc_failure_rows_are_sorted_deterministically(borrow_qc_module, failing_payloads):
    _, failures, _, _ = _run_qc(borrow_qc_module, failing_payloads)

    expected = failures.sort_values(["date", "severity", "check_type", "symbol"], na_position="last").reset_index(drop=True)
    pd.testing.assert_frame_equal(failures.reset_index(drop=True), expected)



def test_cli_manifest_contains_run_metadata_and_contract_hash(tmp_path, monkeypatch, borrow_qc_module, failing_payloads):
    borrow_path = tmp_path / "borrow.csv"
    locate_path = tmp_path / "locate.csv"
    universe_path = tmp_path / "universe.csv"
    outdir = tmp_path / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    failing_payloads["borrow"].to_csv(borrow_path, index=False)
    failing_payloads["locate"].to_csv(locate_path, index=False)
    failing_payloads["universe"].to_csv(universe_path, index=False)

    written_parquet_targets: list[str] = []

    def _fake_write_parquet(df: pd.DataFrame, path: Path) -> None:
        written_parquet_targets.append(path.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Keep the expected path name while avoiding a parquet engine dependency.
        df.to_csv(path.with_suffix(path.suffix + ".csv"), index=False)

    monkeypatch.setattr(borrow_qc_module, "write_parquet", _fake_write_parquet)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "borrow_qc.py",
            "--borrow-proxy-path",
            str(borrow_path),
            "--locate-filter-path",
            str(locate_path),
            "--universe-path",
            str(universe_path),
            "--output-dir",
            str(outdir),
            "--run-id",
            "pytest_cli_contract",
            "--asof-timestamp",
            "2026-03-15T12:00:00Z",
        ],
    )

    borrow_qc_module.main()

    manifest_path = outdir / "manifest.json"
    summary_path = outdir / "borrow_qc_summary.json"
    assert manifest_path.exists()
    assert summary_path.exists()
    assert set(written_parquet_targets) == {
        "borrow_qc_daily.parquet",
        "borrow_qc_failures.parquet",
        "borrow_qc_joined_panel.parquet",
    }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert manifest["module"] == "data/borrow/borrow_qc.py"
    assert manifest["run_id"] == "pytest_cli_contract"
    assert manifest["config_version"] == borrow_qc_module.BorrowQCConfig().config_version
    assert manifest["inputs"]["borrow_proxy_path"] == str(borrow_path)
    assert manifest["outputs"]["manifest"] == str(manifest_path)
    assert manifest["outputs"]["summary"] == str(summary_path)
    assert manifest["column_contract_hash"] == summary["column_contract_hash"]
    assert manifest["n_dates"] == summary["n_dates"]
    assert manifest["n_rows_joined"] == summary["n_rows_joined"]
    assert manifest["market_calendar_name"] == summary["market_calendar_name"]



def test_summary_validation_counts_match_failure_table(borrow_qc_module, failing_payloads):
    _, failures, summary, _ = _run_qc(borrow_qc_module, failing_payloads)

    validation = summary["validation"]
    assert int(validation["structural_failures"]) == int(failures["severity"].eq("FAIL_STRUCTURAL").sum())
    assert int(validation["numeric_failures"]) == int(failures["severity"].eq("FAIL_NUMERIC").sum())
    assert int(validation["warns"]) == int(failures["severity"].eq("WARN").sum())
    assert int(validation["infos"]) == int(failures["severity"].eq("INFO").sum())
