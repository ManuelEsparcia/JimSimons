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
            module_name = f"_qc_prices_test_{abs(hash(str(path)))}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module

    raise ModuleNotFoundError(
        "Could not import qc_prices.py. Expected it at "
        "simons_smallcap_swing.data.price.qc_prices or in a nearby repo path."
    )


@pytest.fixture(scope="session")
def qc_prices_module():
    return _load_qc_prices_module()


# -----------------------------------------------------------------------------
# Helpers / fixtures
# -----------------------------------------------------------------------------


def _cfg(module, **overrides: Any):
    base = dataclasses.asdict(module.QCConfig())
    base.update(overrides)
    return module.QCConfig(**base)



def _calendar(*dates: str) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(list(dates))})



def _raw_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df



def _codes(df: pd.DataFrame) -> set[str]:
    if df is None or df.empty:
        return set()
    return set(df["failure_code"].dropna().astype(str))



def _severities(df: pd.DataFrame) -> set[str]:
    if df is None or df.empty:
        return set()
    return set(df["severity"].dropna().astype(str))



def _write_config(tmp_path: Path, module, **overrides: Any) -> Path:
    cfg = dataclasses.asdict(_cfg(module, **overrides))
    path = tmp_path / "qc_config.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path



def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def clean_calendar_df():
    return _calendar("2024-01-02", "2024-01-03", "2024-01-04")


@pytest.fixture()
def clean_raw_df():
    return _raw_rows(
        [
            {"symbol": "AAA", "date": "2024-01-02", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0, "source_provider": "demo"},
            {"symbol": "AAA", "date": "2024-01-03", "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.0, "volume": 1200.0, "source_provider": "demo"},
            {"symbol": "AAA", "date": "2024-01-04", "open": 102.0, "high": 103.0, "low": 101.0, "close": 102.0, "volume": 1250.0, "source_provider": "demo"},
            {"symbol": "BBB", "date": "2024-01-02", "open": 50.0, "high": 51.0, "low": 49.0, "close": 50.0, "volume": 800.0, "source_provider": "demo"},
            {"symbol": "BBB", "date": "2024-01-03", "open": 51.0, "high": 52.0, "low": 50.0, "close": 51.0, "volume": 850.0, "source_provider": "demo"},
            {"symbol": "BBB", "date": "2024-01-04", "open": 52.0, "high": 53.0, "low": 51.0, "close": 52.0, "volume": 900.0, "source_provider": "demo"},
        ]
    )


@pytest.fixture()
def clean_adjusted_df(clean_raw_df):
    out = clean_raw_df[["symbol", "date", "close"]].copy()
    out["close_adj_split"] = out["close"]
    out["close_adj_total"] = out["close"]
    out["ret_1d_adj_split"] = out.groupby("symbol", observed=True)["close_adj_split"].pct_change()
    out["ret_1d_adj_total"] = out.groupby("symbol", observed=True)["close_adj_total"].pct_change()
    return out.drop(columns=["close"])


# -----------------------------------------------------------------------------
# Unit tests
# -----------------------------------------------------------------------------


def test_validate_raw_schema_missing_required_columns_triggers_fail_structural(qc_prices_module):
    raw = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": [pd.Timestamp("2024-01-02")],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            # volume intentionally missing
            "source_provider": ["demo"],
        }
    )

    findings = qc_prices_module.validate_raw_schema(raw)
    merged = qc_prices_module.concat_findings(findings)

    assert qc_prices_module.FailureCode.MISSING_REQUIRED_COLUMN.value in _codes(merged)
    assert _severities(merged) == {qc_prices_module.Severity.FAIL_STRUCTURAL.value}
    assert "volume" in set(merged["observed_value"].astype(str))



def test_validate_raw_schema_duplicate_pk_triggers_fail_structural(qc_prices_module):
    raw = _raw_rows(
        [
            {"symbol": "AAA", "date": "2024-01-02", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0, "source_provider": "demo"},
            {"symbol": "AAA", "date": "2024-01-02", "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.0, "volume": 1100.0, "source_provider": "demo"},
        ]
    )

    findings = qc_prices_module.validate_raw_schema(raw)
    merged = qc_prices_module.concat_findings(findings)

    assert qc_prices_module.FailureCode.DUPLICATE_PK.value in _codes(merged)
    assert _severities(merged) == {qc_prices_module.Severity.FAIL_STRUCTURAL.value}



def test_check_intrabar_geometry_triggers_fail_row_for_impossible_ohlc(qc_prices_module, clean_calendar_df):
    raw = _raw_rows(
        [
            {"symbol": "AAA", "date": "2024-01-02", "open": 100.0, "high": 99.0, "low": 98.0, "close": 100.0, "volume": 1000.0, "source_provider": "demo"},
        ]
    )
    prepared = qc_prices_module.prepare_raw(raw, clean_calendar_df)

    findings = qc_prices_module.concat_findings(qc_prices_module.check_intrabar_geometry(prepared, _cfg(qc_prices_module)))

    assert qc_prices_module.FailureCode.OHLC_INVALID.value in _codes(findings)
    assert qc_prices_module.Severity.FAIL_ROW.value in _severities(findings)



def test_check_intrabar_geometry_flags_negative_volume_and_penny_price(qc_prices_module, clean_calendar_df):
    raw = _raw_rows(
        [
            {"symbol": "AAA", "date": "2024-01-02", "open": 0.60, "high": 0.70, "low": 0.50, "close": 0.60, "volume": -10.0, "source_provider": "demo"},
        ]
    )
    prepared = qc_prices_module.prepare_raw(raw, clean_calendar_df)

    findings = qc_prices_module.concat_findings(qc_prices_module.check_intrabar_geometry(prepared, _cfg(qc_prices_module, penny_price_threshold=1.0)))

    codes = _codes(findings)
    assert qc_prices_module.FailureCode.NEGATIVE_VOLUME.value in codes
    assert qc_prices_module.FailureCode.PENNY_PRICE.value in codes
    neg = findings.loc[findings["failure_code"] == qc_prices_module.FailureCode.NEGATIVE_VOLUME.value]
    penny = findings.loc[findings["failure_code"] == qc_prices_module.FailureCode.PENNY_PRICE.value]
    assert set(neg["severity"]) == {qc_prices_module.Severity.FAIL_ROW.value}
    assert set(penny["severity"]) == {qc_prices_module.Severity.INFO.value}



def test_check_temporal_coverage_flags_warn_symbol_for_missing_sessions(qc_prices_module):
    calendar = _calendar("2024-01-02", "2024-01-03", "2024-01-04")
    raw = _raw_rows(
        [
            {"symbol": "AAA", "date": "2024-01-02", "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 100.0, "source_provider": "demo"},
            {"symbol": "AAA", "date": "2024-01-04", "open": 11.0, "high": 12.0, "low": 10.0, "close": 11.0, "volume": 100.0, "source_provider": "demo"},
        ]
    )
    prepared = qc_prices_module.prepare_raw(raw, calendar)
    cfg = _cfg(qc_prices_module, symbol_bad_coverage_threshold=0.10, min_symbol_rows_for_coverage=2)

    symbol_cov, findings_list = qc_prices_module.check_temporal_coverage(prepared, calendar, cfg)
    findings = qc_prices_module.concat_findings(findings_list)

    assert bool(symbol_cov.loc[0, "coverage_ok"]) is False
    assert symbol_cov.loc[0, "n_missing_sessions"] == 1
    assert qc_prices_module.FailureCode.BAD_COVERAGE.value in _codes(findings)
    assert qc_prices_module.Severity.WARN_SYMBOL.value in _severities(findings)



def test_check_extreme_return_plausible_is_info_not_fail(qc_prices_module):
    calendar = _calendar("2024-01-02", "2024-01-03")
    raw = _raw_rows(
        [
            {"symbol": "AAA", "date": "2024-01-02", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0, "source_provider": "demo"},
            {"symbol": "AAA", "date": "2024-01-03", "open": 200.0, "high": 205.0, "low": 195.0, "close": 200.0, "volume": 1500.0, "source_provider": "demo"},
        ]
    )
    prepared = qc_prices_module.prepare_raw(raw, calendar)

    findings_list, enriched = qc_prices_module.check_extreme_returns_and_adjusted(
        prepared,
        adj=None,
        config=_cfg(qc_prices_module, extreme_return_threshold=0.40),
        existing_row_fail_mask=pd.Series(False, index=prepared.index),
    )
    findings = qc_prices_module.concat_findings(findings_list)

    assert enriched["extreme_return_flag"].sum() == 1
    plausible = findings.loc[findings["failure_code"] == qc_prices_module.FailureCode.EXTREME_RETURN_PLAUSIBLE.value]
    assert len(plausible) == 1
    assert set(plausible["severity"]) == {qc_prices_module.Severity.INFO.value}
    assert qc_prices_module.FailureCode.EXTREME_RETURN_SUSPECT.value not in _codes(findings)



def test_check_extreme_return_plus_raw_adjusted_inconsistency_triggers_fail_row(qc_prices_module):
    calendar = _calendar("2024-01-02", "2024-01-03")
    raw = _raw_rows(
        [
            {"symbol": "AAA", "date": "2024-01-02", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0, "source_provider": "demo"},
            {"symbol": "AAA", "date": "2024-01-03", "open": 200.0, "high": 205.0, "low": 195.0, "close": 200.0, "volume": 1500.0, "source_provider": "demo"},
        ]
    )
    prepared = qc_prices_module.prepare_raw(raw, calendar)
    adj = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            # No close_adj_split on purpose, so split_event_candidate stays False.
            "ret_1d_adj_split": [pd.NA, -0.10],
        }
    )

    findings_list, enriched = qc_prices_module.check_extreme_returns_and_adjusted(
        prepared,
        adj=adj,
        config=_cfg(qc_prices_module, extreme_return_threshold=0.40, raw_adjusted_mag_ratio_threshold=5.0),
        existing_row_fail_mask=pd.Series(False, index=prepared.index),
    )
    findings = qc_prices_module.concat_findings(findings_list)

    assert enriched["raw_adjusted_inconsistent_flag"].sum() == 1
    inconsistent = findings.loc[findings["failure_code"] == qc_prices_module.FailureCode.RAW_ADJUSTED_INCONSISTENT.value]
    assert len(inconsistent) == 1
    assert set(inconsistent["severity"]) == {qc_prices_module.Severity.FAIL_ROW.value}



def test_determine_gate_fails_when_pct_fail_row_exceeds_threshold(qc_prices_module):
    summary = {
        "pct_fail_row": 0.20,
        "pct_symbols_bad_coverage": 0.0,
        "pct_raw_adjusted_inconsistent": 0.0,
        "warn_count": 0,
        "info_count": 0,
    }
    gate = qc_prices_module.determine_gate(summary, structural_fail_count=0, config=_cfg(qc_prices_module, pct_fail_row_fail_threshold=0.05))
    assert gate == qc_prices_module.Gate.FAIL



def test_run_qc_prices_clean_case_is_pass(qc_prices_module, tmp_path, clean_raw_df, clean_adjusted_df, clean_calendar_df):
    raw_path = _write_csv(clean_raw_df, tmp_path / "raw.csv")
    adj_path = _write_csv(clean_adjusted_df, tmp_path / "adj.csv")
    cal_path = _write_csv(clean_calendar_df, tmp_path / "calendar.csv")
    cfg_path = _write_config(tmp_path, qc_prices_module)

    artifacts = qc_prices_module.run_qc_prices(
        prices_raw_path=raw_path,
        prices_adjusted_path=adj_path,
        calendar_path=cal_path,
        config_path=cfg_path,
        output_dir=tmp_path / "out",
        run_id="qc_clean",
        as_of_ts_utc="2026-03-16T10:00:00Z",
    )

    assert artifacts.summary["gate"] == qc_prices_module.Gate.PASS.value
    assert artifacts.failures.empty
    assert len(artifacts.symbol_level) == 2
    assert set(artifacts.symbol_level["symbol"]) == {"AAA", "BBB"}



def test_run_qc_prices_too_many_bad_symbols_force_gate_fail(qc_prices_module, tmp_path, clean_adjusted_df, clean_calendar_df):
    raw = _raw_rows(
        [
            # AAA misses one session between observed start/end -> bad coverage
            {"symbol": "AAA", "date": "2024-01-02", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0, "source_provider": "demo"},
            {"symbol": "AAA", "date": "2024-01-04", "open": 102.0, "high": 103.0, "low": 101.0, "close": 102.0, "volume": 1200.0, "source_provider": "demo"},
            # BBB is clean
            {"symbol": "BBB", "date": "2024-01-02", "open": 50.0, "high": 51.0, "low": 49.0, "close": 50.0, "volume": 800.0, "source_provider": "demo"},
            {"symbol": "BBB", "date": "2024-01-03", "open": 51.0, "high": 52.0, "low": 50.0, "close": 51.0, "volume": 850.0, "source_provider": "demo"},
            {"symbol": "BBB", "date": "2024-01-04", "open": 52.0, "high": 53.0, "low": 51.0, "close": 52.0, "volume": 900.0, "source_provider": "demo"},
        ]
    )
    # adjusted only for existing rows; enough for the check block.
    adj = raw[["symbol", "date", "close"]].copy()
    adj["close_adj_split"] = adj["close"]
    adj["close_adj_total"] = adj["close"]
    adj["ret_1d_adj_split"] = adj.groupby("symbol", observed=True)["close_adj_split"].pct_change()
    adj["ret_1d_adj_total"] = adj.groupby("symbol", observed=True)["close_adj_total"].pct_change()
    adj = adj.drop(columns=["close"])

    raw_path = _write_csv(raw, tmp_path / "raw.csv")
    adj_path = _write_csv(adj, tmp_path / "adj.csv")
    cal_path = _write_csv(clean_calendar_df, tmp_path / "calendar.csv")
    cfg_path = _write_config(
        tmp_path,
        qc_prices_module,
        symbol_bad_coverage_threshold=0.10,
        pct_symbols_bad_coverage_fail_threshold=0.10,
        min_symbol_rows_for_coverage=2,
    )

    artifacts = qc_prices_module.run_qc_prices(
        prices_raw_path=raw_path,
        prices_adjusted_path=adj_path,
        calendar_path=cal_path,
        config_path=cfg_path,
        output_dir=tmp_path / "out",
        run_id="qc_fail_coverage",
        as_of_ts_utc="2026-03-16T10:00:00Z",
    )

    assert artifacts.summary["pct_symbols_bad_coverage"] == pytest.approx(0.5)
    assert artifacts.summary["gate"] == qc_prices_module.Gate.FAIL.value
    assert qc_prices_module.FailureCode.BAD_COVERAGE.value in _codes(artifacts.row_level)
