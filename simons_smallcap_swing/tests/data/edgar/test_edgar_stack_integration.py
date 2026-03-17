from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# Import adapters
# -----------------------------------------------------------------------------


def _load_module(module_basename: str):
    module_names = [
        f"simons_smallcap_swing.data.edgar.{module_basename}",
        f"data.edgar.{module_basename}",
        module_basename,
    ]
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue

    here = Path(__file__).resolve()
    candidate_paths = [
        here.parents[3] / "simons_smallcap_swing" / "data" / "edgar" / f"{module_basename}.py",
        here.parents[2] / "data" / "edgar" / f"{module_basename}.py",
        here.parents[1] / f"{module_basename}.py",
        Path(f"/mnt/data/{module_basename}.py"),
    ]
    for path in candidate_paths:
        if path.exists():
            spec = importlib.util.spec_from_file_location(f"{module_basename}_under_test", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module

    raise ModuleNotFoundError(f"Could not import {module_basename}.py")


@pytest.fixture(scope="session")
def ticker_cik_module():
    return _load_module("ticker_cik")


@pytest.fixture(scope="session")
def parse_xbrl_module():
    return _load_module("parse_xbrl")


@pytest.fixture(scope="session")
def filings_flags_module():
    return _load_module("filings_flags")


@pytest.fixture(scope="session")
def point_in_time_module():
    return _load_module("point_in_time")


@pytest.fixture(scope="session")
def edgar_qc_module():
    return _load_module("edgar_qc")


@pytest.fixture()
def patch_edgar_qc_write_parquet(monkeypatch: pytest.MonkeyPatch, edgar_qc_module):
    def _writer(path: Path, df: pd.DataFrame, *, index: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=index)

    monkeypatch.setattr(edgar_qc_module, "write_parquet", _writer)
    return _writer


# -----------------------------------------------------------------------------
# Synthetic data builders
# -----------------------------------------------------------------------------


def _sec_row(symbol: str, cik: str | int, *, issuer_name: str, effective_from: str = "2020-01-01") -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "cik": cik,
        "issuer_name": issuer_name,
        "exchange": "NASDAQ",
        "share_class": "COMMON",
        "effective_from": effective_from,
        "effective_to": None,
        "ingest_ts_utc": "2026-03-01T00:00:00Z",
    }


class SymbolSpec(Dict[str, Any]):
    pass


def _submission_payload(spec: Mapping[str, Any]) -> Dict[str, Any]:
    cik = str(spec["cik"]).zfill(10)
    symbol = str(spec["symbol"]).upper()
    accn = str(spec.get("accession_number", f"{cik}-26-000001"))
    filing_date = str(spec.get("filing_date", "2026-02-01"))
    accepted = str(spec.get("acceptance_datetime", "2026-02-01T14:30:00Z"))
    form_type = str(spec.get("form_type", "10-K"))
    return {
        "cik": cik,
        "name": str(spec.get("issuer_name", symbol)),
        "tickers": [symbol],
        "exchanges": [str(spec.get("exchange", "NASDAQ"))],
        "filings": {
            "recent": {
                "accessionNumber": [accn],
                "form": [form_type],
                "filingDate": [filing_date],
                "acceptanceDateTime": [accepted],
            }
        },
    }


def _companyfacts_payload(spec: Mapping[str, Any]) -> Dict[str, Any]:
    cik = str(spec["cik"]).zfill(10)
    filed = str(spec.get("filing_date", "2026-02-01"))
    accepted = str(spec.get("acceptance_datetime", "2026-02-01T14:30:00Z"))
    accn = str(spec.get("accession_number", f"{cik}-26-000001"))
    fy = int(spec.get("fy", 2025))
    facts: Dict[str, Any] = {
        "RevenueFromContractWithCustomerExcludingAssessedTax": {
            "label": "Revenue",
            "description": "Revenue",
            "units": {
                "USD": [
                    {
                        "val": float(spec.get("revenue", 250.0)),
                        "start": str(spec.get("revenue_period_start", "2025-01-01")),
                        "end": str(spec.get("period_end", "2025-12-31")),
                        "filed": filed,
                        "accepted": accepted,
                        "fy": fy,
                        "fp": str(spec.get("revenue_fp", "FY")),
                        "frame": str(spec.get("revenue_frame", "CY2025")),
                        "form": str(spec.get("form_type", "10-K")),
                        "accn": accn,
                    }
                ]
            },
        }
    }
    if bool(spec.get("include_assets", True)):
        facts["Assets"] = {
            "label": "Assets",
            "description": "Assets",
            "units": {
                "USD": [
                    {
                        "val": float(spec.get("assets", 500.0)),
                        "end": str(spec.get("period_end", "2025-12-31")),
                        "filed": filed,
                        "accepted": accepted,
                        "fy": fy,
                        "fp": str(spec.get("assets_fp", "FY")),
                        "form": str(spec.get("form_type", "10-K")),
                        "accn": accn,
                    }
                ]
            },
        }
    return {
        "cik": cik,
        "entityName": str(spec.get("issuer_name", str(spec["symbol"]).upper())),
        "facts": {"us-gaap": facts},
    }


# -----------------------------------------------------------------------------
# Pipeline helper
# -----------------------------------------------------------------------------


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise AssertionError(f"Unsupported table extension: {path}")



def _build_submissions_table(symbol_specs: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for spec in symbol_specs:
        rows.append(
            {
                "cik": str(spec["cik"]).zfill(10),
                "symbol": str(spec["symbol"]).upper(),
                "accession_number": str(spec.get("accession_number", f"{str(spec['cik']).zfill(10)}-26-000001")),
                "form_type": str(spec.get("form_type", "10-K")),
                "filing_date": str(spec.get("filing_date", "2026-02-01")),
                "acceptance_datetime": str(spec.get("acceptance_datetime", "2026-02-01T14:30:00Z")),
                "period_end": str(spec.get("period_end", "2025-12-31")),
                "document_text": str(spec.get("document_text", "Annual report")),
                "filer_status": str(spec.get("filer_status", "large_accelerated_filer")),
            }
        )
    return pd.DataFrame(rows)



def _run_stack(
    *,
    tmp_path: Path,
    ticker_cik_module,
    parse_xbrl_module,
    filings_flags_module,
    point_in_time_module,
    edgar_qc_module,
    sec_rows: Iterable[Mapping[str, Any]],
    symbol_specs: Iterable[Mapping[str, Any]],
    run_qc: bool = True,
    asof_date: str = "2026-03-15",
) -> Dict[str, Any]:
    symbol_specs = [dict(s) for s in symbol_specs]
    root = tmp_path
    root.mkdir(parents=True, exist_ok=True)

    # --- ticker_cik ---
    sec_path = root / "sec_source.csv"
    pd.DataFrame(list(sec_rows)).to_csv(sec_path, index=False)

    map_cfg = copy.deepcopy(ticker_cik_module.DEFAULT_CONFIG)
    map_cfg["storage"]["output_root"] = str(root / "map_out")
    map_cfg["storage"]["allow_csv_fallback"] = True
    map_cfg["identity"]["strict_abort_on_unresolved_conflict"] = False
    map_cfg_path = root / "ticker_cik_config.json"
    map_cfg_path.write_text(json.dumps(map_cfg, indent=2), encoding="utf-8")

    mapping_result = ticker_cik_module.run_ticker_cik(
        sec_source_path=str(sec_path),
        internal_master_path=None,
        config_path=str(map_cfg_path),
        run_id="pytest_stack_mapping",
        asof=asof_date,
    )

    # --- raw JSON inputs for parse_xbrl ---
    submissions_raw_dir = root / "submissions_raw"
    companyfacts_raw_dir = root / "companyfacts_raw"
    submissions_raw_dir.mkdir(parents=True, exist_ok=True)
    companyfacts_raw_dir.mkdir(parents=True, exist_ok=True)
    for spec in symbol_specs:
        cik = str(spec["cik"]).zfill(10)
        (submissions_raw_dir / f"CIK{cik}.json").write_text(
            json.dumps(_submission_payload(spec), indent=2), encoding="utf-8"
        )
        (companyfacts_raw_dir / f"CIK{cik}.json").write_text(
            json.dumps(_companyfacts_payload(spec), indent=2), encoding="utf-8"
        )

    # --- parse_xbrl ---
    parse_cfg = {
        "output_dir": str(root / "parse_out"),
        "table_format": "csv",
        "allow_csv_fallback": True,
        "warn_reject_ratio": 0.95,
        "systemic_mapping_failure_ratio": 0.95,
        "unknown_taxonomy_policy": "reject",
        "unknown_unit_policy": "reject",
        "conflict_policy": "reject",
        "unresolved_tie_policy": "reject",
    }
    parse_cfg_path = root / "parse_xbrl_config.json"
    parse_cfg_path.write_text(json.dumps(parse_cfg, indent=2), encoding="utf-8")
    parse_manifest = parse_xbrl_module.parse_xbrl_run(
        submissions_raw_path=str(submissions_raw_dir),
        companyfacts_raw_path=str(companyfacts_raw_dir),
        config_path=str(parse_cfg_path),
        run_id="pytest_stack_parse",
        asof=asof_date,
    )
    canonical = _read_table(parse_manifest["outputs"]["facts_canonical"]).copy()

    # --- filings_flags ---
    facts_for_flags = canonical.rename(columns={"metric_name": "concept", "unit_base": "unit"}).copy()
    facts_for_flags_path = root / "facts_for_flags.csv"
    facts_for_flags.to_csv(facts_for_flags_path, index=False)

    submissions_table = _build_submissions_table(symbol_specs)
    submissions_table_path = root / "submissions_table.csv"
    submissions_table.to_csv(submissions_table_path, index=False)

    filings_cfg = {
        "output_dir": str(root / "flags_out"),
        "strict_parquet": False,
        "persistence": {"prefer_parquet": False},
    }
    filings_cfg_path = root / "filings_flags_config.json"
    filings_cfg_path.write_text(json.dumps(filings_cfg, indent=2), encoding="utf-8")

    filings_result = filings_flags_module.run_filings_flags(
        submissions_path=submissions_table_path,
        facts_path=facts_for_flags_path,
        config_path=filings_cfg_path,
        run_id="pytest_stack_flags",
        asof=f"{asof_date}T23:59:59Z",
    )

    # --- point_in_time ---
    pit_cfg = {
        "pit_config_version": "pytest_stack_pit_v1",
        "storage": {
            "output_root": str(root / "pit_out"),
            "allow_csv_fallback": True,
            "compression": "snappy",
        },
        "calendar": {"freq": "D", "asof_time_of_day": "23:59:59", "timezone": "UTC"},
        "metrics": {"include": ["revenue", "total_assets"], "exclude": []},
        "flags": {"enabled": True},
        "validation": {
            "abort_on_input_contract_failure": True,
            "coverage_warn_threshold": 0.10,
            "coverage_fail_threshold": 0.00,
            "abort_on_leakage": True,
            "abort_on_duplicate_output": True,
            "abort_on_identity_ambiguity": False,
        },
    }
    pit_cfg_path = root / "pit_config.json"
    pit_cfg_path.write_text(json.dumps(pit_cfg, indent=2), encoding="utf-8")

    pit_result = point_in_time_module.materialize_point_in_time(
        parsed_facts_path=parse_manifest["outputs"]["facts_canonical"],
        ticker_cik_mapping_path=mapping_result["artifacts"]["history_path"],
        pit_config_path=str(pit_cfg_path),
        run_id="pytest_stack_pit",
        start_date=asof_date,
        end_date=asof_date,
        filings_flags_path=filings_result["files"]["filings_flags"],
    )

    out: Dict[str, Any] = {
        "mapping_result": mapping_result,
        "parse_manifest": parse_manifest,
        "canonical": canonical,
        "filings_result": filings_result,
        "pit_result": pit_result,
        "submissions_table": submissions_table,
        "root": root,
    }

    if not run_qc:
        return out

    # --- Adapt outputs for edgar_qc ---
    mapping_for_qc = _read_table(mapping_result["artifacts"]["history_path"]).copy()
    if "effective_to" in mapping_for_qc.columns:
        mapping_for_qc["effective_to"] = mapping_for_qc["effective_to"].fillna("2100-01-01")
    mapping_qc_path = root / "ticker_cik_for_qc.csv"
    mapping_for_qc.to_csv(mapping_qc_path, index=False)

    parsed_for_qc = canonical.rename(
        columns={
            "metric_name": "metric",
            "unit_base": "unit",
            "acceptance_datetime": "acceptance_ts",
            "accession_number": "source_filing_id",
        }
    ).copy()
    parsed_for_qc_path = root / "parsed_facts_for_qc.csv"
    parsed_for_qc.to_csv(parsed_for_qc_path, index=False)

    companyfacts_for_qc = parsed_for_qc[
        ["symbol", "cik", "metric", "unit", "value", "period_end", "acceptance_ts", "source_filing_id"]
    ].copy()
    companyfacts_for_qc["run_id"] = "pytest_companyfacts"
    companyfacts_for_qc_path = root / "companyfacts_for_qc.csv"
    companyfacts_for_qc.to_csv(companyfacts_for_qc_path, index=False)

    submissions_for_qc = submissions_table.rename(
        columns={"accession_number": "source_filing_id", "acceptance_datetime": "acceptance_ts"}
    )[
        ["symbol", "cik", "source_filing_id", "form_type", "acceptance_ts", "filing_date"]
    ].copy()
    submissions_for_qc["status_code"] = 200
    submissions_for_qc["raw_artifact_id"] = "pytest_raw"
    submissions_for_qc["run_id"] = "pytest_submissions"
    submissions_for_qc_path = root / "submissions_for_qc.csv"
    submissions_for_qc.to_csv(submissions_for_qc_path, index=False)

    pit_for_qc = pit_result["pit"].rename(
        columns={
            "metric_name": "metric",
            "metric_value_pit": "value",
            "source_period_end": "period_end",
            "source_acceptance_ts": "acceptance_ts",
            "source_accession_number": "source_filing_id",
            "source_form_type": "form_type",
        }
    ).copy()
    pit_for_qc["unit"] = "USD"
    pit_for_qc = pit_for_qc[
        ["symbol", "cik", "metric", "value", "unit", "period_end", "acceptance_ts", "asof", "source_filing_id", "form_type", "run_id"]
    ].copy()
    pit_for_qc_path = root / "pit_for_qc.csv"
    pit_for_qc.to_csv(pit_for_qc_path, index=False)

    qc_result = edgar_qc_module.run_edgar_qc(
        ticker_cik_outputs_path=mapping_qc_path,
        submissions_raw_path=submissions_for_qc_path,
        companyfacts_raw_path=companyfacts_for_qc_path,
        parsed_facts_path=parsed_for_qc_path,
        pit_store_path=pit_for_qc_path,
        output_dir=root / "qc_out",
        run_id="pytest_stack_qc",
    )

    out.update(
        {
            "mapping_for_qc": mapping_for_qc,
            "qc_result": qc_result,
            "qc_input_paths": {
                "ticker_cik_outputs": mapping_qc_path,
                "submissions_raw": submissions_for_qc_path,
                "companyfacts_raw": companyfacts_for_qc_path,
                "parsed_facts": parsed_for_qc_path,
                "pit_store": pit_for_qc_path,
            },
        }
    )
    return out


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_clean_stack_materializes_pit_and_qc_passes(
    tmp_path: Path,
    patch_edgar_qc_write_parquet,
    ticker_cik_module,
    parse_xbrl_module,
    filings_flags_module,
    point_in_time_module,
    edgar_qc_module,
):
    case = _run_stack(
        tmp_path=tmp_path,
        ticker_cik_module=ticker_cik_module,
        parse_xbrl_module=parse_xbrl_module,
        filings_flags_module=filings_flags_module,
        point_in_time_module=point_in_time_module,
        edgar_qc_module=edgar_qc_module,
        sec_rows=[_sec_row("AAA", "1", issuer_name="AAA INC")],
        symbol_specs=[{"symbol": "AAA", "cik": "1", "issuer_name": "AAA INC", "include_assets": True}],
        run_qc=True,
    )

    canonical = case["canonical"].sort_values(["symbol", "metric_name"], kind="mergesort").reset_index(drop=True)
    pit = case["pit_result"]["pit"].sort_values(["symbol", "metric_name"], kind="mergesort").reset_index(drop=True)
    qc_summary = case["qc_result"]["summary"]

    assert set(canonical["symbol"]) == {"AAA"}
    assert set(canonical["metric_name"]) == {"revenue", "total_assets"}
    assert set(pit["symbol"]) == {"AAA"}
    assert set(pit["metric_name"]) == {"revenue", "total_assets"}
    assert pit["metric_value_pit"].notna().all()
    assert qc_summary["gate"] == "pass"
    assert float(qc_summary["score"]) >= 99.0



def test_filings_flags_penalize_flows_into_point_in_time_quality_weight(
    tmp_path: Path,
    patch_edgar_qc_write_parquet,
    ticker_cik_module,
    parse_xbrl_module,
    filings_flags_module,
    point_in_time_module,
    edgar_qc_module,
):
    # Even with revenue + assets, the current flags implementation can mark
    # missing_core_fields for sparse fundamentals; what matters here is that
    # filings_flags feeds PIT policy and PIT preserves the value while penalizing quality.
    case = _run_stack(
        tmp_path=tmp_path,
        ticker_cik_module=ticker_cik_module,
        parse_xbrl_module=parse_xbrl_module,
        filings_flags_module=filings_flags_module,
        point_in_time_module=point_in_time_module,
        edgar_qc_module=edgar_qc_module,
        sec_rows=[_sec_row("AAA", "1", issuer_name="AAA INC")],
        symbol_specs=[{"symbol": "AAA", "cik": "1", "issuer_name": "AAA INC", "include_assets": True}],
        run_qc=False,
    )

    filings = case["filings_result"]["filings_flags"].copy()
    pit = case["pit_result"]["pit"].copy()

    assert not filings.empty
    assert set(filings["pit_action"].astype(str)) == {"penalize"}
    assert pit["pit_action"].eq("penalize").all()
    assert (pit["quality_weight"].astype(float) < 1.0).all()
    assert pit["metric_value_pit"].notna().all()



def test_unresolved_identity_conflict_symbol_is_excluded_from_pit_but_clean_symbol_survives(
    tmp_path: Path,
    patch_edgar_qc_write_parquet,
    ticker_cik_module,
    parse_xbrl_module,
    filings_flags_module,
    point_in_time_module,
    edgar_qc_module,
):
    sec_rows = [
        _sec_row("AAA", "1", issuer_name="AAA INC"),
        _sec_row("XYZ", "3333333333", issuer_name="XYZ INC"),
        _sec_row("XYZ", "4444444444", issuer_name="XYZ INC"),
    ]
    symbol_specs = [
        {"symbol": "AAA", "cik": "1", "issuer_name": "AAA INC", "include_assets": True},
        {"symbol": "XYZ", "cik": "3333333333", "issuer_name": "XYZ INC", "include_assets": True},
    ]

    case = _run_stack(
        tmp_path=tmp_path,
        ticker_cik_module=ticker_cik_module,
        parse_xbrl_module=parse_xbrl_module,
        filings_flags_module=filings_flags_module,
        point_in_time_module=point_in_time_module,
        edgar_qc_module=edgar_qc_module,
        sec_rows=sec_rows,
        symbol_specs=symbol_specs,
        run_qc=False,
    )

    current = case["mapping_result"]["current"].copy()
    conflicts = case["mapping_result"]["conflicts"].copy()
    pit = case["pit_result"]["pit"].copy()

    assert set(current["symbol"].astype(str)) == {"AAA"}
    assert "unresolved_identity_conflict" in set(conflicts["conflict_class"].astype(str))
    assert set(pit["symbol"].astype(str)) == {"AAA"}
    assert "XYZ" not in set(pit["symbol"].astype(str))



def test_same_asof_rerun_is_stable_across_core_stack(
    tmp_path: Path,
    patch_edgar_qc_write_parquet,
    ticker_cik_module,
    parse_xbrl_module,
    filings_flags_module,
    point_in_time_module,
    edgar_qc_module,
):
    sec_rows = [_sec_row("AAA", "1", issuer_name="AAA INC")]
    symbol_specs = [{"symbol": "AAA", "cik": "1", "issuer_name": "AAA INC", "include_assets": True}]

    case_a = _run_stack(
        tmp_path=tmp_path / "run_a",
        ticker_cik_module=ticker_cik_module,
        parse_xbrl_module=parse_xbrl_module,
        filings_flags_module=filings_flags_module,
        point_in_time_module=point_in_time_module,
        edgar_qc_module=edgar_qc_module,
        sec_rows=sec_rows,
        symbol_specs=symbol_specs,
        run_qc=True,
    )
    case_b = _run_stack(
        tmp_path=tmp_path / "run_b",
        ticker_cik_module=ticker_cik_module,
        parse_xbrl_module=parse_xbrl_module,
        filings_flags_module=filings_flags_module,
        point_in_time_module=point_in_time_module,
        edgar_qc_module=edgar_qc_module,
        sec_rows=sec_rows,
        symbol_specs=symbol_specs,
        run_qc=True,
    )

    canonical_cols = ["symbol", "cik", "metric_name", "value", "period_end", "accession_number", "mapping_status", "selection_reason"]
    pit_cols = ["symbol", "cik", "metric_name", "metric_value_pit", "pit_action", "quality_weight", "flag_severity"]

    canonical_a = case_a["canonical"][canonical_cols].sort_values(canonical_cols[:3], kind="mergesort").reset_index(drop=True)
    canonical_b = case_b["canonical"][canonical_cols].sort_values(canonical_cols[:3], kind="mergesort").reset_index(drop=True)
    pit_a = case_a["pit_result"]["pit"][pit_cols].sort_values(pit_cols[:3], kind="mergesort").reset_index(drop=True)
    pit_b = case_b["pit_result"]["pit"][pit_cols].sort_values(pit_cols[:3], kind="mergesort").reset_index(drop=True)

    pd.testing.assert_frame_equal(canonical_a, canonical_b, check_like=False)
    pd.testing.assert_frame_equal(pit_a, pit_b, check_like=False)
    assert case_a["qc_result"]["summary"]["gate"] == case_b["qc_result"]["summary"]["gate"] == "pass"
    assert float(case_a["qc_result"]["summary"]["score"]) == pytest.approx(float(case_b["qc_result"]["summary"]["score"]))
