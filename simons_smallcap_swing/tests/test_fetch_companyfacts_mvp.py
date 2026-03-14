from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from data.edgar.fetch_companyfacts import fetch_companyfacts
from data.edgar.fetch_submissions import fetch_submissions
from data.edgar.ticker_cik import build_ticker_cik_map
from data.reference.build_reference import build_reference_data
from data.universe.build_universe import build_universe
from simons_core.io.parquet_store import read_parquet
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


COMPANYFACTS_MIN_SCHEMA = DataSchema(
    name="companyfacts_raw_min_test",
    version="1.0.0",
    columns=(
        ColumnSpec("cik", "string", nullable=False),
        ColumnSpec("taxonomy", "string", nullable=False),
        ColumnSpec("tag", "string", nullable=False),
        ColumnSpec("unit", "string", nullable=False),
        ColumnSpec("fact_value", "number", nullable=False),
        ColumnSpec("fact_start_date", "datetime64", nullable=True),
        ColumnSpec("fact_end_date", "datetime64", nullable=True),
        ColumnSpec("filing_date", "datetime64", nullable=False),
        ColumnSpec("acceptance_ts", "datetime64[ns, UTC]", nullable=False),
        ColumnSpec("fiscal_year", "number", nullable=True),
        ColumnSpec("fiscal_period", "string", nullable=True),
        ColumnSpec("form_type", "string", nullable=True),
        ColumnSpec("frame", "string", nullable=True),
        ColumnSpec("accession_number", "string", nullable=True),
    ),
    primary_key=(),
    allow_extra_columns=True,
)


def _build_identity_inputs(tmp_workspace: dict[str, Path]) -> dict[str, Path]:
    reference_root = tmp_workspace["data"] / "reference"
    universe_root = tmp_workspace["data"] / "universe"
    edgar_root = tmp_workspace["data"] / "edgar"

    build_reference_data(output_dir=reference_root, run_id="test_reference_fetch_companyfacts")
    universe_result = build_universe(
        reference_root=reference_root,
        output_dir=universe_root,
        run_id="test_universe_fetch_companyfacts",
    )
    ticker_cik_result = build_ticker_cik_map(
        reference_root=reference_root,
        universe_history_path=universe_result.universe_history,
        output_dir=edgar_root,
        run_id="test_ticker_cik_fetch_companyfacts",
    )
    return {
        "ticker_cik_map": ticker_cik_result.ticker_cik_map_path,
        "universe_history": universe_result.universe_history,
        "edgar_root": edgar_root,
    }


def _write_local_sec_submissions_json(local_dir: Path, ciks: list[str]) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for idx, cik in enumerate(ciks):
        payload = {
            "cik": cik,
            "filings": {
                "recent": {
                    "accessionNumber": [
                        f"0000000000-26-2000{idx + 1}",
                        f"0000000000-26-2001{idx + 1}",
                    ],
                    "filingDate": ["2026-02-20", "2026-02-20"],
                    "acceptanceDateTime": [
                        "2026-02-20T21:35:00Z",
                        "2026-02-20T21:50:00Z",
                    ],
                    "form": ["10-K", "10-K"],
                    "primaryDocument": ["annual.htm", "annual_xbrl.htm"],
                    "primaryDocDescription": [
                        "Annual report",
                        "Annual report inline",
                    ],
                    "reportDate": ["2025-12-31", "2025-12-31"],
                    "isXBRL": [1, 1],
                    "isInlineXBRL": [1, 1],
                }
            },
        }
        (local_dir / f"CIK{cik}.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )


def _write_local_companyfacts_json(local_dir: Path, ciks: list[str]) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for idx, cik in enumerate(ciks):
        payload = {
            "cik": cik,
            "entityName": f"Issuer {idx + 1}",
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2025-12-31",
                                    "val": 1_000_000 + idx * 15_000,
                                    "accn": f"0000000000-26-2000{idx + 1}",
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2026-02-20",
                                    "frame": "CY2025",
                                }
                            ]
                        }
                    },
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {
                                    "start": "2025-01-01",
                                    "end": "2025-12-31",
                                    "val": 120_000 + idx * 2_000,
                                    "accn": f"0000000000-26-2001{idx + 1}",
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2026-02-20",
                                    "acceptance": "2026-02-20T21:45:00Z",
                                    "frame": "CY2025",
                                }
                            ]
                        }
                    },
                }
            },
        }
        (local_dir / f"CIK{cik}.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )


def test_fetch_companyfacts_local_file_mode_generates_outputs(
    tmp_workspace: dict[str, Path],
) -> None:
    inputs = _build_identity_inputs(tmp_workspace)
    ticker_map = read_parquet(inputs["ticker_cik_map"])
    target_ciks = sorted(set(ticker_map["cik"].astype(str).tolist()))[:2]

    submissions_source_dir = tmp_workspace["data"] / "edgar_source" / "submissions_for_companyfacts"
    _write_local_sec_submissions_json(submissions_source_dir, target_ciks)
    submissions_result = fetch_submissions(
        ticker_cik_map_path=inputs["ticker_cik_map"],
        universe_history_path=inputs["universe_history"],
        local_source_path=submissions_source_dir,
        output_dir=inputs["edgar_root"],
        ingestion_mode="local_file",
        run_id="test_submissions_for_companyfacts",
        force_rebuild=True,
    )

    companyfacts_source_dir = tmp_workspace["data"] / "edgar_source" / "companyfacts_local"
    _write_local_companyfacts_json(companyfacts_source_dir, target_ciks)

    result = fetch_companyfacts(
        ticker_cik_map_path=inputs["ticker_cik_map"],
        universe_history_path=inputs["universe_history"],
        submissions_raw_path=submissions_result.submissions_raw_path,
        local_source_path=companyfacts_source_dir,
        output_dir=inputs["edgar_root"],
        ingestion_mode="local_file",
        run_id="test_fetch_companyfacts_local_mode",
        force_rebuild=True,
    )

    assert result.companyfacts_raw_path.exists()
    assert result.ingestion_report_path.exists()
    assert result.row_count > 0
    assert result.source_mode == "local_file"

    companyfacts = read_parquet(result.companyfacts_raw_path)
    assert_schema(companyfacts, COMPANYFACTS_MIN_SCHEMA)
    assert len(companyfacts) > 0
    assert set(companyfacts["cik"].unique().tolist()).issubset(set(target_ciks))
    acceptance_date = (
        companyfacts["acceptance_ts"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    assert (companyfacts["filing_date"] <= acceptance_date).all()

    report = json.loads(result.ingestion_report_path.read_text(encoding="utf-8"))
    assert report["resolved_mode"] == "local_file"
    assert int(report["n_records_output"]) == len(companyfacts)
    assert int(report["acceptance_filled_from_submissions"]) >= 1


def test_fetch_companyfacts_deduplicates_impossible_duplicates(
    tmp_workspace: dict[str, Path],
) -> None:
    inputs = _build_identity_inputs(tmp_workspace)
    cik = str(read_parquet(inputs["ticker_cik_map"])["cik"].astype(str).iloc[0])

    local_dir = tmp_workspace["data"] / "edgar_source" / "companyfacts_dups"
    local_dir.mkdir(parents=True, exist_ok=True)
    duplicate_rows = pd.DataFrame(
        [
            {
                "cik": cik,
                "taxonomy": "us-gaap",
                "tag": "Revenues",
                "unit": "USD",
                "fact_value": 1110000,
                "fact_start_date": "2025-01-01",
                "fact_end_date": "2025-12-31",
                "filing_date": "2026-02-20",
                "acceptance_ts": "2026-02-20T21:00:00Z",
                "fiscal_year": 2025,
                "fiscal_period": "FY",
                "form_type": "10-K",
                "frame": "CY2025",
                "accession_number": "0000000000-26-30001",
            },
            {
                "cik": cik,
                "taxonomy": "us-gaap",
                "tag": "Revenues",
                "unit": "USD",
                "fact_value": 1115000,
                "fact_start_date": "2025-01-01",
                "fact_end_date": "2025-12-31",
                "filing_date": "2026-02-20",
                "acceptance_ts": "2026-02-20T23:00:00Z",
                "fiscal_year": 2025,
                "fiscal_period": "FY",
                "form_type": "10-K",
                "frame": "CY2025",
                "accession_number": "0000000000-26-30001",
            },
        ]
    )
    duplicate_rows.to_csv(local_dir / "dup_rows.csv", index=False)

    result = fetch_companyfacts(
        ticker_cik_map_path=inputs["ticker_cik_map"],
        universe_history_path=inputs["universe_history"],
        local_source_path=local_dir,
        output_dir=inputs["edgar_root"],
        ingestion_mode="local_file",
        run_id="test_fetch_companyfacts_dedup",
        force_rebuild=True,
    )
    companyfacts = read_parquet(result.companyfacts_raw_path)
    rows = companyfacts[
        (companyfacts["cik"] == cik)
        & (companyfacts["tag"] == "Revenues")
        & (companyfacts["accession_number"] == "0000000000-26-30001")
    ]
    assert len(rows) == 1
    assert float(rows.iloc[0]["fact_value"]) == pytest.approx(1115000.0)

    report = json.loads(result.ingestion_report_path.read_text(encoding="utf-8"))
    assert int(report["duplicate_rows_dropped"]) >= 1


def test_fetch_companyfacts_without_local_and_remote_disabled_fails_clearly(
    tmp_workspace: dict[str, Path],
) -> None:
    inputs = _build_identity_inputs(tmp_workspace)
    missing_local = tmp_workspace["data"] / "edgar_source" / "missing_companyfacts"
    with pytest.raises(FileNotFoundError, match="no source files found"):
        fetch_companyfacts(
            ticker_cik_map_path=inputs["ticker_cik_map"],
            universe_history_path=inputs["universe_history"],
            local_source_path=missing_local,
            output_dir=inputs["edgar_root"],
            ingestion_mode="local_file",
            run_id="test_fetch_companyfacts_missing_local",
            force_rebuild=True,
        )


def test_fetch_companyfacts_reuses_cache_when_enabled(
    tmp_workspace: dict[str, Path],
) -> None:
    inputs = _build_identity_inputs(tmp_workspace)
    target_cik = str(read_parquet(inputs["ticker_cik_map"])["cik"].astype(str).iloc[0])

    local_dir = tmp_workspace["data"] / "edgar_source" / "companyfacts_cache"
    _write_local_companyfacts_json(local_dir, [target_cik])

    first = fetch_companyfacts(
        ticker_cik_map_path=inputs["ticker_cik_map"],
        universe_history_path=inputs["universe_history"],
        local_source_path=local_dir,
        output_dir=inputs["edgar_root"],
        ingestion_mode="local_file",
        run_id="test_fetch_companyfacts_cache_first",
        force_rebuild=True,
    )
    assert not first.reused_cache

    second = fetch_companyfacts(
        ticker_cik_map_path=inputs["ticker_cik_map"],
        universe_history_path=inputs["universe_history"],
        local_source_path=tmp_workspace["data"] / "edgar_source" / "missing_after_first",
        output_dir=inputs["edgar_root"],
        ingestion_mode="local_file",
        run_id="test_fetch_companyfacts_cache_second",
        force_rebuild=False,
        reuse_cache=True,
    )
    assert second.reused_cache
    assert second.source_mode == "cache_reuse"
    assert second.row_count == first.row_count
