from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from data.edgar.fetch_submissions import fetch_submissions
from data.edgar.ticker_cik import build_ticker_cik_map
from data.reference.build_reference import build_reference_data
from data.universe.build_universe import build_universe
from simons_core.io.parquet_store import read_parquet
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


SUBMISSIONS_MIN_SCHEMA = DataSchema(
    name="submissions_raw_min_test",
    version="1.0.0",
    columns=(
        ColumnSpec("cik", "string", nullable=False),
        ColumnSpec("accession_number", "string", nullable=False),
        ColumnSpec("filing_date", "datetime64", nullable=False),
        ColumnSpec("acceptance_ts", "datetime64[ns, UTC]", nullable=False),
        ColumnSpec("form_type", "string", nullable=False),
        ColumnSpec("filing_primary_document", "string", nullable=True),
        ColumnSpec("filing_primary_doc_description", "string", nullable=True),
        ColumnSpec("report_date", "datetime64", nullable=True),
        ColumnSpec("is_xbrl", "bool", nullable=False),
        ColumnSpec("is_inline_xbrl", "bool", nullable=False),
    ),
    primary_key=("cik", "accession_number"),
    allow_extra_columns=True,
)


def _build_identity_inputs(tmp_workspace: dict[str, Path]) -> dict[str, Path]:
    reference_root = tmp_workspace["data"] / "reference"
    universe_root = tmp_workspace["data"] / "universe"
    edgar_root = tmp_workspace["data"] / "edgar"

    build_reference_data(output_dir=reference_root, run_id="test_reference_fetch_submissions")
    universe_result = build_universe(
        reference_root=reference_root,
        output_dir=universe_root,
        run_id="test_universe_fetch_submissions",
    )
    ticker_cik_result = build_ticker_cik_map(
        reference_root=reference_root,
        universe_history_path=universe_result.universe_history,
        output_dir=edgar_root,
        run_id="test_ticker_cik_fetch_submissions",
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
                        f"0000000000-26-0000{idx + 1}",
                        f"0000000000-26-0001{idx + 1}",
                    ],
                    "filingDate": ["2026-01-20", "2026-02-15"],
                    "acceptanceDateTime": [
                        "2026-01-20T21:15:00Z",
                        "2026-02-15T22:30:00Z",
                    ],
                    "form": ["10-Q", "8-K"],
                    "primaryDocument": ["q1.htm", "event.htm"],
                    "primaryDocDescription": [
                        "Quarterly report",
                        "Current report",
                    ],
                    "reportDate": ["2025-12-31", "2026-02-10"],
                    "isXBRL": [1, 1],
                    "isInlineXBRL": [1, 0],
                }
            },
        }
        (local_dir / f"CIK{cik}.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )


def test_fetch_submissions_local_file_mode_generates_outputs(
    tmp_workspace: dict[str, Path],
) -> None:
    inputs = _build_identity_inputs(tmp_workspace)
    ticker_map = read_parquet(inputs["ticker_cik_map"])
    target_ciks = sorted(set(ticker_map["cik"].astype(str).tolist()))[:2]

    local_dir = tmp_workspace["data"] / "edgar_source" / "submissions_local"
    _write_local_sec_submissions_json(local_dir, target_ciks)

    result = fetch_submissions(
        ticker_cik_map_path=inputs["ticker_cik_map"],
        universe_history_path=inputs["universe_history"],
        local_source_path=local_dir,
        output_dir=inputs["edgar_root"],
        ingestion_mode="local_file",
        run_id="test_fetch_submissions_local_mode",
        force_rebuild=True,
    )

    assert result.submissions_raw_path.exists()
    assert result.ingestion_report_path.exists()
    assert result.row_count > 0
    assert result.source_mode == "local_file"

    submissions = read_parquet(result.submissions_raw_path)
    assert_schema(submissions, SUBMISSIONS_MIN_SCHEMA)
    assert len(submissions) > 0
    assert submissions["cik"].nunique() >= 1
    assert (submissions["filing_date"] <= submissions["acceptance_ts"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()).all()

    report = json.loads(result.ingestion_report_path.read_text(encoding="utf-8"))
    assert report["resolved_mode"] == "local_file"
    assert report["rows_output"] == len(submissions)


def test_fetch_submissions_deduplicates_impossible_duplicates(
    tmp_workspace: dict[str, Path],
) -> None:
    inputs = _build_identity_inputs(tmp_workspace)
    ticker_map = read_parquet(inputs["ticker_cik_map"])
    cik = sorted(set(ticker_map["cik"].astype(str).tolist()))[0]

    local_dir = tmp_workspace["data"] / "edgar_source" / "submissions_dups"
    local_dir.mkdir(parents=True, exist_ok=True)
    duplicate_rows = pd.DataFrame(
        [
            {
                "cik": cik,
                "accession_number": "0000000000-26-009999",
                "filing_date": "2026-01-31",
                "acceptance_ts": "2026-01-31T20:00:00Z",
                "form_type": "10-Q",
                "filing_primary_document": "q.htm",
                "filing_primary_doc_description": "Quarterly",
                "report_date": "2025-12-31",
                "is_xbrl": 1,
                "is_inline_xbrl": 1,
            },
            {
                "cik": cik,
                "accession_number": "0000000000-26-009999",
                "filing_date": "2026-01-31",
                "acceptance_ts": "2026-01-31T23:00:00Z",
                "form_type": "10-Q",
                "filing_primary_document": "q_revised.htm",
                "filing_primary_doc_description": "Quarterly revised",
                "report_date": "2025-12-31",
                "is_xbrl": 1,
                "is_inline_xbrl": 1,
            },
        ]
    )
    duplicate_path = local_dir / "dup_rows.csv"
    duplicate_rows.to_csv(duplicate_path, index=False)

    result = fetch_submissions(
        ticker_cik_map_path=inputs["ticker_cik_map"],
        universe_history_path=inputs["universe_history"],
        local_source_path=local_dir,
        output_dir=inputs["edgar_root"],
        ingestion_mode="local_file",
        run_id="test_fetch_submissions_dedup",
        force_rebuild=True,
    )
    submissions = read_parquet(result.submissions_raw_path)
    rows = submissions[
        (submissions["cik"] == cik)
        & (submissions["accession_number"] == "0000000000-26-009999")
    ]
    assert len(rows) == 1
    assert str(rows.iloc[0]["filing_primary_document"]) == "q_revised.htm"

    report = json.loads(result.ingestion_report_path.read_text(encoding="utf-8"))
    assert int(report["duplicate_rows_dropped"]) >= 1


def test_fetch_submissions_without_local_and_remote_disabled_fails_clearly(
    tmp_workspace: dict[str, Path],
) -> None:
    inputs = _build_identity_inputs(tmp_workspace)
    missing_local = tmp_workspace["data"] / "edgar_source" / "missing_submissions"
    with pytest.raises(FileNotFoundError, match="no source files found"):
        fetch_submissions(
            ticker_cik_map_path=inputs["ticker_cik_map"],
            universe_history_path=inputs["universe_history"],
            local_source_path=missing_local,
            output_dir=inputs["edgar_root"],
            ingestion_mode="local_file",
            run_id="test_fetch_submissions_missing_local",
            force_rebuild=True,
        )


def test_fetch_submissions_reuses_cache_when_enabled(
    tmp_workspace: dict[str, Path],
) -> None:
    inputs = _build_identity_inputs(tmp_workspace)
    ticker_map = read_parquet(inputs["ticker_cik_map"])
    target_ciks = sorted(set(ticker_map["cik"].astype(str).tolist()))[:1]

    local_dir = tmp_workspace["data"] / "edgar_source" / "submissions_cache"
    _write_local_sec_submissions_json(local_dir, target_ciks)

    first = fetch_submissions(
        ticker_cik_map_path=inputs["ticker_cik_map"],
        universe_history_path=inputs["universe_history"],
        local_source_path=local_dir,
        output_dir=inputs["edgar_root"],
        ingestion_mode="local_file",
        run_id="test_fetch_submissions_cache_first",
        force_rebuild=True,
    )
    assert not first.reused_cache

    # Re-run with a non-existing source path; cached artifact should still be reused.
    second = fetch_submissions(
        ticker_cik_map_path=inputs["ticker_cik_map"],
        universe_history_path=inputs["universe_history"],
        local_source_path=tmp_workspace["data"] / "edgar_source" / "non_existing_now",
        output_dir=inputs["edgar_root"],
        ingestion_mode="local_file",
        run_id="test_fetch_submissions_cache_second",
        force_rebuild=False,
        reuse_cache=True,
    )
    assert second.reused_cache
    assert second.source_mode == "cache_reuse"
    assert second.row_count == first.row_count
