from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.edgar.edgar_qc import run_edgar_qc
from data.edgar.fetch_companyfacts import fetch_companyfacts
from data.edgar.fetch_submissions import fetch_submissions
from data.edgar.point_in_time import build_fundamentals_pit
from data.edgar.ticker_cik import build_ticker_cik_map
from data.reference.build_reference import build_reference_data
from data.universe.build_universe import build_universe
from simons_core.io.parquet_store import read_parquet
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


FUNDAMENTALS_PIT_V2_SCHEMA = DataSchema(
    name="fundamentals_pit_v2_min_test",
    version="2.0.0",
    columns=(
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("cik", "string", nullable=False),
        ColumnSpec("asof_date", "datetime64[ns, UTC]", nullable=False),
        ColumnSpec("acceptance_ts", "datetime64[ns, UTC]", nullable=False),
        ColumnSpec("filing_date", "datetime64", nullable=False),
        ColumnSpec("fact_start_date", "datetime64", nullable=True),
        ColumnSpec("fact_end_date", "datetime64", nullable=True),
        ColumnSpec("fiscal_year", "number", nullable=True),
        ColumnSpec("fiscal_period", "string", nullable=True),
        ColumnSpec("form_type", "string", nullable=True),
        ColumnSpec("taxonomy", "string", nullable=False),
        ColumnSpec("metric_name", "string", nullable=False),
        ColumnSpec("metric_value", "number", nullable=False),
        ColumnSpec("metric_unit", "string", nullable=False),
        ColumnSpec("accession_number", "string", nullable=True),
        ColumnSpec("source_type", "string", nullable=False),
        ColumnSpec("data_quality", "string", nullable=False),
        ColumnSpec("visibility_rule", "string", nullable=False),
    ),
    primary_key=("instrument_id", "asof_date", "metric_name"),
    allow_extra_columns=True,
)


def _build_identity_inputs(tmp_workspace: dict[str, Path]) -> dict[str, Path]:
    reference_root = tmp_workspace["data"] / "reference"
    universe_root = tmp_workspace["data"] / "universe"
    edgar_root = tmp_workspace["data"] / "edgar"

    build_reference_data(output_dir=reference_root, run_id="test_reference_edgar_pit_v2")
    universe_result = build_universe(
        reference_root=reference_root,
        output_dir=universe_root,
        run_id="test_universe_edgar_pit_v2",
    )
    ticker_cik_result = build_ticker_cik_map(
        reference_root=reference_root,
        universe_history_path=universe_result.universe_history,
        output_dir=edgar_root,
        run_id="test_ticker_cik_edgar_pit_v2",
    )
    return {
        "ticker_cik_map": ticker_cik_result.ticker_cik_map_path,
        "universe_history": universe_result.universe_history,
        "edgar_root": edgar_root,
    }


def _build_accession(prefix: str, idx: int) -> str:
    return f"0000000000-26-{prefix}{idx:04d}"


def _write_local_submissions_csv(local_dir: Path, ciks: list[str]) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    accession_kinds = ("R1", "R2", "RF", "N1", "N2", "A1", "S1")
    acceptance_map = {
        "R1": "2026-01-20T21:00:00Z",
        "R2": "2026-02-10T21:00:00Z",
        "RF": "2026-05-10T21:00:00Z",
        "N1": "2026-02-15T21:00:00Z",
        "N2": "2026-02-15T21:00:00Z",
        "A1": "2026-02-18T22:00:00Z",
        "S1": "2026-01-25T21:00:00Z",
    }
    filing_map = {
        "R1": "2026-01-20",
        "R2": "2026-02-10",
        "RF": "2026-05-10",
        "N1": "2026-02-14",
        "N2": "2026-02-15",
        "A1": "2026-02-18",
        "S1": "2026-01-25",
    }
    for idx, cik in enumerate(ciks, start=1):
        for kind in accession_kinds:
            accession = _build_accession(kind, idx)
            rows.append(
                {
                    "cik": cik,
                    "accession_number": accession,
                    "filing_date": filing_map[kind],
                    "acceptance_ts": acceptance_map[kind],
                    "form_type": "10-K" if kind != "S1" else "8-K",
                    "filing_primary_document": "doc.htm",
                    "filing_primary_doc_description": "document",
                    "report_date": "2025-12-31",
                    "is_xbrl": 1,
                    "is_inline_xbrl": 1,
                }
            )
    path = local_dir / "submissions_local.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_local_companyfacts_json(local_dir: Path, ciks: list[str]) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for idx, cik in enumerate(ciks, start=1):
        payload = {
            "cik": cik,
            "entityName": f"Issuer {idx}",
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2025-12-31",
                                    "val": 1_100_000 + idx * 10_000,
                                    "accn": _build_accession("R1", idx),
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2026-01-20",
                                    "acceptance": "2026-01-20T21:00:00Z",
                                    "frame": "CY2025",
                                },
                                {
                                    "end": "2025-12-31",
                                    "val": 1_300_000 + idx * 10_000,
                                    "accn": _build_accession("R2", idx),
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2026-02-10",
                                    "acceptance": "2026-02-10T21:00:00Z",
                                    "frame": "CY2025",
                                },
                                {
                                    "end": "2026-03-31",
                                    "val": 1_500_000 + idx * 10_000,
                                    "accn": _build_accession("RF", idx),
                                    "fy": 2026,
                                    "fp": "Q1",
                                    "form": "10-Q",
                                    "filed": "2026-05-10",
                                    "acceptance": "2026-05-10T21:00:00Z",
                                    "frame": "CY2026Q1",
                                },
                            ]
                        }
                    },
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2025-12-31",
                                    "val": 90_000 + idx * 1_000,
                                    "accn": _build_accession("N1", idx),
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2026-02-14",
                                    "acceptance": "2026-02-15T21:00:00Z",
                                    "frame": "CY2025",
                                },
                                {
                                    "end": "2025-12-31",
                                    "val": 95_000 + idx * 1_000,
                                    "accn": _build_accession("N2", idx),
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2026-02-15",
                                    "acceptance": "2026-02-15T21:00:00Z",
                                    "frame": "CY2025",
                                },
                            ]
                        }
                    },
                    "Assets": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2025-12-31",
                                    "val": 3_200_000 + idx * 5_000,
                                    "accn": _build_accession("A1", idx),
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2026-02-18",
                                    "frame": "CY2025",
                                }
                            ]
                        }
                    },
                },
                "dei": {
                    "EntityCommonStockSharesOutstanding": {
                        "units": {
                            "shares": [
                                {
                                    "end": "2025-12-31",
                                    "val": 22_000_000 + idx * 100_000,
                                    "accn": _build_accession("S1", idx),
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2026-01-25",
                                    "acceptance": "2026-01-25T21:00:00Z",
                                    "frame": "CY2025",
                                }
                            ]
                        }
                    }
                },
            },
        }
        (local_dir / f"CIK{cik}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_edgar_pit_v2(tmp_workspace: dict[str, Path]) -> dict[str, Path]:
    inputs = _build_identity_inputs(tmp_workspace)
    ticker_map = read_parquet(inputs["ticker_cik_map"])
    target_ciks = sorted(set(ticker_map["cik"].astype(str).tolist()))[:2]

    submissions_source_dir = tmp_workspace["data"] / "edgar_source" / "submissions_v2"
    _ = _write_local_submissions_csv(submissions_source_dir, target_ciks)
    submissions_result = fetch_submissions(
        ticker_cik_map_path=inputs["ticker_cik_map"],
        universe_history_path=inputs["universe_history"],
        local_source_path=submissions_source_dir,
        output_dir=inputs["edgar_root"],
        ingestion_mode="local_file",
        run_id="test_submissions_for_pit_v2",
        force_rebuild=True,
    )

    companyfacts_source_dir = tmp_workspace["data"] / "edgar_source" / "companyfacts_v2"
    _write_local_companyfacts_json(companyfacts_source_dir, target_ciks)
    companyfacts_result = fetch_companyfacts(
        ticker_cik_map_path=inputs["ticker_cik_map"],
        universe_history_path=inputs["universe_history"],
        submissions_raw_path=submissions_result.submissions_raw_path,
        local_source_path=companyfacts_source_dir,
        output_dir=inputs["edgar_root"],
        ingestion_mode="local_file",
        run_id="test_companyfacts_for_pit_v2",
        force_rebuild=True,
    )

    pit_result = build_fundamentals_pit(
        ticker_cik_map_path=inputs["ticker_cik_map"],
        universe_history_path=inputs["universe_history"],
        submissions_raw_path=submissions_result.submissions_raw_path,
        companyfacts_raw_path=companyfacts_result.companyfacts_raw_path,
        output_dir=inputs["edgar_root"],
        run_id="test_fundamentals_pit_v2",
    )
    return {
        "ticker_cik_map": inputs["ticker_cik_map"],
        "submissions_raw": submissions_result.submissions_raw_path,
        "companyfacts_raw": companyfacts_result.companyfacts_raw_path,
        "fundamentals_pit": pit_result.fundamentals_pit_path,
        "fundamentals_events": pit_result.fundamentals_events_path,
    }


def test_point_in_time_v2_generates_non_empty_pit_from_raw_sources(
    tmp_workspace: dict[str, Path],
) -> None:
    paths = _build_edgar_pit_v2(tmp_workspace)
    pit = read_parquet(paths["fundamentals_pit"])

    assert paths["fundamentals_pit"].exists()
    assert paths["fundamentals_events"].exists()
    assert len(pit) > 0

    assert_schema(pit, FUNDAMENTALS_PIT_V2_SCHEMA)
    assert set(ALLOWED := {"revenue", "net_income", "total_assets", "shares_outstanding"}).issubset(
        set(pit["metric_name"].unique().tolist())
    )
    assert (
        pd.to_datetime(pit["acceptance_ts"], utc=True)
        <= pd.to_datetime(pit["asof_date"], utc=True)
    ).all()
    assert not pit["source_type"].str.contains("synthetic", case=False, na=False).all()


def test_point_in_time_v2_visibility_and_selection_are_deterministic(
    tmp_workspace: dict[str, Path],
) -> None:
    paths = _build_edgar_pit_v2(tmp_workspace)
    pit = read_parquet(paths["fundamentals_pit"]).copy()
    ticker_map = read_parquet(paths["ticker_cik_map"]).copy()

    # Future filing should never be visible in the generated asof range.
    assert not (pit["accession_number"].astype(str).str.contains("-26-RF", regex=False)).any()

    # Deterministic winner for revenue: latest visible acceptance should win.
    sample_cik = str(pit["cik"].astype(str).iloc[0])
    sample_instrument = str(
        pit[pit["cik"].astype(str) == sample_cik]["instrument_id"].astype(str).iloc[0]
    )
    sample = pit[
        (pit["instrument_id"] == sample_instrument) & (pit["metric_name"] == "revenue")
    ].sort_values("asof_date")
    assert len(sample) > 0
    last_row = sample.iloc[-1]
    assert float(last_row["metric_value"]) == 1_310_000.0

    # Deterministic tie-break for net_income: later filing_date should win.
    net_income = pit[
        (pit["instrument_id"] == sample_instrument) & (pit["metric_name"] == "net_income")
    ].sort_values("asof_date")
    assert len(net_income) > 0
    last_net = net_income.iloc[-1]
    assert float(last_net["metric_value"]) == 96_000.0


def test_edgar_qc_v2_outputs_are_consumable(
    tmp_workspace: dict[str, Path],
) -> None:
    paths = _build_edgar_pit_v2(tmp_workspace)
    qc_result = run_edgar_qc(
        fundamentals_pit_path=paths["fundamentals_pit"],
        ticker_cik_map_path=paths["ticker_cik_map"],
        submissions_raw_path=paths["submissions_raw"],
        output_dir=tmp_workspace["artifacts"] / "edgar_qc_v2",
        run_id="test_edgar_qc_v2",
    )

    assert qc_result.gate_status in {"PASS", "WARN"}
    assert qc_result.summary_path.exists()
    assert qc_result.row_level_path.exists()
    assert qc_result.failures_path.exists()
    assert qc_result.metrics_path.exists()
    assert qc_result.manifest_path.exists()

    summary = json.loads(qc_result.summary_path.read_text(encoding="utf-8"))
    assert summary["gate_status"] in {"PASS", "WARN"}
    assert int(summary["n_rows"]) > 0
