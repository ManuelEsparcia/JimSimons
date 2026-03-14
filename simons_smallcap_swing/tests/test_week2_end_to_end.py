from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_week1_mvp import run_week1_mvp
from run_week2_data_upgrade import run_week2_data_upgrade
from simons_core.io.parquet_store import read_parquet


def _write_local_submissions_source(local_dir: Path, ciks: list[str]) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for idx, cik in enumerate(ciks):
        accession_a = f"0000000000-26-500{idx + 1}0"
        accession_b = f"0000000000-26-500{idx + 1}1"
        payload = {
            "cik": cik,
            "filings": {
                "recent": {
                    "accessionNumber": [accession_a, accession_b],
                    "filingDate": ["2026-01-20", "2026-03-10"],
                    "acceptanceDateTime": ["2026-01-20T21:15:00Z", "2026-03-10T22:10:00Z"],
                    "form": ["10-K", "10-Q"],
                    "primaryDocument": ["annual.htm", "quarterly.htm"],
                    "primaryDocDescription": ["Annual report", "Quarterly report"],
                    "reportDate": ["2025-12-31", "2026-03-31"],
                    "isXBRL": [1, 1],
                    "isInlineXBRL": [1, 1],
                }
            },
        }
        (local_dir / f"CIK{cik}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _write_local_companyfacts_source(local_dir: Path, ciks: list[str]) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for idx, cik in enumerate(ciks):
        accession = f"0000000000-26-500{idx + 1}0"
        payload = {
            "cik": cik,
            "entityName": f"Week2 Issuer {idx + 1}",
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2025-12-31",
                                    "val": 90_000_000 + idx * 5_000_000,
                                    "accn": accession,
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2026-01-20",
                                    "acceptance": "2026-01-20T21:15:00Z",
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
                                    "val": 8_500_000 + idx * 500_000,
                                    "accn": accession,
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2026-01-20",
                                    "acceptance": "2026-01-20T21:20:00Z",
                                    "frame": "CY2025",
                                }
                            ]
                        }
                    },
                    "Assets": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2025-12-31",
                                    "val": 240_000_000 + idx * 10_000_000,
                                    "accn": accession,
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2026-01-20",
                                    "acceptance": "2026-01-20T21:25:00Z",
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
                                    "val": 24_000_000 + idx * 1_000_000,
                                    "accn": accession,
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2026-01-20",
                                    "acceptance": "2026-01-20T21:30:00Z",
                                    "frame": "CY2025",
                                }
                            ]
                        }
                    }
                },
            },
        }
        (local_dir / f"CIK{cik}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def test_week2_runner_end_to_end_smoke(tmp_workspace: dict[str, Path]) -> None:
    week1 = run_week1_mvp(
        run_prefix="test_week2_bootstrap",
        data_root=tmp_workspace["data"],
    )
    ticker_cik_map = read_parquet(week1.artifacts["ticker_cik_map"])
    target_ciks = sorted(set(ticker_cik_map["cik"].astype(str).tolist()))[:2]
    assert target_ciks

    submissions_source = tmp_workspace["data"] / "edgar_source" / "submissions_week2"
    companyfacts_source = tmp_workspace["data"] / "edgar_source" / "companyfacts_week2"
    _write_local_submissions_source(submissions_source, target_ciks)
    _write_local_companyfacts_source(companyfacts_source, target_ciks)

    result = run_week2_data_upgrade(
        run_prefix="test_week2_e2e",
        data_root=tmp_workspace["data"],
        submissions_ingestion_mode="local_file",
        submissions_local_source=submissions_source,
        companyfacts_ingestion_mode="local_file",
        companyfacts_local_source=companyfacts_source,
        force_rebuild_edgar=True,
        allow_fail_gates=True,
    )

    expected_paths = {
        "corporate_actions": result.artifacts["corporate_actions"],
        "adjusted_prices": result.artifacts["adjusted_prices"],
        "survivorship_summary": result.artifacts["survivorship_summary"],
        "market_proxies": result.artifacts["market_proxies"],
        "fundamentals_pit": result.artifacts["fundamentals_pit"],
        "price_qc_summary": result.artifacts["price_qc_summary"],
        "edgar_qc_summary": result.artifacts["edgar_qc_summary"],
    }
    for _, path in expected_paths.items():
        assert path.exists()
        assert path.stat().st_size > 0

    corporate_actions = read_parquet(expected_paths["corporate_actions"])
    adjusted_prices = read_parquet(expected_paths["adjusted_prices"])
    market_proxies = read_parquet(expected_paths["market_proxies"])
    fundamentals_pit = read_parquet(expected_paths["fundamentals_pit"])

    assert len(corporate_actions) > 0
    assert len(adjusted_prices) > 0
    assert len(market_proxies) > 0
    assert len(fundamentals_pit) > 0

    assert (
        pd.to_datetime(fundamentals_pit["acceptance_ts"], utc=True)
        <= pd.to_datetime(fundamentals_pit["asof_date"], utc=True)
    ).all()
    assert fundamentals_pit["visibility_rule"].astype(str).str.contains("acceptance_ts<=asof_date").all()

    for gate_name in ("price_qc", "survivorship", "edgar_qc"):
        assert gate_name in result.gates
        assert result.gates[gate_name] in {"PASS", "WARN", "FAIL"}

    price_qc_summary = json.loads(expected_paths["price_qc_summary"].read_text(encoding="utf-8"))
    edgar_qc_summary = json.loads(expected_paths["edgar_qc_summary"].read_text(encoding="utf-8"))
    survivorship_summary = json.loads(expected_paths["survivorship_summary"].read_text(encoding="utf-8"))
    assert price_qc_summary["gate_status"] in {"PASS", "WARN", "FAIL"}
    assert edgar_qc_summary["gate_status"] in {"PASS", "WARN", "FAIL"}
    assert survivorship_summary["gate_status"] in {"PASS", "WARN", "FAIL"}

    # Compatibility with Week 1 bootstrap artifacts.
    assert week1.manifest_path.exists()
    assert result.artifacts["week1_universe_history"] == week1.artifacts["universe_history"]
    assert result.artifacts["week1_universe_current"] == week1.artifacts["universe_current"]

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_prefix"] == "test_week2_e2e"
    assert set(manifest["gates"]) == {"price_qc", "survivorship", "edgar_qc"}
