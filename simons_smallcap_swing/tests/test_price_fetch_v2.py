from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.price.fetch_prices import fetch_prices
from data.reference.build_reference import build_reference_data
from data.universe.build_universe import build_universe
from simons_core.io.parquet_store import read_parquet


def _prepare_reference_and_universe(tmp_workspace: dict[str, Path]) -> tuple[Path, Path]:
    reference_root = tmp_workspace["data"] / "reference"
    universe_root = tmp_workspace["data"] / "universe"
    build_reference_data(output_dir=reference_root, run_id="test_reference_price_fetch_v2")
    universe_result = build_universe(
        reference_root=reference_root,
        output_dir=universe_root,
        run_id="test_universe_price_fetch_v2",
    )
    return reference_root, universe_result.universe_history


def test_fetch_v2_local_csv_normalizes_schema_and_tracks_unmapped_rows(
    tmp_workspace: dict[str, Path],
) -> None:
    reference_root, universe_history_path = _prepare_reference_and_universe(tmp_workspace)
    price_root = tmp_workspace["data"] / "price"
    source_file = price_root / "source_prices.csv"
    price_root.mkdir(parents=True, exist_ok=True)

    local_df = pd.DataFrame(
        [
            {"Date": "2026-01-05", "Symbol": "AALP", "Open": 12.0, "High": 12.5, "Low": 11.8, "Close": 12.2, "Volume": 120000},
            {"Date": "2026-01-05", "Symbol": "AALP", "Open": 12.1, "High": 12.6, "Low": 11.9, "Close": 12.3, "Volume": 121000},
            {"Date": "2026-01-05", "Symbol": "CRWN", "Open": 21.0, "High": 21.9, "Low": 20.8, "Close": 21.3, "Volume": 98000},
            {"Date": "2026-01-05", "Symbol": "ZZZZ", "Open": 3.0, "High": 3.3, "Low": 2.9, "Close": 3.1, "Volume": 4000},
        ]
    )
    local_df.to_csv(source_file, index=False)

    result = fetch_prices(
        reference_root=reference_root,
        universe_history_path=universe_history_path,
        output_dir=price_root,
        ingestion_mode="local_file",
        local_source_path=source_file,
        allow_synthetic_fallback=False,
        run_id="test_fetch_v2_csv",
    )

    raw = read_parquet(result.raw_prices_path)
    assert result.source_mode == "local_file"
    assert len(raw) > 0
    assert {"date", "instrument_id", "ticker", "open", "high", "low", "close", "volume"}.issubset(raw.columns)
    assert (raw["source_mode"] == "local_file").all()
    assert (~raw["is_synthetic"]).all()
    assert not raw.duplicated(["date", "instrument_id"]).any()
    assert "ZZZZ" not in set(raw["ticker"].tolist())

    report = json.loads(result.ingestion_report_path.read_text(encoding="utf-8"))
    assert report["unmapped_rows"] >= 1
    assert report["duplicate_pk_dropped"] >= 1
    assert len(report["source_files"]) == 1


def test_fetch_v2_local_parquet_pit_ticker_matching_handles_ticker_change(
    tmp_workspace: dict[str, Path],
) -> None:
    reference_root, universe_history_path = _prepare_reference_and_universe(tmp_workspace)
    price_root = tmp_workspace["data"] / "price"
    source_file = price_root / "source_prices.parquet"
    price_root.mkdir(parents=True, exist_ok=True)

    local_df = pd.DataFrame(
        [
            {"trade_date": "2026-01-14", "symbol": "BRVO", "open": 30.0, "high": 30.9, "low": 29.8, "close": 30.4, "volume": 50000},
            {"trade_date": "2026-01-20", "symbol": "BRVX", "open": 31.0, "high": 31.8, "low": 30.6, "close": 31.2, "volume": 52000},
            {"trade_date": "2026-01-20", "symbol": "BRVO", "open": 31.5, "high": 31.7, "low": 31.1, "close": 31.2, "volume": 30000},
        ]
    )
    local_df.to_parquet(source_file, index=False)

    result = fetch_prices(
        reference_root=reference_root,
        universe_history_path=universe_history_path,
        output_dir=price_root,
        ingestion_mode="local_file",
        local_source_path=source_file,
        allow_synthetic_fallback=False,
        run_id="test_fetch_v2_parquet",
    )

    raw = read_parquet(result.raw_prices_path)
    sim2 = raw[raw["instrument_id"] == "SIM0002"].copy()
    assert len(sim2) == 2
    assert set(sim2["ticker"].tolist()) == {"BRVO", "BRVX"}
    assert set(pd.to_datetime(sim2["date"]).dt.strftime("%Y-%m-%d").tolist()) == {"2026-01-14", "2026-01-20"}

    report = json.loads(result.ingestion_report_path.read_text(encoding="utf-8"))
    assert report["unmapped_rows"] == 1
    assert report["resolved_mode"] == "local_file"


def test_fetch_v2_auto_mode_falls_back_to_synthetic_when_no_local_source(
    tmp_workspace: dict[str, Path],
) -> None:
    reference_root, universe_history_path = _prepare_reference_and_universe(tmp_workspace)
    price_root = tmp_workspace["data"] / "price"
    empty_source_dir = price_root / "source"
    empty_source_dir.mkdir(parents=True, exist_ok=True)

    result = fetch_prices(
        reference_root=reference_root,
        universe_history_path=universe_history_path,
        output_dir=price_root,
        ingestion_mode="auto",
        local_source_path=empty_source_dir,
        run_id="test_fetch_v2_auto_fallback",
    )

    raw = read_parquet(result.raw_prices_path)
    assert result.source_mode == "synthetic_fallback"
    assert len(raw) > 0
    assert raw["is_synthetic"].all()
    assert (raw["source_mode"] == "synthetic_fallback").all()
    assert not raw.duplicated(["date", "instrument_id"]).any()

    report = json.loads(result.ingestion_report_path.read_text(encoding="utf-8"))
    assert report["resolved_mode"] == "synthetic_fallback"
    assert report["fallback_reason"] == "local_source_missing_or_empty"


def test_fetch_v2_provider_stub_mode_is_optional_and_graceful(
    tmp_workspace: dict[str, Path],
) -> None:
    reference_root, universe_history_path = _prepare_reference_and_universe(tmp_workspace)
    price_root = tmp_workspace["data"] / "price"

    result = fetch_prices(
        reference_root=reference_root,
        universe_history_path=universe_history_path,
        output_dir=price_root,
        ingestion_mode="provider_stub",
        provider_name="demo_provider_stub",
        run_id="test_fetch_v2_provider_stub",
    )

    raw = read_parquet(result.raw_prices_path)
    assert result.source_mode == "provider_stub"
    assert len(raw) > 0
    assert raw["is_synthetic"].all()
    assert (raw["source_mode"] == "provider_stub").all()
    assert set(raw["source"].unique().tolist()) == {"demo_provider_stub"}
