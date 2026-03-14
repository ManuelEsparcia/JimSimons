from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

# Allow direct script execution: `python simons_smallcap_swing/data/reference/build_reference.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simons_core.io.parquet_store import write_parquet
from simons_core.io.paths import reference_dir
from simons_core.schemas import assert_schema


@dataclass(frozen=True)
class BuildResult:
    trading_calendar: Path
    ticker_history_map: Path
    symbols_metadata: Path
    sector_industry_map: Path


def _build_trading_calendar() -> pd.DataFrame:
    sessions = pd.bdate_range("2026-01-02", "2026-03-31", freq="B")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(sessions),
            "is_session": True,
            "session_idx": range(1, len(sessions) + 1),
            "week": pd.to_datetime(sessions).isocalendar().week.astype("int64"),
            "month": pd.to_datetime(sessions).month.astype("int64"),
            "year": pd.to_datetime(sessions).year.astype("int64"),
        }
    )
    frame = frame.sort_values("date").reset_index(drop=True)
    return frame


def _build_ticker_history_map() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "instrument_id": "SIM0001",
                "ticker": "AALP",
                "start_date": "2019-01-01",
                "end_date": None,
                "is_active": True,
            },
            {
                "instrument_id": "SIM0002",
                "ticker": "BRVO",
                "start_date": "2018-06-01",
                "end_date": "2026-01-15",
                "is_active": False,
            },
            {
                "instrument_id": "SIM0002",
                "ticker": "BRVX",
                "start_date": "2026-01-16",
                "end_date": None,
                "is_active": True,
            },
            {
                "instrument_id": "SIM0003",
                "ticker": "CRWN",
                "start_date": "2020-03-01",
                "end_date": None,
                "is_active": True,
            },
            {
                "instrument_id": "SIM0004",
                "ticker": "DYNM",
                "start_date": "2017-05-15",
                "end_date": "2025-12-31",
                "is_active": False,
            },
            {
                "instrument_id": "SIM0005",
                "ticker": "ELMT",
                "start_date": "2021-09-01",
                "end_date": None,
                "is_active": True,
            },
        ]
    )
    frame["start_date"] = pd.to_datetime(frame["start_date"])
    frame["end_date"] = pd.to_datetime(frame["end_date"])
    frame["is_active"] = frame["is_active"].astype(bool)
    frame = frame.sort_values(["instrument_id", "start_date"]).reset_index(drop=True)
    return frame


def _build_symbols_metadata() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "instrument_id": "SIM0001",
                "ticker": "AALP",
                "name": "Alpha Tools Corp.",
                "exchange": "NASDAQ",
                "asset_type": "COMMON_STOCK",
                "currency": "USD",
                "primary_listing_flag": True,
                "country": "US",
            },
            {
                "instrument_id": "SIM0002",
                "ticker": "BRVX",
                "name": "Bravo Devices Inc.",
                "exchange": "NYSE",
                "asset_type": "COMMON_STOCK",
                "currency": "USD",
                "primary_listing_flag": True,
                "country": "US",
            },
            {
                "instrument_id": "SIM0003",
                "ticker": "CRWN",
                "name": "Crown Logistics Co.",
                "exchange": "NASDAQ",
                "asset_type": "COMMON_STOCK",
                "currency": "USD",
                "primary_listing_flag": True,
                "country": "US",
            },
            {
                "instrument_id": "SIM0004",
                "ticker": "DYNM",
                "name": "Dynamic Retail Group",
                "exchange": "AMEX",
                "asset_type": "COMMON_STOCK",
                "currency": "USD",
                "primary_listing_flag": True,
                "country": "US",
            },
            {
                "instrument_id": "SIM0005",
                "ticker": "ELMT",
                "name": "Element Materials Ltd.",
                "exchange": "NASDAQ",
                "asset_type": "COMMON_STOCK",
                "currency": "USD",
                "primary_listing_flag": True,
                "country": "US",
            },
        ]
    )
    frame["primary_listing_flag"] = frame["primary_listing_flag"].astype(bool)
    frame = frame.sort_values("instrument_id").reset_index(drop=True)
    return frame


def _build_sector_industry_map() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "instrument_id": "SIM0001",
                "sector": "Industrials",
                "industry": "Tools & Equipment",
                "start_date": "2019-01-01",
                "end_date": None,
            },
            {
                "instrument_id": "SIM0002",
                "sector": "Information Technology",
                "industry": "Electronic Equipment",
                "start_date": "2018-06-01",
                "end_date": None,
            },
            {
                "instrument_id": "SIM0003",
                "sector": "Industrials",
                "industry": "Air Freight & Logistics",
                "start_date": "2020-03-01",
                "end_date": None,
            },
            {
                "instrument_id": "SIM0004",
                "sector": "Consumer Discretionary",
                "industry": "Specialty Retail",
                "start_date": "2017-05-15",
                "end_date": "2025-12-31",
            },
            {
                "instrument_id": "SIM0005",
                "sector": "Materials",
                "industry": "Specialty Chemicals",
                "start_date": "2021-09-01",
                "end_date": None,
            },
        ]
    )
    frame["start_date"] = pd.to_datetime(frame["start_date"])
    frame["end_date"] = pd.to_datetime(frame["end_date"])
    frame = frame.sort_values(["instrument_id", "start_date"]).reset_index(drop=True)
    return frame


def build_reference_data(
    *,
    output_dir: str | Path | None = None,
    run_id: str = "reference_mvp_v1",
) -> BuildResult:
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else reference_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    trading_calendar = _build_trading_calendar()
    ticker_history_map = _build_ticker_history_map()
    symbols_metadata = _build_symbols_metadata()
    sector_industry_map = _build_sector_industry_map()

    assert_schema(trading_calendar, "reference_trading_calendar")
    assert_schema(ticker_history_map, "reference_ticker_history_map")
    assert_schema(symbols_metadata, "reference_symbols_metadata")
    assert_schema(sector_industry_map, "reference_sector_industry_map")

    calendar_path = write_parquet(
        trading_calendar,
        target_dir / "trading_calendar.parquet",
        schema_name="reference_trading_calendar",
        run_id=run_id,
    )
    history_path = write_parquet(
        ticker_history_map,
        target_dir / "ticker_history_map.parquet",
        schema_name="reference_ticker_history_map",
        run_id=run_id,
    )
    metadata_path = write_parquet(
        symbols_metadata,
        target_dir / "symbols_metadata.parquet",
        schema_name="reference_symbols_metadata",
        run_id=run_id,
    )
    sector_path = write_parquet(
        sector_industry_map,
        target_dir / "sector_industry_map.parquet",
        schema_name="reference_sector_industry_map",
        run_id=run_id,
    )

    return BuildResult(
        trading_calendar=calendar_path,
        ticker_history_map=history_path,
        symbols_metadata=metadata_path,
        sector_industry_map=sector_path,
    )


def main() -> None:
    result = build_reference_data()
    print("Reference data built:")
    print(f"- {result.trading_calendar}")
    print(f"- {result.ticker_history_map}")
    print(f"- {result.symbols_metadata}")
    print(f"- {result.sector_industry_map}")


if __name__ == "__main__":
    main()
