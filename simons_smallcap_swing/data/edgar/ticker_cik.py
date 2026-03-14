from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

# Allow direct script execution: `python simons_smallcap_swing/data/edgar/ticker_cik.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir, reference_dir
from simons_core.logging import get_logger
from simons_core.schemas import assert_schema


@dataclass(frozen=True)
class TickerCIKResult:
    ticker_cik_map_path: Path
    row_count: int
    n_instruments: int
    config_hash: str


def _normalize_dates(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains non-parseable dates.")
    return parsed.dt.normalize()


def _stable_cik(instrument_id: str) -> str:
    # SEC CIKs are typically shown as zero-padded 10-digit identifiers.
    value = int(hashlib.sha256(instrument_id.encode("utf-8")).hexdigest()[:12], 16) % 10_000_000_000
    return f"{value:010d}"


def _config_hash(version: str = "ticker_cik_mvp_v1") -> str:
    payload = {"version": version, "mode": "deterministic_from_reference_universe"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_inputs(
    *,
    reference_root: str | Path | None,
    universe_history_path: str | Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ref_base = Path(reference_root).expanduser().resolve() if reference_root else reference_dir()
    universe_source = (
        Path(universe_history_path).expanduser().resolve()
        if universe_history_path
        else data_dir() / "universe" / "universe_history.parquet"
    )

    ticker_history_map = read_parquet(ref_base / "ticker_history_map.parquet")
    symbols_metadata = read_parquet(ref_base / "symbols_metadata.parquet")
    universe_history = read_parquet(universe_source)

    assert_schema(ticker_history_map, "reference_ticker_history_map")
    assert_schema(symbols_metadata, "reference_symbols_metadata")

    required_universe = {"date", "instrument_id", "ticker"}
    missing = sorted(required_universe - set(universe_history.columns))
    if missing:
        raise ValueError(f"universe_history is missing required columns: {missing}")

    return ticker_history_map, symbols_metadata, universe_history


def _validate_intervals(ticker_history_map: pd.DataFrame) -> pd.DataFrame:
    frame = ticker_history_map.copy()
    frame["start_date"] = _normalize_dates(frame["start_date"], column="start_date")
    frame["end_date"] = pd.to_datetime(frame["end_date"], errors="coerce").dt.normalize()
    invalid = frame["end_date"].notna() & (frame["end_date"] < frame["start_date"])
    if invalid.any():
        raise ValueError("ticker_history_map contains rows with end_date < start_date.")
    return frame.sort_values(["instrument_id", "start_date", "ticker"]).reset_index(drop=True)


def build_ticker_cik_map(
    *,
    reference_root: str | Path | None = None,
    universe_history_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    run_id: str = "ticker_cik_mvp_v1",
) -> TickerCIKResult:
    logger = get_logger("data.edgar.ticker_cik")
    ticker_history_map, symbols_metadata, universe_history = _load_inputs(
        reference_root=reference_root,
        universe_history_path=universe_history_path,
    )
    ticker_history_map = _validate_intervals(ticker_history_map)

    universe_instruments = set(universe_history["instrument_id"].astype(str).unique().tolist())
    metadata_instruments = set(symbols_metadata["instrument_id"].astype(str).unique().tolist())
    missing_metadata = sorted(universe_instruments - metadata_instruments)
    if missing_metadata:
        raise ValueError(
            "Some universe instruments are missing in symbols_metadata. "
            f"Sample: {missing_metadata[:10]}"
        )

    mapping = ticker_history_map[
        ticker_history_map["instrument_id"].astype(str).isin(universe_instruments)
    ][["instrument_id", "ticker", "start_date", "end_date", "is_active"]].copy()
    if mapping.empty:
        raise ValueError("No ticker history rows matched universe instruments.")

    mapping["instrument_id"] = mapping["instrument_id"].astype(str)
    mapping["ticker"] = mapping["ticker"].astype(str)
    mapping["cik"] = mapping["instrument_id"].map(_stable_cik)
    mapping["start_date"] = _normalize_dates(mapping["start_date"], column="start_date")
    mapping["end_date"] = pd.to_datetime(mapping["end_date"], errors="coerce").dt.normalize()
    mapping["is_active"] = mapping["is_active"].astype(bool)

    duplicated_pk = mapping.duplicated(["instrument_id", "ticker", "start_date"], keep=False)
    if duplicated_pk.any():
        raise ValueError("ticker_cik_map has duplicate (instrument_id, ticker, start_date).")

    config_hash = _config_hash()
    mapping["run_id"] = run_id
    mapping["config_hash"] = config_hash
    mapping["built_ts_utc"] = datetime.now(UTC).isoformat()
    mapping = mapping[
        [
            "instrument_id",
            "ticker",
            "cik",
            "start_date",
            "end_date",
            "is_active",
            "run_id",
            "config_hash",
            "built_ts_utc",
        ]
    ].sort_values(["instrument_id", "start_date", "ticker"]).reset_index(drop=True)

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (data_dir() / "edgar")
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = write_parquet(
        mapping,
        target_dir / "ticker_cik_map.parquet",
        schema_name="edgar_ticker_cik_map_mvp",
        run_id=run_id,
    )

    logger.info(
        "ticker_cik_map_built",
        run_id=run_id,
        row_count=int(len(mapping)),
        n_instruments=int(mapping["instrument_id"].nunique()),
        output_path=str(output_path),
    )

    return TickerCIKResult(
        ticker_cik_map_path=output_path,
        row_count=int(len(mapping)),
        n_instruments=int(mapping["instrument_id"].nunique()),
        config_hash=config_hash,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build MVP PIT-safe ticker/CIK mapping.")
    parser.add_argument("--reference-root", type=str, default=None)
    parser.add_argument("--universe-history-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default="ticker_cik_mvp_v1")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = build_ticker_cik_map(
        reference_root=args.reference_root,
        universe_history_path=args.universe_history_path,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print("Ticker/CIK map built:")
    print(f"- path: {result.ticker_cik_map_path}")
    print(f"- rows: {result.row_count}")
    print(f"- instruments: {result.n_instruments}")


if __name__ == "__main__":
    main()
