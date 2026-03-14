from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

# Allow direct script execution: `python simons_smallcap_swing/data/universe/build_universe.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir, reference_dir, resolve_config_path
from simons_core.logging import get_logger
from simons_core.schemas import assert_schema

DEFAULT_ALLOWED_EXCHANGES: tuple[str, ...] = ("NYSE", "NASDAQ", "AMEX")
DEFAULT_ALLOWED_ASSET_TYPES: tuple[str, ...] = ("COMMON_STOCK",)
DEFAULT_ALLOWED_CURRENCIES: tuple[str, ...] = ("USD",)


@dataclass(frozen=True)
class UniverseBuildPolicy:
    allowed_exchanges: tuple[str, ...] = DEFAULT_ALLOWED_EXCHANGES
    allowed_asset_types: tuple[str, ...] = DEFAULT_ALLOWED_ASSET_TYPES
    allowed_currencies: tuple[str, ...] = DEFAULT_ALLOWED_CURRENCIES

    def as_dict(self) -> dict[str, tuple[str, ...]]:
        return {
            "allowed_exchanges": self.allowed_exchanges,
            "allowed_asset_types": self.allowed_asset_types,
            "allowed_currencies": self.allowed_currencies,
        }


@dataclass(frozen=True)
class UniverseBuildResult:
    universe_history: Path
    universe_current: Path
    row_count_history: int
    row_count_current: int
    n_sessions: int
    n_instruments: int
    config_hash: str


def _normalize_dates(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains non-parseable dates.")
    return parsed.dt.normalize()


def _policy_config_hash(policy: UniverseBuildPolicy) -> str:
    payload = json.dumps(policy.as_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_policy(config_path: str | Path | None = None) -> UniverseBuildPolicy:
    policy = UniverseBuildPolicy()
    config_candidate: Path | None = None
    if config_path is not None:
        config_candidate = Path(config_path)
    else:
        try:
            config_candidate = resolve_config_path("universe.yaml")
        except FileNotFoundError:
            config_candidate = None

    if config_candidate is None or not config_candidate.exists():
        return policy

    try:
        import yaml  # type: ignore
    except Exception:
        # Keep MVP robust even if YAML dependency is unavailable.
        return policy

    loaded: dict[str, Any] = {}
    with config_candidate.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}

    section = loaded.get("universe", {}) if isinstance(loaded, dict) else {}
    exchanges = section.get("include_exchanges", policy.allowed_exchanges)
    asset_types = section.get("security_types", policy.allowed_asset_types)

    project_section = loaded.get("project", {}) if isinstance(loaded, dict) else {}
    currency = project_section.get("currency", None)
    currencies: tuple[str, ...]
    if currency:
        currencies = (str(currency).upper(),)
    else:
        currencies = policy.allowed_currencies

    return UniverseBuildPolicy(
        allowed_exchanges=tuple(str(item).upper() for item in exchanges),
        allowed_asset_types=tuple(str(item).upper() for item in asset_types),
        allowed_currencies=tuple(str(item).upper() for item in currencies),
    )


def _read_reference_tables(
    ref_root: str | Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path(ref_root).expanduser().resolve() if ref_root else reference_dir()
    trading_calendar = read_parquet(base / "trading_calendar.parquet")
    ticker_history_map = read_parquet(base / "ticker_history_map.parquet")
    symbols_metadata = read_parquet(base / "symbols_metadata.parquet")
    sector_industry_map = read_parquet(base / "sector_industry_map.parquet")

    assert_schema(trading_calendar, "reference_trading_calendar")
    assert_schema(ticker_history_map, "reference_ticker_history_map")
    assert_schema(symbols_metadata, "reference_symbols_metadata")
    assert_schema(sector_industry_map, "reference_sector_industry_map")
    return trading_calendar, ticker_history_map, symbols_metadata, sector_industry_map


def _extract_sessions(
    trading_calendar: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    sessions = trading_calendar.loc[trading_calendar["is_session"].astype(bool), ["date"]].copy()
    sessions["date"] = _normalize_dates(sessions["date"], column="date")

    if start_date is not None:
        start_ts = pd.Timestamp(start_date).normalize()
        sessions = sessions[sessions["date"] >= start_ts]
    if end_date is not None:
        end_ts = pd.Timestamp(end_date).normalize()
        sessions = sessions[sessions["date"] <= end_ts]

    sessions = sessions.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    if sessions.empty:
        raise ValueError("No trading sessions available after applying date filters.")
    return sessions


def _validate_ticker_history_map(ticker_history_map: pd.DataFrame) -> pd.DataFrame:
    frame = ticker_history_map.copy()
    frame["start_date"] = _normalize_dates(frame["start_date"], column="start_date")
    frame["end_date"] = pd.to_datetime(frame["end_date"], errors="coerce").dt.normalize()

    invalid_interval = frame["end_date"].notna() & (frame["end_date"] < frame["start_date"])
    if invalid_interval.any():
        n_bad = int(invalid_interval.sum())
        raise ValueError(f"ticker_history_map contains {n_bad} rows with end_date < start_date.")

    return frame.sort_values(["instrument_id", "start_date", "ticker"]).reset_index(drop=True)


def _expand_ticker_mapping_by_session(
    sessions: pd.DataFrame,
    ticker_history_map: pd.DataFrame,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for row in ticker_history_map.itertuples(index=False):
        start = row.start_date
        end = row.end_date
        mask = sessions["date"] >= start
        if pd.notna(end):
            mask &= sessions["date"] <= end
        if not mask.any():
            continue
        chunk = sessions.loc[mask, ["date"]].copy()
        chunk["instrument_id"] = row.instrument_id
        chunk["ticker"] = row.ticker
        pieces.append(chunk)

    if not pieces:
        raise ValueError("Ticker history map produced no active instrument/date rows.")

    panel = pd.concat(pieces, ignore_index=True)
    duplicated_pk = panel.duplicated(["date", "instrument_id"], keep=False)
    if duplicated_pk.any():
        sample = panel.loc[duplicated_pk, ["date", "instrument_id", "ticker"]].head(5)
        raise ValueError(
            "Overlapping ticker intervals generated duplicate (date, instrument_id) rows. "
            f"Sample:\n{sample.to_string(index=False)}"
        )

    return panel.sort_values(["date", "instrument_id"]).reset_index(drop=True)


def _apply_metadata_and_eligibility(
    panel: pd.DataFrame,
    symbols_metadata: pd.DataFrame,
    *,
    policy: UniverseBuildPolicy,
) -> pd.DataFrame:
    metadata = symbols_metadata[
        ["instrument_id", "exchange", "asset_type", "currency", "primary_listing_flag"]
    ].drop_duplicates(subset=["instrument_id"])

    merged = panel.merge(metadata, on="instrument_id", how="left")
    critical_cols = ["exchange", "asset_type", "currency", "primary_listing_flag"]
    missing_critical = merged[critical_cols].isna().any(axis=1)
    if missing_critical.any():
        sample = merged.loc[missing_critical, ["date", "instrument_id", "ticker"]].head(5)
        raise ValueError(
            "Missing symbols metadata for universe rows. "
            f"Sample:\n{sample.to_string(index=False)}"
        )

    merged["exchange"] = merged["exchange"].astype(str).str.upper()
    merged["asset_type"] = merged["asset_type"].astype(str).str.upper()
    merged["currency"] = merged["currency"].astype(str).str.upper()

    merged["is_eligible"] = (
        merged["exchange"].isin(policy.allowed_exchanges)
        & merged["asset_type"].isin(policy.allowed_asset_types)
        & merged["currency"].isin(policy.allowed_currencies)
        & merged["primary_listing_flag"].astype(bool)
    )
    return merged


def _apply_sector_mapping(
    universe: pd.DataFrame,
    sector_industry_map: pd.DataFrame,
) -> pd.DataFrame:
    frame = universe.copy()
    frame["sector"] = "UNKNOWN"
    frame["industry"] = "UNKNOWN"

    sector_map = sector_industry_map.copy()
    sector_map["start_date"] = _normalize_dates(sector_map["start_date"], column="start_date")
    sector_map["end_date"] = pd.to_datetime(sector_map["end_date"], errors="coerce").dt.normalize()

    invalid_interval = sector_map["end_date"].notna() & (sector_map["end_date"] < sector_map["start_date"])
    if invalid_interval.any():
        raise ValueError("sector_industry_map contains rows with end_date < start_date.")

    for row in sector_map.itertuples(index=False):
        mask = (frame["instrument_id"] == row.instrument_id) & (frame["date"] >= row.start_date)
        if pd.notna(row.end_date):
            mask &= frame["date"] <= row.end_date
        if mask.any():
            frame.loc[mask, "sector"] = row.sector
            frame.loc[mask, "industry"] = row.industry
    return frame


def build_universe(
    *,
    reference_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    run_id: str = "universe_mvp_v1",
    config_path: str | Path | None = None,
) -> UniverseBuildResult:
    logger = get_logger("data.universe.build_universe")
    policy = _load_policy(config_path)
    config_hash = _policy_config_hash(policy)

    trading_calendar, ticker_history_map, symbols_metadata, sector_industry_map = _read_reference_tables(
        reference_root
    )
    sessions = _extract_sessions(trading_calendar, start_date=start_date, end_date=end_date)
    ticker_history = _validate_ticker_history_map(ticker_history_map)

    base_panel = _expand_ticker_mapping_by_session(sessions, ticker_history)
    with_metadata = _apply_metadata_and_eligibility(base_panel, symbols_metadata, policy=policy)
    universe = _apply_sector_mapping(with_metadata, sector_industry_map)

    built_ts_utc = datetime.now(UTC).isoformat()
    universe["run_id"] = run_id
    universe["config_hash"] = config_hash
    universe["built_ts_utc"] = built_ts_utc

    ordered_columns = [
        "date",
        "instrument_id",
        "ticker",
        "is_eligible",
        "exchange",
        "asset_type",
        "currency",
        "sector",
        "industry",
        "run_id",
        "config_hash",
        "built_ts_utc",
    ]
    universe = universe[ordered_columns].sort_values(["date", "instrument_id"]).reset_index(drop=True)

    if universe.empty:
        raise ValueError("Universe history is empty after applying MVP eligibility policy.")

    critical_non_null = ["date", "instrument_id", "ticker", "exchange", "asset_type", "currency"]
    if universe[critical_non_null].isna().any().any():
        raise ValueError("Universe history contains nulls in critical columns.")

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (data_dir() / "universe")
    target_dir.mkdir(parents=True, exist_ok=True)

    history_path = write_parquet(
        universe,
        target_dir / "universe_history.parquet",
        schema_name="universe_history_mvp",
        run_id=run_id,
    )

    last_date = universe["date"].max()
    current = universe[(universe["date"] == last_date) & (universe["is_eligible"])].copy()
    if current.empty:
        raise ValueError(f"No eligible instruments in last session ({last_date.date()}).")

    current_path = write_parquet(
        current,
        target_dir / "universe_current.parquet",
        schema_name="universe_current_mvp",
        run_id=run_id,
    )

    logger.info(
        "universe_built",
        run_id=run_id,
        row_count_history=int(len(universe)),
        row_count_current=int(len(current)),
        n_sessions=int(universe["date"].nunique()),
        n_instruments=int(universe["instrument_id"].nunique()),
        output_history=str(history_path),
        output_current=str(current_path),
    )

    return UniverseBuildResult(
        universe_history=history_path,
        universe_current=current_path,
        row_count_history=int(len(universe)),
        row_count_current=int(len(current)),
        n_sessions=int(universe["date"].nunique()),
        n_instruments=int(universe["instrument_id"].nunique()),
        config_hash=config_hash,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build MVP historical PIT universe from reference data.")
    parser.add_argument("--reference-root", type=str, default=None, help="Path to reference parquet folder.")
    parser.add_argument("--output-dir", type=str, default=None, help="Path where universe artifacts are written.")
    parser.add_argument("--start-date", type=str, default=None, help="Inclusive start date filter (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=None, help="Inclusive end date filter (YYYY-MM-DD).")
    parser.add_argument("--config-path", type=str, default=None, help="Optional universe YAML config path.")
    parser.add_argument("--run-id", type=str, default="universe_mvp_v1", help="Run identifier.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = build_universe(
        reference_root=args.reference_root,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        run_id=args.run_id,
        config_path=args.config_path,
    )
    print("Universe artifacts built:")
    print(f"- history: {result.universe_history}")
    print(f"- current: {result.universe_current}")
    print(f"- rows history: {result.row_count_history}")
    print(f"- rows current: {result.row_count_current}")


if __name__ == "__main__":
    main()
