from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import pandas as pd

# Allow direct script execution: `python simons_smallcap_swing/data/price/fetch_prices.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir, reference_dir
from simons_core.logging import get_logger
from simons_core.schemas import assert_schema


@dataclass(frozen=True)
class FetchPricesResult:
    raw_prices_path: Path
    ingestion_report_path: Path
    row_count: int
    n_instruments: int
    n_sessions: int
    config_hash: str
    source_mode: str
    source_files: tuple[str, ...]
    discarded_rows: int
    unmapped_rows: int


LOCAL_SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("date", "trade_date", "datetime", "timestamp"),
    "ticker": ("ticker", "symbol", "ric"),
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c", "price_close"),
    "volume": ("volume", "vol", "v"),
    "adj_close_raw": ("adj_close", "adjclose", "adjusted_close"),
    "vendor": ("vendor", "source_vendor"),
}


def _normalize_dates(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains non-parseable dates.")
    return parsed.dt.normalize()


def _stable_unit(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    value = int(digest, 16)
    return value / float(16**16 - 1)


def _stable_uniform(key: str, lo: float, hi: float) -> float:
    return lo + (hi - lo) * _stable_unit(key)


def _config_hash(payload: dict[str, Any]) -> str:
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

    trading_calendar = read_parquet(ref_base / "trading_calendar.parquet")
    ticker_history_map = read_parquet(ref_base / "ticker_history_map.parquet")
    universe_history = read_parquet(universe_source)

    assert_schema(trading_calendar, "reference_trading_calendar")
    assert_schema(ticker_history_map, "reference_ticker_history_map")

    required_universe_cols = {"date", "instrument_id", "ticker", "is_eligible"}
    missing = sorted(required_universe_cols - set(universe_history.columns))
    if missing:
        raise ValueError(
            f"universe_history is missing required columns: {missing}. "
            f"Path: {universe_source}"
        )

    return trading_calendar, ticker_history_map, universe_history


def _build_eligible_panel(
    universe_history: pd.DataFrame,
    trading_calendar: pd.DataFrame,
) -> pd.DataFrame:
    frame = universe_history.copy()
    frame["date"] = _normalize_dates(frame["date"], column="date")
    frame = frame[frame["is_eligible"].astype(bool)].copy()
    if frame.empty:
        raise ValueError("No eligible rows found in universe_history.")

    sessions = _normalize_dates(
        trading_calendar.loc[trading_calendar["is_session"].astype(bool), "date"],
        column="date",
    )
    valid_sessions = set(sessions.tolist())
    off_calendar = frame.loc[~frame["date"].isin(valid_sessions), ["date"]].drop_duplicates()
    if not off_calendar.empty:
        raise ValueError(
            "universe_history contains dates not present in trading calendar. "
            f"Sample:\n{off_calendar.head(10).to_string(index=False)}"
        )

    duplicated_pk = frame.duplicated(["date", "instrument_id"], keep=False)
    if duplicated_pk.any():
        sample = frame.loc[duplicated_pk, ["date", "instrument_id", "ticker"]].head(10)
        raise ValueError(
            "universe_history has duplicate (date, instrument_id) rows. "
            f"Sample:\n{sample.to_string(index=False)}"
        )

    out = frame[["date", "instrument_id", "ticker"]].copy()
    out["instrument_id"] = out["instrument_id"].astype(str)
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    return out.sort_values(["instrument_id", "date"]).reset_index(drop=True)


def _validate_ticker_pit(
    frame: pd.DataFrame,
    ticker_history_map: pd.DataFrame,
) -> None:
    mapping = ticker_history_map.copy()
    mapping["start_date"] = _normalize_dates(mapping["start_date"], column="start_date")
    mapping["end_date"] = pd.to_datetime(mapping["end_date"], errors="coerce").dt.normalize()

    invalid_interval = mapping["end_date"].notna() & (mapping["end_date"] < mapping["start_date"])
    if invalid_interval.any():
        raise ValueError("ticker_history_map contains invalid intervals (end_date < start_date).")

    check = frame.reset_index(drop=True).copy()
    check["__row_id"] = check.index
    merged = check.merge(
        mapping[["instrument_id", "ticker", "start_date", "end_date"]],
        on=["instrument_id", "ticker"],
        how="left",
    )
    merged["valid"] = (
        merged["start_date"].notna()
        & (merged["date"] >= merged["start_date"])
        & (merged["end_date"].isna() | (merged["date"] <= merged["end_date"]))
    )
    valid_by_row = merged.groupby("__row_id", as_index=True)["valid"].any()
    invalid_row_ids = valid_by_row.index[~valid_by_row]
    if len(invalid_row_ids) > 0:
        bad = check.loc[
            check["__row_id"].isin(invalid_row_ids),
            ["date", "instrument_id", "ticker"],
        ].head(10)
        raise ValueError(
            "Rows contain ticker/date pairs not valid in ticker_history_map. "
            f"Sample:\n{bad.to_string(index=False)}"
        )


def _discover_local_files(local_source_path: str | Path | None) -> list[Path]:
    if local_source_path is None:
        return []

    candidate = Path(local_source_path).expanduser()
    if not candidate.exists():
        return []
    if candidate.is_file():
        return [candidate.resolve()]
    if not candidate.is_dir():
        return []

    files: list[Path] = []
    for ext in ("*.csv", "*.parquet"):
        files.extend(sorted(candidate.glob(ext)))
    return [path.resolve() for path in files]


def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [
        str(col)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        for col in normalized.columns
    ]
    return normalized


def _pick_column(columns: set[str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in columns:
            return alias
    return None


def _load_local_prices(files: list[Path]) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[pd.DataFrame] = []
    stats = {
        "input_rows": 0,
        "parsed_rows": 0,
        "discarded_invalid_rows": 0,
    }

    for file_path in files:
        if file_path.suffix.lower() == ".csv":
            raw = pd.read_csv(file_path)
        elif file_path.suffix.lower() == ".parquet":
            raw = pd.read_parquet(file_path)
        else:
            continue

        stats["input_rows"] += int(len(raw))
        raw = _standardize_columns(raw)
        cols = set(raw.columns)
        selected: dict[str, str] = {}
        for canonical, aliases in LOCAL_SOURCE_ALIASES.items():
            picked = _pick_column(cols, aliases)
            if picked is not None:
                selected[canonical] = picked

        required = ("date", "ticker", "open", "high", "low", "close", "volume")
        if any(req not in selected for req in required):
            stats["discarded_invalid_rows"] += int(len(raw))
            continue

        parsed = pd.DataFrame(
            {
                "date": raw[selected["date"]],
                "ticker": raw[selected["ticker"]],
                "open": raw[selected["open"]],
                "high": raw[selected["high"]],
                "low": raw[selected["low"]],
                "close": raw[selected["close"]],
                "volume": raw[selected["volume"]],
            }
        )
        if "adj_close_raw" in selected:
            parsed["adj_close_raw"] = raw[selected["adj_close_raw"]]
        if "vendor" in selected:
            parsed["vendor"] = raw[selected["vendor"]]
        parsed["source_file"] = str(file_path)
        parsed["__source_row"] = range(len(parsed))
        rows.append(parsed)

    if not rows:
        return pd.DataFrame(), stats

    local = pd.concat(rows, ignore_index=True)
    local["date"] = pd.to_datetime(local["date"], errors="coerce").dt.normalize()
    local["ticker"] = local["ticker"].astype(str).str.upper().str.strip()
    for column in ("open", "high", "low", "close", "volume"):
        local[column] = pd.to_numeric(local[column], errors="coerce")
    if "adj_close_raw" in local.columns:
        local["adj_close_raw"] = pd.to_numeric(local["adj_close_raw"], errors="coerce")

    invalid_required = local[
        ["date", "ticker", "open", "high", "low", "close", "volume"]
    ].isna().any(axis=1)
    stats["discarded_invalid_rows"] += int(invalid_required.sum())
    local = local[~invalid_required].copy()

    stats["parsed_rows"] = int(len(local))
    return local.reset_index(drop=True), stats


def _map_local_to_panel(
    local: pd.DataFrame,
    eligible_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if local.empty:
        return pd.DataFrame(), {"unmapped_rows": 0, "duplicate_pk_dropped": 0}

    panel = eligible_panel[["date", "instrument_id", "ticker"]].copy()
    duplicated_date_ticker = panel.duplicated(["date", "ticker"], keep=False)
    if duplicated_date_ticker.any():
        sample = panel.loc[duplicated_date_ticker, ["date", "ticker", "instrument_id"]].head(10)
        raise ValueError(
            "Ambiguous universe mapping by (date, ticker). "
            f"Sample:\n{sample.to_string(index=False)}"
        )

    merged = local.merge(
        panel,
        on=["date", "ticker"],
        how="left",
    )
    unmapped_rows = int(merged["instrument_id"].isna().sum())
    mapped = merged[merged["instrument_id"].notna()].copy()
    if mapped.empty:
        return pd.DataFrame(), {"unmapped_rows": unmapped_rows, "duplicate_pk_dropped": 0}

    mapped["instrument_id"] = mapped["instrument_id"].astype(str)
    mapped.sort_values(["date", "instrument_id", "source_file", "__source_row"], inplace=True)
    duplicated_pk = mapped.duplicated(["date", "instrument_id"], keep="last")
    duplicate_pk_dropped = int(duplicated_pk.sum())
    mapped = mapped[~duplicated_pk].copy()

    out = mapped[
        [
            "date",
            "instrument_id",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source_file",
        ]
    ].copy()
    if "adj_close_raw" in mapped.columns:
        out["adj_close_raw"] = mapped["adj_close_raw"]
    if "vendor" in mapped.columns:
        out["vendor"] = mapped["vendor"]

    return out.reset_index(drop=True), {
        "unmapped_rows": unmapped_rows,
        "duplicate_pk_dropped": duplicate_pk_dropped,
    }


def _generate_deterministic_prices(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for instrument_id, sub in frame.groupby("instrument_id", sort=True):
        instrument_frame = sub.sort_values("date").reset_index(drop=True)
        base_price = _stable_uniform(f"{instrument_id}:base_price", 8.0, 95.0)
        base_volume = _stable_uniform(f"{instrument_id}:base_volume", 120_000.0, 2_400_000.0)
        phase = _stable_uniform(f"{instrument_id}:phase", 0.0, 2.0 * math.pi)
        drift = _stable_uniform(f"{instrument_id}:drift", -0.0003, 0.0012)

        for idx, row in enumerate(instrument_frame.itertuples(index=False), start=1):
            date_key = pd.Timestamp(row.date).strftime("%Y-%m-%d")
            seasonal = 0.018 * math.sin(idx / 7.0 + phase)
            noise = _stable_uniform(f"{instrument_id}:{date_key}:close_noise", -0.010, 0.010)
            growth = max(0.22, 1.0 + drift * idx + seasonal + noise)
            close = base_price * growth

            open_noise = _stable_uniform(f"{instrument_id}:{date_key}:open_noise", -0.015, 0.015)
            open_price = close * (1.0 + open_noise)
            open_price = max(0.01, open_price)

            spread_up = _stable_uniform(f"{instrument_id}:{date_key}:high_spread", 0.001, 0.030)
            spread_down = _stable_uniform(f"{instrument_id}:{date_key}:low_spread", 0.001, 0.030)
            high = max(open_price, close) * (1.0 + spread_up)
            low = min(open_price, close) * (1.0 - spread_down)
            low = max(0.01, low)

            vol_shock = _stable_uniform(f"{instrument_id}:{date_key}:vol_noise", -0.25, 0.25)
            vol_seasonal = 0.30 * math.sin(idx / 5.0 + phase / 2.0)
            volume = int(max(0, base_volume * max(0.05, 1.0 + vol_shock + vol_seasonal)))

            rows.append(
                {
                    "date": pd.Timestamp(row.date),
                    "instrument_id": row.instrument_id,
                    "ticker": row.ticker,
                    "open": round(float(open_price), 4),
                    "high": round(float(high), 4),
                    "low": round(float(low), 4),
                    "close": round(float(close), 4),
                    "volume": volume,
                }
            )

    raw_prices = pd.DataFrame(rows).sort_values(["date", "instrument_id"]).reset_index(drop=True)
    return raw_prices


def _validate_raw_output(
    raw_prices: pd.DataFrame,
    *,
    trading_calendar: pd.DataFrame,
    ticker_history_map: pd.DataFrame,
) -> None:
    if raw_prices.empty:
        raise ValueError("raw_prices is empty.")

    duplicated_pk = raw_prices.duplicated(["date", "instrument_id"], keep=False)
    if duplicated_pk.any():
        raise ValueError("raw_prices has duplicate (date, instrument_id).")

    critical = ["date", "instrument_id", "ticker", "open", "high", "low", "close", "volume"]
    if raw_prices[critical].isna().any().any():
        raise ValueError("raw_prices has nulls in critical columns.")

    bad_ohlc = (
        (raw_prices["high"] < raw_prices[["open", "close", "low"]].max(axis=1))
        | (raw_prices["low"] > raw_prices[["open", "close", "high"]].min(axis=1))
    )
    if bad_ohlc.any():
        raise ValueError("raw_prices violates OHLC geometry.")

    sessions = _normalize_dates(
        trading_calendar.loc[trading_calendar["is_session"].astype(bool), "date"],
        column="date",
    )
    off_calendar = ~raw_prices["date"].isin(set(sessions.tolist()))
    if off_calendar.any():
        raise ValueError("raw_prices contains dates outside trading_calendar sessions.")

    _validate_ticker_pit(raw_prices[["date", "instrument_id", "ticker"]], ticker_history_map)


def _write_ingestion_report(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def fetch_prices(
    *,
    reference_root: str | Path | None = None,
    universe_history_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    run_id: str = "price_fetch_mvp_v1",
    frequency: str = "1d",
    source: str = "synthetic_mvp_v2",
    ingestion_mode: str = "auto",
    local_source_path: str | Path | None = None,
    allow_synthetic_fallback: bool = True,
    provider_name: str = "provider_stub",
) -> FetchPricesResult:
    logger = get_logger("data.price.fetch_prices")
    generation_version = "fetch_prices_v2"

    trading_calendar, ticker_history_map, universe_history = _load_inputs(
        reference_root=reference_root,
        universe_history_path=universe_history_path,
    )
    eligible_panel = _build_eligible_panel(universe_history, trading_calendar)
    _validate_ticker_pit(eligible_panel, ticker_history_map)

    local_candidate = local_source_path if local_source_path is not None else (data_dir() / "price" / "source")
    local_files = _discover_local_files(local_candidate)
    local_load_stats = {"input_rows": 0, "parsed_rows": 0, "discarded_invalid_rows": 0}
    match_stats = {"unmapped_rows": 0, "duplicate_pk_dropped": 0}
    fallback_reason: str | None = None

    resolved_mode = ingestion_mode
    local_raw = pd.DataFrame()

    if ingestion_mode in {"auto", "local_file"}:
        if local_files:
            loaded, local_load_stats = _load_local_prices(local_files)
            mapped, match_stats = _map_local_to_panel(loaded, eligible_panel)
            if not mapped.empty:
                local_raw = mapped
                resolved_mode = "local_file"
            else:
                fallback_reason = "local_source_has_no_mappable_rows"
        else:
            fallback_reason = "local_source_missing_or_empty"

        if resolved_mode != "local_file":
            if ingestion_mode == "local_file" and not allow_synthetic_fallback:
                raise ValueError(
                    "local_file mode requested but no valid/mappable local rows and fallback is disabled."
                )
            if not allow_synthetic_fallback and ingestion_mode == "auto":
                raise ValueError("auto mode could not use local_file and fallback is disabled.")
            resolved_mode = "synthetic_fallback"

    if ingestion_mode == "synthetic_fallback":
        resolved_mode = "synthetic_fallback"
    elif ingestion_mode == "provider_stub":
        resolved_mode = "provider_stub"
    elif ingestion_mode not in {"auto", "local_file", "synthetic_fallback", "provider_stub"}:
        raise ValueError(f"Unsupported ingestion_mode: {ingestion_mode}")

    if resolved_mode == "local_file":
        raw_prices = local_raw.copy()
        if "adj_close_raw" not in raw_prices.columns:
            raw_prices["adj_close_raw"] = raw_prices["close"]
        raw_prices["source"] = raw_prices["source_file"]
        raw_prices["vendor"] = raw_prices["vendor"] if "vendor" in raw_prices.columns else "local_file"
        raw_prices.drop(columns=[col for col in ("source_file",) if col in raw_prices.columns], inplace=True)
        is_synthetic = False
    else:
        raw_prices = _generate_deterministic_prices(eligible_panel)
        raw_prices["adj_close_raw"] = raw_prices["close"]
        raw_prices["source"] = provider_name if resolved_mode == "provider_stub" else source
        raw_prices["vendor"] = provider_name if resolved_mode == "provider_stub" else "synthetic"
        is_synthetic = True

    raw_prices["date"] = _normalize_dates(raw_prices["date"], column="date")
    raw_prices["instrument_id"] = raw_prices["instrument_id"].astype(str)
    raw_prices["ticker"] = raw_prices["ticker"].astype(str).str.upper().str.strip()
    for col in ("open", "high", "low", "close", "volume", "adj_close_raw"):
        raw_prices[col] = pd.to_numeric(raw_prices[col], errors="coerce")

    numeric_invalid = raw_prices[
        ["open", "high", "low", "close", "volume", "adj_close_raw"]
    ].isna().any(axis=1)
    discarded_numeric = int(numeric_invalid.sum())
    raw_prices = raw_prices[~numeric_invalid].copy()

    raw_prices.sort_values(["date", "instrument_id"], inplace=True)
    raw_prices.reset_index(drop=True, inplace=True)

    config_hash = _config_hash(
        {
            "version": generation_version,
            "requested_mode": ingestion_mode,
            "resolved_mode": resolved_mode,
            "frequency": frequency,
            "source": source,
            "provider_name": provider_name,
            "local_source_path": str(local_source_path) if local_source_path is not None else "",
            "source_files": [str(path) for path in local_files],
        }
    )

    built_ts_utc = datetime.now(UTC).isoformat()
    raw_prices["source_mode"] = resolved_mode
    raw_prices["is_synthetic"] = is_synthetic
    raw_prices["frequency"] = frequency
    raw_prices["run_id"] = run_id
    raw_prices["config_hash"] = config_hash
    raw_prices["built_ts_utc"] = built_ts_utc

    _validate_raw_output(
        raw_prices,
        trading_calendar=trading_calendar,
        ticker_history_map=ticker_history_map,
    )

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (data_dir() / "price")
    target_dir.mkdir(parents=True, exist_ok=True)

    raw_path = write_parquet(
        raw_prices,
        target_dir / "raw_prices.parquet",
        schema_name="price_raw_mvp",
        run_id=run_id,
    )

    report_payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "requested_mode": ingestion_mode,
        "resolved_mode": resolved_mode,
        "local_source_path": str(local_candidate),
        "source_files": [str(path) for path in local_files],
        "fallback_reason": fallback_reason,
        "input_rows": int(local_load_stats["input_rows"]),
        "parsed_rows": int(local_load_stats["parsed_rows"]),
        "discarded_invalid_rows": int(local_load_stats["discarded_invalid_rows"] + discarded_numeric),
        "unmapped_rows": int(match_stats["unmapped_rows"]),
        "duplicate_pk_dropped": int(match_stats["duplicate_pk_dropped"]),
        "final_rows": int(len(raw_prices)),
    }
    report_path = _write_ingestion_report(
        target_dir / "raw_prices.ingestion_report.json",
        report_payload,
    )

    logger.info(
        "raw_prices_built",
        run_id=run_id,
        row_count=int(len(raw_prices)),
        n_instruments=int(raw_prices["instrument_id"].nunique()),
        n_sessions=int(raw_prices["date"].nunique()),
        requested_mode=ingestion_mode,
        resolved_mode=resolved_mode,
        source_files=len(local_files),
        unmapped_rows=int(match_stats["unmapped_rows"]),
        output_path=str(raw_path),
    )

    return FetchPricesResult(
        raw_prices_path=raw_path,
        ingestion_report_path=report_path,
        row_count=int(len(raw_prices)),
        n_instruments=int(raw_prices["instrument_id"].nunique()),
        n_sessions=int(raw_prices["date"].nunique()),
        config_hash=config_hash,
        source_mode=resolved_mode,
        source_files=tuple(str(path) for path in local_files),
        discarded_rows=int(local_load_stats["discarded_invalid_rows"] + discarded_numeric + match_stats["duplicate_pk_dropped"]),
        unmapped_rows=int(match_stats["unmapped_rows"]),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build raw price data v2 (local_file + synthetic fallback).")
    parser.add_argument("--reference-root", type=str, default=None, help="Path to reference data directory.")
    parser.add_argument(
        "--universe-history-path",
        type=str,
        default=None,
        help="Path to universe_history parquet.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Path where raw_prices.parquet is written.")
    parser.add_argument("--run-id", type=str, default="price_fetch_mvp_v1")
    parser.add_argument("--frequency", type=str, default="1d")
    parser.add_argument("--source", type=str, default="synthetic_mvp_v2")
    parser.add_argument(
        "--ingestion-mode",
        type=str,
        default="auto",
        choices=("auto", "local_file", "synthetic_fallback", "provider_stub"),
    )
    parser.add_argument(
        "--local-source-path",
        type=str,
        default=None,
        help="CSV/Parquet file or directory for local_file mode.",
    )
    parser.add_argument(
        "--disable-synthetic-fallback",
        action="store_true",
        help="Disable fallback to synthetic mode when local_file is unavailable/invalid.",
    )
    parser.add_argument("--provider-name", type=str, default="provider_stub")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = fetch_prices(
        reference_root=args.reference_root,
        universe_history_path=args.universe_history_path,
        output_dir=args.output_dir,
        run_id=args.run_id,
        frequency=args.frequency,
        source=args.source,
        ingestion_mode=args.ingestion_mode,
        local_source_path=args.local_source_path,
        allow_synthetic_fallback=not args.disable_synthetic_fallback,
        provider_name=args.provider_name,
    )
    print("Raw prices built:")
    print(f"- path: {result.raw_prices_path}")
    print(f"- mode: {result.source_mode}")
    print(f"- report: {result.ingestion_report_path}")
    print(f"- rows: {result.row_count}")
    print(f"- instruments: {result.n_instruments}")
    print(f"- sessions: {result.n_sessions}")


if __name__ == "__main__":
    main()
