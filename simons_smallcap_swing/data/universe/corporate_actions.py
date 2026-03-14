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

# Allow direct script execution: `python simons_smallcap_swing/data/universe/corporate_actions.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir, reference_dir
from simons_core.logging import get_logger
from simons_core.schemas import assert_schema

ALLOWED_EVENT_TYPES: tuple[str, ...] = (
    "split",
    "reverse_split",
    "ticker_change",
    "delisting",
    "listing_start",
    "listing_end",
)
TERMINAL_EVENT_TYPES: set[str] = {"delisting", "listing_end"}


@dataclass(frozen=True)
class CorporateActionsResult:
    corporate_actions_path: Path
    summary_path: Path
    row_count: int
    event_type_counts: dict[str, int]
    config_hash: str


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains invalid dates.")
    return parsed.dt.normalize()


def _stable_event_id(
    *,
    instrument_id: str,
    event_type: str,
    effective_date: pd.Timestamp,
    old_ticker: str | None,
    new_ticker: str | None,
    split_factor: float | None,
) -> str:
    payload = "|".join(
        [
            instrument_id,
            event_type,
            str(pd.Timestamp(effective_date).date()),
            old_ticker or "",
            new_ticker or "",
            "" if split_factor is None else f"{split_factor:.10f}",
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"ca_{digest}"


def _config_hash(split_source_path: str | Path | None) -> str:
    payload = {
        "version": "corporate_actions_mvp_v1",
        "allowed_event_types": list(ALLOWED_EVENT_TYPES),
        "split_source_path": str(split_source_path) if split_source_path else "",
    }
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
        raise ValueError(f"universe_history missing required columns: {missing}")

    return ticker_history_map, symbols_metadata, universe_history


def _prepare_ticker_history(
    ticker_history_map: pd.DataFrame,
    symbols_metadata: pd.DataFrame,
    universe_history: pd.DataFrame,
) -> pd.DataFrame:
    history = ticker_history_map.copy()
    history["instrument_id"] = history["instrument_id"].astype(str)
    history["ticker"] = history["ticker"].astype(str)
    history["start_date"] = _normalize_date(history["start_date"], column="start_date")
    history["end_date"] = pd.to_datetime(history["end_date"], errors="coerce").dt.normalize()
    history["is_active"] = history["is_active"].astype(bool)

    invalid_interval = history["end_date"].notna() & (history["end_date"] < history["start_date"])
    if invalid_interval.any():
        raise ValueError("ticker_history_map has rows with end_date < start_date.")

    metadata_instruments = set(symbols_metadata["instrument_id"].astype(str).tolist())
    universe_instruments = set(universe_history["instrument_id"].astype(str).tolist())
    # Keep all universe instruments and any historical rows that ended (terminal evidence).
    keep_mask = history["instrument_id"].isin(universe_instruments) | history["end_date"].notna()
    history = history[keep_mask].copy()
    history = history[history["instrument_id"].isin(metadata_instruments)].copy()

    if history.empty:
        raise ValueError("No usable ticker history rows for corporate actions build.")

    return history.sort_values(["instrument_id", "start_date", "ticker"]).reset_index(drop=True)


def _build_inferred_events(ticker_history: pd.DataFrame) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    for instrument_id, sub in ticker_history.groupby("instrument_id", sort=True):
        seq = sub.sort_values("start_date").reset_index(drop=True)
        first = seq.iloc[0]
        events.append(
            {
                "instrument_id": instrument_id,
                "event_type": "listing_start",
                "effective_date": first["start_date"],
                "announced_date": first["start_date"],
                "source_start_date": first["start_date"],
                "source_end_date": first["end_date"],
                "old_ticker": None,
                "new_ticker": first["ticker"],
                "split_factor": None,
                "event_value": None,
                "event_unit": None,
                "source_mode": "inferred_ticker_history",
                "source_ref": "ticker_history_map",
                "is_terminal": False,
            }
        )

        for idx in range(1, len(seq)):
            prev_row = seq.iloc[idx - 1]
            cur_row = seq.iloc[idx]
            if prev_row["ticker"] != cur_row["ticker"]:
                announced = prev_row["end_date"] if pd.notna(prev_row["end_date"]) else cur_row["start_date"]
                events.append(
                    {
                        "instrument_id": instrument_id,
                        "event_type": "ticker_change",
                        "effective_date": cur_row["start_date"],
                        "announced_date": announced,
                        "source_start_date": prev_row["start_date"],
                        "source_end_date": cur_row["start_date"],
                        "old_ticker": prev_row["ticker"],
                        "new_ticker": cur_row["ticker"],
                        "split_factor": None,
                        "event_value": None,
                        "event_unit": None,
                        "source_mode": "inferred_ticker_history",
                        "source_ref": "ticker_history_map",
                        "is_terminal": False,
                    }
                )

        last = seq.iloc[-1]
        if pd.notna(last["end_date"]):
            events.append(
                {
                    "instrument_id": instrument_id,
                    "event_type": "delisting",
                    "effective_date": last["end_date"],
                    "announced_date": last["end_date"],
                    "source_start_date": last["start_date"],
                    "source_end_date": last["end_date"],
                    "old_ticker": last["ticker"],
                    "new_ticker": None,
                    "split_factor": None,
                    "event_value": None,
                    "event_unit": None,
                    "source_mode": "inferred_ticker_history",
                    "source_ref": "ticker_history_map",
                    "is_terminal": True,
                }
            )
            events.append(
                {
                    "instrument_id": instrument_id,
                    "event_type": "listing_end",
                    "effective_date": last["end_date"],
                    "announced_date": last["end_date"],
                    "source_start_date": last["start_date"],
                    "source_end_date": last["end_date"],
                    "old_ticker": last["ticker"],
                    "new_ticker": None,
                    "split_factor": None,
                    "event_value": None,
                    "event_unit": None,
                    "source_mode": "inferred_ticker_history",
                    "source_ref": "ticker_history_map",
                    "is_terminal": True,
                }
            )

    return events


def _load_optional_split_events(split_source_path: str | Path | None) -> list[dict[str, Any]]:
    if split_source_path is None:
        return []

    path = Path(split_source_path).expanduser().resolve()
    if not path.exists():
        return []

    if path.suffix.lower() == ".csv":
        split_df = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        split_df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported split source extension: {path.suffix}")

    split_df.columns = [str(col).strip().lower() for col in split_df.columns]
    required = {"instrument_id", "effective_date", "split_factor"}
    missing = sorted(required - set(split_df.columns))
    if missing:
        raise ValueError(f"split source missing required columns: {missing}")

    split_df["instrument_id"] = split_df["instrument_id"].astype(str)
    split_df["effective_date"] = _normalize_date(split_df["effective_date"], column="effective_date")
    split_df["split_factor"] = pd.to_numeric(split_df["split_factor"], errors="coerce")
    if split_df["split_factor"].isna().any():
        raise ValueError("split source has non-numeric split_factor values.")
    if (split_df["split_factor"] <= 0).any():
        raise ValueError("split source has split_factor <= 0.")

    if "event_type" in split_df.columns:
        split_df["event_type"] = split_df["event_type"].astype(str).str.strip().str.lower()
        allowed_split_events = {"split", "reverse_split"}
        invalid_event_types = sorted(set(split_df["event_type"].unique()) - allowed_split_events)
        if invalid_event_types:
            raise ValueError(
                "split source contains unsupported event_type values: "
                f"{invalid_event_types}. Allowed: {sorted(allowed_split_events)}"
            )
    else:
        split_df["event_type"] = split_df["split_factor"].map(
            lambda value: "split" if float(value) < 1.0 else "reverse_split"
        )

    if "announced_date" in split_df.columns:
        split_df["announced_date"] = pd.to_datetime(split_df["announced_date"], errors="coerce").dt.normalize()
    else:
        split_df["announced_date"] = split_df["effective_date"]

    events: list[dict[str, Any]] = []
    for row in split_df.itertuples(index=False):
        events.append(
            {
                "instrument_id": row.instrument_id,
                "event_type": row.event_type,
                "effective_date": row.effective_date,
                "announced_date": row.announced_date,
                "source_start_date": row.effective_date,
                "source_end_date": row.effective_date,
                "old_ticker": None,
                "new_ticker": None,
                "split_factor": float(row.split_factor),
                "event_value": float(row.split_factor),
                "event_unit": "ratio",
                "source_mode": "local_split_file",
                "source_ref": str(path),
                "is_terminal": False,
            }
        )
    return events


def _validate_corporate_actions(frame: pd.DataFrame) -> None:
    required = {
        "event_id",
        "instrument_id",
        "event_type",
        "effective_date",
        "is_terminal",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required corporate_actions columns: {missing}")

    for col in ("event_id", "instrument_id", "event_type", "effective_date"):
        if frame[col].isna().any():
            raise ValueError(f"Column '{col}' contains null values.")

    if not set(frame["event_type"].unique()).issubset(set(ALLOWED_EVENT_TYPES)):
        bad = sorted(set(frame["event_type"].unique()) - set(ALLOWED_EVENT_TYPES))
        raise ValueError(f"Unsupported event_type values: {bad}")

    split_mask = frame["event_type"].isin(["split", "reverse_split"])
    if split_mask.any():
        invalid_split = frame.loc[split_mask, "split_factor"].isna() | (
            frame.loc[split_mask, "split_factor"] <= 0
        )
        if invalid_split.any():
            raise ValueError("split/reverse_split rows require split_factor > 0.")

    ticker_change_mask = frame["event_type"] == "ticker_change"
    if ticker_change_mask.any():
        subset = frame.loc[ticker_change_mask, ["old_ticker", "new_ticker"]]
        invalid = (
            subset["old_ticker"].isna()
            | subset["new_ticker"].isna()
            | (subset["old_ticker"] == subset["new_ticker"])
        )
        if invalid.any():
            raise ValueError("ticker_change rows require old_ticker != new_ticker and both non-null.")

    terminal_invalid_non_terminal = frame["is_terminal"] & ~frame["event_type"].isin(
        TERMINAL_EVENT_TYPES
    )
    if terminal_invalid_non_terminal.any():
        raise ValueError("is_terminal=True is only allowed for terminal event types.")
    terminal_missing_flag = frame["event_type"].isin(TERMINAL_EVENT_TYPES) & ~frame["is_terminal"]
    if terminal_missing_flag.any():
        raise ValueError("Terminal event types must have is_terminal=True.")

    source_interval_invalid = (
        frame["source_start_date"].notna()
        & frame["source_end_date"].notna()
        & (frame["source_end_date"] < frame["source_start_date"])
    )
    if source_interval_invalid.any():
        raise ValueError("source_end_date must be >= source_start_date when both are present.")

    duplicate_event_id = frame.duplicated(["event_id"], keep=False)
    if duplicate_event_id.any():
        raise ValueError("Duplicate event_id values detected.")

    dedup_key = frame[
        ["instrument_id", "event_type", "effective_date", "old_ticker", "new_ticker"]
    ].copy()
    dedup_key["old_ticker"] = dedup_key["old_ticker"].fillna("__NULL__")
    dedup_key["new_ticker"] = dedup_key["new_ticker"].fillna("__NULL__")
    duplicate_key = dedup_key.duplicated(
        ["instrument_id", "event_type", "effective_date", "old_ticker", "new_ticker"], keep=False
    )
    if duplicate_key.any():
        raise ValueError("Duplicate corporate action logical keys detected.")


def build_corporate_actions(
    *,
    reference_root: str | Path | None = None,
    universe_history_path: str | Path | None = None,
    split_source_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    run_id: str = "corporate_actions_mvp_v1",
) -> CorporateActionsResult:
    logger = get_logger("data.universe.corporate_actions")
    ticker_history_map, symbols_metadata, universe_history = _load_inputs(
        reference_root=reference_root,
        universe_history_path=universe_history_path,
    )
    prepared_history = _prepare_ticker_history(ticker_history_map, symbols_metadata, universe_history)

    inferred_events = _build_inferred_events(prepared_history)
    split_events = _load_optional_split_events(split_source_path)
    all_events = inferred_events + split_events
    if not all_events:
        raise ValueError("No corporate action events were generated.")

    frame = pd.DataFrame(all_events)
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["event_type"] = frame["event_type"].astype(str)
    frame["effective_date"] = _normalize_date(frame["effective_date"], column="effective_date")
    frame["announced_date"] = pd.to_datetime(frame["announced_date"], errors="coerce").dt.normalize()
    frame["source_start_date"] = pd.to_datetime(frame["source_start_date"], errors="coerce").dt.normalize()
    frame["source_end_date"] = pd.to_datetime(frame["source_end_date"], errors="coerce").dt.normalize()
    frame["split_factor"] = pd.to_numeric(frame["split_factor"], errors="coerce")
    frame["event_value"] = pd.to_numeric(frame["event_value"], errors="coerce")
    frame["is_terminal"] = frame["is_terminal"].astype(bool)
    known_instruments = set(prepared_history["instrument_id"].astype(str).unique().tolist())
    unknown_instruments = sorted(set(frame["instrument_id"].unique().tolist()) - known_instruments)
    if unknown_instruments:
        raise ValueError(
            "Corporate actions include unknown instrument_id values not present in "
            f"ticker_history_map/symbols_metadata: {unknown_instruments}"
        )

    event_ids: list[str] = []
    for row in frame.itertuples(index=False):
        event_ids.append(
            _stable_event_id(
                instrument_id=row.instrument_id,
                event_type=row.event_type,
                effective_date=row.effective_date,
                old_ticker=row.old_ticker if pd.notna(row.old_ticker) else None,
                new_ticker=row.new_ticker if pd.notna(row.new_ticker) else None,
                split_factor=float(row.split_factor) if pd.notna(row.split_factor) else None,
            )
        )
    frame["event_id"] = event_ids

    config_hash = _config_hash(split_source_path)
    built_ts = datetime.now(UTC).isoformat()
    frame["run_id"] = run_id
    frame["config_hash"] = config_hash
    frame["built_ts_utc"] = built_ts

    ordered_columns = [
        "event_id",
        "instrument_id",
        "event_type",
        "effective_date",
        "announced_date",
        "source_start_date",
        "source_end_date",
        "old_ticker",
        "new_ticker",
        "split_factor",
        "event_value",
        "event_unit",
        "source_mode",
        "source_ref",
        "is_terminal",
        "run_id",
        "config_hash",
        "built_ts_utc",
    ]
    frame = frame[ordered_columns].sort_values(
        ["instrument_id", "effective_date", "event_type", "event_id"]
    ).reset_index(drop=True)

    _validate_corporate_actions(frame)

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (data_dir() / "universe")
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = write_parquet(
        frame,
        target_dir / "corporate_actions.parquet",
        schema_name="corporate_actions_mvp",
        run_id=run_id,
    )

    counts = frame["event_type"].value_counts().sort_index().to_dict()
    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "row_count": int(len(frame)),
        "n_instruments": int(frame["instrument_id"].nunique()),
        "event_type_counts": counts,
        "terminal_events": int(frame["is_terminal"].sum()),
        "split_source_path": str(split_source_path) if split_source_path else "",
        "output_path": str(output_path),
    }
    summary_path = target_dir / "corporate_actions.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "corporate_actions_built",
        run_id=run_id,
        row_count=int(len(frame)),
        n_instruments=int(frame["instrument_id"].nunique()),
        event_type_counts=counts,
        output_path=str(output_path),
    )

    return CorporateActionsResult(
        corporate_actions_path=output_path,
        summary_path=summary_path,
        row_count=int(len(frame)),
        event_type_counts={str(key): int(value) for key, value in counts.items()},
        config_hash=config_hash,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build canonical corporate actions MVP dataset.")
    parser.add_argument("--reference-root", type=str, default=None)
    parser.add_argument("--universe-history-path", type=str, default=None)
    parser.add_argument("--split-source-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default="corporate_actions_mvp_v1")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = build_corporate_actions(
        reference_root=args.reference_root,
        universe_history_path=args.universe_history_path,
        split_source_path=args.split_source_path,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print("Corporate actions built:")
    print(f"- path: {result.corporate_actions_path}")
    print(f"- summary: {result.summary_path}")
    print(f"- rows: {result.row_count}")
    print(f"- event_type_counts: {result.event_type_counts}")


if __name__ == "__main__":
    main()
