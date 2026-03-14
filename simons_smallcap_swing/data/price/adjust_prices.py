from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Allow direct script execution: `python simons_smallcap_swing/data/price/adjust_prices.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger


@dataclass(frozen=True)
class AdjustPricesResult:
    adjusted_prices_path: Path
    adjustment_report_path: Path
    row_count: int
    mode: str
    config_hash: str
    split_events_total: int
    split_events_consumed: int
    rows_factor_not_one: int


def _config_hash(*, mode: str, convention: str, raw_source: Path, corporate_source: Path) -> str:
    payload = {
        "mode": mode,
        "convention": convention,
        "raw_source": str(raw_source),
        "corporate_actions_source": str(corporate_source),
        "version": "price_adjust_v2_split_only",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _normalize_dates(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains non-parseable dates.")
    return parsed.dt.normalize()


def _validate_raw_schema(raw: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "instrument_id", "ticker", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"raw_prices is missing required columns: {missing}")

    if raw.empty:
        raise ValueError("raw_prices is empty.")

    dup = raw.duplicated(["date", "instrument_id"], keep=False)
    if dup.any():
        raise ValueError("raw_prices has duplicate (date, instrument_id).")

    out = raw.copy()
    out["date"] = _normalize_dates(out["date"], column="date")
    out["instrument_id"] = out["instrument_id"].astype(str)
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if out[required].isna().any().any():
        raise ValueError("raw_prices has nulls in required columns.")

    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("raw_prices has non-positive OHLC values.")
    if (out["volume"] < 0).any():
        raise ValueError("raw_prices has negative volume values.")

    bad_ohlc = (
        (out["high"] < out[["open", "close", "low"]].max(axis=1))
        | (out["low"] > out[["open", "close", "high"]].min(axis=1))
    )
    if bad_ohlc.any():
        raise ValueError("raw_prices violates OHLC geometry.")

    return out.sort_values(["date", "instrument_id"]).reset_index(drop=True)


def _load_corporate_actions(
    corporate_actions_path: str | Path | None,
) -> tuple[pd.DataFrame, Path, bool, str]:
    default_source = data_dir() / "universe" / "corporate_actions.parquet"
    if corporate_actions_path is None:
        source = default_source
        if source.exists():
            return read_parquet(source), source, True, ""
        empty = pd.DataFrame(
            columns=[
                "instrument_id",
                "event_type",
                "effective_date",
                "split_factor",
                "event_id",
            ]
        )
        return empty, source, False, "corporate_actions_default_not_found"

    source = Path(corporate_actions_path).expanduser().resolve()
    return read_parquet(source), source, True, ""


def _extract_split_events(
    corporate_actions: pd.DataFrame,
    *,
    raw_instruments: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {
        "corporate_events_total": int(len(corporate_actions)),
        "ignored_non_split_events": 0,
        "ignored_unknown_instrument_events": 0,
        "split_events_total": 0,
    }

    if corporate_actions.empty:
        return pd.DataFrame(columns=["instrument_id", "effective_date", "split_factor"]), stats

    required = {"instrument_id", "event_type", "effective_date"}
    missing = sorted(required - set(corporate_actions.columns))
    if missing:
        raise ValueError(f"corporate_actions is missing required columns: {missing}")

    frame = corporate_actions.copy()
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["event_type"] = frame["event_type"].astype(str).str.strip().str.lower()
    split_mask = frame["event_type"].isin({"split", "reverse_split"})

    stats["ignored_non_split_events"] = int((~split_mask).sum())
    split_events = frame[split_mask].copy()
    if split_events.empty:
        return pd.DataFrame(columns=["instrument_id", "effective_date", "split_factor"]), stats

    if "split_factor" not in split_events.columns:
        raise ValueError("corporate_actions split events require 'split_factor' column.")

    split_events["effective_date"] = pd.to_datetime(
        split_events["effective_date"], errors="coerce"
    ).dt.normalize()
    if split_events["effective_date"].isna().any():
        raise ValueError("corporate_actions split events contain invalid effective_date values.")

    split_events["split_factor"] = pd.to_numeric(split_events["split_factor"], errors="coerce")
    if split_events["split_factor"].isna().any():
        raise ValueError("corporate_actions split events contain non-numeric split_factor values.")
    if (split_events["split_factor"] <= 0).any():
        raise ValueError("corporate_actions split events require split_factor > 0.")

    if "event_id" in split_events.columns:
        duplicated_event_id = split_events["event_id"].astype(str).duplicated(keep=False)
        if duplicated_event_id.any():
            raise ValueError("corporate_actions has duplicate event_id values for split events.")

    before_filter = len(split_events)
    split_events = split_events[split_events["instrument_id"].isin(raw_instruments)].copy()
    stats["ignored_unknown_instrument_events"] = int(before_filter - len(split_events))
    stats["split_events_total"] = int(len(split_events))

    return split_events[["instrument_id", "effective_date", "split_factor"]], stats


def _compute_split_factors(
    raw: pd.DataFrame,
    split_events: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, int, int]:
    cumulative_factor = pd.Series(1.0, index=raw.index, dtype="float64")
    applied_count = pd.Series(0, index=raw.index, dtype="int64")
    if split_events.empty:
        return cumulative_factor, applied_count, 0, 0

    split_events = split_events.copy()
    min_raw_by_instrument = raw.groupby("instrument_id", as_index=True)["date"].min().to_dict()
    split_events["is_consumed"] = split_events.apply(
        lambda row: row["effective_date"] > min_raw_by_instrument.get(row["instrument_id"], pd.Timestamp.max),
        axis=1,
    )
    split_events_consumed = int(split_events["is_consumed"].sum())
    split_events_no_effect = int((~split_events["is_consumed"]).sum())

    for instrument_id, row_idx in raw.groupby("instrument_id", sort=False).groups.items():
        instrument_events = split_events.loc[split_events["instrument_id"] == instrument_id]
        if instrument_events.empty:
            continue

        day_factors = (
            instrument_events.groupby("effective_date", as_index=False)["split_factor"]
            .prod()
            .sort_values("effective_date")
            .reset_index(drop=True)
        )
        event_dates = day_factors["effective_date"].to_numpy(dtype="datetime64[ns]")
        event_factors = day_factors["split_factor"].to_numpy(dtype="float64")
        suffix_factors = np.cumprod(event_factors[::-1])[::-1]

        row_index = list(row_idx)
        raw_dates = raw.loc[row_index, "date"].to_numpy(dtype="datetime64[ns]")
        insert_idx = np.searchsorted(event_dates, raw_dates, side="right")

        factors = np.ones(len(raw_dates), dtype="float64")
        valid = insert_idx < len(event_dates)
        if valid.any():
            factors[valid] = suffix_factors[insert_idx[valid]]

        counts = (len(event_dates) - insert_idx).astype("int64")
        cumulative_factor.loc[row_index] = factors
        applied_count.loc[row_index] = counts

    return cumulative_factor, applied_count, split_events_consumed, split_events_no_effect


def _write_adjustment_report(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def adjust_prices(
    *,
    raw_prices_path: str | Path | None = None,
    corporate_actions_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    run_id: str = "price_adjust_split_only_v2",
) -> AdjustPricesResult:
    logger = get_logger("data.price.adjust_prices")
    mode = "split_only"
    convention = (
        "backward_split_adjustment: OHLC * product(split_factor for effective_date > date), "
        "volume / same cumulative factor"
    )

    raw_source = (
        Path(raw_prices_path).expanduser().resolve()
        if raw_prices_path
        else data_dir() / "price" / "raw_prices.parquet"
    )
    raw = _validate_raw_schema(read_parquet(raw_source))

    corporate_actions, corporate_source, corporate_actions_found, corporate_missing_reason = (
        _load_corporate_actions(corporate_actions_path)
    )
    split_events, split_stats = _extract_split_events(
        corporate_actions,
        raw_instruments=set(raw["instrument_id"].astype(str).unique().tolist()),
    )
    cumulative_factor, applied_count, split_events_consumed, split_events_no_effect = _compute_split_factors(
        raw, split_events
    )

    if (cumulative_factor <= 0).any():
        raise ValueError("Computed cumulative_split_factor contains non-positive values.")

    config_hash = _config_hash(
        mode=mode,
        convention=convention,
        raw_source=raw_source,
        corporate_source=corporate_source,
    )

    adjusted = raw[["date", "instrument_id", "ticker"]].copy()
    adjusted["cumulative_split_factor"] = cumulative_factor.astype(float)
    adjusted["applied_split_events_count"] = applied_count.astype("int64")

    adjusted["open_adj"] = raw["open"].astype(float) * adjusted["cumulative_split_factor"]
    adjusted["high_adj"] = raw["high"].astype(float) * adjusted["cumulative_split_factor"]
    adjusted["low_adj"] = raw["low"].astype(float) * adjusted["cumulative_split_factor"]
    adjusted["close_adj"] = raw["close"].astype(float) * adjusted["cumulative_split_factor"]
    adjusted["volume_adj"] = raw["volume"].astype(float) / adjusted["cumulative_split_factor"]

    bad_ohlc = (
        (adjusted["high_adj"] < adjusted[["open_adj", "close_adj", "low_adj"]].max(axis=1))
        | (adjusted["low_adj"] > adjusted[["open_adj", "close_adj", "high_adj"]].min(axis=1))
    )
    if bad_ohlc.any():
        raise ValueError("adjusted_prices violates OHLC geometry after split-only adjustment.")
    if (adjusted[["open_adj", "high_adj", "low_adj", "close_adj"]] <= 0).any().any():
        raise ValueError("adjusted_prices has non-positive OHLC values after adjustment.")
    if (adjusted["volume_adj"] < 0).any():
        raise ValueError("adjusted_prices has negative volume_adj values.")

    adjusted["adjustment_mode"] = mode
    adjusted["adjustment_note"] = convention
    adjusted["source_raw_path"] = str(raw_source)
    adjusted["source_corporate_actions_path"] = str(corporate_source)
    adjusted["run_id"] = run_id
    adjusted["config_hash"] = config_hash
    adjusted["built_ts_utc"] = datetime.now(UTC).isoformat()

    adjusted = adjusted[
        [
            "date",
            "instrument_id",
            "ticker",
            "open_adj",
            "high_adj",
            "low_adj",
            "close_adj",
            "volume_adj",
            "adjustment_mode",
            "adjustment_note",
            "cumulative_split_factor",
            "applied_split_events_count",
            "source_raw_path",
            "source_corporate_actions_path",
            "run_id",
            "config_hash",
            "built_ts_utc",
        ]
    ].sort_values(["date", "instrument_id"]).reset_index(drop=True)

    duplicated_pk = adjusted.duplicated(["date", "instrument_id"], keep=False)
    if duplicated_pk.any():
        raise ValueError("adjusted_prices has duplicate (date, instrument_id) rows.")

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (data_dir() / "price")
    target_dir.mkdir(parents=True, exist_ok=True)
    adjusted_path = write_parquet(
        adjusted,
        target_dir / "adjusted_prices.parquet",
        schema_name="price_adjusted_split_only_v2",
        run_id=run_id,
    )

    rows_factor_not_one = int((~np.isclose(adjusted["cumulative_split_factor"], 1.0)).sum())
    adjustment_report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "adjustment_mode": mode,
        "convention": convention,
        "source_raw_path": str(raw_source),
        "source_corporate_actions_path": str(corporate_source),
        "corporate_actions_found": bool(corporate_actions_found),
        "corporate_actions_missing_reason": corporate_missing_reason,
        "corporate_events_total": int(split_stats["corporate_events_total"]),
        "ignored_non_split_events": int(split_stats["ignored_non_split_events"]),
        "ignored_unknown_instrument_events": int(split_stats["ignored_unknown_instrument_events"]),
        "split_events_total": int(split_stats["split_events_total"]),
        "split_events_consumed": int(split_events_consumed),
        "split_events_no_effect": int(split_events_no_effect),
        "rows_total": int(len(adjusted)),
        "rows_factor_not_one": rows_factor_not_one,
        "factor_min": float(adjusted["cumulative_split_factor"].min()),
        "factor_max": float(adjusted["cumulative_split_factor"].max()),
        "output_path": str(adjusted_path),
    }
    report_path = _write_adjustment_report(
        target_dir / "adjusted_prices.adjustment_report.json",
        adjustment_report,
    )

    logger.info(
        "adjusted_prices_built",
        run_id=run_id,
        mode=mode,
        row_count=int(len(adjusted)),
        split_events_total=int(split_stats["split_events_total"]),
        split_events_consumed=int(split_events_consumed),
        rows_factor_not_one=rows_factor_not_one,
        output_path=str(adjusted_path),
    )

    return AdjustPricesResult(
        adjusted_prices_path=adjusted_path,
        adjustment_report_path=report_path,
        row_count=int(len(adjusted)),
        mode=mode,
        config_hash=config_hash,
        split_events_total=int(split_stats["split_events_total"]),
        split_events_consumed=int(split_events_consumed),
        rows_factor_not_one=rows_factor_not_one,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build adjusted prices v2 (split_only mode).")
    parser.add_argument("--raw-prices-path", type=str, default=None, help="Path to raw_prices parquet.")
    parser.add_argument(
        "--corporate-actions-path",
        type=str,
        default=None,
        help="Path to corporate_actions parquet. If omitted and default path is missing, no split events are applied.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path where adjusted_prices.parquet is written.",
    )
    parser.add_argument("--run-id", type=str, default="price_adjust_split_only_v2")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = adjust_prices(
        raw_prices_path=args.raw_prices_path,
        corporate_actions_path=args.corporate_actions_path,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print("Adjusted prices built:")
    print(f"- path: {result.adjusted_prices_path}")
    print(f"- report: {result.adjustment_report_path}")
    print(f"- rows: {result.row_count}")
    print(f"- mode: {result.mode}")
    print(f"- split_events_total: {result.split_events_total}")
    print(f"- split_events_consumed: {result.split_events_consumed}")
    print(f"- rows_factor_not_one: {result.rows_factor_not_one}")


if __name__ == "__main__":
    main()
