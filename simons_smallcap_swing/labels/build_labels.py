from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "labels_forward_mvp_v1"
DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 20)
DEFAULT_DECISION_LAG = 1
DEFAULT_PRICE_FIELD = "close_adj"

ADJUSTED_INPUT_SCHEMA = DataSchema(
    name="labels_adjusted_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("close_adj", "number", nullable=False),
    ),
    primary_key=("date", "instrument_id"),
    allow_extra_columns=True,
)

UNIVERSE_INPUT_SCHEMA = DataSchema(
    name="labels_universe_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("is_eligible", "bool", nullable=False),
    ),
    primary_key=("date", "instrument_id"),
    allow_extra_columns=True,
)

CALENDAR_INPUT_SCHEMA = DataSchema(
    name="labels_calendar_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("is_session", "bool", nullable=False),
    ),
    primary_key=("date",),
    allow_extra_columns=True,
)

LABELS_FORWARD_SCHEMA = DataSchema(
    name="labels_forward_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("horizon_days", "int64", nullable=False),
        ColumnSpec("entry_date", "datetime64", nullable=False),
        ColumnSpec("exit_date", "datetime64", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("label_value", "float64", nullable=False),
        ColumnSpec("price_entry", "float64", nullable=False),
        ColumnSpec("price_exit", "float64", nullable=False),
        ColumnSpec("source_price_field", "string", nullable=False),
    ),
    primary_key=("date", "instrument_id", "horizon_days", "label_name"),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class BuildLabelsResult:
    labels_path: Path
    summary_path: Path
    row_count: int
    n_instruments: int
    horizons: tuple[int, ...]
    label_names: tuple[str, ...]
    config_hash: str


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains invalid dates.")
    return parsed.dt.normalize()


def _normalize_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    normalized = sorted({int(item) for item in horizons})
    if not normalized:
        raise ValueError("At least one horizon is required.")
    if any(item <= 0 for item in normalized):
        raise ValueError(f"Horizons must be positive integers. Received: {normalized}")
    return tuple(normalized)


def _build_config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _prepare_inputs(
    *,
    adjusted_prices_path: str | Path | None,
    universe_history_path: str | Path | None,
    trading_calendar_path: str | Path | None,
    source_price_field: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    base_data = data_dir()
    adjusted_source = (
        Path(adjusted_prices_path).expanduser().resolve()
        if adjusted_prices_path
        else base_data / "price" / "adjusted_prices.parquet"
    )
    universe_source = (
        Path(universe_history_path).expanduser().resolve()
        if universe_history_path
        else base_data / "universe" / "universe_history.parquet"
    )
    calendar_source = (
        Path(trading_calendar_path).expanduser().resolve()
        if trading_calendar_path
        else base_data / "reference" / "trading_calendar.parquet"
    )

    adjusted = read_parquet(adjusted_source)
    universe = read_parquet(universe_source)
    calendar = read_parquet(calendar_source)

    if source_price_field not in adjusted.columns:
        raise ValueError(
            f"Price field '{source_price_field}' is missing in adjusted_prices. "
            f"Available columns: {sorted(adjusted.columns)}"
        )

    assert_schema(adjusted, ADJUSTED_INPUT_SCHEMA)
    assert_schema(universe, UNIVERSE_INPUT_SCHEMA)
    assert_schema(calendar, CALENDAR_INPUT_SCHEMA)

    adjusted = adjusted.copy()
    universe = universe.copy()
    calendar = calendar.copy()

    adjusted["date"] = _normalize_date(adjusted["date"], column="date")
    adjusted["instrument_id"] = adjusted["instrument_id"].astype(str)
    adjusted["ticker"] = adjusted["ticker"].astype(str).str.upper().str.strip()
    adjusted[source_price_field] = pd.to_numeric(adjusted[source_price_field], errors="coerce")
    if adjusted[source_price_field].isna().any():
        raise ValueError(f"adjusted_prices has non-numeric values in '{source_price_field}'.")
    if (adjusted[source_price_field] <= 0).any():
        raise ValueError(f"adjusted_prices has non-positive values in '{source_price_field}'.")
    if adjusted.duplicated(["date", "instrument_id"]).any():
        raise ValueError("adjusted_prices has duplicate (date, instrument_id) rows.")

    universe["date"] = _normalize_date(universe["date"], column="date")
    universe["instrument_id"] = universe["instrument_id"].astype(str)
    universe["ticker"] = universe["ticker"].astype(str).str.upper().str.strip()
    universe["is_eligible"] = universe["is_eligible"].astype(bool)
    if universe.duplicated(["date", "instrument_id"]).any():
        raise ValueError("universe_history has duplicate (date, instrument_id) rows.")

    calendar["date"] = _normalize_date(calendar["date"], column="date")
    calendar["is_session"] = calendar["is_session"].astype(bool)
    if calendar.duplicated(["date"]).any():
        raise ValueError("trading_calendar has duplicate date rows.")

    sessions = pd.DatetimeIndex(sorted(calendar.loc[calendar["is_session"], "date"].unique()))
    if sessions.empty:
        raise ValueError("trading_calendar has no active sessions.")
    valid_sessions = set(sessions.tolist())

    if (~adjusted["date"].isin(valid_sessions)).any():
        sample = sorted(
            {
                str(pd.Timestamp(item).date())
                for item in adjusted.loc[
                    ~adjusted["date"].isin(valid_sessions), "date"
                ].head(10)
            }
        )
        raise ValueError(
            "adjusted_prices contains dates outside trading sessions. "
            f"Sample: {sample}"
        )

    eligible = universe[universe["is_eligible"]].copy()
    if eligible.empty:
        raise ValueError("universe_history has no eligible rows (is_eligible=True).")
    if (~eligible["date"].isin(valid_sessions)).any():
        sample = sorted(
            {
                str(pd.Timestamp(item).date())
                for item in eligible.loc[
                    ~eligible["date"].isin(valid_sessions), "date"
                ].head(10)
            }
        )
        raise ValueError(
            "universe_history has eligible rows outside trading sessions. "
            f"Sample: {sample}"
        )

    return adjusted, eligible, calendar, adjusted_source, universe_source, calendar_source


def build_labels(
    *,
    adjusted_prices_path: str | Path | None = None,
    universe_history_path: str | Path | None = None,
    trading_calendar_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    decision_lag: int = DEFAULT_DECISION_LAG,
    source_price_field: str = DEFAULT_PRICE_FIELD,
    include_binary_direction: bool = False,
    run_id: str = MODULE_VERSION,
) -> BuildLabelsResult:
    logger = get_logger("labels.build_labels")
    if int(decision_lag) < 1:
        raise ValueError("decision_lag must be >= 1 for strict no-leakage semantics.")
    lag = int(decision_lag)
    normalized_horizons = _normalize_horizons(horizons)

    adjusted, eligible, calendar, adjusted_source, universe_source, calendar_source = _prepare_inputs(
        adjusted_prices_path=adjusted_prices_path,
        universe_history_path=universe_history_path,
        trading_calendar_path=trading_calendar_path,
        source_price_field=source_price_field,
    )

    sessions = pd.DatetimeIndex(sorted(calendar.loc[calendar["is_session"], "date"].unique()))
    session_df = pd.DataFrame({"date": sessions, "session_pos": range(len(sessions))})
    max_session_pos = int(len(sessions) - 1)
    session_values = pd.to_datetime(sessions.to_numpy())

    decisions = eligible.merge(session_df, on="date", how="left")
    if decisions["session_pos"].isna().any():
        raise ValueError("Failed to map eligible universe rows to session index.")
    decisions["session_pos"] = decisions["session_pos"].astype(int)

    price_panel = adjusted[
        ["date", "instrument_id", "ticker", source_price_field]
    ].rename(columns={source_price_field: "price"})

    labels_blocks: list[pd.DataFrame] = []
    per_horizon_stats: list[dict[str, Any]] = []
    total_candidates = 0
    total_incomplete = 0

    for horizon in normalized_horizons:
        frame = decisions.copy()
        frame["horizon_days"] = int(horizon)
        frame["entry_pos"] = frame["session_pos"] + lag
        frame["exit_pos"] = frame["session_pos"] + lag + int(horizon)

        candidate_count = int(len(frame))
        total_candidates += candidate_count

        incomplete_mask = (frame["entry_pos"] > max_session_pos) | (frame["exit_pos"] > max_session_pos)
        incomplete_count = int(incomplete_mask.sum())
        total_incomplete += incomplete_count

        frame = frame.loc[~incomplete_mask].copy()
        if frame.empty:
            per_horizon_stats.append(
                {
                    "horizon_days": int(horizon),
                    "candidate_rows": candidate_count,
                    "incomplete_horizon_rows": incomplete_count,
                    "missing_price_rows": 0,
                    "output_rows": 0,
                }
            )
            continue

        frame["entry_date"] = pd.to_datetime(session_values[frame["entry_pos"].to_numpy()])
        frame["exit_date"] = pd.to_datetime(session_values[frame["exit_pos"].to_numpy()])

        frame = frame.merge(
            price_panel[["instrument_id", "date", "price"]].rename(
                columns={"date": "entry_date", "price": "price_entry"}
            ),
            on=["instrument_id", "entry_date"],
            how="left",
        )
        frame = frame.merge(
            price_panel[["instrument_id", "date", "price"]].rename(
                columns={"date": "exit_date", "price": "price_exit"}
            ),
            on=["instrument_id", "exit_date"],
            how="left",
        )

        missing_price_mask = frame["price_entry"].isna() | frame["price_exit"].isna()
        missing_price_count = int(missing_price_mask.sum())
        frame = frame.loc[~missing_price_mask].copy()
        if frame.empty:
            per_horizon_stats.append(
                {
                    "horizon_days": int(horizon),
                    "candidate_rows": candidate_count,
                    "incomplete_horizon_rows": incomplete_count,
                    "missing_price_rows": missing_price_count,
                    "output_rows": 0,
                }
            )
            continue

        frame["label_name"] = f"fwd_ret_{int(horizon)}d"
        frame["label_value"] = frame["price_exit"] / frame["price_entry"] - 1.0
        frame["source_price_field"] = source_price_field

        block = frame[
            [
                "date",
                "instrument_id",
                "ticker",
                "horizon_days",
                "entry_date",
                "exit_date",
                "label_name",
                "label_value",
                "price_entry",
                "price_exit",
                "source_price_field",
            ]
        ].copy()
        labels_blocks.append(block)

        if include_binary_direction:
            binary = block.copy()
            binary["label_name"] = f"fwd_dir_up_{int(horizon)}d"
            binary["label_value"] = (binary["label_value"] > 0.0).astype(float)
            labels_blocks.append(binary)

        per_horizon_stats.append(
            {
                "horizon_days": int(horizon),
                "candidate_rows": candidate_count,
                "incomplete_horizon_rows": incomplete_count,
                "missing_price_rows": missing_price_count,
                "output_rows": int(len(block)),
            }
        )

    if not labels_blocks:
        raise ValueError(
            "No labels generated. Check horizons, decision_lag, and forward price coverage."
        )

    labels = pd.concat(labels_blocks, ignore_index=True)
    labels["label_value"] = labels["label_value"].astype(float)
    labels["price_entry"] = labels["price_entry"].astype(float)
    labels["price_exit"] = labels["price_exit"].astype(float)
    labels["horizon_days"] = labels["horizon_days"].astype("int64")

    if not (labels["entry_date"] > labels["date"]).all():
        raise ValueError("Temporal leakage risk: expected entry_date > decision date for all rows.")
    if not (labels["exit_date"] >= labels["entry_date"]).all():
        raise ValueError("Invalid timing: found exit_date < entry_date.")
    if labels[["label_value", "price_entry", "price_exit"]].isna().any().any():
        raise ValueError("Generated labels contain null numeric fields.")
    if labels.duplicated(["date", "instrument_id", "horizon_days", "label_name"]).any():
        raise ValueError("Generated labels contain duplicate logical PK rows.")

    labels = labels.sort_values(
        ["date", "instrument_id", "horizon_days", "label_name"]
    ).reset_index(drop=True)
    assert_schema(labels, LABELS_FORWARD_SCHEMA)

    config_hash = _build_config_hash(
        {
            "version": MODULE_VERSION,
            "horizons": list(normalized_horizons),
            "decision_lag": lag,
            "source_price_field": source_price_field,
            "include_binary_direction": include_binary_direction,
            "paths": {
                "adjusted_prices": str(adjusted_source),
                "universe_history": str(universe_source),
                "trading_calendar": str(calendar_source),
            },
        }
    )
    built_ts_utc = datetime.now(UTC).isoformat()
    labels["run_id"] = run_id
    labels["config_hash"] = config_hash
    labels["built_ts_utc"] = built_ts_utc

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (data_dir() / "labels")
    target_dir.mkdir(parents=True, exist_ok=True)
    labels_path = write_parquet(
        labels,
        target_dir / "labels_forward.parquet",
        schema_name=LABELS_FORWARD_SCHEMA.name,
        run_id=run_id,
    )

    horizon_stats_df = pd.DataFrame(per_horizon_stats)
    missing_horizon_ratio = (
        float(total_incomplete) / float(total_candidates) if total_candidates > 0 else 0.0
    )
    summary_payload = {
        "created_at_utc": built_ts_utc,
        "run_id": run_id,
        "config_hash": config_hash,
        "label_mode": "forward_return_fixed_horizon",
        "decision_lag": lag,
        "source_price_field": source_price_field,
        "horizons_built": list(normalized_horizons),
        "label_names": sorted(labels["label_name"].astype(str).unique().tolist()),
        "n_rows_output": int(len(labels)),
        "n_instruments": int(labels["instrument_id"].nunique()),
        "start_date": str(pd.Timestamp(labels["date"].min()).date()),
        "end_date": str(pd.Timestamp(labels["date"].max()).date()),
        "pct_missing_due_to_horizon": round(missing_horizon_ratio, 6),
        "n_candidate_rows": int(total_candidates),
        "n_rows_dropped_incomplete_horizon": int(total_incomplete),
        "horizon_stats": horizon_stats_df.to_dict(orient="records"),
        "input_paths": {
            "adjusted_prices": str(adjusted_source),
            "universe_history": str(universe_source),
            "trading_calendar": str(calendar_source),
        },
        "output_path": str(labels_path),
    }
    summary_path = target_dir / "labels_forward.summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "labels_forward_built",
        run_id=run_id,
        row_count=int(len(labels)),
        n_instruments=int(labels["instrument_id"].nunique()),
        horizons=list(normalized_horizons),
        include_binary_direction=bool(include_binary_direction),
        output_path=str(labels_path),
    )

    return BuildLabelsResult(
        labels_path=labels_path,
        summary_path=summary_path,
        row_count=int(len(labels)),
        n_instruments=int(labels["instrument_id"].nunique()),
        horizons=normalized_horizons,
        label_names=tuple(sorted(labels["label_name"].astype(str).unique().tolist())),
        config_hash=config_hash,
    )


def _parse_horizons(text: str) -> tuple[int, ...]:
    parts = [item.strip() for item in text.split(",") if item.strip()]
    try:
        return _normalize_horizons(int(item) for item in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build PIT-safe forward labels (MVP) from adjusted prices and universe."
    )
    parser.add_argument("--adjusted-prices-path", type=str, default=None)
    parser.add_argument("--universe-history-path", type=str, default=None)
    parser.add_argument("--trading-calendar-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--horizons", type=_parse_horizons, default=DEFAULT_HORIZONS)
    parser.add_argument("--decision-lag", type=int, default=DEFAULT_DECISION_LAG)
    parser.add_argument("--source-price-field", type=str, default=DEFAULT_PRICE_FIELD)
    parser.add_argument("--include-binary-direction", action="store_true")
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = build_labels(
        adjusted_prices_path=args.adjusted_prices_path,
        universe_history_path=args.universe_history_path,
        trading_calendar_path=args.trading_calendar_path,
        output_dir=args.output_dir,
        horizons=args.horizons,
        decision_lag=args.decision_lag,
        source_price_field=args.source_price_field,
        include_binary_direction=args.include_binary_direction,
        run_id=args.run_id,
    )
    print("Labels forward built:")
    print(f"- path: {result.labels_path}")
    print(f"- summary: {result.summary_path}")
    print(f"- rows: {result.row_count}")
    print(f"- n_instruments: {result.n_instruments}")
    print(f"- horizons: {list(result.horizons)}")
    print(f"- labels: {list(result.label_names)}")


if __name__ == "__main__":
    main()
