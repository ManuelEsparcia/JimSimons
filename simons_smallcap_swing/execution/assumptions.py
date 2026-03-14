from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

# Allow direct script execution: `python simons_smallcap_swing/execution/assumptions.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "execution_assumptions_mvp_v1"
DEFAULT_EXECUTION_DELAY_SESSIONS = 1
FILL_ASSUMPTION_FULL_FILL = "full_fill"
FILL_ASSUMPTION_SKIP_IF_MISSING_NEXT_SESSION = "skip_if_missing_next_session"
SUPPORTED_FILL_ASSUMPTIONS = (
    FILL_ASSUMPTION_FULL_FILL,
    FILL_ASSUMPTION_SKIP_IF_MISSING_NEXT_SESSION,
)
COST_TIMING_ON_EXECUTION = "apply_on_execution_date"
COST_TIMING_ON_SIGNAL = "apply_on_signal_date"
SUPPORTED_COST_TIMINGS = (
    COST_TIMING_ON_EXECUTION,
    COST_TIMING_ON_SIGNAL,
)
EPS = 1e-12

HOLDINGS_INPUT_SCHEMA = DataSchema(
    name="execution_assumptions_holdings_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("portfolio_mode", "string", nullable=False),
        ColumnSpec("target_weight", "float64", nullable=False),
    ),
    primary_key=("date", "instrument_id", "portfolio_mode", "model_name", "label_name"),
    allow_extra_columns=True,
)

REBALANCE_INPUT_SCHEMA = DataSchema(
    name="execution_assumptions_rebalance_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("prev_weight", "float64", nullable=False),
        ColumnSpec("target_weight", "float64", nullable=False),
        ColumnSpec("weight_change", "float64", nullable=False),
        ColumnSpec("abs_weight_change", "float64", nullable=False),
        ColumnSpec("entered_flag", "bool", nullable=False),
        ColumnSpec("exited_flag", "bool", nullable=False),
        ColumnSpec("portfolio_mode", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
    ),
    primary_key=("date", "instrument_id", "portfolio_mode", "model_name", "label_name"),
    allow_extra_columns=True,
)

CALENDAR_INPUT_SCHEMA = DataSchema(
    name="execution_assumptions_calendar_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("is_session", "bool", nullable=False),
    ),
    primary_key=("date",),
    allow_extra_columns=True,
)

PRICES_INPUT_SCHEMA = DataSchema(
    name="execution_assumptions_prices_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("close_adj", "float64", nullable=False),
    ),
    primary_key=("date", "instrument_id"),
    allow_extra_columns=True,
)

EXECUTION_HOLDINGS_SCHEMA = DataSchema(
    name="execution_holdings_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("signal_date", "datetime64", nullable=False),
        ColumnSpec("execution_date", "datetime64", nullable=True),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("portfolio_mode", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("target_weight", "float64", nullable=False),
        ColumnSpec("execution_weight", "float64", nullable=False),
        ColumnSpec("fill_assumption", "string", nullable=False),
        ColumnSpec("execution_delay_sessions", "int64", nullable=False),
        ColumnSpec("is_executable", "bool", nullable=False),
        ColumnSpec("skip_reason", "string", nullable=True),
    ),
    primary_key=("signal_date", "instrument_id", "portfolio_mode", "model_name", "label_name"),
    allow_extra_columns=True,
)

EXECUTION_REBALANCE_SCHEMA = DataSchema(
    name="execution_rebalance_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("signal_date", "datetime64", nullable=False),
        ColumnSpec("execution_date", "datetime64", nullable=True),
        ColumnSpec("cost_date", "datetime64", nullable=True),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("prev_weight", "float64", nullable=False),
        ColumnSpec("target_weight", "float64", nullable=False),
        ColumnSpec("execution_weight", "float64", nullable=False),
        ColumnSpec("weight_change_signal", "float64", nullable=False),
        ColumnSpec("weight_change_execution", "float64", nullable=False),
        ColumnSpec("entered_flag", "bool", nullable=False),
        ColumnSpec("exited_flag", "bool", nullable=False),
        ColumnSpec("is_executable", "bool", nullable=False),
        ColumnSpec("skip_reason", "string", nullable=True),
        ColumnSpec("portfolio_mode", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
    ),
    primary_key=("signal_date", "instrument_id", "portfolio_mode", "model_name", "label_name"),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class ExecutionAssumptionsResult:
    execution_holdings_path: Path
    execution_rebalance_path: Path
    execution_assumptions_summary_path: Path
    row_count_holdings: int
    row_count_rebalance: int
    model_name: str
    label_name: str
    config_hash: str


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains invalid dates.")
    return parsed.dt.normalize()


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return float(parsed)


def _config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _select_unique_value(frame: pd.DataFrame, column: str, *, provided: str | None) -> str:
    if provided is not None:
        selected = str(provided).strip()
        if not selected:
            raise ValueError(f"Provided {column} is empty.")
        filtered = frame[frame[column].astype(str) == selected]
        if filtered.empty:
            raise ValueError(f"No rows left after filtering {column}='{selected}'.")
        return selected
    uniques = sorted(frame[column].astype(str).unique().tolist())
    if len(uniques) != 1:
        raise ValueError(
            f"Expected exactly one {column} per run. Observed {column} values: {uniques}. "
            f"Pass --{column.replace('_', '-')} explicitly."
        )
    return str(uniques[0])


def _normalize_modes(modes: Iterable[str] | None) -> tuple[str, ...]:
    if modes is None:
        return tuple()
    return tuple(sorted({str(item).strip() for item in modes if str(item).strip()}))


def _prepare_holdings(path: Path) -> pd.DataFrame:
    frame = read_parquet(path)
    assert_schema(frame, HOLDINGS_INPUT_SCHEMA)
    frame = frame.copy()
    frame["date"] = _normalize_date(frame["date"], column="date")
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["portfolio_mode"] = frame["portfolio_mode"].astype(str)
    frame["model_name"] = frame["model_name"].astype(str)
    frame["label_name"] = frame["label_name"].astype(str)
    frame["target_weight"] = pd.to_numeric(frame["target_weight"], errors="coerce")
    if frame["target_weight"].isna().any():
        raise ValueError("portfolio_holdings contains non-numeric target_weight.")
    if frame.duplicated(["date", "instrument_id", "portfolio_mode", "model_name", "label_name"], keep=False).any():
        raise ValueError("portfolio_holdings has duplicate logical PK rows.")
    return frame


def _prepare_rebalance(path: Path) -> pd.DataFrame:
    frame = read_parquet(path)
    assert_schema(frame, REBALANCE_INPUT_SCHEMA)
    frame = frame.copy()
    frame["date"] = _normalize_date(frame["date"], column="date")
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["portfolio_mode"] = frame["portfolio_mode"].astype(str)
    frame["model_name"] = frame["model_name"].astype(str)
    frame["label_name"] = frame["label_name"].astype(str)
    for col in ("prev_weight", "target_weight", "weight_change", "abs_weight_change"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if frame[col].isna().any():
            raise ValueError(f"portfolio_rebalance contains non-numeric '{col}'.")
    frame["entered_flag"] = frame["entered_flag"].astype(bool)
    frame["exited_flag"] = frame["exited_flag"].astype(bool)
    if frame.duplicated(["date", "instrument_id", "portfolio_mode", "model_name", "label_name"], keep=False).any():
        raise ValueError("portfolio_rebalance has duplicate logical PK rows.")
    if not np.allclose(
        (frame["target_weight"] - frame["prev_weight"]).to_numpy(dtype=float),
        frame["weight_change"].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("portfolio_rebalance weight_change must equal target_weight - prev_weight.")
    return frame


def _prepare_calendar(path: Path) -> pd.DataFrame:
    frame = read_parquet(path)
    assert_schema(frame, CALENDAR_INPUT_SCHEMA)
    frame = frame.copy()
    frame["date"] = _normalize_date(frame["date"], column="date")
    frame["is_session"] = frame["is_session"].astype(bool)
    if frame.duplicated(["date"], keep=False).any():
        raise ValueError("trading_calendar has duplicate date rows.")
    return frame


def _prepare_prices(path: Path) -> pd.DataFrame:
    frame = read_parquet(path)
    assert_schema(frame, PRICES_INPUT_SCHEMA)
    frame = frame.copy()
    frame["date"] = _normalize_date(frame["date"], column="date")
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["close_adj"] = pd.to_numeric(frame["close_adj"], errors="coerce")
    if frame["close_adj"].isna().any():
        raise ValueError("adjusted_prices contains non-numeric close_adj.")
    if (frame["close_adj"] <= 0).any():
        raise ValueError("adjusted_prices close_adj must be > 0.")
    if frame.duplicated(["date", "instrument_id"], keep=False).any():
        raise ValueError("adjusted_prices has duplicate (date, instrument_id) rows.")
    return frame


def _build_execution_date_map(
    *,
    calendar: pd.DataFrame,
    delay_sessions: int,
) -> dict[pd.Timestamp, pd.Timestamp | pd.NaT]:
    sessions = sorted(pd.to_datetime(calendar.loc[calendar["is_session"], "date"]).dt.normalize().unique().tolist())
    if not sessions:
        raise ValueError("trading_calendar has no session dates.")
    idx_map = {pd.Timestamp(x).normalize(): i for i, x in enumerate(sessions)}
    out: dict[pd.Timestamp, pd.Timestamp | pd.NaT] = {}
    for date_value in pd.to_datetime(calendar["date"]).dt.normalize().unique().tolist():
        signal_date = pd.Timestamp(date_value).normalize()
        if signal_date not in idx_map:
            out[signal_date] = pd.NaT
            continue
        target_idx = idx_map[signal_date] + delay_sessions
        if target_idx >= len(sessions):
            out[signal_date] = pd.NaT
        else:
            out[signal_date] = pd.Timestamp(sessions[target_idx]).normalize()
    return out


def _skip_reason_for_row(
    *,
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp | pd.NaT,
    signal_is_session: bool,
    has_execution_price: bool | None,
    fill_assumption: str,
) -> str | None:
    if not signal_is_session:
        return "signal_not_session"
    if pd.isna(execution_date):
        return "no_execution_session"
    if fill_assumption == FILL_ASSUMPTION_SKIP_IF_MISSING_NEXT_SESSION and has_execution_price is False:
        return "missing_price_on_execution_date"
    return None


def run_execution_assumptions(
    *,
    holdings_path: str | Path | None = None,
    rebalance_path: str | Path | None = None,
    trading_calendar_path: str | Path | None = None,
    adjusted_prices_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_name: str | None = None,
    label_name: str | None = None,
    portfolio_modes: Iterable[str] | None = None,
    execution_delay_sessions: int = DEFAULT_EXECUTION_DELAY_SESSIONS,
    fill_assumption: str = FILL_ASSUMPTION_FULL_FILL,
    cost_timing: str = COST_TIMING_ON_EXECUTION,
    run_id: str = MODULE_VERSION,
) -> ExecutionAssumptionsResult:
    logger = get_logger("execution.assumptions")

    delay_int = int(execution_delay_sessions)
    if delay_int < 0:
        raise ValueError("execution_delay_sessions must be >= 0.")
    fill_assumption_norm = str(fill_assumption).strip()
    if fill_assumption_norm not in SUPPORTED_FILL_ASSUMPTIONS:
        raise ValueError(
            f"Unsupported fill_assumption='{fill_assumption_norm}'. "
            f"Supported: {list(SUPPORTED_FILL_ASSUMPTIONS)}"
        )
    cost_timing_norm = str(cost_timing).strip()
    if cost_timing_norm not in SUPPORTED_COST_TIMINGS:
        raise ValueError(
            f"Unsupported cost_timing='{cost_timing_norm}'. "
            f"Supported: {list(SUPPORTED_COST_TIMINGS)}"
        )

    holdings_source = (
        Path(holdings_path).expanduser().resolve()
        if holdings_path
        else (data_dir() / "portfolio" / "portfolio_holdings.parquet")
    )
    rebalance_source = (
        Path(rebalance_path).expanduser().resolve()
        if rebalance_path
        else (data_dir() / "portfolio" / "portfolio_rebalance.parquet")
    )
    calendar_source = (
        Path(trading_calendar_path).expanduser().resolve()
        if trading_calendar_path
        else (data_dir() / "reference" / "trading_calendar.parquet")
    )
    prices_source = (
        Path(adjusted_prices_path).expanduser().resolve()
        if adjusted_prices_path
        else None
    )

    holdings = _prepare_holdings(holdings_source)
    rebalance = _prepare_rebalance(rebalance_source)
    calendar = _prepare_calendar(calendar_source)

    selected_model_name = _select_unique_value(holdings, "model_name", provided=model_name)
    selected_label_name = _select_unique_value(holdings, "label_name", provided=label_name)
    holdings = holdings[
        (holdings["model_name"] == selected_model_name) & (holdings["label_name"] == selected_label_name)
    ].copy()
    rebalance = rebalance[
        (rebalance["model_name"] == selected_model_name) & (rebalance["label_name"] == selected_label_name)
    ].copy()
    if holdings.empty:
        raise ValueError("No holdings rows left for selected model_name/label_name.")
    if rebalance.empty:
        raise ValueError("No rebalance rows left for selected model_name/label_name.")

    selected_modes = _normalize_modes(portfolio_modes)
    if selected_modes:
        holdings = holdings[holdings["portfolio_mode"].isin(set(selected_modes))].copy()
        rebalance = rebalance[rebalance["portfolio_mode"].isin(set(selected_modes))].copy()
        if holdings.empty:
            raise ValueError(f"No holdings rows left after portfolio_mode filter: {list(selected_modes)}")
        if rebalance.empty:
            raise ValueError(f"No rebalance rows left after portfolio_mode filter: {list(selected_modes)}")

    execution_date_map = _build_execution_date_map(calendar=calendar, delay_sessions=delay_int)
    session_dates = set(pd.to_datetime(calendar.loc[calendar["is_session"], "date"]).dt.normalize().tolist())

    price_keys: set[tuple[pd.Timestamp, str]] | None = None
    if prices_source is not None:
        prices = _prepare_prices(prices_source)
        price_keys = {
            (pd.Timestamp(row.date).normalize(), str(row.instrument_id))
            for row in prices[["date", "instrument_id"]].itertuples(index=False)
        }

    holdings_exec = holdings.rename(columns={"date": "signal_date"}).copy()
    holdings_exec["signal_date"] = _normalize_date(holdings_exec["signal_date"], column="signal_date")
    holdings_exec["execution_date"] = holdings_exec["signal_date"].map(execution_date_map)
    signal_is_session = holdings_exec["signal_date"].isin(session_dates)

    if price_keys is None:
        has_exec_price_holdings = pd.Series([None] * len(holdings_exec), index=holdings_exec.index, dtype="object")
    else:
        has_exec_price_holdings = holdings_exec.apply(
            lambda row: bool(
                (pd.Timestamp(row["execution_date"]).normalize(), str(row["instrument_id"])) in price_keys
            )
            if not pd.isna(row["execution_date"])
            else None,
            axis=1,
        )

    holdings_exec["skip_reason"] = [
        _skip_reason_for_row(
            signal_date=pd.Timestamp(sig_date),
            execution_date=pd.Timestamp(exec_date) if not pd.isna(exec_date) else pd.NaT,
            signal_is_session=bool(is_session),
            has_execution_price=(
                None if value is None or (isinstance(value, float) and pd.isna(value)) else bool(value)
            ),
            fill_assumption=fill_assumption_norm,
        )
        for sig_date, exec_date, is_session, value in zip(
            holdings_exec["signal_date"],
            holdings_exec["execution_date"],
            signal_is_session,
            has_exec_price_holdings,
        )
    ]
    holdings_exec["is_executable"] = holdings_exec["skip_reason"].isna()
    holdings_exec["execution_weight"] = np.where(
        holdings_exec["is_executable"],
        holdings_exec["target_weight"].astype(float),
        0.0,
    )
    holdings_exec["fill_assumption"] = fill_assumption_norm
    holdings_exec["execution_delay_sessions"] = delay_int

    holdings_exec = holdings_exec[
        [
            "signal_date",
            "execution_date",
            "instrument_id",
            "ticker",
            "portfolio_mode",
            "model_name",
            "label_name",
            "target_weight",
            "execution_weight",
            "fill_assumption",
            "execution_delay_sessions",
            "is_executable",
            "skip_reason",
        ]
    ].sort_values(["signal_date", "portfolio_mode", "instrument_id"]).reset_index(drop=True)

    rebalance_exec = rebalance.rename(columns={"date": "signal_date"}).copy()
    rebalance_exec["signal_date"] = _normalize_date(rebalance_exec["signal_date"], column="signal_date")
    rebalance_exec["execution_date"] = rebalance_exec["signal_date"].map(execution_date_map)
    reb_signal_is_session = rebalance_exec["signal_date"].isin(session_dates)

    if price_keys is None:
        has_exec_price_rebalance = pd.Series([None] * len(rebalance_exec), index=rebalance_exec.index, dtype="object")
    else:
        has_exec_price_rebalance = rebalance_exec.apply(
            lambda row: bool(
                (pd.Timestamp(row["execution_date"]).normalize(), str(row["instrument_id"])) in price_keys
            )
            if not pd.isna(row["execution_date"])
            else None,
            axis=1,
        )

    rebalance_exec["skip_reason"] = [
        _skip_reason_for_row(
            signal_date=pd.Timestamp(sig_date),
            execution_date=pd.Timestamp(exec_date) if not pd.isna(exec_date) else pd.NaT,
            signal_is_session=bool(is_session),
            has_execution_price=(
                None if value is None or (isinstance(value, float) and pd.isna(value)) else bool(value)
            ),
            fill_assumption=fill_assumption_norm,
        )
        for sig_date, exec_date, is_session, value in zip(
            rebalance_exec["signal_date"],
            rebalance_exec["execution_date"],
            reb_signal_is_session,
            has_exec_price_rebalance,
        )
    ]
    rebalance_exec["is_executable"] = rebalance_exec["skip_reason"].isna()
    rebalance_exec["execution_weight"] = np.where(
        rebalance_exec["is_executable"],
        rebalance_exec["target_weight"].astype(float),
        rebalance_exec["prev_weight"].astype(float),
    )
    rebalance_exec["weight_change_signal"] = rebalance_exec["weight_change"].astype(float)
    rebalance_exec["weight_change_execution"] = (
        rebalance_exec["execution_weight"].astype(float) - rebalance_exec["prev_weight"].astype(float)
    )
    if cost_timing_norm == COST_TIMING_ON_EXECUTION:
        rebalance_exec["cost_date"] = np.where(
            rebalance_exec["is_executable"],
            rebalance_exec["execution_date"],
            pd.NaT,
        )
    else:
        rebalance_exec["cost_date"] = np.where(
            rebalance_exec["is_executable"],
            rebalance_exec["signal_date"],
            pd.NaT,
        )
    rebalance_exec["cost_date"] = pd.to_datetime(rebalance_exec["cost_date"], errors="coerce").dt.normalize()

    rebalance_exec = rebalance_exec[
        [
            "signal_date",
            "execution_date",
            "cost_date",
            "instrument_id",
            "ticker",
            "prev_weight",
            "target_weight",
            "execution_weight",
            "weight_change_signal",
            "weight_change_execution",
            "entered_flag",
            "exited_flag",
            "is_executable",
            "skip_reason",
            "portfolio_mode",
            "model_name",
            "label_name",
        ]
    ].sort_values(["signal_date", "portfolio_mode", "instrument_id"]).reset_index(drop=True)

    # Validation checks
    for frame_name, frame in (
        ("execution_holdings", holdings_exec),
        ("execution_rebalance", rebalance_exec),
    ):
        non_exec_missing_reason = (~frame["is_executable"]) & frame["skip_reason"].isna()
        exec_with_reason = frame["is_executable"] & frame["skip_reason"].notna()
        if non_exec_missing_reason.any():
            raise ValueError(f"{frame_name} has non-executable rows without skip_reason.")
        if exec_with_reason.any():
            raise ValueError(f"{frame_name} has executable rows with non-null skip_reason.")
        has_exec_date = frame["execution_date"].notna()
        if (frame.loc[has_exec_date, "signal_date"] > frame.loc[has_exec_date, "execution_date"]).any():
            raise ValueError(f"{frame_name} has signal_date > execution_date.")
        if has_exec_date.any():
            exec_dates = pd.to_datetime(frame.loc[has_exec_date, "execution_date"]).dt.normalize()
            if not exec_dates.isin(session_dates).all():
                raise ValueError(f"{frame_name} has execution_date outside trading sessions.")

    if not np.allclose(
        rebalance_exec["weight_change_execution"].to_numpy(dtype=float),
        (rebalance_exec["execution_weight"] - rebalance_exec["prev_weight"]).to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("weight_change_execution must equal execution_weight - prev_weight.")

    config_hash = _config_hash(
        {
            "version": MODULE_VERSION,
            "holdings_path": str(holdings_source),
            "rebalance_path": str(rebalance_source),
            "trading_calendar_path": str(calendar_source),
            "adjusted_prices_path": str(prices_source) if prices_source is not None else None,
            "model_name": selected_model_name,
            "label_name": selected_label_name,
            "portfolio_modes": sorted(holdings_exec["portfolio_mode"].astype(str).unique().tolist()),
            "execution_delay_sessions": delay_int,
            "fill_assumption": fill_assumption_norm,
            "cost_timing": cost_timing_norm,
        }
    )
    built_ts_utc = datetime.now(UTC).isoformat()

    holdings_exec["run_id"] = run_id
    holdings_exec["config_hash"] = config_hash
    holdings_exec["built_ts_utc"] = built_ts_utc
    rebalance_exec["run_id"] = run_id
    rebalance_exec["config_hash"] = config_hash
    rebalance_exec["built_ts_utc"] = built_ts_utc

    assert_schema(holdings_exec, EXECUTION_HOLDINGS_SCHEMA)
    assert_schema(rebalance_exec, EXECUTION_REBALANCE_SCHEMA)

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (data_dir() / "execution")
    target_dir.mkdir(parents=True, exist_ok=True)

    execution_holdings_path = write_parquet(
        holdings_exec,
        target_dir / "execution_holdings.parquet",
        schema_name=EXECUTION_HOLDINGS_SCHEMA.name,
        run_id=run_id,
    )
    execution_rebalance_path = write_parquet(
        rebalance_exec,
        target_dir / "execution_rebalance.parquet",
        schema_name=EXECUTION_REBALANCE_SCHEMA.name,
        run_id=run_id,
    )

    skip_counts = {
        str(reason): int(count)
        for reason, count in holdings_exec.loc[~holdings_exec["is_executable"], "skip_reason"]
        .value_counts(dropna=False)
        .items()
    }

    summary = {
        "built_ts_utc": built_ts_utc,
        "run_id": run_id,
        "config_hash": config_hash,
        "module_version": MODULE_VERSION,
        "execution_delay_sessions": delay_int,
        "fill_assumption": fill_assumption_norm,
        "cost_timing": cost_timing_norm,
        "model_name": selected_model_name,
        "label_name": selected_label_name,
        "portfolio_modes": sorted(holdings_exec["portfolio_mode"].astype(str).unique().tolist()),
        "n_signal_rows": int(len(holdings_exec)),
        "n_execution_rows": int(holdings_exec["is_executable"].sum()),
        "n_skipped_rows": int((~holdings_exec["is_executable"]).sum()),
        "skip_reasons": skip_counts,
        "first_signal_date": str(pd.Timestamp(holdings_exec["signal_date"].min()).date()),
        "first_execution_date": (
            str(pd.Timestamp(holdings_exec.loc[holdings_exec["execution_date"].notna(), "execution_date"].min()).date())
            if holdings_exec["execution_date"].notna().any()
            else None
        ),
        "last_signal_date": str(pd.Timestamp(holdings_exec["signal_date"].max()).date()),
        "last_execution_date": (
            str(pd.Timestamp(holdings_exec.loc[holdings_exec["execution_date"].notna(), "execution_date"].max()).date())
            if holdings_exec["execution_date"].notna().any()
            else None
        ),
        "output_paths": {
            "execution_holdings": str(execution_holdings_path),
            "execution_rebalance": str(execution_rebalance_path),
        },
    }
    execution_assumptions_summary_path = target_dir / "execution_assumptions_summary.json"
    execution_assumptions_summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    logger.info(
        "execution_assumptions_built",
        run_id=run_id,
        model_name=selected_model_name,
        label_name=selected_label_name,
        execution_delay_sessions=delay_int,
        fill_assumption=fill_assumption_norm,
        cost_timing=cost_timing_norm,
        n_signal_rows=int(len(holdings_exec)),
        n_execution_rows=int(holdings_exec["is_executable"].sum()),
        n_skipped_rows=int((~holdings_exec["is_executable"]).sum()),
        output_dir=str(target_dir),
    )

    return ExecutionAssumptionsResult(
        execution_holdings_path=execution_holdings_path,
        execution_rebalance_path=execution_rebalance_path,
        execution_assumptions_summary_path=execution_assumptions_summary_path,
        row_count_holdings=int(len(holdings_exec)),
        row_count_rebalance=int(len(rebalance_exec)),
        model_name=selected_model_name,
        label_name=selected_label_name,
        config_hash=config_hash,
    )


def _parse_csv_strings(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return values


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize explicit execution assumptions (signal_date -> execution_date) for holdings/rebalance."
    )
    parser.add_argument("--holdings-path", type=str, default=None)
    parser.add_argument("--rebalance-path", type=str, default=None)
    parser.add_argument("--trading-calendar-path", type=str, default=None)
    parser.add_argument("--adjusted-prices-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--label-name", type=str, default=None)
    parser.add_argument("--portfolio-modes", type=_parse_csv_strings, default=tuple())
    parser.add_argument("--execution-delay-sessions", type=int, default=DEFAULT_EXECUTION_DELAY_SESSIONS)
    parser.add_argument("--fill-assumption", type=str, default=FILL_ASSUMPTION_FULL_FILL)
    parser.add_argument("--cost-timing", type=str, default=COST_TIMING_ON_EXECUTION)
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_execution_assumptions(
        holdings_path=args.holdings_path,
        rebalance_path=args.rebalance_path,
        trading_calendar_path=args.trading_calendar_path,
        adjusted_prices_path=args.adjusted_prices_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        label_name=args.label_name,
        portfolio_modes=args.portfolio_modes,
        execution_delay_sessions=args.execution_delay_sessions,
        fill_assumption=args.fill_assumption,
        cost_timing=args.cost_timing,
        run_id=args.run_id,
    )
    print("Execution assumptions built:")
    print(f"- execution_holdings: {result.execution_holdings_path}")
    print(f"- execution_rebalance: {result.execution_rebalance_path}")
    print(f"- execution_assumptions_summary: {result.execution_assumptions_summary_path}")
    print(f"- row_count_holdings: {result.row_count_holdings}")
    print(f"- row_count_rebalance: {result.row_count_rebalance}")
    print(f"- model_name: {result.model_name}")
    print(f"- label_name: {result.label_name}")


if __name__ == "__main__":
    main()
