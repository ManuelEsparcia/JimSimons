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

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "backtest_engine_mvp_v2_execution_timing"
EPS = 1e-12

HOLDINGS_LEGACY_SCHEMA = DataSchema(
    name="backtest_holdings_legacy_input",
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

HOLDINGS_EXEC_SCHEMA = DataSchema(
    name="backtest_holdings_execution_input",
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
        ColumnSpec("is_executable", "bool", nullable=False),
    ),
    primary_key=("signal_date", "instrument_id", "portfolio_mode", "model_name", "label_name"),
    allow_extra_columns=True,
)

COSTS_SCHEMA = DataSchema(
    name="backtest_costs_input",
    version="1.0.0",
    columns=(
        ColumnSpec("cost_date", "datetime64", nullable=True),
        ColumnSpec("date", "datetime64", nullable=True),
        ColumnSpec("portfolio_mode", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("total_cost", "float64", nullable=False),
    ),
    primary_key=("cost_date", "portfolio_mode", "model_name", "label_name"),
    allow_extra_columns=True,
)

PRICES_SCHEMA = DataSchema(
    name="backtest_prices_input",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("close_adj", "float64", nullable=False),
    ),
    primary_key=("date", "instrument_id"),
    allow_extra_columns=True,
)

CALENDAR_SCHEMA = DataSchema(
    name="backtest_calendar_input",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("is_session", "bool", nullable=False),
    ),
    primary_key=("date",),
    allow_extra_columns=True,
)

BACKTEST_DAILY_SCHEMA = DataSchema(
    name="backtest_daily_mvp",
    version="2.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("return_start_date", "datetime64", nullable=False),
        ColumnSpec("return_end_date", "datetime64", nullable=False),
        ColumnSpec("portfolio_mode", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("gross_return", "float64", nullable=False),
        ColumnSpec("total_cost", "float64", nullable=False),
        ColumnSpec("net_return", "float64", nullable=False),
        ColumnSpec("gross_equity", "float64", nullable=False),
        ColumnSpec("net_equity", "float64", nullable=False),
        ColumnSpec("drawdown_net", "float64", nullable=False),
        ColumnSpec("n_positions", "int64", nullable=False),
    ),
    primary_key=("date", "portfolio_mode", "model_name", "label_name"),
    allow_extra_columns=True,
)

BACKTEST_CONTRIB_SCHEMA = DataSchema(
    name="backtest_contributions_mvp",
    version="2.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("signal_date", "datetime64", nullable=False),
        ColumnSpec("execution_date", "datetime64", nullable=False),
        ColumnSpec("return_start_date", "datetime64", nullable=False),
        ColumnSpec("return_end_date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("portfolio_mode", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("target_weight", "float64", nullable=False),
        ColumnSpec("execution_weight", "float64", nullable=False),
        ColumnSpec("realized_return", "float64", nullable=False),
        ColumnSpec("contribution", "float64", nullable=False),
    ),
    primary_key=("signal_date", "instrument_id", "portfolio_mode", "model_name", "label_name"),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class BacktestEngineResult:
    backtest_daily_path: Path
    backtest_contributions_path: Path
    backtest_summary_path: Path
    row_count_daily: int
    row_count_contributions: int
    model_name: str
    label_name: str
    config_hash: str


def _norm_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains invalid dates.")
    return parsed.dt.normalize()


def _norm_modes(modes: Iterable[str] | None) -> tuple[str, ...]:
    if modes is None:
        return tuple()
    return tuple(sorted({str(x).strip() for x in modes if str(x).strip()}))


def _cfg_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(v):
        return None
    return float(v)


def _pick_unique(df: pd.DataFrame, column: str, provided: str | None) -> str:
    if provided is not None:
        val = str(provided).strip()
        if not val:
            raise ValueError(f"Provided {column} is empty.")
        if df[df[column].astype(str) == val].empty:
            raise ValueError(f"No rows left after filtering {column}='{val}'.")
        return val
    vals = sorted(df[column].astype(str).unique().tolist())
    if len(vals) != 1:
        raise ValueError(f"Expected exactly one {column}. Observed: {vals}.")
    return str(vals[0])


def _prepare_holdings(path: Path) -> tuple[pd.DataFrame, str]:
    frame = read_parquet(path)
    cols = set(frame.columns)
    if {"signal_date", "execution_date", "execution_weight", "is_executable"}.issubset(cols):
        assert_schema(frame, HOLDINGS_EXEC_SCHEMA)
        out = frame.copy()
        out["signal_date"] = _norm_date(out["signal_date"], column="signal_date")
        out["execution_date"] = pd.to_datetime(out["execution_date"], errors="coerce").dt.normalize()
        out["target_weight"] = pd.to_numeric(out["target_weight"], errors="coerce")
        out["execution_weight"] = pd.to_numeric(out["execution_weight"], errors="coerce")
        if out[["target_weight", "execution_weight"]].isna().any().any():
            raise ValueError("execution_holdings contains non-numeric weights.")
        out["is_executable"] = out["is_executable"].astype(bool)
        out["skip_reason"] = out.get("skip_reason", pd.Series([None] * len(out))).astype("string")
        mode = "execution_holdings"
    else:
        assert_schema(frame, HOLDINGS_LEGACY_SCHEMA)
        out = frame.copy()
        out["date"] = _norm_date(out["date"], column="date")
        out["signal_date"] = out["date"]
        out["execution_date"] = out["date"]
        out["target_weight"] = pd.to_numeric(out["target_weight"], errors="coerce")
        if out["target_weight"].isna().any():
            raise ValueError("portfolio_holdings contains non-numeric target_weight.")
        out["execution_weight"] = out["target_weight"]
        out["is_executable"] = True
        out["skip_reason"] = None
        mode = "legacy_portfolio_holdings"

    out["instrument_id"] = out["instrument_id"].astype(str)
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["portfolio_mode"] = out["portfolio_mode"].astype(str)
    out["model_name"] = out["model_name"].astype(str)
    out["label_name"] = out["label_name"].astype(str)
    if out.duplicated(["signal_date", "instrument_id", "portfolio_mode", "model_name", "label_name"], keep=False).any():
        raise ValueError("holdings has duplicate logical PK rows.")
    return out[
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
            "is_executable",
            "skip_reason",
        ]
    ].copy(), mode


def _prepare_costs(path: Path) -> pd.DataFrame:
    frame = read_parquet(path).copy()
    frame["cost_date"] = pd.to_datetime(frame["cost_date"], errors="coerce").dt.normalize() if "cost_date" in frame.columns else pd.NaT
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize() if "date" in frame.columns else pd.NaT
    frame["cost_date"] = frame["cost_date"].where(frame["cost_date"].notna(), frame["date"])
    frame["date"] = frame["cost_date"]
    if frame["cost_date"].isna().any():
        raise ValueError("costs_daily must contain valid cost_date or date.")
    assert_schema(frame, COSTS_SCHEMA)

    frame["portfolio_mode"] = frame["portfolio_mode"].astype(str)
    frame["model_name"] = frame["model_name"].astype(str)
    frame["label_name"] = frame["label_name"].astype(str)
    frame["total_cost"] = pd.to_numeric(frame["total_cost"], errors="coerce")
    if frame["total_cost"].isna().any():
        raise ValueError("costs_daily contains non-numeric total_cost.")
    if (frame["total_cost"] < -EPS).any():
        raise ValueError("costs_daily total_cost must be >= 0.")
    if frame.duplicated(["cost_date", "portfolio_mode", "model_name", "label_name"], keep=False).any():
        raise ValueError("costs_daily has duplicate (cost_date, portfolio_mode, model_name, label_name).")
    return frame[["cost_date", "date", "portfolio_mode", "model_name", "label_name", "total_cost"]].copy()


def _prepare_prices(path: Path) -> pd.DataFrame:
    frame = read_parquet(path)
    assert_schema(frame, PRICES_SCHEMA)
    out = frame.copy()
    out["date"] = _norm_date(out["date"], column="date")
    out["instrument_id"] = out["instrument_id"].astype(str)
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["close_adj"] = pd.to_numeric(out["close_adj"], errors="coerce")
    if out["close_adj"].isna().any() or (out["close_adj"] <= 0).any():
        raise ValueError("adjusted_prices close_adj must be numeric > 0.")
    if out.duplicated(["date", "instrument_id"], keep=False).any():
        raise ValueError("adjusted_prices has duplicate (date, instrument_id).")
    return out


def _prepare_calendar(path: Path) -> pd.DataFrame:
    frame = read_parquet(path)
    assert_schema(frame, CALENDAR_SCHEMA)
    out = frame.copy()
    out["date"] = _norm_date(out["date"], column="date")
    out["is_session"] = out["is_session"].astype(bool)
    if out.duplicated(["date"], keep=False).any():
        raise ValueError("trading_calendar has duplicate date rows.")
    return out


def _forward_returns(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices.sort_values(["instrument_id", "date"]).copy()
    out["return_end_date"] = out.groupby("instrument_id", sort=False)["date"].shift(-1)
    out["close_next"] = out.groupby("instrument_id", sort=False)["close_adj"].shift(-1)
    out["realized_return"] = out["close_next"] / out["close_adj"] - 1.0
    out = out.dropna(subset=["return_end_date", "realized_return"]).copy()
    out = out.rename(columns={"date": "return_start_date"})
    return out[["return_start_date", "return_end_date", "instrument_id", "realized_return"]]


def run_backtest_engine(
    *,
    holdings_path: str | Path | None = None,
    costs_daily_path: str | Path | None = None,
    adjusted_prices_path: str | Path | None = None,
    trading_calendar_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_name: str | None = None,
    label_name: str | None = None,
    portfolio_modes: Iterable[str] | None = None,
    run_id: str = MODULE_VERSION,
) -> BacktestEngineResult:
    logger = get_logger("backtest.engine")

    if holdings_path:
        holdings_source = Path(holdings_path).expanduser().resolve()
    else:
        ex_path = data_dir() / "execution" / "execution_holdings.parquet"
        lg_path = data_dir() / "portfolio" / "portfolio_holdings.parquet"
        holdings_source = ex_path if ex_path.exists() else lg_path

    costs_source = Path(costs_daily_path).expanduser().resolve() if costs_daily_path else (data_dir() / "execution" / "costs_daily.parquet")
    prices_source = Path(adjusted_prices_path).expanduser().resolve() if adjusted_prices_path else (data_dir() / "price" / "adjusted_prices.parquet")
    calendar_source = Path(trading_calendar_path).expanduser().resolve() if trading_calendar_path else (data_dir() / "reference" / "trading_calendar.parquet")

    holdings, holdings_mode = _prepare_holdings(holdings_source)
    costs = _prepare_costs(costs_source)
    prices = _prepare_prices(prices_source)
    calendar = _prepare_calendar(calendar_source)

    selected_model = _pick_unique(holdings, "model_name", model_name)
    selected_label = _pick_unique(holdings, "label_name", label_name)
    holdings = holdings[(holdings["model_name"] == selected_model) & (holdings["label_name"] == selected_label)].copy()
    costs = costs[(costs["model_name"] == selected_model) & (costs["label_name"] == selected_label)].copy()

    modes = _norm_modes(portfolio_modes)
    if modes:
        holdings = holdings[holdings["portfolio_mode"].isin(set(modes))].copy()
        costs = costs[costs["portfolio_mode"].isin(set(modes))].copy()
        if holdings.empty:
            raise ValueError(f"No holdings rows left after portfolio_mode filter: {list(modes)}")

    sessions = set(calendar.loc[calendar["is_session"], "date"].tolist())

    before_exec = len(holdings)
    holdings = holdings[holdings["is_executable"]].copy()
    rows_dropped_non_executable = int(before_exec - len(holdings))

    before_session = len(holdings)
    holdings = holdings[holdings["execution_date"].isin(sessions)].copy()
    rows_dropped_non_session_execution_date = int(before_session - len(holdings))
    if holdings.empty:
        raise ValueError("No executable holdings rows left after execution_date session filter.")

    checks = holdings.groupby(["execution_date", "portfolio_mode"], as_index=False).agg(
        long_sum=("execution_weight", lambda s: float(s[s > 0].sum())),
        short_sum=("execution_weight", lambda s: float(s[s < 0].sum())),
    )
    for row in checks.itertuples(index=False):
        if row.portfolio_mode == "long_only_top_n":
            if not np.isclose(float(row.long_sum), 1.0, atol=1e-12):
                raise ValueError(f"Invalid long weight sum for long_only_top_n on {pd.Timestamp(row.execution_date).date()}.")
            if not np.isclose(float(row.short_sum), 0.0, atol=1e-12):
                raise ValueError(f"Invalid short weight sum for long_only_top_n on {pd.Timestamp(row.execution_date).date()}.")
        if row.portfolio_mode == "long_short_top_bottom_n":
            if not np.isclose(float(row.long_sum), 1.0, atol=1e-12):
                raise ValueError(f"Invalid long weight sum for long_short_top_bottom_n on {pd.Timestamp(row.execution_date).date()}.")
            if not np.isclose(float(row.short_sum), -1.0, atol=1e-12):
                raise ValueError(f"Invalid short weight sum for long_short_top_bottom_n on {pd.Timestamp(row.execution_date).date()}.")

    returns = _forward_returns(prices)
    joined = holdings.merge(
        returns,
        left_on=["execution_date", "instrument_id"],
        right_on=["return_start_date", "instrument_id"],
        how="left",
    )
    rows_dropped_missing_tplus1_return = int(joined["realized_return"].isna().sum())
    contrib = joined[joined["realized_return"].notna()].copy()
    if contrib.empty:
        raise ValueError("No executable holdings rows have usable execution_date->next_session returns.")

    contrib["realized_return"] = pd.to_numeric(contrib["realized_return"], errors="coerce")
    contrib["contribution"] = contrib["execution_weight"].astype(float) * contrib["realized_return"]
    contrib["date"] = contrib["return_start_date"]

    contrib = contrib[
        [
            "date",
            "signal_date",
            "execution_date",
            "return_start_date",
            "return_end_date",
            "instrument_id",
            "ticker",
            "portfolio_mode",
            "model_name",
            "label_name",
            "target_weight",
            "execution_weight",
            "realized_return",
            "contribution",
        ]
    ].sort_values(["date", "portfolio_mode", "instrument_id"]).reset_index(drop=True)
    assert_schema(contrib, BACKTEST_CONTRIB_SCHEMA)

    daily = (
        contrib.groupby(["date", "portfolio_mode", "model_name", "label_name"], as_index=False)
        .agg(gross_return=("contribution", "sum"), n_positions=("instrument_id", "count"))
        .sort_values(["portfolio_mode", "date"])
        .reset_index(drop=True)
    )
    map_end = (
        contrib[["date", "return_end_date"]]
        .drop_duplicates(subset=["date"])
        .rename(columns={"date": "return_start_date"})
    )
    daily = daily.rename(columns={"date": "return_start_date"})
    daily = daily.merge(map_end, on="return_start_date", how="left")
    daily["date"] = daily["return_start_date"]

    daily = daily.merge(
        costs[["cost_date", "portfolio_mode", "model_name", "label_name", "total_cost"]],
        left_on=["return_start_date", "portfolio_mode", "model_name", "label_name"],
        right_on=["cost_date", "portfolio_mode", "model_name", "label_name"],
        how="left",
    )
    daily["total_cost"] = pd.to_numeric(daily["total_cost"], errors="coerce").fillna(0.0)
    daily["net_return"] = daily["gross_return"] - daily["total_cost"]

    if not np.allclose(
        (daily["gross_return"] - daily["total_cost"]).to_numpy(dtype=float),
        daily["net_return"].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("net_return must equal gross_return - total_cost.")

    for _, idx in daily.groupby(["portfolio_mode", "model_name", "label_name"], sort=False).groups.items():
        block = daily.loc[idx].sort_values("return_start_date").copy()
        gross_eq = (1.0 + block["gross_return"].astype(float)).cumprod()
        net_eq = (1.0 + block["net_return"].astype(float)).cumprod()
        dd = net_eq / net_eq.cummax() - 1.0
        daily.loc[block.index, "gross_equity"] = gross_eq.to_numpy(dtype=float)
        daily.loc[block.index, "net_equity"] = net_eq.to_numpy(dtype=float)
        daily.loc[block.index, "drawdown_net"] = dd.to_numpy(dtype=float)

    if (daily["drawdown_net"] > EPS).any():
        raise ValueError("drawdown_net must be <= 0.")

    chk = (
        contrib.groupby(["date", "portfolio_mode", "model_name", "label_name"], as_index=False)["contribution"]
        .sum()
        .rename(columns={"contribution": "contrib_sum"})
        .merge(daily, on=["date", "portfolio_mode", "model_name", "label_name"], how="left")
    )
    if not np.allclose(chk["contrib_sum"].to_numpy(dtype=float), chk["gross_return"].to_numpy(dtype=float), atol=1e-12, rtol=0.0):
        raise ValueError("Contributions do not aggregate to gross_return.")

    daily["n_positions"] = pd.to_numeric(daily["n_positions"], errors="coerce").astype("int64")
    daily = daily[
        [
            "date",
            "return_start_date",
            "return_end_date",
            "portfolio_mode",
            "model_name",
            "label_name",
            "gross_return",
            "total_cost",
            "net_return",
            "gross_equity",
            "net_equity",
            "drawdown_net",
            "n_positions",
        ]
    ].sort_values(["return_start_date", "portfolio_mode"]).reset_index(drop=True)
    assert_schema(daily, BACKTEST_DAILY_SCHEMA)

    cfg = _cfg_hash(
        {
            "version": MODULE_VERSION,
            "holdings_path": str(holdings_source),
            "costs_daily_path": str(costs_source),
            "adjusted_prices_path": str(prices_source),
            "trading_calendar_path": str(calendar_source),
            "model_name": selected_model,
            "label_name": selected_label,
            "portfolio_modes": sorted(daily["portfolio_mode"].astype(str).unique().tolist()),
            "holdings_input_mode": holdings_mode,
            "return_convention": "execution_weight_at_execution_date_apply_to_close_adj_execution_date_to_next_session",
            "cost_convention": "cost_date_aligned_to_return_start_date",
        }
    )
    built_ts = datetime.now(UTC).isoformat()

    daily["run_id"] = run_id
    daily["config_hash"] = cfg
    daily["built_ts_utc"] = built_ts
    contrib["run_id"] = run_id
    contrib["config_hash"] = cfg
    contrib["built_ts_utc"] = built_ts

    out_dir = Path(output_dir).expanduser().resolve() if output_dir else (data_dir() / "backtest")
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_path = write_parquet(daily, out_dir / "backtest_daily.parquet", schema_name=BACKTEST_DAILY_SCHEMA.name, run_id=run_id)
    contrib_path = write_parquet(contrib, out_dir / "backtest_contributions.parquet", schema_name=BACKTEST_CONTRIB_SCHEMA.name, run_id=run_id)

    mode_summaries: list[dict[str, Any]] = []
    for mode, block in daily.groupby("portfolio_mode", sort=True):
        block = block.sort_values("return_start_date")
        mode_summaries.append(
            {
                "portfolio_mode": str(mode),
                "start_date": str(pd.Timestamp(block["return_start_date"].min()).date()),
                "end_date": str(pd.Timestamp(block["return_start_date"].max()).date()),
                "n_dates": int(len(block)),
                "cumulative_gross_return": _to_float(block["gross_equity"].iloc[-1] - 1.0),
                "cumulative_net_return": _to_float(block["net_equity"].iloc[-1] - 1.0),
                "mean_daily_gross_return": _to_float(block["gross_return"].mean()),
                "mean_daily_net_return": _to_float(block["net_return"].mean()),
                "std_daily_net_return": _to_float(block["net_return"].std(ddof=0)),
                "positive_net_return_rate": _to_float((block["net_return"] > 0).mean()),
                "max_drawdown_net": _to_float(block["drawdown_net"].min()),
                "total_cost_paid": _to_float(block["total_cost"].sum()),
            }
        )

    gross_vals = [float(x["cumulative_gross_return"]) for x in mode_summaries if x.get("cumulative_gross_return") is not None]
    net_vals = [float(x["cumulative_net_return"]) for x in mode_summaries if x.get("cumulative_net_return") is not None]

    summary = {
        "built_ts_utc": built_ts,
        "run_id": run_id,
        "config_hash": cfg,
        "module_version": MODULE_VERSION,
        "model_name": selected_model,
        "label_name": selected_label,
        "portfolio_modes": sorted(daily["portfolio_mode"].astype(str).unique().tolist()),
        "start_date": str(pd.Timestamp(daily["return_start_date"].min()).date()),
        "end_date": str(pd.Timestamp(daily["return_start_date"].max()).date()),
        "n_dates": int(daily["return_start_date"].nunique()),
        "cumulative_gross_return": _to_float(float(np.mean(gross_vals)) if gross_vals else None),
        "cumulative_net_return": _to_float(float(np.mean(net_vals)) if net_vals else None),
        "mean_daily_gross_return": _to_float(daily["gross_return"].mean()),
        "mean_daily_net_return": _to_float(daily["net_return"].mean()),
        "std_daily_net_return": _to_float(daily["net_return"].std(ddof=0)),
        "positive_net_return_rate": _to_float((daily["net_return"] > 0).mean()),
        "max_drawdown_net": _to_float(daily["drawdown_net"].min()),
        "total_cost_paid": _to_float(daily["total_cost"].sum()),
        "return_convention": "execution_weight_at_execution_date_apply_to_close_adj_execution_date_to_next_session",
        "cost_convention": "total_cost_applied_on_cost_date_aligned_to_return_start_date",
        "holdings_input_mode": holdings_mode,
        "rows_dropped_non_executable": rows_dropped_non_executable,
        "rows_dropped_non_session_execution_date": rows_dropped_non_session_execution_date,
        "rows_dropped_missing_tplus1_return": rows_dropped_missing_tplus1_return,
        "mode_summaries": mode_summaries,
        "output_paths": {
            "backtest_daily": str(daily_path),
            "backtest_contributions": str(contrib_path),
        },
    }
    summary_path = out_dir / "backtest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "backtest_engine_built",
        run_id=run_id,
        model_name=selected_model,
        label_name=selected_label,
        row_count_daily=int(len(daily)),
        row_count_contributions=int(len(contrib)),
        holdings_input_mode=holdings_mode,
        output_dir=str(out_dir),
    )

    return BacktestEngineResult(
        backtest_daily_path=daily_path,
        backtest_contributions_path=contrib_path,
        backtest_summary_path=summary_path,
        row_count_daily=int(len(daily)),
        row_count_contributions=int(len(contrib)),
        model_name=selected_model,
        label_name=selected_label,
        config_hash=cfg,
    )


def _parse_csv_strings(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return values


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run daily backtest net-of-costs with explicit execution_date/cost_date timing."
    )
    parser.add_argument("--holdings-path", type=str, default=None)
    parser.add_argument("--costs-daily-path", type=str, default=None)
    parser.add_argument("--adjusted-prices-path", type=str, default=None)
    parser.add_argument("--trading-calendar-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--label-name", type=str, default=None)
    parser.add_argument("--portfolio-modes", type=_parse_csv_strings, default=tuple())
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_backtest_engine(
        holdings_path=args.holdings_path,
        costs_daily_path=args.costs_daily_path,
        adjusted_prices_path=args.adjusted_prices_path,
        trading_calendar_path=args.trading_calendar_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        label_name=args.label_name,
        portfolio_modes=args.portfolio_modes,
        run_id=args.run_id,
    )
    print("Backtest engine built:")
    print(f"- backtest_daily: {result.backtest_daily_path}")
    print(f"- backtest_contributions: {result.backtest_contributions_path}")
    print(f"- backtest_summary: {result.backtest_summary_path}")
    print(f"- row_count_daily: {result.row_count_daily}")
    print(f"- row_count_contributions: {result.row_count_contributions}")
    print(f"- model_name: {result.model_name}")
    print(f"- label_name: {result.label_name}")


if __name__ == "__main__":
    main()
