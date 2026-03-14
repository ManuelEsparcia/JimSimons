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

# Allow direct script execution: `python simons_smallcap_swing/portfolio/construct_portfolio.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "construct_portfolio_mvp_v1"
DEFAULT_SPLIT_ROLES: tuple[str, ...] = ("valid", "test")
MODE_LONG_ONLY_TOP_N = "long_only_top_n"
MODE_LONG_SHORT_TOP_BOTTOM_N = "long_short_top_bottom_n"
DEFAULT_PORTFOLIO_MODES: tuple[str, ...] = (
    MODE_LONG_ONLY_TOP_N,
    MODE_LONG_SHORT_TOP_BOTTOM_N,
)
SUPPORTED_PORTFOLIO_MODES: tuple[str, ...] = (
    MODE_LONG_ONLY_TOP_N,
    MODE_LONG_SHORT_TOP_BOTTOM_N,
)
DEFAULT_TOP_N = 20
DEFAULT_BOTTOM_N = 20
EPS = 1e-12

SIGNALS_INPUT_SCHEMA = DataSchema(
    name="construct_portfolio_signals_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("split_role", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("raw_score", "number", nullable=True),
        ColumnSpec("rank_pct", "number", nullable=True),
    ),
    primary_key=("date", "instrument_id", "model_name", "label_name"),
    allow_extra_columns=True,
)

UNIVERSE_INPUT_SCHEMA = DataSchema(
    name="construct_portfolio_universe_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("is_eligible", "bool", nullable=False),
    ),
    primary_key=("date", "instrument_id"),
    allow_extra_columns=True,
)

PORTFOLIO_HOLDINGS_SCHEMA = DataSchema(
    name="portfolio_holdings_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("portfolio_mode", "string", nullable=False),
        ColumnSpec("side", "string", nullable=False),
        ColumnSpec("raw_score", "float64", nullable=False),
        ColumnSpec("rank_pct", "float64", nullable=True),
        ColumnSpec("target_weight", "float64", nullable=False),
        ColumnSpec("gross_exposure_side", "float64", nullable=False),
        ColumnSpec("net_exposure", "float64", nullable=False),
    ),
    primary_key=("date", "instrument_id", "portfolio_mode", "model_name", "label_name"),
    allow_extra_columns=True,
)

PORTFOLIO_REBALANCE_SCHEMA = DataSchema(
    name="portfolio_rebalance_mvp",
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
        ColumnSpec("turnover_contribution", "float64", nullable=False),
        ColumnSpec("portfolio_mode", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
    ),
    primary_key=("date", "instrument_id", "portfolio_mode", "model_name", "label_name"),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class ConstructPortfolioResult:
    holdings_path: Path
    rebalance_path: Path
    summary_path: Path
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


def _normalize_split_roles(split_roles: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(sorted({str(item).strip() for item in split_roles if str(item).strip()}))
    if not normalized:
        raise ValueError("At least one split_role is required.")
    return normalized


def _normalize_portfolio_modes(modes: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(sorted({str(item).strip() for item in modes if str(item).strip()}))
    if not normalized:
        raise ValueError("At least one portfolio_mode is required.")
    invalid = sorted(set(normalized) - set(SUPPORTED_PORTFOLIO_MODES))
    if invalid:
        raise ValueError(
            f"Unsupported portfolio_mode values: {invalid}. "
            f"Supported modes: {list(SUPPORTED_PORTFOLIO_MODES)}"
        )
    return normalized


def _select_unique_value(frame: pd.DataFrame, column: str, *, provided: str | None) -> str:
    if provided is not None:
        selected = str(provided).strip()
        if not selected:
            raise ValueError(f"Provided {column} is empty.")
        filtered = frame[frame[column].astype(str) == selected]
        if filtered.empty:
            raise ValueError(f"No rows left after filtering {column}='{selected}'.")
        return selected

    unique_vals = sorted(frame[column].astype(str).unique().tolist())
    if len(unique_vals) != 1:
        raise ValueError(
            f"Expected exactly one {column} per run. "
            f"Observed {column} values: {unique_vals}. "
            f"Pass --{column.replace('_', '-')} explicitly."
        )
    return str(unique_vals[0])


def _prepare_signals(source: Path) -> pd.DataFrame:
    frame = read_parquet(source)
    assert_schema(frame, SIGNALS_INPUT_SCHEMA)
    frame = frame.copy()
    frame["date"] = _normalize_date(frame["date"], column="date")
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["split_role"] = frame["split_role"].astype(str)
    frame["label_name"] = frame["label_name"].astype(str)
    frame["model_name"] = frame["model_name"].astype(str)
    frame["raw_score"] = pd.to_numeric(frame["raw_score"], errors="coerce")
    frame["rank_pct"] = pd.to_numeric(frame["rank_pct"], errors="coerce")
    if "horizon_days" in frame.columns:
        frame["horizon_days"] = pd.to_numeric(frame["horizon_days"], errors="coerce").astype("Int64")
    if "split_name" in frame.columns:
        frame["split_name"] = frame["split_name"].astype(str)

    if frame.duplicated(["date", "instrument_id", "model_name", "label_name"], keep=False).any():
        raise ValueError(
            "signals_daily contains duplicate (date, instrument_id, model_name, label_name) rows."
        )
    return frame


def _prepare_universe(source: Path) -> pd.DataFrame:
    frame = read_parquet(source)
    assert_schema(frame, UNIVERSE_INPUT_SCHEMA)
    frame = frame.copy()
    frame["date"] = _normalize_date(frame["date"], column="date")
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["is_eligible"] = frame["is_eligible"].astype(bool)
    if frame.duplicated(["date", "instrument_id"], keep=False).any():
        raise ValueError("universe_history contains duplicate (date, instrument_id) rows.")
    return frame


def _build_date_holdings(
    *,
    date_frame: pd.DataFrame,
    portfolio_mode: str,
    model_name: str,
    label_name: str,
    top_n: int,
    bottom_n: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    date_value = pd.Timestamp(date_frame["date"].iloc[0]).normalize()
    n_eligible = int(date_frame["instrument_id"].nunique())
    scored = date_frame[date_frame["raw_score"].notna()].copy()
    n_scored = int(len(scored))
    status = {
        "date": date_value,
        "portfolio_mode": portfolio_mode,
        "n_eligible": n_eligible,
        "n_scored": n_scored,
        "n_long": 0,
        "n_short": 0,
        "n_selected": 0,
        "is_executable": False,
        "skip_reason": None,
    }

    if n_scored <= 0:
        status["skip_reason"] = "no_score"
        return pd.DataFrame(), status

    ranked_desc = scored.sort_values(["raw_score", "instrument_id"], ascending=[False, True]).copy()
    if portfolio_mode == MODE_LONG_ONLY_TOP_N:
        n_long = min(int(top_n), int(len(ranked_desc)))
        if n_long <= 0:
            status["skip_reason"] = "missing_long"
            return pd.DataFrame(), status
        long_sel = ranked_desc.head(n_long).copy()
        long_sel["side"] = "long"
        long_sel["target_weight"] = 1.0 / float(n_long)
        selected = long_sel
    elif portfolio_mode == MODE_LONG_SHORT_TOP_BOTTOM_N:
        n_long = min(int(top_n), int(len(ranked_desc)))
        if n_long <= 0:
            status["skip_reason"] = "missing_long"
            return pd.DataFrame(), status
        long_sel = ranked_desc.head(n_long).copy()
        remaining = scored[~scored["instrument_id"].isin(long_sel["instrument_id"])].copy()
        ranked_asc = remaining.sort_values(["raw_score", "instrument_id"], ascending=[True, True]).copy()
        n_short = min(int(bottom_n), int(len(ranked_asc)))
        if n_short <= 0:
            status["skip_reason"] = "missing_short"
            return pd.DataFrame(), status
        short_sel = ranked_asc.head(n_short).copy()
        long_sel["side"] = "long"
        short_sel["side"] = "short"
        long_sel["target_weight"] = 1.0 / float(n_long)
        short_sel["target_weight"] = -1.0 / float(n_short)
        selected = pd.concat([long_sel, short_sel], ignore_index=True)
    else:
        raise ValueError(f"Unsupported portfolio_mode '{portfolio_mode}'.")

    selected["date"] = date_value
    selected["model_name"] = model_name
    selected["label_name"] = label_name
    selected["portfolio_mode"] = portfolio_mode
    long_exposure = float(selected.loc[selected["target_weight"] > 0.0, "target_weight"].sum())
    short_exposure = float(abs(selected.loc[selected["target_weight"] < 0.0, "target_weight"].sum()))
    selected["gross_exposure_side"] = np.where(
        selected["side"] == "long",
        long_exposure,
        short_exposure,
    )
    selected["net_exposure"] = float(selected["target_weight"].sum())
    selected = selected[
        [
            "date",
            "instrument_id",
            "ticker",
            "model_name",
            "label_name",
            "portfolio_mode",
            "side",
            "raw_score",
            "rank_pct",
            "target_weight",
            "gross_exposure_side",
            "net_exposure",
            "split_role",
            *(("horizon_days",) if "horizon_days" in selected.columns else ()),
            *(("split_name",) if "split_name" in selected.columns else ()),
        ]
    ].sort_values(["target_weight", "raw_score", "instrument_id"], ascending=[False, False, True]).reset_index(
        drop=True
    )

    status["n_long"] = int((selected["side"] == "long").sum())
    status["n_short"] = int((selected["side"] == "short").sum())
    status["n_selected"] = int(len(selected))
    status["is_executable"] = True
    return selected, status


def _build_rebalance(
    *,
    mode_dates: list[pd.Timestamp],
    holdings_mode: pd.DataFrame,
    portfolio_mode: str,
    model_name: str,
    label_name: str,
) -> pd.DataFrame:
    if holdings_mode.empty and not mode_dates:
        return pd.DataFrame(
            columns=[
                "date",
                "instrument_id",
                "ticker",
                "prev_weight",
                "target_weight",
                "weight_change",
                "abs_weight_change",
                "entered_flag",
                "exited_flag",
                "turnover_contribution",
                "portfolio_mode",
                "model_name",
                "label_name",
            ]
        )

    rows: list[dict[str, Any]] = []
    prev_weights: dict[str, float] = {}
    prev_tickers: dict[str, str] = {}
    mode_holdings = holdings_mode.copy()

    for date_value in sorted(pd.Timestamp(x).normalize() for x in mode_dates):
        current = mode_holdings[mode_holdings["date"] == date_value].copy()
        curr_weights = {
            str(row.instrument_id): float(row.target_weight)
            for row in current.itertuples(index=False)
        }
        curr_tickers = {
            str(row.instrument_id): str(row.ticker)
            for row in current.itertuples(index=False)
        }
        instrument_ids = sorted(set(prev_weights) | set(curr_weights))
        for instrument_id in instrument_ids:
            prev_weight = float(prev_weights.get(instrument_id, 0.0))
            target_weight = float(curr_weights.get(instrument_id, 0.0))
            weight_change = target_weight - prev_weight
            abs_weight_change = abs(weight_change)
            if abs_weight_change <= EPS:
                continue
            entered_flag = abs(prev_weight) <= EPS and abs(target_weight) > EPS
            exited_flag = abs(prev_weight) > EPS and abs(target_weight) <= EPS
            ticker = curr_tickers.get(instrument_id, prev_tickers.get(instrument_id))
            if ticker is None:
                raise ValueError(f"Missing ticker while building rebalance row for {instrument_id}.")
            rows.append(
                {
                    "date": date_value,
                    "instrument_id": instrument_id,
                    "ticker": str(ticker),
                    "prev_weight": prev_weight,
                    "target_weight": target_weight,
                    "weight_change": weight_change,
                    "abs_weight_change": abs_weight_change,
                    "entered_flag": bool(entered_flag),
                    "exited_flag": bool(exited_flag),
                    "turnover_contribution": 0.5 * abs_weight_change,
                    "portfolio_mode": portfolio_mode,
                    "model_name": model_name,
                    "label_name": label_name,
                }
            )
        prev_weights = curr_weights
        prev_tickers = curr_tickers

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "instrument_id",
                "ticker",
                "prev_weight",
                "target_weight",
                "weight_change",
                "abs_weight_change",
                "entered_flag",
                "exited_flag",
                "turnover_contribution",
                "portfolio_mode",
                "model_name",
                "label_name",
            ]
        )
    return pd.DataFrame(rows).sort_values(["date", "instrument_id"]).reset_index(drop=True)


def _validate_mode_holdings(holdings_mode: pd.DataFrame, portfolio_mode: str) -> None:
    if holdings_mode.empty:
        return
    by_date = holdings_mode.groupby("date", as_index=False).agg(
        long_sum=("target_weight", lambda s: float(s[s > 0.0].sum())),
        short_sum=("target_weight", lambda s: float(s[s < 0.0].sum())),
    )
    for row in by_date.itertuples(index=False):
        if portfolio_mode == MODE_LONG_ONLY_TOP_N:
            if not np.isclose(float(row.long_sum), 1.0, atol=1e-12):
                raise ValueError(
                    f"Invalid long exposure for {portfolio_mode} on {pd.Timestamp(row.date).date()}: "
                    f"{row.long_sum}"
                )
            if not np.isclose(float(row.short_sum), 0.0, atol=1e-12):
                raise ValueError(
                    f"Invalid short exposure for {portfolio_mode} on {pd.Timestamp(row.date).date()}: "
                    f"{row.short_sum}"
                )
        elif portfolio_mode == MODE_LONG_SHORT_TOP_BOTTOM_N:
            if not np.isclose(float(row.long_sum), 1.0, atol=1e-12):
                raise ValueError(
                    f"Invalid long exposure for {portfolio_mode} on {pd.Timestamp(row.date).date()}: "
                    f"{row.long_sum}"
                )
            if not np.isclose(float(row.short_sum), -1.0, atol=1e-12):
                raise ValueError(
                    f"Invalid short exposure for {portfolio_mode} on {pd.Timestamp(row.date).date()}: "
                    f"{row.short_sum}"
                )


def run_construct_portfolio(
    *,
    signals_path: str | Path | None = None,
    universe_history_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_name: str | None = None,
    label_name: str | None = None,
    split_name: str | None = None,
    horizon_days: int | None = None,
    split_roles: Iterable[str] = DEFAULT_SPLIT_ROLES,
    portfolio_modes: Iterable[str] = DEFAULT_PORTFOLIO_MODES,
    top_n: int = DEFAULT_TOP_N,
    bottom_n: int = DEFAULT_BOTTOM_N,
    run_id: str = MODULE_VERSION,
) -> ConstructPortfolioResult:
    logger = get_logger("portfolio.construct_portfolio")

    top_n_int = int(top_n)
    bottom_n_int = int(bottom_n)
    if top_n_int <= 0:
        raise ValueError("top_n must be > 0.")
    if bottom_n_int <= 0:
        raise ValueError("bottom_n must be > 0.")

    selected_split_roles = _normalize_split_roles(split_roles)
    selected_modes = _normalize_portfolio_modes(portfolio_modes)

    signals_source = (
        Path(signals_path).expanduser().resolve()
        if signals_path
        else (data_dir() / "signals" / "signals_daily.parquet")
    )
    universe_source = (
        Path(universe_history_path).expanduser().resolve()
        if universe_history_path
        else (data_dir() / "universe" / "universe_history.parquet")
    )

    signals = _prepare_signals(signals_source)
    universe = _prepare_universe(universe_source)

    filter_stats: dict[str, int] = {"n_rows_signals_input": int(len(signals))}
    signals = signals[signals["split_role"].isin(set(selected_split_roles))].copy()
    filter_stats["n_rows_after_split_roles"] = int(len(signals))
    if signals.empty:
        raise ValueError(f"No rows left after split_role filter: {list(selected_split_roles)}")

    selected_model_name = _select_unique_value(signals, "model_name", provided=model_name)
    signals = signals[signals["model_name"] == selected_model_name].copy()
    selected_label_name = _select_unique_value(signals, "label_name", provided=label_name)
    signals = signals[signals["label_name"] == selected_label_name].copy()

    if "split_name" in signals.columns:
        if split_name is not None:
            selected_split_name = str(split_name).strip()
            if not selected_split_name:
                raise ValueError("Provided split_name is empty.")
            signals = signals[signals["split_name"] == selected_split_name].copy()
        unique_split_names = sorted(signals["split_name"].astype(str).unique().tolist())
        if len(unique_split_names) > 1:
            raise ValueError(
                "Expected exactly one split_name per run. "
                f"Observed split_name values: {unique_split_names}. "
                "Pass --split-name explicitly."
            )

    selected_horizon: int | None = None
    if "horizon_days" in signals.columns:
        if horizon_days is not None:
            signals = signals[signals["horizon_days"] == int(horizon_days)].copy()
        unique_horizons = sorted(signals["horizon_days"].dropna().astype(int).unique().tolist())
        if not unique_horizons:
            raise ValueError("No rows left after horizon_days filtering.")
        if len(unique_horizons) > 1:
            raise ValueError(
                "Expected exactly one horizon_days per run. "
                f"Observed horizon_days values: {unique_horizons}. "
                "Pass --horizon-days explicitly."
            )
        selected_horizon = int(unique_horizons[0])

    if signals.duplicated(["date", "instrument_id", "model_name", "label_name"], keep=False).any():
        raise ValueError(
            "signals_daily contains duplicate rows for (date, instrument_id, model_name, label_name) "
            "after filtering."
        )

    eligible = universe.loc[universe["is_eligible"], ["date", "instrument_id"]].copy()
    before_eligibility = len(signals)
    signals = signals.merge(eligible, on=["date", "instrument_id"], how="inner")
    filter_stats["n_rows_after_eligibility"] = int(len(signals))
    filter_stats["n_rows_excluded_by_eligibility"] = int(before_eligibility - len(signals))
    if signals.empty:
        raise ValueError("No signal rows left after applying eligible universe filter.")

    all_dates = sorted(pd.to_datetime(signals["date"]).dt.normalize().unique().tolist())
    if not all_dates:
        raise ValueError("No portfolio dates available after filtering.")

    holdings_blocks: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []
    mode_dates: dict[str, list[pd.Timestamp]] = {
        mode: [pd.Timestamp(x).normalize() for x in all_dates] for mode in selected_modes
    }

    for date_value in all_dates:
        date_frame = signals[signals["date"] == pd.Timestamp(date_value)].copy()
        for mode in selected_modes:
            mode_holdings, mode_status = _build_date_holdings(
                date_frame=date_frame,
                portfolio_mode=mode,
                model_name=selected_model_name,
                label_name=selected_label_name,
                top_n=top_n_int,
                bottom_n=bottom_n_int,
            )
            status_rows.append(mode_status)
            if not mode_holdings.empty:
                holdings_blocks.append(mode_holdings)

    if not holdings_blocks:
        raise ValueError("No holdings could be constructed for the selected configuration.")

    holdings = pd.concat(holdings_blocks, ignore_index=True)
    holdings = holdings.sort_values(["date", "portfolio_mode", "instrument_id"]).reset_index(drop=True)
    if holdings.duplicated(
        ["date", "instrument_id", "portfolio_mode", "model_name", "label_name"], keep=False
    ).any():
        raise ValueError("portfolio_holdings has duplicate logical PK rows.")

    for mode in selected_modes:
        _validate_mode_holdings(holdings[holdings["portfolio_mode"] == mode].copy(), mode)

    rebalance_blocks: list[pd.DataFrame] = []
    for mode in selected_modes:
        mode_holdings = holdings[holdings["portfolio_mode"] == mode].copy()
        rebalance_mode = _build_rebalance(
            mode_dates=mode_dates[mode],
            holdings_mode=mode_holdings,
            portfolio_mode=mode,
            model_name=selected_model_name,
            label_name=selected_label_name,
        )
        if not rebalance_mode.empty:
            rebalance_blocks.append(rebalance_mode)

    if rebalance_blocks:
        rebalance = pd.concat(rebalance_blocks, ignore_index=True)
        rebalance = rebalance.sort_values(["date", "portfolio_mode", "instrument_id"]).reset_index(drop=True)
    else:
        rebalance = pd.DataFrame(
            columns=[
                "date",
                "instrument_id",
                "ticker",
                "prev_weight",
                "target_weight",
                "weight_change",
                "abs_weight_change",
                "entered_flag",
                "exited_flag",
                "turnover_contribution",
                "portfolio_mode",
                "model_name",
                "label_name",
            ]
        )

    status_df = pd.DataFrame(status_rows).sort_values(["date", "portfolio_mode"]).reset_index(drop=True)

    config_hash = _config_hash(
        {
            "version": MODULE_VERSION,
            "signals_path": str(signals_source),
            "universe_history_path": str(universe_source),
            "model_name": selected_model_name,
            "label_name": selected_label_name,
            "split_name": split_name,
            "horizon_days": selected_horizon,
            "split_roles": list(selected_split_roles),
            "portfolio_modes": list(selected_modes),
            "top_n": top_n_int,
            "bottom_n": bottom_n_int,
        }
    )
    built_ts_utc = datetime.now(UTC).isoformat()

    holdings["run_id"] = run_id
    holdings["config_hash"] = config_hash
    holdings["built_ts_utc"] = built_ts_utc

    rebalance["run_id"] = run_id
    rebalance["config_hash"] = config_hash
    rebalance["built_ts_utc"] = built_ts_utc

    assert_schema(holdings, PORTFOLIO_HOLDINGS_SCHEMA)
    if not rebalance.empty:
        assert_schema(rebalance, PORTFOLIO_REBALANCE_SCHEMA)

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (data_dir() / "portfolio")
    target_dir.mkdir(parents=True, exist_ok=True)

    holdings_path = write_parquet(
        holdings,
        target_dir / "portfolio_holdings.parquet",
        schema_name=PORTFOLIO_HOLDINGS_SCHEMA.name,
        run_id=run_id,
    )

    if rebalance.empty:
        raise ValueError("portfolio_rebalance is empty; expected at least first-date entry rows.")
    rebalance_path = write_parquet(
        rebalance,
        target_dir / "portfolio_rebalance.parquet",
        schema_name=PORTFOLIO_REBALANCE_SCHEMA.name,
        run_id=run_id,
    )

    mode_summaries: list[dict[str, Any]] = []
    for mode in selected_modes:
        mode_status = status_df[status_df["portfolio_mode"] == mode].copy()
        mode_holdings = holdings[holdings["portfolio_mode"] == mode].copy()
        mode_rebalance = rebalance[rebalance["portfolio_mode"] == mode].copy()

        turnovers = (
            mode_rebalance.groupby("date", as_index=False)["turnover_contribution"]
            .sum()
            .rename(columns={"turnover_contribution": "turnover"})
        )
        turnovers = mode_status[["date"]].merge(turnovers, on="date", how="left")
        turnovers["turnover"] = pd.to_numeric(turnovers["turnover"], errors="coerce").fillna(0.0)

        exposures = (
            mode_holdings.groupby("date", as_index=False)
            .agg(
                gross_exposure=("target_weight", lambda s: float(np.abs(s).sum())),
                net_exposure=("target_weight", lambda s: float(s.sum())),
                n_holdings=("instrument_id", "nunique"),
                n_long=("side", lambda s: int((s == "long").sum())),
                n_short=("side", lambda s: int((s == "short").sum())),
            )
        )
        exposures = mode_status[["date"]].merge(exposures, on="date", how="left")
        for col in ("gross_exposure", "net_exposure", "n_holdings", "n_long", "n_short"):
            exposures[col] = pd.to_numeric(exposures[col], errors="coerce").fillna(0.0)

        mode_summaries.append(
            {
                "model_name": selected_model_name,
                "label_name": selected_label_name,
                "horizon_days": selected_horizon,
                "portfolio_mode": mode,
                "n_dates": int(len(mode_status)),
                "n_executable_dates": int(mode_status["is_executable"].sum()),
                "avg_n_holdings": _to_float_or_none(exposures["n_holdings"].mean()),
                "avg_n_long": _to_float_or_none(exposures["n_long"].mean()),
                "avg_n_short": _to_float_or_none(exposures["n_short"].mean()),
                "avg_turnover": _to_float_or_none(turnovers["turnover"].mean()),
                "median_turnover": _to_float_or_none(turnovers["turnover"].median()),
                "max_turnover": _to_float_or_none(turnovers["turnover"].max()),
                "avg_gross_exposure": _to_float_or_none(exposures["gross_exposure"].mean()),
                "avg_net_exposure": _to_float_or_none(exposures["net_exposure"].mean()),
                "n_entries": int(mode_rebalance["entered_flag"].sum()) if not mode_rebalance.empty else 0,
                "n_exits": int(mode_rebalance["exited_flag"].sum()) if not mode_rebalance.empty else 0,
            }
        )

    summary_payload = {
        "created_at_utc": built_ts_utc,
        "run_id": run_id,
        "config_hash": config_hash,
        "module_version": MODULE_VERSION,
        "signals_path": str(signals_source),
        "universe_history_path": str(universe_source),
        "model_name": selected_model_name,
        "label_name": selected_label_name,
        "horizon_days": selected_horizon,
        "split_roles_included": list(selected_split_roles),
        "portfolio_modes": list(selected_modes),
        "top_n": top_n_int,
        "bottom_n": bottom_n_int,
        "filter_stats": filter_stats,
        "n_rows_holdings": int(len(holdings)),
        "n_rows_rebalance": int(len(rebalance)),
        "mode_summaries": mode_summaries,
        "output_paths": {
            "portfolio_holdings": str(holdings_path),
            "portfolio_rebalance": str(rebalance_path),
        },
    }
    summary_path = target_dir / "portfolio_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "portfolio_constructed",
        run_id=run_id,
        model_name=selected_model_name,
        label_name=selected_label_name,
        portfolio_modes=list(selected_modes),
        row_count_holdings=int(len(holdings)),
        row_count_rebalance=int(len(rebalance)),
        holdings_path=str(holdings_path),
        rebalance_path=str(rebalance_path),
    )

    return ConstructPortfolioResult(
        holdings_path=holdings_path,
        rebalance_path=rebalance_path,
        summary_path=summary_path,
        row_count_holdings=int(len(holdings)),
        row_count_rebalance=int(len(rebalance)),
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
        description="Construct formal daily target holdings and rebalance deltas from ranked signals."
    )
    parser.add_argument("--signals-path", type=str, default=None)
    parser.add_argument("--universe-history-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--label-name", type=str, default=None)
    parser.add_argument("--split-name", type=str, default=None)
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument("--split-roles", type=_parse_csv_strings, default=DEFAULT_SPLIT_ROLES)
    parser.add_argument("--portfolio-modes", type=_parse_csv_strings, default=DEFAULT_PORTFOLIO_MODES)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--bottom-n", type=int, default=DEFAULT_BOTTOM_N)
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_construct_portfolio(
        signals_path=args.signals_path,
        universe_history_path=args.universe_history_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        label_name=args.label_name,
        split_name=args.split_name,
        horizon_days=args.horizon_days,
        split_roles=args.split_roles,
        portfolio_modes=args.portfolio_modes,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        run_id=args.run_id,
    )
    print("Portfolio constructed:")
    print(f"- holdings: {result.holdings_path}")
    print(f"- rebalance: {result.rebalance_path}")
    print(f"- summary: {result.summary_path}")
    print(f"- row_count_holdings: {result.row_count_holdings}")
    print(f"- row_count_rebalance: {result.row_count_rebalance}")
    print(f"- model_name: {result.model_name}")
    print(f"- label_name: {result.label_name}")


if __name__ == "__main__":
    main()
