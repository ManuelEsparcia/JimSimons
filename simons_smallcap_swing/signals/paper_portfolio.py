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

# Allow direct script execution: `python simons_smallcap_swing/signals/paper_portfolio.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "paper_portfolio_mvp_v1"
DEFAULT_SPLIT_ROLES: tuple[str, ...] = ("valid", "test")
MODE_LONG_ONLY_TOP = "long_only_top"
MODE_LONG_SHORT_TOP_BOTTOM = "long_short_top_bottom"
MODE_SHORT_ONLY_BOTTOM = "short_only_bottom"
DEFAULT_PORTFOLIO_MODES: tuple[str, ...] = (
    MODE_LONG_ONLY_TOP,
    MODE_LONG_SHORT_TOP_BOTTOM,
)
SUPPORTED_PORTFOLIO_MODES: tuple[str, ...] = (
    MODE_LONG_ONLY_TOP,
    MODE_LONG_SHORT_TOP_BOTTOM,
    MODE_SHORT_ONLY_BOTTOM,
)

SIGNALS_INPUT_SCHEMA = DataSchema(
    name="paper_portfolio_signals_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("split_role", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("is_top", "bool", nullable=False),
        ColumnSpec("is_bottom", "bool", nullable=False),
    ),
    primary_key=("date", "instrument_id", "model_name", "label_name"),
    allow_extra_columns=True,
)

LABELS_INPUT_SCHEMA = DataSchema(
    name="paper_portfolio_labels_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("horizon_days", "int64", nullable=False),
        ColumnSpec("label_value", "number", nullable=False),
    ),
    primary_key=("date", "instrument_id", "horizon_days", "label_name"),
    allow_extra_columns=True,
)

PAPER_DAILY_SCHEMA = DataSchema(
    name="paper_portfolio_daily_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("portfolio_mode", "string", nullable=False),
        ColumnSpec("n_long", "int64", nullable=False),
        ColumnSpec("n_short", "int64", nullable=False),
        ColumnSpec("gross_long_return", "float64", nullable=True),
        ColumnSpec("gross_short_return", "float64", nullable=True),
        ColumnSpec("gross_portfolio_return", "float64", nullable=True),
        ColumnSpec("is_executable", "bool", nullable=False),
        ColumnSpec("skip_reason", "string", nullable=True),
    ),
    primary_key=("date", "portfolio_mode", "model_name", "label_name"),
    allow_extra_columns=True,
)

PAPER_POSITIONS_SCHEMA = DataSchema(
    name="paper_portfolio_positions_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("portfolio_mode", "string", nullable=False),
        ColumnSpec("side", "string", nullable=False),
        ColumnSpec("weight", "float64", nullable=False),
        ColumnSpec("realized_return", "float64", nullable=False),
        ColumnSpec("contribution", "float64", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
    ),
    primary_key=("date", "instrument_id", "portfolio_mode", "model_name", "label_name"),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class PaperPortfolioResult:
    daily_path: Path
    positions_path: Path
    summary_path: Path
    row_count_daily: int
    row_count_positions: int
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
    frame["is_top"] = frame["is_top"].astype(bool)
    frame["is_bottom"] = frame["is_bottom"].astype(bool)
    if "horizon_days" in frame.columns:
        frame["horizon_days"] = pd.to_numeric(frame["horizon_days"], errors="coerce").astype("Int64")

    if frame.duplicated(["date", "instrument_id", "model_name", "label_name"], keep=False).any():
        raise ValueError("signals_daily contains duplicate (date, instrument_id, model_name, label_name) rows.")
    return frame


def _prepare_labels(source: Path) -> pd.DataFrame:
    frame = read_parquet(source)
    assert_schema(frame, LABELS_INPUT_SCHEMA)
    frame = frame.copy()
    frame["date"] = _normalize_date(frame["date"], column="date")
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["label_name"] = frame["label_name"].astype(str)
    frame["horizon_days"] = pd.to_numeric(frame["horizon_days"], errors="coerce").astype("Int64")
    frame["label_value"] = pd.to_numeric(frame["label_value"], errors="coerce")
    if frame["label_value"].isna().any():
        raise ValueError("labels_forward has null/non-numeric label_value values.")
    if frame.duplicated(["date", "instrument_id", "horizon_days", "label_name"], keep=False).any():
        raise ValueError("labels_forward contains duplicate (date, instrument_id, horizon_days, label_name) rows.")
    return frame


def _resolve_join_keys(
    *,
    signals: pd.DataFrame,
    labels: pd.DataFrame,
    horizon_days: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], int | None]:
    join_keys = ["date", "instrument_id", "label_name"]
    selected_horizon: int | None = None
    signals_has_h = "horizon_days" in signals.columns

    if signals_has_h:
        if horizon_days is not None:
            signals = signals[signals["horizon_days"] == int(horizon_days)].copy()
        if signals.empty:
            raise ValueError("No signal rows left after horizon_days filtering.")
        unique_signal_h = sorted(signals["horizon_days"].dropna().astype(int).unique().tolist())
        if len(unique_signal_h) != 1:
            raise ValueError(
                f"signals_daily has multiple horizon_days values: {unique_signal_h}. "
                "Pass --horizon-days explicitly or pre-filter input."
            )
        selected_horizon = unique_signal_h[0]

    if horizon_days is not None:
        labels = labels[labels["horizon_days"] == int(horizon_days)].copy()
    if labels.empty:
        raise ValueError("No label rows left after horizon_days filtering.")

    unique_label_h = sorted(labels["horizon_days"].dropna().astype(int).unique().tolist())
    if signals_has_h:
        labels = labels[labels["horizon_days"] == int(selected_horizon)].copy()
        if labels.empty:
            raise ValueError(
                f"No labels found for selected horizon_days={selected_horizon}."
            )
        join_keys.append("horizon_days")
    else:
        if len(unique_label_h) != 1:
            raise ValueError(
                "signals_daily has no horizon_days column but labels_forward has multiple horizons. "
                f"Observed label horizons: {unique_label_h}. Pass --horizon-days explicitly."
            )
        selected_horizon = unique_label_h[0]
        labels = labels[labels["horizon_days"] == int(selected_horizon)].copy()

    if labels.duplicated(join_keys, keep=False).any():
        raise ValueError(
            f"labels_forward has duplicate rows for join keys {join_keys}; join would be ambiguous."
        )
    if signals.duplicated(join_keys + ["model_name"], keep=False).any():
        raise ValueError(
            f"signals_daily has duplicate rows for logical keys {join_keys + ['model_name']}."
        )
    return signals, labels, join_keys, selected_horizon


def _build_one_mode_for_date(
    *,
    date_frame: pd.DataFrame,
    portfolio_mode: str,
    model_name: str,
    label_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    date_value = pd.Timestamp(date_frame["date"].iloc[0]).normalize()
    long_candidates = date_frame[date_frame["is_top"]].copy()
    short_candidates = date_frame[date_frame["is_bottom"]].copy()
    n_long = int(len(long_candidates))
    n_short = int(len(short_candidates))
    n_ranked = int(date_frame["instrument_id"].nunique())

    def _daily_row(
        *,
        executable: bool,
        reason: str | None,
        gross_long: float | None,
        gross_short: float | None,
        gross_portfolio: float | None,
    ) -> dict[str, Any]:
        return {
            "date": date_value,
            "model_name": model_name,
            "label_name": label_name,
            "portfolio_mode": portfolio_mode,
            "n_long": n_long,
            "n_short": n_short,
            "n_names_ranked": n_ranked,
            "gross_long_return": gross_long,
            "gross_short_return": gross_short,
            "gross_portfolio_return": gross_portfolio,
            "is_executable": bool(executable),
            "skip_reason": reason,
        }

    if portfolio_mode == MODE_LONG_ONLY_TOP:
        if n_long <= 0:
            return pd.DataFrame(), _daily_row(
                executable=False,
                reason="missing_long",
                gross_long=None,
                gross_short=None,
                gross_portfolio=None,
            )
        positions = long_candidates[["date", "instrument_id", "ticker", "split_role", "realized_return"]].copy()
        positions["portfolio_mode"] = portfolio_mode
        positions["side"] = "long"
        positions["weight"] = 1.0 / float(n_long)
        positions["contribution"] = positions["weight"] * positions["realized_return"]
        gross_long = float(long_candidates["realized_return"].mean())
        gross_port = float(positions["contribution"].sum())
        return positions, _daily_row(
            executable=True,
            reason=None,
            gross_long=gross_long,
            gross_short=None,
            gross_portfolio=gross_port,
        )

    if portfolio_mode == MODE_LONG_SHORT_TOP_BOTTOM:
        missing_parts: list[str] = []
        if n_long <= 0:
            missing_parts.append("long")
        if n_short <= 0:
            missing_parts.append("short")
        if missing_parts:
            return pd.DataFrame(), _daily_row(
                executable=False,
                reason="missing_" + "_".join(missing_parts),
                gross_long=None,
                gross_short=None,
                gross_portfolio=None,
            )

        long_pos = long_candidates[["date", "instrument_id", "ticker", "split_role", "realized_return"]].copy()
        long_pos["portfolio_mode"] = portfolio_mode
        long_pos["side"] = "long"
        long_pos["weight"] = 1.0 / float(n_long)

        short_pos = short_candidates[["date", "instrument_id", "ticker", "split_role", "realized_return"]].copy()
        short_pos["portfolio_mode"] = portfolio_mode
        short_pos["side"] = "short"
        short_pos["weight"] = -1.0 / float(n_short)

        positions = pd.concat([long_pos, short_pos], ignore_index=True)
        positions["contribution"] = positions["weight"] * positions["realized_return"]
        gross_long = float(long_candidates["realized_return"].mean())
        gross_short = float(short_candidates["realized_return"].mean())
        gross_port = float(positions["contribution"].sum())
        return positions, _daily_row(
            executable=True,
            reason=None,
            gross_long=gross_long,
            gross_short=gross_short,
            gross_portfolio=gross_port,
        )

    if portfolio_mode == MODE_SHORT_ONLY_BOTTOM:
        if n_short <= 0:
            return pd.DataFrame(), _daily_row(
                executable=False,
                reason="missing_short",
                gross_long=None,
                gross_short=None,
                gross_portfolio=None,
            )
        positions = short_candidates[["date", "instrument_id", "ticker", "split_role", "realized_return"]].copy()
        positions["portfolio_mode"] = portfolio_mode
        positions["side"] = "short"
        positions["weight"] = -1.0 / float(n_short)
        positions["contribution"] = positions["weight"] * positions["realized_return"]
        gross_short = float(short_candidates["realized_return"].mean())
        gross_port = float(positions["contribution"].sum())
        return positions, _daily_row(
            executable=True,
            reason=None,
            gross_long=None,
            gross_short=gross_short,
            gross_portfolio=gross_port,
        )

    raise ValueError(f"Unsupported portfolio_mode '{portfolio_mode}'.")


def _validate_weights_and_contributions(daily: pd.DataFrame, positions: pd.DataFrame) -> None:
    executable = daily[daily["is_executable"]].copy()
    if executable.empty:
        raise ValueError("No executable portfolio dates. Nothing to evaluate.")

    # Contribution sum must match reported daily return for executable rows.
    pos_returns = (
        positions.groupby(["date", "portfolio_mode"], as_index=False)["contribution"]
        .sum()
        .rename(columns={"contribution": "contribution_sum"})
    )
    chk = executable.merge(pos_returns, on=["date", "portfolio_mode"], how="left")
    if chk["contribution_sum"].isna().any():
        raise ValueError("Missing position contributions for executable daily rows.")
    if not np.allclose(
        chk["contribution_sum"].to_numpy(dtype=float),
        chk["gross_portfolio_return"].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("Position contributions do not aggregate to gross_portfolio_return.")

    # Weight checks by mode/date.
    weights = (
        positions.groupby(["date", "portfolio_mode", "side"], as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "weight_sum"})
    )
    for row in executable.itertuples(index=False):
        mode = str(row.portfolio_mode)
        date = pd.Timestamp(row.date).normalize()
        day = weights[(weights["date"] == date) & (weights["portfolio_mode"] == mode)]
        long_sum = float(day.loc[day["side"] == "long", "weight_sum"].sum())
        short_sum = float(day.loc[day["side"] == "short", "weight_sum"].sum())

        if mode == MODE_LONG_ONLY_TOP:
            if not np.isclose(long_sum, 1.0, atol=1e-12):
                raise ValueError(f"Invalid long weight sum for {mode} on {date.date()}: {long_sum}")
            if not np.isclose(short_sum, 0.0, atol=1e-12):
                raise ValueError(f"Unexpected short weights for {mode} on {date.date()}: {short_sum}")
        elif mode == MODE_LONG_SHORT_TOP_BOTTOM:
            if not np.isclose(long_sum, 1.0, atol=1e-12):
                raise ValueError(f"Invalid long weight sum for {mode} on {date.date()}: {long_sum}")
            if not np.isclose(short_sum, -1.0, atol=1e-12):
                raise ValueError(f"Invalid short weight sum for {mode} on {date.date()}: {short_sum}")
        elif mode == MODE_SHORT_ONLY_BOTTOM:
            if not np.isclose(long_sum, 0.0, atol=1e-12):
                raise ValueError(f"Unexpected long weights for {mode} on {date.date()}: {long_sum}")
            if not np.isclose(short_sum, -1.0, atol=1e-12):
                raise ValueError(f"Invalid short weight sum for {mode} on {date.date()}: {short_sum}")


def run_paper_portfolio(
    *,
    signals_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_name: str | None = None,
    label_name: str | None = None,
    horizon_days: int | None = None,
    split_roles: Iterable[str] = DEFAULT_SPLIT_ROLES,
    portfolio_modes: Iterable[str] = DEFAULT_PORTFOLIO_MODES,
    run_id: str = MODULE_VERSION,
) -> PaperPortfolioResult:
    logger = get_logger("signals.paper_portfolio")

    signals_source = (
        Path(signals_path).expanduser().resolve()
        if signals_path
        else (data_dir() / "signals" / "signals_daily.parquet")
    )
    labels_source = (
        Path(labels_path).expanduser().resolve()
        if labels_path
        else (data_dir() / "labels" / "labels_forward.parquet")
    )

    selected_split_roles = _normalize_split_roles(split_roles)
    selected_modes = _normalize_portfolio_modes(portfolio_modes)

    signals = _prepare_signals(signals_source)
    labels = _prepare_labels(labels_source)

    signals = signals[signals["split_role"].isin(set(selected_split_roles))].copy()
    if signals.empty:
        raise ValueError(f"No signal rows left after split_role filter: {list(selected_split_roles)}")

    selected_model_name = _select_unique_value(signals, "model_name", provided=model_name)
    signals = signals[signals["model_name"] == selected_model_name].copy()

    selected_label_name = _select_unique_value(signals, "label_name", provided=label_name)
    signals = signals[signals["label_name"] == selected_label_name].copy()
    labels = labels[labels["label_name"] == selected_label_name].copy()
    if labels.empty:
        raise ValueError(f"No labels found for label_name='{selected_label_name}'.")

    signals, labels, join_keys, selected_horizon = _resolve_join_keys(
        signals=signals,
        labels=labels,
        horizon_days=horizon_days,
    )

    merged = signals.merge(
        labels[join_keys + ["label_value"]],
        on=join_keys,
        how="inner",
    )
    if merged.empty:
        raise ValueError("Join between signals_daily and labels_forward produced no rows.")

    merged["realized_return"] = pd.to_numeric(merged["label_value"], errors="coerce")
    if merged["realized_return"].isna().any():
        raise ValueError("Joined rows contain null/non-numeric realized_return.")

    dates_sorted = sorted(pd.to_datetime(merged["date"]).dt.normalize().unique().tolist())
    if not dates_sorted:
        raise ValueError("No dates available after joining signals and labels.")

    daily_rows: list[dict[str, Any]] = []
    positions_blocks: list[pd.DataFrame] = []

    for dt in dates_sorted:
        date_frame = merged[merged["date"] == pd.Timestamp(dt)].copy()
        if date_frame.empty:
            continue
        for mode in selected_modes:
            positions, daily_row = _build_one_mode_for_date(
                date_frame=date_frame,
                portfolio_mode=mode,
                model_name=selected_model_name,
                label_name=selected_label_name,
            )
            daily_rows.append(daily_row)
            if not positions.empty:
                positions_blocks.append(positions)

    daily = pd.DataFrame(daily_rows).sort_values(["date", "portfolio_mode"]).reset_index(drop=True)
    if daily.empty:
        raise ValueError("paper_portfolio_daily is empty.")
    daily["n_long"] = pd.to_numeric(daily["n_long"], errors="coerce").fillna(0).astype("int64")
    daily["n_short"] = pd.to_numeric(daily["n_short"], errors="coerce").fillna(0).astype("int64")
    daily["n_names_ranked"] = pd.to_numeric(daily["n_names_ranked"], errors="coerce").fillna(0).astype("int64")
    daily["gross_long_return"] = pd.to_numeric(daily["gross_long_return"], errors="coerce").astype("float64")
    daily["gross_short_return"] = pd.to_numeric(daily["gross_short_return"], errors="coerce").astype("float64")
    daily["gross_portfolio_return"] = pd.to_numeric(daily["gross_portfolio_return"], errors="coerce").astype("float64")
    daily["is_executable"] = daily["is_executable"].astype(bool)

    positions = (
        pd.concat(positions_blocks, ignore_index=True)
        if positions_blocks
        else pd.DataFrame(
            columns=["date", "instrument_id", "ticker", "split_role", "realized_return", "portfolio_mode", "side", "weight", "contribution"]
        )
    )
    if positions.empty:
        raise ValueError("No executable positions were generated for selected modes and dates.")

    positions["model_name"] = selected_model_name
    positions["label_name"] = selected_label_name
    if selected_horizon is not None:
        positions["horizon_days"] = int(selected_horizon)
        daily["horizon_days"] = int(selected_horizon)

    positions = positions[
        [
            "date",
            "instrument_id",
            "ticker",
            "portfolio_mode",
            "side",
            "weight",
            "realized_return",
            "contribution",
            "model_name",
            "label_name",
            "split_role",
            *(("horizon_days",) if selected_horizon is not None else ()),
        ]
    ].sort_values(["date", "portfolio_mode", "instrument_id", "side"]).reset_index(drop=True)

    if positions.duplicated(["date", "instrument_id", "portfolio_mode", "model_name", "label_name"], keep=False).any():
        raise ValueError(
            "paper_portfolio_positions has duplicate logical PK rows."
        )
    assert_schema(daily, PAPER_DAILY_SCHEMA)
    assert_schema(positions, PAPER_POSITIONS_SCHEMA)

    _validate_weights_and_contributions(daily, positions)

    config_hash = _config_hash(
        {
            "version": MODULE_VERSION,
            "signals_path": str(signals_source),
            "labels_path": str(labels_source),
            "model_name": selected_model_name,
            "label_name": selected_label_name,
            "horizon_days": selected_horizon,
            "split_roles": list(selected_split_roles),
            "portfolio_modes": list(selected_modes),
        }
    )
    built_ts_utc = datetime.now(UTC).isoformat()

    daily["run_id"] = run_id
    daily["config_hash"] = config_hash
    daily["built_ts_utc"] = built_ts_utc

    positions["run_id"] = run_id
    positions["config_hash"] = config_hash
    positions["built_ts_utc"] = built_ts_utc

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (data_dir() / "signals")
    target_dir.mkdir(parents=True, exist_ok=True)

    daily_path = write_parquet(
        daily,
        target_dir / "paper_portfolio_daily.parquet",
        schema_name=PAPER_DAILY_SCHEMA.name,
        run_id=run_id,
    )
    positions_path = write_parquet(
        positions,
        target_dir / "paper_portfolio_positions.parquet",
        schema_name=PAPER_POSITIONS_SCHEMA.name,
        run_id=run_id,
    )

    mode_summaries: list[dict[str, Any]] = []
    for mode in selected_modes:
        mode_daily = daily[daily["portfolio_mode"] == mode].copy()
        mode_exec = mode_daily[mode_daily["is_executable"]].copy()
        mode_summaries.append(
            {
                "model_name": selected_model_name,
                "label_name": selected_label_name,
                "horizon_days": selected_horizon,
                "portfolio_mode": mode,
                "n_dates": int(len(mode_daily)),
                "n_executable_dates": int(len(mode_exec)),
                "n_skipped_dates": int(len(mode_daily) - len(mode_exec)),
                "avg_n_long": _to_float_or_none(pd.to_numeric(mode_daily["n_long"], errors="coerce").mean(skipna=True)),
                "avg_n_short": _to_float_or_none(pd.to_numeric(mode_daily["n_short"], errors="coerce").mean(skipna=True)),
                "mean_daily_gross_return": _to_float_or_none(pd.to_numeric(mode_exec["gross_portfolio_return"], errors="coerce").mean(skipna=True)),
                "median_daily_gross_return": _to_float_or_none(pd.to_numeric(mode_exec["gross_portfolio_return"], errors="coerce").median(skipna=True)),
                "std_daily_gross_return": _to_float_or_none(pd.to_numeric(mode_exec["gross_portfolio_return"], errors="coerce").std(skipna=True, ddof=0)),
                "positive_return_rate": (
                    float((pd.to_numeric(mode_exec["gross_portfolio_return"], errors="coerce") > 0).mean())
                    if not mode_exec.empty
                    else None
                ),
                "mean_long_leg_return": _to_float_or_none(pd.to_numeric(mode_exec["gross_long_return"], errors="coerce").mean(skipna=True)),
                "mean_short_leg_return": _to_float_or_none(pd.to_numeric(mode_exec["gross_short_return"], errors="coerce").mean(skipna=True)),
            }
        )

    summary_payload = {
        "created_at_utc": built_ts_utc,
        "run_id": run_id,
        "config_hash": config_hash,
        "module_version": MODULE_VERSION,
        "signals_path": str(signals_source),
        "labels_path": str(labels_source),
        "model_name": selected_model_name,
        "label_name": selected_label_name,
        "horizon_days": selected_horizon,
        "split_roles_included": list(selected_split_roles),
        "portfolio_modes": list(selected_modes),
        "join_stats": {
            "n_rows_signals_filtered": int(len(signals)),
            "n_rows_labels_filtered": int(len(labels)),
            "n_rows_joined": int(len(merged)),
            "n_rows_positions": int(len(positions)),
        },
        "mode_summaries": mode_summaries,
        "output_paths": {
            "paper_portfolio_daily": str(daily_path),
            "paper_portfolio_positions": str(positions_path),
        },
    }
    summary_path = target_dir / "paper_portfolio_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "paper_portfolio_built",
        run_id=run_id,
        model_name=selected_model_name,
        label_name=selected_label_name,
        modes=list(selected_modes),
        n_daily=int(len(daily)),
        n_positions=int(len(positions)),
        output_path=str(daily_path),
    )

    return PaperPortfolioResult(
        daily_path=daily_path,
        positions_path=positions_path,
        summary_path=summary_path,
        row_count_daily=int(len(daily)),
        row_count_positions=int(len(positions)),
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
    parser = argparse.ArgumentParser(description="Build simple paper portfolio from ranked signals.")
    parser.add_argument("--signals-path", type=str, default=None)
    parser.add_argument("--labels-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--label-name", type=str, default=None)
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument("--split-roles", type=_parse_csv_strings, default=DEFAULT_SPLIT_ROLES)
    parser.add_argument("--portfolio-modes", type=_parse_csv_strings, default=DEFAULT_PORTFOLIO_MODES)
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_paper_portfolio(
        signals_path=args.signals_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        label_name=args.label_name,
        horizon_days=args.horizon_days,
        split_roles=args.split_roles,
        portfolio_modes=args.portfolio_modes,
        run_id=args.run_id,
    )
    print("Paper portfolio built:")
    print(f"- daily: {result.daily_path}")
    print(f"- positions: {result.positions_path}")
    print(f"- summary: {result.summary_path}")
    print(f"- row_count_daily: {result.row_count_daily}")
    print(f"- row_count_positions: {result.row_count_positions}")
    print(f"- model_name: {result.model_name}")
    print(f"- label_name: {result.label_name}")


if __name__ == "__main__":
    main()
