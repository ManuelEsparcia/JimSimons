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

# Allow direct script execution: `python simons_smallcap_swing/execution/cost_model.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "cost_model_mvp_v2_execution_timing"
DEFAULT_COST_BPS_PER_TURNOVER = 10.0
DEFAULT_ENTRY_BPS = 2.0
DEFAULT_EXIT_BPS = 2.0
EPS = 1e-12

REBALANCE_LEGACY_INPUT_SCHEMA = DataSchema(
    name="cost_model_rebalance_legacy_input_mvp",
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

REBALANCE_EXECUTION_INPUT_SCHEMA = DataSchema(
    name="cost_model_rebalance_execution_input_mvp",
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
        ColumnSpec("portfolio_mode", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
    ),
    primary_key=("signal_date", "instrument_id", "portfolio_mode", "model_name", "label_name"),
    allow_extra_columns=True,
)

COSTS_POSITIONS_SCHEMA = DataSchema(
    name="costs_positions_mvp",
    version="2.0.0",
    columns=(
        ColumnSpec("signal_date", "datetime64", nullable=False),
        ColumnSpec("execution_date", "datetime64", nullable=True),
        ColumnSpec("cost_date", "datetime64", nullable=True),
        ColumnSpec("date", "datetime64", nullable=True),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("portfolio_mode", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("prev_weight", "float64", nullable=False),
        ColumnSpec("target_weight", "float64", nullable=False),
        ColumnSpec("execution_weight", "float64", nullable=False),
        ColumnSpec("weight_change_signal", "float64", nullable=False),
        ColumnSpec("weight_change_execution", "float64", nullable=False),
        ColumnSpec("weight_change", "float64", nullable=False),
        ColumnSpec("abs_weight_change", "float64", nullable=False),
        ColumnSpec("turnover_contribution", "float64", nullable=False),
        ColumnSpec("entered_flag", "bool", nullable=False),
        ColumnSpec("exited_flag", "bool", nullable=False),
        ColumnSpec("is_executable", "bool", nullable=False),
        ColumnSpec("skip_reason", "string", nullable=True),
        ColumnSpec("turnover_cost", "float64", nullable=False),
        ColumnSpec("entry_cost", "float64", nullable=False),
        ColumnSpec("exit_cost", "float64", nullable=False),
        ColumnSpec("total_cost", "float64", nullable=False),
    ),
    primary_key=("signal_date", "instrument_id", "portfolio_mode", "model_name", "label_name"),
    allow_extra_columns=True,
)

COSTS_DAILY_SCHEMA = DataSchema(
    name="costs_daily_mvp",
    version="2.0.0",
    columns=(
        ColumnSpec("cost_date", "datetime64", nullable=False),
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("portfolio_mode", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("n_positions", "int64", nullable=False),
        ColumnSpec("gross_turnover", "float64", nullable=False),
        ColumnSpec("total_turnover_cost", "float64", nullable=False),
        ColumnSpec("total_entry_cost", "float64", nullable=False),
        ColumnSpec("total_exit_cost", "float64", nullable=False),
        ColumnSpec("total_cost", "float64", nullable=False),
    ),
    primary_key=("cost_date", "portfolio_mode", "model_name", "label_name"),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class CostModelResult:
    costs_positions_path: Path
    costs_daily_path: Path
    costs_summary_path: Path
    row_count_positions: int
    row_count_daily: int
    model_name: str
    label_name: str
    config_hash: str


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains invalid dates.")
    return parsed.dt.normalize()


def _normalize_portfolio_modes(modes: Iterable[str] | None) -> tuple[str, ...]:
    if modes is None:
        return tuple()
    return tuple(sorted({str(item).strip() for item in modes if str(item).strip()}))


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


def _config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


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


def _prepare_execution_rebalance(frame: pd.DataFrame) -> pd.DataFrame:
    assert_schema(frame, REBALANCE_EXECUTION_INPUT_SCHEMA)
    out = frame.copy()
    out["signal_date"] = _normalize_date(out["signal_date"], column="signal_date")
    out["execution_date"] = pd.to_datetime(out["execution_date"], errors="coerce").dt.normalize()
    out["cost_date"] = pd.to_datetime(out["cost_date"], errors="coerce").dt.normalize()
    out["instrument_id"] = out["instrument_id"].astype(str)
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["portfolio_mode"] = out["portfolio_mode"].astype(str)
    out["model_name"] = out["model_name"].astype(str)
    out["label_name"] = out["label_name"].astype(str)
    for col in (
        "prev_weight",
        "target_weight",
        "execution_weight",
        "weight_change_signal",
        "weight_change_execution",
    ):
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any():
            raise ValueError(f"execution_rebalance contains non-numeric '{col}'.")
    out["entered_flag"] = out["entered_flag"].astype(bool)
    out["exited_flag"] = out["exited_flag"].astype(bool)
    out["is_executable"] = out["is_executable"].astype(bool)
    out["skip_reason"] = out.get("skip_reason", pd.Series([None] * len(out))).astype("string")

    if out.duplicated(["signal_date", "instrument_id", "portfolio_mode", "model_name", "label_name"], keep=False).any():
        raise ValueError("execution_rebalance has duplicate logical PK rows.")
    if (out.loc[out["execution_date"].notna(), "signal_date"] > out.loc[out["execution_date"].notna(), "execution_date"]).any():
        raise ValueError("execution_rebalance has signal_date > execution_date.")
    if not np.allclose(
        (out["execution_weight"] - out["prev_weight"]).to_numpy(dtype=float),
        out["weight_change_execution"].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("execution_rebalance weight_change_execution must equal execution_weight - prev_weight.")
    if not np.allclose(
        (out["target_weight"] - out["prev_weight"]).to_numpy(dtype=float),
        out["weight_change_signal"].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("execution_rebalance weight_change_signal must equal target_weight - prev_weight.")
    return out


def _prepare_legacy_rebalance(frame: pd.DataFrame) -> pd.DataFrame:
    assert_schema(frame, REBALANCE_LEGACY_INPUT_SCHEMA)
    out = frame.copy()
    out["date"] = _normalize_date(out["date"], column="date")
    out["signal_date"] = out["date"]
    out["execution_date"] = out["date"]
    out["cost_date"] = out["date"]
    out["instrument_id"] = out["instrument_id"].astype(str)
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["portfolio_mode"] = out["portfolio_mode"].astype(str)
    out["model_name"] = out["model_name"].astype(str)
    out["label_name"] = out["label_name"].astype(str)
    for col in ("prev_weight", "target_weight", "weight_change", "abs_weight_change"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any():
            raise ValueError(f"portfolio_rebalance contains non-numeric '{col}'.")
    out["execution_weight"] = out["target_weight"].astype(float)
    out["weight_change_signal"] = out["weight_change"].astype(float)
    out["weight_change_execution"] = out["weight_change"].astype(float)
    out["is_executable"] = True
    out["skip_reason"] = None
    out["entered_flag"] = out["entered_flag"].astype(bool)
    out["exited_flag"] = out["exited_flag"].astype(bool)

    if out.duplicated(["signal_date", "instrument_id", "portfolio_mode", "model_name", "label_name"], keep=False).any():
        raise ValueError("portfolio_rebalance has duplicate logical PK rows.")
    return out


def _prepare_rebalance(source: Path) -> tuple[pd.DataFrame, str]:
    frame = read_parquet(source)
    cols = set(frame.columns.tolist())
    if {"signal_date", "execution_date", "cost_date", "execution_weight", "weight_change_execution", "is_executable"}.issubset(cols):
        return _prepare_execution_rebalance(frame), "execution_rebalance"
    return _prepare_legacy_rebalance(frame), "legacy_portfolio_rebalance"


def run_cost_model(
    *,
    holdings_path: str | Path | None = None,
    rebalance_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_name: str | None = None,
    label_name: str | None = None,
    portfolio_modes: Iterable[str] | None = None,
    cost_bps_per_turnover: float = DEFAULT_COST_BPS_PER_TURNOVER,
    entry_bps: float = DEFAULT_ENTRY_BPS,
    exit_bps: float = DEFAULT_EXIT_BPS,
    run_id: str = MODULE_VERSION,
) -> CostModelResult:
    logger = get_logger("execution.cost_model")

    if float(cost_bps_per_turnover) < 0.0 or float(entry_bps) < 0.0 or float(exit_bps) < 0.0:
        raise ValueError("Cost bps parameters must be >= 0.")

    holdings_source = (
        Path(holdings_path).expanduser().resolve()
        if holdings_path
        else (data_dir() / "portfolio" / "portfolio_holdings.parquet")
    )
    if rebalance_path:
        rebalance_source = Path(rebalance_path).expanduser().resolve()
    else:
        default_execution = data_dir() / "execution" / "execution_rebalance.parquet"
        default_legacy = data_dir() / "portfolio" / "portfolio_rebalance.parquet"
        rebalance_source = default_execution if default_execution.exists() else default_legacy

    rebalance, input_mode = _prepare_rebalance(rebalance_source)

    selected_model_name = _select_unique_value(rebalance, "model_name", provided=model_name)
    selected_label_name = _select_unique_value(rebalance, "label_name", provided=label_name)
    rebalance = rebalance[
        (rebalance["model_name"] == selected_model_name)
        & (rebalance["label_name"] == selected_label_name)
    ].copy()
    if rebalance.empty:
        raise ValueError("rebalance input has no rows for selected model_name/label_name.")

    selected_modes = _normalize_portfolio_modes(portfolio_modes)
    if selected_modes:
        rebalance = rebalance[rebalance["portfolio_mode"].isin(set(selected_modes))].copy()
        if rebalance.empty:
            raise ValueError(f"No rebalance rows left after portfolio_mode filter: {list(selected_modes)}")

    positions = rebalance.copy()

    # Execution-effective transitions and costs
    positions["weight_change_execution"] = pd.to_numeric(positions["weight_change_execution"], errors="coerce")
    if positions["weight_change_execution"].isna().any():
        raise ValueError("weight_change_execution contains NaN values.")
    positions["abs_weight_change"] = np.where(
        positions["is_executable"],
        np.abs(positions["weight_change_execution"].astype(float)),
        0.0,
    )
    positions["turnover_contribution"] = 0.5 * positions["abs_weight_change"].astype(float)

    entered_exec = (
        positions["is_executable"]
        & (np.abs(positions["prev_weight"].astype(float)) <= EPS)
        & (np.abs(positions["execution_weight"].astype(float)) > EPS)
    )
    exited_exec = (
        positions["is_executable"]
        & (np.abs(positions["prev_weight"].astype(float)) > EPS)
        & (np.abs(positions["execution_weight"].astype(float)) <= EPS)
    )
    positions["entered_flag"] = entered_exec.astype(bool)
    positions["exited_flag"] = exited_exec.astype(bool)

    positions["turnover_cost"] = float(cost_bps_per_turnover) * 1e-4 * positions["abs_weight_change"].astype(float)
    positions["entry_cost"] = np.where(
        positions["entered_flag"],
        float(entry_bps) * 1e-4 * np.abs(positions["execution_weight"].astype(float)),
        0.0,
    )
    positions["exit_cost"] = np.where(
        positions["exited_flag"],
        float(exit_bps) * 1e-4 * np.abs(positions["prev_weight"].astype(float)),
        0.0,
    )
    positions["total_cost"] = (
        positions["turnover_cost"] + positions["entry_cost"] + positions["exit_cost"]
    ).astype(float)
    if (positions["total_cost"] < -EPS).any():
        raise ValueError("total_cost must be non-negative.")

    # Keep backward-compatible date alias while making cost_date explicit.
    positions["date"] = positions["cost_date"]
    positions["weight_change"] = positions["weight_change_execution"].astype(float)

    positions = positions[
        [
            "signal_date",
            "execution_date",
            "cost_date",
            "date",
            "instrument_id",
            "ticker",
            "portfolio_mode",
            "model_name",
            "label_name",
            "prev_weight",
            "target_weight",
            "execution_weight",
            "weight_change_signal",
            "weight_change_execution",
            "weight_change",
            "abs_weight_change",
            "turnover_contribution",
            "entered_flag",
            "exited_flag",
            "is_executable",
            "skip_reason",
            "turnover_cost",
            "entry_cost",
            "exit_cost",
            "total_cost",
        ]
    ].sort_values(["signal_date", "portfolio_mode", "instrument_id"]).reset_index(drop=True)
    assert_schema(positions, COSTS_POSITIONS_SCHEMA)

    # Aggregate costs by explicit cost_date only.
    daily_base = positions[(positions["is_executable"]) & (positions["cost_date"].notna())].copy()
    if daily_base.empty:
        raise ValueError("No executable rows with valid cost_date to aggregate costs.")

    daily = (
        daily_base.groupby(["cost_date", "portfolio_mode", "model_name", "label_name"], as_index=False)
        .agg(
            n_positions=("instrument_id", "count"),
            gross_turnover=("abs_weight_change", "sum"),
            total_turnover_cost=("turnover_cost", "sum"),
            total_entry_cost=("entry_cost", "sum"),
            total_exit_cost=("exit_cost", "sum"),
            total_cost=("total_cost", "sum"),
        )
        .sort_values(["cost_date", "portfolio_mode"])
        .reset_index(drop=True)
    )
    daily["date"] = daily["cost_date"]
    daily["n_positions"] = pd.to_numeric(daily["n_positions"], errors="coerce").astype("int64")
    assert_schema(daily, COSTS_DAILY_SCHEMA)

    # Reconciliation: daily total must equal sum of position-level total_cost for executable rows.
    chk = (
        daily_base.groupby(["cost_date", "portfolio_mode"], as_index=False)["total_cost"]
        .sum()
        .rename(columns={"total_cost": "positions_total"})
        .merge(daily[["cost_date", "portfolio_mode", "total_cost"]], on=["cost_date", "portfolio_mode"], how="left")
    )
    if not np.allclose(
        chk["positions_total"].to_numpy(dtype=float),
        chk["total_cost"].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("Daily costs do not reconcile with position-level costs.")

    # Optional holdings existence check only for observability; not required for timing refit.
    holds_exists = holdings_source.exists()

    config_hash = _config_hash(
        {
            "version": MODULE_VERSION,
            "holdings_path": str(holdings_source),
            "rebalance_path": str(rebalance_source),
            "rebalance_input_mode": input_mode,
            "model_name": selected_model_name,
            "label_name": selected_label_name,
            "portfolio_modes": sorted(rebalance["portfolio_mode"].astype(str).unique().tolist()),
            "cost_bps_per_turnover": float(cost_bps_per_turnover),
            "entry_bps": float(entry_bps),
            "exit_bps": float(exit_bps),
            "cost_unit": "weight_fraction_of_nav",
            "cost_aggregation_date": "cost_date",
        }
    )
    built_ts_utc = datetime.now(UTC).isoformat()

    positions["run_id"] = run_id
    positions["config_hash"] = config_hash
    positions["built_ts_utc"] = built_ts_utc

    daily["run_id"] = run_id
    daily["config_hash"] = config_hash
    daily["built_ts_utc"] = built_ts_utc

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (data_dir() / "execution")
    target_dir.mkdir(parents=True, exist_ok=True)

    costs_positions_path = write_parquet(
        positions,
        target_dir / "costs_positions.parquet",
        schema_name=COSTS_POSITIONS_SCHEMA.name,
        run_id=run_id,
    )
    costs_daily_path = write_parquet(
        daily,
        target_dir / "costs_daily.parquet",
        schema_name=COSTS_DAILY_SCHEMA.name,
        run_id=run_id,
    )

    summary = {
        "built_ts_utc": built_ts_utc,
        "run_id": run_id,
        "config_hash": config_hash,
        "module_version": MODULE_VERSION,
        "model_name": selected_model_name,
        "label_name": selected_label_name,
        "portfolio_modes": sorted(daily["portfolio_mode"].astype(str).unique().tolist()),
        "cost_bps_per_turnover": float(cost_bps_per_turnover),
        "entry_bps": float(entry_bps),
        "exit_bps": float(exit_bps),
        "n_dates": int(daily["cost_date"].nunique()),
        "mean_daily_cost": _to_float_or_none(daily["total_cost"].mean()),
        "median_daily_cost": _to_float_or_none(daily["total_cost"].median()),
        "max_daily_cost": _to_float_or_none(daily["total_cost"].max()),
        "mean_turnover": _to_float_or_none(daily["gross_turnover"].mean()),
        "n_positions_rows": int(len(positions)),
        "n_daily_rows": int(len(daily)),
        "n_executable_rows": int(positions["is_executable"].sum()),
        "n_non_executable_rows": int((~positions["is_executable"]).sum()),
        "rebalance_input_mode": input_mode,
        "cost_aggregation_date": "cost_date",
        "cost_unit": "weight_fraction_of_nav",
        "holds_source_exists": bool(holds_exists),
        "output_paths": {
            "costs_positions": str(costs_positions_path),
            "costs_daily": str(costs_daily_path),
        },
    }
    costs_summary_path = target_dir / "costs_summary.json"
    costs_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "cost_model_built",
        run_id=run_id,
        model_name=selected_model_name,
        label_name=selected_label_name,
        row_count_positions=int(len(positions)),
        row_count_daily=int(len(daily)),
        rebalance_input_mode=input_mode,
        output_dir=str(target_dir),
    )

    return CostModelResult(
        costs_positions_path=costs_positions_path,
        costs_daily_path=costs_daily_path,
        costs_summary_path=costs_summary_path,
        row_count_positions=int(len(positions)),
        row_count_daily=int(len(daily)),
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
        description="Compute simple turnover/entry/exit costs using explicit execution timing (signal/execution/cost date)."
    )
    parser.add_argument("--holdings-path", type=str, default=None)
    parser.add_argument("--rebalance-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--label-name", type=str, default=None)
    parser.add_argument("--portfolio-modes", type=_parse_csv_strings, default=tuple())
    parser.add_argument("--cost-bps-per-turnover", type=float, default=DEFAULT_COST_BPS_PER_TURNOVER)
    parser.add_argument("--entry-bps", type=float, default=DEFAULT_ENTRY_BPS)
    parser.add_argument("--exit-bps", type=float, default=DEFAULT_EXIT_BPS)
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_cost_model(
        holdings_path=args.holdings_path,
        rebalance_path=args.rebalance_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        label_name=args.label_name,
        portfolio_modes=args.portfolio_modes,
        cost_bps_per_turnover=args.cost_bps_per_turnover,
        entry_bps=args.entry_bps,
        exit_bps=args.exit_bps,
        run_id=args.run_id,
    )
    print("Cost model built:")
    print(f"- costs_positions: {result.costs_positions_path}")
    print(f"- costs_daily: {result.costs_daily_path}")
    print(f"- costs_summary: {result.costs_summary_path}")
    print(f"- row_count_positions: {result.row_count_positions}")
    print(f"- row_count_daily: {result.row_count_daily}")
    print(f"- model_name: {result.model_name}")
    print(f"- label_name: {result.label_name}")


if __name__ == "__main__":
    main()
