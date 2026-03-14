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


MODULE_VERSION = "backtest_diagnostics_mvp_v1"
EPS = 1e-12

BACKTEST_DAILY_INPUT_SCHEMA = DataSchema(
    name="backtest_diagnostics_daily_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("portfolio_mode", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("gross_return", "float64", nullable=False),
        ColumnSpec("net_return", "float64", nullable=False),
        ColumnSpec("total_cost", "float64", nullable=False),
        ColumnSpec("gross_equity", "float64", nullable=False),
        ColumnSpec("net_equity", "float64", nullable=False),
        ColumnSpec("drawdown_net", "float64", nullable=False),
    ),
    primary_key=("date", "portfolio_mode", "model_name", "label_name"),
    allow_extra_columns=True,
)

COSTS_DAILY_INPUT_V2_SCHEMA = DataSchema(
    name="backtest_diagnostics_costs_daily_input_v2_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("cost_date", "datetime64", nullable=False),
        ColumnSpec("portfolio_mode", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("total_cost", "float64", nullable=False),
    ),
    primary_key=("cost_date", "portfolio_mode", "model_name", "label_name"),
    allow_extra_columns=True,
)

COSTS_DAILY_INPUT_V1_SCHEMA = DataSchema(
    name="backtest_diagnostics_costs_daily_input_v1_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("portfolio_mode", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("total_cost", "float64", nullable=False),
    ),
    primary_key=("date", "portfolio_mode", "model_name", "label_name"),
    allow_extra_columns=True,
)

DIAGNOSTICS_DAILY_SCHEMA = DataSchema(
    name="backtest_diagnostics_daily_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("portfolio_mode", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("gross_return", "float64", nullable=False),
        ColumnSpec("net_return", "float64", nullable=False),
        ColumnSpec("total_cost", "float64", nullable=False),
        ColumnSpec("gross_equity", "float64", nullable=False),
        ColumnSpec("net_equity", "float64", nullable=False),
        ColumnSpec("drawdown_net", "float64", nullable=False),
        ColumnSpec("daily_cost_drag", "float64", nullable=False),
    ),
    primary_key=("date", "portfolio_mode", "model_name", "label_name"),
    allow_extra_columns=True,
)

DIAGNOSTICS_BY_MODE_SCHEMA = DataSchema(
    name="backtest_diagnostics_by_mode_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("portfolio_mode", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("n_dates", "int64", nullable=False),
        ColumnSpec("cumulative_gross_return", "float64", nullable=False),
        ColumnSpec("cumulative_net_return", "float64", nullable=False),
        ColumnSpec("mean_daily_gross_return", "float64", nullable=False),
        ColumnSpec("mean_daily_net_return", "float64", nullable=False),
        ColumnSpec("std_daily_net_return", "float64", nullable=False),
        ColumnSpec("positive_net_return_rate", "float64", nullable=False),
        ColumnSpec("max_drawdown_net", "float64", nullable=False),
        ColumnSpec("total_cost_paid", "float64", nullable=False),
        ColumnSpec("avg_daily_cost", "float64", nullable=False),
        ColumnSpec("avg_turnover_if_available", "float64", nullable=True),
    ),
    primary_key=("portfolio_mode", "model_name", "label_name"),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class BacktestDiagnosticsResult:
    diagnostics_daily_path: Path
    diagnostics_by_mode_path: Path
    diagnostics_summary_path: Path
    row_count_daily: int
    row_count_by_mode: int
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
        raise ValueError(
            f"Expected exactly one {column}. Observed: {vals}. "
            f"Pass --{column.replace('_', '-')} explicitly."
        )
    return str(vals[0])


def _prepare_backtest_daily(path: Path) -> pd.DataFrame:
    frame = read_parquet(path)
    assert_schema(frame, BACKTEST_DAILY_INPUT_SCHEMA)
    out = frame.copy()
    out["date"] = _norm_date(out["date"], column="date")
    out["portfolio_mode"] = out["portfolio_mode"].astype(str)
    out["model_name"] = out["model_name"].astype(str)
    out["label_name"] = out["label_name"].astype(str)

    for col in (
        "gross_return",
        "net_return",
        "total_cost",
        "gross_equity",
        "net_equity",
        "drawdown_net",
    ):
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any():
            raise ValueError(f"backtest_daily has non-numeric values in '{col}'.")

    if out["portfolio_mode"].str.strip().eq("").any():
        raise ValueError("backtest_daily contains empty portfolio_mode values.")
    if out.duplicated(["date", "portfolio_mode", "model_name", "label_name"], keep=False).any():
        raise ValueError("backtest_daily has duplicate logical PK rows.")
    if (out["drawdown_net"] > EPS).any():
        raise ValueError("backtest_daily drawdown_net must be <= 0.")
    if not np.allclose(
        (out["gross_return"] - out["total_cost"]).to_numpy(dtype=float),
        out["net_return"].to_numpy(dtype=float),
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError("backtest_daily must satisfy net_return = gross_return - total_cost.")

    return out


def _prepare_costs_daily(path: Path) -> pd.DataFrame:
    frame = read_parquet(path)
    cols = set(frame.columns.tolist())
    if "cost_date" in cols:
        assert_schema(frame, COSTS_DAILY_INPUT_V2_SCHEMA)
        out = frame.copy()
        out["cost_date"] = _norm_date(out["cost_date"], column="cost_date")
    else:
        assert_schema(frame, COSTS_DAILY_INPUT_V1_SCHEMA)
        out = frame.copy()
        out["cost_date"] = _norm_date(out["date"], column="date")

    out["date"] = out["cost_date"]
    out["portfolio_mode"] = out["portfolio_mode"].astype(str)
    out["model_name"] = out["model_name"].astype(str)
    out["label_name"] = out["label_name"].astype(str)
    out["total_cost"] = pd.to_numeric(out["total_cost"], errors="coerce")
    if out["total_cost"].isna().any():
        raise ValueError("costs_daily contains non-numeric total_cost.")
    if (out["total_cost"] < -EPS).any():
        raise ValueError("costs_daily total_cost must be >= 0.")

    optional_numeric = ("gross_turnover", "total_turnover_cost", "total_entry_cost", "total_exit_cost")
    for col in optional_numeric:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if out.duplicated(["cost_date", "portfolio_mode", "model_name", "label_name"], keep=False).any():
        raise ValueError("costs_daily has duplicate (cost_date, portfolio_mode, model_name, label_name) rows.")

    keep_cols = [
        "cost_date",
        "date",
        "portfolio_mode",
        "model_name",
        "label_name",
        "total_cost",
    ]
    for col in optional_numeric:
        if col in out.columns:
            keep_cols.append(col)
    return out[keep_cols].copy()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mode_sharpe_like(mean_ret: float, std_ret: float) -> float | None:
    if std_ret is None:
        return None
    if abs(float(std_ret)) <= EPS:
        return None
    return float(mean_ret) / float(std_ret)


def run_backtest_diagnostics(
    *,
    backtest_daily_path: str | Path | None = None,
    backtest_summary_path: str | Path | None = None,
    costs_daily_path: str | Path | None = None,
    backtest_contributions_path: str | Path | None = None,
    portfolio_summary_path: str | Path | None = None,
    execution_assumptions_summary_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_name: str | None = None,
    label_name: str | None = None,
    portfolio_modes: Iterable[str] | None = None,
    run_id: str = MODULE_VERSION,
) -> BacktestDiagnosticsResult:
    logger = get_logger("backtest.diagnostics")

    backtest_daily_source = (
        Path(backtest_daily_path).expanduser().resolve()
        if backtest_daily_path
        else (data_dir() / "backtest" / "backtest_daily.parquet")
    )
    backtest_summary_source = (
        Path(backtest_summary_path).expanduser().resolve()
        if backtest_summary_path
        else (data_dir() / "backtest" / "backtest_summary.json")
    )
    costs_daily_source = (
        Path(costs_daily_path).expanduser().resolve()
        if costs_daily_path
        else (data_dir() / "execution" / "costs_daily.parquet")
    )
    contrib_source = (
        Path(backtest_contributions_path).expanduser().resolve()
        if backtest_contributions_path
        else (data_dir() / "backtest" / "backtest_contributions.parquet")
    )
    portfolio_summary_source = (
        Path(portfolio_summary_path).expanduser().resolve()
        if portfolio_summary_path
        else (data_dir() / "portfolio" / "portfolio_summary.json")
    )
    execution_summary_source = (
        Path(execution_assumptions_summary_path).expanduser().resolve()
        if execution_assumptions_summary_path
        else (data_dir() / "execution" / "execution_assumptions_summary.json")
    )

    backtest_daily = _prepare_backtest_daily(backtest_daily_source)
    costs_daily = _prepare_costs_daily(costs_daily_source)
    backtest_summary = _load_json(backtest_summary_source)

    inferred_model = (
        model_name
        if model_name is not None
        else (
            str(backtest_summary.get("model_name")).strip()
            if backtest_summary.get("model_name") is not None
            else None
        )
    )
    inferred_label = (
        label_name
        if label_name is not None
        else (
            str(backtest_summary.get("label_name")).strip()
            if backtest_summary.get("label_name") is not None
            else None
        )
    )

    selected_model = _pick_unique(backtest_daily, "model_name", inferred_model)
    selected_label = _pick_unique(backtest_daily, "label_name", inferred_label)

    backtest_daily = backtest_daily[
        (backtest_daily["model_name"] == selected_model) & (backtest_daily["label_name"] == selected_label)
    ].copy()
    costs_daily = costs_daily[
        (costs_daily["model_name"] == selected_model) & (costs_daily["label_name"] == selected_label)
    ].copy()

    modes = _norm_modes(portfolio_modes)
    if modes:
        mode_set = set(modes)
        backtest_daily = backtest_daily[backtest_daily["portfolio_mode"].isin(mode_set)].copy()
        costs_daily = costs_daily[costs_daily["portfolio_mode"].isin(mode_set)].copy()
        if backtest_daily.empty:
            raise ValueError(f"No backtest rows left after portfolio_mode filter: {list(modes)}")

    if backtest_daily.empty:
        raise ValueError("No backtest rows available for diagnostics after filtering.")

    merge_keys = ["date", "portfolio_mode", "model_name", "label_name"]
    costs_merge = costs_daily.copy()
    costs_merge["date"] = costs_merge["cost_date"]
    for col in ("gross_turnover", "total_turnover_cost", "total_entry_cost", "total_exit_cost"):
        if col not in costs_merge.columns:
            costs_merge[col] = np.nan
    costs_merge = costs_merge[
        ["date", "portfolio_mode", "model_name", "label_name", "total_cost", "gross_turnover", "total_turnover_cost", "total_entry_cost", "total_exit_cost"]
    ].rename(columns={"total_cost": "costs_daily_total_cost"})

    daily = backtest_daily.merge(costs_merge, on=merge_keys, how="left")
    daily["daily_cost_drag"] = daily["gross_return"] - daily["net_return"]
    if not np.allclose(
        daily["daily_cost_drag"].to_numpy(dtype=float),
        daily["total_cost"].to_numpy(dtype=float),
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError("daily_cost_drag must match total_cost from backtest_daily.")

    has_costs_match = daily["costs_daily_total_cost"].notna()
    if has_costs_match.any():
        mismatch = np.abs(
            daily.loc[has_costs_match, "costs_daily_total_cost"].to_numpy(dtype=float)
            - daily.loc[has_costs_match, "total_cost"].to_numpy(dtype=float)
        ) > 1e-10
        n_cost_mismatch_rows = int(mismatch.sum())
    else:
        n_cost_mismatch_rows = 0

    built_ts = datetime.now(UTC).isoformat()
    cfg = _cfg_hash(
        {
            "version": MODULE_VERSION,
            "backtest_daily_path": str(backtest_daily_source),
            "backtest_summary_path": str(backtest_summary_source),
            "costs_daily_path": str(costs_daily_source),
            "backtest_contributions_path": str(contrib_source),
            "portfolio_summary_path": str(portfolio_summary_source),
            "execution_assumptions_summary_path": str(execution_summary_source),
            "model_name": selected_model,
            "label_name": selected_label,
            "portfolio_modes": sorted(backtest_daily["portfolio_mode"].astype(str).unique().tolist()),
        }
    )

    daily["run_id"] = run_id
    daily["config_hash"] = cfg
    daily["built_ts_utc"] = built_ts

    daily_out = daily[
        [
            "date",
            "portfolio_mode",
            "model_name",
            "label_name",
            "gross_return",
            "net_return",
            "total_cost",
            "gross_equity",
            "net_equity",
            "drawdown_net",
            "daily_cost_drag",
            "gross_turnover",
            "total_turnover_cost",
            "total_entry_cost",
            "total_exit_cost",
            "run_id",
            "config_hash",
            "built_ts_utc",
        ]
    ].sort_values(["portfolio_mode", "date"]).reset_index(drop=True)
    assert_schema(daily_out, DIAGNOSTICS_DAILY_SCHEMA)

    by_mode_rows: list[dict[str, Any]] = []
    for mode, block in daily_out.groupby("portfolio_mode", sort=True):
        block = block.sort_values("date")
        mean_net = float(block["net_return"].mean())
        std_net = float(block["net_return"].std(ddof=0))
        sharpe_like = _mode_sharpe_like(mean_net, std_net)
        by_mode_rows.append(
            {
                "portfolio_mode": str(mode),
                "model_name": selected_model,
                "label_name": selected_label,
                "n_dates": int(block["date"].nunique()),
                "cumulative_gross_return": float(block["gross_equity"].iloc[-1] - 1.0),
                "cumulative_net_return": float(block["net_equity"].iloc[-1] - 1.0),
                "mean_daily_gross_return": float(block["gross_return"].mean()),
                "mean_daily_net_return": mean_net,
                "std_daily_net_return": std_net,
                "positive_net_return_rate": float((block["net_return"] > 0).mean()),
                "max_drawdown_net": float(block["drawdown_net"].min()),
                "total_cost_paid": float(block["total_cost"].sum()),
                "avg_daily_cost": float(block["total_cost"].mean()),
                "avg_turnover_if_available": (
                    float(block["gross_turnover"].mean())
                    if block["gross_turnover"].notna().any()
                    else None
                ),
                "cost_drag_total": float(block["daily_cost_drag"].sum()),
                "sharpe_like": sharpe_like,
                "total_turnover_cost_paid": (
                    float(block["total_turnover_cost"].sum())
                    if block["total_turnover_cost"].notna().any()
                    else None
                ),
                "total_entry_cost_paid": (
                    float(block["total_entry_cost"].sum())
                    if block["total_entry_cost"].notna().any()
                    else None
                ),
                "total_exit_cost_paid": (
                    float(block["total_exit_cost"].sum())
                    if block["total_exit_cost"].notna().any()
                    else None
                ),
                "run_id": run_id,
                "config_hash": cfg,
                "built_ts_utc": built_ts,
            }
        )

    by_mode = pd.DataFrame(by_mode_rows).sort_values("portfolio_mode").reset_index(drop=True)
    assert_schema(by_mode, DIAGNOSTICS_BY_MODE_SCHEMA)

    if by_mode.empty:
        raise ValueError("Diagnostics by-mode table is empty.")

    best_mode_by_cum_net = str(
        by_mode.sort_values(["cumulative_net_return", "portfolio_mode"], ascending=[False, True]).iloc[0]["portfolio_mode"]
    )

    sharpe_available = by_mode[by_mode["sharpe_like"].notna()].copy()
    best_mode_by_sharpe: str | None
    if sharpe_available.empty:
        best_mode_by_sharpe = None
    else:
        best_mode_by_sharpe = str(
            sharpe_available.sort_values(["sharpe_like", "portfolio_mode"], ascending=[False, True]).iloc[0]["portfolio_mode"]
        )

    execution_notes: str | None = None
    if execution_summary_source.exists():
        exec_summary = _load_json(execution_summary_source)
        execution_notes = (
            f"execution_delay_sessions={exec_summary.get('execution_delay_sessions')}, "
            f"fill_assumption={exec_summary.get('fill_assumption')}, "
            f"cost_timing={exec_summary.get('cost_timing')}, "
            f"n_skipped_rows={exec_summary.get('n_skipped_rows')}"
        )
    else:
        execution_notes = "execution assumptions summary not available"

    dropped_notes: dict[str, Any] | str
    dropped_keys = [
        key
        for key in (
            "rows_dropped_non_executable",
            "rows_dropped_non_session_execution_date",
            "rows_dropped_missing_tplus1_return",
        )
        if key in backtest_summary
    ]
    if dropped_keys:
        dropped_notes = {k: backtest_summary.get(k) for k in dropped_keys}
    else:
        dropped_notes = "dropped row counters not available"

    summary_payload = {
        "built_ts_utc": built_ts,
        "run_id": run_id,
        "config_hash": cfg,
        "module_version": MODULE_VERSION,
        "model_name": selected_model,
        "label_name": selected_label,
        "portfolio_modes": sorted(by_mode["portfolio_mode"].astype(str).unique().tolist()),
        "best_mode_by_cumulative_net_return": best_mode_by_cum_net,
        "best_mode_by_sharpe_like_if_available": best_mode_by_sharpe,
        "total_cost_paid_all_modes": _to_float(by_mode["total_cost_paid"].sum()),
        "mean_cost_drag": _to_float(daily_out["daily_cost_drag"].mean()),
        "max_drawdown_net_all_modes": _to_float(by_mode["max_drawdown_net"].min()),
        "cost_breakdown_all_modes": {
            "total_turnover_cost": _to_float(by_mode["total_turnover_cost_paid"].sum(skipna=True)),
            "total_entry_cost": _to_float(by_mode["total_entry_cost_paid"].sum(skipna=True)),
            "total_exit_cost": _to_float(by_mode["total_exit_cost_paid"].sum(skipna=True)),
        },
        "notes_on_execution_assumptions": execution_notes,
        "notes_on_dropped_rows_if_available": dropped_notes,
        "n_cost_mismatch_rows_vs_costs_daily": n_cost_mismatch_rows,
        "output_paths": {},
    }

    out_dir = Path(output_dir).expanduser().resolve() if output_dir else (data_dir() / "backtest")
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_path = write_parquet(
        daily_out,
        out_dir / "backtest_diagnostics_daily.parquet",
        schema_name=DIAGNOSTICS_DAILY_SCHEMA.name,
        run_id=run_id,
    )
    by_mode_path = write_parquet(
        by_mode,
        out_dir / "backtest_diagnostics_by_mode.parquet",
        schema_name=DIAGNOSTICS_BY_MODE_SCHEMA.name,
        run_id=run_id,
    )
    summary_path = out_dir / "backtest_diagnostics_summary.json"
    summary_payload["output_paths"] = {
        "backtest_diagnostics_daily": str(daily_path),
        "backtest_diagnostics_by_mode": str(by_mode_path),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "backtest_diagnostics_built",
        run_id=run_id,
        model_name=selected_model,
        label_name=selected_label,
        row_count_daily=int(len(daily_out)),
        row_count_by_mode=int(len(by_mode)),
        best_mode_by_cumulative_net_return=best_mode_by_cum_net,
        output_dir=str(out_dir),
    )

    return BacktestDiagnosticsResult(
        diagnostics_daily_path=daily_path,
        diagnostics_by_mode_path=by_mode_path,
        diagnostics_summary_path=summary_path,
        row_count_daily=int(len(daily_out)),
        row_count_by_mode=int(len(by_mode)),
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
        description="Build MVP diagnostics/reporting artifacts from backtest outputs."
    )
    parser.add_argument("--backtest-daily-path", type=str, default=None)
    parser.add_argument("--backtest-summary-path", type=str, default=None)
    parser.add_argument("--costs-daily-path", type=str, default=None)
    parser.add_argument("--backtest-contributions-path", type=str, default=None)
    parser.add_argument("--portfolio-summary-path", type=str, default=None)
    parser.add_argument("--execution-assumptions-summary-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--label-name", type=str, default=None)
    parser.add_argument("--portfolio-modes", type=_parse_csv_strings, default=tuple())
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_backtest_diagnostics(
        backtest_daily_path=args.backtest_daily_path,
        backtest_summary_path=args.backtest_summary_path,
        costs_daily_path=args.costs_daily_path,
        backtest_contributions_path=args.backtest_contributions_path,
        portfolio_summary_path=args.portfolio_summary_path,
        execution_assumptions_summary_path=args.execution_assumptions_summary_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        label_name=args.label_name,
        portfolio_modes=args.portfolio_modes,
        run_id=args.run_id,
    )
    print("Backtest diagnostics built:")
    print(f"- diagnostics_daily: {result.diagnostics_daily_path}")
    print(f"- diagnostics_by_mode: {result.diagnostics_by_mode_path}")
    print(f"- diagnostics_summary: {result.diagnostics_summary_path}")
    print(f"- row_count_daily: {result.row_count_daily}")
    print(f"- row_count_by_mode: {result.row_count_by_mode}")
    print(f"- model_name: {result.model_name}")
    print(f"- label_name: {result.label_name}")


if __name__ == "__main__":
    main()
