from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time
from typing import Callable, Iterable

# Allow direct script execution: `python simons_smallcap_swing/run_week6_execution_backtest.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.diagnostics import run_backtest_diagnostics
from backtest.engine import run_backtest_engine
from execution.assumptions import run_execution_assumptions
from execution.cost_model import run_cost_model
from portfolio.construct_portfolio import run_construct_portfolio
from simons_core.io.paths import data_dir


DEFAULT_SPLIT_ROLES: tuple[str, ...] = ("valid", "test")
DEFAULT_PORTFOLIO_MODES: tuple[str, ...] = ("long_only_top_n", "long_short_top_bottom_n")


@dataclass(frozen=True)
class Week6RunResult:
    run_prefix: str
    data_root: Path
    manifest_path: Path
    artifacts: dict[str, Path]
    statuses: dict[str, str]


def _run_id(prefix: str, step: str) -> str:
    return f"{prefix}_{step}"


def _run_step(
    idx: int,
    total: int,
    label: str,
    func: Callable[..., object],
    **kwargs: object,
) -> object:
    t0 = time.perf_counter()
    print(f"[{idx}/{total}] {label} ...")
    try:
        out = func(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"Step failed [{idx}/{total}] {label}: {exc}") from exc
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    print(f"[{idx}/{total}] {label} done ({elapsed_ms} ms)")
    return out


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    items = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not items:
        raise ValueError("Expected at least one comma-separated value.")
    return items


def _normalize_list(values: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(sorted({str(item).strip() for item in values if str(item).strip()}))
    if not normalized:
        raise ValueError("Expected at least one value.")
    return normalized


def _resolve_default_paths(
    *,
    base_data: Path,
    signals_path: str | Path | None,
    universe_history_path: str | Path | None,
    trading_calendar_path: str | Path | None,
    adjusted_prices_path: str | Path | None,
) -> dict[str, Path]:
    return {
        "signals_daily": (
            Path(signals_path).expanduser().resolve()
            if signals_path
            else (base_data / "signals" / "signals_daily.parquet").resolve()
        ),
        "universe_history": (
            Path(universe_history_path).expanduser().resolve()
            if universe_history_path
            else (base_data / "universe" / "universe_history.parquet").resolve()
        ),
        "trading_calendar": (
            Path(trading_calendar_path).expanduser().resolve()
            if trading_calendar_path
            else (base_data / "reference" / "trading_calendar.parquet").resolve()
        ),
        "adjusted_prices": (
            Path(adjusted_prices_path).expanduser().resolve()
            if adjusted_prices_path
            else (base_data / "price" / "adjusted_prices.parquet").resolve()
        ),
    }


def _ensure_week6_prerequisites(expected: dict[str, Path]) -> dict[str, Path]:
    missing = [name for name, path in expected.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Week 6 runner requires Week 1-5 artifacts. "
            f"Missing: {missing}. Paths checked: "
            + ", ".join(f"{name}={path}" for name, path in expected.items())
        )
    return expected


def run_week6_execution_backtest(
    *,
    run_prefix: str = "week6_execution_backtest",
    data_root: str | Path | None = None,
    signals_path: str | Path | None = None,
    universe_history_path: str | Path | None = None,
    trading_calendar_path: str | Path | None = None,
    adjusted_prices_path: str | Path | None = None,
    model_name: str = "ridge_baseline",
    label_name: str = "fwd_ret_5d",
    split_name: str | None = None,
    horizon_days: int | None = 5,
    split_roles: Iterable[str] = DEFAULT_SPLIT_ROLES,
    portfolio_modes: Iterable[str] = DEFAULT_PORTFOLIO_MODES,
    top_n: int = 20,
    bottom_n: int = 20,
    execution_delay_sessions: int = 1,
    fill_assumption: str = "full_fill",
    cost_timing: str = "apply_on_execution_date",
    cost_bps_per_turnover: float = 10.0,
    entry_bps: float = 2.0,
    exit_bps: float = 2.0,
) -> Week6RunResult:
    base_data = Path(data_root).expanduser().resolve() if data_root else data_dir()
    portfolio_root = base_data / "portfolio"
    execution_root = base_data / "execution"
    backtest_root = base_data / "backtest"
    for directory in (portfolio_root, execution_root, backtest_root):
        directory.mkdir(parents=True, exist_ok=True)

    prereq = _ensure_week6_prerequisites(
        _resolve_default_paths(
            base_data=base_data,
            signals_path=signals_path,
            universe_history_path=universe_history_path,
            trading_calendar_path=trading_calendar_path,
            adjusted_prices_path=adjusted_prices_path,
        )
    )

    selected_split_roles = _normalize_list(split_roles)
    selected_portfolio_modes = _normalize_list(portfolio_modes)

    total_steps = 5
    step = 1
    statuses: dict[str, str] = {}

    portfolio = _run_step(
        step,
        total_steps,
        "construct portfolio",
        run_construct_portfolio,
        signals_path=prereq["signals_daily"],
        universe_history_path=prereq["universe_history"],
        output_dir=portfolio_root,
        model_name=model_name,
        label_name=label_name,
        split_name=split_name,
        horizon_days=horizon_days,
        split_roles=selected_split_roles,
        portfolio_modes=selected_portfolio_modes,
        top_n=int(top_n),
        bottom_n=int(bottom_n),
        run_id=_run_id(run_prefix, "construct_portfolio"),
    )
    statuses["construct_portfolio"] = "DONE"
    step += 1

    execution = _run_step(
        step,
        total_steps,
        "apply execution assumptions",
        run_execution_assumptions,
        holdings_path=portfolio.holdings_path,
        rebalance_path=portfolio.rebalance_path,
        trading_calendar_path=prereq["trading_calendar"],
        adjusted_prices_path=prereq["adjusted_prices"],
        output_dir=execution_root,
        model_name=model_name,
        label_name=label_name,
        portfolio_modes=selected_portfolio_modes,
        execution_delay_sessions=int(execution_delay_sessions),
        fill_assumption=fill_assumption,
        cost_timing=cost_timing,
        run_id=_run_id(run_prefix, "execution_assumptions"),
    )
    statuses["execution_assumptions"] = "DONE"
    step += 1

    costs = _run_step(
        step,
        total_steps,
        "run cost model",
        run_cost_model,
        holdings_path=portfolio.holdings_path,
        rebalance_path=execution.execution_rebalance_path,
        output_dir=execution_root,
        model_name=model_name,
        label_name=label_name,
        portfolio_modes=selected_portfolio_modes,
        cost_bps_per_turnover=float(cost_bps_per_turnover),
        entry_bps=float(entry_bps),
        exit_bps=float(exit_bps),
        run_id=_run_id(run_prefix, "cost_model"),
    )
    statuses["cost_model"] = "DONE"
    step += 1

    backtest = _run_step(
        step,
        total_steps,
        "run backtest engine",
        run_backtest_engine,
        holdings_path=execution.execution_holdings_path,
        costs_daily_path=costs.costs_daily_path,
        adjusted_prices_path=prereq["adjusted_prices"],
        trading_calendar_path=prereq["trading_calendar"],
        output_dir=backtest_root,
        model_name=model_name,
        label_name=label_name,
        portfolio_modes=selected_portfolio_modes,
        run_id=_run_id(run_prefix, "backtest_engine"),
    )
    statuses["backtest_engine"] = "DONE"
    step += 1

    diagnostics = _run_step(
        step,
        total_steps,
        "run backtest diagnostics",
        run_backtest_diagnostics,
        backtest_daily_path=backtest.backtest_daily_path,
        backtest_summary_path=backtest.backtest_summary_path,
        costs_daily_path=costs.costs_daily_path,
        backtest_contributions_path=backtest.backtest_contributions_path,
        execution_assumptions_summary_path=execution.execution_assumptions_summary_path,
        output_dir=backtest_root,
        model_name=model_name,
        label_name=label_name,
        portfolio_modes=selected_portfolio_modes,
        run_id=_run_id(run_prefix, "backtest_diagnostics"),
    )
    statuses["backtest_diagnostics"] = "DONE"

    artifacts: dict[str, Path] = {
        "portfolio_holdings": portfolio.holdings_path,
        "portfolio_rebalance": portfolio.rebalance_path,
        "portfolio_summary": portfolio.summary_path,
        "execution_holdings": execution.execution_holdings_path,
        "execution_rebalance": execution.execution_rebalance_path,
        "execution_assumptions_summary": execution.execution_assumptions_summary_path,
        "costs_positions": costs.costs_positions_path,
        "costs_daily": costs.costs_daily_path,
        "costs_summary": costs.costs_summary_path,
        "backtest_daily": backtest.backtest_daily_path,
        "backtest_contributions": backtest.backtest_contributions_path,
        "backtest_summary": backtest.backtest_summary_path,
        "backtest_diagnostics_daily": diagnostics.diagnostics_daily_path,
        "backtest_diagnostics_by_mode": diagnostics.diagnostics_by_mode_path,
        "backtest_diagnostics_summary": diagnostics.diagnostics_summary_path,
    }

    manifest_payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_prefix": run_prefix,
        "data_root": str(base_data),
        "steps_total": total_steps,
        "flags": {
            "model_name": model_name,
            "label_name": label_name,
            "split_name": split_name,
            "horizon_days": horizon_days,
            "split_roles": list(selected_split_roles),
            "portfolio_modes": list(selected_portfolio_modes),
            "top_n": int(top_n),
            "bottom_n": int(bottom_n),
            "execution_delay_sessions": int(execution_delay_sessions),
            "fill_assumption": fill_assumption,
            "cost_timing": cost_timing,
            "cost_bps_per_turnover": float(cost_bps_per_turnover),
            "entry_bps": float(entry_bps),
            "exit_bps": float(exit_bps),
        },
        "statuses": statuses,
        "prerequisites": {key: str(path) for key, path in prereq.items()},
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }
    manifest_path = base_data / f"week6_execution_backtest_manifest_{run_prefix}.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] Week 6 execution/backtest pipeline completed. Manifest: {manifest_path}")

    return Week6RunResult(
        run_prefix=run_prefix,
        data_root=base_data,
        manifest_path=manifest_path,
        artifacts=artifacts,
        statuses=statuses,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Week 6 execution + costs + backtest + diagnostics pipeline.")
    parser.add_argument("--run-prefix", type=str, default="week6_execution_backtest")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--signals-path", type=str, default=None)
    parser.add_argument("--universe-history-path", type=str, default=None)
    parser.add_argument("--trading-calendar-path", type=str, default=None)
    parser.add_argument("--adjusted-prices-path", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="ridge_baseline")
    parser.add_argument("--label-name", type=str, default="fwd_ret_5d")
    parser.add_argument("--split-name", type=str, default=None)
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--split-roles", type=_parse_csv_strings, default=DEFAULT_SPLIT_ROLES)
    parser.add_argument("--portfolio-modes", type=_parse_csv_strings, default=DEFAULT_PORTFOLIO_MODES)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--bottom-n", type=int, default=20)
    parser.add_argument("--execution-delay-sessions", type=int, default=1)
    parser.add_argument("--fill-assumption", type=str, default="full_fill")
    parser.add_argument("--cost-timing", type=str, default="apply_on_execution_date")
    parser.add_argument("--cost-bps-per-turnover", type=float, default=10.0)
    parser.add_argument("--entry-bps", type=float, default=2.0)
    parser.add_argument("--exit-bps", type=float, default=2.0)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_week6_execution_backtest(
        run_prefix=args.run_prefix,
        data_root=args.data_root,
        signals_path=args.signals_path,
        universe_history_path=args.universe_history_path,
        trading_calendar_path=args.trading_calendar_path,
        adjusted_prices_path=args.adjusted_prices_path,
        model_name=args.model_name,
        label_name=args.label_name,
        split_name=args.split_name,
        horizon_days=args.horizon_days,
        split_roles=args.split_roles,
        portfolio_modes=args.portfolio_modes,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        execution_delay_sessions=args.execution_delay_sessions,
        fill_assumption=args.fill_assumption,
        cost_timing=args.cost_timing,
        cost_bps_per_turnover=args.cost_bps_per_turnover,
        entry_bps=args.entry_bps,
        exit_bps=args.exit_bps,
    )
    print("Week 6 statuses:")
    for key, value in result.statuses.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()

