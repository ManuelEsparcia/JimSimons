from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

# Allow direct script execution: `python simons_smallcap_swing/data/price/market_proxies.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir, reference_dir
from simons_core.logging import get_logger

MIN_COVERAGE_RATIO = 0.25
EXTREME_MOVE_THRESHOLD = 0.10


@dataclass(frozen=True)
class MarketProxiesResult:
    market_proxies_path: Path
    summary_path: Path
    row_count: int
    start_date: str
    end_date: str
    avg_coverage_ratio: float
    config_hash: str


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains invalid dates.")
    return parsed.dt.normalize()


def _config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_inputs(
    *,
    adjusted_prices_path: str | Path | None,
    universe_history_path: str | Path | None,
    trading_calendar_path: str | Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    price_base = data_dir() / "price"
    universe_base = data_dir() / "universe"
    ref_base = reference_dir()

    adjusted_source = (
        Path(adjusted_prices_path).expanduser().resolve()
        if adjusted_prices_path
        else (price_base / "adjusted_prices.parquet")
    )
    universe_source = (
        Path(universe_history_path).expanduser().resolve()
        if universe_history_path
        else (universe_base / "universe_history.parquet")
    )
    calendar_source = (
        Path(trading_calendar_path).expanduser().resolve()
        if trading_calendar_path
        else (ref_base / "trading_calendar.parquet")
    )

    adjusted = read_parquet(adjusted_source)
    universe = read_parquet(universe_source)
    calendar = read_parquet(calendar_source)
    return adjusted, universe, calendar, adjusted_source, universe_source, calendar_source


def _validate_and_prepare(
    adjusted: pd.DataFrame,
    universe: pd.DataFrame,
    calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    adjusted_required = {"date", "instrument_id", "ticker", "close_adj", "volume_adj"}
    universe_required = {"date", "instrument_id", "ticker", "is_eligible"}
    calendar_required = {"date", "is_session"}

    missing_adjusted = sorted(adjusted_required - set(adjusted.columns))
    missing_universe = sorted(universe_required - set(universe.columns))
    missing_calendar = sorted(calendar_required - set(calendar.columns))
    if missing_adjusted:
        raise ValueError(f"adjusted_prices missing required columns: {missing_adjusted}")
    if missing_universe:
        raise ValueError(f"universe_history missing required columns: {missing_universe}")
    if missing_calendar:
        raise ValueError(f"trading_calendar missing required columns: {missing_calendar}")

    adjusted = adjusted.copy()
    universe = universe.copy()
    calendar = calendar.copy()

    adjusted["date"] = _normalize_date(adjusted["date"], column="date")
    adjusted["instrument_id"] = adjusted["instrument_id"].astype(str)
    adjusted["ticker"] = adjusted["ticker"].astype(str).str.upper().str.strip()
    adjusted["close_adj"] = pd.to_numeric(adjusted["close_adj"], errors="coerce")
    adjusted["volume_adj"] = pd.to_numeric(adjusted["volume_adj"], errors="coerce")

    universe["date"] = _normalize_date(universe["date"], column="date")
    universe["instrument_id"] = universe["instrument_id"].astype(str)
    universe["ticker"] = universe["ticker"].astype(str).str.upper().str.strip()
    universe["is_eligible"] = universe["is_eligible"].astype(bool)

    sessions = _normalize_date(
        calendar.loc[calendar["is_session"].astype(bool), "date"],
        column="date",
    )
    sessions = pd.DatetimeIndex(sorted(sessions.unique()))
    if sessions.empty:
        raise ValueError("Trading calendar has no active sessions.")

    if adjusted.empty:
        raise ValueError("adjusted_prices is empty.")
    if universe.empty:
        raise ValueError("universe_history is empty.")

    dup_adjusted = adjusted.duplicated(["date", "instrument_id"], keep=False)
    if dup_adjusted.any():
        raise ValueError("adjusted_prices has duplicate (date, instrument_id) rows.")

    dup_universe = universe.duplicated(["date", "instrument_id"], keep=False)
    if dup_universe.any():
        raise ValueError("universe_history has duplicate (date, instrument_id) rows.")

    invalid_adjusted_dates = adjusted.loc[~adjusted["date"].isin(set(sessions.tolist())), "date"]
    if not invalid_adjusted_dates.empty:
        sample = sorted({str(pd.Timestamp(item).date()) for item in invalid_adjusted_dates.head(10).tolist()})
        raise ValueError(
            "adjusted_prices contains rows outside trading calendar sessions. "
            f"Sample: {sample}"
        )

    invalid_universe_dates = universe.loc[~universe["date"].isin(set(sessions.tolist())), "date"]
    if not invalid_universe_dates.empty:
        sample = sorted({str(pd.Timestamp(item).date()) for item in invalid_universe_dates.head(10).tolist()})
        raise ValueError(
            "universe_history contains rows outside trading calendar sessions. "
            f"Sample: {sample}"
        )

    if adjusted["close_adj"].isna().all():
        raise ValueError("adjusted_prices has no valid close_adj values.")
    if (adjusted["close_adj"] <= 0).all():
        raise ValueError("adjusted_prices close_adj is non-positive for all rows.")

    adjusted = adjusted.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    universe = universe.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    return adjusted, universe, sessions


def _build_adjusted_metrics(adjusted: pd.DataFrame) -> pd.DataFrame:
    panel = adjusted[["date", "instrument_id", "ticker", "close_adj", "volume_adj"]].copy()
    panel["ret_1d"] = panel.groupby("instrument_id", sort=False)["close_adj"].pct_change()
    panel["volume_roll_median_20"] = panel.groupby("instrument_id", sort=False)["volume_adj"].transform(
        lambda series: series.rolling(20, min_periods=1).median()
    )
    denom = panel["volume_roll_median_20"].replace(0, np.nan)
    panel["turnover_component"] = panel["volume_adj"] / denom
    panel.loc[~np.isfinite(panel["turnover_component"]), "turnover_component"] = np.nan
    panel["turnover_component"] = panel["turnover_component"].clip(lower=0.0)
    return panel


def _aggregate_daily(
    *,
    adjusted_metrics: pd.DataFrame,
    universe: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    run_id: str,
    config_hash: str,
) -> pd.DataFrame:
    eligible = universe[universe["is_eligible"]].copy()
    if eligible.empty:
        raise ValueError("universe_history has no eligible rows (is_eligible=True).")

    min_date = eligible["date"].min()
    max_date = eligible["date"].max()
    sessions_scope = pd.DatetimeIndex(
        sorted(sessions[(sessions >= min_date) & (sessions <= max_date)].tolist())
    )
    if sessions_scope.empty:
        raise ValueError("No trading sessions overlap with eligible universe range.")

    universe_panel = eligible[["date", "instrument_id", "ticker"]].copy()
    merged = universe_panel.merge(
        adjusted_metrics[
            [
                "date",
                "instrument_id",
                "ticker",
                "close_adj",
                "ret_1d",
                "turnover_component",
            ]
        ],
        on=["date", "instrument_id", "ticker"],
        how="left",
    )

    n_names = (
        universe_panel.groupby("date", as_index=False)["instrument_id"]
        .nunique()
        .rename(columns={"instrument_id": "n_names"})
    )

    usable = merged[merged["ret_1d"].notna()].copy()
    daily_returns = (
        usable.groupby("date", as_index=False)
        .agg(
            n_names_with_prices=("instrument_id", "nunique"),
            equal_weight_return=("ret_1d", "mean"),
            median_return=("ret_1d", "median"),
            cross_sectional_vol=("ret_1d", lambda series: float(series.std(ddof=1)) if len(series) > 1 else 0.0),
            breadth_up=("ret_1d", lambda series: float((series > 0).mean())),
            breadth_down=("ret_1d", lambda series: float((series < 0).mean())),
            pct_extreme_moves=("ret_1d", lambda series: float((series.abs() > EXTREME_MOVE_THRESHOLD).mean())),
        )
    )

    daily_turnover = (
        merged[merged["turnover_component"].notna()]
        .groupby("date", as_index=False)
        .agg(turnover_proxy=("turnover_component", "median"))
    )

    daily = pd.DataFrame({"date": sessions_scope})
    daily = daily.merge(n_names, on="date", how="left")
    daily = daily.merge(daily_returns, on="date", how="left")
    daily = daily.merge(daily_turnover, on="date", how="left")

    daily["n_names"] = daily["n_names"].fillna(0).astype(int)
    daily["n_names_with_prices"] = daily["n_names_with_prices"].fillna(0).astype(int)
    denom = daily["n_names"].replace(0, np.nan)
    daily["coverage_ratio"] = (daily["n_names_with_prices"] / denom).fillna(0.0).clip(0.0, 1.0)

    fill_zero_cols = [
        "equal_weight_return",
        "median_return",
        "cross_sectional_vol",
        "breadth_up",
        "breadth_down",
        "turnover_proxy",
        "pct_extreme_moves",
    ]
    for col in fill_zero_cols:
        daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0.0)

    daily["breadth_up"] = daily["breadth_up"].clip(0.0, 1.0)
    daily["breadth_down"] = daily["breadth_down"].clip(0.0, 1.0)
    daily["pct_extreme_moves"] = daily["pct_extreme_moves"].clip(0.0, 1.0)
    daily["cross_sectional_vol"] = daily["cross_sectional_vol"].clip(lower=0.0)
    daily["turnover_proxy"] = daily["turnover_proxy"].clip(lower=0.0)

    daily["rolling_5d_vol"] = (
        daily["equal_weight_return"]
        .rolling(5, min_periods=2)
        .std(ddof=1)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    daily["rolling_20d_vol"] = (
        daily["equal_weight_return"]
        .rolling(20, min_periods=5)
        .std(ddof=1)
        .fillna(0.0)
        .clip(lower=0.0)
    )

    daily["severity_max"] = np.where(
        daily["n_names"] <= 0,
        "FAIL",
        np.where(daily["coverage_ratio"] < MIN_COVERAGE_RATIO, "WARN", "PASS"),
    )
    daily["run_id"] = run_id
    daily["config_hash"] = config_hash
    daily["built_ts_utc"] = datetime.now(UTC).isoformat()

    ordered_cols = [
        "date",
        "n_names",
        "n_names_with_prices",
        "coverage_ratio",
        "equal_weight_return",
        "median_return",
        "breadth_up",
        "breadth_down",
        "cross_sectional_vol",
        "turnover_proxy",
        "pct_extreme_moves",
        "rolling_5d_vol",
        "rolling_20d_vol",
        "severity_max",
        "run_id",
        "config_hash",
        "built_ts_utc",
    ]
    daily = daily[ordered_cols].sort_values("date").reset_index(drop=True)

    if daily.duplicated(["date"], keep=False).any():
        raise ValueError("market_proxies output has duplicate dates.")

    return daily


def _build_summary(
    *,
    daily: pd.DataFrame,
    run_id: str,
    baseline_convention: str,
    adjusted_source: Path,
    universe_source: Path,
    calendar_source: Path,
) -> dict[str, Any]:
    worst_idx = int(daily["coverage_ratio"].idxmin())
    worst_session = pd.Timestamp(daily.loc[worst_idx, "date"]).strftime("%Y-%m-%d")
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "start_date": pd.Timestamp(daily["date"].min()).strftime("%Y-%m-%d"),
        "end_date": pd.Timestamp(daily["date"].max()).strftime("%Y-%m-%d"),
        "n_sessions": int(len(daily)),
        "avg_coverage_ratio": float(daily["coverage_ratio"].mean()),
        "min_coverage_ratio": float(daily["coverage_ratio"].min()),
        "avg_n_names": float(daily["n_names"].mean()),
        "avg_n_names_with_prices": float(daily["n_names_with_prices"].mean()),
        "worst_session_by_coverage": worst_session,
        "pct_sessions_low_coverage": float((daily["coverage_ratio"] < MIN_COVERAGE_RATIO).mean()),
        "baseline_universe_convention": baseline_convention,
        "input_paths": {
            "adjusted_prices": str(adjusted_source),
            "universe_history": str(universe_source),
            "trading_calendar": str(calendar_source),
        },
        "severity_counts": {
            str(level): int(count)
            for level, count in daily["severity_max"].value_counts().sort_index().items()
        },
    }
    return payload


def build_market_proxies(
    *,
    adjusted_prices_path: str | Path | None = None,
    universe_history_path: str | Path | None = None,
    trading_calendar_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    run_id: str = "market_proxies_mvp_v1",
) -> MarketProxiesResult:
    logger = get_logger("data.price.market_proxies")

    adjusted, universe, calendar, adjusted_source, universe_source, calendar_source = _load_inputs(
        adjusted_prices_path=adjusted_prices_path,
        universe_history_path=universe_history_path,
        trading_calendar_path=trading_calendar_path,
    )
    adjusted, universe, sessions = _validate_and_prepare(adjusted, universe, calendar)

    baseline_convention = (
        "Aggregate on PIT universe_history rows where is_eligible=True for each date; "
        "no hindsight projection from current constituents."
    )
    config_hash = _config_hash(
        {
            "version": "market_proxies_mvp_v1",
            "min_coverage_ratio": MIN_COVERAGE_RATIO,
            "extreme_move_threshold": EXTREME_MOVE_THRESHOLD,
            "baseline_convention": baseline_convention,
            "paths": {
                "adjusted_prices": str(adjusted_source),
                "universe_history": str(universe_source),
                "trading_calendar": str(calendar_source),
            },
        }
    )

    adjusted_metrics = _build_adjusted_metrics(adjusted)
    daily = _aggregate_daily(
        adjusted_metrics=adjusted_metrics,
        universe=universe,
        sessions=sessions,
        run_id=run_id,
        config_hash=config_hash,
    )

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (data_dir() / "price")
    target_dir.mkdir(parents=True, exist_ok=True)

    market_proxies_path = write_parquet(
        daily,
        target_dir / "market_proxies.parquet",
        schema_name="market_proxies_mvp_v1",
        run_id=run_id,
    )

    summary_payload = _build_summary(
        daily=daily,
        run_id=run_id,
        baseline_convention=baseline_convention,
        adjusted_source=adjusted_source,
        universe_source=universe_source,
        calendar_source=calendar_source,
    )
    summary_payload["config_hash"] = config_hash
    summary_path = target_dir / "market_proxies.summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger.info(
        "market_proxies_built",
        run_id=run_id,
        row_count=int(len(daily)),
        start_date=summary_payload["start_date"],
        end_date=summary_payload["end_date"],
        avg_coverage_ratio=round(float(summary_payload["avg_coverage_ratio"]), 6),
        output_path=str(market_proxies_path),
    )

    return MarketProxiesResult(
        market_proxies_path=market_proxies_path,
        summary_path=summary_path,
        row_count=int(len(daily)),
        start_date=str(summary_payload["start_date"]),
        end_date=str(summary_payload["end_date"]),
        avg_coverage_ratio=float(summary_payload["avg_coverage_ratio"]),
        config_hash=config_hash,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build MVP market proxies from adjusted prices and PIT universe."
    )
    parser.add_argument("--adjusted-prices-path", type=str, default=None)
    parser.add_argument("--universe-history-path", type=str, default=None)
    parser.add_argument("--trading-calendar-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default="market_proxies_mvp_v1")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = build_market_proxies(
        adjusted_prices_path=args.adjusted_prices_path,
        universe_history_path=args.universe_history_path,
        trading_calendar_path=args.trading_calendar_path,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print("Market proxies built:")
    print(f"- path: {result.market_proxies_path}")
    print(f"- summary: {result.summary_path}")
    print(f"- rows: {result.row_count}")
    print(f"- start_date: {result.start_date}")
    print(f"- end_date: {result.end_date}")
    print(f"- avg_coverage_ratio: {result.avg_coverage_ratio:.6f}")


if __name__ == "__main__":
    main()
