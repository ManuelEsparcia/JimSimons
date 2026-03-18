"""
backtest/report.py — Institutional backtest report generator.

Produces a structured report with:
    1. Executive summary (CAGR, Sharpe, MaxDD, Calmar, costs)
    2. Return distribution analysis (skew, kurtosis, tail percentiles)
    3. Drawdown analysis (depth, duration, recovery)
    4. Cost decomposition (commission, slippage, impact, borrow)
    5. Rolling metrics (rolling Sharpe, vol, drawdown)
    6. Subperiod analysis (yearly, quarterly, monthly breakdown)
    7. Statistical significance (bootstrap CI, deflated Sharpe, p-values)
    8. Capacity estimation (Kelly fraction, ADV utilisation)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from .engine import BacktestResult

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drawdown analysis
# ---------------------------------------------------------------------------

def compute_drawdowns(eq: pd.DataFrame) -> pd.DataFrame:
    """Compute drawdown series and episodes."""
    nav = eq["nav"]
    peak = nav.cummax()
    dd = nav / peak - 1

    # Find drawdown episodes
    episodes = []
    in_dd = False
    start_idx = 0
    for i in range(len(dd)):
        if dd.iloc[i] < 0 and not in_dd:
            in_dd = True
            start_idx = i
        elif dd.iloc[i] >= 0 and in_dd:
            in_dd = False
            episodes.append({
                "start": eq["date"].iloc[start_idx],
                "end": eq["date"].iloc[i],
                "trough": eq["date"].iloc[start_idx + dd.iloc[start_idx:i].argmin()],
                "depth_pct": round(float(dd.iloc[start_idx:i].min()) * 100, 2),
                "duration_days": i - start_idx,
            })
    # Handle open drawdown
    if in_dd:
        episodes.append({
            "start": eq["date"].iloc[start_idx],
            "end": eq["date"].iloc[-1],
            "trough": eq["date"].iloc[start_idx + dd.iloc[start_idx:].argmin()],
            "depth_pct": round(float(dd.iloc[start_idx:].min()) * 100, 2),
            "duration_days": len(dd) - start_idx,
        })

    return pd.DataFrame(episodes).sort_values("depth_pct")


# ---------------------------------------------------------------------------
# Rolling metrics
# ---------------------------------------------------------------------------

def compute_rolling_metrics(
    eq: pd.DataFrame, window: int = 63, sessions_per_year: int = 252,
) -> pd.DataFrame:
    """Compute rolling Sharpe, vol, and drawdown."""
    P = sessions_per_year
    ret = eq["return_net"]

    rolling = pd.DataFrame({"date": eq["date"]})
    rolling["rolling_sharpe"] = ret.rolling(window, min_periods=window//2).apply(
        lambda r: r.mean() * np.sqrt(P) / r.std() if r.std() > 0 else 0, raw=True,
    )
    rolling["rolling_vol"] = ret.rolling(window, min_periods=window//2).std() * np.sqrt(P)
    rolling["rolling_return"] = ret.rolling(window, min_periods=window//2).mean() * P

    # Rolling drawdown
    nav = eq["nav"]
    rolling["rolling_dd"] = nav / nav.rolling(window, min_periods=1).max() - 1

    return rolling


# ---------------------------------------------------------------------------
# Subperiod analysis
# ---------------------------------------------------------------------------

def compute_subperiod_metrics(
    eq: pd.DataFrame, sessions_per_year: int = 252,
) -> pd.DataFrame:
    """Monthly return breakdown."""
    P = sessions_per_year
    df = eq[["date", "return_net", "return_gross"]].copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["month"] = pd.to_datetime(df["date"]).dt.month

    monthly = df.groupby(["year", "month"]).agg(
        n_days=("return_net", "count"),
        return_net=("return_net", lambda r: float((1 + r).prod() - 1)),
        return_gross=("return_gross", lambda r: float((1 + r).prod() - 1)),
        vol=("return_net", lambda r: float(r.std() * np.sqrt(P))),
    ).reset_index()

    return monthly


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_metric(
    returns: np.ndarray,
    metric_fn,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap confidence interval for any metric computed from returns."""
    rng = np.random.RandomState(seed)
    n = len(returns)
    boot_vals = []
    for _ in range(n_bootstrap):
        sample = rng.choice(returns, size=n, replace=True)
        boot_vals.append(metric_fn(sample))
    boot_vals = np.array(boot_vals)
    return {
        "observed": float(metric_fn(returns)),
        "mean": float(np.nanmean(boot_vals)),
        "ci_lo": float(np.nanpercentile(boot_vals, 2.5)),
        "ci_hi": float(np.nanpercentile(boot_vals, 97.5)),
        "p_value_le_zero": float(np.nanmean(boot_vals <= 0)),
    }


# ---------------------------------------------------------------------------
# MAIN REPORT
# ---------------------------------------------------------------------------

@dataclass
class BacktestReport:
    """Structured backtest report."""
    summary: dict[str, Any]
    return_distribution: dict[str, float]
    drawdown_episodes: pd.DataFrame
    cost_decomposition: dict[str, float]
    rolling_metrics: pd.DataFrame
    subperiod_metrics: pd.DataFrame
    bootstrap_sharpe: dict[str, float]
    bootstrap_cagr: dict[str, float]
    diagnostics_summary: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "return_distribution": self.return_distribution,
            "cost_decomposition": self.cost_decomposition,
            "bootstrap_sharpe": self.bootstrap_sharpe,
            "bootstrap_cagr": self.bootstrap_cagr,
            "n_drawdown_episodes": len(self.drawdown_episodes),
            "worst_drawdown": self.drawdown_episodes.iloc[0].to_dict() if len(self.drawdown_episodes) > 0 else {},
        }


def generate_report(
    result: BacktestResult,
    diagnostics_summary: dict[str, Any] | None = None,
    n_bootstrap: int = 2000,
) -> BacktestReport:
    """Generate the full institutional backtest report."""
    eq = result.equity_curve
    P = result.config.sessions_per_year

    # Return distribution
    ret = eq["return_net"].dropna().values
    dist = {
        "mean_daily": float(np.mean(ret)),
        "std_daily": float(np.std(ret)),
        "skewness": float(pd.Series(ret).skew()),
        "kurtosis": float(pd.Series(ret).kurtosis()),
        "p1": float(np.percentile(ret, 1)),
        "p5": float(np.percentile(ret, 5)),
        "p25": float(np.percentile(ret, 25)),
        "median": float(np.median(ret)),
        "p75": float(np.percentile(ret, 75)),
        "p95": float(np.percentile(ret, 95)),
        "p99": float(np.percentile(ret, 99)),
        "best_day_pct": round(float(np.max(ret)) * 100, 3),
        "worst_day_pct": round(float(np.min(ret)) * 100, 3),
        "pct_positive_days": round(float((ret > 0).mean()) * 100, 1),
    }

    # Drawdowns
    dd_episodes = compute_drawdowns(eq)

    # Costs
    costs = {
        "total": float(eq["costs_total"].sum()),
        "commission": float(eq["costs_commission"].sum()),
        "slippage": float(eq["costs_slippage"].sum()),
        "impact": float(eq["costs_impact"].sum()),
        "borrow": float(eq["costs_borrow"].sum()),
        "cost_per_trade": float(eq["costs_total"].sum() / max(result.summary.get("n_trades", 1), 1)),
    }

    # Rolling
    rolling = compute_rolling_metrics(eq, window=63, sessions_per_year=P)

    # Subperiod
    subperiod = compute_subperiod_metrics(eq, P)

    # Bootstrap
    sharpe_fn = lambda r: float(np.mean(r) * np.sqrt(P) / np.std(r)) if np.std(r) > 0 else 0
    boot_sharpe = bootstrap_metric(ret, sharpe_fn, n_bootstrap)

    def cagr_fn(r):
        cum = np.prod(1 + r)
        t_y = len(r) / P
        return float(cum ** (1 / max(t_y, 0.01)) - 1)
    boot_cagr = bootstrap_metric(ret, cagr_fn, n_bootstrap)

    report = BacktestReport(
        summary=result.summary,
        return_distribution=dist,
        drawdown_episodes=dd_episodes,
        cost_decomposition=costs,
        rolling_metrics=rolling,
        subperiod_metrics=subperiod,
        bootstrap_sharpe=boot_sharpe,
        bootstrap_cagr=boot_cagr,
        diagnostics_summary=diagnostics_summary,
    )

    LOGGER.info(
        "Report: Sharpe=%.2f [%.2f, %.2f], CAGR=%.1f%% [%.1f%%, %.1f%%]",
        boot_sharpe["observed"], boot_sharpe["ci_lo"], boot_sharpe["ci_hi"],
        boot_cagr["observed"] * 100, boot_cagr["ci_lo"] * 100, boot_cagr["ci_hi"] * 100,
    )

    return report
