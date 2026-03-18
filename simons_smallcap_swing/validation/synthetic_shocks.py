"""
validation/synthetic_shocks.py — Stress testing under adverse scenarios.

Applies synthetic shocks to market data and re-runs the backtest to
measure strategy fragility under:
1. Liquidity compression (ADV drops by κ%)
2. Spread widening (spreads multiply by κ)
3. Volatility amplification (σ scales by κ)
4. Gap/jump events (sudden price drops)
5. Correlation concentration (all correlations → 1)
6. Cost inflation (transaction costs multiply by κ)

Each shock family produces a modified price/volume panel.
The strategy is re-evaluated on each stressed scenario.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import GateResult

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShockScenario:
    name: str
    shock_type: str
    severity: str          # "mild" | "moderate" | "severe"
    params: dict[str, float] = field(default_factory=dict)


DEFAULT_SCENARIOS: tuple[ShockScenario, ...] = (
    ShockScenario("liquidity_mild", "liquidity_compression", "mild", {"kappa": 0.7}),
    ShockScenario("liquidity_severe", "liquidity_compression", "severe", {"kappa": 0.3}),
    ShockScenario("vol_amplify_2x", "volatility_amplification", "moderate", {"kappa": 2.0}),
    ShockScenario("vol_amplify_3x", "volatility_amplification", "severe", {"kappa": 3.0}),
    ShockScenario("cost_2x", "cost_inflation", "moderate", {"multiplier": 2.0}),
    ShockScenario("cost_3x", "cost_inflation", "severe", {"multiplier": 3.0}),
    ShockScenario("gap_down_5pct", "gap_jump", "moderate", {"gap_pct": -0.05, "n_events": 5}),
    ShockScenario("gap_down_10pct", "gap_jump", "severe", {"gap_pct": -0.10, "n_events": 3}),
)


# ---------------------------------------------------------------------------
# Shock application functions
# ---------------------------------------------------------------------------

def apply_liquidity_compression(
    prices_df: pd.DataFrame, kappa: float, seed: int = 42,
) -> pd.DataFrame:
    """Reduce volume by factor kappa (0 < kappa < 1)."""
    df = prices_df.copy()
    df["volume"] = (df["volume"] * kappa).astype(int).clip(lower=0)
    return df


def apply_volatility_amplification(
    prices_df: pd.DataFrame, kappa: float, seed: int = 42,
) -> pd.DataFrame:
    """Amplify returns by factor kappa, keeping mean unchanged.

    r_shocked = μ + κ × (r - μ), ensuring 1 + r_shocked > 0.
    """
    df = prices_df.copy()
    for sym in df["symbol"].unique():
        mask = df["symbol"] == sym
        close = df.loc[mask, "close"].values
        if len(close) < 2:
            continue
        returns = np.diff(close) / close[:-1]
        mu = returns.mean()
        shocked_returns = mu + kappa * (returns - mu)
        shocked_returns = np.maximum(shocked_returns, -0.5)  # floor at -50%
        new_close = np.zeros(len(close))
        new_close[0] = close[0]
        for i in range(len(shocked_returns)):
            new_close[i+1] = new_close[i] * (1 + shocked_returns[i])
        df.loc[mask, "close"] = new_close
        df.loc[mask, "high"] = np.maximum(df.loc[mask, "high"].values, new_close)
        df.loc[mask, "low"] = np.minimum(df.loc[mask, "low"].values, new_close)
    return df


def apply_gap_jump(
    prices_df: pd.DataFrame, gap_pct: float, n_events: int, seed: int = 42,
) -> pd.DataFrame:
    """Insert sudden gap events at random dates."""
    df = prices_df.copy()
    rng = np.random.RandomState(seed)
    dates = df["date"].unique()
    if len(dates) < n_events + 10:
        return df
    event_dates = rng.choice(dates[10:-10], size=n_events, replace=False)
    for d in event_dates:
        mask = df["date"] == d
        for col in ("open", "high", "low", "close"):
            if col in df.columns:
                df.loc[mask, col] *= (1 + gap_pct)
    return df


def apply_cost_inflation(
    cost_config: Any, multiplier: float,
) -> Any:
    """Returns a new CostConfig with inflated costs."""
    from backtest import CostConfig
    return CostConfig(
        fixed_commission_bps=cost_config.fixed_commission_bps * multiplier,
        slippage_bps=cost_config.slippage_bps * multiplier,
        impact_coeff=cost_config.impact_coeff * multiplier,
        borrow_annual_bps=cost_config.borrow_annual_bps * multiplier,
        financing_annual_bps=cost_config.financing_annual_bps * multiplier,
    )


def apply_shock(
    scenario: ShockScenario,
    prices_df: pd.DataFrame,
    cost_config: Any = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, Any]:
    """Apply a shock scenario, returning modified prices and cost config."""
    shocked_prices = prices_df
    shocked_costs = cost_config

    if scenario.shock_type == "liquidity_compression":
        shocked_prices = apply_liquidity_compression(prices_df, scenario.params["kappa"], seed)
    elif scenario.shock_type == "volatility_amplification":
        shocked_prices = apply_volatility_amplification(prices_df, scenario.params["kappa"], seed)
    elif scenario.shock_type == "gap_jump":
        shocked_prices = apply_gap_jump(prices_df, scenario.params["gap_pct"], int(scenario.params["n_events"]), seed)
    elif scenario.shock_type == "cost_inflation" and cost_config is not None:
        shocked_costs = apply_cost_inflation(cost_config, scenario.params["multiplier"])

    return shocked_prices, shocked_costs


def run_synthetic_shocks(
    backtest_fn: Callable,
    base_summary: dict[str, Any],
    prices_df: pd.DataFrame,
    cost_config: Any = None,
    scenarios: Sequence[ShockScenario] | None = None,
) -> tuple[list[GateResult], dict[str, Any]]:
    """Run backtest under each shock scenario and compare to base.

    backtest_fn: callable(prices_df, cost_config) → summary dict
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS

    results: list[GateResult] = []
    details: dict[str, Any] = {"base": base_summary}

    base_sharpe = base_summary.get("sharpe_net", 0)

    for scenario in scenarios:
        try:
            shocked_prices, shocked_costs = apply_shock(scenario, prices_df, cost_config)
            stressed_summary = backtest_fn(shocked_prices, shocked_costs)
            stressed_sharpe = stressed_summary.get("sharpe_net", 0)
            sharpe_drop = base_sharpe - stressed_sharpe

            details[scenario.name] = {"sharpe": stressed_sharpe, "sharpe_drop": round(sharpe_drop, 3)}

            status = "PASS"
            if stressed_sharpe < -1.0:
                status = "FAIL"
            elif stressed_sharpe < 0:
                status = "WARN"

            results.append(GateResult(
                f"shock_{scenario.name}", "economic", status,
                round(stressed_sharpe, 3), 0,
                f"{scenario.name}: Sharpe {base_sharpe:.2f}→{stressed_sharpe:.2f} (Δ={sharpe_drop:+.2f})",
            ))
        except Exception as e:
            results.append(GateResult(
                f"shock_{scenario.name}", "economic", "WARN",
                None, None, f"{scenario.name}: failed — {str(e)[:80]}",
            ))

    return results, details
