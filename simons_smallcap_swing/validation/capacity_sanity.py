"""
validation/capacity_sanity.py — Strategy capacity estimation under cost scaling.

Answers: "At what AUM does this strategy stop working?"

Scales the strategy by factor α and measures:
- Participation rate: |q| / ADV (>5% → market impact)
- Days to liquidate: |position| / (η × ADV) (>5 days → liquidity risk)
- Cost curve: how costs scale with α (linear + impact = super-linear)
- Retention: net_pnl / gross_pnl at each scale (drops with α)
- Capacity: max α where Sharpe_net > threshold and retention > minimum

The key insight: in small caps, capacity is MUCH smaller than in large caps.
A strategy that works at $1M might fail at $10M because you can't trade
5% of ADV without moving the price.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import GateResult

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CapacityConfig:
    alpha_grid: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
    max_participation_q95: float = 0.05    # 5% of ADV
    max_dtl_q95: float = 5.0              # 5 days to liquidate
    min_retention: float = 0.30            # 30% of gross P&L retained
    min_sharpe_net: float = 0.5            # minimum viable Sharpe
    impact_exponent: float = 0.5           # sqrt impact model
    adv_fill_fraction: float = 0.10        # can fill 10% of ADV per day


def estimate_capacity(
    trade_notionals: np.ndarray,
    adv_values: np.ndarray,
    gross_pnl_series: np.ndarray,
    cost_per_trade_bps: float,
    impact_coeff: float = 0.1,
    config: CapacityConfig | None = None,
) -> dict[str, Any]:
    """Estimate strategy capacity across scaling factors.

    Parameters
    ----------
    trade_notionals : per-trade absolute notional at base scale (α=1)
    adv_values : per-trade ADV
    gross_pnl_series : daily gross P&L at base scale
    cost_per_trade_bps : base cost per trade in bps
    impact_coeff : market impact coefficient
    config : CapacityConfig
    """
    cfg = config or CapacityConfig()
    results_by_alpha: list[dict[str, Any]] = []

    valid = (adv_values > 0) & (trade_notionals > 0)
    base_participation = trade_notionals[valid] / adv_values[valid]
    base_cost_total = trade_notionals.sum() * cost_per_trade_bps / 1e4
    gross_total = float(np.sum(gross_pnl_series))

    for alpha in cfg.alpha_grid:
        # Participation scales linearly
        participation = base_participation * alpha
        p_q50 = float(np.percentile(participation, 50))
        p_q95 = float(np.percentile(participation, 95))
        p_q99 = float(np.percentile(participation, 99))

        # Days to liquidate
        positions = trade_notionals * alpha
        dtl = positions[valid] / (cfg.adv_fill_fraction * adv_values[valid])
        dtl_q95 = float(np.percentile(dtl, 95))

        # Costs: linear part + impact (super-linear)
        linear_cost = base_cost_total * alpha
        impact_cost = impact_coeff * np.sum(
            trade_notionals[valid] * alpha * np.sqrt(participation)
        ) / 1e4
        total_cost = linear_cost + impact_cost

        # Gross P&L scales linearly
        gross_scaled = gross_total * alpha
        net_scaled = gross_scaled - total_cost
        retention = net_scaled / gross_scaled if abs(gross_scaled) > 0 else 0

        # Sharpe approximation (assumes vol scales with alpha too)
        n_days = len(gross_pnl_series)
        daily_net = gross_pnl_series * alpha - (total_cost / max(n_days, 1))
        mu = daily_net.mean()
        sig = daily_net.std()
        sharpe = mu * np.sqrt(252) / sig if sig > 0 else 0

        results_by_alpha.append({
            "alpha": alpha,
            "participation_q95": round(p_q95, 4),
            "dtl_q95": round(dtl_q95, 2),
            "total_cost": round(total_cost, 2),
            "retention": round(retention, 4),
            "sharpe_net": round(float(sharpe), 3),
            "viable": (p_q95 <= cfg.max_participation_q95
                      and dtl_q95 <= cfg.max_dtl_q95
                      and retention >= cfg.min_retention
                      and sharpe >= cfg.min_sharpe_net),
        })

    # Find capacity (max viable alpha)
    viable_alphas = [r["alpha"] for r in results_by_alpha if r["viable"]]
    capacity_alpha = max(viable_alphas) if viable_alphas else 0.0

    # Binding constraint
    binding = "none"
    if not viable_alphas and results_by_alpha:
        last = results_by_alpha[0]
        if last["participation_q95"] > cfg.max_participation_q95:
            binding = "participation"
        elif last["dtl_q95"] > cfg.max_dtl_q95:
            binding = "days_to_liquidate"
        elif last["retention"] < cfg.min_retention:
            binding = "cost_retention"
        elif last["sharpe_net"] < cfg.min_sharpe_net:
            binding = "sharpe"

    return {
        "capacity_alpha": capacity_alpha,
        "binding_constraint": binding,
        "scaling_table": results_by_alpha,
    }


def run_capacity_sanity(
    trade_notionals: np.ndarray,
    adv_values: np.ndarray,
    gross_pnl_series: np.ndarray,
    cost_per_trade_bps: float,
    base_capital: float = 1e6,
    config: CapacityConfig | None = None,
) -> tuple[list[GateResult], dict[str, Any]]:
    """Run capacity analysis and return gate results."""
    cap = estimate_capacity(trade_notionals, adv_values, gross_pnl_series,
                            cost_per_trade_bps, config=config)

    capacity_usd = cap["capacity_alpha"] * base_capital
    results = [GateResult(
        "capacity_sanity", "economic",
        "FAIL" if cap["capacity_alpha"] < 1.0 else ("WARN" if cap["capacity_alpha"] < 5.0 else "PASS"),
        f"${capacity_usd:,.0f}", f">${base_capital:,.0f}",
        f"Capacity: {cap['capacity_alpha']:.0f}× base (${capacity_usd:,.0f}), binding={cap['binding_constraint']}",
    )]

    return results, cap
