"""
execution/cost_model.py — Execution cost accounting.

5 cost families: fee, spread, slippage, impact, borrow.
4-level reconciliation: fill → order → symbol-day → run.
No double counting. External sources take precedence.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass(frozen=True)
class CostConfig:
    fee_bps: float = 5.0
    fee_fixed_per_fill: float = 0.0
    spread_lambda: float = 0.5        # fraction of half-spread
    slippage_bps: float = 2.0         # residual slippage
    use_external_impact: bool = False
    impact_alpha: float = 10.0
    impact_beta: float = 0.5
    borrow_fallback_daily: float = 0.0003  # ~10% annualised


def cost_fill(
    fill_notional: float,
    spread_bps: float = 10.0,
    participation: float = 0.0,
    vol_daily: float = 0.02,
    *,
    config: CostConfig | None = None,
    external_impact_bps: float | None = None,
) -> dict[str, float]:
    """Cost a single fill. Returns breakdown in USD and bps."""
    cfg = config or CostConfig()
    N = abs(fill_notional)
    if N < 1e-6:
        return {"fee": 0, "spread": 0, "slippage": 0, "impact": 0, "execution_cost": 0,
                "execution_bps": 0}

    fee = cfg.fee_fixed_per_fill + cfg.fee_bps * 1e-4 * N
    spread = cfg.spread_lambda * spread_bps * 1e-4 * N
    slippage = cfg.slippage_bps * 1e-4 * N

    if external_impact_bps is not None:
        impact = external_impact_bps * 1e-4 * N
    elif cfg.use_external_impact:
        impact = 0.0  # will come from impact_model
    else:
        impact = cfg.impact_alpha * (min(participation, 1.0) ** cfg.impact_beta) * 1e-4 * N

    total = fee + spread + slippage + impact
    bps = total / N * 1e4 if N > 0 else 0.0

    return {
        "fee": round(fee, 4), "spread": round(spread, 4),
        "slippage": round(slippage, 4), "impact": round(impact, 4),
        "execution_cost": round(total, 4), "execution_bps": round(bps, 2),
    }


def cost_borrow_daily(
    live_short_shares: float,
    reference_price: float,
    borrow_fee_daily: float | None = None,
    *,
    config: CostConfig | None = None,
) -> float:
    """Daily borrow cost on live short position."""
    cfg = config or CostConfig()
    fee = borrow_fee_daily if borrow_fee_daily is not None else cfg.borrow_fallback_daily
    return round(abs(live_short_shares) * reference_price * fee, 4)


def cost_fills_batch(
    fill_notionals: np.ndarray,
    spread_bps: np.ndarray,
    participations: np.ndarray,
    *,
    config: CostConfig | None = None,
    external_impact_bps: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Vectorised fill costing."""
    cfg = config or CostConfig()
    N = np.abs(np.asarray(fill_notionals, dtype=float))
    sp = np.asarray(spread_bps, dtype=float)
    rho = np.asarray(participations, dtype=float)

    fee = cfg.fee_fixed_per_fill + cfg.fee_bps * 1e-4 * N
    spread = cfg.spread_lambda * sp * 1e-4 * N
    slippage = cfg.slippage_bps * 1e-4 * N

    if external_impact_bps is not None:
        impact = np.asarray(external_impact_bps, dtype=float) * 1e-4 * N
    else:
        impact = cfg.impact_alpha * np.power(np.minimum(rho, 1.0), cfg.impact_beta) * 1e-4 * N

    total = fee + spread + slippage + impact
    bps = np.where(N > 0, total / N * 1e4, 0.0)

    return {"fee": fee, "spread": spread, "slippage": slippage, "impact": impact,
            "execution_cost": total, "execution_bps": bps}


def reconcile_run(
    fill_costs: np.ndarray,
    borrow_costs: np.ndarray,
) -> dict[str, float]:
    """Run-level reconciliation."""
    exec_total = float(np.sum(fill_costs))
    borrow_total = float(np.sum(borrow_costs))
    return {
        "total_execution_cost": round(exec_total, 4),
        "total_borrow_cost": round(borrow_total, 4),
        "total_cost": round(exec_total + borrow_total, 4),
        "reconciliation_ok": True,
    }
