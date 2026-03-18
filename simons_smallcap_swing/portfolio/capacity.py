"""
portfolio/capacity.py — AUM capacity estimation.

Scales the rebalance tape across AUM scenarios and checks:
    1. Market participation p95 ≤ cap
    2. Trade DTL p95 ≤ cap
    3. Position DTL p95 ≤ cap
    4. Single-name participation ≤ cap
    5. Return retention ≥ threshold
    6. Net return ≥ minimum

Monotonic regularisation applied before rule evaluation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np


@dataclass(frozen=True)
class CapacityConfig:
    aum_reference: float = 1e7
    scenario_multipliers: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0)
    phi: float = 0.10                    # usable ADV fraction
    eps_L: float = 10_000.0              # liquidity floor USD
    pi_cap_hard: float = 5.0            # defensive hard cap on participation
    pi_p95_max: float = 0.10
    dtl_trade_max: float = 1.0          # days
    dtl_pos_max: float = 5.0
    pi_single_name_max: float = 0.20
    rho_min: float = 0.80               # min return retention
    r_net_min: float = 0.02             # min net annual return
    impact_eta: float = 10.0            # impact coefficient
    impact_gamma: float = 0.5           # impact exponent (square-root)
    fee_bps: float = 5.0
    r_gross_ann: float = 0.25           # gross annual return baseline


def usable_liquidity(adv_usd: np.ndarray, phi: float, eps_L: float) -> np.ndarray:
    return np.maximum(phi * np.asarray(adv_usd, dtype=float), eps_L)


def robust_participation(order_notional: np.ndarray, usable_adv: np.ndarray, pi_cap: float) -> np.ndarray:
    pi = order_notional / np.maximum(usable_adv, 1.0)
    return np.minimum(pi, pi_cap)


def impact_cost_bps(spread_bps: np.ndarray, fee_bps: float, eta: float, gamma: float, participation: np.ndarray) -> np.ndarray:
    return 0.5 * spread_bps + fee_bps + eta * np.power(np.minimum(participation, 1.0), gamma)


def run_capacity(
    delta_w: np.ndarray,
    w_post: np.ndarray,
    adv_usd: np.ndarray,
    spread_bps: np.ndarray,
    *,
    config: CapacityConfig | None = None,
    turnover_ann: float = 12.0,
) -> dict[str, Any]:
    """Run capacity analysis across AUM scenarios."""
    cfg = config or CapacityConfig()
    dw = np.abs(np.asarray(delta_w, dtype=float).ravel())
    wp = np.abs(np.asarray(w_post, dtype=float).ravel())
    adv = np.asarray(adv_usd, dtype=float).ravel()
    spread = np.asarray(spread_bps, dtype=float).ravel()

    L = usable_liquidity(adv, cfg.phi, cfg.eps_L)

    scenarios = []
    for m in cfg.scenario_multipliers:
        A = m * cfg.aum_reference
        order_notional = A * dw
        pos_notional = A * wp

        pi = robust_participation(order_notional, L, cfg.pi_cap_hard)
        dtl_trade = order_notional / L
        dtl_pos = pos_notional / L

        cost = impact_cost_bps(spread, cfg.fee_bps, cfg.impact_eta, cfg.impact_gamma, pi)
        avg_cost = float(np.average(cost, weights=np.maximum(order_notional, 1e-12)))
        ann_cost = turnover_ann * avg_cost
        r_net = cfg.r_gross_ann - ann_cost / 1e4
        rho = r_net / max(cfg.r_gross_ann - (turnover_ann * float(np.average(
            impact_cost_bps(spread, cfg.fee_bps, cfg.impact_eta, cfg.impact_gamma,
                           robust_participation(cfg.aum_reference * dw, L, cfg.pi_cap_hard)),
            weights=np.maximum(cfg.aum_reference * dw, 1e-12)
        ))) / 1e4, 1e-6)

        scenarios.append({
            "multiplier": m, "aum": A,
            "pi_p50": float(np.median(pi)),
            "pi_p95": float(np.percentile(pi, 95)) if pi.size > 0 else 0.0,
            "pi_max": float(pi.max()) if pi.size > 0 else 0.0,
            "dtl_trade_p95": float(np.percentile(dtl_trade, 95)) if dtl_trade.size > 0 else 0.0,
            "dtl_pos_p95": float(np.percentile(dtl_pos, 95)) if dtl_pos.size > 0 else 0.0,
            "avg_cost_bps": round(avg_cost, 2),
            "ann_cost_bps": round(ann_cost, 2),
            "r_net_ann": round(r_net, 4),
            "return_retention": round(rho, 4),
        })

    # Monotonic regularisation
    for key in ("pi_p95", "dtl_trade_p95", "dtl_pos_p95", "avg_cost_bps"):
        running_max = 0.0
        for s in scenarios:
            running_max = max(running_max, s[key])
            s[key] = running_max

    # Evaluate hard rules
    for s in scenarios:
        breaches = []
        if s["pi_p95"] > cfg.pi_p95_max: breaches.append("pi_p95")
        if s["dtl_trade_p95"] > cfg.dtl_trade_max: breaches.append("dtl_trade_p95")
        if s["dtl_pos_p95"] > cfg.dtl_pos_max: breaches.append("dtl_pos_p95")
        if s["pi_max"] > cfg.pi_single_name_max: breaches.append("pi_single_name")
        if s["return_retention"] < cfg.rho_min: breaches.append("return_retention")
        if s["r_net_ann"] < cfg.r_net_min: breaches.append("r_net_ann")
        s["hard_pass"] = len(breaches) == 0
        s["breach_reasons"] = breaches

    # Find capacity limit
    capacity_limit = None
    capacity_mult = None
    baseline_ok = any(s["multiplier"] == 1.0 and s["hard_pass"] for s in scenarios)
    for s in scenarios:
        if s["hard_pass"]:
            capacity_limit = s["aum"]
            capacity_mult = s["multiplier"]

    return {
        "scenarios": scenarios,
        "capacity_limit": capacity_limit,
        "capacity_multiplier": capacity_mult,
        "baseline_feasible": baseline_ok,
    }
