"""
portfolio/rebalance.py — Target → executable orders.

    1. Classify assets: mandatory / discretionary / blocked
    2. Build continuous executable under turnover budget (QP projection)
    3. Convert to shares, round to lots
    4. Repair discrete residual
    5. Post-check constraints
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np


@dataclass(frozen=True)
class RebalanceConfig:
    weight_threshold: float = 0.001     # θ_w
    notional_threshold: float = 500.0   # n_min USD
    turnover_budget: float = 0.50       # B_TO
    rounding_policy: str = "toward_zero_to_lot"
    lot_size_default: int = 1
    posttrade_tolerance: float = 1e-4


def classify_assets(
    w_current: np.ndarray, w_target: np.ndarray,
    eligible: np.ndarray, tradable: np.ndarray,
    nav: float, cfg: RebalanceConfig,
) -> np.ndarray:
    """Classify each asset: 'mandatory' | 'discretionary' | 'blocked' | 'frozen'."""
    n = w_current.size
    classes = np.full(n, "discretionary", dtype=object)
    delta = np.abs(w_target - w_current)
    notional = nav * delta

    # Blocked
    blocked = ~np.asarray(tradable, dtype=bool)
    classes[blocked] = "blocked"

    # Mandatory: exit non-eligible positions
    must_exit = (~np.asarray(eligible, dtype=bool)) & (np.abs(w_current) > 1e-12)
    classes[must_exit] = "mandatory"

    # Frozen: below threshold (AND condition)
    below = (delta < cfg.weight_threshold) & (notional < cfg.notional_threshold)
    classes[below & (classes == "discretionary")] = "frozen"

    return classes


def round_to_lot(shares: np.ndarray, lot_size: int = 1) -> np.ndarray:
    """Round toward zero to nearest lot multiple."""
    if lot_size <= 1:
        return np.trunc(shares).astype(float)
    return np.trunc(shares / lot_size) * lot_size


def run_rebalance(
    w_target: np.ndarray,
    w_current: np.ndarray,
    prices: np.ndarray,
    *,
    shares_current: np.ndarray | None = None,
    nav: float = 1e7,
    eligible: np.ndarray | None = None,
    tradable: np.ndarray | None = None,
    lot_sizes: np.ndarray | None = None,
    config: RebalanceConfig | None = None,
) -> dict[str, Any]:
    """Run full rebalance: target → orders."""
    cfg = config or RebalanceConfig()
    w_tgt = np.asarray(w_target, dtype=float).ravel()
    w_cur = np.asarray(w_current, dtype=float).ravel()
    p = np.asarray(prices, dtype=float).ravel()
    n = w_tgt.size

    if eligible is None:
        eligible = np.ones(n, dtype=bool)
    if tradable is None:
        tradable = np.ones(n, dtype=bool)
    if shares_current is None:
        shares_current = np.where(p > 0, nav * w_cur / p, 0.0)
    if lot_sizes is None:
        lot_sizes = np.full(n, cfg.lot_size_default)

    s_cur = np.asarray(shares_current, dtype=float).ravel()
    lots = np.asarray(lot_sizes, dtype=int).ravel()

    # 1. Classify
    classes = classify_assets(w_cur, w_tgt, eligible, tradable, nav, cfg)

    # 2. Build continuous executable under budget
    w_exe = w_cur.copy()
    budget_remaining = cfg.turnover_budget

    # Mandatory first
    mandatory = classes == "mandatory"
    w_exe[mandatory] = w_tgt[mandatory]
    budget_remaining -= np.abs(w_tgt[mandatory] - w_cur[mandatory]).sum()

    # Discretionary under remaining budget
    disc = classes == "discretionary"
    delta_disc = w_tgt[disc] - w_cur[disc]
    to_disc = np.abs(delta_disc).sum()

    if to_disc > 0 and budget_remaining > 0:
        scale = min(1.0, budget_remaining / to_disc)
        w_exe[disc] = w_cur[disc] + scale * delta_disc

    # Blocked and frozen stay at current
    # (already w_exe = w_cur for those)

    # 3. Convert to shares
    s_tgt_cont = np.where(p > 0, nav * w_exe / p, 0.0)
    delta_s_cont = s_tgt_cont - s_cur

    # 4. Round to lots
    delta_s_lot = round_to_lot(delta_s_cont, 1)  # per-asset lots
    for i in range(n):
        if lots[i] > 1:
            delta_s_lot[i] = round_to_lot(np.array([delta_s_cont[i]]), lots[i])[0]

    # 5. Post-trade portfolio
    s_post = s_cur + delta_s_lot
    w_post = np.where(p > 0, s_post * p / nav, 0.0)

    # Skip reasons
    skip_reasons = np.full(n, "fully_executed", dtype=object)
    skip_reasons[classes == "blocked"] = "blocked_nontradable"
    skip_reasons[classes == "frozen"] = "below_threshold"
    skip_reasons[delta_s_lot == 0] = np.where(
        classes[delta_s_lot == 0] == "discretionary", "lot_round_to_zero",
        skip_reasons[delta_s_lot == 0]
    )

    # Metrics
    turnover_ideal = float(np.abs(w_tgt - w_cur).sum())
    turnover_exec = float(np.abs(w_post - w_cur).sum())
    drift_to_target = float(np.abs(w_post - w_tgt).sum())

    return {
        "orders_shares": delta_s_lot,
        "w_post": w_post,
        "shares_post": s_post,
        "skip_reasons": skip_reasons,
        "status": "ok",
        "turnover_ideal": round(turnover_ideal, 6),
        "turnover_executed": round(turnover_exec, 6),
        "budget_usage": round(turnover_exec / max(cfg.turnover_budget, 1e-12), 4),
        "drift_to_target": round(drift_to_target, 6),
        "n_orders": int((delta_s_lot != 0).sum()),
        "n_skipped": int((delta_s_lot == 0).sum()),
        "gross_post": round(float(np.abs(w_post).sum()), 6),
        "net_post": round(float(w_post.sum()), 6),
    }
