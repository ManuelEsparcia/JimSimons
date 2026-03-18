"""
execution/turnover_control.py — Turnover budgeting and throttling.

    1. Compute raw turnover
    2. Apply no-trade bands (with override for mandatory/risk-reducing)
    3. Classify trades: hard_constraint → risk_reducing → alpha → residual
    4. Allocate budget tier by tier with proportional throttle within tier
    5. Preserve constraints if enabled
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class TurnoverConfig:
    budget_2way: float = 0.50        # B_t
    band_default: float = 0.002      # no-trade band threshold
    band_override_priority: bool = True  # allow mandatory/risk trades through band
    preserve_constraints: bool = True
    stagger_enabled: bool = False
    max_stagger_days: int = 3


def classify_trades(
    delta_w: np.ndarray,
    hard_constraint: np.ndarray | None = None,
    risk_reducing: np.ndarray | None = None,
    alpha_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Assign each trade to a tier: 0=hard, 1=risk, 2=alpha, 3=residual."""
    n = delta_w.size
    tiers = np.full(n, 3, dtype=int)  # default residual
    if alpha_scores is not None:
        tiers[np.abs(np.asarray(alpha_scores)) > 1e-10] = 2
    if risk_reducing is not None:
        tiers[np.asarray(risk_reducing, dtype=bool)] = 1
    if hard_constraint is not None:
        tiers[np.asarray(hard_constraint, dtype=bool)] = 0
    return tiers


def apply_bands(
    delta_w: np.ndarray,
    tiers: np.ndarray,
    band: float,
    override: bool = True,
) -> np.ndarray:
    """Apply no-trade band. Mandatory/risk trades bypass if override=True."""
    out = delta_w.copy()
    small = np.abs(out) < band
    if override:
        # Only zero out residual and alpha trades below band
        killable = small & (tiers >= 2)
        out[killable] = 0.0
    else:
        out[small] = 0.0
    return out


def throttle_proportional(
    delta_w: np.ndarray,
    tiers: np.ndarray,
    budget: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Allocate turnover budget tier by tier, proportional within each tier."""
    out = np.zeros_like(delta_w)
    budget_remaining = budget
    tier_info = {}

    for tier in range(4):  # 0=hard, 1=risk, 2=alpha, 3=residual
        mask = tiers == tier
        if not mask.any():
            continue
        tier_to = np.abs(delta_w[mask]).sum()
        if tier_to <= 0:
            continue

        if tier_to <= budget_remaining:
            out[mask] = delta_w[mask]
            budget_remaining -= tier_to
            tier_info[f"tier_{tier}"] = {"to": round(float(tier_to), 6), "scale": 1.0}
        else:
            scale = budget_remaining / tier_to
            out[mask] = delta_w[mask] * scale
            tier_info[f"tier_{tier}"] = {"to": round(float(tier_to), 6), "scale": round(scale, 4)}
            budget_remaining = 0.0
            break  # no budget for lower tiers

    return out, tier_info


def run_turnover_control(
    w_current: np.ndarray,
    w_target: np.ndarray,
    *,
    hard_constraint: np.ndarray | None = None,
    risk_reducing: np.ndarray | None = None,
    alpha_scores: np.ndarray | None = None,
    config: TurnoverConfig | None = None,
) -> dict[str, Any]:
    """Full turnover control pipeline."""
    cfg = config or TurnoverConfig()
    w_cur = np.asarray(w_current, dtype=float).ravel()
    w_tgt = np.asarray(w_target, dtype=float).ravel()
    n = w_cur.size

    delta_raw = w_tgt - w_cur
    to_raw_2way = float(np.abs(delta_raw).sum())

    # Classify
    tiers = classify_trades(delta_raw, hard_constraint, risk_reducing, alpha_scores)

    # Apply bands
    delta_banded = apply_bands(delta_raw, tiers, cfg.band_default, cfg.band_override_priority)
    to_banded_2way = float(np.abs(delta_banded).sum())
    n_band_clipped = int(((delta_raw != 0) & (delta_banded == 0)).sum()) if to_raw_2way > 0 else 0

    # Throttle
    delta_ctrl, tier_info = throttle_proportional(delta_banded, tiers, cfg.budget_2way)
    w_ctrl = w_cur + delta_ctrl

    to_ctrl_2way = float(np.abs(delta_ctrl).sum())

    # Deferred
    delta_deferred = delta_banded - delta_ctrl
    to_deferred = float(np.abs(delta_deferred).sum())

    # Alpha retention proxy
    if alpha_scores is not None:
        a = np.asarray(alpha_scores, dtype=float).ravel()
        u_raw = float((a * delta_raw).sum())
        u_ctrl = float((a * delta_ctrl).sum())
        retention = u_ctrl / u_raw if abs(u_raw) > 1e-12 else np.nan
    else:
        retention = np.nan

    return {
        "w_ctrl": w_ctrl,
        "delta_ctrl": delta_ctrl,
        "status": "ok",
        "turnover_raw_2way": round(to_raw_2way, 6),
        "turnover_banded_2way": round(to_banded_2way, 6),
        "turnover_ctrl_2way": round(to_ctrl_2way, 6),
        "turnover_deferred": round(to_deferred, 6),
        "budget_usage": round(to_ctrl_2way / max(cfg.budget_2way, 1e-12), 4),
        "n_band_clipped": n_band_clipped,
        "alpha_retention_proxy": round(retention, 4) if np.isfinite(retention) else None,
        "tier_info": tier_info,
    }
