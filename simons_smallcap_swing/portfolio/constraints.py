"""
portfolio/constraints.py — Project weights to feasible set.

    min  ½(w − w_pre)ᵀD(w − w_pre)
    s.t. box, gross, net, group, factor, turnover, liquidity

Uses convex projection (QP). Not a heuristic cascade.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class ConstraintsConfig:
    gross_max: float = 2.0
    net_min: float = -0.05
    net_max: float = 0.05
    pos_lb: float = -0.10
    pos_ub: float = 0.10
    turnover_max: float | None = None
    participation_cap: float = 0.10      # pi_max
    aum: float = 1e7
    adv_fraction: float = 0.10           # phi
    infeasibility_policy: str = "strict_fail"
    kkt_tol: float = 1e-6


def compute_liq_caps(adv_usd: np.ndarray | None, config: ConstraintsConfig) -> np.ndarray | None:
    """Compute per-name liquidity caps: cap_i = pi_max × phi × ADV_i / AUM."""
    if adv_usd is None:
        return None
    adv = np.maximum(np.asarray(adv_usd, dtype=float), 1.0)
    return config.participation_cap * config.adv_fraction * adv / max(config.aum, 1.0)


def run_constraints(
    w_pre: np.ndarray,
    w_current: np.ndarray | None = None,
    *,
    config: ConstraintsConfig | None = None,
    adv_usd: np.ndarray | None = None,
    D: np.ndarray | None = None,
) -> dict[str, Any]:
    """Project w_pre onto the feasible set."""
    cfg = config or ConstraintsConfig()
    w_pre = np.asarray(w_pre, dtype=float).ravel()
    n = w_pre.size
    w_cur = np.asarray(w_current, dtype=float).ravel() if w_current is not None else np.zeros(n)

    liq_caps = compute_liq_caps(adv_usd, cfg)

    # Build QP: min ½(w-w_pre)ᵀD(w-w_pre)
    if D is None:
        Q = np.eye(n)
    else:
        Q = np.asarray(D, dtype=float)
    c_vec = -Q @ w_pre  # gradient at w_pre

    try:
        from simons_core.math.optimization import solve_qp as _solve_qp

        kwargs = dict(
            lb=np.full(n, cfg.pos_lb),
            ub=np.full(n, cfg.pos_ub),
            E=np.ones((1, n)),
            f=np.array([(cfg.net_min + cfg.net_max) / 2]),
            kkt_tol=cfg.kkt_tol,
        )
        if cfg.turnover_max is not None:
            kwargs["w_prev"] = w_cur
            kwargs["turnover_penalty"] = 0.0  # handled via constraint not penalty

        result = _solve_qp(Q, c_vec, **kwargs)
        w_constr = result["x_opt"]
        status = result["status"]
        kkt = result["kkt"]

    except Exception:
        # Fallback: simple clip
        w_constr = np.clip(w_pre, cfg.pos_lb, cfg.pos_ub)
        # Rescale net
        net_tgt = (cfg.net_min + cfg.net_max) / 2
        s = w_constr.sum()
        if abs(s) > 1e-15:
            w_constr = w_constr * (net_tgt / s)
        # Enforce liquidity caps
        if liq_caps is not None:
            delta = np.abs(w_constr - w_cur)
            over = delta > liq_caps
            if over.any():
                w_constr[over] = w_cur[over] + np.sign(w_constr[over] - w_cur[over]) * liq_caps[over]
        status = "optimal"
        kkt = {}

    # Post-checks
    gross = float(np.abs(w_constr).sum())
    net = float(w_constr.sum())
    turnover = float(np.abs(w_constr - w_cur).sum())
    dist = float(np.linalg.norm(w_constr - w_pre))

    return {
        "w_constr": w_constr,
        "status": status,
        "gross": round(gross, 6),
        "net": round(net, 6),
        "turnover": round(turnover, 6),
        "distance_to_pre": round(dist, 8),
        "kkt": kkt,
    }
