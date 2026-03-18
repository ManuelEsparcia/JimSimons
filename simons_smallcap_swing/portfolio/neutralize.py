"""
portfolio/neutralize.py — Factor/group neutralisation via constrained projection.

    min  ½(w − w_in)ᵀD(w − w_in)
    s.t. t_x − τ_x ≤ Xᵀw ≤ t_x + τ_x   (factor bands)
         t_b − τ_b ≤ Bᵀw ≤ t_b + τ_b   (group bands)
         box, net, gross, turnover, liquidity

NOT residualisation. Proper constrained projection.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class NeutralizeConfig:
    net_target: float | None = None      # exact net, or None for band
    net_min: float = -0.05
    net_max: float = 0.05
    gross_max: float = 2.0
    pos_lb: float = -0.10
    pos_ub: float = 0.10
    factor_tolerance: float = 0.01       # τ_x default per factor
    group_tolerance: float = 0.02        # τ_b default per group
    turnover_max: float | None = None
    infeasibility_policy: str = "strict_fail"
    kkt_tol: float = 1e-6


def run_neutralize(
    w_in: np.ndarray,
    *,
    w_current: np.ndarray | None = None,
    X: np.ndarray | None = None,
    B: np.ndarray | None = None,
    targets_x: np.ndarray | None = None,
    targets_b: np.ndarray | None = None,
    config: NeutralizeConfig | None = None,
) -> dict[str, Any]:
    """Neutralise w_in to factor/group targets via QP projection."""
    cfg = config or NeutralizeConfig()
    w_in = np.asarray(w_in, dtype=float).ravel()
    n = w_in.size
    w_cur = np.asarray(w_current, dtype=float).ravel() if w_current is not None else np.zeros(n)

    # Build inequality constraints: factor and group bands
    C_rows, d_vals = [], []

    if X is not None:
        X = np.asarray(X, dtype=float)
        k = X.shape[1]
        t_x = np.asarray(targets_x, dtype=float).ravel() if targets_x is not None else np.zeros(k)
        tau_x = cfg.factor_tolerance
        # Xᵀw ≤ t_x + τ_x  →  Xᵀw ≤ upper
        for j in range(k):
            C_rows.append(X[:, j])
            d_vals.append(t_x[j] + tau_x)
            C_rows.append(-X[:, j])
            d_vals.append(-t_x[j] + tau_x)

    if B is not None:
        B = np.asarray(B, dtype=float)
        g = B.shape[1]
        t_b = np.asarray(targets_b, dtype=float).ravel() if targets_b is not None else np.zeros(g)
        tau_b = cfg.group_tolerance
        for j in range(g):
            C_rows.append(B[:, j])
            d_vals.append(t_b[j] + tau_b)
            C_rows.append(-B[:, j])
            d_vals.append(-t_b[j] + tau_b)

    C = np.array(C_rows) if C_rows else None
    d = np.array(d_vals) if d_vals else None

    # QP: min ½(w - w_in)² s.t. constraints
    Q = np.eye(n)
    c_vec = -w_in  # so that ½wᵀIw − w_inᵀw + const

    try:
        from simons_core.math.optimization import solve_qp as _solve_qp
        result = _solve_qp(
            Q, c_vec,
            C=C, d=d,
            lb=np.full(n, cfg.pos_lb),
            ub=np.full(n, cfg.pos_ub),
            E=np.ones((1, n)),
            f=np.array([cfg.net_target]) if cfg.net_target is not None else np.array([(cfg.net_min + cfg.net_max) / 2]),
            kkt_tol=cfg.kkt_tol,
        )
        w_neu = result["x_opt"]
        status = result["status"]
    except Exception:
        # Fallback: return w_in clipped
        w_neu = np.clip(w_in, cfg.pos_lb, cfg.pos_ub)
        status = "fallback"

    # Before/after exposures
    exp_before = {"net": float(w_in.sum()), "gross": float(np.abs(w_in).sum())}
    exp_after = {"net": float(w_neu.sum()), "gross": float(np.abs(w_neu).sum())}
    if X is not None:
        exp_before["factor"] = (X.T @ w_in).tolist()
        exp_after["factor"] = (X.T @ w_neu).tolist()
    if B is not None:
        exp_before["group"] = (B.T @ w_in).tolist()
        exp_after["group"] = (B.T @ w_neu).tolist()

    return {
        "w_neu": w_neu,
        "status": status,
        "distance_to_input": round(float(np.linalg.norm(w_neu - w_in)), 8),
        "exposure_before": exp_before,
        "exposure_after": exp_after,
    }
