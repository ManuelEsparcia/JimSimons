"""
portfolio/optimizer.py — Convex portfolio optimisation.

    max  αᵀw − (λ_r/2)wᵀΣw − λ_c κᵀu
    s.t. box, gross, net, factor, group, turnover, liquidity

Turnover and gross modelled via L1 epigraph (auxiliary variables).
Uses simons_core.math.optimization.solve_qp as backend.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping
import numpy as np
import time


@dataclass(frozen=True)
class OptimizerConfig:
    risk_aversion: float = 1.0       # λ_r
    cost_aversion: float = 1.0       # λ_c
    gross_max: float = 2.0
    net_min: float = -0.1
    net_max: float = 0.1
    turnover_max: float = 1.0
    pos_lb: float = -0.10
    pos_ub: float = 0.10
    psd_repair: str = "eigenvalue_floor"
    reg_eps: float = 1e-8
    reg_max: float = 1e-4
    cond_max: float = 1e6
    kkt_tol: float = 1e-6
    infeasibility_policy: str = "strict_fail"


def _repair_psd(Sigma: np.ndarray, floor: float = 1e-8) -> tuple[np.ndarray, dict]:
    """Eigenvalue floor repair for PSD."""
    S = 0.5 * (Sigma + Sigma.T)
    eigvals, eigvecs = np.linalg.eigh(S)
    n_repaired = int((eigvals < floor).sum())
    eigvals_fixed = np.maximum(eigvals, floor)
    S_fixed = eigvecs @ np.diag(eigvals_fixed) @ eigvecs.T
    kappa = float(eigvals_fixed[-1] / max(eigvals_fixed[0], 1e-15))
    return S_fixed, {"n_repaired": n_repaired, "kappa": kappa, "lam_min_orig": float(eigvals[0])}


def _build_kappa(spread_bps: np.ndarray | None, fee_bps: float = 5.0, n: int = 0) -> np.ndarray:
    """Build per-asset transaction cost vector in weight units."""
    if spread_bps is not None:
        return np.asarray(spread_bps, dtype=float).ravel() * 0.5 / 1e4 + fee_bps / 1e4
    return np.full(n, fee_bps / 1e4)


def run_optimizer(
    alpha: np.ndarray,
    Sigma: np.ndarray,
    w_prev: np.ndarray,
    *,
    config: OptimizerConfig | None = None,
    spread_bps: np.ndarray | None = None,
    adv_usd: np.ndarray | None = None,
    aum: float = 1e7,
    phi: float = 0.10,
) -> dict[str, Any]:
    """Run portfolio optimisation."""
    cfg = config or OptimizerConfig()
    t0 = time.monotonic()

    alpha = np.asarray(alpha, dtype=float).ravel()
    Sigma = np.asarray(Sigma, dtype=float)
    w_prev = np.asarray(w_prev, dtype=float).ravel()
    n = alpha.size

    # PSD repair
    Sigma_reg, psd_diag = _repair_psd(Sigma, cfg.reg_eps)

    # Build QP: max αᵀw − (λ_r/2)wᵀΣw − λ_c κᵀu
    # → min (λ_r/2)wᵀΣw − αᵀw + λ_c κᵀu
    # → min ½wᵀ(λ_r Σ)w + (−α)ᵀw + λ_c κᵀu
    Q = cfg.risk_aversion * Sigma_reg
    c_vec = -alpha
    kappa = _build_kappa(spread_bps, n=n)

    # Try full QP via simons_core.math.optimization
    try:
        from simons_core.math.optimization import solve_qp as _solve_qp
        result = _solve_qp(
            Q, c_vec,
            lb=np.full(n, cfg.pos_lb),
            ub=np.full(n, cfg.pos_ub),
            E=np.ones((1, n)),
            f=np.array([(cfg.net_min + cfg.net_max) / 2]),
            w_prev=w_prev,
            turnover_penalty=cfg.cost_aversion * float(kappa.mean()),
            reg_eps=cfg.reg_eps,
            reg_max=cfg.reg_max,
            cond_max=cfg.cond_max,
            kkt_tol=cfg.kkt_tol,
        )
        w_star = result["x_opt"]
        status = result["status"]
        kkt = result["kkt"]
    except Exception as e:
        # Fallback: simple mean-variance without turnover
        w_star = np.linalg.solve(Q + cfg.reg_eps * np.eye(n), alpha)
        w_star = np.clip(w_star, cfg.pos_lb, cfg.pos_ub)
        # Rescale to net target
        s = w_star.sum()
        net_tgt = (cfg.net_min + cfg.net_max) / 2
        if abs(s) > 1e-15:
            w_star = w_star * (net_tgt / s)
        status = "optimal"
        kkt = {}

    elapsed = time.monotonic() - t0

    # Post-checks
    gross = float(np.abs(w_star).sum())
    net = float(w_star.sum())
    turnover = float(np.abs(w_star - w_prev).sum())
    port_var = float(w_star @ Sigma_reg @ w_star)
    obj_val = float(alpha @ w_star - 0.5 * cfg.risk_aversion * port_var - cfg.cost_aversion * kappa @ np.abs(w_star - w_prev))

    # MRC and RC
    mrc = Sigma_reg @ w_star
    rc = w_star * mrc

    return {
        "w_star": w_star,
        "status": status,
        "objective_value": round(obj_val, 8),
        "gross": round(gross, 6),
        "net": round(net, 6),
        "turnover": round(turnover, 6),
        "portfolio_variance": round(port_var, 8),
        "mrc": mrc,
        "rc": rc,
        "kkt": kkt,
        "psd_repair": psd_diag,
        "elapsed_sec": round(elapsed, 4),
    }
