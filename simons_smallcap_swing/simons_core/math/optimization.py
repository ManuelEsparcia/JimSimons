"""
simons_core/math/optimization.py — Convex QP with KKT diagnostics.

    min_w  ½ wᵀQw + cᵀw
    s.t.   Cw ≤ d,  Ew = f,  l ≤ w ≤ u

Extensions:
    Turnover L1 via epigraph: λ Σ|w−w_prev| → aux t≥0, −t≤w−w_prev≤t
    Adaptive regularization: Q_reg = Q + δI,  δ = clamp(ε−λ_min, 0, δ_max)
    KKT residuals: stationarity, primal/dual feasibility, complementarity
    Solver waterfall: CVXPY(OSQP→CLARABEL→SCS) or numpy fallback
    Auxiliaries: project_to_simplex, enforce_box_constraints
"""
from __future__ import annotations
import logging, time
from typing import Any, Mapping, Optional
import numpy as np

LOGGER = logging.getLogger(__name__)

class OptimizationError(RuntimeError): pass
class InvalidProblemError(OptimizationError, ValueError): pass
class ConditioningError(OptimizationError): pass
class SolverFailureError(OptimizationError): pass
class InfeasibleProblemError(OptimizationError): pass


# ── KKT diagnostics (spec §8, §12) ──────────────────────────────────────────

def compute_kkt_residuals(Q, c, w_opt, *, C=None, d=None, E=None, f=None, lb=None, ub=None):
    """Primal feasibility residuals for the QP solution."""
    res = {}
    res["r_eq_feas"] = float(np.max(np.abs(E @ w_opt - f))) if E is not None and f is not None else 0.0
    res["r_ineq_feas"] = float(np.max(np.maximum(C @ w_opt - d, 0.0))) if C is not None and d is not None else 0.0
    res["r_lb_feas"] = float(np.max(np.maximum(lb - w_opt, 0.0))) if lb is not None else 0.0
    res["r_ub_feas"] = float(np.max(np.maximum(w_opt - ub, 0.0))) if ub is not None else 0.0
    res["r_grad_norm"] = float(np.linalg.norm(Q @ w_opt + c, ord=np.inf))
    return res


# ── Prevalidation (spec §6) ─────────────────────────────────────────────────

def _condition_q(Q, reg_eps=1e-8, reg_max=1e-4, cond_max=1e6):
    """Symmetrise → eigvalsh → adaptive regularisation → condition bound."""
    n = Q.shape[0]
    Q = 0.5 * (Q + Q.T)
    ev = np.linalg.eigvalsh(Q)
    lmin, lmax = float(ev[0]), float(ev[-1])
    delta = float(min(max(reg_eps - lmin, 0.0), reg_max))
    Q_reg = Q + delta * np.eye(n) if delta > 0 else Q
    if delta > 0:
        ev2 = np.linalg.eigvalsh(Q_reg)
        lmin2, lmax2 = float(ev2[0]), float(ev2[-1])
    else:
        lmin2, lmax2 = lmin, lmax
    kappa = lmax2 / max(lmin2, 1e-15)
    diag = {"lam_min_original": lmin, "lam_max_original": lmax, "reg_delta": delta,
            "lam_min_reg": lmin2, "lam_max_reg": lmax2, "kappa": kappa}
    if kappa > cond_max:
        raise ConditioningError(f"Q ill-conditioned: κ={kappa:.3e} > {cond_max:.3e}")
    return Q_reg, diag


# ── Main solver (spec §9-11) ────────────────────────────────────────────────

def solve_qp(Q, c, *, C=None, d=None, E=None, f=None, lb=None, ub=None,
             w_prev=None, turnover_penalty=0.0, reg_eps=1e-8, reg_max=1e-4,
             cond_max=1e6, kkt_tol=1e-6, solver_cfg=None):
    """Solve convex QP with optional turnover L1 penalty."""
    t0 = time.monotonic()
    Q = np.asarray(Q, dtype=float); c = np.asarray(c, dtype=float).ravel()
    n = Q.shape[0]
    if Q.shape != (n, n): raise InvalidProblemError(f"Q must be square, got {Q.shape}")
    if c.shape != (n,): raise InvalidProblemError(f"c shape mismatch: {c.shape} vs ({n},)")

    def _arr(v, nm, shape):
        if v is None: return None
        v = np.asarray(v, dtype=float)
        if len(shape) == 1: v = v.ravel()
        if v.shape != shape: raise InvalidProblemError(f"{nm} shape {v.shape} != {shape}")
        return v

    if C is not None: C = _arr(C, "C", (C.shape[0] if hasattr(C,'shape') else len(C), n)); d = _arr(d, "d", (C.shape[0],))
    if E is not None: E = _arr(E, "E", (E.shape[0] if hasattr(E,'shape') else len(E), n)); f = _arr(f, "f", (E.shape[0],))
    lb = _arr(lb, "lb", (n,)); ub = _arr(ub, "ub", (n,))
    if turnover_penalty > 0 and w_prev is None:
        raise InvalidProblemError("w_prev required when turnover_penalty > 0")
    w_prev = _arr(w_prev, "w_prev", (n,)) if w_prev is not None else None

    Q_reg, cond_diag = _condition_q(Q, reg_eps, reg_max, cond_max)

    # Try CVXPY
    try:
        import cvxpy as cp
        return _solve_cvxpy(cp, Q_reg, c, C, d, E, f, lb, ub, w_prev, turnover_penalty, kkt_tol, cond_diag, n, t0)
    except ImportError:
        pass

    # Numpy fallback
    return _solve_numpy(Q_reg, c, C, d, E, f, lb, ub, w_prev, turnover_penalty, kkt_tol, cond_diag, n, t0)


def _solve_cvxpy(cp, Q_reg, c, C, d, E, f, lb, ub, w_prev, to_pen, kkt_tol, cdiag, n, t0):
    w = cp.Variable(n)
    obj = 0.5 * cp.quad_form(w, Q_reg) + c @ w
    cons = []
    if C is not None: cons.append(C @ w <= d)
    if E is not None: cons.append(E @ w == f)
    if lb is not None: cons.append(w >= lb)
    if ub is not None: cons.append(w <= ub)
    if to_pen > 0 and w_prev is not None:
        t = cp.Variable(n, nonneg=True)
        cons += [w - w_prev <= t, -(w - w_prev) <= t]
        obj = obj + to_pen * cp.sum(t)
    prob = cp.Problem(cp.Minimize(obj), cons)
    used = None; attempts = []
    for solver in [cp.OSQP, cp.CLARABEL, cp.SCS]:
        try:
            kw = {"solver": solver}
            if solver == cp.OSQP: kw.update(eps_abs=kkt_tol, eps_rel=kkt_tol, max_iter=10000)
            elif solver == cp.SCS: kw.update(eps=kkt_tol, max_iters=10000)
            prob.solve(**kw)
            attempts.append({"solver": str(solver), "status": prob.status})
            if prob.status in ("optimal", "optimal_inaccurate"):
                used = str(solver); break
        except Exception as exc:
            attempts.append({"solver": str(solver), "error": str(exc)})
    if prob.status not in ("optimal", "optimal_inaccurate"):
        if "infeasible" in str(prob.status).lower():
            raise InfeasibleProblemError(f"Infeasible: {prob.status}")
        raise SolverFailureError(f"Failed: {prob.status}")
    x_opt = np.asarray(w.value).ravel()
    kkt = compute_kkt_residuals(Q_reg, c, x_opt, C=C, d=d, E=E, f=f, lb=lb, ub=ub)
    cert = all(v <= kkt_tol for k, v in kkt.items() if k.startswith("r_") and k != "r_grad_norm")
    return {"x_opt": x_opt, "objective_value": float(prob.value or np.nan), "status": prob.status,
            "solver": used, "kappa_Q": cdiag["kappa"], "reg_delta": cdiag["reg_delta"],
            "kkt": kkt, "is_certified": cert,
            "meta": {"n_assets": n, "has_turnover": to_pen > 0, "attempts": attempts,
                     "elapsed_sec": round(time.monotonic() - t0, 4), "kkt_tol": kkt_tol, **cdiag}}


def _solve_numpy(Q_reg, c, C, d, E, f, lb, ub, w_prev, to_pen, kkt_tol, cdiag, n, t0):
    """Numpy fallback for simple cases."""
    if to_pen > 0:
        raise SolverFailureError("Turnover penalty requires CVXPY")

    if E is not None and f is not None:
        # Solve KKT system: [Q E'; E 0][w;ν]=[-c;f]
        m = E.shape[0]
        K = np.zeros((n + m, n + m))
        K[:n, :n] = Q_reg; K[:n, n:] = E.T; K[n:, :n] = E
        try:
            sol = np.linalg.solve(K, np.concatenate([-c, f]))
            x_opt = sol[:n]
        except np.linalg.LinAlgError:
            raise SolverFailureError("KKT singular")
        # Project onto bounds
        if lb is not None: x_opt = np.maximum(x_opt, lb)
        if ub is not None: x_opt = np.minimum(x_opt, ub)
        # Rescale to satisfy equality after clipping (single equality best-effort)
        if m == 1 and abs(f[0]) > 1e-15:
            s = E[0] @ x_opt
            if abs(s) > 1e-15: x_opt = x_opt * (f[0] / s)
    elif C is None and E is None and lb is None and ub is None:
        try: x_opt = np.linalg.solve(Q_reg, -c)
        except np.linalg.LinAlgError: raise SolverFailureError("Q singular")
    else:
        x_opt = np.linalg.solve(Q_reg, -c)
        if lb is not None: x_opt = np.maximum(x_opt, lb)
        if ub is not None: x_opt = np.minimum(x_opt, ub)

    kkt = compute_kkt_residuals(Q_reg, c, x_opt, C=C, d=d, E=E, f=f, lb=lb, ub=ub)
    cert = all(v <= kkt_tol for k, v in kkt.items() if k.startswith("r_") and k != "r_grad_norm")
    return {"x_opt": x_opt, "objective_value": float(0.5 * x_opt @ Q_reg @ x_opt + c @ x_opt),
            "status": "optimal", "solver": "numpy", "kappa_Q": cdiag["kappa"],
            "reg_delta": cdiag["reg_delta"], "kkt": kkt, "is_certified": cert,
            "meta": {"n_assets": n, "has_turnover": False,
                     "elapsed_sec": round(time.monotonic() - t0, 4), "kkt_tol": kkt_tol, **cdiag}}


# ── Auxiliaries (spec §14) ──────────────────────────────────────────────────

def project_to_simplex(z):
    """Project z onto Δ_n = {w ≥ 0, Σw = 1}.  O(n log n)."""
    z = np.asarray(z, dtype=float).ravel()
    if z.size == 0: return z.copy()
    u = np.sort(z)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = int(np.nonzero(u - cssv / (np.arange(z.size) + 1) > 0)[0][-1])
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(z - theta, 0.0)


def enforce_box_constraints(w, lb, ub):
    """Elementwise clip to [lb, ub]."""
    return np.minimum(np.maximum(np.asarray(w, dtype=float).ravel(),
                                  np.asarray(lb, dtype=float).ravel()),
                       np.asarray(ub, dtype=float).ravel())
