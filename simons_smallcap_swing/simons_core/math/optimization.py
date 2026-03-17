
from __future__ import annotations

"""
simons_core.math.optimization
=============================

Institutional convex optimization primitives for portfolio construction and
allocation problems.

This module is intentionally *contract-first* and *diagnostic-heavy*:

- canonical convex QP semantics;
- explicit PSD symmetrization/regularization of the Hessian;
- conditioning checks before solver invocation;
- correct L1 turnover modeling via epigraph variables;
- controlled solver waterfall with persisted attempt history;
- KKT-style certification diagnostics rather than opaque "solver said optimal";
- deterministic geometric helpers such as simplex projection and box clipping.

Primary problem family
----------------------
The canonical form implemented here is::

    minimize_w   0.5 * w.T @ Q @ w + c.T @ w

    subject to   C @ w <= d
                 E @ w  = f
                 lb <= w <= ub

Optionally, turnover is modeled exactly (within convex epigraph semantics) as::

    turnover_penalty * ||w - w_prev||_1

via auxiliary variables ``t >= 0`` and constraints::

    w - w_prev - t <= 0
   -w + w_prev - t <= 0
              -t <= 0

Notes
-----
- The preferred backend is CVXPY with a controlled waterfall over OSQP,
  CLARABEL and SCS.
- If CVXPY is unavailable at runtime, the module still imports cleanly, but
  ``solve_qp`` fails closed with a clear error.
- Diagnostic routines and geometric utilities remain usable without CVXPY.
"""

from dataclasses import dataclass, field
import argparse
import json
import math
from time import perf_counter
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence, TypeAlias

import numpy as np

try:  # pragma: no cover - optional dependency branch
    import cvxpy as cp  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency branch
    cp = None  # type: ignore[assignment]


__all__ = [
    "DEFAULT_REG_EPS",
    "DEFAULT_REG_MAX",
    "DEFAULT_COND_MAX",
    "DEFAULT_KKT_TOL",
    "DEFAULT_SOLVER_ORDER",
    "OptimizationError",
    "InvalidProblemError",
    "ConditioningError",
    "SolverUnavailableError",
    "SolverFailureError",
    "InfeasibleProblemError",
    "UnboundedProblemError",
    "CertificationError",
    "SolverConfig",
    "SolverAttempt",
    "ConditioningReport",
    "KKTDiagnostics",
    "QPSolution",
    "TurnoverEpigraph",
    "available_solvers",
    "validate_qp_inputs",
    "compute_kkt_residuals",
    "project_to_simplex",
    "enforce_box_constraints",
    "build_turnover_epigraph",
    "solve_qp_result",
    "solve_qp",
]


ArrayLike: TypeAlias = Sequence[float] | np.ndarray

DEFAULT_REG_EPS: Final[float] = 1e-8
DEFAULT_REG_MAX: Final[float] = 1e-4
DEFAULT_COND_MAX: Final[float] = 1e6
DEFAULT_KKT_TOL: Final[float] = 1e-8
DEFAULT_SOLVER_ORDER: Final[tuple[str, ...]] = ("OSQP", "CLARABEL", "SCS")
_DEFAULT_EIGEN_EPS: Final[float] = 1e-12


class OptimizationError(RuntimeError):
    """Base class for optimization-layer failures."""


class InvalidProblemError(OptimizationError, ValueError):
    """Raised when shapes, finiteness or semantics of the problem are invalid."""


class ConditioningError(OptimizationError):
    """Raised when the quadratic form is not numerically usable/certifiable."""


class SolverUnavailableError(OptimizationError):
    """Raised when the requested optimization backend is unavailable."""


class SolverFailureError(OptimizationError):
    """Raised when all solver attempts fail or return unusable statuses."""


class InfeasibleProblemError(OptimizationError):
    """Raised when the problem is certified infeasible."""


class UnboundedProblemError(OptimizationError):
    """Raised when the problem is certified unbounded."""


class CertificationError(OptimizationError):
    """Raised when a solution exists but cannot be numerically certified."""


@dataclass(frozen=True)
class SolverConfig:
    """
    Configuration for the controlled solver waterfall.

    Parameters
    ----------
    solver_order:
        Ordered solver names to try. Supported names depend on the runtime
        backend but the institutional default is ``OSQP -> CLARABEL -> SCS``.
    eps_abs / eps_rel:
        Absolute/relative solver tolerances.
    max_iters:
        Iteration cap passed through to solver-specific options when supported.
    verbose:
        Whether the backend solver should log.
    accept_inaccurate:
        Whether ``*_inaccurate`` statuses are accepted as usable.
    solver_options:
        Per-solver extra options, keyed by solver name.
    """

    solver_order: tuple[str, ...] = DEFAULT_SOLVER_ORDER
    eps_abs: float = DEFAULT_KKT_TOL
    eps_rel: float = DEFAULT_KKT_TOL
    max_iters: int = 100_000
    verbose: bool = False
    accept_inaccurate: bool = True
    solver_options: Mapping[str, Mapping[str, Any]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @classmethod
    def from_any(cls, value: Mapping[str, Any] | "SolverConfig" | None) -> "SolverConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        solver_options = value.get("solver_options", {})
        return cls(
            solver_order=tuple(str(s).upper() for s in value.get("solver_order", DEFAULT_SOLVER_ORDER)),
            eps_abs=float(value.get("eps_abs", DEFAULT_KKT_TOL)),
            eps_rel=float(value.get("eps_rel", DEFAULT_KKT_TOL)),
            max_iters=int(value.get("max_iters", 100_000)),
            verbose=bool(value.get("verbose", False)),
            accept_inaccurate=bool(value.get("accept_inaccurate", True)),
            solver_options=MappingProxyType(
                {
                    str(k).upper(): dict(v)
                    for k, v in dict(solver_options).items()
                }
            ),
        )


@dataclass(frozen=True)
class SolverAttempt:
    """Single solver attempt inside the waterfall."""

    solver: str
    status: str
    solve_time_sec: float
    options: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    error_message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "status": self.status,
            "solve_time_sec": float(self.solve_time_sec),
            "options": dict(self.options),
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class ConditioningReport:
    """Spectral/numerical diagnostics for the Hessian."""

    n: int
    lam_min_raw: float
    lam_max_raw: float
    reg_delta: float
    lam_min_reg: float
    lam_max_reg: float
    kappa_q: float
    is_psd_raw: bool
    is_convex_usable: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": int(self.n),
            "lam_min_raw": float(self.lam_min_raw),
            "lam_max_raw": float(self.lam_max_raw),
            "reg_delta": float(self.reg_delta),
            "lam_min_reg": float(self.lam_min_reg),
            "lam_max_reg": float(self.lam_max_reg),
            "kappa_q": float(self.kappa_q),
            "is_psd_raw": bool(self.is_psd_raw),
            "is_convex_usable": bool(self.is_convex_usable),
        }


@dataclass(frozen=True)
class KKTDiagnostics:
    """
    Quantitative KKT-style residual report.

    A residual may be ``math.inf`` when the necessary dual quantity is not
    available from the backend; this intentionally prevents silent
    certification.
    """

    r_eq_feas: float
    r_ineq_feas: float
    r_lb_feas: float
    r_ub_feas: float
    r_dual_feas_ineq: float
    r_dual_feas_lb: float
    r_dual_feas_ub: float
    r_stationarity: float
    r_complementarity_ineq: float
    r_complementarity_lb: float
    r_complementarity_ub: float
    r_turnover_feas_pos: float = 0.0
    r_turnover_feas_neg: float = 0.0
    r_turnover_feas_t_nonneg: float = 0.0
    r_turnover_dual_feas_pos: float = 0.0
    r_turnover_dual_feas_neg: float = 0.0
    r_turnover_dual_feas_t_nonneg: float = 0.0
    r_turnover_complementarity_pos: float = 0.0
    r_turnover_complementarity_neg: float = 0.0
    r_turnover_complementarity_t_nonneg: float = 0.0
    r_turnover_stationarity_t: float = 0.0
    kkt_tol: float = DEFAULT_KKT_TOL

    @property
    def max_residual(self) -> float:
        values = [
            self.r_eq_feas,
            self.r_ineq_feas,
            self.r_lb_feas,
            self.r_ub_feas,
            self.r_dual_feas_ineq,
            self.r_dual_feas_lb,
            self.r_dual_feas_ub,
            self.r_stationarity,
            self.r_complementarity_ineq,
            self.r_complementarity_lb,
            self.r_complementarity_ub,
            self.r_turnover_feas_pos,
            self.r_turnover_feas_neg,
            self.r_turnover_feas_t_nonneg,
            self.r_turnover_dual_feas_pos,
            self.r_turnover_dual_feas_neg,
            self.r_turnover_dual_feas_t_nonneg,
            self.r_turnover_complementarity_pos,
            self.r_turnover_complementarity_neg,
            self.r_turnover_complementarity_t_nonneg,
            self.r_turnover_stationarity_t,
        ]
        return float(max(values))

    @property
    def is_certified(self) -> bool:
        values = [
            self.r_eq_feas,
            self.r_ineq_feas,
            self.r_lb_feas,
            self.r_ub_feas,
            self.r_dual_feas_ineq,
            self.r_dual_feas_lb,
            self.r_dual_feas_ub,
            self.r_stationarity,
            self.r_complementarity_ineq,
            self.r_complementarity_lb,
            self.r_complementarity_ub,
            self.r_turnover_feas_pos,
            self.r_turnover_feas_neg,
            self.r_turnover_feas_t_nonneg,
            self.r_turnover_dual_feas_pos,
            self.r_turnover_dual_feas_neg,
            self.r_turnover_dual_feas_t_nonneg,
            self.r_turnover_complementarity_pos,
            self.r_turnover_complementarity_neg,
            self.r_turnover_complementarity_t_nonneg,
            self.r_turnover_stationarity_t,
        ]
        return all(np.isfinite(v) and v <= self.kkt_tol for v in values)

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "r_eq_feas": float(self.r_eq_feas),
            "r_ineq_feas": float(self.r_ineq_feas),
            "r_lb_feas": float(self.r_lb_feas),
            "r_ub_feas": float(self.r_ub_feas),
            "r_dual_feas_ineq": float(self.r_dual_feas_ineq),
            "r_dual_feas_lb": float(self.r_dual_feas_lb),
            "r_dual_feas_ub": float(self.r_dual_feas_ub),
            "r_stationarity": float(self.r_stationarity),
            "r_complementarity_ineq": float(self.r_complementarity_ineq),
            "r_complementarity_lb": float(self.r_complementarity_lb),
            "r_complementarity_ub": float(self.r_complementarity_ub),
            "r_turnover_feas_pos": float(self.r_turnover_feas_pos),
            "r_turnover_feas_neg": float(self.r_turnover_feas_neg),
            "r_turnover_feas_t_nonneg": float(self.r_turnover_feas_t_nonneg),
            "r_turnover_dual_feas_pos": float(self.r_turnover_dual_feas_pos),
            "r_turnover_dual_feas_neg": float(self.r_turnover_dual_feas_neg),
            "r_turnover_dual_feas_t_nonneg": float(self.r_turnover_dual_feas_t_nonneg),
            "r_turnover_complementarity_pos": float(self.r_turnover_complementarity_pos),
            "r_turnover_complementarity_neg": float(self.r_turnover_complementarity_neg),
            "r_turnover_complementarity_t_nonneg": float(self.r_turnover_complementarity_t_nonneg),
            "r_turnover_stationarity_t": float(self.r_turnover_stationarity_t),
            "max_residual": float(self.max_residual),
            "kkt_tol": float(self.kkt_tol),
            "is_certified": bool(self.is_certified),
        }


@dataclass(frozen=True)
class TurnoverEpigraph:
    """
    Epigraph representation of turnover for a CVXPY problem.

    The fields are typed as ``Any`` to keep this module importable even when
    CVXPY is not installed.
    """

    t_var: Any
    objective_term: Any
    constraints: tuple[Any, ...]
    constraint_pos: Any
    constraint_neg: Any
    constraint_t_nonneg: Any


@dataclass(frozen=True)
class QPSolution:
    """Structured QP solution and diagnostics bundle."""

    x_opt: np.ndarray
    objective_value: float
    objective_value_original_q: float
    status: str
    solver: str
    conditioning: ConditioningReport
    kkt: KKTDiagnostics
    diagnostics_primal: Mapping[str, float]
    attempts: tuple[SolverAttempt, ...]
    meta: Mapping[str, Any]
    lambda_ineq: np.ndarray | None = None
    lambda_eq: np.ndarray | None = None
    alpha_lb: np.ndarray | None = None
    beta_ub: np.ndarray | None = None
    turnover_duals: Mapping[str, np.ndarray | None] | None = None

    @property
    def is_certified(self) -> bool:
        return self.kkt.is_certified

    def as_dict(self) -> dict[str, Any]:
        return {
            "x_opt": self.x_opt.copy(),
            "objective_value": float(self.objective_value),
            "objective_value_original_q": float(self.objective_value_original_q),
            "status": self.status,
            "solver": self.solver,
            "conditioning": self.conditioning.as_dict(),
            "kkt": self.kkt.as_dict(),
            "diagnostics_primal": dict(self.diagnostics_primal),
            "attempts": [attempt.as_dict() for attempt in self.attempts],
            "meta": dict(self.meta),
            "lambda_ineq": None if self.lambda_ineq is None else self.lambda_ineq.copy(),
            "lambda_eq": None if self.lambda_eq is None else self.lambda_eq.copy(),
            "alpha_lb": None if self.alpha_lb is None else self.alpha_lb.copy(),
            "beta_ub": None if self.beta_ub is None else self.beta_ub.copy(),
            "turnover_duals": None
            if self.turnover_duals is None
            else {
                key: None if value is None else value.copy()
                for key, value in self.turnover_duals.items()
            },
            "is_certified": bool(self.is_certified),
        }


def available_solvers() -> tuple[str, ...]:
    """Return installed CVXPY solver names in normalized uppercase form."""
    if cp is None:
        return ()
    try:
        return tuple(sorted(str(name).upper() for name in cp.installed_solvers()))
    except Exception:
        return ()


def _require_cvxpy() -> None:
    if cp is None:
        raise SolverUnavailableError(
            "CVXPY is required at runtime for solve_qp but is not installed. "
            "Install cvxpy (and preferably OSQP/CLARABEL/SCS) in the environment."
        )


def _as_float_vector(x: ArrayLike | None, *, name: str, length: int | None = None) -> np.ndarray | None:
    if x is None:
        return None
    arr = np.asarray(x, dtype=float).reshape(-1)
    if length is not None and arr.shape != (length,):
        raise InvalidProblemError(f"{name} must have shape ({length},), got {arr.shape}.")
    if not np.isfinite(arr).all():
        raise InvalidProblemError(f"{name} must contain only finite values.")
    return arr


def _as_float_matrix(x: ArrayLike | None, *, name: str, ncols: int | None = None) -> np.ndarray | None:
    if x is None:
        return None
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise InvalidProblemError(f"{name} must be 2-dimensional, got ndim={arr.ndim}.")
    if ncols is not None and arr.shape[1] != ncols:
        raise InvalidProblemError(
            f"{name} must have shape (m, {ncols}), got {arr.shape}."
        )
    if not np.isfinite(arr).all():
        raise InvalidProblemError(f"{name} must contain only finite values.")
    return arr


def _symmetrize_q(Q: np.ndarray) -> np.ndarray:
    return 0.5 * (Q + Q.T)


def _max_or_zero(values: np.ndarray | None) -> float:
    if values is None:
        return 0.0
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.max(arr))


def _negative_part_max(values: np.ndarray | None) -> float:
    if values is None:
        return math.inf
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.maximum(-arr, 0.0)))


def _complementarity_residual(multiplier: np.ndarray | None, slack: np.ndarray | None) -> float:
    if multiplier is None or slack is None:
        return math.inf
    mult = np.asarray(multiplier, dtype=float).reshape(-1)
    slk = np.asarray(slack, dtype=float).reshape(-1)
    if mult.shape != slk.shape:
        return math.inf
    if mult.size == 0:
        return 0.0
    return float(np.max(np.abs(mult * slk)))


def _normalize_solver_name(name: str) -> str:
    return str(name).strip().upper()


def _accepted_statuses(accept_inaccurate: bool) -> tuple[str, ...]:
    if accept_inaccurate:
        return ("optimal", "optimal_inaccurate")
    return ("optimal",)


def _status_is_infeasible(status: str) -> bool:
    status_n = status.lower()
    return "infeasible" in status_n


def _status_is_unbounded(status: str) -> bool:
    status_n = status.lower()
    return "unbounded" in status_n


def validate_qp_inputs(
    Q: ArrayLike,
    c: ArrayLike,
    C: ArrayLike | None = None,
    d: ArrayLike | None = None,
    E: ArrayLike | None = None,
    f: ArrayLike | None = None,
    lb: ArrayLike | None = None,
    ub: ArrayLike | None = None,
    w_prev: ArrayLike | None = None,
    turnover_penalty: float = 0.0,
) -> dict[str, np.ndarray | None]:
    """
    Validate and normalize canonical QP inputs.

    Returns
    -------
    dict
        Normalized ``numpy.ndarray`` objects under the canonical names.
    """
    Q_arr = np.asarray(Q, dtype=float)
    if Q_arr.ndim != 2 or Q_arr.shape[0] != Q_arr.shape[1]:
        raise InvalidProblemError(f"Q must be square with shape (n, n), got {Q_arr.shape}.")
    if not np.isfinite(Q_arr).all():
        raise InvalidProblemError("Q must contain only finite values.")

    n = int(Q_arr.shape[0])
    c_arr = _as_float_vector(c, name="c", length=n)
    assert c_arr is not None

    if (C is None) != (d is None):
        raise InvalidProblemError("C and d must be provided together or both be None.")
    if (E is None) != (f is None):
        raise InvalidProblemError("E and f must be provided together or both be None.")

    C_arr = _as_float_matrix(C, name="C", ncols=n)
    d_arr = None if d is None else _as_float_vector(d, name="d")
    if C_arr is not None:
        assert d_arr is not None
        if d_arr.shape != (C_arr.shape[0],):
            raise InvalidProblemError(
                f"d must have shape ({C_arr.shape[0]},), got {d_arr.shape}."
            )

    E_arr = _as_float_matrix(E, name="E", ncols=n)
    f_arr = None if f is None else _as_float_vector(f, name="f")
    if E_arr is not None:
        assert f_arr is not None
        if f_arr.shape != (E_arr.shape[0],):
            raise InvalidProblemError(
                f"f must have shape ({E_arr.shape[0]},), got {f_arr.shape}."
            )

    lb_arr = _as_float_vector(lb, name="lb", length=n)
    ub_arr = _as_float_vector(ub, name="ub", length=n)
    if lb_arr is not None and ub_arr is not None:
        if bool(np.any(lb_arr > ub_arr)):
            raise InvalidProblemError("lb must satisfy lb <= ub elementwise.")

    w_prev_arr = _as_float_vector(w_prev, name="w_prev", length=n)
    turnover_penalty_f = float(turnover_penalty)
    if turnover_penalty_f < 0:
        raise InvalidProblemError("turnover_penalty must be non-negative.")
    if turnover_penalty_f > 0.0 and w_prev_arr is None:
        raise InvalidProblemError("w_prev is required when turnover_penalty > 0.")

    return {
        "Q": Q_arr,
        "c": c_arr,
        "C": C_arr,
        "d": d_arr,
        "E": E_arr,
        "f": f_arr,
        "lb": lb_arr,
        "ub": ub_arr,
        "w_prev": w_prev_arr,
    }


def _condition_q(
    Q: np.ndarray,
    *,
    reg_eps: float,
    reg_max: float,
    cond_max: float,
) -> tuple[np.ndarray, ConditioningReport]:
    if reg_eps < 0 or reg_max < 0:
        raise InvalidProblemError("reg_eps and reg_max must be non-negative.")
    if cond_max <= 0:
        raise InvalidProblemError("cond_max must be positive.")

    Q_sym = _symmetrize_q(Q)
    eigvals = np.linalg.eigvalsh(Q_sym)
    lam_min_raw = float(eigvals[0])
    lam_max_raw = float(eigvals[-1])

    delta = float(min(max(reg_eps - lam_min_raw, 0.0), reg_max))
    Q_reg = Q_sym + delta * np.eye(Q_sym.shape[0], dtype=float)
    eigvals_reg = np.linalg.eigvalsh(Q_reg)
    lam_min_reg = float(eigvals_reg[0])
    lam_max_reg = float(eigvals_reg[-1])

    denom = max(lam_min_reg, _DEFAULT_EIGEN_EPS)
    kappa_q = float(lam_max_reg / denom)
    is_psd_raw = bool(lam_min_raw >= -reg_eps)
    is_convex_usable = bool(lam_min_reg >= -reg_eps and np.isfinite(kappa_q) and kappa_q <= cond_max)

    report = ConditioningReport(
        n=int(Q.shape[0]),
        lam_min_raw=lam_min_raw,
        lam_max_raw=lam_max_raw,
        reg_delta=delta,
        lam_min_reg=lam_min_reg,
        lam_max_reg=lam_max_reg,
        kappa_q=kappa_q,
        is_psd_raw=is_psd_raw,
        is_convex_usable=is_convex_usable,
    )

    if lam_min_raw < -reg_max - reg_eps:
        raise ConditioningError(
            "Q is too far from PSD for the configured regularization budget: "
            f"lam_min_raw={lam_min_raw:.6e}, reg_max={reg_max:.6e}."
        )
    if kappa_q > cond_max:
        raise ConditioningError(
            f"Regularized Q remains ill-conditioned: kappa={kappa_q:.6e} > cond_max={cond_max:.6e}."
        )

    return Q_reg, report


def project_to_simplex(z: ArrayLike) -> np.ndarray:
    """
    Project a vector onto the probability simplex.

    Projects ``z`` onto::

        Δ_n = {w : w_i >= 0, sum_i w_i = 1}

    using the standard sorting-based algorithm.
    """
    z_arr = np.asarray(z, dtype=float).reshape(-1)
    if z_arr.size == 0:
        return z_arr.copy()
    u = np.sort(z_arr)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u - cssv / (np.arange(z_arr.size) + 1) > 0)[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(z_arr - theta, 0.0)


def enforce_box_constraints(w: ArrayLike, lb: ArrayLike, ub: ArrayLike) -> np.ndarray:
    """
    Clip a weight vector into a box defined by elementwise bounds.
    """
    w_arr = np.asarray(w, dtype=float)
    lb_arr = np.asarray(lb, dtype=float)
    ub_arr = np.asarray(ub, dtype=float)
    if w_arr.shape != lb_arr.shape or w_arr.shape != ub_arr.shape:
        raise InvalidProblemError(
            f"w, lb, ub must share the same shape; got {w_arr.shape}, {lb_arr.shape}, {ub_arr.shape}."
        )
    if bool(np.any(lb_arr > ub_arr)):
        raise InvalidProblemError("lb must satisfy lb <= ub elementwise.")
    return np.minimum(np.maximum(w_arr, lb_arr), ub_arr)


def build_turnover_epigraph(w_var: Any, w_prev: ArrayLike, turnover_penalty: float) -> TurnoverEpigraph:
    """
    Construct the exact convex epigraph representation for L1 turnover.

    Parameters
    ----------
    w_var:
        CVXPY primal decision variable ``w``.
    w_prev:
        Previous weights.
    turnover_penalty:
        Non-negative coefficient multiplying the L1 turnover term.

    Returns
    -------
    TurnoverEpigraph
        Auxiliary variable, objective term and individual constraints for dual
        inspection downstream.
    """
    _require_cvxpy()

    if turnover_penalty <= 0:
        raise InvalidProblemError("turnover_penalty must be > 0 for epigraph construction.")

    if getattr(w_var, "shape", None) is None:
        raise InvalidProblemError("w_var must be a valid CVXPY variable/expression with a shape.")

    if len(tuple(w_var.shape)) != 1:
        raise InvalidProblemError("w_var must be one-dimensional.")

    n = int(w_var.shape[0])
    w_prev_arr = _as_float_vector(w_prev, name="w_prev", length=n)
    assert w_prev_arr is not None

    t_var = cp.Variable(n, name="turnover_t")
    c_pos = w_var - w_prev_arr - t_var <= 0
    c_neg = -w_var + w_prev_arr - t_var <= 0
    c_t_nonneg = -t_var <= 0
    objective_term = float(turnover_penalty) * cp.sum(t_var)

    return TurnoverEpigraph(
        t_var=t_var,
        objective_term=objective_term,
        constraints=(c_pos, c_neg, c_t_nonneg),
        constraint_pos=c_pos,
        constraint_neg=c_neg,
        constraint_t_nonneg=c_t_nonneg,
    )


def compute_kkt_residuals(
    Q: np.ndarray,
    c: np.ndarray,
    C: np.ndarray | None,
    d: np.ndarray | None,
    E: np.ndarray | None,
    f: np.ndarray | None,
    lb: np.ndarray | None,
    ub: np.ndarray | None,
    w_opt: np.ndarray,
    *,
    lambda_ineq: np.ndarray | None = None,
    lambda_eq: np.ndarray | None = None,
    alpha_lb: np.ndarray | None = None,
    beta_ub: np.ndarray | None = None,
    turnover_meta: Mapping[str, Any] | None = None,
    kkt_tol: float = DEFAULT_KKT_TOL,
) -> KKTDiagnostics:
    """
    Compute KKT-style residuals for the canonical QP.

    Parameters are assumed to have already passed shape validation.
    """
    Q = np.asarray(Q, dtype=float)
    c = np.asarray(c, dtype=float).reshape(-1)
    w_opt = np.asarray(w_opt, dtype=float).reshape(-1)

    if Q.shape != (w_opt.size, w_opt.size):
        raise InvalidProblemError("Q and w_opt have incompatible shapes in compute_kkt_residuals.")
    if c.shape != (w_opt.size,):
        raise InvalidProblemError("c and w_opt have incompatible shapes in compute_kkt_residuals.")

    r_eq_feas = 0.0 if E is None else float(np.max(np.abs(E @ w_opt - f)))
    r_ineq_feas = 0.0 if C is None else float(np.max(np.maximum(C @ w_opt - d, 0.0)))
    r_lb_feas = 0.0 if lb is None else float(np.max(np.maximum(lb - w_opt, 0.0)))
    r_ub_feas = 0.0 if ub is None else float(np.max(np.maximum(w_opt - ub, 0.0)))

    r_dual_feas_ineq = 0.0 if C is None else _negative_part_max(lambda_ineq)
    r_dual_feas_lb = 0.0 if lb is None else _negative_part_max(alpha_lb)
    r_dual_feas_ub = 0.0 if ub is None else _negative_part_max(beta_ub)

    stat = Q @ w_opt + c
    if C is not None and lambda_ineq is not None:
        stat = stat + C.T @ np.asarray(lambda_ineq, dtype=float).reshape(-1)
    elif C is not None:
        stat = np.full_like(stat, np.inf)

    if E is not None and lambda_eq is not None:
        stat = stat + E.T @ np.asarray(lambda_eq, dtype=float).reshape(-1)
    elif E is not None:
        stat = np.full_like(stat, np.inf)

    if lb is not None and alpha_lb is not None:
        stat = stat - np.asarray(alpha_lb, dtype=float).reshape(-1)
    elif lb is not None:
        stat = np.full_like(stat, np.inf)

    if ub is not None and beta_ub is not None:
        stat = stat + np.asarray(beta_ub, dtype=float).reshape(-1)
    elif ub is not None:
        stat = np.full_like(stat, np.inf)

    r_turnover_feas_pos = 0.0
    r_turnover_feas_neg = 0.0
    r_turnover_feas_t_nonneg = 0.0
    r_turnover_dual_feas_pos = 0.0
    r_turnover_dual_feas_neg = 0.0
    r_turnover_dual_feas_t_nonneg = 0.0
    r_turnover_complementarity_pos = 0.0
    r_turnover_complementarity_neg = 0.0
    r_turnover_complementarity_t_nonneg = 0.0
    r_turnover_stationarity_t = 0.0

    if turnover_meta:
        w_prev = _as_float_vector(turnover_meta.get("w_prev"), name="turnover_meta.w_prev", length=w_opt.size)
        turnover_penalty = float(turnover_meta.get("turnover_penalty", 0.0))
        t_opt = _as_float_vector(turnover_meta.get("t_opt"), name="turnover_meta.t_opt", length=w_opt.size)
        gamma_pos = _as_float_vector(
            turnover_meta.get("lambda_turnover_pos"),
            name="turnover_meta.lambda_turnover_pos",
            length=w_opt.size,
        )
        gamma_neg = _as_float_vector(
            turnover_meta.get("lambda_turnover_neg"),
            name="turnover_meta.lambda_turnover_neg",
            length=w_opt.size,
        )
        gamma_t = _as_float_vector(
            turnover_meta.get("lambda_turnover_t_nonneg"),
            name="turnover_meta.lambda_turnover_t_nonneg",
            length=w_opt.size,
        )

        if w_prev is None or t_opt is None:
            raise InvalidProblemError(
                "turnover_meta must include w_prev and t_opt when provided."
            )

        slack_pos = w_opt - w_prev - t_opt
        slack_neg = -w_opt + w_prev - t_opt
        slack_t = -t_opt

        r_turnover_feas_pos = float(np.max(np.maximum(slack_pos, 0.0)))
        r_turnover_feas_neg = float(np.max(np.maximum(slack_neg, 0.0)))
        r_turnover_feas_t_nonneg = float(np.max(np.maximum(slack_t, 0.0)))
        r_turnover_dual_feas_pos = _negative_part_max(gamma_pos)
        r_turnover_dual_feas_neg = _negative_part_max(gamma_neg)
        r_turnover_dual_feas_t_nonneg = _negative_part_max(gamma_t)
        r_turnover_complementarity_pos = _complementarity_residual(gamma_pos, slack_pos)
        r_turnover_complementarity_neg = _complementarity_residual(gamma_neg, slack_neg)
        r_turnover_complementarity_t_nonneg = _complementarity_residual(gamma_t, slack_t)

        if gamma_pos is None or gamma_neg is None:
            stat = np.full_like(stat, np.inf)
        else:
            stat = stat + gamma_pos - gamma_neg

        if gamma_pos is None or gamma_neg is None or gamma_t is None:
            r_turnover_stationarity_t = math.inf
        else:
            stat_t = (
                turnover_penalty * np.ones_like(t_opt)
                - gamma_pos
                - gamma_neg
                - gamma_t
            )
            r_turnover_stationarity_t = float(np.linalg.norm(stat_t, ord=np.inf))

    r_stationarity = float(np.linalg.norm(stat, ord=np.inf)) if np.all(np.isfinite(stat)) else math.inf

    g_ineq = None if C is None else (C @ w_opt - d)
    g_lb = None if lb is None else (lb - w_opt)
    g_ub = None if ub is None else (w_opt - ub)

    r_complementarity_ineq = 0.0 if C is None else _complementarity_residual(lambda_ineq, g_ineq)
    r_complementarity_lb = 0.0 if lb is None else _complementarity_residual(alpha_lb, g_lb)
    r_complementarity_ub = 0.0 if ub is None else _complementarity_residual(beta_ub, g_ub)

    return KKTDiagnostics(
        r_eq_feas=r_eq_feas,
        r_ineq_feas=r_ineq_feas,
        r_lb_feas=r_lb_feas,
        r_ub_feas=r_ub_feas,
        r_dual_feas_ineq=r_dual_feas_ineq,
        r_dual_feas_lb=r_dual_feas_lb,
        r_dual_feas_ub=r_dual_feas_ub,
        r_stationarity=r_stationarity,
        r_complementarity_ineq=r_complementarity_ineq,
        r_complementarity_lb=r_complementarity_lb,
        r_complementarity_ub=r_complementarity_ub,
        r_turnover_feas_pos=r_turnover_feas_pos,
        r_turnover_feas_neg=r_turnover_feas_neg,
        r_turnover_feas_t_nonneg=r_turnover_feas_t_nonneg,
        r_turnover_dual_feas_pos=r_turnover_dual_feas_pos,
        r_turnover_dual_feas_neg=r_turnover_dual_feas_neg,
        r_turnover_dual_feas_t_nonneg=r_turnover_dual_feas_t_nonneg,
        r_turnover_complementarity_pos=r_turnover_complementarity_pos,
        r_turnover_complementarity_neg=r_turnover_complementarity_neg,
        r_turnover_complementarity_t_nonneg=r_turnover_complementarity_t_nonneg,
        r_turnover_stationarity_t=r_turnover_stationarity_t,
        kkt_tol=float(kkt_tol),
    )


def _solver_option_bundle(solver_name: str, cfg: SolverConfig, *, warm_start: bool) -> dict[str, Any]:
    name = _normalize_solver_name(solver_name)
    opts: dict[str, Any]

    if name == "OSQP":
        opts = {
            "eps_abs": cfg.eps_abs,
            "eps_rel": cfg.eps_rel,
            "max_iter": cfg.max_iters,
            "verbose": cfg.verbose,
            "warm_start": warm_start,
        }
    elif name == "CLARABEL":
        opts = {
            "tol_gap_abs": cfg.eps_abs,
            "tol_gap_rel": cfg.eps_rel,
            "tol_feas": cfg.eps_abs,
            "max_iter": cfg.max_iters,
            "verbose": cfg.verbose,
        }
    elif name == "SCS":
        opts = {
            "eps_abs": cfg.eps_abs,
            "eps_rel": cfg.eps_rel,
            "max_iters": cfg.max_iters,
            "verbose": cfg.verbose,
            "warm_start": warm_start,
        }
    else:
        opts = {"verbose": cfg.verbose}

    extra = dict(cfg.solver_options.get(name, {}))
    opts.update(extra)
    return opts


def _status_label(problem: Any) -> str:
    status = getattr(problem, "status", None)
    return "unknown" if status is None else str(status)


def _problem_value(problem: Any) -> float:
    value = getattr(problem, "value", None)
    if value is None:
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def _try_solver(
    problem: Any,
    solver_name: str,
    cfg: SolverConfig,
    *,
    warm_start_enabled: bool,
) -> SolverAttempt:
    options = _solver_option_bundle(solver_name, cfg, warm_start=warm_start_enabled)
    solver_constant = getattr(cp, _normalize_solver_name(solver_name), None)
    if solver_constant is None:
        raise SolverUnavailableError(f"Requested solver {solver_name!r} is not recognized by CVXPY.")

    t0 = perf_counter()
    try:
        problem.solve(solver=solver_constant, **options)
        elapsed = perf_counter() - t0
        return SolverAttempt(
            solver=_normalize_solver_name(solver_name),
            status=_status_label(problem),
            solve_time_sec=elapsed,
            options=MappingProxyType(dict(options)),
            error_message=None,
        )
    except Exception as exc:
        elapsed = perf_counter() - t0
        return SolverAttempt(
            solver=_normalize_solver_name(solver_name),
            status="solver_error",
            solve_time_sec=elapsed,
            options=MappingProxyType(dict(options)),
            error_message=f"{type(exc).__name__}: {exc}",
        )


def solve_qp_result(
    Q: ArrayLike,
    c: ArrayLike,
    C: ArrayLike | None = None,
    d: ArrayLike | None = None,
    E: ArrayLike | None = None,
    f: ArrayLike | None = None,
    lb: ArrayLike | None = None,
    ub: ArrayLike | None = None,
    w_prev: ArrayLike | None = None,
    turnover_penalty: float = 0.0,
    reg_eps: float = DEFAULT_REG_EPS,
    reg_max: float = DEFAULT_REG_MAX,
    cond_max: float = DEFAULT_COND_MAX,
    kkt_tol: float = DEFAULT_KKT_TOL,
    solver_cfg: Mapping[str, Any] | SolverConfig | None = None,
    warm_start: ArrayLike | None = None,
) -> QPSolution:
    """
    Solve the canonical convex QP and return a structured diagnostics bundle.
    """
    _require_cvxpy()

    normalized = validate_qp_inputs(
        Q=Q,
        c=c,
        C=C,
        d=d,
        E=E,
        f=f,
        lb=lb,
        ub=ub,
        w_prev=w_prev,
        turnover_penalty=turnover_penalty,
    )
    Q_raw = normalized["Q"]
    c_arr = normalized["c"]
    C_arr = normalized["C"]
    d_arr = normalized["d"]
    E_arr = normalized["E"]
    f_arr = normalized["f"]
    lb_arr = normalized["lb"]
    ub_arr = normalized["ub"]
    w_prev_arr = normalized["w_prev"]

    assert Q_raw is not None and c_arr is not None

    cfg = SolverConfig.from_any(solver_cfg)
    Q_reg, conditioning = _condition_q(
        Q_raw,
        reg_eps=float(reg_eps),
        reg_max=float(reg_max),
        cond_max=float(cond_max),
    )

    n = int(Q_raw.shape[0])
    w = cp.Variable(n, name="w")

    objective_expr = 0.5 * cp.quad_form(w, Q_reg) + c_arr @ w
    constraints: list[Any] = []

    c_ineq = None
    c_eq = None
    c_lb = None
    c_ub = None
    turnover_epigraph: TurnoverEpigraph | None = None

    if C_arr is not None and d_arr is not None:
        c_ineq = C_arr @ w - d_arr <= 0
        constraints.append(c_ineq)

    if E_arr is not None and f_arr is not None:
        c_eq = E_arr @ w - f_arr == 0
        constraints.append(c_eq)

    if lb_arr is not None:
        c_lb = lb_arr - w <= 0
        constraints.append(c_lb)

    if ub_arr is not None:
        c_ub = w - ub_arr <= 0
        constraints.append(c_ub)

    if float(turnover_penalty) > 0.0:
        assert w_prev_arr is not None
        turnover_epigraph = build_turnover_epigraph(
            w_var=w,
            w_prev=w_prev_arr,
            turnover_penalty=float(turnover_penalty),
        )
        objective_expr = objective_expr + turnover_epigraph.objective_term
        constraints.extend(turnover_epigraph.constraints)

    problem = cp.Problem(cp.Minimize(objective_expr), constraints)

    warm_start_arr = None if warm_start is None else _as_float_vector(warm_start, name="warm_start", length=n)
    if warm_start_arr is not None:
        w.value = warm_start_arr.copy()
        if turnover_epigraph is not None and w_prev_arr is not None:
            turnover_epigraph.t_var.value = np.abs(warm_start_arr - w_prev_arr)

    attempts: list[SolverAttempt] = []
    accepted = _accepted_statuses(cfg.accept_inaccurate)
    installed = set(available_solvers())
    last_attempt_status = "not_attempted"

    started = perf_counter()
    for solver_name in cfg.solver_order:
        solver_norm = _normalize_solver_name(solver_name)
        if solver_norm not in installed:
            attempts.append(
                SolverAttempt(
                    solver=solver_norm,
                    status="not_installed",
                    solve_time_sec=0.0,
                    options=MappingProxyType({}),
                    error_message="Solver not installed in current environment.",
                )
            )
            continue

        attempt = _try_solver(
            problem=problem,
            solver_name=solver_norm,
            cfg=cfg,
            warm_start_enabled=warm_start_arr is not None,
        )
        attempts.append(attempt)
        last_attempt_status = attempt.status

        if attempt.status.lower() in accepted:
            break
        if _status_is_infeasible(attempt.status):
            raise InfeasibleProblemError(
                f"Optimization problem reported infeasible by solver {solver_norm}."
            )
        if _status_is_unbounded(attempt.status):
            raise UnboundedProblemError(
                f"Optimization problem reported unbounded by solver {solver_norm}."
            )

    elapsed_sec = perf_counter() - started
    status = _status_label(problem)
    if status.lower() not in accepted:
        raise SolverFailureError(
            "Optimization failed after solver waterfall. "
            f"final_status={status!r}, last_attempt_status={last_attempt_status!r}."
        )

    if w.value is None:
        raise SolverFailureError("Solver reported success but primal variable w has no value.")

    x_opt = np.asarray(w.value, dtype=float).reshape(-1)

    lambda_ineq = None if c_ineq is None or c_ineq.dual_value is None else np.asarray(c_ineq.dual_value, dtype=float).reshape(-1)
    lambda_eq = None if c_eq is None or c_eq.dual_value is None else np.asarray(c_eq.dual_value, dtype=float).reshape(-1)
    alpha_lb = None if c_lb is None or c_lb.dual_value is None else np.asarray(c_lb.dual_value, dtype=float).reshape(-1)
    beta_ub = None if c_ub is None or c_ub.dual_value is None else np.asarray(c_ub.dual_value, dtype=float).reshape(-1)

    turnover_duals: dict[str, np.ndarray | None] | None = None
    turnover_meta: dict[str, Any] | None = None
    if turnover_epigraph is not None:
        t_opt = None if turnover_epigraph.t_var.value is None else np.asarray(turnover_epigraph.t_var.value, dtype=float).reshape(-1)
        gamma_pos = (
            None
            if turnover_epigraph.constraint_pos.dual_value is None
            else np.asarray(turnover_epigraph.constraint_pos.dual_value, dtype=float).reshape(-1)
        )
        gamma_neg = (
            None
            if turnover_epigraph.constraint_neg.dual_value is None
            else np.asarray(turnover_epigraph.constraint_neg.dual_value, dtype=float).reshape(-1)
        )
        gamma_t_nonneg = (
            None
            if turnover_epigraph.constraint_t_nonneg.dual_value is None
            else np.asarray(turnover_epigraph.constraint_t_nonneg.dual_value, dtype=float).reshape(-1)
        )
        turnover_duals = {
            "lambda_turnover_pos": gamma_pos,
            "lambda_turnover_neg": gamma_neg,
            "lambda_turnover_t_nonneg": gamma_t_nonneg,
        }
        turnover_meta = {
            "w_prev": w_prev_arr,
            "turnover_penalty": float(turnover_penalty),
            "t_opt": t_opt,
            "lambda_turnover_pos": gamma_pos,
            "lambda_turnover_neg": gamma_neg,
            "lambda_turnover_t_nonneg": gamma_t_nonneg,
        }

    kkt = compute_kkt_residuals(
        Q=Q_reg,
        c=c_arr,
        C=C_arr,
        d=d_arr,
        E=E_arr,
        f=f_arr,
        lb=lb_arr,
        ub=ub_arr,
        w_opt=x_opt,
        lambda_ineq=lambda_ineq,
        lambda_eq=lambda_eq,
        alpha_lb=alpha_lb,
        beta_ub=beta_ub,
        turnover_meta=turnover_meta,
        kkt_tol=float(kkt_tol),
    )

    diagnostics_primal = {
        "r_eq_feas": float(kkt.r_eq_feas),
        "r_ineq_feas": float(kkt.r_ineq_feas),
        "r_lb_feas": float(kkt.r_lb_feas),
        "r_ub_feas": float(kkt.r_ub_feas),
        "max_feas_residual": float(max(kkt.r_eq_feas, kkt.r_ineq_feas, kkt.r_lb_feas, kkt.r_ub_feas)),
    }

    meta = {
        "n_assets": n,
        "n_ineq": 0 if C_arr is None else int(C_arr.shape[0]),
        "n_eq": 0 if E_arr is None else int(E_arr.shape[0]),
        "has_turnover": bool(float(turnover_penalty) > 0.0),
        "solver_order_tried": tuple(_normalize_solver_name(s) for s in cfg.solver_order),
        "elapsed_sec": float(elapsed_sec),
        "tol_used": {
            "eps_abs": float(cfg.eps_abs),
            "eps_rel": float(cfg.eps_rel),
            "kkt_tol": float(kkt_tol),
        },
        "cvxpy_available": True,
        "accept_inaccurate": bool(cfg.accept_inaccurate),
        "installed_solvers": available_solvers(),
    }

    objective_value_original_q = float(0.5 * x_opt @ Q_raw @ x_opt + c_arr @ x_opt)
    if float(turnover_penalty) > 0.0 and w_prev_arr is not None:
        objective_value_original_q += float(turnover_penalty) * float(np.sum(np.abs(x_opt - w_prev_arr)))

    solution = QPSolution(
        x_opt=x_opt,
        objective_value=float(_problem_value(problem)),
        objective_value_original_q=objective_value_original_q,
        status=status,
        solver=attempts[-1].solver if attempts else "unknown",
        conditioning=conditioning,
        kkt=kkt,
        diagnostics_primal=MappingProxyType(diagnostics_primal),
        attempts=tuple(attempts),
        meta=MappingProxyType(meta),
        lambda_ineq=lambda_ineq,
        lambda_eq=lambda_eq,
        alpha_lb=alpha_lb,
        beta_ub=beta_ub,
        turnover_duals=None if turnover_duals is None else MappingProxyType(turnover_duals),
    )

    return solution


def solve_qp(
    Q: ArrayLike,
    c: ArrayLike,
    C: ArrayLike | None = None,
    d: ArrayLike | None = None,
    E: ArrayLike | None = None,
    f: ArrayLike | None = None,
    lb: ArrayLike | None = None,
    ub: ArrayLike | None = None,
    w_prev: ArrayLike | None = None,
    turnover_penalty: float = 0.0,
    reg_eps: float = DEFAULT_REG_EPS,
    reg_max: float = DEFAULT_REG_MAX,
    cond_max: float = DEFAULT_COND_MAX,
    kkt_tol: float = DEFAULT_KKT_TOL,
    solver_cfg: Mapping[str, Any] | SolverConfig | None = None,
    warm_start: ArrayLike | None = None,
) -> dict[str, Any]:
    """
    Solve the canonical convex QP and return a dict-friendly artifact bundle.

    This is the compatibility wrapper expected by higher layers that prefer a
    mapping payload over a dataclass.
    """
    return solve_qp_result(
        Q=Q,
        c=c,
        C=C,
        d=d,
        E=E,
        f=f,
        lb=lb,
        ub=ub,
        w_prev=w_prev,
        turnover_penalty=turnover_penalty,
        reg_eps=reg_eps,
        reg_max=reg_max,
        cond_max=cond_max,
        kkt_tol=kkt_tol,
        solver_cfg=solver_cfg,
        warm_start=warm_start,
    ).as_dict()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, MappingProxyType):
        return dict(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _run_self_test() -> dict[str, Any]:
    """
    Minimal deterministic self-test suite.

    The self-test intentionally covers:
    - simplex projection invariants;
    - box clipping invariants;
    - KKT residual helper on a trivial exact case;
    - an optional tiny QP solve if CVXPY is available.
    """
    results: dict[str, Any] = {
        "cvxpy_available": cp is not None,
        "installed_solvers": available_solvers(),
        "tests": {},
    }

    z = np.array([0.2, -0.1, 1.7, 0.3])
    proj = project_to_simplex(z)
    results["tests"]["project_to_simplex"] = {
        "sum": float(np.sum(proj)),
        "min": float(np.min(proj)),
        "nonnegative": bool(np.all(proj >= -1e-12)),
        "close_to_one": bool(abs(float(np.sum(proj)) - 1.0) <= 1e-10),
    }

    clipped = enforce_box_constraints(
        np.array([1.2, -0.5, 0.3]),
        np.array([0.0, 0.0, -1.0]),
        np.array([1.0, 1.0, 0.5]),
    )
    results["tests"]["enforce_box_constraints"] = {
        "vector": clipped,
        "within_bounds": bool(np.all(clipped >= np.array([0.0, 0.0, -1.0])) and np.all(clipped <= np.array([1.0, 1.0, 0.5]))),
    }

    Q = np.eye(2)
    c = np.array([0.0, 0.0])
    E = np.array([[1.0, 1.0]])
    f = np.array([1.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    w_star = np.array([0.5, 0.5])
    lambda_eq = np.array([-0.5])
    alpha_lb = np.array([0.0, 0.0])
    beta_ub = np.array([0.0, 0.0])

    kkt = compute_kkt_residuals(
        Q=Q,
        c=c,
        C=None,
        d=None,
        E=E,
        f=f,
        lb=lb,
        ub=ub,
        w_opt=w_star,
        lambda_ineq=None,
        lambda_eq=lambda_eq,
        alpha_lb=alpha_lb,
        beta_ub=beta_ub,
        kkt_tol=1e-12,
    )
    results["tests"]["compute_kkt_residuals"] = kkt.as_dict()

    if cp is not None and available_solvers():
        tiny = solve_qp(
            Q=np.eye(2),
            c=np.array([0.0, 0.0]),
            E=np.array([[1.0, 1.0]]),
            f=np.array([1.0]),
            lb=np.array([0.0, 0.0]),
            ub=np.array([1.0, 1.0]),
            kkt_tol=1e-8,
        )
        results["tests"]["tiny_qp"] = {
            "x_opt": tiny["x_opt"],
            "status": tiny["status"],
            "solver": tiny["solver"],
            "is_certified": tiny["kkt"]["is_certified"],
        }
    else:
        results["tests"]["tiny_qp"] = {
            "skipped": True,
            "reason": "CVXPY or solver backend not available in current environment.",
        }

    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Institutional convex optimization primitives for simons_core."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a deterministic built-in self-test and print JSON.",
    )
    parser.add_argument(
        "--show-solvers",
        action="store_true",
        help="Print installed CVXPY solver backends and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.show_solvers:
        print(json.dumps({"installed_solvers": available_solvers()}, indent=2))
        return 0

    if args.self_test:
        payload = _run_self_test()
        print(json.dumps(payload, indent=2, default=_json_default))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
