from __future__ import annotations

import math

import numpy as np
import pytest


pytestmark = [pytest.mark.math, pytest.mark.solver]


def _preferred_solver_cfg(optimization_mod, *, include_fake_first: bool = False) -> dict[str, object]:
    installed = tuple(str(s).upper() for s in optimization_mod.available_solvers())
    preferred = [s for s in optimization_mod.DEFAULT_SOLVER_ORDER if s in installed]
    if not preferred:
        preferred = list(installed)
    if not preferred:
        pytest.skip("No CVXPY solver backend installed in current environment.")

    order = list(preferred)
    if include_fake_first:
        order = ["__NOT_INSTALLED__", *order]

    return {
        "solver_order": tuple(order),
        "eps_abs": 1e-9,
        "eps_rel": 1e-9,
        "max_iters": 50_000,
        "accept_inaccurate": True,
        "verbose": False,
    }


@pytest.fixture()
def require_solver_backend(cvxpy_available: bool) -> None:
    if not cvxpy_available:
        pytest.skip("CVXPY and a supported solver backend are not available in this environment.")


def test_solve_qp_result_raises_when_cvxpy_unavailable(monkeypatch, optimization_mod):
    monkeypatch.setattr(optimization_mod, "cp", None)

    with pytest.raises(optimization_mod.SolverUnavailableError, match="CVXPY is required"):
        optimization_mod.solve_qp_result(
            Q=np.eye(2, dtype=float),
            c=np.array([0.0, 0.0], dtype=float),
        )


def test_solve_qp_result_unconstrained_identity_matches_closed_form(
    optimization_mod,
    qp_unconstrained_identity,
    assert_allclose,
    require_solver_backend,
):
    sol = optimization_mod.solve_qp_result(
        Q=qp_unconstrained_identity.Q,
        c=qp_unconstrained_identity.c,
        solver_cfg=_preferred_solver_cfg(optimization_mod),
    )

    expected = -qp_unconstrained_identity.c
    assert isinstance(sol, optimization_mod.QPSolution)
    assert sol.x_opt.shape == expected.shape
    assert_allclose(sol.x_opt, expected, atol=1e-7, rtol=1e-7)
    assert "optimal" in sol.status.lower()
    assert sol.solver in {attempt.solver for attempt in sol.attempts}
    assert math.isfinite(sol.objective_value)
    assert math.isfinite(sol.objective_value_original_q)
    assert sol.kkt.is_certified is True


def test_solve_qp_wrapper_returns_dict_payload(
    optimization_mod,
    qp_unconstrained_identity,
    assert_allclose,
    require_solver_backend,
):
    artifact = optimization_mod.solve_qp(
        Q=qp_unconstrained_identity.Q,
        c=qp_unconstrained_identity.c,
        solver_cfg=_preferred_solver_cfg(optimization_mod),
    )

    assert isinstance(artifact, dict)
    assert set(
        [
            "x_opt",
            "objective_value",
            "objective_value_original_q",
            "status",
            "solver",
            "conditioning",
            "kkt",
            "diagnostics_primal",
            "attempts",
            "meta",
            "lambda_ineq",
            "lambda_eq",
            "alpha_lb",
            "beta_ub",
            "turnover_duals",
            "is_certified",
        ]
    ).issubset(artifact)
    assert_allclose(artifact["x_opt"], np.array([1.0, 2.0], dtype=float), atol=1e-7, rtol=1e-7)
    assert artifact["is_certified"] is True
    assert "optimal" in str(artifact["status"]).lower()


def test_solve_qp_result_respects_box_constraints(
    optimization_mod,
    qp_box_only,
    require_solver_backend,
):
    sol = optimization_mod.solve_qp_result(
        Q=qp_box_only.Q,
        c=qp_box_only.c,
        lb=qp_box_only.lb,
        ub=qp_box_only.ub,
        solver_cfg=_preferred_solver_cfg(optimization_mod),
    )

    assert np.all(sol.x_opt >= qp_box_only.lb - 1e-8)
    assert np.all(sol.x_opt <= qp_box_only.ub + 1e-8)
    assert np.allclose(sol.x_opt, np.array([0.6, -0.3]), atol=1e-6, rtol=1e-6)
    assert sol.kkt.r_lb_feas <= 1e-7
    assert sol.kkt.r_ub_feas <= 1e-7


def test_solve_qp_result_respects_sum_to_one_and_bounds(
    optimization_mod,
    qp_sum_to_one_box,
    assert_allclose,
    require_solver_backend,
):
    sol = optimization_mod.solve_qp_result(
        Q=qp_sum_to_one_box.Q,
        c=qp_sum_to_one_box.c,
        E=qp_sum_to_one_box.E,
        f=qp_sum_to_one_box.f,
        lb=qp_sum_to_one_box.lb,
        ub=qp_sum_to_one_box.ub,
        w_prev=qp_sum_to_one_box.w_prev,
        turnover_penalty=qp_sum_to_one_box.turnover_penalty,
        solver_cfg=_preferred_solver_cfg(optimization_mod),
    )

    assert_allclose(np.sum(sol.x_opt), 1.0, atol=1e-7, rtol=1e-7)
    assert np.all(sol.x_opt >= qp_sum_to_one_box.lb - 1e-8)
    assert np.all(sol.x_opt <= qp_sum_to_one_box.ub + 1e-8)
    assert sol.kkt.r_eq_feas <= 1e-7
    assert sol.kkt.r_lb_feas <= 1e-7
    assert sol.kkt.r_ub_feas <= 1e-7
    assert sol.meta["has_turnover"] is True


def test_turnover_penalty_pulls_solution_towards_previous_weights(
    optimization_mod,
    qp_sum_to_one_box,
    require_solver_backend,
):
    no_turn = optimization_mod.solve_qp_result(
        Q=qp_sum_to_one_box.Q,
        c=qp_sum_to_one_box.c,
        E=qp_sum_to_one_box.E,
        f=qp_sum_to_one_box.f,
        lb=qp_sum_to_one_box.lb,
        ub=qp_sum_to_one_box.ub,
        solver_cfg=_preferred_solver_cfg(optimization_mod),
    )
    with_turn = optimization_mod.solve_qp_result(
        Q=qp_sum_to_one_box.Q,
        c=qp_sum_to_one_box.c,
        E=qp_sum_to_one_box.E,
        f=qp_sum_to_one_box.f,
        lb=qp_sum_to_one_box.lb,
        ub=qp_sum_to_one_box.ub,
        w_prev=qp_sum_to_one_box.w_prev,
        turnover_penalty=5.0,
        solver_cfg=_preferred_solver_cfg(optimization_mod),
    )

    dist_no_turn = float(np.sum(np.abs(no_turn.x_opt - qp_sum_to_one_box.w_prev)))
    dist_with_turn = float(np.sum(np.abs(with_turn.x_opt - qp_sum_to_one_box.w_prev)))

    assert dist_with_turn <= dist_no_turn + 1e-8
    assert with_turn.turnover_duals is not None
    assert with_turn.meta["has_turnover"] is True


def test_solver_waterfall_records_not_installed_attempt_before_success(
    optimization_mod,
    qp_unconstrained_identity,
    require_solver_backend,
):
    sol = optimization_mod.solve_qp_result(
        Q=qp_unconstrained_identity.Q,
        c=qp_unconstrained_identity.c,
        solver_cfg=_preferred_solver_cfg(optimization_mod, include_fake_first=True),
    )

    assert len(sol.attempts) >= 2
    assert sol.attempts[0].solver == "__NOT_INSTALLED__"
    assert sol.attempts[0].status == "not_installed"
    assert sol.attempts[0].error_message is not None
    assert "optimal" in sol.attempts[-1].status.lower()


def test_solve_qp_result_supports_warm_start(
    optimization_mod,
    qp_with_ineq,
    warm_start_good,
    require_solver_backend,
):
    sol = optimization_mod.solve_qp_result(
        Q=qp_with_ineq.Q,
        c=qp_with_ineq.c,
        C=qp_with_ineq.C,
        d=qp_with_ineq.d,
        lb=qp_with_ineq.lb,
        ub=qp_with_ineq.ub,
        warm_start=warm_start_good,
        solver_cfg=_preferred_solver_cfg(optimization_mod),
    )

    assert np.all(sol.x_opt >= qp_with_ineq.lb - 1e-8)
    assert np.all(sol.x_opt <= qp_with_ineq.ub + 1e-8)
    assert sol.kkt.r_ineq_feas <= 1e-7


def test_solve_qp_result_raises_infeasible_problem(
    optimization_mod,
    require_solver_backend,
):
    with pytest.raises(optimization_mod.InfeasibleProblemError):
        optimization_mod.solve_qp_result(
            Q=np.eye(1, dtype=float),
            c=np.array([0.0], dtype=float),
            lb=np.array([0.0], dtype=float),
            ub=np.array([1.0], dtype=float),
            E=np.array([[1.0]], dtype=float),
            f=np.array([2.0], dtype=float),
            solver_cfg=_preferred_solver_cfg(optimization_mod),
        )


def test_solve_qp_result_raises_unbounded_problem(
    optimization_mod,
    require_solver_backend,
):
    with pytest.raises(optimization_mod.UnboundedProblemError):
        optimization_mod.solve_qp_result(
            Q=np.zeros((1, 1), dtype=float),
            c=np.array([-1.0], dtype=float),
            solver_cfg=_preferred_solver_cfg(optimization_mod),
        )
