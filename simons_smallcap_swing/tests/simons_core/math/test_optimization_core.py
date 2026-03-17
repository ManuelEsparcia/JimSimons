from __future__ import annotations

import math

import numpy as np
import pytest


pytestmark = [pytest.mark.math]


def test_validate_qp_inputs_normalizes_well_formed_problem(optimization_mod, qp_sum_to_one_box):
    normalized = optimization_mod.validate_qp_inputs(
        Q=qp_sum_to_one_box.Q,
        c=qp_sum_to_one_box.c,
        C=qp_sum_to_one_box.C,
        d=qp_sum_to_one_box.d,
        E=qp_sum_to_one_box.E,
        f=qp_sum_to_one_box.f,
        lb=qp_sum_to_one_box.lb,
        ub=qp_sum_to_one_box.ub,
        w_prev=qp_sum_to_one_box.w_prev,
        turnover_penalty=qp_sum_to_one_box.turnover_penalty,
    )

    assert set(normalized) == {"Q", "c", "C", "d", "E", "f", "lb", "ub", "w_prev"}
    assert normalized["Q"].shape == (3, 3)
    assert normalized["c"].shape == (3,)
    assert normalized["E"].shape == (1, 3)
    assert normalized["f"].shape == (1,)
    assert normalized["lb"].shape == (3,)
    assert normalized["ub"].shape == (3,)
    assert normalized["w_prev"].shape == (3,)
    assert normalized["C"] is None and normalized["d"] is None


def test_validate_qp_inputs_rejects_nonsquare_q(optimization_mod):
    with pytest.raises(optimization_mod.InvalidProblemError, match="Q must be square"):
        optimization_mod.validate_qp_inputs(
            Q=np.ones((2, 3), dtype=float),
            c=np.array([1.0, 2.0], dtype=float),
        )


def test_validate_qp_inputs_rejects_nonfinite_q(optimization_mod):
    bad_q = np.array([[1.0, np.nan], [0.0, 1.0]], dtype=float)
    with pytest.raises(optimization_mod.InvalidProblemError, match="finite"):
        optimization_mod.validate_qp_inputs(
            Q=bad_q,
            c=np.array([1.0, 2.0], dtype=float),
        )


def test_validate_qp_inputs_rejects_bad_c_shape(optimization_mod, qp_incompatible_shapes):
    with pytest.raises(optimization_mod.InvalidProblemError, match="c must have shape"):
        optimization_mod.validate_qp_inputs(**qp_incompatible_shapes)


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    [
        ({"C": np.eye(2)}, "C and d must be provided together"),
        ({"d": np.ones(2)}, "C and d must be provided together"),
        ({"E": np.eye(2)}, "E and f must be provided together"),
        ({"f": np.ones(2)}, "E and f must be provided together"),
    ],
)
def test_validate_qp_inputs_requires_paired_constraints(optimization_mod, kwargs, pattern):
    base = {"Q": np.eye(2, dtype=float), "c": np.array([0.0, 0.0], dtype=float)}
    base.update(kwargs)
    with pytest.raises(optimization_mod.InvalidProblemError, match=pattern):
        optimization_mod.validate_qp_inputs(**base)


def test_validate_qp_inputs_rejects_bounds_inversion(optimization_mod):
    with pytest.raises(optimization_mod.InvalidProblemError, match="lb <= ub"):
        optimization_mod.validate_qp_inputs(
            Q=np.eye(2, dtype=float),
            c=np.array([0.0, 0.0], dtype=float),
            lb=np.array([0.0, 1.0], dtype=float),
            ub=np.array([1.0, 0.5], dtype=float),
        )


def test_validate_qp_inputs_requires_w_prev_for_turnover(optimization_mod):
    with pytest.raises(optimization_mod.InvalidProblemError, match="w_prev is required"):
        optimization_mod.validate_qp_inputs(
            Q=np.eye(2, dtype=float),
            c=np.array([0.0, 0.0], dtype=float),
            turnover_penalty=0.1,
        )


def test_validate_qp_inputs_rejects_negative_turnover_penalty(optimization_mod):
    with pytest.raises(optimization_mod.InvalidProblemError, match="non-negative"):
        optimization_mod.validate_qp_inputs(
            Q=np.eye(2, dtype=float),
            c=np.array([0.0, 0.0], dtype=float),
            turnover_penalty=-1.0,
        )


def test_project_to_simplex_enforces_simplex_invariants(optimization_mod, simplex_input, assert_allclose):
    proj = optimization_mod.project_to_simplex(simplex_input)

    assert proj.shape == simplex_input.shape
    assert np.all(proj >= -1e-15)
    assert_allclose(np.sum(proj), 1.0, atol=1e-12)



def test_project_to_simplex_preserves_vector_already_on_simplex(optimization_mod, already_simplex_vector, assert_allclose):
    proj = optimization_mod.project_to_simplex(already_simplex_vector)
    assert_allclose(proj, already_simplex_vector, atol=1e-12)



def test_project_to_simplex_empty_vector_returns_empty(optimization_mod):
    out = optimization_mod.project_to_simplex(np.array([], dtype=float))
    assert isinstance(out, np.ndarray)
    assert out.shape == (0,)



def test_enforce_box_constraints_clips_elementwise(optimization_mod, box_bounds, assert_allclose):
    lb, ub = box_bounds
    w = np.array([-1.0, 0.2, 1.5], dtype=float)
    clipped = optimization_mod.enforce_box_constraints(w, lb, ub)
    expected = np.array([-0.5, 0.2, 0.9], dtype=float)
    assert_allclose(clipped, expected, atol=1e-12)



def test_enforce_box_constraints_rejects_shape_mismatch(optimization_mod):
    with pytest.raises(optimization_mod.InvalidProblemError, match="same shape"):
        optimization_mod.enforce_box_constraints(
            np.array([0.0, 1.0], dtype=float),
            np.array([0.0], dtype=float),
            np.array([1.0, 1.0], dtype=float),
        )



def test_condition_q_leaves_psd_matrix_usable(optimization_mod, psd_matrix, assert_allclose):
    q_reg, report = optimization_mod._condition_q(
        psd_matrix,
        reg_eps=optimization_mod.DEFAULT_REG_EPS,
        reg_max=optimization_mod.DEFAULT_REG_MAX,
        cond_max=1e12,
    )

    assert q_reg.shape == psd_matrix.shape
    assert report.n == psd_matrix.shape[0]
    assert report.is_psd_raw is True
    assert report.is_convex_usable is True
    assert report.reg_delta >= 0.0
    assert np.min(np.linalg.eigvalsh(q_reg)) >= -optimization_mod.DEFAULT_REG_EPS
    # PSD inputs may still get a tiny floor regularization when lam_min < reg_eps.
    assert_allclose(q_reg, q_reg.T, atol=1e-12)



def test_condition_q_regularizes_small_negative_eigenvalue(optimization_mod):
    q = np.diag(np.array([-1e-9, 2.0], dtype=float))
    q_reg, report = optimization_mod._condition_q(
        q,
        reg_eps=1e-8,
        reg_max=1e-4,
        cond_max=1e12,
    )

    assert report.lam_min_raw < 0.0
    assert report.reg_delta > 0.0
    assert report.is_convex_usable is True
    assert np.min(np.linalg.eigvalsh(q_reg)) >= -1e-8



def test_condition_q_raises_when_too_far_from_psd(optimization_mod, indefinite_matrix):
    with pytest.raises(optimization_mod.ConditioningError, match="too far from PSD"):
        optimization_mod._condition_q(
            indefinite_matrix,
            reg_eps=1e-8,
            reg_max=1e-4,
            cond_max=1e12,
        )



def test_condition_q_raises_when_regularized_matrix_is_too_ill_conditioned(optimization_mod):
    q = np.diag(np.array([1e-16, 1.0], dtype=float))
    with pytest.raises(optimization_mod.ConditioningError, match="ill-conditioned"):
        optimization_mod._condition_q(
            q,
            reg_eps=1e-8,
            reg_max=1e-4,
            cond_max=1e6,
        )



def test_compute_kkt_residuals_exact_unconstrained_solution_is_certified(optimization_mod):
    q = np.eye(2, dtype=float)
    c = np.array([-1.0, -2.0], dtype=float)
    w_opt = np.array([1.0, 2.0], dtype=float)

    kkt = optimization_mod.compute_kkt_residuals(
        Q=q,
        c=c,
        C=None,
        d=None,
        E=None,
        f=None,
        lb=None,
        ub=None,
        w_opt=w_opt,
    )

    assert kkt.is_certified is True
    assert kkt.max_residual == pytest.approx(0.0, abs=1e-12)
    as_dict = kkt.as_dict()
    assert as_dict["is_certified"] is True
    assert as_dict["max_residual"] == pytest.approx(0.0, abs=1e-12)



def test_compute_kkt_residuals_detects_missing_duals_as_infinite_stationarity(optimization_mod):
    q = np.eye(2, dtype=float)
    c = np.zeros(2, dtype=float)
    e = np.array([[1.0, 1.0]], dtype=float)
    f = np.array([1.0], dtype=float)
    w_opt = np.array([0.5, 0.5], dtype=float)

    kkt = optimization_mod.compute_kkt_residuals(
        Q=q,
        c=c,
        C=None,
        d=None,
        E=e,
        f=f,
        lb=None,
        ub=None,
        w_opt=w_opt,
        lambda_eq=None,
    )

    assert math.isinf(kkt.r_stationarity)
    assert kkt.is_certified is False



def test_compute_kkt_residuals_detects_primal_and_dual_violations(optimization_mod):
    q = np.eye(2, dtype=float)
    c = np.zeros(2, dtype=float)
    C = np.array([[1.0, 1.0]], dtype=float)
    d = np.array([1.0], dtype=float)
    lb = np.array([0.0, 0.0], dtype=float)
    ub = np.array([1.0, 1.0], dtype=float)
    w_opt = np.array([0.8, 0.6], dtype=float)  # violates x1 + x2 <= 1

    kkt = optimization_mod.compute_kkt_residuals(
        Q=q,
        c=c,
        C=C,
        d=d,
        E=None,
        f=None,
        lb=lb,
        ub=ub,
        w_opt=w_opt,
        lambda_ineq=np.array([-0.5], dtype=float),
        alpha_lb=np.array([0.0, 0.0], dtype=float),
        beta_ub=np.array([0.0, 0.0], dtype=float),
    )

    assert kkt.r_ineq_feas == pytest.approx(0.4, abs=1e-12)
    assert kkt.r_dual_feas_ineq == pytest.approx(0.5, abs=1e-12)
    assert kkt.is_certified is False



def test_compute_kkt_residuals_turnover_exact_case_is_certified(optimization_mod):
    q = np.eye(2, dtype=float)
    w_opt = np.array([0.6, 0.4], dtype=float)
    c = np.array([-1.6, 0.6], dtype=float)

    turnover_meta = {
        "w_prev": np.array([0.5, 0.5], dtype=float),
        "turnover_penalty": 1.0,
        "t_opt": np.array([0.1, 0.1], dtype=float),
        "lambda_turnover_pos": np.array([1.0, 0.0], dtype=float),
        "lambda_turnover_neg": np.array([0.0, 1.0], dtype=float),
        "lambda_turnover_t_nonneg": np.array([0.0, 0.0], dtype=float),
    }

    kkt = optimization_mod.compute_kkt_residuals(
        Q=q,
        c=c,
        C=None,
        d=None,
        E=None,
        f=None,
        lb=None,
        ub=None,
        w_opt=w_opt,
        turnover_meta=turnover_meta,
    )

    assert kkt.is_certified is True
    assert kkt.r_turnover_feas_pos == pytest.approx(0.0, abs=1e-12)
    assert kkt.r_turnover_feas_neg == pytest.approx(0.0, abs=1e-12)
    assert kkt.r_turnover_stationarity_t == pytest.approx(0.0, abs=1e-12)



def test_compute_kkt_residuals_requires_complete_turnover_meta(optimization_mod):
    with pytest.raises(optimization_mod.InvalidProblemError, match="must include w_prev and t_opt"):
        optimization_mod.compute_kkt_residuals(
            Q=np.eye(2, dtype=float),
            c=np.zeros(2, dtype=float),
            C=None,
            d=None,
            E=None,
            f=None,
            lb=None,
            ub=None,
            w_opt=np.zeros(2, dtype=float),
            turnover_meta={"turnover_penalty": 1.0},
        )



def test_kkt_diagnostics_is_certified_false_when_any_residual_exceeds_tol(optimization_mod):
    kkt = optimization_mod.KKTDiagnostics(
        r_eq_feas=0.0,
        r_ineq_feas=0.0,
        r_lb_feas=0.0,
        r_ub_feas=0.0,
        r_dual_feas_ineq=0.0,
        r_dual_feas_lb=0.0,
        r_dual_feas_ub=0.0,
        r_stationarity=1e-3,
        r_complementarity_ineq=0.0,
        r_complementarity_lb=0.0,
        r_complementarity_ub=0.0,
        kkt_tol=1e-6,
    )

    assert kkt.max_residual == pytest.approx(1e-3)
    assert kkt.is_certified is False



def test_build_turnover_epigraph_constructs_expected_shape_when_cvxpy_available(optimization_mod):
    if getattr(optimization_mod, "cp", None) is None:
        pytest.skip("cvxpy not available in this environment")

    cp = optimization_mod.cp
    w = cp.Variable(3, name="w")
    epigraph = optimization_mod.build_turnover_epigraph(
        w_var=w,
        w_prev=np.array([0.2, 0.3, 0.5], dtype=float),
        turnover_penalty=0.25,
    )

    assert tuple(epigraph.t_var.shape) == (3,)
    assert len(epigraph.constraints) == 3
    assert epigraph.constraint_pos is epigraph.constraints[0]
    assert epigraph.constraint_neg is epigraph.constraints[1]
    assert epigraph.constraint_t_nonneg is epigraph.constraints[2]



def test_build_turnover_epigraph_rejects_nonpositive_penalty(optimization_mod):
    if getattr(optimization_mod, "cp", None) is None:
        pytest.skip("cvxpy not available in this environment")

    cp = optimization_mod.cp
    w = cp.Variable(2)
    with pytest.raises(optimization_mod.InvalidProblemError, match="must be > 0"):
        optimization_mod.build_turnover_epigraph(
            w_var=w,
            w_prev=np.array([0.5, 0.5], dtype=float),
            turnover_penalty=0.0,
        )
