from __future__ import annotations

import json
import math

import numpy as np
import pytest


pytestmark = pytest.mark.math


def test_finite_view_omit_filters_nonfinite_and_preserves_order(
    robust_mod,
    vector_with_nonfinite,
    assert_allclose,
):
    x_norm, mask, values = robust_mod.finite_view(vector_with_nonfinite, nan_policy="omit")

    assert_allclose(x_norm, vector_with_nonfinite)
    assert mask.dtype == bool
    assert mask.tolist() == [True, False, True, False, False, True]
    assert_allclose(values, np.array([1.0, 2.0, 3.0]))


def test_finite_view_raise_on_nonfinite(robust_mod, vector_with_nonfinite):
    with pytest.raises(ValueError, match="NaN or infinite"):
        robust_mod.finite_view(vector_with_nonfinite, nan_policy="raise")


def test_finite_view_all_nonfinite_raises(robust_mod, all_nonfinite_vector):
    with pytest.raises(ValueError, match="No finite observations"):
        robust_mod.finite_view(all_nonfinite_vector, nan_policy="omit")


def test_invalid_nan_policy_raises_typed_error(robust_mod, clean_vector):
    with pytest.raises(robust_mod.InvalidNanPolicyError):
        robust_mod.finite_view(clean_vector, nan_policy="drop")


def test_invalid_consistency_raises_typed_error(robust_mod, clean_vector):
    with pytest.raises(robust_mod.InvalidConsistencyError):
        robust_mod.robust_center_scale(clean_vector, consistency="gaussian")


def test_robust_center_scale_raw_is_outlier_resistant(
    robust_mod,
    outlier_vector,
):
    center, scale, meta = robust_mod.robust_center_scale(outlier_vector, consistency="raw")

    assert math.isclose(center, 10.05, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(scale, 0.1, rel_tol=0.0, abs_tol=1e-12)
    assert meta["center_estimator"] == "median"
    assert meta["scale_estimator"] == "mad_raw"
    assert meta["n_valid"] == 6
    assert meta["n_nonfinite"] == 0
    assert meta["degenerate_scale"] is False
    assert math.isclose(meta["breakdown_point_center"], 0.5, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(meta["breakdown_point_scale"], 0.5, rel_tol=0.0, abs_tol=0.0)


def test_robust_center_scale_normal_applies_consistency_factor(
    robust_mod,
    clean_vector,
):
    center, scale, meta = robust_mod.robust_center_scale(clean_vector, consistency="normal")

    assert math.isclose(center, 3.0, rel_tol=0.0, abs_tol=1e-12)
    expected = robust_mod.MAD_NORMAL_CONSISTENCY * 1.0
    assert math.isclose(scale, expected, rel_tol=0.0, abs_tol=1e-12)
    assert meta["scale_estimator"] == "mad_normal"
    assert math.isclose(meta["mad_raw"], 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(meta["scale_before_floor"], expected, rel_tol=0.0, abs_tol=1e-12)


def test_robust_center_scale_constant_vector_flags_degeneracy(
    robust_mod,
    constant_vector,
):
    center, scale, meta = robust_mod.robust_center_scale(constant_vector)

    assert math.isclose(center, 7.5, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(scale, robust_mod.DEFAULT_EPS, rel_tol=0.0, abs_tol=0.0)
    assert meta["degenerate_scale"] is True
    assert meta["degeneracy"]["treated_as_degenerate"] is True
    assert math.isclose(meta["degeneracy"]["mad_raw"], 0.0, rel_tol=0.0, abs_tol=1e-12)


def test_robust_center_scale_requires_positive_eps(robust_mod, clean_vector):
    with pytest.raises(ValueError, match="eps must be strictly positive"):
        robust_mod.robust_center_scale(clean_vector, eps=0.0)


def test_robust_zscore_omit_preserves_nonfinite_positions(
    robust_mod,
    vector_with_nonfinite,
    assert_allclose,
):
    z = robust_mod.robust_zscore(vector_with_nonfinite, nan_policy="omit")

    expected_scale = robust_mod.MAD_NORMAL_CONSISTENCY * 1.0
    expected = np.array(
        [
            (1.0 - 2.0) / expected_scale,
            np.nan,
            0.0,
            np.nan,
            np.nan,
            (3.0 - 2.0) / expected_scale,
        ],
        dtype=float,
    )
    assert z.shape == vector_with_nonfinite.shape
    assert_allclose(z, expected)


def test_robust_zscore_constant_vector_returns_zeros(
    robust_mod,
    constant_vector,
    assert_allclose,
):
    z = robust_mod.robust_zscore(constant_vector)
    assert_allclose(z, np.zeros_like(constant_vector))


def test_robust_zscore_clip_is_symmetric(
    robust_mod,
    signed_residuals,
):
    z = robust_mod.robust_zscore(signed_residuals, consistency="raw", clip=1.25)

    finite = z[np.isfinite(z)]
    assert np.all(finite <= 1.25 + 1e-12)
    assert np.all(finite >= -1.25 - 1e-12)
    assert math.isclose(float(finite[0]), -1.25, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(finite[-1]), 1.25, rel_tol=0.0, abs_tol=1e-12)


def test_robust_zscore_rejects_nonpositive_clip(robust_mod, clean_vector):
    with pytest.raises(ValueError, match="clip must be strictly positive"):
        robust_mod.robust_zscore(clean_vector, clip=0.0)


def test_trimmed_mean_matches_hand_calculation(robust_mod):
    x = np.array([1.0, 2.0, 3.0, 100.0], dtype=float)
    value, meta = robust_mod.trimmed_mean(x, p=0.25)

    assert math.isclose(value, 2.5, rel_tol=0.0, abs_tol=1e-12)
    assert meta["k_trim"] == 1
    assert meta["n_kept"] == 2
    assert math.isclose(meta["alpha_eff"], 0.25, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(meta["breakdown_point"], 0.25, rel_tol=0.0, abs_tol=1e-12)


def test_trimmed_mean_p_zero_reduces_to_mean(robust_mod, clean_vector):
    value, meta = robust_mod.trimmed_mean(clean_vector, p=0.0)

    assert math.isclose(value, float(np.mean(clean_vector)), rel_tol=0.0, abs_tol=1e-12)
    assert meta["k_trim"] == 0
    assert meta["n_kept"] == clean_vector.size


def test_trimmed_mean_rejects_invalid_p(robust_mod, clean_vector):
    with pytest.raises(ValueError, match="0 <= p < 0.5"):
        robust_mod.trimmed_mean(clean_vector, p=0.5)


def test_winsorize_quantile_clipping_matches_self_test(
    robust_mod,
    assert_allclose,
):
    x = np.array([1.0, 2.0, 3.0, 100.0], dtype=float)
    out = robust_mod.winsorize(x, 0.0, 0.75)
    assert_allclose(out, np.array([1.0, 2.0, 3.0, 27.25]))


def test_winsorize_preserves_nonfinite_positions(
    robust_mod,
    vector_with_nonfinite,
):
    out = robust_mod.winsorize(vector_with_nonfinite, 0.0, 1.0)

    assert out.shape == vector_with_nonfinite.shape
    assert np.isnan(out[1]) and np.isnan(out[3]) and np.isnan(out[4])
    assert np.isfinite(out[0]) and np.isfinite(out[2]) and np.isfinite(out[5])


def test_winsorize_rejects_invalid_quantile_order(robust_mod, clean_vector):
    with pytest.raises(ValueError, match="lower_q < upper_q"):
        robust_mod.winsorize(clean_vector, 0.8, 0.2)


def test_huber_psi_is_odd_and_clipped(
    robust_mod,
    signed_residuals,
    assert_allclose,
):
    psi = robust_mod.huber_psi(signed_residuals, c=1.0)

    expected = np.array([-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0], dtype=float)
    assert_allclose(psi, expected)
    assert_allclose(psi, -robust_mod.huber_psi(-signed_residuals, c=1.0))


def test_huber_weights_match_definition(
    robust_mod,
    signed_residuals,
    assert_allclose,
):
    w, meta = robust_mod.huber_weights(signed_residuals, c=1.0)

    expected = np.array([0.1, 0.5, 1.0, 1.0, 1.0, 0.5, 0.1], dtype=float)
    assert_allclose(w, expected)
    assert meta["n_obs"] == signed_residuals.size
    assert meta["n_downweighted"] == 4
    assert math.isclose(meta["fraction_downweighted"], 4 / 7, rel_tol=0.0, abs_tol=1e-12)


def test_huber_functions_require_positive_c(robust_mod, clean_vector):
    with pytest.raises(ValueError, match="strictly positive"):
        robust_mod.huber_psi(clean_vector, c=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        robust_mod.huber_weights(clean_vector, c=0.0)


def test_describe_robust_properties_has_expected_estimators(robust_mod):
    props = robust_mod.describe_robust_properties()

    assert isinstance(props, tuple)
    estimators = {entry["estimator"] for entry in props}
    assert {"median", "mad_raw", "trimmed_mean(p)", "winsorize", "huber_weights"}.issubset(estimators)


def test_main_self_test_prints_valid_json(robust_mod, capsys):
    rc = robust_mod.main(["--self-test"])
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["self_test"]["ok"] is True
    assert "robust_center_scale_clean_and_outlier_resistant" in payload["self_test"]["tests"]


def test_main_describe_prints_valid_json(robust_mod, capsys):
    rc = robust_mod.main(["--describe"])
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert isinstance(payload["properties"], list)
    assert any(item["estimator"] == "median" for item in payload["properties"])
