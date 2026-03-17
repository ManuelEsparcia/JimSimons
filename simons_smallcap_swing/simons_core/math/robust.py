from __future__ import annotations

"""
simons_core.math.robust
=======================

Institutional univariate robust-statistics primitives for noisy, heavy-tailed
financial inputs.

Scope
-----
This module is deliberately narrow and explicit:

- robust center/scale via median + MAD;
- robust z-scoring built on the same center/scale contract;
- symmetric trimmed mean;
- quantile winsorization;
- Huber reweighting / influence helpers.

Non-goals
---------
This is *not* a multivariate robust-statistics framework. It does not implement
robust covariance, robust regression, depth-based methods, robust PCA, or
contamination models beyond the univariate utilities documented here.

Institutional design rules
--------------------------
- NaN/Inf policy is explicit and deterministic.
- Degenerate scales are handled intentionally rather than left to NumPy divide-
  by-zero behaviour.
- MAD "raw" and MAD scaled for Gaussian consistency are distinct modes.
- Vector outputs preserve the shape of the input and write ``NaN`` in positions
  corresponding to non-finite observations under ``nan_policy='omit'``.
- Small-sample effects are surfaced via metadata when low-cost to do so.
"""

from dataclasses import dataclass
import argparse
import json
import math
from typing import Any, Final, Literal, TypeAlias

import numpy as np


__all__ = [
    "MAD_NORMAL_CONSISTENCY",
    "DEFAULT_EPS",
    "DEFAULT_HUBER_C",
    "DEFAULT_TRIM_P",
    "DEFAULT_WINSOR_LOWER_Q",
    "DEFAULT_WINSOR_UPPER_Q",
    "BREAKDOWN_POINT_MEDIAN",
    "BREAKDOWN_POINT_MAD",
    "ARE_NORMAL_MEDIAN",
    "RobustError",
    "InvalidNanPolicyError",
    "InvalidConsistencyError",
    "DegenerateScaleWarningInfo",
    "finite_view",
    "robust_center_scale",
    "robust_zscore",
    "trimmed_mean",
    "winsorize",
    "huber_weights",
    "huber_psi",
    "describe_robust_properties",
]


ArrayLike: TypeAlias = np.ndarray | list[float] | tuple[float, ...]
NanPolicy: TypeAlias = Literal["omit", "raise"]
Consistency: TypeAlias = Literal["raw", "normal"]

MAD_NORMAL_CONSISTENCY: Final[float] = 1.482602218505602
DEFAULT_EPS: Final[float] = 1e-12
DEFAULT_HUBER_C: Final[float] = 1.345
DEFAULT_TRIM_P: Final[float] = 0.05
DEFAULT_WINSOR_LOWER_Q: Final[float] = 0.05
DEFAULT_WINSOR_UPPER_Q: Final[float] = 0.95
BREAKDOWN_POINT_MEDIAN: Final[float] = 0.50
BREAKDOWN_POINT_MAD: Final[float] = 0.50
ARE_NORMAL_MEDIAN: Final[float] = 0.637


class RobustError(RuntimeError):
    """Base class for robust-statistics layer failures."""


class InvalidNanPolicyError(RobustError, ValueError):
    """Raised when ``nan_policy`` is outside the supported contract."""


class InvalidConsistencyError(RobustError, ValueError):
    """Raised when the requested MAD consistency mode is unknown."""


@dataclass(frozen=True)
class DegenerateScaleWarningInfo:
    """Structured metadata describing scale degeneracy handling."""

    mad_raw: float
    scale_before_floor: float
    eps_used: float
    treated_as_degenerate: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "mad_raw": float(self.mad_raw),
            "scale_before_floor": float(self.scale_before_floor),
            "eps_used": float(self.eps_used),
            "treated_as_degenerate": bool(self.treated_as_degenerate),
        }


@dataclass(frozen=True)
class _FiniteView:
    """Internal normalized finite-view representation."""

    x: np.ndarray
    mask: np.ndarray
    values: np.ndarray
    nonfinite_count: int


def _as_1d_float_array(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    else:
        arr = arr.reshape(-1)
    return arr


def _validate_nan_policy(nan_policy: str) -> NanPolicy:
    if nan_policy not in {"omit", "raise"}:
        raise InvalidNanPolicyError(
            f"Unsupported nan_policy={nan_policy!r}. Expected 'omit' or 'raise'."
        )
    return nan_policy  # type: ignore[return-value]


def _validate_consistency(consistency: str) -> Consistency:
    if consistency not in {"raw", "normal"}:
        raise InvalidConsistencyError(
            f"Unsupported consistency={consistency!r}. Expected 'raw' or 'normal'."
        )
    return consistency  # type: ignore[return-value]


def finite_view(x: ArrayLike, *, nan_policy: NanPolicy = "omit") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the canonical finite view used by all robust primitives.

    Parameters
    ----------
    x:
        Input vector. Any array-like object is coerced to a flat float array.
    nan_policy:
        ``'omit'`` ignores non-finite values for aggregation while preserving
        position semantics in vector outputs. ``'raise'`` fails on the first
        presence of NaN/Inf.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(x_norm, finite_mask, x_valid)``.

    Raises
    ------
    ValueError
        If ``nan_policy='raise'`` and non-finite values are present, or if no
        finite observations remain after filtering.
    """
    view = _finite_view_internal(x, nan_policy=nan_policy)
    return view.x, view.mask, view.values


def _finite_view_internal(x: ArrayLike, *, nan_policy: NanPolicy = "omit") -> _FiniteView:
    nan_policy = _validate_nan_policy(nan_policy)
    x_norm = _as_1d_float_array(x)
    mask = np.isfinite(x_norm)
    nonfinite_count = int(mask.size - int(mask.sum()))

    if nan_policy == "raise" and nonfinite_count:
        raise ValueError("Input contains NaN or infinite values.")

    values = x_norm[mask]
    if values.size == 0:
        raise ValueError("No finite observations available.")

    return _FiniteView(
        x=x_norm,
        mask=mask,
        values=values,
        nonfinite_count=nonfinite_count,
    )


def _small_sample_note(n_valid: int, *, threshold: int = 20) -> str | None:
    if n_valid < threshold:
        return (
            "Small-sample regime: quantiles / trimmed statistics may be unstable "
            f"for n_valid={n_valid} < {threshold}."
        )
    return None


def _mad_scale(mad_raw: float, consistency: Consistency) -> float:
    if consistency == "raw":
        return float(mad_raw)
    if consistency == "normal":
        return float(MAD_NORMAL_CONSISTENCY * mad_raw)
    raise InvalidConsistencyError(
        f"Unsupported consistency={consistency!r}. Expected 'raw' or 'normal'."
    )


def robust_center_scale(
    x: ArrayLike,
    *,
    nan_policy: NanPolicy = "omit",
    consistency: Consistency = "normal",
    eps: float = DEFAULT_EPS,
) -> tuple[float, float, dict[str, Any]]:
    """
    Robust center/scale using median and MAD.

    Mathematical definition
    -----------------------
    For finite observations ``x_v``:

    ``center = median(x_v)``

    ``mad_raw = median(|x_v - center|)``

    If ``consistency='normal'``, the scale is multiplied by
    ``1 / Phi^{-1}(0.75) = 1.482602218505602`` so that it is consistent with the
    Gaussian standard deviation. The returned scale is floored by ``eps`` for
    algebraic stability.

    Returns
    -------
    tuple[float, float, dict[str, Any]]
        ``(center, scale, meta)`` where ``meta`` includes robustness and
        degeneracy diagnostics.
    """
    if eps <= 0:
        raise ValueError("eps must be strictly positive.")

    consistency = _validate_consistency(consistency)
    view = _finite_view_internal(x, nan_policy=nan_policy)

    center = float(np.median(view.values))
    abs_dev = np.abs(view.values - center)
    mad_raw = float(np.median(abs_dev))
    scale_before_floor = _mad_scale(mad_raw, consistency)
    treated_as_degenerate = bool(scale_before_floor < eps)
    scale = float(max(scale_before_floor, eps))

    deg_info = DegenerateScaleWarningInfo(
        mad_raw=mad_raw,
        scale_before_floor=scale_before_floor,
        eps_used=float(eps),
        treated_as_degenerate=treated_as_degenerate,
    )

    meta: dict[str, Any] = {
        "n_valid": int(view.values.size),
        "n_nonfinite": int(view.nonfinite_count),
        "nan_policy": nan_policy,
        "consistency": consistency,
        "center_estimator": "median",
        "scale_estimator": "mad_raw" if consistency == "raw" else "mad_normal",
        "breakdown_point_center": BREAKDOWN_POINT_MEDIAN,
        "breakdown_point_scale": BREAKDOWN_POINT_MAD,
        "are_normal_center": ARE_NORMAL_MEDIAN,
        "mad_raw": float(mad_raw),
        "scale_before_floor": float(scale_before_floor),
        "scale": float(scale),
        "eps_used": float(eps),
        "degenerate_scale": bool(treated_as_degenerate),
        "degeneracy": deg_info.as_dict(),
        "small_sample_note": _small_sample_note(int(view.values.size)),
    }
    return center, scale, meta


def robust_zscore(
    x: ArrayLike,
    *,
    nan_policy: NanPolicy = "omit",
    consistency: Consistency = "normal",
    eps: float = DEFAULT_EPS,
    clip: float | None = None,
) -> np.ndarray:
    """
    Robust z-score using median/MAD center-scale.

    Behavioural contract
    --------------------
    - Non-finite input positions become ``NaN`` in the output under
      ``nan_policy='omit'``.
    - If the robust scale collapses below ``eps``, finite observations are
      assigned zero rather than divided by an effectively singular scale.
    - Optional symmetric clipping can be applied to the final robust z-scores.
    """
    if clip is not None and clip <= 0:
        raise ValueError("clip must be strictly positive when provided.")

    view = _finite_view_internal(x, nan_policy=nan_policy)
    out = np.full_like(view.x, np.nan, dtype=float)

    center, scale, meta = robust_center_scale(
        view.x,
        nan_policy=nan_policy,
        consistency=consistency,
        eps=eps,
    )

    xv = view.values
    if bool(meta["degenerate_scale"]):
        z = np.zeros_like(xv, dtype=float)
    else:
        z = (xv - center) / scale

    if clip is not None:
        z = np.clip(z, -clip, clip)

    out[view.mask] = z
    return out


def trimmed_mean(
    x: ArrayLike,
    p: float = DEFAULT_TRIM_P,
    *,
    nan_policy: NanPolicy = "omit",
) -> tuple[float, dict[str, Any]]:
    """
    Symmetric trimmed mean.

    Parameters
    ----------
    x:
        Input vector.
    p:
        Requested trimming fraction per tail. Must satisfy ``0 <= p < 0.5``.
    nan_policy:
        NaN/Inf handling policy.

    Returns
    -------
    tuple[float, dict[str, Any]]
        Trimmed mean and metadata including effective trim fraction.
    """
    if not (0.0 <= p < 0.5):
        raise ValueError("p must satisfy 0 <= p < 0.5.")

    view = _finite_view_internal(x, nan_policy=nan_policy)
    xv = np.sort(view.values)
    n = int(xv.size)
    k = int(math.floor(p * n))

    kept = n - 2 * k
    if kept <= 0:
        raise ValueError("Trim level leaves no observations.")

    value = float(np.mean(xv[k : n - k]))
    alpha_eff = float(k / n) if n else 0.0
    meta: dict[str, Any] = {
        "n_valid": n,
        "n_nonfinite": int(view.nonfinite_count),
        "nan_policy": nan_policy,
        "p_requested": float(p),
        "k_trim": int(k),
        "alpha_eff": alpha_eff,
        "n_kept": int(kept),
        "breakdown_point": float(alpha_eff),
        "small_sample_note": _small_sample_note(n),
    }
    return value, meta


def winsorize(
    x: ArrayLike,
    lower_q: float = DEFAULT_WINSOR_LOWER_Q,
    upper_q: float = DEFAULT_WINSOR_UPPER_Q,
    *,
    nan_policy: NanPolicy = "omit",
) -> np.ndarray:
    """
    Quantile winsorization with shape preservation.

    Non-finite input positions are returned as ``NaN`` under ``omit``. The
    sample size is preserved for finite positions.
    """
    if not (0.0 <= lower_q < upper_q <= 1.0):
        raise ValueError("Require 0 <= lower_q < upper_q <= 1.")

    view = _finite_view_internal(x, nan_policy=nan_policy)
    out = np.full_like(view.x, np.nan, dtype=float)

    lo = float(np.quantile(view.values, lower_q))
    hi = float(np.quantile(view.values, upper_q))
    out[view.mask] = np.clip(view.values, lo, hi)
    return out


def huber_psi(residuals: ArrayLike, c: float = DEFAULT_HUBER_C) -> np.ndarray:
    """
    Huber influence function applied elementwise.

    ``psi(r) = r`` for ``|r| <= c`` and ``c * sign(r)`` otherwise.
    """
    if c <= 0:
        raise ValueError("c must be strictly positive.")

    r = _as_1d_float_array(residuals)
    a = np.abs(r)
    out = np.array(r, copy=True, dtype=float)
    mask = a > c
    out[mask] = c * np.sign(r[mask])
    return out


def huber_weights(
    residuals: ArrayLike,
    c: float = DEFAULT_HUBER_C,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Compute classical Huber weights.

    For standardized residuals ``r`` and threshold ``c > 0``:

    ``w = min(1, c / |r|)``
    """
    if c <= 0:
        raise ValueError("c must be strictly positive.")

    r = _as_1d_float_array(residuals)
    a = np.abs(r)
    w = np.ones_like(a, dtype=float)
    mask = a > c
    # Safe because the mask excludes zeros.
    w[mask] = c / a[mask]

    meta: dict[str, Any] = {
        "c": float(c),
        "psi": "huber",
        "efficiency_note": "classical high-efficiency choice under Gaussian residuals",
        "n_obs": int(r.size),
        "n_downweighted": int(mask.sum()),
        "fraction_downweighted": float(mask.mean()) if r.size else 0.0,
    }
    return w, meta


def describe_robust_properties() -> tuple[dict[str, Any], ...]:
    """Return a compact machine-readable summary of key robust estimators."""
    return (
        {
            "estimator": "median",
            "breakdown_point": BREAKDOWN_POINT_MEDIAN,
            "efficiency_note": "~0.637 ARE under Gaussian model",
            "influence": "bounded local / sign-like",
        },
        {
            "estimator": "mad_raw",
            "breakdown_point": BREAKDOWN_POINT_MAD,
            "efficiency_note": "robust scale, non-smooth",
            "influence": "bounded / non-smooth",
        },
        {
            "estimator": "trimmed_mean(p)",
            "breakdown_point": "~p",
            "efficiency_note": "high when p small under light tails",
            "influence": "clipped tails",
        },
        {
            "estimator": "winsorize",
            "breakdown_point": "n/a",
            "efficiency_note": "stabilizer, preserves n",
            "influence": "bounded through clipping",
        },
        {
            "estimator": "huber_weights",
            "breakdown_point": "depends on full estimator context",
            "efficiency_note": "classical high efficiency near Gaussian residuals",
            "influence": "psi(r)=clip(r,-c,c)",
        },
    )


# ---------------------------------------------------------------------------
# Self-test / lightweight CLI
# ---------------------------------------------------------------------------

def _assert_close(actual: np.ndarray | float, expected: np.ndarray | float, tol: float = 1e-12) -> None:
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        if not np.allclose(np.asarray(actual), np.asarray(expected), atol=tol, rtol=0.0, equal_nan=True):
            raise AssertionError(f"Arrays differ.\nactual={actual!r}\nexpected={expected!r}")
        return
    if not math.isclose(float(actual), float(expected), abs_tol=tol, rel_tol=0.0):
        raise AssertionError(f"Values differ. actual={actual!r}, expected={expected!r}")


def _run_self_test() -> dict[str, Any]:
    x = np.array([1.0, 2.0, 2.0, 3.0, 100.0])
    center, scale, meta = robust_center_scale(x, consistency="raw")
    _assert_close(center, 2.0)
    _assert_close(scale, 1.0)
    assert meta["breakdown_point_center"] == 0.50
    assert meta["breakdown_point_scale"] == 0.50

    z = robust_zscore(np.array([1.0, 1.0, 1.0, np.nan]))
    _assert_close(z[:3], np.array([0.0, 0.0, 0.0]))
    assert math.isnan(float(z[3]))

    tm, tm_meta = trimmed_mean(np.array([1.0, 2.0, 3.0, 100.0]), p=0.25)
    _assert_close(tm, 2.5)
    _assert_close(tm_meta["alpha_eff"], 0.25)

    w = winsorize(np.array([1.0, 2.0, 3.0, 100.0]), 0.0, 0.75)
    _assert_close(w, np.array([1.0, 2.0, 3.0, 27.25]))

    hw, hw_meta = huber_weights(np.array([0.0, 1.0, 2.0]), c=1.0)
    _assert_close(hw, np.array([1.0, 1.0, 0.5]))
    assert hw_meta["n_downweighted"] == 1

    psi = huber_psi(np.array([-2.0, -0.5, 0.0, 0.5, 2.0]), c=1.0)
    _assert_close(psi, np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))

    try:
        robust_center_scale(np.array([np.nan, np.inf]))
    except ValueError:
        pass
    else:  # pragma: no cover - sanity guard
        raise AssertionError("Expected ValueError for all non-finite data.")

    return {
        "ok": True,
        "tests": [
            "robust_center_scale_clean_and_outlier_resistant",
            "robust_zscore_degenerate_scale_and_nan_preservation",
            "trimmed_mean_effective_alpha",
            "winsorize_quantile_clipping",
            "huber_weights_definition",
            "huber_psi_definition",
            "all_nonfinite_raises",
        ],
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Self-test utilities for simons_core.math.robust",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic internal checks and print JSON.",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print the compact robust-properties table as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    payload: dict[str, Any] = {}
    if args.self_test:
        payload["self_test"] = _run_self_test()
    if args.describe:
        payload["properties"] = list(describe_robust_properties())

    if not payload:
        parser.print_help()
        return 0

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
