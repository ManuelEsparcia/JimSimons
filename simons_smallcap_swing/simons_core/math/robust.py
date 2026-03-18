"""
simons_core/math/robust.py — Robust univariate statistics.

    robust_center_scale   median + MAD (β=0.50, ARE≈0.637 under N)
    robust_zscore         (x − median) / MAD_σ, degenerate → 0, NaN preserved
    trimmed_mean          bilateral α-trimmed mean (β=α)
    winsorize             quantile clipping, preserves sample size + NaN positions
    huber_weights         w = min(1, c/|r|), c=1.345 → 95% efficiency Gaussian

All functions: finite-view only, NaN preserved in vector outputs, metadata returned.
"""
from __future__ import annotations
from typing import Any
import numpy as np

MAD_NORMAL_CONSISTENCY = 1.482602218505602  # 1 / Φ⁻¹(0.75)
DEFAULT_HUBER_C = 1.345


def _finite_view(x, nan_policy: str = "omit"):
    """(x_full, mask, x_valid).  Raises on all-NaN or if nan_policy='raise'."""
    x = np.asarray(x, dtype=float).ravel()
    mask = np.isfinite(x)
    if nan_policy == "raise" and not mask.all():
        raise ValueError("Input contains NaN/Inf (nan_policy='raise')")
    xv = x[mask]
    if xv.size == 0:
        raise ValueError("No finite observations")
    return x, mask, xv


def robust_center_scale(x, *, nan_policy="omit", consistency="normal", eps=1e-12):
    """Median + MAD.  consistency='raw' | 'normal' (×1.4826).  β=0.50."""
    _, _, xv = _finite_view(x, nan_policy)
    center = float(np.median(xv))
    mad_raw = float(np.median(np.abs(xv - center)))
    if consistency == "normal":
        scale = MAD_NORMAL_CONSISTENCY * mad_raw
    elif consistency == "raw":
        scale = mad_raw
    else:
        raise ValueError(f"Unknown consistency: {consistency!r}")
    scale = max(scale, eps)
    meta = {"n_valid": int(xv.size), "center_estimator": "median",
            "scale_estimator": f"mad_{consistency}", "mad_raw": mad_raw,
            "breakdown_point_center": 0.50, "breakdown_point_scale": 0.50,
            "are_normal_center": 0.637 if consistency == "normal" else None, "eps_used": float(eps)}
    return center, scale, meta


def robust_zscore(x, *, nan_policy="omit", consistency="normal", eps=1e-12, clip=None):
    """z = (x − median) / MAD_σ.  Degenerate scale → z=0.  NaN preserved."""
    x_full, mask, _ = _finite_view(x, nan_policy)
    center, scale, _ = robust_center_scale(x_full, nan_policy=nan_policy, consistency=consistency, eps=eps)
    out = np.full_like(x_full, np.nan, dtype=float)
    out[mask] = 0.0 if scale <= eps else (x_full[mask] - center) / scale
    if clip is not None:
        fm = np.isfinite(out)
        out[fm] = np.clip(out[fm], -clip, clip)
    return out


def trimmed_mean(x, p=0.05, *, nan_policy="omit"):
    """Bilateral α-trimmed mean.  k=⌊p·n⌋ removed each tail.  β=p."""
    if not (0.0 <= p < 0.5):
        raise ValueError(f"p must satisfy 0 ≤ p < 0.5, got {p}")
    _, _, xv = _finite_view(x, nan_policy)
    xs = np.sort(xv)
    n = xs.size
    k = int(np.floor(p * n))
    if n - 2 * k <= 0:
        raise ValueError(f"Trim p={p} leaves no observations (n={n}, k={k})")
    val = float(np.mean(xs[k:n - k]))
    meta = {"n_valid": int(n), "k_trim": int(k), "p_requested": float(p),
            "alpha_eff": float(k / n), "breakdown_point": float(p)}
    return val, meta


def winsorize(x, lower_q=0.05, upper_q=0.95, *, nan_policy="omit"):
    """Clip at quantiles; preserves sample size and NaN positions."""
    if not (0.0 <= lower_q < upper_q <= 1.0):
        raise ValueError(f"Require 0 ≤ lower_q < upper_q ≤ 1, got ({lower_q}, {upper_q})")
    x_full, mask, xv = _finite_view(x, nan_policy)
    lo, hi = float(np.quantile(xv, lower_q)), float(np.quantile(xv, upper_q))
    out = np.full_like(x_full, np.nan, dtype=float)
    out[mask] = np.clip(xv, lo, hi)
    return out


def huber_weights(residuals, c=DEFAULT_HUBER_C):
    """w = min(1, c/|r|).  ψ_Huber(r)=r if |r|≤c, c·sign(r) otherwise.  c=1.345→95% eff."""
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}")
    r = np.asarray(residuals, dtype=float).ravel()
    a = np.abs(r)
    w = np.ones_like(a)
    big = a > c
    w[big] = c / a[big]
    w[~np.isfinite(w)] = 0.0
    meta = {"c": float(c), "psi": "huber", "n_downweighted": int(big.sum()),
            "pct_downweighted": float(big.mean())}
    return w, meta
