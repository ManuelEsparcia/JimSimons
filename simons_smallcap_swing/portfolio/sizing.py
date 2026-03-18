"""
portfolio/sizing.py — Score → raw weights.

Default mode: rank_symmetric
    r_i = PercentileRank(s_i)  ∈ (0,1)
    q_i = 2r_i − 1             ∈ (−1,1)
    → deadband → top-k → vol/liq adjust → normalise to gross/net → cap-and-redistribute

Output:  w_raw with G(w)=g*, N(w)=n*, |w_i|≤p_max
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np


@dataclass(frozen=True)
class SizingConfig:
    mode: str = "rank_symmetric"
    gross_target: float = 2.0       # long + short gross
    net_target: float = 0.0         # net exposure
    deadband: float = 0.0           # |q_i| < deadband → 0
    topk_long: int | None = None
    topk_short: int | None = None
    vol_adjust_beta: float = 0.0    # power for vol adjustment
    liq_adjust_beta: float = 0.0    # power for liquidity adjustment
    max_name_weight: float = 0.10   # cap per name
    min_effective_n: int = 10
    score_clip: float = 5.0
    eps: float = 1e-12


def score_to_intensity(scores: np.ndarray, mode: str, clip: float = 5.0) -> np.ndarray:
    """Transform scores to signed intensity q ∈ (−1, 1)."""
    s = np.asarray(scores, dtype=float).ravel()
    mask = np.isfinite(s)
    q = np.zeros_like(s)

    if mode == "rank_symmetric":
        valid = s[mask]
        if valid.size < 2:
            return q
        from scipy.stats import rankdata
        try:
            ranks = rankdata(valid, method="average") / (valid.size + 1)
        except ImportError:
            order = np.argsort(np.argsort(valid))
            ranks = (order + 1) / (valid.size + 1)
        q[mask] = 2.0 * ranks - 1.0

    elif mode == "zscore_symmetric":
        valid = s[mask]
        mu, sigma = np.mean(valid), np.std(valid)
        if sigma > 1e-15:
            q[mask] = np.clip((valid - mu) / sigma, -clip, clip)

    elif mode == "proportional_signed":
        q[mask] = np.clip(s[mask], -clip, clip)

    else:
        raise ValueError(f"Unknown sizing mode: {mode!r}")
    return q


def apply_deadband(q: np.ndarray, deadband: float) -> np.ndarray:
    """Zero out intensities below deadband threshold."""
    if deadband <= 0:
        return q
    out = q.copy()
    out[np.abs(out) < deadband] = 0.0
    return out


def apply_topk(q: np.ndarray, topk_long: int | None, topk_short: int | None) -> np.ndarray:
    """Keep only top-k long and top-k short positions."""
    out = q.copy()
    if topk_long is not None:
        long_idx = np.where(out > 0)[0]
        if len(long_idx) > topk_long:
            threshold = np.sort(out[long_idx])[-topk_long]
            out[long_idx[out[long_idx] < threshold]] = 0.0
    if topk_short is not None:
        short_idx = np.where(out < 0)[0]
        if len(short_idx) > topk_short:
            threshold = np.sort(out[short_idx])[topk_short]
            out[short_idx[out[short_idx] > threshold]] = 0.0
    return out


def adjust_vol_liq(q: np.ndarray, vol: np.ndarray | None, adv: np.ndarray | None,
                   beta_sigma: float, beta_liq: float, eps: float = 1e-12) -> np.ndarray:
    """Adjust intensity by volatility and liquidity."""
    out = q.copy()
    if vol is not None and beta_sigma > 0:
        v = np.maximum(np.asarray(vol, dtype=float), eps)
        out = out / np.power(v, beta_sigma)
    if adv is not None and beta_liq > 0:
        a = np.maximum(np.asarray(adv, dtype=float), eps)
        ref = np.median(a[a > eps]) if np.any(a > eps) else 1.0
        out = out * np.power(a / ref, beta_liq)
    return out


def normalise_gross_net(q: np.ndarray, g_star: float, n_star: float, eps: float = 1e-12) -> np.ndarray:
    """Normalise to exact gross and net targets by side decomposition.

    G_L = (g* + n*)/2,  G_S = (g* - n*)/2
    w_long  = (q⁺/Q⁺) × G_L
    w_short = −(q⁻/Q⁻) × G_S
    """
    q_pos = np.maximum(q, 0.0)
    q_neg = np.maximum(-q, 0.0)
    Q_pos = q_pos.sum()
    Q_neg = q_neg.sum()

    G_L = (g_star + n_star) / 2.0
    G_S = (g_star - n_star) / 2.0

    w = np.zeros_like(q)
    if Q_pos > eps and G_L > eps:
        w += (q_pos / Q_pos) * G_L
    if Q_neg > eps and G_S > eps:
        w -= (q_neg / Q_neg) * G_S

    return w


def cap_and_redistribute(w: np.ndarray, p_max: float, eps: float = 1e-12) -> np.ndarray:
    """Cap per-name weights and redistribute excess within same side."""
    out = w.copy()
    for _iter in range(len(w)):
        capped_long = (out > p_max)
        capped_short = (out < -p_max)
        if not capped_long.any() and not capped_short.any():
            break
        # Long side
        if capped_long.any():
            excess = (out[capped_long] - p_max).sum()
            out[capped_long] = p_max
            free_long = (out > 0) & (~capped_long)
            if free_long.any():
                total_free = out[free_long].sum()
                if total_free > eps:
                    out[free_long] += out[free_long] / total_free * excess
        # Short side
        if capped_short.any():
            excess = (-out[capped_short] - p_max).sum()
            out[capped_short] = -p_max
            free_short = (out < 0) & (~capped_short)
            if free_short.any():
                total_free = (-out[free_short]).sum()
                if total_free > eps:
                    out[free_short] -= (-out[free_short]) / total_free * excess
    return out


def effective_n(w: np.ndarray) -> float:
    """N_eff = G²/Σw²."""
    gross = np.abs(w).sum()
    ssq = (w ** 2).sum()
    return gross ** 2 / ssq if ssq > 0 else 0.0


def run_sizing(
    scores: np.ndarray,
    eligible: np.ndarray | None = None,
    *,
    vol: np.ndarray | None = None,
    adv: np.ndarray | None = None,
    config: SizingConfig | None = None,
) -> dict[str, Any]:
    """Full sizing pipeline: score → raw weights."""
    cfg = config or SizingConfig()
    s = np.asarray(scores, dtype=float).ravel()
    n = s.size

    # Filter eligible
    if eligible is not None:
        mask = np.asarray(eligible, dtype=bool).ravel()
        s = np.where(mask, s, np.nan)

    # Transform
    q = score_to_intensity(s, cfg.mode, cfg.score_clip)
    q = apply_deadband(q, cfg.deadband)
    q = apply_topk(q, cfg.topk_long, cfg.topk_short)

    # Adjust
    q = adjust_vol_liq(q, vol, adv, cfg.vol_adjust_beta, cfg.liq_adjust_beta, cfg.eps)

    # Check degeneracy
    if np.abs(q).sum() < cfg.eps:
        return {"w_raw": np.zeros(n), "status": "degenerate",
                "gross_realized": 0.0, "net_realized": 0.0, "n_eff": 0.0}

    # Normalise
    w = normalise_gross_net(q, cfg.gross_target, cfg.net_target, cfg.eps)

    # Cap
    w = cap_and_redistribute(w, cfg.max_name_weight, cfg.eps)

    n_eff = effective_n(w)
    gross = float(np.abs(w).sum())
    net = float(w.sum())

    return {
        "w_raw": w,
        "status": "ok" if n_eff >= cfg.min_effective_n else "concentrated",
        "gross_realized": round(gross, 8),
        "net_realized": round(net, 8),
        "n_eff": round(n_eff, 2),
        "n_long": int((w > cfg.eps).sum()),
        "n_short": int((w < -cfg.eps).sum()),
    }
