"""
execution/impact_model.py — Power-law market impact by liquidity bucket.

    impact_temp_bps = α_b × ρ^β_b × g(ν) + γ_b × s
    impact_perm_bps = η_b × ρ                (optional)

Bucketised: A (high liq) → B (medium) → C (small cap) → D (micro).
Participation saturation: superlinear penalty above ρ_max.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


BUCKET_PARAMS = {
    "A": {"alpha": 5.0,  "beta": 0.5, "gamma": 0.05, "eta": 2.0, "rho_max": 0.15, "vol_ref": 0.015},
    "B": {"alpha": 8.0,  "beta": 0.5, "gamma": 0.08, "eta": 3.0, "rho_max": 0.10, "vol_ref": 0.020},
    "C": {"alpha": 12.0, "beta": 0.6, "gamma": 0.10, "eta": 5.0, "rho_max": 0.05, "vol_ref": 0.025},
    "D": {"alpha": 20.0, "beta": 0.7, "gamma": 0.15, "eta": 8.0, "rho_max": 0.03, "vol_ref": 0.030},
}


@dataclass(frozen=True)
class ImpactConfig:
    delta: float = 0.5               # vol scaling exponent
    use_permanent: bool = False
    saturation_kappa: float = 50.0   # superlinear penalty coeff
    saturation_zeta: float = 2.0     # superlinear exponent
    pi_cap_hard: float = 5.0         # defensive cap
    bucket_params: dict = field(default_factory=lambda: dict(BUCKET_PARAMS))
    scenario: str = "base"


def compute_impact(
    attempt_notional: np.ndarray,
    adv_usd: np.ndarray,
    vol_daily: np.ndarray,
    spread_bps: np.ndarray,
    buckets: np.ndarray,
    *,
    config: ImpactConfig | None = None,
) -> dict[str, np.ndarray]:
    """Compute market impact per attempt."""
    cfg = config or ImpactConfig()
    N = np.abs(np.asarray(attempt_notional, dtype=float))
    adv = np.maximum(np.asarray(adv_usd, dtype=float), 1.0)
    vol = np.maximum(np.asarray(vol_daily, dtype=float), 1e-6)
    sp = np.asarray(spread_bps, dtype=float)
    bkt = np.asarray(buckets, dtype=str)

    n = N.size
    rho = np.minimum(N / adv, cfg.pi_cap_hard)
    impact_temp = np.zeros(n)
    impact_perm = np.zeros(n)
    sat_flags = np.zeros(n, dtype=bool)

    for i in range(n):
        b = bkt[i] if bkt[i] in cfg.bucket_params else "C"
        p = cfg.bucket_params[b]
        rho_i = rho[i]
        vol_scale = (vol[i] / p["vol_ref"]) ** cfg.delta

        if rho_i <= p["rho_max"]:
            impact_temp[i] = p["alpha"] * (rho_i ** p["beta"]) * vol_scale + p["gamma"] * sp[i]
        else:
            # Base + superlinear saturation penalty
            base = p["alpha"] * (p["rho_max"] ** p["beta"]) * vol_scale + p["gamma"] * sp[i]
            excess = rho_i - p["rho_max"]
            penalty = cfg.saturation_kappa * (excess ** cfg.saturation_zeta)
            impact_temp[i] = base + penalty
            sat_flags[i] = True

        if cfg.use_permanent:
            impact_perm[i] = p["eta"] * rho_i

    impact_total = impact_temp + impact_perm
    impact_cost = impact_total * 1e-4 * N

    return {
        "participation_rate": rho,
        "impact_temp_bps": impact_temp,
        "impact_perm_bps": impact_perm,
        "impact_total_bps": impact_total,
        "impact_cost_usd": impact_cost,
        "saturation_flag": sat_flags,
        "bucket_used": bkt,
    }
