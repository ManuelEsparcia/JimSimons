"""
validation/multiple_testing.py — Multiple hypothesis testing correction.

When you've tested 50 strategy variants and picked the best Sharpe,
how much of that Sharpe is real vs luck? This module answers that.

Three levels of correction:
    I.  FWER (Bonferroni/Holm): P(any false positive) < α
    II. FDR (Benjamini-Hochberg): E[false discoveries / discoveries] < q
    III. Deflated Sharpe: adjust for best-of-N selection bias

Plus: effective number of tests (N_eff) from correlation structure.

References:
    Holm (1979) "A Simple Sequentially Rejective Multiple Test Procedure"
    Benjamini & Hochberg (1995) "Controlling the False Discovery Rate"
    Harvey, Liu & Zhu (2016) "…and the Cross-Section of Expected Returns"
    Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import stats

from . import GateResult

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Effective number of tests
# ---------------------------------------------------------------------------

def compute_n_effective(corr_matrix: np.ndarray) -> float:
    """Estimate effective number of independent tests from correlation structure.

    N_eff = (Σ λ_k)² / (Σ λ_k²)

    If all strategies are independent: N_eff ≈ M
    If all are identical: N_eff ≈ 1
    """
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = eigenvalues[eigenvalues > 0]  # numerical stability
    sum_lambda = eigenvalues.sum()
    sum_lambda_sq = (eigenvalues ** 2).sum()
    if sum_lambda_sq < 1e-15:
        return 1.0
    n_eff = sum_lambda ** 2 / sum_lambda_sq
    return float(np.clip(n_eff, 1.0, len(corr_matrix)))


# ---------------------------------------------------------------------------
# Level I: FWER (Holm)
# ---------------------------------------------------------------------------

def holm_correction(p_values: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """Holm step-down procedure for FWER control.

    Sort p-values, reject while p_(k) ≤ α / (m - k + 1).
    """
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    rejected = np.zeros(m, dtype=bool)
    for k in range(m):
        threshold = alpha / (m - k)
        if sorted_p[k] <= threshold:
            rejected[sorted_idx[k]] = True
        else:
            break

    adjusted_p = np.minimum(1.0, sorted_p * np.arange(m, 0, -1))
    # Make monotone
    for i in range(1, m):
        adjusted_p[i] = max(adjusted_p[i], adjusted_p[i-1])
    adj_p_original = np.empty(m)
    adj_p_original[sorted_idx] = adjusted_p

    return {
        "method": "holm",
        "n_rejected": int(rejected.sum()),
        "rejected_mask": rejected,
        "adjusted_p_values": adj_p_original,
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Level II: FDR (Benjamini-Hochberg)
# ---------------------------------------------------------------------------

def bh_correction(p_values: np.ndarray, q: float = 0.05) -> dict[str, Any]:
    """Benjamini-Hochberg procedure for FDR control.

    k* = max{k : p_(k) ≤ k/m × q}. Reject H_(1),...,H_(k*).
    """
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    thresholds = np.arange(1, m+1) / m * q
    k_star = 0
    for k in range(m):
        if sorted_p[k] <= thresholds[k]:
            k_star = k + 1

    rejected = np.zeros(m, dtype=bool)
    if k_star > 0:
        rejected[sorted_idx[:k_star]] = True

    # Adjusted p-values
    adjusted_p = sorted_p * m / np.arange(1, m+1)
    for i in range(m-2, -1, -1):
        adjusted_p[i] = min(adjusted_p[i], adjusted_p[i+1])
    adjusted_p = np.minimum(1.0, adjusted_p)
    adj_p_original = np.empty(m)
    adj_p_original[sorted_idx] = adjusted_p

    return {
        "method": "benjamini_hochberg",
        "n_rejected": int(rejected.sum()),
        "rejected_mask": rejected,
        "adjusted_p_values": adj_p_original,
        "fdr_level": q,
    }


# ---------------------------------------------------------------------------
# Level III: Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

def deflated_sharpe(
    observed_sharpe: float,
    n_obs: int,
    n_tests: int | float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> dict[str, Any]:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.

    Adjusts observed Sharpe for:
    - Selection bias (best of N tests)
    - Non-normality of returns (skew, kurtosis)
    - Sample size

    SR_0 = expected max Sharpe under null with N tests
    DSR = (SR_obs - SR_0) / σ(SR)
    """
    # Expected max Sharpe under null (Euler-Mascheroni approximation)
    gamma_em = 0.5772156649
    if n_tests > 1:
        sr_0 = np.sqrt(2 * np.log(n_tests)) - (gamma_em + np.log(np.pi)) / (2 * np.sqrt(2 * np.log(n_tests)))
    else:
        sr_0 = 0.0

    # Standard error of Sharpe estimator (Lo 2002 + non-normality correction)
    se_sr = np.sqrt((1 + 0.25 * (kurtosis - 3) * observed_sharpe**2
                     - skew * observed_sharpe) / max(n_obs, 1))

    # Deflated Sharpe
    if se_sr > 0:
        dsr_stat = (observed_sharpe - sr_0) / se_sr
        dsr_pval = 1.0 - stats.norm.cdf(dsr_stat)
    else:
        dsr_stat = 0.0
        dsr_pval = 1.0

    return {
        "observed_sharpe": observed_sharpe,
        "sr_0_benchmark": round(sr_0, 4),
        "se_sharpe": round(se_sr, 4),
        "deflated_sharpe_stat": round(dsr_stat, 4),
        "deflated_sharpe_pval": round(dsr_pval, 6),
        "n_tests": n_tests,
        "significant_at_05": dsr_pval < 0.05,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_multiple_testing(
    p_values: np.ndarray,
    strategy_names: Sequence[str] | None = None,
    *,
    observed_sharpe: float | None = None,
    n_obs: int = 252,
    return_skew: float = 0.0,
    return_kurtosis: float = 3.0,
    corr_matrix: np.ndarray | None = None,
    alpha: float = 0.05,
    fdr_q: float = 0.10,
) -> tuple[list[GateResult], dict[str, Any]]:
    """Run full multiple testing correction suite."""
    results: list[GateResult] = []
    details: dict[str, Any] = {}

    m = len(p_values)

    # N_eff
    if corr_matrix is not None:
        n_eff = compute_n_effective(corr_matrix)
    else:
        n_eff = float(m)
    details["n_eff"] = round(n_eff, 1)
    details["n_raw"] = m

    # FWER (Holm)
    holm = holm_correction(p_values, alpha)
    details["holm"] = {k: v for k, v in holm.items() if k != "rejected_mask"}
    results.append(GateResult(
        "multiple_testing_fwer", "statistical",
        "PASS" if holm["n_rejected"] > 0 else "WARN",
        holm["n_rejected"], 1,
        f"Holm FWER: {holm['n_rejected']}/{m} survive at α={alpha}",
    ))

    # FDR (BH)
    bh = bh_correction(p_values, fdr_q)
    details["bh"] = {k: v for k, v in bh.items() if k != "rejected_mask"}
    results.append(GateResult(
        "multiple_testing_fdr", "statistical",
        "PASS" if bh["n_rejected"] > 0 else "WARN",
        bh["n_rejected"], 1,
        f"BH FDR: {bh['n_rejected']}/{m} survive at q={fdr_q}",
    ))

    # Deflated Sharpe
    if observed_sharpe is not None:
        dsr = deflated_sharpe(observed_sharpe, n_obs, n_eff, return_skew, return_kurtosis)
        details["deflated_sharpe"] = dsr
        results.append(GateResult(
            "deflated_sharpe", "statistical",
            "PASS" if dsr["significant_at_05"] else "WARN",
            dsr["deflated_sharpe_pval"], 0.05,
            f"DSR p={dsr['deflated_sharpe_pval']:.4f}, SR_0={dsr['sr_0_benchmark']:.3f} (N_eff={n_eff:.0f})",
        ))

    summary = {"n_tests": m, "n_eff": n_eff,
               "holm_survivors": holm["n_rejected"], "bh_survivors": bh["n_rejected"]}
    return results, summary
