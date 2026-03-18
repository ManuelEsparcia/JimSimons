"""
validation/pbo_cscv.py — Probability of Backtest Overfitting (CSCV).

Bailey, Borwein, Lopez de Prado & Zhu (2017) combinatorially symmetric
cross-validation for detecting overfitting in backtested strategies.

PBO = P(the IS champion underperforms OOS median)

High PBO (>0.5) → the strategy selection process is likely picking noise.
Low PBO (<0.1) → the IS champion is genuinely good OOS.

Algorithm:
1. Partition history into S chronological blocks
2. For J random half-splits (IS, OOS):
   a. Pick the champion on IS (best Sharpe)
   b. Check its OOS rank among all strategies
   c. If OOS rank < median → overfitting signal
3. PBO = fraction of splits where champion underperforms OOS median
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Any, Sequence

import numpy as np

from . import GateResult

LOGGER = logging.getLogger(__name__)


def compute_pbo(
    performance_matrix: np.ndarray,
    n_splits: int = 100,
    n_blocks: int = 10,
    metric: str = "mean",
    seed: int = 42,
) -> dict[str, Any]:
    """Compute PBO via combinatorially symmetric cross-validation.

    Parameters
    ----------
    performance_matrix : (n_blocks, n_strategies) array
        Per-block performance for each strategy. Blocks are chronological.
    n_splits : int
        Number of random IS/OOS half-splits to evaluate.
    n_blocks : int
        Number of chronological blocks (must equal performance_matrix.shape[0]).
    metric : str
        Aggregation metric: "mean" | "sharpe"
    seed : int

    Returns
    -------
    dict with pbo, logit_distribution, omega_distribution, etc.
    """
    S, M = performance_matrix.shape
    assert S == n_blocks, f"Matrix has {S} blocks but n_blocks={n_blocks}"
    assert S >= 4, "Need at least 4 blocks for CSCV"
    assert S % 2 == 0, "Number of blocks must be even"

    half = S // 2
    rng = np.random.RandomState(seed)

    # Generate splits
    all_indices = list(range(S))
    if n_splits >= len(list(combinations(all_indices, half))):
        splits = list(combinations(all_indices, half))
    else:
        splits = []
        seen = set()
        while len(splits) < n_splits:
            is_idx = tuple(sorted(rng.choice(S, size=half, replace=False)))
            if is_idx not in seen:
                seen.add(is_idx)
                splits.append(is_idx)

    # Aggregator
    def aggregate(blocks):
        if metric == "sharpe":
            mu = blocks.mean()
            std = blocks.std()
            return mu / std if std > 0 else 0.0
        return blocks.mean()

    omegas = []
    logits = []

    for is_idx in splits:
        oos_idx = tuple(i for i in range(S) if i not in is_idx)

        # IS and OOS scores for each strategy
        is_scores = np.array([aggregate(performance_matrix[list(is_idx), m]) for m in range(M)])
        oos_scores = np.array([aggregate(performance_matrix[list(oos_idx), m]) for m in range(M)])

        # IS champion
        champion = np.argmax(is_scores)

        # OOS rank of champion (1=worst, M=best)
        rank = 1 + np.sum(oos_scores < oos_scores[champion])
        omega = rank / (M + 1)
        omegas.append(omega)

        # Logit transform
        if 0 < omega < 1:
            logit = np.log(omega / (1 - omega))
        elif omega <= 0:
            logit = -10
        else:
            logit = 10
        logits.append(logit)

    omegas = np.array(omegas)
    logits = np.array(logits)

    # PBO = P(logit < 0) = fraction where champion underperforms OOS median
    pbo = float((logits < 0).mean())

    # Bootstrap CI for PBO
    boot_pbos = []
    for _ in range(1000):
        sample = rng.choice(logits, size=len(logits), replace=True)
        boot_pbos.append(float((sample < 0).mean()))
    boot_pbos = np.array(boot_pbos)

    return {
        "pbo": round(pbo, 4),
        "pbo_ci_lo": round(float(np.percentile(boot_pbos, 2.5)), 4),
        "pbo_ci_hi": round(float(np.percentile(boot_pbos, 97.5)), 4),
        "n_splits": len(splits),
        "n_strategies": M,
        "n_blocks": S,
        "omega_mean": round(float(omegas.mean()), 4),
        "omega_median": round(float(np.median(omegas)), 4),
        "logit_mean": round(float(logits.mean()), 4),
    }


def run_pbo(
    performance_matrix: np.ndarray,
    **kwargs,
) -> tuple[list[GateResult], dict[str, Any]]:
    """Run PBO analysis and return gate results."""
    pbo_result = compute_pbo(performance_matrix, **kwargs)

    results = [GateResult(
        "pbo_cscv", "statistical",
        "FAIL" if pbo_result["pbo"] > 0.5 else ("WARN" if pbo_result["pbo"] > 0.3 else "PASS"),
        pbo_result["pbo"], 0.5,
        f"PBO={pbo_result['pbo']:.1%} [{pbo_result['pbo_ci_lo']:.1%}, {pbo_result['pbo_ci_hi']:.1%}]"
        f" ω_mean={pbo_result['omega_mean']:.3f} ({pbo_result['n_splits']} splits, {pbo_result['n_strategies']} strategies)",
    )]

    return results, pbo_result
