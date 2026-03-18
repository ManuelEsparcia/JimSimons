"""
validation/validation_suite.py — Orchestrator with hard/soft gate taxonomy.

Runs all validation modules and produces a single GO/NO-GO decision:

Gate hierarchy:
    STRUCTURAL (hard gates — non-compensable):
        leakage_audit, walkforward_bias
        Any FAIL → global FAIL, no further analysis needed

    STATISTICAL (soft gates):
        multiple_testing, pbo_cscv
        WARN/FAIL accumulate but don't auto-reject

    ECONOMIC (soft gates):
        synthetic_shocks, capacity_sanity
        WARN/FAIL indicate operational risk

Final decision:
    H = Π h_structural  (product of hard gates: 0 if any FAIL)
    If H = 0: REJECT
    If H = 1: compute soft score from statistical + economic gates
    Final recommendation: PASS | CONDITIONAL | REJECT
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np

from . import GateResult, GateStatus

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SuiteConfig:
    """Configuration for the validation suite."""
    # Structural (hard)
    run_leakage: bool = True
    run_walkforward: bool = True

    # Statistical (soft)
    run_multiple_testing: bool = True
    run_pbo: bool = True

    # Economic (soft)
    run_shocks: bool = False    # requires backtest_fn
    run_capacity: bool = False  # requires trade data

    # Decision thresholds
    max_soft_failures: int = 2     # more than this → REJECT even without structural fail
    max_pbo: float = 0.50
    min_sharpe_deflated_pval: float = 0.10


@dataclass
class ValidationReport:
    """Final validation report."""
    decision: str                      # "PASS" | "CONDITIONAL" | "REJECT"
    structural_gates: list[GateResult]
    statistical_gates: list[GateResult]
    economic_gates: list[GateResult]
    all_gates: list[GateResult]
    summary: dict[str, Any]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "gates": [
                {"name": g.gate_name, "category": g.category,
                 "status": g.status, "message": g.message}
                for g in self.all_gates
            ],
        }


def run_validation_suite(
    *,
    # Leakage inputs
    features_df=None,
    feature_cols=None,
    label_col=None,
    train_dates=None,
    test_dates=None,
    preprocess_states=None,

    # Walk-forward inputs
    folds=None,

    # Multiple testing inputs
    p_values=None,
    observed_sharpe=None,
    n_obs=252,
    return_skew=0.0,
    return_kurtosis=3.0,
    corr_matrix=None,

    # PBO inputs
    performance_matrix=None,

    # Config
    config: SuiteConfig | None = None,
) -> ValidationReport:
    """Run the complete validation suite."""
    cfg = config or SuiteConfig()

    structural: list[GateResult] = []
    statistical: list[GateResult] = []
    economic: list[GateResult] = []

    # --- STRUCTURAL GATES ---
    if cfg.run_leakage and features_df is not None:
        from .leakage_audit import run_leakage_audit
        leak_results, _ = run_leakage_audit(
            features_df, feature_cols=feature_cols, label_col=label_col,
            train_dates=train_dates, test_dates=test_dates,
            preprocess_states=preprocess_states,
        )
        structural.extend(leak_results)

    if cfg.run_walkforward and folds is not None:
        from .walkforward_bias import run_walkforward_bias_audit
        wf_results, _ = run_walkforward_bias_audit(folds)
        structural.extend(wf_results)

    # --- STATISTICAL GATES ---
    if cfg.run_multiple_testing and p_values is not None:
        from .multiple_testing import run_multiple_testing
        mt_results, _ = run_multiple_testing(
            p_values, observed_sharpe=observed_sharpe,
            n_obs=n_obs, return_skew=return_skew,
            return_kurtosis=return_kurtosis, corr_matrix=corr_matrix,
        )
        statistical.extend(mt_results)

    if cfg.run_pbo and performance_matrix is not None:
        from .pbo_cscv import run_pbo
        pbo_results, _ = run_pbo(performance_matrix)
        statistical.extend(pbo_results)

    # --- DECISION ---
    all_gates = structural + statistical + economic

    struct_fail = any(g.status == "FAIL" for g in structural)
    n_soft_fail = sum(1 for g in statistical + economic if g.status == "FAIL")
    n_soft_warn = sum(1 for g in statistical + economic if g.status == "WARN")

    if struct_fail:
        decision = "REJECT"
        reason = "Structural gate failure (non-compensable)"
    elif n_soft_fail > cfg.max_soft_failures:
        decision = "REJECT"
        reason = f"{n_soft_fail} soft gate failures exceed max={cfg.max_soft_failures}"
    elif n_soft_fail > 0 or n_soft_warn > 2:
        decision = "CONDITIONAL"
        reason = f"{n_soft_fail} soft FAIL, {n_soft_warn} WARN — review required"
    else:
        decision = "PASS"
        reason = "All gates passed"

    summary = {
        "decision": decision,
        "reason": reason,
        "n_structural": len(structural),
        "n_structural_fail": sum(1 for g in structural if g.status == "FAIL"),
        "n_statistical": len(statistical),
        "n_statistical_fail": n_soft_fail,
        "n_economic": len(economic),
        "n_total_checks": len(all_gates),
    }

    LOGGER.info("Validation suite: %s — %s", decision, reason)

    return ValidationReport(
        decision=decision,
        structural_gates=structural,
        statistical_gates=statistical,
        economic_gates=economic,
        all_gates=all_gates,
        summary=summary,
        timestamp=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )
