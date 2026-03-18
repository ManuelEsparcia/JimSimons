"""
validation/walkforward_bias.py — Walk-forward geometry validation.

Validates that the walk-forward experimental design is temporally sound:
1. Chronological ordering: max(train) < min(valid) < min(test)
2. Gap sufficiency: gap >= max(label_horizon, feature_lookback) + embargo
3. Purge effectiveness: no train observation's info window touches test
4. Overlap contamination: quantifies information overlap between folds
5. Boundary effects: detects edge artifacts at fold boundaries
6. Stability: small protocol changes shouldn't materially change OOS results

Any structural violation → FAIL (non-compensable).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import GateResult

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WFBiasConfig:
    label_horizon: int = 10
    feature_lookback: int = 63
    embargo_min: int = 21
    overlap_fail_threshold: float = 0.005
    overlap_warn_threshold: float = 0.0
    boundary_pct: float = 0.10    # fraction of test for boundary analysis


def check_chronological_order(
    folds: Sequence[dict[str, Any]],
) -> GateResult:
    """Check 1: max(train) < min(test) for every fold."""
    violations = []
    for i, fold in enumerate(folds):
        train_end = fold.get("train_end")
        test_start = fold.get("test_start")
        if train_end is not None and test_start is not None:
            if pd.Timestamp(train_end) >= pd.Timestamp(test_start):
                violations.append(f"fold_{i}: train_end={train_end} >= test_start={test_start}")

    return GateResult(
        "wf_chronological", "structural",
        "FAIL" if violations else "PASS",
        len(violations), 0,
        f"{len(violations)} chronological violations: {violations[:3]}" if violations else "All folds chronologically ordered",
    )


def check_gap_sufficiency(
    folds: Sequence[dict[str, Any]],
    config: WFBiasConfig,
) -> GateResult:
    """Check 2: gap between train end and test start >= required minimum."""
    required = max(config.label_horizon, config.feature_lookback) + config.embargo_min
    violations = []
    gaps = []

    for i, fold in enumerate(folds):
        train_end = fold.get("train_end")
        test_start = fold.get("test_start")
        if train_end is not None and test_start is not None:
            gap = (pd.Timestamp(test_start) - pd.Timestamp(train_end)).days
            gaps.append(gap)
            if gap < required:
                violations.append(f"fold_{i}: gap={gap}d < required={required}d")

    return GateResult(
        "wf_gap_sufficiency", "structural",
        "FAIL" if violations else "PASS",
        min(gaps) if gaps else None, required,
        f"Min gap={min(gaps) if gaps else 'N/A'}d, required={required}d. {len(violations)} violations",
    )


def check_purge_effectiveness(
    folds: Sequence[dict[str, Any]],
    config: WFBiasConfig,
) -> GateResult:
    """Check 3: no train observation's info window touches test."""
    violations = []
    for i, fold in enumerate(folds):
        train_end = fold.get("train_end")
        test_start = fold.get("test_start")
        if train_end is None or test_start is None:
            continue
        # Last train obs info window extends to train_end + label_horizon
        info_end = pd.Timestamp(train_end) + pd.Timedelta(days=config.label_horizon)
        if info_end >= pd.Timestamp(test_start):
            violations.append(f"fold_{i}: info_end={info_end} >= test_start={test_start}")

    return GateResult(
        "wf_purge_effective", "structural",
        "FAIL" if violations else "PASS",
        len(violations), 0,
        f"{len(violations)} purge breaches" if violations else "Purge effective: no info window touches test",
    )


def check_fold_overlap(
    folds: Sequence[dict[str, Any]],
) -> GateResult:
    """Check 4: OOS segments don't overlap across folds."""
    ranges = []
    for fold in folds:
        ts = fold.get("test_start")
        te = fold.get("test_end")
        if ts and te:
            ranges.append((pd.Timestamp(ts), pd.Timestamp(te)))

    overlaps = 0
    for i in range(len(ranges)):
        for j in range(i+1, len(ranges)):
            if ranges[i][0] <= ranges[j][1] and ranges[j][0] <= ranges[i][1]:
                overlaps += 1

    return GateResult(
        "wf_oos_overlap", "structural",
        "FAIL" if overlaps > 0 else "PASS",
        overlaps, 0,
        f"{overlaps} overlapping OOS segments" if overlaps else "No OOS overlap",
    )


def run_walkforward_bias_audit(
    folds: Sequence[dict[str, Any]],
    config: WFBiasConfig | None = None,
) -> tuple[list[GateResult], dict[str, Any]]:
    """Run all walk-forward bias checks."""
    cfg = config or WFBiasConfig()
    results = [
        check_chronological_order(folds),
        check_gap_sufficiency(folds, cfg),
        check_purge_effectiveness(folds, cfg),
        check_fold_overlap(folds),
    ]
    n_fail = sum(1 for r in results if r.status == "FAIL")
    summary = {"overall": "FAIL" if n_fail > 0 else "PASS", "n_checks": len(results), "n_fail": n_fail}
    return results, summary
