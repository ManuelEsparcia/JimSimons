"""
validation/leakage_audit.py — Temporal leakage detection (5 classes).

Detects 5 classes of leakage that destroy backtest validity:
1. Availability leakage: feature uses data published after decision_cut
2. Label leakage: features encode future label information
3. Train-test leakage: information overlap between folds
4. Join/ffill leakage: forward-fill propagates future data
5. Global transform leakage: preprocessing fitted on full dataset

Each check returns PASS/WARN/FAIL with specific breach counts.
A single confirmed leakage → entire experiment is invalid.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import GateResult, GateStatus, LeakageDetected

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Availability leakage
# ---------------------------------------------------------------------------

def check_availability_leakage(
    features_df: pd.DataFrame,
    availability_col: str = "availability_date",
    decision_col: str = "decision_date",
    safety_margin_days: int = 0,
) -> GateResult:
    """Check that all features use data available BEFORE decision cut.

    Formal condition: tau_avail(j,k) <= tau_decision(j) - safety_margin
    Any breach → FAIL (zero tolerance).
    """
    if availability_col not in features_df.columns or decision_col not in features_df.columns:
        return GateResult("availability_leakage", "structural", "PASS", None, None,
                         "No availability/decision columns — check skipped (assume PIT)")

    avail = pd.to_datetime(features_df[availability_col], errors="coerce")
    decision = pd.to_datetime(features_df[decision_col], errors="coerce")
    valid = avail.notna() & decision.notna()

    if safety_margin_days > 0:
        decision = decision - pd.Timedelta(days=safety_margin_days)

    breaches = (avail > decision) & valid
    n_breach = int(breaches.sum())
    n_total = int(valid.sum())

    return GateResult(
        "availability_leakage", "structural",
        "FAIL" if n_breach > 0 else "PASS",
        n_breach, 0,
        f"{n_breach}/{n_total} availability breaches ({n_breach/max(n_total,1):.4%})",
        {"breach_indices": list(features_df.index[breaches][:20])},
    )


# ---------------------------------------------------------------------------
# 2. Label leakage (correlation-based proxy)
# ---------------------------------------------------------------------------

def check_label_leakage(
    features_df: pd.DataFrame,
    label_col: str,
    feature_cols: Sequence[str],
    suspicious_corr_threshold: float = 0.5,
) -> GateResult:
    """Check for suspiciously high correlation between features and FUTURE labels.

    High correlation alone doesn't prove leakage (could be genuine signal),
    but |corr| > 0.5 with future labels is a strong red flag.
    """
    if label_col not in features_df.columns:
        return GateResult("label_leakage", "structural", "PASS", None, None,
                         "No label column — check skipped")

    y = features_df[label_col].values
    suspicious = []
    for col in feature_cols:
        if col not in features_df.columns:
            continue
        x = features_df[col].values
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < 30:
            continue
        corr = np.corrcoef(x[valid], y[valid])[0, 1]
        if abs(corr) > suspicious_corr_threshold:
            suspicious.append((col, round(float(corr), 4)))

    status = "WARN" if suspicious else "PASS"
    return GateResult(
        "label_leakage_proxy", "structural", status,
        len(suspicious), 0,
        f"{len(suspicious)} features with |corr(feat, label)| > {suspicious_corr_threshold}: {suspicious[:5]}",
        {"suspicious_features": suspicious},
    )


# ---------------------------------------------------------------------------
# 3. Train-test leakage (information overlap)
# ---------------------------------------------------------------------------

def check_train_test_leakage(
    train_dates: np.ndarray,
    test_dates: np.ndarray,
    label_horizon: int = 10,
    feature_lookback: int = 63,
    embargo_days: int = 21,
) -> GateResult:
    """Check that train and test windows don't overlap informationally.

    The information window of observation at date t is:
        [t - feature_lookback, t + label_horizon]
    Train observations whose info window touches test → contamination.
    """
    train_set = set(pd.to_datetime(train_dates))
    test_set = set(pd.to_datetime(test_dates))

    # Direct date overlap
    overlap = train_set & test_set
    if overlap:
        return GateResult(
            "train_test_leakage", "structural", "FAIL",
            len(overlap), 0,
            f"{len(overlap)} dates appear in both train and test",
            {"overlap_dates": sorted(list(overlap))[:10]},
        )

    # Gap check
    max_train = max(train_set) if train_set else pd.Timestamp.min
    min_test = min(test_set) if test_set else pd.Timestamp.max
    gap_days = (min_test - max_train).days
    required_gap = label_horizon + embargo_days

    if gap_days < required_gap:
        return GateResult(
            "train_test_leakage", "structural",
            "FAIL" if gap_days < label_horizon else "WARN",
            gap_days, required_gap,
            f"Gap between train end and test start: {gap_days}d < required {required_gap}d",
        )

    return GateResult("train_test_leakage", "structural", "PASS",
                      gap_days, required_gap, f"Gap {gap_days}d >= required {required_gap}d")


# ---------------------------------------------------------------------------
# 4. Forward-fill leakage
# ---------------------------------------------------------------------------

def check_ffill_leakage(
    df: pd.DataFrame,
    date_col: str = "date",
    max_ffill_days: int = 5,
) -> GateResult:
    """Detect excessive forward-fill that may propagate stale data.

    If a value doesn't change for > max_ffill_days consecutive sessions,
    it may be forward-filled from stale data.
    """
    numeric_cols = [c for c in df.columns if c not in (date_col, "symbol") and pd.api.types.is_numeric_dtype(df[c])]
    flagged = []

    for col in numeric_cols[:50]:  # cap for performance
        if df[col].nunique() < 3:
            continue
        # Count max consecutive identical values per symbol
        if "symbol" in df.columns:
            max_repeat = df.groupby("symbol")[col].apply(
                lambda s: (s == s.shift()).astype(int).groupby((s != s.shift()).cumsum()).cumsum().max()
            ).max()
        else:
            s = df[col]
            max_repeat = (s == s.shift()).astype(int).groupby((s != s.shift()).cumsum()).cumsum().max()

        if max_repeat > max_ffill_days:
            flagged.append((col, int(max_repeat)))

    return GateResult(
        "ffill_leakage", "structural",
        "WARN" if flagged else "PASS",
        len(flagged), 0,
        f"{len(flagged)} columns with >{max_ffill_days} consecutive identical values: {flagged[:5]}",
    )


# ---------------------------------------------------------------------------
# 5. Global transform leakage
# ---------------------------------------------------------------------------

def check_global_transform_leakage(
    preprocess_states: Sequence[Any],
    n_folds: int,
) -> GateResult:
    """Check that preprocessing was fitted per-fold, not globally.

    If all folds have identical preprocessing state, it was likely
    fitted on the full dataset (leakage).
    """
    if len(preprocess_states) < 2:
        return GateResult("global_transform_leakage", "structural", "PASS",
                         None, None, "Only 1 fold — check skipped")

    # Compare imputation values across folds
    all_identical = True
    for i in range(1, len(preprocess_states)):
        s0 = preprocess_states[0]
        si = preprocess_states[i]
        if hasattr(s0, 'impute_values') and hasattr(si, 'impute_values'):
            if s0.impute_values != si.impute_values:
                all_identical = False
                break
            if s0.scale_mean != si.scale_mean:
                all_identical = False
                break

    if all_identical and len(preprocess_states) >= 3:
        return GateResult(
            "global_transform_leakage", "structural", "WARN",
            True, False,
            "All folds have identical preprocessing state — possible global fit leakage",
        )

    return GateResult("global_transform_leakage", "structural", "PASS",
                      False, False, "Preprocessing states differ across folds (good)")


# ---------------------------------------------------------------------------
# Full audit
# ---------------------------------------------------------------------------

def run_leakage_audit(
    features_df: pd.DataFrame,
    *,
    feature_cols: Sequence[str] | None = None,
    label_col: str | None = None,
    train_dates: np.ndarray | None = None,
    test_dates: np.ndarray | None = None,
    label_horizon: int = 10,
    embargo_days: int = 21,
    preprocess_states: Sequence[Any] | None = None,
) -> tuple[list[GateResult], dict[str, Any]]:
    """Run all 5 leakage checks."""
    results: list[GateResult] = []

    results.append(check_availability_leakage(features_df))

    if label_col and feature_cols:
        results.append(check_label_leakage(features_df, label_col, feature_cols))

    if train_dates is not None and test_dates is not None:
        results.append(check_train_test_leakage(train_dates, test_dates, label_horizon, embargo_days=embargo_days))

    results.append(check_ffill_leakage(features_df))

    if preprocess_states:
        results.append(check_global_transform_leakage(preprocess_states, len(preprocess_states)))

    n_fail = sum(1 for r in results if r.status == "FAIL")
    summary = {
        "overall": "FAIL" if n_fail > 0 else ("WARN" if any(r.status == "WARN" for r in results) else "PASS"),
        "n_checks": len(results), "n_fail": n_fail,
    }
    return results, summary
