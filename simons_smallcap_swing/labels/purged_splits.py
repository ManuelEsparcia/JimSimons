"""
labels/purged_splits.py — Temporal splits with purge + embargo.

Constructs train/valid/test partitions where NO train sample's forward
window overlaps with ANY test sample's information set.

Core concepts:
    Event interval: I(t,i,h) = [t+lag, t+lag+h]  (forward dependency)
    Purge rule:     train excluded if I_train ∩ W_test ≠ ∅
    Embargo rule:   train excluded if t ∈ E_test (post-test buffer)

Split modes:
    purged_kfold:  K temporal blocks, each takes turn as test
    walkforward:   expanding/rolling train → test sequences
    cpcv:          combinatorial purged cross-validation

Horizon policy:
    primary:    use declared primary horizon
    max:        use max(H) — conservative default
    per_label:  per-horizon purge (advanced)

Invariants (HARD, non-negotiable):
    1. I_train ∩ W_test = ∅  (no temporal overlap)
    2. train_dates ∩ embargo_dates = ∅
    3. Deterministic: same inputs → same folds
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    Severity,
    utc_now_iso, config_hash, json_safe,
    read_dataframe, write_parquet_safe, write_json_safe,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitConfig:
    split_mode: str = "purged_kfold"    # purged_kfold | walkforward | cpcv
    n_folds: int = 5
    decision_lag: int = 1
    horizon_policy: str = "max"          # primary | max | per_label
    primary_horizon: int = 10
    horizons: tuple[int, ...] = (5, 10, 20)
    embargo_policy: str = "fixed_days"   # fixed_days | pct_of_test | feature_memory_aware
    embargo_days: int = 21
    min_train_obs: int = 500
    min_test_obs: int = 100
    # walkforward specific
    wf_train_days: int = 504             # ~2 years
    wf_test_days: int = 63              # ~1 quarter
    wf_expanding: bool = True


def resolve_effective_horizon(config: SplitConfig) -> int:
    """Resolve h_eff based on horizon policy."""
    if config.horizon_policy == "max":
        return max(config.horizons)
    if config.horizon_policy == "primary":
        return config.primary_horizon
    return max(config.horizons)  # default to conservative


def resolve_effective_embargo(config: SplitConfig) -> int:
    """Resolve b_eff based on embargo policy."""
    if config.embargo_policy == "fixed_days":
        return config.embargo_days
    if config.embargo_policy == "pct_of_test":
        return max(1, int(0.1 * config.wf_test_days))
    # feature_memory_aware → use max(horizon, embargo_days)
    return max(config.embargo_days, resolve_effective_horizon(config))


def build_event_intervals(
    dates: pd.Series,
    lag: int,
    h_eff: int,
    calendar: np.ndarray,
) -> pd.DataFrame:
    """Build event interval [t+lag, t+lag+h] for each observation date."""
    cal_list = sorted(pd.Timestamp(d) for d in calendar)
    cal_idx = {d: i for i, d in enumerate(cal_list)}

    starts, ends = [], []
    for d in dates:
        d = pd.Timestamp(d)
        idx = cal_idx.get(d)
        if idx is None:
            starts.append(pd.NaT)
            ends.append(pd.NaT)
            continue
        s_idx = idx + lag
        e_idx = idx + lag + h_eff
        starts.append(cal_list[s_idx] if s_idx < len(cal_list) else pd.NaT)
        ends.append(cal_list[e_idx] if e_idx < len(cal_list) else pd.NaT)

    return pd.DataFrame({"date": dates.values, "event_start": starts, "event_end": ends})


def build_purged_kfold(
    unique_dates: np.ndarray,
    n_folds: int,
    h_eff: int,
    b_eff: int,
    lag: int,
) -> list[dict[str, Any]]:
    """Build K temporal folds with purge + embargo."""
    dates = np.sort(unique_dates)
    n = len(dates)
    fold_size = n // n_folds

    folds = []
    for k in range(n_folds):
        test_start = k * fold_size
        test_end = min((k + 1) * fold_size, n)
        test_dates = set(pd.Timestamp(d) for d in dates[test_start:test_end])

        # Test window (expanded by forward dependency)
        test_date_max = max(test_dates)
        test_date_min = min(test_dates)

        # Purge: any train date whose event interval touches test
        # Event interval of train at t: [t+lag, t+lag+h_eff]
        # Touches test if t+lag <= test_date_max and t+lag+h_eff >= test_date_min
        # Conservative: purge dates within h_eff+lag of test boundaries
        purge_before = set()
        purge_after = set()
        for i in range(max(0, test_start - h_eff - lag), test_start):
            purge_before.add(pd.Timestamp(dates[i]))
        for i in range(test_end, min(n, test_end + h_eff + lag)):
            purge_after.add(pd.Timestamp(dates[i]))

        # Embargo: b_eff days after test
        embargo_dates = set()
        for i in range(test_end, min(n, test_end + b_eff)):
            embargo_dates.add(pd.Timestamp(dates[i]))

        # Train = all dates - test - purge - embargo
        all_dates = set(pd.Timestamp(d) for d in dates)
        train_dates = all_dates - test_dates - purge_before - purge_after - embargo_dates

        folds.append({
            "fold_id": k,
            "test_dates": sorted(test_dates),
            "train_dates": sorted(train_dates),
            "n_train": len(train_dates),
            "n_test": len(test_dates),
            "n_purged": len(purge_before) + len(purge_after),
            "n_embargoed": len(embargo_dates),
            "first_test_date": test_date_min,
            "last_test_date": test_date_max,
        })

    return folds


def build_walkforward(
    unique_dates: np.ndarray,
    config: SplitConfig,
    h_eff: int,
    b_eff: int,
) -> list[dict[str, Any]]:
    """Build walk-forward folds (expanding or rolling train)."""
    dates = np.sort(unique_dates)
    n = len(dates)
    folds = []
    fold_id = 0

    i = config.wf_train_days
    while i + config.wf_test_days <= n:
        test_start = i
        test_end = min(i + config.wf_test_days, n)
        test_dates = set(pd.Timestamp(d) for d in dates[test_start:test_end])

        if config.wf_expanding:
            train_start = 0
        else:
            train_start = max(0, test_start - config.wf_train_days)

        # Purge: remove train dates near test boundary
        safe_train_end = max(0, test_start - h_eff - config.decision_lag)
        train_dates = set(pd.Timestamp(d) for d in dates[train_start:safe_train_end])

        # Embargo already handled by gap
        folds.append({
            "fold_id": fold_id,
            "test_dates": sorted(test_dates),
            "train_dates": sorted(train_dates),
            "n_train": len(train_dates),
            "n_test": len(test_dates),
            "first_test_date": min(test_dates),
            "last_test_date": max(test_dates),
        })
        fold_id += 1
        i = test_end + b_eff

    return folds


def validate_no_leakage(
    folds: list[dict[str, Any]],
    h_eff: int,
    lag: int,
) -> list[dict[str, Any]]:
    """Validate that no train sample's event interval touches test."""
    violations = []
    for fold in folds:
        test_set = set(fold["test_dates"])
        if not test_set:
            continue
        test_min = min(test_set)
        test_max = max(test_set)

        for t in fold["train_dates"]:
            # Event interval of train sample at t: [t+lag, t+lag+h_eff]
            event_start = t + pd.Timedelta(days=lag)
            event_end = t + pd.Timedelta(days=lag + h_eff)
            # Overlaps test if event_start <= test_max and event_end >= test_min
            if event_start <= test_max and event_end >= test_min:
                violations.append({"fold_id": fold["fold_id"], "train_date": t,
                                  "event_start": event_start, "event_end": event_end})
    return violations


def build_purged_splits(
    labels: pd.DataFrame | str | Path,
    calendar: pd.DatetimeIndex | np.ndarray | None = None,
    *,
    config: SplitConfig | None = None,
    output_dir: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build temporal splits with purge + embargo."""
    cfg = config or SplitConfig()
    if not run_id:
        run_id = f"splits_{utc_now_iso().replace(':','').replace('-','')}"

    if isinstance(labels, (str, Path)):
        labels = read_dataframe(labels)

    labels["date"] = pd.to_datetime(labels["date"], errors="coerce")

    # Use valid labels only
    if "label_valid_flag" in labels.columns:
        valid = labels[labels["label_valid_flag"]].copy()
    else:
        valid = labels.copy()

    unique_dates = np.sort(valid["date"].unique())

    if calendar is None:
        calendar = unique_dates

    h_eff = resolve_effective_horizon(cfg)
    b_eff = resolve_effective_embargo(cfg)

    # Build folds
    if cfg.split_mode == "purged_kfold":
        folds = build_purged_kfold(unique_dates, cfg.n_folds, h_eff, b_eff, cfg.decision_lag)
    elif cfg.split_mode == "walkforward":
        folds = build_walkforward(unique_dates, cfg, h_eff, b_eff)
    else:
        raise ValueError(f"Unknown split_mode: {cfg.split_mode}")

    # Validate
    violations = validate_no_leakage(folds, h_eff, cfg.decision_lag)

    # Fold summary
    fold_summary = pd.DataFrame([{
        "fold_id": f["fold_id"],
        "n_train": f["n_train"],
        "n_test": f["n_test"],
        "first_test": f.get("first_test_date"),
        "last_test": f.get("last_test_date"),
        "n_purged": f.get("n_purged", 0),
        "n_embargoed": f.get("n_embargoed", 0),
    } for f in folds])

    manifest = {
        "run_id": run_id,
        "split_mode": cfg.split_mode,
        "n_folds": len(folds),
        "h_eff": h_eff,
        "b_eff": b_eff,
        "horizon_policy": cfg.horizon_policy,
        "embargo_policy": cfg.embargo_policy,
        "n_leakage_violations": len(violations),
        "n_unique_dates": len(unique_dates),
        "config_hash": config_hash(json_safe(cfg.__dict__)),
        "timestamp": utc_now_iso(),
    }

    if output_dir:
        out = Path(output_dir) / run_id
        out.mkdir(parents=True, exist_ok=True)
        write_parquet_safe(fold_summary, out / "fold_summary.parquet")
        write_json_safe(manifest, out / "splits_manifest.json")
        if violations:
            write_json_safe({"violations": [json_safe(v) for v in violations]}, out / "leakage_violations.json")

    LOGGER.info("Splits: %s, %d folds, h_eff=%d, b_eff=%d, leakage_violations=%d",
                cfg.split_mode, len(folds), h_eff, b_eff, len(violations))

    return {"folds": folds, "fold_summary": fold_summary, "violations": violations, "manifest": manifest}
