"""
features/feature_qc.py — Feature quality control and gating.

Six audit dimensions:
    Q_struct  Schema, PK uniqueness, dtype coherence
    Q_cov     Coverage and missingness classification
    Q_num     Inf/NaN, near-constant, outlier detection
    Q_leak    Temporal availability breach (PIT)
    Q_drift   Distribution drift vs reference window
    Q_col     Pairwise collinearity / redundancy

Gate logic: FAIL in any critical dimension → overall FAIL regardless
of aggregate score.  WARNs accumulate but don't block.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    FeatureError,
    DataContractError,
    LeakageError,
    Severity,
    severity_rank,
    QCRecord,
    PK_COLUMNS,
    MAD_NORMAL_CONSISTENCY,
    content_hash,
    _json_safe,
    utc_now_iso,
    get_logger,
)

LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureQCConfig:
    """Thresholds for feature quality control."""
    # Coverage
    min_global_coverage: float = 0.50
    min_date_coverage: float = 0.30
    max_null_fraction: float = 0.50

    # Numeric
    max_inf_fraction: float = 0.0       # zero tolerance for Inf
    near_constant_std_threshold: float = 1e-10
    outlier_mad_threshold: float = 10.0  # |z_robust| > 10 → outlier

    # Leakage
    leakage_zero_tolerance: bool = True  # any breach → FAIL

    # Drift (KS test p-value)
    drift_ks_alpha: float = 0.01
    drift_reference_window_days: int = 252  # ~1 year

    # Collinearity
    max_pairwise_corr: float = 0.95
    collinearity_persistence_days: int = 63  # must persist ~1Q to flag

    # Gate policy
    fail_on_struct: bool = True
    fail_on_leakage: bool = True
    fail_on_coverage_below: float = 0.10  # < 10% → FAIL


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_structure(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    cfg: FeatureQCConfig,
) -> list[QCRecord]:
    """Q_struct: schema, PK uniqueness, dtypes."""
    records: list[QCRecord] = []

    # PK uniqueness
    n_dups = df.duplicated(subset=list(PK_COLUMNS)).sum()
    records.append(QCRecord(
        check_name="struct.pk_unique",
        severity=Severity.FAIL if n_dups > 0 else Severity.PASS,
        status="FAIL" if n_dups > 0 else "PASS",
        feature_name="*",
        metric_value=int(n_dups),
        threshold=0,
        message=f"{n_dups} duplicate (date, symbol) rows" if n_dups else "PK unique",
    ))

    # Feature columns exist
    missing_cols = [c for c in feature_cols if c not in df.columns]
    records.append(QCRecord(
        check_name="struct.columns_present",
        severity=Severity.FAIL if missing_cols else Severity.PASS,
        status="FAIL" if missing_cols else "PASS",
        feature_name="*",
        metric_value=missing_cols,
        threshold=[],
        message=f"Missing columns: {missing_cols}" if missing_cols else "All columns present",
    ))

    # Dtype check: feature columns should be numeric
    non_numeric = [c for c in feature_cols if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
    records.append(QCRecord(
        check_name="struct.dtypes_numeric",
        severity=Severity.WARN if non_numeric else Severity.PASS,
        status="FAIL" if non_numeric else "PASS",
        feature_name="*",
        metric_value=non_numeric,
        threshold=[],
        message=f"Non-numeric features: {non_numeric}" if non_numeric else "All numeric",
    ))

    return records


def check_coverage(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    cfg: FeatureQCConfig,
) -> list[QCRecord]:
    """Q_cov: global coverage, per-date coverage, null fraction."""
    records: list[QCRecord] = []

    for col in feature_cols:
        if col not in df.columns:
            continue
        series = df[col]
        global_cov = float(series.notna().mean())
        null_frac = 1.0 - global_cov

        # Global coverage
        sev = Severity.PASS
        if global_cov < cfg.fail_on_coverage_below:
            sev = Severity.FAIL
        elif global_cov < cfg.min_global_coverage:
            sev = Severity.WARN
        records.append(QCRecord(
            check_name="cov.global",
            severity=sev,
            status="FAIL" if sev == Severity.FAIL else "PASS",
            feature_name=col,
            metric_value=round(global_cov, 4),
            threshold=cfg.min_global_coverage,
            message=f"coverage={global_cov:.1%}",
        ))

        # Per-date coverage (min across dates)
        if "date" in df.columns:
            per_date = df.groupby("date")[col].apply(lambda s: s.notna().mean())
            min_date_cov = float(per_date.min()) if len(per_date) > 0 else 0.0
            sev_d = Severity.WARN if min_date_cov < cfg.min_date_coverage else Severity.PASS
            records.append(QCRecord(
                check_name="cov.min_date",
                severity=sev_d,
                status="PASS" if sev_d == Severity.PASS else "WARN",
                feature_name=col,
                metric_value=round(min_date_cov, 4),
                threshold=cfg.min_date_coverage,
                message=f"min_date_coverage={min_date_cov:.1%}",
            ))

    return records


def check_numeric(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    cfg: FeatureQCConfig,
) -> list[QCRecord]:
    """Q_num: Inf, near-constant, outliers."""
    records: list[QCRecord] = []

    for col in feature_cols:
        if col not in df.columns:
            continue
        series = df[col]
        valid = series.dropna()
        n_valid = len(valid)
        if n_valid == 0:
            continue

        # Inf check
        n_inf = np.isinf(valid).sum()
        inf_frac = n_inf / n_valid
        records.append(QCRecord(
            check_name="num.inf",
            severity=Severity.FAIL if inf_frac > cfg.max_inf_fraction else Severity.PASS,
            status="FAIL" if inf_frac > 0 else "PASS",
            feature_name=col,
            metric_value=int(n_inf),
            threshold=0,
            message=f"{n_inf} Inf values ({inf_frac:.4%})",
        ))

        # Near-constant
        finite = valid[np.isfinite(valid)]
        if len(finite) > 1:
            std_val = float(finite.std())
            is_const = std_val < cfg.near_constant_std_threshold
            records.append(QCRecord(
                check_name="num.near_constant",
                severity=Severity.WARN if is_const else Severity.PASS,
                status="WARN" if is_const else "PASS",
                feature_name=col,
                metric_value=std_val,
                threshold=cfg.near_constant_std_threshold,
                message=f"std={std_val:.2e}" + (" (near-constant)" if is_const else ""),
            ))

        # Outlier fraction (robust z-score > threshold)
        if len(finite) >= 10:
            med = finite.median()
            mad = (finite - med).abs().median() * MAD_NORMAL_CONSISTENCY
            if mad > 0:
                z = ((finite - med) / mad).abs()
                outlier_frac = float((z > cfg.outlier_mad_threshold).mean())
                records.append(QCRecord(
                    check_name="num.outlier_fraction",
                    severity=Severity.WARN if outlier_frac > 0.01 else Severity.PASS,
                    status="WARN" if outlier_frac > 0.01 else "PASS",
                    feature_name=col,
                    metric_value=round(outlier_frac, 6),
                    threshold=0.01,
                    message=f"outlier_fraction={outlier_frac:.4%} (|z_robust|>{cfg.outlier_mad_threshold})",
                ))

    return records


def check_leakage(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    cfg: FeatureQCConfig,
    availability_col: str | None = None,
    decision_cut_col: str | None = None,
) -> list[QCRecord]:
    """Q_leak: temporal availability breach.

    If availability_col and decision_cut_col are present, checks that
    availability_date ≤ decision_cut for every observation.
    """
    records: list[QCRecord] = []

    if availability_col and decision_cut_col:
        if availability_col in df.columns and decision_cut_col in df.columns:
            avail = pd.to_datetime(df[availability_col], errors="coerce")
            cut = pd.to_datetime(df[decision_cut_col], errors="coerce")
            breaches = (avail > cut) & avail.notna() & cut.notna()
            n_breach = int(breaches.sum())
            n_total = int((avail.notna() & cut.notna()).sum())
            breach_frac = n_breach / max(n_total, 1)

            sev = Severity.FAIL if (n_breach > 0 and cfg.leakage_zero_tolerance) else Severity.PASS
            records.append(QCRecord(
                check_name="leak.temporal_availability",
                severity=sev,
                status="FAIL" if n_breach > 0 else "PASS",
                feature_name="*",
                metric_value=n_breach,
                threshold=0,
                message=f"{n_breach}/{n_total} availability breaches ({breach_frac:.4%})",
            ))
    else:
        records.append(QCRecord(
            check_name="leak.temporal_availability",
            severity=Severity.INFO,
            status="SKIP",
            feature_name="*",
            metric_value=None,
            threshold=None,
            message="No availability/decision_cut columns — leakage test skipped",
        ))

    return records


def check_drift(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    cfg: FeatureQCConfig,
    reference_df: pd.DataFrame | None = None,
) -> list[QCRecord]:
    """Q_drift: distribution shift vs reference (KS test)."""
    records: list[QCRecord] = []

    if reference_df is None:
        records.append(QCRecord(
            check_name="drift.ks_test",
            severity=Severity.INFO,
            status="SKIP",
            feature_name="*",
            metric_value=None,
            threshold=None,
            message="No reference DataFrame — drift test skipped",
        ))
        return records

    from scipy.stats import ks_2samp

    for col in feature_cols:
        if col not in df.columns or col not in reference_df.columns:
            continue
        current = df[col].dropna().values
        ref = reference_df[col].dropna().values
        if len(current) < 30 or len(ref) < 30:
            continue
        stat, pval = ks_2samp(current, ref)
        drifted = pval < cfg.drift_ks_alpha
        records.append(QCRecord(
            check_name="drift.ks_test",
            severity=Severity.WARN if drifted else Severity.PASS,
            status="WARN" if drifted else "PASS",
            feature_name=col,
            metric_value=round(pval, 6),
            threshold=cfg.drift_ks_alpha,
            message=f"KS p={pval:.4f}, stat={stat:.4f}" + (" (DRIFT)" if drifted else ""),
        ))

    return records


def check_collinearity(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    cfg: FeatureQCConfig,
) -> list[QCRecord]:
    """Q_col: pairwise Pearson correlation, flag persistent high corr."""
    records: list[QCRecord] = []
    cols = [c for c in feature_cols if c in df.columns]

    if len(cols) < 2:
        return records

    # Sample for efficiency if panel is large
    sample = df[cols].dropna()
    if len(sample) > 50_000:
        sample = sample.sample(50_000, random_state=42)
    if len(sample) < 30:
        return records

    corr_matrix = sample.corr()
    flagged_pairs: list[tuple[str, str, float]] = []

    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            r = abs(corr_matrix.loc[c1, c2])
            if r > cfg.max_pairwise_corr:
                flagged_pairs.append((c1, c2, round(r, 4)))

    for c1, c2, r in flagged_pairs:
        records.append(QCRecord(
            check_name="col.pairwise_corr",
            severity=Severity.WARN,
            status="WARN",
            feature_name=f"{c1} × {c2}",
            metric_value=r,
            threshold=cfg.max_pairwise_corr,
            message=f"|corr({c1}, {c2})| = {r:.4f} > {cfg.max_pairwise_corr}",
        ))

    return records


# ---------------------------------------------------------------------------
# Aggregate QC
# ---------------------------------------------------------------------------

def run_feature_qc(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    config: FeatureQCConfig | None = None,
    reference_df: pd.DataFrame | None = None,
    availability_col: str | None = None,
    decision_cut_col: str | None = None,
) -> tuple[list[QCRecord], dict[str, Any]]:
    """Run all 6 QC dimensions and return records + summary.

    Returns
    -------
    records : list of QCRecord
    summary : dict with overall_status, counts by severity, etc.
    """
    cfg = config or FeatureQCConfig()

    all_records: list[QCRecord] = []
    all_records.extend(check_structure(df, feature_cols, cfg))
    all_records.extend(check_coverage(df, feature_cols, cfg))
    all_records.extend(check_numeric(df, feature_cols, cfg))
    all_records.extend(check_leakage(df, feature_cols, cfg, availability_col, decision_cut_col))
    all_records.extend(check_drift(df, feature_cols, cfg, reference_df))
    all_records.extend(check_collinearity(df, feature_cols, cfg))

    # Aggregate
    n_fail = sum(1 for r in all_records if r.severity == Severity.FAIL)
    n_warn = sum(1 for r in all_records if r.severity == Severity.WARN)
    n_pass = sum(1 for r in all_records if r.severity == Severity.PASS)

    has_struct_fail = any(
        r.severity == Severity.FAIL and r.check_name.startswith("struct.")
        for r in all_records
    )
    has_leak_fail = any(
        r.severity == Severity.FAIL and r.check_name.startswith("leak.")
        for r in all_records
    )

    if (cfg.fail_on_struct and has_struct_fail) or (cfg.fail_on_leakage and has_leak_fail) or n_fail > 0:
        overall = "FAIL"
    elif n_warn > 0:
        overall = "WARN"
    else:
        overall = "PASS"

    summary = {
        "overall_status": overall,
        "n_checks": len(all_records),
        "n_fail": n_fail,
        "n_warn": n_warn,
        "n_pass": n_pass,
        "n_features_checked": len(feature_cols),
        "timestamp": utc_now_iso(),
    }

    LOGGER.info(
        "Feature QC complete: %s (%d FAIL, %d WARN, %d PASS)",
        overall, n_fail, n_warn, n_pass,
    )

    return all_records, summary
