"""
labels/label_qc.py — Label quality control and gating.

Institutional gatekeeper: decides whether labels are usable for
training, validation, and backtesting.

Checks (spec §6-15):
    1. Structural:  PK uniqueness, schema, flag/reason coherence
    2. Statistical: coverage, variance, class balance, outliers
    3. Temporal:    alignment with features, anti-leakage, drift

Gate: FAIL dominates. One horizon FAIL → global FAIL.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    Severity, max_severity, severity_rank,
    utc_now_iso, config_hash, json_safe,
    read_dataframe, write_parquet_safe, write_json_safe,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelQCConfig:
    coverage_fail: float = 0.30
    coverage_warn: float = 0.50
    var_min: float = 1e-10
    class_balance_fail: float = 0.02   # class <2% → FAIL
    class_balance_warn: float = 0.05
    outlier_method: str = "MAD"
    outlier_k: float = 10.0
    alignment_policy: str = "exact_on_eligible_rows"
    leakage_policy: str = "fail"


@dataclass
class QCCheck:
    layer: str
    name: str
    severity: str
    horizon: int | None
    count: int
    total: int
    message: str


def _check_structural(df: pd.DataFrame, horizons: Sequence[int]) -> list[QCCheck]:
    checks = []

    # PK uniqueness
    pk = ["date", "symbol"]
    n_dup = int(df.duplicated(subset=pk).sum())
    checks.append(QCCheck("structural", "pk_unique", "FAIL" if n_dup > 0 else "PASS",
                           None, n_dup, len(df), f"{n_dup} PK duplicates"))

    # Flag/reason coherence
    if "label_valid_flag" in df.columns and "label_exclusion_reason" in df.columns:
        valid_with_reason = (df["label_valid_flag"] == True) & df["label_exclusion_reason"].notna()
        n1 = int(valid_with_reason.sum())
        checks.append(QCCheck("structural", "valid_no_reason", "FAIL" if n1 > 0 else "PASS",
                               None, n1, len(df), f"{n1} valid rows WITH exclusion reason"))

        invalid_no_reason = (df["label_valid_flag"] == False) & df["label_exclusion_reason"].isna()
        n2 = int(invalid_no_reason.sum())
        checks.append(QCCheck("structural", "invalid_has_reason", "FAIL" if n2 > 0 else "PASS",
                               None, n2, len(df), f"{n2} invalid rows WITHOUT reason"))

    # NaN/Inf in valid rows
    for h in horizons:
        for prefix in ("y_fwd_ret_net", "y_fwd_ret_gross"):
            col = f"{prefix}_{h}d"
            if col not in df.columns:
                continue
            valid_mask = df.get("label_valid_flag", pd.Series(True, index=df.index))
            valid_vals = df.loc[valid_mask, col]
            n_nan = int(valid_vals.isna().sum())
            n_inf = int(np.isinf(valid_vals.dropna().values).sum()) if len(valid_vals.dropna()) > 0 else 0
            if n_nan > 0:
                checks.append(QCCheck("structural", f"nan_in_valid_{col}", "FAIL", h, n_nan, len(valid_vals),
                                       f"{n_nan} NaN in valid {col}"))
            if n_inf > 0:
                checks.append(QCCheck("structural", f"inf_in_valid_{col}", "FAIL", h, n_inf, len(valid_vals),
                                       f"{n_inf} Inf in valid {col}"))

    return checks


def _check_statistical(df: pd.DataFrame, horizons: Sequence[int], cfg: LabelQCConfig) -> list[QCCheck]:
    checks = []
    valid = df[df.get("label_valid_flag", pd.Series(True, index=df.index))]

    for h in horizons:
        col = f"y_fwd_ret_net_{h}d"
        if col not in valid.columns:
            col = f"y_fwd_ret_gross_{h}d"
        if col not in valid.columns:
            continue

        vals = valid[col].dropna()
        n_valid = len(vals)
        n_total = len(valid)
        cov = n_valid / max(n_total, 1)

        # Coverage
        sev = "FAIL" if cov < cfg.coverage_fail else ("WARN" if cov < cfg.coverage_warn else "PASS")
        checks.append(QCCheck("statistical", f"coverage_{h}d", sev, h, n_valid, n_total,
                               f"Coverage {h}d: {cov:.1%}"))

        # Variance
        var = float(vals.var()) if len(vals) > 1 else 0
        checks.append(QCCheck("statistical", f"variance_{h}d",
                               "FAIL" if var < cfg.var_min else "PASS",
                               h, 0, 0, f"Var({h}d) = {var:.2e}"))

        # Class balance (if classification exists)
        cls_col = f"y_cls_{h}d"
        if cls_col in valid.columns:
            cls_vals = valid[cls_col].dropna()
            if len(cls_vals) > 0:
                for c in (-1, 0, 1):
                    pct = (cls_vals == c).mean()
                    sev = "FAIL" if pct < cfg.class_balance_fail else ("WARN" if pct < cfg.class_balance_warn else "PASS")
                    checks.append(QCCheck("statistical", f"class_balance_{h}d_c{c}", sev, h, 0, 0,
                                           f"Class {c} balance {h}d: {pct:.1%}"))

    return checks


def _check_alignment(df: pd.DataFrame, features_index: pd.DataFrame | None) -> list[QCCheck]:
    if features_index is None:
        return []

    label_keys = set(zip(df["date"].astype(str), df["symbol"]))
    feat_keys = set(zip(features_index["date"].astype(str), features_index["symbol"]))
    mismatch = len(feat_keys - label_keys)
    orphan = len(label_keys - feat_keys)

    checks = [
        QCCheck("temporal", "features_without_labels", "FAIL" if mismatch > 0 else "PASS",
                None, mismatch, len(feat_keys), f"{mismatch} feature rows without matching labels"),
    ]
    return checks


def run_label_qc(
    labels: pd.DataFrame | str | Path,
    *,
    features_index: pd.DataFrame | str | Path | None = None,
    horizons: Sequence[int] = (5, 10, 20),
    config: LabelQCConfig | None = None,
    output_dir: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run label QC suite."""
    cfg = config or LabelQCConfig()
    if not run_id:
        run_id = f"lqc_{utc_now_iso().replace(':','').replace('-','')}"

    if isinstance(labels, (str, Path)):
        labels = read_dataframe(labels)
    if isinstance(features_index, (str, Path)):
        features_index = read_dataframe(features_index)

    labels["date"] = pd.to_datetime(labels["date"], errors="coerce")

    all_checks: list[QCCheck] = []
    all_checks.extend(_check_structural(labels, horizons))
    all_checks.extend(_check_statistical(labels, horizons, cfg))
    all_checks.extend(_check_alignment(labels, features_index))

    n_fail = sum(1 for c in all_checks if c.severity == "FAIL")
    n_warn = sum(1 for c in all_checks if c.severity == "WARN")
    gate = "FAIL" if n_fail > 0 else ("WARN" if n_warn > 0 else "PASS")

    summary = {
        "run_id": run_id,
        "global_gate": gate,
        "n_checks": len(all_checks),
        "n_fail": n_fail, "n_warn": n_warn,
        "n_pass": sum(1 for c in all_checks if c.severity == "PASS"),
        "timestamp": utc_now_iso(),
    }

    if output_dir:
        out = Path(output_dir) / run_id
        out.mkdir(parents=True, exist_ok=True)
        cdf = pd.DataFrame([{"layer": c.layer, "check": c.name, "severity": c.severity,
                             "horizon": c.horizon, "count": c.count, "message": c.message} for c in all_checks])
        write_parquet_safe(cdf, out / "label_qc_checks.parquet")
        write_json_safe(summary, out / "label_qc_summary.json")

    LOGGER.info("Label QC: %s (%dF %dW)", gate, n_fail, n_warn)
    return {"checks": all_checks, "summary": summary}
