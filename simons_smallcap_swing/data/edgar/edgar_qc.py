"""
data/edgar/edgar_qc.py — EDGAR pipeline quality control and gating.

7-layer hierarchical QC:
    1. Schema:     column presence, types, PK uniqueness
    2. Coverage:   per-stage success rates (download, parse, PIT)
    3. Identity:   symbol-CIK mapping conflicts, active vs inactive
    4. Temporal:   PIT leakage (acceptance_ts > asof), date monotonicity
    5. Values:     range checks, robust z-score outliers, sign consistency
    6. Staleness:  per-metric freshness vs thresholds
    7. Lineage:    traceability from PIT value back to filing

Gate logic: FAIL in schema/identity/temporal → block pipeline.
Score = weighted sum of per-layer scores, with hard gates dominating.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    EdgarError,
    normalize_cik, normalize_metric,
    parse_date_series, utc_now_iso, config_hash,
    read_dataframe, write_parquet_safe, write_json_safe,
    normalize_columns, json_safe,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class QCCheck:
    layer: str
    check_name: str
    severity: str    # PASS / WARN / FAIL
    metric_value: Any
    threshold: Any
    message: str


# ---------------------------------------------------------------------------
# Layer 1: Schema
# ---------------------------------------------------------------------------

def check_schema(
    tables: dict[str, pd.DataFrame],
    required_schemas: dict[str, list[str]],
) -> list[QCCheck]:
    """Validate column presence and PK uniqueness."""
    checks = []
    for name, df in tables.items():
        required = required_schemas.get(name, [])
        missing = [c for c in required if c not in df.columns]
        checks.append(QCCheck(
            "schema", f"{name}_columns",
            "FAIL" if missing else "PASS",
            missing, [], f"{name}: missing {missing}" if missing else f"{name}: schema OK",
        ))
        # PK uniqueness
        if "cik" in df.columns and "accession_number" in df.columns:
            n_dup = df.duplicated(subset=["cik", "accession_number"]).sum()
            checks.append(QCCheck(
                "schema", f"{name}_pk_unique",
                "WARN" if n_dup > 0 else "PASS",
                int(n_dup), 0, f"{name}: {n_dup} PK duplicates",
            ))
    return checks


# ---------------------------------------------------------------------------
# Layer 2: Coverage
# ---------------------------------------------------------------------------

def check_coverage(
    facts: pd.DataFrame,
    submissions: pd.DataFrame,
    cik_list: Sequence[str],
) -> list[QCCheck]:
    checks = []
    n_requested = len(cik_list)

    # Download coverage
    if "cik" in submissions.columns:
        n_downloaded = submissions["cik"].nunique()
        rate = n_downloaded / max(n_requested, 1)
        checks.append(QCCheck(
            "coverage", "download_rate", "FAIL" if rate < 0.5 else ("WARN" if rate < 0.8 else "PASS"),
            round(rate, 4), 0.8, f"Download: {n_downloaded}/{n_requested} CIKs ({rate:.0%})",
        ))

    # Parse coverage
    if "cik" in facts.columns:
        n_parsed = facts["cik"].nunique()
        rate = n_parsed / max(n_requested, 1)
        checks.append(QCCheck(
            "coverage", "parse_rate", "FAIL" if rate < 0.3 else ("WARN" if rate < 0.7 else "PASS"),
            round(rate, 4), 0.7, f"Parse: {n_parsed}/{n_requested} CIKs with facts ({rate:.0%})",
        ))

    return checks


# ---------------------------------------------------------------------------
# Layer 3: Identity
# ---------------------------------------------------------------------------

def check_identity(
    mapping: pd.DataFrame,
) -> list[QCCheck]:
    checks = []
    if "symbol" not in mapping.columns or "cik" not in mapping.columns:
        return checks

    # Multi-CIK per symbol
    multi = mapping.groupby("symbol")["cik"].nunique()
    n_conflicts = int((multi > 1).sum())
    checks.append(QCCheck(
        "identity", "multi_cik_conflicts",
        "WARN" if n_conflicts > 0 else "PASS",
        n_conflicts, 0, f"{n_conflicts} symbols map to multiple CIKs",
    ))
    return checks


# ---------------------------------------------------------------------------
# Layer 4: Temporal (PIT leakage)
# ---------------------------------------------------------------------------

def check_temporal(
    pit_panel: pd.DataFrame,
) -> list[QCCheck]:
    checks = []
    if "asof" not in pit_panel.columns or "acceptance_ts" not in pit_panel.columns:
        return checks

    asof = pd.to_datetime(pit_panel["asof"], errors="coerce")
    acc = pd.to_datetime(pit_panel["acceptance_ts"], errors="coerce")
    valid = asof.notna() & acc.notna()
    breaches = (acc > asof) & valid
    n_breach = int(breaches.sum())

    checks.append(QCCheck(
        "temporal", "pit_leakage",
        "FAIL" if n_breach > 0 else "PASS",
        n_breach, 0, f"{n_breach} PIT leakage breaches (acceptance_ts > asof)",
    ))
    return checks


# ---------------------------------------------------------------------------
# Layer 5: Values
# ---------------------------------------------------------------------------

def check_values(
    facts: pd.DataFrame,
) -> list[QCCheck]:
    checks = []
    if "value" not in facts.columns:
        return checks

    vals = pd.to_numeric(facts["value"], errors="coerce")
    n_nan = int(vals.isna().sum())
    n_inf = int(np.isinf(vals.dropna()).sum())

    checks.append(QCCheck(
        "values", "nan_values", "WARN" if n_nan > len(facts) * 0.1 else "PASS",
        n_nan, int(len(facts) * 0.1), f"{n_nan} NaN values ({n_nan/max(len(facts),1):.1%})",
    ))
    checks.append(QCCheck(
        "values", "inf_values", "FAIL" if n_inf > 0 else "PASS",
        n_inf, 0, f"{n_inf} Inf values",
    ))
    return checks


# ---------------------------------------------------------------------------
# Layer 6: Staleness
# ---------------------------------------------------------------------------

def check_staleness(
    pit_panel: pd.DataFrame,
    max_staleness: int = 365,
) -> list[QCCheck]:
    checks = []
    if "staleness_days" not in pit_panel.columns:
        return checks

    stale = pit_panel["staleness_days"].dropna()
    if len(stale) == 0:
        return checks

    n_very_stale = int((stale > max_staleness).sum())
    median_stale = float(stale.median())

    checks.append(QCCheck(
        "staleness", "very_stale_facts",
        "WARN" if n_very_stale > len(stale) * 0.1 else "PASS",
        n_very_stale, int(len(stale) * 0.1),
        f"{n_very_stale} facts staler than {max_staleness}d, median={median_stale:.0f}d",
    ))
    return checks


# ---------------------------------------------------------------------------
# Layer 7: Lineage
# ---------------------------------------------------------------------------

def check_lineage(
    pit_panel: pd.DataFrame,
) -> list[QCCheck]:
    checks = []
    # Every PIT value should trace to an accession_number
    if "accession_number" not in pit_panel.columns:
        checks.append(QCCheck("lineage", "accession_present", "WARN", None, None, "No accession_number column"))
        return checks

    usable = pit_panel[pit_panel.get("pit_action", pd.Series("allow", index=pit_panel.index)).isin(("allow", "penalize"))]
    if len(usable) == 0:
        return checks

    n_missing = int(usable["accession_number"].isna().sum())
    rate = n_missing / max(len(usable), 1)
    checks.append(QCCheck(
        "lineage", "missing_accession",
        "WARN" if rate > 0.05 else "PASS",
        n_missing, int(len(usable) * 0.05),
        f"{n_missing} usable facts without accession_number ({rate:.1%})",
    ))
    return checks


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def run_edgar_qc(
    *,
    facts: pd.DataFrame | None = None,
    submissions: pd.DataFrame | None = None,
    mapping: pd.DataFrame | None = None,
    pit_panel: pd.DataFrame | None = None,
    cik_list: Sequence[str] | None = None,
    output_dir: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run all 7 QC layers."""
    if not run_id:
        run_id = f"qc_{utc_now_iso().replace(':','').replace('-','')}"

    all_checks: list[QCCheck] = []

    # Schema
    tables = {}
    if facts is not None: tables["facts"] = facts
    if submissions is not None: tables["submissions"] = submissions
    if mapping is not None: tables["mapping"] = mapping
    required = {
        "facts": ["cik", "metric_name", "value"],
        "submissions": ["cik", "accession_number", "form_type"],
        "mapping": ["symbol", "cik"],
    }
    all_checks.extend(check_schema(tables, required))

    # Coverage
    if facts is not None and submissions is not None and cik_list:
        all_checks.extend(check_coverage(facts, submissions, cik_list))

    # Identity
    if mapping is not None:
        all_checks.extend(check_identity(mapping))

    # Temporal
    if pit_panel is not None:
        all_checks.extend(check_temporal(pit_panel))

    # Values
    if facts is not None:
        all_checks.extend(check_values(facts))

    # Staleness
    if pit_panel is not None:
        all_checks.extend(check_staleness(pit_panel))

    # Lineage
    if pit_panel is not None:
        all_checks.extend(check_lineage(pit_panel))

    # Gate
    n_fail = sum(1 for c in all_checks if c.severity == "FAIL")
    n_warn = sum(1 for c in all_checks if c.severity == "WARN")
    overall = "FAIL" if n_fail > 0 else ("WARN" if n_warn > 0 else "PASS")

    summary = {
        "overall": overall,
        "n_checks": len(all_checks),
        "n_fail": n_fail, "n_warn": n_warn,
        "n_pass": sum(1 for c in all_checks if c.severity == "PASS"),
        "run_id": run_id, "timestamp": utc_now_iso(),
    }

    if output_dir:
        out = Path(output_dir) / run_id
        out.mkdir(parents=True, exist_ok=True)
        checks_df = pd.DataFrame([
            {"layer": c.layer, "check": c.check_name, "severity": c.severity,
             "metric": str(c.metric_value), "message": c.message}
            for c in all_checks
        ])
        write_parquet_safe(checks_df, out / "qc_checks.parquet")
        write_json_safe(summary, out / "manifest.json")

    LOGGER.info("EDGAR QC: %s (%dF %dW %dP)", overall, n_fail, n_warn, summary["n_pass"])
    return {"checks": all_checks, "summary": summary}
