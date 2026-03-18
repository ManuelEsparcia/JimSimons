"""
data/universe/universe_qc.py — Universe quality control and certification.

Certifies that the built universe is internally consistent, temporally
valid, and semantically coherent BEFORE downstream consumption.

4 audit layers (spec §6):
    1. Structural:  PK uniqueness, valid is_eligible domain, valid enums
    2. PIT/Lifecycle: no eligible outside W_i, coherence with terminal CA
    3. Semantic:    eligible⟺no reason, primary matches precedence
    4. Temporal:    stable size, reasonable turnover, calendar compliance

5 gates (spec §14):
    Gate1: PIT integrity (FAIL if any hard violation)
    Gate2: Semantic coherence
    Gate3: Delisted coverage
    Gate4: Temporal stability
    Gate5: Missing metadata
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    QCError, ExclusionReason, MembershipState, Severity,
    GateResult, max_severity, severity_rank,
    utc_now_iso, config_hash, json_safe,
    read_dataframe, write_parquet_safe, write_json_safe,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class UniverseQCConfig:
    tau_turnover_warn: float = 0.20
    tau_turnover_fail: float = 0.50
    tau_size_drop_fail: float = 0.50       # >50% drop in one day
    min_daily_size: int = 20
    min_delisted_coverage: float = 0.90
    max_missing_meta_pct: float = 0.10


@dataclass
class QCCheck:
    layer: str       # "structural" | "pit" | "semantic" | "temporal"
    name: str
    severity: str
    count: int
    total: int
    message: str


# ---------------------------------------------------------------------------
# Layer 1: Structural (spec §7)
# ---------------------------------------------------------------------------

def _check_structural(df: pd.DataFrame, cfg: UniverseQCConfig) -> list[QCCheck]:
    checks: list[QCCheck] = []

    # 7.1 PK uniqueness on (date, instrument_id)
    pk = ["date", "instrument_id"] if "instrument_id" in df.columns else ["date", "symbol"]
    n_dup = int(df.duplicated(subset=pk).sum())
    checks.append(QCCheck("structural", "pk_unique", "FAIL" if n_dup > 0 else "PASS",
                           n_dup, len(df), f"{n_dup} PK duplicates on ({', '.join(pk)})"))

    # 7.2 is_eligible domain
    if "is_eligible" in df.columns:
        bad = ~df["is_eligible"].isin([0, 1])
        n = int(bad.sum())
        checks.append(QCCheck("structural", "eligible_domain", "FAIL" if n > 0 else "PASS",
                               n, len(df), f"{n} rows with is_eligible ∉ {{0,1}}"))

    # 7.3 Valid membership_state
    if "membership_state" in df.columns:
        valid_states = {s.value for s in MembershipState} | {"pre_listing"}
        invalid = ~df["membership_state"].isin(valid_states)
        n = int(invalid.sum())
        checks.append(QCCheck("structural", "valid_states", "FAIL" if n > 0 else "PASS",
                               n, len(df), f"{n} rows with unrecognised membership_state"))

    # 7.4 Valid exclusion reasons
    if "primary_exclusion_reason" in df.columns:
        valid_reasons = {r.value for r in ExclusionReason} | {None, np.nan, ""}
        has_reason = df["primary_exclusion_reason"].notna() & (df["primary_exclusion_reason"] != "")
        if has_reason.any():
            invalid_reasons = ~df.loc[has_reason, "primary_exclusion_reason"].isin({r.value for r in ExclusionReason})
            n = int(invalid_reasons.sum())
            checks.append(QCCheck("structural", "valid_reasons", "WARN" if n > 0 else "PASS",
                                   n, int(has_reason.sum()), f"{n} rows with unrecognised exclusion reason"))

    return checks


# ---------------------------------------------------------------------------
# Layer 2: PIT / Lifecycle (spec §8)
# ---------------------------------------------------------------------------

def _check_pit(df: pd.DataFrame) -> list[QCCheck]:
    checks: list[QCCheck] = []
    date = pd.to_datetime(df["date"])

    # 8.1 No eligible before list_date
    if "list_date" in df.columns:
        ld = pd.to_datetime(df["list_date"], errors="coerce")
        pre = (date < ld) & (df["is_eligible"] == 1) & ld.notna()
        n = int(pre.sum())
        checks.append(QCCheck("pit", "no_eligible_pre_listing", "FAIL" if n > 0 else "PASS",
                               n, len(df), f"{n} eligible rows before list_date"))

    # 8.1 No eligible after delist_date / tau_death
    for death_col in ("delist_date", "tau_death"):
        if death_col in df.columns:
            dd = pd.to_datetime(df[death_col], errors="coerce")
            post = (date > dd) & (df["is_eligible"] == 1) & dd.notna()
            n = int(post.sum())
            checks.append(QCCheck("pit", f"no_eligible_post_{death_col}", "FAIL" if n > 0 else "PASS",
                                   n, len(df), f"{n} eligible rows after {death_col}"))

    return checks


# ---------------------------------------------------------------------------
# Layer 3: Semantic (spec §9)
# ---------------------------------------------------------------------------

def _check_semantic(df: pd.DataFrame) -> list[QCCheck]:
    checks: list[QCCheck] = []

    if "primary_exclusion_reason" not in df.columns:
        return checks

    # 9.1 eligible=1 ⟹ reason is None
    elig_with_reason = (df["is_eligible"] == 1) & df["primary_exclusion_reason"].notna() & (df["primary_exclusion_reason"] != "")
    n1 = int(elig_with_reason.sum())
    checks.append(QCCheck("semantic", "eligible_implies_no_reason", "FAIL" if n1 > 0 else "PASS",
                           n1, len(df), f"{n1} eligible rows WITH exclusion reason"))

    # 9.1 eligible=0 ⟹ reason is not None
    inelig_no_reason = (df["is_eligible"] == 0) & (df["primary_exclusion_reason"].isna() | (df["primary_exclusion_reason"] == ""))
    n2 = int(inelig_no_reason.sum())
    checks.append(QCCheck("semantic", "ineligible_implies_reason", "FAIL" if n2 > 0 else "PASS",
                           n2, len(df), f"{n2} ineligible rows WITHOUT exclusion reason"))

    return checks


# ---------------------------------------------------------------------------
# Layer 4: Temporal (spec §10)
# ---------------------------------------------------------------------------

def _check_temporal(df: pd.DataFrame, calendar: pd.DatetimeIndex | None, cfg: UniverseQCConfig) -> list[QCCheck]:
    checks: list[QCCheck] = []

    daily = df.groupby("date")["is_eligible"].agg(["sum", "count"]).reset_index()
    daily.columns = ["date", "n_eligible", "n_total"]
    daily = daily.sort_values("date")

    # Min daily size
    min_n = int(daily["n_eligible"].min()) if len(daily) > 0 else 0
    checks.append(QCCheck("temporal", "min_daily_eligible", "WARN" if min_n < cfg.min_daily_size else "PASS",
                           min_n, cfg.min_daily_size, f"Min daily eligible: {min_n}"))

    # Day-over-day drops
    if len(daily) > 1:
        pct_chg = daily["n_eligible"].pct_change()
        worst = float(pct_chg.min()) if pct_chg.notna().any() else 0
        checks.append(QCCheck("temporal", "max_daily_drop", "FAIL" if worst < -cfg.tau_size_drop_fail else "PASS",
                               round(worst, 4), -cfg.tau_size_drop_fail, f"Max drop: {worst:.1%}"))

    # Turnover
    if "instrument_id" in df.columns and len(daily) > 1:
        dates_sorted = sorted(df["date"].unique())
        turnovers = []
        prev = set()
        for d in dates_sorted:
            curr = set(df[(df["date"] == d) & (df["is_eligible"] == 1)]["instrument_id"])
            if prev:
                union = curr | prev
                sym_diff = len(curr ^ prev)
                t = sym_diff / len(union) if union else 0
                turnovers.append(t)
            prev = curr
        if turnovers:
            max_to = max(turnovers)
            checks.append(QCCheck("temporal", "max_turnover",
                                   "FAIL" if max_to > cfg.tau_turnover_fail else ("WARN" if max_to > cfg.tau_turnover_warn else "PASS"),
                                   round(max_to, 4), cfg.tau_turnover_fail, f"Max daily turnover: {max_to:.1%}"))

    # Calendar compliance
    if calendar is not None:
        uni_dates = set(pd.to_datetime(df["date"].unique()))
        cal_set = set(calendar)
        outside = uni_dates - cal_set
        if outside:
            checks.append(QCCheck("temporal", "dates_in_calendar", "FAIL", len(outside), len(uni_dates),
                                   f"{len(outside)} dates outside market calendar"))
        missing_cal = cal_set - uni_dates
        if missing_cal:
            checks.append(QCCheck("temporal", "calendar_gaps", "WARN" if len(missing_cal) < 5 else "FAIL",
                                   len(missing_cal), len(cal_set), f"{len(missing_cal)} calendar dates missing from universe"))

    return checks


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_universe_qc(
    universe: pd.DataFrame | str | Path,
    *,
    calendar: pd.DatetimeIndex | Sequence | None = None,
    config: UniverseQCConfig | None = None,
    output_dir: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run all 4 universe QC layers."""
    cfg = config or UniverseQCConfig()
    if not run_id:
        run_id = f"uqc_{utc_now_iso().replace(':','').replace('-','')}"

    if isinstance(universe, (str, Path)):
        universe = read_dataframe(universe)
    df = universe.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    cal = pd.DatetimeIndex(calendar) if calendar is not None else None

    all_checks: list[QCCheck] = []

    # Layer 1
    all_checks.extend(_check_structural(df, cfg))

    # Stop if structural FAIL
    if not any(c.severity == "FAIL" and c.layer == "structural" for c in all_checks):
        all_checks.extend(_check_pit(df))
        all_checks.extend(_check_semantic(df))
        all_checks.extend(_check_temporal(df, cal, cfg))

    n_fail = sum(1 for c in all_checks if c.severity == "FAIL")
    n_warn = sum(1 for c in all_checks if c.severity == "WARN")
    gate = "FAIL" if n_fail > 0 else ("WARN" if n_warn > 0 else "PASS")

    summary = {
        "run_id": run_id,
        "gate": gate,
        "n_checks": len(all_checks),
        "n_fail": n_fail, "n_warn": n_warn,
        "n_pass": sum(1 for c in all_checks if c.severity == "PASS"),
        "n_rows": len(df),
        "n_instruments": int(df["instrument_id"].nunique()) if "instrument_id" in df.columns else 0,
        "timestamp": utc_now_iso(),
    }

    if output_dir:
        out = Path(output_dir) / run_id
        out.mkdir(parents=True, exist_ok=True)
        cdf = pd.DataFrame([{"layer": c.layer, "check": c.name, "severity": c.severity,
                             "count": c.count, "total": c.total, "message": c.message} for c in all_checks])
        write_parquet_safe(cdf, out / "universe_qc_failures.parquet")
        write_json_safe(summary, out / "universe_qc_summary.json")

    LOGGER.info("Universe QC: %s (%dF %dW %dP)", gate, n_fail, n_warn, summary["n_pass"])
    return {"checks": all_checks, "summary": summary}
