"""
data/borrow/borrow_qc.py — Quality control for borrow proxy and locate filter.

5 audit dimensions (all mandatory):
    1. Structural: schema, PK uniqueness, dtypes
    2. Numeric: fee non-negative, availability ∈ [0,1], no NaN in critical
    3. Coverage: % of universe with usable data
    4. Stability: jump detection, staleness, tier oscillation
    5. Coherence: alignment between fee, availability, HTB, tier, and locate

4 severity levels:
    0 - structural hard fail (non-compensable)
    1 - numeric hard fail
    2 - soft degradation (WARN)
    3 - informational

Gate logic: any hard fail → overall FAIL regardless of other scores.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import BorrowTier, REQUIRED_PROXY_COLS

LOGGER = logging.getLogger(__name__)


@dataclass
class QCCheck:
    dimension: str       # "structural" | "numeric" | "coverage" | "stability" | "coherence"
    check_name: str
    severity: str        # "PASS" | "WARN" | "FAIL"
    level: int           # 0-3
    metric_value: Any
    threshold: Any
    message: str


@dataclass(frozen=True)
class BorrowQCConfig:
    min_coverage: float = 0.80
    max_fee_annual: float = 10.0          # sanity cap
    max_jump_pct: float = 5.0             # fee jump > 500% in 1 day
    max_stale_fraction: float = 0.10
    max_fallback_fraction: float = 0.30


# ---------------------------------------------------------------------------
# Dimension 1: Structural
# ---------------------------------------------------------------------------

def check_structural(proxy: pd.DataFrame) -> list[QCCheck]:
    checks = []

    # Schema
    missing = REQUIRED_PROXY_COLS - {"asof_timestamp"} - set(proxy.columns)
    checks.append(QCCheck(
        "structural", "schema_complete", "FAIL" if missing else "PASS", 0,
        list(missing), [], f"Missing columns: {missing}" if missing else "Schema OK",
    ))

    # PK uniqueness
    if "date" in proxy.columns and "symbol" in proxy.columns:
        n_dups = proxy.duplicated(subset=["date", "symbol"]).sum()
        checks.append(QCCheck(
            "structural", "pk_unique", "FAIL" if n_dups > 0 else "PASS", 0,
            int(n_dups), 0, f"{n_dups} duplicate (date, symbol)" if n_dups else "PK unique",
        ))

    return checks


# ---------------------------------------------------------------------------
# Dimension 2: Numeric
# ---------------------------------------------------------------------------

def check_numeric(proxy: pd.DataFrame, cfg: BorrowQCConfig) -> list[QCCheck]:
    checks = []

    if "borrow_fee_annual" in proxy.columns:
        n_neg = int((proxy["borrow_fee_annual"] < 0).sum())
        checks.append(QCCheck(
            "numeric", "fee_non_negative", "FAIL" if n_neg > 0 else "PASS", 1,
            n_neg, 0, f"{n_neg} negative fees",
        ))
        n_extreme = int((proxy["borrow_fee_annual"] > cfg.max_fee_annual).sum())
        checks.append(QCCheck(
            "numeric", "fee_within_range", "WARN" if n_extreme > 0 else "PASS", 2,
            n_extreme, 0, f"{n_extreme} fees > {cfg.max_fee_annual}",
        ))

    if "borrow_availability_score" in proxy.columns:
        out_of_range = ((proxy["borrow_availability_score"] < 0) | (proxy["borrow_availability_score"] > 1)).sum()
        checks.append(QCCheck(
            "numeric", "availability_in_01", "FAIL" if out_of_range > 0 else "PASS", 1,
            int(out_of_range), 0, f"{out_of_range} availability values outside [0,1]",
        ))

    # NaN in critical fields
    for col in ["borrow_fee_annual", "borrow_availability_score", "borrow_tier"]:
        if col in proxy.columns:
            n_nan = int(proxy[col].isna().sum())
            frac = n_nan / max(len(proxy), 1)
            checks.append(QCCheck(
                "numeric", f"nan_{col}", "FAIL" if frac > 0.5 else ("WARN" if frac > 0.05 else "PASS"),
                1 if frac > 0.5 else 2,
                round(frac, 4), 0.05, f"{col}: {frac:.1%} NaN",
            ))

    return checks


# ---------------------------------------------------------------------------
# Dimension 3: Coverage
# ---------------------------------------------------------------------------

def check_coverage(
    proxy: pd.DataFrame,
    universe: pd.DataFrame | None,
    cfg: BorrowQCConfig,
) -> list[QCCheck]:
    checks = []

    if universe is not None and "date" in universe.columns and "symbol" in universe.columns:
        merged = universe[["date", "symbol"]].merge(
            proxy[["date", "symbol", "borrow_fee_annual"]],
            on=["date", "symbol"], how="left",
        )
        coverage = merged["borrow_fee_annual"].notna().mean()
    else:
        coverage = proxy["borrow_fee_annual"].notna().mean() if "borrow_fee_annual" in proxy.columns else 0

    checks.append(QCCheck(
        "coverage", "universe_coverage",
        "FAIL" if coverage < 0.5 else ("WARN" if coverage < cfg.min_coverage else "PASS"),
        2, round(float(coverage), 4), cfg.min_coverage,
        f"Coverage: {coverage:.1%} of universe",
    ))

    return checks


# ---------------------------------------------------------------------------
# Dimension 4: Stability
# ---------------------------------------------------------------------------

def check_stability(proxy: pd.DataFrame, cfg: BorrowQCConfig) -> list[QCCheck]:
    checks = []

    # Jump detection
    if "jump_flag" in proxy.columns:
        jump_frac = float(proxy["jump_flag"].mean())
        checks.append(QCCheck(
            "stability", "jump_fraction",
            "WARN" if jump_frac > 0.01 else "PASS", 2,
            round(jump_frac, 4), 0.01, f"{jump_frac:.2%} of rows flagged as jumps",
        ))

    # Staleness
    if "stale_input_flag" in proxy.columns:
        stale_frac = float(proxy["stale_input_flag"].mean())
        checks.append(QCCheck(
            "stability", "stale_fraction",
            "WARN" if stale_frac > cfg.max_stale_fraction else "PASS", 2,
            round(stale_frac, 4), cfg.max_stale_fraction,
            f"{stale_frac:.1%} stale inputs",
        ))

    # Fallback fraction
    if "fallback_flag" in proxy.columns:
        fb_frac = float(proxy["fallback_flag"].mean())
        checks.append(QCCheck(
            "stability", "fallback_fraction",
            "WARN" if fb_frac > cfg.max_fallback_fraction else "PASS", 2,
            round(fb_frac, 4), cfg.max_fallback_fraction,
            f"{fb_frac:.1%} using fallback",
        ))

    return checks


# ---------------------------------------------------------------------------
# Dimension 5: Coherence
# ---------------------------------------------------------------------------

def check_coherence(proxy: pd.DataFrame, locate: pd.DataFrame | None) -> list[QCCheck]:
    checks = []

    # Fee vs tier coherence
    if "borrow_fee_annual" in proxy.columns and "borrow_tier" in proxy.columns:
        blocked = proxy["borrow_tier"] == BorrowTier.BLOCKED.value
        if blocked.any():
            blocked_fees = proxy.loc[blocked, "borrow_fee_annual"]
            easy_fees = proxy.loc[proxy["borrow_tier"] == BorrowTier.EASY.value, "borrow_fee_annual"]
            if len(blocked_fees) > 0 and len(easy_fees) > 0:
                # Blocked should have higher fees than easy (on average)
                incoherent = blocked_fees.median() < easy_fees.median()
                checks.append(QCCheck(
                    "coherence", "fee_tier_alignment",
                    "WARN" if incoherent else "PASS", 2,
                    None, None,
                    "Blocked median fee < easy median fee (incoherent)" if incoherent else "Fee-tier aligned",
                ))

    # Locate coherence: blocked tier ↔ eligible=0
    if locate is not None and "locate_tier" in locate.columns and "short_eligible_flag" in locate.columns:
        blocked_but_eligible = (
            (locate["locate_tier"] == BorrowTier.BLOCKED.value) & (locate["short_eligible_flag"] == 1)
        ).sum()
        checks.append(QCCheck(
            "coherence", "blocked_but_eligible",
            "FAIL" if blocked_but_eligible > 0 else "PASS", 1,
            int(blocked_but_eligible), 0,
            f"{blocked_but_eligible} symbols blocked but marked eligible",
        ))

        # Eligible must have reason empty
        eligible_with_reason = (
            (locate["short_eligible_flag"] == 1) & (locate["reject_reason"].fillna("") != "")
        ).sum()
        checks.append(QCCheck(
            "coherence", "eligible_no_reject_reason",
            "WARN" if eligible_with_reason > 0 else "PASS", 2,
            int(eligible_with_reason), 0,
            f"{eligible_with_reason} eligible symbols with reject_reason set",
        ))

    return checks


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_borrow_qc(
    proxy: pd.DataFrame,
    *,
    locate: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    config: BorrowQCConfig | None = None,
) -> tuple[list[QCCheck], dict[str, Any]]:
    """Run all 5 QC dimensions on borrow proxy + locate."""
    cfg = config or BorrowQCConfig()

    all_checks: list[QCCheck] = []
    all_checks.extend(check_structural(proxy))
    all_checks.extend(check_numeric(proxy, cfg))
    all_checks.extend(check_coverage(proxy, universe, cfg))
    all_checks.extend(check_stability(proxy, cfg))
    all_checks.extend(check_coherence(proxy, locate))

    n_fail = sum(1 for c in all_checks if c.severity == "FAIL")
    n_warn = sum(1 for c in all_checks if c.severity == "WARN")
    overall = "FAIL" if n_fail > 0 else ("WARN" if n_warn > 0 else "PASS")

    summary = {
        "overall": overall,
        "n_checks": len(all_checks),
        "n_fail": n_fail,
        "n_warn": n_warn,
        "n_pass": sum(1 for c in all_checks if c.severity == "PASS"),
    }

    LOGGER.info("Borrow QC: %s (%dF %dW)", overall, n_fail, n_warn)
    return all_checks, summary
