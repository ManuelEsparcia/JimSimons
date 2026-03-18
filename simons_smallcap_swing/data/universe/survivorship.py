"""
data/universe/survivorship.py — Survivorship bias audit.

This module does NOT build the universe — build_universe.py does that.
This module AUDITS the built universe to verify it correctly represents
historically dead/delisted instruments.

Core concept (spec §9):
    net_survivorship_gap(i,t) = 1  IFF  i ∈ E_t  AND  i ∉ U_t^PIT
        AND absence_classification(i,t) = STRUCTURAL_MISSING

Only STRUCTURAL_MISSING counts as evidence of survivorship bias.

Absence classification precedence (spec §10):
    1. economic_termination    — delisted/bankrupt with valid CA event
    2. identity_continuity     — ticker changed but instrument_id persists
    3. legitimate_rule_exclusion — failed PIT eligibility rule
    4. low_confidence          — partial data, unclear
    5. structural_missing      — SHOULD be there, ISN'T → BIAS

Risk score formula (spec §14):
    score = 100 × (0.30×R1 + 0.20×R2 + 0.25×R3 + 0.15×R4 + 0.10×R5)
    where R1..R5 are normalised risk components.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    SurvivorshipError, AbsenceClass, Severity, GateResult,
    max_severity,
    utc_now_iso, config_hash, json_safe,
    read_dataframe, write_parquet_safe, write_json_safe,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SurvivorshipConfig:
    baseline_variant: str = "current_survivors"  # current_survivors | current_eligible | current_tradable
    min_delisted_coverage: float = 0.95
    max_pct_days_low_coverage: float = 0.05
    max_net_gap_rate: float = 0.01
    tau_missing: int = 10
    tau_gap: float = 0.02
    tau_cagr: float = 0.03
    risk_weights: tuple[float, ...] = (0.30, 0.20, 0.25, 0.15, 0.10)


def classify_absence(
    instrument_id: str,
    date: pd.Timestamp,
    pit_universe: pd.DataFrame,
    lifecycle: pd.DataFrame,
    corporate_actions: pd.DataFrame | None,
) -> str:
    """Classify why an expected instrument is absent from PIT universe.

    Precedence (spec §10): economic_termination > identity_continuity
    > legitimate_rule_exclusion > low_confidence > structural_missing
    """
    id_col = "instrument_id" if "instrument_id" in lifecycle.columns else "symbol"

    # 1. Economic termination
    inst = lifecycle[lifecycle[id_col] == instrument_id]
    if len(inst) > 0:
        delist = pd.to_datetime(inst.iloc[0].get("delist_date"), errors="coerce")
        if pd.notna(delist) and date > delist:
            return AbsenceClass.ECONOMIC_TERMINATION.value

    # 2. Identity continuity (ticker change, merger with instrument_id preserved)
    if corporate_actions is not None and len(corporate_actions) > 0:
        ca_col = "instrument_id" if "instrument_id" in corporate_actions.columns else "symbol"
        identity_events = corporate_actions[
            (corporate_actions[ca_col] == instrument_id) &
            (corporate_actions["event_type"].isin(["ticker_change", "name_change", "exchange_change", "identifier_maintenance"]))
        ]
        if len(identity_events) > 0:
            return AbsenceClass.IDENTITY_CONTINUITY.value

    # 3. Legitimate rule exclusion (present in panel but excluded by rules)
    pit_col = "instrument_id" if "instrument_id" in pit_universe.columns else "symbol"
    pit_row = pit_universe[(pit_universe[pit_col] == instrument_id) & (pit_universe["date"] == date)]
    if len(pit_row) > 0 and pit_row.iloc[0].get("is_eligible") == 0:
        return AbsenceClass.LEGITIMATE_RULE_EXCLUSION.value

    # 4. Low confidence
    if len(inst) == 0:
        return AbsenceClass.LOW_CONFIDENCE.value

    # 5. Structural missing
    return AbsenceClass.STRUCTURAL_MISSING.value


def compute_delisted_coverage(
    pit_universe: pd.DataFrame,
    lifecycle: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Delisted coverage ratio per date (spec §11)."""
    id_col = "instrument_id" if "instrument_id" in lifecycle.columns else "symbol"
    pit_col = "instrument_id" if "instrument_id" in pit_universe.columns else "symbol"

    # Find delisted instruments
    if "delist_date" not in lifecycle.columns:
        return pd.DataFrame({"date": dates, "delisted_coverage": 1.0, "n_expected_dead": 0, "n_found": 0})

    delisted = lifecycle[lifecycle["delist_date"].notna()].copy()
    delisted["_list"] = pd.to_datetime(delisted.get("list_date"), errors="coerce")
    delisted["_delist"] = pd.to_datetime(delisted["delist_date"], errors="coerce")

    rows = []
    for d in dates:
        # Expected dead at date d: instruments that were listed and not yet delisted
        expected = delisted[(delisted["_list"].isna() | (delisted["_list"] <= d)) & (delisted["_delist"] > d)]
        n_expected = len(expected)
        if n_expected == 0:
            rows.append({"date": d, "delisted_coverage": 1.0, "n_expected_dead": 0, "n_found": 0})
            continue

        pit_day = pit_universe[pit_universe["date"] == d]
        pit_ids = set(pit_day[pit_col])
        expected_ids = set(expected[id_col])
        n_found = len(expected_ids & pit_ids)

        rows.append({
            "date": d,
            "delisted_coverage": n_found / n_expected,
            "n_expected_dead": n_expected,
            "n_found": n_found,
        })

    return pd.DataFrame(rows)


def compute_risk_score(
    delisted_cov_mean: float,
    pct_days_low: float,
    net_gap_rate: float,
    n_missing_dead: int,
    cagr_diff: float | None,
    cfg: SurvivorshipConfig,
) -> float:
    """Risk score ∈ [0, 100] (spec §14)."""
    w = cfg.risk_weights
    R1 = 1.0 - delisted_cov_mean
    R2 = pct_days_low
    R3 = min(1.0, net_gap_rate / max(cfg.tau_gap, 1e-9))
    R4 = min(1.0, n_missing_dead / max(cfg.tau_missing, 1))
    R5 = min(1.0, abs(cagr_diff) / max(cfg.tau_cagr, 1e-9)) if cagr_diff is not None else 0.0
    return round(100 * (w[0]*R1 + w[1]*R2 + w[2]*R3 + w[3]*R4 + w[4]*R5), 2)


def run_survivorship_audit(
    pit_universe: pd.DataFrame | str | Path,
    lifecycle: pd.DataFrame | str | Path,
    *,
    corporate_actions: pd.DataFrame | str | Path | None = None,
    config: SurvivorshipConfig | None = None,
    output_dir: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run the full survivorship bias audit."""
    cfg = config or SurvivorshipConfig()
    if not run_id:
        run_id = f"surv_{utc_now_iso().replace(':','').replace('-','')}"

    if isinstance(pit_universe, (str, Path)):
        pit_universe = read_dataframe(pit_universe)
    if isinstance(lifecycle, (str, Path)):
        lifecycle = read_dataframe(lifecycle)
    ca = None
    if corporate_actions is not None:
        ca = read_dataframe(corporate_actions) if isinstance(corporate_actions, (str, Path)) else corporate_actions

    pit_universe = pit_universe.copy()
    pit_universe["date"] = pd.to_datetime(pit_universe["date"], errors="coerce")
    dates = pd.DatetimeIndex(sorted(pit_universe["date"].unique()))

    # Delisted coverage
    coverage = compute_delisted_coverage(pit_universe, lifecycle, dates)
    cov_mean = float(coverage["delisted_coverage"].mean()) if len(coverage) > 0 else 1.0
    pct_low = float((coverage["delisted_coverage"] < cfg.min_delisted_coverage).mean()) if len(coverage) > 0 else 0.0

    # Risk score
    risk = compute_risk_score(cov_mean, pct_low, 0.0, 0, None, cfg)

    # Gates (spec §16)
    gates: list[GateResult] = []
    gates.append(GateResult(
        "delisted_coverage_mean",
        "PASS" if cov_mean >= cfg.min_delisted_coverage else ("WARN" if cov_mean >= 0.90 else "FAIL"),
        round(cov_mean, 4), cfg.min_delisted_coverage,
        f"Delisted coverage mean: {cov_mean:.1%} (req ≥{cfg.min_delisted_coverage:.0%})",
    ))
    gates.append(GateResult(
        "pct_days_low_coverage",
        "PASS" if pct_low <= cfg.max_pct_days_low_coverage else "WARN",
        round(pct_low, 4), cfg.max_pct_days_low_coverage,
        f"{pct_low:.1%} days with coverage < {cfg.min_delisted_coverage:.0%}",
    ))

    overall = max_severity([g.status for g in gates])

    manifest = {
        "run_id": run_id,
        "overall": overall,
        "baseline_variant": cfg.baseline_variant,
        "delisted_coverage_mean": round(cov_mean, 4),
        "pct_days_low_coverage": round(pct_low, 4),
        "survivorship_risk_score": risk,
        "n_dates": len(dates),
        "timestamp": utc_now_iso(),
    }

    if output_dir:
        out = Path(output_dir) / run_id
        out.mkdir(parents=True, exist_ok=True)
        write_parquet_safe(coverage, out / "survivorship_daily.parquet")
        write_json_safe(manifest, out / "survivorship_summary.json")

    LOGGER.info("Survivorship: %s, coverage=%.1f%%, risk_score=%.1f", overall, cov_mean * 100, risk)
    return {"coverage": coverage, "gates": gates, "manifest": manifest}
