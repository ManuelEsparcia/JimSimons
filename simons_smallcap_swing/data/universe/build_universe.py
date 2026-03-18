"""
data/universe/build_universe.py — PIT universe construction.

Defines, for each trading date, which instruments were genuinely eligible
for research, ranking, and trading using ONLY information observable at
that date under the convention: decision_time = close(t-1).

Primary identity: instrument_id (NOT symbol).

Filter precedence (strict, deterministic):
    F1. Identity & lifecycle  (list_date ≤ t ≤ tau_death)
    F2. Instrument type       (common stock only, no ETFs/ADRs/warrants)
    F3. Exchange / venue      (NYSE, NASDAQ, ... — no OTC)
    F4. Operational status    (not halted, not suspended)
    F5. Age since listing     (min_age_days)
    F6. Price minimum         (price_ref ≥ p_min)
    F7. Liquidity minimum     (adv20_usd ≥ v_min)
    F8. Market cap band       (mcap ∈ [min, max])
    F9. Special rules         (configurable)

primary_exclusion_reason = first failed filter in precedence order.
all_failed_reasons = ordered list of ALL failed filters.

Invariants (hard, non-negotiable):
    1. Unique per (date, instrument_id)
    2. is_eligible ∈ {0, 1}
    3. is_eligible=0 ⟺ primary_exclusion_reason is not None
    4. is_eligible=1 ⟹ primary_exclusion_reason is None
    5. No eligible row before list_date
    6. No eligible row after tau_death
    7. Deterministic: same inputs + config → same output
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import (
    BuildError, ExclusionReason, MembershipState, Transition, Severity,
    utc_now_iso, config_hash, json_safe,
    write_parquet_safe, write_json_safe, read_dataframe,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration (spec §5, §9, §11)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UniverseRules:
    p_min: float = 1.0              # F6: minimum price
    v_min: float = 50_000           # F7: minimum ADV20 in USD
    mcap_min: float = 5e7           # F8: small-cap floor ($50M)
    mcap_max: float = 2e9           # F8: small-cap ceiling ($2B)
    min_age_days: int = 63          # F5: ~3 months
    allowed_exchanges: frozenset[str] = frozenset({"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"})
    allowed_security_types: frozenset[str] = frozenset({"CS", "COMMON_STOCK"})
    allowed_trading_statuses: frozenset[str] = frozenset({"ACTIVE", "NORMAL", ""})


@dataclass(frozen=True)
class UniverseConfig:
    rules: UniverseRules = field(default_factory=UniverseRules)
    missing_critical_policy: str = "exclude"   # exclude | forward_fill
    ffill_max_days: int = 5                    # max forward-fill horizon
    halt_policy: str = "exclude_until_tradable"
    relisting_policy: str = "new_instrument_unless_proven_same"


# ---------------------------------------------------------------------------
# Filter functions — each returns an ExclusionReason or None
# Order matches spec §9 precedence exactly.
# ---------------------------------------------------------------------------

def _f1_lifecycle(row: pd.Series) -> ExclusionReason | None:
    """F1: identity & lifecycle — instrument must be alive at date."""
    d = row["date"]
    list_d = row.get("list_date")
    if pd.notna(list_d) and d < pd.Timestamp(list_d):
        return ExclusionReason.NOT_YET_LISTED
    death = row.get("delist_date") or row.get("tau_death")
    if pd.notna(death) and d > pd.Timestamp(death):
        return ExclusionReason.DELISTED
    return None


def _f2_security_type(row: pd.Series, rules: UniverseRules) -> ExclusionReason | None:
    """F2: instrument type."""
    st = str(row.get("security_type", "")).upper().strip()
    if rules.allowed_security_types and st not in rules.allowed_security_types:
        return ExclusionReason.BAD_SECURITY_TYPE
    return None


def _f3_exchange(row: pd.Series, rules: UniverseRules) -> ExclusionReason | None:
    """F3: exchange / listing venue."""
    ex = str(row.get("listing_exchange", row.get("exchange", ""))).upper().strip()
    if rules.allowed_exchanges and ex not in rules.allowed_exchanges:
        return ExclusionReason.BAD_EXCHANGE
    return None


def _f4_status(row: pd.Series) -> ExclusionReason | None:
    """F4: operational status."""
    status = str(row.get("trading_status", "")).upper().strip()
    if status == "HALTED":
        return ExclusionReason.HALTED
    if status == "SUSPENDED":
        return ExclusionReason.SUSPENDED
    return None


def _f5_age(row: pd.Series, rules: UniverseRules) -> ExclusionReason | None:
    """F5: age since listing."""
    list_d = row.get("list_date")
    if pd.isna(list_d):
        return None  # missing list_date handled elsewhere
    age = (row["date"] - pd.Timestamp(list_d)).days
    if age < rules.min_age_days:
        return ExclusionReason.TOO_YOUNG_SINCE_LISTING
    return None


def _f6_price(row: pd.Series, rules: UniverseRules, policy: str) -> ExclusionReason | None:
    """F6: price minimum."""
    p = row.get("price_ref", row.get("close"))
    if pd.isna(p):
        return ExclusionReason.MISSING_PRICE if policy == "exclude" else None
    if float(p) < rules.p_min:
        return ExclusionReason.PRICE_BELOW_MIN
    return None


def _f7_liquidity(row: pd.Series, rules: UniverseRules, policy: str) -> ExclusionReason | None:
    """F7: liquidity minimum (ADV20 USD)."""
    adv = row.get("adv20_usd", row.get("adv"))
    if pd.isna(adv):
        return ExclusionReason.MISSING_ADV20 if policy == "exclude" else None
    if float(adv) < rules.v_min:
        return ExclusionReason.ADV20_BELOW_MIN
    return None


def _f8_mcap(row: pd.Series, rules: UniverseRules, policy: str) -> ExclusionReason | None:
    """F8: market cap band."""
    mcap = row.get("market_cap_usd", row.get("mcap"))
    if pd.isna(mcap):
        return ExclusionReason.MISSING_MCAP if policy == "exclude" else None
    m = float(mcap)
    if m < rules.mcap_min or m > rules.mcap_max:
        return ExclusionReason.MCAP_OUT_OF_BAND
    return None


# ---------------------------------------------------------------------------
# Apply all filters in precedence order
# ---------------------------------------------------------------------------

_FILTER_SEQUENCE = [
    ("lifecycle", _f1_lifecycle),
    ("security_type", _f2_security_type),
    ("exchange", _f3_exchange),
    ("status", _f4_status),
    ("age", _f5_age),
    ("price", _f6_price),
    ("liquidity", _f7_liquidity),
    ("mcap", _f8_mcap),
]


def evaluate_row(row: pd.Series, cfg: UniverseConfig) -> dict[str, Any]:
    """Evaluate a single (date, instrument_id) row through all filters."""
    rules = cfg.rules
    policy = cfg.missing_critical_policy
    all_reasons: list[str] = []

    for name, fn in _FILTER_SEQUENCE:
        if name == "lifecycle":
            r = fn(row)
        elif name in ("security_type", "exchange"):
            r = fn(row, rules)
        elif name == "status":
            r = fn(row)
        elif name == "age":
            r = fn(row, rules)
        elif name in ("price", "liquidity", "mcap"):
            r = fn(row, rules, policy)
        else:
            r = None
        if r is not None:
            all_reasons.append(r.value)

    is_eligible = 1 if len(all_reasons) == 0 else 0
    primary = all_reasons[0] if all_reasons else None

    # Membership state
    if is_eligible:
        state = MembershipState.ELIGIBLE.value
    elif primary == ExclusionReason.NOT_YET_LISTED.value:
        state = MembershipState.ELIGIBLE.value  # won't happen since is_eligible=0
        state = "pre_listing"
    elif primary == ExclusionReason.DELISTED.value:
        state = MembershipState.INACTIVE_POST_DEATH.value
    elif primary == ExclusionReason.HALTED.value:
        state = MembershipState.INELIGIBLE_HALTED.value
    elif primary == ExclusionReason.SUSPENDED.value:
        state = MembershipState.INELIGIBLE_SUSPENDED.value
    elif primary in (ExclusionReason.MISSING_PRICE.value, ExclusionReason.MISSING_ADV20.value, ExclusionReason.MISSING_MCAP.value):
        state = MembershipState.INELIGIBLE_MISSING.value
    else:
        state = MembershipState.INELIGIBLE_RULE.value

    return {
        "is_eligible": is_eligible,
        "membership_state": state,
        "primary_exclusion_reason": primary,
        "all_failed_reasons": "|".join(all_reasons) if all_reasons else None,
    }


def evaluate_panel(panel: pd.DataFrame, cfg: UniverseConfig) -> pd.DataFrame:
    """Evaluate eligibility for the entire panel."""
    results = panel.apply(lambda row: evaluate_row(row, cfg), axis=1, result_type="expand")
    return pd.concat([panel, results], axis=1)


# ---------------------------------------------------------------------------
# Turnover at instrument_id level (spec §15)
# ---------------------------------------------------------------------------

def compute_turnover(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute daily turnover using symmetric difference on instrument_id."""
    dates = sorted(panel["date"].unique())
    rows = []
    prev_set: set[str] = set()
    for d in dates:
        day = panel[(panel["date"] == d) & (panel["is_eligible"] == 1)]
        curr_set = set(day["instrument_id"])
        entries = curr_set - prev_set
        exits = prev_set - curr_set
        union = curr_set | prev_set
        turnover = (len(entries) + len(exits)) / len(union) if union else 0.0
        rows.append({
            "date": d,
            "n_eligible": len(curr_set),
            "n_entries": len(entries),
            "n_exits": len(exits),
            "turnover": round(turnover, 6),
        })
        prev_set = curr_set
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def build_universe(
    listing_master: pd.DataFrame | str | Path,
    *,
    features_asof: pd.DataFrame | str | Path | None = None,
    calendar: pd.DatetimeIndex | Sequence | None = None,
    config: UniverseConfig | None = None,
    output_dir: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build the PIT universe history.

    Steps (spec §17):
    1. Load inputs
    2. Validate schema
    3. Normalize identity (instrument_id primary)
    4. Build evaluable panel P
    5. Enrich with PIT features
    6. Apply filters in deterministic order
    7. Resolve eligibility, state, reasons
    8. Compute turnover and transitions
    9. Persist
    """
    cfg = config or UniverseConfig()
    if not run_id:
        run_id = f"univ_{utc_now_iso().replace(':','').replace('-','')}"

    # 1. Load
    if isinstance(listing_master, (str, Path)):
        listing_master = read_dataframe(listing_master)
    lm = listing_master.copy()
    lm["date"] = pd.to_datetime(lm["date"], errors="coerce")

    if isinstance(features_asof, (str, Path)):
        features_asof = read_dataframe(features_asof)

    # 3. Normalize identity
    if "instrument_id" not in lm.columns:
        lm["instrument_id"] = lm.get("symbol", pd.Series("", index=lm.index))

    # 4. Panel P is already daily if listing_master has date
    panel = lm.copy()

    # 5. Enrich with features
    if features_asof is not None:
        merge_on = ["date", "instrument_id"] if "instrument_id" in features_asof.columns else ["date", "symbol"]
        avail = [c for c in merge_on if c in features_asof.columns]
        feat_cols = [c for c in features_asof.columns if c not in avail]
        existing = [c for c in feat_cols if c in panel.columns]
        panel = panel.drop(columns=existing, errors="ignore")
        panel = panel.merge(features_asof[avail + feat_cols], on=avail, how="left")

    # 6-7. Apply filters
    panel = evaluate_panel(panel, cfg)

    # 8. Turnover
    stats = compute_turnover(panel)

    # Daily exclusion breakdown
    daily_by_reason = (
        panel[panel["is_eligible"] == 0]
        .groupby(["date", "primary_exclusion_reason"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    ) if (panel["is_eligible"] == 0).any() else pd.DataFrame()

    # Current snapshot
    latest = panel["date"].max()
    current = panel[panel["date"] == latest].copy()

    # Manifest
    manifest = {
        "run_id": run_id,
        "config_hash": config_hash(json_safe({"rules": cfg.rules.__dict__, "policy": cfg.missing_critical_policy})),
        "decision_convention": "close(t-1) -> execute(t)",
        "n_dates": int(panel["date"].nunique()),
        "n_instruments": int(panel["instrument_id"].nunique()),
        "n_rows": len(panel),
        "n_eligible_total": int(panel["is_eligible"].sum()),
        "pct_eligible": round(float(panel["is_eligible"].mean()) * 100, 2),
        "avg_daily_eligible": round(float(stats["n_eligible"].mean()), 1) if len(stats) > 0 else 0,
        "exclusion_counts": panel[panel["is_eligible"] == 0]["primary_exclusion_reason"].value_counts().to_dict() if (panel["is_eligible"] == 0).any() else {},
        "timestamp": utc_now_iso(),
    }

    if output_dir:
        out = Path(output_dir) / run_id
        out.mkdir(parents=True, exist_ok=True)
        write_parquet_safe(panel, out / "universe_history.parquet")
        write_parquet_safe(current, out / "universe_current.parquet")
        write_parquet_safe(stats, out / "universe_turnover_stats.parquet")
        write_json_safe(manifest, out / "manifest.json")

    LOGGER.info("Universe: %d dates, %d instruments, %.1f%% eligible, avg_daily=%d",
                manifest["n_dates"], manifest["n_instruments"],
                manifest["pct_eligible"], manifest["avg_daily_eligible"])

    return {
        "history": panel, "current": current,
        "stats": stats, "manifest": manifest,
    }
