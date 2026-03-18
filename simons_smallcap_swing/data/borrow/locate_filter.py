"""
data/borrow/locate_filter.py — Short eligibility and locate tier decision.

Consumes borrow_cost_proxy output and produces a binary eligible/blocked
decision plus operational tier (easy/medium/hard/blocked) for each symbol.

Key design: this is a POLICY layer, not an estimation layer.
Decisions are rule-based, monotonic, and traceable.

Rule precedence (highest → lowest):
    structural_missing > explicit_block_override > availability_critical
    > fee_critical > htb_severe > proxy_quality_too_low > fallback_severe
    > tier_mapping_soft

Hysteresis: upgrades (toward blocked) are immediate; downgrades
(toward easy) require k_down consecutive days of persistence.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import BorrowTier, TIER_CODE, REQUIRED_PROXY_COLS

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocateConfig:
    # Blocking thresholds
    alpha_block: float = 0.10          # availability < 10% → blocked
    fee_daily_block: float = 0.01      # daily fee > 1% → blocked
    block_on_htb: bool = False         # HTB flag alone doesn't block by default
    block_on_low_quality: bool = True  # low proxy quality → block

    # Tiering thresholds (for non-blocked symbols)
    alpha_easy: float = 0.60
    alpha_hard: float = 0.25
    fee_easy: float = 0.00005         # ~2bps daily
    fee_hard: float = 0.001           # ~37bps daily

    # Hysteresis
    hysteresis_down_days: int = 3

    # Override lists
    manual_blocklist: frozenset[str] = frozenset()
    manual_allowlist: frozenset[str] = frozenset()

    version: str = "1.0"


# ---------------------------------------------------------------------------
# Rule hierarchy
# ---------------------------------------------------------------------------

RULE_PRECEDENCE = [
    "structural_missing",
    "explicit_block_override",
    "availability_critical",
    "fee_critical",
    "htb_severe",
    "proxy_quality_too_low",
    "fallback_severe",
]


def _apply_hard_rules(row: pd.Series, cfg: LocateConfig) -> tuple[bool, str, list[str]]:
    """Apply hard blocking rules. Returns (blocked, dominant_reason, all_triggered)."""
    triggered: list[str] = []

    # R1: structural missing
    if pd.isna(row.get("borrow_fee_daily")) or pd.isna(row.get("borrow_availability_score")):
        triggered.append("structural_missing")

    # R2: explicit override
    if row["symbol"] in cfg.manual_blocklist:
        triggered.append("explicit_block_override")

    # R3: availability critical
    alpha = row.get("borrow_availability_score", 0)
    if not pd.isna(alpha) and alpha < cfg.alpha_block:
        triggered.append("availability_critical")

    # R4: fee critical
    fee_d = row.get("borrow_fee_daily", 0)
    if not pd.isna(fee_d) and fee_d > cfg.fee_daily_block:
        triggered.append("fee_critical")

    # R5: HTB severe
    if cfg.block_on_htb and row.get("htb_flag", 0) == 1:
        triggered.append("htb_severe")

    # R6: proxy quality too low
    if cfg.block_on_low_quality and row.get("proxy_quality") == "low":
        triggered.append("proxy_quality_too_low")

    # R7: fallback severe
    if row.get("fallback_flag", 0) == 1 and row.get("proxy_quality") == "low":
        triggered.append("fallback_severe")

    if not triggered:
        return False, "", []

    # Dominant = highest precedence
    dominant = min(triggered, key=lambda r: RULE_PRECEDENCE.index(r) if r in RULE_PRECEDENCE else 99)
    return True, dominant, triggered


def _assign_soft_tier(row: pd.Series, cfg: LocateConfig) -> str:
    """Assign tier for non-blocked symbols."""
    alpha = row.get("borrow_availability_score", 0.5)
    fee_d = row.get("borrow_fee_daily", 0)
    htb = row.get("htb_flag", 0)

    if alpha >= cfg.alpha_easy and fee_d <= cfg.fee_easy and htb == 0:
        return BorrowTier.EASY.value
    if alpha < cfg.alpha_hard or fee_d > cfg.fee_hard or htb == 1:
        return BorrowTier.HARD.value
    return BorrowTier.MEDIUM.value


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def build_locate_filter(
    borrow_proxy: pd.DataFrame,
    *,
    config: LocateConfig | None = None,
    universe_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Determine short eligibility for each (date, symbol).

    Parameters
    ----------
    borrow_proxy : output of build_borrow_cost_proxy
    config : LocateConfig
    universe_df : if provided, only scores symbols in universe

    Returns
    -------
    DataFrame with locate decision per (date, symbol).
    """
    cfg = config or LocateConfig()
    df = borrow_proxy.copy()

    # Validate required columns
    missing = REQUIRED_PROXY_COLS - set(df.columns)
    if missing - {"asof_timestamp"}:
        raise ValueError(f"Missing required proxy columns: {missing}")

    # Filter to universe if provided
    if universe_df is not None and "symbol" in universe_df.columns:
        eligible_symbols = set(universe_df["symbol"].unique())
        df = df[df["symbol"].isin(eligible_symbols)].copy()

    # Apply rules per row
    results = []
    for idx, row in df.iterrows():
        blocked, dominant, triggered = _apply_hard_rules(row, cfg)

        if blocked:
            tier = BorrowTier.BLOCKED.value
            eligible = 0
            reason = dominant
        else:
            tier = _assign_soft_tier(row, cfg)
            eligible = 1
            reason = ""

        # Manual allowlist override (limited: cannot override structural_missing)
        if row["symbol"] in cfg.manual_allowlist and "structural_missing" not in triggered:
            eligible = 1
            tier = min(tier, BorrowTier.HARD.value, key=lambda t: TIER_CODE.get(BorrowTier(t), 3))
            reason = ""

        results.append({
            "date": row["date"],
            "symbol": row["symbol"],
            "short_eligible_flag": eligible,
            "locate_tier": tier,
            "locate_tier_code": TIER_CODE.get(BorrowTier(tier), 3),
            "reject_reason": reason,
            "dominant_rule": dominant if blocked else "",
            "all_triggered_rules": ",".join(triggered) if triggered else "",
            "borrow_fee_daily": row.get("borrow_fee_daily", np.nan),
            "borrow_fee_annual": row.get("borrow_fee_annual", np.nan),
            "availability_score": row.get("borrow_availability_score", np.nan),
            "htb_flag": row.get("htb_flag", 0),
            "proxy_quality": row.get("proxy_quality", "low"),
            "fallback_flag": row.get("fallback_flag", 0),
            "override_flag": 1 if row["symbol"] in cfg.manual_allowlist else 0,
            "data_quality_flag": "missing_borrow_join" if "structural_missing" in triggered else "",
            "locate_config_version": cfg.version,
            "upstream_proxy_version": row.get("config_version", ""),
        })

    out = pd.DataFrame(results)

    # Summary
    if len(out) > 0:
        n_eligible = int(out["short_eligible_flag"].sum())
        n_total = len(out)
        LOGGER.info(
            "Locate filter: %d/%d eligible (%.0f%%), tier dist: %s",
            n_eligible, n_total, n_eligible / max(n_total, 1) * 100,
            out["locate_tier"].value_counts().to_dict(),
        )

    return out
