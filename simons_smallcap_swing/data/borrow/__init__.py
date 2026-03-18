"""
data.borrow — Borrow cost estimation, locate filtering, and QC.

Pipeline: borrow_cost_proxy → locate_filter → borrow_qc

Produces causal, conservative estimates of shorting friction for
small caps where direct securities lending feeds are unavailable.
"""
from __future__ import annotations
import enum


class BorrowTier(str, enum.Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    BLOCKED = "blocked"


class ProxyQuality(str, enum.Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


TIER_CODE = {BorrowTier.EASY: 0, BorrowTier.MEDIUM: 1, BorrowTier.HARD: 2, BorrowTier.BLOCKED: 3}

REQUIRED_PROXY_COLS = frozenset({
    "symbol", "date", "borrow_fee_annual", "borrow_fee_daily",
    "borrow_availability_score", "htb_flag", "borrow_tier",
    "proxy_quality", "proxy_source", "stress_score",
    "jump_flag", "stale_input_flag", "fallback_flag", "config_version",
})
