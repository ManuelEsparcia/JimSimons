"""
execution/borrow_execution.py — Short admission and borrow cost.

    desired → admitted → filled → live
    Borrow cost on live short only.
    Recall → forced cover.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class BorrowConfig:
    fail_closed_missing: bool = True
    allow_reduction_when_not_eligible: bool = True
    fee_daycount: int = 360
    default_fee_annual: float = 0.01     # 1% fallback


def compute_admission(
    current_shares: np.ndarray,
    target_shares: np.ndarray,
    eligible: np.ndarray,
    availability: np.ndarray,
    capacity_max: np.ndarray,
    *,
    config: BorrowConfig | None = None,
) -> dict[str, np.ndarray]:
    """Determine admitted short position per asset."""
    cfg = config or BorrowConfig()
    q_cur = np.asarray(current_shares, dtype=float).ravel()
    q_des = np.asarray(target_shares, dtype=float).ravel()
    elig = np.asarray(eligible, dtype=bool).ravel()
    avail = np.asarray(availability, dtype=float).ravel()
    cap = np.asarray(capacity_max, dtype=float).ravel()
    n = q_cur.size

    q_adm = q_des.copy()
    reasons = np.full(n, "", dtype=object)

    for i in range(n):
        short_des = max(0.0, -q_des[i])
        short_cur = max(0.0, -q_cur[i])
        delta_open = max(0.0, short_des - short_cur)

        if delta_open <= 0:
            # Reducing or flat → always allowed
            continue

        if not elig[i]:
            if cfg.allow_reduction_when_not_eligible and q_des[i] > q_cur[i]:
                # Reducing short is ok, but can't open more
                pass
            # Can't open new short
            q_adm[i] = -short_cur  # keep current short, don't increase
            reasons[i] = "not_eligible"
            continue

        # Capacity free
        b_free = max(0.0, avail[i] * cap[i] - short_cur)
        if b_free <= 0:
            q_adm[i] = -short_cur
            reasons[i] = "no_capacity"
            continue

        admitted_open = min(delta_open, b_free)
        q_adm[i] = -(short_cur + admitted_open)
        if admitted_open < delta_open:
            reasons[i] = "truncated_by_capacity"

    return {
        "admitted_shares": q_adm,
        "reject_reasons": reasons,
        "n_rejected": int((reasons != "").sum()),
        "n_truncated": int((reasons == "truncated_by_capacity").sum()),
    }


def compute_borrow_cost(
    live_short_shares: np.ndarray,
    reference_prices: np.ndarray,
    borrow_fees_daily: np.ndarray | None = None,
    *,
    config: BorrowConfig | None = None,
) -> np.ndarray:
    """Daily borrow cost per asset on live short positions."""
    cfg = config or BorrowConfig()
    shorts = np.maximum(-np.asarray(live_short_shares, dtype=float), 0.0)
    prices = np.asarray(reference_prices, dtype=float)

    if borrow_fees_daily is not None:
        fees = np.asarray(borrow_fees_daily, dtype=float)
    else:
        fees = np.full(shorts.size, cfg.default_fee_annual / cfg.fee_daycount)

    return shorts * prices * fees


def process_recall(
    live_short_shares: np.ndarray,
    new_capacity: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute forced cover from recall events."""
    shorts = np.maximum(-np.asarray(live_short_shares, dtype=float), 0.0)
    cap = np.maximum(np.asarray(new_capacity, dtype=float), 0.0)
    forced = np.maximum(shorts - cap, 0.0)
    return {
        "forced_cover_shares": forced,
        "n_recalls": int((forced > 0).sum()),
        "total_forced_cover": float(forced.sum()),
    }
