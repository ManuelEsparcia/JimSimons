"""
execution/fills_simulator.py — Orders → fills with shared daily capacity.

Shared capacity: Q_daycap = φ × ADV per symbol-date.
Priority: risk-reducing first → priority score → order_id.
Fill price: p_ref + sign × Δp (spread + slippage + impact).
Carry: unfilled remainder persists with max_carry_days.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np


@dataclass(frozen=True)
class FillsConfig:
    phi: float = 0.10                # fraction of ADV available
    price_benchmark: str = "close"   # close | vwap | open
    spread_deterioration: float = 0.5  # half-spread applied to fill price
    slippage_bps: float = 2.0
    carry_unfilled: bool = True
    max_carry_days: int = 3
    min_fill_shares: int = 1


def simulate_fills(
    orders: list[dict[str, Any]],
    market_data: dict[str, dict[str, Any]],
    *,
    config: FillsConfig | None = None,
) -> dict[str, Any]:
    """Simulate fills for a single date.

    orders: list of dicts with keys: symbol, order_id, side, delta_shares, priority
    market_data: {symbol: {close, open, high, low, adv, spread_bps, halt}}
    """
    cfg = config or FillsConfig()
    fills = []
    order_states = []

    # Group by symbol
    by_symbol: dict[str, list[dict]] = {}
    for o in orders:
        by_symbol.setdefault(o["symbol"], []).append(o)

    for sym, sym_orders in by_symbol.items():
        md = market_data.get(sym)
        if md is None or md.get("halt", False):
            for o in sym_orders:
                order_states.append({"order_id": o["order_id"], "symbol": sym,
                                     "fill_shares": 0, "remaining": abs(o["delta_shares"]),
                                     "status": "blocked", "reason": "halt" if md and md.get("halt") else "no_data"})
            continue

        adv = md.get("adv", 0)
        if adv <= 0:
            for o in sym_orders:
                order_states.append({"order_id": o["order_id"], "symbol": sym,
                                     "fill_shares": 0, "remaining": abs(o["delta_shares"]),
                                     "status": "blocked", "reason": "no_adv"})
            continue

        daycap = cfg.phi * adv
        p_ref = md.get(cfg.price_benchmark, md.get("close", 0))
        spread = md.get("spread_bps", 10.0)

        # Sort: risk-reducing first, then priority desc, then order_id
        sym_orders.sort(key=lambda o: (
            0 if o.get("risk_reducing", False) else 1,
            -o.get("priority", 0),
            o["order_id"],
        ))

        cap_remaining = daycap
        for o in sym_orders:
            req = abs(o["delta_shares"])
            alloc = min(req, cap_remaining)
            fill_shares = max(0, int(np.floor(alloc)))
            if fill_shares < cfg.min_fill_shares:
                fill_shares = 0

            cap_remaining -= fill_shares
            remaining = req - fill_shares

            # Fill price with directional deterioration
            if fill_shares > 0 and p_ref > 0:
                sign = 1 if o.get("side", "buy") == "buy" else -1
                delta_p = (cfg.spread_deterioration * spread + cfg.slippage_bps) * 1e-4 * p_ref
                fill_price = p_ref + sign * delta_p

                # Clip to day range if available
                lo, hi = md.get("low", 0), md.get("high", float("inf"))
                if lo > 0 and hi > 0:
                    fill_price = max(lo, min(hi, fill_price))

                fills.append({
                    "symbol": sym, "order_id": o["order_id"],
                    "fill_shares": fill_shares * (1 if o.get("side") == "buy" else -1),
                    "fill_price": round(fill_price, 4),
                    "fill_notional": round(fill_shares * fill_price, 2),
                    "participation": round(fill_shares / adv, 6) if adv > 0 else 0,
                })

            status = "full" if remaining == 0 else ("partial" if fill_shares > 0 else "none")
            order_states.append({
                "order_id": o["order_id"], "symbol": sym,
                "fill_shares": fill_shares, "remaining": int(remaining),
                "status": status,
                "reason": "fully_executed" if status == "full" else (
                    "capacity_exhausted" if cap_remaining <= 0 else "partial_capacity"),
            })

    total_requested = sum(abs(o["delta_shares"]) for o in orders)
    total_filled = sum(abs(f["fill_shares"]) for f in fills)

    return {
        "fills": fills,
        "order_states": order_states,
        "completion_rate": round(total_filled / max(total_requested, 1), 4),
        "n_fills": len(fills),
        "n_orders": len(orders),
        "capacity_utilisation": round(total_filled / max(sum(
            cfg.phi * market_data.get(s, {}).get("adv", 0) for s in by_symbol), 1), 4),
    }
