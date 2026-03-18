"""
labels/event_labels.py — Event-driven episodic labels.

Transforms discrete information triggers (earnings, gaps, shocks) into
PIT episodes with formal entry/exit conventions, overlap resolution,
and outcome computation.

Unit of analysis: episode (NOT panel row).
Each episode = (event_id, symbol, τ_tradable, horizon, outcome).

Timestamp hierarchy (spec §7):
    t_occ ≤ t_pub ≤ t_known ≤ t_trad
    Label anchored at τ_e = t_trad (first tradable moment).

Overlap policy (spec §17): first_event_wins_with_cooldown.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    EventFamily, EventExclusionReason, DEFAULT_EVENT_HORIZONS, Severity,
    utc_now_iso, config_hash, sha256_text, json_safe,
    read_dataframe, write_parquet_safe, write_json_safe,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EventLabelConfig:
    canonical_families: tuple[str, ...] = tuple(f.value for f in EventFamily)
    overlap_policy: str = "first_event_wins_with_cooldown"
    timestamp_policy: str = "strict"
    net_of_costs: bool = True
    missing_cost_policy: str = "strict_invalidate"
    abnormal_mode: str = "none"     # none | market_adjusted | sector_market_adjusted
    decision_lag: int = 0            # 0 = enter at t_trad itself
    classification_default: str = "ternary_tail_20_20_on_net_within_family"
    tail_top_pct: float = 0.20
    tail_bot_pct: float = 0.20


def resolve_tradability(
    events: pd.DataFrame,
    config: EventLabelConfig,
) -> pd.DataFrame:
    """Resolve t_trad from t_pub / t_known timestamps.

    Pre-market event → t_trad = open same day.
    Post-close event → t_trad = open next session.
    Ambiguous → TIMESTAMP_AMBIGUOUS.
    """
    df = events.copy()
    if "event_timestamp_tradable" in df.columns:
        return df

    # Default: use t_pub + 1 session as conservative
    if "event_timestamp_pub" in df.columns:
        df["event_timestamp_tradable"] = pd.to_datetime(df["event_timestamp_pub"], errors="coerce")
    elif "event_timestamp_known" in df.columns:
        df["event_timestamp_tradable"] = pd.to_datetime(df["event_timestamp_known"], errors="coerce")
    else:
        df["event_timestamp_tradable"] = pd.NaT
        df["_ts_ambiguous"] = True

    return df


def resolve_overlaps(
    events: pd.DataFrame,
    config: EventLabelConfig,
) -> pd.DataFrame:
    """Apply overlap policy: first_event_wins_with_cooldown.

    If event e2 arrives within cooldown of e1 (same symbol), e2 is rejected.
    Cooldown = max(horizons) for the event family.
    """
    df = events.sort_values(["symbol", "event_timestamp_tradable"]).copy()
    df["_overlap_rejected"] = False

    for sym, grp in df.groupby("symbol"):
        last_trad = pd.NaT
        last_cooldown = 0
        for idx in grp.index:
            t_trad = grp.loc[idx, "event_timestamp_tradable"]
            family = str(grp.loc[idx, "event_type"]).upper()
            horizons = DEFAULT_EVENT_HORIZONS.get(family, (5,))
            cooldown = max(horizons)

            if pd.notna(last_trad) and pd.notna(t_trad):
                gap = (t_trad - last_trad).days
                if gap <= last_cooldown:
                    df.loc[idx, "_overlap_rejected"] = True
                    continue

            last_trad = t_trad
            last_cooldown = cooldown

    return df


def compute_event_returns(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    config: EventLabelConfig,
) -> pd.DataFrame:
    """Compute event returns for each episode × horizon."""
    # Build price lookup
    price_lookup: dict[tuple[str, Any], float] = {}
    for _, row in prices.iterrows():
        key = (row["symbol"], pd.Timestamp(row["date"]))
        price_lookup[key] = float(row.get("close", row.get("close_adj", np.nan)))

    episodes = []
    for _, ev in events.iterrows():
        sym = ev["symbol"]
        t_trad = pd.Timestamp(ev["event_timestamp_tradable"])
        family = str(ev.get("event_type", "")).upper()
        horizons = DEFAULT_EVENT_HORIZONS.get(family, (5,))

        for h in horizons:
            t_exit = t_trad + pd.Timedelta(days=h)
            entry_px = price_lookup.get((sym, t_trad))
            exit_px = price_lookup.get((sym, t_exit))

            ret = (exit_px / entry_px - 1.0) if (entry_px and exit_px and entry_px > 0) else np.nan

            # Exclusion reason
            reason = None
            if ev.get("_ts_ambiguous"):
                reason = EventExclusionReason.TIMESTAMP_AMBIGUOUS.value
            elif ev.get("_overlap_rejected"):
                reason = EventExclusionReason.OVERLAP_REJECTED.value
            elif pd.isna(t_trad):
                reason = EventExclusionReason.TIMESTAMP_AMBIGUOUS.value
            elif entry_px is None or pd.isna(entry_px):
                reason = EventExclusionReason.MISSING_ENTRY_PRICE.value
            elif exit_px is None or pd.isna(exit_px):
                reason = EventExclusionReason.MISSING_EXIT_PRICE.value

            episodes.append({
                "event_id": ev.get("event_id", sha256_text(f"{sym}|{t_trad}|{family}|{h}")[:12]),
                "symbol": sym,
                "event_type": family,
                "event_timestamp_tradable": t_trad,
                "horizon_days": h,
                "entry_px": entry_px,
                "exit_px": exit_px,
                "event_ret_gross": ret,
                "event_ret_net": ret,  # placeholder — costs attached downstream
                "event_label_valid_flag": reason is None and not pd.isna(ret),
                "event_exclusion_reason": reason,
                "event_start": t_trad,
                "event_end": t_exit,
            })

    return pd.DataFrame(episodes)


def build_event_labels(
    events: pd.DataFrame | str | Path,
    prices: pd.DataFrame | str | Path,
    *,
    config: EventLabelConfig | None = None,
    output_dir: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build event-driven episodic labels."""
    cfg = config or EventLabelConfig()
    if not run_id:
        run_id = f"evlbl_{utc_now_iso().replace(':','').replace('-','')}"

    if isinstance(events, (str, Path)):
        events = read_dataframe(events)
    if isinstance(prices, (str, Path)):
        prices = read_dataframe(prices)

    events = events.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")

    # Resolve timestamps
    events = resolve_tradability(events, cfg)

    # Resolve overlaps
    events = resolve_overlaps(events, cfg)

    # Compute returns
    episodes = compute_event_returns(events, prices, cfg)

    # Classification within family
    for family in episodes["event_type"].unique():
        mask = (episodes["event_type"] == family) & episodes["event_label_valid_flag"]
        if mask.sum() < 10:
            continue
        for h in episodes.loc[mask, "horizon_days"].unique():
            h_mask = mask & (episodes["horizon_days"] == h)
            vals = episodes.loc[h_mask, "event_ret_gross"]
            q_top = vals.quantile(1 - cfg.tail_top_pct)
            q_bot = vals.quantile(cfg.tail_bot_pct)
            cls = pd.Series(0, index=episodes.index)
            cls[episodes["event_ret_gross"] >= q_top] = 1
            cls[episodes["event_ret_gross"] <= q_bot] = -1
            episodes.loc[h_mask, "event_cls"] = cls[h_mask]

    episodes["run_id"] = run_id

    manifest = {
        "run_id": run_id,
        "n_events_raw": len(events),
        "n_episodes": len(episodes),
        "n_valid": int(episodes["event_label_valid_flag"].sum()),
        "n_overlap_rejected": int(events.get("_overlap_rejected", pd.Series(False)).sum()),
        "families": episodes["event_type"].value_counts().to_dict(),
        "overlap_policy": cfg.overlap_policy,
        "timestamp": utc_now_iso(),
    }

    if output_dir:
        out = Path(output_dir) / run_id
        out.mkdir(parents=True, exist_ok=True)
        write_parquet_safe(episodes, out / "event_episodes.parquet")
        write_json_safe(manifest, out / "event_manifest.json")

    LOGGER.info("Event labels: %d episodes, %d valid, %d overlap rejected",
                len(episodes), manifest["n_valid"], manifest["n_overlap_rejected"])
    return {"episodes": episodes, "manifest": manifest}
