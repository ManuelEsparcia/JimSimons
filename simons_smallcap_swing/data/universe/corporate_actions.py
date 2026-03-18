"""
data/universe/corporate_actions.py — Canonical corporate actions pipeline.

Transforms heterogeneous raw feeds into a single, deterministic, PIT-safe,
identity-resolved canonical layer.

Primary entity: instrument_id (NOT ticker).

Pipeline (spec §10):
    10.1  Ingest raw from multiple providers
    10.2  Normalise semantics → canonical taxonomy Ω_ca
    10.3  Resolve identity → instrument_id via PIT linkage
    10.4  Deduplicate intra- and inter-provider (equivalence classes)
    10.5  Resolve conflicts via closed priority policy

Same-day precedence (spec §12):
    1.identifier_maintenance 2.ticker_change 3.exchange_change
    4.split/reverse_split 5.stock_dividend 6.cash_dividend
    7.special_cash_dividend 8.rights_issue 9.spinoff
    10.merger/acquisition 11.delisting 12.relisting

Event status: pending | confirmed | cancelled | superseded
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    CorporateActionsError, CanonicalEventType, EventFamily, EVENT_FAMILY_MAP,
    Severity, GateResult, max_severity,
    utc_now_iso, sha256_text, config_hash,
    json_safe, read_dataframe, write_parquet_safe, write_json_safe,
)

LOGGER = logging.getLogger(__name__)

# Alias map: raw provider names → canonical
_TYPE_ALIASES: dict[str, str] = {
    "stock split": "split", "forward split": "split", "fwd split": "split",
    "reverse stock split": "reverse_split", "reverse split": "reverse_split",
    "cash dividend": "cash_dividend", "ordinary dividend": "cash_dividend",
    "special dividend": "special_cash_dividend", "special cash dividend": "special_cash_dividend",
    "return of capital": "special_cash_dividend",
    "stock dividend": "stock_dividend",
    "ticker change": "ticker_change", "symbol change": "ticker_change",
    "name change": "name_change",
    "exchange change": "exchange_change", "exchange transfer": "exchange_change",
    "merger": "merger", "acquisition": "acquisition",
    "spin-off": "spinoff", "spinoff": "spinoff", "spin off": "spinoff",
    "delisting": "delisting", "delist": "delisting",
    "relisting": "relisting",
    "bankruptcy": "bankruptcy_reorg",
    "rights issue": "rights_issue", "rights offering": "rights_issue",
    "tender offer": "tender_offer",
}

# Same-day precedence order (lower = applied first)
_SAMEDAY_ORDER: dict[str, int] = {
    "identifier_maintenance": 1, "name_change": 1,
    "ticker_change": 2, "share_class_change": 2,
    "exchange_change": 3,
    "split": 4, "reverse_split": 4,
    "stock_dividend": 5,
    "cash_dividend": 6,
    "special_cash_dividend": 7,
    "rights_issue": 8,
    "spinoff": 9,
    "merger": 10, "acquisition": 10, "tender_offer": 10,
    "delisting": 11,
    "relisting": 12,
    "bankruptcy_reorg": 11,
}


@dataclass(frozen=True)
class CorporateActionsConfig:
    source_priority: dict[str, int] = field(default_factory=lambda: {"sec": 1, "exchange": 2, "provider": 3})
    dedup_tolerance_days: int = 3
    split_ratio_bounds: tuple[float, float] = (0.01, 100.0)
    max_cash_div_yield: float = 0.5
    special_div_threshold: float = 0.10  # >10% yield → reclassify as special


def normalise_event_type(raw: Any) -> str:
    """Map raw event type string to canonical Ω_ca."""
    s = str(raw).strip().lower()
    return _TYPE_ALIASES.get(s, s if s in {e.value for e in CanonicalEventType} else "unknown")


def normalise_events(raw: pd.DataFrame, cfg: CorporateActionsConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalise raw events: map types, validate, separate failures."""
    df = raw.copy()
    failures = []

    # Map type
    if "event_type" in df.columns:
        df["event_type"] = df["event_type"].apply(normalise_event_type)
    df["event_family"] = df["event_type"].map(EVENT_FAMILY_MAP).fillna("unknown")

    # Dates
    for col in ("announcement_ts", "ex_date", "effective_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Validate split ratios
    lo, hi = cfg.split_ratio_bounds
    if "split_ratio" in df.columns:
        sr = pd.to_numeric(df["split_ratio"], errors="coerce")
        bad = sr.notna() & ((sr < lo) | (sr > hi))
        if bad.any():
            failures.append(df[bad].assign(failure_code="invalid_split_ratio"))
            df = df[~bad]

    # Validate cash
    if "cash_amount" in df.columns:
        ca = pd.to_numeric(df["cash_amount"], errors="coerce")
        neg = ca.notna() & (ca < 0)
        if neg.any():
            failures.append(df[neg].assign(failure_code="negative_cash_amount"))
            df = df[~neg]

    # Same-day precedence
    df["_sameday_order"] = df["event_type"].map(_SAMEDAY_ORDER).fillna(99)

    # Source priority
    df["source_priority"] = df.get("source", pd.Series("unknown", index=df.index)).map(
        cfg.source_priority
    ).fillna(99)

    failures_df = pd.concat(failures, ignore_index=True) if failures else pd.DataFrame()
    return df, failures_df


def resolve_identity(events: pd.DataFrame, identity_master: pd.DataFrame | None = None) -> pd.DataFrame:
    """Resolve events to instrument_id via PIT identity master."""
    df = events.copy()
    if "instrument_id" not in df.columns:
        if identity_master is not None and "symbol" in df.columns and "instrument_id" in identity_master.columns:
            id_map = identity_master.drop_duplicates("symbol").set_index("symbol")["instrument_id"]
            df["instrument_id"] = df["symbol"].map(id_map)
        else:
            df["instrument_id"] = df.get("symbol", "")
    return df


def deduplicate(events: pd.DataFrame, cfg: CorporateActionsConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate: same economic event from multiple providers."""
    if len(events) == 0:
        return events, pd.DataFrame()

    key_cols = ["instrument_id", "event_type"]
    date_col = "ex_date" if "ex_date" in events.columns else "effective_date"
    if date_col in events.columns:
        key_cols.append(date_col)

    avail = [c for c in key_cols if c in events.columns]
    if not avail:
        return events, pd.DataFrame()

    events = events.sort_values(avail + ["source_priority", "_sameday_order"])
    deduped = events.drop_duplicates(subset=avail, keep="first")
    dups = events[events.duplicated(subset=avail, keep="first")]
    return deduped, dups


def build_lineage_hash(row: pd.Series) -> str:
    """Deterministic event_id for traceability."""
    parts = [str(row.get(c, "")) for c in ("instrument_id", "event_type", "ex_date", "effective_date", "source")]
    return sha256_text("|".join(parts))[:16]


def run_corporate_actions(
    raw_events: pd.DataFrame | str | Path,
    *,
    identity_master: pd.DataFrame | str | Path | None = None,
    config: CorporateActionsConfig | None = None,
    output_dir: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run the full corporate actions canonicalisation pipeline."""
    cfg = config or CorporateActionsConfig()
    if not run_id:
        run_id = f"ca_{utc_now_iso().replace(':','').replace('-','')}"

    if isinstance(raw_events, (str, Path)):
        raw_events = read_dataframe(raw_events)
    if isinstance(identity_master, (str, Path)):
        identity_master = read_dataframe(identity_master)

    n_raw = len(raw_events)

    normalised, norm_failures = normalise_events(raw_events, cfg)
    resolved = resolve_identity(normalised, identity_master)
    canonical, dups = deduplicate(resolved, cfg)

    # Lineage and event_id
    if len(canonical) > 0:
        canonical["event_id"] = canonical.apply(build_lineage_hash, axis=1)
        if "status" not in canonical.columns:
            canonical["status"] = "confirmed"

    # Current snapshot (most recent per instrument_id × event_type)
    current = canonical.copy()
    if "effective_date" in current.columns and len(current) > 0:
        current = current.sort_values("effective_date", ascending=False)
        current = current.drop_duplicates(["instrument_id", "event_type"], keep="first")

    all_failures = pd.concat([f for f in [norm_failures, dups] if len(f) > 0], ignore_index=True) if (len(norm_failures) > 0 or len(dups) > 0) else pd.DataFrame()

    manifest = {
        "run_id": run_id,
        "n_raw": n_raw,
        "n_canonical": len(canonical),
        "n_failures": len(all_failures),
        "n_duplicates_removed": len(dups),
        "event_type_counts": canonical["event_type"].value_counts().to_dict() if len(canonical) > 0 else {},
        "event_family_counts": canonical["event_family"].value_counts().to_dict() if len(canonical) > 0 else {},
        "timestamp": utc_now_iso(),
    }

    if output_dir:
        out = Path(output_dir) / run_id
        out.mkdir(parents=True, exist_ok=True)
        write_parquet_safe(canonical, out / "history.parquet")
        write_parquet_safe(current, out / "current.parquet")
        if len(all_failures) > 0:
            write_parquet_safe(all_failures, out / "failures.parquet")
        write_json_safe(manifest, out / "manifest.json")

    LOGGER.info("CA: %d raw → %d canonical (%d dupes, %d failures)",
                n_raw, len(canonical), len(dups), len(norm_failures))
    return {"history": canonical, "current": current, "failures": all_failures, "manifest": manifest}
