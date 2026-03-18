"""
data/edgar/ticker_cik.py — Master identity resolution: ticker ↔ CIK.

Produces a versioned, PIT-safe mapping between tradeable symbols and
SEC regulatory identities. Handles: ticker changes, delistings,
re-listings, mergers, share classes, and conflicting sources.

Output:
    history:   (symbol, cik, effective_from, effective_to, confidence, source)
    current:   active snapshot at asof
    conflicts: unresolved cases for manual review
    manifest:  run metadata and quality metrics
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    EdgarError, InputValidationError,
    normalize_cik, normalize_symbol, parse_date,
    utc_now_iso, config_hash, sha256_text,
    read_dataframe, write_parquet_safe, write_json_safe,
    deep_merge, normalize_columns, json_safe,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG: dict[str, Any] = {
    "storage": {"output_root": "data/edgar/mappings", "allow_csv_fallback": True},
    "identity": {
        "default_effective_from": "1900-01-01",
        "default_effective_to": "2099-12-31",
    },
    "source_priority": {"sec_tickers": 1, "sec_company_tickers": 2, "internal": 3, "manual": 0},
    "confidence": {"default": 0.5, "sec_official": 0.9, "internal": 0.7, "manual": 1.0},
    "conflict_policy": "prefer_higher_priority",
}

SOURCE_ALIASES: dict[str, Sequence[str]] = {
    "symbol": ["symbol", "ticker", "TICKER"],
    "cik": ["cik", "CIK", "cik_str"],
    "company_name": ["company_name", "companyName", "name", "entity_name"],
    "exchange": ["exchange", "EXCHANGE"],
    "effective_from": ["effective_from", "valid_from"],
    "effective_to": ["effective_to", "valid_to"],
    "source": ["source", "source_name"],
}


def load_and_normalize_source(path: str | Path | pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Load a source file and normalize to canonical columns."""
    if isinstance(path, pd.DataFrame):
        df = path.copy()
    else:
        df = read_dataframe(path)
    df = normalize_columns(df, SOURCE_ALIASES)
    if "symbol" not in df.columns:
        raise InputValidationError(f"Source '{source_name}' missing 'symbol' column")
    if "cik" not in df.columns:
        raise InputValidationError(f"Source '{source_name}' missing 'cik' column")
    df["symbol"] = df["symbol"].apply(normalize_symbol)
    df["cik"] = df["cik"].apply(lambda x: normalize_cik(x) if pd.notna(x) else None)
    df["source"] = source_name
    df = df.dropna(subset=["symbol", "cik"])
    df = df[df["symbol"] != ""]
    return df


def build_candidates(sources: dict[str, pd.DataFrame | str | Path], cfg: dict) -> pd.DataFrame:
    """Load and merge all identity sources into a candidates table."""
    all_dfs = []
    for name, src in sources.items():
        try:
            df = load_and_normalize_source(src, name)
            # Attach priority and confidence
            df["source_priority"] = cfg.get("source_priority", {}).get(name, 5)
            df["confidence"] = cfg.get("confidence", {}).get(name, cfg["confidence"].get("default", 0.5))
            all_dfs.append(df)
        except Exception as e:
            LOGGER.warning("Skipping source '%s': %s", name, e)
    if not all_dfs:
        raise InputValidationError("No valid identity sources loaded")
    return pd.concat(all_dfs, ignore_index=True)


def resolve_conflicts(candidates: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve conflicting mappings: same symbol → different CIKs.

    Returns (resolved_history, unresolved_conflicts).
    """
    conflicts_list = []
    resolved_list = []

    for symbol, group in candidates.groupby("symbol"):
        unique_ciks = group["cik"].unique()
        if len(unique_ciks) == 1:
            # No conflict — take best row
            best = group.sort_values(["source_priority", "confidence"], ascending=[True, False]).iloc[0]
            resolved_list.append(best.to_dict())
        else:
            # Conflict: same symbol, multiple CIKs
            # Resolution: prefer highest priority source, then highest confidence
            group_sorted = group.sort_values(["source_priority", "confidence"], ascending=[True, False])
            winner = group_sorted.iloc[0]
            resolved_list.append(winner.to_dict())
            # Record all non-winners as conflicts
            for _, row in group_sorted.iloc[1:].iterrows():
                if row["cik"] != winner["cik"]:
                    conflicts_list.append({
                        **row.to_dict(),
                        "conflict_class": "multi_cik_for_symbol",
                        "winner_cik": winner["cik"],
                        "winner_source": winner["source"],
                    })

    resolved = pd.DataFrame(resolved_list)
    conflicts = pd.DataFrame(conflicts_list) if conflicts_list else pd.DataFrame()
    return resolved, conflicts


def build_history(resolved: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Build historified mapping with effective_from/to windows."""
    df = resolved.copy()
    default_from = cfg.get("identity", {}).get("default_effective_from", "1900-01-01")
    default_to = cfg.get("identity", {}).get("default_effective_to", "2099-12-31")

    if "effective_from" not in df.columns:
        df["effective_from"] = pd.Timestamp(default_from)
    else:
        df["effective_from"] = pd.to_datetime(df["effective_from"], errors="coerce").fillna(pd.Timestamp(default_from))

    if "effective_to" not in df.columns:
        df["effective_to"] = pd.Timestamp(default_to)
    else:
        df["effective_to"] = pd.to_datetime(df["effective_to"], errors="coerce").fillna(pd.Timestamp(default_to))

    # Ensure no overlapping windows for same symbol
    df = df.sort_values(["symbol", "effective_from"]).reset_index(drop=True)
    return df


def build_current(history: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Get active snapshot at asof date."""
    h = history.copy()
    h["effective_from"] = pd.to_datetime(h["effective_from"])
    h["effective_to"] = pd.to_datetime(h["effective_to"])
    active = h[(h["effective_from"] <= asof) & (asof < h["effective_to"])]
    return active.drop_duplicates("symbol", keep="first")


def run_ticker_cik(
    sources: dict[str, pd.DataFrame | str | Path],
    *,
    config: dict[str, Any] | None = None,
    asof: str | None = None,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run the full ticker-CIK resolution pipeline."""
    cfg = deep_merge(DEFAULT_CONFIG, config or {})
    if not run_id:
        run_id = f"ticker_cik_{utc_now_iso().replace(':','').replace('-','')}"
    asof_ts = pd.Timestamp(asof) if asof else pd.Timestamp.now()

    candidates = build_candidates(sources, cfg)
    resolved, conflicts = resolve_conflicts(candidates, cfg)
    history = build_history(resolved, cfg)
    current = build_current(history, asof_ts)

    manifest = {
        "run_id": run_id,
        "asof": str(asof_ts),
        "n_sources": len(sources),
        "n_candidates": len(candidates),
        "n_resolved": len(resolved),
        "n_conflicts": len(conflicts),
        "n_current": len(current),
        "n_unique_symbols": int(history["symbol"].nunique()),
        "n_unique_ciks": int(history["cik"].nunique()),
        "config_hash": config_hash(cfg),
        "timestamp": utc_now_iso(),
    }

    if output_dir:
        out = Path(output_dir) / run_id
        out.mkdir(parents=True, exist_ok=True)
        write_parquet_safe(history, out / "history.parquet")
        write_parquet_safe(current, out / "current.parquet")
        if len(conflicts) > 0:
            write_parquet_safe(conflicts, out / "conflicts.parquet")
        write_json_safe(manifest, out / "manifest.json")

    LOGGER.info("Ticker-CIK: %d resolved, %d conflicts, %d current", len(resolved), len(conflicts), len(current))
    return {"history": history, "current": current, "conflicts": conflicts, "manifest": manifest}
