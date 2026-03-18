"""
data/edgar/parse_xbrl.py — XBRL fact canonisation and normalisation.

Transforms raw EDGAR facts into canonical financial metrics:
1. Map XBRL tags → internal metric names (versionable mapping)
2. Resolve taxonomy priority (us-gaap > dei > ifrs)
3. Normalise units and scale (millions → absolute, etc.)
4. Resolve period context (instant vs duration, fiscal alignment)
5. Deduplicate: deterministic selection among competing candidates
6. Quality score per fact

References the XBRL US GAAP taxonomy for tag → metric mapping.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    EdgarError, ParseError, InputValidationError,
    normalize_cik, normalize_accession_number, normalize_metric,
    parse_date, parse_datetime_utc, parse_numeric,
    utc_now_iso, config_hash, sha256_text,
    read_dataframe, write_parquet_safe, write_json_safe,
    normalize_columns, json_safe,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tag → Metric mapping (the CRITICAL mapping layer)
# ---------------------------------------------------------------------------

DEFAULT_TAG_MAPPING: dict[str, dict[str, Any]] = {
    # Revenue
    "Revenues": {"metric": "revenue", "priority": 1},
    "RevenueFromContractWithCustomerExcludingAssessedTax": {"metric": "revenue", "priority": 2},
    "SalesRevenueNet": {"metric": "revenue", "priority": 3},
    # Net Income
    "NetIncomeLoss": {"metric": "net_income", "priority": 1},
    "ProfitLoss": {"metric": "net_income", "priority": 2},
    # Assets
    "Assets": {"metric": "total_assets", "priority": 1},
    "AssetsCurrent": {"metric": "current_assets", "priority": 1},
    # Equity
    "StockholdersEquity": {"metric": "total_equity", "priority": 1},
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": {"metric": "total_equity", "priority": 2},
    # Liabilities
    "Liabilities": {"metric": "total_liabilities", "priority": 1},
    "LiabilitiesCurrent": {"metric": "current_liabilities", "priority": 1},
    "LongTermDebt": {"metric": "total_debt", "priority": 1},
    "LongTermDebtNoncurrent": {"metric": "total_debt", "priority": 2},
    # Profitability
    "GrossProfit": {"metric": "gross_profit", "priority": 1},
    "OperatingIncomeLoss": {"metric": "operating_income", "priority": 1},
    # Cash flow
    "NetCashProvidedByOperatingActivities": {"metric": "operating_cash_flow", "priority": 1},
    "PaymentsToAcquirePropertyPlantAndEquipment": {"metric": "capex", "priority": 1},
    # DEI
    "EntityCommonStockSharesOutstanding": {"metric": "shares_outstanding", "priority": 1},
    "CommonStockSharesOutstanding": {"metric": "shares_outstanding", "priority": 2},
    # Retained earnings
    "RetainedEarningsAccumulatedDeficit": {"metric": "retained_earnings", "priority": 1},
    # Interest
    "InterestExpense": {"metric": "interest_expense", "priority": 1},
    # Receivables
    "AccountsReceivableNetCurrent": {"metric": "receivables", "priority": 1},
}

TAXONOMY_PRIORITY: dict[str, int] = {
    "us-gaap": 1,
    "dei": 2,
    "ifrs-full": 3,
}

UNIT_CONVERSIONS: dict[str, float] = {
    "usd": 1.0,
    "usd/shares": 1.0,
    "shares": 1.0,
    "pure": 1.0,
    "usd_millions": 1e6,
    "usd_thousands": 1e3,
    "usd_billions": 1e9,
}


def resolve_tag_mapping(tag: str, taxonomy: str, mapping: dict[str, dict] | None = None) -> Optional[dict]:
    """Map an XBRL tag to an internal metric definition."""
    m = mapping or DEFAULT_TAG_MAPPING
    # Case-insensitive match
    for map_tag, defn in m.items():
        if tag.lower() == map_tag.lower():
            return defn
    return None


def normalize_unit(unit_str: str) -> tuple[str, float]:
    """Normalise unit string and return (canonical_unit, scale_factor)."""
    u = str(unit_str).strip().lower().replace(" ", "")
    if u in UNIT_CONVERSIONS:
        return u, UNIT_CONVERSIONS[u]
    if "usd" in u or "$" in u:
        return "usd", 1.0
    if "share" in u:
        return "shares", 1.0
    return u, 1.0


def compute_fact_quality(row: dict) -> float:
    """Compute a quality score [0, 1] for a single fact."""
    score = 1.0
    if row.get("value") is None:
        return 0.0
    if row.get("acceptance_datetime") is None:
        score -= 0.2
    if row.get("accession_number") is None:
        score -= 0.1
    if row.get("filed_date") is None:
        score -= 0.1
    tax = row.get("taxonomy", "")
    tax_priority = TAXONOMY_PRIORITY.get(tax, 5)
    if tax_priority > 2:
        score -= 0.1
    return max(0.0, score)


def canonicalize_facts(
    raw_facts: pd.DataFrame,
    tag_mapping: dict[str, dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Canonicalize raw XBRL facts.

    Returns (canonical_facts, rejected_facts).
    """
    m = tag_mapping or DEFAULT_TAG_MAPPING
    canonical_rows = []
    rejected_rows = []

    for _, row in raw_facts.iterrows():
        tag = str(row.get("tag", ""))
        taxonomy = str(row.get("taxonomy", ""))

        # Map tag
        defn = resolve_tag_mapping(tag, taxonomy, m)
        if defn is None:
            rejected_rows.append({**row.to_dict(), "rejection_class": "unmapped_tag"})
            continue

        metric_name = defn["metric"]
        mapping_priority = defn.get("priority", 5)

        # Normalise unit
        unit_raw = str(row.get("unit", ""))
        unit_canonical, scale = normalize_unit(unit_raw)

        # Value
        val = parse_numeric(row.get("value"))
        if val is None:
            rejected_rows.append({**row.to_dict(), "rejection_class": "non_numeric_value", "metric_name": metric_name})
            continue
        val_scaled = val * scale

        # Quality score
        quality = compute_fact_quality(row.to_dict())

        canonical_rows.append({
            "cik": row.get("cik"),
            "metric_name": metric_name,
            "value": val_scaled,
            "unit_canonical": unit_canonical,
            "unit_raw": unit_raw,
            "scale_factor": scale,
            "taxonomy": taxonomy,
            "tag": tag,
            "mapping_priority": mapping_priority,
            "taxonomy_priority": TAXONOMY_PRIORITY.get(taxonomy, 5),
            "period_start": row.get("period_start"),
            "period_end": row.get("period_end"),
            "instant": row.get("instant"),
            "fiscal_year": row.get("fiscal_year"),
            "fiscal_period": row.get("fiscal_period"),
            "form_type": row.get("form_type"),
            "filed_date": row.get("filed_date"),
            "acceptance_datetime": row.get("acceptance_datetime") or row.get("acceptance_ts"),
            "accession_number": row.get("accession_number"),
            "quality_score": quality,
        })

    canonical = pd.DataFrame(canonical_rows) if canonical_rows else pd.DataFrame()
    rejected = pd.DataFrame(rejected_rows) if rejected_rows else pd.DataFrame()

    # Deduplicate canonical: per (cik, metric_name, fiscal_year, fiscal_period)
    # Keep best: highest taxonomy_priority (lowest number), then mapping_priority, then quality
    if len(canonical) > 0:
        canonical = canonical.sort_values(
            ["taxonomy_priority", "mapping_priority", "quality_score"],
            ascending=[True, True, False],
        )
        dedup_cols = ["cik", "metric_name"]
        for col in ["fiscal_year", "fiscal_period", "period_end"]:
            if col in canonical.columns:
                dedup_cols.append(col)
        canonical = canonical.drop_duplicates(subset=dedup_cols, keep="first")

    LOGGER.info("Parse XBRL: %d canonical, %d rejected", len(canonical), len(rejected))
    return canonical, rejected


def run_parse_xbrl(
    raw_facts: pd.DataFrame | str | Path,
    *,
    tag_mapping: dict | None = None,
    output_dir: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run the full XBRL parse/canonicalization pipeline."""
    if not run_id:
        run_id = f"parse_{utc_now_iso().replace(':','').replace('-','')}"

    if isinstance(raw_facts, (str, Path)):
        raw_facts = read_dataframe(raw_facts)

    canonical, rejected = canonicalize_facts(raw_facts, tag_mapping)

    # Rename acceptance column for downstream
    if "acceptance_datetime" in canonical.columns:
        canonical = canonical.rename(columns={"acceptance_datetime": "acceptance_ts"})

    manifest = {
        "run_id": run_id,
        "n_raw": len(raw_facts),
        "n_canonical": len(canonical),
        "n_rejected": len(rejected),
        "metrics_found": sorted(canonical["metric_name"].unique().tolist()) if len(canonical) > 0 else [],
        "timestamp": utc_now_iso(),
    }

    if output_dir:
        out = Path(output_dir) / run_id
        out.mkdir(parents=True, exist_ok=True)
        write_parquet_safe(canonical, out / "facts_canonical.parquet")
        if len(rejected) > 0:
            write_parquet_safe(rejected, out / "facts_rejected.parquet")
        write_json_safe(manifest, out / "manifest.json")

    return {"canonical": canonical, "rejected": rejected, "manifest": manifest}
