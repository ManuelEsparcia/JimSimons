"""
data/edgar/fetch_companyfacts.py — SEC company facts (XBRL) ingestion.

Downloads companyfacts JSON from SEC for each CIK and extracts
a minimal tabular representation of financial facts with full
context (taxonomy, unit, period, filing metadata).

Dual persistence: raw JSON payload + extracted facts table.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from . import (
    EdgarError, InputValidationError, PayloadSchemaError,
    normalize_cik, normalize_accession_number,
    parse_date, parse_numeric, normalize_metric,
    utc_now_iso, sha256_text, config_hash,
    read_dataframe, write_parquet_safe, write_json_safe,
    ensure_directory, stable_json_dumps,
    sec_fetch_json, SecClientConfig, RateLimiter,
    json_safe,
)

LOGGER = logging.getLogger(__name__)

SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
DEFAULT_TAXONOMIES = ("us-gaap", "dei")

FACTS_COLUMNS = [
    "cik", "taxonomy", "tag", "label", "unit", "value",
    "period_start", "period_end", "instant", "fiscal_year", "fiscal_period",
    "form_type", "filed_date", "accession_number", "acceptance_datetime",
]


def extract_facts_from_payload(
    payload: dict[str, Any],
    cik: str,
    taxonomies: Sequence[str] = DEFAULT_TAXONOMIES,
) -> list[dict[str, Any]]:
    """Extract tabular facts from SEC companyfacts payload."""
    cik_norm = normalize_cik(cik)
    facts_node = payload.get("facts", {})
    rows = []

    for taxonomy in taxonomies:
        tax_node = facts_node.get(taxonomy, {})
        for tag, tag_data in tax_node.items():
            label = tag_data.get("label", "")
            units_node = tag_data.get("units", {})
            for unit_key, entries in units_node.items():
                for entry in entries:
                    rows.append({
                        "cik": cik_norm,
                        "taxonomy": taxonomy,
                        "tag": tag,
                        "label": label,
                        "unit": unit_key,
                        "value": parse_numeric(entry.get("val")),
                        "period_start": parse_date(entry.get("start")),
                        "period_end": parse_date(entry.get("end")),
                        "instant": parse_date(entry.get("end")) if "start" not in entry else None,
                        "fiscal_year": entry.get("fy"),
                        "fiscal_period": entry.get("fp"),
                        "form_type": entry.get("form"),
                        "filed_date": parse_date(entry.get("filed")),
                        "accession_number": normalize_accession_number(entry.get("accn")),
                    })
    return rows


def run_fetch_companyfacts(
    cik_list: Sequence[str],
    *,
    taxonomies: Sequence[str] = DEFAULT_TAXONOMIES,
    max_rps: float = 8.0,
    user_agent: str = "QuantResearch/1.0 (research@example.com)",
    output_dir: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Fetch companyfacts for all CIKs."""
    if not run_id:
        run_id = f"facts_{utc_now_iso().replace(':','').replace('-','')}"

    client = SecClientConfig(user_agent=user_agent, max_rps=max_rps)
    limiter = RateLimiter(max_rps)

    all_facts = []
    failures = []

    for cik in cik_list:
        cik_norm = normalize_cik(cik)
        url = SEC_COMPANYFACTS_URL.format(cik=cik_norm)
        try:
            payload = sec_fetch_json(url, client_config=client, rate_limiter=limiter)
            facts = extract_facts_from_payload(payload, cik_norm, taxonomies)
            all_facts.extend(facts)
        except Exception as e:
            failures.append({"cik": cik_norm, "error": str(e)})
            LOGGER.warning("Failed CIK %s: %s", cik_norm, e)

    facts_df = pd.DataFrame(all_facts) if all_facts else pd.DataFrame(columns=FACTS_COLUMNS)

    # Deduplicate by (cik, tag, unit, period_end, accession_number)
    dedup_cols = ["cik", "tag", "unit", "period_end", "accession_number"]
    avail = [c for c in dedup_cols if c in facts_df.columns]
    if avail and len(facts_df) > 0:
        facts_df = facts_df.drop_duplicates(subset=avail, keep="first")

    manifest = {
        "run_id": run_id,
        "n_ciks": len(cik_list),
        "n_success": len(cik_list) - len(failures),
        "n_facts": len(facts_df),
        "n_failures": len(failures),
        "taxonomies": list(taxonomies),
        "timestamp": utc_now_iso(),
    }

    if output_dir:
        out = Path(output_dir) / run_id
        ensure_directory(out)
        write_parquet_safe(facts_df, out / "facts_raw.parquet")
        write_json_safe(manifest, out / "manifest.json")

    LOGGER.info("CompanyFacts: %d CIKs → %d facts, %d failures", len(cik_list), len(facts_df), len(failures))
    return {"facts": facts_df, "failures": failures, "manifest": manifest}
