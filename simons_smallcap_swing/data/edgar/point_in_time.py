"""
data/edgar/point_in_time.py — Point-in-time fundamentals materialisation.

THE most P&L-critical file in the EDGAR pipeline. If this file has
look-ahead bias, every fundamental feature is contaminated and the
entire backtest is invalid.

Core invariant (NEVER violate):
    For every observation (asof, symbol, metric):
        acceptance_datetime(source_filing) <= asof

    No exceptions. No "close enough". No "probably available".

Pipeline:
1. Resolve symbol → CIK via PIT-safe identity mapping
2. For each (asof, symbol, metric):
   a. Filter candidates: acceptance_ts <= asof
   b. Select best candidate (latest accepted, highest quality)
   c. Apply staleness policy per metric
   d. Apply quality flags (penalize / exclude / allow)
3. Output: panel (asof, symbol, metric_name) → value

Policies:
    NaN:      no observable data → NaN (never fabricate)
    penalize: data exists but stale or low quality → value with flag
    exclude:  data exists but unacceptable → NaN with reason
    allow:    data is fresh and high quality → value

Staleness per metric type:
    Income statement items: max 180 days (reported quarterly)
    Balance sheet items:    max 120 days
    Cash flow items:        max 180 days
    DEI items:              max 365 days (annual filings)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    EdgarError, PITError, InputValidationError,
    normalize_cik, normalize_symbol,
    parse_date, parse_datetime_utc, parse_date_series,
    utc_now_iso, config_hash, json_safe,
    read_dataframe, write_parquet_safe, write_json_safe,
    deep_merge, normalize_columns,
    LOGGER as _PARENT_LOGGER,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "pit_config_version": "1.0.0",
    "storage": {
        "output_root": "data/edgar/pit",
        "allow_csv_fallback": True,
        "compression": "snappy",
    },
    "temporal": {
        "decision_time": "16:00:00",   # 4pm ET — after market close
        "asof_frequency": "B",         # business days
    },
    "staleness": {
        "default_max_days": 180,
        "per_metric_type": {
            "income_statement": 180,
            "balance_sheet": 120,
            "cash_flow": 180,
            "dei": 365,
        },
        "per_metric_override": {},      # metric_name → max_days
    },
    "quality": {
        "min_quality_score": 0.3,       # below this → exclude
        "penalize_below": 0.7,          # between min and this → penalize
        "exclude_restatement_hard": True,
        "exclude_amended_unresolved": True,
    },
    "coverage": {
        "min_metrics_per_symbol": 3,
        "min_symbols_per_date": 10,
    },
    "amendment_policy": {
        "prefer_latest_amendment": True,
        "amendment_forms": ["10-K/A", "10-Q/A", "20-F/A"],
    },
}

# Metric type classification for staleness
METRIC_TYPE_MAP: dict[str, str] = {
    "revenue": "income_statement",
    "net_income": "income_statement",
    "gross_profit": "income_statement",
    "operating_income": "income_statement",
    "ebitda": "income_statement",
    "total_assets": "balance_sheet",
    "total_equity": "balance_sheet",
    "total_debt": "balance_sheet",
    "current_assets": "balance_sheet",
    "current_liabilities": "balance_sheet",
    "total_liabilities": "balance_sheet",
    "retained_earnings": "balance_sheet",
    "cash_from_operations": "cash_flow",
    "operating_cash_flow": "cash_flow",
    "capex": "cash_flow",
    "free_cash_flow": "cash_flow",
    "shares_outstanding": "dei",
    "common_shares": "dei",
}

# Column aliases for input normalization
FACTS_ALIASES: dict[str, Sequence[str]] = {
    "cik": ["cik", "CIK", "cik_str"],
    "metric_name": ["metric_name", "metric", "concept", "tag"],
    "value": ["value", "val", "amount"],
    "unit": ["unit", "unit_canonical", "uom"],
    "period_start": ["period_start", "start_date", "startDate"],
    "period_end": ["period_end", "end_date", "endDate"],
    "filed_date": ["filed_date", "filedDate", "filing_date"],
    "acceptance_ts": ["acceptance_ts", "acceptance_datetime", "acceptanceDateTime", "accepted_at"],
    "accession_number": ["accession_number", "accessionNumber", "accession"],
    "form_type": ["form_type", "formType", "form"],
    "fiscal_year": ["fiscal_year", "fy", "fiscalYear"],
    "fiscal_period": ["fiscal_period", "fp", "fiscalPeriod"],
    "quality_score": ["quality_score", "fact_quality_score"],
}

MAPPING_ALIASES: dict[str, Sequence[str]] = {
    "symbol": ["symbol", "ticker"],
    "cik": ["cik", "CIK"],
    "effective_from": ["effective_from", "valid_from", "start_date"],
    "effective_to": ["effective_to", "valid_to", "end_date"],
    "confidence": ["confidence", "confidence_score"],
}


# ---------------------------------------------------------------------------
# Staleness computation
# ---------------------------------------------------------------------------

def get_max_staleness_days(metric_name: str, cfg: dict[str, Any]) -> int:
    """Get maximum staleness for a specific metric.

    Priority: per_metric_override > per_metric_type > default.
    """
    staleness_cfg = cfg.get("staleness", {})
    overrides = staleness_cfg.get("per_metric_override", {})
    if metric_name in overrides:
        return int(overrides[metric_name])

    metric_type = METRIC_TYPE_MAP.get(metric_name, "unknown")
    type_map = staleness_cfg.get("per_metric_type", {})
    if metric_type in type_map:
        return int(type_map[metric_type])

    return int(staleness_cfg.get("default_max_days", 180))


def compute_staleness_days(
    acceptance_ts: pd.Series,
    asof: pd.Timestamp,
) -> pd.Series:
    """Days between acceptance and asof date."""
    accepted = pd.to_datetime(acceptance_ts, errors="coerce")
    asof_ts = pd.Timestamp(asof)
    return (asof_ts - accepted).dt.days


# ---------------------------------------------------------------------------
# PIT candidate selection (THE CORE — no leakage allowed)
# ---------------------------------------------------------------------------

def filter_observable_candidates(
    facts: pd.DataFrame,
    asof: pd.Timestamp,
) -> pd.DataFrame:
    """Filter facts to those observable at asof.

    HARD INVARIANT: acceptance_ts <= asof. No exceptions.
    """
    if "acceptance_ts" not in facts.columns:
        raise PITError("Facts table missing 'acceptance_ts' — cannot enforce PIT")

    accepted = pd.to_datetime(facts["acceptance_ts"], errors="coerce")
    asof_ts = pd.Timestamp(asof)

    # THE CRITICAL FILTER — everything before this is preparation
    observable_mask = accepted <= asof_ts

    # Also exclude NaT acceptance (we don't know when it became available)
    observable_mask = observable_mask & accepted.notna()

    return facts[observable_mask].copy()


def select_best_candidate(
    candidates: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.Series:
    """Select the best candidate per (cik, metric_name) group.

    Deterministic selection priority:
    1. Latest acceptance_ts (most recent filing visible)
    2. Higher quality_score (better quality filing)
    3. Latest filed_date (tiebreaker)
    4. Lexicographic accession_number (final deterministic tiebreaker)
    """
    if len(candidates) == 0:
        return pd.Series(dtype=object)

    df = candidates.copy()

    # Ensure sort columns exist
    for col in ["acceptance_ts", "quality_score", "filed_date", "accession_number"]:
        if col not in df.columns:
            if col == "quality_score":
                df[col] = 1.0
            elif col in ("acceptance_ts", "filed_date"):
                df[col] = pd.NaT
            else:
                df[col] = ""

    # Sort by selection priority (descending)
    df["_acceptance_ts"] = pd.to_datetime(df["acceptance_ts"], errors="coerce")
    df["_filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce")
    df["_quality"] = pd.to_numeric(df["quality_score"], errors="coerce").fillna(0)
    df["_accession"] = df["accession_number"].astype(str).fillna("")

    df = df.sort_values(
        ["_acceptance_ts", "_quality", "_filed_date", "_accession"],
        ascending=[False, False, False, False],
    )

    # Group by (cik, metric_name) and take first (= best)
    group_cols = ["cik", "metric_name"]
    if "fiscal_year" in df.columns and "fiscal_period" in df.columns:
        group_cols.extend(["fiscal_year", "fiscal_period"])

    return df.groupby(group_cols).first().reset_index()


# ---------------------------------------------------------------------------
# Quality and staleness policy
# ---------------------------------------------------------------------------

def apply_staleness_and_quality(
    selected: pd.DataFrame,
    asof: pd.Timestamp,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Apply staleness and quality policies to selected candidates.

    For each observation, assigns:
        pit_action: "allow" | "penalize" | "exclude"
        pit_reason: human-readable reason
        staleness_days: days since acceptance
    """
    df = selected.copy()
    quality_cfg = cfg.get("quality", {})
    min_quality = quality_cfg.get("min_quality_score", 0.3)
    penalize_below = quality_cfg.get("penalize_below", 0.7)

    # Staleness
    df["staleness_days"] = compute_staleness_days(df["acceptance_ts"], asof)

    # Per-metric max staleness
    df["max_staleness"] = df["metric_name"].apply(
        lambda m: get_max_staleness_days(m, cfg)
    )

    # Initialize actions
    df["pit_action"] = "allow"
    df["pit_reason"] = ""

    # Rule 1: Stale data
    stale_mask = df["staleness_days"] > df["max_staleness"]
    df.loc[stale_mask, "pit_action"] = "exclude"
    df.loc[stale_mask, "pit_reason"] = "stale: " + df.loc[stale_mask, "staleness_days"].astype(str) + "d > max " + df.loc[stale_mask, "max_staleness"].astype(str) + "d"

    # Rule 2: Low quality
    if "quality_score" in df.columns:
        quality = pd.to_numeric(df["quality_score"], errors="coerce").fillna(1.0)
        low_quality = quality < min_quality
        df.loc[low_quality & (df["pit_action"] == "allow"), "pit_action"] = "exclude"
        df.loc[low_quality & (df["pit_reason"] == ""), "pit_reason"] = "quality_below_minimum"

        penalized = (quality >= min_quality) & (quality < penalize_below)
        df.loc[penalized & (df["pit_action"] == "allow"), "pit_action"] = "penalize"
        df.loc[penalized & (df["pit_reason"] == ""), "pit_reason"] = "quality_penalized"

    # Rule 3: Restatement hard exclusion
    if quality_cfg.get("exclude_restatement_hard", True):
        for col in ("flag_restatement_hard", "restatement_hard"):
            if col in df.columns:
                restate = df[col].fillna(0).astype(bool)
                df.loc[restate, "pit_action"] = "exclude"
                df.loc[restate & (df["pit_reason"] == ""), "pit_reason"] = "restatement_hard"

    return df


# ---------------------------------------------------------------------------
# Identity resolution (PIT-safe symbol → CIK)
# ---------------------------------------------------------------------------

def resolve_pit_identity(
    mapping: pd.DataFrame,
    symbol: str,
    asof: pd.Timestamp,
) -> Optional[str]:
    """Resolve symbol to CIK at a specific point in time.

    Uses the mapping table with effective_from/effective_to windows.
    Returns the CIK that was valid for this symbol at asof.
    """
    sym = normalize_symbol(symbol)
    if not sym:
        return None

    m = mapping[mapping["symbol"] == sym].copy()
    if len(m) == 0:
        return None

    # Parse dates
    m["_from"] = pd.to_datetime(m.get("effective_from"), errors="coerce").fillna(pd.Timestamp.min)
    m["_to"] = pd.to_datetime(m.get("effective_to"), errors="coerce").fillna(pd.Timestamp.max)

    asof_ts = pd.Timestamp(asof)
    active = m[(m["_from"] <= asof_ts) & (asof_ts < m["_to"])]

    if len(active) == 0:
        return None
    if len(active) == 1:
        return normalize_cik(active.iloc[0]["cik"])

    # Multiple active mappings — pick highest confidence
    if "confidence" in active.columns:
        active = active.sort_values("confidence", ascending=False)
    return normalize_cik(active.iloc[0]["cik"])


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def build_pit_panel(
    facts: pd.DataFrame,
    mapping: pd.DataFrame,
    asof_dates: Sequence[pd.Timestamp],
    metrics: Sequence[str],
    cfg: dict[str, Any],
    flags: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the full PIT fundamentals panel.

    Returns: (panel, coverage, issues)
        panel:    (asof, symbol, metric_name) → value + metadata
        coverage: per (asof, symbol) coverage summary
        issues:   per observation issue log
    """
    # Normalize inputs
    facts = normalize_columns(facts, FACTS_ALIASES)
    mapping = normalize_columns(mapping, MAPPING_ALIASES)

    # Ensure required columns
    for col in ("cik", "metric_name", "value", "acceptance_ts"):
        if col not in facts.columns:
            raise PITError(f"Facts table missing required column: {col}")
    for col in ("symbol", "cik"):
        if col not in mapping.columns:
            raise PITError(f"Mapping table missing required column: {col}")

    # Join flags if provided
    if flags is not None and "accession_number" in flags.columns and "accession_number" in facts.columns:
        flag_cols = [c for c in flags.columns if c.startswith("flag_") or c == "quality_score" or c == "accession_number"]
        if flag_cols:
            facts = facts.merge(
                flags[flag_cols].drop_duplicates("accession_number"),
                on="accession_number", how="left", suffixes=("", "_flag"),
            )

    # Get unique symbols from mapping
    symbols = mapping["symbol"].unique()
    all_rows: list[dict[str, Any]] = []
    all_issues: list[dict[str, Any]] = []

    for asof in asof_dates:
        asof_ts = pd.Timestamp(asof)

        for symbol in symbols:
            # PIT-safe identity resolution
            cik = resolve_pit_identity(mapping, symbol, asof_ts)
            if cik is None:
                all_issues.append({
                    "asof": asof_ts, "symbol": symbol,
                    "issue": "no_active_cik_mapping",
                    "severity": "info",
                })
                continue

            # Filter facts for this CIK
            cik_facts = facts[facts["cik"] == cik]
            if len(cik_facts) == 0:
                continue

            for metric in metrics:
                metric_facts = cik_facts[cik_facts["metric_name"] == metric]
                if len(metric_facts) == 0:
                    all_rows.append({
                        "asof": asof_ts, "symbol": symbol, "metric_name": metric,
                        "value": np.nan, "pit_action": "no_data",
                        "staleness_days": np.nan, "cik": cik,
                    })
                    continue

                # CRITICAL: filter to observable candidates only
                observable = filter_observable_candidates(metric_facts, asof_ts)

                if len(observable) == 0:
                    all_rows.append({
                        "asof": asof_ts, "symbol": symbol, "metric_name": metric,
                        "value": np.nan, "pit_action": "not_yet_observable",
                        "staleness_days": np.nan, "cik": cik,
                    })
                    continue

                # Select best candidate
                best = select_best_candidate(observable, cfg)

                if len(best) == 0:
                    all_rows.append({
                        "asof": asof_ts, "symbol": symbol, "metric_name": metric,
                        "value": np.nan, "pit_action": "selection_failed",
                        "staleness_days": np.nan, "cik": cik,
                    })
                    continue

                # Apply staleness and quality
                best = apply_staleness_and_quality(best, asof_ts, cfg)

                row = best.iloc[0]
                val = row["value"]
                action = row.get("pit_action", "allow")

                if action == "exclude":
                    val = np.nan

                all_rows.append({
                    "asof": asof_ts,
                    "symbol": symbol,
                    "metric_name": metric,
                    "value": val,
                    "cik": cik,
                    "pit_action": action,
                    "pit_reason": row.get("pit_reason", ""),
                    "staleness_days": row.get("staleness_days", np.nan),
                    "acceptance_ts": row.get("acceptance_ts"),
                    "filed_date": row.get("filed_date"),
                    "accession_number": row.get("accession_number"),
                    "fiscal_year": row.get("fiscal_year"),
                    "fiscal_period": row.get("fiscal_period"),
                    "quality_score": row.get("quality_score"),
                })

    panel = pd.DataFrame(all_rows)
    issues = pd.DataFrame(all_issues) if all_issues else pd.DataFrame()

    # Coverage summary
    coverage = _compute_coverage(panel, asof_dates, symbols, metrics)

    LOGGER.info(
        "PIT panel built: %d rows, %d asof dates × %d symbols × %d metrics, "
        "coverage=%.1f%%, excluded=%.1f%%",
        len(panel), len(asof_dates), len(symbols), len(metrics),
        (panel["pit_action"] == "allow").mean() * 100 if len(panel) > 0 else 0,
        (panel["pit_action"] == "exclude").mean() * 100 if len(panel) > 0 else 0,
    )

    return panel, coverage, issues


def _compute_coverage(
    panel: pd.DataFrame,
    asof_dates: Sequence,
    symbols: Sequence,
    metrics: Sequence,
) -> pd.DataFrame:
    """Compute coverage: fraction of (asof, symbol) with non-NaN values."""
    if len(panel) == 0:
        return pd.DataFrame()

    cov_rows = []
    for asof in asof_dates:
        asof_panel = panel[panel["asof"] == pd.Timestamp(asof)]
        for sym in symbols:
            sym_panel = asof_panel[asof_panel["symbol"] == str(sym)]
            n_metrics = len(metrics)
            n_allowed = (sym_panel["pit_action"] == "allow").sum()
            n_penalized = (sym_panel["pit_action"] == "penalize").sum()
            n_excluded = (sym_panel["pit_action"] == "exclude").sum()
            n_no_data = n_metrics - len(sym_panel)
            cov_rows.append({
                "asof": pd.Timestamp(asof), "symbol": sym,
                "n_metrics": n_metrics,
                "n_allowed": int(n_allowed),
                "n_penalized": int(n_penalized),
                "n_excluded": int(n_excluded),
                "n_no_data": int(n_no_data),
                "coverage_pct": round(int(n_allowed + n_penalized) / max(n_metrics, 1) * 100, 1),
            })

    return pd.DataFrame(cov_rows)


# ---------------------------------------------------------------------------
# Pivot to wide format for feature consumption
# ---------------------------------------------------------------------------

def pivot_to_wide(panel: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format PIT panel to wide: (asof, symbol) × metric columns.

    Only includes observations with pit_action in ("allow", "penalize").
    Excluded observations become NaN.
    """
    usable = panel[panel["pit_action"].isin(("allow", "penalize"))].copy()
    if len(usable) == 0:
        return pd.DataFrame()

    wide = usable.pivot_table(
        index=["asof", "symbol"],
        columns="metric_name",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    return wide


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def materialize_point_in_time(
    facts_path: str | Path | pd.DataFrame,
    mapping_path: str | Path | pd.DataFrame,
    *,
    asof_dates: Sequence[str] | None = None,
    asof_start: str | None = None,
    asof_end: str | None = None,
    metrics: Sequence[str] | None = None,
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    flags_path: str | Path | pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Materialize the PIT fundamentals panel.

    Parameters
    ----------
    facts_path : path to canonical facts or DataFrame
    mapping_path : path to ticker-CIK mapping or DataFrame
    asof_dates : explicit list of dates, or use asof_start/end
    metrics : list of metric names to materialize
    config : configuration dict (merged with defaults)
    flags_path : path to filings flags or DataFrame (optional)

    Returns
    -------
    dict with keys: panel, coverage, issues, manifest, config
    """
    # Load config
    cfg = dict(DEFAULT_CONFIG)
    if config_path:
        user_cfg = json.loads(Path(config_path).read_text()) if isinstance(config_path, (str, Path)) else {}
        cfg = deep_merge(cfg, user_cfg)
    if config:
        cfg = deep_merge(cfg, config)

    if not run_id:
        run_id = f"pit_{utc_now_iso().replace(':', '').replace('-', '')}"

    # Load data
    if isinstance(facts_path, pd.DataFrame):
        facts = facts_path
    else:
        facts = read_dataframe(facts_path)

    if isinstance(mapping_path, pd.DataFrame):
        mapping = mapping_path
    else:
        mapping = read_dataframe(mapping_path)

    flags = None
    if flags_path is not None:
        if isinstance(flags_path, pd.DataFrame):
            flags = flags_path
        else:
            flags = read_dataframe(flags_path)

    # Resolve asof grid
    if asof_dates is None:
        if asof_start and asof_end:
            freq = cfg.get("temporal", {}).get("asof_frequency", "B")
            asof_dates = pd.bdate_range(asof_start, asof_end, freq=freq)
        else:
            # Use all unique acceptance dates as grid
            all_dates = pd.to_datetime(facts.get("acceptance_ts", facts.get("filed_date")), errors="coerce")
            asof_dates = pd.bdate_range(all_dates.min(), all_dates.max(), freq="B")
    else:
        asof_dates = [pd.Timestamp(d) for d in asof_dates]

    # Resolve metrics
    facts = normalize_columns(facts, FACTS_ALIASES)
    if metrics is None:
        metrics = sorted(facts["metric_name"].dropna().unique())

    # Build panel
    panel, coverage, issues = build_pit_panel(
        facts, mapping, asof_dates, metrics, cfg, flags,
    )

    # Wide format for downstream
    wide = pivot_to_wide(panel)

    # Persist if output_dir specified
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "config_hash": config_hash(cfg),
        "n_asof_dates": len(asof_dates),
        "n_symbols": int(panel["symbol"].nunique()) if len(panel) > 0 else 0,
        "n_metrics": len(metrics),
        "n_rows": len(panel),
        "n_wide_rows": len(wide),
        "coverage_mean_pct": float(coverage["coverage_pct"].mean()) if len(coverage) > 0 else 0,
        "pct_allowed": float((panel["pit_action"] == "allow").mean() * 100) if len(panel) > 0 else 0,
        "pct_excluded": float((panel["pit_action"] == "exclude").mean() * 100) if len(panel) > 0 else 0,
        "pct_penalized": float((panel["pit_action"] == "penalize").mean() * 100) if len(panel) > 0 else 0,
        "timestamp": utc_now_iso(),
    }

    if output_dir:
        out = Path(output_dir) / run_id
        out.mkdir(parents=True, exist_ok=True)
        write_parquet_safe(panel, out / "pit_panel.parquet")
        write_parquet_safe(wide, out / "pit_wide.parquet")
        write_parquet_safe(coverage, out / "pit_coverage.parquet")
        if len(issues) > 0:
            write_parquet_safe(issues, out / "pit_issues.parquet")
        write_json_safe(manifest, out / "manifest.json")

    return {
        "panel": panel,
        "wide": wide,
        "coverage": coverage,
        "issues": issues,
        "manifest": manifest,
        "config": cfg,
    }
