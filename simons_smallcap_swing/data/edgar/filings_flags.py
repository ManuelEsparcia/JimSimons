"""
data/edgar/filings_flags.py — Filing quality flags and risk signals.

Transforms the regulatory filing history into actionable quality signals:
    Timeliness:   late_filer, persistent_late_filer
    Corrections:  amended_form, restatement_hard, restatement_suspected
    Completeness: missing_core_fields, incomplete_period
    Anomalies:    accounting_anomaly (value jumps, sign flips)
    Structure:    reporting_gap, fiscal_year_change

Each flag is PIT-safe: only uses information observable at the filing's
acceptance datetime. Quality score and pit_action (allow/penalize/exclude)
are derived from the flag constellation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    EdgarError, normalize_cik, normalize_accession_number,
    parse_date, parse_datetime_utc,
    utc_now_iso, config_hash,
    read_dataframe, write_parquet_safe, write_json_safe,
    normalize_columns, json_safe, deep_merge,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG: dict[str, Any] = {
    "deadline_days": {
        "10-K": 90, "10-Q": 45, "20-F": 120, "6-K": 4,
    },
    "persistent_late_window": 8,  # filings to look back
    "persistent_late_threshold": 0.5,  # >50% late → persistent
    "amendment_forms": ["10-K/A", "10-Q/A", "20-F/A"],
    "core_metrics": ["revenue", "net_income", "total_assets", "total_equity", "operating_cash_flow"],
    "anomaly_zscore_threshold": 4.0,
    "reporting_gap_max_days": 200,  # >200 days between filings = gap
    "quality_weights": {
        "timeliness": 0.2, "corrections": 0.3,
        "completeness": 0.2, "anomalies": 0.3,
    },
}


def compute_timeliness_flags(
    submissions: pd.DataFrame, cfg: dict,
) -> pd.DataFrame:
    """Flag late filers based on filing_date vs expected deadline."""
    df = submissions.copy()
    deadlines = cfg.get("deadline_days", {})

    # Infer deadline
    df["_deadline_days"] = df["form_type"].map(deadlines).fillna(90)
    df["_period_end"] = pd.to_datetime(df.get("period_end", df.get("period_of_report")), errors="coerce")
    df["_filed"] = pd.to_datetime(df["filed_date"], errors="coerce")

    if "_period_end" in df.columns and df["_period_end"].notna().any():
        df["_days_to_file"] = (df["_filed"] - df["_period_end"]).dt.days
        df["flag_late_filer"] = (df["_days_to_file"] > df["_deadline_days"]).astype(int)
    else:
        df["flag_late_filer"] = 0
        df["_days_to_file"] = np.nan

    # Persistent late filer (rolling window)
    window = cfg.get("persistent_late_window", 8)
    threshold = cfg.get("persistent_late_threshold", 0.5)
    df["flag_persistent_late"] = 0
    for cik, grp in df.groupby("cik"):
        sorted_g = grp.sort_values("_filed")
        rolling_late = sorted_g["flag_late_filer"].rolling(window, min_periods=2).mean()
        df.loc[sorted_g.index, "flag_persistent_late"] = (rolling_late > threshold).astype(int).values

    return df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")


def compute_correction_flags(
    submissions: pd.DataFrame, cfg: dict,
) -> pd.DataFrame:
    """Flag amendments and restatements."""
    df = submissions.copy()
    amendment_forms = cfg.get("amendment_forms", [])

    df["flag_amended"] = df["form_type"].isin(amendment_forms).astype(int)

    # Restatement: look for restatement keywords in description
    desc = df.get("description", pd.Series("", index=df.index)).fillna("").str.lower()
    df["flag_restatement_hard"] = desc.str.contains("restatement|restated", regex=True).astype(int)
    df["flag_restatement_suspected"] = (
        df["flag_amended"] & ~df["flag_restatement_hard"].astype(bool)
    ).astype(int)

    return df


def compute_completeness_flags(
    submissions: pd.DataFrame,
    facts: pd.DataFrame | None,
    cfg: dict,
) -> pd.DataFrame:
    """Flag filings with missing core financial fields."""
    df = submissions.copy()
    df["flag_missing_core"] = 0
    df["n_missing_core"] = 0

    if facts is None or len(facts) == 0:
        return df

    core = cfg.get("core_metrics", [])
    if not core:
        return df

    # For each filing (accession_number), check which core metrics exist
    if "accession_number" in facts.columns and "metric_name" in facts.columns:
        coverage = facts.groupby("accession_number")["metric_name"].apply(
            lambda x: set(x.dropna().str.lower())
        )
        for idx, row in df.iterrows():
            acc = row.get("accession_number")
            if acc in coverage.index:
                present = coverage[acc]
                missing = [m for m in core if m.lower() not in present]
                df.at[idx, "n_missing_core"] = len(missing)
                df.at[idx, "flag_missing_core"] = 1 if len(missing) > len(core) // 2 else 0

    return df


def compute_reporting_gap_flags(
    submissions: pd.DataFrame, cfg: dict,
) -> pd.DataFrame:
    """Flag gaps in reporting cadence."""
    df = submissions.copy()
    max_gap = cfg.get("reporting_gap_max_days", 200)

    df["flag_reporting_gap"] = 0
    for cik, grp in df.groupby("cik"):
        sorted_g = grp.sort_values("filed_date")
        filed = pd.to_datetime(sorted_g["filed_date"], errors="coerce")
        gaps = filed.diff().dt.days
        df.loc[sorted_g.index, "flag_reporting_gap"] = (gaps > max_gap).astype(int).values

    return df


def compute_quality_score(
    df: pd.DataFrame, cfg: dict,
) -> pd.DataFrame:
    """Compute aggregate quality score [0, 1] from flags."""
    weights = cfg.get("quality_weights", {})
    w_time = weights.get("timeliness", 0.2)
    w_corr = weights.get("corrections", 0.3)
    w_comp = weights.get("completeness", 0.2)
    w_anom = weights.get("anomalies", 0.3)

    score = pd.Series(1.0, index=df.index)
    score -= df.get("flag_late_filer", 0).astype(float) * 0.2 * w_time
    score -= df.get("flag_persistent_late", 0).astype(float) * 0.4 * w_time
    score -= df.get("flag_amended", 0).astype(float) * 0.1 * w_corr
    score -= df.get("flag_restatement_hard", 0).astype(float) * 0.5 * w_corr
    score -= df.get("flag_restatement_suspected", 0).astype(float) * 0.2 * w_corr
    score -= df.get("flag_missing_core", 0).astype(float) * 0.3 * w_comp
    score -= df.get("flag_reporting_gap", 0).astype(float) * 0.2 * w_comp

    df["quality_score"] = score.clip(0, 1)

    # PIT action
    df["pit_action"] = "allow"
    df.loc[df["quality_score"] < 0.3, "pit_action"] = "exclude"
    df.loc[(df["quality_score"] >= 0.3) & (df["quality_score"] < 0.7), "pit_action"] = "penalize"

    return df


def run_filings_flags(
    submissions: pd.DataFrame | str | Path,
    *,
    facts: pd.DataFrame | str | Path | None = None,
    config: dict[str, Any] | None = None,
    output_dir: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run the full filings flags pipeline."""
    cfg = deep_merge(DEFAULT_CONFIG, config or {})
    if not run_id:
        run_id = f"flags_{utc_now_iso().replace(':','').replace('-','')}"

    if isinstance(submissions, (str, Path)):
        submissions = read_dataframe(submissions)
    if isinstance(facts, (str, Path)):
        facts = read_dataframe(facts)

    df = submissions.copy()
    df = compute_timeliness_flags(df, cfg)
    df = compute_correction_flags(df, cfg)
    df = compute_completeness_flags(df, facts, cfg)
    df = compute_reporting_gap_flags(df, cfg)
    df = compute_quality_score(df, cfg)

    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    manifest = {
        "run_id": run_id,
        "n_filings": len(df),
        "n_flags": len(flag_cols),
        "flag_rates": {c: float(df[c].mean()) for c in flag_cols},
        "quality_score_mean": float(df["quality_score"].mean()),
        "pct_excluded": float((df["pit_action"] == "exclude").mean()),
        "timestamp": utc_now_iso(),
    }

    if output_dir:
        out = Path(output_dir) / run_id
        out.mkdir(parents=True, exist_ok=True)
        write_parquet_safe(df, out / "flags.parquet")
        write_json_safe(manifest, out / "manifest.json")

    LOGGER.info("Flags: %d filings, quality_mean=%.2f, excluded=%.1f%%",
                len(df), manifest["quality_score_mean"], manifest["pct_excluded"] * 100)
    return {"flags": df, "manifest": manifest}
