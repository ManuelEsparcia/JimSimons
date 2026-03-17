from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "pit_config_version": "1.0.0",
    "storage": {
        "output_root": "data/edgar/pit",
        "allow_csv_fallback": True,
        "compression": "snappy",
    },
    "calendar": {
        "freq": "D",
        "asof_time_of_day": "23:59:59",
        "timezone": "UTC",
    },
    "selection": {
        "filing_authority_rank": {
            "10-K": 100,
            "10-Q": 90,
            "20-F": 100,
            "40-F": 100,
            "8-K": 60,
            "6-K": 50,
            "10-K/A": 95,
            "10-Q/A": 85,
            "20-F/A": 95,
            "40-F/A": 95,
        },
        "amendment_priority": {
            "original": 0,
            "technical": 1,
            "non_substantive": 1,
            "substantive": 3,
            "restatement": 4,
        },
        "prefer_higher_fact_quality": True,
        "max_same_tuple_tolerance": 1,
        "full_panel": True,
    },
    "quality": {
        "severity_rank": {
            "NONE": 0,
            "INFO": 1,
            "WARN": 2,
            "MEDIUM": 2,
            "HIGH": 3,
            "CRITICAL": 4,
            "FAIL": 4,
        },
        "severity_action": {
            "NONE": "allow",
            "INFO": "allow",
            "WARN": "penalize",
            "MEDIUM": "penalize",
            "HIGH": "exclude",
            "CRITICAL": "exclude",
            "FAIL": "exclude",
        },
        "severity_weight": {
            "NONE": 1.0,
            "INFO": 1.0,
            "WARN": 0.85,
            "MEDIUM": 0.75,
            "HIGH": 0.0,
            "CRITICAL": 0.0,
            "FAIL": 0.0,
        },
    },
    "staleness": {
        "default": {
            "max_staleness_days": 180,
            "stale_action": "exclude",
            "stale_quality_weight": 0.0,
        },
        "by_metric": {
            "Revenue": {"max_staleness_days": 140, "stale_action": "exclude", "stale_quality_weight": 0.0},
            "NetIncome": {"max_staleness_days": 140, "stale_action": "exclude", "stale_quality_weight": 0.0},
            "TotalAssets": {"max_staleness_days": 220, "stale_action": "penalize", "stale_quality_weight": 0.5},
            "SharesOutstanding": {"max_staleness_days": 120, "stale_action": "penalize", "stale_quality_weight": 0.6},
        },
        "age_buckets": [30, 60, 90, 120, 180, 270, 365],
    },
    "metrics": {
        "include": None,
        "exclude": [],
    },
    "flags": {
        "enabled": True,
    },
    "validation": {
        "abort_on_input_contract_failure": True,
        "coverage_warn_threshold": 0.60,
        "coverage_fail_threshold": 0.30,
        "abort_on_leakage": True,
        "abort_on_duplicate_output": True,
        "abort_on_identity_ambiguity": False,
    },
}

REQUIRED_FACT_COLS = [
    "cik",
    "metric_name",
    "value",
    "period_end",
    "filed_date",
    "acceptance_datetime",
]
OPTIONAL_FACT_ALIASES = {
    "source_accession_number": ["source_accession_number", "accession_number"],
    "source_form_type": ["source_form_type", "form_type"],
    "fact_quality_score": ["fact_quality_score", "quality_score"],
    "amendment_type": ["amendment_type", "amendment_class", "amendment_flag"],
    "filing_authority": ["filing_authority", "form_type"],
    "metric_unit": ["metric_unit", "unit", "unit_ref"],
    "mapping_version": ["mapping_version"],
}

REQUIRED_MAPPING_COLS = ["symbol", "cik"]
OPTIONAL_MAPPING_ALIASES = {
    "effective_from": ["effective_from", "valid_from", "start_date"],
    "effective_to": ["effective_to", "valid_to", "end_date"],
    "is_active": ["is_active", "active_flag"],
    "confidence_score": ["confidence_score"],
    "mapping_version": ["mapping_version"],
    "resolution_status": ["resolution_status"],
}

OPTIONAL_FLAG_ALIASES = {
    "source_accession_number": ["source_accession_number", "accession_number"],
    "cik": ["cik"],
    "flag_severity": ["flag_severity", "severity", "max_severity"],
    "pit_action": ["pit_action"],
    "quality_weight": ["quality_weight", "weight"],
    "source_form_type": ["source_form_type", "form_type"],
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_convert(None)


def _to_cik(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return ""
    return digits.zfill(10)


def _to_symbol(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def _parse_timestamp_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)


def _parse_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=False, errors="coerce").dt.normalize()


def _deep_merge(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(a)
    for k, v in b.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def _config_hash(cfg: Mapping[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if not path:
        return cfg
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config path not found: {path}")
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML config files.")
        with p.open("r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
    elif p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            user_cfg = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {p.suffix}")
    if not isinstance(user_cfg, Mapping):
        raise ValueError("Config root must be a mapping/object.")
    return _deep_merge(cfg, user_cfg)


def _find_files(path: Path, exts: Sequence[str]) -> List[Path]:
    if path.is_file():
        return [path]
    files: List[Path] = []
    for ext in exts:
        files.extend(sorted(path.rglob(f"*{ext}")))
    return files


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    if p.is_dir():
        parquet_files = _find_files(p, [".parquet"])
        if parquet_files:
            return pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        csv_files = _find_files(p, [".csv"])
        if csv_files:
            return pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        json_files = _find_files(p, [".json"])
        if json_files:
            return pd.concat([pd.read_json(f, lines=f.suffix == ".jsonl") for f in json_files], ignore_index=True)
        raise FileNotFoundError(f"No readable tabular files found in directory: {path}")

    suf = p.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(p)
    if suf == ".csv":
        return pd.read_csv(p)
    if suf in {".json", ".jsonl"}:
        return pd.read_json(p, lines=suf == ".jsonl")
    raise ValueError(f"Unsupported table format: {p.suffix}")


def write_table(df: pd.DataFrame, path: Path, allow_csv_fallback: bool = True, compression: str = "snappy") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False, compression=compression)
        return path
    except Exception as exc:
        if not allow_csv_fallback:
            raise RuntimeError(
                "Parquet write failed and CSV fallback is disabled. Install pyarrow or fastparquet."
            ) from exc
        fallback = path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        return fallback


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _first_existing_col(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    for col in aliases:
        if col in df.columns:
            return col
    return None


def normalize_columns(df: pd.DataFrame, alias_map: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    out = df.copy()
    for canonical, aliases in alias_map.items():
        hit = _first_existing_col(out, aliases)
        if hit and hit != canonical:
            out[canonical] = out[hit]
    return out


def _date_to_asof_timestamp(value: pd.Timestamp, hhmmss: str) -> pd.Timestamp:
    hh, mm, ss = [int(x) for x in hhmmss.split(":")]
    return pd.Timestamp(datetime.combine(value.date(), time(hh, mm, ss)))


def resolve_asof_grid(
    start_date: Optional[str],
    end_date: Optional[str],
    explicit_asofs: Optional[Sequence[str]],
    cfg: Mapping[str, Any],
) -> List[pd.Timestamp]:
    cal = cfg["calendar"]
    hhmmss = cal.get("asof_time_of_day", "23:59:59")
    if explicit_asofs:
        stamps = pd.to_datetime(list(explicit_asofs), utc=True, errors="coerce")
        if pd.isna(stamps).any():
            raise ValueError("One or more explicit asof timestamps could not be parsed.")
        return [pd.Timestamp(ts).tz_convert(None) if getattr(ts, "tzinfo", None) else pd.Timestamp(ts) for ts in stamps]

    if not start_date or not end_date:
        raise ValueError("Provide either explicit asofs or both start_date and end_date.")
    d0 = pd.Timestamp(start_date).normalize()
    d1 = pd.Timestamp(end_date).normalize()
    if d1 < d0:
        raise ValueError("end_date must be >= start_date")
    dates = pd.date_range(d0, d1, freq=cal.get("freq", "D"))
    return [_date_to_asof_timestamp(pd.Timestamp(d), hhmmss) for d in dates]


def _age_bucket(days: Optional[float], buckets: Sequence[int]) -> str:
    if days is None or pd.isna(days):
        return "unknown"
    d = float(days)
    for b in buckets:
        if d <= b:
            return f"<= {b}d"
    return f"> {buckets[-1]}d" if buckets else "aged"


# -----------------------------------------------------------------------------
# Input preparation
# -----------------------------------------------------------------------------
def prepare_facts(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df, OPTIONAL_FACT_ALIASES)
    missing = [c for c in REQUIRED_FACT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"parsed_facts missing required columns: {missing}")

    out = df.copy()
    out["cik"] = out["cik"].map(_to_cik)
    out["metric_name"] = out["metric_name"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["period_end"] = _parse_date_series(out["period_end"])
    out["filed_date"] = _parse_date_series(out["filed_date"])
    out["acceptance_datetime"] = _parse_timestamp_series(out["acceptance_datetime"])

    # Optional columns / defaults
    if "source_accession_number" not in out.columns:
        out["source_accession_number"] = pd.NA
    if "source_form_type" not in out.columns:
        out["source_form_type"] = pd.NA
    if "fact_quality_score" not in out.columns:
        out["fact_quality_score"] = 1.0
    out["fact_quality_score"] = pd.to_numeric(out["fact_quality_score"], errors="coerce").fillna(1.0)
    if "amendment_type" not in out.columns:
        out["amendment_type"] = "original"
    out["amendment_type"] = out["amendment_type"].fillna("original").astype(str)
    if "filing_authority" not in out.columns:
        out["filing_authority"] = out["source_form_type"].fillna("")
    if "metric_unit" not in out.columns:
        out["metric_unit"] = pd.NA
    if "mapping_version" not in out.columns:
        out["mapping_version"] = pd.NA

    # Contract checks
    bad_temporal = out[
        out["period_end"].notna()
        & out["filed_date"].notna()
        & (out["period_end"] > out["filed_date"])
    ]
    if not bad_temporal.empty:
        raise ValueError("Input facts violate period_end <= filed_date for at least one row.")
    bad_temporal_2 = out[
        out["filed_date"].notna()
        & out["acceptance_datetime"].notna()
        & (out["filed_date"] > out["acceptance_datetime"].dt.normalize())
    ]
    if not bad_temporal_2.empty:
        raise ValueError("Input facts violate filed_date <= acceptance_datetime for at least one row.")

    out = out[out["cik"] != ""].copy()
    out = out[out["metric_name"].notna()].copy()
    out["_source_row_id"] = np.arange(len(out), dtype=np.int64)
    return out


def prepare_mapping(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df, OPTIONAL_MAPPING_ALIASES)
    missing = [c for c in REQUIRED_MAPPING_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"ticker_cik mapping missing required columns: {missing}")

    out = df.copy()
    out["symbol"] = out["symbol"].map(_to_symbol)
    out["cik"] = out["cik"].map(_to_cik)
    if "effective_from" not in out.columns:
        out["effective_from"] = pd.Timestamp("1900-01-01")
    out["effective_from"] = _parse_date_series(out["effective_from"]).fillna(pd.Timestamp("1900-01-01"))
    if "effective_to" not in out.columns:
        out["effective_to"] = pd.NaT
    out["effective_to"] = _parse_date_series(out["effective_to"])
    if "is_active" not in out.columns:
        out["is_active"] = True
    out["is_active"] = out["is_active"].fillna(True).astype(bool)
    if "confidence_score" not in out.columns:
        out["confidence_score"] = 1.0
    out["confidence_score"] = pd.to_numeric(out["confidence_score"], errors="coerce").fillna(1.0)
    if "mapping_version" not in out.columns:
        out["mapping_version"] = pd.NA
    if "resolution_status" not in out.columns:
        out["resolution_status"] = "exact"

    invalid = out[(out["effective_to"].notna()) & (out["effective_to"] < out["effective_from"])]
    if not invalid.empty:
        raise ValueError("Mapping contains rows with effective_to < effective_from.")
    out = out[(out["symbol"] != "") & (out["cik"] != "")].copy()
    out["_map_row_id"] = np.arange(len(out), dtype=np.int64)
    return out


def prepare_flags(df: Optional[pd.DataFrame], cfg: Mapping[str, Any]) -> pd.DataFrame:
    if df is None or df.empty or not cfg.get("flags", {}).get("enabled", True):
        return pd.DataFrame(
            columns=[
                "cik",
                "source_accession_number",
                "flag_severity",
                "pit_action",
                "quality_weight",
                "source_form_type",
            ]
        )
    out = normalize_columns(df, OPTIONAL_FLAG_ALIASES).copy()
    if "cik" in out.columns:
        out["cik"] = out["cik"].map(_to_cik)
    else:
        out["cik"] = ""
    if "source_accession_number" not in out.columns:
        out["source_accession_number"] = pd.NA
    sev_rank = cfg["quality"]["severity_rank"]
    sev_action = cfg["quality"]["severity_action"]
    sev_weight = cfg["quality"]["severity_weight"]
    out["flag_severity"] = out.get("flag_severity", "NONE").fillna("NONE").astype(str).str.upper()
    out["pit_action"] = out.get("pit_action", pd.Series(index=out.index, dtype=object))
    out["pit_action"] = out["pit_action"].where(out["pit_action"].notna(), out["flag_severity"].map(sev_action)).fillna("allow")
    out["quality_weight"] = pd.to_numeric(out.get("quality_weight", np.nan), errors="coerce")
    out["quality_weight"] = out["quality_weight"].where(out["quality_weight"].notna(), out["flag_severity"].map(sev_weight)).fillna(1.0)
    out["flag_rank"] = out["flag_severity"].map(sev_rank).fillna(0).astype(int)
    if "source_form_type" not in out.columns:
        out["source_form_type"] = pd.NA
    # Keep most severe row per (cik, accession)
    out = out.sort_values(["cik", "source_accession_number", "flag_rank"], ascending=[True, True, False])
    out = out.drop_duplicates(subset=["cik", "source_accession_number"], keep="first")
    return out


# -----------------------------------------------------------------------------
# Core PIT logic
# -----------------------------------------------------------------------------
def active_mapping_for_asof(mapping: pd.DataFrame, asof: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = asof.normalize()
    mask = (
        mapping["is_active"]
        & (mapping["effective_from"] <= d)
        & (mapping["effective_to"].isna() | (mapping["effective_to"] > d))
    )
    active = mapping.loc[mask].copy()
    if active.empty:
        return active, pd.DataFrame(columns=["asof", "issue_type", "symbol", "cik", "n_active", "details", "severity"])

    # Conflicts on symbol -> multiple active CIKs
    sym_counts = active.groupby("symbol")["cik"].nunique().reset_index(name="n_cik")
    bad_symbols = sym_counts[sym_counts["n_cik"] > 1]["symbol"]
    issues: List[Dict[str, Any]] = []
    if not bad_symbols.empty:
        for sym in bad_symbols.tolist():
            sub = active[active["symbol"] == sym]
            issues.append(
                {
                    "asof": asof,
                    "issue_type": "identity_ambiguity",
                    "symbol": sym,
                    "cik": pd.NA,
                    "n_active": int(sub["cik"].nunique()),
                    "details": f"Multiple active CIKs for symbol {sym}: {sorted(sub['cik'].unique().tolist())}",
                    "severity": "HIGH",
                }
            )
        active = active[~active["symbol"].isin(bad_symbols)].copy()

    # Conflicts on cik -> multiple active symbols are allowed only if identical symbol aliases are absent.
    cik_counts = active.groupby("cik")["symbol"].nunique().reset_index(name="n_symbol")
    bad_ciks = cik_counts[cik_counts["n_symbol"] > 1]["cik"]
    if not bad_ciks.empty:
        for cik in bad_ciks.tolist():
            sub = active[active["cik"] == cik]
            issues.append(
                {
                    "asof": asof,
                    "issue_type": "identity_ambiguity",
                    "symbol": pd.NA,
                    "cik": cik,
                    "n_active": int(sub["symbol"].nunique()),
                    "details": f"Multiple active symbols for cik {cik}: {sorted(sub['symbol'].unique().tolist())}",
                    "severity": "HIGH",
                }
            )
        active = active[~active["cik"].isin(bad_ciks)].copy()

    active = active.sort_values(["symbol", "confidence_score", "effective_from", "_map_row_id"], ascending=[True, False, False, True])
    active = active.drop_duplicates(subset=["symbol", "cik"], keep="first")
    issues_df = pd.DataFrame(issues)
    return active, issues_df


def build_metric_universe(facts: pd.DataFrame, cfg: Mapping[str, Any]) -> List[str]:
    metrics_cfg = cfg.get("metrics", {})
    include = metrics_cfg.get("include")
    exclude = set(metrics_cfg.get("exclude") or [])
    metrics = sorted(set(facts["metric_name"].dropna().astype(str).unique().tolist()))
    if include:
        include_set = set(include)
        metrics = [m for m in metrics if m in include_set]
    metrics = [m for m in metrics if m not in exclude]
    return metrics


def metric_policy(metric_name: str, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    st_cfg = cfg["staleness"]
    specific = (st_cfg.get("by_metric") or {}).get(metric_name)
    return _deep_merge(st_cfg["default"], specific or {})


def _selection_priority_columns(candidates: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    out = candidates.copy()
    form_rank_map = cfg["selection"].get("filing_authority_rank", {})
    amend_rank_map = cfg["selection"].get("amendment_priority", {})
    out["_filing_rank"] = out["source_form_type"].fillna("").astype(str).map(form_rank_map).fillna(0).astype(int)
    out["_amendment_rank"] = out["amendment_type"].fillna("original").astype(str).map(amend_rank_map).fillna(0).astype(int)
    out["_fact_quality"] = pd.to_numeric(out["fact_quality_score"], errors="coerce").fillna(0.0)
    out["_accession_key"] = out["source_accession_number"].fillna("").astype(str)
    return out


def _resolve_selection_conflicts(candidates: pd.DataFrame, asof: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    issues: List[Dict[str, Any]] = []
    if candidates.empty:
        return candidates, pd.DataFrame(columns=["asof", "issue_type", "symbol", "cik", "metric_name", "details", "severity"])

    sel_key_cols = [
        "symbol",
        "cik",
        "metric_name",
        "acceptance_datetime",
        "_filing_rank",
        "_amendment_rank",
        "_fact_quality",
        "filed_date",
        "_accession_key",
    ]
    dup_mask = candidates.duplicated(subset=sel_key_cols, keep=False)
    dup_df = candidates[dup_mask].copy()
    if dup_df.empty:
        return candidates, pd.DataFrame(columns=["asof", "issue_type", "symbol", "cik", "metric_name", "details", "severity"])

    by_key = dup_df.groupby(["symbol", "cik", "metric_name"], dropna=False)
    bad_groups: List[Tuple[str, str, str]] = []
    for (sym, cik, metric), sub in by_key:
        # If same selection tuple but economically different values / lineages, surface as conflict.
        same_tuple_groups = sub.groupby(sel_key_cols, dropna=False)
        group_conflict = False
        for _, same in same_tuple_groups:
            values = set(pd.to_numeric(same["value"], errors="coerce").round(12).dropna().tolist())
            accessions = set(same["source_accession_number"].fillna("<NA>").astype(str).tolist())
            if len(values) > 1 or len(accessions) > 1 or len(same) > 1:
                group_conflict = True
        if group_conflict:
            bad_groups.append((sym, cik, metric))
            issues.append(
                {
                    "asof": asof,
                    "issue_type": "selection_conflict",
                    "symbol": sym,
                    "cik": cik,
                    "metric_name": metric,
                    "details": "Multiple equally ranked observable candidates remain after lexicographic arbitration.",
                    "severity": "HIGH",
                }
            )
    if bad_groups:
        bad_idx = pd.MultiIndex.from_tuples(bad_groups, names=["symbol", "cik", "metric_name"])
        cand_idx = pd.MultiIndex.from_frame(candidates[["symbol", "cik", "metric_name"]])
        keep_mask = ~cand_idx.isin(bad_idx)
        candidates = candidates.loc[keep_mask].copy()
    return candidates, pd.DataFrame(issues)


def join_flags(candidates: pd.DataFrame, flags: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    if flags.empty:
        out = candidates.copy()
        out["flag_severity"] = "NONE"
        out["flag_rank"] = 0
        out["_quality_action"] = "allow"
        out["_quality_weight"] = 1.0
        return out

    on_cols = ["cik", "source_accession_number"]
    merged = candidates.merge(
        flags[["cik", "source_accession_number", "flag_severity", "flag_rank", "pit_action", "quality_weight"]],
        on=on_cols,
        how="left",
        suffixes=("", "_flag"),
    )
    sev_action = cfg["quality"]["severity_action"]
    sev_weight = cfg["quality"]["severity_weight"]
    merged["flag_severity"] = merged["flag_severity"].fillna("NONE").astype(str).str.upper()
    merged["flag_rank"] = pd.to_numeric(merged["flag_rank"], errors="coerce").fillna(0).astype(int)
    merged["_quality_action"] = merged["pit_action"].where(merged["pit_action"].notna(), merged["flag_severity"].map(sev_action)).fillna("allow")
    merged["_quality_weight"] = pd.to_numeric(merged["quality_weight"], errors="coerce")
    merged["_quality_weight"] = merged["_quality_weight"].where(merged["_quality_weight"].notna(), merged["flag_severity"].map(sev_weight)).fillna(1.0)
    return merged


def select_candidates_for_asof(
    facts: pd.DataFrame,
    active_map: pd.DataFrame,
    flags: pd.DataFrame,
    asof: pd.Timestamp,
    cfg: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if active_map.empty:
        empty = pd.DataFrame(columns=["symbol", "cik", "metric_name"])
        issues = pd.DataFrame(columns=["asof", "issue_type", "symbol", "cik", "metric_name", "details", "severity"])
        return empty, issues

    eligible = facts[facts["acceptance_datetime"] <= asof].copy()
    if eligible.empty:
        empty = pd.DataFrame(columns=["symbol", "cik", "metric_name"])
        issues = pd.DataFrame(columns=["asof", "issue_type", "symbol", "cik", "metric_name", "details", "severity"])
        return empty, issues

    # PIT-safe symbol resolution via active CIK mapping.
    merged = eligible.merge(
        active_map[["symbol", "cik", "mapping_version", "confidence_score", "resolution_status"]],
        on="cik",
        how="inner",
        suffixes=("", "_map"),
    )
    if "mapping_version_map" in merged.columns:
        merged["mapping_version"] = merged["mapping_version_map"].where(
            merged["mapping_version_map"].notna(), merged.get("mapping_version")
        )
        merged = merged.drop(columns=["mapping_version_map"])
    if merged.empty:
        empty = pd.DataFrame(columns=["symbol", "cik", "metric_name"])
        issues = pd.DataFrame(columns=["asof", "issue_type", "symbol", "cik", "metric_name", "details", "severity"])
        return empty, issues

    merged = _selection_priority_columns(merged, cfg)
    merged = join_flags(merged, flags, cfg)

    sort_cols = [
        "symbol",
        "metric_name",
        "acceptance_datetime",
        "_filing_rank",
        "_amendment_rank",
        "_fact_quality",
        "filed_date",
        "_accession_key",
        "_source_row_id",
    ]
    merged = merged.sort_values(
        sort_cols,
        ascending=[True, True, False, False, False, False, False, False, True],
    )

    # Expose unresolved ties before dropping to first.
    merged, tie_issues = _resolve_selection_conflicts(merged, asof)
    selected = merged.drop_duplicates(subset=["symbol", "metric_name"], keep="first").copy()
    selected["asof"] = asof
    selected["selection_reason"] = np.where(
        selected["acceptance_datetime"].dt.normalize() < asof.normalize(),
        "latest_observable_ffill",
        "latest_observable_same_day",
    )
    selected["ffill_applied"] = selected["acceptance_datetime"].dt.normalize() < asof.normalize()
    return selected, tie_issues


def apply_staleness_and_quality(selected: pd.DataFrame, asof: pd.Timestamp, cfg: Mapping[str, Any]) -> pd.DataFrame:
    if selected.empty:
        return selected
    out = selected.copy()
    out["staleness_days"] = ((asof - out["acceptance_datetime"]).dt.total_seconds() / 86400.0).astype(float)
    out["quality_weight"] = pd.to_numeric(out["_quality_weight"], errors="coerce").fillna(1.0)
    out["pit_action"] = out["_quality_action"].fillna("allow").astype(str)
    out["is_stale"] = False
    out["max_staleness_days"] = np.nan
    out["age_bucket"] = "unknown"

    actions: List[str] = []
    weights: List[float] = []
    is_stale_list: List[bool] = []
    max_days_list: List[float] = []
    age_buckets: List[str] = []
    values: List[Any] = []

    age_bkt_cfg = cfg["staleness"].get("age_buckets", [])
    for _, row in out.iterrows():
        pol = metric_policy(str(row["metric_name"]), cfg)
        max_days = float(pol.get("max_staleness_days", 180))
        stale_action = str(pol.get("stale_action", "exclude"))
        stale_w = float(pol.get("stale_quality_weight", 0.0))
        staleness_days = float(row["staleness_days"]) if not pd.isna(row["staleness_days"]) else math.inf
        is_stale = staleness_days > max_days
        action = str(row["pit_action"])
        weight = float(row["quality_weight"])
        value = row["value"]
        if is_stale:
            if stale_action == "exclude":
                action = "stale"
                weight = 0.0
                value = np.nan
            elif stale_action == "penalize":
                action = "stale"
                weight = min(weight, stale_w)
            else:
                action = "allow"
                weight = min(weight, stale_w if stale_w > 0 else weight)
        else:
            if action == "exclude":
                value = np.nan
            elif action == "penalize":
                weight = min(weight, 1.0)
            else:
                weight = min(max(weight, 0.0), 1.0)
        if action == "exclude":
            value = np.nan
            weight = 0.0
        actions.append(action)
        weights.append(weight)
        is_stale_list.append(bool(is_stale))
        max_days_list.append(max_days)
        age_buckets.append(_age_bucket(staleness_days, age_bkt_cfg))
        values.append(value)

    out["pit_action"] = actions
    out["quality_weight"] = weights
    out["is_stale"] = is_stale_list
    out["max_staleness_days"] = max_days_list
    out["age_bucket"] = age_buckets
    out["metric_value_pit"] = values
    return out


def build_full_panel(
    asof: pd.Timestamp,
    active_map: pd.DataFrame,
    metrics: Sequence[str],
    selected: pd.DataFrame,
    issues: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    active_symbols = active_map[["symbol", "cik", "mapping_version"]].drop_duplicates().copy()
    if not metrics:
        raise ValueError("Metric universe is empty after include/exclude filters.")

    if cfg["selection"].get("full_panel", True):
        base = active_symbols.assign(_k=1).merge(pd.DataFrame({"metric_name": list(metrics), "_k": 1}), on="_k", how="inner").drop(columns="_k")
    else:
        if selected.empty:
            base = active_symbols.iloc[0:0].copy()
            base["metric_name"] = pd.Series(dtype=object)
        else:
            base = selected[["symbol", "cik", "metric_name"]].drop_duplicates().merge(
                active_symbols[["symbol", "cik", "mapping_version"]], on=["symbol", "cik"], how="left"
            )
    base["asof"] = asof

    panel = base.merge(
        selected[
            [
                "asof",
                "symbol",
                "cik",
                "metric_name",
                "metric_value_pit",
                "period_end",
                "filed_date",
                "acceptance_datetime",
                "source_accession_number",
                "source_form_type",
                "selection_reason",
                "ffill_applied",
                "staleness_days",
                "pit_action",
                "quality_weight",
                "is_stale",
                "age_bucket",
                "flag_severity",
                "mapping_version",
            ]
        ],
        on=["asof", "symbol", "cik", "metric_name", "mapping_version"],
        how="left",
    )

    panel = panel.rename(
        columns={
            "period_end": "source_period_end",
            "filed_date": "source_filed_date",
            "acceptance_datetime": "source_acceptance_ts",
        }
    )

    # Default statuses for rows with no candidate.
    no_candidate = panel["source_acceptance_ts"].isna()
    panel.loc[no_candidate, "pit_action"] = "exclude"
    panel.loc[no_candidate, "quality_weight"] = 0.0
    panel.loc[no_candidate, "is_stale"] = False
    panel.loc[no_candidate, "selection_reason"] = "no_observable_fact"
    panel.loc[no_candidate, "ffill_applied"] = False
    panel.loc[no_candidate, "age_bucket"] = "missing"
    panel.loc[no_candidate, "flag_severity"] = "NONE"
    panel.loc[no_candidate, "metric_value_pit"] = np.nan
    panel["run_id"] = cfg.get("run_id")
    panel["pit_config_version"] = cfg.get("pit_config_version")

    # Materialize identity-ambiguity rows separately in issues; do not backfill panel with ambiguous symbols.
    return panel


# -----------------------------------------------------------------------------
# Coverage, issues, validation
# -----------------------------------------------------------------------------
def summarize_coverage(panel: pd.DataFrame, active_map: pd.DataFrame, metrics: Sequence[str], asof: pd.Timestamp, cfg: Mapping[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    total_symbols = int(active_map["symbol"].nunique())
    total_expected_all = int(total_symbols * len(metrics))
    usable_mask_all = panel["metric_value_pit"].notna() & panel["pit_action"].isin(["allow", "penalize", "stale"])
    usable_all = int(usable_mask_all.sum())
    rows.append(
        {
            "asof": asof,
            "metric_name": "__ALL__",
            "n_symbols": total_symbols,
            "n_expected": total_expected_all,
            "n_usable": usable_all,
            "coverage_ratio": usable_all / total_expected_all if total_expected_all else np.nan,
            "n_no_candidate": int((panel["selection_reason"] == "no_observable_fact").sum()),
            "n_quality_excluded": int(((panel["selection_reason"] != "no_observable_fact") & (panel["pit_action"] == "exclude")).sum()),
            "n_stale": int(panel["is_stale"].fillna(False).sum()),
            "n_penalized": int((panel["pit_action"] == "penalize").sum()),
            "n_ffill": int(panel["ffill_applied"].fillna(False).sum()),
        }
    )

    for metric in metrics:
        sub = panel[panel["metric_name"] == metric]
        expected = len(sub)
        usable_mask = sub["metric_value_pit"].notna() & sub["pit_action"].isin(["allow", "penalize", "stale"])
        usable = int(usable_mask.sum())
        rows.append(
            {
                "asof": asof,
                "metric_name": metric,
                "n_symbols": total_symbols,
                "n_expected": expected,
                "n_usable": usable,
                "coverage_ratio": usable / expected if expected else np.nan,
                "n_no_candidate": int((sub["selection_reason"] == "no_observable_fact").sum()),
                "n_quality_excluded": int(((sub["selection_reason"] != "no_observable_fact") & (sub["pit_action"] == "exclude")).sum()),
                "n_stale": int(sub["is_stale"].fillna(False).sum()),
                "n_penalized": int((sub["pit_action"] == "penalize").sum()),
                "n_ffill": int(sub["ffill_applied"].fillna(False).sum()),
            }
        )
    cov = pd.DataFrame(rows)
    warn_thr = float(cfg["validation"].get("coverage_warn_threshold", 0.60))
    fail_thr = float(cfg["validation"].get("coverage_fail_threshold", 0.30))
    cov["coverage_severity"] = np.where(
        cov["coverage_ratio"] < fail_thr,
        "FAIL",
        np.where(cov["coverage_ratio"] < warn_thr, "WARN", "PASS"),
    )
    return cov


def run_validations(panel: pd.DataFrame, issues: pd.DataFrame, coverage: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    findings: List[Dict[str, Any]] = []

    leakage = panel[
        panel["source_acceptance_ts"].notna() & (panel["source_acceptance_ts"] > panel["asof"])
    ]
    if not leakage.empty:
        findings.append(
            {
                "asof": pd.NaT,
                "issue_type": "temporal_leakage",
                "symbol": pd.NA,
                "cik": pd.NA,
                "metric_name": pd.NA,
                "details": f"Detected {len(leakage)} rows with source_acceptance_ts > asof.",
                "severity": "CRITICAL",
            }
        )

    dups = panel[panel.duplicated(subset=["asof", "symbol", "metric_name"], keep=False)]
    if not dups.empty:
        findings.append(
            {
                "asof": pd.NaT,
                "issue_type": "duplicate_output_key",
                "symbol": pd.NA,
                "cik": pd.NA,
                "metric_name": pd.NA,
                "details": f"Detected {len(dups)} duplicate rows on (asof, symbol, metric_name).",
                "severity": "CRITICAL",
            }
        )

    staleness_bad = panel[
        panel["source_acceptance_ts"].notna()
        & ((panel["asof"] - panel["source_acceptance_ts"]).dt.total_seconds() / 86400.0 - panel["staleness_days"]).abs().fillna(0) > 1e-9
    ]
    if not staleness_bad.empty:
        findings.append(
            {
                "asof": pd.NaT,
                "issue_type": "staleness_inconsistency",
                "symbol": pd.NA,
                "cik": pd.NA,
                "metric_name": pd.NA,
                "details": f"Detected {len(staleness_bad)} rows with inconsistent staleness_days.",
                "severity": "HIGH",
            }
        )

    cov_fail = coverage[(coverage["metric_name"] == "__ALL__") & (coverage["coverage_severity"] == "FAIL")]
    if not cov_fail.empty:
        findings.append(
            {
                "asof": cov_fail.iloc[0]["asof"],
                "issue_type": "coverage_collapse",
                "symbol": pd.NA,
                "cik": pd.NA,
                "metric_name": "__ALL__",
                "details": f"Aggregate coverage ratio {cov_fail.iloc[0]['coverage_ratio']:.4f} below fail threshold.",
                "severity": "HIGH",
            }
        )

    all_issues = pd.concat([issues, pd.DataFrame(findings)], ignore_index=True, sort=False)
    return all_issues


def final_gate(issues: pd.DataFrame) -> str:
    if issues.empty:
        return "PASS"
    sev = issues["severity"].fillna("INFO").astype(str).str.upper()
    if sev.isin(["CRITICAL", "FAIL"]).any():
        return "FAIL"
    if sev.isin(["HIGH", "WARN"]).any():
        return "WARN"
    return "PASS"


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def materialize_point_in_time(
    parsed_facts_path: str,
    ticker_cik_mapping_path: str,
    pit_config_path: Optional[str],
    run_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    asofs: Optional[Sequence[str]] = None,
    filings_flags_path: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_config(pit_config_path)
    cfg["run_id"] = run_id
    started_at = _utc_now()

    facts = prepare_facts(read_table(parsed_facts_path))
    mapping = prepare_mapping(read_table(ticker_cik_mapping_path))
    flags = prepare_flags(read_table(filings_flags_path) if filings_flags_path else None, cfg)
    metric_universe = build_metric_universe(facts, cfg)
    asof_grid = resolve_asof_grid(start_date, end_date, asofs, cfg)
    if not asof_grid:
        raise ValueError("As-of grid resolved to empty set.")

    output_root = Path(cfg["storage"]["output_root"])
    allow_csv_fallback = bool(cfg["storage"].get("allow_csv_fallback", True))
    compression = str(cfg["storage"].get("compression", "snappy"))

    pit_parts: List[pd.DataFrame] = []
    coverage_parts: List[pd.DataFrame] = []
    issue_parts: List[pd.DataFrame] = []

    for asof in asof_grid:
        active_map, identity_issues = active_mapping_for_asof(mapping, asof)
        selected, selection_issues = select_candidates_for_asof(facts, active_map, flags, asof, cfg)
        selected = apply_staleness_and_quality(selected, asof, cfg)
        panel = build_full_panel(asof, active_map, metric_universe, selected, selection_issues, cfg)
        coverage = summarize_coverage(panel, active_map, metric_universe, asof, cfg)
        issues = pd.concat([identity_issues, selection_issues], ignore_index=True, sort=False)
        issues = run_validations(panel, issues, coverage, cfg)

        # Persist date partition.
        date_key = pd.Timestamp(asof).date().isoformat()
        part_path = output_root / f"date={date_key}" / f"part-{run_id}.parquet"
        write_table(panel, part_path, allow_csv_fallback=allow_csv_fallback, compression=compression)

        pit_parts.append(panel)
        coverage_parts.append(coverage)
        issue_parts.append(issues)

    pit_df = pd.concat(pit_parts, ignore_index=True) if pit_parts else pd.DataFrame()
    coverage_df = pd.concat(coverage_parts, ignore_index=True) if coverage_parts else pd.DataFrame()
    issues_df = pd.concat(issue_parts, ignore_index=True, sort=False) if issue_parts else pd.DataFrame()

    # Final validations across the full run.
    gate = final_gate(issues_df)
    if gate == "FAIL":
        if cfg["validation"].get("abort_on_leakage", True):
            leak = issues_df[issues_df["issue_type"] == "temporal_leakage"]
            dups = issues_df[issues_df["issue_type"] == "duplicate_output_key"]
            if not leak.empty or not dups.empty:
                # Artifacts are already persisted; raise after writing summary artifacts below.
                pass

    coverage_path = write_table(
        coverage_df,
        output_root / f"pit_coverage_{run_id}.parquet",
        allow_csv_fallback=allow_csv_fallback,
        compression=compression,
    )
    issues_path = write_table(
        issues_df,
        output_root / f"pit_issues_{run_id}.parquet",
        allow_csv_fallback=allow_csv_fallback,
        compression=compression,
    )

    summary = {
        "run_id": run_id,
        "pit_config_version": cfg.get("pit_config_version"),
        "config_hash": _config_hash(cfg),
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "asof_start": min(asof_grid) if asof_grid else None,
        "asof_end": max(asof_grid) if asof_grid else None,
        "n_asofs": len(asof_grid),
        "n_symbols_max": int(max((p["symbol"].nunique() for p in pit_parts), default=0)),
        "n_metrics": len(metric_universe),
        "n_rows_pit": int(len(pit_df)),
        "n_rows_issues": int(len(issues_df)),
        "gate": gate,
        "coverage_ratio_all_mean": float(
            coverage_df.loc[coverage_df["metric_name"] == "__ALL__", "coverage_ratio"].mean()
        )
        if not coverage_df.empty
        else None,
        "pct_penalized": float((pit_df["pit_action"] == "penalize").mean()) if not pit_df.empty else None,
        "pct_excluded": float((pit_df["pit_action"] == "exclude").mean()) if not pit_df.empty else None,
        "pct_stale": float(pit_df["is_stale"].fillna(False).mean()) if not pit_df.empty else None,
        "paths": {
            "output_root": str(output_root),
            "coverage": str(coverage_path),
            "issues": str(issues_path),
        },
    }
    manifest_path = output_root / f"manifest_{run_id}.json"
    write_json(summary, manifest_path)

    # Explicit hard-fail after artifact emission.
    fatal_issue_types: List[str] = []
    if cfg["validation"].get("abort_on_leakage", True):
        fatal_issue_types.append("temporal_leakage")
    if cfg["validation"].get("abort_on_duplicate_output", True):
        fatal_issue_types.append("duplicate_output_key")
    if cfg["validation"].get("abort_on_identity_ambiguity", False):
        fatal_issue_types.append("identity_ambiguity")
    if fatal_issue_types and not issues_df.empty and issues_df["issue_type"].isin(fatal_issue_types).any():
        fatal = issues_df[issues_df["issue_type"].isin(fatal_issue_types)][["issue_type", "details"]].head(10)
        raise RuntimeError(
            "point_in_time materialization produced fatal issues: "
            + "; ".join(f"{r.issue_type}: {r.details}" for r in fatal.itertuples())
        )

    return {
        "pit": pit_df,
        "coverage": coverage_df,
        "issues": issues_df,
        "manifest": summary,
        "manifest_path": manifest_path,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize EDGAR canonical facts into a strict point-in-time panel.")
    parser.add_argument("--parsed-facts-path", required=True)
    parser.add_argument("--ticker-cik-mapping-path", required=True)
    parser.add_argument("--pit-config-path")
    parser.add_argument("--filings-flags-path")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--asofs", nargs="*", help="Explicit asof timestamps or dates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    materialize_point_in_time(
        parsed_facts_path=args.parsed_facts_path,
        ticker_cik_mapping_path=args.ticker_cik_mapping_path,
        pit_config_path=args.pit_config_path,
        run_id=args.run_id,
        start_date=args.start_date,
        end_date=args.end_date,
        asofs=args.asofs,
        filings_flags_path=args.filings_flags_path,
    )


if __name__ == "__main__":
    main()
