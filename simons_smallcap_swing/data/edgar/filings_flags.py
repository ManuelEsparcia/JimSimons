from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "config_version": "1.0.0",
    "output_dir": "data/edgar/flags",
    "strict_parquet": False,
    "persistence": {"prefer_parquet": True},
    "filer_status_fallback": "unknown",
    "deadline_days": {
        "10-K": {"large_accelerated_filer": 60, "accelerated_filer": 75, "non_accelerated_filer": 90, "smaller_reporting_company": 90, "unknown": 90},
        "10-Q": {"large_accelerated_filer": 40, "accelerated_filer": 40, "non_accelerated_filer": 45, "smaller_reporting_company": 45, "unknown": 45},
        "20-F": {"unknown": 120},
        "6-K": {"unknown": 4},
        "8-K": {"unknown": 4},
    },
    "form_family_map": {
        "10-K": ["10-K", "10-K/A"],
        "10-Q": ["10-Q", "10-Q/A"],
        "20-F": ["20-F", "20-F/A"],
        "6-K": ["6-K", "6-K/A"],
        "8-K": ["8-K", "8-K/A"],
    },
    "form_frequency_days": {
        "10-Q": {"min": 65, "max": 125},
        "10-K": {"min": 280, "max": 460},
        "20-F": {"min": 300, "max": 500},
    },
    "persistent_late": {
        "window": 4,
        "threshold": 0.5,
    },
    "core_fields_by_form_family": {
        "10-Q": ["Revenue", "NetIncomeLoss", "Assets", "Liabilities", "StockholdersEquity"],
        "10-K": ["Revenue", "NetIncomeLoss", "Assets", "Liabilities", "StockholdersEquity"],
        "20-F": ["Revenue", "NetIncomeLoss", "Assets", "Liabilities", "StockholdersEquity"],
    },
    "core_field_weights": {
        "Revenue": 1.2,
        "NetIncomeLoss": 1.3,
        "Assets": 1.1,
        "Liabilities": 1.0,
        "StockholdersEquity": 1.0,
    },
    "coverage": {
        "sparse_threshold": 0.6,
    },
    "restatement": {
        "hard_terms": [
            "restatement",
            "non-reliance",
            "material weakness",
            "revision of previously issued",
            "error in previously issued",
        ],
        "suspected_terms": [
            "revision",
            "corrected",
            "correction",
            "amendment",
            "reclass",
        ],
        "material_change_ratio": 0.05,
        "hard_threshold": 1.0,
        "suspected_threshold": 0.5,
        "evidence_weights": {
            "explicit_hard_term": 1.0,
            "amended_sensitive_form": 0.55,
            "material_fact_change": 0.7,
            "suspected_term": 0.35,
            "multiple_accounting_anomalies": 0.35,
        },
    },
    "anomaly": {
        "concepts": ["Revenue", "NetIncomeLoss", "Assets", "Liabilities", "StockholdersEquity"],
        "concept_thresholds": {
            "Revenue": 6.0,
            "NetIncomeLoss": 7.0,
            "Assets": 6.0,
            "Liabilities": 6.0,
            "StockholdersEquity": 6.0,
        },
        "min_history": 4,
        "bs_epsilon": 0.03,
    },
    "score": {
        "weights": {
            "flag_late_filer": 0.7,
            "flag_persistent_late_filer": 1.1,
            "flag_amended": 0.35,
            "flag_restatement_hard": 2.2,
            "flag_restatement_suspected": 1.2,
            "flag_missing_core_fields": 1.1,
            "flag_sparse_fundamental_coverage": 0.8,
            "flag_accounting_anomaly": 1.0,
            "flag_balance_sheet_inconsistency": 1.6,
            "flag_reporting_gap": 0.6,
            "flag_irregular_filing_sequence": 0.5,
        },
        "severity_penalty_lambda": 0.45,
        "exclude_threshold": 0.35,
        "penalize_threshold": 0.72,
        "critical_flags": [
            "flag_restatement_hard",
            "flag_balance_sheet_inconsistency",
        ],
    },
    "pit_policy": {
        "warn_if_any": [
            "flag_late_filer",
            "flag_missing_core_fields",
            "flag_restatement_suspected",
            "flag_accounting_anomaly",
            "flag_reporting_gap",
            "flag_persistent_late_filer",
        ],
        "exclude_if_any": [
            "flag_restatement_hard",
            "flag_balance_sheet_inconsistency",
        ],
    },
}

SUBMISSION_ALIASES: Dict[str, Sequence[str]] = {
    "cik": ["cik", "CIK"],
    "accession_number": ["accession_number", "accessionNo", "accession", "accessionnumber"],
    "form_type": ["form_type", "form", "formType"],
    "filing_date": ["filing_date", "filingDate"],
    "acceptance_datetime": ["acceptance_datetime", "acceptanceDateTime", "accepted", "accepted_datetime"],
    "period_end": ["period_end", "periodOfReport", "period_of_report", "fy_end", "period"],
    "filer_status": ["filer_status", "issuer_filer_status", "public_float_status"],
    "document_text": ["document_text", "filing_text", "text", "description", "primary_doc_description", "items"],
    "acceptance_date": ["acceptance_date"],
    "ticker": ["ticker", "symbol"],
    "sector": ["sector"],
    "market_cap": ["market_cap", "marketcap", "mkt_cap"],
    "liquidity_bucket": ["liquidity_bucket", "liq_bucket"],
}

FACT_ALIASES: Dict[str, Sequence[str]] = {
    "cik": ["cik", "CIK"],
    "accession_number": ["accession_number", "accessionNo", "accession"],
    "period_end": ["period_end", "period", "periodOfReport", "period_of_report", "fy_end"],
    "concept": ["concept", "fact_name", "tag", "taxonomy_concept"],
    "value": ["value", "fact_value", "numeric_value", "val"],
    "observed_date": ["observed_date", "accepted_date", "filing_date", "fact_observed_date", "asof_date"],
    "acceptance_datetime": ["acceptance_datetime", "acceptanceDateTime", "accepted", "accepted_datetime"],
    "form_type": ["form_type", "form", "fact_form_type"],
    "fy": ["fy", "fiscal_year"],
    "fp": ["fp", "fiscal_period", "quarter"],
}


@dataclass
class FailureRecord:
    stage: str
    severity: str
    code: str
    message: str
    row_count: int = 0


class ModuleError(RuntimeError):
    """Raised for fatal module configuration or data errors."""


# -----------------------------
# Generic IO and config helpers
# -----------------------------


def _utc_now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists():
        raise ModuleError(f"Config path does not exist: {p}")
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ModuleError("PyYAML is required to read YAML config files.")
        user_cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    else:
        user_cfg = json.loads(p.read_text(encoding="utf-8"))
    return _deep_merge(cfg, user_cfg)


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in update.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise ModuleError(f"Input path does not exist: {p}")
    if p.is_dir():
        frames: List[pd.DataFrame] = []
        for child in sorted(p.iterdir()):
            if child.suffix.lower() in {".parquet", ".pq", ".csv", ".json", ".jsonl"}:
                frames.append(_read_table(child))
        if not frames:
            raise ModuleError(f"No readable tabular files found under directory: {p}")
        return pd.concat(frames, ignore_index=True, sort=False)
    suffix = p.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(p)
        except Exception as exc:
            raise ModuleError(f"Unable to read parquet file {p}: {exc}") from exc
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix == ".json":
        try:
            return pd.read_json(p)
        except ValueError:
            return pd.json_normalize(json.loads(p.read_text(encoding="utf-8")))
    if suffix == ".jsonl":
        return pd.read_json(p, lines=True)
    raise ModuleError(f"Unsupported input file type: {p}")


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_table(df: pd.DataFrame, path: Path, *, prefer_parquet: bool = True, strict_parquet: bool = False) -> str:
    if prefer_parquet:
        target = path.with_suffix(".parquet") if path.suffix == "" else path
        try:
            df.to_parquet(target, index=False)
            return str(target)
        except Exception as exc:
            if strict_parquet:
                raise ModuleError(
                    f"Parquet persistence failed for {target}. Install pyarrow or fastparquet. Original error: {exc}"
                ) from exc
    csv_target = path.with_suffix(".csv") if path.suffix == "" else path.with_suffix(".csv")
    df.to_csv(csv_target, index=False)
    return str(csv_target)


def _write_json(payload: Mapping[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    return str(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


# -----------------------------
# Normalization and validation
# -----------------------------


def _first_existing(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    colset = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in colset:
            return colset[cand.lower()]
    return None


def _normalize_columns(df: pd.DataFrame, aliases: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    out = df.copy()
    rename: Dict[str, str] = {}
    for canonical, cands in aliases.items():
        src = _first_existing(out.columns.tolist(), cands)
        if src is not None and src != canonical:
            rename[src] = canonical
    out = out.rename(columns=rename)
    return out


def _coerce_datetime(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce", utc=True)
    return out


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _canonicalize_cik(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(10)


def _normalize_accession(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"[^0-9-]", "", regex=True)


def _normalize_submissions(submissions: pd.DataFrame, failures: List[FailureRecord]) -> pd.DataFrame:
    df = _normalize_columns(submissions, SUBMISSION_ALIASES)
    required = ["cik", "accession_number", "form_type", "filing_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        failures.append(FailureRecord("load_submissions", "critical", "MISSING_COLUMNS", f"Missing required columns: {missing}"))
        raise ModuleError(f"Submissions missing required columns: {missing}")

    df = _coerce_datetime(df, ["filing_date", "acceptance_datetime", "acceptance_date", "period_end"])
    df["cik"] = _canonicalize_cik(df["cik"])
    df["accession_number"] = _normalize_accession(df["accession_number"])
    if "acceptance_datetime" not in df.columns:
        if "acceptance_date" in df.columns:
            df["acceptance_datetime"] = df["acceptance_date"]
        else:
            df["acceptance_datetime"] = df["filing_date"]
    if "period_end" not in df.columns:
        df["period_end"] = pd.NaT
    if "filer_status" not in df.columns:
        df["filer_status"] = pd.NA
    if "document_text" not in df.columns:
        df["document_text"] = pd.NA

    bad_dates = df["filing_date"].isna().sum()
    if bad_dates:
        failures.append(FailureRecord("load_submissions", "critical", "BAD_FILING_DATE", "Some filing_date values are not parseable.", int(bad_dates)))
        raise ModuleError("Submissions contain unparseable filing_date values.")

    dup_mask = df.duplicated(subset=["cik", "accession_number"], keep=False)
    if dup_mask.any():
        failures.append(FailureRecord("load_submissions", "warn", "DUPLICATE_FILING_KEYS", "Duplicate (cik, accession_number) detected; keeping latest acceptance_datetime.", int(dup_mask.sum())))
        df = (
            df.sort_values(["cik", "accession_number", "acceptance_datetime", "filing_date"], ascending=[True, True, True, True])
            .drop_duplicates(["cik", "accession_number"], keep="last")
            .reset_index(drop=True)
        )
    return df


def _normalize_facts(facts: pd.DataFrame, failures: List[FailureRecord]) -> pd.DataFrame:
    df = _normalize_columns(facts, FACT_ALIASES)
    required = ["cik", "concept", "value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        failures.append(FailureRecord("load_facts", "critical", "MISSING_COLUMNS", f"Missing required fact columns: {missing}"))
        raise ModuleError(f"Facts missing required columns: {missing}")

    df = _coerce_datetime(df, ["period_end", "observed_date", "acceptance_datetime"])
    df = _coerce_numeric(df, ["value"])
    df["cik"] = _canonicalize_cik(df["cik"])
    if "accession_number" in df.columns:
        df["accession_number"] = _normalize_accession(df["accession_number"].fillna(""))
    else:
        df["accession_number"] = ""
    if "observed_date" not in df.columns:
        if "acceptance_datetime" in df.columns:
            df["observed_date"] = df["acceptance_datetime"]
        else:
            df["observed_date"] = pd.NaT
    if "form_type" not in df.columns:
        df["form_type"] = pd.NA
    if "period_end" not in df.columns:
        df["period_end"] = pd.NaT
    return df


def _apply_pit_filters(submissions: pd.DataFrame, facts: pd.DataFrame, asof: Optional[pd.Timestamp]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if asof is None:
        return submissions.copy(), facts.copy()
    subs = submissions.copy()
    facs = facts.copy()
    subs = subs[subs["filing_date"] <= asof].copy()
    if "acceptance_datetime" in subs.columns:
        acc = subs["acceptance_datetime"].fillna(subs["filing_date"])
        subs = subs[acc <= asof].copy()
    acc_col = facs["acceptance_datetime"] if "acceptance_datetime" in facs.columns else pd.Series(pd.NaT, index=facs.index)
    obs_col = facs["observed_date"].fillna(acc_col)
    far_future = pd.Timestamp("2262-04-11", tz="UTC")
    obs_col = obs_col.fillna(far_future)
    facs = facs[obs_col <= asof].copy()
    return subs.reset_index(drop=True), facs.reset_index(drop=True)


# -----------------------------
# Domain helpers
# -----------------------------


def _reverse_form_family_map(cfg: Mapping[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for family, forms in cfg.get("form_family_map", {}).items():
        for form in forms:
            out[str(form).upper()] = str(family)
    return out


def _classify_form_family(form_type: Any, reverse_map: Mapping[str, str]) -> str:
    form = str(form_type or "").upper().strip()
    if form in reverse_map:
        return reverse_map[form]
    if form.endswith("/A") and form[:-2] in reverse_map:
        return reverse_map[form[:-2]]
    return form.replace("/A", "")


def _infer_deadline_date(row: pd.Series, cfg: Mapping[str, Any]) -> Tuple[pd.Timestamp | pd.NaT, str]:
    period_end = row.get("period_end")
    if pd.isna(period_end):
        return pd.NaT, "no_period_end"
    family = str(row.get("form_family", ""))
    deadlines = cfg.get("deadline_days", {}).get(family, {})
    filer_status = str(row.get("filer_status") or cfg.get("filer_status_fallback", "unknown")).strip().lower().replace(" ", "_")
    if filer_status in deadlines:
        days = deadlines[filer_status]
        fallback_used = "matched"
    else:
        days = deadlines.get("unknown")
        fallback_used = "fallback_unknown"
    if days is None:
        return pd.NaT, "no_deadline_policy"
    return pd.Timestamp(period_end) + pd.Timedelta(days=int(days)), fallback_used


def _safe_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).lower()


def _rolling_flag_mean(values: Sequence[int], window: int) -> List[float]:
    arr = np.asarray(values, dtype=float)
    out: List[float] = []
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        out.append(float(arr[lo : i + 1].mean()))
    return out


def _mad(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


# -----------------------------
# Linkage and period facts shaping
# -----------------------------


def _link_facts_to_filings(submissions: pd.DataFrame, facts: pd.DataFrame, failures: List[FailureRecord]) -> pd.DataFrame:
    """
    Deterministic linkage policy:
    1) exact (cik, accession_number) when facts carry accession_number;
    2) fallback on (cik, period_end, form_family) assigning the latest filing_date <= observed_date,
       otherwise latest filing for same period_end.
    """
    subs = submissions[["cik", "accession_number", "form_family", "filing_date", "period_end"]].copy()
    subs = subs.rename(columns={"accession_number": "submission_accession"})
    facs = facts.copy()
    facs["link_method"] = pd.NA
    facs["linked_accession_number"] = pd.NA

    has_accession = facs["accession_number"].astype(str).str.len().gt(0)
    exact = facs.loc[has_accession].merge(
        subs,
        left_on=["cik", "accession_number"],
        right_on=["cik", "submission_accession"],
        how="left",
        suffixes=("", "_sub"),
    )
    exact["linked_accession_number"] = exact["submission_accession"]
    exact["link_method"] = np.where(exact["submission_accession"].notna(), "exact_accession", pd.NA)

    needs_fallback = exact["submission_accession"].isna()
    unresolved_exact = exact.loc[needs_fallback, facs.columns].copy() if len(exact) else pd.DataFrame(columns=facs.columns)
    direct_matched = exact.loc[~needs_fallback].copy() if len(exact) else pd.DataFrame(columns=list(facs.columns) + list(subs.columns))

    remaining = pd.concat([facs.loc[~has_accession].copy(), unresolved_exact], ignore_index=True, sort=False)
    if remaining.empty:
        linked = direct_matched.copy()
    else:
        fallback_candidates = submissions[["cik", "accession_number", "period_end", "form_family", "filing_date"]].copy()
        merged = remaining.merge(
            fallback_candidates,
            on=["cik", "period_end"],
            how="left",
            suffixes=("", "_sub"),
        )
        if "form_type" in merged.columns:
            fact_form_family = merged["form_type"].fillna("")
            ff = fact_form_family.where(fact_form_family.eq(""), fact_form_family)
            _ = ff  # placeholder for readability
        if "form_family_sub" in merged.columns:
            pass
        observed = merged["observed_date"].fillna(merged.get("acceptance_datetime", pd.NaT))
        filing = merged["filing_date"].fillna(pd.Timestamp.max.tz_localize("UTC"))
        merged["lag_vs_submission"] = (observed - filing).dt.total_seconds()
        merged["lag_vs_submission"] = merged["lag_vs_submission"].where(merged["lag_vs_submission"] >= 0, np.inf)
        merged = merged.sort_values(["cik", "period_end", "concept", "observed_date", "lag_vs_submission", "filing_date"])
        best = merged.drop_duplicates(subset=["cik", "period_end", "concept", "value", "observed_date"], keep="first").copy()
        best["linked_accession_number"] = best["accession_number_sub"]
        best["link_method"] = np.where(best["accession_number_sub"].notna(), "fallback_period", "unlinked")
        linked = pd.concat([direct_matched, best], ignore_index=True, sort=False)

    if linked.empty:
        failures.append(FailureRecord("linkage", "critical", "NO_LINKABLE_FACTS", "No facts could be linked to filings."))
        raise ModuleError("No linkable facts available for filings_flags.")

    linked["linked_accession_number"] = linked["linked_accession_number"].fillna(linked.get("accession_number", ""))
    unlinked = linked["link_method"].eq("unlinked").sum()
    if unlinked:
        failures.append(FailureRecord("linkage", "warn", "UNLINKED_FACTS", "Some facts could not be deterministically linked to a filing; they remain period-linked only.", int(unlinked)))
    return linked


def _build_period_fact_matrix(linked_facts: pd.DataFrame) -> pd.DataFrame:
    """Create one row per (cik, period_end) with latest value per concept observable as of run."""
    if linked_facts.empty:
        return pd.DataFrame(columns=["cik", "period_end"])
    facs = linked_facts.copy()
    acc_col = facs["acceptance_datetime"] if "acceptance_datetime" in facs.columns else pd.Series(pd.NaT, index=facs.index)
    obs = facs["observed_date"].fillna(acc_col)
    facs = facs.assign(_obs=obs)
    facs = facs.sort_values(["cik", "period_end", "concept", "_obs", "linked_accession_number"])
    facs = facs.drop_duplicates(subset=["cik", "period_end", "concept"], keep="last")
    wide = facs.pivot_table(index=["cik", "period_end"], columns="concept", values="value", aggfunc="last")
    wide = wide.reset_index()
    return wide


# -----------------------------
# Flag engines
# -----------------------------


def _compute_timeliness_flags(subs: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    out = subs.copy()
    out[["deadline_date", "deadline_policy_used"]] = out.apply(
        lambda r: pd.Series(_infer_deadline_date(r, cfg)), axis=1
    )
    out["days_late"] = np.where(
        out["deadline_date"].notna(),
        (out["filing_date"] - out["deadline_date"]).dt.days.clip(lower=0),
        np.nan,
    )
    out["flag_late_filer"] = out["days_late"].fillna(0).gt(0).astype(int)
    window = int(cfg.get("persistent_late", {}).get("window", 4))
    threshold = float(cfg.get("persistent_late", {}).get("threshold", 0.5))
    out = out.sort_values(["cik", "filing_date", "acceptance_datetime", "accession_number"]).reset_index(drop=True)
    out["late_ratio_window"] = (
        out.groupby("cik")["flag_late_filer"]
        .transform(lambda s: pd.Series(_rolling_flag_mean(s.tolist(), window), index=s.index))
        .astype(float)
    )
    out["flag_persistent_late_filer"] = out["late_ratio_window"].ge(threshold).astype(int)
    return out


def _compute_amendment_flags(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out["flag_amended"] = out["form_type"].astype(str).str.upper().str.endswith("/A").astype(int)
    text = out["document_text"].map(_safe_text)
    hard_terms = [t.lower() for t in cfg.get("restatement", {}).get("hard_terms", [])]
    suspected_terms = [t.lower() for t in cfg.get("restatement", {}).get("suspected_terms", [])]

    has_hard_term = text.apply(lambda x: int(any(term in x for term in hard_terms)))
    has_suspected_term = text.apply(lambda x: int(any(term in x for term in suspected_terms)))

    amended_sensitive = out["flag_amended"].eq(1) & out["form_family"].isin(["10-K", "10-Q", "20-F"])
    amendment_class = np.where(
        out["flag_amended"].eq(0),
        "none",
        np.where(has_hard_term.eq(1), "substantive", np.where(has_suspected_term.eq(1), "substantive", "technical")),
    )
    out["amendment_class"] = amendment_class
    out["evidence_explicit_hard_term"] = has_hard_term.astype(int)
    out["evidence_suspected_term"] = has_suspected_term.astype(int)
    out["evidence_amended_sensitive_form"] = amended_sensitive.astype(int)
    return out


def _compute_completeness_flags(df: pd.DataFrame, linked_facts: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    family_to_core = cfg.get("core_fields_by_form_family", {})
    weights = cfg.get("core_field_weights", {})

    # direct filing-level fact availability via linked filing accession when possible
    linked = linked_facts.copy()
    linked["fact_accession"] = linked["linked_accession_number"].fillna("")
    filing_concepts = (
        linked.groupby(["cik", "fact_accession", "concept"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    filing_concepts["present"] = 1
    concept_map = filing_concepts.pivot_table(
        index=["cik", "fact_accession"], columns="concept", values="present", aggfunc="max", fill_value=0
    ).reset_index()

    out = out.merge(
        concept_map,
        left_on=["cik", "accession_number"],
        right_on=["cik", "fact_accession"],
        how="left",
    )

    missing_counts: List[int] = []
    coverage_ratios: List[float] = []
    missing_lists: List[str] = []
    weighted_missing_scores: List[float] = []
    for _, row in out.iterrows():
        family = str(row.get("form_family", ""))
        expected = list(family_to_core.get(family, []))
        if not expected:
            missing_counts.append(0)
            coverage_ratios.append(np.nan)
            missing_lists.append("")
            weighted_missing_scores.append(0.0)
            continue
        present = []
        missing = []
        weighted_missing = 0.0
        weight_total = 0.0
        for concept in expected:
            weight = float(weights.get(concept, 1.0))
            weight_total += weight
            value_present = 0
            if concept in out.columns:
                try:
                    value_present = int(pd.notna(row.get(concept)) and float(row.get(concept, 0)) == 1.0)
                except Exception:
                    value_present = int(pd.notna(row.get(concept)))
            if value_present:
                present.append(concept)
            else:
                missing.append(concept)
                weighted_missing += weight
        missing_counts.append(len(missing))
        coverage_ratios.append(len(present) / len(expected) if expected else np.nan)
        missing_lists.append("|".join(missing))
        weighted_missing_scores.append(weighted_missing / weight_total if weight_total else 0.0)
    out["num_missing_core_fields"] = missing_counts
    out["core_coverage_ratio"] = coverage_ratios
    out["missing_core_fields_list"] = missing_lists
    out["weighted_missing_core_score"] = weighted_missing_scores
    out["flag_missing_core_fields"] = out["num_missing_core_fields"].gt(0).astype(int)
    sparse_thr = float(cfg.get("coverage", {}).get("sparse_threshold", 0.6))
    out["flag_sparse_fundamental_coverage"] = out["core_coverage_ratio"].lt(sparse_thr).fillna(False).astype(int)
    return out


def _compute_reporting_structure_flags(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy().sort_values(["cik", "form_family", "period_end", "filing_date", "accession_number"]).reset_index(drop=True)
    out["days_since_prev_period"] = (
        out.groupby(["cik", "form_family"]) ["period_end"].diff().dt.days
    )
    freq_map = cfg.get("form_frequency_days", {})
    gap_flags: List[int] = []
    irregular_flags: List[int] = []
    for _, row in out.iterrows():
        family = str(row.get("form_family", ""))
        rules = freq_map.get(family, {})
        d = row.get("days_since_prev_period")
        if pd.isna(d) or not rules:
            gap_flags.append(0)
            irregular_flags.append(0)
            continue
        min_days = float(rules.get("min", -np.inf))
        max_days = float(rules.get("max", np.inf))
        gap_flags.append(int(d > max_days))
        irregular_flags.append(int(d < min_days))
    out["flag_reporting_gap"] = gap_flags
    out["flag_irregular_filing_sequence"] = irregular_flags
    return out


def _compute_period_anomalies(period_matrix: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    if period_matrix.empty:
        return pd.DataFrame(columns=["cik", "period_end", "flag_accounting_anomaly", "flag_balance_sheet_inconsistency", "anomaly_magnitude", "anomalous_concepts"]) 

    out = period_matrix.copy().sort_values(["cik", "period_end"]).reset_index(drop=True)
    concepts = list(cfg.get("anomaly", {}).get("concepts", []))
    thresholds = cfg.get("anomaly", {}).get("concept_thresholds", {})
    min_history = int(cfg.get("anomaly", {}).get("min_history", 4))
    bs_eps = float(cfg.get("anomaly", {}).get("bs_epsilon", 0.03))

    out["anomaly_magnitude"] = 0.0
    out["anomalous_concepts"] = ""
    out["flag_accounting_anomaly"] = 0
    for concept in concepts:
        if concept not in out.columns:
            continue
        concept_z = []
        for _, grp in out.groupby("cik"):
            vals = grp[concept].tolist()
            idxs = grp.index.tolist()
            for i, idx in enumerate(idxs):
                history = [v for v in vals[:i] if pd.notna(v)]
                current = vals[i]
                if len(history) < min_history or pd.isna(current):
                    concept_z.append((idx, np.nan))
                    continue
                med = float(np.median(history))
                mad = _mad(history)
                if not np.isfinite(mad) or mad == 0:
                    concept_z.append((idx, np.nan))
                    continue
                z = float((current - med) / (1.4826 * mad))
                concept_z.append((idx, z))
        z_col = f"robust_z_{concept}"
        out[z_col] = np.nan
        for idx, z in concept_z:
            out.at[idx, z_col] = z
        thr = float(thresholds.get(concept, 6.0))
        mask = out[z_col].abs() > thr
        out.loc[mask, "flag_accounting_anomaly"] = 1
        out.loc[mask, "anomaly_magnitude"] = np.maximum(out.loc[mask, "anomaly_magnitude"], out.loc[mask, z_col].abs())
        out.loc[mask, "anomalous_concepts"] = out.loc[mask, "anomalous_concepts"].where(
            out.loc[mask, "anomalous_concepts"].eq(""), out.loc[mask, "anomalous_concepts"] + "|"
        ) + concept

    assets = out["Assets"] if "Assets" in out.columns else np.nan
    liab = out["Liabilities"] if "Liabilities" in out.columns else np.nan
    equity = out["StockholdersEquity"] if "StockholdersEquity" in out.columns else np.nan
    if isinstance(assets, pd.Series) and isinstance(liab, pd.Series) and isinstance(equity, pd.Series):
        residual = assets - liab - equity
        denom = 1.0 + assets.abs()
        ratio = (residual.abs() / denom).replace([np.inf, -np.inf], np.nan)
        out["balance_sheet_residual_ratio"] = ratio
        out["flag_balance_sheet_inconsistency"] = ratio.gt(bs_eps).fillna(False).astype(int)
        out.loc[out["flag_balance_sheet_inconsistency"].eq(1), "flag_accounting_anomaly"] = 1
        out["anomaly_magnitude"] = np.maximum(out["anomaly_magnitude"], ratio.fillna(0.0))
    else:
        out["balance_sheet_residual_ratio"] = np.nan
        out["flag_balance_sheet_inconsistency"] = 0

    out["anomalous_concepts"] = out["anomalous_concepts"].str.strip("|")
    return out[[
        "cik",
        "period_end",
        "flag_accounting_anomaly",
        "flag_balance_sheet_inconsistency",
        "anomaly_magnitude",
        "anomalous_concepts",
        "balance_sheet_residual_ratio",
    ] + [c for c in out.columns if c.startswith("robust_z_")]]


def _compute_restatement_flags(df: pd.DataFrame, linked_facts: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    rest_cfg = cfg.get("restatement", {})
    weights = rest_cfg.get("evidence_weights", {})
    material_change_ratio = float(rest_cfg.get("material_change_ratio", 0.05))

    # detect materially changed facts for same cik/period/concept across accessions
    facs = linked_facts.copy()
    facs = facs[pd.notna(facs["period_end"])].copy()
    facs["value_abs"] = facs["value"].abs()
    facs = facs.sort_values(["cik", "period_end", "concept", "observed_date", "linked_accession_number"])
    material_change: Dict[Tuple[str, str], int] = {}
    evidence_type: Dict[Tuple[str, str], str] = {}
    for (cik, period_end, concept), grp in facs.groupby(["cik", "period_end", "concept"], dropna=False):
        grp = grp.dropna(subset=["value"])
        if grp.empty or grp["linked_accession_number"].nunique() < 2:
            continue
        first_val = grp["value"].iloc[0]
        last_val = grp["value"].iloc[-1]
        denom = 1.0 + abs(first_val)
        rel = abs(last_val - first_val) / denom
        if rel >= material_change_ratio:
            latest_acc = str(grp["linked_accession_number"].iloc[-1])
            material_change[(str(cik), latest_acc)] = material_change.get((str(cik), latest_acc), 0) + 1
            evidence_type[(str(cik), latest_acc)] = "material_fact_change"

    score_vals: List[float] = []
    hard_flags: List[int] = []
    suspected_flags: List[int] = []
    evidence_labels: List[str] = []
    for _, row in out.iterrows():
        key = (str(row["cik"]), str(row["accession_number"]))
        score = 0.0
        ev: List[str] = []
        if int(row.get("evidence_explicit_hard_term", 0)) == 1:
            score += float(weights.get("explicit_hard_term", 1.0))
            ev.append("explicit_hard_term")
        if int(row.get("evidence_amended_sensitive_form", 0)) == 1:
            score += float(weights.get("amended_sensitive_form", 0.55))
            ev.append("amended_sensitive_form")
        if int(row.get("evidence_suspected_term", 0)) == 1:
            score += float(weights.get("suspected_term", 0.35))
            ev.append("suspected_term")
        changes = material_change.get(key, 0)
        if changes > 0:
            score += float(weights.get("material_fact_change", 0.7)) * min(changes, 3)
            ev.append("material_fact_change")
        # multiple accounting anomalies can contribute suspected evidence
        if int(row.get("flag_accounting_anomaly", 0)) + int(row.get("flag_balance_sheet_inconsistency", 0)) >= 2:
            score += float(weights.get("multiple_accounting_anomalies", 0.35))
            ev.append("multiple_accounting_anomalies")
        hard = int(score >= float(rest_cfg.get("hard_threshold", 1.0)))
        suspected = int((not hard) and score >= float(rest_cfg.get("suspected_threshold", 0.5)))
        score_vals.append(score)
        hard_flags.append(hard)
        suspected_flags.append(suspected)
        evidence_labels.append("|".join(ev))
    out["restatement_evidence_score"] = score_vals
    out["restatement_evidence_type"] = evidence_labels
    out["flag_restatement_hard"] = hard_flags
    out["flag_restatement_suspected"] = suspected_flags
    return out


def _aggregate_severity_and_action(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    score_cfg = cfg.get("score", {})
    weights: Mapping[str, float] = score_cfg.get("weights", {})
    critical_flags: Sequence[str] = score_cfg.get("critical_flags", [])
    warn_if_any: Sequence[str] = cfg.get("pit_policy", {}).get("warn_if_any", [])
    exclude_if_any: Sequence[str] = cfg.get("pit_policy", {}).get("exclude_if_any", [])

    weighted_sum = np.zeros(len(out), dtype=float)
    total_weight = float(sum(float(v) for v in weights.values())) or 1.0
    for flag, w in weights.items():
        if flag in out.columns:
            weighted_sum += out[flag].fillna(0).astype(float).to_numpy() * float(w)
    quality_raw = 1.0 - (weighted_sum / total_weight)
    quality_raw = np.clip(quality_raw, 0.0, 1.0)
    high_sev_count = np.zeros(len(out), dtype=float)
    for flag in critical_flags:
        if flag in out.columns:
            high_sev_count += out[flag].fillna(0).astype(float).to_numpy()
    lam = float(score_cfg.get("severity_penalty_lambda", 0.45))
    quality_final = quality_raw * np.exp(-lam * high_sev_count)
    out["quality_score_raw"] = quality_raw
    out["quality_score"] = quality_final
    out["quality_score_final"] = quality_final
    out["score_version"] = score_cfg.get("version", "1.0")

    severities: List[str] = []
    pit_actions: List[str] = []
    dominant_reasons: List[str] = []
    exclude_thr = float(score_cfg.get("exclude_threshold", 0.35))
    penalize_thr = float(score_cfg.get("penalize_threshold", 0.72))

    for _, row in out.iterrows():
        active_critical = [flag for flag in critical_flags if int(row.get(flag, 0)) == 1]
        active_warn = [flag for flag in warn_if_any if int(row.get(flag, 0)) == 1]
        if active_critical:
            severity = "critical"
            reason = active_critical[0]
        elif int(row.get("flag_amended", 0)) == 1 and int(row.get("flag_missing_core_fields", 0)) == 1:
            severity = "warn"
            reason = "amended_plus_missing_core"
        elif active_warn:
            severity = "warn"
            reason = active_warn[0]
        elif int(row.get("flag_amended", 0)) == 1 or int(row.get("flag_irregular_filing_sequence", 0)) == 1:
            severity = "info"
            reason = "minor_reporting_issue"
        else:
            severity = "info"
            reason = "clean"

        if any(int(row.get(flag, 0)) == 1 for flag in exclude_if_any) or float(row["quality_score"]) <= exclude_thr:
            pit = "exclude"
        elif severity == "warn" or any(int(row.get(flag, 0)) == 1 for flag in warn_if_any) or float(row["quality_score"]) <= penalize_thr:
            pit = "penalize"
        else:
            pit = "allow"
        severities.append(severity)
        pit_actions.append(pit)
        dominant_reasons.append(reason)
    out["severity"] = severities
    out["pit_action"] = pit_actions
    out["dominant_quality_reason"] = dominant_reasons
    return out


# -----------------------------
# Metrics and summaries
# -----------------------------


def _bucket_market_cap(series: pd.Series) -> pd.Series:
    if series.isna().all():
        return pd.Series([pd.NA] * len(series), index=series.index)
    bins = [-np.inf, 3e8, 2e9, 1e10, np.inf]
    labels = ["micro", "small", "mid", "large"]
    return pd.cut(series.astype(float), bins=bins, labels=labels)


def _build_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics: List[Dict[str, Any]] = []
    metrics.append({"group": "all", "metric": "n_filings", "value": int(len(df))})
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    for col in flag_cols:
        metrics.append({"group": "all", "metric": f"rate::{col}", "value": float(df[col].fillna(0).mean())})
    for action in ["allow", "penalize", "exclude"]:
        metrics.append({"group": "all", "metric": f"rate::pit_action::{action}", "value": float((df["pit_action"] == action).mean())})
    metrics.append({"group": "all", "metric": "quality_score_mean", "value": float(df["quality_score"].mean())})
    metrics.append({"group": "all", "metric": "quality_score_median", "value": float(df["quality_score"].median())})

    for group_col in ["form_family", "sector", "liquidity_bucket"]:
        if group_col in df.columns:
            for g, grp in df.groupby(group_col, dropna=False):
                gname = f"{group_col}::{g}"
                metrics.append({"group": gname, "metric": "n_filings", "value": int(len(grp))})
                metrics.append({"group": gname, "metric": "quality_score_mean", "value": float(grp["quality_score"].mean())})
                for col in [
                    "flag_late_filer",
                    "flag_persistent_late_filer",
                    "flag_amended",
                    "flag_restatement_hard",
                    "flag_restatement_suspected",
                    "flag_missing_core_fields",
                    "flag_accounting_anomaly",
                    "flag_reporting_gap",
                ]:
                    if col in grp.columns:
                        metrics.append({"group": gname, "metric": f"rate::{col}", "value": float(grp[col].fillna(0).mean())})
    if "market_cap" in df.columns:
        cap_bucket = _bucket_market_cap(df["market_cap"])
        tmp = df.copy()
        tmp["market_cap_bucket"] = cap_bucket
        for g, grp in tmp.groupby("market_cap_bucket", dropna=False):
            gname = f"market_cap_bucket::{g}"
            metrics.append({"group": gname, "metric": "n_filings", "value": int(len(grp))})
            metrics.append({"group": gname, "metric": "quality_score_mean", "value": float(grp["quality_score"].mean())})
    return pd.DataFrame(metrics)


def _build_summary(df: pd.DataFrame, failures: Sequence[FailureRecord], cfg: Mapping[str, Any], run_id: str, asof: Optional[pd.Timestamp]) -> Dict[str, Any]:
    summary = {
        "run_id": run_id,
        "asof": asof.isoformat() if asof is not None else None,
        "generated_at_utc": _utc_now_iso(),
        "config_version": cfg.get("config_version", "unknown"),
        "n_filings": int(len(df)),
        "n_unique_cik": int(df["cik"].nunique()) if not df.empty else 0,
        "quality_score_mean": float(df["quality_score"].mean()) if not df.empty else None,
        "quality_score_median": float(df["quality_score"].median()) if not df.empty else None,
        "pit_action_distribution": df["pit_action"].value_counts(dropna=False).to_dict() if not df.empty else {},
        "severity_distribution": df["severity"].value_counts(dropna=False).to_dict() if not df.empty else {},
        "failure_count": len(failures),
        "failures_by_severity": pd.Series([f.severity for f in failures]).value_counts().to_dict() if failures else {},
    }
    for col in [
        "flag_late_filer",
        "flag_persistent_late_filer",
        "flag_amended",
        "flag_restatement_hard",
        "flag_restatement_suspected",
        "flag_missing_core_fields",
        "flag_sparse_fundamental_coverage",
        "flag_accounting_anomaly",
        "flag_balance_sheet_inconsistency",
        "flag_reporting_gap",
        "flag_irregular_filing_sequence",
    ]:
        if col in df.columns and not df.empty:
            summary[f"rate::{col}"] = float(df[col].fillna(0).mean())
    return summary


# -----------------------------
# Main engine
# -----------------------------


def run_filings_flags(
    submissions_path: str | Path,
    facts_path: str | Path,
    config_path: Optional[str | Path] = None,
    run_id: str = "manual_run",
    asof: Optional[str] = None,
) -> Dict[str, Any]:
    failures: List[FailureRecord] = []
    cfg = _load_config(str(config_path) if config_path is not None else None)
    asof_ts = pd.Timestamp(asof, tz="UTC") if asof else None

    submissions = _normalize_submissions(_read_table(submissions_path), failures)
    facts = _normalize_facts(_read_table(facts_path), failures)
    submissions, facts = _apply_pit_filters(submissions, facts, asof_ts)
    if submissions.empty:
        failures.append(FailureRecord("pit_filter", "critical", "NO_SUBMISSIONS_AFTER_ASOF", "No submissions remain after PIT filter."))
        raise ModuleError("No submissions remain after PIT filter.")
    if facts.empty:
        failures.append(FailureRecord("pit_filter", "critical", "NO_FACTS_AFTER_ASOF", "No facts remain after PIT filter."))
        raise ModuleError("No facts remain after PIT filter.")

    reverse_form_map = _reverse_form_family_map(cfg)
    submissions = submissions.copy()
    submissions["form_family"] = submissions["form_type"].map(lambda x: _classify_form_family(x, reverse_form_map))
    facts = facts.copy()
    facts["form_family"] = facts["form_type"].map(lambda x: _classify_form_family(x, reverse_form_map))

    linked_facts = _link_facts_to_filings(submissions, facts, failures)
    period_matrix = _build_period_fact_matrix(linked_facts)

    filings = _compute_timeliness_flags(submissions, cfg)
    filings = _compute_amendment_flags(filings, cfg)
    filings = _compute_completeness_flags(filings, linked_facts, cfg)
    filings = _compute_reporting_structure_flags(filings, cfg)

    anomalies = _compute_period_anomalies(period_matrix, cfg)
    filings = filings.merge(anomalies, on=["cik", "period_end"], how="left")
    for col in ["flag_accounting_anomaly", "flag_balance_sheet_inconsistency"]:
        if col in filings.columns:
            filings[col] = filings[col].fillna(0).astype(int)
    filings["anomaly_magnitude"] = filings.get("anomaly_magnitude", pd.Series(np.nan, index=filings.index))
    filings["restatement_evidence_type"] = ""

    filings = _compute_restatement_flags(filings, linked_facts, cfg)
    filings = _aggregate_severity_and_action(filings, cfg)
    filings["run_id"] = run_id
    filings["config_version"] = cfg.get("config_version", "unknown")

    # Preserve required ordering and context columns.
    required_order = [
        "cik",
        "accession_number",
        "form_type",
        "filing_date",
        "acceptance_datetime",
        "period_end",
        "form_family",
        "deadline_date",
        "deadline_policy_used",
        "days_late",
        "flag_late_filer",
        "late_ratio_window",
        "flag_persistent_late_filer",
        "flag_amended",
        "amendment_class",
        "flag_restatement_hard",
        "flag_restatement_suspected",
        "restatement_evidence_type",
        "restatement_evidence_score",
        "flag_missing_core_fields",
        "num_missing_core_fields",
        "core_coverage_ratio",
        "missing_core_fields_list",
        "flag_sparse_fundamental_coverage",
        "flag_accounting_anomaly",
        "flag_balance_sheet_inconsistency",
        "anomaly_magnitude",
        "anomalous_concepts",
        "balance_sheet_residual_ratio",
        "flag_reporting_gap",
        "flag_irregular_filing_sequence",
        "quality_score_raw",
        "quality_score",
        "quality_score_final",
        "score_version",
        "severity",
        "pit_action",
        "dominant_quality_reason",
        "run_id",
        "config_version",
    ]
    remaining_cols = [c for c in filings.columns if c not in required_order]
    filings = filings[required_order + remaining_cols]

    metrics = _build_metrics(filings)
    failures_df = pd.DataFrame([asdict(f) for f in failures])
    summary = _build_summary(filings, failures, cfg, run_id, asof_ts)

    output_dir = _ensure_dir(cfg.get("output_dir", "data/edgar/flags"))
    prefer_parquet = bool(cfg.get("persistence", {}).get("prefer_parquet", True))
    strict_parquet = bool(cfg.get("strict_parquet", False))

    files = {
        "filings_flags": _write_table(filings, output_dir / f"filings_flags_{run_id}", prefer_parquet=prefer_parquet, strict_parquet=strict_parquet),
        "filings_flags_metrics": _write_table(metrics, output_dir / f"filings_flags_metrics_{run_id}", prefer_parquet=prefer_parquet, strict_parquet=strict_parquet),
        "filings_flags_summary": _write_json(summary, output_dir / f"filings_flags_summary_{run_id}.json"),
    }
    if not failures_df.empty:
        files["filings_flags_failures"] = _write_table(failures_df, output_dir / f"filings_flags_failures_{run_id}", prefer_parquet=prefer_parquet, strict_parquet=strict_parquet)

    manifest = {
        "module": "data.edgar.filings_flags",
        "run_id": run_id,
        "generated_at_utc": _utc_now_iso(),
        "asof": asof_ts.isoformat() if asof_ts is not None else None,
        "inputs": {
            "submissions_path": str(submissions_path),
            "facts_path": str(facts_path),
            "config_path": str(config_path) if config_path is not None else None,
        },
        "config_version": cfg.get("config_version", "unknown"),
        "row_counts": {
            "submissions_after_pit": int(len(submissions)),
            "facts_after_pit": int(len(facts)),
            "linked_facts": int(len(linked_facts)),
            "filings_flags": int(len(filings)),
            "metrics": int(len(metrics)),
            "failures": int(len(failures_df)),
        },
        "linkage_methods": linked_facts["link_method"].value_counts(dropna=False).to_dict() if "link_method" in linked_facts.columns else {},
        "files": files,
        "summary": {
            "quality_score_mean": summary.get("quality_score_mean"),
            "pit_action_distribution": summary.get("pit_action_distribution"),
            "severity_distribution": summary.get("severity_distribution"),
        },
    }
    files["manifest"] = _write_json(manifest, output_dir / f"manifest_{run_id}.json")

    return {
        "filings_flags": filings,
        "metrics": metrics,
        "failures": failures_df,
        "summary": summary,
        "manifest": manifest,
        "files": files,
    }


# -----------------------------
# CLI
# -----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate PIT-safe EDGAR filings quality flags.")
    parser.add_argument("--submissions-path", required=True, help="Path to submissions table/directory.")
    parser.add_argument("--facts-path", required=True, help="Path to canonized facts table/directory.")
    parser.add_argument("--config-path", default=None, help="Path to JSON/YAML config file.")
    parser.add_argument("--run-id", required=True, help="Run identifier for artifacts.")
    parser.add_argument("--asof", default=None, help="UTC as-of timestamp (ISO-8601 recommended).")
    parser.add_argument("--print-summary", action="store_true", help="Print summary JSON to stdout.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = run_filings_flags(
            submissions_path=args.submissions_path,
            facts_path=args.facts_path,
            config_path=args.config_path,
            run_id=args.run_id,
            asof=args.asof,
        )
        if args.print_summary:
            print(json.dumps(result["summary"], indent=2, sort_keys=True, default=_json_default))
        else:
            print(json.dumps({"status": "ok", "run_id": args.run_id, "files": result["files"]}, indent=2, sort_keys=True))
        return 0
    except ModuleError as exc:
        print(json.dumps({"status": "error", "error": str(exc), "module": "data.edgar.filings_flags"}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
