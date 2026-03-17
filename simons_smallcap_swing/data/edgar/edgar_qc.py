from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# -----------------------------------------------------------------------------
# Errors / constants
# -----------------------------------------------------------------------------


class EdgarQCError(RuntimeError):
    """Raised when EDGAR QC cannot be evaluated safely."""


VALID_GATES = {"pass", "warn", "fail"}
VALID_SEVERITIES = {"info", "warn", "critical"}


REQUIRED_TABLES = {
    "ticker_cik_outputs",
    "submissions_raw",
    "companyfacts_raw",
    "parsed_facts",
    "pit_store",
}


PIT_REQUIRED_COLUMNS = {
    "symbol",
    "cik",
    "metric",
    "value",
    "unit",
    "period_end",
    "acceptance_ts",
    "asof",
    "source_filing_id",
    "run_id",
}


REQUIRED_COLUMNS: Dict[str, set[str]] = {
    "ticker_cik_outputs": {"symbol", "cik"},
    "submissions_raw": {"symbol", "cik", "source_filing_id", "acceptance_ts"},
    "companyfacts_raw": {"symbol", "cik"},
    "parsed_facts": {"symbol", "cik", "metric", "value", "unit", "period_end", "acceptance_ts", "source_filing_id"},
    "pit_store": PIT_REQUIRED_COLUMNS,
}


UNIQUE_KEYS: Dict[str, List[str]] = {
    "ticker_cik_outputs": ["symbol", "cik", "valid_from", "valid_to"],
    "submissions_raw": ["source_filing_id"],
    "companyfacts_raw": ["symbol", "cik", "metric", "unit", "period_end", "acceptance_ts", "source_filing_id"],
    "parsed_facts": ["symbol", "cik", "metric", "period_end", "acceptance_ts", "source_filing_id"],
    "pit_store": ["symbol", "cik", "metric", "asof", "period_end"],
}


TABLE_ALIASES: Dict[str, Dict[str, Sequence[str]]] = {
    "ticker_cik_outputs": {
        "symbol": ["symbol", "ticker", "security", "asset", "issuer_ticker"],
        "cik": ["cik", "issuer_cik", "sec_cik"],
        "valid_from": ["valid_from", "effective_from", "start_date", "active_from", "from_date"],
        "valid_to": ["valid_to", "effective_to", "end_date", "active_to", "to_date"],
        "source_priority": ["source_priority", "priority", "mapping_source_priority"],
        "source_type": ["source_type", "source", "mapping_source", "identity_source"],
        "sector": ["sector", "gics_sector", "industry_sector"],
        "size_bucket": ["size_bucket", "cap_bucket", "market_cap_bucket"],
        "liquidity_decile": ["liquidity_decile", "liq_decile", "liquidity_bucket", "adv_decile"],
        "run_id": ["run_id", "mapping_run_id"],
    },
    "submissions_raw": {
        "symbol": ["symbol", "ticker", "asset"],
        "cik": ["cik", "issuer_cik", "sec_cik"],
        "source_filing_id": ["source_filing_id", "filing_id", "accession_number", "accessionNo"],
        "form_type": ["form_type", "form", "filing_type"],
        "acceptance_ts": ["acceptance_ts", "accepted", "acceptance_datetime", "filing_acceptance_ts"],
        "filing_date": ["filing_date", "filed_date", "date_filed"],
        "status_code": ["status_code", "http_status", "response_code", "download_status"],
        "raw_artifact_id": ["raw_artifact_id", "artifact_id", "payload_id"],
        "run_id": ["run_id", "download_run_id"],
    },
    "companyfacts_raw": {
        "symbol": ["symbol", "ticker", "asset"],
        "cik": ["cik", "issuer_cik", "sec_cik"],
        "metric": ["metric", "fact", "taxonomy_metric", "concept"],
        "unit": ["unit", "units", "fact_unit"],
        "value": ["value", "fact_value"],
        "period_end": ["period_end", "fy_end", "end_date"],
        "acceptance_ts": ["acceptance_ts", "accepted", "acceptance_datetime"],
        "source_filing_id": ["source_filing_id", "filing_id", "accession_number"],
        "run_id": ["run_id", "companyfacts_run_id"],
    },
    "parsed_facts": {
        "symbol": ["symbol", "ticker", "asset"],
        "cik": ["cik", "issuer_cik", "sec_cik"],
        "metric": ["metric", "canonical_metric", "fact_metric", "concept"],
        "value": ["value", "fact_value", "numeric_value"],
        "unit": ["unit", "canonical_unit", "fact_unit"],
        "period_end": ["period_end", "fiscal_period_end", "end_date"],
        "acceptance_ts": ["acceptance_ts", "accepted", "acceptance_datetime"],
        "source_filing_id": ["source_filing_id", "filing_id", "accession_number"],
        "form_type": ["form_type", "form", "filing_type"],
        "amendment_flag": ["amendment_flag", "is_amended", "amended_flag"],
        "raw_artifact_id": ["raw_artifact_id", "artifact_id", "payload_id"],
        "run_id": ["run_id", "parse_run_id"],
    },
    "pit_store": {
        "symbol": ["symbol", "ticker", "asset"],
        "cik": ["cik", "issuer_cik", "sec_cik"],
        "metric": ["metric", "canonical_metric", "fact_metric", "concept"],
        "value": ["value", "fact_value", "numeric_value"],
        "unit": ["unit", "canonical_unit", "fact_unit"],
        "period_end": ["period_end", "fiscal_period_end", "end_date"],
        "acceptance_ts": ["acceptance_ts", "accepted", "acceptance_datetime"],
        "asof": ["asof", "asof_ts", "asof_date", "decision_ts"],
        "source_filing_id": ["source_filing_id", "filing_id", "accession_number"],
        "form_type": ["form_type", "form", "filing_type"],
        "raw_artifact_id": ["raw_artifact_id", "artifact_id", "payload_id"],
        "amendment_flag": ["amendment_flag", "is_amended", "amended_flag"],
        "run_id": ["run_id", "pit_run_id"],
        "sector": ["sector", "gics_sector", "industry_sector"],
        "size_bucket": ["size_bucket", "cap_bucket", "market_cap_bucket"],
        "liquidity_decile": ["liquidity_decile", "liq_decile", "liquidity_bucket", "adv_decile"],
    },
}


DATETIME_COLUMNS = {
    "valid_from",
    "valid_to",
    "acceptance_ts",
    "filing_date",
    "period_end",
    "asof",
}


CANONICAL_CIK_WIDTH = 10


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class StageCoverageThresholds:
    map_warn_min: float = 0.99
    map_fail_min: float = 0.95
    submissions_warn_min: float = 0.95
    submissions_fail_min: float = 0.85
    facts_warn_min: float = 0.93
    facts_fail_min: float = 0.82
    pit_warn_min: float = 0.90
    pit_fail_min: float = 0.75
    absolute_pit_fail_min: float = 0.60
    stratified_warn_min: float = 0.80
    stratified_fail_min: float = 0.65
    daily_warn_min: float = 0.85
    daily_fail_min: float = 0.70


@dataclass
class IdentityThresholds:
    max_mapping_conflict_share_warn: float = 0.0
    max_mapping_conflict_share_fail: float = 0.0
    max_table_mismatch_share_warn: float = 0.005
    max_table_mismatch_share_fail: float = 0.02
    allow_multi_ticker_per_cik: bool = False


@dataclass
class ValueQualityThresholds:
    robust_z_warn: float = 8.0
    robust_z_fail: float = 15.0
    outlier_share_warn: float = 0.02
    outlier_share_fail: float = 0.07
    accounting_residual_warn: float = 0.10
    accounting_residual_fail: float = 0.25
    negative_share_warn: float = 0.01
    negative_share_fail: float = 0.03
    discontinuity_warn_q: float = 0.995
    discontinuity_fail_q: float = 0.999


@dataclass
class LineageThresholds:
    min_traceability_warn: float = 0.995
    min_traceability_fail: float = 0.98
    critical_metric_lineage_fail: float = 0.0


@dataclass
class StalenessThresholds:
    default_warn_days: int = 120
    default_fail_days: int = 220
    essential_metric_fail_share: float = 0.10
    essential_metric_warn_share: float = 0.03
    exact_warn_days: Dict[str, int] = field(
        default_factory=lambda: {
            "assets": 120,
            "liabilities": 120,
            "equity": 120,
            "net_income": 120,
            "revenue": 120,
            "shares_outstanding": 75,
        }
    )
    exact_fail_days: Dict[str, int] = field(
        default_factory=lambda: {
            "assets": 220,
            "liabilities": 220,
            "equity": 220,
            "net_income": 220,
            "revenue": 220,
            "shares_outstanding": 120,
        }
    )
    family_warn_days: Dict[str, int] = field(
        default_factory=lambda: {
            "balance_sheet": 120,
            "income_statement": 120,
            "share_count": 75,
            "cash_flow": 130,
        }
    )
    family_fail_days: Dict[str, int] = field(
        default_factory=lambda: {
            "balance_sheet": 220,
            "income_statement": 220,
            "share_count": 120,
            "cash_flow": 240,
        }
    )


@dataclass
class ScoreWeights:
    coverage: float = 0.22
    identity: float = 0.18
    parse_download: float = 0.10
    value_quality: float = 0.18
    staleness: float = 0.12
    lineage: float = 0.20


@dataclass
class ScoreThresholds:
    pass_min: float = 85.0
    fail_min: float = 65.0


@dataclass
class EdgarQCConfig:
    config_version: str = "edgar_qc_v1"
    coverage: StageCoverageThresholds = field(default_factory=StageCoverageThresholds)
    identity: IdentityThresholds = field(default_factory=IdentityThresholds)
    value_quality: ValueQualityThresholds = field(default_factory=ValueQualityThresholds)
    lineage: LineageThresholds = field(default_factory=LineageThresholds)
    staleness: StalenessThresholds = field(default_factory=StalenessThresholds)
    scoring: ScoreWeights = field(default_factory=ScoreWeights)
    score_thresholds: ScoreThresholds = field(default_factory=ScoreThresholds)
    required_lineage_columns: List[str] = field(
        default_factory=lambda: [
            "source_filing_id",
            "form_type",
            "raw_artifact_id",
            "run_id",
        ]
    )
    critical_metrics: List[str] = field(
        default_factory=lambda: [
            "assets",
            "liabilities",
            "equity",
            "net_income",
            "revenue",
            "shares_outstanding",
        ]
    )
    positive_only_metrics: List[str] = field(
        default_factory=lambda: [
            "shares_outstanding",
            "weighted_average_shares",
            "basic_shares_outstanding",
        ]
    )
    nonnegative_metrics: List[str] = field(
        default_factory=lambda: [
            "assets",
            "current_assets",
            "liabilities",
            "current_liabilities",
            "cash_and_equivalents",
            "revenue",
        ]
    )
    allowed_identity_sources: List[str] = field(
        default_factory=lambda: ["official_sec_source", "historical_internal_mapping", "heuristic_inference"]
    )
    persist_index: bool = False


DEFAULT_CONFIG = EdgarQCConfig()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def utc_now_iso() -> str:
    ts = pd.Timestamp.utcnow()
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")



def git_revision() -> Optional[str]:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return proc.stdout.strip()
    except Exception:
        return None



def file_sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def load_yaml_or_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise EdgarQCError("PyYAML is required to load YAML configs but is not installed.")
        return yaml.safe_load(text)
    return json.loads(text)



def deep_update_dataclass(instance: Any, updates: Mapping[str, Any]) -> Any:
    for key, value in updates.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            deep_update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance



def load_config(path: Optional[Path]) -> EdgarQCConfig:
    cfg = EdgarQCConfig()
    if path is None:
        return cfg
    payload = load_yaml_or_json(path)
    if not isinstance(payload, Mapping):
        raise EdgarQCError("QC config must deserialize to a mapping.")
    return deep_update_dataclass(cfg, payload)



def choose_parquet_engine() -> str:
    for engine in ("pyarrow", "fastparquet"):
        try:
            __import__(engine)
            return engine
        except Exception:
            continue
    raise EdgarQCError(
        "Parquet support is required but neither 'pyarrow' nor 'fastparquet' is installed in this environment."
    )



def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise EdgarQCError(f"Input path does not exist: {path}")

    if path.is_dir():
        files = [
            p
            for p in sorted(path.rglob("*"))
            if p.is_file() and p.suffix.lower() in {".parquet", ".csv", ".json", ".jsonl", ".ndjson"}
        ]
        if not files:
            raise EdgarQCError(f"Directory contains no supported files: {path}")
        frames = [load_dataframe(p) for p in files]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, engine=choose_parquet_engine())
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            if "data" in payload and isinstance(payload["data"], list):
                return pd.DataFrame(payload["data"])
            return pd.DataFrame([payload])
    raise EdgarQCError(f"Unsupported file format for dataframe load: {path}")



def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")



def write_parquet(path: Path, df: pd.DataFrame, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=index, engine=choose_parquet_engine())



def maybe_rename_columns(df: pd.DataFrame, aliases: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    existing = {str(c): c for c in df.columns}
    rename: Dict[Any, str] = {}
    for canonical, choices in aliases.items():
        if canonical in df.columns:
            continue
        for candidate in choices:
            if candidate in existing:
                rename[existing[candidate]] = canonical
                break
    out = df.rename(columns=rename).copy()
    out.columns = [str(c) for c in out.columns]
    return out



def safe_to_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if getattr(parsed.dt, "tz", None) is None:
        return parsed.dt.tz_localize("UTC")
    return parsed.dt.tz_convert("UTC")



def canonicalize_cik(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    return digits.zfill(CANONICAL_CIK_WIDTH)



def normalize_metric(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_").lower()

    metric_aliases = {
        "assets": {"assets", "totalassets", "us_gaap_assets"},
        "liabilities": {"liabilities", "liabilitiesandstockholdersequity", "totalliabilities", "us_gaap_liabilities"},
        "equity": {"equity", "stockholdersequity", "stockholdersequityincludingportionattributabletononcontrollinginterest", "us_gaap_stockholdersequity"},
        "net_income": {"netincome", "netincomeloss", "profitloss", "us_gaap_netincomeloss"},
        "revenue": {"revenue", "revenues", "salesrevenue", "salesrevenuenet", "us_gaap_revenues", "us_gaap_salesrevenuenet"},
        "shares_outstanding": {"sharesoutstanding", "commonstocksharesoutstanding", "entitycommonstocksharesoutstanding", "weightedaveragenumberofdilutedsharesoutstanding", "weightedaveragebasicsharesoutstanding"},
        "cash_and_equivalents": {"cashandequivalentsatcarryingvalue", "cashcash equivalents and short term investments", "cashandcashequivalentsatcarryingvalue"},
        "current_assets": {"currentassets", "assetscurrent"},
        "current_liabilities": {"currentliabilities", "liabilitiescurrent"},
        "weighted_average_shares": {"weightedaveragebasicsharesoutstanding", "weightedaveragenumberofdilutedsharesoutstanding"},
    }
    dense = text.replace("_", "")
    for canonical, aliases in metric_aliases.items():
        if dense in {a.replace("_", "") for a in aliases}:
            return canonical
    return text



def normalize_table(name: str, df: pd.DataFrame) -> pd.DataFrame:
    out = maybe_rename_columns(df, TABLE_ALIASES.get(name, {}))
    if "cik" in out.columns:
        out["cik"] = out["cik"].map(canonicalize_cik)
    if "metric" in out.columns:
        out["metric"] = out["metric"].map(normalize_metric)
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype("string").str.strip().str.upper()
    for col in set(out.columns).intersection(DATETIME_COLUMNS):
        out[col] = safe_to_datetime(out[col])
    if "value" in out.columns:
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
    if "amendment_flag" in out.columns:
        out["amendment_flag"] = out["amendment_flag"].fillna(False).astype(bool)
    return out



def hash_config(cfg: EdgarQCConfig) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()



def coerce_jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [coerce_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): coerce_jsonable(v) for k, v in value.items()}
    return value


# -----------------------------------------------------------------------------
# Check / failure row factories
# -----------------------------------------------------------------------------



def make_check(
    *,
    stage: str,
    check_name: str,
    value: Any,
    threshold_warn: Any = None,
    threshold_fail: Any = None,
    severity: str = "info",
    blocking: bool = False,
    passed: bool = True,
    message: str = "",
    dimensions: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    severity = severity.lower()
    if severity not in VALID_SEVERITIES:
        raise EdgarQCError(f"Invalid severity '{severity}' for check row.")
    return {
        "stage": stage,
        "check_name": check_name,
        "value": coerce_jsonable(value),
        "threshold_warn": coerce_jsonable(threshold_warn),
        "threshold_fail": coerce_jsonable(threshold_fail),
        "severity": severity,
        "blocking": bool(blocking),
        "passed": bool(passed),
        "message": message,
        "dimensions": json.dumps(coerce_jsonable(dict(dimensions or {})), ensure_ascii=False, sort_keys=True),
    }



def make_failure(
    *,
    stage: str,
    check_name: str,
    severity: str,
    issue: str,
    blocking: bool = False,
    symbol: Any = None,
    cik: Any = None,
    metric: Any = None,
    asof: Any = None,
    row_index: Any = None,
    value: Any = None,
    threshold: Any = None,
    details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    severity = severity.lower()
    if severity not in VALID_SEVERITIES:
        raise EdgarQCError(f"Invalid severity '{severity}' for failure row.")
    return {
        "stage": stage,
        "check_name": check_name,
        "severity": severity,
        "blocking": bool(blocking),
        "issue": issue,
        "symbol": None if pd.isna(symbol) else symbol,
        "cik": None if pd.isna(cik) else cik,
        "metric": None if pd.isna(metric) else metric,
        "asof": coerce_jsonable(asof),
        "row_index": None if pd.isna(row_index) else row_index,
        "value": coerce_jsonable(value),
        "threshold": coerce_jsonable(threshold),
        "details": json.dumps(coerce_jsonable(dict(details or {})), ensure_ascii=False, sort_keys=True),
    }


# -----------------------------------------------------------------------------
# Schema / structural checks
# -----------------------------------------------------------------------------



def _safe_unique_key_cols(df: pd.DataFrame, table_name: str) -> List[str]:
    cols = [c for c in UNIQUE_KEYS.get(table_name, []) if c in df.columns]
    if not cols:
        cols = [c for c in REQUIRED_COLUMNS[table_name] if c in df.columns]
    return cols



def run_schema_checks(
    tables: Mapping[str, pd.DataFrame],
    checks: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> Tuple[bool, Dict[str, Any]]:
    schema_ok = True
    metrics: Dict[str, Any] = {}

    for table_name in REQUIRED_TABLES:
        df = tables.get(table_name)
        if df is None:
            schema_ok = False
            checks.append(
                make_check(
                    stage="schema",
                    check_name=f"missing_table::{table_name}",
                    value=0,
                    severity="critical",
                    blocking=True,
                    passed=False,
                    message=f"Required input table '{table_name}' is missing.",
                )
            )
            failures.append(
                make_failure(
                    stage="schema",
                    check_name=f"missing_table::{table_name}",
                    severity="critical",
                    blocking=True,
                    issue="required_input_missing",
                    details={"table": table_name},
                )
            )
            continue

        missing_cols = sorted(REQUIRED_COLUMNS[table_name] - set(df.columns))
        if missing_cols:
            schema_ok = False
            checks.append(
                make_check(
                    stage="schema",
                    check_name=f"required_columns::{table_name}",
                    value=len(missing_cols),
                    threshold_fail=0,
                    severity="critical",
                    blocking=True,
                    passed=False,
                    message=f"Missing required columns in {table_name}: {missing_cols}",
                    dimensions={"table": table_name, "missing_columns": missing_cols},
                )
            )
            for col in missing_cols:
                failures.append(
                    make_failure(
                        stage="schema",
                        check_name=f"required_columns::{table_name}",
                        severity="critical",
                        blocking=True,
                        issue="missing_required_column",
                        details={"table": table_name, "column": col},
                    )
                )
        else:
            checks.append(
                make_check(
                    stage="schema",
                    check_name=f"required_columns::{table_name}",
                    value=0,
                    threshold_fail=0,
                    severity="info",
                    blocking=True,
                    passed=True,
                    message=f"Required columns satisfied for {table_name}.",
                    dimensions={"table": table_name},
                )
            )

        table_identity_cols = [c for c in ("symbol", "cik") if c in df.columns]
        for col in table_identity_cols:
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                schema_ok = False
                checks.append(
                    make_check(
                        stage="schema",
                        check_name=f"null_identity::{table_name}::{col}",
                        value=null_count,
                        threshold_fail=0,
                        severity="critical",
                        blocking=True,
                        passed=False,
                        message=f"Identity column {col} has nulls in {table_name}.",
                        dimensions={"table": table_name, "column": col},
                    )
                )
                failures.append(
                    make_failure(
                        stage="schema",
                        check_name=f"null_identity::{table_name}::{col}",
                        severity="critical",
                        blocking=True,
                        issue="null_identity_key",
                        details={"table": table_name, "column": col, "null_count": null_count},
                    )
                )

        dt_cols = sorted(set(df.columns).intersection(DATETIME_COLUMNS))
        for col in dt_cols:
            bad_ts = int(df[col].isna().sum())
            if bad_ts > 0:
                schema_ok = False
                checks.append(
                    make_check(
                        stage="schema",
                        check_name=f"unparseable_timestamp::{table_name}::{col}",
                        value=bad_ts,
                        threshold_fail=0,
                        severity="critical",
                        blocking=True,
                        passed=False,
                        message=f"Timestamp column {col} contains null/unparseable values in {table_name}.",
                        dimensions={"table": table_name, "column": col},
                    )
                )
                failures.append(
                    make_failure(
                        stage="schema",
                        check_name=f"unparseable_timestamp::{table_name}::{col}",
                        severity="critical",
                        blocking=True,
                        issue="unparseable_timestamp",
                        details={"table": table_name, "column": col, "bad_count": bad_ts},
                    )
                )

        key_cols = _safe_unique_key_cols(df, table_name)
        if key_cols:
            dup_mask = df.duplicated(subset=key_cols, keep=False)
            dup_count = int(dup_mask.sum())
            if dup_count > 0:
                is_critical = table_name in {"submissions_raw", "pit_store"}
                sev = "critical" if is_critical else "warn"
                schema_ok = schema_ok and not is_critical
                checks.append(
                    make_check(
                        stage="schema",
                        check_name=f"duplicate_keys::{table_name}",
                        value=dup_count,
                        threshold_fail=0,
                        severity=sev,
                        blocking=is_critical,
                        passed=False,
                        message=f"Duplicate logical keys in {table_name} over {key_cols}.",
                        dimensions={"table": table_name, "key_cols": key_cols},
                    )
                )
                dup_preview = df.loc[dup_mask, [c for c in key_cols if c in df.columns]].head(100)
                for _, row in dup_preview.iterrows():
                    failures.append(
                        make_failure(
                            stage="schema",
                            check_name=f"duplicate_keys::{table_name}",
                            severity=sev,
                            blocking=is_critical,
                            issue="duplicate_logical_key",
                            symbol=row.get("symbol"),
                            cik=row.get("cik"),
                            details={"table": table_name, "key_values": row.to_dict()},
                        )
                    )
            else:
                checks.append(
                    make_check(
                        stage="schema",
                        check_name=f"duplicate_keys::{table_name}",
                        value=0,
                        threshold_fail=0,
                        severity="info",
                        blocking=table_name in {"submissions_raw", "pit_store"},
                        passed=True,
                        message=f"No duplicate logical keys in {table_name}.",
                        dimensions={"table": table_name, "key_cols": key_cols},
                    )
                )

        metrics[f"rows::{table_name}"] = int(len(df))

    return schema_ok, metrics


# -----------------------------------------------------------------------------
# Coverage checks
# -----------------------------------------------------------------------------



def active_universe_by_date(mapping_df: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    if mapping_df.empty:
        return pd.DataFrame(columns=["asof", "active_universe_size"])
    if "valid_from" not in mapping_df.columns and "valid_to" not in mapping_df.columns:
        size = int(mapping_df["symbol"].dropna().nunique())
        return pd.DataFrame({"asof": pd.Series(pd.unique(dates.dropna())).sort_values(), "active_universe_size": size})

    work = mapping_df.copy()
    if "valid_from" not in work.columns:
        work["valid_from"] = pd.Timestamp("1900-01-01", tz="UTC")
    if "valid_to" not in work.columns:
        work["valid_to"] = pd.Timestamp("2262-04-11", tz="UTC")
    work["valid_from"] = work["valid_from"].fillna(pd.Timestamp("1900-01-01", tz="UTC"))
    work["valid_to"] = work["valid_to"].fillna(pd.Timestamp("2262-04-11", tz="UTC"))

    out_rows: List[Dict[str, Any]] = []
    for d in pd.Series(pd.unique(dates.dropna())).sort_values():
        active = work.loc[(work["valid_from"] <= d) & (d <= work["valid_to"]), "symbol"].dropna().nunique()
        out_rows.append({"asof": d, "active_universe_size": int(active)})
    return pd.DataFrame(out_rows)



def run_coverage_checks(
    tables: Mapping[str, pd.DataFrame],
    cfg: EdgarQCConfig,
    checks: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    mapping = tables["ticker_cik_outputs"]
    submissions = tables["submissions_raw"]
    parsed = tables["parsed_facts"]
    pit = tables["pit_store"]

    universe_size = int(mapping["symbol"].dropna().nunique()) if not mapping.empty else 0
    if universe_size == 0:
        checks.append(
            make_check(
                stage="coverage",
                check_name="empty_mapping_universe",
                value=0,
                threshold_fail=1,
                severity="critical",
                blocking=True,
                passed=False,
                message="Mapping universe is empty.",
            )
        )
        failures.append(
            make_failure(
                stage="coverage",
                check_name="empty_mapping_universe",
                severity="critical",
                blocking=True,
                issue="universe_empty",
            )
        )
        return {"universe_size": 0}, pd.DataFrame(columns=["asof"])

    stage_symbols = {
        "map": mapping["symbol"].dropna().nunique(),
        "submissions": submissions["symbol"].dropna().nunique(),
        "facts": parsed["symbol"].dropna().nunique(),
        "pit": pit["symbol"].dropna().nunique(),
    }
    stage_thresholds = {
        "map": (cfg.coverage.map_warn_min, cfg.coverage.map_fail_min),
        "submissions": (cfg.coverage.submissions_warn_min, cfg.coverage.submissions_fail_min),
        "facts": (cfg.coverage.facts_warn_min, cfg.coverage.facts_fail_min),
        "pit": (cfg.coverage.pit_warn_min, cfg.coverage.pit_fail_min),
    }

    metrics: Dict[str, Any] = {"universe_size": universe_size}
    for stage_name, count in stage_symbols.items():
        ratio = float(count / universe_size) if universe_size else np.nan
        warn_thr, fail_thr = stage_thresholds[stage_name]
        severity = "info"
        passed = True
        blocking = False
        if ratio < fail_thr:
            severity = "critical" if stage_name == "pit" else "warn"
            blocking = stage_name == "pit" and ratio < cfg.coverage.absolute_pit_fail_min
            passed = False
        elif ratio < warn_thr:
            severity = "warn"
            passed = False
        checks.append(
            make_check(
                stage="coverage",
                check_name=f"stage_coverage::{stage_name}",
                value=ratio,
                threshold_warn=warn_thr,
                threshold_fail=fail_thr,
                severity=severity,
                blocking=blocking,
                passed=passed,
                message=f"Coverage at stage '{stage_name}' computed over unique symbols.",
                dimensions={"stage_symbol_count": int(count), "universe_size": universe_size},
            )
        )
        if not passed:
            failures.append(
                make_failure(
                    stage="coverage",
                    check_name=f"stage_coverage::{stage_name}",
                    severity=severity,
                    blocking=blocking,
                    issue="coverage_below_threshold",
                    value=ratio,
                    threshold=fail_thr if ratio < fail_thr else warn_thr,
                    details={"stage": stage_name, "symbol_count": int(count), "universe_size": universe_size},
                )
            )
        metrics[f"coverage::{stage_name}"] = ratio

    daily = pd.DataFrame(columns=["asof"])
    if not pit.empty and "asof" in pit.columns:
        active = active_universe_by_date(mapping, pit["asof"])
        observed = (
            pit.groupby("asof", dropna=False)["symbol"]
            .nunique()
            .rename("pit_symbol_count")
            .reset_index()
        )
        daily = observed.merge(active, on="asof", how="left")
        daily["active_universe_size"] = daily["active_universe_size"].fillna(universe_size)
        daily["pit_coverage"] = np.where(
            daily["active_universe_size"] > 0,
            daily["pit_symbol_count"] / daily["active_universe_size"],
            np.nan,
        )
        daily_warn_share = float((daily["pit_coverage"] < cfg.coverage.daily_warn_min).mean()) if len(daily) else 0.0
        daily_fail_share = float((daily["pit_coverage"] < cfg.coverage.daily_fail_min).mean()) if len(daily) else 0.0
        sev = "info"
        passed = True
        if daily_fail_share > 0.0:
            sev = "warn"
            passed = False
        checks.append(
            make_check(
                stage="coverage",
                check_name="daily_pit_coverage_share_below_warn",
                value=daily_warn_share,
                threshold_warn=0.0,
                threshold_fail=0.0,
                severity=sev,
                blocking=False,
                passed=passed,
                message="Share of PIT dates below daily coverage thresholds.",
                dimensions={"daily_fail_share": daily_fail_share},
            )
        )
        metrics["daily_pit_coverage_warn_share"] = daily_warn_share
        metrics["daily_pit_coverage_fail_share"] = daily_fail_share

        strat_cols = [c for c in ("sector", "size_bucket", "liquidity_decile") if c in mapping.columns]
        for col in strat_cols:
            base = mapping[["symbol", col]].dropna().drop_duplicates()
            obs = pit[["symbol", col]].dropna().drop_duplicates() if col in pit.columns else pit[["symbol"]].merge(base, on="symbol", how="left")
            expected = base.groupby(col)["symbol"].nunique().rename("expected")
            observed_g = obs.groupby(col)["symbol"].nunique().rename("observed")
            strat = pd.concat([expected, observed_g], axis=1).fillna(0.0)
            strat["ratio"] = np.where(strat["expected"] > 0, strat["observed"] / strat["expected"], np.nan)
            min_ratio = float(strat["ratio"].min()) if len(strat) else np.nan
            sev = "info"
            passed = True
            blocking = False
            if len(strat) and min_ratio < cfg.coverage.stratified_fail_min:
                sev = "warn"
                passed = False
            elif len(strat) and min_ratio < cfg.coverage.stratified_warn_min:
                sev = "warn"
                passed = False
            checks.append(
                make_check(
                    stage="coverage",
                    check_name=f"stratified_min_coverage::{col}",
                    value=min_ratio,
                    threshold_warn=cfg.coverage.stratified_warn_min,
                    threshold_fail=cfg.coverage.stratified_fail_min,
                    severity=sev,
                    blocking=blocking,
                    passed=passed,
                    message=f"Minimum stratified PIT coverage over {col} buckets.",
                )
            )
            if not passed:
                worst = strat.sort_values("ratio").head(25)
                for bucket, row in worst.iterrows():
                    failures.append(
                        make_failure(
                            stage="coverage",
                            check_name=f"stratified_min_coverage::{col}",
                            severity=sev,
                            blocking=blocking,
                            issue="stratified_coverage_below_threshold",
                            value=float(row["ratio"]),
                            threshold=cfg.coverage.stratified_fail_min,
                            details={"dimension": col, "bucket": bucket, "expected": int(row["expected"]), "observed": int(row["observed"])} ,
                        )
                    )
            metrics[f"stratified_min_coverage::{col}"] = min_ratio

    return metrics, daily


# -----------------------------------------------------------------------------
# Identity checks
# -----------------------------------------------------------------------------



def _prepare_mapping_intervals(mapping: pd.DataFrame) -> pd.DataFrame:
    work = mapping.copy()
    if "valid_from" not in work.columns:
        work["valid_from"] = pd.Timestamp("1900-01-01", tz="UTC")
    if "valid_to" not in work.columns:
        work["valid_to"] = pd.Timestamp("2262-04-11", tz="UTC")
    work["valid_from"] = work["valid_from"].fillna(pd.Timestamp("1900-01-01", tz="UTC"))
    work["valid_to"] = work["valid_to"].fillna(pd.Timestamp("2262-04-11", tz="UTC"))
    return work



def _find_symbol_cik_conflicts(mapping: pd.DataFrame) -> pd.DataFrame:
    cols = ["symbol", "cik", "other_cik", "valid_from", "valid_to"]
    if mapping.empty:
        return pd.DataFrame(columns=cols)
    work = _prepare_mapping_intervals(mapping)
    rows: List[Dict[str, Any]] = []
    for symbol, g in work.groupby("symbol"):
        g = g.sort_values(["valid_from", "valid_to", "cik"]).reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                r1 = g.iloc[i]
                r2 = g.iloc[j]
                if r1["cik"] == r2["cik"]:
                    continue
                overlap = max(r1["valid_from"], r2["valid_from"]) <= min(r1["valid_to"], r2["valid_to"])
                if overlap:
                    rows.append(
                        {
                            "symbol": symbol,
                            "cik": r1["cik"],
                            "other_cik": r2["cik"],
                            "valid_from": max(r1["valid_from"], r2["valid_from"]),
                            "valid_to": min(r1["valid_to"], r2["valid_to"]),
                        }
                    )
    return pd.DataFrame(rows, columns=cols)



def resolve_active_mapping(mapping: pd.DataFrame, asof: pd.Series) -> pd.DataFrame:
    if mapping.empty:
        return pd.DataFrame(columns=["symbol", "asof", "cik_expected"])
    work = _prepare_mapping_intervals(mapping)
    dates = pd.Series(pd.unique(asof.dropna())).sort_values()
    rows: List[Dict[str, Any]] = []
    for d in dates:
        active = work.loc[(work["valid_from"] <= d) & (d <= work["valid_to"]), ["symbol", "cik"]].drop_duplicates()
        active = active.rename(columns={"cik": "cik_expected"})
        active["asof"] = d
        rows.append(active)
    if not rows:
        return pd.DataFrame(columns=["symbol", "asof", "cik_expected"])
    return pd.concat(rows, ignore_index=True)



def run_identity_checks(
    tables: Mapping[str, pd.DataFrame],
    cfg: EdgarQCConfig,
    checks: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    mapping = tables["ticker_cik_outputs"]
    parsed = tables["parsed_facts"]
    pit = tables["pit_store"]
    metrics: Dict[str, Any] = {}

    mapping_conflicts = _find_symbol_cik_conflicts(mapping)
    conflict_share = float(mapping_conflicts["symbol"].nunique() / max(mapping["symbol"].nunique(), 1)) if not mapping.empty else 0.0
    sev = "info"
    blocking = False
    passed = True
    if conflict_share > cfg.identity.max_mapping_conflict_share_fail:
        sev = "critical"
        blocking = True
        passed = False
    elif conflict_share > cfg.identity.max_mapping_conflict_share_warn:
        sev = "warn"
        passed = False
    checks.append(
        make_check(
            stage="identity",
            check_name="active_symbol_cik_conflict_share",
            value=conflict_share,
            threshold_warn=cfg.identity.max_mapping_conflict_share_warn,
            threshold_fail=cfg.identity.max_mapping_conflict_share_fail,
            severity=sev,
            blocking=blocking,
            passed=passed,
            message="Share of symbols with overlapping active CIK mappings.",
            dimensions={"conflict_count": int(len(mapping_conflicts))},
        )
    )
    for _, row in mapping_conflicts.head(200).iterrows():
        failures.append(
            make_failure(
                stage="identity",
                check_name="active_symbol_cik_conflict_share",
                severity=sev,
                blocking=blocking,
                issue="overlapping_active_mapping",
                symbol=row["symbol"],
                cik=row["cik"],
                value=1,
                threshold=0,
                details={"other_cik": row["other_cik"], "valid_from": row["valid_from"], "valid_to": row["valid_to"]},
            )
        )
    metrics["identity_conflict_share"] = conflict_share

    if not cfg.identity.allow_multi_ticker_per_cik and not mapping.empty:
        work = _prepare_mapping_intervals(mapping)
        by_cik = work.groupby("cik")["symbol"].nunique()
        multi = by_cik[by_cik > 1]
        share = float(len(multi) / max(len(by_cik), 1))
        sev = "critical" if share > 0 else "info"
        checks.append(
            make_check(
                stage="identity",
                check_name="multi_ticker_per_cik",
                value=share,
                threshold_fail=0.0,
                severity=sev,
                blocking=share > 0,
                passed=share == 0,
                message="Multiple active tickers per CIK are not allowed by config.",
                dimensions={"multi_cik_count": int(len(multi))},
            )
        )
        for cik, count in multi.head(200).items():
            failures.append(
                make_failure(
                    stage="identity",
                    check_name="multi_ticker_per_cik",
                    severity="critical",
                    blocking=True,
                    issue="multiple_tickers_for_cik",
                    cik=cik,
                    value=int(count),
                    threshold=1,
                )
            )
        metrics["multi_ticker_per_cik_share"] = share

    if not pit.empty and "asof" in pit.columns:
        active = resolve_active_mapping(mapping, pit["asof"])
        if not active.empty:
            joined = pit[["symbol", "cik", "asof"]].merge(active, on=["symbol", "asof"], how="left")
            mismatch = joined.loc[joined["cik_expected"].notna() & (joined["cik"] != joined["cik_expected"])]
            mismatch_share = float(len(mismatch) / max(len(joined), 1))
        else:
            mismatch = pd.DataFrame(columns=["symbol", "cik", "asof", "cik_expected"])
            mismatch_share = 0.0
        sev = "info"
        blocking = False
        passed = True
        if mismatch_share > cfg.identity.max_table_mismatch_share_fail:
            sev = "critical"
            blocking = True
            passed = False
        elif mismatch_share > cfg.identity.max_table_mismatch_share_warn:
            sev = "warn"
            passed = False
        checks.append(
            make_check(
                stage="identity",
                check_name="pit_vs_mapping_cik_mismatch_share",
                value=mismatch_share,
                threshold_warn=cfg.identity.max_table_mismatch_share_warn,
                threshold_fail=cfg.identity.max_table_mismatch_share_fail,
                severity=sev,
                blocking=blocking,
                passed=passed,
                message="Share of PIT rows whose active mapping CIK disagrees with PIT CIK.",
                dimensions={"mismatch_count": int(len(mismatch))},
            )
        )
        for _, row in mismatch.head(200).iterrows():
            failures.append(
                make_failure(
                    stage="identity",
                    check_name="pit_vs_mapping_cik_mismatch_share",
                    severity=sev,
                    blocking=blocking,
                    issue="pit_mapping_cik_mismatch",
                    symbol=row.get("symbol"),
                    cik=row.get("cik"),
                    asof=row.get("asof"),
                    details={"expected_cik": row.get("cik_expected")},
                )
            )
        metrics["pit_mapping_mismatch_share"] = mismatch_share

    if not parsed.empty and "acceptance_ts" in parsed.columns:
        accepted_active = resolve_active_mapping(mapping, parsed["acceptance_ts"].rename("asof"))
        if not accepted_active.empty:
            joined_p = parsed[["symbol", "cik", "acceptance_ts"]].rename(columns={"acceptance_ts": "asof"}).merge(accepted_active, on=["symbol", "asof"], how="left")
            mismatch_p = joined_p.loc[joined_p["cik_expected"].notna() & (joined_p["cik"] != joined_p["cik_expected"])]
            metrics["parsed_mapping_mismatch_share"] = float(len(mismatch_p) / max(len(joined_p), 1))

    return metrics


# -----------------------------------------------------------------------------
# Download / parse checks
# -----------------------------------------------------------------------------



def run_download_parse_checks(
    tables: Mapping[str, pd.DataFrame],
    checks: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    submissions = tables["submissions_raw"]
    companyfacts = tables["companyfacts_raw"]
    parsed = tables["parsed_facts"]
    metrics: Dict[str, Any] = {}

    if "status_code" in submissions.columns and not submissions.empty:
        status = pd.to_numeric(submissions["status_code"], errors="coerce")
        permanent = int(status.eq(404).sum())
        transient = int(status.isin([429, 500, 502, 503, 504]).sum())
        total = int(status.notna().sum())
        fail_rate = float(1.0 - ((total - transient - permanent) / max(total, 1))) if total > 0 else 0.0
        severity = "warn" if fail_rate > 0.05 else "info"
        checks.append(
            make_check(
                stage="download",
                check_name="submission_download_failure_rate",
                value=fail_rate,
                threshold_warn=0.05,
                threshold_fail=0.15,
                severity="critical" if fail_rate > 0.15 else severity,
                blocking=fail_rate > 0.15,
                passed=fail_rate <= 0.05,
                message="Failure rate over explicit submission download status codes.",
                dimensions={"permanent_404": permanent, "transient": transient, "status_rows": total},
            )
        )
        metrics["submission_download_failure_rate"] = fail_rate

    raw_count = int(len(companyfacts))
    parsed_count = int(len(parsed))
    parse_success = float(parsed_count / max(raw_count, 1)) if raw_count else np.nan
    sev = "info"
    passed = True
    if raw_count > 0 and parse_success < 0.50:
        sev = "warn"
        passed = False
    checks.append(
        make_check(
            stage="parse",
            check_name="parse_success_ratio",
            value=parse_success,
            threshold_warn=0.75,
            threshold_fail=0.50,
            severity="critical" if raw_count > 0 and parse_success < 0.50 else sev,
            blocking=raw_count > 0 and parse_success < 0.40,
            passed=passed,
            message="Ratio of canonical parsed facts to raw company facts rows.",
            dimensions={"raw_rows": raw_count, "parsed_rows": parsed_count},
        )
    )
    metrics["parse_success_ratio"] = parse_success

    if not parsed.empty:
        bad_units = int(parsed["unit"].isna().sum()) if "unit" in parsed.columns else len(parsed)
        share = float(bad_units / max(len(parsed), 1))
        sev = "warn" if share > 0.01 else "info"
        checks.append(
            make_check(
                stage="parse",
                check_name="missing_unit_share",
                value=share,
                threshold_warn=0.01,
                threshold_fail=0.05,
                severity="critical" if share > 0.05 else sev,
                blocking=share > 0.10,
                passed=share <= 0.01,
                message="Share of parsed facts without canonical units.",
            )
        )
        metrics["parsed_missing_unit_share"] = share

    return metrics


# -----------------------------------------------------------------------------
# Temporal checks
# -----------------------------------------------------------------------------



def run_temporal_checks(
    tables: Mapping[str, pd.DataFrame],
    checks: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    pit = tables["pit_store"]
    metrics: Dict[str, Any] = {}

    if pit.empty:
        return metrics

    leakage = pit.loc[pit["acceptance_ts"] > pit["asof"]].copy()
    leakage_count = int(len(leakage))
    checks.append(
        make_check(
            stage="temporal",
            check_name="pit_leakage_count",
            value=leakage_count,
            threshold_fail=0,
            severity="critical" if leakage_count > 0 else "info",
            blocking=leakage_count > 0,
            passed=leakage_count == 0,
            message="Rows with acceptance timestamp later than PIT asof.",
        )
    )
    for idx, row in leakage.head(500).iterrows():
        failures.append(
            make_failure(
                stage="temporal",
                check_name="pit_leakage_count",
                severity="critical",
                blocking=True,
                issue="acceptance_after_asof",
                symbol=row.get("symbol"),
                cik=row.get("cik"),
                metric=row.get("metric"),
                asof=row.get("asof"),
                row_index=int(idx),
                details={"acceptance_ts": row.get("acceptance_ts")},
            )
        )
    metrics["pit_leakage_count"] = leakage_count

    bad_period = pit.loc[pit["period_end"] > pit["acceptance_ts"]].copy()
    bad_count = int(len(bad_period))
    checks.append(
        make_check(
            stage="temporal",
            check_name="period_end_after_acceptance_count",
            value=bad_count,
            threshold_warn=0,
            threshold_fail=0,
            severity="critical" if bad_count > 0 else "info",
            blocking=bad_count > 0,
            passed=bad_count == 0,
            message="Rows whose period_end is later than the acceptance timestamp.",
        )
    )
    for idx, row in bad_period.head(500).iterrows():
        failures.append(
            make_failure(
                stage="temporal",
                check_name="period_end_after_acceptance_count",
                severity="critical",
                blocking=True,
                issue="period_end_after_acceptance",
                symbol=row.get("symbol"),
                cik=row.get("cik"),
                metric=row.get("metric"),
                asof=row.get("asof"),
                row_index=int(idx),
            )
        )
    metrics["period_end_after_acceptance_count"] = bad_count

    comp_keys = [c for c in ["symbol", "metric", "asof"] if c in pit.columns]
    if comp_keys:
        dup = pit.duplicated(subset=comp_keys, keep=False)
        dup_count = int(dup.sum())
        sev = "critical" if dup_count > 0 else "info"
        checks.append(
            make_check(
                stage="temporal",
                check_name="non_deterministic_candidate_resolution",
                value=dup_count,
                threshold_fail=0,
                severity=sev,
                blocking=dup_count > 0,
                passed=dup_count == 0,
                message="Duplicate PIT candidates remain for (symbol, metric, asof).",
                dimensions={"key_cols": comp_keys},
            )
        )
        if dup_count > 0:
            for _, row in pit.loc[dup, comp_keys + [c for c in ["cik", "period_end", "source_filing_id", "acceptance_ts"] if c in pit.columns]].head(500).iterrows():
                failures.append(
                    make_failure(
                        stage="temporal",
                        check_name="non_deterministic_candidate_resolution",
                        severity="critical",
                        blocking=True,
                        issue="duplicate_pit_candidate",
                        symbol=row.get("symbol"),
                        cik=row.get("cik"),
                        metric=row.get("metric"),
                        asof=row.get("asof"),
                        details=row.to_dict(),
                    )
                )
        metrics["pit_duplicate_candidate_count"] = dup_count

    return metrics


# -----------------------------------------------------------------------------
# Value quality checks
# -----------------------------------------------------------------------------



def robust_zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    med = x.median(skipna=True)
    mad = (x - med).abs().median(skipna=True)
    denom = 1.4826 * mad
    if pd.isna(denom) or denom <= 0:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - med) / denom



def metric_family(metric: Optional[str]) -> str:
    m = normalize_metric(metric)
    if m is None:
        return "other"
    if m in {"assets", "liabilities", "equity", "current_assets", "current_liabilities"}:
        return "balance_sheet"
    if m in {"net_income", "revenue"}:
        return "income_statement"
    if "cash" in m:
        return "cash_flow"
    if "shares" in m:
        return "share_count"
    return "other"



def run_value_checks(
    tables: Mapping[str, pd.DataFrame],
    cfg: EdgarQCConfig,
    checks: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    pit = tables["pit_store"]
    metrics: Dict[str, Any] = {}
    if pit.empty:
        return metrics

    work = pit.copy()
    work["metric_norm"] = work["metric"].map(normalize_metric)
    work["value_num"] = pd.to_numeric(work["value"], errors="coerce")

    outlier_rows: List[pd.Index] = []
    metric_outlier_share: Dict[str, float] = {}
    for metric, g in work.groupby("metric_norm"):
        if metric is None or len(g) < 5:
            continue
        rz = robust_zscore(g["value_num"]).abs()
        severe = rz > cfg.value_quality.robust_z_fail
        warn = rz > cfg.value_quality.robust_z_warn
        share = float(warn.mean())
        metric_outlier_share[str(metric)] = share
        if warn.any():
            outlier_rows.append(g.index[warn])
    if outlier_rows:
        outlier_idx = pd.Index(np.concatenate([idx.to_numpy() for idx in outlier_rows]))
    else:
        outlier_idx = pd.Index([], dtype=int)
    outlier_share = float(len(outlier_idx.unique()) / max(len(work), 1))
    sev = "info"
    passed = True
    if outlier_share > cfg.value_quality.outlier_share_fail:
        sev = "warn"
        passed = False
    elif outlier_share > cfg.value_quality.outlier_share_warn:
        sev = "warn"
        passed = False
    checks.append(
        make_check(
            stage="value_quality",
            check_name="robust_outlier_share",
            value=outlier_share,
            threshold_warn=cfg.value_quality.outlier_share_warn,
            threshold_fail=cfg.value_quality.outlier_share_fail,
            severity=sev,
            blocking=False,
            passed=passed,
            message="Share of PIT rows exceeding robust metric-wise z-score thresholds.",
        )
    )
    for idx in outlier_idx.unique()[:500]:
        row = work.loc[idx]
        failures.append(
            make_failure(
                stage="value_quality",
                check_name="robust_outlier_share",
                severity="warn",
                blocking=False,
                issue="robust_outlier",
                symbol=row.get("symbol"),
                cik=row.get("cik"),
                metric=row.get("metric_norm"),
                asof=row.get("asof"),
                row_index=int(idx),
                value=row.get("value_num"),
            )
        )
    metrics["robust_outlier_share"] = outlier_share

    positive_set = {normalize_metric(m) for m in cfg.positive_only_metrics}
    nonneg_set = {normalize_metric(m) for m in cfg.nonnegative_metrics}
    negative_mask = pd.Series(False, index=work.index)
    negative_mask |= work["metric_norm"].isin(list(positive_set)) & (work["value_num"] <= 0)
    negative_mask |= work["metric_norm"].isin(list(nonneg_set)) & (work["value_num"] < 0)
    negative_share = float(negative_mask.mean()) if len(work) else 0.0
    sev = "info"
    passed = True
    if negative_share > cfg.value_quality.negative_share_fail:
        sev = "warn"
        passed = False
    elif negative_share > cfg.value_quality.negative_share_warn:
        sev = "warn"
        passed = False
    checks.append(
        make_check(
            stage="value_quality",
            check_name="sign_rule_violation_share",
            value=negative_share,
            threshold_warn=cfg.value_quality.negative_share_warn,
            threshold_fail=cfg.value_quality.negative_share_fail,
            severity=sev,
            blocking=False,
            passed=passed,
            message="Share of rows violating metric sign constraints.",
        )
    )
    for idx, row in work.loc[negative_mask].head(500).iterrows():
        failures.append(
            make_failure(
                stage="value_quality",
                check_name="sign_rule_violation_share",
                severity="warn",
                issue="sign_rule_violation",
                symbol=row.get("symbol"),
                cik=row.get("cik"),
                metric=row.get("metric_norm"),
                asof=row.get("asof"),
                row_index=int(idx),
                value=row.get("value_num"),
            )
        )
    metrics["sign_rule_violation_share"] = negative_share

    # Accounting residual where metrics coexist.
    needed = {"assets", "liabilities", "equity"}
    pivotable = work.loc[work["metric_norm"].isin(needed), ["symbol", "asof", "period_end", "metric_norm", "value_num"]]
    if not pivotable.empty:
        pivot = pivotable.drop_duplicates(subset=["symbol", "asof", "period_end", "metric_norm"], keep="last")
        wide = pivot.pivot_table(index=["symbol", "asof", "period_end"], columns="metric_norm", values="value_num", aggfunc="last")
        available = wide.dropna(subset=["assets", "liabilities", "equity"], how="any") if all(c in wide.columns for c in needed) else pd.DataFrame()
        if not available.empty:
            residual = (available["assets"] - available["liabilities"] - available["equity"]).abs() / (1.0 + available["assets"].abs())
            resid_share_warn = float((residual > cfg.value_quality.accounting_residual_warn).mean())
            resid_share_fail = float((residual > cfg.value_quality.accounting_residual_fail).mean())
            sev = "warn" if resid_share_warn > 0 else "info"
            checks.append(
                make_check(
                    stage="value_quality",
                    check_name="accounting_residual_share",
                    value=resid_share_warn,
                    threshold_warn=cfg.value_quality.accounting_residual_warn,
                    threshold_fail=cfg.value_quality.accounting_residual_fail,
                    severity="critical" if resid_share_fail > 0 else sev,
                    blocking=False,
                    passed=resid_share_warn == 0,
                    message="Share of triads with material accounting residuals.",
                    dimensions={"residual_fail_share": resid_share_fail},
                )
            )
            metrics["accounting_residual_share"] = resid_share_warn
            bad = residual[residual > cfg.value_quality.accounting_residual_warn].sort_values(ascending=False).head(500)
            for idx, val in bad.items():
                sym, asof, period_end = idx
                failures.append(
                    make_failure(
                        stage="value_quality",
                        check_name="accounting_residual_share",
                        severity="warn" if val <= cfg.value_quality.accounting_residual_fail else "critical",
                        issue="accounting_residual_large",
                        symbol=sym,
                        asof=asof,
                        value=float(val),
                        threshold=cfg.value_quality.accounting_residual_warn,
                        details={"period_end": period_end},
                    )
                )

    return metrics


# -----------------------------------------------------------------------------
# Staleness / lineage checks
# -----------------------------------------------------------------------------



def staleness_thresholds_for_metric(metric: Optional[str], cfg: EdgarQCConfig) -> Tuple[int, int]:
    m = normalize_metric(metric)
    if m in cfg.staleness.exact_warn_days:
        return cfg.staleness.exact_warn_days[m], cfg.staleness.exact_fail_days.get(m, cfg.staleness.default_fail_days)
    fam = metric_family(m)
    return (
        cfg.staleness.family_warn_days.get(fam, cfg.staleness.default_warn_days),
        cfg.staleness.family_fail_days.get(fam, cfg.staleness.default_fail_days),
    )



def run_staleness_checks(
    tables: Mapping[str, pd.DataFrame],
    cfg: EdgarQCConfig,
    checks: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    pit = tables["pit_store"]
    metrics: Dict[str, Any] = {}
    if pit.empty:
        return metrics

    work = pit.copy()
    work["metric_norm"] = work["metric"].map(normalize_metric)
    age_days = (work["asof"] - work["acceptance_ts"]).dt.days
    work["staleness_days"] = age_days.astype("float")

    warn_flags = pd.Series(False, index=work.index)
    fail_flags = pd.Series(False, index=work.index)
    metric_rows: List[Dict[str, Any]] = []
    for metric, g in work.groupby("metric_norm"):
        warn_thr, fail_thr = staleness_thresholds_for_metric(metric, cfg)
        warn = g["staleness_days"] > warn_thr
        fail = g["staleness_days"] > fail_thr
        warn_flags.loc[g.index] = warn
        fail_flags.loc[g.index] = fail
        metric_rows.append(
            {
                "metric": metric,
                "median_staleness_days": float(g["staleness_days"].median()),
                "warn_share": float(warn.mean()),
                "fail_share": float(fail.mean()),
                "warn_thr": warn_thr,
                "fail_thr": fail_thr,
            }
        )

    stale_warn_share = float(warn_flags.mean()) if len(work) else 0.0
    stale_fail_share = float(fail_flags.mean()) if len(work) else 0.0
    sev = "info"
    passed = True
    if stale_fail_share > 0.0:
        sev = "warn"
        passed = False
    elif stale_warn_share > 0.0:
        sev = "warn"
        passed = False
    checks.append(
        make_check(
            stage="staleness",
            check_name="row_staleness_share",
            value=stale_warn_share,
            threshold_warn=0.0,
            threshold_fail=0.0,
            severity=sev,
            blocking=False,
            passed=passed,
            message="Share of PIT rows above metric-specific staleness warning thresholds.",
            dimensions={"stale_fail_share": stale_fail_share},
        )
    )
    metrics["staleness_warn_share"] = stale_warn_share
    metrics["staleness_fail_share"] = stale_fail_share

    critical_set = {normalize_metric(m) for m in cfg.critical_metrics}
    if critical_set:
        crit = work[work["metric_norm"].isin(critical_set)]
        if not crit.empty:
            crit_fail_share = float((fail_flags.loc[crit.index]).mean())
            sev = "critical" if crit_fail_share > cfg.staleness.essential_metric_fail_share else ("warn" if crit_fail_share > cfg.staleness.essential_metric_warn_share else "info")
            checks.append(
                make_check(
                    stage="staleness",
                    check_name="critical_metric_fail_share",
                    value=crit_fail_share,
                    threshold_warn=cfg.staleness.essential_metric_warn_share,
                    threshold_fail=cfg.staleness.essential_metric_fail_share,
                    severity=sev,
                    blocking=crit_fail_share > cfg.staleness.essential_metric_fail_share,
                    passed=crit_fail_share <= cfg.staleness.essential_metric_warn_share,
                    message="Fail-share over critical metrics only.",
                )
            )
            metrics["critical_metric_staleness_fail_share"] = crit_fail_share

    worst = work.loc[warn_flags].sort_values("staleness_days", ascending=False).head(500)
    for idx, row in worst.iterrows():
        warn_thr, fail_thr = staleness_thresholds_for_metric(row.get("metric_norm"), cfg)
        failures.append(
            make_failure(
                stage="staleness",
                check_name="row_staleness_share",
                severity="critical" if row["staleness_days"] > fail_thr else "warn",
                issue="stale_metric",
                symbol=row.get("symbol"),
                cik=row.get("cik"),
                metric=row.get("metric_norm"),
                asof=row.get("asof"),
                row_index=int(idx),
                value=float(row["staleness_days"]),
                threshold=warn_thr,
            )
        )

    return metrics



def run_lineage_checks(
    tables: Mapping[str, pd.DataFrame],
    cfg: EdgarQCConfig,
    checks: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    pit = tables["pit_store"]
    metrics: Dict[str, Any] = {}
    if pit.empty:
        return metrics

    required = [c for c in cfg.required_lineage_columns if c in pit.columns]
    base_required = [c for c in ["source_filing_id", "metric", "unit", "acceptance_ts", "period_end", "run_id"] if c in pit.columns]
    use_cols = sorted(set(required + base_required))
    traceable = pit[use_cols].notna().all(axis=1) if use_cols else pd.Series(False, index=pit.index)
    traceability = float(traceable.mean()) if len(pit) else 0.0
    sev = "info"
    blocking = False
    passed = True
    if traceability < cfg.lineage.min_traceability_fail:
        sev = "critical"
        blocking = True
        passed = False
    elif traceability < cfg.lineage.min_traceability_warn:
        sev = "warn"
        passed = False
    checks.append(
        make_check(
            stage="lineage",
            check_name="traceability_rate",
            value=traceability,
            threshold_warn=cfg.lineage.min_traceability_warn,
            threshold_fail=cfg.lineage.min_traceability_fail,
            severity=sev,
            blocking=blocking,
            passed=passed,
            message="Fraction of PIT rows with complete lineage payload.",
            dimensions={"required_lineage_cols": use_cols},
        )
    )
    metrics["traceability_rate"] = traceability

    work = pit.copy()
    work["metric_norm"] = work["metric"].map(normalize_metric)
    critical_set = {normalize_metric(m) for m in cfg.critical_metrics}
    crit_missing = work.loc[work["metric_norm"].isin(critical_set) & (~traceable)]
    crit_missing_share = float(len(crit_missing) / max(len(work.loc[work["metric_norm"].isin(critical_set)]), 1)) if critical_set else 0.0
    sev = "critical" if crit_missing_share > cfg.lineage.critical_metric_lineage_fail else "info"
    checks.append(
        make_check(
            stage="lineage",
            check_name="critical_metric_lineage_missing_share",
            value=crit_missing_share,
            threshold_fail=cfg.lineage.critical_metric_lineage_fail,
            severity=sev,
            blocking=crit_missing_share > cfg.lineage.critical_metric_lineage_fail,
            passed=crit_missing_share <= cfg.lineage.critical_metric_lineage_fail,
            message="Share of critical-metric rows missing required lineage.",
        )
    )
    metrics["critical_metric_lineage_missing_share"] = crit_missing_share

    for idx, row in work.loc[~traceable].head(500).iterrows():
        missing = [c for c in use_cols if pd.isna(row.get(c))]
        failures.append(
            make_failure(
                stage="lineage",
                check_name="traceability_rate",
                severity="critical" if row.get("metric_norm") in critical_set else "warn",
                blocking=row.get("metric_norm") in critical_set,
                issue="missing_lineage",
                symbol=row.get("symbol"),
                cik=row.get("cik"),
                metric=row.get("metric_norm"),
                asof=row.get("asof"),
                row_index=int(idx),
                details={"missing_columns": missing},
            )
        )

    return metrics


# -----------------------------------------------------------------------------
# Aggregation / daily metrics / manifest
# -----------------------------------------------------------------------------



def build_daily_metrics(
    tables: Mapping[str, pd.DataFrame],
    coverage_daily: pd.DataFrame,
) -> pd.DataFrame:
    pit = tables["pit_store"]
    if pit.empty:
        return pd.DataFrame(columns=["asof"])

    work = pit.copy()
    work["asof_date"] = work["asof"].dt.floor("D")
    work["leakage_flag"] = work["acceptance_ts"] > work["asof"]
    work["staleness_days"] = (work["asof"] - work["acceptance_ts"]).dt.days.astype(float)
    lineage_cols = [c for c in ["source_filing_id", "form_type", "raw_artifact_id", "run_id"] if c in work.columns]
    work["traceable_flag"] = work[lineage_cols].notna().all(axis=1) if lineage_cols else False

    daily = (
        work.groupby("asof_date")
        .agg(
            pit_rows=("symbol", "size"),
            pit_symbol_count=("symbol", "nunique"),
            leakage_count=("leakage_flag", "sum"),
            median_staleness_days=("staleness_days", "median"),
            traceability_rate=("traceable_flag", "mean"),
        )
        .reset_index()
        .rename(columns={"asof_date": "asof"})
    )
    if not coverage_daily.empty:
        daily = daily.merge(coverage_daily, on="asof", how="left")
    return daily.sort_values("asof").reset_index(drop=True)



def aggregate_score(metrics: Mapping[str, Any], cfg: EdgarQCConfig) -> Tuple[float, Dict[str, float]]:
    losses: Dict[str, float] = {}
    cov_candidates = [
        metrics.get("coverage::submissions", 1.0),
        metrics.get("coverage::facts", 1.0),
        metrics.get("coverage::pit", 1.0),
    ]
    cov = float(np.nanmean([v for v in cov_candidates if v is not None])) if cov_candidates else 1.0
    losses["coverage"] = max(0.0, 1.0 - cov)

    identity_loss = max(
        float(metrics.get("identity_conflict_share", 0.0) or 0.0),
        float(metrics.get("pit_mapping_mismatch_share", 0.0) or 0.0),
        float(metrics.get("multi_ticker_per_cik_share", 0.0) or 0.0),
    )
    losses["identity"] = min(1.0, identity_loss * 10.0)

    pd_loss = 0.0
    parse_success = metrics.get("parse_success_ratio")
    if parse_success is not None and not pd.isna(parse_success):
        pd_loss = max(pd_loss, max(0.0, 1.0 - float(parse_success)))
    dl_fail = metrics.get("submission_download_failure_rate")
    if dl_fail is not None and not pd.isna(dl_fail):
        pd_loss = max(pd_loss, float(dl_fail))
    losses["parse_download"] = min(1.0, pd_loss)

    value_loss = max(
        float(metrics.get("robust_outlier_share", 0.0) or 0.0),
        float(metrics.get("sign_rule_violation_share", 0.0) or 0.0),
        float(metrics.get("accounting_residual_share", 0.0) or 0.0),
    )
    losses["value_quality"] = min(1.0, value_loss * 8.0)

    stale_loss = max(
        float(metrics.get("staleness_warn_share", 0.0) or 0.0),
        float(metrics.get("critical_metric_staleness_fail_share", 0.0) or 0.0),
    )
    losses["staleness"] = min(1.0, stale_loss * 5.0)

    traceability = float(metrics.get("traceability_rate", 1.0) or 1.0)
    crit_lineage = float(metrics.get("critical_metric_lineage_missing_share", 0.0) or 0.0)
    losses["lineage"] = min(1.0, max(1.0 - traceability, crit_lineage * 10.0))

    weighted_loss = (
        cfg.scoring.coverage * losses["coverage"]
        + cfg.scoring.identity * losses["identity"]
        + cfg.scoring.parse_download * losses["parse_download"]
        + cfg.scoring.value_quality * losses["value_quality"]
        + cfg.scoring.staleness * losses["staleness"]
        + cfg.scoring.lineage * losses["lineage"]
    )
    score = float(max(0.0, 100.0 - 100.0 * weighted_loss))
    return score, losses



def decide_gate(checks_df: pd.DataFrame, score: float, cfg: EdgarQCConfig) -> Tuple[str, List[str]]:
    if checks_df.empty:
        return "fail", ["No checks executed."]

    blocking_bad = checks_df.loc[checks_df["blocking"] & (~checks_df["passed"])]
    critical_bad = checks_df.loc[(checks_df["severity"] == "critical") & (~checks_df["passed"])]
    warn_bad = checks_df.loc[(checks_df["severity"] == "warn") & (~checks_df["passed"])]

    reasons: List[str] = []
    if not blocking_bad.empty:
        reasons.extend([f"blocking::{n}" for n in blocking_bad["check_name"].head(20).tolist()])
        return "fail", reasons
    if score < cfg.score_thresholds.fail_min:
        reasons.append(f"score<{cfg.score_thresholds.fail_min}")
        if not critical_bad.empty:
            reasons.extend([f"critical::{n}" for n in critical_bad["check_name"].head(10).tolist()])
        return "fail", reasons
    if score < cfg.score_thresholds.pass_min:
        reasons.append(f"score<{cfg.score_thresholds.pass_min}")
        if not warn_bad.empty:
            reasons.extend([f"warn::{n}" for n in warn_bad["check_name"].head(10).tolist()])
        return "warn", reasons
    if not critical_bad.empty or not warn_bad.empty:
        reasons.extend([f"warn::{n}" for n in pd.concat([critical_bad, warn_bad])["check_name"].head(20).tolist()])
        return "warn", reasons
    return "pass", ["all_checks_passed"]



def build_manifest(
    *,
    run_id: str,
    cfg: EdgarQCConfig,
    input_paths: Mapping[str, Path],
    output_dir: Path,
    gate: str,
    score: float,
    reasons: Sequence[str],
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "module": "data.edgar.edgar_qc",
        "generated_at_utc": utc_now_iso(),
        "gate": gate,
        "score": score,
        "decision_reasons": list(reasons),
        "config_version": cfg.config_version,
        "config_hash": hash_config(cfg),
        "git_revision": git_revision(),
        "python_version": platform.python_version(),
        "artifacts": {
            "summary": str(output_dir / "edgar_qc_summary.json"),
            "metrics": str(output_dir / "edgar_qc_metrics.parquet"),
            "failures": str(output_dir / "edgar_qc_failures.parquet"),
            "checks": str(output_dir / "edgar_qc_checks.parquet"),
            "manifest": str(output_dir / "manifest.json"),
        },
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in input_paths.items()
        },
    }


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------



def run_edgar_qc(
    *,
    ticker_cik_outputs_path: Path,
    submissions_raw_path: Path,
    companyfacts_raw_path: Path,
    parsed_facts_path: Path,
    pit_store_path: Path,
    output_dir: Path,
    config_path: Optional[Path] = None,
    run_id: str = "edgar_qc_run",
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    checks: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    input_paths = {
        "ticker_cik_outputs": ticker_cik_outputs_path,
        "submissions_raw": submissions_raw_path,
        "companyfacts_raw": companyfacts_raw_path,
        "parsed_facts": parsed_facts_path,
        "pit_store": pit_store_path,
    }

    tables: Dict[str, pd.DataFrame] = {}
    for name, path in input_paths.items():
        try:
            tables[name] = normalize_table(name, load_dataframe(path))
        except Exception as exc:
            checks.append(
                make_check(
                    stage="schema",
                    check_name=f"load_input::{name}",
                    value=0,
                    threshold_fail=1,
                    severity="critical",
                    blocking=True,
                    passed=False,
                    message=f"Failed to load input {name}: {exc}",
                    dimensions={"path": str(path)},
                )
            )
            failures.append(
                make_failure(
                    stage="schema",
                    check_name=f"load_input::{name}",
                    severity="critical",
                    blocking=True,
                    issue="input_load_failed",
                    details={"path": str(path), "error": str(exc)},
                )
            )
            tables[name] = pd.DataFrame()

    schema_ok, schema_metrics = run_schema_checks(tables, checks, failures)
    coverage_metrics, coverage_daily = run_coverage_checks(tables, cfg, checks, failures)
    identity_metrics = run_identity_checks(tables, cfg, checks, failures)
    parse_metrics = run_download_parse_checks(tables, checks, failures)
    temporal_metrics = run_temporal_checks(tables, checks, failures)
    value_metrics = run_value_checks(tables, cfg, checks, failures)
    staleness_metrics = run_staleness_checks(tables, cfg, checks, failures)
    lineage_metrics = run_lineage_checks(tables, cfg, checks, failures)

    all_metrics: Dict[str, Any] = {}
    for block in (
        schema_metrics,
        coverage_metrics,
        identity_metrics,
        parse_metrics,
        temporal_metrics,
        value_metrics,
        staleness_metrics,
        lineage_metrics,
    ):
        all_metrics.update(block)

    checks_df = pd.DataFrame(checks)
    failures_df = pd.DataFrame(failures)
    daily_metrics_df = build_daily_metrics(tables, coverage_daily)
    score, score_losses = aggregate_score(all_metrics, cfg)
    gate, reasons = decide_gate(checks_df, score, cfg)
    if not schema_ok and gate != "fail":
        gate = "fail"
        reasons = ["schema_failed"] + list(reasons)

    summary = {
        "run_id": run_id,
        "gate": gate,
        "score": score,
        "decision_reasons": list(reasons),
        "config_version": cfg.config_version,
        "config_hash": hash_config(cfg),
        "timestamp_utc": utc_now_iso(),
        "metrics": {k: coerce_jsonable(v) for k, v in sorted(all_metrics.items())},
        "score_losses": {k: float(v) for k, v in score_losses.items()},
        "counts": {
            "checks_total": int(len(checks_df)),
            "checks_failed": int((~checks_df["passed"]).sum()) if not checks_df.empty else 0,
            "failures_total": int(len(failures_df)),
            "critical_failures": int((failures_df["severity"] == "critical").sum()) if not failures_df.empty else 0,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "edgar_qc_summary.json", summary)
    write_parquet(output_dir / "edgar_qc_metrics.parquet", daily_metrics_df, index=cfg.persist_index)
    write_parquet(output_dir / "edgar_qc_failures.parquet", failures_df, index=cfg.persist_index)
    write_parquet(output_dir / "edgar_qc_checks.parquet", checks_df, index=cfg.persist_index)

    manifest = build_manifest(
        run_id=run_id,
        cfg=cfg,
        input_paths=input_paths,
        output_dir=output_dir,
        gate=gate,
        score=score,
        reasons=reasons,
    )
    write_json(output_dir / "manifest.json", manifest)

    return {
        "summary": summary,
        "metrics": daily_metrics_df,
        "failures": failures_df,
        "checks": checks_df,
        "manifest": manifest,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDGAR subsystem quality-control and risk gate.")
    parser.add_argument("--ticker-cik-outputs", required=True, type=Path, dest="ticker_cik_outputs")
    parser.add_argument("--submissions-raw", required=True, type=Path, dest="submissions_raw")
    parser.add_argument("--companyfacts-raw", required=True, type=Path, dest="companyfacts_raw")
    parser.add_argument("--parsed-facts", required=True, type=Path, dest="parsed_facts")
    parser.add_argument("--pit-store", required=True, type=Path, dest="pit_store")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default="edgar_qc_run")
    return parser.parse_args(argv)



def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = run_edgar_qc(
        ticker_cik_outputs_path=args.ticker_cik_outputs,
        submissions_raw_path=args.submissions_raw,
        companyfacts_raw_path=args.companyfacts_raw,
        parsed_facts_path=args.parsed_facts,
        pit_store_path=args.pit_store,
        output_dir=args.output_dir,
        config_path=args.config_path,
        run_id=args.run_id,
    )
    gate = result["summary"]["gate"]
    return 0 if gate in {"pass", "warn"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
