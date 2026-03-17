from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import requests

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_SUBMISSIONS_HISTORY_URL = "https://data.sec.gov/submissions/{filename}"
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
PERMANENT_STATUS_CODES = {400, 401, 403, 404, 410, 422}
REQUIRED_EVENT_FIELDS = (
    "cik",
    "accession_number",
    "form_type",
    "filing_date",
    "acceptance_datetime",
    "run_id",
    "asof",
)
MATERIAL_RECONCILIATION_FIELDS = (
    "form_type",
    "filing_date",
    "report_date",
    "acceptance_datetime",
    "primary_doc",
    "primary_doc_description",
    "source_payload_url",
    "file_number",
    "film_number",
    "is_xbrl",
    "is_inline_xbrl",
    "items",
    "size",
)


class InputValidationError(ValueError):
    """Raised when inputs or configuration fail prechecks."""


class PayloadSchemaError(RuntimeError):
    """Raised when a JSON payload is technically readable but structurally invalid."""


@dataclass(slots=True)
class IngestConfig:
    output_dir: str = "data/edgar"
    user_agent: str = "ManuelEsparcia research pipeline contact@example.com"
    timeout_seconds: float = 20.0
    connect_timeout_seconds: float = 10.0
    max_retries: int = 4
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 30.0
    backoff_jitter_seconds: float = 0.25
    rate_limit_per_second: float = 5.0
    max_workers: int = 1
    table_format: str = "parquet"
    raw_envelope_format: str = "json"
    allow_csv_fallback: bool = True
    skip_existing_success: bool = True
    force_refresh: bool = False
    include_history_files: bool = True
    strict_history_fetch: bool = False
    max_history_files_per_cik: Optional[int] = None
    min_coverage_ratio: float = 0.80
    systemic_source_failure_ratio: float = 0.50
    payload_schema_failure_ratio_warn: float = 0.05
    preserve_payload_fingerprint: bool = True
    fail_on_gate_fail: bool = False

    def validate(self) -> None:
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise InputValidationError("config.output_dir must be a non-empty string")
        if not isinstance(self.user_agent, str) or len(self.user_agent.strip()) < 8:
            raise InputValidationError("config.user_agent must be a meaningful SEC-compliant user agent")
        if self.timeout_seconds <= 0 or self.connect_timeout_seconds <= 0:
            raise InputValidationError("timeouts must be positive")
        if self.max_retries < 0:
            raise InputValidationError("max_retries must be >= 0")
        if self.backoff_base_seconds <= 0 or self.backoff_max_seconds <= 0:
            raise InputValidationError("backoff seconds must be positive")
        if self.backoff_jitter_seconds < 0:
            raise InputValidationError("backoff_jitter_seconds must be >= 0")
        if self.rate_limit_per_second <= 0:
            raise InputValidationError("rate_limit_per_second must be > 0")
        if self.max_workers <= 0:
            raise InputValidationError("max_workers must be >= 1")
        if self.max_history_files_per_cik is not None and self.max_history_files_per_cik < 0:
            raise InputValidationError("max_history_files_per_cik must be >= 0 when provided")
        if not (0 <= self.min_coverage_ratio <= 1):
            raise InputValidationError("min_coverage_ratio must be in [0, 1]")
        if not (0 <= self.systemic_source_failure_ratio <= 1):
            raise InputValidationError("systemic_source_failure_ratio must be in [0, 1]")
        if not (0 <= self.payload_schema_failure_ratio_warn <= 1):
            raise InputValidationError("payload_schema_failure_ratio_warn must be in [0, 1]")
        fmt = self.table_format.strip().lower()
        if fmt not in {"parquet", "csv"}:
            raise InputValidationError("table_format must be 'parquet' or 'csv'")
        self.table_format = fmt
        env_fmt = self.raw_envelope_format.strip().lower()
        if env_fmt not in {"json"}:
            raise InputValidationError("raw_envelope_format must currently be 'json'")
        self.raw_envelope_format = env_fmt


@dataclass(slots=True)
class FetchResult:
    cik: str
    url: str
    status: str
    http_status: Optional[int]
    attempts: int
    fetched_at_utc: Optional[str]
    payload_path: Optional[str]
    payload_sha256: Optional[str]
    payload_rows: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class RateLimiter:
    def __init__(self, rate_per_second: float) -> None:
        self.min_interval = 1.0 / rate_per_second
        self._lock = threading.Lock()
        self._last_ts = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            sleep_for = self.min_interval - (now - self._last_ts)
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_ts = time.monotonic()


class SecSubmissionsClient:
    def __init__(self, config: IngestConfig, rate_limiter: RateLimiter) -> None:
        self.config = config
        self.rate_limiter = rate_limiter
        self.headers = {
            "User-Agent": config.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json, text/plain, */*",
            "Host": "data.sec.gov",
        }

    def fetch_json(self, url: str) -> tuple[str, Optional[dict[str, Any]], Optional[int], int, Optional[str]]:
        attempts = 0
        last_reason: Optional[str] = None
        while attempts <= self.config.max_retries:
            attempts += 1
            self.rate_limiter.acquire()
            try:
                resp = requests.get(
                    url,
                    headers=self.headers,
                    timeout=(self.config.connect_timeout_seconds, self.config.timeout_seconds),
                )
                status_code = int(resp.status_code)
                if status_code == 200:
                    try:
                        payload = resp.json()
                    except Exception as exc:
                        return "schema_failure", None, status_code, attempts, f"invalid_json: {exc}"
                    if not isinstance(payload, dict):
                        return "schema_failure", None, status_code, attempts, "payload_not_dict"
                    return "success", payload, status_code, attempts, None
                if status_code in RETRYABLE_STATUS_CODES and attempts <= self.config.max_retries:
                    last_reason = f"retryable_http_{status_code}"
                    self._sleep_backoff(attempts)
                    continue
                if status_code in RETRYABLE_STATUS_CODES:
                    return "retryable_failure", None, status_code, attempts, f"retry_exhausted_http_{status_code}"
                if status_code in PERMANENT_STATUS_CODES:
                    return "permanent_failure", None, status_code, attempts, f"permanent_http_{status_code}"
                if attempts <= self.config.max_retries:
                    last_reason = f"unexpected_http_{status_code}"
                    self._sleep_backoff(attempts)
                    continue
                return "retryable_failure", None, status_code, attempts, f"unexpected_http_{status_code}"
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_reason = f"{type(exc).__name__}: {exc}"
                if attempts <= self.config.max_retries:
                    self._sleep_backoff(attempts)
                    continue
                return "retryable_failure", None, None, attempts, f"retry_exhausted_network: {last_reason}"
            except requests.RequestException as exc:
                last_reason = f"request_exception: {exc}"
                if attempts <= self.config.max_retries:
                    self._sleep_backoff(attempts)
                    continue
                return "retryable_failure", None, None, attempts, f"retry_exhausted_request_exception: {last_reason}"
        return "retryable_failure", None, None, attempts, last_reason

    def _sleep_backoff(self, attempts: int) -> None:
        base = min(self.config.backoff_max_seconds, self.config.backoff_base_seconds * (2 ** max(attempts - 1, 0)))
        jitter = random.uniform(0.0, self.config.backoff_jitter_seconds)
        time.sleep(base + jitter)


# -----------------------------
# Generic helpers
# -----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_hash(config: IngestConfig) -> str:
    return sha256_text(stable_json_dumps(asdict(config)))


def normalize_run_id(value: str) -> str:
    value = (value or "").strip()
    if not value:
        raise InputValidationError("run_id is required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise InputValidationError("run_id must match [A-Za-z0-9_.-]+")
    return value


def normalize_asof(value: str) -> str:
    value = (value or "").strip()
    if not value:
        raise InputValidationError("asof is required")
    try:
        ts = pd.Timestamp(value)
    except Exception as exc:
        raise InputValidationError(f"invalid asof: {exc}") from exc
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def normalize_optional_date(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        ts = pd.Timestamp(text)
    except Exception as exc:
        raise InputValidationError(f"invalid date filter {value!r}: {exc}") from exc
    return ts.date().isoformat()


def normalize_cik(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise InputValidationError("CIK contains empty value")
    digits = re.sub(r"\D", "", text)
    if not digits:
        raise InputValidationError(f"invalid CIK: {value!r}")
    if len(digits) > 10:
        raise InputValidationError(f"CIK has more than 10 digits: {value!r}")
    return digits.zfill(10)


def load_config(config_path: Optional[str]) -> IngestConfig:
    if not config_path:
        cfg = IngestConfig()
        cfg.validate()
        return cfg
    path = Path(config_path)
    if not path.exists() or not path.is_file():
        raise InputValidationError(f"config_path does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise InputValidationError("PyYAML is required to read YAML config files")
        payload = yaml.safe_load(text) or {}
    elif suffix == ".json":
        payload = json.loads(text)
    else:
        raise InputValidationError("config_path must end with .json, .yaml or .yml")
    if not isinstance(payload, dict):
        raise InputValidationError("config file must contain a mapping/object")
    cfg = IngestConfig(**payload)
    cfg.validate()
    return cfg


def merge_config_with_cli(config: IngestConfig, args: argparse.Namespace) -> IngestConfig:
    cfg_dict = asdict(config)
    if args.output_dir:
        cfg_dict["output_dir"] = args.output_dir
    if args.max_workers is not None:
        cfg_dict["max_workers"] = int(args.max_workers)
    if args.force_refresh:
        cfg_dict["force_refresh"] = True
    if args.no_skip_existing_success:
        cfg_dict["skip_existing_success"] = False
    if args.no_history_files:
        cfg_dict["include_history_files"] = False
    cfg = IngestConfig(**cfg_dict)
    cfg.validate()
    return cfg


def load_cik_list(cik_list_path: str) -> list[str]:
    path = Path(cik_list_path)
    if not path.exists() or not path.is_file():
        raise InputValidationError(f"cik_list_path does not exist: {path}")
    suffix = path.suffix.lower()
    values: list[Any] = []
    if suffix in {".csv", ".txt"}:
        try:
            df = pd.read_csv(path)
            if df.shape[1] == 1:
                values = df.iloc[:, 0].tolist()
            else:
                col = next((c for c in df.columns if str(c).lower() in {"cik", "cik_str", "sec_cik"}), None)
                values = df[col].tolist() if col is not None else df.iloc[:, 0].tolist()
        except Exception:
            with path.open("r", encoding="utf-8") as fh:
                values = [line.strip() for line in fh if line.strip()]
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            values = payload
        elif isinstance(payload, dict):
            key = next((k for k in payload.keys() if str(k).lower() in {"cik", "ciks", "cik_list"}), None)
            if key is None or not isinstance(payload[key], list):
                raise InputValidationError("JSON cik list must be a list or contain key cik/ciks/cik_list with a list")
            values = payload[key]
        else:
            raise InputValidationError("Unsupported JSON cik list structure")
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
        if df.empty:
            raise InputValidationError("CIK parquet file is empty")
        col = next((c for c in df.columns if str(c).lower() in {"cik", "cik_str", "sec_cik"}), None)
        values = df[col].tolist() if col is not None else df.iloc[:, 0].tolist()
    else:
        raise InputValidationError("cik_list_path must be .csv, .txt, .json or .parquet")
    if not values:
        raise InputValidationError("CIK list is empty")
    normalized = [normalize_cik(v) for v in values if str(v).strip()]
    if not normalized:
        raise InputValidationError("CIK list contains no valid values after normalization")
    return sorted(set(normalized))


# -----------------------------
# Parsing helpers
# -----------------------------

def parse_date(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        ts = pd.Timestamp(text)
    except Exception:
        return None
    return ts.date().isoformat()


def parse_acceptance_datetime(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        ts = pd.Timestamp(text)
    except Exception:
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def parse_boolish(value: Any) -> Optional[bool]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "t", "y", "yes"}:
        return True
    if text in {"0", "false", "f", "n", "no"}:
        return False
    return None


def parse_intish(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def sanitize_accession_number(value: Any) -> Optional[str]:
    text = safe_str(value)
    if not text:
        return None
    return text


def build_submission_source_url(cik: str, accession_number: Optional[str], primary_doc: Optional[str]) -> Optional[str]:
    if not accession_number or not primary_doc:
        return None
    accn_no_dash = accession_number.replace("-", "")
    try:
        cik_int = int(cik)
    except Exception:
        cik_int = int(cik.lstrip("0") or "0")
    primary_doc = primary_doc.lstrip("/")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_no_dash}/{primary_doc}"


def normalize_column_values(node: dict[str, Any], key: str, n: int) -> list[Any]:
    raw = node.get(key)
    if raw is None:
        return [None] * n
    if not isinstance(raw, list):
        raise PayloadSchemaError(f"submissions array {key!r} is not a list")
    if len(raw) == n:
        return raw
    if len(raw) == 0:
        return [None] * n
    raise PayloadSchemaError(f"submissions array {key!r} has length {len(raw)} but expected {n}")


def validate_recent_node(recent_node: dict[str, Any], *, node_name: str) -> int:
    if not isinstance(recent_node, dict):
        raise PayloadSchemaError(f"{node_name} is not a dict")
    critical = ["accessionNumber", "filingDate", "form", "acceptanceDateTime"]
    lengths: list[int] = []
    for key in critical:
        values = recent_node.get(key)
        if not isinstance(values, list):
            raise PayloadSchemaError(f"{node_name}.{key} missing or not a list")
        lengths.append(len(values))
    unique = sorted(set(lengths))
    if len(unique) != 1:
        raise PayloadSchemaError(f"{node_name} critical arrays not aligned: {dict(zip(critical, lengths))}")
    return unique[0]


def validate_submissions_payload(payload: dict[str, Any], cik: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(payload, dict):
        raise PayloadSchemaError("payload is not a dict")
    payload_cik = payload.get("cik")
    if payload_cik is not None:
        try:
            parsed_cik = normalize_cik(payload_cik)
            if parsed_cik != cik:
                raise PayloadSchemaError(f"payload cik mismatch: requested={cik} payload={parsed_cik}")
        except InputValidationError as exc:
            raise PayloadSchemaError(f"payload cik invalid: {exc}") from exc
    filings = payload.get("filings")
    if not isinstance(filings, dict):
        raise PayloadSchemaError("payload.filings missing or not a dict")
    recent = filings.get("recent")
    n_recent = validate_recent_node(recent, node_name="filings.recent")
    files = filings.get("files", [])
    if files is None:
        files = []
    if not isinstance(files, list):
        raise PayloadSchemaError("payload.filings.files must be a list when present")
    normalized_files: list[dict[str, Any]] = []
    for idx, item in enumerate(files):
        if not isinstance(item, dict):
            raise PayloadSchemaError(f"payload.filings.files[{idx}] is not a dict")
        name = safe_str(item.get("name"))
        if not name:
            raise PayloadSchemaError(f"payload.filings.files[{idx}].name missing")
        normalized_files.append(
            {
                "name": name,
                "filingCount": parse_intish(item.get("filingCount")),
                "filingFrom": parse_date(item.get("filingFrom")),
                "filingTo": parse_date(item.get("filingTo")),
            }
        )
    meta = {
        "recent_filing_count": n_recent,
        "history_file_count": len(normalized_files),
        "entity_name": safe_str(payload.get("name")) or safe_str(payload.get("entityName")),
        "tickers": payload.get("tickers"),
        "exchanges": payload.get("exchanges"),
        "sic": safe_str(payload.get("sic")),
        "sic_description": safe_str(payload.get("sicDescription")),
        "state_of_incorporation": safe_str(payload.get("stateOfIncorporation")),
        "fiscal_year_end": safe_str(payload.get("fiscalYearEnd")),
    }
    return recent, normalized_files, meta


def extract_events_from_node(
    *,
    cik: str,
    node: dict[str, Any],
    source_payload_url: str,
    source_scope: str,
    source_file_name: Optional[str],
    run_id: str,
    asof: str,
    payload_sha256: Optional[str],
    payload_kind: str,
    entity_meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    n = validate_recent_node(node, node_name=source_scope)
    columns = {
        "accessionNumber": normalize_column_values(node, "accessionNumber", n),
        "filingDate": normalize_column_values(node, "filingDate", n),
        "reportDate": normalize_column_values(node, "reportDate", n),
        "acceptanceDateTime": normalize_column_values(node, "acceptanceDateTime", n),
        "act": normalize_column_values(node, "act", n),
        "form": normalize_column_values(node, "form", n),
        "fileNumber": normalize_column_values(node, "fileNumber", n),
        "filmNumber": normalize_column_values(node, "filmNumber", n),
        "items": normalize_column_values(node, "items", n),
        "size": normalize_column_values(node, "size", n),
        "isXBRL": normalize_column_values(node, "isXBRL", n),
        "isInlineXBRL": normalize_column_values(node, "isInlineXBRL", n),
        "primaryDocument": normalize_column_values(node, "primaryDocument", n),
        "primaryDocDescription": normalize_column_values(node, "primaryDocDescription", n),
    }
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    exact_duplicate_counter = 0
    temporal_inconsistency_counter = 0
    missing_acceptance_counter = 0

    for i in range(n):
        accession_number = sanitize_accession_number(columns["accessionNumber"][i])
        filing_date = parse_date(columns["filingDate"][i])
        report_date = parse_date(columns["reportDate"][i])
        acceptance_dt = parse_acceptance_datetime(columns["acceptanceDateTime"][i])
        primary_doc = safe_str(columns["primaryDocument"][i])
        primary_doc_description = safe_str(columns["primaryDocDescription"][i])
        row = {
            "cik": cik,
            "accession_number": accession_number,
            "form_type": safe_str(columns["form"][i]),
            "filing_date": filing_date,
            "report_date": report_date,
            "acceptance_datetime": acceptance_dt,
            "primary_doc": primary_doc,
            "primary_doc_description": primary_doc_description,
            "submission_document_url": build_submission_source_url(cik, accession_number, primary_doc),
            "source_payload_url": source_payload_url,
            "source_scope": source_scope,
            "source_file_name": source_file_name,
            "payload_kind": payload_kind,
            "act": safe_str(columns["act"][i]),
            "file_number": safe_str(columns["fileNumber"][i]),
            "film_number": safe_str(columns["filmNumber"][i]),
            "items": safe_str(columns["items"][i]),
            "size": parse_intish(columns["size"][i]),
            "is_xbrl": parse_boolish(columns["isXBRL"][i]),
            "is_inline_xbrl": parse_boolish(columns["isInlineXBRL"][i]),
            "entity_name": entity_meta.get("entity_name"),
            "sic": entity_meta.get("sic"),
            "sic_description": entity_meta.get("sic_description"),
            "state_of_incorporation": entity_meta.get("state_of_incorporation"),
            "fiscal_year_end": entity_meta.get("fiscal_year_end"),
            "run_id": run_id,
            "asof": asof,
            "payload_sha256": payload_sha256,
        }
        missing_required = [name for name in REQUIRED_EVENT_FIELDS if row.get(name) in {None, ""}]
        if missing_required:
            failures.append(
                {
                    "cik": cik,
                    "accession_number": accession_number,
                    "failure_class": "record_level_issue",
                    "stage": "tabular_extraction",
                    "reason": f"missing_required_fields: {','.join(missing_required)}",
                    "source_scope": source_scope,
                    "source_payload_url": source_payload_url,
                    "run_id": run_id,
                    "asof": asof,
                }
            )
            continue

        if acceptance_dt is None:
            missing_acceptance_counter += 1
            failures.append(
                {
                    "cik": cik,
                    "accession_number": accession_number,
                    "failure_class": "record_level_issue",
                    "stage": "temporal_validation",
                    "reason": "missing_acceptance_datetime",
                    "source_scope": source_scope,
                    "source_payload_url": source_payload_url,
                    "run_id": run_id,
                    "asof": asof,
                }
            )
        else:
            filing_ts = pd.Timestamp(filing_date) if filing_date is not None else None
            acc_ts = pd.Timestamp(acceptance_dt)
            acc_date = acc_ts.date().isoformat()
            if filing_ts is not None and filing_ts.date().isoformat() > acc_date:
                temporal_inconsistency_counter += 1
                failures.append(
                    {
                        "cik": cik,
                        "accession_number": accession_number,
                        "failure_class": "record_level_issue",
                        "stage": "temporal_validation",
                        "reason": "filing_date_gt_acceptance_datetime",
                        "filing_date": filing_date,
                        "acceptance_datetime": acceptance_dt,
                        "source_scope": source_scope,
                        "source_payload_url": source_payload_url,
                        "run_id": run_id,
                        "asof": asof,
                    }
                )
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df["event_fingerprint"] = df.apply(event_fingerprint_from_row, axis=1)
        before = len(df)
        df = df.drop_duplicates(subset=["cik", "accession_number", "event_fingerprint"], keep="first").reset_index(drop=True)
        exact_duplicate_counter = before - len(df)
        rows = df.to_dict(orient="records")

    metrics = {
        "rows_emitted_pre_dedup": n,
        "rows_valid_post_dedup": len(rows),
        "exact_duplicates_removed": exact_duplicate_counter,
        "temporal_inconsistencies": temporal_inconsistency_counter,
        "missing_acceptance_datetime": missing_acceptance_counter,
        "row_level_failures": len([f for f in failures if f.get("failure_class") == "record_level_issue"]),
    }
    return rows, failures, metrics


def event_fingerprint_from_row(row: pd.Series | dict[str, Any]) -> str:
    payload = {field: (row.get(field) if isinstance(row, dict) else row[field]) for field in MATERIAL_RECONCILIATION_FIELDS}
    return sha256_text(stable_json_dumps(payload))


# -----------------------------
# Paths and persistence
# -----------------------------

def asof_path_token(asof: str) -> str:
    return asof.replace(":", "-")


def raw_payload_json_path(base_dir: Path, asof: str, run_id: str, cik: str) -> Path:
    return base_dir / "raw" / "submissions_payloads_json" / f"date={asof_path_token(asof)}" / f"run_id={run_id}" / f"CIK{cik}.json"


def events_output_base(base_dir: Path, asof: str, run_id: str) -> Path:
    return base_dir / "raw" / "submissions" / f"date={asof_path_token(asof)}" / f"run_id={run_id}" / "part-00000"


def payloads_output_base(base_dir: Path, asof: str, run_id: str) -> Path:
    return base_dir / "raw" / "submissions_payloads" / f"date={asof_path_token(asof)}" / f"run_id={run_id}" / "part-00000"


def failures_output_base(base_dir: Path, run_id: str) -> Path:
    return base_dir / "raw" / f"submissions_failures_{run_id}"


def metrics_output_base(base_dir: Path, run_id: str) -> Path:
    return base_dir / "raw" / f"submissions_metrics_{run_id}"


def reconciliation_output_base(base_dir: Path, run_id: str) -> Path:
    return base_dir / "raw" / f"submissions_reconciliation_{run_id}"


def manifest_output_path(base_dir: Path, run_id: str) -> Path:
    return base_dir / "raw" / f"submissions_manifest_{run_id}.json"


def read_existing_raw_envelope(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def persist_raw_envelope(path: Path, envelope: dict[str, Any]) -> str:
    ensure_directory(path.parent)
    text = json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2)
    path.write_text(text, encoding="utf-8")
    return sha256_text(text)


def write_dataframe(df: pd.DataFrame, target_base: Path, table_format: str, allow_csv_fallback: bool) -> tuple[str, int]:
    ensure_directory(target_base.parent)
    table_format = table_format.lower().strip()
    if table_format == "parquet":
        try:
            target = target_base.with_suffix(".parquet")
            df.to_parquet(target, index=False)
            return str(target), len(df)
        except Exception as exc:
            if not allow_csv_fallback:
                raise RuntimeError(
                    "Unable to write parquet. Install pyarrow or fastparquet, or enable CSV fallback."
                ) from exc
    target = target_base.with_suffix(".csv")
    df.to_csv(target, index=False, quoting=csv.QUOTE_MINIMAL)
    return str(target), len(df)


# -----------------------------
# Incremental state and reconciliation
# -----------------------------

def read_table_auto(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported table extension: {path}")


def load_previous_event_state(base_dir: Path, *, current_asof: str, current_run_id: str) -> pd.DataFrame:
    root = base_dir / "raw" / "submissions"
    if not root.exists():
        return pd.DataFrame(columns=["cik", "accession_number", "event_fingerprint", "asof", "run_id", "filing_date"])
    frames: list[pd.DataFrame] = []
    current_token = asof_path_token(current_asof)
    for path in root.rglob("part-*.csv"):
        if f"date={current_token}" in path.as_posix() and f"run_id={current_run_id}" in path.as_posix():
            continue
        try:
            frames.append(read_table_auto(path))
        except Exception:
            continue
    for path in root.rglob("part-*.parquet"):
        if f"date={current_token}" in path.as_posix() and f"run_id={current_run_id}" in path.as_posix():
            continue
        try:
            frames.append(read_table_auto(path))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["cik", "accession_number", "event_fingerprint", "asof", "run_id", "filing_date"])
    df = pd.concat(frames, axis=0, ignore_index=True)
    required = {"cik", "accession_number", "asof", "run_id"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["cik", "accession_number", "event_fingerprint", "asof", "run_id", "filing_date"])
    if "event_fingerprint" not in df.columns:
        df["event_fingerprint"] = df.apply(event_fingerprint_from_row, axis=1)
    if "filing_date" not in df.columns:
        df["filing_date"] = None
    df["_asof_ts"] = pd.to_datetime(df["asof"], errors="coerce", utc=True)
    df = df.sort_values(["_asof_ts", "run_id"], ascending=[True, True])
    latest = df.groupby(["cik", "accession_number"], as_index=False).tail(1).reset_index(drop=True)
    return latest.drop(columns=["_asof_ts"], errors="ignore")


def reconcile_current_vs_previous(current_events: pd.DataFrame, previous_events: pd.DataFrame) -> pd.DataFrame:
    if current_events.empty:
        return pd.DataFrame(
            columns=[
                "cik",
                "accession_number",
                "reconciliation_class",
                "previous_run_id",
                "previous_asof",
                "current_run_id",
                "current_asof",
                "current_fingerprint",
                "previous_fingerprint",
                "filing_date",
                "late_threshold_asof",
            ]
        )
    current = current_events.copy()
    if "event_fingerprint" not in current.columns:
        current["event_fingerprint"] = current.apply(event_fingerprint_from_row, axis=1)
    if previous_events.empty:
        prev_lookup = pd.DataFrame(columns=["cik", "accession_number", "previous_fingerprint", "previous_run_id", "previous_asof", "previous_filing_date"])
        late_threshold: Optional[pd.Timestamp] = None
    else:
        prev_lookup = previous_events[["cik", "accession_number", "event_fingerprint", "run_id", "asof", "filing_date"]].copy()
        prev_lookup = prev_lookup.rename(
            columns={
                "event_fingerprint": "previous_fingerprint",
                "run_id": "previous_run_id",
                "asof": "previous_asof",
                "filing_date": "previous_filing_date",
            }
        )
        late_threshold = pd.to_datetime(previous_events["asof"], errors="coerce", utc=True).max()
    merged = current.merge(prev_lookup, on=["cik", "accession_number"], how="left")
    current_asof = current["asof"].iloc[0] if not current.empty else None
    late_threshold_iso = late_threshold.isoformat().replace("+00:00", "Z") if late_threshold is not None and not pd.isna(late_threshold) else None

    classes: list[str] = []
    for row in merged.itertuples(index=False):
        prev_fp = getattr(row, "previous_fingerprint", None)
        if prev_fp is None or (isinstance(prev_fp, float) and pd.isna(prev_fp)):
            filing_date = getattr(row, "filing_date", None)
            filing_ts = pd.to_datetime(filing_date, errors="coerce") if filing_date is not None else pd.NaT
            if late_threshold is not None and not pd.isna(late_threshold) and not pd.isna(filing_ts) and filing_ts.date() < late_threshold.date():
                classes.append("late_arrival")
            else:
                classes.append("novel")
        elif str(prev_fp) == str(getattr(row, "event_fingerprint")):
            classes.append("seen")
        else:
            classes.append("changed_record")
    merged["reconciliation_class"] = classes
    merged["current_run_id"] = merged["run_id"]
    merged["current_asof"] = merged["asof"]
    merged["current_fingerprint"] = merged["event_fingerprint"]
    merged["late_threshold_asof"] = late_threshold_iso
    return merged[
        [
            "cik",
            "accession_number",
            "reconciliation_class",
            "previous_run_id",
            "previous_asof",
            "current_run_id",
            "current_asof",
            "current_fingerprint",
            "previous_fingerprint",
            "filing_date",
            "late_threshold_asof",
        ]
    ].reset_index(drop=True)


# -----------------------------
# Per-CIK processing
# -----------------------------

def build_payload_index_row(
    *,
    cik: str,
    payload_kind: str,
    source_url: str,
    file_name: Optional[str],
    http_status: Optional[int],
    fetched_at_utc: Optional[str],
    attempts: int,
    payload_sha256: Optional[str],
    payload_json: Optional[dict[str, Any]],
    raw_json_path: Optional[str],
    run_id: str,
    asof: str,
) -> dict[str, Any]:
    return {
        "cik": cik,
        "payload_kind": payload_kind,
        "file_name": file_name,
        "source_url": source_url,
        "http_status": http_status,
        "fetched_at_utc": fetched_at_utc,
        "attempts": attempts,
        "payload_sha256": payload_sha256,
        "payload_json": stable_json_dumps(payload_json) if payload_json is not None else None,
        "raw_json_path": raw_json_path,
        "run_id": run_id,
        "asof": asof,
    }


def process_single_cik(
    *,
    cik: str,
    client: SecSubmissionsClient,
    config: IngestConfig,
    output_dir: Path,
    asof: str,
    run_id: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> FetchResult:
    started_ts = time.monotonic()
    root_url = SEC_SUBMISSIONS_URL.format(cik=cik)
    raw_json_file = raw_payload_json_path(output_dir, asof, run_id, cik)

    if config.skip_existing_success and not config.force_refresh and raw_json_file.exists():
        try:
            envelope = read_existing_raw_envelope(raw_json_file)
            fetched_at_utc = envelope.get("fetched_at_utc")
            payload_sha = envelope.get("envelope_sha256")
            payload_rows, events, failures, metrics = rebuild_from_saved_envelope(
                envelope=envelope,
                cik=cik,
                run_id=run_id,
                asof=asof,
                start_date=start_date,
                end_date=end_date,
                raw_json_path=str(raw_json_file),
            )
            metrics.update(
                {
                    "cik": cik,
                    "status": "success",
                    "http_status": envelope.get("http_status", 200),
                    "attempts": 0,
                    "latency_seconds": 0.0,
                    "fetched_at_utc": fetched_at_utc,
                    "payload_path": str(raw_json_file),
                    "source_mode": "reuse_existing_raw",
                }
            )
            return FetchResult(
                cik=cik,
                url=root_url,
                status="success",
                http_status=int(envelope.get("http_status", 200)),
                attempts=0,
                fetched_at_utc=fetched_at_utc,
                payload_path=str(raw_json_file),
                payload_sha256=payload_sha,
                payload_rows=payload_rows,
                events=events,
                failures=failures,
                metrics=metrics,
            )
        except Exception as exc:
            return FetchResult(
                cik=cik,
                url=root_url,
                status="schema_failure",
                http_status=None,
                attempts=0,
                fetched_at_utc=None,
                payload_path=str(raw_json_file),
                payload_sha256=None,
                payload_rows=[],
                events=[],
                failures=[
                    {
                        "cik": cik,
                        "failure_class": "schema_failure",
                        "stage": "reuse_existing_raw",
                        "reason": f"failed_to_read_existing_raw: {exc}",
                        "run_id": run_id,
                        "asof": asof,
                    }
                ],
                metrics={
                    "cik": cik,
                    "status": "schema_failure",
                    "attempts": 0,
                    "latency_seconds": 0.0,
                    "source_mode": "reuse_existing_raw_failed",
                },
            )

    root_status, root_payload, http_status, attempts, reason = client.fetch_json(root_url)
    fetched_at_utc = utc_now_iso()
    latency_seconds = round(time.monotonic() - started_ts, 6)
    if root_status != "success" or root_payload is None:
        return FetchResult(
            cik=cik,
            url=root_url,
            status=root_status,
            http_status=http_status,
            attempts=attempts,
            fetched_at_utc=fetched_at_utc,
            payload_path=None,
            payload_sha256=None,
            payload_rows=[],
            events=[],
            failures=[
                {
                    "cik": cik,
                    "failure_class": root_status,
                    "stage": "http_fetch",
                    "http_status": http_status,
                    "attempts": attempts,
                    "reason": reason,
                    "source_url": root_url,
                    "run_id": run_id,
                    "asof": asof,
                }
            ],
            metrics={
                "cik": cik,
                "status": root_status,
                "http_status": http_status,
                "attempts": attempts,
                "latency_seconds": latency_seconds,
                "fetched_at_utc": fetched_at_utc,
                "payload_path": None,
                "source_mode": "downloaded",
            },
        )

    try:
        recent_node, history_files, entity_meta = validate_submissions_payload(root_payload, cik)
    except Exception as exc:
        return FetchResult(
            cik=cik,
            url=root_url,
            status="schema_failure",
            http_status=http_status,
            attempts=attempts,
            fetched_at_utc=fetched_at_utc,
            payload_path=None,
            payload_sha256=None,
            payload_rows=[],
            events=[],
            failures=[
                {
                    "cik": cik,
                    "failure_class": "schema_failure",
                    "stage": "payload_validation",
                    "http_status": http_status,
                    "attempts": attempts,
                    "reason": str(exc),
                    "source_url": root_url,
                    "run_id": run_id,
                    "asof": asof,
                }
            ],
            metrics={
                "cik": cik,
                "status": "schema_failure",
                "http_status": http_status,
                "attempts": attempts,
                "latency_seconds": latency_seconds,
                "fetched_at_utc": fetched_at_utc,
                "payload_path": None,
                "source_mode": "downloaded",
            },
        )

    main_payload_text = stable_json_dumps(root_payload)
    main_payload_sha = sha256_text(main_payload_text)
    history_payloads: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    payload_rows: list[dict[str, Any]] = [
        build_payload_index_row(
            cik=cik,
            payload_kind="root_submissions",
            source_url=root_url,
            file_name=None,
            http_status=http_status,
            fetched_at_utc=fetched_at_utc,
            attempts=attempts,
            payload_sha256=main_payload_sha,
            payload_json=root_payload,
            raw_json_path=str(raw_json_file),
            run_id=run_id,
            asof=asof,
        )
    ]

    history_fetch_failures = 0
    history_attempts_total = 0
    history_files_used = 0
    if config.include_history_files and history_files:
        if config.max_history_files_per_cik is not None:
            history_files = history_files[: config.max_history_files_per_cik]
        for descriptor in history_files:
            filename = descriptor["name"]
            history_url = SEC_SUBMISSIONS_HISTORY_URL.format(filename=filename)
            hist_status, hist_payload, hist_http_status, hist_attempts, hist_reason = client.fetch_json(history_url)
            history_attempts_total += hist_attempts
            if hist_status != "success" or hist_payload is None:
                history_fetch_failures += 1
                failures.append(
                    {
                        "cik": cik,
                        "failure_class": hist_status,
                        "stage": "history_file_fetch",
                        "http_status": hist_http_status,
                        "attempts": hist_attempts,
                        "reason": hist_reason,
                        "source_url": history_url,
                        "source_file_name": filename,
                        "run_id": run_id,
                        "asof": asof,
                    }
                )
                if config.strict_history_fetch:
                    return FetchResult(
                        cik=cik,
                        url=root_url,
                        status=hist_status,
                        http_status=http_status,
                        attempts=attempts + history_attempts_total,
                        fetched_at_utc=fetched_at_utc,
                        payload_path=None,
                        payload_sha256=None,
                        payload_rows=payload_rows,
                        events=[],
                        failures=failures,
                        metrics={
                            "cik": cik,
                            "status": hist_status,
                            "http_status": http_status,
                            "attempts": attempts + history_attempts_total,
                            "latency_seconds": round(time.monotonic() - started_ts, 6),
                            "fetched_at_utc": fetched_at_utc,
                            "payload_path": None,
                            "source_mode": "downloaded",
                        },
                    )
                continue
            try:
                validate_recent_node(hist_payload, node_name=f"history_file:{filename}")
            except Exception as exc:
                history_fetch_failures += 1
                failures.append(
                    {
                        "cik": cik,
                        "failure_class": "schema_failure",
                        "stage": "history_file_validation",
                        "http_status": hist_http_status,
                        "attempts": hist_attempts,
                        "reason": str(exc),
                        "source_url": history_url,
                        "source_file_name": filename,
                        "run_id": run_id,
                        "asof": asof,
                    }
                )
                if config.strict_history_fetch:
                    return FetchResult(
                        cik=cik,
                        url=root_url,
                        status="schema_failure",
                        http_status=http_status,
                        attempts=attempts + history_attempts_total,
                        fetched_at_utc=fetched_at_utc,
                        payload_path=None,
                        payload_sha256=None,
                        payload_rows=payload_rows,
                        events=[],
                        failures=failures,
                        metrics={
                            "cik": cik,
                            "status": "schema_failure",
                            "http_status": http_status,
                            "attempts": attempts + history_attempts_total,
                            "latency_seconds": round(time.monotonic() - started_ts, 6),
                            "fetched_at_utc": fetched_at_utc,
                            "payload_path": None,
                            "source_mode": "downloaded",
                        },
                    )
                continue
            hist_sha = sha256_text(stable_json_dumps(hist_payload))
            history_payloads.append(
                {
                    "name": filename,
                    "url": history_url,
                    "http_status": hist_http_status,
                    "attempts": hist_attempts,
                    "payload_sha256": hist_sha,
                    "payload": hist_payload,
                    "filingCount": descriptor.get("filingCount"),
                    "filingFrom": descriptor.get("filingFrom"),
                    "filingTo": descriptor.get("filingTo"),
                }
            )
            payload_rows.append(
                build_payload_index_row(
                    cik=cik,
                    payload_kind="history_submissions",
                    source_url=history_url,
                    file_name=filename,
                    http_status=hist_http_status,
                    fetched_at_utc=fetched_at_utc,
                    attempts=hist_attempts,
                    payload_sha256=hist_sha,
                    payload_json=hist_payload,
                    raw_json_path=str(raw_json_file),
                    run_id=run_id,
                    asof=asof,
                )
            )
            history_files_used += 1

    envelope = {
        "schema_version": "1.0",
        "module": "data.edgar.fetch_submissions",
        "cik": cik,
        "requested_url": root_url,
        "http_status": http_status,
        "fetched_at_utc": fetched_at_utc,
        "run_id": run_id,
        "asof": asof,
        "entity_meta": entity_meta,
        "main_payload_sha256": main_payload_sha,
        "main_payload": root_payload,
        "history_payloads": history_payloads,
        "history_fetch_failures": [f for f in failures if f.get("stage") in {"history_file_fetch", "history_file_validation"}],
    }
    try:
        envelope_sha = persist_raw_envelope(raw_json_file, envelope)
    except Exception as exc:
        return FetchResult(
            cik=cik,
            url=root_url,
            status="local_failure",
            http_status=http_status,
            attempts=attempts + history_attempts_total,
            fetched_at_utc=fetched_at_utc,
            payload_path=str(raw_json_file),
            payload_sha256=None,
            payload_rows=payload_rows,
            events=[],
            failures=failures
            + [
                {
                    "cik": cik,
                    "failure_class": "local_failure",
                    "stage": "persist_raw_envelope",
                    "reason": str(exc),
                    "source_url": root_url,
                    "run_id": run_id,
                    "asof": asof,
                }
            ],
            metrics={
                "cik": cik,
                "status": "local_failure",
                "http_status": http_status,
                "attempts": attempts + history_attempts_total,
                "latency_seconds": round(time.monotonic() - started_ts, 6),
                "fetched_at_utc": fetched_at_utc,
                "payload_path": str(raw_json_file),
                "source_mode": "downloaded",
            },
        )

    payload_rows = [dict(row, raw_json_path=str(raw_json_file)) for row in payload_rows]
    events, extract_failures, extract_metrics = extract_all_events_from_envelope(
        envelope=envelope,
        cik=cik,
        run_id=run_id,
        asof=asof,
        start_date=start_date,
        end_date=end_date,
    )
    failures.extend(extract_failures)
    total_latency = round(time.monotonic() - started_ts, 6)
    metrics = {
        "cik": cik,
        "status": "success",
        "http_status": http_status,
        "attempts": attempts + history_attempts_total,
        "latency_seconds": total_latency,
        "fetched_at_utc": fetched_at_utc,
        "payload_path": str(raw_json_file),
        "payload_sha256": envelope_sha,
        "source_mode": "downloaded",
        "root_attempts": attempts,
        "history_files_declared": len(history_files),
        "history_files_used": history_files_used,
        "history_fetch_failures": history_fetch_failures,
    }
    metrics.update(extract_metrics)
    return FetchResult(
        cik=cik,
        url=root_url,
        status="success",
        http_status=http_status,
        attempts=attempts + history_attempts_total,
        fetched_at_utc=fetched_at_utc,
        payload_path=str(raw_json_file),
        payload_sha256=envelope_sha,
        payload_rows=payload_rows,
        events=events,
        failures=failures,
        metrics=metrics,
    )


def extract_all_events_from_envelope(
    *,
    envelope: dict[str, Any],
    cik: str,
    run_id: str,
    asof: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    entity_meta = envelope.get("entity_meta", {}) if isinstance(envelope.get("entity_meta"), dict) else {}
    main_payload = envelope.get("main_payload")
    if not isinstance(main_payload, dict):
        raise PayloadSchemaError("saved envelope missing main_payload")
    recent_node, _, _ = validate_submissions_payload(main_payload, cik)
    main_sha = envelope.get("main_payload_sha256")
    all_rows: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    exact_duplicates_removed = 0
    filtered_out_count = 0

    root_rows, root_failures, root_metrics = extract_events_from_node(
        cik=cik,
        node=recent_node,
        source_payload_url=envelope.get("requested_url") or SEC_SUBMISSIONS_URL.format(cik=cik),
        source_scope="filings.recent",
        source_file_name=None,
        run_id=run_id,
        asof=asof,
        payload_sha256=main_sha,
        payload_kind="root_submissions",
        entity_meta=entity_meta,
    )
    all_rows.extend(root_rows)
    all_failures.extend(root_failures)
    exact_duplicates_removed += int(root_metrics.get("exact_duplicates_removed", 0))

    history_payloads = envelope.get("history_payloads", [])
    if history_payloads and not isinstance(history_payloads, list):
        raise PayloadSchemaError("saved envelope history_payloads is not a list")
    for item in history_payloads:
        if not isinstance(item, dict):
            continue
        payload = item.get("payload")
        if not isinstance(payload, dict):
            all_failures.append(
                {
                    "cik": cik,
                    "failure_class": "schema_failure",
                    "stage": "history_payload_structure",
                    "reason": "history payload not a dict",
                    "source_file_name": item.get("name"),
                    "run_id": run_id,
                    "asof": asof,
                }
            )
            continue
        history_rows, history_failures, history_metrics = extract_events_from_node(
            cik=cik,
            node=payload,
            source_payload_url=safe_str(item.get("url")) or SEC_SUBMISSIONS_HISTORY_URL.format(filename=item.get("name")),
            source_scope=f"history_file:{item.get('name')}",
            source_file_name=safe_str(item.get("name")),
            run_id=run_id,
            asof=asof,
            payload_sha256=safe_str(item.get("payload_sha256")),
            payload_kind="history_submissions",
            entity_meta=entity_meta,
        )
        all_rows.extend(history_rows)
        all_failures.extend(history_failures)
        exact_duplicates_removed += int(history_metrics.get("exact_duplicates_removed", 0))

    if all_rows:
        df = pd.DataFrame(all_rows)
        if "event_fingerprint" not in df.columns:
            df["event_fingerprint"] = df.apply(event_fingerprint_from_row, axis=1)
        before_exact = len(df)
        df = df.drop_duplicates(subset=["cik", "accession_number", "event_fingerprint"], keep="first").reset_index(drop=True)
        exact_duplicates_removed += before_exact - len(df)

        conflict_rows: list[dict[str, Any]] = []
        keep_indices: list[int] = []
        source_priority = df["source_scope"].apply(lambda x: 0 if str(x) == "filings.recent" else 1)
        df = df.assign(_source_priority=source_priority)
        for (_, _), grp in df.groupby(["cik", "accession_number"], sort=False):
            fps = grp["event_fingerprint"].astype(str).dropna().unique().tolist()
            if len(fps) <= 1:
                keep_indices.append(int(grp.index[0]))
                continue
            grp_sorted = grp.sort_values(
                ["_source_priority", "acceptance_datetime", "source_payload_url"],
                ascending=[True, True, True],
                na_position="last",
            )
            keep_indices.append(int(grp_sorted.index[0]))
            for _, row in grp_sorted.iloc[1:].iterrows():
                conflict_rows.append(
                    {
                        "cik": row["cik"],
                        "accession_number": row["accession_number"],
                        "failure_class": "record_level_issue",
                        "stage": "intra_run_deduplication",
                        "reason": "material_conflict_same_key_within_run",
                        "source_scope": row.get("source_scope"),
                        "source_payload_url": row.get("source_payload_url"),
                        "run_id": run_id,
                        "asof": asof,
                    }
                )
        if conflict_rows:
            all_failures.extend(conflict_rows)
        df = df.loc[sorted(set(keep_indices))].copy().reset_index(drop=True)

        if start_date is not None or end_date is not None:
            filing_ts = pd.to_datetime(df["filing_date"], errors="coerce")
            mask = pd.Series(True, index=df.index)
            if start_date is not None:
                mask &= filing_ts.dt.date >= pd.Timestamp(start_date).date()
            if end_date is not None:
                mask &= filing_ts.dt.date <= pd.Timestamp(end_date).date()
            filtered_out_count = int((~mask).sum())
            if filtered_out_count > 0:
                all_failures.append(
                    {
                        "cik": cik,
                        "failure_class": "record_level_issue",
                        "stage": "temporal_filtering",
                        "reason": f"temporal_filter_removed_{filtered_out_count}_rows",
                        "run_id": run_id,
                        "asof": asof,
                    }
                )
            df = df.loc[mask].reset_index(drop=True)

        all_rows = df.to_dict(orient="records")
    else:
        all_rows = []

    form_distribution = {}
    if all_rows:
        form_distribution = pd.Series([row.get("form_type") for row in all_rows]).value_counts(dropna=False).to_dict()
    metrics = {
        "submissions_extracted": len(all_rows),
        "exact_duplicates_removed": exact_duplicates_removed,
        "row_level_failures": len([f for f in all_failures if f.get("failure_class") == "record_level_issue"]),
        "history_payloads_available": len(history_payloads) if isinstance(history_payloads, list) else 0,
        "filtered_out_count": filtered_out_count,
        "form_distribution_json": stable_json_dumps(form_distribution),
    }
    return all_rows, all_failures, metrics


def rebuild_from_saved_envelope(
    *,
    envelope: dict[str, Any],
    cik: str,
    run_id: str,
    asof: str,
    start_date: Optional[str],
    end_date: Optional[str],
    raw_json_path: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    payload_rows: list[dict[str, Any]] = []
    root_payload = envelope.get("main_payload")
    if not isinstance(root_payload, dict):
        raise PayloadSchemaError("saved envelope missing main_payload")
    payload_rows.append(
        build_payload_index_row(
            cik=cik,
            payload_kind="root_submissions",
            source_url=safe_str(envelope.get("requested_url")) or SEC_SUBMISSIONS_URL.format(cik=cik),
            file_name=None,
            http_status=parse_intish(envelope.get("http_status")),
            fetched_at_utc=safe_str(envelope.get("fetched_at_utc")),
            attempts=0,
            payload_sha256=safe_str(envelope.get("main_payload_sha256")),
            payload_json=root_payload,
            raw_json_path=raw_json_path,
            run_id=run_id,
            asof=asof,
        )
    )
    history_payloads = envelope.get("history_payloads", [])
    if isinstance(history_payloads, list):
        for item in history_payloads:
            if not isinstance(item, dict):
                continue
            payload_rows.append(
                build_payload_index_row(
                    cik=cik,
                    payload_kind="history_submissions",
                    source_url=safe_str(item.get("url")) or SEC_SUBMISSIONS_HISTORY_URL.format(filename=item.get("name")),
                    file_name=safe_str(item.get("name")),
                    http_status=parse_intish(item.get("http_status")),
                    fetched_at_utc=safe_str(envelope.get("fetched_at_utc")),
                    attempts=parse_intish(item.get("attempts")) or 0,
                    payload_sha256=safe_str(item.get("payload_sha256")),
                    payload_json=item.get("payload") if isinstance(item.get("payload"), dict) else None,
                    raw_json_path=raw_json_path,
                    run_id=run_id,
                    asof=asof,
                )
            )
    events, failures, metrics = extract_all_events_from_envelope(
        envelope=envelope,
        cik=cik,
        run_id=run_id,
        asof=asof,
        start_date=start_date,
        end_date=end_date,
    )
    return payload_rows, events, failures, metrics


# -----------------------------
# Metrics and run gate
# -----------------------------

def classify_run_gate(
    *,
    total_ciks: int,
    success_ciks: int,
    retryable_failures: int,
    permanent_failures: int,
    schema_failures: int,
    local_failures: int,
    config: IngestConfig,
) -> tuple[str, str]:
    if total_ciks <= 0:
        return "fail", "input_or_config_failure"
    source_failure_ratio = (retryable_failures + permanent_failures) / total_ciks
    coverage_ratio = success_ciks / total_ciks
    schema_ratio = schema_failures / total_ciks
    if source_failure_ratio >= config.systemic_source_failure_ratio:
        return "fail", "systemic_source_failure"
    if coverage_ratio < config.min_coverage_ratio:
        return "fail", "coverage_collapse"
    if local_failures > 0:
        return "fail", "local_failure"
    if schema_ratio >= config.payload_schema_failure_ratio_warn:
        return "warn", "payload_schema_failure"
    return "pass", "ok"


def build_metrics_dataframe(
    *,
    cik_metrics_rows: list[dict[str, Any]],
    events_df: pd.DataFrame,
    reconciliation_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    total_ciks: int,
    config: IngestConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    per_cik = pd.DataFrame(cik_metrics_rows)
    if per_cik.empty:
        per_cik = pd.DataFrame(columns=["cik", "status"])
    success_ciks = int((per_cik["status"] == "success").sum()) if "status" in per_cik.columns else 0
    retryable_failures = int((per_cik["status"] == "retryable_failure").sum()) if "status" in per_cik.columns else 0
    permanent_failures = int((per_cik["status"] == "permanent_failure").sum()) if "status" in per_cik.columns else 0
    schema_failures = int((per_cik["status"] == "schema_failure").sum()) if "status" in per_cik.columns else 0
    local_failures = int((per_cik["status"] == "local_failure").sum()) if "status" in per_cik.columns else 0
    coverage_ratio = success_ciks / total_ciks if total_ciks else 0.0
    gate, gate_reason = classify_run_gate(
        total_ciks=total_ciks,
        success_ciks=success_ciks,
        retryable_failures=retryable_failures,
        permanent_failures=permanent_failures,
        schema_failures=schema_failures,
        local_failures=local_failures,
        config=config,
    )
    total_events = int(len(events_df))
    per_cik_event_counts = events_df.groupby("cik").size() if not events_df.empty else pd.Series(dtype=int)
    filings_mean = float(per_cik_event_counts.mean()) if len(per_cik_event_counts) else 0.0
    filings_p50 = float(per_cik_event_counts.quantile(0.50)) if len(per_cik_event_counts) else 0.0
    filings_p95 = float(per_cik_event_counts.quantile(0.95)) if len(per_cik_event_counts) else 0.0
    filings_p99 = float(per_cik_event_counts.quantile(0.99)) if len(per_cik_event_counts) else 0.0
    form_distribution = events_df["form_type"].value_counts(dropna=False).to_dict() if not events_df.empty else {}
    coverage_by_period = (
        events_df.assign(filing_period=pd.to_datetime(events_df["filing_date"], errors="coerce").dt.to_period("M").astype(str))
        .groupby("filing_period")
        .size()
        .to_dict()
        if not events_df.empty
        else {}
    )
    retry_ratio = float((per_cik["attempts"].fillna(0) > 1).mean()) if "attempts" in per_cik.columns and len(per_cik) else 0.0
    lat_mean = float(per_cik["latency_seconds"].fillna(0).mean()) if "latency_seconds" in per_cik.columns and len(per_cik) else 0.0
    lat_p95 = float(per_cik["latency_seconds"].fillna(0).quantile(0.95)) if "latency_seconds" in per_cik.columns and len(per_cik) else 0.0
    lat_p99 = float(per_cik["latency_seconds"].fillna(0).quantile(0.99)) if "latency_seconds" in per_cik.columns and len(per_cik) else 0.0
    late_arrivals = int((reconciliation_df.get("reconciliation_class") == "late_arrival").sum()) if not reconciliation_df.empty else 0
    changed_records = int((reconciliation_df.get("reconciliation_class") == "changed_record").sum()) if not reconciliation_df.empty else 0
    ciks_no_useful = int(success_ciks - len(per_cik_event_counts)) if success_ciks >= len(per_cik_event_counts) else 0
    exact_duplicates_removed = int(per_cik.get("exact_duplicates_removed", pd.Series(dtype=int)).fillna(0).sum()) if len(per_cik) else 0

    aggregate = {
        "cik": "__GLOBAL__",
        "status": gate,
        "gate_reason": gate_reason,
        "target_ciks": total_ciks,
        "success_ciks": success_ciks,
        "coverage_ratio": coverage_ratio,
        "retryable_failures": retryable_failures,
        "permanent_failures": permanent_failures,
        "schema_failures": schema_failures,
        "local_failures": local_failures,
        "total_submissions": total_events,
        "filings_mean_per_cik": filings_mean,
        "filings_p50_per_cik": filings_p50,
        "filings_p95_per_cik": filings_p95,
        "filings_p99_per_cik": filings_p99,
        "distribution_by_form_json": stable_json_dumps(form_distribution),
        "ciks_without_useful_submissions": ciks_no_useful,
        "exact_duplicates_removed": exact_duplicates_removed,
        "late_arrivals": late_arrivals,
        "changed_records": changed_records,
        "latency_mean_seconds": lat_mean,
        "latency_p95_seconds": lat_p95,
        "latency_p99_seconds": lat_p99,
        "retried_call_ratio": retry_ratio,
        "coverage_by_period_json": stable_json_dumps(coverage_by_period),
        "failure_rows": int(len(failures_df)),
    }
    metrics_df = pd.concat([per_cik, pd.DataFrame([aggregate])], axis=0, ignore_index=True, sort=False)
    return metrics_df, aggregate


# -----------------------------
# Main orchestration
# -----------------------------

def run_fetch_submissions(
    *,
    cik_list_path: str,
    config_path: Optional[str],
    run_id: str,
    asof: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir_override: Optional[str] = None,
    max_workers_override: Optional[int] = None,
    force_refresh: bool = False,
    no_skip_existing_success: bool = False,
    no_history_files: bool = False,
) -> dict[str, Any]:
    start_wall = utc_now_iso()
    norm_run_id = normalize_run_id(run_id)
    norm_asof = normalize_asof(asof)
    norm_start_date = normalize_optional_date(start_date)
    norm_end_date = normalize_optional_date(end_date)
    if norm_start_date is not None and norm_end_date is not None and norm_start_date > norm_end_date:
        raise InputValidationError("start_date must be <= end_date")

    config = load_config(config_path)
    cli_ns = argparse.Namespace(
        output_dir=output_dir_override,
        max_workers=max_workers_override,
        force_refresh=force_refresh,
        no_skip_existing_success=no_skip_existing_success,
        no_history_files=no_history_files,
    )
    config = merge_config_with_cli(config, cli_ns)
    ciks = load_cik_list(cik_list_path)
    output_dir = Path(config.output_dir)
    ensure_directory(output_dir / "raw")

    previous_state = load_previous_event_state(output_dir, current_asof=norm_asof, current_run_id=norm_run_id)

    rate_limiter = RateLimiter(config.rate_limit_per_second)
    client = SecSubmissionsClient(config=config, rate_limiter=rate_limiter)

    results: list[FetchResult] = []
    if config.max_workers == 1:
        for cik in ciks:
            results.append(
                process_single_cik(
                    cik=cik,
                    client=client,
                    config=config,
                    output_dir=output_dir,
                    asof=norm_asof,
                    run_id=norm_run_id,
                    start_date=norm_start_date,
                    end_date=norm_end_date,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            fut_map = {
                executor.submit(
                    process_single_cik,
                    cik=cik,
                    client=client,
                    config=config,
                    output_dir=output_dir,
                    asof=norm_asof,
                    run_id=norm_run_id,
                    start_date=norm_start_date,
                    end_date=norm_end_date,
                ): cik
                for cik in ciks
            }
            for fut in as_completed(fut_map):
                results.append(fut.result())
        results.sort(key=lambda x: x.cik)

    payload_rows = [row for result in results for row in result.payload_rows]
    event_rows = [row for result in results for row in result.events]
    failure_rows = [row for result in results for row in result.failures]
    cik_metrics_rows = [result.metrics for result in results]

    events_df = pd.DataFrame(event_rows)
    if events_df.empty:
        events_df = pd.DataFrame(columns=list(REQUIRED_EVENT_FIELDS) + ["event_fingerprint"])
    else:
        if "event_fingerprint" not in events_df.columns:
            events_df["event_fingerprint"] = events_df.apply(event_fingerprint_from_row, axis=1)
        events_df = events_df.sort_values(["cik", "filing_date", "acceptance_datetime", "accession_number"], na_position="last").reset_index(drop=True)

    reconciliation_df = reconcile_current_vs_previous(events_df, previous_state)
    failures_df = pd.DataFrame(failure_rows)
    if failures_df.empty:
        failures_df = pd.DataFrame(columns=["cik", "failure_class", "stage", "reason", "run_id", "asof"])
    payloads_df = pd.DataFrame(payload_rows)
    if payloads_df.empty:
        payloads_df = pd.DataFrame(columns=["cik", "payload_kind", "source_url", "payload_json", "run_id", "asof"])

    metrics_df, aggregate_metrics = build_metrics_dataframe(
        cik_metrics_rows=cik_metrics_rows,
        events_df=events_df,
        reconciliation_df=reconciliation_df,
        failures_df=failures_df,
        total_ciks=len(ciks),
        config=config,
    )

    artifacts: dict[str, Any] = {}
    events_path, events_rows = write_dataframe(events_df, events_output_base(output_dir, norm_asof, norm_run_id), config.table_format, config.allow_csv_fallback)
    artifacts["events_table"] = {"path": events_path, "rows": events_rows}
    payloads_path, payloads_rows = write_dataframe(payloads_df, payloads_output_base(output_dir, norm_asof, norm_run_id), config.table_format, config.allow_csv_fallback)
    artifacts["payloads_table"] = {"path": payloads_path, "rows": payloads_rows}
    failures_path, failures_rows = write_dataframe(failures_df, failures_output_base(output_dir, norm_run_id), config.table_format, config.allow_csv_fallback)
    artifacts["failures_table"] = {"path": failures_path, "rows": failures_rows}
    reconciliation_path, reconciliation_rows = write_dataframe(reconciliation_df, reconciliation_output_base(output_dir, norm_run_id), config.table_format, config.allow_csv_fallback)
    artifacts["reconciliation_table"] = {"path": reconciliation_path, "rows": reconciliation_rows}
    metrics_path, metrics_rows = write_dataframe(metrics_df, metrics_output_base(output_dir, norm_run_id), config.table_format, config.allow_csv_fallback)
    artifacts["metrics_table"] = {"path": metrics_path, "rows": metrics_rows}

    end_wall = utc_now_iso()
    manifest = {
        "module": "data.edgar.fetch_submissions",
        "schema_version": "1.0",
        "run_id": norm_run_id,
        "asof": norm_asof,
        "start_date": norm_start_date,
        "end_date": norm_end_date,
        "config_hash": config_hash(config),
        "target_universe_size": len(ciks),
        "successful_ciks": int((((metrics_df["cik"] != "__GLOBAL__") & (metrics_df["status"] == "success")).sum())) if not metrics_df.empty else 0,
        "failures_by_class": failures_df["failure_class"].value_counts(dropna=False).to_dict() if not failures_df.empty else {},
        "total_submissions_extracted": int(len(events_df)),
        "late_arrivals": int((reconciliation_df.get("reconciliation_class") == "late_arrival").sum()) if not reconciliation_df.empty else 0,
        "changed_records": int((reconciliation_df.get("reconciliation_class") == "changed_record").sum()) if not reconciliation_df.empty else 0,
        "gate": aggregate_metrics.get("status"),
        "gate_reason": aggregate_metrics.get("gate_reason"),
        "aggregate_metrics": aggregate_metrics,
        "artifacts": artifacts,
        "started_at_utc": start_wall,
        "finished_at_utc": end_wall,
    }
    manifest_path = manifest_output_path(output_dir, norm_run_id)
    ensure_directory(manifest_path.parent)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    artifacts["manifest"] = {"path": str(manifest_path), "rows": 1}

    if config.fail_on_gate_fail and aggregate_metrics.get("status") == "fail":
        raise RuntimeError(f"fetch_submissions gate failed: {aggregate_metrics.get('gate_reason')}")
    return manifest


# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and reconcile EDGAR submissions metadata for a CIK universe.")
    parser.add_argument("--cik-list-path", required=True, help="Path to CSV/TXT/JSON/parquet file containing target CIKs")
    parser.add_argument("--config-path", default=None, help="Optional JSON/YAML config path")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument("--asof", required=True, help="Logical execution timestamp")
    parser.add_argument("--start-date", default=None, help="Optional filing_date lower bound (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Optional filing_date upper bound (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default=None, help="Optional override for config.output_dir")
    parser.add_argument("--max-workers", type=int, default=None, help="Optional override for config.max_workers")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cached raw envelopes for the current run")
    parser.add_argument("--no-skip-existing-success", action="store_true", help="Disable reuse of already persisted successful raw envelopes")
    parser.add_argument("--no-history-files", action="store_true", help="Disable fetching historical submissions shards referenced by the root payload")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    manifest = run_fetch_submissions(
        cik_list_path=args.cik_list_path,
        config_path=args.config_path,
        run_id=args.run_id,
        asof=args.asof,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir_override=args.output_dir,
        max_workers_override=args.max_workers,
        force_refresh=args.force_refresh,
        no_skip_existing_success=args.no_skip_existing_success,
        no_history_files=args.no_history_files,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
