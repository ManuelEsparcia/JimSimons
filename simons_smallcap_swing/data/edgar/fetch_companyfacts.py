from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
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


SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
DEFAULT_TAXONOMIES = ["us-gaap", "dei"]
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
PERMANENT_STATUS_CODES = {400, 401, 403, 404, 410, 422}
REQUIRED_FACT_FIELDS = (
    "cik",
    "taxonomy",
    "tag",
    "unit",
    "end_date",
    "filed_date",
    "value",
    "run_id",
    "asof",
)
DEDUP_KEY = (
    "cik",
    "taxonomy",
    "tag",
    "unit",
    "end_date",
    "filed_date",
    "value",
    "accession_number",
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
    taxonomy_filter: list[str] = field(default_factory=lambda: list(DEFAULT_TAXONOMIES))
    min_coverage_ratio: float = 0.80
    systemic_api_failure_ratio: float = 0.50
    payload_anomaly_ratio_warn: float = 0.05
    allow_csv_fallback: bool = True
    table_format: str = "parquet"
    raw_envelope_format: str = "json"
    skip_existing_success: bool = True
    force_refresh: bool = False
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
        if self.rate_limit_per_second <= 0:
            raise InputValidationError("rate_limit_per_second must be > 0")
        if self.max_workers <= 0:
            raise InputValidationError("max_workers must be >= 1")
        if not (0 <= self.min_coverage_ratio <= 1):
            raise InputValidationError("min_coverage_ratio must be in [0, 1]")
        if not (0 <= self.systemic_api_failure_ratio <= 1):
            raise InputValidationError("systemic_api_failure_ratio must be in [0, 1]")
        if not (0 <= self.payload_anomaly_ratio_warn <= 1):
            raise InputValidationError("payload_anomaly_ratio_warn must be in [0, 1]")
        fmt = self.table_format.lower().strip()
        if fmt not in {"parquet", "csv"}:
            raise InputValidationError("table_format must be 'parquet' or 'csv'")
        self.table_format = fmt
        env_fmt = self.raw_envelope_format.lower().strip()
        if env_fmt not in {"json"}:
            raise InputValidationError("raw_envelope_format must currently be 'json'")
        self.raw_envelope_format = env_fmt
        if not isinstance(self.taxonomy_filter, list) or not all(isinstance(x, str) and x.strip() for x in self.taxonomy_filter):
            raise InputValidationError("taxonomy_filter must be a list[str]")
        self.taxonomy_filter = sorted({x.strip() for x in self.taxonomy_filter})


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
    facts: list[dict[str, Any]] = field(default_factory=list)
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


class SecCompanyFactsClient:
    def __init__(self, config: IngestConfig, rate_limiter: RateLimiter) -> None:
        self.config = config
        self.rate_limiter = rate_limiter
        self.headers = {
            "User-Agent": config.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json, text/plain, */*",
            "Host": "data.sec.gov",
        }

    def fetch_json(self, cik: str) -> tuple[str, Optional[dict[str, Any]], Optional[int], int, Optional[str]]:
        url = SEC_COMPANYFACTS_URL.format(cik=cik)
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
    if args.taxonomy_filter:
        cfg_dict["taxonomy_filter"] = [x.strip() for x in args.taxonomy_filter.split(",") if x.strip()]
    if args.force_refresh:
        cfg_dict["force_refresh"] = True
    if args.no_skip_existing_success:
        cfg_dict["skip_existing_success"] = False
    if args.max_workers is not None:
        cfg_dict["max_workers"] = int(args.max_workers)
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
                if col is None:
                    values = df.iloc[:, 0].tolist()
                else:
                    values = df[col].tolist()
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
        if col is None:
            values = df.iloc[:, 0].tolist()
        else:
            values = df[col].tolist()
    else:
        raise InputValidationError("cik_list_path must be .csv, .txt, .json or .parquet")
    if not values:
        raise InputValidationError("CIK list is empty")
    normalized = [normalize_cik(v) for v in values if str(v).strip()]
    if not normalized:
        raise InputValidationError("CIK list contains no valid values after normalization")
    return sorted(set(normalized))


def validate_payload_structure(payload: dict[str, Any], cik: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PayloadSchemaError("payload is not a dict")
    facts = payload.get("facts")
    if not isinstance(facts, dict) or not facts:
        raise PayloadSchemaError("payload.facts missing or empty")
    entity_cik = payload.get("cik")
    if entity_cik is not None:
        try:
            payload_cik = normalize_cik(entity_cik)
            if payload_cik != cik:
                raise PayloadSchemaError(f"payload cik mismatch: payload={payload_cik} requested={cik}")
        except InputValidationError as exc:
            raise PayloadSchemaError(f"payload cik invalid: {exc}") from exc
    parseable_taxonomies = [k for k, v in facts.items() if isinstance(k, str) and isinstance(v, dict)]
    if not parseable_taxonomies:
        raise PayloadSchemaError("payload.facts has no parseable taxonomy nodes")
    return facts


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


def parse_numeric_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def raw_payload_path(base_dir: Path, asof: str, run_id: str, cik: str) -> Path:
    asof_part = asof.replace(":", "-")
    return base_dir / "raw" / "companyfacts" / asof_part / run_id / f"CIK{cik}.json"


def read_existing_raw_envelope(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def persist_raw_envelope(path: Path, envelope: dict[str, Any]) -> str:
    ensure_directory(path.parent)
    text = json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2)
    path.write_text(text, encoding="utf-8")
    return sha256_text(text)


def build_fact_row(
    *,
    cik: str,
    entity_name: Optional[str],
    taxonomy: str,
    tag: str,
    unit: str,
    obs: dict[str, Any],
    tag_meta: dict[str, Any],
    run_id: str,
    asof: str,
    source_url: str,
    payload_sha256: Optional[str],
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    row = {
        "cik": cik,
        "entity_name": entity_name,
        "taxonomy": taxonomy,
        "tag": tag,
        "unit": unit,
        "end_date": parse_date(obs.get("end")),
        "filed_date": parse_date(obs.get("filed")),
        "start_date": parse_date(obs.get("start")),
        "value": parse_numeric_value(obs.get("val")),
        "form": obs.get("form"),
        "accession_number": obs.get("accn"),
        "frame": obs.get("frame"),
        "fy": obs.get("fy"),
        "fp": obs.get("fp"),
        "tag_label": tag_meta.get("label"),
        "tag_description": tag_meta.get("description"),
        "source_url": source_url,
        "payload_sha256": payload_sha256,
        "run_id": run_id,
        "asof": asof,
    }
    missing_required = [field for field in REQUIRED_FACT_FIELDS if row.get(field) is None or row.get(field) == ""]
    if missing_required:
        failure = {
            "cik": cik,
            "failure_class": "row_level_parse_issue",
            "stage": "tabular_extraction",
            "taxonomy": taxonomy,
            "tag": tag,
            "unit": unit,
            "accession_number": row.get("accession_number"),
            "reason": f"missing_required_fields: {','.join(missing_required)}",
            "run_id": run_id,
            "asof": asof,
        }
        return None, failure
    return row, None


def extract_minimal_facts(
    *,
    cik: str,
    payload: dict[str, Any],
    taxonomy_filter: set[str],
    run_id: str,
    asof: str,
    source_url: str,
    payload_sha256: Optional[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    facts_node = validate_payload_structure(payload, cik)
    entity_name = payload.get("entityName")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    observed_taxonomies = sorted(k for k in facts_node.keys() if isinstance(k, str))
    kept_taxonomies = sorted([k for k in observed_taxonomies if k in taxonomy_filter])
    discarded_taxonomies = sorted([k for k in observed_taxonomies if k not in taxonomy_filter])
    exact_duplicate_counter = 0

    for taxonomy in kept_taxonomies:
        tax_node = facts_node.get(taxonomy)
        if not isinstance(tax_node, dict):
            failures.append(
                {
                    "cik": cik,
                    "failure_class": "schema_failure",
                    "stage": "taxonomy_iteration",
                    "taxonomy": taxonomy,
                    "reason": "taxonomy node is not a dict",
                    "run_id": run_id,
                    "asof": asof,
                }
            )
            continue
        for tag, tag_meta in tax_node.items():
            if not isinstance(tag_meta, dict):
                failures.append(
                    {
                        "cik": cik,
                        "failure_class": "schema_failure",
                        "stage": "tag_iteration",
                        "taxonomy": taxonomy,
                        "tag": tag,
                        "reason": "tag node is not a dict",
                        "run_id": run_id,
                        "asof": asof,
                    }
                )
                continue
            units = tag_meta.get("units")
            if not isinstance(units, dict) or not units:
                failures.append(
                    {
                        "cik": cik,
                        "failure_class": "schema_failure",
                        "stage": "unit_iteration",
                        "taxonomy": taxonomy,
                        "tag": tag,
                        "reason": "units missing or empty",
                        "run_id": run_id,
                        "asof": asof,
                    }
                )
                continue
            for unit, observations in units.items():
                if not isinstance(observations, list):
                    failures.append(
                        {
                            "cik": cik,
                            "failure_class": "schema_failure",
                            "stage": "observation_iteration",
                            "taxonomy": taxonomy,
                            "tag": tag,
                            "unit": unit,
                            "reason": "observations are not a list",
                            "run_id": run_id,
                            "asof": asof,
                        }
                    )
                    continue
                for obs in observations:
                    if not isinstance(obs, dict):
                        failures.append(
                            {
                                "cik": cik,
                                "failure_class": "schema_failure",
                                "stage": "observation_iteration",
                                "taxonomy": taxonomy,
                                "tag": tag,
                                "unit": unit,
                                "reason": "observation is not a dict",
                                "run_id": run_id,
                                "asof": asof,
                            }
                        )
                        continue
                    row, failure = build_fact_row(
                        cik=cik,
                        entity_name=entity_name,
                        taxonomy=taxonomy,
                        tag=tag,
                        unit=unit,
                        obs=obs,
                        tag_meta=tag_meta,
                        run_id=run_id,
                        asof=asof,
                        source_url=source_url,
                        payload_sha256=payload_sha256,
                    )
                    if failure is not None:
                        failures.append(failure)
                    elif row is not None:
                        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        before = len(df)
        df = df.drop_duplicates(subset=list(DEDUP_KEY), keep="first").reset_index(drop=True)
        exact_duplicate_counter = before - len(df)
        rows = df.to_dict(orient="records")

    metrics = {
        "observed_taxonomies": observed_taxonomies,
        "kept_taxonomies": kept_taxonomies,
        "discarded_taxonomies": discarded_taxonomies,
        "facts_extracted_pre_dedup": len(rows) + exact_duplicate_counter,
        "facts_extracted_post_dedup": len(rows),
        "exact_duplicates_removed": exact_duplicate_counter,
        "row_level_failures": sum(1 for f in failures if f.get("failure_class") == "row_level_parse_issue"),
        "schema_failures_local": sum(1 for f in failures if f.get("failure_class") == "schema_failure"),
    }
    return rows, failures, metrics


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


def classify_run_gate(
    *,
    total_ciks: int,
    success_ciks: int,
    retryable_failures: int,
    permanent_failures: int,
    schema_failures: int,
    input_failures: int,
    config: IngestConfig,
) -> tuple[str, str]:
    if total_ciks <= 0:
        return "fail", "input_failure"
    if input_failures > 0:
        return "fail", "input_failure"
    api_failure_ratio = (retryable_failures + permanent_failures) / total_ciks
    coverage_ratio = success_ciks / total_ciks
    schema_ratio = schema_failures / total_ciks
    if api_failure_ratio >= config.systemic_api_failure_ratio:
        return "fail", "systemic_api_failure"
    if coverage_ratio < config.min_coverage_ratio:
        return "fail", "coverage_collapse"
    if schema_ratio >= config.payload_anomaly_ratio_warn:
        return "warn", "payload_anomaly"
    return "pass", "ok"


def process_single_cik(
    *,
    cik: str,
    client: SecCompanyFactsClient,
    config: IngestConfig,
    output_dir: Path,
    asof: str,
    run_id: str,
    taxonomy_filter: set[str],
) -> FetchResult:
    url = SEC_COMPANYFACTS_URL.format(cik=cik)
    started_ts = time.monotonic()
    payload_file = raw_payload_path(output_dir, asof, run_id, cik)

    if config.skip_existing_success and not config.force_refresh and payload_file.exists():
        try:
            envelope = read_existing_raw_envelope(payload_file)
            payload = envelope.get("payload")
            fetched_at_utc = envelope.get("fetched_at_utc")
            payload_sha = envelope.get("payload_sha256")
            facts, failures, metrics = extract_minimal_facts(
                cik=cik,
                payload=payload,
                taxonomy_filter=taxonomy_filter,
                run_id=run_id,
                asof=asof,
                source_url=url,
                payload_sha256=payload_sha,
            )
            metrics.update(
                {
                    "cik": cik,
                    "status": "success",
                    "http_status": envelope.get("http_status", 200),
                    "attempts": 0,
                    "latency_seconds": 0.0,
                    "fetched_at_utc": fetched_at_utc,
                    "payload_path": str(payload_file),
                    "source_mode": "reuse_existing_raw",
                }
            )
            return FetchResult(
                cik=cik,
                url=url,
                status="success",
                http_status=int(envelope.get("http_status", 200)),
                attempts=0,
                fetched_at_utc=fetched_at_utc,
                payload_path=str(payload_file),
                payload_sha256=payload_sha,
                facts=facts,
                failures=failures,
                metrics=metrics,
            )
        except Exception as exc:
            fallback_failure = {
                "cik": cik,
                "failure_class": "schema_failure",
                "stage": "reuse_existing_raw",
                "reason": f"failed_to_read_existing_raw: {exc}",
                "run_id": run_id,
                "asof": asof,
            }
            return FetchResult(
                cik=cik,
                url=url,
                status="schema_failure",
                http_status=None,
                attempts=0,
                fetched_at_utc=None,
                payload_path=str(payload_file),
                payload_sha256=None,
                facts=[],
                failures=[fallback_failure],
                metrics={
                    "cik": cik,
                    "status": "schema_failure",
                    "http_status": None,
                    "attempts": 0,
                    "latency_seconds": 0.0,
                    "fetched_at_utc": None,
                    "payload_path": str(payload_file),
                    "source_mode": "reuse_existing_raw_failed",
                },
            )

    status, payload, http_status, attempts, reason = client.fetch_json(cik)
    latency_seconds = round(time.monotonic() - started_ts, 6)
    fetched_at_utc = utc_now_iso()

    if status != "success" or payload is None:
        failure_class = status
        failure_record = {
            "cik": cik,
            "failure_class": failure_class,
            "stage": "http_fetch",
            "http_status": http_status,
            "attempts": attempts,
            "reason": reason,
            "source_url": url,
            "run_id": run_id,
            "asof": asof,
        }
        return FetchResult(
            cik=cik,
            url=url,
            status=status,
            http_status=http_status,
            attempts=attempts,
            fetched_at_utc=fetched_at_utc,
            payload_path=None,
            payload_sha256=None,
            facts=[],
            failures=[failure_record],
            metrics={
                "cik": cik,
                "status": status,
                "http_status": http_status,
                "attempts": attempts,
                "latency_seconds": latency_seconds,
                "fetched_at_utc": fetched_at_utc,
                "payload_path": None,
                "source_mode": "network_fetch",
            },
        )

    try:
        validate_payload_structure(payload, cik)
        raw_json_text = stable_json_dumps(payload)
        payload_sha256 = sha256_text(raw_json_text)
        envelope = {
            "cik": cik,
            "requested_url": url,
            "http_status": http_status,
            "fetched_at_utc": fetched_at_utc,
            "run_id": run_id,
            "asof": asof,
            "payload_sha256": payload_sha256,
            "payload": payload,
        }
        persist_raw_envelope(payload_file, envelope)
        facts, failures, extraction_metrics = extract_minimal_facts(
            cik=cik,
            payload=payload,
            taxonomy_filter=taxonomy_filter,
            run_id=run_id,
            asof=asof,
            source_url=url,
            payload_sha256=payload_sha256,
        )
        metrics = {
            "cik": cik,
            "status": "success",
            "http_status": http_status,
            "attempts": attempts,
            "latency_seconds": latency_seconds,
            "fetched_at_utc": fetched_at_utc,
            "payload_path": str(payload_file),
            "payload_sha256": payload_sha256,
            "source_mode": "network_fetch",
            **extraction_metrics,
        }
        return FetchResult(
            cik=cik,
            url=url,
            status="success",
            http_status=http_status,
            attempts=attempts,
            fetched_at_utc=fetched_at_utc,
            payload_path=str(payload_file),
            payload_sha256=payload_sha256,
            facts=facts,
            failures=failures,
            metrics=metrics,
        )
    except PayloadSchemaError as exc:
        failure_record = {
            "cik": cik,
            "failure_class": "schema_failure",
            "stage": "payload_validation",
            "http_status": http_status,
            "attempts": attempts,
            "reason": str(exc),
            "source_url": url,
            "run_id": run_id,
            "asof": asof,
        }
        return FetchResult(
            cik=cik,
            url=url,
            status="schema_failure",
            http_status=http_status,
            attempts=attempts,
            fetched_at_utc=fetched_at_utc,
            payload_path=None,
            payload_sha256=None,
            facts=[],
            failures=[failure_record],
            metrics={
                "cik": cik,
                "status": "schema_failure",
                "http_status": http_status,
                "attempts": attempts,
                "latency_seconds": latency_seconds,
                "fetched_at_utc": fetched_at_utc,
                "payload_path": None,
                "source_mode": "network_fetch",
            },
        )
    except Exception as exc:
        failure_record = {
            "cik": cik,
            "failure_class": "schema_failure",
            "stage": "payload_persist_or_extract",
            "http_status": http_status,
            "attempts": attempts,
            "reason": f"unexpected_exception: {type(exc).__name__}: {exc}",
            "source_url": url,
            "run_id": run_id,
            "asof": asof,
        }
        return FetchResult(
            cik=cik,
            url=url,
            status="schema_failure",
            http_status=http_status,
            attempts=attempts,
            fetched_at_utc=fetched_at_utc,
            payload_path=None,
            payload_sha256=None,
            facts=[],
            failures=[failure_record],
            metrics={
                "cik": cik,
                "status": "schema_failure",
                "http_status": http_status,
                "attempts": attempts,
                "latency_seconds": latency_seconds,
                "fetched_at_utc": fetched_at_utc,
                "payload_path": None,
                "source_mode": "network_fetch",
            },
        )


def records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def run_ingestion(
    *,
    cik_list_path: str,
    config: IngestConfig,
    run_id: str,
    asof: str,
) -> dict[str, Any]:
    started_at_utc = utc_now_iso()
    ciks = load_cik_list(cik_list_path)
    taxonomy_filter = set(config.taxonomy_filter)
    output_dir = Path(config.output_dir)
    ensure_directory(output_dir)

    rate_limiter = RateLimiter(config.rate_limit_per_second)
    client = SecCompanyFactsClient(config=config, rate_limiter=rate_limiter)

    results: list[FetchResult] = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(
                process_single_cik,
                cik=cik,
                client=client,
                config=config,
                output_dir=output_dir,
                asof=asof,
                run_id=run_id,
                taxonomy_filter=taxonomy_filter,
            ): cik
            for cik in ciks
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda x: x.cik)

    facts_records: list[dict[str, Any]] = []
    failures_records: list[dict[str, Any]] = []
    metrics_records: list[dict[str, Any]] = []
    payload_index_records: list[dict[str, Any]] = []

    for res in results:
        facts_records.extend(res.facts)
        failures_records.extend(res.failures)
        metrics_records.append(res.metrics)
        payload_index_records.append(
            {
                "cik": res.cik,
                "status": res.status,
                "http_status": res.http_status,
                "attempts": res.attempts,
                "fetched_at_utc": res.fetched_at_utc,
                "payload_path": res.payload_path,
                "payload_sha256": res.payload_sha256,
                "run_id": run_id,
                "asof": asof,
            }
        )

    facts_df = records_to_dataframe(facts_records)
    if not facts_df.empty:
        facts_df = facts_df.drop_duplicates(subset=list(DEDUP_KEY), keep="first").reset_index(drop=True)
    failures_df = records_to_dataframe(failures_records)
    metrics_df = records_to_dataframe(metrics_records)
    payload_index_df = records_to_dataframe(payload_index_records)

    success_ciks = int(sum(1 for r in results if r.status == "success"))
    retryable_failures = int(sum(1 for r in results if r.status == "retryable_failure"))
    permanent_failures = int(sum(1 for r in results if r.status == "permanent_failure"))
    schema_failures = int(sum(1 for r in results if r.status == "schema_failure"))
    input_failures = 0

    total_ciks = len(ciks)
    coverage_ratio = success_ciks / total_ciks if total_ciks else 0.0
    gate, gate_reason = classify_run_gate(
        total_ciks=total_ciks,
        success_ciks=success_ciks,
        retryable_failures=retryable_failures,
        permanent_failures=permanent_failures,
        schema_failures=schema_failures,
        input_failures=input_failures,
        config=config,
    )

    facts_path, facts_rows = write_dataframe(
        facts_df,
        output_dir / f"companyfacts_facts_{run_id}",
        config.table_format,
        config.allow_csv_fallback,
    )
    failures_path, failures_rows = write_dataframe(
        failures_df,
        output_dir / f"companyfacts_failures_{run_id}",
        config.table_format,
        config.allow_csv_fallback,
    )
    metrics_path, metrics_rows = write_dataframe(
        metrics_df,
        output_dir / f"companyfacts_metrics_{run_id}",
        config.table_format,
        config.allow_csv_fallback,
    )
    payload_index_path, payload_index_rows = write_dataframe(
        payload_index_df,
        output_dir / f"companyfacts_payload_index_{run_id}",
        config.table_format,
        config.allow_csv_fallback,
    )

    taxonomies_seen = sorted(
        {
            str(t)
            for rec in metrics_records
            for t in rec.get("observed_taxonomies", [])
            if isinstance(t, str) and t.strip()
        }
    )
    discarded_taxonomies_seen = sorted(
        {
            str(t)
            for rec in metrics_records
            for t in rec.get("discarded_taxonomies", [])
            if isinstance(t, str) and t.strip()
        }
    )

    ended_at_utc = utc_now_iso()
    manifest = {
        "module": "data.edgar.fetch_companyfacts",
        "run_id": run_id,
        "asof": asof,
        "config_hash": config_hash(config),
        "config": asdict(config),
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "requested_cik_count": total_ciks,
        "success_cik_count": success_ciks,
        "retryable_failure_cik_count": retryable_failures,
        "permanent_failure_cik_count": permanent_failures,
        "schema_failure_cik_count": schema_failures,
        "coverage_ratio": round(coverage_ratio, 6),
        "facts_total_rows": int(facts_rows),
        "failures_total_rows": int(failures_rows),
        "metrics_total_rows": int(metrics_rows),
        "payload_index_total_rows": int(payload_index_rows),
        "gate": gate,
        "gate_reason": gate_reason,
        "taxonomy_filter": sorted(config.taxonomy_filter),
        "taxonomies_seen": taxonomies_seen,
        "discarded_taxonomies_seen": discarded_taxonomies_seen,
        "artifacts": {
            "facts": facts_path,
            "failures": failures_path,
            "metrics": metrics_path,
            "payload_index": payload_index_path,
            "raw_payload_dir": str((output_dir / "raw" / "companyfacts" / asof.replace(":", "-") / run_id).resolve()),
        },
    }
    manifest_path = output_dir / f"companyfacts_manifest_{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "facts_df": facts_df,
        "failures_df": failures_df,
        "metrics_df": metrics_df,
        "payload_index_df": payload_index_df,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch raw SEC companyfacts payloads and extract a conservative minimal facts table."
    )
    parser.add_argument("--cik-list-path", required=True, help="Path to the input CIK universe (.csv/.txt/.json/.parquet)")
    parser.add_argument("--config-path", required=False, help="Path to JSON/YAML config")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument("--asof", required=True, help="Logical execution timestamp in ISO-8601 format")
    parser.add_argument("--output-dir", required=False, help="Override output directory")
    parser.add_argument(
        "--taxonomy-filter",
        required=False,
        help="Comma-separated list of allowed taxonomies, e.g. us-gaap,dei",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Ignore existing raw payloads and re-fetch from SEC")
    parser.add_argument(
        "--no-skip-existing-success",
        action="store_true",
        help="Do not reuse existing raw payloads even when present",
    )
    parser.add_argument("--max-workers", type=int, required=False, help="Override max_workers from config")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_id = normalize_run_id(args.run_id)
        asof = normalize_asof(args.asof)
        base_config = load_config(args.config_path)
        config = merge_config_with_cli(base_config, args)
        result = run_ingestion(
            cik_list_path=args.cik_list_path,
            config=config,
            run_id=run_id,
            asof=asof,
        )
        manifest = result["manifest"]
        print(json.dumps({
            "status": manifest["gate"],
            "gate_reason": manifest["gate_reason"],
            "manifest_path": result["manifest_path"],
            "requested_cik_count": manifest["requested_cik_count"],
            "success_cik_count": manifest["success_cik_count"],
            "coverage_ratio": manifest["coverage_ratio"],
            "facts_total_rows": manifest["facts_total_rows"],
        }, ensure_ascii=False))
        if config.fail_on_gate_fail and manifest["gate"] == "fail":
            return 2
        return 0
    except InputValidationError as exc:
        print(json.dumps({"status": "fail", "gate_reason": "input_failure", "error": str(exc)}), file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive top-level boundary
        print(
            json.dumps(
                {
                    "status": "fail",
                    "gate_reason": "unexpected_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
