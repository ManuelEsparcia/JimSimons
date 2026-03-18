"""
data.edgar — SEC EDGAR fundamentals pipeline (point-in-time).

Pipeline: ticker_cik → fetch_submissions → fetch_companyfacts
          → parse_xbrl → point_in_time → filings_flags → edgar_qc

Shared infrastructure centralised here to eliminate ~920 lines of
duplicated utilities across the 7 submodules. Every submodule imports
from this __init__ instead of redefining its own copy.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time as _time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Try parent-level imports (graceful if standalone)
try:
    from .. import Severity, utc_now_iso as _parent_utc
except ImportError:
    Severity = None
    _parent_utc = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEC_USER_AGENT = "QuantResearch/1.0 (research@example.com)"
SEC_MAX_RPS = 10  # SEC rate limit
RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
PERMANENT_STATUS_CODES = frozenset({400, 401, 403, 404, 410, 422})

CIK_WIDTH = 10
CIK_REGEX = re.compile(r"^\d{1,10}$")


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

class EdgarError(RuntimeError):
    """Base error for the EDGAR pipeline."""

class InputValidationError(EdgarError, ValueError):
    """Invalid input data or configuration."""

class PayloadSchemaError(EdgarError):
    """SEC payload doesn't match expected schema."""

class NetworkError(EdgarError):
    """Network/HTTP error fetching from SEC."""

class ParseError(EdgarError):
    """Error parsing XBRL/filing data."""

class PITError(EdgarError):
    """Point-in-time construction error."""


# ---------------------------------------------------------------------------
# Identity normalization (was 4 copies across files)
# ---------------------------------------------------------------------------

def normalize_cik(value: Any) -> str:
    """Normalize CIK to zero-padded 10-digit string.

    Handles: int, float, str with/without leading zeros.
    Returns: '0000012345' format, or raises InputValidationError.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise InputValidationError("CIK is None/NaN")
    raw = str(value).strip().lstrip("0") or "0"
    if not CIK_REGEX.match(raw):
        raise InputValidationError(f"Invalid CIK: {value!r}")
    return raw.zfill(CIK_WIDTH)


def normalize_symbol(value: Any) -> str:
    """Normalize ticker symbol: uppercase, stripped, no whitespace."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip().upper().replace(" ", "")


def normalize_accession_number(value: Any) -> Optional[str]:
    """Normalize SEC accession number: strip, lowercase, validate format."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw.lower() in ("nan", "none", ""):
        return None
    return raw


# ---------------------------------------------------------------------------
# Date/time parsing (was 5 copies)
# ---------------------------------------------------------------------------

def parse_date(value: Any) -> Optional[pd.Timestamp]:
    """Parse a date value to pd.Timestamp, return None on failure."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        return ts.normalize()  # strip time component
    except Exception:
        return None


def parse_datetime_utc(value: Any) -> Optional[pd.Timestamp]:
    """Parse a datetime value to pd.Timestamp (UTC), return None on failure."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        return ts
    except Exception:
        return None


def parse_date_series(series: pd.Series) -> pd.Series:
    """Parse a Series of date values to datetime64."""
    return pd.to_datetime(series, errors="coerce")


def parse_numeric(value: Any) -> Optional[float]:
    """Parse a numeric value, return None on failure."""
    if value is None:
        return None
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def parse_boolish(value: Any) -> Optional[bool]:
    """Parse a boolean-ish value (handles strings, ints, etc.)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "t", "y"):
        return True
    if s in ("false", "0", "no", "f", "n"):
        return False
    return None


# ---------------------------------------------------------------------------
# Hashing and IDs (was 5 copies)
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    """Current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_text(text: str) -> str:
    """SHA-256 hash of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> Optional[str]:
    """SHA-256 hash of a file's contents."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(cfg: Mapping[str, Any]) -> str:
    """Deterministic hash of a configuration dict."""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def json_safe(value: Any) -> Any:
    """Convert numpy/pandas types to JSON-serializable."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat() if pd.notna(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(str(x) for x in value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def stable_json_dumps(obj: Any) -> str:
    """Deterministic JSON serialization."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


# ---------------------------------------------------------------------------
# Config loading (was 6 copies)
# ---------------------------------------------------------------------------

def load_yaml_or_json(path: Path) -> Any:
    """Load YAML or JSON file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
            return yaml.safe_load(text)
        except ImportError:
            raise EdgarError("PyYAML required for .yaml config files")
    return json.loads(text)


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Deep merge two dicts (update wins on leaf conflicts)."""
    result = dict(base)
    for k, v in update.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# IO: read/write tables (was 6 copies)
# ---------------------------------------------------------------------------

def read_dataframe(path: str | Path) -> pd.DataFrame:
    """Read a DataFrame from parquet or CSV (auto-detect)."""
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    # Try parquet first, fallback to csv
    pq = p.with_suffix(".parquet")
    if pq.exists():
        return pd.read_parquet(pq)
    csv = p.with_suffix(".csv")
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No parquet or csv found for {p}")


def write_parquet_safe(
    df: pd.DataFrame,
    path: Path,
    *,
    allow_csv_fallback: bool = True,
    compression: str = "snappy",
) -> str:
    """Write DataFrame to parquet with CSV fallback.

    Returns the actual path written (may differ if fallback used).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False, compression=compression)
        return str(path)
    except Exception as e:
        if allow_csv_fallback:
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            LOGGER.warning("Parquet write failed (%s), fell back to CSV: %s", e, csv_path)
            return str(csv_path)
        raise


def write_json_safe(payload: Mapping[str, Any], path: Path) -> str:
    """Write JSON file with directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return str(path)


def ensure_directory(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Column normalization (was 3 copies)
# ---------------------------------------------------------------------------

def normalize_columns(
    df: pd.DataFrame,
    aliases: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Rename columns using alias map: {canonical_name: [alias1, alias2, ...]}."""
    lowered = {c.lower(): c for c in df.columns}
    rename_map = {}
    for canonical, candidates in aliases.items():
        if canonical.lower() in lowered:
            continue  # already has canonical name
        for alias in candidates:
            if alias.lower() in lowered:
                rename_map[lowered[alias.lower()]] = canonical
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def normalize_metric(value: Any) -> Optional[str]:
    """Normalize XBRL metric name: lowercase, strip whitespace."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value).strip().lower()


# ---------------------------------------------------------------------------
# SEC HTTP client infrastructure (was 2 copies of RateLimiter)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Thread-safe rate limiter for SEC API calls."""

    def __init__(self, max_rps: float = SEC_MAX_RPS):
        self._interval = 1.0 / max_rps
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = _time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._interval:
                _time.sleep(self._interval - elapsed)
            self._last_call = _time.monotonic()


@dataclass
class SecClientConfig:
    """Configuration for SEC HTTP client."""
    user_agent: str = SEC_USER_AGENT
    max_rps: float = SEC_MAX_RPS
    max_retries: int = 3
    backoff_base: float = 1.0
    backoff_max: float = 30.0
    timeout: int = 30


def sec_fetch_json(
    url: str,
    *,
    client_config: SecClientConfig | None = None,
    rate_limiter: RateLimiter | None = None,
) -> dict[str, Any]:
    """Fetch JSON from SEC with rate limiting and retries.

    Raises NetworkError on permanent failure.
    """
    import requests

    cfg = client_config or SecClientConfig()
    limiter = rate_limiter or RateLimiter(cfg.max_rps)
    headers = {"User-Agent": cfg.user_agent, "Accept": "application/json"}

    last_error = None
    for attempt in range(cfg.max_retries + 1):
        limiter.wait()
        try:
            resp = requests.get(url, headers=headers, timeout=cfg.timeout)

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in PERMANENT_STATUS_CODES:
                raise NetworkError(f"Permanent HTTP {resp.status_code} for {url}")

            if resp.status_code in RETRYABLE_STATUS_CODES:
                last_error = NetworkError(f"HTTP {resp.status_code} for {url}")
                backoff = min(cfg.backoff_base * (2 ** attempt), cfg.backoff_max)
                import random
                _time.sleep(backoff + random.uniform(0, backoff * 0.1))
                continue

            raise NetworkError(f"Unexpected HTTP {resp.status_code} for {url}")

        except requests.RequestException as e:
            last_error = NetworkError(f"Request failed for {url}: {e}")
            if attempt < cfg.max_retries:
                backoff = min(cfg.backoff_base * (2 ** attempt), cfg.backoff_max)
                _time.sleep(backoff)

    raise last_error or NetworkError(f"All retries exhausted for {url}")
