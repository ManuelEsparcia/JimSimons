from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import logging
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# -----------------------------------------------------------------------------
# Exceptions and enums
# -----------------------------------------------------------------------------


class FetchPricesError(RuntimeError):
    """Raised when a hard contract, IO or persistence invariant is violated."""


class ProviderError(RuntimeError):
    """Base class for provider-side failures."""


class RetriableProviderError(ProviderError):
    """Transient error that may succeed on retry."""


class PermanentProviderError(ProviderError):
    """Permanent provider error for a symbol or request."""


class MissingCredentialsError(PermanentProviderError):
    """Raised when provider credentials are invalid or absent."""


class Severity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    FAIL = "FAIL"


class FetchMode(str, Enum):
    FULL_REFRESH = "full_refresh"
    INCREMENTAL = "incremental"
    RECONCILE = "reconcile"
    BACKFILL = "backfill"


class FailureCode(str, Enum):
    UNIVERSE_READ_ERROR = "UNIVERSE_READ_ERROR"
    CALENDAR_READ_ERROR = "CALENDAR_READ_ERROR"
    CONFIG_ERROR = "CONFIG_ERROR"
    PROVIDER_CLIENT_ERROR = "PROVIDER_CLIENT_ERROR"
    MISSING_REQUIRED_COLUMN = "MISSING_REQUIRED_COLUMN"
    SCHEMA_DRIFT = "SCHEMA_DRIFT"
    INVALID_SYMBOL = "INVALID_SYMBOL"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_RATE_LIMIT = "PROVIDER_RATE_LIMIT"
    PROVIDER_HTTP_5XX = "PROVIDER_HTTP_5XX"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    SYMBOL_NOT_FOUND = "SYMBOL_NOT_FOUND"
    EMPTY_PAYLOAD = "EMPTY_PAYLOAD"
    UNPARSABLE_TIMESTAMP = "UNPARSABLE_TIMESTAMP"
    INVALID_TRADE_DATE = "INVALID_TRADE_DATE"
    INVALID_PRICE_GEOMETRY = "INVALID_PRICE_GEOMETRY"
    INVALID_PRICE_VALUE = "INVALID_PRICE_VALUE"
    INVALID_VOLUME_VALUE = "INVALID_VOLUME_VALUE"
    DUPLICATE_PROVIDER_ROW = "DUPLICATE_PROVIDER_ROW"
    PERSISTENCE_ERROR = "PERSISTENCE_ERROR"
    REVISIONS_DETECTED = "REVISIONS_DETECTED"
    NO_NEW_DATES = "NO_NEW_DATES"
    NO_ACTIVITY_OBSERVED = "NO_ACTIVITY_OBSERVED"
    FALLBACK_USED = "FALLBACK_USED"
    COVERAGE_GAPS = "COVERAGE_GAPS"
    BACKFILL_NO_GAPS = "BACKFILL_NO_GAPS"


SEVERITY_ORDER = {Severity.INFO.value: 0, Severity.WARN.value: 1, Severity.FAIL.value: 2}

REQUIRED_CANONICAL_COLUMNS = [
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "trades_count",
    "currency",
    "source_provider",
    "provider_dataset_id",
    "provider_row_id",
    "provider_timestamp_utc",
    "ingest_ts_utc",
    "run_id",
    "config_hash",
    "universe_snapshot_id",
    "fetch_mode",
    "fallback_used",
]

COMPARISON_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "trades_count",
    "currency",
    "provider_dataset_id",
]


# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    base_delay_sec: float = 0.5
    alpha: float = 2.0
    cap_delay_sec: float = 8.0
    jitter_low: float = 0.9
    jitter_high: float = 1.1


@dataclass(frozen=True)
class RateLimitPolicy:
    min_interval_sec: float = 0.0


@dataclass(frozen=True)
class ValidationPolicy:
    require_positive_prices: bool = True
    allow_zero_volume: bool = True
    drop_invalid_rows: bool = True
    duplicate_resolution: str = "last"
    warn_on_optional_missing: bool = True
    missing_warn_threshold: float = 0.01
    missing_fail_threshold: float = 0.20


@dataclass(frozen=True)
class PersistencePolicy:
    output_root: str = "data/price"
    compression: str = "snappy"
    persist_canonical: bool = True
    persist_raw: bool = True
    raw_append_dedupe: bool = True
    overwrite_existing_run: bool = False
    full_refresh_overwrite_canonical: bool = True
    reconcile_overwrite_canonical: bool = True
    backfill_overwrite_canonical: bool = True
    incremental_update_existing_range: bool = False
    canonical_relpath: str = "canonical/canonical_prices.parquet"


@dataclass(frozen=True)
class FetchPolicy:
    frequency: str = "1d"
    chunk_size_tickers: int = 250
    mode: str = FetchMode.INCREMENTAL.value
    allow_fallback: bool = True
    backfill_fetch_full_gap_span: bool = True
    compare_existing_for_revisions: bool = True
    provider_priority: Sequence[str] = field(default_factory=lambda: ["primary", "fallback"])


@dataclass(frozen=True)
class NormalizationPolicy:
    market_timezone: str = "America/New_York"
    timestamp_to_trade_date: str = "market_tz_date"
    trim_symbols: bool = True
    uppercase_symbols: bool = True


@dataclass(frozen=True)
class FetchPricesConfig:
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    rate_limit: RateLimitPolicy = field(default_factory=RateLimitPolicy)
    validation: ValidationPolicy = field(default_factory=ValidationPolicy)
    persistence: PersistencePolicy = field(default_factory=PersistencePolicy)
    fetch: FetchPolicy = field(default_factory=FetchPolicy)
    normalization: NormalizationPolicy = field(default_factory=NormalizationPolicy)


@dataclass(frozen=True)
class ProviderSpec:
    alias: str
    kind: str
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    dataset_id: Optional[str] = None
    timezone: Optional[str] = None
    enabled: bool = True


@dataclass
class UniverseWindow:
    symbol: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    expected_sessions: List[pd.Timestamp]


@dataclass
class FetchArtifacts:
    raw: pd.DataFrame
    canonical: pd.DataFrame
    coverage: pd.DataFrame
    failures: pd.DataFrame
    revisions: pd.DataFrame
    summary: Dict[str, Any]
    manifest: Dict[str, Any]


# -----------------------------------------------------------------------------
# Provider protocol and built-in provider clients
# -----------------------------------------------------------------------------


class ProviderClient(Protocol):
    provider_name: str
    provider_alias: str
    dataset_id: Optional[str]

    def fetch_daily_bars(self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp, frequency: str) -> pd.DataFrame:
        ...


class CSVDirectoryProvider:
    """Fetches one CSV/Parquet file per symbol from local storage.

    Supported params:
    - root_dir
    - file_pattern (default "{symbol}.csv")
    - format in {csv, parquet}
    - symbol_column optional for single file with many symbols
    - column_map
    - dataset_id
    """

    def __init__(self, spec: ProviderSpec):
        params = dict(spec.params)
        self.provider_alias = spec.alias
        self.provider_name = spec.name
        self.dataset_id = spec.dataset_id or params.get("dataset_id")
        self.root_dir = Path(str(params.get("root_dir", ""))).expanduser().resolve()
        self.file_pattern = str(params.get("file_pattern", "{symbol}.csv"))
        self.file_format = str(params.get("format", "csv")).lower()
        self.symbol_column = params.get("symbol_column")
        self.column_map = dict(params.get("column_map", {}))
        if not self.root_dir.exists():
            raise PermanentProviderError(f"CSVDirectoryProvider root_dir does not exist: {self.root_dir}")

    def _read_frame(self, path: Path) -> pd.DataFrame:
        if self.file_format == "csv":
            return pd.read_csv(path)
        if self.file_format == "parquet":
            return pd.read_parquet(path)
        raise PermanentProviderError(f"Unsupported CSVDirectoryProvider format: {self.file_format}")

    def fetch_daily_bars(self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp, frequency: str) -> pd.DataFrame:
        path = self.root_dir / self.file_pattern.format(symbol=symbol)
        if not path.exists():
            raise PermanentProviderError(f"Symbol file not found for {symbol}: {path}")
        frame = self._read_frame(path)
        if self.symbol_column and self.symbol_column in frame.columns:
            frame = frame.loc[frame[self.symbol_column].astype(str) == str(symbol)].copy()
        if frame.empty:
            return pd.DataFrame()
        if self.column_map:
            frame = frame.rename(columns=self.column_map)
        if "symbol" not in frame.columns:
            frame["symbol"] = symbol
        return frame


class CSVFileProvider:
    """Fetches many symbols from one CSV/Parquet file."""

    def __init__(self, spec: ProviderSpec):
        params = dict(spec.params)
        self.provider_alias = spec.alias
        self.provider_name = spec.name
        self.dataset_id = spec.dataset_id or params.get("dataset_id")
        self.path = Path(str(params.get("path", ""))).expanduser().resolve()
        self.file_format = str(params.get("format", self.path.suffix.lstrip(".") or "csv")).lower()
        self.symbol_column = str(params.get("symbol_column", "symbol"))
        self.column_map = dict(params.get("column_map", {}))
        if not self.path.exists():
            raise PermanentProviderError(f"CSVFileProvider path does not exist: {self.path}")
        self._cache: Optional[pd.DataFrame] = None

    def _read_all(self) -> pd.DataFrame:
        if self._cache is not None:
            return self._cache.copy()
        if self.file_format == "csv":
            frame = pd.read_csv(self.path)
        elif self.file_format == "parquet":
            frame = pd.read_parquet(self.path)
        else:
            raise PermanentProviderError(f"Unsupported CSVFileProvider format: {self.file_format}")
        if self.column_map:
            frame = frame.rename(columns=self.column_map)
        self._cache = frame.copy()
        return frame

    def fetch_daily_bars(self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp, frequency: str) -> pd.DataFrame:
        frame = self._read_all()
        if self.symbol_column not in frame.columns:
            raise PermanentProviderError(f"CSVFileProvider missing symbol column: {self.symbol_column}")
        frame = frame.loc[frame[self.symbol_column].astype(str) == str(symbol)].copy()
        if frame.empty:
            return pd.DataFrame()
        if "symbol" not in frame.columns:
            frame["symbol"] = symbol
        return frame


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_timestamp_utc(value: str | pd.Timestamp | datetime | None) -> Optional[pd.Timestamp]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def normalize_date(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    try:
        if pd.isna(value) and not isinstance(value, (str, bytes)):
            return None
    except Exception:
        pass
    return value


def stable_hash_payload(payload: Any, prefix: str = "") -> str:
    raw = json.dumps(json_safe(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return f"{prefix}_{digest}" if prefix else digest


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(json_safe(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def git_code_version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return f"unknown::{platform.python_version()}"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def chunked(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def highest_severity(values: Iterable[str]) -> str:
    best = Severity.INFO.value
    for value in values:
        if SEVERITY_ORDER.get(str(value), -1) > SEVERITY_ORDER.get(best, -1):
            best = str(value)
    return best


class RateLimiter:
    def __init__(self, min_interval_sec: float = 0.0):
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self._last_call = 0.0

    def acquire(self) -> None:
        if self.min_interval_sec <= 0:
            return
        now = time.monotonic()
        wait = self.min_interval_sec - (now - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()


# -----------------------------------------------------------------------------
# Configuration loading
# -----------------------------------------------------------------------------


def read_data_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise FetchPricesError("PyYAML is required to load YAML config files.")
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise FetchPricesError(f"Unsupported config file format: {path}")


def deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_merge(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def dataclass_from_mapping(cls: Any, mapping: Optional[Mapping[str, Any]]) -> Any:
    mapping = dict(mapping or {})
    kwargs: Dict[str, Any] = {}
    for field_info in dataclasses.fields(cls):
        if field_info.name not in mapping:
            continue
        value = mapping[field_info.name]
        field_type = field_info.type
        if dataclasses.is_dataclass(field_type):
            kwargs[field_info.name] = dataclass_from_mapping(field_type, value)
        else:
            kwargs[field_info.name] = value
    return cls(**kwargs)


def load_config(config_path: Optional[Path], provider_config_path: Path) -> Tuple[FetchPricesConfig, Dict[str, ProviderSpec], Dict[str, Any], str]:
    provider_payload = read_data_file(provider_config_path)
    payload: Dict[str, Any] = {}
    if config_path is not None:
        payload = dict(read_data_file(config_path) or {})
    payload = deep_merge(payload, provider_payload)

    cfg = FetchPricesConfig(
        retry=dataclass_from_mapping(RetryPolicy, payload.get("retry")),
        rate_limit=dataclass_from_mapping(RateLimitPolicy, payload.get("rate_limit")),
        validation=dataclass_from_mapping(ValidationPolicy, payload.get("validation")),
        persistence=dataclass_from_mapping(PersistencePolicy, payload.get("persistence")),
        fetch=dataclass_from_mapping(FetchPolicy, payload.get("fetch")),
        normalization=dataclass_from_mapping(NormalizationPolicy, payload.get("normalization")),
    )

    providers_raw = payload.get("providers") or payload.get("provider")
    if not providers_raw or not isinstance(providers_raw, Mapping):
        raise FetchPricesError("Provider config must define a 'providers' mapping with at least one provider.")

    providers: Dict[str, ProviderSpec] = {}
    for alias, spec in providers_raw.items():
        if not isinstance(spec, Mapping):
            raise FetchPricesError(f"Provider spec for alias '{alias}' must be a mapping.")
        enabled = bool(spec.get("enabled", True))
        provider = ProviderSpec(
            alias=str(alias),
            kind=str(spec.get("kind", spec.get("type", ""))).strip(),
            name=str(spec.get("name", alias)).strip(),
            params=dict(spec.get("params", {})),
            dataset_id=spec.get("dataset_id"),
            timezone=spec.get("timezone"),
            enabled=enabled,
        )
        if not provider.kind:
            raise FetchPricesError(f"Provider '{alias}' must declare a non-empty 'kind'.")
        providers[provider.alias] = provider

    active = [alias for alias in cfg.fetch.provider_priority if alias in providers and providers[alias].enabled]
    if not active:
        active = [alias for alias, spec in providers.items() if spec.enabled]
    if not active:
        raise FetchPricesError("No enabled providers available after reading config.")

    config_hash = stable_hash_payload(payload, prefix="cfg")
    return cfg, providers, payload, config_hash


# -----------------------------------------------------------------------------
# Input loading
# -----------------------------------------------------------------------------


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(data)
    raise FetchPricesError(f"Unsupported table format: {path}")


SYMBOL_COLUMN_CANDIDATES = ["symbol", "ticker", "security_symbol", "provider_symbol"]
DATE_COLUMN_CANDIDATES = ["date", "session_date", "trade_date", "calendar_date"]
ELIGIBILITY_COLUMN_CANDIDATES = ["is_eligible", "eligible", "in_universe", "is_in_universe", "active"]


def choose_first_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    cols = {str(c) for c in columns}
    for candidate in candidates:
        if candidate in cols:
            return candidate
    return None


def normalize_symbol(symbol: Any, trim: bool = True, uppercase: bool = True) -> str:
    value = str(symbol)
    if trim:
        value = value.strip()
    if uppercase:
        value = value.upper()
    return value


def load_universe_snapshot(path: Path, start_date: pd.Timestamp, end_date: pd.Timestamp, cfg: FetchPricesConfig) -> Tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FetchPricesError(f"Universe snapshot path does not exist: {path}")
    frame = load_table(path)
    if frame.empty:
        raise FetchPricesError("Universe snapshot is empty.")
    symbol_col = choose_first_column(frame.columns, SYMBOL_COLUMN_CANDIDATES)
    if symbol_col is None:
        raise FetchPricesError(f"Universe snapshot missing symbol column; candidates={SYMBOL_COLUMN_CANDIDATES}")
    frame = frame.rename(columns={symbol_col: "symbol"}).copy()
    frame["symbol"] = frame["symbol"].map(lambda x: normalize_symbol(x, cfg.normalization.trim_symbols, cfg.normalization.uppercase_symbols))
    frame = frame.loc[frame["symbol"].astype(str).str.len() > 0].copy()
    if frame.empty:
        raise FetchPricesError("Universe snapshot contains no usable symbols after normalization.")

    date_col = choose_first_column(frame.columns, DATE_COLUMN_CANDIDATES)
    if date_col is not None:
        frame = frame.rename(columns={date_col: "date"}).copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame = frame.loc[frame["date"].notna()].copy()
    else:
        frame["date"] = pd.NaT

    eligibility_col = choose_first_column(frame.columns, ELIGIBILITY_COLUMN_CANDIDATES)
    if eligibility_col is not None:
        frame["is_eligible"] = frame[eligibility_col].astype(bool)
    elif "membership_state" in frame.columns:
        frame["is_eligible"] = frame["membership_state"].astype(str).str.upper().isin({"ENTER", "IN", "STAY", "CURRENT", "ELIGIBLE"})
    else:
        frame["is_eligible"] = True

    if "date" in frame.columns and frame["date"].notna().any():
        frame = frame.loc[(frame["date"].isna()) | ((frame["date"] >= start_date) & (frame["date"] <= end_date))].copy()
        if frame.empty:
            raise FetchPricesError("Universe snapshot has no rows overlapping the requested date range.")

    snapshot_id = stable_hash_payload({"path": str(path), "sha256": file_sha256(path)}, prefix="universe")
    return frame.reset_index(drop=True), snapshot_id


def load_market_calendar(path: Path, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DatetimeIndex:
    if not path.exists():
        raise FetchPricesError(f"Calendar path does not exist: {path}")
    frame = load_table(path)
    if frame.empty:
        raise FetchPricesError("Calendar file is empty.")
    date_col = choose_first_column(frame.columns, DATE_COLUMN_CANDIDATES)
    if date_col is None:
        raise FetchPricesError(f"Calendar file missing date column; candidates={DATE_COLUMN_CANDIDATES}")
    frame = frame.rename(columns={date_col: "date"}).copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.loc[frame["date"].notna()].copy()
    if "is_session" in frame.columns:
        frame = frame.loc[frame["is_session"].astype(bool)].copy()
    sessions = pd.DatetimeIndex(sorted(pd.unique(frame["date"])))
    sessions = sessions[(sessions >= start_date) & (sessions <= end_date)]
    if len(sessions) == 0:
        raise FetchPricesError("Calendar has zero sessions in requested range.")
    return sessions


# -----------------------------------------------------------------------------
# Universe to fetch windows
# -----------------------------------------------------------------------------


def build_universe_windows(universe: pd.DataFrame, sessions: pd.DatetimeIndex, start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[UniverseWindow]:
    if len(sessions) == 0:
        return []

    sessions_set = set(pd.Timestamp(x).normalize() for x in sessions)
    windows: List[UniverseWindow] = []

    dated_rows = universe.loc[universe["date"].notna()].copy() if "date" in universe.columns else pd.DataFrame()
    if not dated_rows.empty:
        eligible = dated_rows.loc[dated_rows["is_eligible"].fillna(False)].copy()
        if not eligible.empty:
            for symbol, group in eligible.groupby("symbol", sort=True):
                active_dates = sorted(d for d in pd.unique(group["date"]) if pd.Timestamp(d).normalize() in sessions_set)
                if not active_dates:
                    continue
                expected = [d for d in sessions if d in set(active_dates)]
                if not expected:
                    continue
                windows.append(
                    UniverseWindow(
                        symbol=str(symbol),
                        start_date=pd.Timestamp(expected[0]).normalize(),
                        end_date=pd.Timestamp(expected[-1]).normalize(),
                        expected_sessions=[pd.Timestamp(x).normalize() for x in expected],
                    )
                )
            if windows:
                return windows

    # Snapshot without per-date history: assume active throughout requested range.
    eligible = universe.loc[universe["is_eligible"].fillna(True)].copy()
    symbols = sorted(pd.unique(eligible["symbol"].astype(str)))
    return [
        UniverseWindow(symbol=symbol, start_date=start_date, end_date=end_date, expected_sessions=[pd.Timestamp(x).normalize() for x in sessions])
        for symbol in symbols
    ]


# -----------------------------------------------------------------------------
# Provider construction and fetching
# -----------------------------------------------------------------------------


def import_object(dotted_path: str) -> Any:
    module_name, _, object_name = dotted_path.rpartition(".")
    if not module_name:
        raise FetchPricesError(f"Invalid dotted path: {dotted_path}")
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def build_provider_client(spec: ProviderSpec) -> ProviderClient:
    kind = spec.kind.lower()
    if kind in {"csv_directory", "csvdir", "directory"}:
        return CSVDirectoryProvider(spec)
    if kind in {"csv_file", "csvfile", "file"}:
        return CSVFileProvider(spec)
    if kind in {"python", "custom", "class"}:
        class_path = spec.params.get("class_path")
        if not class_path:
            raise FetchPricesError(f"Custom provider '{spec.alias}' requires params.class_path")
        cls = import_object(str(class_path))
        client = cls(spec)  # type: ignore[call-arg]
        return client
    raise FetchPricesError(f"Unsupported provider kind for '{spec.alias}': {spec.kind}")


@dataclass
class FetchAttemptResult:
    frame: Optional[pd.DataFrame]
    provider_alias: str
    provider_name: str
    dataset_id: Optional[str]
    retry_count: int
    error_code: Optional[str]
    error_class: Optional[str]
    message: Optional[str]
    fallback_used: bool
    http_like_error: bool = False


def classify_provider_exception(exc: Exception) -> Tuple[str, str, bool]:
    message = str(exc)
    text = message.lower()
    if isinstance(exc, MissingCredentialsError):
        return FailureCode.PROVIDER_CLIENT_ERROR.value, exc.__class__.__name__, False
    if isinstance(exc, RetriableProviderError):
        if "429" in text or "rate" in text:
            return FailureCode.PROVIDER_RATE_LIMIT.value, exc.__class__.__name__, True
        if "timeout" in text:
            return FailureCode.PROVIDER_TIMEOUT.value, exc.__class__.__name__, True
        if "5xx" in text or "503" in text or "502" in text or "500" in text:
            return FailureCode.PROVIDER_HTTP_5XX.value, exc.__class__.__name__, True
        return FailureCode.PROVIDER_UNAVAILABLE.value, exc.__class__.__name__, True
    if isinstance(exc, PermanentProviderError):
        if "not found" in text or "invalid symbol" in text or "unsupported" in text:
            return FailureCode.SYMBOL_NOT_FOUND.value, exc.__class__.__name__, False
        return FailureCode.INVALID_SYMBOL.value, exc.__class__.__name__, False
    return FailureCode.PROVIDER_UNAVAILABLE.value, exc.__class__.__name__, False


def backoff_sleep(policy: RetryPolicy, retry_idx: int) -> None:
    delay = min(policy.cap_delay_sec, policy.base_delay_sec * (policy.alpha ** retry_idx))
    jitter = random.uniform(policy.jitter_low, policy.jitter_high)
    time.sleep(delay * jitter)


def fetch_symbol_with_fallback(
    clients: Sequence[ProviderClient],
    window: UniverseWindow,
    frequency: str,
    retry_policy: RetryPolicy,
    rate_limiter: RateLimiter,
) -> FetchAttemptResult:
    last_failure: Optional[FetchAttemptResult] = None
    for provider_idx, client in enumerate(clients):
        retry_count = 0
        while True:
            try:
                rate_limiter.acquire()
                frame = client.fetch_daily_bars(window.symbol, window.start_date, window.end_date, frequency)
                return FetchAttemptResult(
                    frame=frame,
                    provider_alias=client.provider_alias,
                    provider_name=client.provider_name,
                    dataset_id=getattr(client, "dataset_id", None),
                    retry_count=retry_count,
                    error_code=None,
                    error_class=None,
                    message=None,
                    fallback_used=provider_idx > 0,
                    http_like_error=False,
                )
            except Exception as exc:  # noqa: BLE001
                error_code, error_class, http_like = classify_provider_exception(exc)
                retriable = isinstance(exc, RetriableProviderError)
                if retriable and retry_count < retry_policy.max_retries:
                    backoff_sleep(retry_policy, retry_count)
                    retry_count += 1
                    continue
                last_failure = FetchAttemptResult(
                    frame=None,
                    provider_alias=client.provider_alias,
                    provider_name=client.provider_name,
                    dataset_id=getattr(client, "dataset_id", None),
                    retry_count=retry_count,
                    error_code=error_code,
                    error_class=error_class,
                    message=str(exc),
                    fallback_used=provider_idx > 0,
                    http_like_error=http_like,
                )
                break
    if last_failure is None:
        raise FetchPricesError("Internal error: fetch_symbol_with_fallback finished without result or failure.")
    return last_failure


# -----------------------------------------------------------------------------
# Window logic by fetch mode
# -----------------------------------------------------------------------------


def get_fetch_window(
    window: UniverseWindow,
    mode: str,
    existing_canonical: pd.DataFrame,
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp, List[pd.Timestamp], Optional[str]]]:
    mode_enum = FetchMode(mode)
    expected = [pd.Timestamp(x).normalize() for x in window.expected_sessions]

    if existing_canonical.empty:
        return window.start_date, window.end_date, expected, None

    existing_symbol = existing_canonical.loc[existing_canonical["symbol"] == window.symbol].copy()
    existing_dates = set(pd.to_datetime(existing_symbol["trade_date"], errors="coerce").dt.normalize().dropna().tolist())

    if mode_enum == FetchMode.FULL_REFRESH:
        return window.start_date, window.end_date, expected, None

    if mode_enum == FetchMode.INCREMENTAL:
        if not existing_dates:
            return window.start_date, window.end_date, expected, None
        max_existing = max(existing_dates)
        new_start = max(window.start_date, pd.Timestamp(max_existing) + pd.Timedelta(days=1))
        if new_start > window.end_date:
            return None
        expected_filtered = [d for d in expected if d >= new_start]
        if not expected_filtered:
            return None
        return pd.Timestamp(expected_filtered[0]).normalize(), pd.Timestamp(expected_filtered[-1]).normalize(), expected_filtered, None

    if mode_enum == FetchMode.RECONCILE:
        return window.start_date, window.end_date, expected, None

    if mode_enum == FetchMode.BACKFILL:
        missing = [d for d in expected if d not in existing_dates]
        if not missing:
            return None
        return pd.Timestamp(missing[0]).normalize(), pd.Timestamp(missing[-1]).normalize(), missing, None

    raise FetchPricesError(f"Unsupported fetch mode: {mode}")


# -----------------------------------------------------------------------------
# Normalization and validation
# -----------------------------------------------------------------------------


def map_trade_dates(frame: pd.DataFrame, cfg: FetchPricesConfig, provider_timezone: Optional[str]) -> Tuple[pd.Series, pd.Series]:
    market_tz = provider_timezone or cfg.normalization.market_timezone
    ts_col = None
    for candidate in ["trade_date", "date", "session_date"]:
        if candidate in frame.columns:
            trade_date = pd.to_datetime(frame[candidate], errors="coerce").dt.normalize()
            provider_ts = pd.Series([pd.NaT] * len(frame), index=frame.index, dtype="datetime64[ns, UTC]")
            return trade_date, provider_ts
    for candidate in ["timestamp", "datetime", "provider_timestamp", "time", "t"]:
        if candidate in frame.columns:
            ts_col = candidate
            break
    if ts_col is None:
        raise FetchPricesError("Provider payload must contain trade_date/date/session_date or a timestamp-like column.")

    provider_ts = pd.to_datetime(frame[ts_col], errors="coerce", utc=True)
    if cfg.normalization.timestamp_to_trade_date == "utc_date":
        trade_date = provider_ts.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    else:
        trade_date = provider_ts.dt.tz_convert(market_tz).dt.tz_localize(None).dt.normalize()
    return trade_date, provider_ts


def normalize_provider_frame(
    raw_frame: pd.DataFrame,
    symbol: str,
    attempt: FetchAttemptResult,
    cfg: FetchPricesConfig,
    config_hash: str,
    universe_snapshot_id: str,
    fetch_mode: str,
    ingest_ts_utc: str,
    provider_timezone: Optional[str],
) -> pd.DataFrame:
    frame = raw_frame.copy()
    if frame.empty:
        return pd.DataFrame(columns=REQUIRED_CANONICAL_COLUMNS)

    lower_map = {str(c).lower(): c for c in frame.columns}
    rename_map: Dict[str, str] = {}
    for source_name, target_name in [
        ("o", "open"),
        ("h", "high"),
        ("l", "low"),
        ("c", "close"),
        ("v", "volume"),
        ("vw", "vwap"),
        ("n", "trades_count"),
        ("ticker", "symbol"),
    ]:
        if source_name in lower_map and target_name not in frame.columns:
            rename_map[lower_map[source_name]] = target_name
    if rename_map:
        frame = frame.rename(columns=rename_map)

    if "symbol" not in frame.columns:
        frame["symbol"] = symbol
    frame["symbol"] = frame["symbol"].map(lambda x: normalize_symbol(x, cfg.normalization.trim_symbols, cfg.normalization.uppercase_symbols))
    frame = frame.loc[frame["symbol"] == symbol].copy()

    trade_date, provider_ts = map_trade_dates(frame, cfg, provider_timezone)
    frame["trade_date"] = trade_date
    frame["provider_timestamp_utc"] = provider_ts

    for col in ["open", "high", "low", "close", "volume", "vwap", "trades_count"]:
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if "currency" not in frame.columns:
        frame["currency"] = None
    if "provider_dataset_id" not in frame.columns:
        frame["provider_dataset_id"] = attempt.dataset_id
    if "provider_row_id" not in frame.columns:
        frame["provider_row_id"] = pd.Series(np.arange(len(frame)), index=frame.index).astype(str)

    frame["source_provider"] = attempt.provider_alias
    frame["ingest_ts_utc"] = ingest_ts_utc
    frame["run_id"] = None  # will be filled by caller
    frame["config_hash"] = config_hash
    frame["universe_snapshot_id"] = universe_snapshot_id
    frame["fetch_mode"] = fetch_mode
    frame["fallback_used"] = bool(attempt.fallback_used)

    out = frame[[c for c in REQUIRED_CANONICAL_COLUMNS if c in frame.columns] + [c for c in frame.columns if c not in REQUIRED_CANONICAL_COLUMNS]].copy()
    for required in REQUIRED_CANONICAL_COLUMNS:
        if required not in out.columns:
            out[required] = np.nan
    out = out[REQUIRED_CANONICAL_COLUMNS]
    return out.reset_index(drop=True)


@dataclass
class ValidationOutput:
    clean: pd.DataFrame
    failures: List[Dict[str, Any]]


def build_failure(
    symbol: str,
    provider: str,
    error_code: str,
    error_class: str,
    message: str,
    retry_count: int,
    final_status: str,
    severity: str,
    trade_date: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "trade_date": pd.Timestamp(trade_date).normalize() if trade_date is not None else pd.NaT,
        "error_code": error_code,
        "error_class": error_class,
        "message": message,
        "retry_count": int(retry_count),
        "provider": provider,
        "final_status": final_status,
        "severity": severity,
    }


def validate_ingestion_frame(
    frame: pd.DataFrame,
    window: UniverseWindow,
    run_id: str,
    cfg: FetchPricesConfig,
) -> ValidationOutput:
    failures: List[Dict[str, Any]] = []
    if frame.empty:
        return ValidationOutput(clean=frame.copy(), failures=failures)

    work = frame.copy()
    work["run_id"] = run_id

    missing_required = [c for c in ["symbol", "trade_date", "open", "high", "low", "close", "volume", "source_provider"] if c not in work.columns]
    if missing_required:
        raise FetchPricesError(f"Normalized frame missing required columns: {missing_required}")

    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.normalize()
    work = work.sort_values(["symbol", "trade_date", "provider_timestamp_utc", "provider_row_id"], kind="stable").reset_index(drop=True)

    valid_mask = pd.Series(True, index=work.index)
    for idx, row in work.iterrows():
        if pd.isna(row["trade_date"]):
            failures.append(build_failure(row["symbol"], row["source_provider"], FailureCode.INVALID_TRADE_DATE.value, "RowValidation", "trade_date could not be parsed", 0, "dropped_row", Severity.FAIL.value))
            valid_mask.loc[idx] = False
            continue
        if row["trade_date"] < window.start_date or row["trade_date"] > window.end_date:
            failures.append(build_failure(row["symbol"], row["source_provider"], FailureCode.INVALID_TRADE_DATE.value, "RowValidation", "trade_date outside requested range", 0, "dropped_row", Severity.WARN.value, row["trade_date"]))
            valid_mask.loc[idx] = False
            continue
        prices = [row["open"], row["high"], row["low"], row["close"]]
        if any(pd.isna(x) for x in prices):
            failures.append(build_failure(row["symbol"], row["source_provider"], FailureCode.MISSING_REQUIRED_COLUMN.value, "RowValidation", "OHLC contains null values", 0, "dropped_row", Severity.FAIL.value, row["trade_date"]))
            valid_mask.loc[idx] = False
            continue
        if cfg.validation.require_positive_prices and any(float(x) <= 0 for x in prices):
            failures.append(build_failure(row["symbol"], row["source_provider"], FailureCode.INVALID_PRICE_VALUE.value, "RowValidation", "OHLC must be positive", 0, "dropped_row", Severity.FAIL.value, row["trade_date"]))
            valid_mask.loc[idx] = False
            continue
        high = float(row["high"])
        low = float(row["low"])
        open_ = float(row["open"])
        close = float(row["close"])
        if high < max(open_, close, low) or low > min(open_, close, high):
            failures.append(build_failure(row["symbol"], row["source_provider"], FailureCode.INVALID_PRICE_GEOMETRY.value, "RowValidation", "OHLC geometry is inconsistent", 0, "dropped_row", Severity.FAIL.value, row["trade_date"]))
            valid_mask.loc[idx] = False
            continue
        volume = row["volume"]
        if pd.isna(volume) or float(volume) < 0:
            failures.append(build_failure(row["symbol"], row["source_provider"], FailureCode.INVALID_VOLUME_VALUE.value, "RowValidation", "volume must be >= 0", 0, "dropped_row", Severity.FAIL.value, row["trade_date"]))
            valid_mask.loc[idx] = False
            continue

    work = work.loc[valid_mask].copy()
    if work.empty:
        return ValidationOutput(clean=work, failures=failures)

    dedupe_keep = "last" if cfg.validation.duplicate_resolution == "last" else "first"
    dup_mask = work.duplicated(subset=["symbol", "trade_date", "source_provider"], keep=dedupe_keep)
    if dup_mask.any():
        dropped = work.loc[dup_mask].copy()
        for _, row in dropped.iterrows():
            failures.append(build_failure(row["symbol"], row["source_provider"], FailureCode.DUPLICATE_PROVIDER_ROW.value, "DuplicateResolution", "duplicate provider row resolved deterministically", 0, "dropped_duplicate", Severity.WARN.value, row["trade_date"]))
        work = work.loc[~dup_mask].copy()

    work = work.sort_values(["symbol", "trade_date"], kind="stable").reset_index(drop=True)
    return ValidationOutput(clean=work, failures=failures)


# -----------------------------------------------------------------------------
# Coverage, revisions and canonical merge
# -----------------------------------------------------------------------------


def build_coverage_report(
    windows: Sequence[UniverseWindow],
    accepted: pd.DataFrame,
    failures: pd.DataFrame,
    fallback_symbols: Sequence[str],
    cfg: FetchPricesConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    accepted = accepted.copy()
    if not accepted.empty:
        accepted["trade_date"] = pd.to_datetime(accepted["trade_date"], errors="coerce").dt.normalize()

    fallback_set = set(fallback_symbols)
    failures_by_symbol = failures.groupby("symbol") if not failures.empty else None

    for window in windows:
        symbol_df = accepted.loc[accepted["symbol"] == window.symbol].copy() if not accepted.empty else pd.DataFrame()
        observed = sorted(pd.unique(symbol_df["trade_date"])) if not symbol_df.empty else []
        expected = [pd.Timestamp(x).normalize() for x in window.expected_sessions]
        expected_set = set(expected)
        observed_set = set(pd.Timestamp(x).normalize() for x in observed)
        missing = sorted(expected_set - observed_set)
        if symbol_df.empty:
            first_date = pd.NaT
            last_date = pd.NaT
            provider_used = None
        else:
            first_date = min(observed)
            last_date = max(observed)
            provider_used = ",".join(sorted(pd.unique(symbol_df["source_provider"].astype(str))))
        pct_missing = float(len(missing) / len(expected)) if expected else 0.0
        gap_count = len(missing)
        severities = [Severity.INFO.value]
        if pct_missing > 0:
            if pct_missing >= cfg.validation.missing_fail_threshold:
                severities.append(Severity.FAIL.value)
            elif pct_missing >= cfg.validation.missing_warn_threshold:
                severities.append(Severity.WARN.value)
            else:
                severities.append(Severity.INFO.value)
        if window.symbol in fallback_set:
            severities.append(Severity.WARN.value)
        if failures_by_symbol is not None and window.symbol in failures_by_symbol.groups:
            severities.extend(failures_by_symbol.get_group(window.symbol)["severity"].astype(str).tolist())
        rows.append(
            {
                "symbol": window.symbol,
                "expected_sessions": int(len(expected)),
                "observed_sessions": int(len(observed_set)),
                "pct_missing_sessions": pct_missing,
                "first_date": pd.Timestamp(first_date).normalize() if pd.notna(first_date) else pd.NaT,
                "last_date": pd.Timestamp(last_date).normalize() if pd.notna(last_date) else pd.NaT,
                "provider_used": provider_used,
                "gap_count": int(gap_count),
                "missing_dates": [pd.Timestamp(x).date().isoformat() for x in missing],
                "severity_max": highest_severity(severities),
                "fallback_used": window.symbol in fallback_set,
            }
        )
    return pd.DataFrame(rows)


def load_existing_canonical(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=REQUIRED_CANONICAL_COLUMNS)
    return pd.read_parquet(path)


def build_revisions_report(existing: pd.DataFrame, new: pd.DataFrame, run_id: str) -> pd.DataFrame:
    if existing.empty or new.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", "field_name", "old_value", "new_value", "old_run_id", "new_run_id", "provider", "update_applied"])
    left = existing[["symbol", "trade_date", "source_provider", "run_id"] + COMPARISON_FIELDS].copy()
    right = new[["symbol", "trade_date", "source_provider", "run_id"] + COMPARISON_FIELDS].copy()
    merged = left.merge(right, on=["symbol", "trade_date", "source_provider"], how="inner", suffixes=("_old", "_new"))
    rows: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        for field in COMPARISON_FIELDS:
            old = row[f"{field}_old"]
            newv = row[f"{field}_new"]
            if (pd.isna(old) and pd.isna(newv)) or old == newv:
                continue
            rows.append(
                {
                    "symbol": row["symbol"],
                    "trade_date": pd.Timestamp(row["trade_date"]).normalize(),
                    "field_name": field,
                    "old_value": old,
                    "new_value": newv,
                    "old_run_id": row["run_id_old"],
                    "new_run_id": run_id,
                    "provider": row["source_provider"],
                    "update_applied": False,
                }
            )
    return pd.DataFrame(rows)


def merge_with_existing_snapshot(existing: pd.DataFrame, new: pd.DataFrame, mode: str, cfg: FetchPricesConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if new.empty and existing.empty:
        empty = pd.DataFrame(columns=["symbol", "trade_date", "field_name", "old_value", "new_value", "old_run_id", "new_run_id", "provider", "update_applied"])
        return pd.DataFrame(columns=REQUIRED_CANONICAL_COLUMNS), empty
    if existing.empty:
        revisions = build_revisions_report(existing, new, run_id=str(new["run_id"].iloc[0]) if not new.empty else "")
        return new.sort_values(["symbol", "trade_date", "source_provider"]).reset_index(drop=True), revisions

    revisions = build_revisions_report(existing, new, run_id=str(new["run_id"].iloc[0]) if not new.empty else "")
    overwrite = False
    if mode == FetchMode.FULL_REFRESH.value:
        overwrite = cfg.persistence.full_refresh_overwrite_canonical
    elif mode == FetchMode.RECONCILE.value:
        overwrite = cfg.persistence.reconcile_overwrite_canonical
    elif mode == FetchMode.BACKFILL.value:
        overwrite = cfg.persistence.backfill_overwrite_canonical
    elif mode == FetchMode.INCREMENTAL.value:
        overwrite = cfg.persistence.incremental_update_existing_range

    key_cols = ["symbol", "trade_date", "source_provider"]
    new_keys = new[key_cols].drop_duplicates() if not new.empty else pd.DataFrame(columns=key_cols)
    if not overwrite and not new_keys.empty:
        merged_keys = existing[key_cols].merge(new_keys, on=key_cols, how="left", indicator=True)
        existing_only = existing.copy()
        new_only = new.merge(existing[key_cols], on=key_cols, how="left", indicator=True)
        new_only = new_only.loc[new_only["_merge"] == "left_only"].drop(columns=["_merge"])
        canonical = pd.concat([existing_only, new_only], ignore_index=True, sort=False)
    else:
        existing_keep = existing.merge(new_keys, on=key_cols, how="left", indicator=True)
        existing_keep = existing_keep.loc[existing_keep["_merge"] == "left_only"].drop(columns=["_merge"])
        canonical = pd.concat([existing_keep, new], ignore_index=True, sort=False)
        if not revisions.empty:
            revisions["update_applied"] = True

    canonical = canonical.sort_values(["symbol", "trade_date", "source_provider", "ingest_ts_utc"], kind="stable")
    canonical = canonical.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    for required in REQUIRED_CANONICAL_COLUMNS:
        if required not in canonical.columns:
            canonical[required] = np.nan
    return canonical[REQUIRED_CANONICAL_COLUMNS].copy(), revisions


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------


def require_parquet_engine() -> None:
    try:
        import pyarrow  # noqa: F401

        return
    except Exception:
        pass
    try:
        import fastparquet  # noqa: F401

        return
    except Exception:
        pass
    raise FetchPricesError("Writing parquet requires 'pyarrow' or 'fastparquet' in the runtime environment.")


def write_parquet_deduped(path: Path, frame: pd.DataFrame, dedupe_keys: Optional[Sequence[str]] = None, overwrite: bool = False, compression: str = "snappy") -> None:
    require_parquet_engine()
    ensure_parent(path)
    if path.exists() and not overwrite:
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, frame], ignore_index=True, sort=False)
    else:
        combined = frame.copy()
    if dedupe_keys:
        combined = combined.sort_values(list(dedupe_keys), kind="stable").drop_duplicates(subset=list(dedupe_keys), keep="last")
    combined.to_parquet(path, index=False, compression=compression)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def run_fetch_prices(
    universe_snapshot_path: Path,
    calendar_path: Path,
    provider_config_path: Path,
    start_date: str,
    end_date: str,
    frequency: str,
    fetch_mode: str,
    run_id: str,
    asof_ts_utc: str,
    config_path: Optional[Path] = None,
) -> FetchArtifacts:
    t0 = time.perf_counter()
    cfg, providers, raw_config, config_hash = load_config(config_path, provider_config_path)
    start_ts = normalize_date(start_date)
    end_ts = normalize_date(end_date)
    if end_ts < start_ts:
        raise FetchPricesError("end_date must be >= start_date")
    if frequency != cfg.fetch.frequency:
        logging.getLogger(__name__).info("CLI frequency=%s differs from config frequency=%s; CLI value wins.", frequency, cfg.fetch.frequency)
    mode = FetchMode(fetch_mode).value

    universe_df, universe_snapshot_id = load_universe_snapshot(universe_snapshot_path, start_ts, end_ts, cfg)
    sessions = load_market_calendar(calendar_path, start_ts, end_ts)
    windows = build_universe_windows(universe_df, sessions, start_ts, end_ts)
    if not windows:
        raise FetchPricesError("No symbols/windows to fetch after applying universe snapshot and calendar.")

    provider_aliases = [alias for alias in cfg.fetch.provider_priority if alias in providers and providers[alias].enabled]
    if not provider_aliases:
        provider_aliases = [alias for alias, spec in providers.items() if spec.enabled]
    clients = [build_provider_client(providers[alias]) for alias in provider_aliases]
    if not clients:
        raise FetchPricesError("No provider clients could be built.")

    output_root = Path(cfg.persistence.output_root)
    canonical_path = output_root / cfg.persistence.canonical_relpath
    existing_canonical = load_existing_canonical(canonical_path) if cfg.persistence.persist_canonical else pd.DataFrame(columns=REQUIRED_CANONICAL_COLUMNS)
    rate_limiter = RateLimiter(cfg.rate_limit.min_interval_sec)

    ingest_ts_utc = parse_timestamp_utc(asof_ts_utc)
    if ingest_ts_utc is None:
        raise FetchPricesError("asof_ts_utc could not be parsed.")
    ingest_ts_utc_str = ingest_ts_utc.isoformat()

    raw_rows: List[pd.DataFrame] = []
    failures_rows: List[Dict[str, Any]] = []
    fallback_symbols: List[str] = []
    request_count = 0
    http_like_failures = 0
    tickers_ok = 0
    tickers_failed = 0

    for symbol_chunk in chunked([w.symbol for w in windows], cfg.fetch.chunk_size_tickers):
        chunk_lookup = {w.symbol: w for w in windows if w.symbol in set(symbol_chunk)}
        for symbol in symbol_chunk:
            window = chunk_lookup[symbol]
            actual_window = get_fetch_window(window, mode, existing_canonical)
            if actual_window is None:
                failures_rows.append(
                    build_failure(symbol, provider_aliases[0], FailureCode.NO_NEW_DATES.value if mode == FetchMode.INCREMENTAL.value else FailureCode.BACKFILL_NO_GAPS.value, "FetchWindow", "no fetch required under current mode", 0, "skipped", Severity.INFO.value)
                )
                continue
            fetch_start, fetch_end, expected_sessions, _ = actual_window
            request_count += 1
            temp_window = UniverseWindow(symbol=symbol, start_date=fetch_start, end_date=fetch_end, expected_sessions=expected_sessions)
            attempt = fetch_symbol_with_fallback(clients, temp_window, frequency, cfg.retry, rate_limiter)
            if attempt.http_like_error:
                http_like_failures += 1
            if attempt.frame is None:
                tickers_failed += 1
                failures_rows.append(
                    build_failure(symbol, attempt.provider_alias, attempt.error_code or FailureCode.PROVIDER_UNAVAILABLE.value, attempt.error_class or "ProviderError", attempt.message or "provider failure", attempt.retry_count, "failed_symbol", Severity.FAIL.value)
                )
                continue
            if attempt.fallback_used:
                fallback_symbols.append(symbol)
                failures_rows.append(build_failure(symbol, attempt.provider_alias, FailureCode.FALLBACK_USED.value, "ProviderFallback", f"fallback provider '{attempt.provider_alias}' was used", attempt.retry_count, "provider_fallback", Severity.WARN.value))
            if attempt.frame.empty:
                tickers_ok += 1
                failures_rows.append(build_failure(symbol, attempt.provider_alias, FailureCode.NO_ACTIVITY_OBSERVED.value, "ProviderFetch", "provider returned empty payload for symbol", attempt.retry_count, "empty_payload", Severity.INFO.value))
                continue
            normalized = normalize_provider_frame(
                attempt.frame,
                symbol=symbol,
                attempt=attempt,
                cfg=cfg,
                config_hash=config_hash,
                universe_snapshot_id=universe_snapshot_id,
                fetch_mode=mode,
                ingest_ts_utc=ingest_ts_utc_str,
                provider_timezone=providers[attempt.provider_alias].timezone,
            )
            normalized["run_id"] = run_id
            validation = validate_ingestion_frame(normalized, temp_window, run_id, cfg)
            failures_rows.extend(validation.failures)
            if validation.clean.empty:
                tickers_failed += 1
                failures_rows.append(build_failure(symbol, attempt.provider_alias, FailureCode.EMPTY_PAYLOAD.value, "Validation", "all fetched rows were dropped by validation", attempt.retry_count, "failed_symbol", Severity.FAIL.value))
                continue
            raw_rows.append(validation.clean)
            tickers_ok += 1

    raw_df = pd.concat(raw_rows, ignore_index=True, sort=False) if raw_rows else pd.DataFrame(columns=REQUIRED_CANONICAL_COLUMNS)
    failures_df = pd.DataFrame(failures_rows)
    if not failures_df.empty:
        failures_df["trade_date"] = pd.to_datetime(failures_df["trade_date"], errors="coerce").dt.normalize()
        failures_df = failures_df.sort_values(["symbol", "trade_date", "severity", "error_code"], kind="stable").reset_index(drop=True)

    coverage_df = build_coverage_report(windows, raw_df, failures_df, fallback_symbols, cfg)
    canonical_df, revisions_df = merge_with_existing_snapshot(existing_canonical, raw_df, mode, cfg)
    if not revisions_df.empty:
        revision_notice = revisions_df.copy()
        revision_notice["severity"] = Severity.WARN.value
        revision_notice["error_code"] = FailureCode.REVISIONS_DETECTED.value
        revision_notice["error_class"] = "RevisionDetection"
        revision_notice["message"] = revision_notice.apply(lambda r: f"historical revision detected for field={r['field_name']}", axis=1)
        revision_notice["retry_count"] = 0
        revision_notice["provider"] = revision_notice["provider"]
        revision_notice["final_status"] = revision_notice["update_applied"].map(lambda x: "canonical_updated" if bool(x) else "reported_only")
        failures_df = pd.concat([
            failures_df,
            revision_notice[["symbol", "trade_date", "error_code", "error_class", "message", "retry_count", "provider", "final_status", "severity"]],
        ], ignore_index=True, sort=False)

    summary = {
        "run_id": run_id,
        "fetch_mode": mode,
        "frequency": frequency,
        "start_date": start_ts.date().isoformat(),
        "end_date": end_ts.date().isoformat(),
        "symbols_requested": len(windows),
        "tickers_ok": tickers_ok,
        "tickers_failed": tickers_failed,
        "rows_ingested": int(len(raw_df)),
        "rows_canonical": int(len(canonical_df)),
        "coverage_warn_symbols": int((coverage_df["severity_max"] == Severity.WARN.value).sum()) if not coverage_df.empty else 0,
        "coverage_fail_symbols": int((coverage_df["severity_max"] == Severity.FAIL.value).sum()) if not coverage_df.empty else 0,
        "revisions_detected": int(len(revisions_df)),
        "fallback_usage_rate": float(len(set(fallback_symbols)) / len(windows)) if windows else 0.0,
        "provider_http_error_rate": float(http_like_failures / request_count) if request_count else 0.0,
    }

    duration_sec = time.perf_counter() - t0
    manifest = {
        "run_id": run_id,
        "provider_priority": provider_aliases,
        "fetch_mode": mode,
        "start_date": start_ts.date().isoformat(),
        "end_date": end_ts.date().isoformat(),
        "frequency": frequency,
        "config_hash": config_hash,
        "universe_snapshot_id": universe_snapshot_id,
        "universe_snapshot_path": str(universe_snapshot_path),
        "calendar_path": str(calendar_path),
        "provider_config_path": str(provider_config_path),
        "config_path": str(config_path) if config_path is not None else None,
        "code_version": git_code_version(),
        "tickers_requested": len(windows),
        "tickers_ok": tickers_ok,
        "tickers_failed": tickers_failed,
        "rows_ingested": int(len(raw_df)),
        "provider_http_error_rate": summary["provider_http_error_rate"],
        "fallback_usage_rate": summary["fallback_usage_rate"],
        "revisions_detected": int(len(revisions_df)),
        "run_duration_sec": duration_sec,
        "asof_ts_utc": ingest_ts_utc_str,
        "created_ts_utc": utc_now(),
        "python_version": platform.python_version(),
        "host_platform": platform.platform(),
        "raw_config_digest": stable_hash_payload(raw_config, prefix="provider_cfg"),
        "output_root": str(output_root),
    }

    if cfg.persistence.persist_raw and not raw_df.empty:
        raw_path = output_root / f"raw/provider={provider_aliases[0]}/freq={frequency}/run_id={run_id}/prices.parquet"
        write_parquet_deduped(
            raw_path,
            raw_df,
            dedupe_keys=["symbol", "trade_date", "source_provider", "run_id"] if cfg.persistence.raw_append_dedupe else None,
            overwrite=cfg.persistence.overwrite_existing_run,
            compression=cfg.persistence.compression,
        )
    elif cfg.persistence.persist_raw:
        raw_path = output_root / f"raw/provider={provider_aliases[0]}/freq={frequency}/run_id={run_id}/prices.parquet"

    coverage_path = output_root / f"reports/coverage_{run_id}.parquet"
    failures_path = output_root / f"reports/failures_{run_id}.parquet"
    revisions_path = output_root / f"reports/revisions_{run_id}.parquet"
    summary_path = output_root / f"reports/summary_{run_id}.json"
    manifest_path = output_root / f"reports/manifest_{run_id}.json"

    if not coverage_df.empty:
        write_parquet_deduped(coverage_path, coverage_df, dedupe_keys=["symbol"], overwrite=True, compression=cfg.persistence.compression)
    else:
        require_parquet_engine()
        ensure_parent(coverage_path)
        coverage_df.to_parquet(coverage_path, index=False, compression=cfg.persistence.compression)
    if not failures_df.empty:
        write_parquet_deduped(failures_path, failures_df, dedupe_keys=["symbol", "trade_date", "error_code", "provider", "message"], overwrite=True, compression=cfg.persistence.compression)
    else:
        require_parquet_engine()
        ensure_parent(failures_path)
        failures_df.to_parquet(failures_path, index=False, compression=cfg.persistence.compression)
    if not revisions_df.empty:
        write_parquet_deduped(revisions_path, revisions_df, dedupe_keys=["symbol", "trade_date", "field_name", "provider", "new_run_id"], overwrite=True, compression=cfg.persistence.compression)
    else:
        require_parquet_engine()
        ensure_parent(revisions_path)
        revisions_df.to_parquet(revisions_path, index=False, compression=cfg.persistence.compression)

    write_json(summary_path, summary)
    write_json(manifest_path, manifest)

    if cfg.persistence.persist_canonical:
        write_parquet_deduped(canonical_path, canonical_df, dedupe_keys=["symbol", "trade_date", "source_provider"], overwrite=True, compression=cfg.persistence.compression)

    return FetchArtifacts(
        raw=raw_df,
        canonical=canonical_df,
        coverage=coverage_df,
        failures=failures_df,
        revisions=revisions_df,
        summary=summary,
        manifest=manifest,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Institutional raw OHLCV ingestion for PIT universe snapshots.")
    parser.add_argument("--universe-snapshot-path", required=True, help="Path to PIT universe snapshot (csv/parquet/json).")
    parser.add_argument("--calendar-path", required=True, help="Path to official market calendar (csv/parquet/json).")
    parser.add_argument("--provider-config-path", required=True, help="Path to provider configuration (json/yaml).")
    parser.add_argument("--config-path", default=None, help="Optional general fetch config (json/yaml) merged before provider config.")
    parser.add_argument("--start-date", required=True, help="Inclusive start trade_date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="Inclusive end trade_date (YYYY-MM-DD).")
    parser.add_argument("--frequency", default="1d", help="Frequency code; MVP expects 1d.")
    parser.add_argument("--fetch-mode", default=FetchMode.INCREMENTAL.value, choices=[e.value for e in FetchMode], help="full_refresh, incremental, reconcile, backfill")
    parser.add_argument("--run-id", required=True, help="Stable run identifier used for raw versioning and reports.")
    parser.add_argument("--as-of-ts-utc", required=True, help="UTC ingestion timestamp for versioning and auditability.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")

    artifacts = run_fetch_prices(
        universe_snapshot_path=Path(args.universe_snapshot_path),
        calendar_path=Path(args.calendar_path),
        provider_config_path=Path(args.provider_config_path),
        start_date=args.start_date,
        end_date=args.end_date,
        frequency=args.frequency,
        fetch_mode=args.fetch_mode,
        run_id=args.run_id,
        asof_ts_utc=args.as_of_ts_utc,
        config_path=Path(args.config_path) if args.config_path else None,
    )
    print(json.dumps(json_safe(artifacts.summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
