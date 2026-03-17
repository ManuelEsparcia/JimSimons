from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import logging
import math
import platform
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# -----------------------------------------------------------------------------
# Constants, enums and exceptions
# -----------------------------------------------------------------------------


class CorporateActionsError(RuntimeError):
    """Raised when a hard validation, canonicalization or IO invariant fails."""


class Severity(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class EventStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


class FailureCode(str, Enum):
    MISSING_PROVIDER = "MISSING_PROVIDER"
    UNSUPPORTED_EVENT_TYPE = "UNSUPPORTED_EVENT_TYPE"
    INVALID_TEMPORAL_FIELDS = "INVALID_TEMPORAL_FIELDS"
    UNRESOLVED_IDENTITY = "UNRESOLVED_IDENTITY"
    AMBIGUOUS_IDENTITY = "AMBIGUOUS_IDENTITY"
    LOW_CONFIDENCE_IDENTITY = "LOW_CONFIDENCE_IDENTITY"
    INVALID_SPLIT_RATIO = "INVALID_SPLIT_RATIO"
    INVALID_CASH_AMOUNT = "INVALID_CASH_AMOUNT"
    MATERIAL_CONFLICT = "MATERIAL_CONFLICT"
    MISSING_CRITICAL_FIELDS = "MISSING_CRITICAL_FIELDS"
    RAW_DUPLICATE_COLLISION = "RAW_DUPLICATE_COLLISION"


class CanonicalEventType(str, Enum):
    SPLIT = "split"
    REVERSE_SPLIT = "reverse_split"
    STOCK_DIVIDEND = "stock_dividend"
    CASH_DIVIDEND = "cash_dividend"
    SPECIAL_CASH_DIVIDEND = "special_cash_dividend"
    TICKER_CHANGE = "ticker_change"
    NAME_CHANGE = "name_change"
    SHARE_CLASS_CHANGE = "share_class_change"
    EXCHANGE_CHANGE = "exchange_change"
    MERGER = "merger"
    ACQUISITION = "acquisition"
    SPINOFF = "spinoff"
    RIGHTS_ISSUE = "rights_issue"
    TENDER_OFFER = "tender_offer"
    DELISTING = "delisting"
    RELISTING = "relisting"
    BANKRUPTCY_REORG = "bankruptcy_reorg"
    IDENTIFIER_MAINTENANCE = "identifier_maintenance"


class EventFamily(str, Enum):
    MECHANICAL_ADJUSTMENT = "mechanical_adjustment"
    IDENTITY = "identity"
    ECONOMIC_TRANSFORMATION = "economic_transformation"
    TERMINAL = "terminal_or_quasi_terminal"


EVENT_TYPE_TO_FAMILY: Dict[str, str] = {
    CanonicalEventType.SPLIT.value: EventFamily.MECHANICAL_ADJUSTMENT.value,
    CanonicalEventType.REVERSE_SPLIT.value: EventFamily.MECHANICAL_ADJUSTMENT.value,
    CanonicalEventType.STOCK_DIVIDEND.value: EventFamily.MECHANICAL_ADJUSTMENT.value,
    CanonicalEventType.CASH_DIVIDEND.value: EventFamily.MECHANICAL_ADJUSTMENT.value,
    CanonicalEventType.SPECIAL_CASH_DIVIDEND.value: EventFamily.MECHANICAL_ADJUSTMENT.value,
    CanonicalEventType.TICKER_CHANGE.value: EventFamily.IDENTITY.value,
    CanonicalEventType.NAME_CHANGE.value: EventFamily.IDENTITY.value,
    CanonicalEventType.SHARE_CLASS_CHANGE.value: EventFamily.IDENTITY.value,
    CanonicalEventType.EXCHANGE_CHANGE.value: EventFamily.IDENTITY.value,
    CanonicalEventType.IDENTIFIER_MAINTENANCE.value: EventFamily.IDENTITY.value,
    CanonicalEventType.MERGER.value: EventFamily.ECONOMIC_TRANSFORMATION.value,
    CanonicalEventType.ACQUISITION.value: EventFamily.ECONOMIC_TRANSFORMATION.value,
    CanonicalEventType.SPINOFF.value: EventFamily.ECONOMIC_TRANSFORMATION.value,
    CanonicalEventType.RIGHTS_ISSUE.value: EventFamily.ECONOMIC_TRANSFORMATION.value,
    CanonicalEventType.TENDER_OFFER.value: EventFamily.ECONOMIC_TRANSFORMATION.value,
    CanonicalEventType.BANKRUPTCY_REORG.value: EventFamily.TERMINAL.value,
    CanonicalEventType.DELISTING.value: EventFamily.TERMINAL.value,
    CanonicalEventType.RELISTING.value: EventFamily.TERMINAL.value,
}

DEFAULT_PROVIDER_PRIORITIES = {
    "official": 1,
    "exchange": 2,
    "company": 3,
    "vendor_a": 4,
    "vendor_b": 5,
    "vendor_c": 6,
}

DEFAULT_CRITICAL_FIELDS = (
    "event_type",
    "instrument_id",
    "ex_date",
    "effective_date",
    "split_ratio",
    "cash_amount",
    "survivor_instrument_id",
)

DEFAULT_SAME_DAY_PRECEDENCE = (
    CanonicalEventType.IDENTIFIER_MAINTENANCE.value,
    CanonicalEventType.NAME_CHANGE.value,
    CanonicalEventType.TICKER_CHANGE.value,
    CanonicalEventType.EXCHANGE_CHANGE.value,
    CanonicalEventType.SPLIT.value,
    CanonicalEventType.REVERSE_SPLIT.value,
    CanonicalEventType.STOCK_DIVIDEND.value,
    CanonicalEventType.CASH_DIVIDEND.value,
    CanonicalEventType.SPECIAL_CASH_DIVIDEND.value,
    CanonicalEventType.RIGHTS_ISSUE.value,
    CanonicalEventType.SPINOFF.value,
    CanonicalEventType.MERGER.value,
    CanonicalEventType.ACQUISITION.value,
    CanonicalEventType.DELISTING.value,
    CanonicalEventType.RELISTING.value,
    CanonicalEventType.BANKRUPTCY_REORG.value,
    CanonicalEventType.SHARE_CLASS_CHANGE.value,
    CanonicalEventType.TENDER_OFFER.value,
)

DEFAULT_EVENT_TYPE_ALIASES: Dict[str, str] = {
    "split": CanonicalEventType.SPLIT.value,
    "stock split": CanonicalEventType.SPLIT.value,
    "forward split": CanonicalEventType.SPLIT.value,
    "reverse split": CanonicalEventType.REVERSE_SPLIT.value,
    "reverse_stock_split": CanonicalEventType.REVERSE_SPLIT.value,
    "stock dividend": CanonicalEventType.STOCK_DIVIDEND.value,
    "bonus issue": CanonicalEventType.STOCK_DIVIDEND.value,
    "dividend": CanonicalEventType.CASH_DIVIDEND.value,
    "cash dividend": CanonicalEventType.CASH_DIVIDEND.value,
    "special dividend": CanonicalEventType.SPECIAL_CASH_DIVIDEND.value,
    "special cash dividend": CanonicalEventType.SPECIAL_CASH_DIVIDEND.value,
    "ticker change": CanonicalEventType.TICKER_CHANGE.value,
    "symbol change": CanonicalEventType.TICKER_CHANGE.value,
    "name change": CanonicalEventType.NAME_CHANGE.value,
    "share class change": CanonicalEventType.SHARE_CLASS_CHANGE.value,
    "exchange change": CanonicalEventType.EXCHANGE_CHANGE.value,
    "exchange move": CanonicalEventType.EXCHANGE_CHANGE.value,
    "merger": CanonicalEventType.MERGER.value,
    "acquisition": CanonicalEventType.ACQUISITION.value,
    "spin off": CanonicalEventType.SPINOFF.value,
    "spinoff": CanonicalEventType.SPINOFF.value,
    "rights issue": CanonicalEventType.RIGHTS_ISSUE.value,
    "rights offering": CanonicalEventType.RIGHTS_ISSUE.value,
    "tender offer": CanonicalEventType.TENDER_OFFER.value,
    "delist": CanonicalEventType.DELISTING.value,
    "delisting": CanonicalEventType.DELISTING.value,
    "relist": CanonicalEventType.RELISTING.value,
    "relisting": CanonicalEventType.RELISTING.value,
    "bankruptcy": CanonicalEventType.BANKRUPTCY_REORG.value,
    "bankruptcy reorg": CanonicalEventType.BANKRUPTCY_REORG.value,
    "identifier maintenance": CanonicalEventType.IDENTIFIER_MAINTENANCE.value,
}

REQUIRED_RAW_CORE_COLUMNS = {"provider_event_id"}
OPTIONAL_RAW_DATE_COLUMNS = ["announcement_ts", "announcement_date", "ex_date", "effective_date"]
OPTIONAL_RAW_ID_COLUMNS = [
    "instrument_id",
    "issuer_id",
    "symbol",
    "exchange",
    "share_class",
    "figi",
    "cusip",
    "cik",
    "isin",
    "sedol",
    "vendor_instrument_id",
]
PRICE_REQUIRED_COLUMNS = {"date", "instrument_id", "close"}


# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizationConfig:
    decision_cutoff_policy: str = "close(t-1)"
    timezone: str = "UTC"
    default_currency: str = "USD"
    event_type_aliases: Mapping[str, str] = field(default_factory=lambda: DEFAULT_EVENT_TYPE_ALIASES)
    special_dividend_threshold: float = 0.10
    split_ratio_tolerance: float = 1e-8
    cash_amount_tolerance: float = 1e-8
    date_tolerance_days: int = 1
    allow_compound_event_splitting: bool = True


@dataclass(frozen=True)
class MatchingConfig:
    min_confidence_score: float = 0.70
    relisting_resolution: str = "new_instrument_unless_proven_continuous"
    prefer_explicit_instrument_id: bool = True
    external_id_columns: Tuple[str, ...] = ("figi", "cusip", "isin", "sedol", "cik", "vendor_instrument_id")
    active_window_start_columns: Tuple[str, ...] = ("effective_from", "ticker_start_date", "list_date")
    active_window_end_columns: Tuple[str, ...] = ("effective_to", "ticker_end_date", "delist_date")


@dataclass(frozen=True)
class ConflictConfig:
    provider_priority: Mapping[str, int] = field(default_factory=lambda: DEFAULT_PROVIDER_PRIORITIES)
    critical_fields: Tuple[str, ...] = DEFAULT_CRITICAL_FIELDS
    same_day_precedence: Tuple[str, ...] = DEFAULT_SAME_DAY_PRECEDENCE


@dataclass(frozen=True)
class ThresholdConfig:
    unresolved_rate_warn: float = 0.02
    unresolved_rate_fail: float = 0.05
    conflict_rate_warn: float = 0.01
    conflict_rate_fail: float = 0.03
    single_provider_concentration_warn: float = 0.80
    single_provider_concentration_fail: float = 0.95
    special_dividend_rate_warn: float = 0.10
    relisting_rate_warn: float = 0.05


@dataclass(frozen=True)
class OutputConfig:
    output_dir: str = "data/universe/corporate_actions"
    compression: str = "snappy"


@dataclass(frozen=True)
class CorporateActionsConfig:
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    conflict: ConflictConfig = field(default_factory=ConflictConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass(frozen=True)
class GateResult:
    name: str
    severity: Severity
    passed: bool
    observed: Any
    threshold: Any
    message: str


@dataclass(frozen=True)
class CanonicalArtifacts:
    history: pd.DataFrame
    current: pd.DataFrame
    failures: pd.DataFrame
    summary: Dict[str, Any]
    manifest: Dict[str, Any]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.exists():
        raise CorporateActionsError(f"Path does not exist: {path}")
    return path


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def maybe_git_code_version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    out: MutableMapping[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


def ensure_parquet_engine_available() -> None:
    if importlib.util.find_spec("pyarrow") is None and importlib.util.find_spec("fastparquet") is None:
        raise CorporateActionsError(
            "Parquet support requires either 'pyarrow' or 'fastparquet'. Install one before running corporate_actions."
        )


def read_dataframe(path_like: str | Path) -> pd.DataFrame:
    path = ensure_path(path_like)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        ensure_parquet_engine_available()
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix == ".json":
        try:
            return pd.read_json(path, orient="records")
        except ValueError:
            return pd.read_json(path, lines=True)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise CorporateActionsError(f"Unsupported tabular format: {path}")


def read_many_dataframes(paths: Sequence[str | Path]) -> pd.DataFrame:
    frames = []
    for path_like in paths:
        path = ensure_path(path_like)
        if path.is_dir():
            parts = sorted(
                p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".parquet", ".csv", ".txt", ".json", ".jsonl"}
            )
            if not parts:
                raise CorporateActionsError(f"Directory has no supported tabular files: {path}")
            frames.extend([read_dataframe(p) for p in parts])
        else:
            frames.append(read_dataframe(path))
    if not frames:
        raise CorporateActionsError("No raw corporate action frames could be loaded")
    return pd.concat(frames, ignore_index=True, sort=False)


def normalize_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def normalize_ts_series(series: pd.Series, default_tz: str = "UTC") -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if getattr(parsed.dt, "tz", None) is None:
        parsed = pd.to_datetime(series, errors="coerce").dt.tz_localize(default_tz).dt.tz_convert("UTC")
    return parsed


def coerce_str_series(series: pd.Series) -> pd.Series:
    out = series.astype("string")
    out = out.str.strip()
    out = out.mask(out.eq(""), pd.NA)
    return out


def coerce_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_lower(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.lower()


def json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, set, str, bytes)) else False:
        return None
    return value


def stable_hash_payload(payload: Mapping[str, Any], prefix: str) -> str:
    return f"{prefix}_{sha256_text(canonical_json(json_safe(payload)))[:20]}"


def listify_non_null(values: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    for value in values:
        if isinstance(value, list):
            for inner in value:
                if inner is not None and not pd.isna(inner):
                    out.append(inner)
        elif value is not None and not pd.isna(value):
            out.append(value)
    return out


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


def load_config(config_path: Optional[str | Path]) -> Tuple[CorporateActionsConfig, Dict[str, Any], str]:
    defaults = dataclasses.asdict(CorporateActionsConfig())
    if config_path is None:
        config = CorporateActionsConfig()
        return config, defaults, sha256_text(canonical_json(defaults))

    path = ensure_path(config_path)
    raw_text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise CorporateActionsError("PyYAML is required to read YAML configs.")
        payload = yaml.safe_load(raw_text) or {}
    elif suffix == ".json":
        payload = json.loads(raw_text)
    else:
        raise CorporateActionsError(f"Unsupported config format: {path.suffix}")
    if not isinstance(payload, Mapping):
        raise CorporateActionsError("Config must deserialize to a mapping/object.")

    merged = deep_merge(defaults, dict(payload))
    config = CorporateActionsConfig(
        normalization=NormalizationConfig(**merged.get("normalization", {})),
        matching=MatchingConfig(**merged.get("matching", {})),
        conflict=ConflictConfig(**merged.get("conflict", {})),
        thresholds=ThresholdConfig(**merged.get("thresholds", {})),
        output=OutputConfig(**merged.get("output", {})),
    )
    config_hash = sha256_text(canonical_json(dataclasses.asdict(config)))
    return config, merged, config_hash


# -----------------------------------------------------------------------------
# Input loading
# -----------------------------------------------------------------------------


def _split_paths(raw_paths: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for item in raw_paths:
        parts = [p.strip() for p in str(item).split(",") if p.strip()]
        expanded.extend(parts)
    if not expanded:
        raise CorporateActionsError("At least one raw corporate actions path is required.")
    return expanded


def _first_present(df: pd.DataFrame, names: Sequence[str], default: Any = pd.NA) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([default] * len(df), index=df.index)


def load_raw_corporate_actions(raw_paths: Sequence[str | Path]) -> pd.DataFrame:
    df = read_many_dataframes(raw_paths).copy()
    if df.empty:
        raise CorporateActionsError("Raw corporate actions input is empty.")

    for col in REQUIRED_RAW_CORE_COLUMNS:
        if col not in df.columns:
            raise CorporateActionsError(f"raw corporate actions missing required column: {col}")

    provider_series = _first_present(df, ["provider_name", "provider", "source_provider"]) 
    df["provider_name"] = coerce_str_series(provider_series).fillna("unknown")
    df["provider_event_id"] = coerce_str_series(df["provider_event_id"])
    if df["provider_event_id"].isna().any():
        missing_idx = df["provider_event_id"].isna()
        fallback_payloads = [stable_hash_payload({"i": int(i)}, "raw") for i in df.index[missing_idx]]
        df.loc[missing_idx, "provider_event_id"] = fallback_payloads

    df["raw_event_type"] = coerce_str_series(
        _first_present(df, ["raw_event_type", "event_type", "action_type", "ca_type", "event", "type"])
    )
    df["raw_status"] = coerce_str_series(_first_present(df, ["status", "event_status", "raw_status"]))
    df["announcement_ts_raw"] = _first_present(df, ["announcement_ts", "announcement_datetime", "announced_at", "announcement_date"])
    df["ex_date_raw"] = _first_present(df, ["ex_date", "ex_dt", "exDate"])
    df["effective_date_raw"] = _first_present(df, ["effective_date", "effective_dt", "pay_date", "date"])
    df["symbol_raw"] = coerce_str_series(_first_present(df, ["symbol", "ticker", "ric"]))
    df["exchange_raw"] = coerce_str_series(_first_present(df, ["exchange", "listing_exchange", "venue"]))
    df["share_class_raw"] = coerce_str_series(_first_present(df, ["share_class", "class", "shareClass"]))
    df["currency_raw"] = coerce_str_series(_first_present(df, ["currency", "ccy"]))
    df["instrument_id_raw"] = coerce_str_series(_first_present(df, ["instrument_id"]))
    df["issuer_id_raw"] = coerce_str_series(_first_present(df, ["issuer_id"]))

    for col in ["cash_amount", "gross_amount", "dividend_amount", "amount", "prev_close", "close_t_minus_1"]:
        if col in df.columns:
            df[col] = coerce_float_series(df[col])
    return df.reset_index(drop=True)


def load_identity_master(path_like: str | Path, matching: MatchingConfig) -> pd.DataFrame:
    df = read_dataframe(path_like).copy()
    if "instrument_id" not in df.columns:
        raise CorporateActionsError("identity master must contain instrument_id")
    if "symbol" not in df.columns:
        raise CorporateActionsError("identity master must contain symbol")

    for col in set(["instrument_id", "issuer_id", "symbol", "exchange", "listing_exchange", "share_class"] + list(matching.external_id_columns)):
        if col in df.columns:
            df[col] = coerce_str_series(df[col])

    start_col = next((c for c in matching.active_window_start_columns if c in df.columns), None)
    end_col = next((c for c in matching.active_window_end_columns if c in df.columns), None)
    if start_col is not None:
        df[start_col] = normalize_date_series(df[start_col])
    if end_col is not None:
        df[end_col] = normalize_date_series(df[end_col])
    df["_active_start"] = df[start_col] if start_col is not None else pd.Timestamp("1900-01-01")
    df["_active_end"] = df[end_col] if end_col is not None else pd.Timestamp("2100-12-31")
    df["_active_start"] = normalize_date_series(df["_active_start"])
    df["_active_end"] = normalize_date_series(df["_active_end"])
    df["_active_start"] = df["_active_start"].fillna(pd.Timestamp("1900-01-01"))
    df["_active_end"] = df["_active_end"].fillna(pd.Timestamp("2100-12-31"))
    return df.reset_index(drop=True)


def load_price_reference(path_like: Optional[str | Path]) -> Optional[pd.DataFrame]:
    if path_like is None:
        return None
    df = read_dataframe(path_like).copy()
    missing = PRICE_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise CorporateActionsError(f"prices input missing required columns: {sorted(missing)}")
    df["date"] = normalize_date_series(df["date"])
    df["instrument_id"] = coerce_str_series(df["instrument_id"])
    df["close"] = coerce_float_series(df["close"])
    if "adj_close" in df.columns:
        df["adj_close"] = coerce_float_series(df["adj_close"])
    return df.sort_values(["instrument_id", "date"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------


def _normalize_status(raw_status: Any) -> str:
    token = safe_lower(raw_status)
    if token is None:
        return EventStatus.CONFIRMED.value
    if token in {"pending", "announced", "declared", "proposed"}:
        return EventStatus.PENDING.value
    if token in {"confirmed", "effective", "completed", "done", "active"}:
        return EventStatus.CONFIRMED.value
    if token in {"cancelled", "canceled", "withdrawn", "terminated"}:
        return EventStatus.CANCELLED.value
    if token in {"superseded", "replaced"}:
        return EventStatus.SUPERSEDED.value
    return EventStatus.CONFIRMED.value


def _split_compound_tokens(raw_event_type: str) -> List[str]:
    token = raw_event_type.strip().lower()
    if not token:
        return []
    if re.search(r"\+|/|;|\|| and ", token):
        parts = re.split(r"\+|/|;|\|| and ", token)
        return [p.strip() for p in parts if p.strip()]
    return [token]


def _map_event_type(raw_event_type: Any, aliases: Mapping[str, str]) -> List[str]:
    token = safe_lower(raw_event_type)
    if token is None:
        return []
    mapped: List[str] = []
    for piece in _split_compound_tokens(token):
        if piece in aliases:
            mapped.append(aliases[piece])
            continue
        if "special" in piece and "dividend" in piece:
            mapped.append(CanonicalEventType.SPECIAL_CASH_DIVIDEND.value)
        elif "dividend" in piece and "stock" in piece:
            mapped.append(CanonicalEventType.STOCK_DIVIDEND.value)
        elif "dividend" in piece:
            mapped.append(CanonicalEventType.CASH_DIVIDEND.value)
        elif "reverse" in piece and "split" in piece:
            mapped.append(CanonicalEventType.REVERSE_SPLIT.value)
        elif "split" in piece:
            mapped.append(CanonicalEventType.SPLIT.value)
        elif "ticker" in piece or "symbol" in piece:
            mapped.append(CanonicalEventType.TICKER_CHANGE.value)
        elif "exchange" in piece:
            mapped.append(CanonicalEventType.EXCHANGE_CHANGE.value)
        elif "share class" in piece:
            mapped.append(CanonicalEventType.SHARE_CLASS_CHANGE.value)
        elif "name" in piece:
            mapped.append(CanonicalEventType.NAME_CHANGE.value)
        elif "spin" in piece:
            mapped.append(CanonicalEventType.SPINOFF.value)
        elif "merger" in piece:
            mapped.append(CanonicalEventType.MERGER.value)
        elif "acquisition" in piece or "acquired" in piece:
            mapped.append(CanonicalEventType.ACQUISITION.value)
        elif "rights" in piece:
            mapped.append(CanonicalEventType.RIGHTS_ISSUE.value)
        elif "tender" in piece:
            mapped.append(CanonicalEventType.TENDER_OFFER.value)
        elif "delist" in piece:
            mapped.append(CanonicalEventType.DELISTING.value)
        elif "relist" in piece:
            mapped.append(CanonicalEventType.RELISTING.value)
        elif "bankruptcy" in piece or "reorg" in piece:
            mapped.append(CanonicalEventType.BANKRUPTCY_REORG.value)
        elif "identifier" in piece or "cusip" in piece or "isin" in piece or "figi" in piece:
            mapped.append(CanonicalEventType.IDENTIFIER_MAINTENANCE.value)
    # preserve order, unique
    seen = set()
    ordered = []
    for ev in mapped:
        if ev not in seen:
            seen.add(ev)
            ordered.append(ev)
    return ordered


def _parse_split_ratio(row: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    ratio_fields = ["split_ratio", "ratio", "split_factor", "factor", "conversion_ratio"]
    ratio_value = None
    for field in ratio_fields:
        if field in row and row[field] is not None and not pd.isna(row[field]):
            ratio_value = row[field]
            break
    split_from = row.get("split_from")
    split_to = row.get("split_to")
    if split_from is not None and split_to is not None and not pd.isna(split_from) and not pd.isna(split_to):
        a = float(split_from)
        b = float(split_to)
        if a > 0 and b > 0:
            return a, b, b / a
    if ratio_value is None or pd.isna(ratio_value):
        return None, None, None
    if isinstance(ratio_value, str):
        token = ratio_value.strip().replace(" ", "")
        if ":" in token:
            left, right = token.split(":", 1)
            try:
                a = float(left)
                b = float(right)
                if a > 0 and b > 0:
                    return a, b, b / a
            except ValueError:
                return None, None, None
        if "/" in token:
            left, right = token.split("/", 1)
            try:
                a = float(left)
                b = float(right)
                if a > 0 and b > 0:
                    return a, b, b / a
            except ValueError:
                return None, None, None
    try:
        ratio = float(ratio_value)
    except Exception:
        return None, None, None
    if ratio <= 0:
        return None, None, None
    if ratio >= 1:
        return 1.0, ratio, ratio
    return 1.0 / ratio, 1.0, ratio


def _extract_cash_amount(row: Mapping[str, Any]) -> Optional[float]:
    for field in ["cash_amount", "gross_amount", "dividend_amount", "amount", "cash_consideration"]:
        value = row.get(field)
        if value is not None and not pd.isna(value):
            try:
                return float(value)
            except Exception:
                return None
    return None


def _row_payload(row: pd.Series) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for k, v in row.to_dict().items():
        if isinstance(v, (pd.Timestamp, datetime)):
            payload[k] = pd.Timestamp(v).isoformat()
        elif pd.isna(v) if not isinstance(v, (dict, list, tuple, str, bytes)) else False:
            payload[k] = None
        else:
            payload[k] = v
    return payload


def normalize_raw_events(raw_df: pd.DataFrame, config: CorporateActionsConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for _, raw_row in raw_df.iterrows():
        provider = raw_row["provider_name"]
        provider_priority = config.conflict.provider_priority.get(str(provider).lower(), 999)
        raw_event_type = raw_row.get("raw_event_type")
        mapped_types = _map_event_type(raw_event_type, config.normalization.event_type_aliases)
        if not mapped_types:
            failures.append(
                {
                    "provider_name": provider,
                    "provider_event_id": raw_row.get("provider_event_id"),
                    "failure_code": FailureCode.UNSUPPORTED_EVENT_TYPE.value,
                    "failure_detail": f"Could not map raw_event_type={raw_event_type!r}",
                    "raw_payload": canonical_json(_row_payload(raw_row)),
                }
            )
            continue

        announcement_ts = normalize_ts_series(pd.Series([raw_row.get("announcement_ts_raw")]), config.normalization.timezone).iloc[0]
        ex_date = normalize_date_series(pd.Series([raw_row.get("ex_date_raw")])).iloc[0]
        effective_date = normalize_date_series(pd.Series([raw_row.get("effective_date_raw")])).iloc[0]
        if pd.isna(announcement_ts):
            # conservative fallback: if ex/effective exist, event becomes known only at 23:59 UTC the previous day is not safe;
            # use effective/ex date at midnight UTC to avoid pre-knowledge.
            fallback_date = effective_date if not pd.isna(effective_date) else ex_date
            announcement_ts = pd.Timestamp(fallback_date).tz_localize("UTC") if not pd.isna(fallback_date) else pd.NaT
        if pd.isna(ex_date) and not pd.isna(effective_date):
            ex_date = effective_date
        if pd.isna(effective_date) and not pd.isna(ex_date):
            effective_date = ex_date
        if pd.isna(announcement_ts) and pd.isna(ex_date) and pd.isna(effective_date):
            failures.append(
                {
                    "provider_name": provider,
                    "provider_event_id": raw_row.get("provider_event_id"),
                    "failure_code": FailureCode.INVALID_TEMPORAL_FIELDS.value,
                    "failure_detail": "announcement_ts, ex_date and effective_date are all missing/unparseable",
                    "raw_payload": canonical_json(_row_payload(raw_row)),
                }
            )
            continue

        split_from, split_to, split_ratio = _parse_split_ratio(raw_row)
        cash_amount = _extract_cash_amount(raw_row)
        quality_flags: List[str] = []
        if split_ratio is not None and split_ratio <= 0:
            quality_flags.append(FailureCode.INVALID_SPLIT_RATIO.value)
        if cash_amount is not None and cash_amount < 0:
            quality_flags.append(FailureCode.INVALID_CASH_AMOUNT.value)

        currency = raw_row.get("currency_raw")
        if currency is None or pd.isna(currency):
            currency = config.normalization.default_currency

        prev_close = raw_row.get("prev_close")
        if prev_close is None or pd.isna(prev_close):
            prev_close = raw_row.get("close_t_minus_1")
        prev_close = float(prev_close) if prev_close is not None and not pd.isna(prev_close) else None

        for type_idx, event_type in enumerate(mapped_types, start=1):
            event_cash_amount = cash_amount
            event_quality = list(quality_flags)
            if event_type == CanonicalEventType.CASH_DIVIDEND.value and event_cash_amount is not None and prev_close is not None and prev_close > 0:
                if event_cash_amount / prev_close >= config.normalization.special_dividend_threshold:
                    event_type = CanonicalEventType.SPECIAL_CASH_DIVIDEND.value
                    event_quality.append("AUTO_RECLASSIFIED_SPECIAL_DIVIDEND")
            event_family = EVENT_TYPE_TO_FAMILY.get(event_type)
            raw_payload = _row_payload(raw_row)
            rows.append(
                {
                    "provider_name": provider,
                    "provider_priority": provider_priority,
                    "provider_event_id": raw_row.get("provider_event_id"),
                    "raw_event_type": raw_event_type,
                    "canonical_event_type": event_type,
                    "event_family": event_family,
                    "compound_event_count": len(mapped_types),
                    "compound_position": type_idx,
                    "compound_event_id": stable_hash_payload(
                        {"provider": provider, "provider_event_id": raw_row.get("provider_event_id")}, "compound"
                    ) if len(mapped_types) > 1 else None,
                    "announcement_ts": announcement_ts,
                    "ex_date": ex_date,
                    "effective_date": effective_date,
                    "raw_status": raw_row.get("raw_status"),
                    "status": _normalize_status(raw_row.get("raw_status")),
                    "symbol": raw_row.get("symbol_raw"),
                    "exchange": raw_row.get("exchange_raw"),
                    "share_class": raw_row.get("share_class_raw"),
                    "instrument_id_input": raw_row.get("instrument_id_raw"),
                    "issuer_id_input": raw_row.get("issuer_id_raw"),
                    "currency": currency,
                    "cash_amount": event_cash_amount,
                    "prev_close_t_minus_1": prev_close,
                    "split_from": split_from,
                    "split_to": split_to,
                    "split_ratio": split_ratio,
                    "raw_payload": canonical_json(raw_payload),
                    "quality_flags": json.dumps(event_quality, ensure_ascii=False),
                    "figi": raw_row.get("figi"),
                    "cusip": raw_row.get("cusip"),
                    "isin": raw_row.get("isin"),
                    "sedol": raw_row.get("sedol"),
                    "cik": raw_row.get("cik"),
                    "vendor_instrument_id": raw_row.get("vendor_instrument_id"),
                    "new_symbol": raw_row.get("new_symbol") if "new_symbol" in raw_row.index else raw_row.get("to_symbol"),
                    "new_exchange": raw_row.get("new_exchange") if "new_exchange" in raw_row.index else raw_row.get("to_exchange"),
                    "new_share_class": raw_row.get("new_share_class") if "new_share_class" in raw_row.index else raw_row.get("to_share_class"),
                    "new_name": raw_row.get("new_name") if "new_name" in raw_row.index else raw_row.get("to_name"),
                    "survivor_instrument_id_input": raw_row.get("survivor_instrument_id") if "survivor_instrument_id" in raw_row.index else raw_row.get("to_instrument_id"),
                    "child_instrument_id_input": raw_row.get("child_instrument_id") if "child_instrument_id" in raw_row.index else raw_row.get("spin_child_instrument_id"),
                    "spinoff_ratio": coerce_float_series(pd.Series([raw_row.get("spinoff_ratio") if "spinoff_ratio" in raw_row.index else raw_row.get("alpha")])).iloc[0],
                    "conversion_ratio": coerce_float_series(pd.Series([raw_row.get("conversion_ratio")])).iloc[0],
                    "stock_cash_mix": raw_row.get("stock_cash_mix") if "stock_cash_mix" in raw_row.index else pd.NA,
                    "ingest_ts_utc": normalize_ts_series(pd.Series([raw_row.get("ingest_ts") if "ingest_ts" in raw_row.index else pd.NaT])).iloc[0],
                }
            )
    normalized = pd.DataFrame(rows)
    failure_df = pd.DataFrame(failures)
    if normalized.empty:
        return normalized, failure_df

    normalized["provider_name"] = coerce_str_series(normalized["provider_name"]).fillna("unknown")
    normalized["provider_event_id"] = coerce_str_series(normalized["provider_event_id"])
    normalized["status"] = coerce_str_series(normalized["status"]).fillna(EventStatus.CONFIRMED.value)
    for col in [
        "symbol",
        "exchange",
        "share_class",
        "instrument_id_input",
        "issuer_id_input",
        "new_symbol",
        "new_exchange",
        "new_share_class",
        "new_name",
        "survivor_instrument_id_input",
        "child_instrument_id_input",
        "figi",
        "cusip",
        "isin",
        "sedol",
        "cik",
        "vendor_instrument_id",
        "currency",
    ]:
        if col in normalized.columns:
            normalized[col] = coerce_str_series(normalized[col])

    normalized["canonical_date"] = normalize_date_series(
        normalized["effective_date"].where(normalized["effective_date"].notna(), normalized["ex_date"])
    )
    normalized["canonical_date"] = normalized["canonical_date"].where(
        normalized["canonical_date"].notna(), normalized["announcement_ts"].dt.tz_convert("UTC").dt.normalize()
    )
    return normalized.reset_index(drop=True), failure_df


# -----------------------------------------------------------------------------
# Identity resolution
# -----------------------------------------------------------------------------


def _active_slice(identity_df: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    if pd.isna(ref_date):
        return identity_df
    return identity_df[(identity_df["_active_start"] <= ref_date) & (identity_df["_active_end"].fillna(pd.Timestamp("2100-12-31")) >= ref_date)]


def _resolve_by_external_ids(row: pd.Series, active_id: pd.DataFrame, matching: MatchingConfig) -> Tuple[Optional[pd.Series], Optional[str], float]:
    for col in matching.external_id_columns:
        if col not in row.index or row[col] is None or pd.isna(row[col]) or col not in active_id.columns:
            continue
        candidates = active_id[active_id[col].eq(row[col])]
        if len(candidates) == 1:
            return candidates.iloc[0], f"external_id:{col}", 0.98
        if len(candidates) > 1:
            unique_instruments = candidates["instrument_id"].dropna().unique() if "instrument_id" in candidates.columns else []
            if len(unique_instruments) == 1:
                return candidates.iloc[0], f"external_id_single_instrument:{col}", 0.97
            return None, f"ambiguous_external_id:{col}", 0.0
    return None, None, 0.0


def _resolve_by_symbol(row: pd.Series, active_id: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[str], float]:
    symbol = row.get("symbol")
    exchange = row.get("exchange")
    share_class = row.get("share_class")
    if symbol is None or pd.isna(symbol):
        return None, None, 0.0

    candidates = active_id[active_id["symbol"].eq(symbol)] if "symbol" in active_id.columns else active_id.iloc[0:0]
    if exchange is not None and not pd.isna(exchange):
        exch_col = "exchange" if "exchange" in candidates.columns else ("listing_exchange" if "listing_exchange" in candidates.columns else None)
        if exch_col is not None:
            exact = candidates[candidates[exch_col].eq(exchange)]
            if not exact.empty:
                candidates = exact
    if share_class is not None and not pd.isna(share_class) and "share_class" in candidates.columns:
        exact_sc = candidates[candidates["share_class"].eq(share_class)]
        if not exact_sc.empty:
            candidates = exact_sc

    if len(candidates) == 1:
        method = "pit_symbol_exchange_shareclass"
        score = 0.90
        if exchange is None or pd.isna(exchange):
            method = "pit_symbol"
            score = 0.80
        return candidates.iloc[0], method, score
    if len(candidates) > 1:
        unique_instrument = candidates["instrument_id"].dropna().unique()
        if len(unique_instrument) == 1:
            return candidates.iloc[0], "pit_symbol_multirow_single_instrument", 0.88
        return None, "ambiguous_symbol_match", 0.0
    return None, None, 0.0


def resolve_identity(normalized: pd.DataFrame, identity_df: pd.DataFrame, matching: MatchingConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if normalized.empty:
        return normalized.copy(), pd.DataFrame()

    resolved_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    identity_index = identity_df.copy()
    if "instrument_id" in identity_index.columns:
        identity_index["instrument_id"] = coerce_str_series(identity_index["instrument_id"])

    for _, row in normalized.iterrows():
        ref_date = row.get("effective_date")
        if pd.isna(ref_date):
            ref_date = row.get("ex_date")
        if pd.isna(ref_date):
            ref_date = row.get("canonical_date")
        active = _active_slice(identity_index, ref_date)

        chosen: Optional[pd.Series] = None
        linkage_method: Optional[str] = None
        confidence = 0.0
        failure_code: Optional[str] = None
        failure_detail: Optional[str] = None

        explicit_instrument = row.get("instrument_id_input")
        if matching.prefer_explicit_instrument_id and explicit_instrument is not None and not pd.isna(explicit_instrument):
            matches = active[active["instrument_id"].eq(explicit_instrument)]
            if len(matches) == 1:
                chosen = matches.iloc[0]
                linkage_method = "explicit_instrument_id"
                confidence = 1.0
            elif len(matches) > 1:
                failure_code = FailureCode.AMBIGUOUS_IDENTITY.value
                failure_detail = f"explicit instrument_id={explicit_instrument} matched multiple active rows"
            else:
                # if explicit id not active at ref_date, still allow exact historical match conservatively.
                hist = identity_index[identity_index["instrument_id"].eq(explicit_instrument)]
                if len(hist) == 1:
                    chosen = hist.iloc[0]
                    linkage_method = "explicit_instrument_id_outside_window"
                    confidence = 0.95
                else:
                    failure_code = FailureCode.UNRESOLVED_IDENTITY.value
                    failure_detail = f"explicit instrument_id={explicit_instrument} not found in identity master"

        if chosen is None and failure_code is None:
            chosen, linkage_method, confidence = _resolve_by_external_ids(row, active, matching)
            if linkage_method and linkage_method.startswith("ambiguous"):
                failure_code = FailureCode.AMBIGUOUS_IDENTITY.value
                failure_detail = linkage_method
                chosen = None

        if chosen is None and failure_code is None:
            chosen, linkage_method, confidence = _resolve_by_symbol(row, active)
            if linkage_method and linkage_method.startswith("ambiguous"):
                failure_code = FailureCode.AMBIGUOUS_IDENTITY.value
                failure_detail = linkage_method
                chosen = None

        if chosen is None and failure_code is None:
            failure_code = FailureCode.UNRESOLVED_IDENTITY.value
            failure_detail = "No explicit id, external-id or PIT symbol mapping resolved uniquely"

        if chosen is None:
            failures.append(
                {
                    "provider_name": row.get("provider_name"),
                    "provider_event_id": row.get("provider_event_id"),
                    "canonical_event_type": row.get("canonical_event_type"),
                    "failure_code": failure_code,
                    "failure_detail": failure_detail,
                    "symbol": row.get("symbol"),
                    "exchange": row.get("exchange"),
                    "share_class": row.get("share_class"),
                    "canonical_date": row.get("canonical_date"),
                    "raw_payload": row.get("raw_payload"),
                }
            )
            continue

        if confidence < matching.min_confidence_score:
            failures.append(
                {
                    "provider_name": row.get("provider_name"),
                    "provider_event_id": row.get("provider_event_id"),
                    "canonical_event_type": row.get("canonical_event_type"),
                    "failure_code": FailureCode.LOW_CONFIDENCE_IDENTITY.value,
                    "failure_detail": f"confidence_score={confidence:.3f} below threshold={matching.min_confidence_score:.3f}",
                    "symbol": row.get("symbol"),
                    "exchange": row.get("exchange"),
                    "share_class": row.get("share_class"),
                    "canonical_date": row.get("canonical_date"),
                    "raw_payload": row.get("raw_payload"),
                }
            )
            continue

        enriched = row.to_dict()
        enriched["instrument_id"] = chosen.get("instrument_id")
        enriched["issuer_id"] = chosen.get("issuer_id") if "issuer_id" in chosen.index else row.get("issuer_id_input")
        enriched["symbol"] = chosen.get("symbol") if "symbol" in chosen.index and pd.notna(chosen.get("symbol")) else row.get("symbol")
        exch_col = "exchange" if "exchange" in chosen.index else ("listing_exchange" if "listing_exchange" in chosen.index else None)
        if exch_col is not None:
            enriched["exchange"] = chosen.get(exch_col)
        enriched["share_class"] = chosen.get("share_class") if "share_class" in chosen.index else row.get("share_class")
        enriched["linkage_method"] = linkage_method
        enriched["confidence_score"] = confidence
        resolved_rows.append(enriched)

    resolved_df = pd.DataFrame(resolved_rows)
    failure_df = pd.DataFrame(failures)
    return resolved_df.reset_index(drop=True), failure_df


# -----------------------------------------------------------------------------
# Deduplication, conflicts and canonicalization
# -----------------------------------------------------------------------------


def _economic_signature(row: pd.Series) -> str:
    payload = {
        "event_type": row.get("canonical_event_type"),
        "split_ratio": None if pd.isna(row.get("split_ratio")) else round(float(row.get("split_ratio")), 12),
        "cash_amount": None if pd.isna(row.get("cash_amount")) else round(float(row.get("cash_amount")), 8),
        "currency": row.get("currency"),
        "new_symbol": row.get("new_symbol"),
        "new_exchange": row.get("new_exchange"),
        "new_share_class": row.get("new_share_class"),
        "survivor_instrument_id": row.get("survivor_instrument_id_input"),
        "child_instrument_id": row.get("child_instrument_id_input"),
        "spinoff_ratio": None if pd.isna(row.get("spinoff_ratio")) else round(float(row.get("spinoff_ratio")), 12),
        "conversion_ratio": None if pd.isna(row.get("conversion_ratio")) else round(float(row.get("conversion_ratio")), 12),
    }
    return sha256_text(canonical_json(payload))[:16]


@dataclass
class Cluster:
    anchor_date: Optional[pd.Timestamp]
    rows: List[pd.Series]


def _cluster_resolved_events(resolved: pd.DataFrame, tolerance_days: int) -> List[Cluster]:
    if resolved.empty:
        return []
    work = resolved.copy()
    work = work.sort_values(
        ["instrument_id", "canonical_event_type", "canonical_date", "provider_priority", "provider_name", "provider_event_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    clusters: List[Cluster] = []
    for _, row in work.iterrows():
        row_date = row.get("canonical_date")
        if pd.isna(row_date):
            row_date = None
        placed = False
        for cluster in reversed(clusters):
            first = cluster.rows[0]
            if first.get("instrument_id") != row.get("instrument_id"):
                break
            if first.get("canonical_event_type") != row.get("canonical_event_type"):
                continue
            if cluster.anchor_date is None or row_date is None:
                cluster.rows.append(row)
                placed = True
                break
            if abs((pd.Timestamp(row_date) - pd.Timestamp(cluster.anchor_date)).days) <= tolerance_days:
                cluster.rows.append(row)
                placed = True
                break
        if not placed:
            clusters.append(Cluster(anchor_date=row_date if row_date is not None else None, rows=[row]))
    return clusters


def _field_materially_conflicts(values: List[Any], field: str, config: CorporateActionsConfig) -> bool:
    cleaned = [v for v in values if v is not None and not (pd.isna(v) if not isinstance(v, (dict, list, tuple, str, bytes)) else False)]
    if len(cleaned) <= 1:
        return False
    if field in {"split_ratio", "cash_amount"}:
        nums = [float(v) for v in cleaned]
        tol = config.normalization.split_ratio_tolerance if field == "split_ratio" else config.normalization.cash_amount_tolerance
        return max(nums) - min(nums) > tol
    if field in {"ex_date", "effective_date"}:
        dates = [pd.Timestamp(v).normalize() for v in cleaned]
        return len({d.isoformat() for d in dates}) > 1
    return len({str(v) for v in cleaned}) > 1


def _choose_canonical_value(rows: List[pd.Series], field: str) -> Any:
    ordered = sorted(rows, key=lambda r: (r.get("provider_priority", 999), str(r.get("provider_name")), str(r.get("provider_event_id"))))
    for row in ordered:
        value = row.get(field)
        if value is not None and not (pd.isna(value) if not isinstance(value, (dict, list, tuple, str, bytes)) else False):
            return value
    return None


def _derive_status(selected_status: Any, effective_date: Any, asof_ts_utc: str) -> str:
    token = safe_lower(selected_status)
    if token in {EventStatus.CANCELLED.value, EventStatus.SUPERSEDED.value}:
        return token  # type: ignore[return-value]
    asof_ts = pd.Timestamp(asof_ts_utc, tz="UTC")
    if effective_date is not None and not pd.isna(effective_date):
        effective_ts = pd.Timestamp(effective_date).tz_localize("UTC") if pd.Timestamp(effective_date).tzinfo is None else pd.Timestamp(effective_date).tz_convert("UTC")
        if effective_ts > asof_ts:
            return EventStatus.PENDING.value
    if token == EventStatus.PENDING.value:
        return EventStatus.PENDING.value
    return EventStatus.CONFIRMED.value


def _build_lineage(rows: List[pd.Series]) -> List[Dict[str, Any]]:
    lineage = []
    for row in sorted(rows, key=lambda r: (r.get("provider_priority", 999), str(r.get("provider_name")), str(r.get("provider_event_id")))):
        lineage.append(
            {
                "provider_name": row.get("provider_name"),
                "provider_priority": row.get("provider_priority"),
                "provider_event_id": row.get("provider_event_id"),
                "raw_event_type": row.get("raw_event_type"),
                "raw_status": row.get("raw_status"),
                "announcement_ts": json_safe(row.get("announcement_ts")),
                "ex_date": json_safe(row.get("ex_date")),
                "effective_date": json_safe(row.get("effective_date")),
            }
        )
    return lineage


def canonicalize_events(resolved: pd.DataFrame, config: CorporateActionsConfig, asof_ts_utc: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if resolved.empty:
        return pd.DataFrame(), pd.DataFrame()

    canonical_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    clusters = _cluster_resolved_events(resolved, config.normalization.date_tolerance_days)
    for cluster in clusters:
        rows = cluster.rows
        critical_conflicts: Dict[str, List[Any]] = {}
        for field in config.conflict.critical_fields:
            values = [r.get(field) for r in rows]
            if _field_materially_conflicts(values, field, config):
                critical_conflicts[field] = [json_safe(v) for v in values]

        if critical_conflicts:
            representative = rows[0]
            failures.append(
                {
                    "provider_name": representative.get("provider_name"),
                    "provider_event_id": representative.get("provider_event_id"),
                    "canonical_event_type": representative.get("canonical_event_type"),
                    "failure_code": FailureCode.MATERIAL_CONFLICT.value,
                    "failure_detail": f"Material conflict in fields: {sorted(critical_conflicts)}",
                    "instrument_id": representative.get("instrument_id"),
                    "canonical_date": representative.get("canonical_date"),
                    "conflict_fields": canonical_json(critical_conflicts),
                    "raw_lineage": canonical_json(_build_lineage(rows)),
                }
            )
            continue

        first = sorted(rows, key=lambda r: (r.get("provider_priority", 999), str(r.get("provider_name")), str(r.get("provider_event_id"))))[0]
        event_type = _choose_canonical_value(rows, "canonical_event_type")
        event_family = EVENT_TYPE_TO_FAMILY.get(str(event_type), None)
        instrument_id = _choose_canonical_value(rows, "instrument_id")
        issuer_id = _choose_canonical_value(rows, "issuer_id")
        symbol = _choose_canonical_value(rows, "symbol")
        exchange = _choose_canonical_value(rows, "exchange")
        share_class = _choose_canonical_value(rows, "share_class")
        announcement_ts = _choose_canonical_value(rows, "announcement_ts")
        ex_date = _choose_canonical_value(rows, "ex_date")
        effective_date = _choose_canonical_value(rows, "effective_date")
        split_from = _choose_canonical_value(rows, "split_from")
        split_to = _choose_canonical_value(rows, "split_to")
        split_ratio = _choose_canonical_value(rows, "split_ratio")
        cash_amount = _choose_canonical_value(rows, "cash_amount")
        currency = _choose_canonical_value(rows, "currency")
        prev_close = _choose_canonical_value(rows, "prev_close_t_minus_1")
        survivor_instrument_id = _choose_canonical_value(rows, "survivor_instrument_id_input")
        child_instrument_id = _choose_canonical_value(rows, "child_instrument_id_input")
        spinoff_ratio = _choose_canonical_value(rows, "spinoff_ratio")
        conversion_ratio = _choose_canonical_value(rows, "conversion_ratio")
        stock_cash_mix = _choose_canonical_value(rows, "stock_cash_mix")
        new_symbol = _choose_canonical_value(rows, "new_symbol")
        new_exchange = _choose_canonical_value(rows, "new_exchange")
        new_share_class = _choose_canonical_value(rows, "new_share_class")
        new_name = _choose_canonical_value(rows, "new_name")
        status = _derive_status(_choose_canonical_value(rows, "status"), effective_date, asof_ts_utc)
        confidence_score = max(float(r.get("confidence_score") or 0.0) for r in rows)
        source_provider = first.get("provider_name")
        source_priority = first.get("provider_priority")
        quality_flags = listify_non_null([json.loads(r.get("quality_flags") or "[]") for r in rows])
        raw_lineage = _build_lineage(rows)
        economic_signature = _economic_signature(first)
        event_key_payload = {
            "instrument_id": instrument_id,
            "event_type": event_type,
            "announcement_ts": json_safe(announcement_ts),
            "ex_date": json_safe(ex_date),
            "effective_date": json_safe(effective_date),
            "economic_signature": economic_signature,
        }
        event_id = stable_hash_payload(event_key_payload, "ca")

        if event_type in {CanonicalEventType.SPLIT.value, CanonicalEventType.REVERSE_SPLIT.value, CanonicalEventType.STOCK_DIVIDEND.value}:
            if split_ratio is None or pd.isna(split_ratio) or float(split_ratio) <= 0:
                failures.append(
                    {
                        "provider_name": source_provider,
                        "provider_event_id": first.get("provider_event_id"),
                        "canonical_event_type": event_type,
                        "failure_code": FailureCode.INVALID_SPLIT_RATIO.value,
                        "failure_detail": "Mechanical adjustment event requires a valid positive split_ratio",
                        "instrument_id": instrument_id,
                        "canonical_date": first.get("canonical_date"),
                        "raw_lineage": canonical_json(raw_lineage),
                    }
                )
                continue

        if event_type in {CanonicalEventType.CASH_DIVIDEND.value, CanonicalEventType.SPECIAL_CASH_DIVIDEND.value}:
            if cash_amount is not None and not pd.isna(cash_amount) and float(cash_amount) < 0:
                failures.append(
                    {
                        "provider_name": source_provider,
                        "provider_event_id": first.get("provider_event_id"),
                        "canonical_event_type": event_type,
                        "failure_code": FailureCode.INVALID_CASH_AMOUNT.value,
                        "failure_detail": "cash_amount must be non-negative",
                        "instrument_id": instrument_id,
                        "canonical_date": first.get("canonical_date"),
                        "raw_lineage": canonical_json(raw_lineage),
                    }
                )
                continue

        adjustment_factor = None
        adjustment_factor_kind = None
        if event_type in {CanonicalEventType.SPLIT.value, CanonicalEventType.REVERSE_SPLIT.value, CanonicalEventType.STOCK_DIVIDEND.value} and split_ratio is not None and not pd.isna(split_ratio):
            adjustment_factor = float(split_ratio)
            adjustment_factor_kind = "backward_price_factor"
        elif event_type in {CanonicalEventType.CASH_DIVIDEND.value, CanonicalEventType.SPECIAL_CASH_DIVIDEND.value}:
            if cash_amount is not None and prev_close is not None and prev_close > 0 and 0 < float(cash_amount) < float(prev_close):
                adjustment_factor = 1.0 - float(cash_amount) / float(prev_close)
                adjustment_factor_kind = "backward_price_factor"
            elif cash_amount is not None:
                quality_flags.append("DIVIDEND_FACTOR_NOT_COMPUTABLE")

        consideration = None
        if event_type in {CanonicalEventType.MERGER.value, CanonicalEventType.ACQUISITION.value}:
            consideration = {
                "conversion_ratio": None if conversion_ratio is None or pd.isna(conversion_ratio) else float(conversion_ratio),
                "cash_consideration": None if cash_amount is None or pd.isna(cash_amount) else float(cash_amount),
                "stock_cash_mix": stock_cash_mix,
                "survivor_instrument_id": survivor_instrument_id,
            }
        elif event_type == CanonicalEventType.SPINOFF.value:
            consideration = {
                "instrument_parent": instrument_id,
                "instrument_child": child_instrument_id,
                "alpha": None if spinoff_ratio is None or pd.isna(spinoff_ratio) else float(spinoff_ratio),
                "ex_date": json_safe(ex_date),
            }

        canonical_rows.append(
            {
                "event_id": event_id,
                "instrument_id": instrument_id,
                "issuer_id": issuer_id,
                "symbol": symbol,
                "exchange": exchange,
                "share_class": share_class,
                "event_type": event_type,
                "event_family": event_family,
                "announcement_ts": announcement_ts,
                "ex_date": ex_date,
                "effective_date": effective_date,
                "status": status,
                "source_provider": source_provider,
                "source_priority": source_priority,
                "confidence_score": confidence_score,
                "raw_event_refs": canonical_json(raw_lineage),
                "raw_records_count": len(rows),
                "quality_flags": canonical_json(sorted(set(str(q) for q in quality_flags))),
                "split_from": split_from,
                "split_to": split_to,
                "split_ratio": None if split_ratio is None or pd.isna(split_ratio) else float(split_ratio),
                "cash_amount": None if cash_amount is None or pd.isna(cash_amount) else float(cash_amount),
                "currency": currency,
                "prev_close_t_minus_1": None if prev_close is None or pd.isna(prev_close) else float(prev_close),
                "adjustment_factor": adjustment_factor,
                "adjustment_factor_kind": adjustment_factor_kind,
                "survivor_instrument_id": survivor_instrument_id,
                "child_instrument_id": child_instrument_id,
                "spinoff_ratio": None if spinoff_ratio is None or pd.isna(spinoff_ratio) else float(spinoff_ratio),
                "consideration": canonical_json(consideration) if consideration is not None else None,
                "new_symbol": new_symbol,
                "new_exchange": new_exchange,
                "new_share_class": new_share_class,
                "new_name": new_name,
                "compound_event_id": _choose_canonical_value(rows, "compound_event_id"),
                "same_day_precedence_rank": None,
                "same_day_sequence": None,
            }
        )

    canonical_df = pd.DataFrame(canonical_rows)
    failure_df = pd.DataFrame(failures)
    if canonical_df.empty:
        return canonical_df, failure_df

    dup_event_id = canonical_df["event_id"].duplicated(keep=False)
    if dup_event_id.any():
        dups = canonical_df.loc[dup_event_id, "event_id"].tolist()
        raise CorporateActionsError(f"Canonical event_id collision detected: {dups[:5]}")

    precedence_map = {ev: i + 1 for i, ev in enumerate(config.conflict.same_day_precedence)}
    canonical_df["same_day_precedence_rank"] = canonical_df["event_type"].map(precedence_map).fillna(999).astype(int)
    canonical_df["application_date"] = normalize_date_series(
        canonical_df["effective_date"].where(canonical_df["effective_date"].notna(), canonical_df["ex_date"])
    )
    canonical_df = canonical_df.sort_values(
        ["instrument_id", "application_date", "same_day_precedence_rank", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    canonical_df["same_day_sequence"] = (
        canonical_df.groupby(["instrument_id", "application_date"], dropna=False).cumcount() + 1
    )
    return canonical_df, failure_df


# -----------------------------------------------------------------------------
# Snapshot, QC and outputs
# -----------------------------------------------------------------------------


def build_current_snapshot(history: pd.DataFrame, asof_ts_utc: str) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    asof_ts = pd.Timestamp(asof_ts_utc, tz="UTC")
    work = history.copy()
    ann = pd.to_datetime(work["announcement_ts"], utc=True, errors="coerce")
    work["known_asof"] = ann.le(asof_ts)
    current = work[
        work["known_asof"]
        & work["status"].isin([EventStatus.PENDING.value, EventStatus.CONFIRMED.value])
    ].copy()
    current = current.sort_values(["effective_date", "ex_date", "event_id"], kind="mergesort").reset_index(drop=True)
    return current


def evaluate_gates(history: pd.DataFrame, failures: pd.DataFrame, raw_count: int, config: CorporateActionsConfig) -> List[GateResult]:
    n_canonical = int(len(history))
    n_failures = int(len(failures))
    unresolved_rate = n_failures / raw_count if raw_count else 0.0
    if unresolved_rate > config.thresholds.unresolved_rate_fail:
        sev = Severity.FAIL
    elif unresolved_rate > config.thresholds.unresolved_rate_warn:
        sev = Severity.WARN
    else:
        sev = Severity.PASS
    gates = [
        GateResult(
            name="unresolved_rate",
            severity=sev,
            passed=sev != Severity.FAIL,
            observed=unresolved_rate,
            threshold={"warn": config.thresholds.unresolved_rate_warn, "fail": config.thresholds.unresolved_rate_fail},
            message="Rate of unresolved/conflicting/unlinked raw events.",
        )
    ]

    conflict_rate = 0.0
    if not failures.empty and "failure_code" in failures.columns:
        conflict_rate = float((failures["failure_code"] == FailureCode.MATERIAL_CONFLICT.value).sum()) / raw_count if raw_count else 0.0
    if conflict_rate > config.thresholds.conflict_rate_fail:
        sev = Severity.FAIL
    elif conflict_rate > config.thresholds.conflict_rate_warn:
        sev = Severity.WARN
    else:
        sev = Severity.PASS
    gates.append(
        GateResult(
            name="material_conflict_rate",
            severity=sev,
            passed=sev != Severity.FAIL,
            observed=conflict_rate,
            threshold={"warn": config.thresholds.conflict_rate_warn, "fail": config.thresholds.conflict_rate_fail},
            message="Share of failed normalized events attributable to material inter-provider conflicts.",
        )
    )

    provider_concentration = 0.0
    if not history.empty and "source_provider" in history.columns:
        provider_counts = history["source_provider"].value_counts(normalize=True)
        provider_concentration = float(provider_counts.iloc[0]) if len(provider_counts) else 0.0
    if provider_concentration > config.thresholds.single_provider_concentration_fail:
        sev = Severity.FAIL
    elif provider_concentration > config.thresholds.single_provider_concentration_warn:
        sev = Severity.WARN
    else:
        sev = Severity.PASS
    gates.append(
        GateResult(
            name="provider_concentration",
            severity=sev,
            passed=sev != Severity.FAIL,
            observed=provider_concentration,
            threshold={
                "warn": config.thresholds.single_provider_concentration_warn,
                "fail": config.thresholds.single_provider_concentration_fail,
            },
            message="Concentration of canonical source-of-truth on a single provider.",
        )
    )

    special_div_rate = 0.0
    relisting_rate = 0.0
    if not history.empty:
        type_norm = history["event_type"].astype(str)
        special_div_rate = float(type_norm.eq(CanonicalEventType.SPECIAL_CASH_DIVIDEND.value).mean())
        relisting_rate = float(type_norm.eq(CanonicalEventType.RELISTING.value).mean())
    gates.append(
        GateResult(
            name="special_dividend_rate",
            severity=Severity.WARN if special_div_rate > config.thresholds.special_dividend_rate_warn else Severity.PASS,
            passed=True,
            observed=special_div_rate,
            threshold=config.thresholds.special_dividend_rate_warn,
            message="Monitor high share of special cash dividends.",
        )
    )
    gates.append(
        GateResult(
            name="relisting_rate",
            severity=Severity.WARN if relisting_rate > config.thresholds.relisting_rate_warn else Severity.PASS,
            passed=True,
            observed=relisting_rate,
            threshold=config.thresholds.relisting_rate_warn,
            message="Monitor unusual share of relisting events.",
        )
    )

    hard_fail = False
    if not history.empty:
        if history["event_id"].duplicated().any():
            hard_fail = True
        mech = history[history["event_type"].isin([
            CanonicalEventType.SPLIT.value,
            CanonicalEventType.REVERSE_SPLIT.value,
            CanonicalEventType.STOCK_DIVIDEND.value,
        ])]
        if not mech.empty and mech["split_ratio"].isna().any():
            hard_fail = True
        cash = history[history["event_type"].isin([CanonicalEventType.CASH_DIVIDEND.value, CanonicalEventType.SPECIAL_CASH_DIVIDEND.value])]
        if not cash.empty and (cash["cash_amount"].fillna(0) < 0).any():
            hard_fail = True
        confirmed = history[history["status"] == EventStatus.CONFIRMED.value]
        if not confirmed.empty and confirmed["instrument_id"].isna().any():
            hard_fail = True
    gates.append(
        GateResult(
            name="hard_invariants",
            severity=Severity.FAIL if hard_fail else Severity.PASS,
            passed=not hard_fail,
            observed="ok" if not hard_fail else "violated",
            threshold="no violations",
            message="Uniqueness, resolved identity, valid split ratios and non-negative cash dividends.",
        )
    )
    return gates


def build_summary(history: pd.DataFrame, current: pd.DataFrame, failures: pd.DataFrame, gates: List[GateResult], raw_count: int) -> Dict[str, Any]:
    gate_status = Severity.PASS.value
    if any(g.severity == Severity.FAIL for g in gates):
        gate_status = Severity.FAIL.value
    elif any(g.severity == Severity.WARN for g in gates):
        gate_status = Severity.WARN.value

    counts_by_type = history["event_type"].value_counts(dropna=False).to_dict() if not history.empty else {}
    coverage_by_provider = history["source_provider"].value_counts(dropna=False).to_dict() if not history.empty else {}
    failure_counts = failures["failure_code"].value_counts(dropna=False).to_dict() if not failures.empty and "failure_code" in failures.columns else {}

    return {
        "raw_event_count": int(raw_count),
        "canonical_event_count": int(len(history)),
        "current_snapshot_count": int(len(current)),
        "failure_count": int(len(failures)),
        "counts_by_event_type": {str(k): int(v) for k, v in counts_by_type.items()},
        "coverage_by_provider": {str(k): int(v) for k, v in coverage_by_provider.items()},
        "failure_counts": {str(k): int(v) for k, v in failure_counts.items()},
        "n_fail": int(sum(g.severity == Severity.FAIL for g in gates)),
        "n_warn": int(sum(g.severity == Severity.WARN for g in gates)),
        "gate_status": gate_status,
    }


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------


def canonicalize_corporate_actions(
    raw_paths: Sequence[str | Path],
    identity_master_path: str | Path,
    run_id: str,
    asof_ts_utc: str,
    config: CorporateActionsConfig,
    config_hash: str,
    config_payload: Mapping[str, Any],
    prices_path: Optional[str | Path] = None,
) -> CanonicalArtifacts:
    build_started_utc = utc_now_iso()
    raw_paths_list = _split_paths([str(p) for p in raw_paths])
    raw_df = load_raw_corporate_actions(raw_paths_list)
    identity_df = load_identity_master(identity_master_path, config.matching)
    prices_df = load_price_reference(prices_path)
    raw_count = len(raw_df)

    logger.info("Normalizing raw corporate action feeds: n_raw=%s", raw_count)
    normalized, failures_norm = normalize_raw_events(raw_df, config)

    if normalized.empty and failures_norm.empty:
        raise CorporateActionsError("Normalization produced neither canonical candidates nor failure records.")

    if prices_df is not None and not normalized.empty:
        price_prev = prices_df.rename(columns={"date": "price_ref_date", "close": "price_ref_close"})
        merge_dates = normalize_date_series(normalized["ex_date"] - pd.Timedelta(days=1))
        normalized["price_ref_date"] = merge_dates
        normalized = normalized.merge(
            price_prev[["instrument_id", "price_ref_date", "price_ref_close"]],
            left_on=["instrument_id_input", "price_ref_date"],
            right_on=["instrument_id", "price_ref_date"],
            how="left",
            suffixes=("", "_price"),
        )
        normalized["prev_close_t_minus_1"] = normalized["prev_close_t_minus_1"].where(
            normalized["prev_close_t_minus_1"].notna(), normalized["price_ref_close"]
        )
        normalized = normalized.drop(columns=[c for c in ["instrument_id_price", "price_ref_close"] if c in normalized.columns])

    logger.info("Resolving identity on normalized events: n_candidates=%s", len(normalized))
    resolved, failures_id = resolve_identity(normalized, identity_df, config.matching)

    logger.info("Deduplicating and canonicalizing resolved events: n_resolved=%s", len(resolved))
    history, failures_canon = canonicalize_events(resolved, config, asof_ts_utc)

    failures = pd.concat([df for df in [failures_norm, failures_id, failures_canon] if not df.empty], ignore_index=True, sort=False)
    if not failures.empty:
        for col in ["provider_name", "provider_event_id", "failure_code", "failure_detail"]:
            if col not in failures.columns:
                failures[col] = pd.NA

    current = build_current_snapshot(history, asof_ts_utc)
    gates = evaluate_gates(history, failures, raw_count, config)
    summary = build_summary(history, current, failures, gates, raw_count)

    manifest = {
        "run_id": run_id,
        "generated_at_utc": utc_now_iso(),
        "build_started_utc": build_started_utc,
        "asof_ts_utc": asof_ts_utc,
        "decision_cutoff_policy": config.normalization.decision_cutoff_policy,
        "code_version": maybe_git_code_version(),
        "python_version": platform.python_version(),
        "config_hash": config_hash,
        "config": json_safe(config_payload),
        "inputs": {
            "raw_paths": raw_paths_list,
            "identity_master_path": str(identity_master_path),
            "prices_path": None if prices_path is None else str(prices_path),
            "raw_hashes": {str(p): sha256_file(ensure_path(p)) if ensure_path(p).is_file() else "directory" for p in raw_paths_list},
            "identity_master_hash": sha256_file(ensure_path(identity_master_path)),
            "prices_hash": sha256_file(ensure_path(prices_path)) if prices_path is not None and ensure_path(prices_path).is_file() else None,
        },
        "outputs": {
            "history": "history.parquet",
            "current": "current.parquet",
            "failures": f"failures_{run_id}.parquet",
            "manifest": f"manifest_{run_id}.json",
            "summary": f"summary_{run_id}.json",
        },
        "counts_by_event_type": summary["counts_by_event_type"],
        "coverage_by_provider": summary["coverage_by_provider"],
        "failure_counts": summary["failure_counts"],
        "checks_executed": [dataclasses.asdict(g) for g in gates],
        "severity_final": summary["gate_status"],
        "temporal_convention": {
            "known_asof(a,t)": "1[announcement_ts(a) <= decision_cutoff(t)]",
            "decision_cutoff(t)": config.normalization.decision_cutoff_policy,
        },
    }

    history = history.reset_index(drop=True)
    current = current.reset_index(drop=True)
    failures = failures.reset_index(drop=True)
    return CanonicalArtifacts(history=history, current=current, failures=failures, summary=summary, manifest=manifest)


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------


def persist_outputs(artifacts: CanonicalArtifacts, output_dir: str | Path, run_id: str, compression: str) -> None:
    ensure_parquet_engine_available()
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    artifacts.history.to_parquet(outdir / "history.parquet", index=False, compression=compression)
    artifacts.current.to_parquet(outdir / "current.parquet", index=False, compression=compression)
    artifacts.failures.to_parquet(outdir / f"failures_{run_id}.parquet", index=False, compression=compression)
    (outdir / f"summary_{run_id}.json").write_text(
        json.dumps(json_safe(artifacts.summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (outdir / f"manifest_{run_id}.json").write_text(
        json.dumps(json_safe(artifacts.manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Canonicalize raw multi-provider corporate actions into a single PIT-safe, deterministic and auditable layer."
    )
    parser.add_argument("--raw-paths", nargs="+", required=True, help="One or more raw corporate-action paths or directories. Comma-separated values are also allowed.")
    parser.add_argument("--identity-master-path", required=True, help="Path to PIT identity master/mapping.")
    parser.add_argument("--prices-path", default=None, help="Optional price reference path for dividend backward factors.")
    parser.add_argument("--config-path", default=None, help="Optional YAML/JSON config.")
    parser.add_argument("--run-id", required=True, help="Stable run identifier.")
    parser.add_argument("--asof-ts-utc", required=True, help="As-of timestamp in UTC, e.g. 2026-03-15T11:00:00Z")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    config, payload, config_hash = load_config(args.config_path)
    output_dir = Path(args.output_dir or config.output.output_dir)

    artifacts = canonicalize_corporate_actions(
        raw_paths=args.raw_paths,
        identity_master_path=args.identity_master_path,
        prices_path=args.prices_path,
        run_id=args.run_id,
        asof_ts_utc=args.asof_ts_utc,
        config=config,
        config_hash=config_hash,
        config_payload=payload,
    )
    persist_outputs(artifacts, output_dir, args.run_id, config.output.compression)
    logger.info(
        "Corporate actions canonicalization complete. gate_status=%s | n_canonical=%s | n_failures=%s | output=%s",
        artifacts.summary["gate_status"],
        artifacts.summary["canonical_event_count"],
        artifacts.summary["failure_count"],
        output_dir,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
