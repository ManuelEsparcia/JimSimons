
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import logging
import math
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# -----------------------------------------------------------------------------
# Constants, enums and exceptions
# -----------------------------------------------------------------------------


class PriceAdjustmentError(RuntimeError):
    """Raised when a hard validation, PIT invariant or adjustment contract fails."""


class Severity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    FAIL = "FAIL"


class AdjustmentMode(str, Enum):
    SPLIT_ONLY = "split_only"
    TOTAL_RETURN_LIKE = "total_return_like"
    DUAL_OUTPUT = "dual_output"


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


class EventOutcome(str, Enum):
    APPLIED = "APPLIED"
    SKIPPED_UNSUPPORTED = "SKIPPED_UNSUPPORTED"
    SKIPPED_SPECIAL_DIVIDEND_EXCLUDED = "SKIPPED_SPECIAL_DIVIDEND_EXCLUDED"
    SKIPPED_INVALID_DIVIDEND_FACTOR = "SKIPPED_INVALID_DIVIDEND_FACTOR"
    SKIPPED_MISSING_PREV_CLOSE = "SKIPPED_MISSING_PREV_CLOSE"
    SKIPPED_OUTSIDE_RANGE = "SKIPPED_OUTSIDE_RANGE"
    SKIPPED_NOT_VISIBLE_PIT = "SKIPPED_NOT_VISIBLE_PIT"
    SKIPPED_MISSING_FIELDS = "SKIPPED_MISSING_FIELDS"


class FailureCode(str, Enum):
    MISSING_REQUIRED_COLUMN = "MISSING_REQUIRED_COLUMN"
    DUPLICATE_PRICE_KEY = "DUPLICATE_PRICE_KEY"
    DUPLICATE_EVENT_KEY = "DUPLICATE_EVENT_KEY"
    NON_POSITIVE_FACTOR = "NON_POSITIVE_FACTOR"
    INVALID_DIVIDEND_FACTOR = "INVALID_DIVIDEND_FACTOR"
    MISSING_PREV_CLOSE_FOR_DIVIDEND = "MISSING_PREV_CLOSE_FOR_DIVIDEND"
    UNSUPPORTED_EVENT_TYPE = "UNSUPPORTED_EVENT_TYPE"
    MATERIAL_CONFLICT = "MATERIAL_CONFLICT"
    PIT_SNAPSHOT_MISSING = "PIT_SNAPSHOT_MISSING"
    INVALID_ADJUST_MODE = "INVALID_ADJUST_MODE"
    INVALID_PRICE_GEOMETRY = "INVALID_PRICE_GEOMETRY"
    INVALID_PRICE_VALUE = "INVALID_PRICE_VALUE"
    INVALID_VOLUME_VALUE = "INVALID_VOLUME_VALUE"
    INVALID_EVENT_TEMPORAL = "INVALID_EVENT_TEMPORAL"


SAME_DAY_PRECEDENCE = [
    CanonicalEventType.SPLIT.value,
    CanonicalEventType.STOCK_DIVIDEND.value,
    CanonicalEventType.CASH_DIVIDEND.value,
    CanonicalEventType.SPECIAL_CASH_DIVIDEND.value,
]

RAW_PRICE_REQUIRED = {"symbol", "date", "open", "high", "low", "close", "volume"}
RAW_EVENT_REQUIRED_ANY = {"symbol", "event_type", "action_type", "ex_date", "effective_date", "announcement_ts", "effective_ts_utc"}
SUPPORTED_EVENT_TYPES = {
    CanonicalEventType.SPLIT.value,
    CanonicalEventType.REVERSE_SPLIT.value,
    CanonicalEventType.STOCK_DIVIDEND.value,
    CanonicalEventType.CASH_DIVIDEND.value,
    CanonicalEventType.SPECIAL_CASH_DIVIDEND.value,
}
UNSUPPORTED_PASS_THROUGH_TYPES = {
    CanonicalEventType.TICKER_CHANGE.value,
    CanonicalEventType.NAME_CHANGE.value,
    CanonicalEventType.SHARE_CLASS_CHANGE.value,
    CanonicalEventType.EXCHANGE_CHANGE.value,
    CanonicalEventType.MERGER.value,
    CanonicalEventType.ACQUISITION.value,
    CanonicalEventType.SPINOFF.value,
    CanonicalEventType.RIGHTS_ISSUE.value,
    CanonicalEventType.TENDER_OFFER.value,
    CanonicalEventType.DELISTING.value,
    CanonicalEventType.RELISTING.value,
    CanonicalEventType.BANKRUPTCY_REORG.value,
}

SEVERITY_ORDER = {Severity.INFO.value: 0, Severity.WARN.value: 1, Severity.FAIL.value: 2}


# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ReverseSplitPolicy:
    extreme_threshold: float = 20.0


@dataclass(frozen=True)
class ConflictPolicy:
    provider_priority: Sequence[str] = field(default_factory=lambda: ["primary_provider", "secondary_provider", "vendor_fallback"])
    same_day_precedence: Sequence[str] = field(default_factory=lambda: list(SAME_DAY_PRECEDENCE))
    split_ratio_tolerance: float = 1e-10
    cash_amount_tolerance: float = 1e-8
    fail_on_material_conflict: bool = False


@dataclass(frozen=True)
class AdjustmentPolicy:
    include_special_cash_dividends: bool = False
    allow_empty_corporate_actions: bool = True
    mode: str = AdjustmentMode.DUAL_OUTPUT.value
    unsupported_event_severity: str = Severity.WARN.value
    missing_prev_close_severity: str = Severity.WARN.value
    invalid_dividend_severity: str = Severity.WARN.value
    abort_on_fail_symbol: bool = False


@dataclass(frozen=True)
class OutputPolicy:
    compression: str = "snappy"


@dataclass(frozen=True)
class AdjustPricesConfig:
    reverse_split: ReverseSplitPolicy = field(default_factory=ReverseSplitPolicy)
    conflict: ConflictPolicy = field(default_factory=ConflictPolicy)
    adjustment: AdjustmentPolicy = field(default_factory=AdjustmentPolicy)
    output: OutputPolicy = field(default_factory=OutputPolicy)


@dataclass
class AdjustmentArtifacts:
    adjusted: pd.DataFrame
    factors: pd.DataFrame
    events_applied: pd.DataFrame
    conflicts: pd.DataFrame
    summary: Dict[str, Any]
    manifest: Dict[str, Any]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_hash_payload(payload: Any, prefix: str = "") -> str:
    raw = json.dumps(json_safe(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return f"{prefix}_{digest}" if prefix else digest


def json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if pd.isna(value) if not isinstance(value, (str, bytes, dict, list, tuple, set)) else False:
        return None
    return value


def canonical_json(payload: Any) -> str:
    return json.dumps(json_safe(payload), sort_keys=True, ensure_ascii=False)


def maybe_import_git_revision() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        return out or None
    except Exception:
        return None


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise PriceAdjustmentError(f"Input file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise PriceAdjustmentError(f"Unsupported file extension for input: {path.suffix}")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_parquet_support() -> None:
    pyarrow_ok = importlib.util.find_spec("pyarrow") is not None
    fastparquet_ok = importlib.util.find_spec("fastparquet") is not None
    if not (pyarrow_ok or fastparquet_ok):
        raise PriceAdjustmentError(
            "Parquet output requires an installed engine ('pyarrow' or 'fastparquet'). "
            "Install one of them before running adjust_prices persistence."
        )


def parse_iso_ts(value: Any) -> Optional[pd.Timestamp]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def normalize_date_series(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, errors="coerce")
    if out.dt.tz is not None:
        out = out.dt.tz_convert(None)
    return out.dt.normalize()


def normalize_timestamp_series(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, utc=True, errors="coerce")
    return out


def build_symbol_key(df: pd.DataFrame) -> pd.Series:
    if "instrument_id" in df.columns:
        instrument = df["instrument_id"].astype(str)
        return instrument.where(instrument.str.len() > 0, df["symbol"].astype(str))
    return df["symbol"].astype(str)


def severity_max(values: Iterable[str]) -> str:
    values_list = [str(v) for v in values if v is not None and str(v) != ""]
    if not values_list:
        return Severity.INFO.value
    return max(values_list, key=lambda x: SEVERITY_ORDER.get(x, -1))


def merge_config_dict(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            merge_config_dict(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def load_config(config_path: Optional[Path], cli_mode: Optional[str] = None) -> AdjustPricesConfig:
    payload: Dict[str, Any] = json.loads(json.dumps(dataclasses.asdict(AdjustPricesConfig())))
    if config_path is not None:
        if not config_path.exists():
            raise PriceAdjustmentError(f"Config file does not exist: {config_path}")
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise PriceAdjustmentError("PyYAML is required to load YAML config files")
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        elif config_path.suffix.lower() == ".json":
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            raise PriceAdjustmentError(f"Unsupported config format: {config_path.suffix}")
        if not isinstance(loaded, dict):
            raise PriceAdjustmentError("Config payload must be a JSON/YAML object")
        merge_config_dict(payload, loaded)
    if cli_mode is not None:
        payload.setdefault("adjustment", {})
        payload["adjustment"]["mode"] = cli_mode
    try:
        cfg = AdjustPricesConfig(
            reverse_split=ReverseSplitPolicy(**payload.get("reverse_split", {})),
            conflict=ConflictPolicy(**payload.get("conflict", {})),
            adjustment=AdjustmentPolicy(**payload.get("adjustment", {})),
            output=OutputPolicy(**payload.get("output", {})),
        )
    except TypeError as exc:
        raise PriceAdjustmentError(f"Invalid config payload: {exc}") from exc
    if cfg.adjustment.mode not in {m.value for m in AdjustmentMode}:
        raise PriceAdjustmentError(f"Invalid adjustment mode in config: {cfg.adjustment.mode}")
    return cfg


# -----------------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------------


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(RAW_PRICE_REQUIRED - set(prices.columns))
    if missing:
        raise PriceAdjustmentError(f"Prices input is missing required columns: {missing}")

    df = prices.copy()
    df["symbol"] = df["symbol"].astype(str)
    if "instrument_id" in df.columns:
        df["instrument_id"] = df["instrument_id"].astype(str)
    df["date"] = normalize_date_series(df["date"])
    if df["date"].isna().any():
        raise PriceAdjustmentError("Prices input contains unparseable dates")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["open", "high", "low", "close"]].isna().any().any():
        raise PriceAdjustmentError("Prices input contains unparseable OHLC values")
    if df["volume"].isna().any():
        raise PriceAdjustmentError("Prices input contains unparseable volume values")

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise PriceAdjustmentError("Prices input contains non-positive OHLC values")
    if (df["volume"] < 0).any():
        raise PriceAdjustmentError("Prices input contains negative volume values")

    if not ((df["low"] <= df[["open", "close"]].min(axis=1)) & (df[["open", "close"]].max(axis=1) <= df["high"])).all():
        raise PriceAdjustmentError("Prices input violates OHLC geometry")

    key_cols = ["symbol", "date"] if "instrument_id" not in df.columns else ["instrument_id", "date"]
    duplicated = df.duplicated(key_cols, keep=False)
    if duplicated.any():
        sample = df.loc[duplicated, key_cols].head(10).to_dict(orient="records")
        raise PriceAdjustmentError(f"Duplicate logical price keys detected: {sample}")

    sort_cols = ["symbol", "date"] if "instrument_id" not in df.columns else ["instrument_id", "date"]
    df = df.sort_values(sort_cols + (["symbol"] if "instrument_id" in df.columns else []), kind="mergesort").reset_index(drop=True)
    return df


def validate_events(events: pd.DataFrame, allow_empty: bool) -> pd.DataFrame:
    if events.empty:
        if allow_empty:
            return events.copy()
        raise PriceAdjustmentError("Corporate actions input is empty and allow_empty_corporate_actions=false")

    df = events.copy()
    if "symbol" not in df.columns and "instrument_id" not in df.columns:
        raise PriceAdjustmentError("Corporate actions input must contain at least 'symbol' or 'instrument_id'")

    if "symbol" not in df.columns:
        df["symbol"] = df["instrument_id"].astype(str)
    else:
        df["symbol"] = df["symbol"].astype(str)
    if "instrument_id" in df.columns:
        df["instrument_id"] = df["instrument_id"].astype(str)

    type_col = "event_type" if "event_type" in df.columns else "action_type" if "action_type" in df.columns else None
    if type_col is None:
        raise PriceAdjustmentError("Corporate actions input must contain 'event_type' or 'action_type'")
    if "event_type" not in df.columns:
        df["event_type"] = df[type_col]

    for date_col in ["ex_date", "effective_date"]:
        if date_col not in df.columns:
            df[date_col] = pd.NaT
        df[date_col] = normalize_date_series(df[date_col])
    for ts_col in ["announcement_ts", "effective_ts_utc", "updated_ts_utc"]:
        if ts_col not in df.columns:
            df[ts_col] = pd.NaT
        df[ts_col] = normalize_timestamp_series(df[ts_col])

    for col in ["split_ratio", "cash_amount", "prev_close_t_minus_1", "adjustment_factor"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "source_provider" not in df.columns:
        df["source_provider"] = "unknown_provider"
    if "provider_priority" not in df.columns:
        df["provider_priority"] = np.nan
    else:
        df["provider_priority"] = pd.to_numeric(df["provider_priority"], errors="coerce")

    df["event_type"] = df["event_type"].map(normalize_event_type)
    app_date = df["effective_date"].where(df["effective_date"].notna(), df["ex_date"])
    if app_date.isna().any():
        raise PriceAdjustmentError("Corporate actions input must contain ex_date or effective_date for every event")
    df["application_date"] = app_date

    bad_temporal = (df["announcement_ts"].notna()) & (df["effective_ts_utc"].notna()) & (df["effective_ts_utc"] < df["announcement_ts"])
    if bad_temporal.any():
        raise PriceAdjustmentError("Corporate actions input contains effective_ts_utc earlier than announcement_ts")

    df["visible_ts_utc"] = df["effective_ts_utc"]
    missing_visible = df["visible_ts_utc"].isna()
    df.loc[missing_visible, "visible_ts_utc"] = df.loc[missing_visible, "announcement_ts"]
    missing_visible = df["visible_ts_utc"].isna()
    if missing_visible.any():
        df.loc[missing_visible, "visible_ts_utc"] = pd.to_datetime(df.loc[missing_visible, "application_date"]).dt.tz_localize("UTC")

    sort_cols = ["instrument_id", "symbol", "application_date"] if "instrument_id" in df.columns else ["symbol", "application_date"]
    df = df.sort_values(sort_cols + ["visible_ts_utc"], kind="mergesort").reset_index(drop=True)
    return df


def normalize_event_type(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "split": CanonicalEventType.SPLIT.value,
        "stock_split": CanonicalEventType.SPLIT.value,
        "reverse_split": CanonicalEventType.REVERSE_SPLIT.value,
        "reverse_stock_split": CanonicalEventType.REVERSE_SPLIT.value,
        "stock_dividend": CanonicalEventType.STOCK_DIVIDEND.value,
        "share_dividend": CanonicalEventType.STOCK_DIVIDEND.value,
        "cash_dividend": CanonicalEventType.CASH_DIVIDEND.value,
        "dividend": CanonicalEventType.CASH_DIVIDEND.value,
        "ordinary_dividend": CanonicalEventType.CASH_DIVIDEND.value,
        "cash_ordinary": CanonicalEventType.CASH_DIVIDEND.value,
        "special_cash_dividend": CanonicalEventType.SPECIAL_CASH_DIVIDEND.value,
        "special_dividend": CanonicalEventType.SPECIAL_CASH_DIVIDEND.value,
        "ticker_change": CanonicalEventType.TICKER_CHANGE.value,
        "name_change": CanonicalEventType.NAME_CHANGE.value,
        "share_class_change": CanonicalEventType.SHARE_CLASS_CHANGE.value,
        "exchange_change": CanonicalEventType.EXCHANGE_CHANGE.value,
        "merger": CanonicalEventType.MERGER.value,
        "acquisition": CanonicalEventType.ACQUISITION.value,
        "spinoff": CanonicalEventType.SPINOFF.value,
        "rights_issue": CanonicalEventType.RIGHTS_ISSUE.value,
        "tender_offer": CanonicalEventType.TENDER_OFFER.value,
        "delisting": CanonicalEventType.DELISTING.value,
        "relisting": CanonicalEventType.RELISTING.value,
        "bankruptcy_reorg": CanonicalEventType.BANKRUPTCY_REORG.value,
    }
    return mapping.get(token, token or "unknown")


# -----------------------------------------------------------------------------
# Corporate actions normalization and conflict resolution
# -----------------------------------------------------------------------------


def provider_priority_resolver(value: Any, provider: str, config: AdjustPricesConfig) -> int:
    if value is not None and not (isinstance(value, float) and np.isnan(value)):
        try:
            return int(value)
        except Exception:
            pass
    try:
        return list(config.conflict.provider_priority).index(str(provider))
    except ValueError:
        return len(config.conflict.provider_priority) + 10


def event_revision_key(row: pd.Series) -> str:
    payload = {
        "event_id": row.get("event_id"),
        "provider_event_id": row.get("provider_event_id"),
        "instrument_id": row.get("instrument_id"),
        "symbol": row.get("symbol"),
        "event_type": row.get("event_type"),
        "application_date": json_safe(row.get("application_date")),
        "source_provider": row.get("source_provider"),
    }
    return stable_hash_payload(payload, "rev")


def logical_event_key(row: pd.Series) -> str:
    payload = {
        "instrument_id": row.get("instrument_id"),
        "symbol": row.get("symbol"),
        "event_type": row.get("event_type"),
        "ex_date": json_safe(row.get("ex_date")),
        "effective_date": json_safe(row.get("effective_date")),
        "application_date": json_safe(row.get("application_date")),
    }
    return stable_hash_payload(payload, "evt")


def choose_latest_visible_revisions(events: pd.DataFrame, asof_ts_utc: str) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    asof = parse_iso_ts(asof_ts_utc)
    if asof is None:
        raise PriceAdjustmentError(f"Invalid asof_ts_utc: {asof_ts_utc}")

    visible = events[events["visible_ts_utc"] <= asof].copy()
    if visible.empty:
        return visible

    visible["revision_key"] = visible.apply(event_revision_key, axis=1)
    visible = visible.sort_values(["revision_key", "visible_ts_utc"], kind="mergesort")
    visible = visible.groupby("revision_key", as_index=False, sort=False).tail(1).reset_index(drop=True)
    return visible


def detect_material_conflicts(events: pd.DataFrame, config: AdjustPricesConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return events.copy(), pd.DataFrame(
            columns=[
                "symbol_key", "application_date", "event_type", "conflict_code", "severity", "chosen_provider",
                "providers_present", "details", "logical_event_key",
            ]
        )

    work = events.copy()
    work["provider_priority_resolved"] = [
        provider_priority_resolver(v, p, config) for v, p in zip(work["provider_priority"], work["source_provider"])
    ]
    work["logical_event_key"] = work.apply(logical_event_key, axis=1)

    winners: List[pd.Series] = []
    conflicts: List[Dict[str, Any]] = []
    group_cols = ["logical_event_key"]
    for _, grp in work.groupby(group_cols, sort=False):
        grp = grp.sort_values(["provider_priority_resolved", "visible_ts_utc"], kind="mergesort")
        chosen = grp.iloc[0]
        providers_present = grp["source_provider"].astype(str).tolist()

        split_vals = [v for v in grp["split_ratio"].tolist() if pd.notna(v)]
        cash_vals = [v for v in grp["cash_amount"].tolist() if pd.notna(v)]
        ex_dates = [json_safe(v) for v in grp["ex_date"].tolist()]
        eff_dates = [json_safe(v) for v in grp["effective_date"].tolist()]
        conflict_fields: List[str] = []

        if len(split_vals) > 1 and (max(split_vals) - min(split_vals)) > config.conflict.split_ratio_tolerance:
            conflict_fields.append("split_ratio")
        if len(cash_vals) > 1 and (max(cash_vals) - min(cash_vals)) > config.conflict.cash_amount_tolerance:
            conflict_fields.append("cash_amount")
        if len(set(ex_dates)) > 1:
            conflict_fields.append("ex_date")
        if len(set(eff_dates)) > 1:
            conflict_fields.append("effective_date")

        if conflict_fields:
            conflicts.append(
                {
                    "symbol_key": chosen.get("symbol_key"),
                    "application_date": json_safe(chosen.get("application_date")),
                    "event_type": chosen.get("event_type"),
                    "conflict_code": FailureCode.MATERIAL_CONFLICT.value,
                    "severity": Severity.FAIL.value if config.conflict.fail_on_material_conflict else Severity.WARN.value,
                    "chosen_provider": chosen.get("source_provider"),
                    "providers_present": canonical_json(providers_present),
                    "details": canonical_json(
                        {
                            "conflict_fields": conflict_fields,
                            "rows": grp[
                                [
                                    c for c in [
                                        "source_provider", "provider_event_id", "event_id", "split_ratio", "cash_amount",
                                        "ex_date", "effective_date", "visible_ts_utc"
                                    ] if c in grp.columns
                                ]
                            ].to_dict(orient="records")
                        }
                    ),
                    "logical_event_key": chosen.get("logical_event_key"),
                }
            )
        winners.append(chosen)

    resolved = pd.DataFrame(winners).reset_index(drop=True)
    conflicts_df = pd.DataFrame(conflicts)
    return resolved, conflicts_df


def resolve_events(events: pd.DataFrame, asof_ts_utc: str, config: AdjustPricesConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    visible_latest = choose_latest_visible_revisions(events, asof_ts_utc)
    if visible_latest.empty:
        return visible_latest, pd.DataFrame(
            columns=[
                "symbol_key", "application_date", "event_type", "conflict_code", "severity",
                "chosen_provider", "providers_present", "details", "logical_event_key"
            ]
        )
    resolved, conflicts = detect_material_conflicts(visible_latest, config)
    precedence = {ev: i + 1 for i, ev in enumerate(config.conflict.same_day_precedence)}
    resolved["same_day_precedence_rank"] = resolved["event_type"].map(precedence).fillna(999).astype(int)
    resolved = resolved.sort_values(
        ["symbol_key", "application_date", "same_day_precedence_rank", "provider_priority_resolved", "visible_ts_utc"],
        kind="mergesort",
    ).reset_index(drop=True)
    resolved["same_day_sequence"] = resolved.groupby(["symbol_key", "application_date"], sort=False).cumcount() + 1
    return resolved, conflicts


# -----------------------------------------------------------------------------
# Factor construction
# -----------------------------------------------------------------------------


def severity_for_policy(value: str) -> str:
    value = str(value).upper()
    if value not in SEVERITY_ORDER:
        raise PriceAdjustmentError(f"Invalid severity policy value: {value}")
    return value


def compute_dividend_factor(cash_amount: float, prev_close: float) -> Optional[float]:
    if cash_amount == 0:
        return 1.0
    if prev_close <= 0:
        return None
    if cash_amount >= prev_close:
        return None
    return 1.0 - cash_amount / prev_close


def parse_backward_split_factor(row: pd.Series) -> Optional[float]:
    ratio = row.get("split_ratio")
    if ratio is not None and not pd.isna(ratio):
        ratio = float(ratio)
        if ratio > 0:
            return ratio
        return None
    split_from = row.get("split_from")
    split_to = row.get("split_to")
    if split_from is not None and split_to is not None and not pd.isna(split_from) and not pd.isna(split_to):
        a = float(split_from)
        b = float(split_to)
        if a > 0 and b > 0:
            return b / a
    adjustment_factor = row.get("adjustment_factor")
    if adjustment_factor is not None and not pd.isna(adjustment_factor):
        adjustment_factor = float(adjustment_factor)
        if adjustment_factor > 0:
            return adjustment_factor
    return None


def prep_prices_slice(prices: pd.DataFrame, symbol_key: str) -> pd.DataFrame:
    grp = prices[prices["symbol_key"] == symbol_key].copy()
    grp = grp.sort_values("date", kind="mergesort").reset_index(drop=True)
    grp["ret_1d_raw"] = grp["close"].pct_change()
    return grp


def find_prev_close_before_date(prices: pd.DataFrame, event_date: pd.Timestamp) -> Optional[float]:
    history = prices[prices["date"] < event_date]
    if history.empty:
        return None
    return float(history.iloc[-1]["close"])


def build_event_rows_for_symbol(
    symbol_prices: pd.DataFrame,
    symbol_events: pd.DataFrame,
    config: AdjustPricesConfig,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows_events: List[Dict[str, Any]] = []
    rows_conflicts: List[Dict[str, Any]] = []

    if symbol_events.empty:
        return rows_events, rows_conflicts, []

    symbol_events = symbol_events.sort_values(
        ["application_date", "same_day_precedence_rank", "same_day_sequence"], kind="mergesort"
    ).reset_index(drop=True)

    event_date_groups: Dict[pd.Timestamp, List[Dict[str, Any]]] = {}
    symbol_failures: List[Dict[str, Any]] = []

    for _, ev in symbol_events.iterrows():
        event_type = str(ev["event_type"])
        app_date = pd.Timestamp(ev["application_date"])
        base = {
            "symbol": ev.get("symbol"),
            "instrument_id": ev.get("instrument_id"),
            "symbol_key": ev.get("symbol_key"),
            "event_type": event_type,
            "application_date": app_date,
            "ex_date": ev.get("ex_date"),
            "effective_date": ev.get("effective_date"),
            "visible_ts_utc": ev.get("visible_ts_utc"),
            "source_provider": ev.get("source_provider"),
            "provider_event_id": ev.get("provider_event_id"),
            "event_id": ev.get("event_id") or logical_event_key(ev),
            "same_day_precedence_rank": int(ev.get("same_day_precedence_rank", 999)),
            "same_day_sequence": int(ev.get("same_day_sequence", 1)),
            "split_ratio_input": None if pd.isna(ev.get("split_ratio")) else float(ev.get("split_ratio")),
            "cash_amount_input": None if pd.isna(ev.get("cash_amount")) else float(ev.get("cash_amount")),
            "prev_close_t_minus_1_input": None if pd.isna(ev.get("prev_close_t_minus_1")) else float(ev.get("prev_close_t_minus_1")),
            "quality_flags": ev.get("quality_flags"),
        }

        if app_date < start_date or app_date > end_date:
            rows_events.append(
                {
                    **base,
                    "outcome": EventOutcome.SKIPPED_OUTSIDE_RANGE.value,
                    "severity": Severity.INFO.value,
                    "failure_code": None,
                    "failure_detail": "Event application_date outside requested processing range",
                    "split_factor_elementary": None,
                    "dividend_factor_elementary": None,
                    "prev_close_for_dividend": None,
                    "include_in_split_chain": False,
                    "include_in_dividend_chain": False,
                }
            )
            continue

        if event_type not in SUPPORTED_EVENT_TYPES:
            sev = severity_for_policy(config.adjustment.unsupported_event_severity)
            rows_events.append(
                {
                    **base,
                    "outcome": EventOutcome.SKIPPED_UNSUPPORTED.value,
                    "severity": sev,
                    "failure_code": FailureCode.UNSUPPORTED_EVENT_TYPE.value,
                    "failure_detail": f"Unsupported event type for adjust_prices: {event_type}",
                    "split_factor_elementary": None,
                    "dividend_factor_elementary": None,
                    "prev_close_for_dividend": None,
                    "include_in_split_chain": False,
                    "include_in_dividend_chain": False,
                }
            )
            continue

        if event_type == CanonicalEventType.SPECIAL_CASH_DIVIDEND.value and not config.adjustment.include_special_cash_dividends:
            rows_events.append(
                {
                    **base,
                    "outcome": EventOutcome.SKIPPED_SPECIAL_DIVIDEND_EXCLUDED.value,
                    "severity": Severity.WARN.value,
                    "failure_code": None,
                    "failure_detail": "Special cash dividend excluded by policy",
                    "split_factor_elementary": None,
                    "dividend_factor_elementary": None,
                    "prev_close_for_dividend": None,
                    "include_in_split_chain": False,
                    "include_in_dividend_chain": False,
                }
            )
            continue

        split_factor = None
        dividend_factor = None
        prev_close = None
        outcome = EventOutcome.APPLIED.value
        severity = Severity.INFO.value
        failure_code = None
        failure_detail = None
        include_split = False
        include_div = False

        if event_type in {
            CanonicalEventType.SPLIT.value,
            CanonicalEventType.REVERSE_SPLIT.value,
            CanonicalEventType.STOCK_DIVIDEND.value,
        }:
            split_factor = parse_backward_split_factor(ev)
            if split_factor is None or split_factor <= 0:
                outcome = EventOutcome.SKIPPED_MISSING_FIELDS.value
                severity = Severity.FAIL.value
                failure_code = FailureCode.NON_POSITIVE_FACTOR.value
                failure_detail = "Mechanical event is missing a valid positive backward split factor"
            else:
                include_split = True
                if split_factor > config.reverse_split.extreme_threshold:
                    severity = Severity.WARN.value

        elif event_type in {CanonicalEventType.CASH_DIVIDEND.value, CanonicalEventType.SPECIAL_CASH_DIVIDEND.value}:
            cash_amount = ev.get("cash_amount")
            if cash_amount is None or pd.isna(cash_amount):
                outcome = EventOutcome.SKIPPED_MISSING_FIELDS.value
                severity = Severity.FAIL.value
                failure_code = FailureCode.MISSING_REQUIRED_COLUMN.value
                failure_detail = "Cash dividend requires cash_amount"
            else:
                prev_close = find_prev_close_before_date(symbol_prices, app_date)
                if prev_close is None:
                    outcome = EventOutcome.SKIPPED_MISSING_PREV_CLOSE.value
                    severity = severity_for_policy(config.adjustment.missing_prev_close_severity)
                    failure_code = FailureCode.MISSING_PREV_CLOSE_FOR_DIVIDEND.value
                    failure_detail = "No prior raw close exists before dividend ex-date"
                else:
                    dividend_factor = compute_dividend_factor(float(cash_amount), float(prev_close))
                    if dividend_factor is None or dividend_factor <= 0:
                        outcome = EventOutcome.SKIPPED_INVALID_DIVIDEND_FACTOR.value
                        severity = severity_for_policy(config.adjustment.invalid_dividend_severity)
                        failure_code = FailureCode.INVALID_DIVIDEND_FACTOR.value
                        failure_detail = f"Invalid dividend factor for cash_amount={float(cash_amount)} and prev_close={float(prev_close)}"
                    else:
                        include_div = True

        event_row = {
            **base,
            "outcome": outcome,
            "severity": severity,
            "failure_code": failure_code,
            "failure_detail": failure_detail,
            "split_factor_elementary": None if split_factor is None else float(split_factor),
            "dividend_factor_elementary": None if dividend_factor is None else float(dividend_factor),
            "prev_close_for_dividend": None if prev_close is None else float(prev_close),
            "include_in_split_chain": include_split,
            "include_in_dividend_chain": include_div,
        }
        rows_events.append(event_row)
        if failure_code is not None:
            symbol_failures.append(event_row)
        if include_split or include_div or outcome == EventOutcome.APPLIED.value:
            event_date_groups.setdefault(app_date, []).append(event_row)

    applied_events = [row for row in rows_events if row["outcome"] == EventOutcome.APPLIED.value]
    return rows_events, rows_conflicts, applied_events


def build_adjusted_rows_for_symbol(
    symbol_prices: pd.DataFrame,
    event_rows: List[Dict[str, Any]],
    mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prices = symbol_prices.sort_values("date", kind="mergesort").reset_index(drop=True)
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame()

    events_by_date: Dict[pd.Timestamp, List[Dict[str, Any]]] = {}
    for row in event_rows:
        events_by_date.setdefault(pd.Timestamp(row["application_date"]), []).append(row)

    split_cum = 1.0
    div_cum = 1.0
    applied_count = 0
    split_count = 0
    div_count = 0
    sev_state = Severity.INFO.value
    unsupported_state = False

    adjusted_rows: List[Dict[str, Any]] = []
    factor_rows: List[Dict[str, Any]] = []

    for _, px in prices.sort_values("date", ascending=False, kind="mergesort").iterrows():
        date = pd.Timestamp(px["date"])
        combined_factor = split_cum * div_cum

        row = {
            "symbol": px.get("symbol"),
            "date": date,
            "run_id": None,
            "adjustment_mode": mode,
            "asof_ts_utc": None,
            "open_raw": float(px["open"]),
            "high_raw": float(px["high"]),
            "low_raw": float(px["low"]),
            "close_raw": float(px["close"]),
            "volume_raw": float(px["volume"]),
            "split_factor_cum": float(split_cum),
            "dividend_factor_cum": float(div_cum),
            "volume_adj": float(px["volume"]) / float(split_cum),
            "event_count_applied": int(applied_count),
            "has_unsupported_events": bool(unsupported_state),
            "severity_max": sev_state,
        }
        if "instrument_id" in px.index:
            row["instrument_id"] = px.get("instrument_id")

        row["open_adj_split"] = float(px["open"]) * float(split_cum)
        row["high_adj_split"] = float(px["high"]) * float(split_cum)
        row["low_adj_split"] = float(px["low"]) * float(split_cum)
        row["close_adj_split"] = float(px["close"]) * float(split_cum)

        if mode in {AdjustmentMode.TOTAL_RETURN_LIKE.value, AdjustmentMode.DUAL_OUTPUT.value}:
            row["open_adj_total"] = float(px["open"]) * float(combined_factor)
            row["high_adj_total"] = float(px["high"]) * float(combined_factor)
            row["low_adj_total"] = float(px["low"]) * float(combined_factor)
            row["close_adj_total"] = float(px["close"]) * float(combined_factor)

        factor_rows.append(
            {
                "symbol": px.get("symbol"),
                "date": date,
                "instrument_id": px.get("instrument_id") if "instrument_id" in px.index else None,
                "split_factor_cum": float(split_cum),
                "dividend_factor_cum": float(div_cum),
                "combined_factor_cum": float(combined_factor),
                "volume_factor_cum": float(split_cum),
                "event_count_applied": int(applied_count),
                "split_event_count_applied": int(split_count),
                "dividend_event_count_applied": int(div_count),
                "severity_max": sev_state,
                "has_unsupported_events": bool(unsupported_state),
            }
        )
        adjusted_rows.append(row)

        for ev in sorted(events_by_date.get(date, []), key=lambda x: (x["same_day_precedence_rank"], x["same_day_sequence"])):
            sev_state = severity_max([sev_state, ev["severity"]])
            if ev["outcome"] != EventOutcome.APPLIED.value and ev["outcome"] != EventOutcome.SKIPPED_UNSUPPORTED.value:
                unsupported_state = unsupported_state or (ev["outcome"] == EventOutcome.SKIPPED_UNSUPPORTED.value)
            if ev["outcome"] == EventOutcome.SKIPPED_UNSUPPORTED.value:
                unsupported_state = True
                continue
            if ev["include_in_split_chain"]:
                split_factor = float(ev["split_factor_elementary"])
                split_cum *= split_factor
                applied_count += 1
                split_count += 1
            if ev["include_in_dividend_chain"]:
                div_factor = float(ev["dividend_factor_elementary"])
                div_cum *= div_factor
                applied_count += 1
                div_count += 1

            if split_cum <= 0 or div_cum <= 0:
                raise PriceAdjustmentError(f"Non-positive cumulative factor detected for symbol={px.get('symbol')} on date={date}")

    adjusted = pd.DataFrame(adjusted_rows).sort_values("date", kind="mergesort").reset_index(drop=True)
    factors = pd.DataFrame(factor_rows).sort_values("date", kind="mergesort").reset_index(drop=True)
    adjusted["ret_1d_raw"] = adjusted["close_raw"].pct_change()
    adjusted["ret_1d_adj_split"] = adjusted["close_adj_split"].pct_change()
    if mode in {AdjustmentMode.TOTAL_RETURN_LIKE.value, AdjustmentMode.DUAL_OUTPUT.value}:
        adjusted["ret_1d_adj_total"] = adjusted["close_adj_total"].pct_change()
    return adjusted, factors


# -----------------------------------------------------------------------------
# Invariants and orchestration
# -----------------------------------------------------------------------------


def validate_adjusted_invariants(adjusted: pd.DataFrame, mode: str) -> pd.DataFrame:
    if adjusted.empty:
        return adjusted

    work = adjusted.copy().sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    grouped = work.groupby("symbol", sort=False)

    if grouped["date"].apply(lambda s: s.is_monotonic_increasing and s.is_unique).eq(False).any():
        raise PriceAdjustmentError("Adjusted output violates strictly increasing unique dates per symbol")

    numeric_positive_cols = ["split_factor_cum", "dividend_factor_cum", "open_adj_split", "high_adj_split", "low_adj_split", "close_adj_split"]
    if mode in {AdjustmentMode.TOTAL_RETURN_LIKE.value, AdjustmentMode.DUAL_OUTPUT.value}:
        numeric_positive_cols.extend(["open_adj_total", "high_adj_total", "low_adj_total", "close_adj_total"])

    for col in numeric_positive_cols:
        if (work[col] <= 0).any():
            raise PriceAdjustmentError(f"Adjusted output contains non-positive values in column: {col}")

    if (work["volume_adj"] < 0).any():
        raise PriceAdjustmentError("Adjusted output contains negative volume_adj")

    ok_split_geom = (
        (work["low_adj_split"] <= work[["open_adj_split", "close_adj_split"]].min(axis=1))
        & (work[["open_adj_split", "close_adj_split"]].max(axis=1) <= work["high_adj_split"])
    )
    if not ok_split_geom.all():
        raise PriceAdjustmentError("Adjusted split-only OHLC geometry invariant failed")

    if mode in {AdjustmentMode.TOTAL_RETURN_LIKE.value, AdjustmentMode.DUAL_OUTPUT.value}:
        ok_total_geom = (
            (work["low_adj_total"] <= work[["open_adj_total", "close_adj_total"]].min(axis=1))
            & (work[["open_adj_total", "close_adj_total"]].max(axis=1) <= work["high_adj_total"])
        )
        if not ok_total_geom.all():
            raise PriceAdjustmentError("Adjusted total-return-like OHLC geometry invariant failed")

    return work


def build_summary(
    adjusted: pd.DataFrame,
    events_applied: pd.DataFrame,
    conflicts: pd.DataFrame,
    mode: str,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    symbol_count = int(adjusted["symbol"].nunique()) if not adjusted.empty else 0
    event_warn_count = int((events_applied["severity"] == Severity.WARN.value).sum()) if ("severity" in events_applied.columns and not events_applied.empty) else 0
    event_fail_count = int((events_applied["severity"] == Severity.FAIL.value).sum()) if ("severity" in events_applied.columns and not events_applied.empty) else 0
    conflict_warn_count = int((conflicts["severity"] == Severity.WARN.value).sum()) if ("severity" in conflicts.columns and not conflicts.empty) else 0
    conflict_fail_count = int((conflicts["severity"] == Severity.FAIL.value).sum()) if ("severity" in conflicts.columns and not conflicts.empty) else 0

    out = {
        "mode": mode,
        "start_date": start_date,
        "end_date": end_date,
        "symbol_count": symbol_count,
        "row_count": int(len(adjusted)),
        "event_rows_total": int(len(events_applied)),
        "event_rows_applied": int((events_applied["outcome"] == EventOutcome.APPLIED.value).sum()) if ("outcome" in events_applied.columns and not events_applied.empty) else 0,
        "unsupported_event_rows": int((events_applied["outcome"] == EventOutcome.SKIPPED_UNSUPPORTED.value).sum()) if ("outcome" in events_applied.columns and not events_applied.empty) else 0,
        "invalid_dividend_rows": int((events_applied["failure_code"] == FailureCode.INVALID_DIVIDEND_FACTOR.value).sum()) if ("failure_code" in events_applied.columns and not events_applied.empty) else 0,
        "missing_prev_close_rows": int((events_applied["failure_code"] == FailureCode.MISSING_PREV_CLOSE_FOR_DIVIDEND.value).sum()) if ("failure_code" in events_applied.columns and not events_applied.empty) else 0,
        "conflict_count": int(len(conflicts)),
        "warn_count": event_warn_count + conflict_warn_count,
        "fail_count": event_fail_count + conflict_fail_count,
        "symbols_with_events": int(events_applied["symbol"].nunique()) if ("symbol" in events_applied.columns and not events_applied.empty) else 0,
    }
    if not adjusted.empty:
        out["coverage_by_symbol"] = adjusted.groupby("symbol", sort=False).size().to_dict()
    else:
        out["coverage_by_symbol"] = {}
    if not events_applied.empty:
        out["events_by_type"] = events_applied["event_type"].value_counts(dropna=False).to_dict()
        out["events_by_outcome"] = events_applied["outcome"].value_counts(dropna=False).to_dict()
    else:
        out["events_by_type"] = {}
        out["events_by_outcome"] = {}
    return out


def build_manifest(
    run_id: str,
    asof_ts_utc: str,
    mode: str,
    prices_path: Path,
    events_path: Path,
    config: AdjustPricesConfig,
    adjusted: pd.DataFrame,
    events_applied: pd.DataFrame,
    conflicts: pd.DataFrame,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "code_version": maybe_import_git_revision(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "module": "data/price/adjust_prices.py",
        "input_prices_snapshot": str(prices_path),
        "input_corporate_actions_snapshot": str(events_path),
        "input_corporate_actions_hash": stable_hash_payload(events_applied.to_dict(orient="records"), "ca_snapshot"),
        "adjustment_mode": mode,
        "asof_ts_utc": asof_ts_utc,
        "symbol_count": int(adjusted["symbol"].nunique()) if not adjusted.empty else 0,
        "row_count": int(len(adjusted)),
        "fail_count": summary["fail_count"],
        "warn_count": summary["warn_count"],
        "config": json_safe(dataclasses.asdict(config)),
        "artifacts": {
            "adjusted": "adjusted_prices.parquet",
            "factors": "adjustment_factors.parquet",
            "events_applied": "adjustment_events_applied.parquet",
            "conflicts": "adjustment_conflicts.parquet",
            "manifest": f"adjustment_manifest_{run_id}.json",
            "summary": f"adjustment_summary_{run_id}.json",
        },
    }


def adjust_prices(
    prices_raw: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    *,
    run_id: str,
    asof_ts_utc: str,
    start_date: str,
    end_date: str,
    config: AdjustPricesConfig,
) -> AdjustmentArtifacts:
    mode = config.adjustment.mode
    if mode not in {m.value for m in AdjustmentMode}:
        raise PriceAdjustmentError(f"Invalid adjustment mode: {mode}")

    prices = validate_prices(prices_raw)
    prices["symbol_key"] = build_symbol_key(prices)

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise PriceAdjustmentError("start_date and end_date must be parseable dates")
    if start_ts > end_ts:
        raise PriceAdjustmentError("start_date cannot be after end_date")

    prices = prices[(prices["date"] >= start_ts) & (prices["date"] <= end_ts)].copy()
    if prices.empty:
        raise PriceAdjustmentError("No raw prices remain after applying requested date range")

    events = validate_events(corporate_actions, config.adjustment.allow_empty_corporate_actions)
    if not events.empty:
        events["symbol_key"] = build_symbol_key(events)
        resolved_events, conflicts = resolve_events(events, asof_ts_utc, config)
    else:
        resolved_events = pd.DataFrame()
        conflicts = pd.DataFrame(columns=["symbol_key", "application_date", "event_type", "conflict_code", "severity", "chosen_provider", "providers_present", "details", "logical_event_key"])

    adjusted_parts: List[pd.DataFrame] = []
    factor_parts: List[pd.DataFrame] = []
    event_rows_all: List[Dict[str, Any]] = []

    for symbol_key, symbol_prices in prices.groupby("symbol_key", sort=False):
        symbol_events = resolved_events[resolved_events["symbol_key"] == symbol_key].copy() if not resolved_events.empty else pd.DataFrame()
        event_rows, _rows_conf_unused, applied_events = build_event_rows_for_symbol(symbol_prices.sort_values("date"), symbol_events, config, start_ts, end_ts)
        event_rows_all.extend(event_rows)
        adjusted_sym, factors_sym = build_adjusted_rows_for_symbol(symbol_prices, event_rows, mode)
        if not adjusted_sym.empty:
            adjusted_sym["run_id"] = run_id
            adjusted_sym["asof_ts_utc"] = asof_ts_utc
            adjusted_parts.append(adjusted_sym)
        if not factors_sym.empty:
            factor_parts.append(factors_sym)

    adjusted = pd.concat(adjusted_parts, ignore_index=True) if adjusted_parts else pd.DataFrame()
    factors = pd.concat(factor_parts, ignore_index=True) if factor_parts else pd.DataFrame()
    events_applied = pd.DataFrame(event_rows_all)

    if adjusted.empty and not prices.empty:
        raise PriceAdjustmentError("Adjustment produced no output rows")

    adjusted = validate_adjusted_invariants(adjusted, mode)
    adjusted = adjusted.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    factors = factors.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True) if not factors.empty else factors
    if not events_applied.empty:
        events_applied = events_applied.sort_values(
            ["symbol", "application_date", "same_day_precedence_rank", "same_day_sequence"], kind="mergesort"
        ).reset_index(drop=True)

    summary = build_summary(adjusted, events_applied, conflicts, mode, str(start_ts.date()), str(end_ts.date()))
    manifest = {}  # created in wrapper with paths
    return AdjustmentArtifacts(
        adjusted=adjusted.reset_index(drop=True),
        factors=factors.reset_index(drop=True),
        events_applied=events_applied.reset_index(drop=True),
        conflicts=conflicts.reset_index(drop=True) if not conflicts.empty else conflicts,
        summary=summary,
        manifest=manifest,
    )


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------


def write_outputs(outdir: Path, artifacts: AdjustmentArtifacts, run_id: str, config: AdjustPricesConfig) -> None:
    ensure_parquet_support()
    outdir.mkdir(parents=True, exist_ok=True)
    compression = config.output.compression

    artifacts.adjusted.to_parquet(outdir / "adjusted_prices.parquet", index=False, compression=compression)
    artifacts.factors.to_parquet(outdir / "adjustment_factors.parquet", index=False, compression=compression)
    artifacts.events_applied.to_parquet(outdir / "adjustment_events_applied.parquet", index=False, compression=compression)
    if artifacts.conflicts.empty:
        pd.DataFrame(
            columns=["symbol_key", "application_date", "event_type", "conflict_code", "severity", "chosen_provider", "providers_present", "details", "logical_event_key"]
        ).to_parquet(outdir / "adjustment_conflicts.parquet", index=False, compression=compression)
    else:
        artifacts.conflicts.to_parquet(outdir / "adjustment_conflicts.parquet", index=False, compression=compression)

    (outdir / f"adjustment_summary_{run_id}.json").write_text(
        json.dumps(json_safe(artifacts.summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (outdir / f"adjustment_manifest_{run_id}.json").write_text(
        json.dumps(json_safe(artifacts.manifest), indent=2, ensure_ascii=False), encoding="utf-8"
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PIT-safe OHLCV adjustment by visible corporate actions with split-only / total-return-like / dual output modes."
    )
    parser.add_argument("--prices-raw-path", required=True, help="Path to raw daily OHLCV prices (parquet/csv)")
    parser.add_argument("--corporate-actions-path", required=True, help="Path to PIT corporate actions snapshot (parquet/csv)")
    parser.add_argument("--outdir", required=True, help="Output directory for adjusted artifacts")
    parser.add_argument("--run-id", required=True, help="Stable run identifier")
    parser.add_argument("--asof-ts-utc", required=True, help="Logical PIT timestamp in UTC")
    parser.add_argument("--start-date", required=True, help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="Inclusive end date (YYYY-MM-DD)")
    parser.add_argument(
        "--adjust-mode",
        choices=[m.value for m in AdjustmentMode],
        default=None,
        help="Adjustment mode override: split_only | total_return_like | dual_output",
    )
    parser.add_argument("--config", default=None, help="Optional JSON/YAML config path")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    prices_path = Path(args.prices_raw_path)
    events_path = Path(args.corporate_actions_path)
    outdir = Path(args.outdir)
    config = load_config(Path(args.config) if args.config else None, cli_mode=args.adjust_mode)

    prices = load_table(prices_path)
    events = load_table(events_path)

    artifacts = adjust_prices(
        prices,
        events,
        run_id=args.run_id,
        asof_ts_utc=args.asof_ts_utc,
        start_date=args.start_date,
        end_date=args.end_date,
        config=config,
    )
    artifacts.manifest = build_manifest(
        run_id=args.run_id,
        asof_ts_utc=args.asof_ts_utc,
        mode=config.adjustment.mode,
        prices_path=prices_path,
        events_path=events_path,
        config=config,
        adjusted=artifacts.adjusted,
        events_applied=artifacts.events_applied,
        conflicts=artifacts.conflicts,
        summary=artifacts.summary,
    )
    write_outputs(outdir, artifacts, args.run_id, config)

    logging.info(
        "Adjusted prices complete | mode=%s | symbols=%s | rows=%s | events=%s | conflicts=%s",
        config.adjustment.mode,
        artifacts.summary["symbol_count"],
        artifacts.summary["row_count"],
        artifacts.summary["event_rows_total"],
        artifacts.summary["conflict_count"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
