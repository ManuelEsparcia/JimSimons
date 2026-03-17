from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import logging
import os
import platform
import subprocess
import sys
import textwrap
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
# Enums and constants
# -----------------------------------------------------------------------------


class Severity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    FAIL = "FAIL"


class ExclusionReason(str, Enum):
    NOT_YET_LISTED = "NOT_YET_LISTED"
    DELISTED = "DELISTED"
    BAD_SECURITY_TYPE = "BAD_SECURITY_TYPE"
    BAD_EXCHANGE = "BAD_EXCHANGE"
    HALTED = "HALTED"
    SUSPENDED = "SUSPENDED"
    TOO_YOUNG_SINCE_LISTING = "TOO_YOUNG_SINCE_LISTING"
    PRICE_BELOW_MIN = "PRICE_BELOW_MIN"
    ADV20_BELOW_MIN = "ADV20_BELOW_MIN"
    MCAP_OUT_OF_BAND = "MCAP_OUT_OF_BAND"
    MISSING_PRICE = "MISSING_PRICE"
    MISSING_ADV20 = "MISSING_ADV20"
    MISSING_MCAP = "MISSING_MCAP"
    MISSING_SECURITY_TYPE = "MISSING_SECURITY_TYPE"
    MISSING_EXCHANGE = "MISSING_EXCHANGE"
    MISSING_LIST_DATE = "MISSING_LIST_DATE"
    SPECIAL_RULE_EXCLUSION = "SPECIAL_RULE_EXCLUSION"


class MembershipState(str, Enum):
    ELIGIBLE = "eligible"
    INELIGIBLE_RULE = "ineligible_rule"
    INELIGIBLE_MISSING = "ineligible_missing"
    INELIGIBLE_HALTED = "ineligible_halted"
    INELIGIBLE_SUSPENDED = "ineligible_suspended"
    INACTIVE_POST_DEATH = "inactive_post_death"


class Transition(str, Enum):
    ENTER = "ENTER"
    EXIT = "EXIT"
    STAY_IN = "STAY_IN"
    STAY_OUT = "STAY_OUT"
    STATE_CHANGE_OUTSIDE_UNIVERSE = "STATE_CHANGE_OUTSIDE_UNIVERSE"


TEMPORAL_CONVENTION = {
    "decision_time": "close(t-1)",
    "execution_date": "t",
}

REQUIRED_LISTING_COLUMNS = {
    "instrument_id",
    "issuer_id",
    "symbol",
    "listing_exchange",
    "security_type",
    "share_class",
    "list_date",
    "delist_date",
    "trading_status",
    "is_primary_listing",
}

REQUIRED_FEATURE_COLUMNS = {
    "date",
    "instrument_id",
    "price_ref",
    "adv20_usd",
    "market_cap_usd",
}

OPTIONAL_PIT_INTERVAL_COLS = ["effective_from", "effective_to"]
OPTIONAL_METADATA_COLUMNS = ["field_source_flags"]


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


class UniverseBuildError(RuntimeError):
    """Raised when a hard validation or invariant fails."""


# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class UniverseRules:
    allowed_security_types: Tuple[str, ...] = ("COMMON_STOCK", "ORDINARY_SHARE")
    allowed_exchanges: Tuple[str, ...] = ("NYSE", "NASDAQ", "NYSE_ARCA", "NYSE_AMERICAN")
    allowed_trading_statuses: Tuple[str, ...] = ("ACTIVE", "TRADABLE")
    halted_statuses: Tuple[str, ...] = ("HALTED",)
    suspended_statuses: Tuple[str, ...] = ("SUSPENDED",)
    min_listing_age_days: int = 20
    min_price_ref: float = 3.0
    min_adv20_usd: float = 1_000_000.0
    min_market_cap_usd: float = 150_000_000.0
    max_market_cap_usd: float = 5_000_000_000.0
    require_primary_listing: bool = True
    allowed_share_classes: Tuple[str, ...] = tuple()
    excluded_share_classes: Tuple[str, ...] = tuple()
    excluded_instrument_ids: Tuple[str, ...] = tuple()
    excluded_symbols: Tuple[str, ...] = tuple()


@dataclass(frozen=True)
class MissingDataPolicy:
    mode: str = "exclude"  # exclude | forward_fill
    max_staleness_days: int = 0
    forward_fill_fields: Tuple[str, ...] = tuple()


@dataclass(frozen=True)
class EdgeCasePolicy:
    halt_policy: str = "exclude_until_tradable"
    suspension_policy: str = "exclude_until_tradable"
    relisting_policy: str = "new_instrument_unless_proven_same"


@dataclass(frozen=True)
class QCThresholds:
    max_daily_constituent_jump_fraction: float = 0.25
    max_turnover_warn: float = 0.35
    max_critical_missing_fraction_warn: float = 0.05
    max_top_reason_fraction_warn: float = 0.65


@dataclass(frozen=True)
class OutputConfig:
    output_dir: str = "data/universe/output"
    save_daily_constituents_for_end_date_only: bool = True
    compression: str = "snappy"


@dataclass(frozen=True)
class UniverseConfig:
    rules: UniverseRules = field(default_factory=UniverseRules)
    missing_data: MissingDataPolicy = field(default_factory=MissingDataPolicy)
    edge_cases: EdgeCasePolicy = field(default_factory=EdgeCasePolicy)
    qc: QCThresholds = field(default_factory=QCThresholds)
    output: OutputConfig = field(default_factory=OutputConfig)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.exists():
        raise UniverseBuildError(f"Input path does not exist: {path}")
    return path


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


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
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def parse_date(value: Any, field_name: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=False, errors="coerce")
    if pd.isna(ts):
        raise UniverseBuildError(f"Could not parse {field_name} as date: {value!r}")
    if isinstance(ts, pd.DatetimeIndex):  # defensive
        ts = ts[0]
    return pd.Timestamp(ts).normalize()


def normalize_optional_date(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, utc=False, errors="coerce")
    return pd.Series(pd.DatetimeIndex(out).normalize(), index=series.index)


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        1: True,
        0: False,
        True: True,
        False: False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
    }
    return series.map(lambda x: mapping.get(str(x).strip().lower(), False) if pd.notna(x) else False)


def listify_reasons(values: Sequence[str]) -> Optional[str]:
    vals = [v for v in values if v]
    return "|".join(vals) if vals else None


def severity_rank(sev: Severity) -> int:
    return {Severity.INFO: 0, Severity.WARN: 1, Severity.FAIL: 2}[sev]


def max_severity(severities: Iterable[Severity]) -> Severity:
    current = Severity.INFO
    for sev in severities:
        if severity_rank(sev) > severity_rank(current):
            current = sev
    return current


# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------


def _deep_merge_dict(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            _deep_merge_dict(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def _dataclass_from_dict(dc_type: Any, data: Mapping[str, Any]) -> Any:
    kwargs: Dict[str, Any] = {}
    for field_def in dataclasses.fields(dc_type):
        if field_def.name not in data:
            continue
        value = data[field_def.name]
        field_type = field_def.type
        if dataclasses.is_dataclass(field_def.default) or dataclasses.is_dataclass(field_type):
            kwargs[field_def.name] = _dataclass_from_dict(field_type, value)
        elif hasattr(field_type, "__dataclass_fields__"):
            kwargs[field_def.name] = _dataclass_from_dict(field_type, value)
        elif isinstance(value, list) and isinstance(getattr(dc_type, field_def.name, None), tuple):
            kwargs[field_def.name] = tuple(value)
        else:
            kwargs[field_def.name] = value
    return dc_type(**kwargs)


def load_config(config_path: str | Path) -> Tuple[UniverseConfig, Dict[str, Any], str]:
    path = ensure_path(config_path)
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as fh:
        raw_text = fh.read()

    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise UniverseBuildError("PyYAML is required to load YAML config files.")
        payload = yaml.safe_load(raw_text) or {}
    elif suffix == ".json":
        payload = json.loads(raw_text)
    else:
        raise UniverseBuildError(f"Unsupported config format: {path.suffix}")

    if not isinstance(payload, Mapping):
        raise UniverseBuildError("Config file must deserialize to a mapping/object.")

    defaults = dataclasses.asdict(UniverseConfig())
    merged = _deep_merge_dict(defaults, dict(payload))
    config = UniverseConfig(
        rules=UniverseRules(**merged.get("rules", {})),
        missing_data=MissingDataPolicy(**merged.get("missing_data", {})),
        edge_cases=EdgeCasePolicy(**merged.get("edge_cases", {})),
        qc=QCThresholds(**merged.get("qc", {})),
        output=OutputConfig(**merged.get("output", {})),
    )
    config_hash = sha256_text(canonical_json(dataclasses.asdict(config)))
    return config, merged, config_hash


# -----------------------------------------------------------------------------
# IO loading
# -----------------------------------------------------------------------------




def ensure_parquet_engine_available() -> None:
    if importlib.util.find_spec("pyarrow") is None and importlib.util.find_spec("fastparquet") is None:
        raise UniverseBuildError(
            "Parquet support requires either 'pyarrow' or 'fastparquet'. "
            "Install one of them before running build_universe."
        )


def read_dataframe(path_like: str | Path) -> pd.DataFrame:
    path = ensure_path(path_like)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        ensure_parquet_engine_available()
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise UniverseBuildError(f"Unsupported input tabular format: {path}")


def load_listing_master(path_like: str | Path) -> pd.DataFrame:
    df = read_dataframe(path_like)
    missing = REQUIRED_LISTING_COLUMNS - set(df.columns)
    if missing:
        raise UniverseBuildError(f"listing_master missing required columns: {sorted(missing)}")

    df = df.copy()
    df["list_date"] = normalize_optional_date(df["list_date"])
    df["delist_date"] = normalize_optional_date(df["delist_date"])

    if df["list_date"].isna().any():
        bad = int(df["list_date"].isna().sum())
        raise UniverseBuildError(f"listing_master has {bad} rows with non-parseable list_date")

    if "effective_from" in df.columns:
        df["effective_from"] = normalize_optional_date(df["effective_from"])
    if "effective_to" in df.columns:
        df["effective_to"] = normalize_optional_date(df["effective_to"])

    df["instrument_id"] = df["instrument_id"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    df["listing_exchange"] = df["listing_exchange"].astype(str)
    df["security_type"] = df["security_type"].astype(str)
    df["share_class"] = df["share_class"].astype(str)
    df["trading_status"] = df["trading_status"].astype(str)
    df["is_primary_listing"] = as_bool(df["is_primary_listing"])

    if "effective_from" in df.columns and "effective_to" in df.columns:
        unresolved = (
            df.groupby(["instrument_id", "effective_from", "effective_to"], dropna=False)
            .size()
            .reset_index(name="n")
        )
        if (unresolved["n"] > 1).any():
            raise UniverseBuildError(
                "listing_master contains duplicate PIT rows with identical instrument/effective interval."
            )
    else:
        dup = df.duplicated(subset=["instrument_id"], keep=False)
        if dup.any():
            # Multiple rows per instrument without explicit PIT interval are ambiguous and unsafe.
            raise UniverseBuildError(
                "listing_master has duplicated instrument_id without PIT interval columns effective_from/effective_to."
            )

    return df


def load_calendar(path_like: str | Path, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    df = read_dataframe(path_like)
    if "date" not in df.columns:
        if len(df.columns) == 1:
            df = df.rename(columns={df.columns[0]: "date"})
        else:
            raise UniverseBuildError("calendar must contain a 'date' column")
    df = df.copy()
    df["date"] = normalize_optional_date(df["date"])
    if df["date"].isna().any():
        raise UniverseBuildError("calendar contains non-parseable dates")
    df = df[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].reset_index(drop=True)
    if df.empty:
        raise UniverseBuildError("calendar has no sessions inside requested date range")
    return df


def load_features_asof(path_like: str | Path) -> pd.DataFrame:
    path = ensure_path(path_like)
    if path.is_dir():
        ensure_parquet_engine_available()
        parts = sorted(p for p in path.glob("*.parquet") if p.is_file())
        if not parts:
            raise UniverseBuildError(f"features_asof directory has no parquet parts: {path}")
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    else:
        df = read_dataframe(path)

    missing = REQUIRED_FEATURE_COLUMNS - set(df.columns)
    if missing:
        raise UniverseBuildError(f"features_asof missing required columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = normalize_optional_date(df["date"])
    if df["date"].isna().any():
        raise UniverseBuildError("features_asof contains non-parseable dates")
    df["instrument_id"] = df["instrument_id"].astype(str)

    dup = df.duplicated(subset=["date", "instrument_id"], keep=False)
    if dup.any():
        sample = df.loc[dup, ["date", "instrument_id"]].head(10)
        raise UniverseBuildError(
            f"features_asof violates uniqueness on (date, instrument_id). Sample duplicates:\n{sample}"
        )

    if "field_source_flags" not in df.columns:
        df["field_source_flags"] = "{}"
    else:
        df["field_source_flags"] = df["field_source_flags"].fillna("{}")

    return df


# -----------------------------------------------------------------------------
# PIT identity resolution and panel construction
# -----------------------------------------------------------------------------


def _resolve_static_listing_attributes(listing_master: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "instrument_id",
        "issuer_id",
        "symbol",
        "listing_exchange",
        "security_type",
        "share_class",
        "list_date",
        "delist_date",
        "trading_status",
        "is_primary_listing",
    ]
    return listing_master[cols].copy()


def _build_life_windows(
    listing_master: pd.DataFrame,
    sessions: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    sessions = sessions.sort_values().reset_index(drop=True)
    frames: List[pd.DataFrame] = []

    static_lm = _resolve_static_listing_attributes(listing_master)

    for row in static_lm.itertuples(index=False):
        tau_list = pd.Timestamp(row.list_date)
        if pd.isna(tau_list):
            continue
        tau_death = pd.Timestamp(row.delist_date) if pd.notna(row.delist_date) else end_date
        life_start = max(tau_list, start_date)
        life_end = min(tau_death, end_date)
        if life_end < life_start:
            continue
        life_sessions = sessions[(sessions >= life_start) & (sessions <= life_end)]
        if life_sessions.empty:
            continue
        n = len(life_sessions)
        frames.append(
            pd.DataFrame(
                {
                    "date": life_sessions.values,
                    "instrument_id": [row.instrument_id] * n,
                    "issuer_id": [row.issuer_id] * n,
                    "symbol": [row.symbol] * n,
                    "listing_exchange": [row.listing_exchange] * n,
                    "security_type": [row.security_type] * n,
                    "share_class": [row.share_class] * n,
                    "list_date": [tau_list] * n,
                    "delist_date": [pd.Timestamp(row.delist_date) if pd.notna(row.delist_date) else pd.NaT] * n,
                    "trading_status": [row.trading_status] * n,
                    "is_primary_listing": [bool(row.is_primary_listing)] * n,
                }
            )
        )

    if not frames:
        raise UniverseBuildError("No evaluable instrument life windows intersect the requested date range.")

    panel = pd.concat(frames, ignore_index=True)

    # If PIT intervals exist, override static attributes with the correct effective slice.
    if all(col in listing_master.columns for col in OPTIONAL_PIT_INTERVAL_COLS):
        interval_df = listing_master.copy()
        interval_df["effective_from"] = interval_df["effective_from"].fillna(interval_df["list_date"])
        interval_df["effective_to"] = interval_df["effective_to"].fillna(interval_df["delist_date"])
        interval_df["effective_to"] = interval_df["effective_to"].fillna(end_date)

        merged = panel.merge(interval_df, on="instrument_id", suffixes=("", "__pit"), how="left")
        mask = (merged["date"] >= merged["effective_from"]) & (merged["date"] <= merged["effective_to"])
        merged = merged.loc[mask].copy()

        if merged.empty:
            raise UniverseBuildError("PIT listing intervals eliminated all panel rows; inspect listing master.")

        merged.sort_values(["date", "instrument_id", "effective_from"], inplace=True)
        merged = merged.groupby(["date", "instrument_id"], as_index=False).tail(1)

        for base_col in [
            "issuer_id",
            "symbol",
            "listing_exchange",
            "security_type",
            "share_class",
            "list_date",
            "delist_date",
            "trading_status",
            "is_primary_listing",
        ]:
            pit_col = f"{base_col}__pit"
            if pit_col in merged.columns:
                merged[base_col] = merged[pit_col]

        keep_cols = [
            "date",
            "instrument_id",
            "issuer_id",
            "symbol",
            "listing_exchange",
            "security_type",
            "share_class",
            "list_date",
            "delist_date",
            "trading_status",
            "is_primary_listing",
        ]
        panel = merged[keep_cols].copy()

    dup = panel.duplicated(subset=["date", "instrument_id"], keep=False)
    if dup.any():
        sample = panel.loc[dup, ["date", "instrument_id"]].head(10)
        raise UniverseBuildError(
            f"Panel construction violated uniqueness on (date, instrument_id). Sample:\n{sample}"
        )

    return panel.sort_values(["date", "instrument_id"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Missing data policy
# -----------------------------------------------------------------------------


def _apply_forward_fill_if_enabled(
    panel: pd.DataFrame,
    config: UniverseConfig,
) -> pd.DataFrame:
    mode = config.missing_data.mode.lower().strip()
    if mode != "forward_fill":
        return panel

    fields = tuple(config.missing_data.forward_fill_fields)
    if not fields:
        return panel

    max_staleness = int(config.missing_data.max_staleness_days)
    if max_staleness <= 0:
        return panel

    panel = panel.sort_values(["instrument_id", "date"]).copy()

    source_flags: List[Dict[str, Any]] = []
    if "field_source_flags" in panel.columns:
        for raw in panel["field_source_flags"].fillna("{}").astype(str):
            try:
                source_flags.append(json.loads(raw))
            except Exception:
                source_flags.append({})
    else:
        source_flags = [{} for _ in range(len(panel))]
        panel["field_source_flags"] = "{}"

    for field_name in fields:
        if field_name not in panel.columns:
            continue

        original = panel[field_name].copy()
        source_date_col = f"__source_date_{field_name}"
        panel[source_date_col] = panel["date"].where(panel[field_name].notna(), pd.NaT)

        panel[field_name] = panel.groupby("instrument_id", sort=False)[field_name].ffill()
        panel[source_date_col] = panel.groupby("instrument_id", sort=False)[source_date_col].ffill()

        age_days = (panel["date"] - panel[source_date_col]).dt.days
        stale_mask = panel[field_name].notna() & panel[source_date_col].notna() & (age_days > max_staleness)
        panel.loc[stale_mask, field_name] = pd.NA
        panel.loc[stale_mask, source_date_col] = pd.NaT

        used_ffill = original.isna() & panel[field_name].notna() & panel[source_date_col].notna()
        used_positions = [pos for pos, flag in enumerate(used_ffill.tolist()) if flag]
        for pos in used_positions:
            source_flags[pos][field_name] = {
                "source": "forward_fill",
                "source_date": str(pd.Timestamp(panel.iloc[pos][source_date_col]).date()),
                "age_days": int((panel.iloc[pos]["date"] - panel.iloc[pos][source_date_col]).days),
            }

        panel.drop(columns=[source_date_col], inplace=True)

    panel["field_source_flags"] = [canonical_json(x) for x in source_flags]
    return panel


# -----------------------------------------------------------------------------
# Filter application
# -----------------------------------------------------------------------------


def _append_failures(row_failures: List[str], *reasons: Optional[ExclusionReason]) -> List[str]:
    for reason in reasons:
        if reason is not None:
            row_failures.append(reason.value)
    return row_failures


def _reason_from_missing(field_name: str) -> ExclusionReason:
    mapping = {
        "price_ref": ExclusionReason.MISSING_PRICE,
        "adv20_usd": ExclusionReason.MISSING_ADV20,
        "market_cap_usd": ExclusionReason.MISSING_MCAP,
        "security_type": ExclusionReason.MISSING_SECURITY_TYPE,
        "listing_exchange": ExclusionReason.MISSING_EXCHANGE,
        "list_date": ExclusionReason.MISSING_LIST_DATE,
    }
    return mapping[field_name]


def apply_filters(panel: pd.DataFrame, config: UniverseConfig) -> pd.DataFrame:
    rules = config.rules
    panel = panel.copy()

    # Age since listing under the official decision convention close(t-1) -> execute(t).
    panel["days_since_listing"] = (panel["date"] - panel["list_date"]).dt.days

    failed_reasons: List[List[str]] = []

    allowed_security_types = {x.upper() for x in rules.allowed_security_types}
    allowed_exchanges = {x.upper() for x in rules.allowed_exchanges}
    allowed_statuses = {x.upper() for x in rules.allowed_trading_statuses}
    halted_statuses = {x.upper() for x in rules.halted_statuses}
    suspended_statuses = {x.upper() for x in rules.suspended_statuses}
    allowed_share_classes = {x.upper() for x in rules.allowed_share_classes}
    excluded_share_classes = {x.upper() for x in rules.excluded_share_classes}
    excluded_instrument_ids = {str(x) for x in rules.excluded_instrument_ids}
    excluded_symbols = {str(x).upper() for x in rules.excluded_symbols}

    for row in panel.itertuples(index=False):
        failures: List[str] = []

        # 1) identity and life economics -- panel already clipped to life window,
        # but we still defensively protect invariants.
        if pd.isna(row.list_date):
            _append_failures(failures, ExclusionReason.MISSING_LIST_DATE)
        elif pd.Timestamp(row.date) < pd.Timestamp(row.list_date):
            _append_failures(failures, ExclusionReason.NOT_YET_LISTED)
        if pd.notna(row.delist_date) and pd.Timestamp(row.date) > pd.Timestamp(row.delist_date):
            _append_failures(failures, ExclusionReason.DELISTED)

        # 2) security type.
        if pd.isna(row.security_type) or str(row.security_type).strip() == "":
            _append_failures(failures, ExclusionReason.MISSING_SECURITY_TYPE)
        elif str(row.security_type).upper() not in allowed_security_types:
            _append_failures(failures, ExclusionReason.BAD_SECURITY_TYPE)

        # 3) exchange / venue.
        if pd.isna(row.listing_exchange) or str(row.listing_exchange).strip() == "":
            _append_failures(failures, ExclusionReason.MISSING_EXCHANGE)
        elif str(row.listing_exchange).upper() not in allowed_exchanges:
            _append_failures(failures, ExclusionReason.BAD_EXCHANGE)

        # 4) operational status.
        status = str(row.trading_status).upper() if pd.notna(row.trading_status) else ""
        if status in halted_statuses:
            _append_failures(failures, ExclusionReason.HALTED)
        elif status in suspended_statuses:
            _append_failures(failures, ExclusionReason.SUSPENDED)
        elif status and status not in allowed_statuses:
            _append_failures(failures, ExclusionReason.SPECIAL_RULE_EXCLUSION)

        # 5) age since listing.
        if pd.notna(row.days_since_listing) and int(row.days_since_listing) < int(rules.min_listing_age_days):
            _append_failures(failures, ExclusionReason.TOO_YOUNG_SINCE_LISTING)

        # 6) price minimum.
        if pd.isna(row.price_ref):
            _append_failures(failures, ExclusionReason.MISSING_PRICE)
        elif float(row.price_ref) < float(rules.min_price_ref):
            _append_failures(failures, ExclusionReason.PRICE_BELOW_MIN)

        # 7) liquidity minimum.
        if pd.isna(row.adv20_usd):
            _append_failures(failures, ExclusionReason.MISSING_ADV20)
        elif float(row.adv20_usd) < float(rules.min_adv20_usd):
            _append_failures(failures, ExclusionReason.ADV20_BELOW_MIN)

        # 8) market cap band.
        if pd.isna(row.market_cap_usd):
            _append_failures(failures, ExclusionReason.MISSING_MCAP)
        else:
            mcap = float(row.market_cap_usd)
            if mcap < float(rules.min_market_cap_usd) or mcap > float(rules.max_market_cap_usd):
                _append_failures(failures, ExclusionReason.MCAP_OUT_OF_BAND)

        # 9) special rules.
        if rules.require_primary_listing and not bool(row.is_primary_listing):
            _append_failures(failures, ExclusionReason.SPECIAL_RULE_EXCLUSION)
        if allowed_share_classes and str(row.share_class).upper() not in allowed_share_classes:
            _append_failures(failures, ExclusionReason.SPECIAL_RULE_EXCLUSION)
        if excluded_share_classes and str(row.share_class).upper() in excluded_share_classes:
            _append_failures(failures, ExclusionReason.SPECIAL_RULE_EXCLUSION)
        if row.instrument_id in excluded_instrument_ids:
            _append_failures(failures, ExclusionReason.SPECIAL_RULE_EXCLUSION)
        if str(row.symbol).upper() in excluded_symbols:
            _append_failures(failures, ExclusionReason.SPECIAL_RULE_EXCLUSION)

        # Deduplicate while preserving filter order / precedence.
        deduped = []
        seen = set()
        for reason in failures:
            if reason not in seen:
                seen.add(reason)
                deduped.append(reason)
        failed_reasons.append(deduped)

    panel["all_failed_reasons"] = [listify_reasons(x) for x in failed_reasons]
    panel["primary_exclusion_reason"] = [x[0] if x else None for x in failed_reasons]
    panel["is_eligible"] = panel["primary_exclusion_reason"].isna().astype("int8")

    def _membership(primary_reason: Optional[str]) -> str:
        if primary_reason is None:
            return MembershipState.ELIGIBLE.value
        if primary_reason in {
            ExclusionReason.MISSING_PRICE.value,
            ExclusionReason.MISSING_ADV20.value,
            ExclusionReason.MISSING_MCAP.value,
            ExclusionReason.MISSING_SECURITY_TYPE.value,
            ExclusionReason.MISSING_EXCHANGE.value,
            ExclusionReason.MISSING_LIST_DATE.value,
        }:
            return MembershipState.INELIGIBLE_MISSING.value
        if primary_reason == ExclusionReason.HALTED.value:
            return MembershipState.INELIGIBLE_HALTED.value
        if primary_reason == ExclusionReason.SUSPENDED.value:
            return MembershipState.INELIGIBLE_SUSPENDED.value
        return MembershipState.INELIGIBLE_RULE.value

    panel["membership_state"] = panel["primary_exclusion_reason"].map(_membership)
    return panel


# -----------------------------------------------------------------------------
# Transitions, stats, invariants, QC
# -----------------------------------------------------------------------------


def compute_transitions(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["instrument_id", "date"]).copy()
    prev_eligible = panel.groupby("instrument_id", sort=False)["is_eligible"].shift(1)
    prev_state = panel.groupby("instrument_id", sort=False)["membership_state"].shift(1)

    def _transition(curr_eligible: int, curr_state: str, prev_el: Any, prev_st: Any) -> str:
        if pd.isna(prev_el):
            return Transition.ENTER.value if int(curr_eligible) == 1 else Transition.STAY_OUT.value
        prev_el = int(prev_el)
        curr_eligible = int(curr_eligible)
        if prev_el == 0 and curr_eligible == 1:
            return Transition.ENTER.value
        if prev_el == 1 and curr_eligible == 0:
            return Transition.EXIT.value
        if prev_el == 1 and curr_eligible == 1:
            return Transition.STAY_IN.value
        if prev_el == 0 and curr_eligible == 0 and prev_st != curr_state:
            return Transition.STATE_CHANGE_OUTSIDE_UNIVERSE.value
        return Transition.STAY_OUT.value

    panel["transition"] = [
        _transition(curr_el, curr_st, prev_el, prev_st)
        for curr_el, curr_st, prev_el, prev_st in zip(
            panel["is_eligible"], panel["membership_state"], prev_eligible, prev_state
        )
    ]
    return panel


@dataclass
class QCCheck:
    name: str
    severity: Severity
    passed: bool
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildArtifacts:
    history: pd.DataFrame
    current: pd.DataFrame
    constituents_latest: pd.DataFrame
    exclusions_latest: pd.DataFrame
    counts_by_day: pd.DataFrame
    turnover_stats: pd.DataFrame
    reason_breakdown_by_day: pd.DataFrame
    manifest: Dict[str, Any]
    qc_checks: List[QCCheck]


def compute_daily_stats(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = panel.loc[panel["is_eligible"] == 1, ["date", "instrument_id"]].copy()
    counts = (
        panel.groupby("date", as_index=False)
        .agg(
            n_panel=("instrument_id", "size"),
            n_constituents=("is_eligible", "sum"),
            n_ineligible=("is_eligible", lambda s: int((1 - s).sum())),
            n_missing=("membership_state", lambda s: int((s == MembershipState.INELIGIBLE_MISSING.value).sum())),
            n_halted=("membership_state", lambda s: int((s == MembershipState.INELIGIBLE_HALTED.value).sum())),
            n_suspended=("membership_state", lambda s: int((s == MembershipState.INELIGIBLE_SUSPENDED.value).sum())),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    eligible_sets = {
        date: set(group["instrument_id"].astype(str).tolist())
        for date, group in eligible.groupby("date", sort=True)
    }

    turnover_rows: List[Dict[str, Any]] = []
    previous_date: Optional[pd.Timestamp] = None
    previous_set: Optional[set[str]] = None
    for date in counts["date"]:
        current_set = eligible_sets.get(date, set())
        if previous_set is None:
            turnover_rows.append(
                {
                    "date": date,
                    "entries": len(current_set),
                    "exits": 0,
                    "n_constituents": len(current_set),
                    "turnover": 0.0,
                }
            )
        else:
            union = current_set | previous_set
            sym_diff = current_set.symmetric_difference(previous_set)
            entries = current_set - previous_set
            exits = previous_set - current_set
            turnover = 0.0 if not union else len(sym_diff) / len(union)
            turnover_rows.append(
                {
                    "date": date,
                    "prev_date": previous_date,
                    "entries": len(entries),
                    "exits": len(exits),
                    "n_constituents": len(current_set),
                    "turnover": float(turnover),
                }
            )
        previous_date = date
        previous_set = current_set

    turnover_stats = pd.DataFrame(turnover_rows)

    reason_breakdown = (
        panel.loc[panel["is_eligible"] == 0]
        .assign(primary_exclusion_reason=lambda x: x["primary_exclusion_reason"].fillna("UNKNOWN"))
        .groupby(["date", "primary_exclusion_reason"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["date", "n", "primary_exclusion_reason"], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    return counts, turnover_stats, reason_breakdown


def validate_hard_invariants(panel: pd.DataFrame) -> List[QCCheck]:
    checks: List[QCCheck] = []

    dup = panel.duplicated(subset=["date", "instrument_id"], keep=False)
    checks.append(
        QCCheck(
            name="unique_date_instrument",
            severity=Severity.FAIL,
            passed=not dup.any(),
            message="Uniqueness on (date, instrument_id)",
            metrics={"n_violations": int(dup.sum())},
        )
    )

    valid_binary = panel["is_eligible"].isin([0, 1]).all()
    checks.append(
        QCCheck(
            name="eligible_binary",
            severity=Severity.FAIL,
            passed=bool(valid_binary),
            message="is_eligible must be binary {0,1}",
        )
    )

    ok_reason_link1 = ((panel["is_eligible"] == 0) == panel["primary_exclusion_reason"].notna()).all()
    checks.append(
        QCCheck(
            name="eligible_reason_bijection",
            severity=Severity.FAIL,
            passed=bool(ok_reason_link1),
            message="Ineligible rows must have a primary reason; eligible rows must not.",
        )
    )

    before_listing = (panel["date"] < panel["list_date"]) & (panel["is_eligible"] == 1)
    checks.append(
        QCCheck(
            name="no_eligible_before_list_date",
            severity=Severity.FAIL,
            passed=not before_listing.any(),
            message="No instrument can be eligible before list_date.",
            metrics={"n_violations": int(before_listing.sum())},
        )
    )

    after_death = panel["delist_date"].notna() & (panel["date"] > panel["delist_date"]) & (panel["is_eligible"] == 1)
    checks.append(
        QCCheck(
            name="no_eligible_after_delist_date",
            severity=Severity.FAIL,
            passed=not after_death.any(),
            message="No instrument can be eligible after delist_date/tau_death.",
            metrics={"n_violations": int(after_death.sum())},
        )
    )

    return checks


def compute_soft_qc_checks(
    panel: pd.DataFrame,
    counts_by_day: pd.DataFrame,
    turnover_stats: pd.DataFrame,
    reason_breakdown_by_day: pd.DataFrame,
    config: UniverseConfig,
) -> List[QCCheck]:
    checks: List[QCCheck] = []
    qc = config.qc

    if not counts_by_day.empty:
        prev = counts_by_day["n_constituents"].shift(1)
        jump_frac = ((counts_by_day["n_constituents"] - prev).abs() / prev.replace({0: pd.NA})).fillna(0.0)
        max_jump = float(jump_frac.max()) if len(jump_frac) else 0.0
        checks.append(
            QCCheck(
                name="constituent_jump_fraction",
                severity=Severity.WARN if max_jump > qc.max_daily_constituent_jump_fraction else Severity.INFO,
                passed=max_jump <= qc.max_daily_constituent_jump_fraction,
                message="Daily constituent count jump fraction within threshold.",
                metrics={"max_jump_fraction": max_jump, "threshold": qc.max_daily_constituent_jump_fraction},
            )
        )

    max_turnover = float(turnover_stats["turnover"].max()) if not turnover_stats.empty else 0.0
    checks.append(
        QCCheck(
            name="turnover_level",
            severity=Severity.WARN if max_turnover > qc.max_turnover_warn else Severity.INFO,
            passed=max_turnover <= qc.max_turnover_warn,
            message="Universe turnover remains within expected range.",
            metrics={"max_turnover": max_turnover, "threshold": qc.max_turnover_warn},
        )
    )

    if not counts_by_day.empty:
        crit_missing_frac = (
            counts_by_day["n_missing"] / counts_by_day["n_panel"].replace({0: pd.NA})
        ).fillna(0.0)
        max_missing = float(crit_missing_frac.max()) if len(crit_missing_frac) else 0.0
        checks.append(
            QCCheck(
                name="critical_missing_fraction",
                severity=Severity.WARN if max_missing > qc.max_critical_missing_fraction_warn else Severity.INFO,
                passed=max_missing <= qc.max_critical_missing_fraction_warn,
                message="Critical missingness within threshold.",
                metrics={"max_missing_fraction": max_missing, "threshold": qc.max_critical_missing_fraction_warn},
            )
        )

    if not reason_breakdown_by_day.empty:
        totals = reason_breakdown_by_day.groupby("date")["n"].sum().rename("total")
        tmp = reason_breakdown_by_day.merge(totals, on="date", how="left")
        tmp["reason_fraction"] = tmp["n"] / tmp["total"].replace({0: pd.NA})
        max_reason_frac = float(tmp["reason_fraction"].fillna(0.0).max()) if not tmp.empty else 0.0
        checks.append(
            QCCheck(
                name="top_reason_concentration",
                severity=Severity.WARN if max_reason_frac > qc.max_top_reason_fraction_warn else Severity.INFO,
                passed=max_reason_frac <= qc.max_top_reason_fraction_warn,
                message="No single exclusion reason dominates unusually strongly.",
                metrics={"max_reason_fraction": max_reason_frac, "threshold": qc.max_top_reason_fraction_warn},
            )
        )

    return checks


# -----------------------------------------------------------------------------
# Persistence and manifest
# -----------------------------------------------------------------------------


def _write_parquet(df: pd.DataFrame, path: Path, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression=compression)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _hash_material_inputs(input_paths: Mapping[str, Path]) -> Dict[str, str]:
    return {name: sha256_file(path) for name, path in input_paths.items()}


def _frame_logical_hash(df: pd.DataFrame) -> str:
    safe = df.copy()
    safe = safe.sort_values(list(safe.columns)).reset_index(drop=True)
    safe = safe.where(pd.notna(safe), None)
    return sha256_text(canonical_json(safe.to_dict(orient="records")))


def persist_outputs(
    artifacts: BuildArtifacts,
    output_dir: str | Path,
    compression: str,
) -> None:
    ensure_parquet_engine_available()
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    _write_parquet(artifacts.history, base / "history.parquet", compression)
    _write_parquet(artifacts.current, base / "current.parquet", compression)
    _write_parquet(artifacts.constituents_latest, base / f"constituents_{artifacts.manifest['end_date']}.parquet", compression)
    _write_parquet(artifacts.exclusions_latest, base / f"exclusions_{artifacts.manifest['end_date']}.parquet", compression)
    _write_parquet(artifacts.counts_by_day, base / "universe_counts_by_day.parquet", compression)
    _write_parquet(artifacts.turnover_stats, base / "universe_turnover_stats.parquet", compression)
    _write_parquet(artifacts.reason_breakdown_by_day, base / "universe_reason_breakdown_by_day.parquet", compression)

    manifest_path = base / "universe_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(artifacts.manifest), fh, indent=2, sort_keys=True, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Main builder
# -----------------------------------------------------------------------------


def build_universe(
    listing_master_path: str | Path,
    calendar_path: str | Path,
    features_asof_path: str | Path,
    config_path: str | Path,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    run_id: str,
    asof_ts_utc: str,
    output_dir: Optional[str | Path] = None,
    log_level: str = "INFO",
) -> BuildArtifacts:
    setup_logging(log_level)
    built_ts_utc = utc_now_iso()
    build_started_utc = built_ts_utc

    start_ts = parse_date(start_date, "start_date")
    end_ts = parse_date(end_date, "end_date")
    if end_ts < start_ts:
        raise UniverseBuildError("end_date must be >= start_date")

    asof_parsed = pd.to_datetime(asof_ts_utc, utc=True, errors="coerce")
    if pd.isna(asof_parsed):
        raise UniverseBuildError(f"Invalid asof_ts_utc: {asof_ts_utc!r}")

    input_paths = {
        "listing_master": ensure_path(listing_master_path),
        "calendar": ensure_path(calendar_path),
        "features_asof": ensure_path(features_asof_path),
        "config": ensure_path(config_path),
    }

    config, config_payload, config_hash = load_config(config_path)
    output_dir = str(output_dir or config.output.output_dir)

    logger.info("Loading listing master from %s", input_paths["listing_master"])
    listing_master = load_listing_master(input_paths["listing_master"])

    logger.info("Loading calendar from %s", input_paths["calendar"])
    calendar = load_calendar(input_paths["calendar"], start_ts, end_ts)

    logger.info("Loading features-asof from %s", input_paths["features_asof"])
    features_asof = load_features_asof(input_paths["features_asof"])
    features_asof = features_asof[(features_asof["date"] >= start_ts) & (features_asof["date"] <= end_ts)].copy()
    if features_asof.empty:
        raise UniverseBuildError("features_asof has no rows inside the requested date range")

    logger.info("Building evaluable PIT panel")
    panel = _build_life_windows(listing_master, calendar["date"], start_ts, end_ts)

    logger.info("Enriching panel with as-of features")
    panel = panel.merge(features_asof, on=["date", "instrument_id"], how="left", suffixes=("", "__feat"))

    # Allow feature-level operational attributes to override listing-master snapshot if present.
    for optional_col in ["trading_status", "symbol", "listing_exchange", "security_type"]:
        feat_col = f"{optional_col}__feat"
        if feat_col in panel.columns:
            panel[optional_col] = panel[feat_col].combine_first(panel[optional_col])
            panel.drop(columns=[feat_col], inplace=True)

    if "field_source_flags__feat" in panel.columns:
        panel["field_source_flags"] = panel["field_source_flags__feat"].combine_first(panel.get("field_source_flags"))
        panel.drop(columns=["field_source_flags__feat"], inplace=True)
    elif "field_source_flags" not in panel.columns:
        panel["field_source_flags"] = "{}"

    panel = _apply_forward_fill_if_enabled(panel, config)
    panel = apply_filters(panel, config)
    panel = compute_transitions(panel)

    code_version = maybe_git_code_version()
    input_snapshot_hash = sha256_text(canonical_json(_hash_material_inputs(input_paths)))
    panel["run_id"] = run_id
    panel["config_hash"] = config_hash
    panel["code_version"] = code_version
    panel["input_snapshot_hash"] = input_snapshot_hash
    panel["built_ts_utc"] = built_ts_utc

    # Keep the contract tight and predictable.
    desired_columns = [
        "date",
        "instrument_id",
        "issuer_id",
        "symbol",
        "is_eligible",
        "membership_state",
        "transition",
        "primary_exclusion_reason",
        "all_failed_reasons",
        "market_cap_usd",
        "adv20_usd",
        "price_ref",
        "listing_exchange",
        "security_type",
        "share_class",
        "trading_status",
        "is_primary_listing",
        "list_date",
        "delist_date",
        "days_since_listing",
        "field_source_flags",
        "run_id",
        "config_hash",
        "code_version",
        "input_snapshot_hash",
        "built_ts_utc",
    ]
    history = panel[[c for c in desired_columns if c in panel.columns]].copy()
    history.sort_values(["date", "instrument_id"], inplace=True)
    history.reset_index(drop=True, inplace=True)

    hard_checks = validate_hard_invariants(history)
    counts_by_day, turnover_stats, reason_breakdown_by_day = compute_daily_stats(history)
    soft_checks = compute_soft_qc_checks(history, counts_by_day, turnover_stats, reason_breakdown_by_day, config)
    qc_checks = hard_checks + soft_checks

    fail_checks = [c for c in qc_checks if c.severity == Severity.FAIL and not c.passed]
    if fail_checks:
        messages = "\n".join(f"- {c.name}: {c.message} | {c.metrics}" for c in fail_checks)
        raise UniverseBuildError(f"Hard invariant failure(s) detected:\n{messages}")

    latest_date = pd.Timestamp(history["date"].max())
    current = history.loc[history["date"] == latest_date].copy()
    constituents_latest = current.loc[current["is_eligible"] == 1].copy()
    exclusions_latest = current.loc[current["is_eligible"] == 0].copy()

    reason_counts_total = (
        history.loc[history["is_eligible"] == 0, "primary_exclusion_reason"]
        .fillna("UNKNOWN")
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    manifest = {
        "run_id": run_id,
        "start_date": str(start_ts.date()),
        "end_date": str(end_ts.date()),
        "asof_ts_utc": str(pd.Timestamp(asof_parsed).isoformat()),
        "built_ts_utc": built_ts_utc,
        "build_started_ts_utc": build_started_utc,
        "build_finished_ts_utc": utc_now_iso(),
        "config_hash": config_hash,
        "config": dataclasses.asdict(config),
        "code_version": code_version,
        "python_version": sys.version,
        "platform": platform.platform(),
        "input_hashes": _hash_material_inputs(input_paths),
        "input_snapshot_hash": input_snapshot_hash,
        "temporal_convention": TEMPORAL_CONVENTION,
        "missing_data_policy": dataclasses.asdict(config.missing_data),
        "edge_case_policy": dataclasses.asdict(config.edge_cases),
        "n_rows_history": int(len(history)),
        "n_days": int(history["date"].nunique()),
        "n_instruments": int(history["instrument_id"].nunique()),
        "latest_date": str(latest_date.date()),
        "latest_n_constituents": int(constituents_latest["instrument_id"].nunique()),
        "counts_by_day_summary": {
            "min_constituents": int(counts_by_day["n_constituents"].min()) if not counts_by_day.empty else 0,
            "max_constituents": int(counts_by_day["n_constituents"].max()) if not counts_by_day.empty else 0,
            "median_constituents": float(counts_by_day["n_constituents"].median()) if not counts_by_day.empty else 0.0,
        },
        "exclusion_reason_counts_total": reason_counts_total,
        "qc_checks": [dataclasses.asdict(c) for c in qc_checks],
        "run_status": max_severity([c.severity for c in qc_checks if not c.passed] or [Severity.INFO]).value,
        "history_logical_hash": _frame_logical_hash(history),
    }

    artifacts = BuildArtifacts(
        history=history,
        current=current,
        constituents_latest=constituents_latest,
        exclusions_latest=exclusions_latest,
        counts_by_day=counts_by_day,
        turnover_stats=turnover_stats,
        reason_breakdown_by_day=reason_breakdown_by_day,
        manifest=manifest,
        qc_checks=qc_checks,
    )

    persist_outputs(artifacts, output_dir=output_dir, compression=config.output.compression)
    logger.info("Universe build completed. Output written to %s", output_dir)
    return artifacts


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m data.universe.build_universe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Build the canonical daily point-in-time investable universe.

            Temporal convention:
              decision_time = close(t-1)
              execution_date = t
            """
        ).strip(),
    )
    parser.add_argument("--listing-master-path", required=True)
    parser.add_argument("--calendar-path", required=True)
    parser.add_argument("--features-asof-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--as-of-ts-utc", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        build_universe(
            listing_master_path=args.listing_master_path,
            calendar_path=args.calendar_path,
            features_asof_path=args.features_asof_path,
            config_path=args.config_path,
            start_date=args.start_date,
            end_date=args.end_date,
            run_id=args.run_id,
            asof_ts_utc=args.as_of_ts_utc,
            output_dir=args.output_dir,
            log_level=args.log_level,
        )
        return 0
    except UniverseBuildError as exc:
        logger.error("Universe build failed: %s", exc)
        return 2
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error: %s", exc)
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
