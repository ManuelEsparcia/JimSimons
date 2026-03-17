from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import logging
import math
import platform
import random
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# -----------------------------------------------------------------------------
# Constants, enums and exceptions
# -----------------------------------------------------------------------------


class UniverseQCError(RuntimeError):
    """Raised when a hard validation or IO invariant fails."""


class Severity(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class CheckType(str, Enum):
    STRUCTURAL_DUPLICATE_LOGICAL_KEY = "STRUCTURAL_DUPLICATE_LOGICAL_KEY"
    STRUCTURAL_DUPLICATE_SYMBOL = "STRUCTURAL_DUPLICATE_SYMBOL"
    STRUCTURAL_INVALID_IS_ELIGIBLE = "STRUCTURAL_INVALID_IS_ELIGIBLE"
    STRUCTURAL_INVALID_MEMBERSHIP_STATE = "STRUCTURAL_INVALID_MEMBERSHIP_STATE"
    STRUCTURAL_INVALID_EXCLUSION_REASON = "STRUCTURAL_INVALID_EXCLUSION_REASON"
    STRUCTURAL_MISSING_REQUIRED_COLUMN = "STRUCTURAL_MISSING_REQUIRED_COLUMN"
    STRUCTURAL_CONFIG_HASH_DRIFT = "STRUCTURAL_CONFIG_HASH_DRIFT"
    STRUCTURAL_RUN_ID_DRIFT = "STRUCTURAL_RUN_ID_DRIFT"
    PIT_OUTSIDE_LIFECYCLE_WINDOW = "PIT_OUTSIDE_LIFECYCLE_WINDOW"
    PIT_SYMBOL_WINDOW_MISMATCH = "PIT_SYMBOL_WINDOW_MISMATCH"
    PIT_ELIGIBLE_AFTER_TERMINATION = "PIT_ELIGIBLE_AFTER_TERMINATION"
    CALENDAR_INVALID_SESSION = "CALENDAR_INVALID_SESSION"
    CALENDAR_MISSING_SESSION = "CALENDAR_MISSING_SESSION"
    SEMANTIC_ELIGIBILITY_REASON_MISMATCH = "SEMANTIC_ELIGIBILITY_REASON_MISMATCH"
    SEMANTIC_PRIMARY_REASON_ORDER_MISMATCH = "SEMANTIC_PRIMARY_REASON_ORDER_MISMATCH"
    SEMANTIC_MEMBERSHIP_STATE_MISMATCH = "SEMANTIC_MEMBERSHIP_STATE_MISMATCH"
    SEMANTIC_CRITICAL_RULE_REVALIDATION = "SEMANTIC_CRITICAL_RULE_REVALIDATION"
    TEMPORAL_EXTREME_TURNOVER = "TEMPORAL_EXTREME_TURNOVER"
    TEMPORAL_HIGH_TURNOVER = "TEMPORAL_HIGH_TURNOVER"
    TEMPORAL_EXTREME_SIZE_JUMP = "TEMPORAL_EXTREME_SIZE_JUMP"
    TEMPORAL_HIGH_SIZE_JUMP = "TEMPORAL_HIGH_SIZE_JUMP"
    TEMPORAL_REASON_SPIKE = "TEMPORAL_REASON_SPIKE"
    COVERAGE_LOW_DELISTED_COVERAGE = "COVERAGE_LOW_DELISTED_COVERAGE"
    COVERAGE_REACTIVATION_AFTER_DEATH = "COVERAGE_REACTIVATION_AFTER_DEATH"


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


TERMINAL_EVENT_TYPES = {
    "delisting",
    "delisted",
    "acquisition",
    "acquired",
    "merger",
    "liquidation",
    "bankruptcy",
    "termination",
    "cancellation",
    "cash_out",
}

CONTINUITY_EVENT_TYPES = {
    "ticker_change",
    "symbol_change",
    "exchange_change",
    "name_change",
    "relisting_continuous",
    "continuity",
    "identity_continuity",
}

DEFAULT_ALLOWED_SECURITY_TYPES = ("COMMON_STOCK", "ORDINARY_SHARE")
DEFAULT_ALLOWED_EXCHANGES = ("NYSE", "NASDAQ", "NYSE_ARCA", "NYSE_AMERICAN")
DEFAULT_MISSING_CRITICAL_META_COLS = (
    "market_cap_usd",
    "adv20_usd",
    "price_ref",
    "listing_exchange",
    "security_type",
    "trading_status",
    "list_date",
)
DEFAULT_SAMPLE_SEED = 17

REQUIRED_UNIVERSE_COLUMNS = {
    "date",
    "instrument_id",
    "symbol",
    "is_eligible",
    "membership_state",
    "primary_exclusion_reason",
}

RECOMMENDED_UNIVERSE_COLUMNS = {
    "issuer_id",
    "all_failed_reasons",
    "market_cap_usd",
    "adv20_usd",
    "price_ref",
    "listing_exchange",
    "security_type",
    "trading_status",
    "config_hash",
    "run_id",
}

REQUIRED_LISTINGS_COLUMNS = {
    "instrument_id",
    "symbol",
    "list_date",
}

OPTIONAL_LISTINGS_COLUMNS = {
    "issuer_id",
    "delist_date",
    "ticker_start_date",
    "ticker_end_date",
    "status",
    "listing_exchange",
    "security_type",
    "share_class",
}

OPTIONAL_CORPORATE_ACTION_COLUMNS = {
    "event_type",
    "effective_date",
    "instrument_id",
    "from_instrument_id",
    "to_instrument_id",
    "old_instrument_id",
    "new_instrument_id",
    "predecessor_instrument_id",
    "successor_instrument_id",
    "linked_corporate_action_id",
}


# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CriticalRulesConfig:
    min_price_ref: float = 3.0
    min_adv20_usd: float = 1_000_000.0
    min_market_cap_usd: float = 150_000_000.0
    max_market_cap_usd: float = 5_000_000_000.0
    min_listing_age_days: int = 20
    allowed_security_types: Tuple[str, ...] = DEFAULT_ALLOWED_SECURITY_TYPES
    allowed_exchanges: Tuple[str, ...] = DEFAULT_ALLOWED_EXCHANGES


@dataclass(frozen=True)
class ThresholdConfig:
    tau_semantic_warn: float = 0.001
    tau_semantic_fail: float = 0.005
    tau_turnover_warn: float = 0.35
    tau_turnover_fail: float = 0.55
    tau_size_jump_warn: float = 0.20
    tau_size_jump_fail: float = 0.35
    delisted_coverage_warn: float = 0.90
    delisted_coverage_pass: float = 0.95
    pct_days_high_turnover_warn: float = 0.05
    pct_days_high_turnover_fail: float = 0.15
    pct_days_extreme_turnover_fail: float = 0.02
    pct_missing_critical_meta_warn: float = 0.02
    pct_missing_critical_meta_fail: float = 0.05
    reason_spike_warn_multiple: float = 3.0
    reason_spike_fail_multiple: float = 6.0


@dataclass(frozen=True)
class ValidationConfig:
    strict_mode: bool = True
    revalidate_sample_frac: float = 1.0
    revalidate_max_rows: int = 100_000
    random_seed: int = DEFAULT_SAMPLE_SEED
    warn_on_symbol_duplicates: bool = True
    allow_symbol_window_warn_if_continuity: bool = True


@dataclass(frozen=True)
class OutputConfig:
    output_dir: str = "data/universe/qc"
    compression: str = "snappy"


@dataclass(frozen=True)
class UniverseQCConfig:
    critical_rules: CriticalRulesConfig = field(default_factory=CriticalRulesConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    missing_critical_meta_cols: Tuple[str, ...] = DEFAULT_MISSING_CRITICAL_META_COLS


@dataclass(frozen=True)
class GateResult:
    name: str
    severity: Severity
    observed: Any
    threshold: Any
    message: str


@dataclass(frozen=True)
class QCArtifacts:
    daily: pd.DataFrame
    failures: pd.DataFrame
    summary: Dict[str, Any]
    manifest: Dict[str, Any]


# -----------------------------------------------------------------------------
# Utilities
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
        raise UniverseQCError(f"Input path does not exist: {path}")
    return path


def ensure_parquet_engine_available() -> None:
    if importlib.util.find_spec("pyarrow") is None and importlib.util.find_spec("fastparquet") is None:
        raise UniverseQCError(
            "Parquet support requires either 'pyarrow' or 'fastparquet'. Install one before running universe_qc.py."
        )


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
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def normalize_date_series(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, utc=False, errors="coerce")
    return pd.Series(pd.DatetimeIndex(out).normalize(), index=series.index)


def deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            deep_merge(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def severity_rank(sev: Severity | str) -> int:
    mapping = {Severity.PASS: 0, Severity.WARN: 1, Severity.FAIL: 2, "PASS": 0, "WARN": 1, "FAIL": 2}
    return mapping[sev]


def max_severity(values: Iterable[Severity | str], default: Severity = Severity.PASS) -> Severity:
    best = default
    for value in values:
        sev = Severity(value)
        if severity_rank(sev) > severity_rank(best):
            best = sev
    return best


def parse_reason_list(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(x) for x in value if pd.notna(x) and str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            decoded = json.loads(text)
            if isinstance(decoded, list):
                return [str(x) for x in decoded if pd.notna(x) and str(x).strip()]
        except Exception:
            pass
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def first_notna(series: pd.Series) -> Any:
    non_na = series.dropna()
    if non_na.empty:
        return pd.NA
    return non_na.iloc[0]


def safe_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def as_str(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------


def load_config(config_path: Optional[str | Path]) -> Tuple[UniverseQCConfig, Dict[str, Any], str]:
    default_payload = dataclasses.asdict(UniverseQCConfig())
    if config_path is None:
        cfg = UniverseQCConfig()
        cfg_hash = sha256_text(canonical_json(default_payload))
        return cfg, default_payload, cfg_hash

    path = ensure_path(config_path)
    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise UniverseQCError("PyYAML is required to parse YAML configs.")
        payload = yaml.safe_load(raw) or {}
    elif suffix == ".json":
        payload = json.loads(raw)
    else:
        raise UniverseQCError(f"Unsupported config format: {path.suffix}")
    if not isinstance(payload, Mapping):
        raise UniverseQCError("Config file must deserialize to a mapping/object.")

    merged = deep_merge(default_payload, dict(payload))
    cfg = UniverseQCConfig(
        critical_rules=CriticalRulesConfig(**merged.get("critical_rules", {})),
        thresholds=ThresholdConfig(**merged.get("thresholds", {})),
        validation=ValidationConfig(**merged.get("validation", {})),
        output=OutputConfig(**merged.get("output", {})),
        missing_critical_meta_cols=tuple(merged.get("missing_critical_meta_cols", DEFAULT_MISSING_CRITICAL_META_COLS)),
    )
    if not (0 < cfg.validation.revalidate_sample_frac <= 1.0):
        raise UniverseQCError("validation.revalidate_sample_frac must be in (0, 1].")

    cfg_hash = sha256_text(canonical_json(dataclasses.asdict(cfg)))
    return cfg, merged, cfg_hash


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------


def read_dataframe(path_like: str | Path) -> pd.DataFrame:
    path = ensure_path(path_like)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        ensure_parquet_engine_available()
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise UniverseQCError(f"Unsupported tabular format: {path}")


def read_maybe_partitioned(path_like: str | Path) -> pd.DataFrame:
    path = ensure_path(path_like)
    if path.is_dir():
        ensure_parquet_engine_available()
        parts = sorted(p for p in path.glob("*.parquet") if p.is_file())
        if not parts:
            raise UniverseQCError(f"Directory has no parquet parts: {path}")
        return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return read_dataframe(path)


def load_universe_history(path_like: str | Path) -> pd.DataFrame:
    df = read_maybe_partitioned(path_like).copy()
    missing = REQUIRED_UNIVERSE_COLUMNS - set(df.columns)
    if missing:
        raise UniverseQCError(f"universe_history missing required columns: {sorted(missing)}")

    df["date"] = normalize_date_series(df["date"])
    if df["date"].isna().any():
        raise UniverseQCError("universe_history contains invalid dates")

    df["instrument_id"] = df["instrument_id"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    if "issuer_id" not in df.columns:
        df["issuer_id"] = pd.NA
    else:
        df["issuer_id"] = df["issuer_id"].astype(str)

    for col in ["market_cap_usd", "adv20_usd", "price_ref"]:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["listing_exchange", "security_type", "trading_status", "membership_state", "primary_exclusion_reason"]:
        if col not in df.columns:
            df[col] = pd.NA

    for col in ["config_hash", "run_id", "all_failed_reasons"]:
        if col not in df.columns:
            df[col] = pd.NA

    for col in ["list_date", "delist_date"]:
        if col in df.columns:
            df[col] = normalize_date_series(df[col])
        else:
            df[col] = pd.NaT

    return df.sort_values(["date", "instrument_id"]).reset_index(drop=True)


def load_listings_master(path_like: str | Path) -> pd.DataFrame:
    df = read_maybe_partitioned(path_like).copy()
    missing = REQUIRED_LISTINGS_COLUMNS - set(df.columns)
    if missing:
        raise UniverseQCError(f"listings_master missing required columns: {sorted(missing)}")

    df["instrument_id"] = df["instrument_id"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    if "issuer_id" not in df.columns:
        df["issuer_id"] = pd.NA
    else:
        df["issuer_id"] = df["issuer_id"].astype(str)

    for col in ["list_date", "delist_date", "ticker_start_date", "ticker_end_date"]:
        if col not in df.columns:
            df[col] = pd.NaT
        else:
            df[col] = normalize_date_series(df[col])

    if df["list_date"].isna().any():
        raise UniverseQCError("listings_master.list_date contains invalid values")

    for col in ["status", "listing_exchange", "security_type", "share_class"]:
        if col not in df.columns:
            df[col] = pd.NA

    return df.sort_values(["instrument_id", "list_date", "ticker_start_date"], na_position="last").reset_index(drop=True)


def load_calendar(path_like: str | Path) -> pd.DatetimeIndex:
    df = read_maybe_partitioned(path_like).copy()
    date_col = None
    for candidate in ["date", "session_date", "trading_date"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        if len(df.columns) == 1:
            date_col = df.columns[0]
        else:
            raise UniverseQCError("calendar must contain a date/session_date/trading_date column")
    dates = pd.DatetimeIndex(pd.to_datetime(df[date_col], utc=False, errors="coerce")).normalize().dropna().unique().sort_values()
    if len(dates) == 0:
        raise UniverseQCError("calendar has no valid sessions")
    return dates


def load_corporate_actions(path_like: Optional[str | Path]) -> pd.DataFrame:
    if not path_like:
        return pd.DataFrame(columns=sorted(OPTIONAL_CORPORATE_ACTION_COLUMNS))
    df = read_maybe_partitioned(path_like).copy()
    required = {"event_type", "effective_date"}
    missing = required - set(df.columns)
    if missing:
        raise UniverseQCError(f"corporate_actions missing required columns: {sorted(missing)}")
    df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()
    df["effective_date"] = normalize_date_series(df["effective_date"])
    if df["effective_date"].isna().any():
        raise UniverseQCError("corporate_actions contains invalid effective_date values")
    for col in OPTIONAL_CORPORATE_ACTION_COLUMNS - {"event_type", "effective_date"}:
        if col not in df.columns:
            df[col] = pd.NA
        else:
            df[col] = df[col].astype(str)
    return df


# -----------------------------------------------------------------------------
# Supporting maps
# -----------------------------------------------------------------------------


def build_listing_lifecycle(listings: pd.DataFrame, ca: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        listings.groupby("instrument_id", as_index=False)
        .agg(
            {
                "issuer_id": first_notna,
                "symbol": first_notna,
                "list_date": "min",
                "delist_date": lambda s: s.dropna().min() if s.dropna().size else pd.NaT,
                "status": first_notna,
                "listing_exchange": first_notna,
                "security_type": first_notna,
                "share_class": first_notna,
            }
        )
        .sort_values("instrument_id")
        .reset_index(drop=True)
    )
    grouped["death_from_ca"] = pd.NaT
    grouped["terminal_ca_id"] = pd.NA

    if not ca.empty:
        terminal_rows = ca.loc[ca["event_type"].isin(TERMINAL_EVENT_TYPES)].copy()
        if not terminal_rows.empty:
            pieces: List[pd.DataFrame] = []
            for col in ["instrument_id", "from_instrument_id", "old_instrument_id", "predecessor_instrument_id"]:
                tmp = terminal_rows[[col, "effective_date", "linked_corporate_action_id"]].rename(columns={col: "instrument_id"})
                tmp = tmp[tmp["instrument_id"].notna()]
                pieces.append(tmp)
            terminal_map = pd.concat(pieces, ignore_index=True)
            terminal_map["instrument_id"] = terminal_map["instrument_id"].astype(str)
            terminal_map = terminal_map.sort_values(["instrument_id", "effective_date"]).groupby("instrument_id", as_index=False).first()
            grouped = grouped.merge(
                terminal_map.rename(columns={"effective_date": "death_from_ca", "linked_corporate_action_id": "terminal_ca_id"}),
                on="instrument_id",
                how="left",
                suffixes=("", "__dup"),
            )
            if "death_from_ca__dup" in grouped.columns:
                grouped["death_from_ca"] = grouped["death_from_ca"].combine_first(grouped["death_from_ca__dup"])
                grouped.drop(columns=["death_from_ca__dup"], inplace=True)
            if "terminal_ca_id__dup" in grouped.columns:
                grouped["terminal_ca_id"] = grouped["terminal_ca_id"].combine_first(grouped["terminal_ca_id__dup"])
                grouped.drop(columns=["terminal_ca_id__dup"], inplace=True)

    grouped["birth_date"] = grouped["list_date"]
    grouped["death_date"] = grouped[["delist_date", "death_from_ca"]].min(axis=1)
    grouped["is_dead_observed"] = grouped["death_date"].notna()
    return grouped


def build_symbol_windows(listings: pd.DataFrame) -> pd.DataFrame:
    win = listings[["instrument_id", "symbol", "ticker_start_date", "ticker_end_date"]].copy()
    win["ticker_start_date"] = win["ticker_start_date"].fillna(pd.NaT)
    win["ticker_end_date"] = win["ticker_end_date"].fillna(pd.NaT)
    return win


def build_continuity_map(ca: pd.DataFrame) -> Dict[str, Set[str]]:
    continuity: Dict[str, Set[str]] = {}

    def _link(left: Optional[str], right: Optional[str]) -> None:
        if left is None or right is None or pd.isna(left) or pd.isna(right):
            return
        a, b = str(left), str(right)
        continuity.setdefault(a, set()).add(b)
        continuity.setdefault(b, set()).add(a)

    if ca.empty:
        return continuity
    rows = ca.loc[ca["event_type"].isin(CONTINUITY_EVENT_TYPES)].copy()
    for row in rows.itertuples(index=False):
        for a, b in [
            (getattr(row, "from_instrument_id", None), getattr(row, "to_instrument_id", None)),
            (getattr(row, "old_instrument_id", None), getattr(row, "new_instrument_id", None)),
            (getattr(row, "predecessor_instrument_id", None), getattr(row, "successor_instrument_id", None)),
        ]:
            _link(a, b)
    return continuity


# -----------------------------------------------------------------------------
# Failures builder
# -----------------------------------------------------------------------------


def make_failure(
    *,
    date: Any,
    instrument_id: Any,
    symbol: Any,
    check_type: CheckType | str,
    severity: Severity | str,
    flag_reason: str,
    evidence_summary: str,
    linked_corporate_action_id: Any = pd.NA,
) -> Dict[str, Any]:
    return {
        "date": pd.NaT if date is None or pd.isna(date) else pd.Timestamp(date).normalize(),
        "instrument_id": pd.NA if instrument_id is None or pd.isna(instrument_id) else str(instrument_id),
        "symbol": pd.NA if symbol is None or pd.isna(symbol) else str(symbol),
        "check_type": check_type.value if isinstance(check_type, Enum) else str(check_type),
        "severity": severity.value if isinstance(severity, Enum) else str(severity),
        "flag_reason": flag_reason,
        "evidence_summary": evidence_summary,
        "linked_corporate_action_id": linked_corporate_action_id,
    }


# -----------------------------------------------------------------------------
# Structural checks
# -----------------------------------------------------------------------------


def structural_fail_fast_checks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []

    dup_logic = df.duplicated(subset=["date", "instrument_id"], keep=False)
    if dup_logic.any():
        for row in df.loc[dup_logic, ["date", "instrument_id", "symbol"]].head(500).itertuples(index=False):
            failures.append(
                make_failure(
                    date=row.date,
                    instrument_id=row.instrument_id,
                    symbol=row.symbol,
                    check_type=CheckType.STRUCTURAL_DUPLICATE_LOGICAL_KEY,
                    severity=Severity.FAIL,
                    flag_reason="duplicate_(date,instrument_id)",
                    evidence_summary="Logical key (date, instrument_id) appears more than once.",
                )
            )

    valid_is_eligible = df["is_eligible"].isin([0, 1, False, True])
    invalid_is_el = df.loc[~valid_is_eligible | df["is_eligible"].isna(), ["date", "instrument_id", "symbol", "is_eligible"]]
    for row in invalid_is_el.head(500).itertuples(index=False):
        failures.append(
            make_failure(
                date=row.date,
                instrument_id=row.instrument_id,
                symbol=row.symbol,
                check_type=CheckType.STRUCTURAL_INVALID_IS_ELIGIBLE,
                severity=Severity.FAIL,
                flag_reason="is_eligible_out_of_domain",
                evidence_summary=f"is_eligible={row.is_eligible!r} not in {{0,1}}.",
            )
        )

    valid_states = {x.value for x in MembershipState}
    invalid_states = df.loc[df["membership_state"].notna() & ~df["membership_state"].astype(str).isin(valid_states)]
    for row in invalid_states[["date", "instrument_id", "symbol", "membership_state"]].head(500).itertuples(index=False):
        failures.append(
            make_failure(
                date=row.date,
                instrument_id=row.instrument_id,
                symbol=row.symbol,
                check_type=CheckType.STRUCTURAL_INVALID_MEMBERSHIP_STATE,
                severity=Severity.FAIL,
                flag_reason="membership_state_out_of_enum",
                evidence_summary=f"membership_state={row.membership_state!r} is not in the official enum.",
            )
        )

    valid_reasons = {x.value for x in ExclusionReason}
    invalid_reasons = df.loc[
        df["primary_exclusion_reason"].notna()
        & ~df["primary_exclusion_reason"].astype(str).isin(valid_reasons)
    ]
    for row in invalid_reasons[["date", "instrument_id", "symbol", "primary_exclusion_reason"]].head(500).itertuples(index=False):
        failures.append(
            make_failure(
                date=row.date,
                instrument_id=row.instrument_id,
                symbol=row.symbol,
                check_type=CheckType.STRUCTURAL_INVALID_EXCLUSION_REASON,
                severity=Severity.FAIL,
                flag_reason="primary_exclusion_reason_out_of_enum",
                evidence_summary=f"primary_exclusion_reason={row.primary_exclusion_reason!r} is not recognized.",
            )
        )

    dup_symbol = df[df["symbol"].notna()].duplicated(subset=["date", "symbol"], keep=False)
    for row in df.loc[dup_symbol, ["date", "instrument_id", "symbol"]].head(500).itertuples(index=False):
        failures.append(
            make_failure(
                date=row.date,
                instrument_id=row.instrument_id,
                symbol=row.symbol,
                check_type=CheckType.STRUCTURAL_DUPLICATE_SYMBOL,
                severity=Severity.WARN,
                flag_reason="duplicate_(date,symbol)",
                evidence_summary="Secondary symbol key duplicates observed; may indicate ticker recycling or provider issue.",
            )
        )
    return failures


# -----------------------------------------------------------------------------
# PIT and semantic checks
# -----------------------------------------------------------------------------


def attach_lifecycle(df: pd.DataFrame, lifecycle: pd.DataFrame) -> pd.DataFrame:
    lifecycle_cols = [
        "instrument_id",
        "birth_date",
        "death_date",
        "terminal_ca_id",
        "is_dead_observed",
        "list_date",
        "delist_date",
        "status",
        "listing_exchange",
        "security_type",
    ]
    available = [c for c in lifecycle_cols if c in lifecycle.columns]
    out = df.merge(lifecycle[available], on="instrument_id", how="left", suffixes=("", "__lifecycle"))
    # Prefer explicit PIT fields from universe when present.
    for col in ["listing_exchange", "security_type"]:
        lc = f"{col}__lifecycle"
        if lc in out.columns:
            out[col] = out[col].combine_first(out[lc])
            out.drop(columns=[lc], inplace=True)
    return out


def run_pit_checks(
    df: pd.DataFrame,
    symbol_windows: pd.DataFrame,
    continuity_map: Dict[str, Set[str]],
    cfg: UniverseQCConfig,
) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []

    missing_lifecycle = df["birth_date"].isna()
    for row in df.loc[missing_lifecycle, ["date", "instrument_id", "symbol"]].head(500).itertuples(index=False):
        failures.append(
            make_failure(
                date=row.date,
                instrument_id=row.instrument_id,
                symbol=row.symbol,
                check_type=CheckType.PIT_OUTSIDE_LIFECYCLE_WINDOW,
                severity=Severity.FAIL,
                flag_reason="missing_lifecycle_window",
                evidence_summary="Instrument not found in lifecycle master; cannot certify PIT existence.",
            )
        )

    pre_listing = df["birth_date"].notna() & (df["date"] < df["birth_date"])
    for row in df.loc[pre_listing, ["date", "instrument_id", "symbol", "birth_date"]].head(500).itertuples(index=False):
        failures.append(
            make_failure(
                date=row.date,
                instrument_id=row.instrument_id,
                symbol=row.symbol,
                check_type=CheckType.PIT_OUTSIDE_LIFECYCLE_WINDOW,
                severity=Severity.FAIL,
                flag_reason="pre_listing_row",
                evidence_summary=f"Row date {row.date.date()} precedes list_date {row.birth_date.date()}.",
            )
        )

    post_death = df["death_date"].notna() & (df["date"] > df["death_date"])
    for row in df.loc[post_death, ["date", "instrument_id", "symbol", "death_date"]].head(500).itertuples(index=False):
        failures.append(
            make_failure(
                date=row.date,
                instrument_id=row.instrument_id,
                symbol=row.symbol,
                check_type=CheckType.PIT_OUTSIDE_LIFECYCLE_WINDOW,
                severity=Severity.FAIL,
                flag_reason="post_death_row",
                evidence_summary=f"Row date {row.date.date()} is after death_date {row.death_date.date()}.",
            )
        )

    eligible_after_term = df["death_date"].notna() & (df["date"] > df["death_date"]) & (df["is_eligible"].astype(str).isin(["1", "True", "true"]))
    for row in df.loc[eligible_after_term, ["date", "instrument_id", "symbol", "death_date", "terminal_ca_id"]].head(500).itertuples(index=False):
        failures.append(
            make_failure(
                date=row.date,
                instrument_id=row.instrument_id,
                symbol=row.symbol,
                check_type=CheckType.PIT_ELIGIBLE_AFTER_TERMINATION,
                severity=Severity.FAIL,
                flag_reason="eligible_after_terminal_event",
                evidence_summary=f"Instrument remains eligible after economic termination on {row.death_date.date()}.",
                linked_corporate_action_id=row.terminal_ca_id,
            )
        )

    if not symbol_windows.empty:
        merged = df[["date", "instrument_id", "symbol"]].merge(
            symbol_windows,
            on=["instrument_id", "symbol"],
            how="left",
        )
        no_window = merged["ticker_start_date"].isna() & merged["ticker_end_date"].isna()
        bad_start = merged["ticker_start_date"].notna() & (merged["date"] < merged["ticker_start_date"])
        bad_end = merged["ticker_end_date"].notna() & (merged["date"] > merged["ticker_end_date"])
        mismatched = no_window | bad_start | bad_end
        for row in merged.loc[mismatched, ["date", "instrument_id", "symbol", "ticker_start_date", "ticker_end_date"]].head(500).itertuples(index=False):
            has_continuity = bool(continuity_map.get(str(row.instrument_id)))
            severity = Severity.WARN if cfg.validation.allow_symbol_window_warn_if_continuity and has_continuity else Severity.FAIL
            failures.append(
                make_failure(
                    date=row.date,
                    instrument_id=row.instrument_id,
                    symbol=row.symbol,
                    check_type=CheckType.PIT_SYMBOL_WINDOW_MISMATCH,
                    severity=severity,
                    flag_reason="symbol_not_valid_for_date",
                    evidence_summary=(
                        f"Observed symbol not valid on {row.date.date()} under ticker window "
                        f"[{row.ticker_start_date}, {row.ticker_end_date}]"
                    ),
                )
            )

    return failures


def expected_membership_state(primary_reason: Optional[str]) -> str:
    if primary_reason is None or pd.isna(primary_reason):
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
    if primary_reason == ExclusionReason.DELISTED.value:
        return MembershipState.INACTIVE_POST_DEATH.value
    return MembershipState.INELIGIBLE_RULE.value


def revalidate_critical_rules(sample: pd.DataFrame, cfg: UniverseQCConfig) -> List[str]:
    rules = cfg.critical_rules
    reasons: List[str] = []
    for row in sample.itertuples(index=False):
        row_reasons: List[str] = []
        sec_type = as_str(getattr(row, "security_type", None))
        exch = as_str(getattr(row, "listing_exchange", None))
        list_date = getattr(row, "list_date", pd.NaT)
        date = getattr(row, "date", pd.NaT)
        price_ref = safe_float(getattr(row, "price_ref", None))
        adv20 = safe_float(getattr(row, "adv20_usd", None))
        mcap = safe_float(getattr(row, "market_cap_usd", None))

        if sec_type is None:
            row_reasons.append(ExclusionReason.MISSING_SECURITY_TYPE.value)
        elif sec_type.upper() not in {x.upper() for x in rules.allowed_security_types}:
            row_reasons.append(ExclusionReason.BAD_SECURITY_TYPE.value)

        if exch is None:
            row_reasons.append(ExclusionReason.MISSING_EXCHANGE.value)
        elif exch.upper() not in {x.upper() for x in rules.allowed_exchanges}:
            row_reasons.append(ExclusionReason.BAD_EXCHANGE.value)

        if pd.isna(list_date):
            row_reasons.append(ExclusionReason.MISSING_LIST_DATE.value)
        elif pd.notna(date) and int((pd.Timestamp(date) - pd.Timestamp(list_date)).days) < int(rules.min_listing_age_days):
            row_reasons.append(ExclusionReason.TOO_YOUNG_SINCE_LISTING.value)

        if price_ref is None:
            row_reasons.append(ExclusionReason.MISSING_PRICE.value)
        elif price_ref < rules.min_price_ref:
            row_reasons.append(ExclusionReason.PRICE_BELOW_MIN.value)

        if adv20 is None:
            row_reasons.append(ExclusionReason.MISSING_ADV20.value)
        elif adv20 < rules.min_adv20_usd:
            row_reasons.append(ExclusionReason.ADV20_BELOW_MIN.value)

        if mcap is None:
            row_reasons.append(ExclusionReason.MISSING_MCAP.value)
        elif mcap < rules.min_market_cap_usd or mcap > rules.max_market_cap_usd:
            row_reasons.append(ExclusionReason.MCAP_OUT_OF_BAND.value)

        reasons.append("|".join(row_reasons))
    return reasons


def run_semantic_checks(df: pd.DataFrame, cfg: UniverseQCConfig) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []

    # Binary reason <-> eligibility coherence.
    elig_but_reason = df["is_eligible"].astype(int).eq(1) & df["primary_exclusion_reason"].notna()
    inelig_no_reason = df["is_eligible"].astype(int).eq(0) & df["primary_exclusion_reason"].isna()
    for row in df.loc[elig_but_reason | inelig_no_reason, ["date", "instrument_id", "symbol", "is_eligible", "primary_exclusion_reason"]].head(500).itertuples(index=False):
        failures.append(
            make_failure(
                date=row.date,
                instrument_id=row.instrument_id,
                symbol=row.symbol,
                check_type=CheckType.SEMANTIC_ELIGIBILITY_REASON_MISMATCH,
                severity=Severity.FAIL,
                flag_reason="eligibility_reason_incoherence",
                evidence_summary=(
                    f"is_eligible={row.is_eligible}, primary_exclusion_reason={row.primary_exclusion_reason!r} violate binary semantic equivalence."
                ),
            )
        )

    # all_failed_reasons order.
    if "all_failed_reasons" in df.columns:
        parsed = df["all_failed_reasons"].map(parse_reason_list)
        parsed_first = parsed.map(lambda x: x[0] if x else pd.NA)
        bad_order = df["primary_exclusion_reason"].notna() & parsed_first.notna() & (parsed_first.astype(str) != df["primary_exclusion_reason"].astype(str))
        for row, parsed_list in zip(df.loc[bad_order, ["date", "instrument_id", "symbol", "primary_exclusion_reason"]].head(500).itertuples(index=False), parsed.loc[bad_order].head(500)):
            failures.append(
                make_failure(
                    date=row.date,
                    instrument_id=row.instrument_id,
                    symbol=row.symbol,
                    check_type=CheckType.SEMANTIC_PRIMARY_REASON_ORDER_MISMATCH,
                    severity=Severity.FAIL if cfg.validation.strict_mode else Severity.WARN,
                    flag_reason="primary_reason_not_first_failed_reason",
                    evidence_summary=f"primary_exclusion_reason={row.primary_exclusion_reason!r}, all_failed_reasons={parsed_list!r}",
                )
            )

    # membership_state coherence.
    expected_state = df["primary_exclusion_reason"].map(expected_membership_state)
    bad_state = df["membership_state"].astype(str) != expected_state.astype(str)
    for row, exp_state in zip(df.loc[bad_state, ["date", "instrument_id", "symbol", "membership_state"]].head(500).itertuples(index=False), expected_state.loc[bad_state].head(500)):
        failures.append(
            make_failure(
                date=row.date,
                instrument_id=row.instrument_id,
                symbol=row.symbol,
                check_type=CheckType.SEMANTIC_MEMBERSHIP_STATE_MISMATCH,
                severity=Severity.FAIL,
                flag_reason="membership_state_semantic_mismatch",
                evidence_summary=f"Observed membership_state={row.membership_state!r}; expected={exp_state!r} from primary reason.",
            )
        )

    # Revalidate critical rules only on rows marked eligible.
    eligible = df.loc[df["is_eligible"].astype(int).eq(1)].copy()
    if not eligible.empty:
        rng = random.Random(cfg.validation.random_seed)
        max_rows = min(cfg.validation.revalidate_max_rows, len(eligible))
        sample_frac = cfg.validation.revalidate_sample_frac
        n_take = min(max_rows, max(1, int(math.ceil(len(eligible) * sample_frac))))
        idx = list(eligible.index)
        if n_take < len(idx):
            idx = rng.sample(idx, n_take)
        sample = eligible.loc[idx].copy().sort_values(["date", "instrument_id"])  # deterministic output order
        sample["revalidated_reasons"] = revalidate_critical_rules(sample, cfg)
        violated = sample["revalidated_reasons"].astype(str).str.len() > 0
        for row in sample.loc[violated, ["date", "instrument_id", "symbol", "revalidated_reasons"]].head(1000).itertuples(index=False):
            failures.append(
                make_failure(
                    date=row.date,
                    instrument_id=row.instrument_id,
                    symbol=row.symbol,
                    check_type=CheckType.SEMANTIC_CRITICAL_RULE_REVALIDATION,
                    severity=Severity.FAIL if cfg.validation.strict_mode else Severity.WARN,
                    flag_reason="eligible_but_fails_revalidated_critical_rule",
                    evidence_summary=f"Eligible row fails independent rule revalidation: {row.revalidated_reasons}",
                )
            )

    return failures


# -----------------------------------------------------------------------------
# Temporal, coverage and calendar checks
# -----------------------------------------------------------------------------


def compute_turnover(prev_set: Set[str], curr_set: Set[str]) -> Tuple[float, int, int]:
    union = prev_set | curr_set
    enters = len(curr_set - prev_set)
    exits = len(prev_set - curr_set)
    if not union:
        return 0.0, enters, exits
    return (enters + exits) / len(union), enters, exits


def compute_delisted_expected_sets(lifecycle: pd.DataFrame, audit_dates: pd.DatetimeIndex) -> Dict[pd.Timestamp, Set[str]]:
    expected: Dict[pd.Timestamp, Set[str]] = {pd.Timestamp(d): set() for d in audit_dates}
    dead = lifecycle.loc[lifecycle["is_dead_observed"]].copy()
    if dead.empty:
        return expected
    for row in dead.itertuples(index=False):
        start = max(pd.Timestamp(row.birth_date), audit_dates.min())
        end = min(pd.Timestamp(row.death_date), audit_dates.max())
        if start > end:
            continue
        for d in audit_dates[(audit_dates >= start) & (audit_dates <= end)]:
            expected[pd.Timestamp(d)].add(str(row.instrument_id))
    return expected


def run_calendar_temporal_coverage_checks(
    df: pd.DataFrame,
    lifecycle: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    continuity_map: Dict[str, Set[str]],
    cfg: UniverseQCConfig,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    failures: List[Dict[str, Any]] = []

    observed_dates = pd.DatetimeIndex(df["date"].sort_values().unique())
    calendar_set = set(pd.Timestamp(x) for x in calendar)
    bad_dates = [d for d in observed_dates if pd.Timestamp(d) not in calendar_set]
    for d in bad_dates:
        failures.append(
            make_failure(
                date=d,
                instrument_id=pd.NA,
                symbol=pd.NA,
                check_type=CheckType.CALENDAR_INVALID_SESSION,
                severity=Severity.FAIL,
                flag_reason="date_not_in_market_calendar",
                evidence_summary=f"Observed universe session {pd.Timestamp(d).date()} is not present in official calendar.",
            )
        )

    daily_records: List[Dict[str, Any]] = []
    prev_eligible: Set[str] = set()
    eligible_sets_by_date: Dict[pd.Timestamp, Set[str]] = {}
    reason_counts = (
        df.assign(primary_exclusion_reason=lambda x: x["primary_exclusion_reason"].fillna("__NONE__"))
        .groupby(["date", "primary_exclusion_reason"], as_index=False)
        .size()
        .rename(columns={"size": "n_reason"})
    )
    pivot_reason = reason_counts.pivot(index="date", columns="primary_exclusion_reason", values="n_reason").fillna(0)
    expected_dead = compute_delisted_expected_sets(lifecycle, observed_dates)

    for i, date in enumerate(observed_dates):
        day = df.loc[df["date"] == date].copy()
        eligible = set(day.loc[day["is_eligible"].astype(int).eq(1), "instrument_id"].astype(str))
        eligible_sets_by_date[pd.Timestamp(date)] = eligible
        if i == 0:
            turnover, n_enter, n_exit = 0.0, len(eligible), 0
        else:
            turnover, n_enter, n_exit = compute_turnover(prev_eligible, eligible)
        size_today = len(eligible)
        size_prev = len(prev_eligible)
        size_jump_abs = size_today - size_prev if i > 0 else 0
        size_jump_rel = 0.0 if i == 0 or size_prev == 0 else size_jump_abs / size_prev

        missing_critical = float(day[list(cfg.missing_critical_meta_cols)].isna().any(axis=1).mean()) if cfg.missing_critical_meta_cols else 0.0
        dead_expected = expected_dead.get(pd.Timestamp(date), set())
        dead_appearing = len(dead_expected & set(day["instrument_id"].astype(str)))
        delisted_coverage_ratio = np.nan if len(dead_expected) == 0 else dead_appearing / len(dead_expected)

        daily_records.append(
            {
                "date": pd.Timestamp(date),
                "n_rows": int(len(day)),
                "n_eligible": int(size_today),
                "n_ineligible": int((day["is_eligible"].astype(int) == 0).sum()),
                "turnover": float(turnover),
                "n_enter": int(n_enter),
                "n_exit": int(n_exit),
                "size_jump_abs": int(size_jump_abs),
                "size_jump_rel": float(size_jump_rel),
                "pct_missing_critical_meta": float(missing_critical),
                "n_expected_dead": int(len(dead_expected)),
                "n_dead_appearing": int(dead_appearing),
                "delisted_coverage_ratio": float(delisted_coverage_ratio) if pd.notna(delisted_coverage_ratio) else np.nan,
            }
        )
        prev_eligible = eligible

    daily = pd.DataFrame(daily_records).sort_values("date").reset_index(drop=True)
    if daily.empty:
        raise UniverseQCError("Universe history has no rows after loading.")

    # Missing sessions between min/max observed dates.
    expected_between = calendar[(calendar >= observed_dates.min()) & (calendar <= observed_dates.max())]
    missing_sessions = [d for d in expected_between if d not in observed_dates]
    for d in missing_sessions:
        failures.append(
            make_failure(
                date=d,
                instrument_id=pd.NA,
                symbol=pd.NA,
                check_type=CheckType.CALENDAR_MISSING_SESSION,
                severity=Severity.FAIL,
                flag_reason="missing_expected_session",
                evidence_summary=f"Missing expected calendar session {pd.Timestamp(d).date()} between observed universe endpoints.",
            )
        )

    # High/extreme turnover and size jumps.
    for row in daily.itertuples(index=False):
        if row.turnover > cfg.thresholds.tau_turnover_fail:
            failures.append(
                make_failure(
                    date=row.date,
                    instrument_id=pd.NA,
                    symbol=pd.NA,
                    check_type=CheckType.TEMPORAL_EXTREME_TURNOVER,
                    severity=Severity.FAIL,
                    flag_reason="turnover_above_fail_threshold",
                    evidence_summary=f"turnover={row.turnover:.4f} > tau_turnover_fail={cfg.thresholds.tau_turnover_fail:.4f}",
                )
            )
        elif row.turnover > cfg.thresholds.tau_turnover_warn:
            failures.append(
                make_failure(
                    date=row.date,
                    instrument_id=pd.NA,
                    symbol=pd.NA,
                    check_type=CheckType.TEMPORAL_HIGH_TURNOVER,
                    severity=Severity.WARN,
                    flag_reason="turnover_above_warn_threshold",
                    evidence_summary=f"turnover={row.turnover:.4f} > tau_turnover_warn={cfg.thresholds.tau_turnover_warn:.4f}",
                )
            )
        if abs(row.size_jump_rel) > cfg.thresholds.tau_size_jump_fail:
            failures.append(
                make_failure(
                    date=row.date,
                    instrument_id=pd.NA,
                    symbol=pd.NA,
                    check_type=CheckType.TEMPORAL_EXTREME_SIZE_JUMP,
                    severity=Severity.FAIL,
                    flag_reason="size_jump_above_fail_threshold",
                    evidence_summary=f"size_jump_rel={row.size_jump_rel:.4f} exceeds fail threshold {cfg.thresholds.tau_size_jump_fail:.4f}",
                )
            )
        elif abs(row.size_jump_rel) > cfg.thresholds.tau_size_jump_warn:
            failures.append(
                make_failure(
                    date=row.date,
                    instrument_id=pd.NA,
                    symbol=pd.NA,
                    check_type=CheckType.TEMPORAL_HIGH_SIZE_JUMP,
                    severity=Severity.WARN,
                    flag_reason="size_jump_above_warn_threshold",
                    evidence_summary=f"size_jump_rel={row.size_jump_rel:.4f} exceeds warn threshold {cfg.thresholds.tau_size_jump_warn:.4f}",
                )
            )
        if pd.notna(row.delisted_coverage_ratio) and row.delisted_coverage_ratio < cfg.thresholds.delisted_coverage_warn:
            failures.append(
                make_failure(
                    date=row.date,
                    instrument_id=pd.NA,
                    symbol=pd.NA,
                    check_type=CheckType.COVERAGE_LOW_DELISTED_COVERAGE,
                    severity=Severity.FAIL if row.delisted_coverage_ratio < cfg.thresholds.delisted_coverage_warn else Severity.WARN,
                    flag_reason="low_delisted_coverage_ratio",
                    evidence_summary=f"delisted_coverage_ratio={row.delisted_coverage_ratio:.4f}",
                )
            )

    # Reason spikes: compare daily counts with rolling median of previous 20 observations.
    if not pivot_reason.empty:
        pivot_reason = pivot_reason.sort_index()
        for reason in [c for c in pivot_reason.columns if c != "__NONE__"]:
            series = pivot_reason[reason].astype(float)
            baseline = series.shift(1).rolling(20, min_periods=5).median()
            ratio = series / baseline.replace({0.0: np.nan})
            for date, obs, base, rat in zip(series.index, series.values, baseline.values, ratio.values):
                if np.isnan(base) or obs <= 0:
                    continue
                if rat >= cfg.thresholds.reason_spike_fail_multiple:
                    failures.append(
                        make_failure(
                            date=date,
                            instrument_id=pd.NA,
                            symbol=pd.NA,
                            check_type=CheckType.TEMPORAL_REASON_SPIKE,
                            severity=Severity.FAIL,
                            flag_reason=f"reason_spike_{reason}",
                            evidence_summary=f"reason={reason}, count={obs:.0f}, rolling_median={base:.2f}, multiple={rat:.2f}",
                        )
                    )
                elif rat >= cfg.thresholds.reason_spike_warn_multiple:
                    failures.append(
                        make_failure(
                            date=date,
                            instrument_id=pd.NA,
                            symbol=pd.NA,
                            check_type=CheckType.TEMPORAL_REASON_SPIKE,
                            severity=Severity.WARN,
                            flag_reason=f"reason_spike_{reason}",
                            evidence_summary=f"reason={reason}, count={obs:.0f}, rolling_median={base:.2f}, multiple={rat:.2f}",
                        )
                    )

    # Reactivations after death. Fail only when instrument becomes eligible after a death period and no continuity links exist.
    temp = df.sort_values(["instrument_id", "date"]).copy()
    temp["prev_eligible"] = temp.groupby("instrument_id", sort=False)["is_eligible"].shift(1)
    reactivated = temp[
        temp["death_date"].notna()
        & (temp["date"] > temp["death_date"])
        & temp["is_eligible"].astype(int).eq(1)
    ]
    for row in reactivated[["date", "instrument_id", "symbol", "death_date", "terminal_ca_id"]].head(500).itertuples(index=False):
        if continuity_map.get(str(row.instrument_id)):
            continue
        failures.append(
            make_failure(
                date=row.date,
                instrument_id=row.instrument_id,
                symbol=row.symbol,
                check_type=CheckType.COVERAGE_REACTIVATION_AFTER_DEATH,
                severity=Severity.FAIL,
                flag_reason="reactivation_after_death_without_continuity",
                evidence_summary=f"Instrument appears eligible on {row.date.date()} after death_date {row.death_date.date()}.",
                linked_corporate_action_id=row.terminal_ca_id,
            )
        )

    # Add reason count columns to daily output.
    if not pivot_reason.empty:
        pivot_reason = pivot_reason.add_prefix("n_reason__").reset_index()
        daily = daily.merge(pivot_reason, on="date", how="left")

    return daily, failures


# -----------------------------------------------------------------------------
# Gates and summary
# -----------------------------------------------------------------------------


def build_gates(daily: pd.DataFrame, failures: pd.DataFrame, cfg: UniverseQCConfig) -> List[GateResult]:
    critical_checks = {
        CheckType.STRUCTURAL_DUPLICATE_LOGICAL_KEY.value,
        CheckType.STRUCTURAL_INVALID_IS_ELIGIBLE.value,
        CheckType.PIT_OUTSIDE_LIFECYCLE_WINDOW.value,
        CheckType.CALENDAR_INVALID_SESSION.value,
        CheckType.CALENDAR_MISSING_SESSION.value,
        CheckType.PIT_ELIGIBLE_AFTER_TERMINATION.value,
    }
    has_critical_fail = not failures.loc[
        failures["check_type"].isin(critical_checks) & failures["severity"].eq(Severity.FAIL.value)
    ].empty
    gate1 = GateResult(
        name="gate1_integrity_pit_critical",
        severity=Severity.FAIL if has_critical_fail else Severity.PASS,
        observed=int(
            failures.loc[
                failures["check_type"].isin(critical_checks) & failures["severity"].eq(Severity.FAIL.value)
            ].shape[0]
        ),
        threshold=0,
        message="Critical PIT/integrity blockers present" if has_critical_fail else "No critical PIT/integrity blockers",
    )

    semantic_checks = {
        CheckType.SEMANTIC_ELIGIBILITY_REASON_MISMATCH.value,
        CheckType.SEMANTIC_PRIMARY_REASON_ORDER_MISMATCH.value,
        CheckType.SEMANTIC_MEMBERSHIP_STATE_MISMATCH.value,
        CheckType.SEMANTIC_CRITICAL_RULE_REVALIDATION.value,
    }
    semantic_rows = failures.loc[failures["check_type"].isin(semantic_checks), ["date", "instrument_id"]].drop_duplicates()
    semantic_ratio = 0.0 if daily["n_rows"].sum() == 0 else len(semantic_rows) / int(daily["n_rows"].sum())
    if semantic_ratio >= cfg.thresholds.tau_semantic_fail:
        sem_sev = Severity.FAIL
    elif semantic_ratio >= cfg.thresholds.tau_semantic_warn:
        sem_sev = Severity.WARN
    else:
        sem_sev = Severity.PASS
    gate2 = GateResult(
        name="gate2_semantic_coherence",
        severity=sem_sev,
        observed=float(semantic_ratio),
        threshold={"warn": cfg.thresholds.tau_semantic_warn, "fail": cfg.thresholds.tau_semantic_fail},
        message="Semantic inconsistency ratio over threshold" if sem_sev != Severity.PASS else "Semantic coherence acceptable",
    )

    coverage_mean = float(daily["delisted_coverage_ratio"].dropna().mean()) if daily["delisted_coverage_ratio"].notna().any() else 1.0
    if coverage_mean < cfg.thresholds.delisted_coverage_warn:
        cov_sev = Severity.FAIL
    elif coverage_mean < cfg.thresholds.delisted_coverage_pass:
        cov_sev = Severity.WARN
    else:
        cov_sev = Severity.PASS
    gate3 = GateResult(
        name="gate3_delisted_coverage",
        severity=cov_sev,
        observed=float(coverage_mean),
        threshold={"warn": cfg.thresholds.delisted_coverage_warn, "pass": cfg.thresholds.delisted_coverage_pass},
        message="Delisted coverage below policy threshold" if cov_sev != Severity.PASS else "Delisted coverage acceptable",
    )

    pct_days_high_turnover = float((daily["turnover"] > cfg.thresholds.tau_turnover_warn).mean())
    pct_days_extreme_turnover = float((daily["turnover"] > cfg.thresholds.tau_turnover_fail).mean())
    if (
        pct_days_extreme_turnover > cfg.thresholds.pct_days_extreme_turnover_fail
        or pct_days_high_turnover > cfg.thresholds.pct_days_high_turnover_fail
    ):
        stab_sev = Severity.FAIL
    elif pct_days_high_turnover > cfg.thresholds.pct_days_high_turnover_warn:
        stab_sev = Severity.WARN
    else:
        stab_sev = Severity.PASS
    gate4 = GateResult(
        name="gate4_temporal_stability",
        severity=stab_sev,
        observed={
            "pct_days_high_turnover": pct_days_high_turnover,
            "pct_days_extreme_turnover": pct_days_extreme_turnover,
        },
        threshold={
            "warn": cfg.thresholds.pct_days_high_turnover_warn,
            "fail": {
                "pct_days_high_turnover": cfg.thresholds.pct_days_high_turnover_fail,
                "pct_days_extreme_turnover": cfg.thresholds.pct_days_extreme_turnover_fail,
            },
        },
        message="Temporal stability degraded" if stab_sev != Severity.PASS else "Temporal stability acceptable",
    )

    pct_missing_critical_meta = float(daily["pct_missing_critical_meta"].mean()) if not daily.empty else 0.0
    if pct_missing_critical_meta >= cfg.thresholds.pct_missing_critical_meta_fail:
        miss_sev = Severity.FAIL
    elif pct_missing_critical_meta >= cfg.thresholds.pct_missing_critical_meta_warn:
        miss_sev = Severity.WARN
    else:
        miss_sev = Severity.PASS
    gate5 = GateResult(
        name="gate5_missing_critical_meta",
        severity=miss_sev,
        observed=float(pct_missing_critical_meta),
        threshold={"warn": cfg.thresholds.pct_missing_critical_meta_warn, "fail": cfg.thresholds.pct_missing_critical_meta_fail},
        message="Missing critical metadata exceeds threshold" if miss_sev != Severity.PASS else "Missing critical metadata acceptable",
    )
    return [gate1, gate2, gate3, gate4, gate5]


def build_summary(
    df: pd.DataFrame,
    daily: pd.DataFrame,
    failures: pd.DataFrame,
    gates: Sequence[GateResult],
    cfg: UniverseQCConfig,
    cfg_hash: str,
    run_id: str,
) -> Dict[str, Any]:
    gate_status = max_severity([g.severity for g in gates] + [Severity(x) for x in failures["severity"].tolist()], default=Severity.PASS).value
    top_failure_types = (
        failures.groupby(["check_type", "severity"], as_index=False)
        .size()
        .sort_values(["size", "check_type"], ascending=[False, True])
        .head(15)
        .to_dict(orient="records")
    )
    return {
        "gate_status": gate_status,
        "n_rows_processed": int(len(df)),
        "n_distinct_instruments": int(df["instrument_id"].nunique()),
        "n_fail": int((failures["severity"] == Severity.FAIL.value).sum()),
        "n_warn": int((failures["severity"] == Severity.WARN.value).sum()),
        "delisted_coverage_ratio_mean": float(daily["delisted_coverage_ratio"].dropna().mean()) if daily["delisted_coverage_ratio"].notna().any() else 1.0,
        "pct_days_high_turnover": float((daily["turnover"] > cfg.thresholds.tau_turnover_warn).mean()),
        "pct_days_extreme_turnover": float((daily["turnover"] > cfg.thresholds.tau_turnover_fail).mean()),
        "pct_missing_critical_meta": float(daily["pct_missing_critical_meta"].mean()) if not daily.empty else 0.0,
        "top_failure_types": top_failure_types,
        "run_id": run_id,
        "config_hash": cfg_hash,
        "code_version": maybe_git_code_version(),
        "gates": [
            {
                "name": g.name,
                "severity": g.severity.value if isinstance(g.severity, Enum) else str(g.severity),
                "observed": g.observed,
                "threshold": g.threshold,
                "message": g.message,
            }
            for g in gates
        ],
    }


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def audit_universe(
    *,
    universe_history_path: str | Path,
    listings_master_path: str | Path,
    calendar_path: str | Path,
    corporate_actions_path: Optional[str | Path],
    run_id: str,
    config: UniverseQCConfig,
    config_hash: str,
    config_payload: Dict[str, Any],
) -> QCArtifacts:
    universe = load_universe_history(universe_history_path)
    listings = load_listings_master(listings_master_path)
    calendar = load_calendar(calendar_path)
    corporate_actions = load_corporate_actions(corporate_actions_path)

    lifecycle = build_listing_lifecycle(listings, corporate_actions)
    symbol_windows = build_symbol_windows(listings)
    continuity_map = build_continuity_map(corporate_actions)

    universe = attach_lifecycle(universe, lifecycle)

    failures_list: List[Dict[str, Any]] = []
    failures_list.extend(structural_fail_fast_checks(universe))
    failures_list.extend(run_pit_checks(universe, symbol_windows, continuity_map, config))
    failures_list.extend(run_semantic_checks(universe, config))
    daily, extra_failures = run_calendar_temporal_coverage_checks(universe, lifecycle, calendar, continuity_map, config)
    failures_list.extend(extra_failures)

    # Detect config/run id drift in the persisted universe if columns exist.
    if "config_hash" in universe.columns and universe["config_hash"].notna().any():
        seen_hashes = sorted({str(x) for x in universe["config_hash"].dropna().unique()})
        if seen_hashes != [config_hash]:
            failures_list.append(
                make_failure(
                    date=pd.NaT,
                    instrument_id=pd.NA,
                    symbol=pd.NA,
                    check_type=CheckType.STRUCTURAL_CONFIG_HASH_DRIFT,
                    severity=Severity.WARN,
                    flag_reason="config_hash_drift_in_universe_history",
                    evidence_summary=f"Observed config_hash values {seen_hashes}; current QC config_hash={config_hash}",
                )
            )
    if "run_id" in universe.columns and universe["run_id"].notna().any():
        seen_run_ids = sorted({str(x) for x in universe["run_id"].dropna().unique()})
        if len(seen_run_ids) > 1:
            failures_list.append(
                make_failure(
                    date=pd.NaT,
                    instrument_id=pd.NA,
                    symbol=pd.NA,
                    check_type=CheckType.STRUCTURAL_RUN_ID_DRIFT,
                    severity=Severity.WARN,
                    flag_reason="multiple_run_ids_present_in_universe_history",
                    evidence_summary=f"Observed run_ids in universe history: {seen_run_ids[:20]}",
                )
            )

    failures = pd.DataFrame(failures_list)
    if failures.empty:
        failures = pd.DataFrame(
            columns=[
                "date",
                "instrument_id",
                "symbol",
                "check_type",
                "severity",
                "flag_reason",
                "evidence_summary",
                "linked_corporate_action_id",
            ]
        )
    else:
        failures = failures.sort_values(["date", "severity", "check_type", "instrument_id"], na_position="last").reset_index(drop=True)

    gates = build_gates(daily, failures, config)
    summary = build_summary(universe, daily, failures, gates, config, config_hash, run_id)

    manifest = {
        "run_id": run_id,
        "generated_at_utc": utc_now_iso(),
        "code_version": maybe_git_code_version(),
        "python_version": platform.python_version(),
        "config_hash": config_hash,
        "config": config_payload,
        "inputs": {
            "universe_history_path": str(universe_history_path),
            "listings_master_path": str(listings_master_path),
            "calendar_path": str(calendar_path),
            "corporate_actions_path": None if corporate_actions_path is None else str(corporate_actions_path),
            "universe_history_sha256": sha256_file(ensure_path(universe_history_path)) if ensure_path(universe_history_path).is_file() else "directory",
            "listings_master_sha256": sha256_file(ensure_path(listings_master_path)) if ensure_path(listings_master_path).is_file() else "directory",
            "calendar_sha256": sha256_file(ensure_path(calendar_path)) if ensure_path(calendar_path).is_file() else "directory",
            "corporate_actions_sha256": (
                sha256_file(ensure_path(corporate_actions_path))
                if corporate_actions_path is not None and ensure_path(corporate_actions_path).is_file()
                else ("directory" if corporate_actions_path is not None else None)
            ),
        },
        "outputs": {
            "summary": f"{run_id}/universe_qc_summary.json",
            "daily": f"{run_id}/universe_qc_daily.parquet",
            "failures": f"{run_id}/universe_qc_failures.parquet",
            "manifest": f"{run_id}/manifest.json",
        },
        "gate_status": summary["gate_status"],
        "n_fail": summary["n_fail"],
        "n_warn": summary["n_warn"],
    }
    return QCArtifacts(daily=daily, failures=failures, summary=summary, manifest=manifest)


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------


def persist_outputs(artifacts: QCArtifacts, output_dir: str | Path, compression: str) -> None:
    ensure_parquet_engine_available()
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    artifacts.daily.to_parquet(outdir / "universe_qc_daily.parquet", index=False, compression=compression)
    artifacts.failures.to_parquet(outdir / "universe_qc_failures.parquet", index=False, compression=compression)
    (outdir / "universe_qc_summary.json").write_text(
        json.dumps(artifacts.summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (outdir / "manifest.json").write_text(
        json.dumps(artifacts.manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Certify PIT universe integrity, lifecycle consistency, semantic coherence and temporal stability."
    )
    parser.add_argument("--universe-history-path", required=True)
    parser.add_argument("--listings-master-path", required=True)
    parser.add_argument("--calendar-path", required=True)
    parser.add_argument("--corporate-actions-path", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    config, config_payload, config_hash = load_config(args.config_path)
    base_output = Path(args.output_dir or config.output.output_dir)
    run_output_dir = base_output / args.run_id

    logger.info("Loading inputs and running universe QC for run_id=%s", args.run_id)
    artifacts = audit_universe(
        universe_history_path=args.universe_history_path,
        listings_master_path=args.listings_master_path,
        calendar_path=args.calendar_path,
        corporate_actions_path=args.corporate_actions_path,
        run_id=args.run_id,
        config=config,
        config_hash=config_hash,
        config_payload=config_payload,
    )
    persist_outputs(artifacts, run_output_dir, config.output.compression)
    logger.info(
        "Universe QC completed. gate_status=%s | n_fail=%s | n_warn=%s | output=%s",
        artifacts.summary["gate_status"],
        artifacts.summary["n_fail"],
        artifacts.summary["n_warn"],
        run_output_dir,
    )
    return 0 if artifacts.summary["gate_status"] != Severity.FAIL.value else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
