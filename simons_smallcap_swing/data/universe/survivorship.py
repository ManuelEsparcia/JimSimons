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


class SurvivorshipAuditError(RuntimeError):
    """Raised when a hard validation or invariant fails."""


class Severity(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class AbsenceClassification(str, Enum):
    LEGITIMATE_RULE_EXCLUSION = "legitimate_rule_exclusion"
    IDENTITY_CONTINUITY = "identity_continuity"
    ECONOMIC_TERMINATION = "economic_termination"
    STRUCTURAL_MISSING = "structural_missing"
    LOW_CONFIDENCE = "low_confidence"


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

STRUCTURAL_EXCLUSION_REASONS = {
    "BAD_SECURITY_TYPE",
    "BAD_EXCHANGE",
    "SPECIAL_RULE_EXCLUSION",
    "MISSING_SECURITY_TYPE",
    "MISSING_EXCHANGE",
}

DEFAULT_TRADABLE_STATUSES = ("ACTIVE", "TRADABLE")
DEFAULT_BASELINE_VARIANTS = ("current_survivors", "current_eligible", "current_tradable")


# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineConfig:
    variant: str = "current_eligible"
    tradable_statuses: Tuple[str, ...] = DEFAULT_TRADABLE_STATUSES


@dataclass(frozen=True)
class LifecycleConfig:
    required_security_types: Tuple[str, ...] = tuple()
    allowed_exchanges: Tuple[str, ...] = tuple()
    excluded_share_classes: Tuple[str, ...] = tuple()
    include_statuses: Tuple[str, ...] = tuple()
    relisting_policy: str = "new_instrument_unless_proven_continuous"
    low_confidence_when_missing_lifecycle: bool = True


@dataclass(frozen=True)
class ThresholdConfig:
    low_delisted_coverage_threshold: float = 0.95
    mean_net_gap_rate_threshold: float = 0.01
    missing_dead_threshold: int = 10
    cagr_diff_threshold: float = 0.02


@dataclass(frozen=True)
class RiskWeightsConfig:
    r1_weight: float = 0.30
    r2_weight: float = 0.20
    r3_weight: float = 0.25
    r4_weight: float = 0.15
    r5_weight: float = 0.10
    tau_gap: float = 0.01
    tau_missing: int = 10
    tau_cagr: float = 0.02


@dataclass(frozen=True)
class OutputConfig:
    output_dir: str = "data/universe/audit"
    compression: str = "snappy"
    include_problem_cases_all: bool = True


@dataclass(frozen=True)
class AuditConfig:
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    risk: RiskWeightsConfig = field(default_factory=RiskWeightsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# -----------------------------------------------------------------------------
# Dataclasses for outputs
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    name: str
    severity: Severity
    passed: bool
    threshold: Any
    observed: Any
    message: str


@dataclass(frozen=True)
class AuditArtifacts:
    daily: pd.DataFrame
    summary: Dict[str, Any]
    problem_cases: pd.DataFrame
    missing_dead: pd.DataFrame
    comparison: Optional[pd.DataFrame]
    manifest: Dict[str, Any]


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
        raise SurvivorshipAuditError(f"Input path does not exist: {path}")
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
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def ensure_parquet_engine_available() -> None:
    if importlib.util.find_spec("pyarrow") is None and importlib.util.find_spec("fastparquet") is None:
        raise SurvivorshipAuditError(
            "Parquet support requires either 'pyarrow' or 'fastparquet'. "
            "Install one of them before running survivorship.py."
        )


def normalize_date_series(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, utc=False, errors="coerce")
    return pd.Series(pd.DatetimeIndex(out).normalize(), index=series.index)


def severity_rank(sev: Severity) -> int:
    return {Severity.PASS: 0, Severity.WARN: 1, Severity.FAIL: 2}[sev]


def max_severity(values: Iterable[Severity]) -> Severity:
    result = Severity.PASS
    for value in values:
        if severity_rank(value) > severity_rank(result):
            result = value
    return result


def _deep_merge_dict(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            _deep_merge_dict(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def load_config(config_path: str | Path) -> Tuple[AuditConfig, Dict[str, Any], str]:
    path = ensure_path(config_path)
    raw_text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise SurvivorshipAuditError("PyYAML is required to parse YAML config files.")
        payload = yaml.safe_load(raw_text) or {}
    elif suffix == ".json":
        payload = json.loads(raw_text)
    else:
        raise SurvivorshipAuditError(f"Unsupported config format: {path.suffix}")

    if not isinstance(payload, Mapping):
        raise SurvivorshipAuditError("Config file must deserialize to a mapping/object.")

    merged = _deep_merge_dict(dataclasses.asdict(AuditConfig()), dict(payload))
    cfg = AuditConfig(
        baseline=BaselineConfig(**merged.get("baseline", {})),
        lifecycle=LifecycleConfig(**merged.get("lifecycle", {})),
        thresholds=ThresholdConfig(**merged.get("thresholds", {})),
        risk=RiskWeightsConfig(**merged.get("risk", {})),
        output=OutputConfig(**merged.get("output", {})),
    )
    if cfg.baseline.variant not in DEFAULT_BASELINE_VARIANTS:
        raise SurvivorshipAuditError(
            f"baseline.variant must be one of {DEFAULT_BASELINE_VARIANTS}, got {cfg.baseline.variant!r}"
        )
    cfg_hash = sha256_text(canonical_json(dataclasses.asdict(cfg)))
    return cfg, merged, cfg_hash


# -----------------------------------------------------------------------------
# IO loading
# -----------------------------------------------------------------------------


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
    "listing_exchange",
    "security_type",
    "share_class",
    "trading_status",
    "list_date",
    "delist_date",
}

REQUIRED_LIFECYCLE_COLUMNS = {
    "instrument_id",
    "symbol",
    "list_date",
}

OPTIONAL_LIFECYCLE_COLUMNS = {
    "issuer_id",
    "delist_date",
    "ticker_start_date",
    "ticker_end_date",
    "status",
    "security_type",
    "listing_exchange",
    "share_class",
    "canonical_instrument_id",
}

OPTIONAL_CA_COLUMNS = {
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


PRICE_COL_CANDIDATES = [
    "adj_close",
    "adjusted_close",
    "close_adj",
    "close",
    "price",
    "px_last",
]


def read_dataframe(path_like: str | Path) -> pd.DataFrame:
    path = ensure_path(path_like)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        ensure_parquet_engine_available()
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise SurvivorshipAuditError(f"Unsupported tabular format: {path}")


def read_maybe_partitioned(path_like: str | Path) -> pd.DataFrame:
    path = ensure_path(path_like)
    if path.is_dir():
        ensure_parquet_engine_available()
        parts = sorted(p for p in path.glob("*.parquet") if p.is_file())
        if not parts:
            raise SurvivorshipAuditError(f"Directory has no parquet parts: {path}")
        return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return read_dataframe(path)


def load_universe_history(path_like: str | Path) -> pd.DataFrame:
    df = read_maybe_partitioned(path_like).copy()
    missing = REQUIRED_UNIVERSE_COLUMNS - set(df.columns)
    if missing:
        raise SurvivorshipAuditError(f"universe_history missing required columns: {sorted(missing)}")

    df["date"] = normalize_date_series(df["date"])
    if df["date"].isna().any():
        raise SurvivorshipAuditError("universe_history contains non-parseable dates")

    df["instrument_id"] = df["instrument_id"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    if "issuer_id" in df.columns:
        df["issuer_id"] = df["issuer_id"].astype(str)
    else:
        df["issuer_id"] = pd.NA

    if not set(pd.unique(df["is_eligible"].dropna())).issubset({0, 1, False, True, np.int8(0), np.int8(1)}):
        raise SurvivorshipAuditError("universe_history.is_eligible must only contain 0/1 values")
    df["is_eligible"] = df["is_eligible"].astype("int8")

    dup = df.duplicated(subset=["date", "instrument_id"], keep=False)
    if dup.any():
        sample = df.loc[dup, ["date", "instrument_id"]].head(10)
        raise SurvivorshipAuditError(
            "universe_history violates uniqueness on (date, instrument_id). Sample:\n"
            f"{sample.to_string(index=False)}"
        )

    if "all_failed_reasons" not in df.columns:
        df["all_failed_reasons"] = None

    for col in ["list_date", "delist_date"]:
        if col in df.columns:
            df[col] = normalize_date_series(df[col])

    return df.sort_values(["date", "instrument_id"]).reset_index(drop=True)


def load_lifecycle_master(path_like: str | Path) -> pd.DataFrame:
    df = read_maybe_partitioned(path_like).copy()
    missing = REQUIRED_LIFECYCLE_COLUMNS - set(df.columns)
    if missing:
        raise SurvivorshipAuditError(f"lifecycle master missing required columns: {sorted(missing)}")

    for col in ["list_date", "delist_date", "ticker_start_date", "ticker_end_date"]:
        if col not in df.columns:
            df[col] = pd.NaT
        else:
            df[col] = normalize_date_series(df[col])
    if df["list_date"].isna().any():
        raise SurvivorshipAuditError("lifecycle master has invalid list_date values")

    df["instrument_id"] = df["instrument_id"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    if "issuer_id" not in df.columns:
        df["issuer_id"] = pd.NA
    else:
        df["issuer_id"] = df["issuer_id"].astype(str)
    if "canonical_instrument_id" not in df.columns:
        df["canonical_instrument_id"] = df["instrument_id"]
    else:
        df["canonical_instrument_id"] = df["canonical_instrument_id"].fillna(df["instrument_id"]).astype(str)

    static_cols = [
        "instrument_id",
        "issuer_id",
        "symbol",
        "list_date",
        "delist_date",
        "canonical_instrument_id",
    ]
    for col in ["status", "security_type", "listing_exchange", "share_class"]:
        if col in df.columns:
            static_cols.append(col)

    # Keep the earliest list date and earliest non-null death date per instrument.
    def _first_notna(series: pd.Series) -> Any:
        non_na = series.dropna()
        return non_na.iloc[0] if not non_na.empty else pd.NA

    grouped = (
        df.sort_values(["instrument_id", "list_date", "ticker_start_date"], na_position="last")
        .groupby("instrument_id", as_index=False)
        .agg(
            {
                "issuer_id": _first_notna,
                "symbol": _first_notna,
                "list_date": "min",
                "delist_date": lambda s: s.dropna().min() if s.dropna().size else pd.NaT,
                "canonical_instrument_id": _first_notna,
                **{
                    col: _first_notna
                    for col in ["status", "security_type", "listing_exchange", "share_class"]
                    if col in df.columns
                },
            }
        )
    )
    return grouped.sort_values("instrument_id").reset_index(drop=True)


def load_corporate_actions(path_like: Optional[str | Path]) -> pd.DataFrame:
    if not path_like:
        return pd.DataFrame(columns=sorted(OPTIONAL_CA_COLUMNS))
    df = read_maybe_partitioned(path_like).copy()
    required = {"event_type", "effective_date"}
    missing = required - set(df.columns)
    if missing:
        raise SurvivorshipAuditError(f"corporate_actions missing required columns: {sorted(missing)}")
    df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()
    df["effective_date"] = normalize_date_series(df["effective_date"])
    if df["effective_date"].isna().any():
        raise SurvivorshipAuditError("corporate_actions contains invalid effective_date values")
    for col in [
        "instrument_id",
        "from_instrument_id",
        "to_instrument_id",
        "old_instrument_id",
        "new_instrument_id",
        "predecessor_instrument_id",
        "successor_instrument_id",
        "linked_corporate_action_id",
    ]:
        if col not in df.columns:
            df[col] = pd.NA
        else:
            df[col] = df[col].astype(str)
    return df


def load_prices(path_like: Optional[str | Path]) -> Optional[pd.DataFrame]:
    if not path_like:
        return None
    df = read_maybe_partitioned(path_like).copy()
    needed = {"date", "instrument_id"}
    missing = needed - set(df.columns)
    if missing:
        raise SurvivorshipAuditError(f"prices dataset missing required columns: {sorted(missing)}")
    price_col = None
    for candidate in PRICE_COL_CANDIDATES:
        if candidate in df.columns:
            price_col = candidate
            break
    if price_col is None:
        raise SurvivorshipAuditError(
            f"prices dataset must contain one of {PRICE_COL_CANDIDATES}; got columns {list(df.columns)}"
        )
    df["date"] = normalize_date_series(df["date"])
    df["instrument_id"] = df["instrument_id"].astype(str)
    df = df[["date", "instrument_id", price_col]].rename(columns={price_col: "price"})
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "instrument_id"]).drop_duplicates(["date", "instrument_id"])
    return df.sort_values(["instrument_id", "date"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Normalization and supporting maps
# -----------------------------------------------------------------------------


def build_lifecycle_windows(lifecycle: pd.DataFrame, ca: pd.DataFrame) -> pd.DataFrame:
    windows = lifecycle.copy()
    windows["death_from_ca"] = pd.NaT

    if not ca.empty:
        terminal_rows = ca.loc[ca["event_type"].isin(TERMINAL_EVENT_TYPES)].copy()
        if not terminal_rows.empty:
            candidate_pairs: List[pd.DataFrame] = []
            for col in [
                "instrument_id",
                "from_instrument_id",
                "old_instrument_id",
                "predecessor_instrument_id",
            ]:
                if col in terminal_rows.columns:
                    tmp = terminal_rows[[col, "effective_date"]].rename(columns={col: "instrument_id"}).copy()
                    tmp = tmp[tmp["instrument_id"].notna()]
                    candidate_pairs.append(tmp)
            if candidate_pairs:
                deaths = pd.concat(candidate_pairs, ignore_index=True)
                deaths["instrument_id"] = deaths["instrument_id"].astype(str)
                death_min = deaths.groupby("instrument_id", as_index=False)["effective_date"].min()
                windows = windows.merge(
                    death_min.rename(columns={"effective_date": "death_from_ca"}),
                    on="instrument_id",
                    how="left",
                    suffixes=("", "__dup"),
                )
                if "death_from_ca__dup" in windows.columns:
                    windows["death_from_ca"] = windows["death_from_ca"].combine_first(windows["death_from_ca__dup"])
                    windows.drop(columns=["death_from_ca__dup"], inplace=True)

    windows["birth_date"] = windows["list_date"]
    windows["death_date"] = windows[["delist_date", "death_from_ca"]].min(axis=1)
    windows["is_dead_observed"] = windows["death_date"].notna()
    return windows


def build_continuity_map(ca: pd.DataFrame, lifecycle: pd.DataFrame) -> Dict[str, Set[str]]:
    continuity: Dict[str, Set[str]] = {}

    def _link(a: Optional[str], b: Optional[str]) -> None:
        if not a or not b or pd.isna(a) or pd.isna(b):
            return
        a_str, b_str = str(a), str(b)
        continuity.setdefault(a_str, set()).add(b_str)
        continuity.setdefault(b_str, set()).add(a_str)

    if not ca.empty:
        continuity_rows = ca.loc[ca["event_type"].isin(CONTINUITY_EVENT_TYPES)].copy()
        for row in continuity_rows.itertuples(index=False):
            pairs = [
                (getattr(row, "from_instrument_id", None), getattr(row, "to_instrument_id", None)),
                (getattr(row, "old_instrument_id", None), getattr(row, "new_instrument_id", None)),
                (
                    getattr(row, "predecessor_instrument_id", None),
                    getattr(row, "successor_instrument_id", None),
                ),
            ]
            for left, right in pairs:
                _link(left, right)

    if "canonical_instrument_id" in lifecycle.columns:
        for canon, group in lifecycle.groupby("canonical_instrument_id"):
            ids = [str(x) for x in group["instrument_id"].dropna().astype(str).tolist()]
            for i, left in enumerate(ids):
                for right in ids[i + 1 :]:
                    _link(left, right)
    return continuity


def parse_reasons(value: Any) -> List[str]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x not in (None, "", np.nan)]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            decoded = json.loads(text)
            if isinstance(decoded, list):
                return [str(x) for x in decoded if x not in (None, "", np.nan)]
        except Exception:
            pass
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


# -----------------------------------------------------------------------------
# Core audit computations
# -----------------------------------------------------------------------------


def build_current_snapshot(universe_history: pd.DataFrame, lifecycle_windows: pd.DataFrame) -> pd.DataFrame:
    latest_date = pd.Timestamp(universe_history["date"].max())
    current = universe_history.loc[universe_history["date"] == latest_date].copy()

    extra_current_survivors = lifecycle_windows.loc[
        (lifecycle_windows["birth_date"] <= latest_date)
        & (
            lifecycle_windows["death_date"].isna()
            | (lifecycle_windows["death_date"] >= latest_date)
        )
    ][["instrument_id", "issuer_id", "symbol", "birth_date", "death_date"]].copy()

    if not extra_current_survivors.empty:
        missing_ids = sorted(set(extra_current_survivors["instrument_id"]) - set(current["instrument_id"]))
        if missing_ids:
            extras = extra_current_survivors.loc[extra_current_survivors["instrument_id"].isin(missing_ids)].copy()
            extras["date"] = latest_date
            extras["is_eligible"] = 0
            extras["membership_state"] = "not_observed_in_history_latest"
            extras["primary_exclusion_reason"] = "MISSING_CURRENT_HISTORY_ROW"
            extras["all_failed_reasons"] = "MISSING_CURRENT_HISTORY_ROW"
            current = pd.concat([current, extras], ignore_index=True, sort=False)

    return current.sort_values(["date", "instrument_id"]).reset_index(drop=True)


def _structurally_eligible_mask(df: pd.DataFrame, config: AuditConfig) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if config.lifecycle.required_security_types and "security_type" in df.columns:
        mask &= df["security_type"].isin(config.lifecycle.required_security_types)
    if config.lifecycle.allowed_exchanges and "listing_exchange" in df.columns:
        mask &= df["listing_exchange"].isin(config.lifecycle.allowed_exchanges)
    if config.lifecycle.excluded_share_classes and "share_class" in df.columns:
        mask &= ~df["share_class"].isin(config.lifecycle.excluded_share_classes)
    if config.lifecycle.include_statuses and "status" in df.columns:
        mask &= df["status"].isin(config.lifecycle.include_statuses)
    return mask.fillna(False)


@dataclass(frozen=True)
class PreparedUniverseContext:
    universe_history: pd.DataFrame
    current_snapshot: pd.DataFrame
    lifecycle_windows: pd.DataFrame
    expected_panel: pd.DataFrame
    eligible_by_date: Dict[pd.Timestamp, Set[str]]
    universe_rows_by_key: Dict[Tuple[pd.Timestamp, str], Mapping[str, Any]]
    naive_by_date: Dict[pd.Timestamp, Set[str]]
    continuity_map: Dict[str, Set[str]]
    latest_date: pd.Timestamp



def prepare_context(
    universe_history: pd.DataFrame,
    lifecycle: pd.DataFrame,
    ca: pd.DataFrame,
    config: AuditConfig,
) -> PreparedUniverseContext:
    lifecycle_windows = build_lifecycle_windows(lifecycle, ca)
    current_snapshot = build_current_snapshot(universe_history, lifecycle_windows)
    latest_date = pd.Timestamp(universe_history["date"].max())

    expected = lifecycle_windows.loc[_structurally_eligible_mask(lifecycle_windows, config)].copy()
    expected["death_date_effective"] = expected["death_date"].fillna(pd.Timestamp.max.normalize())

    frames: List[pd.DataFrame] = []
    audit_dates = sorted(pd.unique(universe_history["date"]))
    date_index = pd.DatetimeIndex(audit_dates)
    for row in expected.itertuples(index=False):
        active_dates = date_index[
            (date_index >= pd.Timestamp(row.birth_date))
            & (
                pd.isna(row.death_date)
                | (date_index <= pd.Timestamp(row.death_date))
            )
        ]
        if active_dates.empty:
            continue
        n = len(active_dates)
        frames.append(
            pd.DataFrame(
                {
                    "date": active_dates.values,
                    "instrument_id": [row.instrument_id] * n,
                    "issuer_id": [row.issuer_id] * n,
                    "symbol_lifecycle": [row.symbol] * n,
                    "birth_date": [row.birth_date] * n,
                    "death_date": [row.death_date] * n,
                    "is_dead_observed": [bool(row.is_dead_observed)] * n,
                }
            )
        )
    expected_panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "instrument_id", "issuer_id", "symbol_lifecycle", "birth_date", "death_date", "is_dead_observed"]
    )

    eligible_rows = universe_history.loc[universe_history["is_eligible"] == 1, ["date", "instrument_id"]].copy()
    eligible_by_date = (
        eligible_rows.groupby("date")["instrument_id"].apply(lambda s: set(s.astype(str))).to_dict()
        if not eligible_rows.empty
        else {}
    )

    universe_rows_by_key: Dict[Tuple[pd.Timestamp, str], Mapping[str, Any]] = {
        (pd.Timestamp(row.date), str(row.instrument_id)): row._asdict() if hasattr(row, "_asdict") else {}
        for row in universe_history.itertuples(index=False)
    }
    if not universe_rows_by_key:
        for _, row in universe_history.iterrows():
            universe_rows_by_key[(pd.Timestamp(row["date"]), str(row["instrument_id"]))] = row.to_dict()

    naive_by_date = build_naive_universe_by_date(current_snapshot, expected_panel, latest_date, config)
    continuity_map = build_continuity_map(ca, lifecycle_windows)

    return PreparedUniverseContext(
        universe_history=universe_history,
        current_snapshot=current_snapshot,
        lifecycle_windows=lifecycle_windows,
        expected_panel=expected_panel,
        eligible_by_date=eligible_by_date,
        universe_rows_by_key=universe_rows_by_key,
        naive_by_date=naive_by_date,
        continuity_map=continuity_map,
        latest_date=latest_date,
    )



def build_naive_universe_by_date(
    current_snapshot: pd.DataFrame,
    expected_panel: pd.DataFrame,
    latest_date: pd.Timestamp,
    config: AuditConfig,
) -> Dict[pd.Timestamp, Set[str]]:
    audit_dates = sorted(pd.unique(expected_panel["date"])) if not expected_panel.empty else sorted(pd.unique(current_snapshot["date"]))
    if not audit_dates:
        return {}

    variant = config.baseline.variant
    if variant == "current_eligible":
        base_set = set(current_snapshot.loc[current_snapshot["is_eligible"] == 1, "instrument_id"].astype(str))
    elif variant == "current_tradable":
        base_set = set(
            current_snapshot.loc[
                (current_snapshot["is_eligible"] == 1)
                & current_snapshot.get("trading_status", pd.Series(index=current_snapshot.index, dtype=object)).isin(
                    config.baseline.tradable_statuses
                ),
                "instrument_id",
            ].astype(str)
        )
    elif variant == "current_survivors":
        # Survivors = economically alive at the latest audit date, independent of current rule eligibility.
        base_set = set(current_snapshot["instrument_id"].astype(str))
    else:  # pragma: no cover - validated earlier
        raise SurvivorshipAuditError(f"Unknown baseline variant: {variant}")

    return {pd.Timestamp(dt): set(base_set) for dt in audit_dates}



def classify_absence(
    date: pd.Timestamp,
    instrument_id: str,
    ctx: PreparedUniverseContext,
    config: AuditConfig,
) -> Tuple[AbsenceClassification, str, Optional[str], Optional[str]]:
    key = (pd.Timestamp(date), str(instrument_id))
    row = ctx.universe_rows_by_key.get(key)
    current_present_ids = ctx.eligible_by_date.get(pd.Timestamp(date), set())

    lifecycle_row = ctx.lifecycle_windows.loc[ctx.lifecycle_windows["instrument_id"] == instrument_id]
    lifecycle = lifecycle_row.iloc[0].to_dict() if not lifecycle_row.empty else None

    if lifecycle is not None:
        death_date = lifecycle.get("death_date")
        if pd.notna(death_date) and pd.Timestamp(date) > pd.Timestamp(death_date):
            return (
                AbsenceClassification.ECONOMIC_TERMINATION,
                f"date {date.date()} is after economic death date {pd.Timestamp(death_date).date()}",
                None,
                None,
            )

    linked_id = None
    continuity_neighbors = ctx.continuity_map.get(str(instrument_id), set())
    for candidate in continuity_neighbors:
        if candidate in current_present_ids:
            linked_id = candidate
            return (
                AbsenceClassification.IDENTITY_CONTINUITY,
                f"continuity link maps {instrument_id} to active instrument {candidate} on {date.date()}",
                candidate,
                None,
            )

    if row is not None:
        primary_reason = row.get("primary_exclusion_reason")
        reasons = parse_reasons(row.get("all_failed_reasons"))
        if int(row.get("is_eligible", 0)) == 0:
            if primary_reason and str(primary_reason) != "nan":
                return (
                    AbsenceClassification.LEGITIMATE_RULE_EXCLUSION,
                    f"instrument observed in PIT universe history as ineligible with reason {primary_reason}",
                    None,
                    str(primary_reason),
                )
            if reasons:
                return (
                    AbsenceClassification.LEGITIMATE_RULE_EXCLUSION,
                    f"instrument observed in PIT universe history as ineligible with reasons {reasons}",
                    None,
                    str(reasons[0]),
                )

    if lifecycle is None:
        if config.lifecycle.low_confidence_when_missing_lifecycle:
            return (
                AbsenceClassification.LOW_CONFIDENCE,
                "instrument absent and lifecycle evidence missing",
                None,
                None,
            )
        return (
            AbsenceClassification.STRUCTURAL_MISSING,
            "instrument absent and lifecycle master has no corresponding record",
            None,
            None,
        )

    if pd.isna(lifecycle.get("birth_date")):
        return (
            AbsenceClassification.LOW_CONFIDENCE,
            "instrument absent with incomplete birth_date evidence in lifecycle master",
            None,
            None,
        )

    return (
        AbsenceClassification.STRUCTURAL_MISSING,
        "instrument expected historically but missing from PIT eligible set without valid explanation",
        None,
        None,
    )



def build_problem_cases(ctx: PreparedUniverseContext, config: AuditConfig) -> pd.DataFrame:
    problems: List[Dict[str, Any]] = []
    expected_panel = ctx.expected_panel.sort_values(["date", "instrument_id"]).reset_index(drop=True)
    if expected_panel.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "instrument_id",
                "issuer_id",
                "symbol",
                "classification",
                "classification_detail",
                "linked_instrument_id",
                "primary_exclusion_reason",
                "is_dead_expected",
                "gross_survivorship_gap",
                "net_survivorship_gap",
            ]
        )

    for row in expected_panel.itertuples(index=False):
        date = pd.Timestamp(row.date)
        instrument_id = str(row.instrument_id)
        eligible_set = ctx.eligible_by_date.get(date, set())
        is_present_in_pit = instrument_id in eligible_set

        classification = None
        detail = None
        linked_instrument_id = None
        primary_reason = None
        if not is_present_in_pit:
            classification, detail, linked_instrument_id, primary_reason = classify_absence(
                date=date,
                instrument_id=instrument_id,
                ctx=ctx,
                config=config,
            )
        else:
            classification = None
            detail = None
            linked_instrument_id = None
            primary_reason = None

        gross_gap = 0 if is_present_in_pit else 1
        net_gap = 1 if classification == AbsenceClassification.STRUCTURAL_MISSING else 0

        problems.append(
            {
                "date": date,
                "instrument_id": instrument_id,
                "issuer_id": row.issuer_id,
                "symbol": row.symbol_lifecycle,
                "classification": classification.value if isinstance(classification, AbsenceClassification) else None,
                "classification_detail": detail,
                "linked_instrument_id": linked_instrument_id,
                "primary_exclusion_reason": primary_reason,
                "is_dead_expected": bool(row.is_dead_observed),
                "gross_survivorship_gap": gross_gap,
                "net_survivorship_gap": net_gap,
            }
        )

    return pd.DataFrame(problems)



def compute_daily_metrics(problem_cases: pd.DataFrame, ctx: PreparedUniverseContext) -> pd.DataFrame:
    all_dates = sorted(pd.unique(problem_cases["date"])) if not problem_cases.empty else sorted(ctx.naive_by_date)
    rows: List[Dict[str, Any]] = []

    for date in all_dates:
        date = pd.Timestamp(date)
        pit_set = ctx.eligible_by_date.get(date, set())
        naive_set = ctx.naive_by_date.get(date, set())
        union = pit_set | naive_set
        overlap = (len(pit_set & naive_set) / len(union)) if union else np.nan

        problems_t = problem_cases.loc[problem_cases["date"] == date].copy()
        expected_count = int(len(problems_t))
        dead_t = problems_t.loc[problems_t["is_dead_expected"] == True].copy()  # noqa: E712
        n_dead_expected = int(len(dead_t))
        n_missing_dead = int((dead_t["classification"] == AbsenceClassification.STRUCTURAL_MISSING.value).sum())
        n_dead_covered = int(
            (
                dead_t["classification"].isna()
                | (dead_t["classification"] != AbsenceClassification.STRUCTURAL_MISSING.value)
            ).sum()
        )
        delisted_coverage_ratio = (n_dead_covered / n_dead_expected) if n_dead_expected > 0 else np.nan
        mean_net_gap = (
            float(problems_t["net_survivorship_gap"].sum()) / expected_count if expected_count > 0 else np.nan
        )
        gross_gap_rate = (
            float(problems_t["gross_survivorship_gap"].sum()) / expected_count if expected_count > 0 else np.nan
        )

        counts_by_class = problems_t["classification"].fillna("present").value_counts(dropna=False).to_dict()

        rows.append(
            {
                "date": date,
                "n_pit": int(len(pit_set)),
                "n_naive": int(len(naive_set)),
                "n_expected": expected_count,
                "n_dead_expected": n_dead_expected,
                "overlap_ratio": overlap,
                "delisted_coverage_ratio": delisted_coverage_ratio,
                "net_gap_rate": mean_net_gap,
                "gross_gap_rate": gross_gap_rate,
                "n_missing_dead_instruments": n_missing_dead,
                "n_structural_missing": int(
                    (problems_t["classification"] == AbsenceClassification.STRUCTURAL_MISSING.value).sum()
                ),
                "n_legitimate_rule_exclusion": int(
                    (problems_t["classification"] == AbsenceClassification.LEGITIMATE_RULE_EXCLUSION.value).sum()
                ),
                "n_identity_continuity": int(
                    (problems_t["classification"] == AbsenceClassification.IDENTITY_CONTINUITY.value).sum()
                ),
                "n_economic_termination": int(
                    (problems_t["classification"] == AbsenceClassification.ECONOMIC_TERMINATION.value).sum()
                ),
                "n_low_confidence": int(
                    (problems_t["classification"] == AbsenceClassification.LOW_CONFIDENCE.value).sum()
                ),
                "absence_classification_counts": json.dumps(counts_by_class, sort_keys=True),
            }
        )

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return daily



def compute_economic_comparison(
    prices: Optional[pd.DataFrame],
    ctx: PreparedUniverseContext,
) -> Optional[pd.DataFrame]:
    if prices is None or prices.empty:
        return None

    px = prices.copy().sort_values(["instrument_id", "date"]).reset_index(drop=True)
    px["ret_1d"] = px.groupby("instrument_id", sort=False)["price"].pct_change()
    px = px.dropna(subset=["ret_1d"])
    if px.empty:
        return None

    price_rows: List[Dict[str, Any]] = []
    for date, grp in px.groupby("date", sort=True):
        date = pd.Timestamp(date)
        pit_set = ctx.eligible_by_date.get(date, set())
        naive_set = ctx.naive_by_date.get(date, set())
        grp = grp.copy()
        grp["instrument_id"] = grp["instrument_id"].astype(str)
        pit_mask = grp["instrument_id"].isin(pit_set)
        naive_mask = grp["instrument_id"].isin(naive_set)
        pit_ret = float(grp.loc[pit_mask, "ret_1d"].mean()) if pit_mask.any() else np.nan
        naive_ret = float(grp.loc[naive_mask, "ret_1d"].mean()) if naive_mask.any() else np.nan
        price_rows.append(
            {
                "date": date,
                "pit_return": pit_ret,
                "naive_return": naive_ret,
                "delta_return_naive_minus_pit": (naive_ret - pit_ret) if pd.notna(pit_ret) and pd.notna(naive_ret) else np.nan,
                "pit_constituents_with_return": int(pit_mask.sum()),
                "naive_constituents_with_return": int(naive_mask.sum()),
            }
        )

    comp = pd.DataFrame(price_rows).sort_values("date").reset_index(drop=True)
    if comp.empty:
        return None

    comp["pit_nav"] = (1.0 + comp["pit_return"].fillna(0.0)).cumprod()
    comp["naive_nav"] = (1.0 + comp["naive_return"].fillna(0.0)).cumprod()
    return comp



def _annualized_cagr(nav: pd.Series, n_periods_per_year: int = 252) -> float:
    nav = nav.dropna()
    if nav.empty or len(nav) < 2:
        return float("nan")
    total_return = float(nav.iloc[-1] / nav.iloc[0])
    years = max(len(nav) / n_periods_per_year, 1 / n_periods_per_year)
    if total_return <= 0:
        return float("nan")
    return total_return ** (1 / years) - 1



def _annualized_sharpe(rets: pd.Series, n_periods_per_year: int = 252) -> float:
    rets = rets.dropna()
    if rets.empty or rets.std(ddof=0) == 0:
        return float("nan")
    return float(np.sqrt(n_periods_per_year) * rets.mean() / rets.std(ddof=0))



def _max_drawdown(nav: pd.Series) -> float:
    nav = nav.dropna()
    if nav.empty:
        return float("nan")
    running_max = nav.cummax()
    dd = nav / running_max - 1.0
    return float(dd.min())



def compute_aggregate_metrics(
    daily: pd.DataFrame,
    comparison: Optional[pd.DataFrame],
    config: AuditConfig,
) -> Dict[str, Any]:
    coverage_mean = float(daily["delisted_coverage_ratio"].dropna().mean()) if daily["delisted_coverage_ratio"].notna().any() else np.nan
    pct_days_low_coverage = float(
        (daily["delisted_coverage_ratio"].fillna(1.0) < config.thresholds.low_delisted_coverage_threshold).mean()
    )
    mean_net_gap_rate = float(daily["net_gap_rate"].dropna().mean()) if daily["net_gap_rate"].notna().any() else np.nan
    n_missing_dead = int(daily["n_missing_dead_instruments"].fillna(0).sum())

    cagr_pit = cagr_naive = sharpe_pit = sharpe_naive = maxdd_pit = maxdd_naive = np.nan
    cagr_diff = sharpe_diff = maxdd_diff = np.nan
    if comparison is not None and not comparison.empty:
        cagr_pit = _annualized_cagr(comparison["pit_nav"])
        cagr_naive = _annualized_cagr(comparison["naive_nav"])
        sharpe_pit = _annualized_sharpe(comparison["pit_return"])
        sharpe_naive = _annualized_sharpe(comparison["naive_return"])
        maxdd_pit = _max_drawdown(comparison["pit_nav"])
        maxdd_naive = _max_drawdown(comparison["naive_nav"])
        cagr_diff = cagr_naive - cagr_pit if pd.notna(cagr_pit) and pd.notna(cagr_naive) else np.nan
        sharpe_diff = sharpe_naive - sharpe_pit if pd.notna(sharpe_pit) and pd.notna(sharpe_naive) else np.nan
        maxdd_diff = maxdd_naive - maxdd_pit if pd.notna(maxdd_pit) and pd.notna(maxdd_naive) else np.nan

    r1 = 1.0 - coverage_mean if pd.notna(coverage_mean) else 0.0
    r2 = pct_days_low_coverage
    r3 = min(1.0, mean_net_gap_rate / config.risk.tau_gap) if pd.notna(mean_net_gap_rate) and config.risk.tau_gap > 0 else 0.0
    r4 = min(1.0, n_missing_dead / max(config.risk.tau_missing, 1))
    r5 = (
        min(1.0, abs(cagr_diff) / config.risk.tau_cagr)
        if pd.notna(cagr_diff) and config.risk.tau_cagr > 0
        else 0.0
    )
    score = 100.0 * (
        config.risk.r1_weight * r1
        + config.risk.r2_weight * r2
        + config.risk.r3_weight * r3
        + config.risk.r4_weight * r4
        + config.risk.r5_weight * r5
    )

    return {
        "delisted_coverage_ratio_mean": coverage_mean,
        "pct_days_low_delisted_coverage": pct_days_low_coverage,
        "mean_net_gap_rate": mean_net_gap_rate,
        "n_missing_dead_instruments": n_missing_dead,
        "CAGR_pit": cagr_pit,
        "CAGR_naive": cagr_naive,
        "CAGR_diff_pit_vs_naive": cagr_diff,
        "Sharpe_pit": sharpe_pit,
        "Sharpe_naive": sharpe_naive,
        "Sharpe_diff_pit_vs_naive": sharpe_diff,
        "MaxDD_pit": maxdd_pit,
        "MaxDD_naive": maxdd_naive,
        "MaxDD_diff_pit_vs_naive": maxdd_diff,
        "survivorship_risk_score": float(score),
        "risk_components": {
            "R1": float(r1),
            "R2": float(r2),
            "R3": float(r3),
            "R4": float(r4),
            "R5": float(r5),
        },
    }



def apply_gates(summary_metrics: Dict[str, Any], config: AuditConfig, has_prices: bool) -> List[GateResult]:
    coverage_mean = summary_metrics.get("delisted_coverage_ratio_mean")
    pct_days_low = summary_metrics.get("pct_days_low_delisted_coverage")
    mean_net_gap_rate = summary_metrics.get("mean_net_gap_rate")
    n_missing_dead = summary_metrics.get("n_missing_dead_instruments")
    cagr_diff = summary_metrics.get("CAGR_diff_pit_vs_naive")

    gates: List[GateResult] = []

    cov_pass = pd.notna(coverage_mean) and coverage_mean >= config.thresholds.low_delisted_coverage_threshold
    gates.append(
        GateResult(
            name="delisted_coverage_ratio_mean",
            severity=Severity.PASS if cov_pass else Severity.FAIL,
            passed=bool(cov_pass),
            threshold=config.thresholds.low_delisted_coverage_threshold,
            observed=coverage_mean,
            message="Mean delisted coverage must remain above configured threshold.",
        )
    )

    pct_pass = pd.notna(pct_days_low) and pct_days_low <= 0.05
    gates.append(
        GateResult(
            name="pct_days_low_delisted_coverage",
            severity=Severity.PASS if pct_pass else Severity.FAIL,
            passed=bool(pct_pass),
            threshold=0.05,
            observed=pct_days_low,
            message="Fraction of days with low delisted coverage must remain <= 5%.",
        )
    )

    gap_pass = pd.notna(mean_net_gap_rate) and mean_net_gap_rate <= config.thresholds.mean_net_gap_rate_threshold
    gates.append(
        GateResult(
            name="mean_net_gap_rate",
            severity=Severity.PASS if gap_pass else Severity.FAIL,
            passed=bool(gap_pass),
            threshold=config.thresholds.mean_net_gap_rate_threshold,
            observed=mean_net_gap_rate,
            message="Mean net survivorship gap rate must remain below threshold.",
        )
    )

    missing_pass = n_missing_dead <= config.thresholds.missing_dead_threshold
    gates.append(
        GateResult(
            name="n_missing_dead_instruments",
            severity=Severity.PASS if missing_pass else Severity.FAIL,
            passed=bool(missing_pass),
            threshold=config.thresholds.missing_dead_threshold,
            observed=n_missing_dead,
            message="Total structurally missing dead instruments must remain below threshold.",
        )
    )

    if has_prices:
        if pd.isna(cagr_diff):
            gates.append(
                GateResult(
                    name="abs_CAGR_diff_pit_vs_naive",
                    severity=Severity.WARN,
                    passed=False,
                    threshold=config.thresholds.cagr_diff_threshold,
                    observed=np.nan,
                    message="Economic comparison was requested but insufficient price overlap prevented a stable CAGR diagnostic.",
                )
            )
        else:
            cagr_pass = abs(float(cagr_diff)) <= config.thresholds.cagr_diff_threshold
            gates.append(
                GateResult(
                    name="abs_CAGR_diff_pit_vs_naive",
                    severity=Severity.PASS if cagr_pass else Severity.FAIL,
                    passed=bool(cagr_pass),
                    threshold=config.thresholds.cagr_diff_threshold,
                    observed=abs(float(cagr_diff)),
                    message="Absolute CAGR difference between naïve and PIT proxy universes must remain bounded.",
                )
            )

    return gates


# -----------------------------------------------------------------------------
# Orchestration and persistence
# -----------------------------------------------------------------------------


def run_survivorship_audit(
    universe_history: pd.DataFrame,
    lifecycle_master: pd.DataFrame,
    corporate_actions: Optional[pd.DataFrame],
    prices: Optional[pd.DataFrame],
    config: AuditConfig,
    config_hash: str,
    run_id: str,
    asof_ts_utc: str,
    input_paths: Mapping[str, str],
) -> AuditArtifacts:
    ca = corporate_actions if corporate_actions is not None else pd.DataFrame(columns=sorted(OPTIONAL_CA_COLUMNS))
    ctx = prepare_context(universe_history=universe_history, lifecycle=lifecycle_master, ca=ca, config=config)
    problem_cases = build_problem_cases(ctx, config)
    daily = compute_daily_metrics(problem_cases, ctx)
    comparison = compute_economic_comparison(prices, ctx)
    summary_metrics = compute_aggregate_metrics(daily, comparison, config)
    gates = apply_gates(summary_metrics, config, has_prices=comparison is not None and not comparison.empty)
    gate_status = max_severity(g.severity for g in gates) if gates else Severity.PASS

    missing_dead = problem_cases.loc[
        (problem_cases["is_dead_expected"] == True)  # noqa: E712
        & (problem_cases["classification"] == AbsenceClassification.STRUCTURAL_MISSING.value)
    ].copy()

    built_ts_utc = utc_now_iso()
    code_version = maybe_git_code_version()
    input_snapshot_hash = sha256_text(canonical_json(input_paths))
    problem_counts = problem_cases["classification"].fillna("present").value_counts(dropna=False).sort_index().to_dict()

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "asof_ts_utc": asof_ts_utc,
        "built_ts_utc": built_ts_utc,
        "config_hash": config_hash,
        "code_version": code_version,
        "baseline_definition": config.baseline.variant,
        "gate_status": gate_status.value,
        "n_problem_cases": int(len(problem_cases)),
        "n_expected_rows_audited": int(len(problem_cases)),
        "n_audit_days": int(daily["date"].nunique()) if not daily.empty else 0,
        **summary_metrics,
        "absence_classification_counts": problem_counts,
        "gate_results": [dataclasses.asdict(g) for g in gates],
    }

    manifest = {
        "run_id": run_id,
        "asof_ts_utc": asof_ts_utc,
        "built_ts_utc": built_ts_utc,
        "config_hash": config_hash,
        "code_version": code_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "baseline_variant": config.baseline.variant,
        "baseline_tradable_statuses": list(config.baseline.tradable_statuses),
        "relisting_policy": config.lifecycle.relisting_policy,
        "thresholds": dataclasses.asdict(config.thresholds),
        "risk_weights": dataclasses.asdict(config.risk),
        "absence_classification_catalog": [c.value for c in AbsenceClassification],
        "absence_classification_counts": problem_counts,
        "input_paths": dict(input_paths),
        "input_snapshot_hash": input_snapshot_hash,
        "gate_status": gate_status.value,
        "gate_results": [dataclasses.asdict(g) for g in gates],
        "summary_metrics": {
            k: (float(v) if isinstance(v, (np.floating, np.float64, np.float32)) else v)
            for k, v in summary_metrics.items()
        },
        "overrides": {},
    }

    return AuditArtifacts(
        daily=daily,
        summary=summary,
        problem_cases=problem_cases,
        missing_dead=missing_dead,
        comparison=comparison,
        manifest=manifest,
    )



def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")



def _write_parquet(df: pd.DataFrame, path: Path, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression=compression)



def persist_outputs(artifacts: AuditArtifacts, output_dir: str | Path, compression: str) -> None:
    ensure_parquet_engine_available()
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    _write_json(artifacts.summary, base / "survivorship_summary.json")
    _write_parquet(artifacts.daily, base / "survivorship_daily.parquet", compression)
    _write_parquet(artifacts.missing_dead, base / "missing_dead_instruments.parquet", compression)
    _write_parquet(artifacts.problem_cases, base / "survivorship_problem_cases.parquet", compression)
    if artifacts.comparison is not None and not artifacts.comparison.empty:
        _write_parquet(artifacts.comparison, base / "naive_vs_pit_comparison.parquet", compression)
    _write_json(artifacts.manifest, base / "survivorship_manifest.json")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="survivorship",
        description="Audit a PIT historical universe for survivorship bias, delisted coverage and naïve-vs-PIT drift.",
    )
    parser.add_argument("--universe-history-path", required=True, help="Path to universe history produced by build_universe.py")
    parser.add_argument("--lifecycle-master-path", required=True, help="Path to PIT lifecycle/listings master")
    parser.add_argument("--corporate-actions-path", default=None, help="Optional path to canonical corporate actions")
    parser.add_argument("--prices-path", default=None, help="Optional path to adjusted prices for PIT-vs-naïve economic comparison")
    parser.add_argument("--config-path", required=True, help="Path to YAML/JSON config for survivorship audit")
    parser.add_argument("--run-id", required=True, help="Stable run identifier")
    parser.add_argument("--asof-ts-utc", required=True, help="As-of timestamp in UTC, e.g. 2026-03-15T10:00:00Z")
    parser.add_argument("--output-dir", default=None, help="Optional override for output directory")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser



def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    cfg, _, cfg_hash = load_config(args.config_path)
    output_dir = args.output_dir or cfg.output.output_dir

    input_paths = {
        "universe_history_path": str(Path(args.universe_history_path)),
        "lifecycle_master_path": str(Path(args.lifecycle_master_path)),
        "corporate_actions_path": str(Path(args.corporate_actions_path)) if args.corporate_actions_path else None,
        "prices_path": str(Path(args.prices_path)) if args.prices_path else None,
        "config_path": str(Path(args.config_path)),
    }

    universe_history = load_universe_history(args.universe_history_path)
    lifecycle_master = load_lifecycle_master(args.lifecycle_master_path)
    corporate_actions = load_corporate_actions(args.corporate_actions_path)
    prices = load_prices(args.prices_path)

    artifacts = run_survivorship_audit(
        universe_history=universe_history,
        lifecycle_master=lifecycle_master,
        corporate_actions=corporate_actions,
        prices=prices,
        config=cfg,
        config_hash=cfg_hash,
        run_id=args.run_id,
        asof_ts_utc=args.asof_ts_utc,
        input_paths={k: v for k, v in input_paths.items() if v is not None},
    )
    persist_outputs(artifacts, output_dir=output_dir, compression=cfg.output.compression)

    logger.info("Survivorship audit completed | run_id=%s | gate_status=%s", args.run_id, artifacts.summary["gate_status"])
    logger.info("Output directory: %s", Path(output_dir).resolve())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
