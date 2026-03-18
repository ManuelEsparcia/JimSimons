"""
data.universe — PIT universe construction, corporate actions, survivorship, QC.

Pipeline: corporate_actions → build_universe → survivorship → universe_qc

Primary entity of the entire module: instrument_id (NOT symbol/ticker).
Ticker is an observational label that changes over time; instrument_id
is the stable economic identity that persists across ticker changes,
exchange moves, and name changes.
"""
from __future__ import annotations

import enum
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

class UniverseError(RuntimeError):
    """Base error for the universe pipeline."""

class BuildError(UniverseError):
    pass

class CorporateActionsError(UniverseError):
    pass

class SurvivorshipError(UniverseError):
    pass

class QCError(UniverseError):
    pass


# ---------------------------------------------------------------------------
# ExclusionReason — stable enum (spec §12)
# Names EXACTLY as specified in build_universe.txt
# ---------------------------------------------------------------------------

class ExclusionReason(str, enum.Enum):
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
    SPECIAL_RULE_EXCLUSION = "SPECIAL_RULE_EXCLUSION"


# ---------------------------------------------------------------------------
# MembershipState (spec §13)
# ---------------------------------------------------------------------------

class MembershipState(str, enum.Enum):
    ELIGIBLE = "eligible"
    INELIGIBLE_RULE = "ineligible_rule"
    INELIGIBLE_MISSING = "ineligible_missing"
    INELIGIBLE_HALTED = "ineligible_halted"
    INELIGIBLE_SUSPENDED = "ineligible_suspended"
    INACTIVE_POST_DEATH = "inactive_post_death"


# ---------------------------------------------------------------------------
# Transition (spec §13)
# ---------------------------------------------------------------------------

class Transition(str, enum.Enum):
    ENTER = "ENTER"
    EXIT = "EXIT"
    STAY_IN = "STAY_IN"
    STAY_OUT = "STAY_OUT"
    STATE_CHANGE_OUTSIDE = "STATE_CHANGE_OUTSIDE_UNIVERSE"


# ---------------------------------------------------------------------------
# Canonical corporate action taxonomy (spec corporate_actions §6)
# ---------------------------------------------------------------------------

class CanonicalEventType(str, enum.Enum):
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


class EventFamily(str, enum.Enum):
    ADJUSTMENT = "adjustment"      # split, reverse_split, stock_div, cash_div
    IDENTITY = "identity"          # ticker_change, name_change, exchange_change
    TRANSFORM = "transform"        # merger, acquisition, spinoff, rights_issue
    TERMINAL = "terminal"          # delisting, bankruptcy


# Map event types to families
EVENT_FAMILY_MAP: dict[str, str] = {
    "split": "adjustment", "reverse_split": "adjustment",
    "stock_dividend": "adjustment", "cash_dividend": "adjustment",
    "special_cash_dividend": "adjustment",
    "ticker_change": "identity", "name_change": "identity",
    "share_class_change": "identity", "exchange_change": "identity",
    "identifier_maintenance": "identity",
    "merger": "transform", "acquisition": "transform",
    "spinoff": "transform", "rights_issue": "transform",
    "tender_offer": "transform",
    "delisting": "terminal", "bankruptcy_reorg": "terminal",
    "relisting": "identity",
}


# ---------------------------------------------------------------------------
# Absence classification for survivorship (spec survivorship §10)
# ---------------------------------------------------------------------------

class AbsenceClass(str, enum.Enum):
    ECONOMIC_TERMINATION = "economic_termination"
    IDENTITY_CONTINUITY = "identity_continuity"
    LEGITIMATE_RULE_EXCLUSION = "legitimate_rule_exclusion"
    LOW_CONFIDENCE = "low_confidence"
    STRUCTURAL_MISSING = "structural_missing"


# ---------------------------------------------------------------------------
# Severity hierarchy (shared)
# ---------------------------------------------------------------------------

class Severity(str, enum.Enum):
    PASS = "PASS"
    INFO = "INFO"
    WARN = "WARN"
    FAIL = "FAIL"


_SEV_RANK = {"PASS": 0, "INFO": 1, "WARN": 2, "FAIL": 3}


def severity_rank(sev: Severity | str) -> int:
    key = sev.value if isinstance(sev, Severity) else str(sev).upper()
    return _SEV_RANK.get(key, -1)


def max_severity(values, default: str = "PASS") -> str:
    best = default
    for v in values:
        s = v.value if isinstance(v, Severity) else str(v).upper()
        if _SEV_RANK.get(s, -1) > _SEV_RANK.get(best, -1):
            best = s
    return best


# ---------------------------------------------------------------------------
# GateResult
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    gate_name: str
    status: str
    metric_value: Any
    threshold: Any
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


# ---------------------------------------------------------------------------
# Shared utilities (eliminate duplication across 4 files)
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def config_hash(cfg: Mapping[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict:
    result = dict(base)
    for k, v in update.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, pd.Timestamp): return value.isoformat() if pd.notna(value) else None
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Mapping): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(x) for x in value]
    if isinstance(value, (set, frozenset)): return sorted(str(x) for x in value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)): return None
    if isinstance(value, enum.Enum): return value.value
    return value


def read_dataframe(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet": return pd.read_parquet(p)
    if p.suffix == ".csv": return pd.read_csv(p)
    for ext in (".parquet", ".csv"):
        if p.with_suffix(ext).exists():
            return read_dataframe(p.with_suffix(ext))
    raise FileNotFoundError(f"No data file: {p}")


def write_parquet_safe(df: pd.DataFrame, path: Path, **kw) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False, compression=kw.get("compression", "snappy"))
        return str(path)
    except Exception:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return str(csv_path)


def write_json_safe(payload: Mapping[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, default=str), encoding="utf-8")
    return str(path)
