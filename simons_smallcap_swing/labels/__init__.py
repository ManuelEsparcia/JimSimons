"""
labels — Canonical target construction for supervised learning.

Pipeline:
    build_labels → neutralized_targets → label_qc → purged_splits → models/*

This module defines WHAT the model learns, under WHAT execution convention,
with WHAT frictions, and with WHAT temporal validity rules.

The primary target of the system is:
    y_fwd_ret_net_10d  (forward 10-day return, net of transaction costs)

Everything else (rankings, classifications, neutralized residuals) are
derived transformations of this base target.
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
# Exclusion hierarchy (spec build_labels §17) — strict precedence order
# ---------------------------------------------------------------------------

class LabelExclusionReason(str, enum.Enum):
    OUT_OF_SAMPLE_DATE = "OUT_OF_SAMPLE_DATE"
    NOT_IN_UNIVERSE = "NOT_IN_UNIVERSE"
    MISSING_ENTRY_PRICE = "MISSING_ENTRY_PRICE"
    MISSING_EXIT_PRICE = "MISSING_EXIT_PRICE"
    INCOMPLETE_FORWARD_WINDOW = "INCOMPLETE_FORWARD_WINDOW"
    CORPORATE_ACTION_UNSAFE = "CORPORATE_ACTION_UNSAFE"
    TRADING_HALT_OR_SUSPENSION = "TRADING_HALT_OR_SUSPENSION"
    DELIST_AMBIGUOUS = "DELIST_AMBIGUOUS"
    MISSING_COST_INPUT = "MISSING_COST_INPUT"
    MISSING_EXPOSURES = "MISSING_EXPOSURES"
    NEUTRALIZATION_FAILED = "NEUTRALIZATION_FAILED"
    QC_REJECTED = "QC_REJECTED"


# Ordered by priority (index 0 = highest priority)
EXCLUSION_PRIORITY = [r.value for r in LabelExclusionReason]


class EventExclusionReason(str, enum.Enum):
    """Event-label specific exclusions (spec event_labels §20)."""
    UNKNOWN_EVENT_TYPE = "UNKNOWN_EVENT_TYPE"
    TIMESTAMP_AMBIGUOUS = "TIMESTAMP_AMBIGUOUS"
    NOT_IN_UNIVERSE = "NOT_IN_UNIVERSE"
    MISSING_ENTRY_PRICE = "MISSING_ENTRY_PRICE"
    MISSING_EXIT_PRICE = "MISSING_EXIT_PRICE"
    INCOMPLETE_OUTCOME_WINDOW = "INCOMPLETE_OUTCOME_WINDOW"
    OVERLAP_REJECTED = "OVERLAP_REJECTED"
    TRADING_HALT_OR_SUSPENSION = "TRADING_HALT_OR_SUSPENSION"
    CORPORATE_ACTION_UNSAFE = "CORPORATE_ACTION_UNSAFE"
    DELIST_AMBIGUOUS = "DELIST_AMBIGUOUS"
    MISSING_COST_INPUT = "MISSING_COST_INPUT"
    BENCHMARK_UNAVAILABLE = "BENCHMARK_UNAVAILABLE"
    QC_REJECTED = "QC_REJECTED"


# ---------------------------------------------------------------------------
# Event taxonomy (spec event_labels §5)
# ---------------------------------------------------------------------------

class EventFamily(str, enum.Enum):
    EARNINGS = "EARNINGS"
    CORP_NEWS = "CORP_NEWS"
    OPEN_GAP = "OPEN_GAP"
    TECH_BREAK = "TECH_BREAK"
    VOL_SHOCK = "VOL_SHOCK"
    MICROSTRUCTURE_SHOCK = "MICROSTRUCTURE_SHOCK"


# Default horizons per event family (spec §16)
DEFAULT_EVENT_HORIZONS: dict[str, tuple[int, ...]] = {
    "EARNINGS": (1, 3, 5, 10),
    "CORP_NEWS": (1, 3, 5, 10),
    "OPEN_GAP": (1, 2, 5),
    "TECH_BREAK": (3, 5, 10),
    "VOL_SHOCK": (1, 3, 5),
    "MICROSTRUCTURE_SHOCK": (1, 2, 3),
}


# ---------------------------------------------------------------------------
# Neutralization failure reasons (spec neutralized_targets §11.3)
# ---------------------------------------------------------------------------

class NeutFailure(str, enum.Enum):
    N_OBS_TOO_SMALL = "N_OBS_TOO_SMALL"
    MISSING_ESSENTIAL_EXPOSURES = "MISSING_ESSENTIAL_EXPOSURES"
    DESIGN_SINGULAR = "DESIGN_SINGULAR"
    CONDITION_NUMBER_TOO_HIGH = "CONDITION_NUMBER_TOO_HIGH"
    LOW_SECTOR_SUPPORT = "LOW_SECTOR_SUPPORT"
    ESTIMATOR_FAILED = "ESTIMATOR_FAILED"
    RESIDUAL_VARIANCE_COLLAPSED = "RESIDUAL_VARIANCE_COLLAPSED"


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------

class Severity(str, enum.Enum):
    PASS = "PASS"
    INFO = "INFO"
    WARN = "WARN"
    FAIL = "FAIL"


_SEV_RANK = {"PASS": 0, "INFO": 1, "WARN": 2, "FAIL": 3}

def severity_rank(s: str) -> int:
    return _SEV_RANK.get(str(s).upper(), -1)

def max_severity(values) -> str:
    best = "PASS"
    for v in values:
        s = v.value if isinstance(v, Severity) else str(v).upper()
        if _SEV_RANK.get(s, -1) > _SEV_RANK.get(best, -1):
            best = s
    return best


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def config_hash(cfg: Mapping[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]

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
