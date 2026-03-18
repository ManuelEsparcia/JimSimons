"""
features — Canonical feature construction for the quant research pipeline.

Pipeline: microstructure → fundamentals_core → fundamentals_deltas
          → interactions → cross_sectional → feature_qc → feature_store
          (orchestrated by build_features)

Shared infrastructure lives here to eliminate duplication across submodules.
"""
from __future__ import annotations

import enum
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful simons_core integration
# ---------------------------------------------------------------------------
try:
    from simons_core.interfaces import FeatureMatrix, RunContext, GateResult
    from simons_core.math.robust import (
        median_absolute_deviation,
        robust_clip,
        contamination_fraction,
    )
    from simons_core.logging import get_logger as _core_logger

    _HAS_CORE = True
except ImportError:
    FeatureMatrix = None
    RunContext = None
    GateResult = None
    median_absolute_deviation = None
    robust_clip = None
    contamination_fraction = None
    _core_logger = None
    _HAS_CORE = False


def has_core() -> bool:
    return _HAS_CORE


def get_logger(name: str) -> Any:
    if _core_logger is not None:
        try:
            return _core_logger(name)
        except Exception:
            pass
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAD_NORMAL_CONSISTENCY: float = 1.4826
"""Consistency factor so that MAD × 1.4826 ≈ σ for Gaussian data."""

DEFAULT_WINSOR_LOWER: float = 0.01
DEFAULT_WINSOR_UPPER: float = 0.99
DEFAULT_MAD_FLOOR: float = 1e-9
DEFAULT_MIN_OBS_CS: int = 20
DEFAULT_DECISION_LAG: int = 1

PK_COLUMNS: tuple[str, ...] = ("date", "symbol")
"""Primary key of the features panel."""


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

class FeatureError(RuntimeError):
    """Base error for the features package."""


class ConfigError(FeatureError, ValueError):
    """Invalid feature configuration."""


class DataContractError(FeatureError):
    """Input table violates expected contract."""


class CoverageError(FeatureError):
    """Insufficient coverage for feature computation."""


class LeakageError(FeatureError):
    """Temporal leakage detected."""


# ---------------------------------------------------------------------------
# Severity / Gate
# ---------------------------------------------------------------------------

class Severity(str, enum.Enum):
    PASS = "PASS"
    INFO = "INFO"
    WARN = "WARN"
    FAIL = "FAIL"


def severity_rank(sev: Severity | str) -> int:
    key = sev.value if isinstance(sev, Severity) else str(sev).upper()
    return {"PASS": 0, "INFO": 1, "WARN": 2, "FAIL": 3}.get(key, -1)


# ---------------------------------------------------------------------------
# Feature families
# ---------------------------------------------------------------------------

class FeatureFamily(str, enum.Enum):
    """Canonical feature family taxonomy."""
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    VOLATILITY = "volatility"
    MICROSTRUCTURE = "microstructure"
    LIQUIDITY = "liquidity"
    FUNDAMENTAL_LEVEL = "fundamental_level"
    FUNDAMENTAL_DELTA = "fundamental_delta"
    BORROW = "borrow"
    INTERACTION = "interaction"
    META = "meta"  # coverage, staleness, quality flags


# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureDef:
    """Metadata for a single feature column."""
    name: str
    family: FeatureFamily
    formula: str
    lookback_days: int = 0
    decision_lag: int = DEFAULT_DECISION_LAG
    source: str = "ohlcv"
    is_cross_sectional: bool = False
    version: str = "1.0"

    @property
    def min_history_required(self) -> int:
        return self.lookback_days + self.decision_lag


# ---------------------------------------------------------------------------
# QC record
# ---------------------------------------------------------------------------

@dataclass
class QCRecord:
    """Structured QC check result for features."""
    check_name: str
    severity: Severity
    status: str  # PASS / FAIL
    feature_name: str
    metric_value: Any
    threshold: Any
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_name": self.check_name,
            "severity": self.severity.value,
            "status": self.status,
            "feature_name": self.feature_name,
            "metric_value": _json_safe(self.metric_value),
            "threshold": _json_safe(self.threshold),
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas types to JSON-serializable."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(x) for x in value]
    return value


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def content_hash(df: pd.DataFrame) -> str:
    """Compute a reproducible hash of a DataFrame's content."""
    blob = pd.util.hash_pandas_object(df, index=False).values.tobytes()
    return hashlib.sha256(blob).hexdigest()


def validate_panel_pk(df: pd.DataFrame) -> None:
    """Ensure (date, symbol) is unique."""
    if df.duplicated(subset=list(PK_COLUMNS)).any():
        n_dups = df.duplicated(subset=list(PK_COLUMNS)).sum()
        raise DataContractError(
            f"Panel has {n_dups} duplicate (date, symbol) rows"
        )


def resolve_date_col(df: pd.DataFrame) -> str:
    """Find the date column."""
    for c in ("date", "trade_date", "asof_date", "session_date"):
        if c in df.columns:
            return c
    raise DataContractError("No date column found")


def resolve_symbol_col(df: pd.DataFrame) -> str:
    """Find the symbol column."""
    for c in ("symbol", "ticker", "instrument_id"):
        if c in df.columns:
            return c
    raise DataContractError("No symbol column found")
