from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class RunContext:
    """Small immutable run metadata shared across modules."""

    run_id: str
    asof_date: pd.Timestamp
    seed: int
    config_hash: str
    pipeline_version: str = "1.0.0"
    market: str = "US_EQ"
    parent_run_id: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("RunContext.run_id must be non-empty.")
        if not self.config_hash:
            raise ValueError("RunContext.config_hash must be non-empty.")
        ts = pd.Timestamp(self.asof_date)
        if pd.isna(ts):
            raise ValueError("RunContext.asof_date must be a valid timestamp.")
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        object.__setattr__(self, "asof_date", ts)


@dataclass(frozen=True)
class GateResult:
    """Normalized gate decision object."""

    passed: bool
    score: float
    threshold: float = 0.8
    reasons: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError("GateResult.score must belong to [0,1].")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("GateResult.threshold must belong to [0,1].")
        if self.passed != (self.score >= self.threshold):
            raise ValueError("GateResult.passed inconsistent with score >= threshold.")
        object.__setattr__(self, "reasons", tuple(self.reasons))


class ContractViolation(Exception):
    """Base class for contract-level violations."""


class DataLeakageError(ContractViolation):
    """Temporal eligibility or look-ahead violations."""


class SchemaViolation(ContractViolation):
    """Input/output structure violates expected schema contract."""


class FeasibilityError(ContractViolation):
    """Inputs are well-typed but not feasible under constraints."""


class ReproducibilityError(ContractViolation):
    """Non-deterministic behavior detected under fixed context."""


class GateSemanticsError(ContractViolation):
    """Inconsistent gate semantics or malformed gate output."""


@runtime_checkable
class DataProvider(Protocol):
    def fetch(self, asof_date: pd.Timestamp, context: RunContext) -> pd.DataFrame:
        ...


@runtime_checkable
class FeatureBuilder(Protocol):
    def build(self, raw_data: pd.DataFrame, context: RunContext) -> pd.DataFrame:
        ...


@runtime_checkable
class ModelTrainer(Protocol):
    def fit(self, features: pd.DataFrame, labels: pd.DataFrame, context: RunContext) -> Any:
        ...

    def predict(self, features: pd.DataFrame, context: RunContext) -> pd.DataFrame:
        ...


@runtime_checkable
class PortfolioEngine(Protocol):
    def rebalance(self, signals: pd.DataFrame, context: RunContext) -> pd.DataFrame:
        ...


@runtime_checkable
class Validator(Protocol):
    def validate(self, artifact: object, context: RunContext) -> GateResult:
        ...


__all__ = [
    "ContractViolation",
    "DataLeakageError",
    "DataProvider",
    "FeatureBuilder",
    "FeasibilityError",
    "GateResult",
    "GateSemanticsError",
    "ModelTrainer",
    "PortfolioEngine",
    "ReproducibilityError",
    "RunContext",
    "SchemaViolation",
    "Validator",
]
