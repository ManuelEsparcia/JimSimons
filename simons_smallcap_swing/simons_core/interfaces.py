"""
simons_core/interfaces.py — Abstract contracts for the quant pipeline.

    DataProvider    (asof_date, RunContext) → PriceData
    FeatureBuilder  (PriceData, RunContext) → FeatureMatrix
    ModelTrainer    fit/predict with reproducibility
    PortfolioEngine signals → feasible portfolio decision
    Validator       artifact → GateResult ∈ [0,1]

Plus transversal types: RunContext (immutable), GateResult (normalised).
Error hierarchy: ContractViolation → DataLeakageError, SchemaViolation, etc.

Core invariant:
    output(Stage_i) ⊨ input(Stage_{i+1})  (compositional safety)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import pandas as pd


# ── Transversal types (spec §5) ─────────────────────────────────────────────

@dataclass(frozen=True)
class RunContext:
    """Minimal immutable execution context."""
    run_id: str
    asof_date: pd.Timestamp
    seed: int
    config_hash: str
    pipeline_version: str = "1.0.0"
    market: str = "US_EQ"
    parent_run_id: str | None = None


@dataclass(frozen=True)
class GateResult:
    """Normalised validation gate. score ∈ [0,1], passed ⟺ score ≥ threshold."""
    passed: bool
    score: float
    threshold: float = 0.8
    reasons: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0,1], got {self.score}")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be in [0,1], got {self.threshold}")
        if self.passed != (self.score >= self.threshold):
            raise ValueError(f"passed={self.passed} inconsistent with score={self.score} >= threshold={self.threshold}")


# ── I/O semantic types (spec §6) ────────────────────────────────────────────

@dataclass
class PriceData:
    asof_date: pd.Timestamp
    symbols: list[str]
    prices: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class FeatureMatrix:
    asof_date: pd.Timestamp
    features: pd.DataFrame
    feature_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelArtifact:
    model_ref: str
    artifact_path: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

@dataclass
class PortfolioDecision:
    timestamp: pd.Timestamp
    weights: pd.Series
    turnover_l1: float = 0.0
    notional_exposure: float = 0.0
    constraints_active: list[str] = field(default_factory=list)


# ── Abstract contracts (spec §7-11) ─────────────────────────────────────────

class DataProvider(ABC):
    """PIT data fetch. Invariant: max(timestamp(output)) ≤ asof_date."""
    @abstractmethod
    def fetch(self, asof_date: pd.Timestamp, context: RunContext) -> PriceData:
        raise NotImplementedError

class FeatureBuilder(ABC):
    @abstractmethod
    def build(self, raw_data: PriceData, context: RunContext) -> FeatureMatrix:
        raise NotImplementedError

class ModelTrainer(ABC):
    @abstractmethod
    def fit(self, features: FeatureMatrix, labels: pd.Series, context: RunContext) -> ModelArtifact:
        raise NotImplementedError

    @abstractmethod
    def predict(self, features: FeatureMatrix, model: ModelArtifact, context: RunContext) -> pd.DataFrame:
        raise NotImplementedError

class PortfolioEngine(ABC):
    @abstractmethod
    def rebalance(self, signals: pd.DataFrame, context: RunContext) -> PortfolioDecision:
        raise NotImplementedError

class Validator(ABC):
    @abstractmethod
    def validate(self, artifact: object, context: RunContext) -> GateResult:
        raise NotImplementedError


# ── Error hierarchy (spec §12) ──────────────────────────────────────────────

class ContractViolation(Exception):
    """Base for pipeline contract violations."""

class DataLeakageError(ContractViolation):
    """Future information beyond asof_date detected."""

class SchemaViolation(ContractViolation):
    """I/O structure doesn't match expected schema."""

class FeasibilityError(ContractViolation):
    """Portfolio decision violates declared constraints."""

class ReproducibilityError(ContractViolation):
    """Same seed/input produced inconsistent outputs."""

class GateSemanticsError(ContractViolation):
    """GateResult internally inconsistent."""
