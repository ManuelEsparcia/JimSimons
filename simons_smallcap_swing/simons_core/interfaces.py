from __future__ import annotations

"""
simons_core.interfaces
======================

Contract-first abstract interfaces and semantic I/O types for the quant
research pipeline.

This module exists to make stages of the stack *interchangeable, testable and
composable* under explicit contracts. It intentionally does not implement data
loading, feature engineering, model fitting, portfolio optimization or gating
logic. Instead, it defines:

- small, stable execution types (`RunContext`, `GateResult`);
- semantically meaningful I/O containers (`PriceData`, `FeatureMatrix`,
  `Labels`, `ModelArtifact`, `PredictionFrame`, `PortfolioDecision`);
- abstract stage contracts (`DataProvider`, `FeatureBuilder`, `ModelTrainer`,
  `PortfolioEngine`, `Validator`);
- typed contract-level failures;
- reusable helpers for anti-look-ahead, alignment and reproducibility checks.

Design principles
-----------------
1. Contract-first.
2. Explicit semantics beyond signatures.
3. Minimal public surface.
4. Fail fast on contract violations.
5. No look-ahead as a first-class invariant.
6. Reproducibility via explicit `RunContext`.
7. Versioned contract evolution.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Generic, Mapping, Protocol, Sequence, TypeAlias, TypeVar

import pandas as pd
from pandas.api.types import is_numeric_dtype


__all__ = [
    "DEFAULT_PIPELINE_VERSION",
    "DEFAULT_MARKET",
    "CONTRACT_VERSION",
    "RunContext",
    "GateResult",
    "PriceData",
    "FeatureMatrix",
    "Labels",
    "ModelArtifact",
    "PredictionFrame",
    "PortfolioDecision",
    "ContractViolation",
    "DataLeakageError",
    "SchemaViolation",
    "FeasibilityError",
    "ReproducibilityError",
    "GateSemanticsError",
    "AlignmentError",
    "ContractVersionError",
    "StructuredLoggerLike",
    "DataProvider",
    "FeatureBuilder",
    "ModelTrainer",
    "PortfolioEngine",
    "Validator",
    "ensure_no_future_data",
    "ensure_index_aligned",
    "ensure_same_asof_date",
    "ensure_single_effective_timestamp",
    "ensure_numeric_frame",
    "ensure_numeric_series",
    "ensure_reproducible_run_context",
]


DEFAULT_PIPELINE_VERSION: Final[str] = "1.0.0"
DEFAULT_MARKET: Final[str] = "US_EQ"
CONTRACT_VERSION: Final[str] = "1.0.0"

TimestampLike: TypeAlias = pd.Timestamp | datetime | date | str
MetricsMap: TypeAlias = Mapping[str, float]
MetadataMap: TypeAlias = Mapping[str, Any]


class ContractViolation(Exception):
    """Base class for contract-level violations."""


class DataLeakageError(ContractViolation):
    """Detected future information beyond the declared as-of boundary."""


class SchemaViolation(ContractViolation):
    """Input or output structure does not satisfy the expected contract."""


class FeasibilityError(ContractViolation):
    """A portfolio decision violates declared feasibility rules."""


class ReproducibilityError(ContractViolation):
    """Same inputs/seed/context produced non-reproducible behavior."""


class GateSemanticsError(ContractViolation):
    """A gate result is internally inconsistent or ambiguous."""


class AlignmentError(ContractViolation):
    """Two or more pipeline objects are structurally or temporally misaligned."""


class ContractVersionError(ContractViolation):
    """Raised when a component advertises an incompatible contract version."""


class StructuredLoggerLike(Protocol):
    """Small logging protocol used by contract-aware implementations.

    The intent is to avoid coupling the interfaces module to a specific logging
    backend while still allowing strongly typed logger injection.
    """

    def info(self, event: str, /, **payload: Any) -> None:
        ...

    def warning(self, event: str, /, **payload: Any) -> None:
        ...

    def error(self, event: str, /, **payload: Any) -> None:
        ...


TArtifact = TypeVar("TArtifact")



def _to_timestamp(value: TimestampLike) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        return ts.tz_convert(None)
    return ts



def _freeze_mapping(mapping: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping or {}))



def _freeze_sequence(values: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(str(value) for value in (values or ()))



def _validate_metric_mapping(metrics: Mapping[str, Any], *, field_name: str) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise SchemaViolation(f"{field_name} keys must be non-empty strings.")
        if not isinstance(value, (int, float)):
            raise SchemaViolation(f"{field_name}[{key!r}] must be numeric, got {type(value).__name__}.")
        normalized[key] = float(value)
    return normalized



def ensure_no_future_data(
    frame: pd.DataFrame,
    *,
    asof_date: TimestampLike,
    timestamp_column: str = "date",
) -> None:
    """Enforce the anti-look-ahead invariant on a DataFrame.

    Parameters
    ----------
    frame:
        DataFrame containing the observations.
    asof_date:
        Latest timestamp allowed by contract.
    timestamp_column:
        Name of the column holding observation timestamps.

    Raises
    ------
    SchemaViolation
        If ``frame`` is not a DataFrame.
    DataLeakageError
        If any timestamp is strictly greater than ``asof_date``.
    """
    if not isinstance(frame, pd.DataFrame):
        raise SchemaViolation(f"Expected pandas.DataFrame, got {type(frame).__name__}.")

    if timestamp_column not in frame.columns:
        return

    cutoff = _to_timestamp(asof_date)
    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
    if timestamps.isna().any():
        raise SchemaViolation(
            f"Column {timestamp_column!r} contains non-coercible timestamps; cannot prove PIT integrity."
        )
    if bool((timestamps > cutoff).any()):
        max_observed = timestamps.max()
        raise DataLeakageError(
            f"Found data beyond asof_date={cutoff}: max({timestamp_column})={max_observed}."
        )



def ensure_index_aligned(left: pd.Index, right: pd.Index, *, label: str = "objects") -> None:
    """Require exact index equality for two pipeline objects."""
    if not left.equals(right):
        raise AlignmentError(f"Misaligned {label}: indices are not identical.")



def ensure_same_asof_date(*objects: Any) -> pd.Timestamp:
    """Require a shared ``asof_date`` attribute across contract objects.

    Returns the common timestamp on success.
    """
    if not objects:
        raise ValueError("ensure_same_asof_date requires at least one object.")

    values: list[pd.Timestamp] = []
    for obj in objects:
        if not hasattr(obj, "asof_date"):
            raise AlignmentError(f"Object {type(obj).__name__} has no asof_date attribute.")
        values.append(_to_timestamp(getattr(obj, "asof_date")))

    first = values[0]
    if any(value != first for value in values[1:]):
        raise AlignmentError("Objects do not share the same asof_date.")
    return first



def ensure_single_effective_timestamp(
    signals: pd.DataFrame,
    *,
    timestamp_column: str = "date",
) -> pd.Timestamp:
    """Require that a decision input refers to a single effective timestamp."""
    if not isinstance(signals, pd.DataFrame):
        raise SchemaViolation(f"Expected signals as DataFrame, got {type(signals).__name__}.")
    if timestamp_column not in signals.columns:
        raise SchemaViolation(f"Signals must contain timestamp column {timestamp_column!r}.")

    timestamps = pd.to_datetime(signals[timestamp_column], errors="coerce")
    if timestamps.isna().any():
        raise SchemaViolation(f"Signals column {timestamp_column!r} contains invalid timestamps.")

    unique_values = pd.Index(timestamps.unique())
    if len(unique_values) != 1:
        raise AlignmentError(
            f"Signals must refer to a single effective timestamp, found {len(unique_values)}."
        )
    return pd.Timestamp(unique_values[0])



def ensure_numeric_frame(frame: pd.DataFrame, *, label: str) -> None:
    """Require all columns of a DataFrame to be numeric."""
    if not isinstance(frame, pd.DataFrame):
        raise SchemaViolation(f"Expected {label} as DataFrame, got {type(frame).__name__}.")
    non_numeric = [str(col) for col in frame.columns if not is_numeric_dtype(frame[col])]
    if non_numeric:
        raise SchemaViolation(f"{label} contains non-numeric columns: {non_numeric}.")



def ensure_numeric_series(series: pd.Series, *, label: str) -> None:
    """Require a Series to be numeric."""
    if not isinstance(series, pd.Series):
        raise SchemaViolation(f"Expected {label} as Series, got {type(series).__name__}.")
    if not is_numeric_dtype(series):
        raise SchemaViolation(f"{label} must be numeric, got dtype {series.dtype!s}.")



def ensure_reproducible_run_context(context: "RunContext") -> None:
    """Validate the minimal reproducibility footprint of a run context."""
    if not context.run_id:
        raise ReproducibilityError("RunContext.run_id must be a non-empty string.")
    if not isinstance(context.seed, int):
        raise ReproducibilityError("RunContext.seed must be an integer.")
    if not context.config_hash:
        raise ReproducibilityError("RunContext.config_hash must be a non-empty string.")
    if not context.pipeline_version:
        raise ReproducibilityError("RunContext.pipeline_version must be a non-empty string.")


@dataclass(frozen=True)
class RunContext:
    """Minimal immutable execution context for reproducible pipeline runs.

    Parameters
    ----------
    run_id:
        Unique identifier for the current run.
    asof_date:
        Temporal upper bound for all observations consumed in the run.
    seed:
        Seed controlling pseudo-randomness in conforming implementations.
    config_hash:
        Fingerprint of the effective configuration.
    pipeline_version:
        Semantic version of the pipeline / contract bundle.
    market:
        Market or venue scope (e.g. ``US_EQ``).
    parent_run_id:
        Optional lineage pointer when the run descends from another run.
    metadata:
        Small immutable metadata mapping for tracing and observability.
    """

    run_id: str
    asof_date: pd.Timestamp
    seed: int
    config_hash: str
    pipeline_version: str = DEFAULT_PIPELINE_VERSION
    market: str = DEFAULT_MARKET
    parent_run_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "asof_date", _to_timestamp(self.asof_date))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        ensure_reproducible_run_context(self)

    def child(self, *, run_id: str, asof_date: TimestampLike | None = None, seed: int | None = None) -> "RunContext":
        """Create a lineage-preserving child context."""
        return RunContext(
            run_id=run_id,
            asof_date=self.asof_date if asof_date is None else _to_timestamp(asof_date),
            seed=self.seed if seed is None else seed,
            config_hash=self.config_hash,
            pipeline_version=self.pipeline_version,
            market=self.market,
            parent_run_id=self.run_id,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "run_id": self.run_id,
            "asof_date": self.asof_date.isoformat(),
            "seed": self.seed,
            "config_hash": self.config_hash,
            "pipeline_version": self.pipeline_version,
            "market": self.market,
            "parent_run_id": self.parent_run_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GateResult:
    """Normalized validation decision.

    Contract invariants
    -------------------
    - ``score`` belongs to ``[0, 1]``.
    - ``threshold`` belongs to ``[0, 1]``.
    - ``passed`` is equivalent to ``score >= threshold``.
    - ``metrics`` is numeric and audit-friendly.
    """

    passed: bool
    score: float
    threshold: float = 0.8
    reasons: tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        score = float(self.score)
        threshold = float(self.threshold)
        metrics = _validate_metric_mapping(self.metrics, field_name="GateResult.metrics")

        if not 0.0 <= score <= 1.0:
            raise GateSemanticsError("GateResult.score must belong to [0, 1].")
        if not 0.0 <= threshold <= 1.0:
            raise GateSemanticsError("GateResult.threshold must belong to [0, 1].")
        if bool(self.passed) != (score >= threshold):
            raise GateSemanticsError("GateResult.passed is inconsistent with score >= threshold.")

        object.__setattr__(self, "score", score)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "reasons", _freeze_sequence(self.reasons))
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @classmethod
    def from_score(
        cls,
        score: float,
        *,
        threshold: float = 0.8,
        reasons: Sequence[str] = (),
        metrics: Mapping[str, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "GateResult":
        """Build a gate result from the normalized score convention."""
        normalized_score = float(score)
        normalized_threshold = float(threshold)
        return cls(
            passed=normalized_score >= normalized_threshold,
            score=normalized_score,
            threshold=normalized_threshold,
            reasons=tuple(str(reason) for reason in reasons),
            metrics=metrics or {},
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class PriceData:
    """Point-in-time price-like input to downstream stages.

    Expected shape
    --------------
    ``prices`` is a DataFrame containing at least one row and usually a
    timestamp column plus one or more instrument columns/identifiers.

    Contract notes
    --------------
    This object intentionally does not prescribe a full schema for the DataFrame
    because that belongs to the dedicated schemas module. It does, however,
    enforce the core anti-look-ahead invariant and a minimal semantic payload.
    """

    asof_date: pd.Timestamp
    prices: pd.DataFrame
    symbols: tuple[str, ...] = ()
    timestamp_column: str = "date"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prices, pd.DataFrame):
            raise SchemaViolation(f"PriceData.prices must be a DataFrame, got {type(self.prices).__name__}.")
        if self.prices.empty:
            raise SchemaViolation("PriceData.prices must not be empty.")

        object.__setattr__(self, "asof_date", _to_timestamp(self.asof_date))
        object.__setattr__(self, "symbols", _freeze_sequence(self.symbols))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

        ensure_no_future_data(
            self.prices,
            asof_date=self.asof_date,
            timestamp_column=self.timestamp_column,
        )

    @property
    def n_rows(self) -> int:
        return int(len(self.prices))


@dataclass(frozen=True)
class FeatureMatrix:
    """Point-in-time aligned feature matrix.

    ``features`` should typically be indexed by the modeling unit (for example,
    date × instrument) and contain only the columns intended for a trainer.
    Any identifier columns that remain should be explicitly documented in
    metadata or handled before model fitting.
    """

    asof_date: pd.Timestamp
    features: pd.DataFrame
    feature_names: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.features, pd.DataFrame):
            raise SchemaViolation(
                f"FeatureMatrix.features must be a DataFrame, got {type(self.features).__name__}."
            )
        if self.features.empty:
            raise SchemaViolation("FeatureMatrix.features must not be empty.")

        object.__setattr__(self, "asof_date", _to_timestamp(self.asof_date))
        object.__setattr__(self, "feature_names", _freeze_sequence(self.feature_names))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

        if self.feature_names:
            missing = [name for name in self.feature_names if name not in self.features.columns]
            if missing:
                raise SchemaViolation(
                    f"FeatureMatrix.feature_names contains columns absent from features: {missing}."
                )
            ensure_numeric_frame(self.features[list(self.feature_names)], label="FeatureMatrix.features")

    @property
    def shape(self) -> tuple[int, int]:
        return self.features.shape

    def feature_frame(self) -> pd.DataFrame:
        """Return the effective model feature frame.

        If ``feature_names`` is declared, only those columns are returned;
        otherwise the whole DataFrame is returned.
        """
        if self.feature_names:
            return self.features.loc[:, list(self.feature_names)]
        return self.features


@dataclass(frozen=True)
class Labels:
    """Target vector aligned to a feature matrix."""

    asof_date: pd.Timestamp
    values: pd.Series
    name: str = "label"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.values, pd.Series):
            raise SchemaViolation(f"Labels.values must be a Series, got {type(self.values).__name__}.")
        if self.values.empty:
            raise SchemaViolation("Labels.values must not be empty.")

        object.__setattr__(self, "asof_date", _to_timestamp(self.asof_date))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        ensure_numeric_series(self.values, label="Labels.values")

        if self.name and (self.values.name is None):
            object.__setattr__(self, "values", self.values.rename(self.name))


@dataclass(frozen=True)
class ModelArtifact:
    """Serializable reference to a fitted model artifact and its evidence."""

    model_ref: str
    artifact_path: str
    params: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    created_at_utc: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_ref:
            raise SchemaViolation("ModelArtifact.model_ref must be a non-empty string.")
        if not self.artifact_path:
            raise SchemaViolation("ModelArtifact.artifact_path must be a non-empty string.")

        object.__setattr__(self, "metrics", MappingProxyType(_validate_metric_mapping(self.metrics, field_name="ModelArtifact.metrics")))
        object.__setattr__(self, "params", _freeze_mapping(self.params))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        object.__setattr__(self, "created_at_utc", _to_timestamp(self.created_at_utc).to_pydatetime())

    @property
    def artifact_path_obj(self) -> Path:
        return Path(self.artifact_path)


@dataclass(frozen=True)
class PredictionFrame:
    """Prediction or signal frame produced by a model artifact.

    Required semantics
    ------------------
    The DataFrame should be aligned with the consuming `PortfolioEngine`.
    If score columns are declared, they must exist and be numeric.
    """

    asof_date: pd.Timestamp
    predictions: pd.DataFrame
    score_columns: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.predictions, pd.DataFrame):
            raise SchemaViolation(
                f"PredictionFrame.predictions must be a DataFrame, got {type(self.predictions).__name__}."
            )
        if self.predictions.empty:
            raise SchemaViolation("PredictionFrame.predictions must not be empty.")

        object.__setattr__(self, "asof_date", _to_timestamp(self.asof_date))
        object.__setattr__(self, "score_columns", _freeze_sequence(self.score_columns))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

        if self.score_columns:
            missing = [name for name in self.score_columns if name not in self.predictions.columns]
            if missing:
                raise SchemaViolation(
                    f"PredictionFrame.score_columns contains columns absent from predictions: {missing}."
                )
            ensure_numeric_frame(self.predictions[list(self.score_columns)], label="PredictionFrame.predictions")


@dataclass(frozen=True)
class PortfolioDecision:
    """Feasible portfolio decision emitted by a portfolio engine.

    The canonical semantics here are target weights, but the metadata can carry
    auxiliary execution details. The object enforces only generic feasibility
    checks that are independent from any strategy-specific constraint set.
    """

    timestamp: pd.Timestamp
    weights: pd.Series
    turnover_l1: float
    notional_exposure: float
    constraints_active: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.weights, pd.Series):
            raise SchemaViolation(
                f"PortfolioDecision.weights must be a Series, got {type(self.weights).__name__}."
            )
        if self.weights.empty:
            raise FeasibilityError("PortfolioDecision.weights must not be empty.")
        if bool(self.weights.isna().any()):
            raise FeasibilityError("PortfolioDecision.weights contains NaN values.")
        ensure_numeric_series(self.weights, label="PortfolioDecision.weights")

        timestamp = _to_timestamp(self.timestamp)
        turnover_l1 = float(self.turnover_l1)
        notional_exposure = float(self.notional_exposure)

        if turnover_l1 < 0.0:
            raise FeasibilityError("PortfolioDecision.turnover_l1 must be non-negative.")
        if notional_exposure < 0.0:
            raise FeasibilityError("PortfolioDecision.notional_exposure must be non-negative.")

        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "turnover_l1", turnover_l1)
        object.__setattr__(self, "notional_exposure", notional_exposure)
        object.__setattr__(self, "constraints_active", _freeze_sequence(self.constraints_active))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def gross_exposure(self) -> float:
        return float(self.weights.abs().sum())

    @property
    def net_exposure(self) -> float:
        return float(self.weights.sum())


class ContractStage(ABC):
    """Common base for contract-aware pipeline stages."""

    contract_name: str = "stage"
    contract_version: str = CONTRACT_VERSION

    def assert_compatible_version(self, expected: str = CONTRACT_VERSION) -> None:
        if self.contract_version != expected:
            raise ContractVersionError(
                f"{type(self).__name__} advertises contract_version={self.contract_version!r}, "
                f"expected {expected!r}."
            )


class DataProvider(ContractStage, ABC):
    """Produce point-in-time source data for a given as-of date."""

    contract_name = "data_provider"

    @abstractmethod
    def fetch(self, asof_date: pd.Timestamp, context: RunContext) -> PriceData:
        """Fetch point-in-time data.

        Contract invariant
        ------------------
        ``max(timestamp(output)) <= asof_date`` whenever a timestamp column is
        present in the output payload.
        """
        raise NotImplementedError


class FeatureBuilder(ContractStage, ABC):
    """Transform valid source data into a feature matrix."""

    contract_name = "feature_builder"

    @abstractmethod
    def build(self, raw_data: PriceData, context: RunContext) -> FeatureMatrix:
        """Build a point-in-time aligned feature matrix."""
        raise NotImplementedError


class ModelTrainer(ContractStage, ABC):
    """Fit and score models under reproducible contracts."""

    contract_name = "model_trainer"

    @abstractmethod
    def fit(
        self,
        features: FeatureMatrix,
        labels: Labels,
        context: RunContext,
    ) -> ModelArtifact:
        """Fit a reproducible model artifact from aligned features and labels."""
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        features: FeatureMatrix,
        model: ModelArtifact,
        context: RunContext,
    ) -> PredictionFrame:
        """Produce prediction scores aligned with the feature matrix."""
        raise NotImplementedError


class PortfolioEngine(ContractStage, ABC):
    """Convert signals into a feasible target portfolio decision."""

    contract_name = "portfolio_engine"

    @abstractmethod
    def rebalance(
        self,
        signals: PredictionFrame,
        context: RunContext,
    ) -> PortfolioDecision:
        """Map signals to a feasible portfolio decision."""
        raise NotImplementedError


class Validator(ContractStage, Generic[TArtifact], ABC):
    """Validate an artifact and emit a normalized gate decision."""

    contract_name = "validator"

    @abstractmethod
    def validate(self, artifact: TArtifact, context: RunContext) -> GateResult:
        """Validate an artifact and return a normalized gate result."""
        raise NotImplementedError


def validate_provider_output(output: PriceData, *, requested_asof_date: TimestampLike) -> None:
    """Reusable contract check for `DataProvider.fetch` implementations."""
    if not isinstance(output, PriceData):
        raise SchemaViolation(f"DataProvider output must be PriceData, got {type(output).__name__}.")
    requested = _to_timestamp(requested_asof_date)
    if output.asof_date != requested:
        raise AlignmentError(
            f"DataProvider returned asof_date={output.asof_date}, expected {requested}."
        )


def validate_feature_builder_output(output: FeatureMatrix, *, source: PriceData) -> None:
    """Reusable contract check for `FeatureBuilder.build` implementations."""
    if not isinstance(output, FeatureMatrix):
        raise SchemaViolation(
            f"FeatureBuilder output must be FeatureMatrix, got {type(output).__name__}."
        )
    ensure_same_asof_date(output, source)


def validate_trainer_inputs(features: FeatureMatrix, labels: Labels, context: RunContext) -> None:
    """Validate canonical trainer preconditions."""
    ensure_same_asof_date(features, labels)
    if features.asof_date > context.asof_date or labels.asof_date > context.asof_date:
        raise DataLeakageError("Trainer inputs exceed RunContext.asof_date.")
    ensure_index_aligned(features.features.index, labels.values.index, label="features and labels")


def validate_prediction_output(output: PredictionFrame, *, features: FeatureMatrix) -> None:
    """Validate canonical predictor postconditions."""
    if not isinstance(output, PredictionFrame):
        raise SchemaViolation(
            f"ModelTrainer.predict output must be PredictionFrame, got {type(output).__name__}."
        )
    ensure_same_asof_date(output, features)
    ensure_index_aligned(output.predictions.index, features.features.index, label="predictions and features")


def validate_portfolio_output(output: PortfolioDecision, *, context: RunContext) -> None:
    """Validate canonical portfolio engine postconditions."""
    if not isinstance(output, PortfolioDecision):
        raise SchemaViolation(
            f"PortfolioEngine output must be PortfolioDecision, got {type(output).__name__}."
        )
    if output.timestamp > context.asof_date:
        raise DataLeakageError(
            f"PortfolioDecision.timestamp={output.timestamp} exceeds RunContext.asof_date={context.asof_date}."
        )


def validate_gate_output(output: GateResult) -> None:
    """Validate canonical validator postcondition."""
    if not isinstance(output, GateResult):
        raise SchemaViolation(f"Validator output must be GateResult, got {type(output).__name__}.")
