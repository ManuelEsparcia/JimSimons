"""
simons_core — Public API surface.

Re-exports stable contracts, types, and services that define
the package's supported interface. Import is deterministic,
cheap, and side-effect free.

Usage:
    from simons_core import RunContext, GateResult, MarketCalendar
"""

from .interfaces import (
    DataProvider, FeatureBuilder, ModelTrainer, PortfolioEngine, Validator,
    RunContext, GateResult,
    PriceData, FeatureMatrix, ModelArtifact, PortfolioDecision,
    ContractViolation, DataLeakageError, SchemaViolation,
    FeasibilityError, ReproducibilityError,
)
from .schemas import (
    DataSchema, ColumnSpec, ValidationResult, ValidationIssue,
    validate_schema, assert_schema, get_schema, list_schemas,
    SCHEMA_REGISTRY,
)
from .calendar import MarketCalendar, SessionType, load_market_calendar
from .logging import get_logger, StructuredLogger, LogEvent, with_step, serialize_exception

__all__ = [
    # Interfaces
    "DataProvider", "FeatureBuilder", "ModelTrainer", "PortfolioEngine", "Validator",
    # Transversal types
    "RunContext", "GateResult",
    # I/O semantic types
    "PriceData", "FeatureMatrix", "ModelArtifact", "PortfolioDecision",
    # Errors
    "ContractViolation", "DataLeakageError", "SchemaViolation",
    "FeasibilityError", "ReproducibilityError",
    # Schemas
    "DataSchema", "ColumnSpec", "ValidationResult", "ValidationIssue",
    "validate_schema", "assert_schema", "get_schema", "list_schemas",
    "SCHEMA_REGISTRY",
    # Calendar
    "MarketCalendar", "SessionType", "load_market_calendar",
    # Logging
    "get_logger", "StructuredLogger", "LogEvent", "with_step", "serialize_exception",
]
