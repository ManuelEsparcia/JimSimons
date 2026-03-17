from __future__ import annotations

"""
Public API for :mod:`simons_core`.

This module intentionally exposes a *small, stable and ergonomic* namespace for
consumers of the core package while keeping import-time behavior cheap and free
of operational side effects.

Design notes
------------
- Re-export only stable contracts, canonical schema machinery and foundational
  services.
- Avoid eager imports of optional or heavier submodules (for example the market
  calendar backend) by resolving exports lazily via ``__getattr__``.
- Keep the root namespace explicit through ``__all__`` so the supported public
  surface is visible and testable.

The root package must remain suitable for use in research notebooks, tests,
backtests and CI where repeated imports are common and import-time side effects
are unacceptable.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Final


# ---------------------------------------------------------------------------
# Public API contract
# ---------------------------------------------------------------------------

__all__ = [
    # interfaces: execution context, semantic containers, contracts, errors
    "DEFAULT_MARKET",
    "DEFAULT_PIPELINE_VERSION",
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
    # schemas: canonical structural contracts and validation entry points
    "ColumnSpec",
    "ColumnPatternSpec",
    "DataSchema",
    "ValidationIssue",
    "ValidationResult",
    "SchemaValidationError",
    "PRICE_ADJUSTED_SCHEMA",
    "FEATURES_SCHEMA",
    "LABELS_SCHEMA",
    "PREDICTIONS_SCHEMA",
    "get_schema",
    "list_schemas",
    "validate_schema",
    "assert_schema",
    "format_validation_issues",
    # calendar: market session geometry
    "CANONICAL_TIMEZONE",
    "SessionType",
    "MarketCalendar",
    "load_market_calendar",
    "align_to_session",
    "next_trading_day",
    "prev_trading_day",
    "trading_days_between",
    # logging: structured observability primitives
    "DEFAULT_SCHEMA_VERSION",
    "LoggerError",
    "LogSink",
    "MemorySink",
    "MultiSink",
    "StructuredLogger",
    "StreamJsonSink",
    "StdoutJsonSink",
    "bind_context",
    "get_logger",
    "serialize_exception",
    "with_step",
]


_EXPORT_MAP: Final[dict[str, tuple[str, str]]] = {
    # interfaces
    "DEFAULT_MARKET": (".interfaces", "DEFAULT_MARKET"),
    "DEFAULT_PIPELINE_VERSION": (".interfaces", "DEFAULT_PIPELINE_VERSION"),
    "CONTRACT_VERSION": (".interfaces", "CONTRACT_VERSION"),
    "RunContext": (".interfaces", "RunContext"),
    "GateResult": (".interfaces", "GateResult"),
    "PriceData": (".interfaces", "PriceData"),
    "FeatureMatrix": (".interfaces", "FeatureMatrix"),
    "Labels": (".interfaces", "Labels"),
    "ModelArtifact": (".interfaces", "ModelArtifact"),
    "PredictionFrame": (".interfaces", "PredictionFrame"),
    "PortfolioDecision": (".interfaces", "PortfolioDecision"),
    "ContractViolation": (".interfaces", "ContractViolation"),
    "DataLeakageError": (".interfaces", "DataLeakageError"),
    "SchemaViolation": (".interfaces", "SchemaViolation"),
    "FeasibilityError": (".interfaces", "FeasibilityError"),
    "ReproducibilityError": (".interfaces", "ReproducibilityError"),
    "GateSemanticsError": (".interfaces", "GateSemanticsError"),
    "AlignmentError": (".interfaces", "AlignmentError"),
    "ContractVersionError": (".interfaces", "ContractVersionError"),
    "DataProvider": (".interfaces", "DataProvider"),
    "FeatureBuilder": (".interfaces", "FeatureBuilder"),
    "ModelTrainer": (".interfaces", "ModelTrainer"),
    "PortfolioEngine": (".interfaces", "PortfolioEngine"),
    "Validator": (".interfaces", "Validator"),
    "ensure_no_future_data": (".interfaces", "ensure_no_future_data"),
    "ensure_index_aligned": (".interfaces", "ensure_index_aligned"),
    "ensure_same_asof_date": (".interfaces", "ensure_same_asof_date"),
    "ensure_single_effective_timestamp": (".interfaces", "ensure_single_effective_timestamp"),
    "ensure_numeric_frame": (".interfaces", "ensure_numeric_frame"),
    "ensure_numeric_series": (".interfaces", "ensure_numeric_series"),
    "ensure_reproducible_run_context": (".interfaces", "ensure_reproducible_run_context"),
    # schemas
    "ColumnSpec": (".schemas", "ColumnSpec"),
    "ColumnPatternSpec": (".schemas", "ColumnPatternSpec"),
    "DataSchema": (".schemas", "DataSchema"),
    "ValidationIssue": (".schemas", "ValidationIssue"),
    "ValidationResult": (".schemas", "ValidationResult"),
    "SchemaValidationError": (".schemas", "SchemaValidationError"),
    "PRICE_ADJUSTED_SCHEMA": (".schemas", "PRICE_ADJUSTED_SCHEMA"),
    "FEATURES_SCHEMA": (".schemas", "FEATURES_SCHEMA"),
    "LABELS_SCHEMA": (".schemas", "LABELS_SCHEMA"),
    "PREDICTIONS_SCHEMA": (".schemas", "PREDICTIONS_SCHEMA"),
    "get_schema": (".schemas", "get_schema"),
    "list_schemas": (".schemas", "list_schemas"),
    "validate_schema": (".schemas", "validate_schema"),
    "assert_schema": (".schemas", "assert_schema"),
    "format_validation_issues": (".schemas", "format_validation_issues"),
    # calendar
    "CANONICAL_TIMEZONE": (".calendar", "CANONICAL_TIMEZONE"),
    "SessionType": (".calendar", "SessionType"),
    "MarketCalendar": (".calendar", "MarketCalendar"),
    "load_market_calendar": (".calendar", "load_market_calendar"),
    "align_to_session": (".calendar", "align_to_session"),
    "next_trading_day": (".calendar", "next_trading_day"),
    "prev_trading_day": (".calendar", "prev_trading_day"),
    "trading_days_between": (".calendar", "trading_days_between"),
    # logging
    "DEFAULT_SCHEMA_VERSION": (".logging", "DEFAULT_SCHEMA_VERSION"),
    "LoggerError": (".logging", "LoggerError"),
    "LogSink": (".logging", "LogSink"),
    "MemorySink": (".logging", "MemorySink"),
    "MultiSink": (".logging", "MultiSink"),
    "StructuredLogger": (".logging", "StructuredLogger"),
    "StreamJsonSink": (".logging", "StreamJsonSink"),
    "StdoutJsonSink": (".logging", "StdoutJsonSink"),
    "bind_context": (".logging", "bind_context"),
    "get_logger": (".logging", "get_logger"),
    "serialize_exception": (".logging", "serialize_exception"),
    "with_step": (".logging", "with_step"),
}


if TYPE_CHECKING:
    from .calendar import (  # noqa: F401
        CANONICAL_TIMEZONE,
        MarketCalendar,
        SessionType,
        align_to_session,
        load_market_calendar,
        next_trading_day,
        prev_trading_day,
        trading_days_between,
    )
    from .interfaces import (  # noqa: F401
        CONTRACT_VERSION,
        DEFAULT_MARKET,
        DEFAULT_PIPELINE_VERSION,
        AlignmentError,
        ContractVersionError,
        ContractViolation,
        DataLeakageError,
        DataProvider,
        FeatureBuilder,
        FeatureMatrix,
        FeasibilityError,
        GateResult,
        GateSemanticsError,
        Labels,
        ModelArtifact,
        ModelTrainer,
        PortfolioDecision,
        PortfolioEngine,
        PredictionFrame,
        PriceData,
        ReproducibilityError,
        RunContext,
        SchemaViolation,
        Validator,
        ensure_index_aligned,
        ensure_no_future_data,
        ensure_numeric_frame,
        ensure_numeric_series,
        ensure_reproducible_run_context,
        ensure_same_asof_date,
        ensure_single_effective_timestamp,
    )
    from .logging import (  # noqa: F401
        DEFAULT_SCHEMA_VERSION,
        LoggerError,
        LogSink,
        MemorySink,
        MultiSink,
        StdoutJsonSink,
        StreamJsonSink,
        StructuredLogger,
        bind_context,
        get_logger,
        serialize_exception,
        with_step,
    )
    from .schemas import (  # noqa: F401
        FEATURES_SCHEMA,
        LABELS_SCHEMA,
        PREDICTIONS_SCHEMA,
        PRICE_ADJUSTED_SCHEMA,
        ColumnPatternSpec,
        ColumnSpec,
        DataSchema,
        SchemaValidationError,
        ValidationIssue,
        ValidationResult,
        assert_schema,
        format_validation_issues,
        get_schema,
        list_schemas,
        validate_schema,
    )


def __getattr__(name: str):
    """Resolve public exports lazily and cache them in the module globals."""
    try:
        module_name, attr_name = _EXPORT_MAP[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Keep interactive discovery aligned with the explicit public contract."""
    return sorted(set(globals()) | set(__all__))
