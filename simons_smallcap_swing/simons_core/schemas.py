from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str | None = None
    nullable: bool = True


@dataclass(frozen=True)
class DataSchema:
    name: str
    version: str
    columns: tuple[ColumnSpec, ...]
    primary_key: tuple[str, ...] = ()
    allow_extra_columns: bool = True

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(col.name for col in self.columns)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    column: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    schema_name: str
    schema_version: str
    ok: bool
    issues: tuple[ValidationIssue, ...] = ()


class SchemaValidationError(ValueError):
    def __init__(self, result: ValidationResult) -> None:
        self.result = result
        message = "; ".join(issue.message for issue in result.issues) or "Schema validation failed."
        super().__init__(message)


class PITTimestampError(ValueError):
    """Raised when a row violates availability_ts <= decision_ts semantics."""


_SCHEMA_REGISTRY: dict[str, DataSchema] = {}


def register_schema(schema: DataSchema) -> None:
    _SCHEMA_REGISTRY[schema.name] = schema


def get_schema(name: str) -> DataSchema:
    try:
        return _SCHEMA_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_SCHEMA_REGISTRY))
        raise KeyError(f"Unknown schema '{name}'. Available: [{available}]") from exc


def list_schemas() -> list[str]:
    return sorted(_SCHEMA_REGISTRY)


def _dtype_compatible(series: pd.Series, expected: str) -> bool:
    normalized = expected.lower().strip()
    if normalized in {"string", "str"}:
        # Allow pandas "string" and object-backed string columns for MVP flexibility.
        return is_string_dtype(series) or series.dtype == object
    if normalized in {"float", "float64", "float32"}:
        return is_float_dtype(series)
    if normalized in {"int", "int64", "int32"}:
        return is_integer_dtype(series)
    if normalized in {"number", "numeric"}:
        return is_numeric_dtype(series)
    if normalized in {"bool", "boolean"}:
        return is_bool_dtype(series)
    if normalized.startswith("datetime64"):
        if not is_datetime64_any_dtype(series):
            return False
        if "utc" in normalized:
            tz = getattr(series.dt, "tz", None)
            return tz is not None and str(tz).upper() == "UTC"
        return True
    return True


def validate_schema(df: pd.DataFrame, schema: str | DataSchema) -> ValidationResult:
    spec = get_schema(schema) if isinstance(schema, str) else schema
    issues: list[ValidationIssue] = []

    missing = [name for name in spec.required_columns if name not in df.columns]
    if missing:
        issues.extend(
            ValidationIssue(
                code="missing_column",
                column=col,
                message=f"Missing required column '{col}'.",
            )
            for col in missing
        )
        return ValidationResult(spec.name, spec.version, ok=False, issues=tuple(issues))

    if not spec.allow_extra_columns:
        extras = sorted(set(df.columns) - set(spec.required_columns))
        for col in extras:
            issues.append(
                ValidationIssue(
                    code="unexpected_column",
                    column=col,
                    message=f"Unexpected column '{col}' for schema '{spec.name}'.",
                )
            )

    col_specs = {col.name: col for col in spec.columns}
    for name in spec.required_columns:
        col_spec = col_specs[name]
        series = df[name]

        if col_spec.dtype and not _dtype_compatible(series, col_spec.dtype):
            issues.append(
                ValidationIssue(
                    code="dtype_mismatch",
                    column=name,
                    message=(
                        f"Column '{name}' has dtype '{series.dtype}', "
                        f"expected '{col_spec.dtype}'."
                    ),
                )
            )
        if not col_spec.nullable and series.isna().any():
            issues.append(
                ValidationIssue(
                    code="nullability_violation",
                    column=name,
                    message=f"Column '{name}' contains null values but is non-nullable.",
                )
            )

    if spec.primary_key and set(spec.primary_key).issubset(df.columns):
        duplicates = df.duplicated(list(spec.primary_key), keep=False)
        if duplicates.any():
            issues.append(
                ValidationIssue(
                    code="primary_key_violation",
                    column=",".join(spec.primary_key),
                    message=(
                        f"Primary key {spec.primary_key} has duplicate rows "
                        f"({int(duplicates.sum())} duplicates)."
                    ),
                )
            )

    return ValidationResult(
        schema_name=spec.name,
        schema_version=spec.version,
        ok=not issues,
        issues=tuple(issues),
    )


def assert_schema(df: pd.DataFrame, schema: str | DataSchema) -> None:
    result = validate_schema(df, schema)
    if not result.ok:
        raise SchemaValidationError(result)


def assert_pit_no_lookahead(
    df: pd.DataFrame,
    *,
    decision_col: str = "asof",
    available_col: str = "acceptance_ts",
) -> None:
    required = (decision_col, available_col)
    missing = [col for col in required if col not in df.columns]
    if missing:
        schema = DataSchema(
            name="pit_lookahead_check",
            version="1.0.0",
            columns=tuple(ColumnSpec(col, nullable=False) for col in required),
            allow_extra_columns=True,
        )
        result = ValidationResult(
            schema_name=schema.name,
            schema_version=schema.version,
            ok=False,
            issues=tuple(
                ValidationIssue(
                    code="missing_column",
                    column=col,
                    message=f"Missing required PIT column '{col}'.",
                )
                for col in missing
            ),
        )
        raise SchemaValidationError(result)

    decision_ts = pd.to_datetime(df[decision_col], utc=True, errors="coerce")
    available_ts = pd.to_datetime(df[available_col], utc=True, errors="coerce")

    if decision_ts.isna().any() or available_ts.isna().any():
        raise PITTimestampError(
            f"Invalid timestamp values in '{decision_col}' or '{available_col}'."
        )

    leakage_mask = available_ts > decision_ts
    if leakage_mask.any():
        n_bad = int(leakage_mask.sum())
        raise PITTimestampError(
            f"PIT violation: {n_bad} rows with {available_col} > {decision_col}."
        )


def _register_builtin_schemas() -> None:
    register_schema(
        DataSchema(
            name="prices_adjusted",
            version="1.0.0",
            columns=(
                ColumnSpec("date", "datetime64", nullable=False),
                ColumnSpec("symbol", "string", nullable=False),
                ColumnSpec("open", "float64", nullable=False),
                ColumnSpec("high", "float64", nullable=False),
                ColumnSpec("low", "float64", nullable=False),
                ColumnSpec("close", "float64", nullable=False),
            ),
            primary_key=("date", "symbol"),
            allow_extra_columns=True,
        )
    )
    register_schema(
        DataSchema(
            name="pit_observation",
            version="1.0.0",
            columns=(
                ColumnSpec("symbol", "string", nullable=False),
                ColumnSpec("asof", "datetime64[ns, UTC]", nullable=False),
                ColumnSpec("acceptance_ts", "datetime64[ns, UTC]", nullable=False),
                ColumnSpec("value", "float64", nullable=True),
            ),
            primary_key=("symbol", "asof"),
            allow_extra_columns=True,
        )
    )
    register_schema(
        DataSchema(
            name="reference_trading_calendar",
            version="1.0.0",
            columns=(
                ColumnSpec("date", "datetime64", nullable=False),
                ColumnSpec("is_session", "bool", nullable=False),
                ColumnSpec("session_idx", "int64", nullable=False),
            ),
            primary_key=("date",),
            allow_extra_columns=True,
        )
    )
    register_schema(
        DataSchema(
            name="reference_ticker_history_map",
            version="1.0.0",
            columns=(
                ColumnSpec("instrument_id", "string", nullable=False),
                ColumnSpec("ticker", "string", nullable=False),
                ColumnSpec("start_date", "datetime64", nullable=False),
                ColumnSpec("end_date", "datetime64", nullable=True),
                ColumnSpec("is_active", "bool", nullable=False),
            ),
            primary_key=("instrument_id", "ticker", "start_date"),
            allow_extra_columns=True,
        )
    )
    register_schema(
        DataSchema(
            name="reference_symbols_metadata",
            version="1.0.0",
            columns=(
                ColumnSpec("instrument_id", "string", nullable=False),
                ColumnSpec("ticker", "string", nullable=False),
                ColumnSpec("name", "string", nullable=False),
                ColumnSpec("exchange", "string", nullable=False),
                ColumnSpec("asset_type", "string", nullable=False),
                ColumnSpec("currency", "string", nullable=False),
                ColumnSpec("primary_listing_flag", "bool", nullable=False),
                ColumnSpec("country", "string", nullable=False),
            ),
            primary_key=("instrument_id",),
            allow_extra_columns=True,
        )
    )
    register_schema(
        DataSchema(
            name="reference_sector_industry_map",
            version="1.0.0",
            columns=(
                ColumnSpec("instrument_id", "string", nullable=False),
                ColumnSpec("sector", "string", nullable=False),
                ColumnSpec("industry", "string", nullable=False),
                ColumnSpec("start_date", "datetime64", nullable=False),
                ColumnSpec("end_date", "datetime64", nullable=True),
            ),
            primary_key=("instrument_id", "start_date"),
            allow_extra_columns=True,
        )
    )


_register_builtin_schemas()


__all__ = [
    "ColumnSpec",
    "DataSchema",
    "PITTimestampError",
    "SchemaValidationError",
    "ValidationIssue",
    "ValidationResult",
    "assert_pit_no_lookahead",
    "assert_schema",
    "get_schema",
    "list_schemas",
    "register_schema",
    "validate_schema",
]
