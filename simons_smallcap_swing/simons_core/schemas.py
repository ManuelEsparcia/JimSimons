from __future__ import annotations

"""
simons_core.schemas
===================

Institutional structural contracts for canonical datasets moving through the
quant research stack.

This module is intentionally contract-first and fail-fast:
- it declares explicit schemas for important datasets;
- it validates shape, dtypes, nullability, keys and domain constraints;
- it centralizes schema evolution/versioning;
- it raises actionable, structured validation failures.

It does *not* impute, repair, transform, or infer business logic. Any automatic
repair belongs elsewhere in the stack.
"""

from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
import re
from typing import Any, Callable, Final, Mapping, Sequence, TypeAlias

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

IssueCode: TypeAlias = str
SchemaValidator: TypeAlias = Callable[[pd.DataFrame, "DataSchema"], list["ValidationIssue"]]


@dataclass(frozen=True)
class ColumnSpec:
    """Declarative contract for a single named column."""

    name: str
    dtype: str
    nullable: bool = False
    required: bool = True
    allowed_values: tuple[Any, ...] | None = None
    min_value: float | int | None = None
    max_value: float | int | None = None
    description: str | None = None


@dataclass(frozen=True)
class ColumnPatternSpec:
    """
    Declarative contract for dynamic column families such as ``feature_*``.

    Parameters
    ----------
    pattern:
        Full regex pattern to match dynamic column names.
    dtype:
        Expected dtype alias for all matching columns. If ``None``, dtype is
        not checked for the dynamic family.
    min_count:
        Minimum number of matching columns required.
    """

    pattern: str
    dtype: str | None = None
    nullable: bool = True
    min_count: int = 0
    allowed_values: tuple[Any, ...] | None = None
    min_value: float | int | None = None
    max_value: float | int | None = None
    description: str | None = None

    def compiled(self) -> re.Pattern[str]:
        return re.compile(self.pattern)


@dataclass(frozen=True)
class ValidationIssue:
    """Single actionable schema validation issue."""

    code: IssueCode
    message: str
    column: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    """Structured validation outcome."""

    passed: bool
    issues: tuple[ValidationIssue, ...] = ()
    schema_name: str | None = None
    schema_version: str | None = None

    def by_code(self, code: IssueCode) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.code == code)

    def summary(self) -> str:
        if self.passed:
            label = self.schema_name or "schema"
            version = f" v{self.schema_version}" if self.schema_version else ""
            return f"Validation passed for {label}{version}."
        return format_validation_issues(self)


@dataclass(frozen=True)
class DataSchema:
    """
    Structural contract for a canonical dataset.

    Notes
    -----
    ``columns`` defines explicit named columns.
    ``column_patterns`` defines dynamic families such as ``feature_*``.
    ``primary_key`` is treated as a strict uniqueness invariant.
    ``validators`` host schema-specific semantic/domain checks.
    """

    name: str
    version: str
    columns: tuple[ColumnSpec, ...]
    primary_key: tuple[str, ...] = ()
    allow_extra_columns: bool = False
    column_patterns: tuple[ColumnPatternSpec, ...] = ()
    date_column: str | None = None
    require_monotonic_date: bool = False
    allow_empty: bool = True
    validators: tuple[SchemaValidator, ...] = ()
    description: str | None = None

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns if column.required)

    def get_column(self, name: str) -> ColumnSpec | None:
        for column in self.columns:
            if column.name == name:
                return column
        return None


class SchemaValidationError(ValueError):
    """Raised when a DataFrame violates a declared data contract."""


MISSING_COLUMN: Final = "MISSING_COLUMN"
UNEXPECTED_COLUMN: Final = "UNEXPECTED_COLUMN"
DUPLICATE_COLUMN_NAME: Final = "DUPLICATE_COLUMN_NAME"
INVALID_DTYPE: Final = "INVALID_DTYPE"
NULL_IN_NONNULLABLE: Final = "NULL_IN_NONNULLABLE"
PRIMARY_KEY_VIOLATION: Final = "PRIMARY_KEY_VIOLATION"
DOMAIN_VIOLATION: Final = "DOMAIN_VIOLATION"
NON_MONOTONIC_DATE: Final = "NON_MONOTONIC_DATE"
FUTURE_TIMESTAMP_VIOLATION: Final = "FUTURE_TIMESTAMP_VIOLATION"
EMPTY_DATASET: Final = "EMPTY_DATASET"
SCHEMA_NOT_FOUND: Final = "SCHEMA_NOT_FOUND"


def _build_issue(code: IssueCode, message: str, column: str | None = None) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, column=column)


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"validate_schema expects a pandas.DataFrame, got {type(df).__name__}"
        )
    return df


def _count_column_name(df: pd.DataFrame, name: str) -> int:
    return sum(1 for column in df.columns if str(column) == name)


def _get_unique_series(df: pd.DataFrame, name: str) -> pd.Series | None:
    if _count_column_name(df, name) != 1:
        return None
    result = df[name]
    return result if isinstance(result, pd.Series) else None


def _is_string_like_object(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return True
    return bool(non_null.map(lambda value: isinstance(value, str)).all())


def _dtype_compatible(series: pd.Series, expected: str) -> bool:
    """
    Runtime dtype compatibility check.

    This is intentionally broader than strict dtype-string equality because data
    providers and pandas extension dtypes can vary (e.g. ``Int64`` vs ``int64``).
    """
    alias = expected.lower()

    if alias in {"float", "float64", "float32"}:
        return is_float_dtype(series)

    if alias in {"int", "int64", "int32"}:
        return is_integer_dtype(series)

    if alias in {"numeric", "number"}:
        return is_numeric_dtype(series)

    if alias == "bool":
        return is_bool_dtype(series) or (
            is_object_dtype(series)
            and bool(series.dropna().map(lambda value: isinstance(value, bool)).all())
        )

    if alias == "binary":
        if is_bool_dtype(series):
            return True
        if is_integer_dtype(series):
            non_null = series.dropna()
            return bool(non_null.isin([0, 1]).all())
        return False

    if alias in {"string", "str"}:
        return is_string_dtype(series) or (
            is_object_dtype(series) and _is_string_like_object(series)
        )

    if alias in {"date", "datetime", "timestamp"}:
        return is_datetime64_any_dtype(series)

    if alias == "category":
        return str(series.dtype) == "category"

    if alias == "any":
        return True

    raise ValueError(f"Unsupported expected dtype alias: {expected!r}")


def _validate_duplicate_columns(df: pd.DataFrame) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    duplicated_names = df.columns[df.columns.duplicated()].tolist()
    for name in sorted({str(col) for col in duplicated_names}):
        issues.append(
            _build_issue(
                DUPLICATE_COLUMN_NAME,
                f"Duplicate column name detected: {name}",
                column=name,
            )
        )
    return issues


def _validate_domain_bounds(
    series: pd.Series,
    *,
    column: str,
    allowed_values: tuple[Any, ...] | None,
    min_value: float | int | None,
    max_value: float | int | None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    non_null = series.dropna()

    if allowed_values is not None and not bool(non_null.isin(allowed_values).all()):
        issues.append(
            _build_issue(
                DOMAIN_VIOLATION,
                f"Column {column} contains values outside allowed domain {allowed_values!r}",
                column=column,
            )
        )

    if min_value is not None:
        try:
            below = non_null < min_value
        except TypeError:
            below = pd.Series([True] * len(non_null), index=non_null.index)
        if bool(below.any()):
            issues.append(
                _build_issue(
                    DOMAIN_VIOLATION,
                    f"Column {column} contains values < {min_value}",
                    column=column,
                )
            )

    if max_value is not None:
        try:
            above = non_null > max_value
        except TypeError:
            above = pd.Series([True] * len(non_null), index=non_null.index)
        if bool(above.any()):
            issues.append(
                _build_issue(
                    DOMAIN_VIOLATION,
                    f"Column {column} contains values > {max_value}",
                    column=column,
                )
            )

    return issues


def _validate_missing_and_extra_columns(df: pd.DataFrame, schema: DataSchema) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    actual_columns = tuple(str(col) for col in df.columns)
    actual_set = set(actual_columns)
    explicit_required = set(schema.required_columns)
    explicit_all = set(schema.column_names)

    for column in sorted(explicit_required - actual_set):
        issues.append(
            _build_issue(
                MISSING_COLUMN,
                f"Missing required column: {column}",
                column=column,
            )
        )

    matched_pattern_columns: set[str] = set()
    for pattern_spec in schema.column_patterns:
        regex = pattern_spec.compiled()
        matched = [column for column in actual_columns if regex.fullmatch(column)]
        matched_pattern_columns.update(matched)

        if len(matched) < pattern_spec.min_count:
            issues.append(
                _build_issue(
                    MISSING_COLUMN,
                    "Expected at least "
                    f"{pattern_spec.min_count} column(s) matching pattern "
                    f"{pattern_spec.pattern!r}, found {len(matched)}.",
                )
            )

    if not schema.allow_extra_columns:
        allowed = explicit_all | matched_pattern_columns
        for column in sorted(actual_set - allowed):
            issues.append(
                _build_issue(
                    UNEXPECTED_COLUMN,
                    f"Unexpected column: {column}",
                    column=column,
                )
            )

    return issues


def _validate_column_spec(df: pd.DataFrame, spec: ColumnSpec) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    series = _get_unique_series(df, spec.name)
    if series is None:
        return issues

    if not _dtype_compatible(series, spec.dtype):
        issues.append(
            _build_issue(
                INVALID_DTYPE,
                f"Column {spec.name} has dtype {series.dtype!s}, expected compatible with {spec.dtype!r}",
                column=spec.name,
            )
        )

    if not spec.nullable and bool(series.isna().any()):
        issues.append(
            _build_issue(
                NULL_IN_NONNULLABLE,
                f"Column {spec.name} contains nulls but is not nullable",
                column=spec.name,
            )
        )

    issues.extend(
        _validate_domain_bounds(
            series,
            column=spec.name,
            allowed_values=spec.allowed_values,
            min_value=spec.min_value,
            max_value=spec.max_value,
        )
    )
    return issues


def _validate_pattern_spec(df: pd.DataFrame, pattern_spec: ColumnPatternSpec) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    regex = pattern_spec.compiled()

    unique_names = sorted({str(col) for col in df.columns})
    for column in unique_names:
        if not regex.fullmatch(column):
            continue

        series = _get_unique_series(df, column)
        if series is None:
            continue

        if pattern_spec.dtype is not None and not _dtype_compatible(series, pattern_spec.dtype):
            issues.append(
                _build_issue(
                    INVALID_DTYPE,
                    f"Column {column} has dtype {series.dtype!s}, expected compatible with {pattern_spec.dtype!r}",
                    column=column,
                )
            )

        if not pattern_spec.nullable and bool(series.isna().any()):
            issues.append(
                _build_issue(
                    NULL_IN_NONNULLABLE,
                    f"Column {column} contains nulls but is not nullable",
                    column=column,
                )
            )

        issues.extend(
            _validate_domain_bounds(
                series,
                column=column,
                allowed_values=pattern_spec.allowed_values,
                min_value=pattern_spec.min_value,
                max_value=pattern_spec.max_value,
            )
        )

    return issues


def _validate_primary_key(df: pd.DataFrame, schema: DataSchema) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not schema.primary_key:
        return issues

    missing_pk = [column for column in schema.primary_key if column not in df.columns]
    if missing_pk:
        return issues

    if bool(df.duplicated(subset=list(schema.primary_key)).any()):
        issues.append(
            _build_issue(
                PRIMARY_KEY_VIOLATION,
                f"Primary key {schema.primary_key!r} is not unique",
            )
        )
    return issues


def _validate_temporal_semantics(
    df: pd.DataFrame,
    schema: DataSchema,
    *,
    asof_date: pd.Timestamp | datetime | date | str | None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if schema.date_column is None:
        return issues

    series = _get_unique_series(df, schema.date_column)
    if series is None or not _dtype_compatible(series, "date"):
        return issues

    if schema.require_monotonic_date and not bool(series.is_monotonic_increasing):
        issues.append(
            _build_issue(
                NON_MONOTONIC_DATE,
                f"Date column {schema.date_column} is not monotonic increasing",
                column=schema.date_column,
            )
        )

    if asof_date is not None:
        cutoff = pd.Timestamp(asof_date)
        if bool((series.dropna() > cutoff).any()):
            issues.append(
                _build_issue(
                    FUTURE_TIMESTAMP_VIOLATION,
                    f"Date column {schema.date_column} contains timestamps later than asof_date={cutoff}",
                    column=schema.date_column,
                )
            )

    return issues


def _validate_prices_adjusted_domains(df: pd.DataFrame, _: DataSchema) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    high = _get_unique_series(df, "high")
    low = _get_unique_series(df, "low")
    if high is not None and low is not None:
        bad = high < low
        if bool(bad.any()):
            issues.append(
                _build_issue(
                    DOMAIN_VIOLATION,
                    "Found rows with high < low",
                )
            )

    for column in ("open", "high", "low", "close"):
        series = _get_unique_series(df, column)
        if series is None:
            continue
        bad = series <= 0
        if bool(bad.any()):
            issues.append(
                _build_issue(
                    DOMAIN_VIOLATION,
                    f"Found rows with {column} <= 0",
                    column=column,
                )
            )

    volume = _get_unique_series(df, "volume")
    if volume is not None:
        bad = volume.dropna() < 0
        if bool(bad.any()):
            issues.append(
                _build_issue(
                    DOMAIN_VIOLATION,
                    "Found rows with volume < 0",
                    column="volume",
                )
            )

    return issues


def _validate_features_contract(df: pd.DataFrame, schema: DataSchema) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    feature_columns = [
        str(column)
        for column in df.columns
        if re.fullmatch(r"feature_[A-Za-z0-9_]+", str(column))
    ]

    if not feature_columns:
        issues.append(
            _build_issue(
                MISSING_COLUMN,
                "Features dataset must contain at least one column matching 'feature_[A-Za-z0-9_]+'",
            )
        )
        return issues

    for column in sorted(set(feature_columns)):
        series = _get_unique_series(df, column)
        if series is None:
            continue
        if not _dtype_compatible(series, "numeric"):
            issues.append(
                _build_issue(
                    INVALID_DTYPE,
                    f"Feature column {column} must be numeric",
                    column=column,
                )
            )

    if schema.primary_key:
        missing_pk = [column for column in schema.primary_key if column not in df.columns]
        if not missing_pk:
            feature_name_set = set(feature_columns)
            pk_name_set = set(schema.primary_key)
            overlap = sorted(feature_name_set & pk_name_set)
            for column in overlap:
                issues.append(
                    _build_issue(
                        DOMAIN_VIOLATION,
                        f"Column {column} cannot be both a primary-key field and a feature column",
                        column=column,
                    )
                )

    return issues


def _validate_labels_contract(df: pd.DataFrame, _: DataSchema) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    future_ret = _get_unique_series(df, "future_ret_5d")
    if future_ret is not None:
        if not _dtype_compatible(future_ret, "numeric"):
            issues.append(
                _build_issue(
                    INVALID_DTYPE,
                    "Column future_ret_5d must be numeric",
                    column="future_ret_5d",
                )
            )

    binary = _get_unique_series(df, "binary_swing")
    if binary is not None:
        if not _dtype_compatible(binary, "binary"):
            issues.append(
                _build_issue(
                    INVALID_DTYPE,
                    "Column binary_swing must be boolean or integer-binary",
                    column="binary_swing",
                )
            )
        elif not bool(binary.dropna().isin([0, 1, True, False]).all()):
            issues.append(
                _build_issue(
                    DOMAIN_VIOLATION,
                    "Column binary_swing must only contain values in {0, 1, True, False}",
                    column="binary_swing",
                )
            )

    return issues


def _validate_predictions_contract(df: pd.DataFrame, _: DataSchema) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for column in ("prob_long", "confidence"):
        series = _get_unique_series(df, column)
        if series is None:
            continue
        if not _dtype_compatible(series, "numeric"):
            issues.append(
                _build_issue(
                    INVALID_DTYPE,
                    f"Column {column} must be numeric",
                    column=column,
                )
            )
            continue

        bad = (series.dropna() < 0) | (series.dropna() > 1)
        if bool(bad.any()):
            issues.append(
                _build_issue(
                    DOMAIN_VIOLATION,
                    f"Column {column} must lie in [0, 1]",
                    column=column,
                )
            )

    score = _get_unique_series(df, "score")
    if score is not None and not _dtype_compatible(score, "numeric"):
        issues.append(
            _build_issue(
                INVALID_DTYPE,
                "Column score must be numeric",
                column="score",
            )
        )

    return issues


PRICE_ADJUSTED_SCHEMA = DataSchema(
    name="prices_adjusted",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "date", nullable=False, description="Trading session date."),
        ColumnSpec("symbol", "string", nullable=False, description="Ticker or canonical symbol."),
        ColumnSpec("open", "float64", nullable=False),
        ColumnSpec("high", "float64", nullable=False),
        ColumnSpec("low", "float64", nullable=False),
        ColumnSpec("close", "float64", nullable=False),
        ColumnSpec("volume", "int64", nullable=True, description="Nullable depending on provider completeness."),
    ),
    primary_key=("date", "symbol"),
    allow_extra_columns=False,
    date_column="date",
    require_monotonic_date=False,
    allow_empty=True,
    validators=(_validate_prices_adjusted_domains,),
    description="Adjusted OHLCV daily bars keyed by (date, symbol).",
)


FEATURES_SCHEMA = DataSchema(
    name="features",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "date", nullable=False, description="Decision date for feature observation."),
        ColumnSpec("symbol", "string", nullable=False, description="Ticker or canonical symbol."),
    ),
    primary_key=("date", "symbol"),
    allow_extra_columns=False,
    column_patterns=(
        ColumnPatternSpec(
            pattern=r"feature_[A-Za-z0-9_]+",
            dtype="numeric",
            nullable=True,
            min_count=1,
            description="Model-ready numeric feature columns.",
        ),
    ),
    date_column="date",
    require_monotonic_date=False,
    allow_empty=True,
    validators=(_validate_features_contract,),
    description="Feature matrix keyed by (date, symbol) with dynamic feature_* columns.",
)


LABELS_SCHEMA = DataSchema(
    name="labels",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "date", nullable=False, description="Base observation date."),
        ColumnSpec("symbol", "string", nullable=False, description="Ticker or canonical symbol."),
        ColumnSpec(
            "future_ret_5d",
            "float64",
            nullable=False,
            min_value=-1.0,
            max_value=1.0,
            description="Forward 5-day return under normalized [-1, 1] policy.",
        ),
        ColumnSpec(
            "binary_swing",
            "binary",
            nullable=False,
            allowed_values=(0, 1, True, False),
            description="Binary classification target.",
        ),
    ),
    primary_key=("date", "symbol"),
    allow_extra_columns=False,
    date_column="date",
    require_monotonic_date=False,
    allow_empty=True,
    validators=(_validate_labels_contract,),
    description="Canonical supervised labels keyed by (date, symbol).",
)


PREDICTIONS_SCHEMA = DataSchema(
    name="predictions",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "date", nullable=False, description="Prediction timestamp/date."),
        ColumnSpec("symbol", "string", nullable=False, description="Ticker or canonical symbol."),
        ColumnSpec("score", "float64", nullable=False, description="Primary model score."),
        ColumnSpec(
            "prob_long",
            "float64",
            nullable=True,
            required=False,
            min_value=0.0,
            max_value=1.0,
            description="Optional calibrated long probability.",
        ),
        ColumnSpec(
            "confidence",
            "float64",
            nullable=True,
            required=False,
            min_value=0.0,
            max_value=1.0,
            description="Optional confidence score constrained to [0, 1].",
        ),
    ),
    primary_key=("date", "symbol"),
    allow_extra_columns=False,
    date_column="date",
    require_monotonic_date=False,
    allow_empty=True,
    validators=(_validate_predictions_contract,),
    description="Model inference outputs keyed by (date, symbol).",
)


_SCHEMA_REGISTRY_INTERNAL: Mapping[str, DataSchema] = {
    PRICE_ADJUSTED_SCHEMA.name: PRICE_ADJUSTED_SCHEMA,
    FEATURES_SCHEMA.name: FEATURES_SCHEMA,
    LABELS_SCHEMA.name: LABELS_SCHEMA,
    PREDICTIONS_SCHEMA.name: PREDICTIONS_SCHEMA,
}
SCHEMA_REGISTRY: Final[Mapping[str, DataSchema]] = MappingProxyType(_SCHEMA_REGISTRY_INTERNAL)


SchemaRef: TypeAlias = str | DataSchema


def register_schema(schema: DataSchema, *, overwrite: bool = False) -> DataSchema:
    """Register a schema in the canonical registry."""
    if not isinstance(schema, DataSchema):
        raise TypeError(f"schema must be a DataSchema, got {type(schema).__name__}")
    if not overwrite and schema.name in _SCHEMA_REGISTRY_INTERNAL:
        raise ValueError(f"Schema {schema.name!r} already exists; pass overwrite=True to replace it")
    _SCHEMA_REGISTRY_INTERNAL[schema.name] = schema
    return schema


def get_schema(name: str) -> DataSchema:
    """Return a schema from the canonical registry."""
    try:
        return SCHEMA_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(SCHEMA_REGISTRY))
        raise KeyError(f"Unknown schema {name!r}. Available schemas: {available}") from exc


def list_schemas() -> list[str]:
    """Return registered schema names in deterministic order."""
    return sorted(SCHEMA_REGISTRY.keys())


def format_validation_issues(result: ValidationResult | Sequence[ValidationIssue]) -> str:
    """Render issues into a deterministic human-readable string."""
    issues = result.issues if isinstance(result, ValidationResult) else tuple(result)
    if not issues:
        return "No validation issues."

    rendered: list[str] = []
    for issue in issues:
        location = f" [{issue.column}]" if issue.column is not None else ""
        rendered.append(f"{issue.code}{location}: {issue.message}")
    return "; ".join(rendered)


def validate_schema(
    df: pd.DataFrame,
    schema: SchemaRef,
    *,
    asof_date: pd.Timestamp | datetime | date | str | None = None,
) -> ValidationResult:
    """
    Validate a DataFrame against a declared schema.

    Parameters
    ----------
    df:
        DataFrame to validate.
    schema:
        Schema name or explicit ``DataSchema`` instance.
    asof_date:
        Optional cutoff used by temporal checks to reject future timestamps.
    """
    frame = _normalize_frame(df)
    spec = get_schema(schema) if isinstance(schema, str) else schema
    issues: list[ValidationIssue] = []

    if frame.empty and not spec.allow_empty:
        issues.append(
            _build_issue(
                EMPTY_DATASET,
                f"Schema {spec.name!r} does not allow empty datasets",
            )
        )

    issues.extend(_validate_duplicate_columns(frame))
    issues.extend(_validate_missing_and_extra_columns(frame, spec))

    for column_spec in spec.columns:
        issues.extend(_validate_column_spec(frame, column_spec))

    for pattern_spec in spec.column_patterns:
        issues.extend(_validate_pattern_spec(frame, pattern_spec))

    issues.extend(_validate_primary_key(frame, spec))
    issues.extend(_validate_temporal_semantics(frame, spec, asof_date=asof_date))

    for validator in spec.validators:
        issues.extend(validator(frame, spec))

    return ValidationResult(
        passed=(len(issues) == 0),
        issues=tuple(issues),
        schema_name=spec.name,
        schema_version=spec.version,
    )


def assert_schema(
    df: pd.DataFrame,
    schema: SchemaRef,
    *,
    asof_date: pd.Timestamp | datetime | date | str | None = None,
) -> None:
    """Raise ``SchemaValidationError`` if schema validation fails."""
    result = validate_schema(df, schema, asof_date=asof_date)
    if result.passed:
        return
    raise SchemaValidationError(format_validation_issues(result))


__all__ = [
    "ColumnPatternSpec",
    "ColumnSpec",
    "DataSchema",
    "EMPTY_DATASET",
    "DOMAIN_VIOLATION",
    "DUPLICATE_COLUMN_NAME",
    "FEATURES_SCHEMA",
    "FUTURE_TIMESTAMP_VIOLATION",
    "INVALID_DTYPE",
    "LABELS_SCHEMA",
    "MISSING_COLUMN",
    "NON_MONOTONIC_DATE",
    "NULL_IN_NONNULLABLE",
    "PREDICTIONS_SCHEMA",
    "PRICE_ADJUSTED_SCHEMA",
    "PRIMARY_KEY_VIOLATION",
    "SCHEMA_REGISTRY",
    "SCHEMA_NOT_FOUND",
    "SchemaValidationError",
    "UNEXPECTED_COLUMN",
    "ValidationIssue",
    "ValidationResult",
    "assert_schema",
    "format_validation_issues",
    "get_schema",
    "list_schemas",
    "validate_schema",
]
