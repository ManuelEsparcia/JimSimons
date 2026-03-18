"""
simons_core/schemas.py — Canonical data contracts.

Declares structural schemas for pipeline datasets:
    prices_adjusted, features, labels, predictions

Validation: columns, dtypes, nullable, PK uniqueness, domain constraints.
Registry: SCHEMA_REGISTRY — single source of truth.

D ⊨ S iff: required columns exist, types compatible, non-nullable cols
have no nulls, PK is unique, values in declared domains.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import pandas as pd
from pandas.api.types import (
    is_bool_dtype, is_datetime64_any_dtype, is_float_dtype,
    is_integer_dtype, is_object_dtype, is_string_dtype,
)


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str                          # float64 | int64 | string | date | bool
    nullable: bool = False
    allowed_values: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class DataSchema:
    name: str
    version: str
    columns: tuple[ColumnSpec, ...]
    primary_key: tuple[str, ...] = ()
    allow_extra_columns: bool = False

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    column: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    issues: tuple[ValidationIssue, ...] = ()


# ── Dtype compatibility (spec §14) ──────────────────────────────────────────

def _dtype_compatible(series: pd.Series, expected: str) -> bool:
    if expected == "float64": return is_float_dtype(series)
    if expected == "int64":   return is_integer_dtype(series)
    if expected == "bool":    return is_bool_dtype(series)
    if expected == "string":  return is_string_dtype(series) or is_object_dtype(series)
    if expected == "date":    return is_datetime64_any_dtype(series) or is_object_dtype(series)
    return True


# ── Canonical schemas (spec §7-10) ──────────────────────────────────────────

PRICE_ADJUSTED_SCHEMA = DataSchema(
    name="prices_adjusted", version="1.0.0",
    columns=(
        ColumnSpec("date", "date", nullable=False),
        ColumnSpec("symbol", "string", nullable=False),
        ColumnSpec("open", "float64", nullable=False),
        ColumnSpec("high", "float64", nullable=False),
        ColumnSpec("low", "float64", nullable=False),
        ColumnSpec("close", "float64", nullable=False),
        ColumnSpec("volume", "int64", nullable=True),
    ),
    primary_key=("date", "symbol"),
    allow_extra_columns=False,
)

FEATURES_SCHEMA = DataSchema(
    name="features", version="1.0.0",
    columns=(
        ColumnSpec("date", "date", nullable=False),
        ColumnSpec("symbol", "string", nullable=False),
    ),
    primary_key=("date", "symbol"),
    allow_extra_columns=True,  # feature columns are dynamic
)

LABELS_SCHEMA = DataSchema(
    name="labels", version="1.0.0",
    columns=(
        ColumnSpec("date", "date", nullable=False),
        ColumnSpec("symbol", "string", nullable=False),
    ),
    primary_key=("date", "symbol"),
    allow_extra_columns=True,
)

PREDICTIONS_SCHEMA = DataSchema(
    name="predictions", version="1.0.0",
    columns=(
        ColumnSpec("date", "date", nullable=False),
        ColumnSpec("symbol", "string", nullable=False),
        ColumnSpec("score", "float64", nullable=False),
    ),
    primary_key=("date", "symbol"),
    allow_extra_columns=True,
)

SCHEMA_REGISTRY: dict[str, DataSchema] = {
    "prices_adjusted": PRICE_ADJUSTED_SCHEMA,
    "features": FEATURES_SCHEMA,
    "labels": LABELS_SCHEMA,
    "predictions": PREDICTIONS_SCHEMA,
}


# ── Validation API (spec §12-13) ────────────────────────────────────────────

def get_schema(name: str) -> DataSchema:
    if name not in SCHEMA_REGISTRY:
        raise KeyError(f"Unknown schema: {name!r}. Available: {list(SCHEMA_REGISTRY)}")
    return SCHEMA_REGISTRY[name]

def list_schemas() -> list[str]:
    return sorted(SCHEMA_REGISTRY.keys())

def validate_schema(df: pd.DataFrame, schema: str | DataSchema) -> ValidationResult:
    spec = get_schema(schema) if isinstance(schema, str) else schema
    issues: list[ValidationIssue] = []
    actual = set(df.columns)
    expected = set(spec.column_names)

    for col in sorted(expected - actual):
        issues.append(ValidationIssue("MISSING_COLUMN", f"Missing: {col}", col))

    if not spec.allow_extra_columns:
        for col in sorted(actual - expected):
            issues.append(ValidationIssue("UNEXPECTED_COLUMN", f"Unexpected: {col}", col))

    for cs in spec.columns:
        if cs.name not in df.columns: continue
        s = df[cs.name]
        if not _dtype_compatible(s, cs.dtype):
            issues.append(ValidationIssue("INVALID_DTYPE", f"{cs.name}: expected {cs.dtype}, got {s.dtype}", cs.name))
        if not cs.nullable and s.isna().any():
            issues.append(ValidationIssue("NULL_IN_NONNULLABLE", f"{cs.name} has nulls", cs.name))
        if cs.allowed_values is not None and not s.dropna().isin(cs.allowed_values).all():
            issues.append(ValidationIssue("DOMAIN_VIOLATION", f"{cs.name} has values outside domain", cs.name))

    if spec.primary_key and all(c in df.columns for c in spec.primary_key):
        if df.duplicated(subset=list(spec.primary_key)).any():
            issues.append(ValidationIssue("PRIMARY_KEY_VIOLATION", f"PK {spec.primary_key} not unique"))

    # Domain checks for prices
    if spec.name == "prices_adjusted":
        issues.extend(_validate_price_domains(df))

    return ValidationResult(passed=len(issues) == 0, issues=tuple(issues))

def assert_schema(df: pd.DataFrame, schema: str | DataSchema) -> None:
    r = validate_schema(df, schema)
    if not r.passed:
        msgs = "; ".join(f"{i.code}: {i.message}" for i in r.issues[:5])
        raise ValueError(f"Schema validation failed ({len(r.issues)} issues): {msgs}")


def _validate_price_domains(df: pd.DataFrame) -> list[ValidationIssue]:
    issues = []
    if {"high", "low"}.issubset(df.columns):
        if (df["high"] < df["low"]).any():
            issues.append(ValidationIssue("DOMAIN_VIOLATION", "high < low", None))
    for c in ("open", "high", "low", "close"):
        if c in df.columns and (df[c] <= 0).any():
            issues.append(ValidationIssue("DOMAIN_VIOLATION", f"{c} ≤ 0", c))
    if "volume" in df.columns and (df["volume"].dropna() < 0).any():
        issues.append(ValidationIssue("DOMAIN_VIOLATION", "volume < 0", "volume"))
    return issues
