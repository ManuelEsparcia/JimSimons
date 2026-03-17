from __future__ import annotations

"""
simons_core.io.parquet_store
============================

Institutional deterministic Parquet snapshot store for the quant stack.

This module treats a dataset as a *logical snapshot* rather than as a bag of
files. The visible unit for readers is a coherent tuple:

    S = (Π, Σ, M, τ)

where:
- Π is the set of physical Parquet files,
- Σ is the validated logical schema,
- M is the manifest,
- τ is the publication timestamp.

Core guarantees
---------------
- Contract-first schema validation before publication.
- Atomic publication via staging -> fsync -> rename -> pointer update.
- Versioned snapshots with manifest-governed visibility.
- Optional deterministic writes via canonical row ordering.
- Read-time projection and filter support, including partition pruning.
- Explicit schema-compatibility assessment.
- Corruption detection during validation; never served silently in strict mode.

Scope boundaries
----------------
This module intentionally does *not* attempt ACID table-format semantics
(Delta/Iceberg/Hudi), distributed concurrency control, or automatic schema
migration. It is a local/filesystem-first snapshot store with strong contracts,
not a lakehouse transaction manager.
"""

from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Iterable, Mapping, MutableMapping, Sequence
from uuid import uuid4

import pandas as pd


try:  # Optional structured logger from the same stack.
    from ..logging import get_logger  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - package-layout dependent
    try:
        from logging import getLogger as _stdlib_get_logger

        def get_logger(name: str = "simons_core.io.parquet_store"):
            return _stdlib_get_logger(name)

    except Exception:  # pragma: no cover - defensive fallback
        get_logger = None  # type: ignore[assignment]


try:  # Preferred shared schema layer from simons_core.
    from ..schemas import (  # type: ignore[import-not-found]
        DataSchema,
        SchemaValidationError as CoreSchemaValidationError,
        assert_schema,
        format_validation_issues,
        get_schema,
        validate_schema,
    )
except Exception:  # pragma: no cover - package-layout dependent
    try:
        from schemas import (  # type: ignore[no-redef]
            DataSchema,
            SchemaValidationError as CoreSchemaValidationError,
            assert_schema,
            format_validation_issues,
            get_schema,
            validate_schema,
        )
    except Exception:  # pragma: no cover - graceful degradation
        DataSchema = Any  # type: ignore[assignment,misc]
        CoreSchemaValidationError = ValueError  # type: ignore[assignment]
        assert_schema = None  # type: ignore[assignment]
        format_validation_issues = None  # type: ignore[assignment]
        get_schema = None  # type: ignore[assignment]
        validate_schema = None  # type: ignore[assignment]


__all__ = [
    "CURRENT_POINTER_FILENAME",
    "DEFAULT_ALLOWED_ROOTS_ENV",
    "DEFAULT_COMPRESSION",
    "DEFAULT_LOGICAL_SORT_KEY",
    "DEFAULT_MANIFEST_FILENAME",
    "DEFAULT_SCHEMA_FILENAME",
    "DatasetNotFoundError",
    "InvalidPathError",
    "ManifestError",
    "ParquetStoreError",
    "SchemaCompatibilityIssue",
    "SchemaCompatibilityReport",
    "SchemaValidationError",
    "SnapshotManifest",
    "WriteConflictError",
    "CorruptPartitionError",
    "inspect_snapshot",
    "read_parquet",
    "validate_dataset",
    "write_parquet",
]


DEFAULT_ALLOWED_ROOTS_ENV = "SIMONS_PARQUET_ALLOWED_ROOTS"
DEFAULT_MANIFEST_FILENAME = "manifest.json"
DEFAULT_SCHEMA_FILENAME = "schema.json"
CURRENT_POINTER_FILENAME = "CURRENT.json"
DEFAULT_COMPRESSION = "snappy"
DEFAULT_LOGICAL_SORT_KEY = ("date", "symbol")
_WRITER_VERSION = "simons_core.io.parquet_store/1.0.0"
_PARQUET_SUFFIX = ".parquet"


class ParquetStoreError(RuntimeError):
    """Base exception for the Parquet snapshot store."""


class SchemaValidationError(ParquetStoreError):
    """Raised when a DataFrame or stored snapshot violates schema contracts."""


class ManifestError(ParquetStoreError):
    """Raised when a snapshot manifest is missing, unreadable or inconsistent."""


class DatasetNotFoundError(ParquetStoreError):
    """Raised when a dataset root or requested version cannot be resolved."""


class CorruptPartitionError(ParquetStoreError):
    """Raised when a physical Parquet partition/file is corrupt or unreadable."""


class InvalidPathError(ParquetStoreError):
    """Raised when a path escapes the allowed local filesystem contract."""


class WriteConflictError(ParquetStoreError):
    """Raised on destination/version collisions under protected write modes."""


@dataclass(frozen=True)
class SchemaCompatibilityIssue:
    level: str  # PASS | WARN | FAIL
    category: str
    column: str | None
    message: str


@dataclass(frozen=True)
class SchemaCompatibilityReport:
    status: str
    issues: tuple[SchemaCompatibilityIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass(frozen=True)
class SnapshotManifest:
    dataset_version: str
    schema_ref: str
    schema_hash: str
    created_at_utc: str
    row_count: int
    file_count: int
    partition_cols: tuple[str, ...]
    partitions: tuple[dict[str, Any], ...]
    checksums: dict[str, str]
    writer_version: str
    metadata: dict[str, Any] = field(default_factory=dict)
    total_bytes: int = 0
    deterministic_write_enabled: bool = True
    compression: str = DEFAULT_COMPRESSION
    logical_sort_key: tuple[str, ...] = DEFAULT_LOGICAL_SORT_KEY
    mean_file_size: float = 0.0
    pct_files_below_128mb: float = 0.0
    pct_files_above_1gb: float = 0.0
    physical_files: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "schema_ref": self.schema_ref,
            "schema_hash": self.schema_hash,
            "created_at_utc": self.created_at_utc,
            "row_count": self.row_count,
            "file_count": self.file_count,
            "partition_cols": list(self.partition_cols),
            "partitions": [dict(p) for p in self.partitions],
            "checksums": dict(self.checksums),
            "writer_version": self.writer_version,
            "metadata": _jsonable(self.metadata),
            "total_bytes": self.total_bytes,
            "deterministic_write_enabled": self.deterministic_write_enabled,
            "compression": self.compression,
            "logical_sort_key": list(self.logical_sort_key),
            "mean_file_size": self.mean_file_size,
            "pct_files_below_128mb": self.pct_files_below_128mb,
            "pct_files_above_1gb": self.pct_files_above_1gb,
            "physical_files": list(self.physical_files),
        }


def _logger():
    if get_logger is None:  # pragma: no cover - defensive
        return None
    try:
        return get_logger("simons_core.io.parquet_store")
    except Exception:  # pragma: no cover - defensive
        return None


def _emit(event: str, **payload: Any) -> None:
    logger = _logger()
    if logger is None:
        return
    try:
        log_fn = getattr(logger, "info", None)
        if callable(log_fn):
            log_fn(event, **payload)
    except TypeError:
        try:
            logger.info("%s %s", event, payload)
        except Exception:  # pragma: no cover - defensive
            return


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso_now() -> str:
    return _utc_now().isoformat()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    return repr(value)


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    encoded = json.dumps(_jsonable(payload), sort_keys=True, indent=2).encode("utf-8")
    with tmp.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_directory(path.parent)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise ManifestError(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Invalid JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestError(f"Expected JSON object at {path}, got {type(payload).__name__}")
    return payload


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - platform dependent
        return
    finally:
        os.close(fd)


def _resolve_allowed_roots(allowed_roots: Sequence[str | os.PathLike[str]] | None = None) -> tuple[Path, ...]:
    if allowed_roots is not None:
        roots = [Path(p).expanduser().resolve() for p in allowed_roots]
        return tuple(roots)

    raw = os.environ.get(DEFAULT_ALLOWED_ROOTS_ENV, "").strip()
    if not raw:
        return ()
    parts = [part for part in raw.split(os.pathsep) if part.strip()]
    return tuple(Path(part).expanduser().resolve() for part in parts)


def _resolve_local_dataset_root(
    path: str | os.PathLike[str],
    *,
    allowed_roots: Sequence[str | os.PathLike[str]] | None = None,
) -> Path:
    raw = str(path)
    if "://" in raw and not raw.startswith("file://"):
        raise InvalidPathError(
            "This MVP supports local filesystem paths only. "
            f"Got unsupported URI: {raw!r}"
        )

    if raw.startswith("file://"):
        raw = raw[len("file://") :]

    resolved = Path(raw).expanduser().resolve()
    roots = _resolve_allowed_roots(allowed_roots)
    if roots and not any(resolved == root or root in resolved.parents for root in roots):
        allowed = ", ".join(str(root) for root in roots)
        raise InvalidPathError(
            f"Path {resolved} escapes allowed roots. Allowed roots: {allowed}"
        )
    return resolved


def _versions_dir(root: Path) -> Path:
    return root / "versions"


def _pointer_path(root: Path) -> Path:
    return root / CURRENT_POINTER_FILENAME


def _version_dir(root: Path, dataset_version: str) -> Path:
    return _versions_dir(root) / dataset_version


def _manifest_path(version_dir: Path) -> Path:
    return version_dir / DEFAULT_MANIFEST_FILENAME


def _schema_path(version_dir: Path) -> Path:
    return version_dir / DEFAULT_SCHEMA_FILENAME


def _data_dir(version_dir: Path) -> Path:
    return version_dir / "data"


def _staging_dir(root: Path, dataset_version: str) -> Path:
    return root / ".staging" / f"{dataset_version}.{uuid4().hex}"


def _generate_dataset_version() -> str:
    now = _utc_now()
    return now.strftime("%Y%m%dT%H%M%S.%fZ")


def _canonical_sort_columns(df: pd.DataFrame, preferred: Sequence[str]) -> list[str]:
    return [column for column in preferred if column in df.columns]


def _normalize_partition_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if pd.isna(value):
        return "__NULL__"
    if isinstance(value, (int, float, str, bool)):
        return value
    return repr(value)


def _normalize_comparable(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return pd.Timestamp(value)
        except Exception:
            return value
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(value)
    return value


def _schema_to_metadata(schema_ref: str, df: pd.DataFrame) -> dict[str, Any]:
    if get_schema is None:
        return {
            "schema_ref": schema_ref,
            "schema_version": None,
            "columns": [
                {
                    "name": str(column),
                    "dtype": str(dtype),
                    "nullable": bool(df[column].isna().any()),
                    "required": True,
                }
                for column, dtype in df.dtypes.items()
            ],
            "primary_key": [],
        }

    schema = get_schema(schema_ref)
    columns: list[dict[str, Any]] = []
    for spec in getattr(schema, "columns", ()):
        columns.append(
            {
                "name": getattr(spec, "name", None),
                "dtype": getattr(spec, "dtype", None),
                "nullable": bool(getattr(spec, "nullable", False)),
                "required": bool(getattr(spec, "required", True)),
            }
        )
    return {
        "schema_ref": getattr(schema, "name", schema_ref),
        "schema_version": getattr(schema, "version", None),
        "columns": columns,
        "primary_key": list(getattr(schema, "primary_key", ()) or ()),
        "allow_extra_columns": bool(getattr(schema, "allow_extra_columns", False)),
    }


def _schema_hash(schema_meta: Mapping[str, Any]) -> str:
    blob = json.dumps(_jsonable(schema_meta), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(blob)


def _dtype_family(dtype_name: str | None) -> str:
    if dtype_name is None:
        return "unknown"
    name = str(dtype_name).lower()
    if any(token in name for token in ("int",)):
        return "int"
    if any(token in name for token in ("float", "double")):
        return "float"
    if name in {"numeric", "number"}:
        return "numeric"
    if "bool" in name or name == "binary":
        return "bool"
    if any(token in name for token in ("datetime", "timestamp", "date")):
        return "datetime"
    if any(token in name for token in ("string", "str", "object", "category")):
        return "string"
    return name


def _type_widen_or_same(stored_dtype: str | None, expected_dtype: str | None) -> str:
    stored = _dtype_family(stored_dtype)
    expected = _dtype_family(expected_dtype)
    if stored == expected:
        return "same"
    if stored == "int" and expected in {"float", "numeric"}:
        return "widen"
    if stored == "float" and expected == "numeric":
        return "widen"
    if stored == "numeric" and expected in {"int", "float"}:
        return "narrow"
    if stored == "float" and expected == "int":
        return "narrow"
    return "incompatible"


def assess_schema_compatibility(
    stored_schema: Mapping[str, Any],
    expected_schema: Mapping[str, Any],
) -> SchemaCompatibilityReport:
    """
    Assess compatibility using the explicit matrix requested by the specification.

    Practical interpretation used here:
    - stored schema = what the snapshot contains.
    - expected schema = what the caller wants to enforce.
    """
    issues: list[SchemaCompatibilityIssue] = []

    stored_columns = {
        str(col["name"]): col
        for col in stored_schema.get("columns", [])
        if isinstance(col, Mapping) and "name" in col
    }
    expected_columns = {
        str(col["name"]): col
        for col in expected_schema.get("columns", [])
        if isinstance(col, Mapping) and "name" in col
    }

    for column in sorted(expected_columns.keys() - stored_columns.keys()):
        issues.append(
            SchemaCompatibilityIssue(
                level="FAIL",
                category="drop_column",
                column=column,
                message=f"Snapshot is missing expected column {column!r}.",
            )
        )

    for column in sorted(stored_columns.keys() - expected_columns.keys()):
        issues.append(
            SchemaCompatibilityIssue(
                level="WARN",
                category="add_column",
                column=column,
                message=f"Snapshot has extra column {column!r} relative to expected schema.",
            )
        )

    for column in sorted(stored_columns.keys() & expected_columns.keys()):
        stored = stored_columns[column]
        expected = expected_columns[column]
        type_relation = _type_widen_or_same(
            stored.get("dtype"), expected.get("dtype")
        )
        if type_relation == "same":
            pass
        elif type_relation == "widen":
            issues.append(
                SchemaCompatibilityIssue(
                    level="WARN",
                    category="type_widen",
                    column=column,
                    message=(
                        f"Snapshot column {column!r} dtype {stored.get('dtype')!r} widens "
                        f"to expected {expected.get('dtype')!r}."
                    ),
                )
            )
        else:
            issues.append(
                SchemaCompatibilityIssue(
                    level="FAIL",
                    category="type_narrow_or_incompatible",
                    column=column,
                    message=(
                        f"Snapshot column {column!r} dtype {stored.get('dtype')!r} is not "
                        f"compatible with expected {expected.get('dtype')!r}."
                    ),
                )
            )

        stored_nullable = bool(stored.get("nullable", True))
        expected_nullable = bool(expected.get("nullable", True))
        if stored_nullable and not expected_nullable:
            issues.append(
                SchemaCompatibilityIssue(
                    level="FAIL",
                    category="nullable_strengthening_required",
                    column=column,
                    message=(
                        f"Snapshot column {column!r} is nullable but expected contract is non-nullable."
                    ),
                )
            )

    status = "PASS"
    if any(issue.level == "FAIL" for issue in issues):
        status = "FAIL"
    elif any(issue.level == "WARN" for issue in issues):
        status = "WARN"

    return SchemaCompatibilityReport(status=status, issues=tuple(issues))


def _validate_against_contract(df: pd.DataFrame, schema_ref: str) -> None:
    if assert_schema is None:
        return
    try:
        assert_schema(df, schema_ref)
    except CoreSchemaValidationError as exc:
        raise SchemaValidationError(str(exc)) from exc


def _partition_relative_path(values: Mapping[str, Any], partition_cols: Sequence[str]) -> Path:
    parts = [f"{column}={_normalize_partition_value(values[column])}" for column in partition_cols]
    return Path(*parts) if parts else Path()


def _collect_partition_values(df: pd.DataFrame, partition_cols: Sequence[str]) -> list[dict[str, Any]]:
    if not partition_cols:
        return [{}]
    if df.empty:
        return []
    unique = df.loc[:, list(partition_cols)].drop_duplicates().reset_index(drop=True)
    values: list[dict[str, Any]] = []
    for _, row in unique.iterrows():
        values.append({col: _normalize_partition_value(row[col]) for col in partition_cols})
    return values


def _apply_filter_to_scalar(value: Any, op: str, rhs: Any) -> bool:
    lhs = _normalize_comparable(value)
    rhs_norm = _normalize_comparable(rhs)
    op_norm = op.lower().strip()

    if op_norm in {"=", "=="}:
        return lhs == rhs_norm
    if op_norm in {"!=", "<>"}:
        return lhs != rhs_norm
    if op_norm == ">":
        return lhs > rhs_norm
    if op_norm == ">=":
        return lhs >= rhs_norm
    if op_norm == "<":
        return lhs < rhs_norm
    if op_norm == "<=":
        return lhs <= rhs_norm
    if op_norm == "in":
        return lhs in {_normalize_comparable(v) for v in rhs}
    if op_norm in {"not in", "not_in"}:
        return lhs not in {_normalize_comparable(v) for v in rhs}
    raise ValueError(f"Unsupported filter operator: {op!r}")


def _normalize_filters(filters: Iterable[tuple] | None) -> list[tuple[str, str, Any]]:
    if filters is None:
        return []
    normalized: list[tuple[str, str, Any]] = []
    for item in filters:
        if not isinstance(item, tuple) or len(item) != 3:
            raise ValueError(
                "Each filter must be a 3-tuple: (column, operator, value). "
                f"Got {item!r}"
            )
        column, operator, value = item
        normalized.append((str(column), str(operator), value))
    return normalized


def _partition_matches_filters(
    partition_values: Mapping[str, Any],
    filters: Sequence[tuple[str, str, Any]],
) -> bool:
    for column, operator, value in filters:
        if column not in partition_values:
            continue
        if not _apply_filter_to_scalar(partition_values[column], operator, value):
            return False
    return True


def _dataframe_matches_filters(df: pd.DataFrame, filters: Sequence[tuple[str, str, Any]]) -> pd.DataFrame:
    if not filters or df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    for column, operator, value in filters:
        if column not in df.columns:
            continue
        op = operator.lower().strip()
        series = df[column]
        if op in {"=", "=="}:
            mask &= series == value
        elif op in {"!=", "<>"}:
            mask &= series != value
        elif op == ">":
            mask &= series > value
        elif op == ">=":
            mask &= series >= value
        elif op == "<":
            mask &= series < value
        elif op == "<=":
            mask &= series <= value
        elif op == "in":
            mask &= series.isin(list(value))
        elif op in {"not in", "not_in"}:
            mask &= ~series.isin(list(value))
        else:
            raise ValueError(f"Unsupported filter operator: {operator!r}")
    return df.loc[mask]


def _iter_parquet_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        file
        for file in path.rglob(f"*{_PARQUET_SUFFIX}")
        if file.is_file() and ".staging" not in file.parts
    )


def _require_parquet_engine() -> None:
    try:
        pd.DataFrame({"x": [1]}).to_parquet  # noqa: B018 - attribute existence check
    except Exception as exc:  # pragma: no cover - defensive
        raise ParquetStoreError("pandas parquet support is unavailable in this runtime") from exc


def _write_parquet_file(df: pd.DataFrame, path: Path, compression: str) -> None:
    _require_parquet_engine()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression=compression)


def _read_parquet_file(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    _require_parquet_engine()
    try:
        return pd.read_parquet(path, columns=list(columns) if columns is not None else None)
    except Exception as exc:
        raise CorruptPartitionError(f"Failed to read Parquet file {path}: {exc}") from exc


def _infer_partition_values_from_path(path: Path, data_root: Path, partition_cols: Sequence[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if not partition_cols:
        return values
    try:
        relative_parent = path.parent.relative_to(data_root)
    except ValueError:
        return values
    for part in relative_parent.parts:
        if "=" not in part:
            continue
        column, raw_value = part.split("=", 1)
        if column in partition_cols:
            values[column] = raw_value
    return values


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _write_partitioned_dataset(
    df: pd.DataFrame,
    data_root: Path,
    *,
    partition_cols: Sequence[str],
    compression: str,
) -> None:
    data_root.mkdir(parents=True, exist_ok=True)

    if df.empty:
        empty_path = data_root / f"part-00000{_PARQUET_SUFFIX}"
        _write_parquet_file(df, empty_path, compression=compression)
        return

    if not partition_cols:
        single_path = data_root / f"part-00000{_PARQUET_SUFFIX}"
        _write_parquet_file(df.reset_index(drop=True), single_path, compression=compression)
        return

    grouped = df.groupby(list(partition_cols), dropna=False, sort=True)
    for idx, (keys, subdf) in enumerate(grouped):
        if not isinstance(keys, tuple):
            keys = (keys,)
        value_map = {
            column: _normalize_partition_value(value)
            for column, value in zip(partition_cols, keys, strict=False)
        }
        partition_path = data_root / _partition_relative_path(value_map, partition_cols)
        part_path = partition_path / f"part-{idx:05d}{_PARQUET_SUFFIX}"
        _write_parquet_file(subdf.reset_index(drop=True), part_path, compression=compression)


def _load_current_version(root: Path) -> str:
    pointer = _pointer_path(root)
    if not pointer.exists():
        raise DatasetNotFoundError(f"No current snapshot pointer found under {root}")
    payload = _read_json(pointer)
    version = payload.get("dataset_version")
    if not isinstance(version, str) or not version:
        raise ManifestError(f"Invalid current pointer under {root}: missing dataset_version")
    return version


def _resolve_requested_version(root: Path, dataset_version: str | None) -> str:
    if dataset_version is not None:
        return dataset_version
    return _load_current_version(root)


def _load_manifest(root: Path, dataset_version: str | None = None) -> dict[str, Any]:
    version = _resolve_requested_version(root, dataset_version)
    version_dir = _version_dir(root, version)
    if not version_dir.exists():
        raise DatasetNotFoundError(f"Dataset version {version!r} not found under {root}")
    manifest = _read_json(_manifest_path(version_dir))
    if manifest.get("dataset_version") != version:
        raise ManifestError(
            f"Manifest/version mismatch under {version_dir}: expected {version!r}, "
            f"found {manifest.get('dataset_version')!r}"
        )
    return manifest


def _load_schema_meta(root: Path, dataset_version: str | None = None) -> dict[str, Any]:
    version = _resolve_requested_version(root, dataset_version)
    version_dir = _version_dir(root, version)
    schema_meta = _read_json(_schema_path(version_dir))
    return schema_meta


def _build_partitions_summary(data_root: Path, partition_cols: Sequence[str]) -> tuple[dict[str, Any], ...]:
    files = _iter_parquet_files(data_root)
    if not files:
        return tuple()

    grouped: MutableMapping[str, dict[str, Any]] = {}
    for file in files:
        values = _infer_partition_values_from_path(file, data_root, partition_cols)
        key = json.dumps(_jsonable(values), sort_keys=True)
        entry = grouped.setdefault(
            key,
            {
                "values": values,
                "row_count": 0,
                "file_count": 0,
                "files": [],
                "total_bytes": 0,
            },
        )
        entry["file_count"] += 1
        entry["files"].append(str(file.relative_to(data_root)))
        entry["total_bytes"] += file.stat().st_size
        try:
            entry["row_count"] += len(_read_parquet_file(file))
        except CorruptPartitionError:
            # Row count for corrupt partition is intentionally not accumulated here.
            pass

    partitions = sorted(grouped.values(), key=lambda x: json.dumps(x["values"], sort_keys=True))
    for entry in partitions:
        entry["files"] = sorted(entry["files"])
    return tuple(partitions)


def _build_manifest(
    version_dir: Path,
    *,
    dataset_version: str,
    schema_ref: str,
    schema_meta: Mapping[str, Any],
    row_count: int,
    partition_cols: Sequence[str],
    metadata: Mapping[str, Any] | None,
    deterministic: bool,
    compression: str,
    logical_sort_key: Sequence[str],
) -> SnapshotManifest:
    data_root = _data_dir(version_dir)
    files = _iter_parquet_files(data_root)
    checksums = {
        str(file.relative_to(version_dir)): _sha256_file(file)
        for file in files
    }
    total_bytes = sum(file.stat().st_size for file in files)
    file_count = len(files)
    mean_file_size = float(total_bytes / file_count) if file_count else 0.0
    below_128 = sum(1 for file in files if file.stat().st_size < 128 * 1024 * 1024)
    above_1gb = sum(1 for file in files if file.stat().st_size > 1024 * 1024 * 1024)
    partitions = _build_partitions_summary(data_root, partition_cols)

    return SnapshotManifest(
        dataset_version=dataset_version,
        schema_ref=schema_ref,
        schema_hash=_schema_hash(schema_meta),
        created_at_utc=_utc_iso_now(),
        row_count=int(row_count),
        file_count=file_count,
        partition_cols=tuple(partition_cols),
        partitions=partitions,
        checksums=checksums,
        writer_version=_WRITER_VERSION,
        metadata=dict(_jsonable(metadata or {})),
        total_bytes=total_bytes,
        deterministic_write_enabled=deterministic,
        compression=compression,
        logical_sort_key=tuple(logical_sort_key),
        mean_file_size=mean_file_size,
        pct_files_below_128mb=float(below_128 / file_count) if file_count else 0.0,
        pct_files_above_1gb=float(above_1gb / file_count) if file_count else 0.0,
        physical_files=tuple(sorted(str(file.relative_to(version_dir)) for file in files)),
    )


def _validate_post_write_files(version_dir: Path) -> None:
    data_root = _data_dir(version_dir)
    files = _iter_parquet_files(data_root)
    if not files:
        raise ManifestError(f"No Parquet files produced under {data_root}")

    for file in files:
        _ = _read_parquet_file(file, columns=[])


def _publish_version(root: Path, staging_version_dir: Path, dataset_version: str) -> Path:
    final_version_dir = _version_dir(root, dataset_version)
    final_version_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_version_dir.exists():
        raise WriteConflictError(
            f"Version {dataset_version!r} already exists under {root}."
        )
    os.replace(staging_version_dir, final_version_dir)
    _fsync_directory(final_version_dir.parent)
    _write_json_atomic(
        _pointer_path(root),
        {
            "dataset_version": dataset_version,
            "updated_at_utc": _utc_iso_now(),
            "writer_version": _WRITER_VERSION,
        },
    )
    return final_version_dir


def _resolve_overwrite_base(root: Path, dataset_version: str | None) -> tuple[str, Path]:
    base_version = _resolve_requested_version(root, dataset_version)
    base_dir = _version_dir(root, base_version)
    if not base_dir.exists():
        raise DatasetNotFoundError(
            f"Base dataset version {base_version!r} not found for overwrite_partition."
        )
    return base_version, base_dir


def _prepare_staging_from_base(
    staging_version_dir: Path,
    base_version_dir: Path,
    *,
    partition_cols: Sequence[str],
    incoming_df: pd.DataFrame,
) -> None:
    _copy_tree(base_version_dir, staging_version_dir)
    stage_data_root = _data_dir(staging_version_dir)
    stage_data_root.mkdir(parents=True, exist_ok=True)

    if not partition_cols:
        _safe_rmtree(stage_data_root)
        stage_data_root.mkdir(parents=True, exist_ok=True)
        return

    for values in _collect_partition_values(incoming_df, partition_cols):
        rel = _partition_relative_path(values, partition_cols)
        _safe_rmtree(stage_data_root / rel)


def inspect_snapshot(
    path: str | os.PathLike[str],
    *,
    dataset_version: str | None = None,
    allowed_roots: Sequence[str | os.PathLike[str]] | None = None,
) -> dict[str, Any]:
    """Return manifest + schema metadata for a snapshot without materializing rows."""
    root = _resolve_local_dataset_root(path, allowed_roots=allowed_roots)
    manifest = _load_manifest(root, dataset_version)
    schema_meta = _load_schema_meta(root, dataset_version)
    return {
        "root": str(root),
        "dataset_version": manifest["dataset_version"],
        "manifest": manifest,
        "schema": schema_meta,
    }


def write_parquet(
    df: pd.DataFrame,
    path: str | os.PathLike[str],
    schema_ref: str,
    partition_cols: Sequence[str] | None = None,
    write_mode: str = "fail_if_exists",
    dataset_version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    deterministic: bool = True,
    *,
    compression: str = DEFAULT_COMPRESSION,
    logical_sort_key: Sequence[str] = DEFAULT_LOGICAL_SORT_KEY,
    allowed_roots: Sequence[str | os.PathLike[str]] | None = None,
) -> dict[str, Any]:
    """
    Publish a validated Parquet snapshot with a manifest and atomic visibility.

    Supported write modes
    ---------------------
    - ``fail_if_exists``:
        Fails if the logical destination already exists (when ``dataset_version``
        is omitted) or if the target version already exists (when specified).
    - ``append_versioned``:
        Publishes a new version and updates the current pointer.
    - ``overwrite_partition``:
        Creates a new version by copying the base snapshot and replacing only
        affected partitions from ``df``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"write_parquet expects a pandas.DataFrame, got {type(df).__name__}")

    mode = write_mode.strip().lower()
    if mode not in {"fail_if_exists", "append_versioned", "overwrite_partition"}:
        raise ValueError(
            "write_mode must be one of {'fail_if_exists', 'append_versioned', 'overwrite_partition'}"
        )

    root = _resolve_local_dataset_root(path, allowed_roots=allowed_roots)
    partition_cols = tuple(partition_cols or ())

    _validate_against_contract(df, schema_ref)
    schema_meta = _schema_to_metadata(schema_ref, df)

    sort_key = tuple(_canonical_sort_columns(df, logical_sort_key))
    frame = df.copy()
    if deterministic and sort_key:
        frame = frame.sort_values(list(sort_key)).reset_index(drop=True)

    root.mkdir(parents=True, exist_ok=True)

    if mode == "fail_if_exists":
        if dataset_version is None:
            has_versions = _versions_dir(root).exists() and any(_versions_dir(root).glob("*"))
            if _pointer_path(root).exists() or has_versions:
                raise WriteConflictError(
                    f"Logical destination {root} already has a visible snapshot."
                )
            effective_version = _generate_dataset_version()
        else:
            effective_version = dataset_version
            if _version_dir(root, effective_version).exists():
                raise WriteConflictError(
                    f"Dataset version {effective_version!r} already exists under {root}."
                )
    else:
        effective_version = dataset_version or _generate_dataset_version()
        if _version_dir(root, effective_version).exists():
            raise WriteConflictError(
                f"Dataset version {effective_version!r} already exists under {root}."
            )

    staging_root = _staging_dir(root, effective_version)
    staging_version_dir = staging_root / effective_version
    stage_data_root = _data_dir(staging_version_dir)

    start = time.perf_counter()
    try:
        staging_version_dir.mkdir(parents=True, exist_ok=True)

        if mode == "overwrite_partition":
            base_version, base_dir = _resolve_overwrite_base(root, None)
            base_manifest = _load_manifest(root, base_version)
            base_partition_cols = tuple(base_manifest.get("partition_cols", []))
            if base_partition_cols != partition_cols:
                raise WriteConflictError(
                    "overwrite_partition requires incoming partition_cols to match the base snapshot. "
                    f"Base={base_partition_cols!r}, incoming={partition_cols!r}"
                )
            _prepare_staging_from_base(
                staging_version_dir,
                base_dir,
                partition_cols=partition_cols,
                incoming_df=frame,
            )
        else:
            stage_data_root.mkdir(parents=True, exist_ok=True)

        if not (mode == "overwrite_partition" and frame.empty):
            _write_partitioned_dataset(
                frame,
                _data_dir(staging_version_dir),
                partition_cols=partition_cols,
                compression=compression,
            )

        schema_payload = {
            **schema_meta,
            "schema_ref": schema_ref,
            "schema_hash": _schema_hash(schema_meta),
            "written_at_utc": _utc_iso_now(),
        }
        _write_json_atomic(_schema_path(staging_version_dir), schema_payload)

        _validate_post_write_files(staging_version_dir)

        manifest = _build_manifest(
            staging_version_dir,
            dataset_version=effective_version,
            schema_ref=schema_ref,
            schema_meta=schema_payload,
            row_count=len(frame) if mode != "overwrite_partition" else _compute_total_rows(_data_dir(staging_version_dir)),
            partition_cols=partition_cols,
            metadata=metadata,
            deterministic=deterministic,
            compression=compression,
            logical_sort_key=sort_key,
        )
        _write_json_atomic(_manifest_path(staging_version_dir), manifest.to_dict())

        published_version_dir = _publish_version(root, staging_version_dir, effective_version)
        elapsed = time.perf_counter() - start
        _emit(
            "parquet_store.write",
            path=str(root),
            dataset_version=effective_version,
            write_mode=mode,
            rows_written=manifest.row_count,
            bytes_written=manifest.total_bytes,
            file_count=manifest.file_count,
            write_latency_sec=elapsed,
            partition_cols=list(partition_cols),
            schema_ref=schema_ref,
        )
        return {
            "path": str(root),
            "version_dir": str(published_version_dir),
            "dataset_version": effective_version,
            "manifest": manifest.to_dict(),
            "schema": schema_payload,
            "metrics": {
                "rows_written": manifest.row_count,
                "bytes_written": manifest.total_bytes,
                "write_latency_sec": elapsed,
                "file_count": manifest.file_count,
                "mean_file_size": manifest.mean_file_size,
                "pct_files_below_128mb": manifest.pct_files_below_128mb,
                "pct_files_above_1gb": manifest.pct_files_above_1gb,
            },
        }
    except Exception:
        _safe_rmtree(staging_root)
        raise
    finally:
        _safe_rmtree(staging_root)


def _compute_total_rows(data_root: Path) -> int:
    total = 0
    for file in _iter_parquet_files(data_root):
        total += len(_read_parquet_file(file))
    return total


def _select_files_for_read(
    manifest: Mapping[str, Any],
    version_dir: Path,
    filters: Sequence[tuple[str, str, Any]],
) -> tuple[list[Path], int, int]:
    data_root = _data_dir(version_dir)
    partition_cols = tuple(manifest.get("partition_cols", []))
    files: list[Path] = []
    pruned_partitions = 0
    selected_partitions = 0

    partitions = manifest.get("partitions") or []
    if partitions:
        for part in partitions:
            values = dict(part.get("values", {})) if isinstance(part, Mapping) else {}
            if _partition_matches_filters(values, filters):
                selected_partitions += 1
                for rel in part.get("files", []):
                    files.append(data_root / rel)
            else:
                pruned_partitions += 1
    else:
        files = _iter_parquet_files(data_root)

    if not partitions:
        selected_partitions = len(files)

    files = [file for file in sorted(set(files)) if file.exists()]
    return files, selected_partitions, pruned_partitions


def read_parquet(
    path: str | os.PathLike[str],
    columns: Sequence[str] | None = None,
    filters: Iterable[tuple] | None = None,
    schema_ref: str | None = None,
    dataset_version: str | None = None,
    strict_schema: bool = True,
    *,
    allowed_roots: Sequence[str | os.PathLike[str]] | None = None,
) -> pd.DataFrame:
    """
    Read a visible snapshot or explicit dataset version.

    The reader materializes a coherent version only; it never mixes files from
    different snapshots.
    """
    root = _resolve_local_dataset_root(path, allowed_roots=allowed_roots)
    normalized_filters = _normalize_filters(filters)
    manifest = _load_manifest(root, dataset_version)
    version = manifest["dataset_version"]
    version_dir = _version_dir(root, version)

    stored_schema = _load_schema_meta(root, version)
    if schema_ref is not None:
        expected_schema = _schema_to_metadata(schema_ref, pd.DataFrame())
        compatibility = assess_schema_compatibility(stored_schema, expected_schema)
        if strict_schema and compatibility.status == "FAIL":
            details = "; ".join(issue.message for issue in compatibility.issues)
            raise SchemaValidationError(
                f"Schema compatibility check failed for version {version!r}: {details}"
            )
    else:
        compatibility = None

    files, selected_partitions, pruned_partitions = _select_files_for_read(
        manifest,
        version_dir,
        normalized_filters,
    )
    if not files:
        selected_columns = list(columns) if columns is not None else None
        empty = pd.DataFrame(columns=selected_columns)
        empty.attrs["snapshot_manifest"] = manifest
        return empty

    start = time.perf_counter()
    frames: list[pd.DataFrame] = []
    corrupt_files: list[str] = []
    total_bytes_read = 0

    for file in files:
        total_bytes_read += file.stat().st_size
        try:
            frame = _read_parquet_file(file, columns=columns)
        except CorruptPartitionError:
            if strict_schema:
                raise
            corrupt_files.append(str(file))
            continue
        frames.append(frame)

    if not frames:
        if corrupt_files:
            raise CorruptPartitionError(
                "All selected partitions were corrupt or unreadable. "
                f"Files: {corrupt_files}"
            )
        selected_columns = list(columns) if columns is not None else None
        result = pd.DataFrame(columns=selected_columns)
        result.attrs["snapshot_manifest"] = manifest
        return result

    result = pd.concat(frames, axis=0, ignore_index=True, copy=False)
    result = _dataframe_matches_filters(result, normalized_filters)

    if schema_ref is not None:
        validation_result = validate_schema(result, schema_ref) if validate_schema is not None else None
        if validation_result is not None and not validation_result.passed and strict_schema:
            if format_validation_issues is not None:
                raise SchemaValidationError(format_validation_issues(validation_result))
            raise SchemaValidationError("Read result failed schema validation.")

    elapsed = time.perf_counter() - start
    projected_columns = len(columns) if columns is not None else len(result.columns)
    rho_c_est = float(projected_columns / max(len(result.columns), 1)) if len(result.columns) else 1.0
    rho_p_est = float(selected_partitions / max(selected_partitions + pruned_partitions, 1))

    _emit(
        "parquet_store.read",
        path=str(root),
        dataset_version=version,
        rows_read=int(len(result)),
        bytes_read=int(total_bytes_read),
        projected_columns=int(projected_columns),
        pruned_partitions=int(pruned_partitions),
        read_latency_sec=float(elapsed),
        rho_c_est=float(rho_c_est),
        rho_p_est=float(rho_p_est),
        corrupt_partitions=int(len(corrupt_files)),
    )

    result.attrs["snapshot_manifest"] = manifest
    result.attrs["schema_compatibility"] = compatibility.to_dict() if compatibility else None
    result.attrs["read_metrics"] = {
        "rows_read": int(len(result)),
        "bytes_read": int(total_bytes_read),
        "projected_columns": int(projected_columns),
        "pruned_partitions": int(pruned_partitions),
        "rho_c_est": float(rho_c_est),
        "rho_p_est": float(rho_p_est),
        "read_latency_sec": float(elapsed),
        "corrupt_partitions": int(len(corrupt_files)),
    }
    return result


def validate_dataset(
    path: str | os.PathLike[str],
    schema_ref: str,
    dataset_version: str | None = None,
    *,
    allowed_roots: Sequence[str | os.PathLike[str]] | None = None,
    quarantine_corrupt: bool = False,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Validate snapshot integrity, schema metadata and physical file health.

    Parameters beyond the minimal public API are intentionally conservative:
    ``quarantine_corrupt`` moves corrupt files under ``_quarantine`` inside the
    version directory; ``strict`` turns corruption into an exception.
    """
    root = _resolve_local_dataset_root(path, allowed_roots=allowed_roots)
    manifest = _load_manifest(root, dataset_version)
    version = manifest["dataset_version"]
    version_dir = _version_dir(root, version)
    stored_schema = _load_schema_meta(root, version)
    expected_schema = _schema_to_metadata(schema_ref, pd.DataFrame())
    compatibility = assess_schema_compatibility(stored_schema, expected_schema)

    data_root = _data_dir(version_dir)
    files = _iter_parquet_files(data_root)
    missing_files: list[str] = []
    corrupt_files: list[str] = []
    checksum_mismatches: list[str] = []
    rows_observed = 0
    total_bytes = 0

    for rel, expected_checksum in dict(manifest.get("checksums", {})).items():
        path_obj = version_dir / rel
        if not path_obj.exists():
            missing_files.append(rel)

    for file in files:
        rel = str(file.relative_to(version_dir))
        total_bytes += file.stat().st_size
        try:
            frame = _read_parquet_file(file)
            rows_observed += len(frame)
        except CorruptPartitionError:
            corrupt_files.append(rel)
            if quarantine_corrupt:
                quarantine_target = version_dir / "_quarantine" / rel
                quarantine_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file), str(quarantine_target))
            continue

        expected_checksum = dict(manifest.get("checksums", {})).get(rel)
        if expected_checksum is not None:
            observed_checksum = _sha256_file(file)
            if observed_checksum != expected_checksum:
                checksum_mismatches.append(rel)

    issues: list[str] = []
    if compatibility.status == "FAIL":
        issues.extend(issue.message for issue in compatibility.issues)
    if missing_files:
        issues.append(f"Missing files referenced by manifest: {missing_files}")
    if corrupt_files:
        issues.append(f"Corrupt partitions detected: {corrupt_files}")
    if checksum_mismatches:
        issues.append(f"Checksum mismatches detected: {checksum_mismatches}")
    if rows_observed != int(manifest.get("row_count", -1)):
        issues.append(
            f"Row-count mismatch: manifest={manifest.get('row_count')} observed={rows_observed}"
        )

    file_count = len(files)
    mean_file_size = float(total_bytes / file_count) if file_count else 0.0
    summary = {
        "path": str(root),
        "dataset_version": version,
        "schema_ref": schema_ref,
        "schema_compatibility": compatibility.to_dict(),
        "row_count_manifest": int(manifest.get("row_count", 0)),
        "row_count_observed": int(rows_observed),
        "file_count_manifest": int(manifest.get("file_count", 0)),
        "file_count_observed": int(file_count),
        "missing_files": missing_files,
        "corrupt_partitions": corrupt_files,
        "checksum_mismatches": checksum_mismatches,
        "total_bytes": int(total_bytes),
        "mean_file_size": mean_file_size,
        "pct_files_below_128mb": float(
            sum(1 for file in files if file.stat().st_size < 128 * 1024 * 1024) / max(file_count, 1)
        ),
        "pct_files_above_1gb": float(
            sum(1 for file in files if file.stat().st_size > 1024 * 1024 * 1024) / max(file_count, 1)
        ),
        "valid": len(issues) == 0,
        "issues": issues,
    }

    _emit(
        "parquet_store.validate",
        path=str(root),
        dataset_version=version,
        schema_validation_failures=int(sum(1 for _ in compatibility.issues if _.level == "FAIL")),
        corrupt_partitions=int(len(corrupt_files)),
        file_count=int(file_count),
        rows_read=int(rows_observed),
        bytes_read=int(total_bytes),
    )

    if strict and issues:
        raise ManifestError("; ".join(issues))
    return summary


def _parse_cli_filter(value: str) -> tuple[str, str, Any]:
    operators = [">=", "<=", "!=", "==", "=", ">", "<"]
    for operator in operators:
        if operator in value:
            left, right = value.split(operator, 1)
            return left.strip(), operator, right.strip()
    if " in " in value:
        left, right = value.split(" in ", 1)
        items = [item.strip() for item in right.split(",") if item.strip()]
        return left.strip(), "in", items
    raise ValueError(f"Unable to parse filter expression: {value!r}")


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="python -m simons_core.io.parquet_store")
    sub = parser.add_subparsers(dest="command", required=True)

    validate_cmd = sub.add_parser("validate", help="Validate a snapshot")
    validate_cmd.add_argument("--path", required=True)
    validate_cmd.add_argument("--schema", required=True)
    validate_cmd.add_argument("--version", default=None)
    validate_cmd.add_argument("--quarantine-corrupt", action="store_true")
    validate_cmd.add_argument("--no-strict", action="store_true")

    inspect_cmd = sub.add_parser("inspect", help="Inspect manifest/schema metadata")
    inspect_cmd.add_argument("--path", required=True)
    inspect_cmd.add_argument("--version", default=None)

    read_cmd = sub.add_parser("read", help="Read a snapshot with projection/filters")
    read_cmd.add_argument("--path", required=True)
    read_cmd.add_argument("--version", default=None)
    read_cmd.add_argument("--schema", default=None)
    read_cmd.add_argument("--columns", nargs="*")
    read_cmd.add_argument("--filter", action="append", default=[])
    read_cmd.add_argument("--no-strict-schema", action="store_true")
    read_cmd.add_argument("--head", type=int, default=5)

    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        payload = validate_dataset(
            path=args.path,
            schema_ref=args.schema,
            dataset_version=args.version,
            quarantine_corrupt=args.quarantine_corrupt,
            strict=not args.no_strict,
        )
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0

    if args.command == "inspect":
        payload = inspect_snapshot(path=args.path, dataset_version=args.version)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0

    if args.command == "read":
        filters = [_parse_cli_filter(expr) for expr in args.filter]
        df = read_parquet(
            path=args.path,
            columns=args.columns,
            filters=filters,
            schema_ref=args.schema,
            dataset_version=args.version,
            strict_schema=not args.no_strict_schema,
        )
        print(df.head(args.head).to_string(index=False))
        print()
        print(json.dumps(df.attrs.get("read_metrics", {}), indent=2, sort_keys=True, default=str))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_main())
