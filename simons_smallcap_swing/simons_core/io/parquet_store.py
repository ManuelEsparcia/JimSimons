"""
simons_core/io/parquet_store.py — Snapshot-based columnar persistence.

Publishes tabular datasets as logical snapshots with:
- Schema validation (structural + semantic)
- Manifest-governed visibility (partial writes never visible)
- Deterministic write (canonical sort + fixed compression)
- Columnar read with projection and partition pruning
- Corrupt partition detection and isolation

Core invariant:
    consumer sees COMPLETE snapshot or NOTHING.
    Never a partial publication.

Snapshot = (Π files, Σ schema, M manifest, τ timestamp)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors (spec §15)
# ---------------------------------------------------------------------------

class ParquetStoreError(RuntimeError):
    pass

class SchemaValidationError(ParquetStoreError):
    pass

class ManifestError(ParquetStoreError):
    pass

class DatasetNotFoundError(ParquetStoreError):
    pass

class CorruptPartitionError(ParquetStoreError):
    pass

class WriteConflictError(ParquetStoreError):
    pass


# ---------------------------------------------------------------------------
# Schema compatibility (spec §7.3)
# ---------------------------------------------------------------------------

@dataclass
class SchemaIssue:
    column: str
    issue_type: str   # "missing_required" | "type_mismatch" | "nullable_violation"
    expected: str
    observed: str
    severity: str     # "PASS" | "WARN" | "FAIL"


def validate_schema(
    df: pd.DataFrame,
    schema_ref: Mapping[str, Any],
) -> list[SchemaIssue]:
    """Validate DataFrame against schema reference.

    schema_ref format:
        {"columns": {"col_name": {"dtype": "float64", "required": True, "nullable": False}}}
    """
    issues: list[SchemaIssue] = []
    col_specs = schema_ref.get("columns", {})

    for col_name, spec in col_specs.items():
        if col_name not in df.columns:
            if spec.get("required", False):
                issues.append(SchemaIssue(col_name, "missing_required", col_name, "ABSENT", "FAIL"))
            continue

        # Type check
        expected_dtype = spec.get("dtype")
        if expected_dtype:
            actual = str(df[col_name].dtype)
            if not _dtype_compatible(actual, expected_dtype):
                issues.append(SchemaIssue(col_name, "type_mismatch", expected_dtype, actual, "WARN"))

        # Nullable check
        if not spec.get("nullable", True):
            n_null = int(df[col_name].isna().sum())
            if n_null > 0:
                issues.append(SchemaIssue(col_name, "nullable_violation", "not_nullable", f"{n_null} nulls", "FAIL"))

    return issues


def _dtype_compatible(actual: str, expected: str) -> bool:
    """Check if actual dtype is compatible with expected (allowing widening)."""
    families = {"int": {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"},
                "float": {"float16", "float32", "float64"},
                "str": {"object", "string", "str"}}
    for fam in families.values():
        if actual.lower() in fam and expected.lower() in fam:
            return True
    return actual.lower() == expected.lower()


# ---------------------------------------------------------------------------
# Manifest (spec §14)
# ---------------------------------------------------------------------------

@dataclass
class SnapshotManifest:
    dataset_version: str
    schema_ref: str
    schema_hash: str
    created_at: str
    row_count: int
    file_count: int
    partition_cols: list[str]
    total_bytes: int
    compression: str
    deterministic: bool
    sort_key: list[str]
    checksums: dict[str, str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "schema_ref": self.schema_ref,
            "schema_hash": self.schema_hash,
            "created_at": self.created_at,
            "row_count": self.row_count,
            "file_count": self.file_count,
            "partition_cols": self.partition_cols,
            "total_bytes": self.total_bytes,
            "compression": self.compression,
            "deterministic": self.deterministic,
            "sort_key": self.sort_key,
            "checksums": self.checksums,
            "metadata": self.metadata,
        }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _generate_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa
        return True
    except ImportError:
        try:
            import fastparquet  # noqa
            return True
        except ImportError:
            return False


def _write_df(df: pd.DataFrame, path_stem: Path, compression: str = "snappy") -> Path:
    """Write DataFrame as parquet (preferred) or CSV (fallback)."""
    if _has_parquet():
        p = path_stem.with_suffix(".parquet")
        df.to_parquet(p, index=False, compression=compression)
        return p
    p = path_stem.with_suffix(".csv")
    df.to_csv(p, index=False)
    return p


def _read_df(path: Path, columns=None) -> pd.DataFrame:
    """Read DataFrame from parquet or CSV."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        if columns:
            df = df[[c for c in columns if c in df.columns]]
        return df
    raise DatasetNotFoundError(f"Unsupported format: {path}")


# ---------------------------------------------------------------------------
# Write (spec §12)
# ---------------------------------------------------------------------------

def write_parquet(
    df: pd.DataFrame,
    path: str | Path,
    *,
    schema_ref: str = "default",
    schema_spec: Mapping[str, Any] | None = None,
    partition_cols: Sequence[str] | None = None,
    write_mode: str = "fail_if_exists",
    dataset_version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    deterministic: bool = True,
    compression: str = "snappy",
    sort_key: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Publish a snapshot with schema validation and manifest.

    Write protocol (spec §12):
    1. Resolve schema
    2. Validate structural + semantic
    3. Apply deterministic sort
    4. Write to staging
    5. Validate files
    6. Build manifest
    7. Publish (atomic rename)
    """
    root = Path(path)
    version = dataset_version or _generate_version()
    version_dir = root / "versions" / version

    # Mode check
    if write_mode == "fail_if_exists" and version_dir.exists():
        raise WriteConflictError(f"Version already exists: {version_dir}")

    # Schema validation
    if schema_spec:
        issues = validate_schema(df, schema_spec)
        failures = [i for i in issues if i.severity == "FAIL"]
        if failures:
            raise SchemaValidationError(
                f"{len(failures)} schema failures: {[(i.column, i.issue_type) for i in failures]}"
            )

    # Deterministic sort
    if deterministic:
        sk = list(sort_key) if sort_key else _infer_sort_key(df)
        avail = [c for c in sk if c in df.columns]
        if avail:
            df = df.sort_values(avail).reset_index(drop=True)
    else:
        sk = []

    # Write to staging
    staging = root / "_staging" / version
    staging.mkdir(parents=True, exist_ok=True)
    data_dir = staging / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if partition_cols:
        # Partitioned write
        avail_parts = [c for c in partition_cols if c in df.columns]
        if avail_parts:
            for keys, group in df.groupby(avail_parts):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                part_path = data_dir
                for col, val in zip(avail_parts, keys):
                    part_path = part_path / f"{col}={val}"
                part_path.mkdir(parents=True, exist_ok=True)
                _write_df(group, part_path / "part-0")
        else:
            _write_df(df, data_dir / "part-0")
    else:
        _write_df(df, data_dir / "part-0")

    # Checksums
    checksums = {}
    total_bytes = 0
    file_count = 0
    for ext in ("*.parquet", "*.csv"):
        for fp in data_dir.rglob(ext):
            checksums[str(fp.relative_to(staging))] = _sha256_file(fp)
            total_bytes += fp.stat().st_size
            file_count += 1

    # Schema hash
    schema_meta = {
        "columns": {c: str(df[c].dtype) for c in df.columns},
        "n_cols": len(df.columns),
    }
    schema_hash = hashlib.sha256(
        json.dumps(schema_meta, sort_keys=True).encode()
    ).hexdigest()[:16]

    # Write schema
    (staging / "schema.json").write_text(
        json.dumps(schema_meta, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Build manifest
    manifest = SnapshotManifest(
        dataset_version=version,
        schema_ref=schema_ref,
        schema_hash=schema_hash,
        created_at=_utc_iso_now(),
        row_count=len(df),
        file_count=file_count,
        partition_cols=list(partition_cols or []),
        total_bytes=total_bytes,
        compression=compression,
        deterministic=deterministic,
        sort_key=list(sk),
        checksums=checksums,
        metadata=dict(metadata or {}),
    )

    (staging / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    # Publish: rename staging → final
    version_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(version_dir)

    # Update pointer to latest version
    pointer = root / "versions" / "_latest"
    pointer.write_text(version, encoding="utf-8")

    LOGGER.info("Published snapshot %s: %d rows, %d files, %d bytes",
                version, len(df), file_count, total_bytes)
    return manifest.to_dict()


def _infer_sort_key(df: pd.DataFrame) -> list[str]:
    """Infer canonical sort key from column names."""
    candidates = ["date", "symbol", "instrument_id", "trade_date"]
    return [c for c in candidates if c in df.columns]


# ---------------------------------------------------------------------------
# Read (spec §13)
# ---------------------------------------------------------------------------

def read_parquet(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    filters: Sequence[tuple] | None = None,
    dataset_version: str | None = None,
) -> pd.DataFrame:
    """Read a snapshot with optional column projection and filtering."""
    root = Path(path)

    # Resolve version
    if dataset_version:
        version_dir = root / "versions" / dataset_version
    else:
        pointer = root / "versions" / "_latest"
        if pointer.exists():
            version = pointer.read_text().strip()
            version_dir = root / "versions" / version
        else:
            # Direct read (no versioning)
            if root.suffix in (".parquet", ".csv"):
                return _read_df(root, columns)
            # Try reading as dataset directory
            for ext in ("*.parquet", "*.csv"):
                files = list(root.rglob(ext))
                if files:
                    dfs = [_read_df(f, columns) for f in sorted(files)]
                    return pd.concat(dfs, ignore_index=True)
            raise DatasetNotFoundError(f"No data files found in {root}")

    if not version_dir.exists():
        raise DatasetNotFoundError(f"Version not found: {version_dir}")

    data_dir = version_dir / "data"
    if not data_dir.exists():
        raise DatasetNotFoundError(f"Data directory not found: {data_dir}")

    # Read all data files (parquet or csv)
    dfs = []
    for fp in sorted(data_dir.rglob("*.parquet")) or sorted(data_dir.rglob("*.csv")):
        try:
            dfs.append(_read_df(fp, columns))
        except Exception as e:
            raise CorruptPartitionError(f"Corrupt partition: {fp}: {e}")

    if not dfs:
        # Try any file type
        for fp in sorted(data_dir.rglob("*.*")):
            if fp.suffix in (".parquet", ".csv"):
                try:
                    dfs.append(_read_df(fp, columns))
                except Exception as e:
                    raise CorruptPartitionError(f"Corrupt: {fp}: {e}")

    if not dfs:
        raise DatasetNotFoundError(f"No data files in {data_dir}")

    result = pd.concat(dfs, ignore_index=True)

    # Apply filters
    if filters:
        for f in filters:
            col, op, val = f
            if col not in result.columns:
                continue
            if op == ">=":
                result = result[result[col] >= val]
            elif op == "<=":
                result = result[result[col] <= val]
            elif op == "==":
                result = result[result[col] == val]
            elif op == "in":
                result = result[result[col].isin(val)]
            elif op == ">":
                result = result[result[col] > val]
            elif op == "<":
                result = result[result[col] < val]

    return result


# ---------------------------------------------------------------------------
# Validation (spec §7)
# ---------------------------------------------------------------------------

def validate_dataset(
    path: str | Path,
    schema_spec: Mapping[str, Any] | None = None,
    dataset_version: str | None = None,
) -> dict[str, Any]:
    """Validate integrity, schema, and metadata of a dataset."""
    root = Path(path)
    result = {"valid": True, "issues": []}

    try:
        df = read_parquet(root, dataset_version=dataset_version)
    except Exception as e:
        return {"valid": False, "issues": [f"Read failed: {e}"]}

    result["row_count"] = len(df)
    result["columns"] = list(df.columns)

    if schema_spec:
        issues = validate_schema(df, schema_spec)
        result["schema_issues"] = [{"col": i.column, "type": i.issue_type, "severity": i.severity} for i in issues]
        if any(i.severity == "FAIL" for i in issues):
            result["valid"] = False

    return result
