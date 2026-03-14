from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import uuid
from typing import Any, Iterable

import pandas as pd

from . import paths as path_utils


class ParquetStoreError(Exception):
    """Base parquet store error."""


class ParquetValidationError(ParquetStoreError):
    """Raised when a parquet dataset violates minimal contract checks."""


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        return (path_utils.app_root() / candidate).resolve(strict=False)
    return candidate.resolve(strict=False)


def _write_json_atomic(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + f".tmp.{uuid.uuid4().hex}")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, destination)


def _build_manifest(
    df: pd.DataFrame,
    parquet_path: Path,
    *,
    schema_name: str | None,
    run_id: str | None,
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_path": str(parquet_path),
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "schema_name": schema_name,
        "run_id": run_id,
    }


def write_parquet(
    df: pd.DataFrame,
    path: str | Path,
    *,
    compression: str = "zstd",
    index: bool = False,
    atomic: bool = True,
    create_manifest: bool = True,
    schema_name: str | None = None,
    run_id: str | None = None,
) -> Path:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("write_parquet expects a pandas DataFrame.")
    if df.empty:
        raise ParquetValidationError("write_parquet received an empty DataFrame.")

    destination = _resolve_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if atomic:
        tmp_path = destination.with_suffix(destination.suffix + f".tmp.{uuid.uuid4().hex}")
        df.to_parquet(tmp_path, compression=compression, index=index)
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            raise ParquetStoreError("Atomic parquet write produced an empty temporary file.")
        os.replace(tmp_path, destination)
    else:
        df.to_parquet(destination, compression=compression, index=index)

    if not destination.exists() or destination.stat().st_size == 0:
        raise ParquetStoreError(f"Parquet write failed or produced empty file: {destination}")

    if create_manifest:
        manifest = _build_manifest(df, destination, schema_name=schema_name, run_id=run_id)
        manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
        _write_json_atomic(manifest, manifest_path)

    return destination


def read_parquet(
    path: str | Path,
    *,
    columns: Iterable[str] | None = None,
    required_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    source = _resolve_path(path)
    if not source.exists():
        raise FileNotFoundError(f"Parquet dataset does not exist: {source}")
    if source.stat().st_size == 0:
        raise ParquetValidationError(f"Parquet dataset is empty: {source}")

    try:
        df = pd.read_parquet(source, columns=list(columns) if columns else None)
    except Exception as exc:  # pragma: no cover - dependent on parquet backend internals
        raise ParquetStoreError(f"Failed reading parquet dataset: {source}") from exc

    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ParquetValidationError(
                f"Missing required columns {missing} in dataset: {source}"
            )

    return df


def validate_dataset(path: str | Path, *, required_columns: Iterable[str] | None = None) -> None:
    _ = read_parquet(path, required_columns=required_columns)


__all__ = [
    "ParquetStoreError",
    "ParquetValidationError",
    "read_parquet",
    "validate_dataset",
    "write_parquet",
]
