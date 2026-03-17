from __future__ import annotations

"""
simons_core.io.cache
====================

Deterministic local cache layer for the quant stack.

This module is intentionally **operational** rather than statistical: it does
not change alpha, labels, risk estimates or portfolio logic. Its purpose is to
speed up recomputation while preserving strict semantic equivalence between a
valid cache hit and recomputing the canonical producer with the same logical
inputs.

Design goals
------------
- Deterministic canonical keys based on cryptographic hashes.
- Atomic single-entry writes via temp file -> fsync -> rename.
- Fail-closed reads on expiry, corruption or unsupported serializer.
- Explicit invalidation by namespace, tag or producer.
- Byte-budgeted LRU eviction using file mtimes as the recency signal.
- Sufficient observability for research and production operations.

Scope limitations
-----------------
This is a **local single-store cache**. The supported concurrency model is
single-writer / multi-reader. The module does not attempt distributed
coordination, consensus, replication or multi-node cache coherence.
"""

from argparse import ArgumentParser, Namespace
import builtins
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import struct
import tempfile
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Protocol, Sequence
from uuid import uuid4

try:  # Optional dependency; the rest of the module works without pandas.
    import pandas as pd
except ImportError:  # pragma: no cover - environment dependent
    pd = None  # type: ignore[assignment]


__all__ = [
    "CACHE_FILE_SUFFIX",
    "DEFAULT_CACHE_DIR_ENV",
    "DEFAULT_CACHE_VERSION",
    "CacheCorruptionError",
    "CacheEntryMetadata",
    "CacheError",
    "CacheKeyError",
    "CacheOverwriteError",
    "CacheSerializerError",
    "CacheStatsSnapshot",
    "LocalCache",
    "cache_key",
    "clear",
    "default_cache",
    "delete",
    "exists",
    "get",
    "invalidate_by_producer",
    "invalidate_by_tags",
    "invalidate_namespace",
    "list_entries",
    "set",
    "stats",
]


CACHE_MAGIC = b"SIMCACH1"
CACHE_FORMAT_VERSION = 1
CACHE_FILE_SUFFIX = ".cache"
DEFAULT_CACHE_DIR_ENV = "SIMONS_CACHE_DIR"
DEFAULT_CACHE_VERSION = "v1"
DEFAULT_MAX_BYTES = 5 * 1024 * 1024 * 1024  # 5 GiB
_HEX_256_LEN = 64


class CacheError(RuntimeError):
    """Base class for cache-related failures."""


class CacheKeyError(CacheError, ValueError):
    """Raised when a cache key is malformed or semantically invalid."""


class CacheSerializerError(CacheError):
    """Raised when a serializer cannot encode/decode a value safely."""


class CacheOverwriteError(CacheError, FileExistsError):
    """Raised when a caller attempts to overwrite an existing key implicitly."""


class CacheCorruptionError(CacheError):
    """Raised when an on-disk cache entry is malformed or fails integrity checks."""


class UnsupportedSerializerError(CacheSerializerError):
    """Raised when a serializer name is unknown or unavailable."""


@dataclass(frozen=True, slots=True)
class CacheEntryMetadata:
    """Structured metadata persisted alongside each cache payload."""

    key: str
    namespace: str
    version: str
    fn_id: str
    created_at_utc: str
    expires_at_utc: str | None
    serializer: str
    payload_bytes: int
    checksum: str
    producer: str | None
    context_digest: str | None
    params_digest: str | None
    tags: tuple[str, ...] = ()
    user_metadata: Mapping[str, Any] = field(default_factory=dict)
    producer_runtime_ms: float | None = None
    file_format_version: int = CACHE_FORMAT_VERSION

    def is_expired(self, *, now_utc: datetime | None = None) -> bool:
        if self.expires_at_utc is None:
            return False
        now = now_utc or _utc_now()
        expiry = _parse_utc_timestamp(self.expires_at_utc)
        return now >= expiry

    def to_header(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "namespace": self.namespace,
            "version": self.version,
            "fn_id": self.fn_id,
            "created_at_utc": self.created_at_utc,
            "expires_at_utc": self.expires_at_utc,
            "serializer": self.serializer,
            "payload_bytes": self.payload_bytes,
            "checksum": self.checksum,
            "producer": self.producer,
            "context_digest": self.context_digest,
            "params_digest": self.params_digest,
            "tags": list(self.tags),
            "user_metadata": dict(self.user_metadata),
            "producer_runtime_ms": self.producer_runtime_ms,
            "file_format_version": self.file_format_version,
        }

    @classmethod
    def from_header(cls, data: Mapping[str, Any]) -> "CacheEntryMetadata":
        required = {
            "key",
            "namespace",
            "version",
            "fn_id",
            "created_at_utc",
            "serializer",
            "payload_bytes",
            "checksum",
        }
        missing = sorted(required - builtins.set(data))
        if missing:
            raise CacheCorruptionError(f"Cache header is missing required fields: {missing}")

        payload_bytes = int(data["payload_bytes"])
        if payload_bytes < 0:
            raise CacheCorruptionError("payload_bytes cannot be negative")

        tags_raw = data.get("tags") or []
        if not isinstance(tags_raw, (list, tuple)):
            raise CacheCorruptionError("tags must be a sequence when present")

        user_metadata = data.get("user_metadata") or {}
        if not isinstance(user_metadata, Mapping):
            raise CacheCorruptionError("user_metadata must be a mapping when present")

        return cls(
            key=str(data["key"]),
            namespace=str(data["namespace"]),
            version=str(data["version"]),
            fn_id=str(data["fn_id"]),
            created_at_utc=str(data["created_at_utc"]),
            expires_at_utc=(str(data["expires_at_utc"]) if data.get("expires_at_utc") is not None else None),
            serializer=str(data["serializer"]),
            payload_bytes=payload_bytes,
            checksum=str(data["checksum"]),
            producer=(str(data["producer"]) if data.get("producer") is not None else None),
            context_digest=(str(data["context_digest"]) if data.get("context_digest") is not None else None),
            params_digest=(str(data["params_digest"]) if data.get("params_digest") is not None else None),
            tags=tuple(str(tag) for tag in tags_raw),
            user_metadata=dict(user_metadata),
            producer_runtime_ms=(float(data["producer_runtime_ms"]) if data.get("producer_runtime_ms") is not None else None),
            file_format_version=int(data.get("file_format_version", CACHE_FORMAT_VERSION)),
        )


@dataclass(frozen=True, slots=True)
class CacheStatsSnapshot:
    """User-facing cache metrics snapshot."""

    root: str
    hits: int
    misses: int
    hit_rate: float
    expired_misses: int
    corruption_detected: int
    bytes_used: int
    n_entries: int
    eviction_count: int
    bytes_evicted: int
    mean_get_latency_ms: float
    mean_set_latency_ms: float
    recompute_time_saved_est_ms: float
    per_namespace_stats: Mapping[str, Mapping[str, float | int]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "expired_misses": self.expired_misses,
            "corruption_detected": self.corruption_detected,
            "bytes_used": self.bytes_used,
            "n_entries": self.n_entries,
            "eviction_count": self.eviction_count,
            "bytes_evicted": self.bytes_evicted,
            "mean_get_latency_ms": self.mean_get_latency_ms,
            "mean_set_latency_ms": self.mean_set_latency_ms,
            "recompute_time_saved_est_ms": self.recompute_time_saved_est_ms,
            "per_namespace_stats": {k: dict(v) for k, v in self.per_namespace_stats.items()},
        }


class Serializer(Protocol):
    name: str

    def dumps(self, value: Any) -> bytes:
        ...

    def loads(self, payload: bytes) -> Any:
        ...


class _JsonSerializer:
    name = "json"

    def dumps(self, value: Any) -> bytes:
        try:
            return _canonical_json_bytes(value)
        except Exception as exc:  # noqa: BLE001
            raise CacheSerializerError(
                "Value is not JSON-serializable under canonical encoding"
            ) from exc

    def loads(self, payload: bytes) -> Any:
        try:
            return json.loads(payload.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise CacheSerializerError("Failed to decode JSON payload") from exc


class _BytesSerializer:
    name = "bytes"

    def dumps(self, value: Any) -> bytes:
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise CacheSerializerError("bytes serializer requires bytes-like input")
        return bytes(value)

    def loads(self, payload: bytes) -> bytes:
        return payload


class _TextSerializer:
    name = "text"

    def dumps(self, value: Any) -> bytes:
        if not isinstance(value, str):
            raise CacheSerializerError("text serializer requires str input")
        return value.encode("utf-8")

    def loads(self, payload: bytes) -> str:
        try:
            return payload.decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            raise CacheSerializerError("Failed to decode UTF-8 text payload") from exc


class _PickleSerializer:
    name = "pickle"

    def dumps(self, value: Any) -> bytes:
        try:
            return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:  # noqa: BLE001
            raise CacheSerializerError("Pickle serialization failed") from exc

    def loads(self, payload: bytes) -> Any:
        try:
            return pickle.loads(payload)
        except Exception as exc:  # noqa: BLE001
            raise CacheSerializerError("Pickle deserialization failed") from exc


class _PandasPickleSerializer:
    name = "pandas_pickle"

    def dumps(self, value: Any) -> bytes:
        if pd is None:
            raise UnsupportedSerializerError("pandas is not installed")
        if not isinstance(value, (pd.DataFrame, pd.Series, pd.Index)):  # type: ignore[arg-type]
            raise CacheSerializerError("pandas_pickle serializer requires pandas objects")
        try:
            return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:  # noqa: BLE001
            raise CacheSerializerError("Failed to serialize pandas object") from exc

    def loads(self, payload: bytes) -> Any:
        if pd is None:
            raise UnsupportedSerializerError("pandas is not installed")
        try:
            value = pickle.loads(payload)
        except Exception as exc:  # noqa: BLE001
            raise CacheSerializerError("Failed to deserialize pandas object") from exc
        if not isinstance(value, (pd.DataFrame, pd.Series, pd.Index)):  # type: ignore[arg-type]
            raise CacheSerializerError("Decoded object is not a pandas object")
        return value


class _NumpyNpySerializer:
    name = "numpy_npy"

    def dumps(self, value: Any) -> bytes:
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise UnsupportedSerializerError("numpy is not installed") from exc

        if not isinstance(value, np.ndarray):
            raise CacheSerializerError("numpy_npy serializer requires numpy.ndarray")
        buffer = io.BytesIO()
        try:
            np.save(buffer, value, allow_pickle=False)
        except Exception as exc:  # noqa: BLE001
            raise CacheSerializerError("Failed to serialize numpy array") from exc
        return buffer.getvalue()

    def loads(self, payload: bytes) -> Any:
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise UnsupportedSerializerError("numpy is not installed") from exc

        try:
            return np.load(io.BytesIO(payload), allow_pickle=False)
        except Exception as exc:  # noqa: BLE001
            raise CacheSerializerError("Failed to deserialize numpy array") from exc


_SERIALIZERS: dict[str, Serializer] = {
    "json": _JsonSerializer(),
    "bytes": _BytesSerializer(),
    "text": _TextSerializer(),
    "pickle": _PickleSerializer(),
    "pandas_pickle": _PandasPickleSerializer(),
    "numpy_npy": _NumpyNpySerializer(),
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def _parse_utc_timestamp(value: str) -> datetime:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    ts = datetime.fromisoformat(text)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}

    if isinstance(value, bytearray):
        return {"__bytes__": bytes(value).hex()}

    if isinstance(value, memoryview):
        return {"__bytes__": bytes(value).hex()}

    if isinstance(value, (datetime, date)):
        return {"__datetime__": pd.Timestamp(value).isoformat() if pd is not None else str(value)}

    if isinstance(value, Path):
        return {"__path__": str(value)}

    if isinstance(value, Mapping):
        return {str(key): _canonicalize(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}

    if isinstance(value, builtins.set):
        return {"__set__": sorted(_canonicalize(item) for item in value)}

    if isinstance(value, tuple):
        return {"__tuple__": [_canonicalize(item) for item in value]}

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        return [_canonicalize(item) for item in value]

    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            data = value.to_dict()
        except Exception:  # noqa: BLE001
            pass
        else:
            return {"__to_dict__": _canonicalize(data)}

    raise TypeError(f"Value of type {type(value).__name__} is not canonically serializable")


def _canonical_json_bytes(value: Any) -> bytes:
    payload = _canonicalize(value)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _validate_cache_key(key: str) -> str:
    if not isinstance(key, str):
        raise CacheKeyError(f"Cache key must be a str, got {type(key).__name__}")
    if len(key) != _HEX_256_LEN:
        raise CacheKeyError(f"Cache key must have length {_HEX_256_LEN}, got {len(key)}")
    if any(ch not in "0123456789abcdef" for ch in key):
        raise CacheKeyError("Cache key must be a lowercase SHA-256 hexadecimal string")
    return key


def _normalize_tags(tags: Iterable[str] | None) -> tuple[str, ...]:
    if tags is None:
        return ()
    normalized = sorted({str(tag).strip() for tag in tags if str(tag).strip()})
    return tuple(normalized)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def cache_key(
    namespace: str,
    fn_id: str,
    params: Mapping[str, Any],
    version: str = DEFAULT_CACHE_VERSION,
    context: Mapping[str, Any] | None = None,
) -> str:
    """
    Build a deterministic cryptographic cache key.

    Notes
    -----
    The returned key is the SHA-256 digest of a canonical JSON serialization of
    the logical input payload. Operationally this makes accidental collisions
    negligible for the intended use of the cache.
    """
    if not str(namespace).strip():
        raise CacheKeyError("namespace must be a non-empty string")
    if not str(fn_id).strip():
        raise CacheKeyError("fn_id must be a non-empty string")

    payload = {
        "namespace": str(namespace),
        "version": str(version),
        "fn_id": str(fn_id),
        "params": dict(params),
        "context": dict(context or {}),
    }
    return _sha256_hex(_canonical_json_bytes(payload))


class LocalCache:
    """
    Deterministic local cache with atomic writes and byte-budgeted LRU eviction.

    Parameters
    ----------
    root:
        Root directory for the cache store.
    max_bytes:
        Byte budget enforced through LRU-style eviction. The implementation uses
        file modification times as the recency signal and guarantees that, after
        eviction completes, the total on-disk footprint does not persistently
        exceed ``max_bytes`` except for very brief write transients.
    validate_checksum:
        Whether reads verify the persisted payload checksum.
    quarantine_corrupt:
        Whether corrupt entries are moved aside instead of deleted immediately.
    durability_fsync_dir:
        Whether to fsync parent directories after atomic renames for stronger
        durability semantics on POSIX filesystems.
    """

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        validate_checksum: bool = True,
        quarantine_corrupt: bool = True,
        durability_fsync_dir: bool = True,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be strictly positive")

        root_path = Path(root or os.environ.get(DEFAULT_CACHE_DIR_ENV, ".cache/simons_core"))
        self.root = root_path.expanduser().resolve()
        self.max_bytes = int(max_bytes)
        self.validate_checksum = bool(validate_checksum)
        self.quarantine_corrupt = bool(quarantine_corrupt)
        self.durability_fsync_dir = bool(durability_fsync_dir)

        self.objects_dir = self.root / "objects"
        self.quarantine_dir = self.root / "quarantine"
        self.tmp_dir = self.root / "tmp"

        self.root.mkdir(parents=True, exist_ok=True)
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._expired_misses = 0
        self._corruption_detected = 0
        self._eviction_count = 0
        self._bytes_evicted = 0
        self._get_calls = 0
        self._set_calls = 0
        self._delete_calls = 0
        self._sum_get_latency_ms = 0.0
        self._sum_set_latency_ms = 0.0
        self._recompute_time_saved_est_ms = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Return the cached payload for *key* if valid, else *default*."""
        started = time.perf_counter()
        key = _validate_cache_key(key)
        path = self._path_for_key(key)

        try:
            if not path.exists():
                self._record_miss(reason="not_found")
                return default

            meta, payload = self._read_entry(path)

            if meta.is_expired():
                self._record_miss(reason="expired")
                self._invalidate_path(path)
                return default

            if self.validate_checksum:
                checksum = _sha256_hex(payload)
                if checksum != meta.checksum:
                    raise CacheCorruptionError(
                        f"Checksum mismatch for key={key}: {checksum} != {meta.checksum}"
                    )

            serializer = self._resolve_serializer(meta.serializer)
            value = serializer.loads(payload)
            self._touch_lru(path)
            self._record_hit(meta=meta)
            return value
        except CacheCorruptionError:
            self._record_corruption()
            self._handle_corrupt_entry(path)
            return default
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1_000.0
            with self._lock:
                self._get_calls += 1
                self._sum_get_latency_ms += elapsed_ms

    def set(
        self,
        key: str,
        value: Any,
        *,
        ttl_seconds: int | float | None = None,
        metadata: Mapping[str, Any] | None = None,
        serializer: str = "auto",
        overwrite: bool = False,
    ) -> None:
        """
        Persist a cache entry atomically.

        Notes
        -----
        - ``overwrite=False`` protects against silent mutation of an existing
          key under the supported single-writer operational model.
        - The payload is written to a private temp file, fsynced, verified and
          only then atomically renamed into visibility.
        """
        started = time.perf_counter()
        key = _validate_cache_key(key)
        path = self._path_for_key(key)
        tmp_path = self._tmp_path_for_key(key)
        metadata = dict(metadata or {})

        if ttl_seconds is not None and float(ttl_seconds) < 0:
            raise ValueError("ttl_seconds cannot be negative")

        serializer_impl = self._select_serializer(serializer, value)
        payload = serializer_impl.dumps(value)
        checksum = _sha256_hex(payload)
        meta = self._build_metadata(
            key=key,
            serializer=serializer_impl.name,
            payload=payload,
            checksum=checksum,
            ttl_seconds=ttl_seconds,
            metadata=metadata,
        )
        header_bytes = _canonical_json_bytes(meta.to_header())

        if not overwrite and path.exists():
            raise CacheOverwriteError(f"Cache entry already exists for key={key}")

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            self._write_atomic_file(tmp_path, header_bytes=header_bytes, payload=payload)

            try:
                written_meta, written_payload = self._read_entry(tmp_path)
            except Exception as exc:  # noqa: BLE001
                tmp_path.unlink(missing_ok=True)
                raise CacheCorruptionError("Post-write verification of temp entry failed") from exc

            if written_meta.checksum != checksum or _sha256_hex(written_payload) != checksum:
                tmp_path.unlink(missing_ok=True)
                raise CacheCorruptionError("Post-write checksum verification failed")

            if not overwrite and path.exists():
                tmp_path.unlink(missing_ok=True)
                raise CacheOverwriteError(f"Cache entry already exists for key={key}")

            os.replace(tmp_path, path)
            if self.durability_fsync_dir:
                _fsync_directory(path.parent)

            self._touch_lru(path)
            self._enforce_budget_locked(exclude=path)

        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        with self._lock:
            self._set_calls += 1
            self._sum_set_latency_ms += elapsed_ms

    def exists(self, key: str, *, validate_expiry: bool = True) -> bool:
        """Return True iff *key* exists and, optionally, is not logically expired."""
        key = _validate_cache_key(key)
        path = self._path_for_key(key)
        if not path.exists():
            return False
        if not validate_expiry:
            return True
        try:
            meta = self._read_metadata(path)
        except CacheCorruptionError:
            return False
        return not meta.is_expired()

    def delete(self, key: str) -> bool:
        """Delete a specific key if present. Returns True iff something was removed."""
        key = _validate_cache_key(key)
        path = self._path_for_key(key)
        removed = self._invalidate_path(path)
        with self._lock:
            self._delete_calls += int(removed)
        return removed

    def invalidate_namespace(self, namespace: str) -> int:
        """Invalidate all entries belonging to *namespace*."""
        target = str(namespace)
        removed = 0
        for path, meta in self._iter_entries(include_metadata=True):
            if meta.namespace == target:
                removed += int(self._invalidate_path(path))
        return removed

    def invalidate_by_tags(self, tags: Iterable[str]) -> int:
        """Invalidate all entries containing any of the supplied tags."""
        target_tags = builtins.set(_normalize_tags(tags))
        if not target_tags:
            return 0
        removed = 0
        for path, meta in self._iter_entries(include_metadata=True):
            if target_tags.intersection(meta.tags):
                removed += int(self._invalidate_path(path))
        return removed

    def invalidate_by_producer(self, producer: str) -> int:
        """Invalidate all entries whose producer field matches *producer*."""
        target = str(producer)
        removed = 0
        for path, meta in self._iter_entries(include_metadata=True):
            if meta.producer == target:
                removed += int(self._invalidate_path(path))
        return removed

    def clear(self, *, expired_only: bool = False) -> int:
        """Remove either all entries or only those that are logically expired."""
        removed = 0
        for path, meta in self._iter_entries(include_metadata=True):
            if expired_only and not meta.is_expired():
                continue
            removed += int(self._invalidate_path(path))
        return removed

    def list_entries(self, *, namespace: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        """Return lightweight metadata views for inspection/CLI use."""
        rows: list[dict[str, Any]] = []
        for path, meta in self._iter_entries(include_metadata=True):
            if namespace is not None and meta.namespace != namespace:
                continue
            rows.append(
                {
                    "key": meta.key,
                    "namespace": meta.namespace,
                    "version": meta.version,
                    "fn_id": meta.fn_id,
                    "serializer": meta.serializer,
                    "payload_bytes": meta.payload_bytes,
                    "created_at_utc": meta.created_at_utc,
                    "expires_at_utc": meta.expires_at_utc,
                    "producer": meta.producer,
                    "tags": list(meta.tags),
                    "path": str(path),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
        rows.sort(key=lambda row: (row["namespace"], row["created_at_utc"], row["key"]))
        return rows

    def stats(self) -> CacheStatsSnapshot:
        """Return an operational metrics snapshot for the cache store."""
        aggregate = self._scan_disk_usage()
        with self._lock:
            hits = self._hits
            misses = self._misses
            expired = self._expired_misses
            corruption = self._corruption_detected
            evictions = self._eviction_count
            bytes_evicted = self._bytes_evicted
            get_calls = self._get_calls
            set_calls = self._set_calls
            sum_get = self._sum_get_latency_ms
            sum_set = self._sum_set_latency_ms
            recompute_saved = self._recompute_time_saved_est_ms

        total_resolution_events = hits + misses
        hit_rate = (hits / total_resolution_events) if total_resolution_events else 0.0
        mean_get = (sum_get / get_calls) if get_calls else 0.0
        mean_set = (sum_set / set_calls) if set_calls else 0.0

        return CacheStatsSnapshot(
            root=str(self.root),
            hits=hits,
            misses=misses,
            hit_rate=hit_rate,
            expired_misses=expired,
            corruption_detected=corruption,
            bytes_used=aggregate["bytes_used"],
            n_entries=aggregate["n_entries"],
            eviction_count=evictions,
            bytes_evicted=bytes_evicted,
            mean_get_latency_ms=mean_get,
            mean_set_latency_ms=mean_set,
            recompute_time_saved_est_ms=recompute_saved,
            per_namespace_stats=aggregate["per_namespace_stats"],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _path_for_key(self, key: str) -> Path:
        return self.objects_dir / key[:2] / f"{key}{CACHE_FILE_SUFFIX}"

    def _tmp_path_for_key(self, key: str) -> Path:
        return self.tmp_dir / key[:2] / f"{key}.{uuid4().hex}.tmp"

    def _build_metadata(
        self,
        *,
        key: str,
        serializer: str,
        payload: bytes,
        checksum: str,
        ttl_seconds: int | float | None,
        metadata: Mapping[str, Any],
    ) -> CacheEntryMetadata:
        namespace = str(metadata.get("namespace", "default"))
        version = str(metadata.get("version", DEFAULT_CACHE_VERSION))
        fn_id = str(metadata.get("fn_id", "unknown"))
        producer = metadata.get("producer")
        created_at = _utc_now()
        expires_at = None
        if ttl_seconds is not None:
            expires_at = created_at + timedelta(seconds=float(ttl_seconds))

        user_metadata = {
            str(k): v
            for k, v in metadata.items()
            if k
            not in {
                "namespace",
                "version",
                "fn_id",
                "producer",
                "context_digest",
                "params_digest",
                "tags",
                "producer_runtime_ms",
            }
        }

        return CacheEntryMetadata(
            key=key,
            namespace=namespace,
            version=version,
            fn_id=fn_id,
            created_at_utc=_format_utc_timestamp(created_at),
            expires_at_utc=(_format_utc_timestamp(expires_at) if expires_at is not None else None),
            serializer=serializer,
            payload_bytes=len(payload),
            checksum=checksum,
            producer=(str(producer) if producer is not None else None),
            context_digest=(str(metadata["context_digest"]) if metadata.get("context_digest") is not None else None),
            params_digest=(str(metadata["params_digest"]) if metadata.get("params_digest") is not None else None),
            tags=_normalize_tags(metadata.get("tags")),
            user_metadata=user_metadata,
            producer_runtime_ms=(float(metadata["producer_runtime_ms"]) if metadata.get("producer_runtime_ms") is not None else None),
        )

    def _select_serializer(self, requested: str, value: Any) -> Serializer:
        if requested != "auto":
            return self._resolve_serializer(requested)

        if isinstance(value, (bytes, bytearray, memoryview)):
            return self._resolve_serializer("bytes")
        if isinstance(value, str):
            return self._resolve_serializer("text")
        if pd is not None and isinstance(value, (pd.DataFrame, pd.Series, pd.Index)):  # type: ignore[arg-type]
            return self._resolve_serializer("pandas_pickle")
        try:
            import numpy as np  # type: ignore
        except ImportError:  # pragma: no cover - environment dependent
            np = None  # type: ignore[assignment]
        if np is not None and isinstance(value, np.ndarray):  # type: ignore[attr-defined]
            return self._resolve_serializer("numpy_npy")

        try:
            _canonical_json_bytes(value)
        except Exception:  # noqa: BLE001
            raise CacheSerializerError(
                "auto serializer only supports canonical JSON values, bytes/text, "
                "pandas objects and numpy arrays. Use serializer='pickle' explicitly "
                "for arbitrary Python objects."
            )
        return self._resolve_serializer("json")

    def _resolve_serializer(self, name: str) -> Serializer:
        try:
            return _SERIALIZERS[name]
        except KeyError as exc:
            raise UnsupportedSerializerError(f"Unsupported serializer: {name!r}") from exc

    def _write_atomic_file(self, tmp_path: Path, *, header_bytes: bytes, payload: bytes) -> None:
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("wb") as handle:
            handle.write(CACHE_MAGIC)
            handle.write(struct.pack(">Q", len(header_bytes)))
            handle.write(header_bytes)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def _read_metadata(self, path: Path) -> CacheEntryMetadata:
        with path.open("rb") as handle:
            magic = handle.read(len(CACHE_MAGIC))
            if magic != CACHE_MAGIC:
                raise CacheCorruptionError(f"Invalid cache magic for {path}")

            header_len_blob = handle.read(8)
            if len(header_len_blob) != 8:
                raise CacheCorruptionError(f"Truncated header length for {path}")

            (header_len,) = struct.unpack(">Q", header_len_blob)
            if header_len <= 0:
                raise CacheCorruptionError(f"Non-positive header length for {path}")

            header_bytes = handle.read(header_len)
            if len(header_bytes) != header_len:
                raise CacheCorruptionError(f"Truncated header for {path}")

            try:
                header = json.loads(header_bytes.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                raise CacheCorruptionError(f"Invalid JSON header in {path}") from exc

        return CacheEntryMetadata.from_header(header)

    def _read_entry(self, path: Path) -> tuple[CacheEntryMetadata, bytes]:
        with path.open("rb") as handle:
            magic = handle.read(len(CACHE_MAGIC))
            if magic != CACHE_MAGIC:
                raise CacheCorruptionError(f"Invalid cache magic for {path}")

            header_len_blob = handle.read(8)
            if len(header_len_blob) != 8:
                raise CacheCorruptionError(f"Truncated header length for {path}")
            (header_len,) = struct.unpack(">Q", header_len_blob)
            if header_len <= 0:
                raise CacheCorruptionError(f"Non-positive header length for {path}")

            header_bytes = handle.read(header_len)
            if len(header_bytes) != header_len:
                raise CacheCorruptionError(f"Truncated header for {path}")

            try:
                header = json.loads(header_bytes.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                raise CacheCorruptionError(f"Invalid JSON header in {path}") from exc

            payload = handle.read()

        meta = CacheEntryMetadata.from_header(header)
        if len(payload) != meta.payload_bytes:
            raise CacheCorruptionError(
                f"Payload length mismatch for key={meta.key}: {len(payload)} != {meta.payload_bytes}"
            )
        return meta, payload

    def _touch_lru(self, path: Path) -> None:
        now = time.time()
        os.utime(path, (now, now), follow_symlinks=False)

    def _invalidate_path(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    def _handle_corrupt_entry(self, path: Path) -> None:
        if not path.exists():
            return
        if self.quarantine_corrupt:
            self.quarantine_dir.mkdir(parents=True, exist_ok=True)
            target = self.quarantine_dir / f"{path.stem}.{uuid4().hex}.corrupt"
            try:
                os.replace(path, target)
            except FileNotFoundError:
                return
        else:
            self._invalidate_path(path)

    def _record_hit(self, *, meta: CacheEntryMetadata) -> None:
        with self._lock:
            self._hits += 1
            if meta.producer_runtime_ms is not None:
                self._recompute_time_saved_est_ms += meta.producer_runtime_ms

    def _record_miss(self, *, reason: str) -> None:
        with self._lock:
            self._misses += 1
            if reason == "expired":
                self._expired_misses += 1

    def _record_corruption(self) -> None:
        with self._lock:
            self._corruption_detected += 1
            self._misses += 1

    def _iter_entry_paths(self) -> Iterator[Path]:
        if not self.objects_dir.exists():
            return
        for subdir in sorted(self.objects_dir.glob("*")):
            if not subdir.is_dir():
                continue
            for path in sorted(subdir.glob(f"*{CACHE_FILE_SUFFIX}")):
                if path.is_file():
                    yield path

    def _iter_entries(self, *, include_metadata: bool) -> Iterator[tuple[Path, CacheEntryMetadata]]:
        for path in self._iter_entry_paths():
            try:
                meta = self._read_metadata(path)
            except CacheCorruptionError:
                self._record_corruption()
                self._handle_corrupt_entry(path)
                continue
            yield path, meta

    def _scan_disk_usage(self) -> dict[str, Any]:
        bytes_used = 0
        n_entries = 0
        per_namespace: dict[str, dict[str, float | int]] = {}

        for path, meta in self._iter_entries(include_metadata=True):
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                continue
            bytes_used += size
            n_entries += 1
            bucket = per_namespace.setdefault(
                meta.namespace,
                {
                    "bytes_used": 0,
                    "n_entries": 0,
                    "expired_entries": 0,
                },
            )
            bucket["bytes_used"] = int(bucket["bytes_used"]) + size
            bucket["n_entries"] = int(bucket["n_entries"]) + 1
            if meta.is_expired():
                bucket["expired_entries"] = int(bucket["expired_entries"]) + 1

        return {
            "bytes_used": bytes_used,
            "n_entries": n_entries,
            "per_namespace_stats": per_namespace,
        }

    def _enforce_budget_locked(self, *, exclude: Path | None = None) -> None:
        entries: list[tuple[float, Path, int]] = []
        total_bytes = 0

        for path in self._iter_entry_paths():
            if exclude is not None and path == exclude:
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            total_bytes += stat.st_size
            entries.append((stat.st_mtime, path, stat.st_size))

        if exclude is not None and exclude.exists():
            try:
                total_bytes += exclude.stat().st_size
            except FileNotFoundError:
                pass

        if total_bytes <= self.max_bytes:
            return

        entries.sort(key=lambda item: item[0])
        for _, path, size in entries:
            if total_bytes <= self.max_bytes:
                break
            if not self._invalidate_path(path):
                continue
            total_bytes -= size
            self._eviction_count += 1
            self._bytes_evicted += size

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def get_or_compute(
        self,
        *,
        key: str,
        producer: Callable[[], Any],
        ttl_seconds: int | float | None = None,
        metadata: Mapping[str, Any] | None = None,
        serializer: str = "auto",
        overwrite: bool = False,
    ) -> Any:
        """
        Convenience helper: return cached value or compute/store it.

        This helper is intentionally thin and does not attempt stampede control
        beyond the documented single-writer operating model.
        """
        cached = self.get(key, default=None)
        if cached is not None:
            return cached

        started = time.perf_counter()
        value = producer()
        runtime_ms = (time.perf_counter() - started) * 1_000.0
        merged_metadata = dict(metadata or {})
        merged_metadata.setdefault("producer_runtime_ms", runtime_ms)
        self.set(
            key,
            value,
            ttl_seconds=ttl_seconds,
            metadata=merged_metadata,
            serializer=serializer,
            overwrite=overwrite,
        )
        return value


_default_cache: LocalCache | None = None
_default_cache_lock = threading.Lock()


def default_cache() -> LocalCache:
    global _default_cache
    with _default_cache_lock:
        if _default_cache is None:
            _default_cache = LocalCache()
        return _default_cache


def get(key: str, default: Any = None) -> Any:
    return default_cache().get(key, default=default)


def set(
    key: str,
    value: Any,
    ttl_seconds: int | float | None = None,
    metadata: Mapping[str, Any] | None = None,
    serializer: str = "auto",
    overwrite: bool = False,
) -> None:
    default_cache().set(
        key,
        value,
        ttl_seconds=ttl_seconds,
        metadata=metadata,
        serializer=serializer,
        overwrite=overwrite,
    )


def exists(key: str, validate_expiry: bool = True) -> bool:
    return default_cache().exists(key, validate_expiry=validate_expiry)


def delete(key: str) -> bool:
    return default_cache().delete(key)


def invalidate_namespace(namespace: str) -> int:
    return default_cache().invalidate_namespace(namespace)


def invalidate_by_tags(tags: Iterable[str]) -> int:
    return default_cache().invalidate_by_tags(tags)


def invalidate_by_producer(producer: str) -> int:
    return default_cache().invalidate_by_producer(producer)


def clear(expired_only: bool = False) -> int:
    return default_cache().clear(expired_only=expired_only)


def list_entries(*, namespace: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    return default_cache().list_entries(namespace=namespace, limit=limit)


def stats() -> dict[str, Any]:
    return default_cache().stats().as_dict()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _build_cli() -> ArgumentParser:
    parser = ArgumentParser(prog="python -m simons_core.io.cache")
    parser.add_argument("--root", default=None, help="Override cache root directory")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_stats = subparsers.add_parser("stats", help="Show cache metrics")
    p_stats.add_argument("--json", action="store_true", help="Emit full JSON")

    p_ls = subparsers.add_parser("ls", help="List cache entries")
    p_ls.add_argument("--namespace", default=None)
    p_ls.add_argument("--limit", type=int, default=None)
    p_ls.add_argument("--json", action="store_true")

    p_invalidate = subparsers.add_parser("invalidate", help="Invalidate a subset of the cache")
    p_invalidate.add_argument("--namespace", default=None)
    p_invalidate.add_argument("--producer", default=None)
    p_invalidate.add_argument("--tags", nargs="*", default=None)

    p_clear = subparsers.add_parser("clear", help="Clear all or only expired entries")
    p_clear.add_argument("--expired-only", action="store_true")

    p_delete = subparsers.add_parser("delete", help="Delete a specific cache key")
    p_delete.add_argument("key")

    return parser


def _cache_from_cli(args: Namespace) -> LocalCache:
    return LocalCache(root=args.root, max_bytes=args.max_bytes)


def _cli_stats(cache: LocalCache, args: Namespace) -> int:
    snapshot = cache.stats().as_dict()
    if args.json:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
        return 0

    print(f"root: {snapshot['root']}")
    print(f"hits: {snapshot['hits']}")
    print(f"misses: {snapshot['misses']}")
    print(f"hit_rate: {snapshot['hit_rate']:.4f}")
    print(f"expired_misses: {snapshot['expired_misses']}")
    print(f"corruption_detected: {snapshot['corruption_detected']}")
    print(f"bytes_used: {snapshot['bytes_used']}")
    print(f"n_entries: {snapshot['n_entries']}")
    print(f"eviction_count: {snapshot['eviction_count']}")
    print(f"bytes_evicted: {snapshot['bytes_evicted']}")
    print(f"mean_get_latency_ms: {snapshot['mean_get_latency_ms']:.3f}")
    print(f"mean_set_latency_ms: {snapshot['mean_set_latency_ms']:.3f}")
    print(f"recompute_time_saved_est_ms: {snapshot['recompute_time_saved_est_ms']:.3f}")
    if snapshot["per_namespace_stats"]:
        print("per_namespace_stats:")
        for ns, values in sorted(snapshot["per_namespace_stats"].items()):
            print(f"  - {ns}: {values}")
    return 0


def _cli_ls(cache: LocalCache, args: Namespace) -> int:
    rows = cache.list_entries(namespace=args.namespace, limit=args.limit)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    for row in rows:
        print(
            f"{row['namespace']:>16}  {row['payload_bytes']:>10} B  "
            f"{row['created_at_utc']}  {row['key']}  {row['fn_id']}"
        )
    return 0


def _cli_invalidate(cache: LocalCache, args: Namespace) -> int:
    removed = 0
    if args.namespace:
        removed += cache.invalidate_namespace(args.namespace)
    if args.producer:
        removed += cache.invalidate_by_producer(args.producer)
    if args.tags:
        removed += cache.invalidate_by_tags(args.tags)
    print(removed)
    return 0


def _cli_clear(cache: LocalCache, args: Namespace) -> int:
    removed = cache.clear(expired_only=args.expired_only)
    print(removed)
    return 0


def _cli_delete(cache: LocalCache, args: Namespace) -> int:
    print(int(cache.delete(args.key)))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)
    cache = _cache_from_cli(args)

    if args.command == "stats":
        return _cli_stats(cache, args)
    if args.command == "ls":
        return _cli_ls(cache, args)
    if args.command == "invalidate":
        return _cli_invalidate(cache, args)
    if args.command == "clear":
        return _cli_clear(cache, args)
    if args.command == "delete":
        return _cli_delete(cache, args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
