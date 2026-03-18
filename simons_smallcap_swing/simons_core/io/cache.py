"""
simons_core/io/cache.py — Atomic local cache with deterministic keys.

Core invariant:
    get(k) ≡ f(θ_k)  (valid cache hit = recomputation)

Key construction: SHA-256 over canonical JSON of (namespace, fn_id, params, version, context).
Write protocol: serialize → temp file → fsync → rename (atomic).
Eviction: byte-LRU with configurable budget B_max.
Corruption: detected via checksum, never served.

Serializers: json, bytes, text, pickle, pandas_pickle, numpy_npy.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import pickle
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class CacheError(RuntimeError):
    pass

class CacheKeyError(CacheError, ValueError):
    pass

class CacheCorruptionError(CacheError):
    pass


# ---------------------------------------------------------------------------
# Metadata per entry (spec §6)
# ---------------------------------------------------------------------------

@dataclass
class CacheEntryMeta:
    key: str
    namespace: str
    version: str
    fn_id: str
    created_at_utc: str
    expires_at_utc: str | None
    serializer: str
    payload_bytes: int
    checksum: str
    producer: str | None = None
    context_digest: str | None = None
    tags: list[str] = field(default_factory=list)

    def is_expired(self) -> bool:
        if self.expires_at_utc is None:
            return False
        now = datetime.now(timezone.utc)
        exp = datetime.fromisoformat(self.expires_at_utc)
        return now >= exp

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "namespace": self.namespace,
            "version": self.version, "fn_id": self.fn_id,
            "created_at_utc": self.created_at_utc,
            "expires_at_utc": self.expires_at_utc,
            "serializer": self.serializer,
            "payload_bytes": self.payload_bytes,
            "checksum": self.checksum,
            "producer": self.producer,
            "context_digest": self.context_digest,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CacheEntryMeta:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Serializers (spec §8)
# ---------------------------------------------------------------------------

def _serialize_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()

def _deserialize_json(payload: bytes) -> Any:
    return json.loads(payload)

def _serialize_pickle(value: Any) -> bytes:
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

def _deserialize_pickle(payload: bytes) -> Any:
    return pickle.loads(payload)

def _serialize_pandas(value: Any) -> bytes:
    buf = io.BytesIO()
    try:
        value.to_parquet(buf, index=False)
    except ImportError:
        pickle.dump(value, buf)
    return buf.getvalue()

def _deserialize_pandas(payload: bytes) -> pd.DataFrame:
    try:
        return pd.read_parquet(io.BytesIO(payload))
    except (ImportError, Exception):
        return pickle.loads(payload)

def _serialize_numpy(value: Any) -> bytes:
    buf = io.BytesIO()
    np.save(buf, value, allow_pickle=False)
    return buf.getvalue()

def _deserialize_numpy(payload: bytes) -> np.ndarray:
    return np.load(io.BytesIO(payload), allow_pickle=False)

SERIALIZERS = {
    "json": (_serialize_json, _deserialize_json),
    "pickle": (_serialize_pickle, _deserialize_pickle),
    "pandas": (_serialize_pandas, _deserialize_pandas),
    "numpy": (_serialize_numpy, _deserialize_numpy),
}


def _auto_serializer(value: Any) -> str:
    if isinstance(value, pd.DataFrame):
        return "pandas"
    if isinstance(value, np.ndarray):
        return "numpy"
    if isinstance(value, (dict, list, str, int, float, bool)):
        return "json"
    return "pickle"


# ---------------------------------------------------------------------------
# Cache key (spec §4) — SHA-256 over canonical JSON
# ---------------------------------------------------------------------------

def cache_key(
    namespace: str,
    fn_id: str,
    params: Mapping[str, Any],
    version: str = "v1",
    context: Mapping[str, Any] | None = None,
) -> str:
    """Build deterministic cache key via SHA-256."""
    payload = {
        "namespace": namespace, "version": version,
        "fn_id": fn_id, "params": dict(params),
        "context": dict(context) if context else {},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Byte-LRU eviction (spec §12)
# ---------------------------------------------------------------------------

class ByteLRU:
    """LRU eviction tracker by byte budget."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.entries: OrderedDict[str, int] = OrderedDict()
        self.total_bytes = 0

    def touch(self, key: str, size: int) -> list[str]:
        """Touch a key, return list of evicted keys."""
        evicted = []
        if key in self.entries:
            self.total_bytes -= self.entries.pop(key)
        self.entries[key] = size
        self.entries.move_to_end(key)
        self.total_bytes += size

        while self.total_bytes > self.max_bytes and self.entries:
            old_key, old_size = self.entries.popitem(last=False)
            self.total_bytes -= old_size
            evicted.append(old_key)

        return evicted

    def remove(self, key: str) -> None:
        if key in self.entries:
            self.total_bytes -= self.entries.pop(key)


# ---------------------------------------------------------------------------
# Cache store (spec §9, §10, §12)
# ---------------------------------------------------------------------------

class LocalCache:
    """Atomic local cache with byte-LRU eviction.

    Write protocol (spec §9):
        1. Serialize to temp file
        2. fsync temp file
        3. Verify checksum
        4. Atomic rename to final path
        5. Update LRU
    """

    def __init__(self, root: str | Path, max_bytes: int = 10 * 1024**3):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lru = ByteLRU(max_bytes)
        self._stats = {"hits": 0, "misses": 0, "expired": 0,
                       "corruptions": 0, "evictions": 0, "sets": 0}

    def _key_dir(self, key: str) -> Path:
        return self.root / key[:2] / key[2:4] / key

    def _payload_path(self, key: str) -> Path:
        return self._key_dir(key) / "payload.bin"

    def _meta_path(self, key: str) -> Path:
        return self._key_dir(key) / "meta.json"

    def set(
        self,
        key: str,
        value: Any,
        *,
        namespace: str = "default",
        fn_id: str = "",
        version: str = "v1",
        ttl_seconds: int | None = None,
        serializer: str = "auto",
        overwrite: bool = False,
        tags: Sequence[str] = (),
    ) -> None:
        """Write entry atomically: temp → fsync → rename."""
        final_dir = self._key_dir(key)
        if not overwrite and (final_dir / "meta.json").exists():
            return  # Already cached

        # Serialize
        ser_name = serializer if serializer != "auto" else _auto_serializer(value)
        if ser_name not in SERIALIZERS:
            raise CacheError(f"Unknown serializer: {ser_name}")
        ser_fn, _ = SERIALIZERS[ser_name]
        payload = ser_fn(value)
        checksum = _sha256_bytes(payload)

        # Metadata
        expires = None
        if ttl_seconds is not None:
            from datetime import timedelta
            exp_dt = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
            expires = exp_dt.replace(microsecond=0).isoformat()

        meta = CacheEntryMeta(
            key=key, namespace=namespace, version=version, fn_id=fn_id,
            created_at_utc=_utc_now_iso(), expires_at_utc=expires,
            serializer=ser_name, payload_bytes=len(payload),
            checksum=checksum, tags=list(tags),
        )

        # Atomic write: temp → fsync → rename
        final_dir.mkdir(parents=True, exist_ok=True)
        tmp_payload = final_dir / f"_tmp_payload_{os.getpid()}"
        tmp_meta = final_dir / f"_tmp_meta_{os.getpid()}"

        try:
            # Write payload
            with open(tmp_payload, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())

            # Write metadata
            meta_bytes = json.dumps(meta.to_dict(), indent=2, sort_keys=True).encode()
            with open(tmp_meta, "wb") as f:
                f.write(meta_bytes)
                f.flush()
                os.fsync(f.fileno())

            # Verify checksum
            if _sha256_bytes(tmp_payload.read_bytes()) != checksum:
                raise CacheCorruptionError("Checksum mismatch after write")

            # Atomic rename
            tmp_payload.rename(self._payload_path(key))
            tmp_meta.rename(self._meta_path(key))

        except Exception:
            tmp_payload.unlink(missing_ok=True)
            tmp_meta.unlink(missing_ok=True)
            raise

        # Update LRU
        evicted = self.lru.touch(key, len(payload))
        for ek in evicted:
            self._delete_files(ek)
            self._stats["evictions"] += 1

        self._stats["sets"] += 1

    def get(self, key: str, default: Any = None) -> Any:
        """Read entry with expiry and corruption checks."""
        meta_path = self._meta_path(key)
        payload_path = self._payload_path(key)

        if not meta_path.exists() or not payload_path.exists():
            self._stats["misses"] += 1
            return default

        # Load metadata
        try:
            meta = CacheEntryMeta.from_dict(json.loads(meta_path.read_text()))
        except Exception:
            self._stats["corruptions"] += 1
            self._delete_files(key)
            return default

        # Check expiry
        if meta.is_expired():
            self._stats["expired"] += 1
            self._delete_files(key)
            return default

        # Read and verify payload
        payload = payload_path.read_bytes()
        if _sha256_bytes(payload) != meta.checksum:
            self._stats["corruptions"] += 1
            self._delete_files(key)
            return default

        # Deserialize
        if meta.serializer not in SERIALIZERS:
            self._stats["corruptions"] += 1
            return default

        _, deser_fn = SERIALIZERS[meta.serializer]
        value = deser_fn(payload)

        self.lru.touch(key, len(payload))
        self._stats["hits"] += 1
        return value

    def exists(self, key: str, *, validate_expiry: bool = True) -> bool:
        meta_path = self._meta_path(key)
        if not meta_path.exists():
            return False
        if validate_expiry:
            try:
                meta = CacheEntryMeta.from_dict(json.loads(meta_path.read_text()))
                return not meta.is_expired()
            except Exception:
                return False
        return True

    def delete(self, key: str) -> bool:
        existed = self._meta_path(key).exists()
        self._delete_files(key)
        self.lru.remove(key)
        return existed

    def invalidate_namespace(self, namespace: str) -> int:
        """Invalidate all entries in a namespace."""
        count = 0
        for meta_path in self.root.rglob("meta.json"):
            try:
                meta = json.loads(meta_path.read_text())
                if meta.get("namespace") == namespace:
                    key = meta["key"]
                    self._delete_files(key)
                    self.lru.remove(key)
                    count += 1
            except Exception:
                pass
        return count

    def clear(self, *, expired_only: bool = False) -> int:
        """Clear cache entries."""
        count = 0
        for meta_path in self.root.rglob("meta.json"):
            try:
                meta = CacheEntryMeta.from_dict(json.loads(meta_path.read_text()))
                if expired_only and not meta.is_expired():
                    continue
                self._delete_files(meta.key)
                self.lru.remove(meta.key)
                count += 1
            except Exception:
                pass
        return count

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            **self._stats,
            "hit_rate": self._stats["hits"] / max(self._stats["hits"] + self._stats["misses"], 1),
            "bytes_used": self.lru.total_bytes,
            "n_entries": len(self.lru.entries),
            "max_bytes": self.lru.max_bytes,
        }

    def _delete_files(self, key: str) -> None:
        d = self._key_dir(key)
        if d.exists():
            for f in d.iterdir():
                f.unlink(missing_ok=True)
            try:
                d.rmdir()
            except OSError:
                pass
