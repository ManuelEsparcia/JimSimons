from __future__ import annotations

import json
from pathlib import Path
import struct
import time

import numpy as np
import pandas as pd
import pytest


pytestmark = [pytest.mark.io, pytest.mark.filesystem]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry_paths(cache) -> list[Path]:
    return sorted(cache.objects_dir.rglob(f"*{cache.CACHE_FILE_SUFFIX}")) if hasattr(cache, "CACHE_FILE_SUFFIX") else sorted(cache.objects_dir.rglob("*.cache"))


def _single_entry_path(cache) -> Path:
    paths = sorted(cache.objects_dir.rglob("*.cache"))
    assert len(paths) == 1, f"Expected exactly one cache entry, found {len(paths)}"
    return paths[0]


def _read_raw_entry(path: Path, cache_mod):
    with path.open("rb") as handle:
        magic = handle.read(len(cache_mod.CACHE_MAGIC))
        assert magic == cache_mod.CACHE_MAGIC
        header_len_blob = handle.read(8)
        (header_len,) = struct.unpack(">Q", header_len_blob)
        header_bytes = handle.read(header_len)
        payload = handle.read()
    return json.loads(header_bytes.decode("utf-8")), payload


# ---------------------------------------------------------------------------
# cache_key tests
# ---------------------------------------------------------------------------


def test_cache_key_is_deterministic_under_reordered_mappings(cache_mod):
    key1 = cache_mod.cache_key(
        namespace="features",
        fn_id="build_features.daily",
        params={"window": 20, "groups": ["sector", "exchange"], "flags": {"winsor": True, "neutralize": False}},
        context={"asof": "2024-01-03", "universe": "smallcap"},
    )
    key2 = cache_mod.cache_key(
        namespace="features",
        fn_id="build_features.daily",
        params={"flags": {"neutralize": False, "winsor": True}, "groups": ["sector", "exchange"], "window": 20},
        context={"universe": "smallcap", "asof": "2024-01-03"},
    )
    assert key1 == key2
    assert len(key1) == 64


@pytest.mark.parametrize(
    ("kwargs", "expected_exc"),
    [
        ({"namespace": "", "fn_id": "f", "params": {}}, Exception),
        ({"namespace": "x", "fn_id": "", "params": {}}, Exception),
    ],
)
def test_cache_key_rejects_empty_namespace_or_fnid(cache_mod, kwargs, expected_exc):
    with pytest.raises(expected_exc):
        cache_mod.cache_key(**kwargs)


# ---------------------------------------------------------------------------
# LocalCache construction and basic roundtrips
# ---------------------------------------------------------------------------


def test_local_cache_initialization_creates_expected_directories(make_local_cache):
    cache = make_local_cache("init_case")
    assert cache.root.exists()
    assert cache.objects_dir.exists()
    assert cache.tmp_dir.exists()
    assert cache.quarantine_dir.exists()


@pytest.mark.parametrize(
    ("value", "serializer"),
    [
        ({"a": 1, "b": [1, 2, 3]}, "auto"),
        (b"alpha-binary", "auto"),
        ("hello cache", "auto"),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), "auto"),
        (pd.DataFrame({"x": [1, 2], "y": [3.5, 4.5]}), "auto"),
        ({"nested": [1, 2, 3]}, "json"),
        ({"opaque": {1, 2, 3}, "path": Path("/tmp/example")}, "pickle"),
    ],
)
def test_set_get_roundtrip_for_supported_payloads(local_cache, cache_mod, value, serializer):
    key = cache_mod.cache_key("features", "roundtrip", {"serializer": serializer, "kind": type(value).__name__})
    local_cache.set(
        key,
        value,
        serializer=serializer,
        metadata={
            "namespace": "features",
            "fn_id": "roundtrip",
            "producer": "unit.test",
            "tags": ["smoke", "roundtrip"],
        },
    )
    observed = local_cache.get(key)

    if isinstance(value, np.ndarray):
        np.testing.assert_allclose(observed, value)
    elif isinstance(value, pd.DataFrame):
        pd.testing.assert_frame_equal(observed, value)
    else:
        assert observed == value



def test_set_auto_rejects_arbitrary_python_object_without_pickle(local_cache, cache_mod):
    key = cache_mod.cache_key("features", "serializer.auto.reject", {"case": 1})
    with pytest.raises(cache_mod.CacheSerializerError):
        local_cache.set(key, {"opaque": object()}, serializer="auto")



def test_set_rejects_unsupported_serializer(local_cache, cache_mod):
    key = cache_mod.cache_key("features", "serializer.unsupported", {"case": 1})
    with pytest.raises(cache_mod.CacheSerializerError):
        local_cache.set(key, {"a": 1}, serializer="definitely_not_real")



def test_overwrite_protection_and_explicit_overwrite(local_cache, cache_mod):
    key = cache_mod.cache_key("features", "overwrite", {"case": 1})
    local_cache.set(key, {"value": 1}, metadata={"namespace": "features", "fn_id": "overwrite"})

    with pytest.raises(cache_mod.CacheOverwriteError):
        local_cache.set(key, {"value": 2}, metadata={"namespace": "features", "fn_id": "overwrite"})

    local_cache.set(
        key,
        {"value": 2},
        overwrite=True,
        metadata={"namespace": "features", "fn_id": "overwrite"},
    )
    assert local_cache.get(key) == {"value": 2}



def test_exists_delete_and_missing_default(local_cache, cache_mod):
    missing_key = cache_mod.cache_key("features", "missing", {"case": 0})
    assert local_cache.get(missing_key, default="MISS") == "MISS"
    assert local_cache.exists(missing_key) is False
    assert local_cache.delete(missing_key) is False

    key = cache_mod.cache_key("features", "exists_delete", {"case": 1})
    local_cache.set(key, {"ok": True}, metadata={"namespace": "features", "fn_id": "exists_delete"})
    assert local_cache.exists(key) is True
    assert local_cache.delete(key) is True
    assert local_cache.exists(key) is False


@pytest.mark.parametrize("bad_key", [123, "abc", "A" * 64, "g" * 64])
def test_public_methods_reject_malformed_cache_keys(local_cache, bad_key):
    with pytest.raises(Exception):
        local_cache.get(bad_key)
    with pytest.raises(Exception):
        local_cache.exists(bad_key)
    with pytest.raises(Exception):
        local_cache.delete(bad_key)


# ---------------------------------------------------------------------------
# TTL / expiry semantics
# ---------------------------------------------------------------------------


def test_ttl_zero_behaves_as_immediately_expired(local_cache, cache_mod):
    key = cache_mod.cache_key("features", "ttl.zero", {"case": 1})
    local_cache.set(
        key,
        {"x": 1},
        ttl_seconds=0,
        metadata={"namespace": "features", "fn_id": "ttl.zero"},
    )
    assert local_cache.exists(key, validate_expiry=False) is True
    assert local_cache.exists(key, validate_expiry=True) is False
    assert local_cache.get(key, default="EXPIRED") == "EXPIRED"
    assert local_cache.exists(key, validate_expiry=False) is False



def test_clear_expired_only_removes_only_expired_entries(local_cache, cache_mod):
    expired_key = cache_mod.cache_key("features", "ttl.expired", {"case": 1})
    live_key = cache_mod.cache_key("features", "ttl.live", {"case": 2})

    local_cache.set(expired_key, 1, ttl_seconds=0, metadata={"namespace": "features", "fn_id": "ttl.expired"})
    local_cache.set(live_key, 2, ttl_seconds=60, metadata={"namespace": "features", "fn_id": "ttl.live"})

    removed = local_cache.clear(expired_only=True)
    assert removed == 1
    assert local_cache.exists(expired_key, validate_expiry=False) is False
    assert local_cache.exists(live_key) is True



def test_negative_ttl_is_rejected(local_cache, cache_mod):
    key = cache_mod.cache_key("features", "ttl.negative", {"case": 1})
    with pytest.raises(ValueError):
        local_cache.set(key, {"x": 1}, ttl_seconds=-1, metadata={"namespace": "features", "fn_id": "ttl.negative"})


# ---------------------------------------------------------------------------
# Listing, invalidation and metadata semantics
# ---------------------------------------------------------------------------


def test_list_entries_exposes_lightweight_metadata_and_namespace_filter(local_cache, cache_mod):
    keys = [
        cache_mod.cache_key("features", "f1", {"i": 1}),
        cache_mod.cache_key("features", "f2", {"i": 2}),
        cache_mod.cache_key("labels", "l1", {"i": 3}),
    ]
    local_cache.set(keys[0], 1, metadata={"namespace": "features", "fn_id": "f1", "tags": ["a"]})
    local_cache.set(keys[1], 2, metadata={"namespace": "features", "fn_id": "f2", "tags": ["b"]})
    local_cache.set(keys[2], 3, metadata={"namespace": "labels", "fn_id": "l1", "tags": ["c"]})

    rows = local_cache.list_entries()
    assert len(rows) == 3
    assert set(rows[0]) >= {"key", "namespace", "version", "fn_id", "serializer", "payload_bytes", "created_at_utc", "path", "tags"}

    feature_rows = local_cache.list_entries(namespace="features")
    assert len(feature_rows) == 2
    assert {row["namespace"] for row in feature_rows} == {"features"}

    limited = local_cache.list_entries(limit=2)
    assert len(limited) == 2



def test_invalidate_namespace_tags_and_producer(local_cache, cache_mod):
    k_feat_a = cache_mod.cache_key("features", "a", {"i": 1})
    k_feat_b = cache_mod.cache_key("features", "b", {"i": 2})
    k_lbl_c = cache_mod.cache_key("labels", "c", {"i": 3})

    local_cache.set(k_feat_a, 1, metadata={"namespace": "features", "fn_id": "a", "tags": ["common", "alpha"], "producer": "research.a"})
    local_cache.set(k_feat_b, 2, metadata={"namespace": "features", "fn_id": "b", "tags": ["beta"], "producer": "research.b"})
    local_cache.set(k_lbl_c, 3, metadata={"namespace": "labels", "fn_id": "c", "tags": ["common", "gamma"], "producer": "research.a"})

    removed = local_cache.invalidate_by_tags(["common"])
    assert removed == 2
    assert local_cache.exists(k_feat_a) is False
    assert local_cache.exists(k_lbl_c) is False
    assert local_cache.exists(k_feat_b) is True

    removed = local_cache.invalidate_by_producer("research.b")
    assert removed == 1
    assert local_cache.exists(k_feat_b) is False

    # Recreate and test namespace-wide invalidation.
    local_cache.set(k_feat_a, 1, metadata={"namespace": "features", "fn_id": "a"})
    local_cache.set(k_feat_b, 2, metadata={"namespace": "features", "fn_id": "b"})
    local_cache.set(k_lbl_c, 3, metadata={"namespace": "labels", "fn_id": "c"})
    removed = local_cache.invalidate_namespace("features")
    assert removed == 2
    assert local_cache.exists(k_feat_a) is False
    assert local_cache.exists(k_feat_b) is False
    assert local_cache.exists(k_lbl_c) is True



def test_clear_all_removes_everything(local_cache, cache_mod):
    for i in range(3):
        key = cache_mod.cache_key("features", "clear_all", {"i": i})
        local_cache.set(key, i, metadata={"namespace": "features", "fn_id": "clear_all"})
    removed = local_cache.clear()
    assert removed == 3
    assert local_cache.list_entries() == []


# ---------------------------------------------------------------------------
# Corruption handling / quarantine
# ---------------------------------------------------------------------------


def test_corrupt_magic_returns_default_and_moves_entry_to_quarantine(make_local_cache, cache_mod):
    cache = make_local_cache("corrupt_quarantine", quarantine_corrupt=True)
    key = cache_mod.cache_key("features", "corrupt.magic", {"case": 1})
    cache.set(key, {"ok": True}, metadata={"namespace": "features", "fn_id": "corrupt.magic"})

    path = _single_entry_path(cache)
    raw = path.read_bytes()
    path.write_bytes(b"BROKEN!!" + raw[8:])

    assert cache.get(key, default="CORRUPT") == "CORRUPT"
    assert cache.stats().corruption_detected == 1
    assert cache.exists(key, validate_expiry=False) is False
    quarantined = list(cache.quarantine_dir.glob("*.corrupt"))
    assert len(quarantined) == 1



def test_corrupt_entry_can_be_deleted_instead_of_quarantined(make_local_cache, cache_mod):
    cache = make_local_cache("corrupt_delete", quarantine_corrupt=False)
    key = cache_mod.cache_key("features", "corrupt.delete", {"case": 1})
    cache.set(key, {"ok": True}, metadata={"namespace": "features", "fn_id": "corrupt.delete"})

    path = _single_entry_path(cache)
    path.write_bytes(path.read_bytes()[:5])

    assert cache.get(key, default=None) is None
    assert cache.stats().corruption_detected == 1
    assert list(cache.quarantine_dir.glob("*.corrupt")) == []
    assert path.exists() is False


# ---------------------------------------------------------------------------
# Checksum / metadata internals that are observable from disk
# ---------------------------------------------------------------------------


def test_set_persists_normalized_tags_and_user_metadata(local_cache, cache_mod):
    key = cache_mod.cache_key("features", "metadata", {"case": 1})
    local_cache.set(
        key,
        {"x": 1},
        metadata={
            "namespace": "features",
            "fn_id": "metadata",
            "tags": [" beta ", "alpha", "alpha", ""],
            "producer": "unit.test",
            "freeform": 123,
            "other": {"nested": True},
        },
    )
    path = _single_entry_path(local_cache)
    header, _ = _read_raw_entry(path, cache_mod)
    assert header["tags"] == ["alpha", "beta"]
    assert header["user_metadata"] == {"freeform": 123, "other": {"nested": True}}
    assert header["namespace"] == "features"
    assert header["fn_id"] == "metadata"



def test_read_still_succeeds_when_checksum_validation_disabled(make_local_cache, cache_mod):
    cache = make_local_cache("checksum_disabled", validate_checksum=False)
    key = cache_mod.cache_key("features", "checksum.disabled", {"case": 1})
    value = "plain-text-value"
    cache.set(key, value, serializer="text", metadata={"namespace": "features", "fn_id": "checksum.disabled"})

    path = _single_entry_path(cache)
    header, payload = _read_raw_entry(path, cache_mod)
    header["checksum"] = "0" * 64
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(cache_mod.CACHE_MAGIC)
        handle.write(struct.pack(">Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(payload)

    assert cache.get(key) == value


# ---------------------------------------------------------------------------
# LRU / budget and stats
# ---------------------------------------------------------------------------


def test_budget_enforcement_evicts_oldest_entries(make_local_cache, cache_mod):
    cache = make_local_cache("eviction_case", max_bytes=2600)
    old_key = cache_mod.cache_key("features", "evict.old", {"case": 1})
    new_key = cache_mod.cache_key("features", "evict.new", {"case": 2})

    cache.set(old_key, b"a" * 1600, metadata={"namespace": "features", "fn_id": "evict.old"}, serializer="bytes")
    time.sleep(0.01)
    cache.set(new_key, b"b" * 1600, metadata={"namespace": "features", "fn_id": "evict.new"}, serializer="bytes")

    assert cache.exists(new_key) is True
    assert cache.exists(old_key, validate_expiry=False) is False
    st = cache.stats()
    assert st.eviction_count >= 1
    assert st.bytes_evicted > 0
    assert st.n_entries == 1



def test_stats_track_hits_misses_and_saved_recompute_time(local_cache, cache_mod):
    hit_key = cache_mod.cache_key("features", "stats.hit", {"case": 1})
    miss_key = cache_mod.cache_key("features", "stats.miss", {"case": 2})

    local_cache.set(
        hit_key,
        {"value": 1},
        metadata={
            "namespace": "features",
            "fn_id": "stats.hit",
            "producer_runtime_ms": 12.5,
        },
    )

    assert local_cache.get(hit_key) == {"value": 1}
    assert local_cache.get(miss_key, default=None) is None

    st = local_cache.stats()
    assert st.hits == 1
    assert st.misses == 1
    assert st.hit_rate == pytest.approx(0.5)
    assert st.recompute_time_saved_est_ms == pytest.approx(12.5)
    assert st.n_entries == 1
    assert st.per_namespace_stats["features"]["n_entries"] == 1


# ---------------------------------------------------------------------------
# get_or_compute and module-level wrappers
# ---------------------------------------------------------------------------


def test_get_or_compute_computes_once_then_hits_cache(local_cache, cache_mod):
    key = cache_mod.cache_key("features", "get_or_compute", {"case": 1})
    calls = {"n": 0}

    def producer():
        calls["n"] += 1
        return {"value": 123}

    first = local_cache.get_or_compute(
        key=key,
        producer=producer,
        metadata={"namespace": "features", "fn_id": "get_or_compute"},
    )
    second = local_cache.get_or_compute(
        key=key,
        producer=producer,
        metadata={"namespace": "features", "fn_id": "get_or_compute"},
    )

    assert first == {"value": 123}
    assert second == {"value": 123}
    assert calls["n"] == 1
    assert local_cache.stats().hits == 1



def test_module_level_wrappers_delegate_to_default_cache(monkeypatch, make_local_cache, cache_mod):
    cache = make_local_cache("module_wrappers")
    monkeypatch.setattr(cache_mod, "_default_cache", cache)

    key = cache_mod.cache_key("features", "wrapper", {"case": 1})
    cache_mod.set(key, {"x": 1}, metadata={"namespace": "features", "fn_id": "wrapper"})
    assert cache_mod.exists(key) is True
    assert cache_mod.get(key) == {"x": 1}
    rows = cache_mod.list_entries(namespace="features")
    assert len(rows) == 1
    assert rows[0]["fn_id"] == "wrapper"
    stats = cache_mod.stats()
    assert isinstance(stats, dict)
    assert stats["hits"] >= 1
    assert cache_mod.delete(key) is True
    assert cache_mod.clear() == 0
