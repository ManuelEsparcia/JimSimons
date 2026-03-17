from __future__ import annotations

"""
simons_core.io.paths
====================

Deterministic canonical locator resolution for the quant stack.

This module defines the institutional naming layer for artifact locations:
data, models, reports, cache, ops and temporary materializations. Its purpose
is not to perform I/O, transactions or snapshot management, but to impose a
single reproducible mapping from a logical artifact descriptor to a canonical
material locator while preventing traversal attacks and environment leakage.

Core guarantees
---------------
- Deterministic descriptor -> locator resolution.
- Strict segment normalization and typed validation.
- Closed-set environments and artifact classes.
- Local filesystem and S3 URI compatibility.
- Fail-closed confinement inside the authorized root.
- Explicit separation between persistent and temporary namespaces.
- Support for layout versioning without silent collisions.
"""

from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import unicodedata
from typing import Any, Final, Iterable, Literal, Mapping, Sequence
from urllib.parse import SplitResult, urlsplit, urlunsplit
from uuid import uuid4


__all__ = [
    "ALLOWED_ARTIFACT_CLASSES",
    "ALLOWED_ENVS",
    "DEFAULT_ENV_ROOTS",
    "DEFAULT_LAYOUT_VERSION",
    "ArtifactClass",
    "Backend",
    "EnvName",
    "Locator",
    "LocatorMetricsSnapshot",
    "InvalidDateFormatError",
    "InvalidSegmentError",
    "InvalidVersionError",
    "LocatorError",
    "LocatorSemanticError",
    "PathTraversalError",
    "UnauthorizedRootError",
    "UnknownEnvironmentError",
    "build_locator",
    "build_tmp_locator",
    "canonical_descriptor_digest",
    "confined_to_root",
    "ensure_parent_dirs",
    "get_metrics_snapshot",
    "get_root",
    "locator_from_uri",
    "normalize_segment",
    "reset_metrics",
    "validate_locator",
]


Backend = Literal["file", "s3"]
EnvName = Literal["local", "dev", "prod", "test"]
ArtifactClass = Literal["data", "models", "reports", "cache", "ops", "tmp"]
SegmentKind = Literal[
    "generic",
    "date",
    "version",
    "run_id",
    "digest",
    "partition",
    "env",
    "artifact_name",
    "artifact_class",
]


DEFAULT_LAYOUT_VERSION: Final[str] = "v1"
DEFAULT_ENV_ROOTS: Final[dict[EnvName, str]] = {
    "local": "file:///srv/jimsimons/local",
    "dev": "s3://jimsimons-dev",
    "prod": "s3://jimsimons-prod",
    "test": "file:///tmp/jimsimons-test",
}
ALLOWED_ENVS: Final[tuple[EnvName, ...]] = ("local", "dev", "prod", "test")
ALLOWED_ARTIFACT_CLASSES: Final[tuple[ArtifactClass, ...]] = (
    "data",
    "models",
    "reports",
    "cache",
    "ops",
    "tmp",
)

SEGMENT_MAX_LEN: Final[int] = 63
ARTIFACT_NAME_MAX_LEN: Final[int] = 127
MAX_URI_LENGTH: Final[int] = 4096
DIGEST_DEFAULT_LEN: Final[int] = 12

SEGMENT_REGEX: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,62}$")
PARTITION_REGEX: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9_\-=]{0,127}$")
ARTIFACT_NAME_REGEX: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9_.\-]{0,127}$")
DATE_REGEX: Final[re.Pattern[str]] = re.compile(r"^\d{4}-\d{2}-\d{2}$")
VERSION_REGEX: Final[re.Pattern[str]] = re.compile(r"^v\d+(?:\.\d+){0,2}$")
RUN_ID_REGEX: Final[re.Pattern[str]] = re.compile(r"^run_[a-z0-9][a-z0-9_\-]{2,63}$")
DIGEST_REGEX: Final[re.Pattern[str]] = re.compile(r"^[a-f0-9]{8,64}$")

_RESERVED_GENERIC_SEGMENTS: Final[frozenset[str]] = frozenset({"latest", "null", "none"})


class LocatorError(RuntimeError):
    """Base class for canonical locator failures."""


class UnknownEnvironmentError(LocatorError, ValueError):
    """Raised when an environment is unknown or unsupported."""


class InvalidSegmentError(LocatorError, ValueError):
    """Raised when a locator segment is invalid after normalization."""


class InvalidDateFormatError(InvalidSegmentError):
    """Raised when a date segment is not in YYYY-MM-DD format."""


class InvalidVersionError(InvalidSegmentError):
    """Raised when a version segment is not in the accepted form."""


class PathTraversalError(LocatorError, ValueError):
    """Raised when a locator would escape its authorized root."""


class UnauthorizedRootError(LocatorError, ValueError):
    """Raised when a URI does not belong to any authorized root."""


class LocatorSemanticError(LocatorError, ValueError):
    """Raised when the segment layout is semantically inconsistent."""


@dataclass(frozen=True, slots=True)
class Locator:
    """Strong immutable abstraction for a canonical artifact locator."""

    uri: str
    backend: Backend
    env: EnvName
    artifact_class: ArtifactClass
    segments: tuple[str, ...]
    root: str
    layout_version: str = DEFAULT_LAYOUT_VERSION

    @property
    def is_temporary(self) -> bool:
        return self.artifact_class == "tmp"

    @property
    def leaf(self) -> str | None:
        return self.segments[-1] if self.segments else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def local_path(self) -> Path:
        if self.backend != "file":
            raise LocatorSemanticError("local_path() is only valid for file:// locators")
        path = _file_uri_to_path(self.uri)
        return path


@dataclass(frozen=True, slots=True)
class LocatorMetricsSnapshot:
    n_locators_built: int
    n_validation_failures: int
    n_unknown_env_errors: int
    n_traversal_rejections: int
    n_reserved_word_rejections: int
    distribution_by_env: dict[str, int]
    distribution_by_artifact_class: dict[str, int]


@dataclass(slots=True)
class _Metrics:
    n_locators_built: int = 0
    n_validation_failures: int = 0
    n_unknown_env_errors: int = 0
    n_traversal_rejections: int = 0
    n_reserved_word_rejections: int = 0
    distribution_by_env: dict[str, int] = None  # type: ignore[assignment]
    distribution_by_artifact_class: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.distribution_by_env is None:
            self.distribution_by_env = {}
        if self.distribution_by_artifact_class is None:
            self.distribution_by_artifact_class = {}


_METRICS = _Metrics()
_METRICS_LOCK = threading.Lock()


def reset_metrics() -> None:
    with _METRICS_LOCK:
        _METRICS.n_locators_built = 0
        _METRICS.n_validation_failures = 0
        _METRICS.n_unknown_env_errors = 0
        _METRICS.n_traversal_rejections = 0
        _METRICS.n_reserved_word_rejections = 0
        _METRICS.distribution_by_env = {}
        _METRICS.distribution_by_artifact_class = {}


def get_metrics_snapshot() -> LocatorMetricsSnapshot:
    with _METRICS_LOCK:
        return LocatorMetricsSnapshot(
            n_locators_built=_METRICS.n_locators_built,
            n_validation_failures=_METRICS.n_validation_failures,
            n_unknown_env_errors=_METRICS.n_unknown_env_errors,
            n_traversal_rejections=_METRICS.n_traversal_rejections,
            n_reserved_word_rejections=_METRICS.n_reserved_word_rejections,
            distribution_by_env=dict(_METRICS.distribution_by_env),
            distribution_by_artifact_class=dict(_METRICS.distribution_by_artifact_class),
        )


@dataclass(frozen=True, slots=True)
class _ArtifactLayoutPolicy:
    require_domain: bool
    require_entity: bool = False
    require_run_id: bool = False
    require_one_of: tuple[str, ...] = ()


_LAYOUT_POLICIES: Final[dict[ArtifactClass, _ArtifactLayoutPolicy]] = {
    "data": _ArtifactLayoutPolicy(require_domain=True, require_entity=True),
    "models": _ArtifactLayoutPolicy(require_domain=True, require_entity=True),
    "reports": _ArtifactLayoutPolicy(require_domain=True),
    "cache": _ArtifactLayoutPolicy(require_domain=True, require_one_of=("entity", "digest", "artifact_name")),
    "ops": _ArtifactLayoutPolicy(require_domain=True),
    "tmp": _ArtifactLayoutPolicy(require_domain=True, require_run_id=True),
}


def _bump(counter: str, *, env: str | None = None, artifact_class: str | None = None) -> None:
    with _METRICS_LOCK:
        if counter == "built":
            _METRICS.n_locators_built += 1
        elif counter == "validation_failures":
            _METRICS.n_validation_failures += 1
        elif counter == "unknown_env_errors":
            _METRICS.n_unknown_env_errors += 1
        elif counter == "traversal_rejections":
            _METRICS.n_traversal_rejections += 1
        elif counter == "reserved_word_rejections":
            _METRICS.n_reserved_word_rejections += 1

        if env is not None:
            _METRICS.distribution_by_env[env] = _METRICS.distribution_by_env.get(env, 0) + 1
        if artifact_class is not None:
            _METRICS.distribution_by_artifact_class[artifact_class] = (
                _METRICS.distribution_by_artifact_class.get(artifact_class, 0) + 1
            )


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        if isinstance(value, datetime) and value.tzinfo is not None:
            return value.astimezone(timezone.utc).isoformat()
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_descriptor_digest(descriptor: Mapping[str, Any], *, length: int = DIGEST_DEFAULT_LEN) -> str:
    """Return a deterministic SHA-256 hex prefix for a logical descriptor."""
    if length < 8 or length > 64:
        raise ValueError("Digest length must be between 8 and 64 hex characters")
    digest = hashlib.sha256(_canonical_json_bytes(dict(descriptor))).hexdigest()
    return digest[:length]


def _normalize_env_roots(env_roots: Mapping[str, str] | None = None) -> dict[EnvName, str]:
    roots_in = dict(DEFAULT_ENV_ROOTS)
    if env_roots is not None:
        roots_in.update({str(k): str(v) for k, v in env_roots.items()})

    normalized: dict[EnvName, str] = {}
    for env in ALLOWED_ENVS:
        root = roots_in.get(env)
        if root is None:
            raise UnknownEnvironmentError(f"Missing configured root for environment {env!r}")
        normalized[env] = _normalize_root_uri(root)
    return normalized


def get_root(env: str, *, env_roots: Mapping[str, str] | None = None) -> str:
    """Return the authorized canonical root URI for a closed-set environment."""
    if env not in ALLOWED_ENVS:
        _bump("unknown_env_errors")
        raise UnknownEnvironmentError(f"Unknown environment: {env!r}")
    roots = _normalize_env_roots(env_roots)
    return roots[env]  # type: ignore[index]


def normalize_segment(value: str, *, kind: SegmentKind = "generic") -> str:
    """Normalize and validate a typed logical segment."""
    if not isinstance(value, str):
        raise InvalidSegmentError(f"Segment must be a string, got {type(value).__name__}")

    raw = value.strip()
    if not raw:
        raise InvalidSegmentError("Empty segment")
    if "/" in raw or "\\" in raw:
        raise PathTraversalError(f"Separators are forbidden inside a segment: {value!r}")
    if raw in {".", ".."}:
        raise PathTraversalError(f"Forbidden traversal segment: {value!r}")

    if kind == "date":
        if not DATE_REGEX.fullmatch(raw):
            raise InvalidDateFormatError(f"Invalid date segment: {value!r}; expected YYYY-MM-DD")
        return raw

    if kind == "version":
        s = raw.lower()
        if not VERSION_REGEX.fullmatch(s):
            raise InvalidVersionError(f"Invalid version segment: {value!r}")
        return s

    if kind == "env":
        s = raw.lower()
        if s not in ALLOWED_ENVS:
            _bump("unknown_env_errors")
            raise UnknownEnvironmentError(f"Unknown environment segment: {value!r}")
        return s

    if kind == "artifact_class":
        s = raw.lower()
        if s not in ALLOWED_ARTIFACT_CLASSES:
            raise InvalidSegmentError(f"Invalid artifact_class segment: {value!r}")
        return s

    if kind == "run_id":
        s = _canonicalize_free_text(raw)
        if not s.startswith("run_"):
            s = f"run_{s}"
        if not RUN_ID_REGEX.fullmatch(s):
            raise InvalidSegmentError(f"Invalid run_id segment: {value!r}")
        return s

    if kind == "digest":
        s = raw.lower().strip()
        if not DIGEST_REGEX.fullmatch(s):
            raise InvalidSegmentError(f"Invalid digest segment: {value!r}")
        return s

    if kind == "artifact_name":
        s = _canonicalize_free_text(raw, allow_dot=True, max_len=ARTIFACT_NAME_MAX_LEN)
        if s in _RESERVED_GENERIC_SEGMENTS:
            _bump("reserved_word_rejections")
            raise InvalidSegmentError(f"Reserved artifact_name segment: {s!r}")
        if not ARTIFACT_NAME_REGEX.fullmatch(s):
            raise InvalidSegmentError(f"Invalid artifact_name segment after normalization: {s!r}")
        return s

    if kind == "partition":
        s = _canonicalize_free_text(raw, allow_equals=True, max_len=127)
        if s in _RESERVED_GENERIC_SEGMENTS:
            _bump("reserved_word_rejections")
            raise InvalidSegmentError(f"Reserved partition segment: {s!r}")
        if not PARTITION_REGEX.fullmatch(s):
            raise InvalidSegmentError(f"Invalid partition segment after normalization: {s!r}")
        return s

    s = _canonicalize_free_text(raw, max_len=SEGMENT_MAX_LEN)
    if s in _RESERVED_GENERIC_SEGMENTS:
        _bump("reserved_word_rejections")
        raise InvalidSegmentError(f"Reserved segment: {s!r}")
    if not SEGMENT_REGEX.fullmatch(s):
        raise InvalidSegmentError(f"Invalid generic segment after normalization: {s!r}")
    return s


def _canonicalize_free_text(
    value: str,
    *,
    allow_dot: bool = False,
    allow_equals: bool = False,
    max_len: int = SEGMENT_MAX_LEN,
) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.strip().lower()
    replacement_re = r"[^a-z0-9_\-.]" if allow_dot else r"[^a-z0-9_\-]"
    if allow_equals:
        replacement_re = r"[^a-z0-9_\-=]"
    text = re.sub(replacement_re, "-", text)
    text = re.sub(r"-+", "-", text).strip("-._")
    if not text:
        raise InvalidSegmentError("Empty segment after normalization")
    if text in {".", ".."}:
        raise PathTraversalError("Forbidden traversal segment after normalization")
    if len(text) > max_len:
        raise InvalidSegmentError(
            f"Segment exceeds max length {max_len}: {text!r}"
        )
    return text


def build_locator(
    artifact_class: str,
    env: str,
    domain: str | None = None,
    entity: str | None = None,
    version: str | None = None,
    date: str | None = None,
    run_id: str | None = None,
    partition: str | None = None,
    artifact_name: str | None = None,
    digest: str | None = None,
    *,
    layout_version: str = DEFAULT_LAYOUT_VERSION,
    env_roots: Mapping[str, str] | None = None,
) -> Locator:
    """Build a canonical locator from a logical artifact descriptor."""
    norm_env = normalize_segment(env, kind="env")
    norm_class = normalize_segment(artifact_class, kind="artifact_class")
    root = get_root(norm_env, env_roots=env_roots)
    layout = normalize_segment(layout_version, kind="version")

    _enforce_policy(
        artifact_class=norm_class,
        domain=domain,
        entity=entity,
        run_id=run_id,
        digest=digest,
        artifact_name=artifact_name,
    )

    segments: list[str] = [norm_class, layout]
    if domain is not None:
        segments.append(normalize_segment(domain))
    if entity is not None:
        segments.append(normalize_segment(entity))
    if version is not None:
        segments.append(normalize_segment(version, kind="version"))
    if date is not None:
        segments.append(normalize_segment(date, kind="date"))
    if run_id is not None:
        segments.append(normalize_segment(run_id, kind="run_id"))
    if partition is not None:
        segments.append(normalize_segment(partition, kind="partition"))
    if digest is not None:
        segments.append(normalize_segment(digest, kind="digest"))
    if artifact_name is not None:
        segments.append(normalize_segment(artifact_name, kind="artifact_name"))

    uri = _join_root_and_segments(root, segments)
    locator = Locator(
        uri=uri,
        backend=_backend_from_uri(uri),
        env=norm_env,  # type: ignore[arg-type]
        artifact_class=norm_class,  # type: ignore[arg-type]
        segments=tuple(segments),
        root=root,
        layout_version=layout,
    )
    validate_locator(locator, env_roots=env_roots)
    _bump("built", env=locator.env, artifact_class=locator.artifact_class)
    return locator


def build_tmp_locator(
    env: str,
    scope: str,
    run_id: str,
    *,
    artifact_name: str | None = None,
    date: str | None = None,
    env_roots: Mapping[str, str] | None = None,
    layout_version: str = DEFAULT_LAYOUT_VERSION,
) -> Locator:
    """Build an isolated non-authoritative temporary locator."""
    tmp_date = date or datetime.now(timezone.utc).date().isoformat()
    tmp_leaf = artifact_name or f"tmp-{uuid4().hex[:12]}"
    digest = canonical_descriptor_digest(
        {
            "env": env,
            "scope": scope,
            "run_id": run_id,
            "date": tmp_date,
            "artifact_name": tmp_leaf,
        },
        length=12,
    )
    return build_locator(
        artifact_class="tmp",
        env=env,
        domain=scope,
        entity="staging",
        version=None,
        date=tmp_date,
        run_id=run_id,
        partition=None,
        artifact_name=tmp_leaf,
        digest=digest,
        layout_version=layout_version,
        env_roots=env_roots,
    )


def locator_from_uri(uri: str, *, env_roots: Mapping[str, str] | None = None) -> Locator:
    """Parse a material URI back into a validated Locator."""
    normalized_uri = _normalize_root_uri(uri) if _looks_like_root_uri(uri) else _normalize_artifact_uri(uri)
    roots = _normalize_env_roots(env_roots)

    matches: list[tuple[str, str]] = []
    for env_name, root in roots.items():
        if confined_to_root(normalized_uri, root):
            matches.append((env_name, root))

    if not matches:
        raise UnauthorizedRootError(f"URI is not under any authorized root: {uri!r}")

    env_name, root = max(matches, key=lambda pair: len(pair[1]))
    rel_segments = _relative_segments(normalized_uri, root)
    if len(rel_segments) < 2:
        raise LocatorSemanticError(
            "A locator must contain at least artifact_class and layout_version segments"
        )

    artifact_class = normalize_segment(rel_segments[0], kind="artifact_class")
    layout_version = normalize_segment(rel_segments[1], kind="version")
    locator = Locator(
        uri=normalized_uri,
        backend=_backend_from_uri(normalized_uri),
        env=env_name,  # type: ignore[arg-type]
        artifact_class=artifact_class,  # type: ignore[arg-type]
        segments=tuple(rel_segments),
        root=root,
        layout_version=layout_version,
    )
    validate_locator(locator, env_roots=env_roots)
    return locator


def validate_locator(locator: Locator, *, env_roots: Mapping[str, str] | None = None) -> None:
    """Validate confinement, backend coherence and semantic layout."""
    try:
        roots = _normalize_env_roots(env_roots)
        expected_root = roots[locator.env]
        if locator.root != expected_root:
            raise UnauthorizedRootError(
                f"Locator root {locator.root!r} does not match configured root {expected_root!r}"
            )

        if locator.backend != _backend_from_uri(locator.uri):
            raise LocatorSemanticError("Locator backend is inconsistent with its URI")

        if len(locator.uri) > MAX_URI_LENGTH:
            raise LocatorSemanticError(
                f"Locator exceeds maximum URI length {MAX_URI_LENGTH}: {len(locator.uri)}"
            )

        if not locator.segments:
            raise LocatorSemanticError("Locator must contain at least one segment")

        if locator.segments[0] != locator.artifact_class:
            raise LocatorSemanticError("First segment must equal artifact_class")
        if len(locator.segments) < 2:
            raise LocatorSemanticError("Locator must include layout_version as second segment")
        if locator.segments[1] != locator.layout_version:
            raise LocatorSemanticError("Second segment must equal layout_version")

        normalize_segment(locator.artifact_class, kind="artifact_class")
        normalize_segment(locator.layout_version, kind="version")

        for idx, segment in enumerate(locator.segments[2:], start=2):
            if not segment:
                raise InvalidSegmentError(f"Empty segment at position {idx}")
            if "/" in segment or "\\" in segment:
                raise PathTraversalError(f"Separator found inside segment at position {idx}")
            if segment in {".", ".."}:
                raise PathTraversalError(f"Traversal segment found at position {idx}")

        if not confined_to_root(locator.uri, locator.root):
            _bump("traversal_rejections")
            raise PathTraversalError(
                f"Locator escapes authorized root: locator={locator.uri!r} root={locator.root!r}"
            )

        _validate_semantic_layout(locator)
    except Exception:
        _bump("validation_failures")
        raise


def confined_to_root(candidate_uri: str, root_uri: str) -> bool:
    """Return True iff a normalized candidate URI remains under the normalized root."""
    candidate = _normalize_artifact_uri(candidate_uri)
    root = _normalize_root_uri(root_uri)
    candidate_split = urlsplit(candidate)
    root_split = urlsplit(root)

    if candidate_split.scheme != root_split.scheme or candidate_split.netloc != root_split.netloc:
        return False

    if candidate_split.scheme == "file":
        candidate_path = os.path.normpath(candidate_split.path)
        root_path = os.path.normpath(root_split.path)
        try:
            common = os.path.commonpath([candidate_path, root_path])
        except ValueError:
            return False
        return common == root_path

    candidate_path = _normalize_posix_path(candidate_split.path)
    root_path = _normalize_posix_path(root_split.path)
    if root_path == "/":
        return candidate_path.startswith("/")
    return candidate_path == root_path or candidate_path.startswith(root_path + "/")


def ensure_parent_dirs(locator: Locator) -> None:
    """Create parent directories iff the backend is local file://."""
    validate_locator(locator, env_roots={locator.env: locator.root})
    if locator.backend != "file":
        return
    path = locator.local_path()
    directory = path.parent if _looks_like_file_leaf(locator.leaf) else path
    directory.mkdir(parents=True, exist_ok=True)


def _validate_semantic_layout(locator: Locator) -> None:
    policy = _LAYOUT_POLICIES[locator.artifact_class]
    fields = _semantic_fields(locator)

    if policy.require_domain and fields["domain"] is None:
        raise LocatorSemanticError(f"artifact_class={locator.artifact_class!r} requires a domain segment")
    if policy.require_entity and fields["entity"] is None:
        raise LocatorSemanticError(f"artifact_class={locator.artifact_class!r} requires an entity segment")
    if policy.require_run_id and fields["run_id"] is None:
        raise LocatorSemanticError(f"artifact_class={locator.artifact_class!r} requires a run_id segment")

    if policy.require_one_of and not any(fields[name] is not None for name in policy.require_one_of):
        joined = ", ".join(policy.require_one_of)
        raise LocatorSemanticError(
            f"artifact_class={locator.artifact_class!r} requires at least one of: {joined}"
        )

    if locator.is_temporary and locator.artifact_class != "tmp":
        raise LocatorSemanticError("Temporary locator invariant broken")


def _semantic_fields(locator: Locator) -> dict[str, str | None]:
    tail = list(locator.segments[2:])
    fields: dict[str, str | None] = {
        "domain": None,
        "entity": None,
        "version": None,
        "date": None,
        "run_id": None,
        "partition": None,
        "digest": None,
        "artifact_name": None,
    }
    order = ["domain", "entity", "version", "date", "run_id", "partition", "digest", "artifact_name"]
    for idx, name in enumerate(order):
        if idx < len(tail):
            fields[name] = tail[idx]
    return fields


def _enforce_policy(
    *,
    artifact_class: str,
    domain: str | None,
    entity: str | None,
    run_id: str | None,
    digest: str | None,
    artifact_name: str | None,
) -> None:
    policy = _LAYOUT_POLICIES[artifact_class]  # type: ignore[index]
    if policy.require_domain and domain is None:
        raise LocatorSemanticError(f"artifact_class={artifact_class!r} requires domain")
    if policy.require_entity and entity is None:
        raise LocatorSemanticError(f"artifact_class={artifact_class!r} requires entity")
    if policy.require_run_id and run_id is None:
        raise LocatorSemanticError(f"artifact_class={artifact_class!r} requires run_id")
    if policy.require_one_of:
        values = {
            "entity": entity,
            "digest": digest,
            "artifact_name": artifact_name,
        }
        if not any(values.get(name) is not None for name in policy.require_one_of):
            raise LocatorSemanticError(
                f"artifact_class={artifact_class!r} requires at least one of {policy.require_one_of!r}"
            )


def _normalize_root_uri(root: str) -> str:
    split = _parse_uri(root)
    if split.scheme not in {"file", "s3"}:
        raise UnauthorizedRootError(f"Unsupported root scheme: {split.scheme!r}")

    if split.scheme == "file":
        if split.netloc not in {"", "localhost"}:
            raise UnauthorizedRootError(f"Unsupported file root netloc: {split.netloc!r}")
        path = split.path or "/"
        if not path.startswith("/"):
            raise UnauthorizedRootError("file:// roots must be absolute")
        normalized_path = os.path.normpath(path)
        return urlunsplit(("file", "", normalized_path, "", ""))

    bucket = split.netloc.strip().lower()
    if not bucket:
        raise UnauthorizedRootError("s3:// roots must include a bucket")
    normalized_path = _normalize_posix_path(split.path)
    return urlunsplit(("s3", bucket, normalized_path, "", ""))


def _normalize_artifact_uri(uri: str) -> str:
    split = _parse_uri(uri)
    if split.scheme not in {"file", "s3"}:
        raise UnauthorizedRootError(f"Unsupported artifact URI scheme: {split.scheme!r}")

    if split.scheme == "file":
        if split.netloc not in {"", "localhost"}:
            raise UnauthorizedRootError(f"Unsupported file URI netloc: {split.netloc!r}")
        path = os.path.normpath(split.path)
        return urlunsplit(("file", "", path, "", ""))

    bucket = split.netloc.strip().lower()
    if not bucket:
        raise UnauthorizedRootError("s3:// URIs must include a bucket")
    path = _normalize_posix_path(split.path)
    return urlunsplit(("s3", bucket, path, "", ""))


def _parse_uri(value: str) -> SplitResult:
    if not isinstance(value, str) or not value.strip():
        raise UnauthorizedRootError("URI/root must be a non-empty string")
    split = urlsplit(value.strip())
    if not split.scheme:
        raise UnauthorizedRootError(
            "URIs must be explicit (file://... or s3://...); implicit local paths are forbidden"
        )
    return split


def _backend_from_uri(uri: str) -> Backend:
    scheme = urlsplit(uri).scheme
    if scheme == "file":
        return "file"
    if scheme == "s3":
        return "s3"
    raise UnauthorizedRootError(f"Unsupported URI scheme: {scheme!r}")


def _join_root_and_segments(root: str, segments: Sequence[str]) -> str:
    root_norm = _normalize_root_uri(root)
    if not segments:
        return root_norm

    split = urlsplit(root_norm)
    if split.scheme == "file":
        path = Path(split.path)
        joined = path.joinpath(*segments)
        uri = joined.as_uri()
    else:
        base_path = _normalize_posix_path(split.path)
        joined_path = "/".join([base_path.lstrip("/")] + list(segments)).strip("/")
        uri = urlunsplit(("s3", split.netloc, f"/{joined_path}", "", ""))
    if len(uri) > MAX_URI_LENGTH:
        raise LocatorSemanticError(f"Locator exceeds maximum URI length {MAX_URI_LENGTH}")
    return uri


def _normalize_posix_path(path: str) -> str:
    parts = [part for part in path.replace("\\", "/").split("/") if part not in {"", "."}]
    normalized: list[str] = []
    for part in parts:
        if part == "..":
            raise PathTraversalError("Parent traversal is forbidden in URI paths")
        normalized.append(part)
    return "/" + "/".join(normalized)


def _relative_segments(uri: str, root: str) -> tuple[str, ...]:
    uri_split = urlsplit(_normalize_artifact_uri(uri))
    root_split = urlsplit(_normalize_root_uri(root))
    if uri_split.scheme != root_split.scheme or uri_split.netloc != root_split.netloc:
        raise UnauthorizedRootError("URI/root backend mismatch")

    uri_path = _normalize_posix_path(uri_split.path)
    root_path = _normalize_posix_path(root_split.path)
    if not confined_to_root(uri, root):
        raise PathTraversalError(f"URI escapes root: uri={uri!r} root={root!r}")

    if uri_path == root_path:
        return ()
    rel = uri_path[len(root_path):].lstrip("/")
    if not rel:
        return ()
    return tuple(seg for seg in rel.split("/") if seg)


def _file_uri_to_path(uri: str) -> Path:
    split = urlsplit(_normalize_artifact_uri(uri))
    if split.scheme != "file":
        raise LocatorSemanticError("Only file:// URIs can be converted to local paths")
    return Path(split.path)


def _looks_like_file_leaf(segment: str | None) -> bool:
    if not segment:
        return False
    return "." in segment and not segment.startswith(".") and not segment.endswith(".")


def _looks_like_root_uri(value: str) -> bool:
    split = urlsplit(value)
    if split.scheme == "file":
        return bool(split.path) and not _looks_like_file_leaf(Path(split.path).name)
    if split.scheme == "s3":
        return not _looks_like_file_leaf(Path(split.path).name)
    return False


def _cli_build_locator(args: Namespace) -> int:
    locator = build_locator(
        artifact_class=args.artifact_class,
        env=args.env,
        domain=args.domain,
        entity=args.entity,
        version=args.version,
        date=args.date,
        run_id=args.run_id,
        partition=args.partition,
        artifact_name=args.artifact_name,
        digest=args.digest,
        layout_version=args.layout_version,
    )
    print(locator.uri)
    return 0


def _cli_validate(args: Namespace) -> int:
    locator = locator_from_uri(args.uri)
    print(json.dumps(locator.to_dict(), indent=2, sort_keys=True))
    return 0


def _cli_roots(_: Namespace) -> int:
    print(json.dumps(_normalize_env_roots(), indent=2, sort_keys=True))
    return 0


def _cli_tmp(args: Namespace) -> int:
    locator = build_tmp_locator(
        env=args.env,
        scope=args.scope,
        run_id=args.run_id,
        artifact_name=args.artifact_name,
        date=args.date,
        layout_version=args.layout_version,
    )
    print(locator.uri)
    return 0


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Canonical locator builder/validator for simons_core.io.paths")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_print = subparsers.add_parser("print", help="Build and print a canonical locator")
    p_print.add_argument("--artifact-class", required=True)
    p_print.add_argument("--env", required=True)
    p_print.add_argument("--domain")
    p_print.add_argument("--entity")
    p_print.add_argument("--version")
    p_print.add_argument("--date")
    p_print.add_argument("--run-id")
    p_print.add_argument("--partition")
    p_print.add_argument("--artifact-name")
    p_print.add_argument("--digest")
    p_print.add_argument("--layout-version", default=DEFAULT_LAYOUT_VERSION)
    p_print.set_defaults(func=_cli_build_locator)

    p_validate = subparsers.add_parser("validate", help="Validate and parse a canonical locator URI")
    p_validate.add_argument("--uri", required=True)
    p_validate.set_defaults(func=_cli_validate)

    p_roots = subparsers.add_parser("roots", help="Print authorized roots")
    p_roots.set_defaults(func=_cli_roots)

    p_tmp = subparsers.add_parser("tmp", help="Build and print an isolated tmp locator")
    p_tmp.add_argument("--env", required=True)
    p_tmp.add_argument("--scope", required=True)
    p_tmp.add_argument("--run-id", required=True)
    p_tmp.add_argument("--artifact-name")
    p_tmp.add_argument("--date")
    p_tmp.add_argument("--layout-version", default=DEFAULT_LAYOUT_VERSION)
    p_tmp.set_defaults(func=_cli_tmp)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
