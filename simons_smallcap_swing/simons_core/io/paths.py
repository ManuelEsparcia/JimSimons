"""
simons_core/io/paths.py — Deterministic locator resolution.

Resolves logical artifact descriptors to canonical paths/URIs with:
- Strict segment normalization (lowercase, regex-validated)
- Environment-based root isolation (local/dev/prod/test)
- Anti-traversal confinement (no "..", no escape from root)
- Typed segments: generic, date, version, run_id, digest
- Backend awareness: file:// vs s3://

Core invariant:
    same descriptor → same locator (deterministic)
    every valid locator ∈ prefix_closure(root(env))
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EnvName = Literal["local", "dev", "prod", "test"]
Backend = Literal["file", "s3"]
SegmentKind = Literal["generic", "date", "version", "run_id", "digest", "env"]
ArtifactClass = Literal["data", "models", "reports", "cache", "ops", "tmp"]

SEGMENT_REGEX = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,62}$")
DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")
VERSION_REGEX = re.compile(r"^v\d+(?:\.\d+){0,2}$")
DIGEST_REGEX = re.compile(r"^[a-f0-9]{8,64}$")

RESERVED = frozenset({"null", "none", "latest", "default"})
VALID_ENVS = frozenset({"local", "dev", "prod", "test"})
VALID_ARTIFACT_CLASSES = frozenset({"data", "models", "reports", "cache", "ops", "tmp"})

DEFAULT_ENV_ROOTS: dict[str, str] = {
    "local": "file:///srv/jimsimons/local",
    "dev": "file:///srv/jimsimons/dev",
    "prod": "file:///srv/jimsimons/prod",
    "test": "file:///tmp/jimsimons-test",
}

DIGEST_DEFAULT_LEN = 16
MAX_PATH_LEN = 4096


# ---------------------------------------------------------------------------
# Errors (spec §18)
# ---------------------------------------------------------------------------

class LocatorError(RuntimeError):
    pass

class UnknownEnvironmentError(LocatorError, ValueError):
    pass

class InvalidSegmentError(LocatorError, ValueError):
    pass

class InvalidDateFormatError(InvalidSegmentError):
    pass

class InvalidVersionError(InvalidSegmentError):
    pass

class PathTraversalError(LocatorError, ValueError):
    pass

class UnauthorizedRootError(LocatorError, ValueError):
    pass

class LocatorSemanticError(LocatorError, ValueError):
    pass


# ---------------------------------------------------------------------------
# Locator (spec §12)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Locator:
    """Immutable artifact locator."""
    uri: str
    backend: str          # "file" | "s3"
    env: str
    artifact_class: str
    segments: tuple[str, ...]

    @property
    def is_temporary(self) -> bool:
        return self.artifact_class == "tmp"

    @property
    def leaf(self) -> str | None:
        return self.segments[-1] if self.segments else None

    def local_path(self) -> Path:
        if self.backend != "file":
            raise LocatorError(f"Cannot get local path for backend={self.backend}")
        return Path(self.uri.replace("file://", ""))


# ---------------------------------------------------------------------------
# Environment roots (spec §8)
# ---------------------------------------------------------------------------

def get_root(env: str, *, env_roots: Mapping[str, str] | None = None) -> str:
    """Return the authorised root for an environment. Fails if unknown."""
    roots = env_roots or DEFAULT_ENV_ROOTS
    if env not in roots:
        raise UnknownEnvironmentError(f"Unknown environment: {env!r}. Valid: {sorted(roots.keys())}")
    return roots[env]


# ---------------------------------------------------------------------------
# Segment normalization (spec §6)
# ---------------------------------------------------------------------------

def normalize_segment(value: str, *, kind: SegmentKind = "generic") -> str:
    """Normalize and validate a path segment.

    Pure, deterministic, total on valid inputs.
    Raises InvalidSegmentError on invalid inputs.
    """
    s = unicodedata.normalize("NFC", str(value)).strip().lower()

    if kind == "date":
        if not DATE_REGEX.fullmatch(s):
            raise InvalidDateFormatError(f"Invalid date segment: {value!r} (expected YYYY-MM-DD)")
        return s

    if kind == "version":
        if not VERSION_REGEX.fullmatch(s):
            raise InvalidVersionError(f"Invalid version segment: {value!r} (expected vN or vN.N.N)")
        return s

    if kind == "digest":
        if not DIGEST_REGEX.fullmatch(s):
            raise InvalidSegmentError(f"Invalid digest segment: {value!r}")
        return s

    # Check traversal BEFORE normalization
    if s in (".", "..") or ".." in s:
        raise PathTraversalError(f"Forbidden traversal segment: {value!r}")

    # Generic normalization
    s = re.sub(r"[^a-z0-9_\-]", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")

    if not s:
        raise InvalidSegmentError(f"Empty segment after normalization: {value!r}")
    if s in RESERVED:
        raise InvalidSegmentError(f"Reserved segment: {s!r}")
    if not SEGMENT_REGEX.fullmatch(s):
        raise InvalidSegmentError(f"Invalid segment: {s!r} (from {value!r})")
    return s


# ---------------------------------------------------------------------------
# Confinement (spec §10)
# ---------------------------------------------------------------------------

def confined_to_root(candidate: str, root: str) -> bool:
    """Check that candidate path is confined within root."""
    # Strip file:// prefix for comparison
    c = candidate.replace("file://", "").replace("s3://", "")
    r = root.replace("file://", "").replace("s3://", "")
    c_norm = os.path.normpath(c)
    r_norm = os.path.normpath(r)
    return c_norm == r_norm or c_norm.startswith(r_norm + os.sep)


# ---------------------------------------------------------------------------
# Digest for parametric artifacts (spec §11)
# ---------------------------------------------------------------------------

def canonical_descriptor_digest(
    descriptor: Mapping[str, Any],
    *,
    length: int = DIGEST_DEFAULT_LEN,
) -> str:
    """SHA-256 digest of a canonical JSON serialization."""
    blob = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:length]


# ---------------------------------------------------------------------------
# Builder (spec §14.2)
# ---------------------------------------------------------------------------

def build_locator(
    artifact_class: str,
    env: str,
    *,
    domain: str | None = None,
    entity: str | None = None,
    version: str | None = None,
    date: str | None = None,
    run_id: str | None = None,
    partition: str | None = None,
    artifact_name: str | None = None,
    digest: str | None = None,
    env_roots: Mapping[str, str] | None = None,
) -> Locator:
    """Build a canonical locator from logical descriptor."""
    root = get_root(env, env_roots=env_roots)

    if artifact_class not in VALID_ARTIFACT_CLASSES:
        raise LocatorSemanticError(f"Invalid artifact_class: {artifact_class!r}")

    segs: list[str] = [normalize_segment(artifact_class)]

    for val, kind in [
        (domain, "generic"), (entity, "generic"),
        (version, "version"), (date, "date"),
        (run_id, "generic"), (partition, "generic"),
        (digest, "digest"), (artifact_name, "generic"),
    ]:
        if val is not None:
            segs.append(normalize_segment(val, kind=kind))

    uri = root.rstrip("/") + "/" + "/".join(segs)

    if len(uri) > MAX_PATH_LEN:
        raise LocatorError(f"Path too long: {len(uri)} > {MAX_PATH_LEN}")

    backend: str = "s3" if uri.startswith("s3://") else "file"

    loc = Locator(uri=uri, backend=backend, env=env,
                  artifact_class=artifact_class, segments=tuple(segs))

    validate_locator(loc, env_roots=env_roots)
    return loc


def build_tmp_locator(
    env: str,
    scope: str,
    run_id: str,
    *,
    env_roots: Mapping[str, str] | None = None,
) -> Locator:
    """Build an isolated temporary locator."""
    return build_locator("tmp", env, domain=scope, run_id=run_id, env_roots=env_roots)


# ---------------------------------------------------------------------------
# Validation (spec §15.4)
# ---------------------------------------------------------------------------

def validate_locator(
    locator: Locator,
    *,
    env_roots: Mapping[str, str] | None = None,
) -> None:
    """Validate confinement, format, and semantic consistency."""
    if locator.env not in VALID_ENVS:
        raise UnknownEnvironmentError(f"Invalid env: {locator.env!r}")

    root = get_root(locator.env, env_roots=env_roots)

    if not confined_to_root(locator.uri, root):
        raise UnauthorizedRootError(
            f"Locator escapes root: {locator.uri!r} not confined to {root!r}"
        )

    # Check for traversal in segments
    for seg in locator.segments:
        if ".." in seg:
            raise PathTraversalError(f"Traversal detected in segment: {seg!r}")

    if not locator.segments:
        raise LocatorSemanticError("Locator has no segments")


# ---------------------------------------------------------------------------
# Filesystem helpers (spec §15.5)
# ---------------------------------------------------------------------------

def ensure_parent_dirs(locator: Locator) -> None:
    """Create parent directories (local backend only)."""
    if locator.backend != "file":
        return
    p = locator.local_path()
    p.parent.mkdir(parents=True, exist_ok=True)
