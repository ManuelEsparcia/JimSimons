from __future__ import annotations

import os
from pathlib import Path


APP_DIR_NAME = "simons_smallcap_swing"
REPO_ROOT_ENV = "SIMONS_REPO_ROOT"
APP_ROOT_ENV = "SIMONS_APP_ROOT"


class PathResolutionError(ValueError):
    """Raised when canonical project roots cannot be located."""


class PathSecurityError(ValueError):
    """Raised when a path escapes the allowed root."""


def _resolve_start(start: str | Path | None) -> Path:
    base = Path(start) if start is not None else Path(__file__)
    base = base.expanduser().resolve()
    if base.is_file():
        return base.parent
    return base


def find_repo_root(start: str | Path | None = None) -> Path:
    env_override = os.getenv(REPO_ROOT_ENV)
    if env_override:
        candidate = Path(env_override).expanduser().resolve()
        if not (candidate / APP_DIR_NAME).is_dir():
            raise PathResolutionError(
                f"{REPO_ROOT_ENV} is set but '{APP_DIR_NAME}' is missing: {candidate}"
            )
        return candidate

    cursor = _resolve_start(start)
    for candidate in (cursor, *cursor.parents):
        if (candidate / APP_DIR_NAME).is_dir():
            return candidate

    raise PathResolutionError(
        f"Could not locate repo root containing '{APP_DIR_NAME}' from start={cursor}"
    )


def repo_root() -> Path:
    return find_repo_root()


def app_root() -> Path:
    env_override = os.getenv(APP_ROOT_ENV)
    if env_override:
        candidate = Path(env_override).expanduser().resolve()
        if not candidate.is_dir():
            raise PathResolutionError(f"{APP_ROOT_ENV} does not point to a directory: {candidate}")
        return candidate

    candidate = repo_root() / APP_DIR_NAME
    if not candidate.is_dir():
        raise PathResolutionError(f"Application root not found: {candidate}")
    return candidate


def data_dir() -> Path:
    return app_root() / "data"


def reference_dir() -> Path:
    return data_dir() / "reference"


def configs_dir() -> Path:
    return app_root() / "configs"


def configs_data_dir() -> Path:
    return configs_dir() / "data"


def artifacts_dir() -> Path:
    return app_root() / "artifacts"


def output_dir() -> Path:
    return artifacts_dir()


def cache_dir() -> Path:
    return app_root() / "cache"


def ensure_within_root(path: str | Path, *, root: str | Path | None = None) -> Path:
    root_path = Path(root).expanduser().resolve() if root is not None else app_root().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root_path / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise PathSecurityError(
            f"Path escapes root. resolved={resolved}, root={root_path}"
        ) from exc
    return resolved


def safe_join(base: str | Path, *parts: str | Path) -> Path:
    base_path = Path(base).expanduser().resolve()
    candidate = base_path.joinpath(*(str(part) for part in parts))
    return ensure_within_root(candidate, root=base_path)


def ensure_dir(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = app_root() / candidate
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def resolve_config_path(config: str | Path, *, must_exist: bool = True) -> Path:
    """
    Canonical policy:
    - Preferred location is `configs/data/`.
    - Compatibility fallback supports legacy `configs/*.yaml`.
    """
    incoming = Path(config)

    candidates: list[Path] = []
    if incoming.is_absolute():
        candidates.append(incoming)
    else:
        # 1) explicit relative from app root
        candidates.append(app_root() / incoming)
        # 2) canonical location for module configs
        candidates.append(configs_data_dir() / incoming)
        # 3) legacy location in current repo layout
        candidates.append(configs_dir() / incoming)

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(resolved)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    if must_exist:
        raise FileNotFoundError(
            "Config not found. Checked: " + ", ".join(str(path) for path in unique_candidates)
        )

    # For new callers, default to canonical `configs/data`.
    if not incoming.is_absolute():
        return (configs_data_dir() / incoming).resolve(strict=False)
    return incoming.resolve(strict=False)


__all__ = [
    "APP_DIR_NAME",
    "APP_ROOT_ENV",
    "PathResolutionError",
    "PathSecurityError",
    "REPO_ROOT_ENV",
    "app_root",
    "artifacts_dir",
    "cache_dir",
    "configs_data_dir",
    "configs_dir",
    "data_dir",
    "ensure_dir",
    "ensure_within_root",
    "find_repo_root",
    "output_dir",
    "reference_dir",
    "repo_root",
    "resolve_config_path",
    "safe_join",
]
