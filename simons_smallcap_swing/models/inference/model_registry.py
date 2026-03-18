"""
models/inference/model_registry.py — Model lifecycle governance.

Not a passive catalog — governs the operational right of a model version
to exist, resolve, and serve predictions. Every mutation is an append-only
event. No ambiguity: if resolution is unclear, fail conservatively.

State machine: candidate → staging → production → archived

Aliases: latest_candidate, staging, production — each points to exactly
one version at a time. Alias changes are events, not silent mutations.

Key design:
1. Immutable versions: once registered, artifact_hash is permanent.
2. Feature set hash tracking: ensures train/inference feature compatibility.
3. Validation gate: a version cannot promote without validation_passed=True.
4. Append-only event log: full audit trail, never rewritten.
5. Filesystem-backed: JSON files, no database dependency.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from . import (
    Stage,
    EventType,
    VALID_TRANSITIONS,
    RegistryError,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelVersion:
    """Immutable record of a trained model version."""
    model_id: str
    version: str
    stage: str = Stage.CANDIDATE.value
    artifact_path: str = ""
    artifact_hash: str = ""
    training_run_id: str = ""
    training_data_hash: str = ""
    feature_set_hash: str = ""
    feature_names: list[str] = field(default_factory=list)
    config_hash: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    validation_passed: bool = False
    validation_run_id: str = ""
    created_at: str = ""
    registered_by: str = "system"
    preprocess_state: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.model_id}:{self.version}"


@dataclass
class RegistryEvent:
    """Append-only audit log entry."""
    event_type: str
    model_id: str
    version: str
    timestamp: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Filesystem-backed model registry with lifecycle governance.

    Usage
    -----
    registry = ModelRegistry(base_dir="model_registry")
    registry.register(version_obj)
    registry.promote(model_id, version, Stage.STAGING)
    resolved = registry.resolve("ridge_baseline", alias="production")
    """

    def __init__(self, base_dir: str | Path = "model_registry"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._versions: dict[str, ModelVersion] = {}
        self._aliases: dict[str, dict[str, str]] = {}  # {model_id: {alias: version}}
        self._events: list[RegistryEvent] = []
        self._load()

    # --- Persistence ---

    def _state_path(self) -> Path:
        return self.base_dir / "registry_state.json"

    def _events_path(self) -> Path:
        return self.base_dir / "registry_events.jsonl"

    def _load(self) -> None:
        state_path = self._state_path()
        if state_path.exists():
            data = json.loads(state_path.read_text(encoding="utf-8"))
            for vd in data.get("versions", []):
                mv = ModelVersion(**vd)
                self._versions[mv.key] = mv
            self._aliases = data.get("aliases", {})

        events_path = self._events_path()
        if events_path.exists():
            for line in events_path.read_text(encoding="utf-8").strip().split("\n"):
                if line.strip():
                    self._events.append(RegistryEvent(**json.loads(line)))

    def _save(self) -> None:
        state = {
            "versions": [asdict(v) for v in self._versions.values()],
            "aliases": self._aliases,
        }
        self._state_path().write_text(
            json.dumps(state, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    def _append_event(self, event: RegistryEvent) -> None:
        self._events.append(event)
        with open(self._events_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), sort_keys=True, default=str) + "\n")

    # --- Registration ---

    def register(self, version: ModelVersion) -> None:
        """Register a new model version. Immutable once registered."""
        if version.key in self._versions:
            existing = self._versions[version.key]
            if existing.artifact_hash != version.artifact_hash:
                raise RegistryError(
                    f"Version {version.key} already registered with different artifact_hash. "
                    f"Versions are immutable — create a new version instead."
                )
            LOGGER.info("Version %s already registered (idempotent)", version.key)
            return

        if not version.created_at:
            version.created_at = _utc_now()
        version.stage = Stage.CANDIDATE.value

        self._versions[version.key] = version

        # Auto-assign latest_candidate alias
        model_aliases = self._aliases.setdefault(version.model_id, {})
        model_aliases["latest_candidate"] = version.version

        self._append_event(RegistryEvent(
            event_type=EventType.REGISTER_VERSION.value,
            model_id=version.model_id,
            version=version.version,
            timestamp=_utc_now(),
            details={"artifact_hash": version.artifact_hash, "metrics": version.metrics},
        ))
        self._save()

        LOGGER.info("Registered %s (artifact_hash=%s)", version.key, version.artifact_hash[:12])

    # --- Promotion ---

    def promote(
        self,
        model_id: str,
        version: str,
        target_stage: Stage,
        *,
        force: bool = False,
    ) -> None:
        """Promote a version to a new stage.

        Validates: transition is legal, validation passed (for production).
        """
        key = f"{model_id}:{version}"
        if key not in self._versions:
            raise RegistryError(f"Version {key} not found in registry")

        mv = self._versions[key]
        current_stage = Stage(mv.stage)
        valid_targets = VALID_TRANSITIONS.get(current_stage, ())

        if target_stage not in valid_targets:
            raise RegistryError(
                f"Invalid transition: {current_stage.value} → {target_stage.value}. "
                f"Valid targets from {current_stage.value}: {[s.value for s in valid_targets]}"
            )

        # Validation gate for production
        if target_stage == Stage.PRODUCTION and not mv.validation_passed and not force:
            raise RegistryError(
                f"Cannot promote {key} to production: validation_passed=False. "
                f"Pass force=True to override (logged as OVERRIDE)."
            )

        # Demote current holder of target alias (if any)
        alias_name = target_stage.value
        model_aliases = self._aliases.setdefault(model_id, {})
        old_version = model_aliases.get(alias_name)
        if old_version and old_version != version:
            old_key = f"{model_id}:{old_version}"
            if old_key in self._versions:
                old_mv = self._versions[old_key]
                # Demote old version back one stage
                if target_stage == Stage.PRODUCTION:
                    old_mv.stage = Stage.STAGING.value
                elif target_stage == Stage.STAGING:
                    old_mv.stage = Stage.CANDIDATE.value

        # Apply promotion
        mv.stage = target_stage.value
        model_aliases[alias_name] = version

        event_map = {
            Stage.STAGING: EventType.PROMOTE_TO_STAGING,
            Stage.PRODUCTION: EventType.PROMOTE_TO_PRODUCTION,
            Stage.ARCHIVED: EventType.ARCHIVE_VERSION,
        }
        self._append_event(RegistryEvent(
            event_type=event_map.get(target_stage, EventType.PROMOTE_TO_STAGING).value,
            model_id=model_id,
            version=version,
            timestamp=_utc_now(),
            details={"from_stage": current_stage.value, "to_stage": target_stage.value,
                      "force": force},
        ))
        self._save()

        LOGGER.info("Promoted %s: %s → %s", key, current_stage.value, target_stage.value)

    # --- Resolution ---

    def resolve(
        self,
        model_id: str,
        *,
        version: str | None = None,
        alias: str | None = None,
    ) -> ModelVersion:
        """Resolve a model reference to a specific version.

        Either version or alias must be provided. Resolution is deterministic:
        if ambiguous, fail (never guess).
        """
        if version is not None:
            key = f"{model_id}:{version}"
            if key not in self._versions:
                raise RegistryError(f"Version {key} not found")
            return self._versions[key]

        if alias is not None:
            model_aliases = self._aliases.get(model_id, {})
            resolved_version = model_aliases.get(alias)
            if resolved_version is None:
                raise RegistryError(
                    f"Alias '{alias}' not set for model '{model_id}'. "
                    f"Available aliases: {list(model_aliases.keys())}"
                )
            key = f"{model_id}:{resolved_version}"
            if key not in self._versions:
                raise RegistryError(f"Alias '{alias}' → {key} but version not found")
            return self._versions[key]

        raise RegistryError("Must provide either version or alias")

    # --- Queries ---

    def list_versions(self, model_id: str) -> list[ModelVersion]:
        """List all versions for a model, sorted by creation time."""
        versions = [v for v in self._versions.values() if v.model_id == model_id]
        versions.sort(key=lambda v: v.created_at)
        return versions

    def list_models(self) -> list[str]:
        """List all registered model IDs."""
        return sorted(set(v.model_id for v in self._versions.values()))

    def get_aliases(self, model_id: str) -> dict[str, str]:
        """Get current alias → version mapping for a model."""
        return dict(self._aliases.get(model_id, {}))

    def get_events(self, model_id: str | None = None) -> list[RegistryEvent]:
        """Get audit log, optionally filtered by model_id."""
        if model_id is None:
            return list(self._events)
        return [e for e in self._events if e.model_id == model_id]

    def get_production_version(self, model_id: str) -> ModelVersion | None:
        """Get the current production version, or None."""
        try:
            return self.resolve(model_id, alias="production")
        except RegistryError:
            return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
