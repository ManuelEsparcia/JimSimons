"""
models.inference — Model registry, prediction, and explainability.

Pipeline: model_registry (resolve) → predict (score) → explain (attribute)
"""
from __future__ import annotations
import enum


class Stage(str, enum.Enum):
    CANDIDATE = "candidate"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class EventType(str, enum.Enum):
    REGISTER_VERSION = "REGISTER_VERSION"
    PROMOTE_TO_STAGING = "PROMOTE_TO_STAGING"
    PROMOTE_TO_PRODUCTION = "PROMOTE_TO_PRODUCTION"
    ROLLBACK_PRODUCTION = "ROLLBACK_PRODUCTION"
    ARCHIVE_VERSION = "ARCHIVE_VERSION"


VALID_TRANSITIONS: dict[Stage, tuple[Stage, ...]] = {
    Stage.CANDIDATE: (Stage.STAGING, Stage.ARCHIVED),
    Stage.STAGING: (Stage.PRODUCTION, Stage.CANDIDATE, Stage.ARCHIVED),
    Stage.PRODUCTION: (Stage.STAGING, Stage.ARCHIVED),
    Stage.ARCHIVED: (),
}


class InferenceError(RuntimeError):
    """Base error for inference subpackage."""

class RegistryError(InferenceError):
    """Model registry error."""

class PredictionError(InferenceError):
    """Prediction pipeline error."""

class ExplainError(InferenceError):
    """Explainability error."""

class FeatureCompatibilityError(PredictionError):
    """Feature schema mismatch between train and inference."""
