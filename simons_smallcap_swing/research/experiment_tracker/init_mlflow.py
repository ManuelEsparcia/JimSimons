"""
experiment_tracker/init_mlflow.py - Deterministic MLflow bootstrap.

Initializes tracking backend connectivity, resolves/creates the target experiment,
and persists local evidence artifacts (`init_summary.json`, `manifest.json`) to
guarantee reproducibility of research tracking context.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from simons_smallcap_swing.research._shared import (
    config_hash,
    json_safe,
    run_id_with_prefix,
    utc_now_iso,
    write_json_safe,
)


class InitMlflowError(RuntimeError):
    pass


class ConfigError(InitMlflowError, ValueError):
    pass


class CredentialError(InitMlflowError):
    pass


class BackendConnectionError(InitMlflowError):
    pass


class ExperimentResolutionError(InitMlflowError):
    pass


class ManifestWriteError(InitMlflowError):
    pass


class RecoveryError(InitMlflowError):
    pass


@dataclass(frozen=True)
class InitConfig:
    tracking_uri: str | None = None
    artifact_location: str | None = None
    experiment_name: str = "simons_smallcap_swing"
    schema_version: str = "1.0"
    timeout_seconds: int = 20
    max_retries: int = 2
    required_tags: tuple[str, ...] = ("git_commit", "config_hash", "schema_version")


@dataclass(frozen=True)
class InitContext:
    project_name: str
    module_name: str
    git_commit: str
    config_hash: str
    data_snapshot_hash: str | None = None


@dataclass
class InitResult:
    status: str
    tracking_uri_sanitized: str
    artifact_location_sanitized: str | None
    experiment_name: str
    experiment_id: str | None
    manifest_path: str
    summary_path: str
    retry_count: int
    latency_ms: int


def _sanitize_uri(uri: str | None) -> str | None:
    if uri is None:
        return None
    sanitized = re.sub(r"(https?://)([^/@:]+):([^/@]+)@", r"\1***:***@", str(uri))
    sanitized = re.sub(r"([?&](token|access_token|password|pwd)=)[^&]+", r"\1***", sanitized, flags=re.I)
    return sanitized


def _resolve_tracking_uri(config: InitConfig, override: str | None = None) -> str:
    if override:
        return override
    if config.tracking_uri:
        return config.tracking_uri
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    raise ConfigError("tracking_uri could not be resolved (override/config/env missing)")


def _required_tag_map(config: InitConfig, context: InitContext) -> dict[str, str]:
    tag_values = {
        "git_commit": context.git_commit,
        "config_hash": context.config_hash,
        "schema_version": config.schema_version,
        "project_name": context.project_name,
        "module_name": context.module_name,
    }
    if context.data_snapshot_hash:
        tag_values["data_snapshot_hash"] = context.data_snapshot_hash

    missing = [k for k in config.required_tags if not tag_values.get(k)]
    if missing:
        raise ConfigError(f"required tags missing or empty: {missing}")
    return {k: str(v) for k, v in tag_values.items() if k in config.required_tags}


def _build_init_key(
    tracking_uri_sanitized: str,
    experiment_name: str,
    context: InitContext,
    config: InitConfig,
) -> str:
    payload = {
        "tracking_uri_sanitized": tracking_uri_sanitized,
        "experiment_name": experiment_name,
        "context_config_hash": context.config_hash,
        "git_commit": context.git_commit,
        "schema_version": config.schema_version,
    }
    return config_hash(payload, n=24)


def _resolve_or_create_experiment(
    mlflow_mod: Any,
    experiment_name: str,
    artifact_location: str | None,
) -> str:
    try:
        client = mlflow_mod.tracking.MlflowClient()
    except Exception as exc:
        raise BackendConnectionError(f"cannot create MlflowClient: {exc}") from exc

    try:
        existing = client.get_experiment_by_name(experiment_name)
        if existing is not None:
            return str(existing.experiment_id)
        experiment_id = client.create_experiment(experiment_name, artifact_location=artifact_location)
        return str(experiment_id)
    except Exception as exc:
        raise ExperimentResolutionError(
            f"failed to resolve/create experiment '{experiment_name}': {exc}"
        ) from exc


def init_mlflow(
    config: InitConfig,
    context: InitContext,
    *,
    output_dir: str | Path,
    run_id: str | None = None,
    tracking_uri_override: str | None = None,
) -> InitResult:
    """
    Bootstrap MLflow tracking context with deterministic evidence artifacts.
    """
    rid = run_id or run_id_with_prefix("initmlf")
    start = time.perf_counter()
    retries = 0

    tracking_uri = _resolve_tracking_uri(config, override=tracking_uri_override)
    tracking_uri_s = _sanitize_uri(tracking_uri) or ""
    artifact_location_s = _sanitize_uri(config.artifact_location)
    required_tags = _required_tag_map(config, context)
    init_key = _build_init_key(tracking_uri_s, config.experiment_name, context, config)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "init_summary.json"
    manifest_path = out / "manifest.json"

    experiment_id: str | None = None
    status = "failed"
    failure: str | None = None

    for attempt in range(config.max_retries + 1):
        retries = attempt
        try:
            try:
                import mlflow  # type: ignore
            except Exception as exc:
                raise BackendConnectionError(
                    "mlflow package is not installed or cannot be imported"
                ) from exc

            mlflow.set_tracking_uri(tracking_uri)
            experiment_id = _resolve_or_create_experiment(mlflow, config.experiment_name, config.artifact_location)
            status = "ready"
            failure = None
            break
        except (BackendConnectionError, ExperimentResolutionError) as exc:
            failure = str(exc)
            status = "failed"
            # Retry only transient backend errors
            if attempt < config.max_retries:
                time.sleep(min(1.0 * (attempt + 1), 3.0))
            else:
                break

    latency_ms = int((time.perf_counter() - start) * 1000)
    resolved_at = utc_now_iso()

    summary = {
        "run_id": rid,
        "status": status,
        "tracking_uri_sanitized": tracking_uri_s,
        "artifact_location_sanitized": artifact_location_s,
        "experiment_name": config.experiment_name,
        "experiment_id": experiment_id,
        "schema_version": config.schema_version,
        "resolved_at_utc": resolved_at,
        "latency_ms": latency_ms,
        "retry_count": retries,
        "failure_reason": failure,
    }
    manifest = {
        "run_id": rid,
        "project_name": context.project_name,
        "module_name": context.module_name,
        "git_commit": context.git_commit,
        "config_hash": context.config_hash,
        "data_snapshot_hash": context.data_snapshot_hash,
        "required_tags": required_tags,
        "experiment_name": config.experiment_name,
        "experiment_id": experiment_id,
        "init_key": init_key,
        "tracking_uri_sanitized": tracking_uri_s,
        "artifact_location_sanitized": artifact_location_s,
        "generated_at": resolved_at,
    }

    try:
        write_json_safe(summary, summary_path)
        write_json_safe(manifest, manifest_path)
    except Exception as exc:
        raise ManifestWriteError(f"failed writing init artifacts: {exc}") from exc

    if status != "ready":
        raise RecoveryError(
            f"MLflow init finished with status={status}. "
            f"Check summary at {summary_path}"
        )

    return InitResult(
        status=status,
        tracking_uri_sanitized=tracking_uri_s,
        artifact_location_sanitized=artifact_location_s,
        experiment_name=config.experiment_name,
        experiment_id=experiment_id,
        manifest_path=str(manifest_path),
        summary_path=str(summary_path),
        retry_count=retries,
        latency_ms=latency_ms,
    )


__all__ = [
    "ConfigError",
    "CredentialError",
    "BackendConnectionError",
    "ExperimentResolutionError",
    "ManifestWriteError",
    "RecoveryError",
    "InitConfig",
    "InitContext",
    "InitResult",
    "init_mlflow",
]
