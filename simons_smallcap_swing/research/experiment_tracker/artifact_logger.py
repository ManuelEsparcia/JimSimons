"""
experiment_tracker/artifact_logger.py - Deterministic artifact publication.

Publishes run artifacts with content hashing, logical/physical versioning,
append-only lineage, operational journal, and idempotent replay semantics.
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from simons_smallcap_swing.research._shared import (
    append_jsonl,
    config_hash,
    json_safe,
    run_id_with_prefix,
    sha256_file,
    utc_now_iso,
    write_json_safe,
    write_parquet_safe,
)


class ArtifactLoggerError(RuntimeError):
    pass


class ValidationError(ArtifactLoggerError, ValueError):
    pass


class ConflictError(ArtifactLoggerError):
    pass


class UploadError(ArtifactLoggerError):
    pass


class CommitError(ArtifactLoggerError):
    pass


class RecoveryError(ArtifactLoggerError):
    pass


@dataclass(frozen=True)
class ArtifactSpec:
    logical_name: str
    kind: str
    local_path: str
    producer_step: str
    logical_version: str = "v1"
    schema_version: str = "1.0"
    compression: str = "none"


@dataclass
class LoggedArtifact:
    artifact_id: str
    run_id: str
    logical_name: str
    kind: str
    producer_step: str
    logical_version: str
    physical_version: str
    content_hash: str
    content_size: int
    compression: str
    local_path: str
    store_uri: str
    schema_version: str
    status: str
    created_at: str


DEFAULT_ALLOWED_KINDS = {
    "config_snapshot",
    "data_snapshot",
    "feature_frame",
    "label_frame",
    "trained_model",
    "prediction_frame",
    "backtest_report",
    "risk_report",
    "diagnostic_plot",
    "environment_snapshot",
}


def _lineage_columns() -> list[str]:
    return [
        "artifact_id",
        "run_id",
        "logical_name",
        "kind",
        "producer_step",
        "logical_version",
        "physical_version",
        "content_hash",
        "content_size",
        "compression",
        "local_path",
        "store_uri",
        "schema_version",
        "status",
        "created_at",
    ]


def _read_lineage(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=_lineage_columns())
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    try:
        return pd.read_parquet(path)
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame(columns=_lineage_columns())


def _validate_specs(specs: Sequence[ArtifactSpec], allowed_kinds: set[str]) -> None:
    if not specs:
        raise ValidationError("artifact manifest is empty")
    seen = set()
    for s in specs:
        if not s.logical_name:
            raise ValidationError("logical_name cannot be empty")
        if s.kind not in allowed_kinds:
            raise ValidationError(f"kind not allowed: {s.kind}")
        p = Path(s.local_path)
        if not p.exists() or not p.is_file():
            raise ValidationError(f"local_path does not exist: {s.local_path}")
        key = (s.logical_name, s.logical_version)
        if key in seen:
            raise ValidationError(f"duplicate logical key in same call: {key}")
        seen.add(key)


def log_artifacts(
    specs: Sequence[ArtifactSpec],
    *,
    run_id: str | None = None,
    output_dir: str | Path,
    store_root: str | Path,
    max_artifacts_per_call: int = 500,
    max_bytes_per_artifact: int = 1_500_000_000,
    allowed_kinds: set[str] | None = None,
) -> list[LoggedArtifact]:
    """
    Publish artifacts and persist:
    - artifact_summary.json
    - lineage.parquet
    - artifact_journal.jsonl
    """
    rid = run_id or run_id_with_prefix("artlog")
    kinds = allowed_kinds or set(DEFAULT_ALLOWED_KINDS)
    if len(specs) > max_artifacts_per_call:
        raise ValidationError(
            f"too many artifacts in one call: {len(specs)} > {max_artifacts_per_call}"
        )
    _validate_specs(specs, kinds)

    started = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    journal_path = out / "artifact_journal.jsonl"
    lineage_path = out / "lineage.parquet"
    summary_path = out / "artifact_summary.json"
    store = Path(store_root)
    store.mkdir(parents=True, exist_ok=True)

    existing = _read_lineage(lineage_path)
    logged: list[LoggedArtifact] = []
    n_uploaded = 0
    n_dedup = 0
    n_failed = 0
    total_bytes = 0

    for spec in specs:
        now = utc_now_iso()
        local = Path(spec.local_path)
        content_size = int(local.stat().st_size)
        if content_size > max_bytes_per_artifact:
            append_jsonl(
                journal_path,
                {
                    "run_id": rid,
                    "event_time": now,
                    "artifact_id": None,
                    "phase": "validate",
                    "result": "failed",
                    "status": "invalid",
                    "details": f"too_large:{content_size}",
                },
            )
            n_failed += 1
            continue

        content_hash = sha256_file(local)
        logical_key_mask = (
            (existing.get("run_id", pd.Series(dtype=object)) == rid)
            & (existing.get("logical_name", pd.Series(dtype=object)) == spec.logical_name)
            & (existing.get("logical_version", pd.Series(dtype=object)) == spec.logical_version)
        )
        logical_matches = existing.loc[logical_key_mask] if len(existing) else pd.DataFrame()

        if len(logical_matches) > 0:
            prior_hashes = set(logical_matches["content_hash"].astype(str))
            if content_hash not in prior_hashes:
                raise ConflictError(
                    f"logical key collision with different content hash: "
                    f"{spec.logical_name}@{spec.logical_version}"
                )
            # Idempotent replay
            row = logical_matches.iloc[0].to_dict()
            logged.append(LoggedArtifact(**{k: row[k] for k in _lineage_columns()}))
            n_dedup += 1
            append_jsonl(
                journal_path,
                {
                    "run_id": rid,
                    "event_time": now,
                    "artifact_id": row["artifact_id"],
                    "phase": "replay",
                    "result": "deduplicated",
                    "status": "committed",
                    "details": "idempotent_replay",
                },
            )
            continue

        artifact_id = config_hash(
            {
                "run_id": rid,
                "logical_name": spec.logical_name,
                "logical_version": spec.logical_version,
                "content_hash": content_hash,
            },
            n=24,
        )
        physical_version = content_hash[:12]
        ext = local.suffix if local.suffix else ".bin"
        store_rel = Path(spec.kind) / content_hash[:2] / f"{content_hash}{ext}"
        store_path = store / store_rel
        store_path.parent.mkdir(parents=True, exist_ok=True)

        append_jsonl(
            journal_path,
            {
                "run_id": rid,
                "event_time": now,
                "artifact_id": artifact_id,
                "phase": "upload",
                "result": "started",
                "status": "pending",
                "details": str(local),
            },
        )

        try:
            if not store_path.exists():
                shutil.copy2(local, store_path)
                n_uploaded += 1
            else:
                n_dedup += 1
        except Exception as exc:
            n_failed += 1
            append_jsonl(
                journal_path,
                {
                    "run_id": rid,
                    "event_time": utc_now_iso(),
                    "artifact_id": artifact_id,
                    "phase": "upload",
                    "result": "failed",
                    "status": "failed",
                    "details": str(exc),
                },
            )
            continue

        total_bytes += content_size
        row = LoggedArtifact(
            artifact_id=artifact_id,
            run_id=rid,
            logical_name=spec.logical_name,
            kind=spec.kind,
            producer_step=spec.producer_step,
            logical_version=spec.logical_version,
            physical_version=physical_version,
            content_hash=content_hash,
            content_size=content_size,
            compression=spec.compression,
            local_path=str(local),
            store_uri=f"file://{store_path}",
            schema_version=spec.schema_version,
            status="committed",
            created_at=utc_now_iso(),
        )
        logged.append(row)
        append_jsonl(
            journal_path,
            {
                "run_id": rid,
                "event_time": row.created_at,
                "artifact_id": artifact_id,
                "phase": "commit",
                "result": "success",
                "status": "committed",
                "details": row.store_uri,
            },
        )

    # Append-only lineage
    new_rows = pd.DataFrame([json_safe(a.__dict__) for a in logged], columns=_lineage_columns())
    combined = pd.concat([existing, new_rows], ignore_index=True)
    # De-dup exact rows to keep replay idempotent
    combined = combined.drop_duplicates(
        subset=["run_id", "logical_name", "logical_version", "content_hash", "store_uri"],
        keep="first",
    )
    write_parquet_safe(combined, lineage_path)

    elapsed = time.perf_counter() - started
    summary = {
        "run_id": rid,
        "n_requested": len(specs),
        "n_uploaded": n_uploaded,
        "n_deduplicated": n_dedup,
        "n_committed": len(logged),
        "n_failed": n_failed,
        "total_bytes": int(total_bytes),
        "wall_clock_seconds": round(float(elapsed), 4),
        "lineage_path": str(lineage_path),
        "journal_path": str(journal_path),
    }
    write_json_safe(summary, summary_path)

    if n_failed > 0 and len(logged) == 0:
        raise RecoveryError(
            f"all artifact publications failed for run_id={rid}. "
            f"See journal: {journal_path}"
        )

    return logged


__all__ = [
    "ArtifactSpec",
    "LoggedArtifact",
    "ValidationError",
    "ConflictError",
    "UploadError",
    "CommitError",
    "RecoveryError",
    "log_artifacts",
]
