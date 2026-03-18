"""
research/alpha_discovery/alpha_library.py - Versioned alpha registry.

Maintains a formal catalog of alpha hypotheses with:
- semantic identity (`alpha_id`)
- deterministic spec identity (`spec_hash`)
- evidence linkage (`run_id`)
- lifecycle history (append-only)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from simons_smallcap_swing.research._shared import (
    config_hash,
    json_safe,
    run_id_with_prefix,
    utc_now_iso,
    write_json_safe,
    write_parquet_safe,
)


ALLOWED_FAMILIES = {
    "momentum",
    "reversal",
    "value",
    "quality",
    "microstructure",
    "event",
    "interaction",
    "regime",
    "alternative_data",
    "composite",
}

ALLOWED_STATUSES = {
    "idea",
    "registered",
    "candidate",
    "promoted",
    "deprecated",
    "rejected",
    "archived",
}


@dataclass(frozen=True)
class AlphaLibraryConfig:
    registry_version: str = "1.0"
    schema_version: str = "1.0"
    hash_method: str = "sha256"
    allowed_families: tuple[str, ...] = tuple(sorted(ALLOWED_FAMILIES))
    allowed_statuses: tuple[str, ...] = tuple(sorted(ALLOWED_STATUSES))
    semantic_fields_for_hash: tuple[str, ...] = (
        "family",
        "hypothesis_text",
        "formula_dsl",
        "inputs",
        "params",
        "normalization",
        "universe_scope",
        "intended_horizon",
        "label_definition",
        "leakage_constraints",
    )


@dataclass
class AlphaSpec:
    alpha_id: str
    name: str
    family: str
    hypothesis_text: str
    formula_dsl: Any
    inputs: Any
    params: Any
    normalization: Any
    universe_scope: str
    intended_horizon: str
    label_definition: str
    leakage_constraints: Any
    owner: str
    status: str = "registered"
    version: int = 1
    tags: list[str] = field(default_factory=list)
    economic_rationale: str | None = None
    dependencies_upstream: Any | None = None
    invalidation_conditions: Any | None = None
    notes: str | None = None


@dataclass
class EvidenceRecord:
    alpha_id: str
    spec_hash: str
    run_id: str
    ic_mean: float
    ic_std: float
    hit_rate: float
    coverage: float
    n_folds: int
    eval_start: str
    eval_end: str
    warnings: str | None = None


def _spec_store_path(root: Path) -> Path:
    return root / "specs.parquet"


def _evidence_store_path(root: Path) -> Path:
    return root / "evidence.parquet"


def _history_store_path(root: Path) -> Path:
    return root / "history.parquet"


def _read_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _spec_columns() -> list[str]:
    return [
        "alpha_id",
        "name",
        "family",
        "formula_dsl",
        "spec_hash",
        "version",
        "status",
        "owner",
        "created_at",
        "updated_at",
        "tags",
        "hypothesis_text",
        "inputs",
        "params",
        "normalization",
        "universe_scope",
        "intended_horizon",
        "label_definition",
        "leakage_constraints",
        "economic_rationale",
        "dependencies_upstream",
        "invalidation_conditions",
        "notes",
    ]


def _evidence_columns() -> list[str]:
    return [
        "alpha_id",
        "spec_hash",
        "run_id",
        "ic_mean",
        "ic_std",
        "hit_rate",
        "coverage",
        "n_folds",
        "eval_start",
        "eval_end",
        "warnings",
        "attached_at",
    ]


def _history_columns() -> list[str]:
    return [
        "alpha_id",
        "timestamp",
        "event_type",
        "actor",
        "reason",
        "old_status",
        "new_status",
        "spec_hash_before",
        "spec_hash_after",
        "related_run_id",
    ]


def _manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def _canonical_spec_payload(spec: AlphaSpec, cfg: AlphaLibraryConfig) -> dict[str, Any]:
    payload = {k: json_safe(getattr(spec, k)) for k in cfg.semantic_fields_for_hash}
    return payload


def _compute_spec_hash(spec: AlphaSpec, cfg: AlphaLibraryConfig) -> str:
    return config_hash(_canonical_spec_payload(spec, cfg), n=24)


def validate_alpha(spec: AlphaSpec, config: AlphaLibraryConfig | None = None) -> None:
    cfg = config or AlphaLibraryConfig()
    required = [
        "alpha_id",
        "name",
        "family",
        "hypothesis_text",
        "formula_dsl",
        "inputs",
        "params",
        "normalization",
        "universe_scope",
        "intended_horizon",
        "label_definition",
        "leakage_constraints",
        "owner",
        "status",
    ]
    missing = [f for f in required if getattr(spec, f, None) in (None, "", [])]
    if missing:
        raise ValueError(f"alpha spec missing required fields: {missing}")
    if spec.family not in cfg.allowed_families:
        raise ValueError(f"family '{spec.family}' not in allowed_families")
    if spec.status not in cfg.allowed_statuses:
        raise ValueError(f"status '{spec.status}' not in allowed_statuses")


def _persist_manifest(root: Path, cfg: AlphaLibraryConfig, run_id: str | None = None) -> str:
    manifest = {
        "registry_version": cfg.registry_version,
        "schema_version": cfg.schema_version,
        "config_hash": config_hash(cfg.__dict__, n=24),
        "run_id": run_id,
        "generated_at": utc_now_iso(),
    }
    return write_json_safe(manifest, _manifest_path(root))


def register_alpha(
    spec: AlphaSpec,
    *,
    storage_dir: str | Path,
    config: AlphaLibraryConfig | None = None,
    actor: str = "system",
    reason: str = "register_alpha",
    run_id: str | None = None,
) -> dict[str, Any]:
    cfg = config or AlphaLibraryConfig()
    validate_alpha(spec, cfg)
    root = Path(storage_dir)
    root.mkdir(parents=True, exist_ok=True)

    specs = _read_or_empty(_spec_store_path(root), _spec_columns())
    evidence = _read_or_empty(_evidence_store_path(root), _evidence_columns())
    history = _read_or_empty(_history_store_path(root), _history_columns())

    spec_hash = _compute_spec_hash(spec, cfg)
    now = utc_now_iso()

    existing_same = specs[(specs["alpha_id"] == spec.alpha_id) & (specs["spec_hash"] == spec_hash)]
    if len(existing_same) > 0:
        _persist_manifest(root, cfg, run_id=run_id)
        return {
            "status": "idempotent_replay",
            "alpha_id": spec.alpha_id,
            "spec_hash": spec_hash,
            "storage_dir": str(root),
        }

    row = {
        "alpha_id": spec.alpha_id,
        "name": spec.name,
        "family": spec.family,
        "formula_dsl": json_safe(spec.formula_dsl),
        "spec_hash": spec_hash,
        "version": int(spec.version),
        "status": spec.status,
        "owner": spec.owner,
        "created_at": now,
        "updated_at": now,
        "tags": json_safe(spec.tags),
        "hypothesis_text": spec.hypothesis_text,
        "inputs": json_safe(spec.inputs),
        "params": json_safe(spec.params),
        "normalization": json_safe(spec.normalization),
        "universe_scope": spec.universe_scope,
        "intended_horizon": spec.intended_horizon,
        "label_definition": spec.label_definition,
        "leakage_constraints": json_safe(spec.leakage_constraints),
        "economic_rationale": spec.economic_rationale,
        "dependencies_upstream": json_safe(spec.dependencies_upstream),
        "invalidation_conditions": json_safe(spec.invalidation_conditions),
        "notes": spec.notes,
    }
    specs = pd.concat([specs, pd.DataFrame([row])], ignore_index=True)
    history = pd.concat(
        [
            history,
            pd.DataFrame(
                [
                    {
                        "alpha_id": spec.alpha_id,
                        "timestamp": now,
                        "event_type": "register",
                        "actor": actor,
                        "reason": reason,
                        "old_status": None,
                        "new_status": spec.status,
                        "spec_hash_before": None,
                        "spec_hash_after": spec_hash,
                        "related_run_id": run_id,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    write_parquet_safe(specs, _spec_store_path(root))
    write_parquet_safe(evidence, _evidence_store_path(root))
    write_parquet_safe(history, _history_store_path(root))
    _persist_manifest(root, cfg, run_id=run_id)
    return {"status": "registered", "alpha_id": spec.alpha_id, "spec_hash": spec_hash}


def get_alpha(alpha_id: str, *, storage_dir: str | Path) -> pd.DataFrame:
    root = Path(storage_dir)
    specs = _read_or_empty(_spec_store_path(root), _spec_columns())
    return specs[specs["alpha_id"] == alpha_id].sort_values(["version", "updated_at"]).reset_index(drop=True)


def list_alphas(*, storage_dir: str | Path, filters: Mapping[str, Any] | None = None) -> pd.DataFrame:
    root = Path(storage_dir)
    specs = _read_or_empty(_spec_store_path(root), _spec_columns())
    if not filters:
        return specs
    out = specs.copy()
    for key, value in filters.items():
        if key in out.columns:
            out = out[out[key] == value]
    return out.reset_index(drop=True)


def attach_evidence(
    alpha_id: str,
    evidence: EvidenceRecord,
    *,
    storage_dir: str | Path,
    run_id: str | None = None,
    config: AlphaLibraryConfig | None = None,
) -> dict[str, Any]:
    cfg = config or AlphaLibraryConfig()
    root = Path(storage_dir)
    specs = _read_or_empty(_spec_store_path(root), _spec_columns())
    evidence_df = _read_or_empty(_evidence_store_path(root), _evidence_columns())
    history = _read_or_empty(_history_store_path(root), _history_columns())

    if len(specs[specs["alpha_id"] == alpha_id]) == 0:
        raise ValueError(f"alpha_id not found: {alpha_id}")

    row = {
        "alpha_id": alpha_id,
        "spec_hash": evidence.spec_hash,
        "run_id": evidence.run_id,
        "ic_mean": float(evidence.ic_mean),
        "ic_std": float(evidence.ic_std),
        "hit_rate": float(evidence.hit_rate),
        "coverage": float(evidence.coverage),
        "n_folds": int(evidence.n_folds),
        "eval_start": evidence.eval_start,
        "eval_end": evidence.eval_end,
        "warnings": evidence.warnings,
        "attached_at": utc_now_iso(),
    }
    dup = evidence_df[
        (evidence_df["alpha_id"] == row["alpha_id"])
        & (evidence_df["spec_hash"] == row["spec_hash"])
        & (evidence_df["run_id"] == row["run_id"])
    ]
    if len(dup) == 0:
        evidence_df = pd.concat([evidence_df, pd.DataFrame([row])], ignore_index=True)

    history = pd.concat(
        [
            history,
            pd.DataFrame(
                [
                    {
                        "alpha_id": alpha_id,
                        "timestamp": utc_now_iso(),
                        "event_type": "attach_evidence",
                        "actor": "system",
                        "reason": "attach_evidence",
                        "old_status": None,
                        "new_status": None,
                        "spec_hash_before": evidence.spec_hash,
                        "spec_hash_after": evidence.spec_hash,
                        "related_run_id": run_id or evidence.run_id,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    write_parquet_safe(specs, _spec_store_path(root))
    write_parquet_safe(evidence_df, _evidence_store_path(root))
    write_parquet_safe(history, _history_store_path(root))
    _persist_manifest(root, cfg, run_id=run_id or evidence.run_id)
    return {"status": "evidence_attached", "alpha_id": alpha_id, "run_id": evidence.run_id}


def transition_status(
    alpha_id: str,
    new_status: str,
    reason: str,
    *,
    storage_dir: str | Path,
    actor: str = "system",
    run_id: str | None = None,
    config: AlphaLibraryConfig | None = None,
) -> dict[str, Any]:
    cfg = config or AlphaLibraryConfig()
    if new_status not in cfg.allowed_statuses:
        raise ValueError(f"new_status '{new_status}' not in allowed_statuses")
    root = Path(storage_dir)
    specs = _read_or_empty(_spec_store_path(root), _spec_columns())
    history = _read_or_empty(_history_store_path(root), _history_columns())
    mask = specs["alpha_id"] == alpha_id
    if mask.sum() == 0:
        raise ValueError(f"alpha_id not found: {alpha_id}")
    idx = specs.loc[mask].sort_values(["version", "updated_at"]).index[-1]
    old_status = specs.loc[idx, "status"]
    old_hash = specs.loc[idx, "spec_hash"]
    specs.loc[idx, "status"] = new_status
    specs.loc[idx, "updated_at"] = utc_now_iso()

    history = pd.concat(
        [
            history,
            pd.DataFrame(
                [
                    {
                        "alpha_id": alpha_id,
                        "timestamp": utc_now_iso(),
                        "event_type": "status_changed",
                        "actor": actor,
                        "reason": reason,
                        "old_status": old_status,
                        "new_status": new_status,
                        "spec_hash_before": old_hash,
                        "spec_hash_after": old_hash,
                        "related_run_id": run_id,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    write_parquet_safe(specs, _spec_store_path(root))
    write_parquet_safe(history, _history_store_path(root))
    _persist_manifest(root, cfg, run_id=run_id)
    return {"status": "status_changed", "alpha_id": alpha_id, "old_status": old_status, "new_status": new_status}


def update_alpha(
    alpha_id: str,
    patch: Mapping[str, Any],
    *,
    storage_dir: str | Path,
    actor: str = "system",
    run_id: str | None = None,
    config: AlphaLibraryConfig | None = None,
) -> dict[str, Any]:
    cfg = config or AlphaLibraryConfig()
    root = Path(storage_dir)
    specs = _read_or_empty(_spec_store_path(root), _spec_columns())
    history = _read_or_empty(_history_store_path(root), _history_columns())
    cur = specs[specs["alpha_id"] == alpha_id].sort_values(["version", "updated_at"])
    if len(cur) == 0:
        raise ValueError(f"alpha_id not found: {alpha_id}")
    prev = cur.iloc[-1].to_dict()
    old_hash = str(prev["spec_hash"])
    old_status = str(prev["status"])

    merged = {**prev, **{k: json_safe(v) for k, v in patch.items()}}
    new_version = int(prev.get("version", 1)) + 1
    merged["version"] = new_version
    merged["updated_at"] = utc_now_iso()
    merged["created_at"] = prev.get("created_at", utc_now_iso())

    spec_obj = AlphaSpec(
        alpha_id=str(merged["alpha_id"]),
        name=str(merged["name"]),
        family=str(merged["family"]),
        hypothesis_text=str(merged["hypothesis_text"]),
        formula_dsl=merged["formula_dsl"],
        inputs=merged["inputs"],
        params=merged["params"],
        normalization=merged["normalization"],
        universe_scope=str(merged["universe_scope"]),
        intended_horizon=str(merged["intended_horizon"]),
        label_definition=str(merged["label_definition"]),
        leakage_constraints=merged["leakage_constraints"],
        owner=str(merged["owner"]),
        status=str(merged.get("status", old_status)),
        version=new_version,
        tags=list(merged.get("tags", [])) if isinstance(merged.get("tags", []), list) else [],
        economic_rationale=merged.get("economic_rationale"),
        dependencies_upstream=merged.get("dependencies_upstream"),
        invalidation_conditions=merged.get("invalidation_conditions"),
        notes=merged.get("notes"),
    )
    validate_alpha(spec_obj, cfg)
    new_hash = _compute_spec_hash(spec_obj, cfg)
    merged["spec_hash"] = new_hash

    specs = pd.concat([specs, pd.DataFrame([merged], columns=_spec_columns())], ignore_index=True)
    history = pd.concat(
        [
            history,
            pd.DataFrame(
                [
                    {
                        "alpha_id": alpha_id,
                        "timestamp": utc_now_iso(),
                        "event_type": "spec_updated",
                        "actor": actor,
                        "reason": "update_alpha",
                        "old_status": old_status,
                        "new_status": merged["status"],
                        "spec_hash_before": old_hash,
                        "spec_hash_after": new_hash,
                        "related_run_id": run_id,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    write_parquet_safe(specs, _spec_store_path(root))
    write_parquet_safe(history, _history_store_path(root))
    _persist_manifest(root, cfg, run_id=run_id)
    return {"status": "updated", "alpha_id": alpha_id, "version": new_version, "spec_hash": new_hash}


def deprecate_alpha(
    alpha_id: str,
    cause: str,
    *,
    storage_dir: str | Path,
    actor: str = "system",
    run_id: str | None = None,
    config: AlphaLibraryConfig | None = None,
) -> dict[str, Any]:
    return transition_status(
        alpha_id=alpha_id,
        new_status="deprecated",
        reason=cause,
        storage_dir=storage_dir,
        actor=actor,
        run_id=run_id,
        config=config,
    )


__all__ = [
    "AlphaSpec",
    "EvidenceRecord",
    "AlphaLibraryConfig",
    "validate_alpha",
    "register_alpha",
    "get_alpha",
    "list_alphas",
    "update_alpha",
    "attach_evidence",
    "transition_status",
    "deprecate_alpha",
]
