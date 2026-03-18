"""
features/feature_store.py — Immutable feature snapshot persistence.

Workflow: staging → validation → atomic commit.

Each snapshot is identified by (run_id, asof_date, config_hash) and
contains: feature matrix (parquet), feature dictionary (JSON),
manifest (JSON), and content_hash for bit-exact reproducibility.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    FeatureError,
    DataContractError,
    FeatureDef,
    PK_COLUMNS,
    content_hash,
    validate_panel_pk,
    _json_safe,
    utc_now_iso,
    get_logger,
)

LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureStoreConfig:
    """Configuration for the feature store."""
    base_dir: str = "feature_store"
    format: str = "parquet"           # "parquet" or "csv"
    compression: str = "snappy"
    validate_on_commit: bool = True
    max_null_fraction: float = 0.95   # reject features that are >95% null


# ---------------------------------------------------------------------------
# Snapshot metadata
# ---------------------------------------------------------------------------

@dataclass
class SnapshotManifest:
    """Manifest for a committed feature snapshot."""
    run_id: str
    asof_date: str
    config_hash: str
    content_hash: str
    n_rows: int
    n_features: int
    n_dates: int
    n_symbols: int
    feature_names: list[str]
    created_at: str = ""
    store_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

class FeatureStore:
    """Manages versioned, immutable feature snapshots.

    Usage
    -----
    store = FeatureStore(config)
    store.stage(df, feature_defs, run_id, asof_date, config_hash)
    manifest = store.commit()  # atomic: all-or-nothing
    df = store.load(run_id)    # read back
    latest = store.load_latest()
    """

    def __init__(self, config: FeatureStoreConfig | None = None):
        self.config = config or FeatureStoreConfig()
        self.base_dir = Path(self.config.base_dir)
        self._staged: dict[str, Any] | None = None

    # --- Staging ---

    def stage(
        self,
        df: pd.DataFrame,
        feature_defs: Sequence[FeatureDef],
        run_id: str,
        asof_date: str,
        config_hash: str,
    ) -> None:
        """Stage a feature snapshot for validation and commit."""
        self._staged = {
            "df": df.copy(),
            "feature_defs": list(feature_defs),
            "run_id": run_id,
            "asof_date": asof_date,
            "config_hash": config_hash,
        }
        LOGGER.info("Snapshot staged: run_id=%s, shape=%s", run_id, df.shape)

    # --- Validation ---

    def _validate_staged(self) -> list[str]:
        """Validate staged snapshot. Returns list of issues (empty = OK)."""
        if self._staged is None:
            return ["Nothing staged"]

        issues: list[str] = []
        df = self._staged["df"]

        # PK check
        for col in PK_COLUMNS:
            if col not in df.columns:
                issues.append(f"Missing PK column: {col}")

        if not issues:
            n_dups = df.duplicated(subset=list(PK_COLUMNS)).sum()
            if n_dups > 0:
                issues.append(f"{n_dups} duplicate PK rows")

        # Feature coverage
        feature_names = [fd.name for fd in self._staged["feature_defs"]]
        for name in feature_names:
            if name not in df.columns:
                issues.append(f"Feature '{name}' not in DataFrame")
                continue
            null_frac = df[name].isna().mean()
            if null_frac > self.config.max_null_fraction:
                issues.append(f"Feature '{name}' is {null_frac:.1%} null (>{self.config.max_null_fraction:.0%})")

        # No Inf
        numeric_cols = [c for c in feature_names if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        for col in numeric_cols:
            n_inf = np.isinf(df[col]).sum()
            if n_inf > 0:
                issues.append(f"Feature '{col}' has {n_inf} Inf values")

        return issues

    # --- Commit ---

    def commit(self) -> SnapshotManifest:
        """Validate and atomically commit the staged snapshot.

        Raises FeatureError if validation fails.
        """
        if self._staged is None:
            raise FeatureError("Nothing staged — call stage() first")

        if self.config.validate_on_commit:
            issues = self._validate_staged()
            if issues:
                self._staged = None
                raise DataContractError(
                    f"Staged snapshot failed validation: {issues}"
                )

        df = self._staged["df"]
        feature_defs = self._staged["feature_defs"]
        run_id = self._staged["run_id"]
        asof_date = self._staged["asof_date"]
        config_hash_val = self._staged["config_hash"]
        feature_names = [fd.name for fd in feature_defs]

        # Compute content hash
        chash = content_hash(df)

        # Build paths
        snapshot_dir = self.base_dir / run_id
        staging_dir = snapshot_dir.with_name(f".staging_{run_id}")

        # Write to staging first (atomic pattern)
        staging_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Feature matrix
            matrix_path = staging_dir / f"features.{self.config.format}"
            if self.config.format == "parquet":
                df.to_parquet(matrix_path, index=False, compression=self.config.compression)
            else:
                df.to_csv(matrix_path, index=False)

            # Feature dictionary
            dict_path = staging_dir / "feature_dictionary.json"
            dictionary = {
                fd.name: {
                    "family": fd.family.value,
                    "formula": fd.formula,
                    "lookback_days": fd.lookback_days,
                    "decision_lag": fd.decision_lag,
                    "source": fd.source,
                    "version": fd.version,
                }
                for fd in feature_defs
            }
            dict_path.write_text(
                json.dumps(dictionary, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            # Manifest
            manifest = SnapshotManifest(
                run_id=run_id,
                asof_date=asof_date,
                config_hash=config_hash_val,
                content_hash=chash,
                n_rows=len(df),
                n_features=len(feature_names),
                n_dates=df["date"].nunique() if "date" in df.columns else 0,
                n_symbols=df["symbol"].nunique() if "symbol" in df.columns else 0,
                feature_names=feature_names,
                created_at=utc_now_iso(),
                store_path=str(snapshot_dir),
            )
            manifest_path = staging_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )

            # Atomic move: staging → final
            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir)
            staging_dir.rename(snapshot_dir)

        except Exception:
            # Cleanup staging on failure
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            self._staged = None
            raise

        self._staged = None
        LOGGER.info(
            "Snapshot committed: run_id=%s, %d features × %d rows, hash=%s",
            run_id, manifest.n_features, manifest.n_rows, chash[:12],
        )
        return manifest

    # --- Read ---

    def load(self, run_id: str) -> pd.DataFrame:
        """Load a committed feature snapshot by run_id."""
        snapshot_dir = self.base_dir / run_id
        parquet_path = snapshot_dir / "features.parquet"
        csv_path = snapshot_dir / "features.csv"

        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(f"No snapshot found for run_id={run_id} in {self.base_dir}")

    def load_manifest(self, run_id: str) -> SnapshotManifest:
        """Load the manifest for a committed snapshot."""
        manifest_path = self.base_dir / run_id / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest for run_id={run_id}")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return SnapshotManifest(**data)

    def load_latest(self) -> pd.DataFrame:
        """Load the most recently committed snapshot."""
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Feature store not found: {self.base_dir}")
        snapshots = sorted(
            [d for d in self.base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if not snapshots:
            raise FileNotFoundError("No snapshots in feature store")
        run_id = snapshots[0].name
        return self.load(run_id)

    def list_snapshots(self) -> list[str]:
        """List all committed run_ids."""
        if not self.base_dir.exists():
            return []
        return sorted(
            d.name for d in self.base_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
