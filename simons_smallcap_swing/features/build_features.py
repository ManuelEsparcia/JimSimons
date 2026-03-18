"""
features/build_features.py — Canonical feature matrix construction.

Orchestrates all feature families into a single PIT-strict, QC-validated,
immutable feature snapshot. This is the ONLY entry point for producing
the feature matrix consumed by models, labels, and backtests.

Pipeline:
    1. Load prices, universe, fundamentals (PIT)
    2. Build microstructure features (OHLCV proxies)
    3. Build fundamental level features (if available)
    4. Build fundamental delta features (if available)
    5. Apply cross-sectional transforms (rank, z-score, winsor)
    6. Build interactions (if enabled)
    7. Run feature QC
    8. Commit to feature store
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    FeatureError,
    DataContractError,
    FeatureDef,
    FeatureFamily,
    PK_COLUMNS,
    validate_panel_pk,
    content_hash,
    _json_safe,
    utc_now_iso,
    get_logger,
)
from .microstructure import (
    MicrostructureConfig,
    build_microstructure_features,
)
from .cross_sectional import (
    CrossSectionalConfig,
    transform_panel,
)
from .feature_qc import (
    FeatureQCConfig,
    run_feature_qc,
)
from .feature_store import (
    FeatureStore,
    FeatureStoreConfig,
    SnapshotManifest,
)
from .fundamentals_core import (
    FundamentalsConfig,
    build_fundamental_features,
)
from .fundamentals_deltas import (
    DeltaConfig,
    build_fundamental_deltas,
)
from .interactions import (
    InteractionConfig,
    build_interactions,
)

LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BuildFeaturesConfig:
    """Master configuration for feature matrix construction."""
    # Sub-configs
    microstructure: MicrostructureConfig = field(default_factory=MicrostructureConfig)
    cross_sectional: CrossSectionalConfig = field(default_factory=CrossSectionalConfig)
    fundamentals: FundamentalsConfig = field(default_factory=FundamentalsConfig)
    deltas: DeltaConfig = field(default_factory=DeltaConfig)
    interactions: InteractionConfig = field(default_factory=InteractionConfig)
    qc: FeatureQCConfig = field(default_factory=FeatureQCConfig)
    store: FeatureStoreConfig = field(default_factory=FeatureStoreConfig)

    # Pipeline control
    enable_microstructure: bool = True
    enable_fundamentals: bool = False   # requires EDGAR PIT data
    enable_deltas: bool = False         # requires fundamentals
    enable_interactions: bool = False   # optional for MVP
    enable_qc: bool = True
    enable_store: bool = True

    # Cross-sectional transform to apply
    cs_transform: str = "cs_zscore_robust"  # or "cs_rank", "cs_rank_gauss", etc.
    cs_winsor_first: bool = True            # winsorize before z-score

    # Run metadata
    run_id: str = ""
    asof_date: str = ""

    def config_hash(self) -> str:
        blob = json.dumps(_json_safe(asdict(self)), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Panel merging
# ---------------------------------------------------------------------------

def _merge_feature_panels(
    panels: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Merge multiple feature panels on (date, symbol).

    All panels must have 'date' and 'symbol' columns. Features from
    later panels overwrite earlier ones if names collide.
    """
    if not panels:
        raise FeatureError("No feature panels to merge")

    base = panels[0].copy()
    for panel in panels[1:]:
        # Only merge feature columns (not date/symbol again)
        merge_cols = ["date", "symbol"] + [
            c for c in panel.columns if c not in ("date", "symbol")
        ]
        base = base.merge(
            panel[merge_cols],
            on=["date", "symbol"],
            how="outer",
            suffixes=("", "_dup"),
        )
        # Drop duplicate columns
        dup_cols = [c for c in base.columns if c.endswith("_dup")]
        base.drop(columns=dup_cols, inplace=True)

    return base


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_features(
    prices: pd.DataFrame,
    *,
    universe: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    price_mcap: pd.DataFrame | None = None,
    config: BuildFeaturesConfig | None = None,
) -> dict[str, Any]:
    """Build the canonical feature matrix.

    Parameters
    ----------
    prices : DataFrame
        Adjusted OHLCV panel with (date, symbol, open, high, low, close, volume).
    universe : DataFrame, optional
        Universe panel with (date, symbol, eligible_flag). If provided,
        features are NaN for non-eligible rows.
    fundamentals : DataFrame, optional
        PIT fundamentals panel from EDGAR. Required if enable_fundamentals=True.
    price_mcap : DataFrame, optional
        Market cap panel for valuation features.
    config : BuildFeaturesConfig, optional

    Returns
    -------
    dict with keys:
        features_df : DataFrame — the final feature matrix
        feature_defs : list[FeatureDef] — metadata catalog
        qc_records : list[dict] — QC results (if enabled)
        qc_summary : dict — QC summary
        manifest : SnapshotManifest | None — store manifest (if enabled)
    """
    cfg = config or BuildFeaturesConfig()
    if not cfg.run_id:
        cfg.run_id = f"features_{utc_now_iso().replace(':', '').replace('-', '')}"
    if not cfg.asof_date:
        cfg.asof_date = utc_now_iso()[:10]

    LOGGER.info("Building features: run_id=%s", cfg.run_id)

    # --- Universe mask ---
    universe_mask = None
    if universe is not None:
        eligible_col = None
        for c in ("eligible_flag", "is_eligible", "eligible"):
            if c in universe.columns:
                eligible_col = c
                break
        if eligible_col:
            # Merge on (date, symbol) to get mask aligned with prices
            mask_df = universe[["date", "symbol", eligible_col]].copy()
            mask_df = mask_df.rename(columns={eligible_col: "_eligible"})
            merged = prices.merge(mask_df, on=["date", "symbol"], how="left")
            universe_mask = merged["_eligible"].fillna(False).astype(bool)

    # --- Step 1: Microstructure features ---
    panels: list[pd.DataFrame] = []
    all_defs: list[FeatureDef] = []

    if cfg.enable_microstructure:
        micro_df, micro_defs = build_microstructure_features(
            prices, config=cfg.microstructure, universe_mask=universe_mask,
        )
        panels.append(micro_df)
        all_defs.extend(micro_defs)

    # --- Step 2: Fundamental level features ---
    if cfg.enable_fundamentals and fundamentals is not None:
        fund_df, fund_defs = build_fundamental_features(
            fundamentals, config=cfg.fundamentals, price_mcap=price_mcap,
        )
        panels.append(fund_df)
        all_defs.extend(fund_defs)

        # --- Step 3: Fundamental deltas ---
        if cfg.enable_deltas:
            delta_df, delta_defs = build_fundamental_deltas(
                fundamentals, config=cfg.deltas,
            )
            panels.append(delta_df)
            all_defs.extend(delta_defs)

    # --- Step 4: Merge all panels ---
    if not panels:
        raise FeatureError("No feature panels were produced (check config)")

    features_df = _merge_feature_panels(panels)

    # --- Step 5: Cross-sectional transforms ---
    feature_cols = [c for c in features_df.columns if c not in ("date", "symbol")]

    # Optional winsorization first
    if cfg.cs_winsor_first and cfg.cs_transform != "cs_winsor":
        features_df = transform_panel(
            features_df, feature_cols,
            transform="cs_winsor",
            config=cfg.cross_sectional,
        )

    # Main transform
    features_df = transform_panel(
        features_df, feature_cols,
        transform=cfg.cs_transform,
        config=cfg.cross_sectional,
    )

    # --- Step 6: Interactions (on transformed features) ---
    if cfg.enable_interactions:
        inter_df, inter_defs = build_interactions(
            features_df, config=cfg.interactions,
        )
        # Merge interaction columns
        inter_cols = [c for c in inter_df.columns if c not in ("date", "symbol")]
        if inter_cols:
            features_df = features_df.merge(
                inter_df[["date", "symbol"] + inter_cols],
                on=["date", "symbol"],
                how="left",
            )
            all_defs.extend(inter_defs)
            feature_cols.extend(inter_cols)

    # --- Final cleanup ---
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Update feature_cols after all merges
    feature_cols = [c for c in features_df.columns if c not in ("date", "symbol")]

    # --- Step 7: QC ---
    qc_records_list: list[dict] = []
    qc_summary: dict[str, Any] = {}
    if cfg.enable_qc:
        records, summary = run_feature_qc(
            features_df, feature_cols, config=cfg.qc,
        )
        qc_records_list = [r.to_dict() for r in records]
        qc_summary = summary

        if summary.get("overall_status") == "FAIL":
            LOGGER.warning("Feature QC FAILED: %s", summary)

    # --- Step 8: Store ---
    manifest: SnapshotManifest | None = None
    if cfg.enable_store:
        store = FeatureStore(cfg.store)
        store.stage(
            features_df, all_defs,
            run_id=cfg.run_id,
            asof_date=cfg.asof_date,
            config_hash=cfg.config_hash(),
        )
        try:
            manifest = store.commit()
        except Exception as e:
            LOGGER.error("Feature store commit failed: %s", e)

    # --- Result ---
    result = {
        "features_df": features_df,
        "feature_defs": all_defs,
        "feature_cols": feature_cols,
        "qc_records": qc_records_list,
        "qc_summary": qc_summary,
        "manifest": manifest,
        "run_id": cfg.run_id,
        "config_hash": cfg.config_hash(),
        "n_rows": len(features_df),
        "n_features": len(feature_cols),
        "coverage": float(features_df[feature_cols].notna().mean().mean()),
    }

    LOGGER.info(
        "Feature build complete: %d features × %d rows, coverage=%.1f%%, QC=%s",
        len(feature_cols), len(features_df),
        result["coverage"] * 100,
        qc_summary.get("overall_status", "SKIP"),
    )

    return result
