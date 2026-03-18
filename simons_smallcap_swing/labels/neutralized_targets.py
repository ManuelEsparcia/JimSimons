"""
labels/neutralized_targets.py — Cross-sectional target neutralization.

Removes systematic exposure (sector, beta, size, liquidity) from forward
returns to isolate the idiosyncratic component for supervised learning.

Estimator: weighted ridge regression per (date, horizon).
    β_hat = argmin (y - Xβ)'W(y - Xβ) + λ||β||²
    y_neut = y - Xβ_hat  (residual)

Default canonical exposures:
    intercept + sector dummies + market_beta + log(mcap) + liquidity

Failure policy: invalidate_derived_target (y_base survives, y_neut → NaN).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    NeutFailure, Severity,
    utc_now_iso, config_hash, json_safe,
    read_dataframe, write_parquet_safe, write_json_safe,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NeutralizationConfig:
    input_col_prefix: str = "y_fwd_ret_net"
    horizons: tuple[int, ...] = (5, 10, 20)
    estimator: str = "weighted_ridge"
    ridge_lambda: float = 1.0
    min_obs_per_date: int = 50
    cond_max: float = 1000.0
    residual_var_min: float = 1e-8
    exposure_cols: tuple[str, ...] = ("sector", "market_beta", "log_mcap", "liquidity")
    failure_policy: str = "invalidate_derived_target"


def _weighted_ridge(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, lmbda: float,
) -> tuple[np.ndarray, float]:
    """Solve weighted ridge: (X'WX + λI)β = X'Wy."""
    W = np.diag(w)
    XtWX = X.T @ W @ X + lmbda * np.eye(X.shape[1])
    XtWy = X.T @ W @ y
    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        return np.full(X.shape[1], np.nan), np.nan
    residual = y - X @ beta
    r2 = 1 - np.sum(residual**2) / max(np.sum((y - y.mean())**2), 1e-15)
    return beta, float(r2)


def neutralize_one_group(
    y: np.ndarray,
    X: np.ndarray,
    config: NeutralizationConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Neutralize a single (date, horizon) cross-section."""
    n, k = X.shape
    diag: dict[str, Any] = {"n_obs": n, "n_features": k}

    # Check minimum obs
    if n < config.min_obs_per_date:
        diag["failure"] = NeutFailure.N_OBS_TOO_SMALL.value
        return np.full(n, np.nan), diag

    # Check condition number
    try:
        cond = np.linalg.cond(X.T @ X)
    except Exception:
        cond = np.inf
    diag["cond_number"] = float(cond) if np.isfinite(cond) else 1e12

    if cond > config.cond_max:
        diag["failure"] = NeutFailure.CONDITION_NUMBER_TOO_HIGH.value
        return np.full(n, np.nan), diag

    # Fit
    w = np.ones(n)
    beta, r2 = _weighted_ridge(X, y, w, config.ridge_lambda)
    diag["r2"] = round(r2, 6)
    diag["lambda_used"] = config.ridge_lambda

    if np.any(np.isnan(beta)):
        diag["failure"] = NeutFailure.ESTIMATOR_FAILED.value
        return np.full(n, np.nan), diag

    residual = y - X @ beta
    res_var = float(np.var(residual))
    diag["residual_var"] = round(res_var, 8)

    if res_var < config.residual_var_min:
        diag["failure"] = NeutFailure.RESIDUAL_VARIANCE_COLLAPSED.value
        return np.full(n, np.nan), diag

    diag["failure"] = None
    return residual, diag


def run_neutralized_targets(
    labels: pd.DataFrame | str | Path,
    exposures: pd.DataFrame | str | Path,
    *,
    config: NeutralizationConfig | None = None,
    output_dir: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run cross-sectional neutralization per (date, horizon)."""
    cfg = config or NeutralizationConfig()
    if not run_id:
        run_id = f"neut_{utc_now_iso().replace(':','').replace('-','')}"

    if isinstance(labels, (str, Path)):
        labels = read_dataframe(labels)
    if isinstance(exposures, (str, Path)):
        exposures = read_dataframe(exposures)

    labels = labels.copy()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce")
    exposures["date"] = pd.to_datetime(exposures["date"], errors="coerce")

    # Merge
    merged = labels.merge(exposures, on=["date", "symbol"], how="left", suffixes=("", "_exp"))

    # Build design matrix columns
    exp_cols = [c for c in cfg.exposure_cols if c in merged.columns]

    # Handle sector dummies
    design_cols = []
    if "sector" in exp_cols:
        dummies = pd.get_dummies(merged["sector"], prefix="sector", drop_first=True)
        merged = pd.concat([merged, dummies], axis=1)
        design_cols.extend(dummies.columns.tolist())
        exp_cols = [c for c in exp_cols if c != "sector"]
    design_cols.extend(exp_cols)

    all_diags = []
    for h in cfg.horizons:
        src_col = f"{cfg.input_col_prefix}_{h}d"
        dst_col = f"y_neut_{h}d"

        if src_col not in merged.columns:
            merged[dst_col] = np.nan
            continue

        merged[dst_col] = np.nan

        for date, grp in merged.groupby("date"):
            valid = grp[src_col].notna()
            for dc in design_cols:
                valid = valid & grp[dc].notna()

            idx = grp.index[valid]
            if len(idx) < cfg.min_obs_per_date:
                all_diags.append({"date": date, "horizon": h, "failure": NeutFailure.N_OBS_TOO_SMALL.value, "n_obs": len(idx)})
                continue

            y = grp.loc[idx, src_col].values.astype(float)
            X = np.column_stack([np.ones(len(idx))] + [grp.loc[idx, c].values.astype(float) for c in design_cols])

            residual, diag = neutralize_one_group(y, X, cfg)
            diag["date"] = date
            diag["horizon"] = h
            all_diags.append(diag)

            merged.loc[idx, dst_col] = residual

    # Stats
    diags_df = pd.DataFrame(all_diags)
    n_failed = int(diags_df["failure"].notna().sum()) if "failure" in diags_df.columns else 0
    n_total = len(diags_df)

    manifest = {
        "run_id": run_id,
        "estimator": cfg.estimator,
        "ridge_lambda": cfg.ridge_lambda,
        "n_date_horizon_groups": n_total,
        "n_failed": n_failed,
        "pct_failed": round(n_failed / max(n_total, 1) * 100, 2),
        "exposure_cols": list(cfg.exposure_cols),
        "timestamp": utc_now_iso(),
    }

    if output_dir:
        out = Path(output_dir) / run_id
        out.mkdir(parents=True, exist_ok=True)
        write_parquet_safe(merged, out / "neutralized_labels.parquet")
        if len(diags_df) > 0:
            write_parquet_safe(diags_df, out / "neutralization_diagnostics.parquet")
        write_json_safe(manifest, out / "neutralization_manifest.json")

    LOGGER.info("Neutralization: %d groups, %d failed (%.1f%%)", n_total, n_failed, manifest["pct_failed"])
    return {"neutralized": merged, "diagnostics": diags_df, "manifest": manifest}
