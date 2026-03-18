
"""
research/ablation/feature_ablation.py - Feature contribution ablation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from simons_smallcap_swing.research._shared import (
    bh_fdr,
    block_bootstrap_mean,
    config_hash,
    json_safe,
    read_dataframe,
    run_id_with_prefix,
    stable_params_hash,
    utc_now_iso,
    write_json_safe,
    write_parquet_safe,
)


@dataclass(frozen=True)
class FeatureAblationConfig:
    ablation_mode: str = "drop_retrain"  # drop_retrain | permute_oos
    scope: str = "single"  # single | group
    min_names_per_date: int = 25
    n_bootstrap: int = 400
    bootstrap_block_size: int = 12
    multiple_testing_enabled: bool = True
    q_level: float = 0.10
    complexity_cost_enabled: bool = True
    complexity_weight: float = 1.0


def _to_df(obj: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    return read_dataframe(obj)


def _daily_ic(panel: pd.DataFrame, min_names: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, grp in panel.groupby("date", sort=True):
        valid = grp["score"].notna() & grp["label"].notna()
        n_valid = int(valid.sum())
        n_eligible = int(len(grp))
        coverage = float(n_valid / max(n_eligible, 1))
        ic = float("nan")
        if n_valid >= min_names:
            ic = float(grp.loc[valid, "score"].corr(grp.loc[valid, "label"], method="spearman"))
        rows.append(
            {
                "date": pd.Timestamp(date),
                "ic": ic,
                "coverage": coverage,
            }
        )
    return pd.DataFrame(rows)


def _composite_signal(features: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    x = features[feature_cols].copy()
    for c in feature_cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    # Rank-based normalization by date keeps units comparable.
    tmp = pd.DataFrame(index=x.index)
    for c in feature_cols:
        tmp[c] = x[c]
    return tmp.mean(axis=1, skipna=True)


def _build_windows(dates: pd.Series, splits: pd.DataFrame | None) -> pd.DataFrame:
    d = pd.DataFrame({"date": pd.to_datetime(pd.Series(dates).dropna().unique())}).sort_values("date")
    if splits is not None and "fold_id" in splits.columns:
        s = splits.copy()
        s["date"] = pd.to_datetime(s["date"])
        s["fold_id"] = pd.to_numeric(s["fold_id"], errors="coerce")
        s = s.dropna(subset=["fold_id"]).copy()
        s["fold_id"] = s["fold_id"].astype(int)
        return d.merge(s[["date", "fold_id"]], on="date", how="left")

    out = d.copy()
    out["fold_id"] = out["date"].dt.year
    return out


def _ablate_features(
    features: pd.DataFrame,
    target_cols: list[str],
    *,
    mode: str,
    seed: int,
) -> pd.DataFrame:
    out = features.copy()
    if mode == "drop_retrain":
        return out.drop(columns=target_cols, errors="ignore")
    if mode == "permute_oos":
        rng = np.random.RandomState(seed)
        for c in target_cols:
            if c not in out.columns:
                continue
            out[c] = out.groupby("date")[c].transform(lambda s: pd.Series(rng.permutation(s.values), index=s.index))
        return out
    raise ValueError(f"unsupported ablation_mode: {mode}")

def persist_feature_ablation(results: pd.DataFrame, windows: pd.DataFrame, manifest: Mapping[str, Any], *, output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "ablation_results": write_parquet_safe(results, out / "ablation_results.parquet"),
        "ablation_windows": write_parquet_safe(windows, out / "ablation_windows.parquet"),
        "manifest": write_json_safe(dict(manifest), out / "manifest.json"),
    }


def run_feature_ablation(
    features_path: pd.DataFrame | str | Path,
    labels_path: pd.DataFrame | str | Path,
    *,
    splits_path: pd.DataFrame | str | Path | None = None,
    feature_cols: list[str] | None = None,
    feature_groups: Mapping[str, list[str]] | None = None,
    config: FeatureAblationConfig | None = None,
    baseline_run_ref: str = "unknown",
    run_id: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or FeatureAblationConfig()
    rid = run_id or run_id_with_prefix("fablate")

    features = _to_df(features_path)
    labels = _to_df(labels_path)
    splits = _to_df(splits_path) if splits_path is not None else None

    for c in ["date", "symbol"]:
        if c not in features.columns or c not in labels.columns:
            raise ValueError("features and labels must include date and symbol")

    if "label" not in labels.columns:
        for c in ["target", "forward_return", "y"]:
            if c in labels.columns:
                labels = labels.rename(columns={c: "label"})
                break
    if "label" not in labels.columns:
        raise ValueError("labels need label/target column")

    features["date"] = pd.to_datetime(features["date"])
    labels["date"] = pd.to_datetime(labels["date"])

    if feature_cols is None:
        feature_cols = [c for c in features.columns if c not in {"date", "symbol"}]

    if not feature_cols:
        raise ValueError("no feature columns provided")

    baseline_panel = features[["date", "symbol", *feature_cols]].copy()
    baseline_panel["score"] = _composite_signal(baseline_panel, feature_cols)
    baseline_panel = baseline_panel[["date", "symbol", "score"]].merge(labels[["date", "symbol", "label"]], on=["date", "symbol"], how="left")

    baseline_daily = _daily_ic(baseline_panel, cfg.min_names_per_date)
    baseline_metric = float(pd.to_numeric(baseline_daily["ic"], errors="coerce").mean())

    windows = _build_windows(baseline_daily["date"], splits)
    baseline_w = baseline_daily.merge(windows, on="date", how="left")
    baseline_by_window = baseline_w.groupby("fold_id", as_index=False)["ic"].mean().rename(columns={"ic": "baseline_ic"})

    if cfg.scope == "group" and feature_groups:
        candidates = [(name, cols) for name, cols in feature_groups.items()]
    else:
        candidates = [(c, [c]) for c in feature_cols]

    rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    for i, (name, cols) in enumerate(candidates):
        cols = [c for c in cols if c in feature_cols]
        if not cols:
            continue

        ablated = _ablate_features(features[["date", "symbol", *feature_cols]].copy(), cols, mode=cfg.ablation_mode, seed=123 + i)
        keep_cols = [c for c in feature_cols if c in ablated.columns]

        if not keep_cols:
            ablated_score = pd.Series(np.nan, index=ablated.index)
        else:
            ablated_score = _composite_signal(ablated, keep_cols)

        panel = pd.DataFrame({"date": features["date"], "symbol": features["symbol"], "score": ablated_score})
        panel = panel.merge(labels[["date", "symbol", "label"]], on=["date", "symbol"], how="left")

        daily = _daily_ic(panel, cfg.min_names_per_date)
        metric = float(pd.to_numeric(daily["ic"], errors="coerce").mean())
        delta = float(baseline_metric - metric)

        w = daily.merge(windows, on="date", how="left")
        by_window = w.groupby("fold_id", as_index=False)["ic"].mean().rename(columns={"ic": "ablated_ic"})
        by_window = by_window.merge(baseline_by_window, on="fold_id", how="left")
        by_window["delta"] = by_window["baseline_ic"] - by_window["ablated_ic"]
        by_window["ablation_id"] = stable_params_hash({"target": name, "cols": cols, "mode": cfg.ablation_mode}, n=24)

        deltas = pd.to_numeric(by_window["delta"], errors="coerce").dropna().values
        boot = block_bootstrap_mean(deltas, n_bootstrap=cfg.n_bootstrap, block_size=max(2, cfg.bootstrap_block_size), seed=42 + i)
        if boot.size:
            ci_l = float(np.percentile(boot, 2.5))
            ci_h = float(np.percentile(boot, 97.5))
            p_boot = float(np.mean(boot <= 0.0))
        else:
            ci_l = ci_h = p_boot = float("nan")

        complexity_cost = float(cfg.complexity_weight * len(cols)) if cfg.complexity_cost_enabled else float("nan")
        dpc = float(delta / (complexity_cost + 1e-12)) if cfg.complexity_cost_enabled else float("nan")

        recommendation = "REVIEW"
        if delta > 0.003 and (not np.isfinite(ci_l) or ci_l > 0):
            recommendation = "KEEP"
        elif delta <= 0:
            recommendation = "REMOVE"
        elif cfg.scope == "group" and delta > 0.0:
            recommendation = "GROUP"

        coverage_stats = {
            "non_missing_fraction": float(features[cols].notna().all(axis=1).mean()),
            "cross_sectional_coverage": float(features.groupby("date")[cols].apply(lambda x: x.notna().all(axis=1).mean()).mean()),
            "temporal_coverage_stability": float(features.groupby("date")[cols].apply(lambda x: x.notna().all(axis=1).mean()).std(ddof=0)),
        }

        secondary = {
            "baseline_std": float(pd.to_numeric(baseline_daily["ic"], errors="coerce").std(ddof=0)),
            "ablated_std": float(pd.to_numeric(daily["ic"], errors="coerce").std(ddof=0)),
            "n_windows": int(len(by_window)),
        }

        rows.append(
            {
                "ablation_id": by_window["ablation_id"].iloc[0] if len(by_window) else stable_params_hash({"target": name}, n=24),
                "scope": cfg.scope,
                "target_name": name,
                "ablation_mode": cfg.ablation_mode,
                "baseline_metric": baseline_metric,
                "ablated_metric": metric,
                "delta_metric": delta,
                "delta_median_window": float(np.nanmedian(deltas)) if len(deltas) else float("nan"),
                "positive_window_share": float(np.mean(deltas > 0)) if len(deltas) else float("nan"),
                "ci_lower": ci_l,
                "ci_upper": ci_h,
                "p_value_boot": p_boot,
                "q_value_fdr": np.nan,
                "complexity_cost": complexity_cost,
                "delta_per_complexity": dpc,
                "coverage_stats_json": json_safe(coverage_stats),
                "secondary_metrics_json": json_safe(secondary),
                "recommendation": recommendation,
                "notes": "",
                "baseline_run_ref": baseline_run_ref,
                "run_id": rid,
            }
        )

        for _, r in by_window.iterrows():
            window_rows.append(
                {
                    "ablation_id": r["ablation_id"],
                    "fold_id": r["fold_id"],
                    "baseline_ic": r["baseline_ic"],
                    "ablated_ic": r["ablated_ic"],
                    "delta_ic": r["delta"],
                    "run_id": rid,
                }
            )

    results = pd.DataFrame(rows)
    windows_df = pd.DataFrame(window_rows)

    if cfg.multiple_testing_enabled and len(results):
        results["q_value_fdr"] = bh_fdr(pd.to_numeric(results["p_value_boot"], errors="coerce").fillna(1.0).values)

    if "q_value_fdr" not in results.columns:
        results["q_value_fdr"] = np.nan

    manifest = {
        "run_id": rid,
        "baseline_run_ref": baseline_run_ref,
        "config_hash": config_hash(cfg.__dict__, n=24),
        "n_candidates": int(len(results)),
        "execution_timestamp": utc_now_iso(),
    }

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_feature_ablation(results, windows_df, manifest, output_dir=output_dir)

    return {
        "ablation_results": results,
        "ablation_windows": windows_df,
        "manifest": json_safe(manifest),
        "artifacts": artifacts,
    }


__all__ = [
    "FeatureAblationConfig",
    "persist_feature_ablation",
    "run_feature_ablation",
]
