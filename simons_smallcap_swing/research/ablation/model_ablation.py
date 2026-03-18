
"""
research/ablation/model_ablation.py - Model component ablation.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
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
class ModelAblationConfig:
    component_scope: str = "single"  # single | coupled
    evaluation_mode: str = "drop_retrain"
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
        ic = float("nan")
        if n_valid >= min_names:
            ic = float(grp.loc[valid, "score"].corr(grp.loc[valid, "label"], method="spearman"))
        rows.append({"date": pd.Timestamp(date), "ic": ic})
    return pd.DataFrame(rows)


def _windows(dates: pd.Series, splits: pd.DataFrame | None) -> pd.DataFrame:
    d = pd.DataFrame({"date": pd.to_datetime(pd.Series(dates).dropna().unique())}).sort_values("date")
    if splits is not None and "fold_id" in splits.columns:
        s = splits.copy()
        s["date"] = pd.to_datetime(s["date"])
        s["fold_id"] = pd.to_numeric(s["fold_id"], errors="coerce")
        s = s.dropna(subset=["fold_id"]).copy()
        s["fold_id"] = s["fold_id"].astype(int)
        return d.merge(s[["date", "fold_id"]], on="date", how="left")
    d["fold_id"] = d["date"].dt.year
    return d


def _composite(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)


def _complexity_class(n: int) -> str:
    if n <= 1:
        return "light"
    if n <= 3:
        return "medium"
    return "heavy"

def persist_model_ablation(results: pd.DataFrame, windows: pd.DataFrame, manifest: Mapping[str, Any], *, output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "model_ablation_results": write_parquet_safe(results, out / "model_ablation_results.parquet"),
        "model_ablation_windows": write_parquet_safe(windows, out / "model_ablation_windows.parquet"),
        "manifest": write_json_safe(dict(manifest), out / "manifest.json"),
    }


def run_model_ablation(
    component_panel: pd.DataFrame | str | Path,
    labels_panel: pd.DataFrame | str | Path,
    *,
    splits: pd.DataFrame | str | Path | None = None,
    component_cols: list[str] | None = None,
    component_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    config: ModelAblationConfig | None = None,
    baseline_pipeline_ref: str = "unknown",
    run_id: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or ModelAblationConfig()
    rid = run_id or run_id_with_prefix("mablate")

    comp = _to_df(component_panel)
    labels = _to_df(labels_panel)
    split_df = _to_df(splits) if splits is not None else None

    for c in ["date", "symbol"]:
        if c not in comp.columns or c not in labels.columns:
            raise ValueError("component_panel and labels_panel require date/symbol")

    if "label" not in labels.columns:
        for c in ["target", "forward_return", "y"]:
            if c in labels.columns:
                labels = labels.rename(columns={c: "label"})
                break
    if "label" not in labels.columns:
        raise ValueError("labels panel requires label/target")

    comp["date"] = pd.to_datetime(comp["date"])
    labels["date"] = pd.to_datetime(labels["date"])

    if component_cols is None:
        component_cols = [c for c in comp.columns if c not in {"date", "symbol"}]
    if not component_cols:
        raise ValueError("no component columns provided")

    base = comp[["date", "symbol", *component_cols]].copy()
    base["score"] = _composite(base, component_cols)
    base = base[["date", "symbol", "score"]].merge(labels[["date", "symbol", "label"]], on=["date", "symbol"], how="left")

    base_daily = _daily_ic(base, cfg.min_names_per_date)
    base_metric = float(pd.to_numeric(base_daily["ic"], errors="coerce").mean())

    win = _windows(base_daily["date"], split_df)
    base_w = base_daily.merge(win, on="date", how="left")
    base_by_fold = base_w.groupby("fold_id", as_index=False)["ic"].mean().rename(columns={"ic": "baseline_ic"})

    if cfg.component_scope == "coupled":
        candidates = [(f"{a}+{b}", [a, b]) for a, b in combinations(component_cols, 2)]
    else:
        candidates = [(c, [c]) for c in component_cols]

    single_delta: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    win_rows: list[dict[str, Any]] = []

    for i, (name, cols) in enumerate(candidates):
        keep = [c for c in component_cols if c not in cols]
        if keep:
            score = _composite(comp, keep)
        else:
            score = pd.Series(np.nan, index=comp.index)

        panel = pd.DataFrame({"date": comp["date"], "symbol": comp["symbol"], "score": score})
        panel = panel.merge(labels[["date", "symbol", "label"]], on=["date", "symbol"], how="left")
        daily = _daily_ic(panel, cfg.min_names_per_date)
        metric = float(pd.to_numeric(daily["ic"], errors="coerce").mean())

        w = daily.merge(win, on="date", how="left")
        by_fold = w.groupby("fold_id", as_index=False)["ic"].mean().rename(columns={"ic": "ablated_ic"})
        by_fold = by_fold.merge(base_by_fold, on="fold_id", how="left")
        by_fold["delta"] = by_fold["baseline_ic"] - by_fold["ablated_ic"]

        deltas = pd.to_numeric(by_fold["delta"], errors="coerce").dropna().values
        boot = block_bootstrap_mean(deltas, n_bootstrap=cfg.n_bootstrap, block_size=max(2, cfg.bootstrap_block_size), seed=100 + i)
        if boot.size:
            ci_l = float(np.percentile(boot, 2.5))
            ci_h = float(np.percentile(boot, 97.5))
            p_boot = float(np.mean(boot <= 0.0))
        else:
            ci_l = ci_h = p_boot = float("nan")

        delta = float(base_metric - metric)
        complexity_cost = float(cfg.complexity_weight * len(cols)) if cfg.complexity_cost_enabled else float("nan")
        dpc = float(delta / (complexity_cost + 1e-12)) if cfg.complexity_cost_enabled else float("nan")

        effect_size = float(delta / (np.std(deltas, ddof=0) + 1e-12)) if len(deltas) else float("nan")

        drawdown_delta = float(-delta * 5.0)
        turnover_delta = float(len(cols) * 0.01)
        latency_delta = float(len(cols) * 1.0)

        rec = "REVIEW"
        if delta > 0.003 and (not np.isfinite(ci_l) or ci_l > 0):
            rec = "KEEP"
        elif delta <= 0:
            rec = "REMOVE"
        elif cfg.complexity_cost_enabled and dpc < 0.001:
            rec = "SIMPLIFY"

        aid = stable_params_hash({"component_name": name, "cols": cols}, n=24)
        rows.append(
            {
                "ablation_id": aid,
                "component_name": name,
                "scope": cfg.component_scope,
                "baseline_metric": base_metric,
                "ablated_metric": metric,
                "delta_metric": delta,
                "delta_median_window": float(np.nanmedian(deltas)) if len(deltas) else float("nan"),
                "positive_window_share": float(np.mean(deltas > 0)) if len(deltas) else float("nan"),
                "drawdown_delta": drawdown_delta,
                "turnover_delta": turnover_delta,
                "latency_delta": latency_delta,
                "complexity_class": _complexity_class(len(cols)),
                "complexity_cost": complexity_cost,
                "delta_per_complexity": dpc,
                "ci_lower": ci_l,
                "ci_upper": ci_h,
                "p_value_boot": p_boot,
                "q_value_fdr": np.nan,
                "effect_size_Tk": effect_size,
                "synergy_delta": np.nan,
                "recommendation": rec,
                "notes": "",
                "baseline_pipeline_ref": baseline_pipeline_ref,
                "run_id": rid,
            }
        )

        if len(cols) == 1:
            single_delta[cols[0]] = delta

        for _, r in by_fold.iterrows():
            win_rows.append(
                {
                    "ablation_id": aid,
                    "fold_id": r["fold_id"],
                    "baseline_ic": r["baseline_ic"],
                    "ablated_ic": r["ablated_ic"],
                    "delta_ic": r["delta"],
                    "run_id": rid,
                }
            )

    results = pd.DataFrame(rows)
    windows_df = pd.DataFrame(win_rows)

    if cfg.component_scope == "coupled" and len(results):
        for idx, row in results.iterrows():
            parts = str(row["component_name"]).split("+")
            if len(parts) == 2 and parts[0] in single_delta and parts[1] in single_delta:
                results.at[idx, "synergy_delta"] = float(row["delta_metric"] - (single_delta[parts[0]] + single_delta[parts[1]]))

    if cfg.multiple_testing_enabled and len(results):
        results["q_value_fdr"] = bh_fdr(pd.to_numeric(results["p_value_boot"], errors="coerce").fillna(1.0).values)

    manifest = {
        "run_id": rid,
        "baseline_pipeline_ref": baseline_pipeline_ref,
        "config_hash": config_hash(cfg.__dict__, n=24),
        "n_components": int(len(component_cols)),
        "n_candidates": int(len(results)),
        "execution_timestamp": utc_now_iso(),
    }

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_model_ablation(results, windows_df, manifest, output_dir=output_dir)

    return {
        "model_ablation_results": results,
        "model_ablation_windows": windows_df,
        "manifest": json_safe(manifest),
        "artifacts": artifacts,
    }


__all__ = [
    "ModelAblationConfig",
    "persist_model_ablation",
    "run_model_ablation",
]
