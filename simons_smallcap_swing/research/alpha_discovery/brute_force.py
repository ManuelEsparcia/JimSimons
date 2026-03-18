"""
research/alpha_discovery/brute_force.py - Constrained alpha exploration.

Generates symbolic candidates from a restricted grammar, evaluates OOS IC quality,
controls multiplicity, prunes redundancy, and emits a reproducible shortlist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from simons_smallcap_swing.research._shared import (
    bh_fdr,
    compute_daily_ic,
    compute_newey_west_tstat,
    config_hash,
    json_safe,
    run_id_with_prefix,
    stable_params_hash,
    utc_now_iso,
    write_json_safe,
    write_parquet_safe,
)


@dataclass(frozen=True)
class BruteForceConfig:
    candidate_budget: int = 500
    unary_ops: tuple[str, ...] = ("identity", "rank_cs", "zscore_cs", "delta_1", "ts_mean_5")
    binary_ops: tuple[str, ...] = ("add", "sub", "mul", "safe_div")
    min_coverage: float = 0.70
    min_valid_dates: int = 80
    min_ic_abs: float = 0.005
    max_ic_fold_std: float = 0.05
    max_redundancy_corr: float = 0.95
    bh_q_level: float = 0.10
    random_seed: int = 42


def canonicalize_formula(formula: Mapping[str, Any]) -> str:
    op = str(formula["op"]).lower()
    args = list(formula.get("args", []))
    if op in {"add", "mul", "min", "max"}:
        args = sorted(args)
    return f"{op}(" + ",".join(map(str, args)) + ")"


def hash_formula(formula: Mapping[str, Any]) -> str:
    return stable_params_hash({"canonical_formula": canonicalize_formula(formula)}, n=24)


def validate_formula(formula: Mapping[str, Any], terminals: set[str]) -> tuple[bool, str]:
    op = str(formula.get("op", ""))
    args = list(formula.get("args", []))
    if op == "" or len(args) == 0:
        return False, "parse_error"
    for a in args:
        if isinstance(a, str) and a not in terminals:
            return False, "unknown_terminal"
    if op == "sub" and len(args) == 2 and args[0] == args[1]:
        return False, "degenerate_formula"
    if op == "safe_div" and len(args) == 2 and args[0] == args[1]:
        return False, "degenerate_formula"
    return True, ""


def generate_candidates(config: BruteForceConfig, terminals: Sequence[str]) -> list[dict[str, Any]]:
    terms = sorted(set(terminals))
    candidates: list[dict[str, Any]] = []

    for t in terms:
        for op in config.unary_ops:
            candidates.append({"op": op, "args": [t]})
            if len(candidates) >= config.candidate_budget:
                return candidates

    for i, a in enumerate(terms):
        for b in terms[i:]:
            for op in config.binary_ops:
                candidates.append({"op": op, "args": [a, b]})
                if len(candidates) >= config.candidate_budget:
                    return candidates

    return candidates


def _rank_cs(x: pd.Series, date_idx: pd.Series) -> pd.Series:
    return x.groupby(date_idx).rank(pct=True, method="average")


def _zscore_cs(x: pd.Series, date_idx: pd.Series) -> pd.Series:
    g = x.groupby(date_idx)
    mu = g.transform("mean")
    sd = g.transform("std").replace(0, np.nan)
    return (x - mu) / sd


def materialize_formula(formula: Mapping[str, Any], feature_panel: pd.DataFrame) -> pd.Series:
    op = str(formula["op"]).lower()
    args = list(formula["args"])
    date_idx = pd.to_datetime(feature_panel["date"])

    def col(name: str) -> pd.Series:
        return pd.to_numeric(feature_panel[name], errors="coerce")

    if op == "identity":
        return col(args[0])
    if op == "rank_cs":
        return _rank_cs(col(args[0]), date_idx)
    if op == "zscore_cs":
        return _zscore_cs(col(args[0]), date_idx)
    if op == "delta_1":
        return col(args[0]).groupby(feature_panel["symbol"]).diff(1)
    if op == "ts_mean_5":
        return col(args[0]).groupby(feature_panel["symbol"]).transform(lambda s: s.rolling(5, min_periods=3).mean())
    if op == "add":
        return col(args[0]) + col(args[1])
    if op == "sub":
        return col(args[0]) - col(args[1])
    if op == "mul":
        return col(args[0]) * col(args[1])
    if op == "safe_div":
        den = col(args[1]).replace(0, np.nan)
        return col(args[0]) / den
    raise ValueError(f"unsupported operator: {op}")


def evaluate_candidate(
    candidate: Mapping[str, Any],
    feature_panel: pd.DataFrame,
    label_panel: pd.DataFrame,
    *,
    splits: pd.DataFrame | None = None,
    min_obs: int = 10,
) -> dict[str, Any]:
    formula_hash = hash_formula(candidate)
    canonical = canonicalize_formula(candidate)
    score = materialize_formula(candidate, feature_panel)

    panel = feature_panel[["date", "symbol"]].copy()
    panel["score"] = score
    panel = panel.merge(
        label_panel[["date", "symbol", "label"]],
        on=["date", "symbol"],
        how="left",
    )
    panel["date"] = pd.to_datetime(panel["date"])

    if panel["score"].dropna().std() < 1e-12:
        return {
            "formula_hash": formula_hash,
            "formula_dsl": candidate,
            "canonical_formula": canonical,
            "drop_reason": "zero_variance",
        }

    daily = compute_daily_ic(panel, score_col="score", label_col="label", date_col="date", min_obs=min_obs)
    n_valid_dates = int(daily["ic_value"].notna().sum())
    coverage = float((daily["n_valid"] >= min_obs).mean()) if len(daily) else 0.0
    ic_vals = daily["ic_value"].dropna()

    if n_valid_dates == 0:
        return {
            "formula_hash": formula_hash,
            "formula_dsl": candidate,
            "canonical_formula": canonical,
            "drop_reason": "low_coverage",
        }

    nw_t, nw_p = compute_newey_west_tstat(ic_vals.values, lags=5)

    fold_std = np.nan
    overfit_gap = np.nan
    if splits is not None and len(splits) > 0 and "date" in splits.columns:
        s = splits.copy()
        s["date"] = pd.to_datetime(s["date"])
        tmp = daily.merge(s, on="date", how="left")
        if "fold_id" in tmp.columns:
            fold_ic = tmp.groupby("fold_id")["ic_value"].mean()
            fold_std = float(fold_ic.std()) if len(fold_ic) > 1 else 0.0
        if "split_role" in tmp.columns:
            tr = float(tmp.loc[tmp["split_role"].astype(str).str.lower() == "train", "ic_value"].mean())
            oos = float(tmp.loc[tmp["split_role"].astype(str).str.lower().isin({"oos", "test", "valid"}), "ic_value"].mean())
            if np.isfinite(tr) and np.isfinite(oos):
                overfit_gap = tr - oos

    return {
        "formula_hash": formula_hash,
        "formula_dsl": candidate,
        "canonical_formula": canonical,
        "family_tag": str(candidate["op"]),
        "n_nodes": int(1 + len(candidate.get("args", []))),
        "depth": 2,
        "n_terminals": int(len(set(candidate.get("args", [])))),
        "complexity_score": float(1 + len(candidate.get("args", []))),
        "coverage": coverage,
        "n_valid_dates": n_valid_dates,
        "ic_oos_mean": float(ic_vals.mean()),
        "ic_oos_std": float(ic_vals.std()),
        "ic_hit_rate": float((ic_vals > 0).mean()),
        "ic_hac_tstat": float(nw_t),
        "ic_hac_pvalue": float(nw_p),
        "overfit_gap": float(overfit_gap) if np.isfinite(overfit_gap) else np.nan,
        "fold_std": float(fold_std) if np.isfinite(fold_std) else np.nan,
        "drop_reason": "",
        "daily_ic": daily,
    }


def adjust_multiple_testing(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    p = out["ic_hac_pvalue"].fillna(1.0).values if "ic_hac_pvalue" in out.columns else np.ones(len(out))
    out["bh_qvalue"] = bh_fdr(p)
    return out


def cluster_results(results: pd.DataFrame, *, max_corr: float = 0.95) -> pd.DataFrame:
    out = results.copy()
    out = out.sort_values("ic_oos_mean", ascending=False).reset_index(drop=True)
    cluster_ids = []
    rep_hashes: list[str] = []
    for i, row in out.iterrows():
        assigned = None
        if rep_hashes and "daily_ic_map" in out.columns:
            for cid, rep_h in enumerate(rep_hashes):
                a = out.at[i, "daily_ic_map"]
                b = out.loc[out["formula_hash"] == rep_h, "daily_ic_map"].iloc[0]
                corr = pd.Series(a).corr(pd.Series(b))
                if pd.notna(corr) and abs(float(corr)) >= max_corr:
                    assigned = cid
                    break
        if assigned is None:
            assigned = len(rep_hashes)
            rep_hashes.append(row["formula_hash"])
        cluster_ids.append(assigned)
    out["cluster_id"] = cluster_ids
    return out


def build_shortlist(results: pd.DataFrame, config: BruteForceConfig) -> pd.DataFrame:
    out = results.copy()
    out["selection_flag"] = (
        (out["coverage"] >= config.min_coverage)
        & (out["n_valid_dates"] >= config.min_valid_dates)
        & (out["ic_oos_mean"].abs() >= config.min_ic_abs)
        & (out["bh_qvalue"] <= config.bh_q_level)
        & ((out["fold_std"].isna()) | (out["fold_std"] <= config.max_ic_fold_std))
    )
    out["drop_reason"] = np.where(out["selection_flag"], "", np.where(out["drop_reason"] == "", "selection_reject", out["drop_reason"]))
    return out


def persist_run(
    results: pd.DataFrame,
    shortlist: pd.DataFrame,
    generation_log: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p1 = write_parquet_safe(results, out / "candidates_full.parquet")
    p2 = write_parquet_safe(shortlist, out / "shortlist.parquet")
    p3 = write_parquet_safe(generation_log, out / "generation_log.parquet")
    p4 = write_json_safe(manifest, out / "manifest.json")
    return {"candidates_full": p1, "shortlist": p2, "generation_log": p3, "manifest": p4}


def run_brute_force(
    feature_panel: pd.DataFrame | str | Path,
    label_panel: pd.DataFrame | str | Path,
    *,
    splits: pd.DataFrame | str | Path | None = None,
    config: BruteForceConfig | None = None,
    run_id: str | None = None,
    data_snapshot_id: str = "unknown",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or BruteForceConfig()
    rid = run_id or run_id_with_prefix("bf")

    if isinstance(feature_panel, (str, Path)):
        feature_panel = pd.read_parquet(feature_panel) if str(feature_panel).endswith(".parquet") else pd.read_csv(feature_panel)
    if isinstance(label_panel, (str, Path)):
        label_panel = pd.read_parquet(label_panel) if str(label_panel).endswith(".parquet") else pd.read_csv(label_panel)
    if isinstance(splits, (str, Path)):
        splits = pd.read_parquet(splits) if str(splits).endswith(".parquet") else pd.read_csv(splits)

    feature_panel = feature_panel.copy()
    feature_panel["date"] = pd.to_datetime(feature_panel["date"])
    label_panel = label_panel.copy()
    label_panel["date"] = pd.to_datetime(label_panel["date"])

    terminals = [c for c in feature_panel.columns if c not in {"date", "symbol"}]
    candidates = generate_candidates(cfg, terminals)
    generation_events: list[str] = []

    records: list[dict[str, Any]] = []
    daily_ic_map: dict[str, list[float]] = {}
    seen_hashes: set[str] = set()
    for cand in candidates:
        ok, reason = validate_formula(cand, set(terminals))
        if not ok:
            generation_events.append(reason)
            continue
        h = hash_formula(cand)
        if h in seen_hashes:
            generation_events.append("duplicate_hash")
            continue
        seen_hashes.add(h)
        result = evaluate_candidate(cand, feature_panel, label_panel, splits=splits)
        if result.get("drop_reason"):
            generation_events.append(str(result["drop_reason"]))
            result.setdefault("selection_flag", False)
            result.setdefault("run_id", rid)
            records.append({k: v for k, v in result.items() if k != "daily_ic"})
            continue
        daily_ic = result.pop("daily_ic")
        daily_ic_map[h] = daily_ic["ic_value"].fillna(0.0).tolist()
        result["run_id"] = rid
        records.append(result)

    results = pd.DataFrame(records)
    if len(results) == 0:
        results = pd.DataFrame(columns=[
            "formula_hash", "formula_dsl", "canonical_formula", "family_tag", "n_nodes", "depth",
            "n_terminals", "complexity_score", "coverage", "n_valid_dates", "ic_oos_mean", "ic_oos_std",
            "ic_hit_rate", "ic_hac_tstat", "ic_hac_pvalue", "overfit_gap", "fold_std", "drop_reason", "run_id"
        ])
    else:
        results["daily_ic_map"] = results["formula_hash"].map(daily_ic_map)
        results = adjust_multiple_testing(results)
        results = cluster_results(results, max_corr=cfg.max_redundancy_corr)
        results = build_shortlist(results, cfg)

    shortlist = results[results.get("selection_flag", pd.Series(dtype=bool)) == True].copy()  # noqa: E712
    generation_log = (
        pd.Series(generation_events, name="drop_reason")
        .value_counts()
        .rename_axis("drop_reason")
        .reset_index(name="count")
    )

    manifest = {
        "run_id": rid,
        "config_hash": config_hash(cfg.__dict__, n=24),
        "data_snapshot_id": data_snapshot_id,
        "labels_version": "unknown",
        "splits_version": "unknown",
        "candidate_budget": cfg.candidate_budget,
        "n_generated_raw": len(candidates),
        "n_unique_after_hash": len(seen_hashes),
        "n_evaluated": int(len(results)),
        "n_shortlisted": int(len(shortlist)),
        "execution_timestamp": utc_now_iso(),
    }

    artifacts = {}
    if output_dir:
        artifacts = persist_run(results.drop(columns=["daily_ic_map"], errors="ignore"), shortlist.drop(columns=["daily_ic_map"], errors="ignore"), generation_log, manifest, output_dir=output_dir)

    return {
        "candidates_full": results.drop(columns=["daily_ic_map"], errors="ignore"),
        "shortlist": shortlist.drop(columns=["daily_ic_map"], errors="ignore"),
        "generation_log": generation_log,
        "manifest": manifest,
        "artifacts": artifacts,
    }


__all__ = [
    "BruteForceConfig",
    "generate_candidates",
    "canonicalize_formula",
    "hash_formula",
    "validate_formula",
    "materialize_formula",
    "evaluate_candidate",
    "adjust_multiple_testing",
    "cluster_results",
    "build_shortlist",
    "persist_run",
    "run_brute_force",
]
