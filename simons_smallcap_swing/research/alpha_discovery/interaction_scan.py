
"""
research/alpha_discovery/interaction_scan.py - Incremental interaction screening.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from simons_smallcap_swing.research._shared import (
    bh_fdr,
    block_bootstrap_mean,
    compute_newey_west_tstat,
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
class InteractionScanConfig:
    max_base_signals: int = 30
    min_mean_ic: float = 0.010
    min_mean_coverage: float = 0.80
    min_valid_dates: int = 126
    min_names_per_date: int = 25
    max_order: int = 2
    operators: tuple[str, ...] = (
        "product",
        "min",
        "max",
        "gated_product",
        "sigmoid_gate",
    )
    max_params_per_interaction: int = 2
    n_folds: int = 6
    n_bootstrap: int = 500
    bootstrap_block_size: int = 21
    random_seed: int = 42
    min_delta_ic_mean: float = 0.002
    min_delta_hit_rate: float = 0.60
    max_boot_pvalue: float = 0.10
    min_partial_ic: float = 0.001
    min_coverage: float = 0.80
    max_complexity: float = 6.0
    max_q_value: float = 0.10
    max_abs_corr_with_base: float = 0.95
    max_abs_corr_with_shortlist: float = 0.90
    top_n: int = 100


def _to_df(obj: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    return read_dataframe(obj)


def _normalize_signals(signals: pd.DataFrame, signal_col: str) -> pd.DataFrame:
    for c in ("date", "symbol"):
        if c not in signals.columns:
            raise ValueError("signals panel must contain date and symbol")

    out = signals.copy()
    if "alpha_id" not in out.columns:
        raise ValueError("signals panel must contain alpha_id")

    if signal_col not in out.columns:
        for c in ["signal", "score", "alpha_score", "value"]:
            if c in out.columns:
                signal_col = c
                break
        else:
            raise ValueError(f"signals panel missing signal column: {signal_col}")

    out = out[["date", "symbol", "alpha_id", signal_col]].rename(columns={signal_col: "signal"})
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)
    out["alpha_id"] = out["alpha_id"].astype(str)
    out["signal"] = pd.to_numeric(out["signal"], errors="coerce")

    dup = out.duplicated(subset=["date", "symbol", "alpha_id"])
    if bool(dup.any()):
        out = out.sort_values(["date", "symbol", "alpha_id"]).drop_duplicates(
            subset=["date", "symbol", "alpha_id"], keep="last"
        )

    return out


def _normalize_labels(labels: pd.DataFrame, label_col: str) -> pd.DataFrame:
    for c in ("date", "symbol"):
        if c not in labels.columns:
            raise ValueError("labels panel must contain date and symbol")

    out = labels.copy()
    if label_col not in out.columns:
        for c in ["label", "target", "forward_return", "y"]:
            if c in out.columns:
                label_col = c
                break
        else:
            raise ValueError(f"labels panel missing label column: {label_col}")

    out = out[["date", "symbol", label_col]].rename(columns={label_col: "label"})
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)
    out["label"] = pd.to_numeric(out["label"], errors="coerce")

    dup = out.duplicated(subset=["date", "symbol"])
    if bool(dup.any()):
        out = out.sort_values(["date", "symbol"]).drop_duplicates(subset=["date", "symbol"], keep="last")

    return out


def _normalize_universe(universe: pd.DataFrame | None) -> pd.DataFrame | None:
    if universe is None:
        return None
    if "date" not in universe.columns or "symbol" not in universe.columns:
        raise ValueError("universe mask must contain date and symbol")

    out = universe.copy()
    col = "eligible"
    if col not in out.columns:
        for c in ["in_universe", "is_eligible", "mask", "active"]:
            if c in out.columns:
                col = c
                break
        else:
            out[col] = 1
    out = out[["date", "symbol", col]].rename(columns={col: "eligible"})
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)
    out["eligible"] = out["eligible"].fillna(0).astype(int).clip(0, 1)
    out = out.drop_duplicates(subset=["date", "symbol"], keep="last")
    return out


def _normalize_splits(splits: pd.DataFrame | None) -> pd.DataFrame | None:
    if splits is None:
        return None
    if "date" not in splits.columns:
        raise ValueError("splits must contain date")

    out = splits.copy()
    fold_col = "fold_id"
    if fold_col not in out.columns:
        out[fold_col] = np.nan
    out["date"] = pd.to_datetime(out["date"])
    out = out[["date", fold_col]].rename(columns={fold_col: "fold_id"})
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out


def _compute_daily_ic(panel: pd.DataFrame, *, min_names: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, grp in panel.groupby("date", sort=True):
        eligible = grp["eligible"].fillna(0).astype(int) > 0
        n_eligible = int(eligible.sum())

        valid = eligible & grp["score"].notna() & grp["label"].notna()
        n_valid = int(valid.sum())

        coverage = float(n_valid / n_eligible) if n_eligible > 0 else float("nan")
        ic_val = float("nan")
        if n_valid >= min_names:
            pair = grp.loc[valid, ["score", "label"]].dropna()
            if len(pair) >= 3:
                ic_val = float(pair["score"].corr(pair["label"], method="spearman"))

        rows.append(
            {
                "date": pd.Timestamp(date),
                "ic_value": ic_val,
                "n_valid": n_valid,
                "n_eligible": n_eligible,
                "coverage": coverage,
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _build_fold_map(dates: pd.Series, splits: pd.DataFrame | None, n_folds: int) -> pd.DataFrame:
    unique_dates = pd.Series(pd.to_datetime(pd.Series(dates).dropna().unique())).sort_values()
    if splits is not None and splits["fold_id"].notna().any():
        out = splits.copy()
        out["date"] = pd.to_datetime(out["date"])
        out["fold_id"] = pd.to_numeric(out["fold_id"], errors="coerce")
        out = out.dropna(subset=["fold_id"]).copy()
        if len(out):
            out["fold_id"] = out["fold_id"].astype(int)
            return out[["date", "fold_id"]].drop_duplicates(subset=["date"], keep="last")

    if len(unique_dates) == 0:
        return pd.DataFrame(columns=["date", "fold_id"])

    bins = np.linspace(0, len(unique_dates), n_folds + 1)
    rows: list[dict[str, Any]] = []
    for fold_id in range(n_folds):
        a = int(math.floor(bins[fold_id]))
        b = int(math.floor(bins[fold_id + 1]))
        if fold_id == n_folds - 1:
            b = len(unique_dates)
        for d in unique_dates.iloc[a:b]:
            rows.append({"date": pd.Timestamp(d), "fold_id": int(fold_id)})

    return pd.DataFrame(rows)

def _operator_family(op: str) -> str:
    op = op.lower()
    if op in {"product"}:
        return "product"
    if op in {"min", "max"}:
        return "minmax"
    if op in {"gated_product", "gated_abs"}:
        return "gated"
    if op in {"sigmoid_gate"}:
        return "sigmoid"
    return op


def _interaction_complexity(op: str, n_params: int, order: int) -> float:
    base = 1.0
    op_weight = {
        "product": 1.0,
        "min": 1.0,
        "max": 1.0,
        "sum_weighted": 2.0,
        "safe_ratio": 2.0,
        "gated_product": 3.0,
        "gated_abs": 3.0,
        "sigmoid_gate": 4.0,
    }.get(op, 2.0)
    return float(base + op_weight + 0.5 * n_params + 0.5 * max(order - 1, 0))


def _canonical_interaction(candidate: Mapping[str, Any]) -> dict[str, Any]:
    op = str(candidate["operator"]).lower()
    bases = sorted(map(str, candidate["base_alpha_ids"]))
    params = dict(candidate.get("params", {}))
    if op in {"product", "min", "max", "sum_weighted"}:
        bases = sorted(bases)
    return {"operator": op, "base_alpha_ids": bases, "params": params}


def _interaction_id(candidate: Mapping[str, Any]) -> str:
    canonical = _canonical_interaction(candidate)
    return stable_params_hash(canonical, n=24)


def enumerate_interactions(base_alpha_ids: list[str], config: InteractionScanConfig) -> list[dict[str, Any]]:
    alphas = sorted(set(base_alpha_ids))
    candidates: list[dict[str, Any]] = []

    if config.max_order < 2:
        return candidates

    for a, b in combinations(alphas, 2):
        for op in config.operators:
            op = op.lower()
            if op == "sum_weighted":
                for w in (0.25, 0.50, 0.75):
                    candidates.append(
                        {
                            "base_alpha_ids": [a, b],
                            "operator": op,
                            "params": {"w": float(w)},
                        }
                    )
            elif op == "safe_ratio":
                candidates.append(
                    {
                        "base_alpha_ids": [a, b],
                        "operator": op,
                        "params": {"eps": 1e-6},
                    }
                )
            elif op == "gated_product":
                candidates.append(
                    {
                        "base_alpha_ids": [a, b],
                        "operator": op,
                        "params": {"c": 0.0},
                    }
                )
            elif op == "gated_abs":
                candidates.append(
                    {
                        "base_alpha_ids": [a, b],
                        "operator": op,
                        "params": {"c": 0.0},
                    }
                )
            elif op == "sigmoid_gate":
                for slope in (1.0, 2.0):
                    candidates.append(
                        {
                            "base_alpha_ids": [a, b],
                            "operator": op,
                            "params": {"slope": float(slope), "center": 0.0},
                        }
                    )
            else:
                candidates.append(
                    {
                        "base_alpha_ids": [a, b],
                        "operator": op,
                        "params": {},
                    }
                )

    dedup: dict[str, dict[str, Any]] = {}
    for c in candidates:
        cid = _interaction_id(c)
        cc = _canonical_interaction(c)
        c = {
            "interaction_id": cid,
            "base_alpha_ids": cc["base_alpha_ids"],
            "operator": cc["operator"],
            "params": cc["params"],
            "operator_family": _operator_family(cc["operator"]),
            "operator_spec": f"{cc['operator']}({','.join(cc['base_alpha_ids'])})",
            "complexity": _interaction_complexity(cc["operator"], len(cc["params"]), len(cc["base_alpha_ids"])),
        }
        dedup[cid] = c

    return [dedup[k] for k in sorted(dedup.keys())]


def _apply_operator(op: str, a: pd.Series, b: pd.Series, params: Mapping[str, Any]) -> pd.Series:
    op = op.lower()

    if op == "product":
        return a * b
    if op == "min":
        return pd.concat([a, b], axis=1).min(axis=1)
    if op == "max":
        return pd.concat([a, b], axis=1).max(axis=1)
    if op == "sum_weighted":
        w = float(params.get("w", 0.5))
        return w * a + (1.0 - w) * b
    if op == "safe_ratio":
        eps = float(params.get("eps", 1e-6))
        den = b.replace(0, np.nan)
        return a / (den + eps)
    if op == "gated_product":
        c = float(params.get("c", 0.0))
        return a * (b > c).astype(float)
    if op == "gated_abs":
        c = float(params.get("c", 0.0))
        return a * (b.abs() > abs(c)).astype(float)
    if op == "sigmoid_gate":
        slope = float(params.get("slope", 1.0))
        center = float(params.get("center", 0.0))
        gate = 1.0 / (1.0 + np.exp(-slope * (b - center)))
        return a * gate

    raise ValueError(f"unsupported operator: {op}")


def _partial_ic_mean(
    merged: pd.DataFrame,
    base_cols: list[str],
    *,
    min_names: int,
) -> float:
    pic_vals: list[float] = []

    for _, grp in merged.groupby("date", sort=True):
        use_cols = ["score", "label", *base_cols]
        part = grp[use_cols].dropna()
        if len(part) < min_names:
            continue

        z = part["score"].values.astype(float)
        y = part["label"].values.astype(float)
        X = part[base_cols].values.astype(float)
        X = np.column_stack([np.ones(len(X)), X])

        try:
            beta_z, *_ = np.linalg.lstsq(X, z, rcond=None)
            beta_y, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        z_res = z - X @ beta_z
        y_res = y - X @ beta_y

        if np.std(z_res) < 1e-12 or np.std(y_res) < 1e-12:
            continue

        corr = pd.Series(z_res).corr(pd.Series(y_res), method="spearman")
        if pd.notna(corr):
            pic_vals.append(float(corr))

    if not pic_vals:
        return float("nan")
    return float(np.mean(pic_vals))

def _evaluate_base_signals(
    wide_signals: pd.DataFrame,
    labels: pd.DataFrame,
    universe: pd.DataFrame | None,
    *,
    config: InteractionScanConfig,
) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, pd.Series]]:
    base_rows: list[dict[str, Any]] = []
    base_ic_map: dict[str, pd.Series] = {}
    base_score_map: dict[str, pd.Series] = {}

    idx_df = wide_signals.reset_index()[["date", "symbol"]]

    for alpha_id in sorted(wide_signals.columns):
        panel = idx_df.copy()
        panel["score"] = wide_signals[alpha_id].values
        panel = panel.merge(labels, on=["date", "symbol"], how="left")
        if universe is not None:
            panel = panel.merge(universe, on=["date", "symbol"], how="left")
            panel["eligible"] = panel["eligible"].fillna(0).astype(int).clip(0, 1)
        else:
            panel["eligible"] = 1

        daily = _compute_daily_ic(panel, min_names=config.min_names_per_date)
        ic = pd.Series(daily["ic_value"].values, index=pd.to_datetime(daily["date"]), dtype=float)

        mean_ic = float(pd.to_numeric(daily["ic_value"], errors="coerce").mean())
        coverage_mean = float(pd.to_numeric(daily["coverage"], errors="coerce").mean())
        n_valid_dates = int(pd.to_numeric(daily["ic_value"], errors="coerce").notna().sum())

        base_rows.append(
            {
                "alpha_id": alpha_id,
                "mean_ic": mean_ic,
                "coverage_mean": coverage_mean,
                "n_valid_dates": n_valid_dates,
            }
        )
        base_ic_map[alpha_id] = ic
        base_score_map[alpha_id] = wide_signals[alpha_id].astype(float)

    base_df = pd.DataFrame(base_rows)
    if len(base_df):
        eligible = (
            (base_df["mean_ic"] >= config.min_mean_ic)
            & (base_df["coverage_mean"] >= config.min_mean_coverage)
            & (base_df["n_valid_dates"] >= config.min_valid_dates)
        )
        base_df["eligible"] = eligible
        base_df = base_df.sort_values(["eligible", "mean_ic", "coverage_mean"], ascending=[False, False, False])
        eligible_ids = base_df.loc[base_df["eligible"], "alpha_id"].tolist()[: config.max_base_signals]
        base_df["selected"] = base_df["alpha_id"].isin(eligible_ids)
    else:
        base_df = base_df.assign(eligible=pd.Series(dtype=bool), selected=pd.Series(dtype=bool))

    return base_df.reset_index(drop=True), base_ic_map, base_score_map


def _evaluate_candidate(
    candidate: Mapping[str, Any],
    *,
    wide_signals: pd.DataFrame,
    labels: pd.DataFrame,
    universe: pd.DataFrame | None,
    fold_map: pd.DataFrame,
    base_ic_map: Mapping[str, pd.Series],
    base_score_map: Mapping[str, pd.Series],
    config: InteractionScanConfig,
) -> tuple[dict[str, Any], pd.DataFrame, pd.Series]:
    bases = list(candidate["base_alpha_ids"])
    if any(b not in wide_signals.columns for b in bases):
        return {"interaction_id": candidate["interaction_id"], "drop_reason": "missing_base_signal"}, pd.DataFrame(), pd.Series(dtype=float)

    a = wide_signals[bases[0]].astype(float)
    b = wide_signals[bases[1]].astype(float)
    score = _apply_operator(str(candidate["operator"]), a, b, candidate.get("params", {}))

    idx_df = wide_signals.reset_index()[["date", "symbol"]]
    merged = idx_df.copy()
    merged["score"] = score.values
    merged[bases[0]] = a.values
    merged[bases[1]] = b.values
    merged = merged.merge(labels, on=["date", "symbol"], how="left")

    if universe is not None:
        merged = merged.merge(universe, on=["date", "symbol"], how="left")
        merged["eligible"] = merged["eligible"].fillna(0).astype(int).clip(0, 1)
    else:
        merged["eligible"] = 1

    if merged["score"].dropna().std() < 1e-12:
        return {
            "interaction_id": candidate["interaction_id"],
            "drop_reason": "degenerate_interaction",
        }, pd.DataFrame(), score

    daily = _compute_daily_ic(merged, min_names=config.min_names_per_date)
    n_valid_dates = int(pd.to_numeric(daily["ic_value"], errors="coerce").notna().sum())
    coverage_mean = float(pd.to_numeric(daily["coverage"], errors="coerce").mean())
    coverage_std = float(pd.to_numeric(daily["coverage"], errors="coerce").std(ddof=0))

    if n_valid_dates == 0:
        return {"interaction_id": candidate["interaction_id"], "drop_reason": "no_valid_dates"}, pd.DataFrame(), score

    daily = daily.merge(fold_map, on="date", how="left")
    daily["fold_id"] = pd.to_numeric(daily["fold_id"], errors="coerce")
    daily = daily.dropna(subset=["fold_id"]).copy()
    daily["fold_id"] = daily["fold_id"].astype(int)
    if len(daily) == 0:
        return {"interaction_id": candidate["interaction_id"], "drop_reason": "no_fold_assignment"}, pd.DataFrame(), score

    ic_int_series = pd.Series(daily["ic_value"].values, index=pd.to_datetime(daily["date"]), dtype=float)
    base_daily = []
    for base_id in bases:
        s = base_ic_map.get(base_id, pd.Series(dtype=float))
        s = s.reindex(ic_int_series.index)
        base_daily.append(s)
    base_stack = pd.concat(base_daily, axis=1)
    best_base_daily = base_stack.max(axis=1)
    delta_daily = ic_int_series - best_base_daily

    ic_mean_int = float(ic_int_series.mean())
    ic_mean_best_base = float(best_base_daily.mean())
    delta_ic_mean = float(delta_daily.mean())
    delta_ic_std = float(delta_daily.std(ddof=0))
    delta_ic_hit_rate = float((delta_daily > 0).mean())

    fold_rows: list[dict[str, Any]] = []
    for fid, grp in daily.groupby("fold_id", sort=True):
        d = pd.to_datetime(grp["date"])
        ic_fold = ic_int_series.reindex(d).astype(float)
        base_fold = best_base_daily.reindex(d).astype(float)
        delta_fold = ic_fold - base_fold
        fold_rows.append(
            {
                "interaction_id": candidate["interaction_id"],
                "fold_id": int(fid),
                "ic_int_fold": float(ic_fold.mean()),
                "ic_best_base_fold": float(base_fold.mean()),
                "delta_ic_fold": float(delta_fold.mean()),
                "coverage_fold": float(pd.to_numeric(grp["coverage"], errors="coerce").mean()),
            }
        )

    boot = block_bootstrap_mean(
        delta_daily.dropna().values,
        n_bootstrap=config.n_bootstrap,
        block_size=config.bootstrap_block_size,
        seed=config.random_seed,
    )
    if boot.size:
        boot_p = float(np.mean(boot <= 0.0))
        ci_low = float(np.percentile(boot, 2.5))
        ci_high = float(np.percentile(boot, 97.5))
    else:
        boot_p = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")

    pic = _partial_ic_mean(merged, base_cols=bases, min_names=config.min_names_per_date)

    v_int = score.astype(float)
    red_base = []
    for b_id in bases:
        vb = base_score_map[b_id]
        corr = v_int.corr(vb)
        if pd.notna(corr):
            red_base.append(abs(float(corr)))
    redundancy_base_max = float(max(red_base)) if red_base else float("nan")

    nw_t, nw_p = compute_newey_west_tstat(delta_daily.dropna().values, lags=5)

    rec = {
        "interaction_id": candidate["interaction_id"],
        "base_alpha_ids": list(bases),
        "operator_family": str(candidate["operator_family"]),
        "operator_spec": str(candidate["operator_spec"]),
        "params_json": json_safe(candidate.get("params", {})),
        "complexity": float(candidate["complexity"]),
        "n_valid_dates": n_valid_dates,
        "coverage_mean": coverage_mean,
        "coverage_std": coverage_std,
        "ic_mean_int": ic_mean_int,
        "ic_mean_best_base": ic_mean_best_base,
        "delta_ic_mean": delta_ic_mean,
        "delta_ic_std": delta_ic_std,
        "delta_ic_hit_rate": delta_ic_hit_rate,
        "partial_ic_mean": float(pic),
        "bootstrap_pvalue": boot_p,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "delta_ic_hac_tstat": float(nw_t),
        "delta_ic_hac_pvalue": float(nw_p),
        "redundancy_base_max": redundancy_base_max,
        "drop_reason": "",
    }

    return rec, pd.DataFrame(fold_rows), score


def adjust_multiple_testing(full: pd.DataFrame) -> pd.DataFrame:
    out = full.copy()
    out["bh_qvalue"] = np.nan
    if len(out) == 0:
        return out

    for _, idx in out.groupby("operator_family").groups.items():
        p = pd.to_numeric(out.loc[idx, "bootstrap_pvalue"], errors="coerce").fillna(1.0).values
        q = bh_fdr(p)
        out.loc[idx, "bh_qvalue"] = q

    out["bh_qvalue"] = pd.to_numeric(out["bh_qvalue"], errors="coerce").fillna(1.0)
    return out


def apply_selection_gates(full: pd.DataFrame, config: InteractionScanConfig) -> pd.DataFrame:
    out = full.copy()
    gate: list[str] = []
    reasons: list[str] = []

    for _, r in out.iterrows():
        fails: list[str] = []
        warns: list[str] = []

        if float(r.get("delta_ic_mean", np.nan)) <= 0.0:
            fails.append("fail_delta_ic")
        if float(r.get("coverage_mean", np.nan)) < config.min_coverage:
            fails.append("coverage")
        if float(r.get("bootstrap_pvalue", np.nan)) > config.max_boot_pvalue:
            fails.append("bootstrap")
        if float(r.get("partial_ic_mean", np.nan)) < config.min_partial_ic:
            fails.append("partial_ic")
        if float(r.get("redundancy_base_max", np.nan)) > config.max_abs_corr_with_base:
            fails.append("redundancy_base")
        if float(r.get("complexity", np.nan)) > config.max_complexity:
            fails.append("complexity")
        if float(r.get("bh_qvalue", np.nan)) > config.max_q_value:
            fails.append("qvalue")
        if int(r.get("n_valid_dates", 0)) < config.min_valid_dates:
            fails.append("n_valid_dates")

        if not fails:
            if float(r.get("delta_ic_mean", np.nan)) < config.min_delta_ic_mean:
                warns.append("delta_ic")
            if float(r.get("delta_ic_hit_rate", np.nan)) < config.min_delta_hit_rate:
                warns.append("delta_hit")

        if fails:
            gate.append("FAIL")
            reasons.append(";".join(fails))
        elif warns:
            gate.append("WARN")
            reasons.append(";".join(warns))
        else:
            gate.append("PASS")
            reasons.append("pass")

    out["selection_gate"] = gate
    out["drop_reason"] = reasons
    return out


def cluster_survivors(
    full: pd.DataFrame,
    score_map: Mapping[str, pd.Series],
    *,
    max_abs_corr: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = full.copy()
    out["redundancy_shortlist_max"] = np.nan

    candidates = out[out["selection_gate"] == "PASS"].copy()
    candidates = candidates.sort_values(["delta_ic_mean", "partial_ic_mean", "complexity"], ascending=[False, False, True])

    selected_ids: list[str] = []
    selected_rows: list[dict[str, Any]] = []

    for idx, row in candidates.iterrows():
        cid = str(row["interaction_id"])
        v = score_map.get(cid)
        max_corr = 0.0
        if v is not None and selected_ids:
            corr_vals: list[float] = []
            for sid in selected_ids:
                sv = score_map.get(sid)
                if sv is None:
                    continue
                corr = v.corr(sv)
                if pd.notna(corr):
                    corr_vals.append(abs(float(corr)))
            if corr_vals:
                max_corr = float(max(corr_vals))

        out.loc[idx, "redundancy_shortlist_max"] = max_corr if selected_ids else 0.0

        if selected_ids and max_corr > max_abs_corr:
            out.loc[idx, "selection_gate"] = "FAIL"
            out.loc[idx, "drop_reason"] = "redundancy_shortlist"
            continue

        selected_ids.append(cid)
        selected_rows.append(row.to_dict())

    shortlist = pd.DataFrame(selected_rows)
    if len(shortlist) == 0:
        shortlist = pd.DataFrame(columns=out.columns)

    return out, shortlist

def persist_results(
    interactions_full: pd.DataFrame,
    interactions_folds: pd.DataFrame,
    shortlist: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "interactions_full": write_parquet_safe(interactions_full, out / "interactions_full.parquet"),
        "interactions_folds": write_parquet_safe(interactions_folds, out / "interactions_folds.parquet"),
        "shortlist": write_parquet_safe(shortlist, out / "shortlist.parquet"),
        "manifest": write_json_safe(dict(manifest), out / "manifest.json"),
    }


def run_interaction_scan(
    signals_panel: pd.DataFrame | str | Path,
    labels_panel: pd.DataFrame | str | Path,
    *,
    universe_mask: pd.DataFrame | str | Path | None = None,
    splits: pd.DataFrame | str | Path | None = None,
    signal_col: str = "signal",
    label_col: str = "label",
    config: InteractionScanConfig | None = None,
    run_id: str | None = None,
    data_snapshot_id: str = "unknown",
    signals_version: str = "unknown",
    labels_version: str = "unknown",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or InteractionScanConfig()
    rid = run_id or run_id_with_prefix("intscan")

    signals = _normalize_signals(_to_df(signals_panel), signal_col=signal_col)
    labels = _normalize_labels(_to_df(labels_panel), label_col=label_col)
    universe = _normalize_universe(_to_df(universe_mask)) if universe_mask is not None else None
    split_df = _normalize_splits(_to_df(splits)) if splits is not None else None

    wide = signals.pivot_table(index=["date", "symbol"], columns="alpha_id", values="signal", aggfunc="last")
    wide = wide.sort_index()

    base_eval, base_ic_map, base_score_map = _evaluate_base_signals(
        wide,
        labels,
        universe,
        config=cfg,
    )

    selected_bases = base_eval.loc[base_eval["selected"], "alpha_id"].astype(str).tolist()
    wide = wide[selected_bases] if selected_bases else wide.iloc[:, :0]

    candidates = enumerate_interactions(selected_bases, cfg)
    fold_map = _build_fold_map(wide.reset_index()["date"] if len(wide) else labels["date"], split_df, cfg.n_folds)

    records: list[dict[str, Any]] = []
    fold_rows: list[pd.DataFrame] = []
    score_map: dict[str, pd.Series] = {}

    for cand in candidates:
        rec, fold_metrics, score = _evaluate_candidate(
            cand,
            wide_signals=wide,
            labels=labels,
            universe=universe,
            fold_map=fold_map,
            base_ic_map=base_ic_map,
            base_score_map=base_score_map,
            config=cfg,
        )

        records.append(rec)
        if len(fold_metrics):
            fold_rows.append(fold_metrics)
        if len(score):
            score_map[cand["interaction_id"]] = score.astype(float)

    full = pd.DataFrame(records)
    required_cols = [
        "interaction_id",
        "base_alpha_ids",
        "operator_family",
        "operator_spec",
        "params_json",
        "complexity",
        "n_valid_dates",
        "coverage_mean",
        "coverage_std",
        "ic_mean_int",
        "ic_mean_best_base",
        "delta_ic_mean",
        "delta_ic_std",
        "delta_ic_hit_rate",
        "partial_ic_mean",
        "bootstrap_pvalue",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "bh_qvalue",
        "redundancy_base_max",
        "redundancy_shortlist_max",
        "selection_gate",
        "drop_reason",
        "run_id",
    ]

    if len(full) == 0:
        full = pd.DataFrame(columns=required_cols)
        folds = pd.DataFrame(
            columns=["interaction_id", "fold_id", "ic_int_fold", "ic_best_base_fold", "delta_ic_fold", "coverage_fold"]
        )
        shortlist = pd.DataFrame(columns=required_cols)
    else:
        full = adjust_multiple_testing(full)
        full = apply_selection_gates(full, cfg)
        full, shortlist = cluster_survivors(full, score_map, max_abs_corr=cfg.max_abs_corr_with_shortlist)

        shortlist = shortlist.sort_values(["delta_ic_mean", "partial_ic_mean", "complexity"], ascending=[False, False, True])
        shortlist = shortlist.head(cfg.top_n).reset_index(drop=True)

        folds = pd.concat(fold_rows, axis=0, ignore_index=True) if fold_rows else pd.DataFrame(
            columns=["interaction_id", "fold_id", "ic_int_fold", "ic_best_base_fold", "delta_ic_fold", "coverage_fold"]
        )

        for c in required_cols:
            if c not in full.columns:
                full[c] = np.nan
        for c in required_cols:
            if c not in shortlist.columns:
                shortlist[c] = np.nan

    full["run_id"] = rid
    shortlist["run_id"] = rid

    if len(folds):
        folds = folds.sort_values(["interaction_id", "fold_id"]).reset_index(drop=True)

    manifest = {
        "run_id": rid,
        "config_hash": config_hash(cfg.__dict__, n=24),
        "data_snapshot_id": data_snapshot_id,
        "signals_version": signals_version,
        "labels_version": labels_version,
        "n_base_eligible": int(len(selected_bases)),
        "n_interactions_generated": int(len(candidates)),
        "n_interactions_valid": int(len(full)),
        "n_interactions_selected": int(len(shortlist)),
        "execution_timestamp": utc_now_iso(),
    }

    full = full[required_cols].copy()
    shortlist = shortlist[required_cols].copy()

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_results(full, folds, shortlist, manifest, output_dir=output_dir)

    return {
        "interactions_full": full,
        "interactions_folds": folds,
        "shortlist": shortlist,
        "base_selection": base_eval,
        "manifest": json_safe(manifest),
        "artifacts": artifacts,
    }


__all__ = [
    "InteractionScanConfig",
    "enumerate_interactions",
    "adjust_multiple_testing",
    "apply_selection_gates",
    "cluster_survivors",
    "persist_results",
    "run_interaction_scan",
]
