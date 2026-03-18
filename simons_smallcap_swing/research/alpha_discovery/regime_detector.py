
"""
research/alpha_discovery/regime_detector.py - PIT regime segmentation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from simons_smallcap_swing.research._shared import (
    config_hash,
    json_safe,
    read_dataframe,
    rolling_zscore_shifted,
    run_id_with_prefix,
    utc_now_iso,
    write_json_safe,
    write_parquet_safe,
)


@dataclass(frozen=True)
class RegimeDetectorConfig:
    method_name: str = "threshold"  # threshold | gmm | hmm
    k_candidates: tuple[int, ...] = (2, 3, 4, 5)
    criterion: str = "bic"
    covariance_type: str = "full"
    online_labeling: str = "filter"
    random_state: int = 42
    normalization_method: str = "rolling_zscore"
    normalization_lookback: int = 252
    normalization_eps: float = 1e-8
    min_dates_per_regime: int = 30
    min_mean_duration: int = 3
    max_entropy_mean: float = 1.2
    conditional_eval_enabled: bool = True
    conditional_metric: str = "spearman_ic"
    min_names_per_date: int = 25


def _to_df(obj: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    return read_dataframe(obj)


def load_market_proxies(market_proxies_panel: pd.DataFrame | str | Path) -> pd.DataFrame:
    df = _to_df(market_proxies_panel)
    if "date" not in df.columns:
        raise ValueError("market proxies panel must include date column")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    numeric_cols = [c for c in out.columns if c != "date" and pd.api.types.is_numeric_dtype(out[c])]
    if not numeric_cols:
        raise ValueError("market proxies panel requires at least one numeric proxy column")

    out = out[["date", *numeric_cols]].sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out.reset_index(drop=True)


def validate_proxy_panel(proxies: pd.DataFrame) -> dict[str, Any]:
    if proxies["date"].isna().any():
        raise ValueError("proxies contain null dates")

    proxy_cols = [c for c in proxies.columns if c != "date"]
    null_rate = float(proxies[proxy_cols].isna().mean().mean()) if proxy_cols else 1.0

    return {
        "n_dates": int(len(proxies)),
        "n_proxies": int(len(proxy_cols)),
        "mean_null_rate": null_rate,
    }


def normalize_proxies_pit(proxies: pd.DataFrame, config: RegimeDetectorConfig) -> pd.DataFrame:
    out = proxies.copy()
    proxy_cols = [c for c in out.columns if c != "date"]

    if config.normalization_method.lower() == "none":
        return out

    for c in proxy_cols:
        s = pd.to_numeric(out[c], errors="coerce")
        out[c] = rolling_zscore_shifted(s, lookback=config.normalization_lookback, eps=config.normalization_eps)

    return out


def _fallback_proxy_col(cols: list[str], preferred: list[str]) -> str | None:
    for p in preferred:
        for c in cols:
            if p in c.lower():
                return c
    return cols[0] if cols else None


def fit_threshold_regimes(raw_proxies: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = [c for c in raw_proxies.columns if c != "date"]
    vol_col = _fallback_proxy_col(cols, ["vix", "vol", "rv", "turbulence"])
    ret_col = _fallback_proxy_col(cols, ["ret", "momentum", "mkt", "benchmark"])

    x = raw_proxies.copy()
    if vol_col is None or ret_col is None:
        x["regime_id"] = 0
        x["regime_name"] = "neutral"
        x["posterior_max"] = 1.0
        x["posterior_entropy"] = 0.0
        return x[["date", "regime_id", "regime_name", "posterior_max", "posterior_entropy"]], {
            "rule": "fallback_neutral",
            "vol_col": vol_col,
            "ret_col": ret_col,
        }

    vol = pd.to_numeric(x[vol_col], errors="coerce")
    ret = pd.to_numeric(x[ret_col], errors="coerce")

    qv_hi = float(vol.quantile(0.75))
    qv_lo = float(vol.quantile(0.45))
    qr_dn = float(ret.quantile(0.30))
    qr_up = float(ret.quantile(0.60))

    regime_name = np.where(
        vol > qv_hi,
        "high_vol",
        np.where(
            ret < qr_dn,
            "down_trend",
            np.where((vol <= qv_lo) & (ret >= qr_up), "calm_uptrend", "neutral"),
        ),
    )

    name_to_id = {n: i for i, n in enumerate(sorted(pd.Series(regime_name).unique()))}
    regime_id = pd.Series(regime_name).map(name_to_id).astype(int)

    out = pd.DataFrame(
        {
            "date": x["date"],
            "regime_id": regime_id,
            "regime_name": regime_name,
            "posterior_max": 1.0,
            "posterior_entropy": 0.0,
        }
    )

    meta = {
        "rule": "threshold",
        "vol_col": vol_col,
        "ret_col": ret_col,
        "vol_q75": qv_hi,
        "ret_q30": qr_dn,
    }
    return out, meta

def _fit_gmm_labels(features: np.ndarray, k: int, random_state: int, covariance_type: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    try:
        from sklearn.mixture import GaussianMixture
    except Exception as exc:
        raise RuntimeError("sklearn is required for gmm method") from exc

    model = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    model.fit(features)
    probs = model.predict_proba(features)
    labels = probs.argmax(axis=1)
    bic = float(model.bic(features))
    aic = float(model.aic(features))
    return labels.astype(int), probs, bic, aic


def fit_gmm_regimes(
    proxies_norm: pd.DataFrame,
    *,
    config: RegimeDetectorConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = [c for c in proxies_norm.columns if c != "date"]
    x = proxies_norm.copy()
    x_num = x[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(x_num) < max(config.k_candidates):
        out, meta = fit_threshold_regimes(proxies_norm)
        meta["fallback_reason"] = "insufficient_samples_for_gmm"
        return out, meta

    feats = x_num.values.astype(float)

    tried: list[dict[str, Any]] = []
    for k in config.k_candidates:
        try:
            labels, probs, bic, aic = _fit_gmm_labels(feats, k=k, random_state=config.random_state, covariance_type=config.covariance_type)
        except Exception:
            continue

        temp = pd.DataFrame({"label": labels})
        counts = temp["label"].value_counts()
        viable = bool((counts >= config.min_dates_per_regime).all())
        tried.append(
            {
                "k": int(k),
                "labels": labels,
                "probs": probs,
                "bic": bic,
                "aic": aic,
                "viable": viable,
                "min_count": int(counts.min()) if len(counts) else 0,
            }
        )

    if not tried:
        out, meta = fit_threshold_regimes(proxies_norm)
        meta["fallback_reason"] = "gmm_fit_failed"
        return out, meta

    viable = [t for t in tried if t["viable"]]
    candidate_pool = viable if viable else tried

    key = "bic" if config.criterion.lower() == "bic" else "aic"
    best = sorted(candidate_pool, key=lambda d: d[key])[0]

    res = x[["date"]].merge(x_num.assign(_row_id=np.arange(len(x_num))), left_index=True, right_index=True, how="left")
    out = x[["date"]].copy()
    out["regime_id"] = np.nan
    out["posterior_max"] = np.nan
    out["posterior_entropy"] = np.nan

    idx = x_num.index.to_numpy()
    out.loc[idx, "regime_id"] = best["labels"]
    p = best["probs"]
    out.loc[idx, "posterior_max"] = p.max(axis=1)
    out.loc[idx, "posterior_entropy"] = -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1)

    out["regime_id"] = out["regime_id"].fillna(-1).astype(int)
    out["regime_name"] = out["regime_id"].map(lambda i: f"state_{i}" if i >= 0 else "unknown")

    meta = {
        "selector_method": key,
        "chosen_k": int(best["k"]),
        "bic": float(best["bic"]),
        "aic": float(best["aic"]),
        "k_trials": [{"k": int(t["k"]), "bic": float(t["bic"]), "aic": float(t["aic"]), "viable": bool(t["viable"])} for t in tried],
    }
    return out, meta


def fit_hmm_regimes(
    proxies_norm: pd.DataFrame,
    *,
    config: RegimeDetectorConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception:
        out, meta = fit_gmm_regimes(proxies_norm, config=config)
        meta["fallback_reason"] = "hmmlearn_not_available"
        return out, meta

    cols = [c for c in proxies_norm.columns if c != "date"]
    x = proxies_norm.copy()
    x_num = x[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(x_num) < max(config.k_candidates):
        out, meta = fit_threshold_regimes(proxies_norm)
        meta["fallback_reason"] = "insufficient_samples_for_hmm"
        return out, meta

    feats = x_num.values.astype(float)

    tried: list[dict[str, Any]] = []
    for k in config.k_candidates:
        try:
            model = GaussianHMM(n_components=int(k), covariance_type="diag", random_state=config.random_state, n_iter=200)
            model.fit(feats)
            probs = model.predict_proba(feats)
            labels = probs.argmax(axis=1)
            loglik = float(model.score(feats))
            n_params = k * feats.shape[1] * 2 + k * k
            bic = float(-2.0 * loglik + n_params * math.log(max(len(feats), 2)))
            aic = float(-2.0 * loglik + 2.0 * n_params)
        except Exception:
            continue

        counts = pd.Series(labels).value_counts()
        viable = bool((counts >= config.min_dates_per_regime).all())
        tried.append(
            {
                "k": int(k),
                "labels": labels,
                "probs": probs,
                "bic": bic,
                "aic": aic,
                "viable": viable,
            }
        )

    if not tried:
        out, meta = fit_gmm_regimes(proxies_norm, config=config)
        meta["fallback_reason"] = "hmm_fit_failed"
        return out, meta

    viable = [t for t in tried if t["viable"]]
    candidate_pool = viable if viable else tried

    key = "bic" if config.criterion.lower() == "bic" else "aic"
    best = sorted(candidate_pool, key=lambda d: d[key])[0]

    out = x[["date"]].copy()
    out["regime_id"] = -1
    out["posterior_max"] = np.nan
    out["posterior_entropy"] = np.nan

    idx = x_num.index.to_numpy()
    out.loc[idx, "regime_id"] = best["labels"]
    p = best["probs"]
    out.loc[idx, "posterior_max"] = p.max(axis=1)
    out.loc[idx, "posterior_entropy"] = -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1)
    out["regime_id"] = out["regime_id"].astype(int)
    out["regime_name"] = out["regime_id"].map(lambda i: f"state_{i}" if i >= 0 else "unknown")

    meta = {
        "selector_method": key,
        "chosen_k": int(best["k"]),
        "bic": float(best["bic"]),
        "aic": float(best["aic"]),
        "k_trials": [{"k": int(t["k"]), "bic": float(t["bic"]), "aic": float(t["aic"]), "viable": bool(t["viable"])} for t in tried],
    }
    return out, meta


def assign_semantic_labels(regimes: pd.DataFrame, proxies_raw: pd.DataFrame) -> pd.DataFrame:
    out = regimes.copy()
    merged = out.merge(proxies_raw, on="date", how="left")
    proxy_cols = [c for c in proxies_raw.columns if c != "date"]
    vol_col = _fallback_proxy_col(proxy_cols, ["vix", "vol", "rv", "turbulence"])

    if vol_col is None:
        return out

    state_score = merged.groupby("regime_id")[vol_col].mean().sort_values(ascending=False)
    ordered = list(state_score.index)
    semantic_order = {
        0: "stress",
        1: "risk_off",
        2: "neutral",
        3: "risk_on",
        4: "calm",
    }
    mapping: dict[int, str] = {}
    for i, sid in enumerate(ordered):
        mapping[int(sid)] = semantic_order.get(i, f"state_{sid}")

    out["regime_name"] = out["regime_id"].map(mapping).fillna(out["regime_name"])
    return out

def compute_regime_durations(regimes: pd.DataFrame) -> pd.DataFrame:
    if len(regimes) == 0:
        return pd.DataFrame(columns=["regime_id", "n_blocks", "mean_duration", "median_duration"])

    r = regimes.sort_values("date").copy()
    r["change"] = (r["regime_id"] != r["regime_id"].shift(1)).astype(int)
    r["block"] = r["change"].cumsum()
    block_sizes = r.groupby(["regime_id", "block"], as_index=False).size()

    stats = block_sizes.groupby("regime_id")["size"].agg(["count", "mean", "median"]).reset_index()
    stats = stats.rename(
        columns={
            "count": "n_blocks",
            "mean": "mean_duration",
            "median": "median_duration",
        }
    )
    return stats


def compute_transition_stats(regimes: pd.DataFrame) -> dict[str, Any]:
    if len(regimes) < 2:
        return {}

    r = regimes.sort_values("date")["regime_id"].astype(int).values
    transitions: dict[str, int] = {}
    for a, b in zip(r[:-1], r[1:]):
        key = f"{a}->{b}"
        transitions[key] = transitions.get(key, 0) + 1

    total = max(len(r) - 1, 1)
    transition_prob = {k: float(v / total) for k, v in transitions.items()}
    return {"counts": transitions, "probs": transition_prob}


def _daily_ic(panel: pd.DataFrame, min_names: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (alpha_id, date), grp in panel.groupby(["alpha_id", "date"], sort=True):
        valid = grp["score"].notna() & grp["label"].notna() & (grp["eligible"] > 0)
        n_valid = int(valid.sum())
        n_eligible = int((grp["eligible"] > 0).sum())
        cov = float(n_valid / n_eligible) if n_eligible > 0 else float("nan")
        ic_val = float("nan")
        if n_valid >= min_names:
            ic_val = float(grp.loc[valid, "score"].corr(grp.loc[valid, "label"], method="spearman"))
        rows.append(
            {
                "alpha_id": str(alpha_id),
                "date": pd.Timestamp(date),
                "ic_value": ic_val,
                "coverage": cov,
            }
        )
    return pd.DataFrame(rows)


def compute_conditional_ic(
    regimes: pd.DataFrame,
    alpha_scores_panel: pd.DataFrame,
    forward_labels_panel: pd.DataFrame,
    *,
    universe_mask: pd.DataFrame | None,
    min_names_per_date: int,
) -> pd.DataFrame:
    if len(alpha_scores_panel) == 0 or len(forward_labels_panel) == 0:
        return pd.DataFrame(
            columns=[
                "alpha_id",
                "regime_id",
                "n_dates",
                "mean_ic_cond",
                "std_ic_cond",
                "hit_rate_cond",
                "mean_ic_uncond",
                "delta_ic_vs_uncond",
                "coverage_cond",
            ]
        )

    s = alpha_scores_panel.copy()
    l = forward_labels_panel.copy()
    for col in ["date", "symbol"]:
        if col not in s.columns or col not in l.columns:
            raise ValueError("alpha_scores_panel and forward_labels_panel need date/symbol")

    if "alpha_id" not in s.columns:
        s["alpha_id"] = "alpha_0"

    score_col = "score" if "score" in s.columns else "signal"
    label_col = "label" if "label" in l.columns else "target"
    if score_col not in s.columns:
        raise ValueError("alpha scores panel needs score/signal column")
    if label_col not in l.columns:
        raise ValueError("forward labels panel needs label/target column")

    s = s[["date", "symbol", "alpha_id", score_col]].rename(columns={score_col: "score"})
    l = l[["date", "symbol", label_col]].rename(columns={label_col: "label"})

    s["date"] = pd.to_datetime(s["date"])
    l["date"] = pd.to_datetime(l["date"])

    panel = s.merge(l, on=["date", "symbol"], how="left")
    if universe_mask is not None and len(universe_mask):
        um = universe_mask.copy()
        um["date"] = pd.to_datetime(um["date"])
        if "eligible" not in um.columns:
            um["eligible"] = 1
        panel = panel.merge(um[["date", "symbol", "eligible"]], on=["date", "symbol"], how="left")
        panel["eligible"] = panel["eligible"].fillna(0).astype(int).clip(0, 1)
    else:
        panel["eligible"] = 1

    daily = _daily_ic(panel, min_names=min_names_per_date)
    daily = daily.merge(regimes[["date", "regime_id"]], on="date", how="left")

    uncond = daily.groupby("alpha_id", as_index=False)["ic_value"].mean().rename(columns={"ic_value": "mean_ic_uncond"})

    rows: list[dict[str, Any]] = []
    for (alpha_id, regime_id), grp in daily.groupby(["alpha_id", "regime_id"], sort=True):
        ic = pd.to_numeric(grp["ic_value"], errors="coerce").dropna()
        if len(ic) == 0:
            continue
        mean_ic = float(ic.mean())
        mean_uncond = float(uncond.loc[uncond["alpha_id"] == alpha_id, "mean_ic_uncond"].iloc[0])
        rows.append(
            {
                "alpha_id": str(alpha_id),
                "regime_id": int(regime_id),
                "n_dates": int(len(ic)),
                "mean_ic_cond": mean_ic,
                "std_ic_cond": float(ic.std(ddof=0)),
                "hit_rate_cond": float((ic > 0).mean()),
                "mean_ic_uncond": mean_uncond,
                "delta_ic_vs_uncond": float(mean_ic - mean_uncond),
                "coverage_cond": float(pd.to_numeric(grp["coverage"], errors="coerce").mean()),
            }
        )

    return pd.DataFrame(rows)


def persist_regime_outputs(
    regimes: pd.DataFrame,
    regime_summary: pd.DataFrame,
    conditional_alpha_stats: pd.DataFrame,
    diagnostics: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "regimes": write_parquet_safe(regimes, out / "regimes.parquet"),
        "regime_summary": write_parquet_safe(regime_summary, out / "regime_summary.parquet"),
        "conditional_alpha_stats": write_parquet_safe(conditional_alpha_stats, out / "conditional_alpha_stats.parquet"),
        "diagnostics": write_json_safe(dict(diagnostics), out / "diagnostics.json"),
    }


def run_regime_detector(
    market_proxies_panel: pd.DataFrame | str | Path,
    *,
    config: RegimeDetectorConfig | None = None,
    run_id: str | None = None,
    data_snapshot_id: str = "unknown",
    alpha_scores_panel: pd.DataFrame | str | Path | None = None,
    forward_labels_panel: pd.DataFrame | str | Path | None = None,
    universe_mask: pd.DataFrame | str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or RegimeDetectorConfig()
    rid = run_id or run_id_with_prefix("regime")

    raw = load_market_proxies(market_proxies_panel)
    validation = validate_proxy_panel(raw)
    norm = normalize_proxies_pit(raw, cfg)

    warnings: list[str] = []
    failures: list[str] = []

    method = cfg.method_name.lower()
    if method == "threshold":
        regimes, meta = fit_threshold_regimes(raw)
        selector = "threshold"
    elif method == "gmm":
        regimes, meta = fit_gmm_regimes(norm, config=cfg)
        selector = "gmm"
    elif method == "hmm":
        regimes, meta = fit_hmm_regimes(norm, config=cfg)
        selector = "hmm"
        if cfg.online_labeling.lower() != "filter":
            warnings.append("online_labeling_not_filter")
    else:
        raise ValueError(f"unsupported method_name: {cfg.method_name}")

    regimes = assign_semantic_labels(regimes, raw)
    regimes["method"] = selector
    regimes["run_id"] = rid

    durations = compute_regime_durations(regimes)
    transition_summary = compute_transition_stats(regimes)

    occ = regimes["regime_id"].value_counts().sort_index()
    min_dates = int(occ.min()) if len(occ) else 0
    mean_dur = float(durations["mean_duration"].mean()) if len(durations) else float("nan")
    entropy_mean = float(pd.to_numeric(regimes["posterior_entropy"], errors="coerce").mean()) if "posterior_entropy" in regimes.columns else 0.0

    if min_dates < cfg.min_dates_per_regime:
        failures.append("min_dates_per_regime")
    if np.isfinite(mean_dur) and mean_dur < cfg.min_mean_duration:
        warnings.append("mean_duration_low")
    if np.isfinite(entropy_mean) and entropy_mean > cfg.max_entropy_mean:
        warnings.append("posterior_entropy_high")

    if selector == "hmm" and cfg.online_labeling.lower() != "filter":
        failures.append("hmm_smoothing_not_allowed")

    if validation["n_dates"] <= 0:
        failures.append("empty_proxy_panel")

    status = "PASS"
    if failures:
        status = "FAIL"
    elif warnings:
        status = "WARN"

    summary = pd.DataFrame(
        [
            {
                "method": selector,
                "n_regimes": int(regimes["regime_id"].nunique()),
                "bic": float(meta.get("bic", np.nan)),
                "aic": float(meta.get("aic", np.nan)),
                "min_dates_per_regime": min_dates,
                "occupancy_by_regime": json_safe({int(k): int(v) for k, v in occ.to_dict().items()}),
                "mean_duration_by_regime": json_safe(
                    {int(r["regime_id"]): float(r["mean_duration"]) for _, r in durations.iterrows()} if len(durations) else {}
                ),
                "median_duration_by_regime": json_safe(
                    {int(r["regime_id"]): float(r["median_duration"]) for _, r in durations.iterrows()} if len(durations) else {}
                ),
                "transition_summary": json_safe(transition_summary),
                "config_hash": config_hash(cfg.__dict__, n=24),
                "run_id": rid,
            }
        ]
    )

    cond_stats = pd.DataFrame(
        columns=[
            "alpha_id",
            "regime_id",
            "n_dates",
            "mean_ic_cond",
            "std_ic_cond",
            "hit_rate_cond",
            "mean_ic_uncond",
            "delta_ic_vs_uncond",
            "coverage_cond",
            "run_id",
        ]
    )

    if cfg.conditional_eval_enabled and alpha_scores_panel is not None and forward_labels_panel is not None:
        alpha_df = _to_df(alpha_scores_panel)
        label_df = _to_df(forward_labels_panel)
        univ_df = _to_df(universe_mask) if universe_mask is not None else None
        cond_stats = compute_conditional_ic(
            regimes,
            alpha_df,
            label_df,
            universe_mask=univ_df,
            min_names_per_date=cfg.min_names_per_date,
        )
        if len(cond_stats):
            cond_stats["run_id"] = rid

    diagnostics = {
        "selector_method": selector,
        "K_candidates": list(cfg.k_candidates),
        "chosen_K": int(meta.get("chosen_k", regimes["regime_id"].nunique())),
        "PIT_validation_passed": bool(len(failures) == 0),
        "label_alignment_method": "date_merge",
        "warnings": sorted(set(warnings)),
        "failure_reasons": sorted(set(failures)),
        "gate_status": status,
        "data_snapshot_id": data_snapshot_id,
        "execution_timestamp": utc_now_iso(),
    }

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_regime_outputs(regimes, summary, cond_stats, diagnostics, output_dir=output_dir)

    return {
        "regimes": regimes[["date", "regime_id", "regime_name", "method", "posterior_max", "posterior_entropy", "run_id"]],
        "regime_summary": summary,
        "conditional_alpha_stats": cond_stats,
        "diagnostics": diagnostics,
        "artifacts": artifacts,
    }


__all__ = [
    "RegimeDetectorConfig",
    "load_market_proxies",
    "validate_proxy_panel",
    "normalize_proxies_pit",
    "fit_threshold_regimes",
    "fit_gmm_regimes",
    "fit_hmm_regimes",
    "assign_semantic_labels",
    "compute_regime_durations",
    "compute_transition_stats",
    "compute_conditional_ic",
    "persist_regime_outputs",
    "run_regime_detector",
]
