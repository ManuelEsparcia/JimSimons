"""
models/regimes/regime_features.py - Causal market-state representation for regime inference.

Builds PIT-safe regime features (volatility, breadth, dispersion, correlation
concentration, liquidity, stress), with robust normalization and governance
artifacts for downstream HMM/gating modules.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_id_with_prefix(prefix: str) -> str:
    from datetime import datetime, timezone

    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if np.isnan(v) or np.isinf(v) else v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat() if pd.notna(value) else None
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def config_hash(cfg: Mapping[str, Any], n: int = 24) -> str:
    blob = json.dumps(json_safe(dict(cfg)), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:n]


def _to_df(obj: pd.DataFrame | str | Path | None) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, (str, Path)):
        p = Path(obj)
        if not p.exists():
            return pd.DataFrame()
        if p.suffix.lower() == ".parquet":
            return pd.read_parquet(p)
        if p.suffix.lower() in {".json", ".jsonl"}:
            return pd.read_json(p)
        return pd.read_csv(p)
    return obj.copy()


def _write_parquet_safe(df: pd.DataFrame, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(p, index=False)
        return str(p)
    except Exception:
        alt = p.with_suffix(".csv")
        df.to_csv(alt, index=False)
        return str(alt)


def _write_json_safe(payload: Mapping[str, Any], path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(json_safe(dict(payload)), sort_keys=True, indent=2, default=str),
        encoding="utf-8",
    )
    return str(p)


@dataclass(frozen=True)
class RegimeFeaturesConfig:
    benchmark_mode: str = "internal_equal_weight"  # internal_equal_weight | internal_cap_weight | external_index
    lookbacks_vol: tuple[int, ...] = (10, 20, 63)
    lookbacks_trend: tuple[int, ...] = (20, 63)
    lookback_corr: int = 20
    lookback_norm: int = 252
    lookback_breadth: tuple[int, ...] = (50, 200)
    winsor_limits: tuple[float, float] = (0.01, 0.99)

    smoothing_half_life_by_family: Mapping[str, float] = field(
        default_factory=lambda: {
            "vol": 5.0,
            "vov": 5.0,
            "trend": 8.0,
            "breadth": 5.0,
            "dispersion": 5.0,
            "corr": 8.0,
            "illiq": 5.0,
            "stress": 8.0,
        }
    )

    min_universe_size: int = 40
    min_symbol_coverage_corr: int = 25
    max_proxy_staleness: int = 5
    redundancy_corr_threshold: float = 0.95

    use_rank_gauss: bool = False
    use_corr_shrinkage: bool = True
    use_equal_weight_or_liquidity_weight: str = "equal_weight"

    policy_version: str = "v1"
    random_seed: int = 42
    run_id: str = ""


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def load_market_panel(
    market_prices: pd.DataFrame | str | Path,
    daily_bars: pd.DataFrame | str | Path | None = None,
    universe_history: pd.DataFrame | str | Path | None = None,
) -> pd.DataFrame:
    prices = _to_df(market_prices)
    bars = _to_df(daily_bars)
    univ = _to_df(universe_history)

    if len(prices) == 0 and len(bars) == 0:
        raise ValueError("need market_prices or daily_bars input")

    base = prices if len(prices) else bars
    out = base.copy()

    rename = {}
    if "asset" in out.columns and "symbol" not in out.columns:
        rename["asset"] = "symbol"
    if "adj_close" in out.columns and "close" not in out.columns:
        rename["adj_close"] = "close"
    out = out.rename(columns=rename)

    _require_columns(out, ["date", "symbol", "close"], "market panel")

    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")

    if "volume" not in out.columns and len(bars):
        bars2 = bars.copy()
        if "asset" in bars2.columns and "symbol" not in bars2.columns:
            bars2 = bars2.rename(columns={"asset": "symbol"})
        if "date" in bars2.columns and "symbol" in bars2.columns and "volume" in bars2.columns:
            bars2["date"] = pd.to_datetime(bars2["date"])
            out = out.merge(bars2[["date", "symbol", "volume"]], on=["date", "symbol"], how="left")

    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    else:
        out["volume"] = np.nan

    if "dollar_volume" in out.columns:
        out["dollar_volume"] = pd.to_numeric(out["dollar_volume"], errors="coerce")
    else:
        out["dollar_volume"] = out["close"] * out["volume"]

    out = out.sort_values(["symbol", "date"]).drop_duplicates(["date", "symbol"], keep="last")

    # PIT universe eligibility.
    if len(univ):
        u = univ.copy()
        if "asset" in u.columns and "symbol" not in u.columns:
            u = u.rename(columns={"asset": "symbol"})
        _require_columns(u, ["date", "symbol"], "universe_history")
        u["date"] = pd.to_datetime(u["date"])
        u["symbol"] = u["symbol"].astype(str)
        if "eligible" not in u.columns:
            for c in ["is_eligible", "eligible_flag"]:
                if c in u.columns:
                    u["eligible"] = u[c]
                    break
        if "eligible" not in u.columns:
            u["eligible"] = 1
        u["eligible"] = pd.to_numeric(u["eligible"], errors="coerce").fillna(0).clip(0, 1)
        out = out.merge(u[["date", "symbol", "eligible"]], on=["date", "symbol"], how="left")
        out["eligible"] = out["eligible"].fillna(0).astype(int)
    else:
        out["eligible"] = 1

    out["ret_1d"] = out.groupby("symbol", sort=False)["close"].pct_change()
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    return out


def _median_abs_dev(x: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))


def _rolling_robust_zscore(s: pd.Series, lookback: int, eps: float = 1e-8) -> pd.Series:
    med = s.rolling(lookback, min_periods=max(20, lookback // 5)).median().shift(1)
    mad = s.rolling(lookback, min_periods=max(20, lookback // 5)).apply(_median_abs_dev, raw=True).shift(1)
    return (s - med) / (1.4826 * mad + eps)


def _winsorize_causal(s: pd.Series, lower_q: float, upper_q: float, lookback: int) -> pd.Series:
    lo = s.rolling(lookback, min_periods=max(20, lookback // 5)).quantile(lower_q).shift(1)
    hi = s.rolling(lookback, min_periods=max(20, lookback // 5)).quantile(upper_q).shift(1)
    return s.clip(lower=lo, upper=hi)


def _ema_half_life(s: pd.Series, half_life: float) -> pd.Series:
    if half_life <= 0:
        return s
    alpha = 1.0 - 2.0 ** (-1.0 / float(half_life))
    return s.ewm(alpha=alpha, adjust=False).mean()


def _rank_gauss_approx(s: pd.Series, lookback: int, eps: float = 1e-4) -> pd.Series:
    try:
        from scipy.stats import norm

        def _rank_last(x: np.ndarray) -> float:
            if len(x) == 0:
                return np.nan
            val = x[-1]
            if not np.isfinite(val):
                return np.nan
            arr = x[np.isfinite(x)]
            if len(arr) == 0:
                return np.nan
            rank = (arr <= val).sum() / max(len(arr), 1)
            rank = min(max(rank, eps), 1.0 - eps)
            return float(norm.ppf(rank))

        return s.rolling(lookback, min_periods=max(20, lookback // 5)).apply(_rank_last, raw=True)
    except Exception:
        return s

def compute_regime_feature_families(
    panel: pd.DataFrame,
    cfg: RegimeFeaturesConfig,
) -> pd.DataFrame:
    out = pd.DataFrame({"date": sorted(panel["date"].unique())})

    # Universe and benchmark (equal-weight on eligible names).
    elig = panel[panel["eligible"] > 0].copy()
    universe_size = elig.groupby("date")["symbol"].nunique().rename("universe_size_t")
    bench_ret = elig.groupby("date")["ret_1d"].mean().rename("benchmark_ret_1d")

    out = out.merge(universe_size.reset_index(), on="date", how="left")
    out = out.merge(bench_ret.reset_index(), on="date", how="left")

    # Volatility and trend families.
    for k in cfg.lookbacks_vol:
        out[f"vol_{k}d"] = np.sqrt(252.0 * out["benchmark_ret_1d"].pow(2).rolling(k, min_periods=max(5, k // 3)).mean())
    if 20 in cfg.lookbacks_vol:
        out["vov_20d_on_vol_20d"] = out["vol_20d"].rolling(20, min_periods=8).std()
    else:
        base = f"vol_{cfg.lookbacks_vol[0]}d"
        out["vov_20d_on_vol_20d"] = out[base].rolling(20, min_periods=8).std()

    for k in cfg.lookbacks_trend:
        out[f"trend_ret_{k}d"] = out["benchmark_ret_1d"].rolling(k, min_periods=max(5, k // 3)).mean()

    # Breadth family.
    work = panel.copy().sort_values(["symbol", "date"])
    for k in cfg.lookback_breadth:
        ma_col = f"ma_{k}"
        work[ma_col] = work.groupby("symbol", sort=False)["close"].transform(lambda s: s.rolling(k, min_periods=max(5, k // 4)).mean())
        work[f"above_ma_{k}"] = (work["close"] > work[ma_col]).astype(float)
        b = (
            work[work["eligible"] > 0]
            .groupby("date")[f"above_ma_{k}"]
            .mean()
            .rename(f"pct_above_ma_{k}")
            .reset_index()
        )
        out = out.merge(b, on="date", how="left")

    # Dispersion family (robust MAD cross-sectional returns).
    disp_rows = []
    for d, g in work[work["eligible"] > 0].groupby("date", sort=True):
        x = pd.to_numeric(g["ret_1d"], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            disp = np.nan
        else:
            med = np.median(x)
            disp = 1.4826 * np.median(np.abs(x - med))
        disp_rows.append({"date": d, "dispersion_mad": disp})
    out = out.merge(pd.DataFrame(disp_rows), on="date", how="left")

    # Illiquidity family.
    work["illiq_raw"] = np.abs(work["ret_1d"]) / np.maximum(pd.to_numeric(work["dollar_volume"], errors="coerce"), 1e-9)
    illiq = (
        work[work["eligible"] > 0]
        .groupby("date")["illiq_raw"]
        .median()
        .rename("illiq_median")
        .reset_index()
    )
    out = out.merge(illiq, on="date", how="left")

    # Stress proxy family.
    q10 = out["benchmark_ret_1d"].rolling(63, min_periods=20).quantile(0.10)
    tail_hit = (out["benchmark_ret_1d"] < q10).astype(float)
    out["tail_loss_freq_63d_10pct"] = tail_hit.rolling(63, min_periods=20).mean()

    # Correlation concentration family.
    corr_rows = []
    returns_pivot = work[work["eligible"] > 0].pivot(index="date", columns="symbol", values="ret_1d").sort_index()
    dates = returns_pivot.index.to_list()

    for i, d in enumerate(dates):
        if i + 1 < cfg.lookback_corr:
            corr_rows.append({"date": d, "corr_conc": np.nan, "n_symbols_corr_window_t": 0})
            continue

        wnd = returns_pivot.iloc[i + 1 - cfg.lookback_corr : i + 1].copy()
        # Keep symbols with decent coverage in the window.
        keep = wnd.columns[wnd.notna().sum(axis=0) >= max(5, int(cfg.lookback_corr * 0.6))]
        wnd = wnd[keep]
        n_symbols = int(wnd.shape[1])

        if n_symbols < 2:
            corr_rows.append({"date": d, "corr_conc": np.nan, "n_symbols_corr_window_t": n_symbols})
            continue

        C = wnd.corr().to_numpy(dtype=float)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 1.0)

        if cfg.use_corr_shrinkage:
            delta = 0.10
            C = (1.0 - delta) * C + delta * np.eye(C.shape[0])

        try:
            eigvals = np.linalg.eigvalsh(C)
            lam1 = float(np.nanmax(eigvals))
            conc = float(lam1 / max(np.trace(C), 1e-12))
        except Exception:
            conc = np.nan

        corr_rows.append({"date": d, "corr_conc": conc, "n_symbols_corr_window_t": n_symbols})

    out = out.merge(pd.DataFrame(corr_rows), on="date", how="left")

    return out.sort_values("date").reset_index(drop=True)


def _feature_family(feature_name: str) -> str:
    if feature_name.startswith("vol_"):
        return "vol"
    if feature_name.startswith("vov_"):
        return "vov"
    if feature_name.startswith("trend_"):
        return "trend"
    if feature_name.startswith("pct_above_ma"):
        return "breadth"
    if feature_name.startswith("dispersion"):
        return "dispersion"
    if feature_name.startswith("corr_"):
        return "corr"
    if feature_name.startswith("illiq"):
        return "illiq"
    if feature_name.startswith("tail_"):
        return "stress"
    return "other"


def transform_regime_features(raw: pd.DataFrame, cfg: RegimeFeaturesConfig) -> pd.DataFrame:
    out = raw.copy()
    feature_cols = [
        c
        for c in out.columns
        if c
        not in {
            "date",
            "universe_size_t",
            "benchmark_ret_1d",
            "n_symbols_corr_window_t",
        }
    ]

    low_q, high_q = cfg.winsor_limits
    for c in feature_cols:
        s = pd.to_numeric(out[c], errors="coerce")
        s = _winsorize_causal(s, low_q, high_q, cfg.lookback_norm)
        s = _rolling_robust_zscore(s, cfg.lookback_norm)
        if cfg.use_rank_gauss:
            s = _rank_gauss_approx(s, cfg.lookback_norm)
        family = _feature_family(c)
        half_life = float(cfg.smoothing_half_life_by_family.get(family, 0.0))
        s = _ema_half_life(s, half_life)
        out[c] = s

    return out


def select_canonical_feature_set(transformed: pd.DataFrame, cfg: RegimeFeaturesConfig) -> tuple[list[str], dict[str, Any]]:
    candidates = [
        c
        for c in transformed.columns
        if c
        not in {
            "date",
            "universe_size_t",
            "benchmark_ret_1d",
            "n_symbols_corr_window_t",
        }
    ]

    dropped: dict[str, str] = {}
    keep = []
    for c in candidates:
        s = pd.to_numeric(transformed[c], errors="coerce")
        miss = float(s.isna().mean())
        var = float(np.nanvar(s))
        if miss > 0.40:
            dropped[c] = "high_missing"
            continue
        if not np.isfinite(var) or var <= 1e-10:
            dropped[c] = "near_zero_variance"
            continue
        keep.append(c)

    if len(keep) >= 2:
        corr = transformed[keep].corr(method="spearman").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        for col in upper.columns:
            bad = upper.index[(upper[col] >= cfg.redundancy_corr_threshold) & upper[col].notna()].tolist()
            for r in bad:
                if col not in keep or r not in keep:
                    continue
                # Keep the more stable one (lower missingness, then larger variance).
                miss_col = float(transformed[col].isna().mean())
                miss_r = float(transformed[r].isna().mean())
                var_col = float(np.nanvar(pd.to_numeric(transformed[col], errors="coerce")))
                var_r = float(np.nanvar(pd.to_numeric(transformed[r], errors="coerce")))
                if miss_col < miss_r:
                    drop = r
                elif miss_r < miss_col:
                    drop = col
                else:
                    drop = col if var_col <= var_r else r
                if drop in keep:
                    keep.remove(drop)
                    dropped[drop] = f"redundant_corr>={cfg.redundancy_corr_threshold}"

    report = {
        "features_candidates": sorted(candidates),
        "features_selected": sorted(keep),
        "features_dropped": {k: dropped[k] for k in sorted(dropped)},
        "n_candidates": int(len(candidates)),
        "n_selected": int(len(keep)),
    }

    return sorted(keep), report

def build_confidence_and_flags(df: pd.DataFrame, selected_features: list[str], cfg: RegimeFeaturesConfig) -> pd.DataFrame:
    out = df.copy()

    out["vol_family_valid"] = out[[c for c in selected_features if c.startswith("vol_")]].notna().all(axis=1) if any(c.startswith("vol_") for c in selected_features) else True
    out["breadth_family_valid"] = out[[c for c in selected_features if c.startswith("pct_above_ma")]].notna().all(axis=1) if any(c.startswith("pct_above_ma") for c in selected_features) else True
    out["dispersion_family_valid"] = out[[c for c in selected_features if c.startswith("dispersion")]].notna().all(axis=1) if any(c.startswith("dispersion") for c in selected_features) else True
    out["corr_family_valid"] = out[[c for c in selected_features if c.startswith("corr_")]].notna().all(axis=1) if any(c.startswith("corr_") for c in selected_features) else True
    out["illiq_family_valid"] = out[[c for c in selected_features if c.startswith("illiq")]].notna().all(axis=1) if any(c.startswith("illiq") for c in selected_features) else True
    out["stress_family_valid"] = out[[c for c in selected_features if c.startswith("tail_")]].notna().all(axis=1) if any(c.startswith("tail_") for c in selected_features) else True

    valid_cols = [
        "vol_family_valid",
        "breadth_family_valid",
        "dispersion_family_valid",
        "corr_family_valid",
        "illiq_family_valid",
        "stress_family_valid",
    ]

    c_u = np.clip(pd.to_numeric(out["universe_size_t"], errors="coerce") / max(cfg.min_universe_size, 1), 0.0, 1.0)
    if "n_symbols_corr_window_t" in out.columns:
        c_corr = np.clip(pd.to_numeric(out["n_symbols_corr_window_t"], errors="coerce") / max(cfg.min_symbol_coverage_corr, 1), 0.0, 1.0)
    else:
        c_corr = pd.Series(1.0, index=out.index)

    miss_rate = out[selected_features].isna().mean(axis=1) if selected_features else pd.Series(1.0, index=out.index)
    c_miss = 1.0 - miss_rate
    c_family = out[valid_cols].astype(float).mean(axis=1)

    out["confidence_score_t"] = (0.30 * c_u + 0.20 * c_corr + 0.25 * c_miss + 0.25 * c_family).clip(0.0, 1.0)
    out["low_confidence_flag_t"] = out["confidence_score_t"] < 0.50

    return out


def build_feature_dictionary(selected_features: list[str]) -> dict[str, Any]:
    formulas = {
        "vol": "sqrt(252 * rolling_mean(benchmark_ret^2, k))",
        "vov": "rolling_std(vol_k, 20)",
        "trend": "rolling_mean(benchmark_ret, k)",
        "breadth": "mean(1{close > MA_k}) over eligible universe",
        "dispersion": "1.4826 * median(|r_i - median(r)|)",
        "corr": "lambda1(corr_matrix) / trace(corr_matrix)",
        "illiq": "median(|ret| / max(dollar_volume, eps))",
        "stress": "rolling_mean(1{ret < rolling_q10(ret)})",
    }

    out = {}
    for f in selected_features:
        fam = _feature_family(f)
        out[f] = {
            "name": f,
            "family": fam,
            "formula": formulas.get(fam, "transformed_feature"),
            "frequency": "daily",
            "lag_policy": "point_in_time",
            "units": "normalized_zscore",
            "validity_conditions": "non_null and universe_coverage_ok",
        }
    return out


def persist_regime_feature_outputs(
    regime_features: pd.DataFrame,
    feature_dictionary: Mapping[str, Any],
    summary: Mapping[str, Any],
    selection_report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    output_dir: str | Path,
    run_id: str,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    return {
        "regime_features": _write_parquet_safe(regime_features, out / "regime_features.parquet"),
        "regime_feature_dictionary": _write_json_safe(dict(feature_dictionary), out / f"regime_feature_dictionary_{run_id}.json"),
        "regime_feature_summary": _write_json_safe(dict(summary), out / f"regime_feature_summary_{run_id}.json"),
        "feature_selection_report": _write_json_safe(dict(selection_report), out / f"feature_selection_report_{run_id}.json"),
        "manifest": _write_json_safe(dict(manifest), out / f"manifest_{run_id}.json"),
    }


def run_regime_features(
    market_prices: pd.DataFrame | str | Path,
    *,
    daily_bars: pd.DataFrame | str | Path | None = None,
    universe_history: pd.DataFrame | str | Path | None = None,
    config: RegimeFeaturesConfig | None = None,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    data_snapshot_id: str = "unknown",
    macro_proxy: pd.DataFrame | str | Path | None = None,
    sector_indices: pd.DataFrame | str | Path | None = None,
    cross_asset_proxy: pd.DataFrame | str | Path | None = None,
    breadth_proxy: pd.DataFrame | str | Path | None = None,
) -> dict[str, Any]:
    _ = macro_proxy, sector_indices, cross_asset_proxy, breadth_proxy  # Optional hooks for phase-2 enrichment.

    t0 = time.perf_counter()
    cfg = config or RegimeFeaturesConfig()
    rid = run_id or cfg.run_id or run_id_with_prefix("regfeat")

    panel = load_market_panel(market_prices, daily_bars=daily_bars, universe_history=universe_history)
    raw = compute_regime_feature_families(panel, cfg)
    transformed = transform_regime_features(raw, cfg)

    selected, selection_report = select_canonical_feature_set(transformed, cfg)
    if not selected:
        raise ValueError("no canonical regime features survived filters")

    final = transformed[["date", "universe_size_t", "n_symbols_corr_window_t", *selected]].copy()
    final = build_confidence_and_flags(final, selected, cfg)
    final["run_id"] = rid

    feature_dict = build_feature_dictionary(selected)

    summary = {
        "run_id": rid,
        "timestamp": utc_now_iso(),
        "n_dates": int(len(final)),
        "n_features_selected": int(len(selected)),
        "features_selected": selected,
        "pct_low_confidence": float(final["low_confidence_flag_t"].mean()),
        "median_universe_size": float(pd.to_numeric(final["universe_size_t"], errors="coerce").median()),
        "median_corr_window_symbols": float(pd.to_numeric(final["n_symbols_corr_window_t"], errors="coerce").median()),
        "missingness_by_feature": {
            f: float(final[f].isna().mean()) for f in selected
        },
        "feature_corr_abs_mean": float(final[selected].corr(method="spearman").abs().mean().mean()) if len(selected) >= 2 else 0.0,
    }

    manifest = {
        "run_id": rid,
        "timestamp": utc_now_iso(),
        "config_hash": config_hash(asdict(cfg), n=24),
        "data_snapshot_id": data_snapshot_id,
        "benchmark_mode": cfg.benchmark_mode,
        "min_universe_size": cfg.min_universe_size,
        "lookbacks": {
            "vol": list(cfg.lookbacks_vol),
            "trend": list(cfg.lookbacks_trend),
            "corr": cfg.lookback_corr,
            "norm": cfg.lookback_norm,
            "breadth": list(cfg.lookback_breadth),
        },
        "winsor_limits": list(cfg.winsor_limits),
        "policy_version": cfg.policy_version,
        "runtime_sec": float(time.perf_counter() - t0),
    }

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_regime_feature_outputs(
            final,
            feature_dict,
            summary,
            selection_report,
            manifest,
            output_dir=output_dir,
            run_id=rid,
        )

    return {
        "regime_features": final,
        "regime_feature_dictionary": feature_dict,
        "regime_feature_summary": summary,
        "feature_selection_report": selection_report,
        "manifest": manifest,
        "artifacts": artifacts,
    }


__all__ = [
    "RegimeFeaturesConfig",
    "load_market_panel",
    "compute_regime_feature_families",
    "transform_regime_features",
    "select_canonical_feature_set",
    "build_confidence_and_flags",
    "build_feature_dictionary",
    "persist_regime_feature_outputs",
    "run_regime_features",
]
