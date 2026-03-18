"""
risk/factor_model.py - Canonical PIT multifactor risk model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorModelConfig:
    factors_continuous: tuple[str, ...] = ("size", "momentum", "volatility", "liquidity")
    market_factor_name: str = "beta_mkt"
    sector_col: str = "sector"
    return_col_candidates: tuple[str, ...] = ("return", "ret", "r", "return_1d")
    lookback_beta: int = 126
    min_obs_beta: int = 42
    lookback_factor_cov: int = 252
    min_obs_factor_cov: int = 40
    ewma_lambda_eps: float = 0.97
    eps_D: float = 1e-8
    winsor_z: float = 5.0
    mad_floor: float = 1e-8
    min_factor_breadth: int = 8
    near_const_std: float = 1e-10
    max_cond_BWB: float = 1e8
    ridge_alpha: float = 1e-8
    use_wls: bool = True
    drop_first_sector: bool = True
    coverage_returns_min: float = 0.60
    coverage_features_min: float = 0.70
    fallback_beta_value: float = 1.0
    factor_cov_target_type: str = "diag"
    factor_cov_shrinkage_method: str = "ledoit_wolf"
    factor_cov_provider: str = "cov_shrinkage"  # cov_shrinkage | sample
    psd_diag_floor: float = 1e-8
    psd_eps: float = 1e-10
    store_upper_triangle_only: bool = False


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
    import hashlib
    import json

    blob = json.dumps(json_safe(dict(cfg)), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:n]


def _to_df(obj: pd.DataFrame | str | Path | None) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, (str, Path)):
        p = Path(obj)
        if p.suffix.lower() == ".parquet":
            return pd.read_parquet(p)
        return pd.read_csv(p)
    return obj.copy()


def _to_returns_matrix(returns: pd.DataFrame | str | Path, ret_cols: tuple[str, ...]) -> pd.DataFrame:
    df = _to_df(returns)
    if len(df) == 0:
        raise ValueError("returns input is empty")

    if {"date", "symbol"}.issubset(df.columns):
        c = next((x for x in ret_cols if x in df.columns), None)
        if c is None:
            raise ValueError("returns long format needs return column")
        d = df[["date", "symbol", c]].copy()
        d["date"] = pd.to_datetime(d["date"])
        d["symbol"] = d["symbol"].astype(str)
        d[c] = pd.to_numeric(d[c], errors="coerce")
        return d.pivot_table(index="date", columns="symbol", values=c, aggfunc="last").sort_index()

    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        out = out.set_index("date")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.sort_index()


def _normalize_features(features: pd.DataFrame | str | Path | None, sector_col: str) -> pd.DataFrame:
    df = _to_df(features)
    if len(df) == 0:
        return pd.DataFrame(columns=["date", "symbol", sector_col])
    if {"date", "symbol", "factor_name", "exposure_value"}.issubset(df.columns):
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        d["symbol"] = d["symbol"].astype(str)
        d["factor_name"] = d["factor_name"].astype(str)
        d["exposure_value"] = pd.to_numeric(d["exposure_value"], errors="coerce")
        out = d.pivot_table(index=["date", "symbol"], columns="factor_name", values="exposure_value", aggfunc="last").reset_index()
        if sector_col not in out.columns:
            out[sector_col] = "UNKNOWN"
        return out
    if "date" not in df.columns:
        raise ValueError("features must include date")
    sym = "symbol" if "symbol" in df.columns else "asset"
    if sym not in df.columns:
        raise ValueError("features must include symbol/asset")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out[sym] = out[sym].astype(str)
    out = out.rename(columns={sym: "symbol"})
    if sector_col not in out.columns:
        out[sector_col] = "UNKNOWN"
    out[sector_col] = out[sector_col].fillna("UNKNOWN").astype(str)
    return out


def _normalize_sectors(sectors: pd.DataFrame | str | Path | None, sector_col: str) -> pd.DataFrame:
    df = _to_df(sectors)
    if len(df) == 0:
        return pd.DataFrame(columns=["date", "symbol", sector_col])
    if "date" not in df.columns:
        raise ValueError("sectors must include date")
    sym = "symbol" if "symbol" in df.columns else "asset"
    if sym not in df.columns:
        raise ValueError("sectors must include symbol/asset")
    sec = sector_col if sector_col in df.columns else ("sector" if "sector" in df.columns else "group")
    if sec not in df.columns:
        raise ValueError("sectors must include sector/group")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out[sym] = out[sym].astype(str)
    out[sec] = out[sec].fillna("UNKNOWN").astype(str)
    return out[["date", sym, sec]].rename(columns={sym: "symbol", sec: sector_col})


def _normalize_market(market_returns: pd.DataFrame | pd.Series | str | Path | None, rmat: pd.DataFrame) -> pd.Series:
    if market_returns is None:
        return rmat.mean(axis=1, skipna=True).rename("market_return")
    if isinstance(market_returns, (str, Path)):
        p = Path(market_returns)
        market_returns = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
    if isinstance(market_returns, pd.Series):
        s = pd.to_numeric(market_returns, errors="coerce")
        s.index = pd.to_datetime(s.index)
        return s.sort_index().rename("market_return")
    df = market_returns.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    col = next((c for c in ["market_return", "benchmark_return", "ret", "return", "r"] if c in df.columns), None)
    if col is None:
        if df.shape[1] != 1:
            raise ValueError("market_returns needs market_return-like column")
        col = df.columns[0]
    s = pd.to_numeric(df[col], errors="coerce")
    s.index = pd.to_datetime(s.index)
    return s.sort_index().rename("market_return")


def _zrobust(x: pd.Series, winsor_z: float, mad_floor: float) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    med = float(np.nanmedian(s.values)) if np.isfinite(s.values).any() else 0.0
    mad = float(np.nanmedian(np.abs(s.values - med))) if np.isfinite(s.values).any() else 0.0
    z = (s - med) / max(1.4826 * mad, mad_floor)
    return z.clip(-winsor_z, winsor_z)


def _beta_matrix(rmat: pd.DataFrame, mkt: pd.Series, lookback: int, min_obs: int) -> pd.DataFrame:
    m = pd.to_numeric(mkt, errors="coerce").reindex(rmat.index)
    out = pd.DataFrame(index=rmat.index, columns=rmat.columns, dtype=float)
    var_m = m.rolling(lookback, min_periods=min_obs).var()
    for sym in rmat.columns:
        r = pd.to_numeric(rmat[sym], errors="coerce")
        cov = r.rolling(lookback, min_periods=min_obs).cov(m)
        out[sym] = (cov / var_m).shift(1)
    return out


def _build_B(symbols: list[str], feat_d: pd.DataFrame, sec_d: pd.DataFrame, beta_d: pd.Series, cfg: FactorModelConfig) -> tuple[pd.DataFrame, pd.Series, dict[str, Any], float, int]:
    B = pd.DataFrame(index=pd.Index(symbols, name="symbol"))
    fb = pd.Series(False, index=B.index)

    feat = feat_d.drop_duplicates(subset=["symbol"], keep="last").set_index("symbol") if len(feat_d) else pd.DataFrame()
    sec_map = pd.Series(index=B.index, dtype=object)
    if len(feat) and cfg.sector_col in feat.columns:
        sec_map = feat[cfg.sector_col].reindex(B.index)
    if len(sec_d):
        sec = sec_d.drop_duplicates(subset=["symbol"], keep="last").set_index("symbol")
        sec_map = sec[cfg.sector_col].reindex(B.index).combine_first(sec_map)
    sec_map = sec_map.fillna("UNKNOWN").astype(str)

    beta = pd.to_numeric(beta_d.reindex(B.index), errors="coerce")
    gmed = float(np.nanmedian(beta.values)) if beta.notna().any() else float(cfg.fallback_beta_value)
    for _, idx in sec_map.groupby(sec_map).groups.items():
        miss = beta.loc[list(idx)].isna()
        if miss.any():
            sec_med = float(np.nanmedian(beta.loc[list(idx)].values)) if beta.loc[list(idx)].notna().any() else np.nan
            beta.loc[list(idx)] = beta.loc[list(idx)].fillna(sec_med if np.isfinite(sec_med) else gmed)
            fb.loc[list(idx)] = fb.loc[list(idx)] | miss
    beta = _zrobust(beta.fillna(gmed), cfg.winsor_z, cfg.mad_floor)
    B[cfg.market_factor_name] = beta

    n_nonmiss = 0
    n_tot = max(len(B) * max(len(cfg.factors_continuous), 1), 1)
    for f in cfg.factors_continuous:
        v = pd.to_numeric(feat[f].reindex(B.index), errors="coerce") if len(feat) and f in feat.columns else pd.Series(index=B.index, data=np.nan)
        n_nonmiss += int(v.notna().sum())
        v = _zrobust(v, cfg.winsor_z, cfg.mad_floor)
        m = v.isna()
        if m.any():
            v = v.fillna(0.0)
            fb.loc[m.index[m]] = True
        B[f] = v
    cov_feat = float(n_nonmiss / n_tot)

    dummies = pd.get_dummies(sec_map, prefix="sector", dtype=float).reindex(B.index).fillna(0.0)
    ref = None
    if cfg.drop_first_sector and dummies.shape[1] > 1:
        ref = sorted(dummies.columns)[0]
        dummies = dummies.drop(columns=[ref])
    if dummies.shape[1] > 0:
        B = pd.concat([B, dummies], axis=1)

    meta = {"k_input": int(B.shape[1]), "dropped_sector_reference": ref}
    return B, fb, meta, cov_feat, int(fb.sum())


def _prune(B: pd.DataFrame, cfg: FactorModelConfig) -> tuple[pd.DataFrame, list[str], bool, float]:
    x = B.copy()
    dropped: list[str] = []
    min_b = max(2, min(cfg.min_factor_breadth, max(2, int(0.25 * len(x)))))
    bad = []
    for c in x.columns:
        v = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
        if float(np.nanstd(v.values)) < cfg.near_const_std or int((np.abs(v.values) > 1e-12).sum()) < min_b:
            bad.append(c)
    if bad:
        x = x.drop(columns=bad, errors="ignore")
        dropped.extend(bad)
    if x.shape[1] == 0:
        return x, dropped, True, float("inf")

    colf = False
    for _ in range(max(1, x.shape[1] * 2)):
        if x.shape[1] <= 1:
            break
        m = x.values.astype(float)
        g = m.T @ m + np.eye(x.shape[1]) * cfg.ridge_alpha
        ev = np.clip(np.linalg.eigvalsh(g), 1e-15, None)
        cond = float(np.max(ev) / np.min(ev))
        if np.isfinite(cond) and cond <= cfg.max_cond_BWB:
            return x, dropped, colf, cond
        colf = True
        cor = np.nan_to_num(np.corrcoef(m, rowvar=False), nan=0.0)
        np.fill_diagonal(cor, 0.0)
        i, j = np.unravel_index(np.argmax(np.abs(cor)), cor.shape)
        ci, cj = x.columns[i], x.columns[j]
        bi = int((np.abs(pd.to_numeric(x[ci], errors="coerce").fillna(0.0).values) > 1e-12).sum())
        bj = int((np.abs(pd.to_numeric(x[cj], errors="coerce").fillna(0.0).values) > 1e-12).sum())
        drop = ci if bi <= bj else cj
        x = x.drop(columns=[drop], errors="ignore")
        dropped.append(drop)
    if x.shape[1] == 0:
        return x, dropped, True, float("inf")
    m = x.values.astype(float)
    g = m.T @ m + np.eye(x.shape[1]) * cfg.ridge_alpha
    ev = np.clip(np.linalg.eigvalsh(g), 1e-15, None)
    return x, dropped, colf, float(np.max(ev) / np.min(ev))


def _regress(B: pd.DataFrame, r: pd.Series, sv_prev: pd.Series | None, cfg: FactorModelConfig) -> tuple[pd.Series, pd.Series, float]:
    x = B.copy()
    y = pd.to_numeric(r.reindex(x.index), errors="coerce")
    keep = y.notna() & x.notna().all(axis=1)
    x = x.loc[keep]
    y = y.loc[keep]
    if len(x) < max(3, x.shape[1]):
        raise ValueError("insufficient rows for regression")

    if cfg.use_wls and sv_prev is not None:
        w = pd.to_numeric(sv_prev.reindex(x.index), errors="coerce")
        w = 1.0 / np.clip(w.fillna(w.median() if w.notna().any() else 1.0).values, cfg.eps_D, None)
    else:
        w = np.ones(len(x), dtype=float)
    sw = np.sqrt(np.clip(w, 1e-12, None))

    X = x.values.astype(float) * sw[:, None]
    Y = y.values.astype(float) * sw
    G = X.T @ X + np.eye(X.shape[1]) * cfg.ridge_alpha
    ev = np.clip(np.linalg.eigvalsh(G), 1e-15, None)
    cond = float(np.max(ev) / np.min(ev))
    b = X.T @ Y
    try:
        f = np.linalg.solve(G, b)
    except np.linalg.LinAlgError:
        f = np.linalg.pinv(G) @ b
    fhat = pd.Series(index=x.columns, data=f, dtype=float)
    resid = y - (x @ fhat)
    return fhat, resid.reindex(B.index), cond


def _update_spec(resid: pd.Series, prev: pd.Series | None, lam: float, floor: float) -> pd.Series:
    e2 = pd.to_numeric(resid, errors="coerce") ** 2
    if prev is None:
        init = float(np.nanpercentile(e2.dropna().values, 75)) if e2.notna().any() else max(floor * 10, 1e-6)
        p = pd.Series(index=resid.index, data=init)
    else:
        p = pd.to_numeric(prev.reindex(resid.index), errors="coerce")
        fb = float(np.nanpercentile(p.dropna().values, 75)) if p.notna().any() else max(floor * 10, 1e-6)
        p = p.fillna(fb)
    out = lam * p + (1.0 - lam) * e2.fillna(p)
    return out.clip(lower=floor).astype(float)


def _sym(m: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(0.5 * (m.values + m.values.T), index=m.index, columns=m.columns)


def _psd(m: pd.DataFrame, eps: float, d_floor: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    s = _sym(m)
    ev, vec = np.linalg.eigh(s.values)
    lam_max = float(np.max(ev)) if ev.size else 0.0
    floor = max(float(eps), float(d_floor) * max(lam_max, 1.0))
    mn0 = float(np.min(ev)) if ev.size else float("nan")
    evf = np.maximum(ev, floor)
    nfix = int(np.sum(ev < floor))
    out = vec @ np.diag(evf) @ vec.T
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, np.maximum(np.diag(out), d_floor))
    mn1 = float(np.min(np.linalg.eigvalsh(out))) if out.size else float("nan")
    return pd.DataFrame(out, index=m.index, columns=m.columns), {"min_eig_before": mn0, "min_eig_after": mn1, "num_fixed_eigs": nfix}

def _cond(m: pd.DataFrame) -> float:
    ev = np.clip(np.linalg.eigvalsh(_sym(m).values), 1e-15, None)
    return float(np.max(ev) / np.min(ev))


def _eff_rank(m: pd.DataFrame) -> float:
    ev = np.clip(np.linalg.eigvalsh(_sym(m).values), 0.0, None)
    tot = float(np.sum(ev))
    if tot <= 0:
        return float("nan")
    p = np.clip(ev / tot, 1e-15, 1.0)
    return float(np.exp(-np.sum(p * np.log(p))))


def _sample_omega(fret: pd.DataFrame, facs: list[str], cfg: FactorModelConfig) -> pd.DataFrame:
    cov = fret.reindex(columns=facs).cov(min_periods=2).reindex(index=facs, columns=facs).fillna(0.0)
    d = np.diag(cov.values)
    if np.any(d <= 0):
        np.fill_diagonal(cov.values, np.maximum(d, cfg.psd_diag_floor))
    cov = _sym(cov)
    cov, _ = _psd(cov, cfg.psd_eps, cfg.psd_diag_floor)
    return cov


def _omega(fret: pd.DataFrame, facs: list[str], cfg: FactorModelConfig) -> tuple[pd.DataFrame, str, bool]:
    if len(facs) == 0:
        raise ValueError("no factors for omega")
    win = fret.reindex(columns=facs).tail(cfg.lookback_factor_cov)
    if len(win) < 3:
        v = win.var(ddof=1).reindex(facs).fillna(cfg.psd_diag_floor).clip(lower=cfg.psd_diag_floor)
        return pd.DataFrame(np.diag(v.values), index=facs, columns=facs), "diag_short_history", True

    if cfg.factor_cov_provider.lower() == "cov_shrinkage":
        try:
            try:
                from .cov_shrinkage import CovShrinkageConfig, run_cov_shrinkage
            except Exception:
                from cov_shrinkage import CovShrinkageConfig, run_cov_shrinkage  # type: ignore

            min_obs = int(max(5, min(cfg.min_obs_factor_cov, len(win) - 1)))
            c = CovShrinkageConfig(
                lookback=min(cfg.lookback_factor_cov, len(win)),
                min_obs=min_obs,
                target_type=cfg.factor_cov_target_type,
                shrinkage_method=cfg.factor_cov_shrinkage_method,
                diag_floor=cfg.psd_diag_floor,
                eps_psd=cfg.psd_eps,
                fallback_policy="conservative",
            )
            out = run_cov_shrinkage(win, config=c)
            om = out["cov_shrunk"].reindex(index=facs, columns=facs)
            if om.isna().values.any():
                om = om.combine_first(_sample_omega(win, facs, cfg))
            om = om.fillna(0.0)
            om, _ = _psd(om, cfg.psd_eps, cfg.psd_diag_floor)
            return om, f"cov_shrinkage::{cfg.factor_cov_shrinkage_method}", bool(out["diagnostics"].get("fallback_used", False))
        except Exception:
            pass

    return _sample_omega(win, facs, cfg), "sample_cov", True


def _to_long(m: pd.DataFrame, date: pd.Timestamp, i: str, j: str, v: str, upper: bool) -> pd.DataFrame:
    a = m.values
    n = a.shape[0]
    if upper:
        ii, jj = np.triu_indices(n)
    else:
        ii, jj = np.indices((n, n))
        ii = ii.ravel()
        jj = jj.ravel()
    out = pd.DataFrame({"date": date, i: m.index.values[ii], j: m.columns.values[jj], v: a[ii, jj]})
    out[v] = pd.to_numeric(out[v], errors="coerce")
    return out


def _save_df(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return str(path)
    except Exception:
        p = path.with_suffix(".csv")
        df.to_csv(p, index=False)
        return str(p)


def _save_json(payload: Mapping[str, Any], path: Path) -> str:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(dict(payload)), sort_keys=True, indent=2, default=str), encoding="utf-8")
    return str(path)


def persist_factor_model_outputs(
    exposures: pd.DataFrame,
    factor_returns: pd.DataFrame,
    specific_risk: pd.DataFrame,
    factor_cov: pd.DataFrame,
    total_cov: pd.DataFrame,
    diagnostics: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "exposures": _save_df(exposures, out / "exposures.parquet"),
        "factor_returns": _save_df(factor_returns, out / "factor_returns.parquet"),
        "specific_risk": _save_df(specific_risk, out / "specific_risk.parquet"),
        "factor_cov": _save_df(factor_cov, out / "factor_cov.parquet"),
        "total_cov": _save_df(total_cov, out / "total_cov.parquet"),
        "diagnostics": _save_json(dict(diagnostics), out / "diagnostics.json"),
    }


def run_factor_model(
    returns: pd.DataFrame | str | Path,
    *,
    features: pd.DataFrame | str | Path | None = None,
    sectors: pd.DataFrame | str | Path | None = None,
    market_returns: pd.DataFrame | pd.Series | str | Path | None = None,
    config: FactorModelConfig | None = None,
    calc_date: str | pd.Timestamp | None = None,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or FactorModelConfig()
    rid = run_id or run_id_with_prefix("factorm")

    rmat = _to_returns_matrix(returns, cfg.return_col_candidates)
    if calc_date is not None:
        rmat = rmat.loc[rmat.index <= pd.Timestamp(calc_date)]
    if len(rmat) == 0 or rmat.shape[1] == 0:
        raise ValueError("empty return matrix after filtering")

    feats = _normalize_features(features, cfg.sector_col)
    secs = _normalize_sectors(sectors, cfg.sector_col)
    mkt = _normalize_market(market_returns, rmat).reindex(rmat.index)
    betas = _beta_matrix(rmat, mkt, cfg.lookback_beta, cfg.min_obs_beta)

    exp_rows: list[dict[str, Any]] = []
    fr_rows: list[dict[str, Any]] = []
    sr_rows: list[dict[str, Any]] = []
    fc_rows: list[pd.DataFrame] = []
    tc_rows: list[pd.DataFrame] = []
    d_rows: list[dict[str, Any]] = []

    fret_wide = pd.DataFrame(dtype=float)
    sv_prev = pd.Series(index=rmat.columns, data=np.nan)
    any_fb = False
    n_psd_fix = 0

    for date in rmat.index:
        ret = pd.to_numeric(rmat.loc[date], errors="coerce")
        valid = ret.notna()
        symbols = ret.index[valid].astype(str).tolist()
        if len(symbols) < 3:
            continue

        cov_r = float(valid.mean())
        r = ret.loc[symbols].astype(float)
        fd = feats[feats["date"] == date] if len(feats) else pd.DataFrame(columns=["date", "symbol", cfg.sector_col])
        sd = secs[secs["date"] == date] if len(secs) else pd.DataFrame(columns=["date", "symbol", cfg.sector_col])

        B_raw, fb_sym, bmeta, cov_f, n_fb = _build_B(symbols, fd, sd, betas.loc[date], cfg)
        B, dropped, colf, cond_pruned = _prune(B_raw, cfg)

        if B.shape[1] == 0:
            any_fb = True
            B = pd.DataFrame(index=B_raw.index, data={cfg.market_factor_name: 1.0})
            dropped = dropped + list(B_raw.columns)
            colf = True
            cond_pruned = float("inf")
            fb_sym[:] = True
            n_fb = int(len(fb_sym))

        sv_on = sv_prev.reindex(B.index)
        try:
            fhat, resid, cond_bwb = _regress(B, r, sv_on, cfg)
        except Exception:
            any_fb = True
            B = pd.DataFrame(index=B.index, data={cfg.market_factor_name: 1.0})
            fhat, resid, cond_bwb = _regress(B, r, sv_on, cfg)
            dropped = [c for c in B_raw.columns if c not in B.columns]
            colf = True

        sv_t = _update_spec(resid, sv_on, cfg.ewma_lambda_eps, cfg.eps_D)
        sv_prev.loc[sv_t.index] = sv_t.values

        fret_wide = pd.concat([fret_wide, pd.DataFrame([fhat.to_dict()], index=[date])], axis=0).sort_index()
        fret_wide = fret_wide[~fret_wide.index.duplicated(keep="last")]

        omega, omega_m, omega_fb = _omega(fret_wide, B.columns.tolist(), cfg)
        any_fb = any_fb or omega_fb

        d = sv_t.reindex(B.index).fillna(np.nanmedian(sv_t.values) if sv_t.notna().any() else cfg.eps_D).clip(lower=cfg.eps_D)
        D = pd.DataFrame(np.diag(d.values), index=B.index, columns=B.index)
        sigma = pd.DataFrame(B.values @ omega.values @ B.values.T + D.values, index=B.index, columns=B.index)
        sigma = _sym(sigma)
        sigma, psd = _psd(sigma, cfg.psd_eps, cfg.psd_diag_floor)
        n_psd_fix += int(psd["num_fixed_eigs"])

        cond_o = _cond(omega)
        cond_s = _cond(sigma)
        var_r = float(np.nanvar(r.values))
        var_e = float(np.nanvar(resid.reindex(r.index).values))
        pct = float(1.0 - var_e / var_r) if var_r > 1e-15 else float("nan")

        for sym in B.index:
            fbs = bool(fb_sym.reindex([sym]).fillna(True).iloc[0])
            for fac in B.columns:
                src = "rolling_beta" if fac == cfg.market_factor_name else ("categorical_dummy" if fac.startswith("sector_") else "continuous_structural")
                exp_rows.append({"date": date, "asset_id": sym, "factor_name": fac, "exposure_value": float(B.loc[sym, fac]), "source_type": src, "fallback_used": fbs})

        for fac, fr in fhat.items():
            fr_rows.append({"date": date, "factor_name": str(fac), "factor_return": float(fr)})

        for sym, sv in sv_t.items():
            sr_rows.append({"date": date, "asset_id": str(sym), "spec_var": float(sv), "fallback_used": bool(fb_sym.reindex([sym]).fillna(True).iloc[0])})

        fc_rows.append(_to_long(omega, date, "factor_i", "factor_j", "cov_value", cfg.store_upper_triangle_only))
        tc_rows.append(_to_long(sigma, date, "asset_i", "asset_j", "cov_value", cfg.store_upper_triangle_only))

        d_rows.append(
            {
                "date": date,
                "N_input": int(len(ret.index)),
                "N_effective": int(len(symbols)),
                "K_input": int(bmeta["k_input"]),
                "K_used": int(B.shape[1]),
                "coverage_returns": cov_r,
                "coverage_features": cov_f,
                "num_fallback_symbols": int(n_fb),
                "dropped_factors": dropped,
                "collinearity_flag": bool(colf),
                "cond_BWB": float(cond_bwb),
                "cond_B_pruned": float(cond_pruned),
                "cond_Omega_f": float(cond_o),
                "cond_Sigma": float(cond_s),
                "pct_variance_explained": pct,
                "residual_variance": var_e,
                "num_psd_fixes": int(psd["num_fixed_eigs"]),
                "effective_rank_Omega_f": _eff_rank(omega),
                "effective_rank_Sigma": _eff_rank(sigma),
                "omega_method": omega_m,
            }
        )

    if len(d_rows) == 0:
        raise ValueError("factor model produced no valid dates")

    exposures = pd.DataFrame(exp_rows)
    factor_returns = pd.DataFrame(fr_rows)
    specific_risk = pd.DataFrame(sr_rows)
    factor_cov = pd.concat(fc_rows, ignore_index=True) if fc_rows else pd.DataFrame(columns=["date", "factor_i", "factor_j", "cov_value"])
    total_cov = pd.concat(tc_rows, ignore_index=True) if tc_rows else pd.DataFrame(columns=["date", "asset_i", "asset_j", "cov_value"])

    diag_df = pd.DataFrame(d_rows).sort_values("date").reset_index(drop=True)
    last = diag_df.iloc[-1].to_dict()
    diagnostics = {
        "run_id": rid,
        "calc_date": str(pd.Timestamp(last["date"]).date()),
        "N_symbols_input": int(diag_df["N_input"].iloc[-1]),
        "N_symbols_effective": int(diag_df["N_effective"].iloc[-1]),
        "K_factors_input": int(diag_df["K_input"].iloc[-1]),
        "K_factors_used": int(diag_df["K_used"].iloc[-1]),
        "coverage_returns": float(diag_df["coverage_returns"].mean()),
        "coverage_features": float(diag_df["coverage_features"].mean()),
        "num_fallback_symbols": int(diag_df["num_fallback_symbols"].sum()),
        "cond_BWB": float(diag_df["cond_BWB"].iloc[-1]),
        "cond_Omega_f": float(diag_df["cond_Omega_f"].iloc[-1]),
        "cond_Sigma": float(diag_df["cond_Sigma"].iloc[-1]),
        "pct_variance_explained": float(diag_df["pct_variance_explained"].mean()),
        "residual_variance_summary": {
            "mean": float(diag_df["residual_variance"].mean()),
            "median": float(diag_df["residual_variance"].median()),
            "p95": float(diag_df["residual_variance"].quantile(0.95)),
        },
        "num_psd_fixes": int(n_psd_fix),
        "fallback_used": bool(any_fb),
        "effective_rank_Omega_f": float(diag_df["effective_rank_Omega_f"].iloc[-1]),
        "effective_rank_Sigma": float(diag_df["effective_rank_Sigma"].iloc[-1]),
        "coverage_alert": bool((diag_df["coverage_returns"].mean() < cfg.coverage_returns_min) or (diag_df["coverage_features"].mean() < cfg.coverage_features_min)),
        "dropped_factors_last_date": last.get("dropped_factors", []),
        "config_hash": config_hash(cfg.__dict__, n=24),
        "execution_timestamp": utc_now_iso(),
        "per_date": json_safe(d_rows),
    }

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_factor_model_outputs(exposures, factor_returns, specific_risk, factor_cov, total_cov, diagnostics, output_dir=output_dir)

    return {
        "exposures": exposures,
        "factor_returns": factor_returns,
        "specific_risk": specific_risk,
        "factor_cov": factor_cov,
        "total_cov": total_cov,
        "diagnostics": diagnostics,
        "artifacts": artifacts,
    }


__all__ = ["FactorModelConfig", "persist_factor_model_outputs", "run_factor_model"]
