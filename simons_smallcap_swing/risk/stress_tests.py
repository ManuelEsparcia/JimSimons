"""
risk/stress_tests.py - Scenario-based stress testing and governance gating.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StressTestsConfig:
    es_alpha: float = 0.05
    capital: float = 100_000_000.0
    phi_adv: float = 0.20
    trade_fraction: float = 1.0
    c1_liq: float = 0.002
    c2_liq: float = 0.02
    coverage_adv_min: float = 0.95
    market_sigma_levels: tuple[float, ...] = (1.0, 2.0, 3.0)
    vol_kappa_levels: tuple[float, ...] = (0.25, 0.50, 1.00)
    corr_delta_levels: tuple[float, ...] = (0.10, 0.20, 0.30)
    liquidity_eta_levels: tuple[float, ...] = (0.75, 0.50, 0.25)
    factor_shocks: Mapping[str, float] = field(
        default_factory=lambda: {
            "size": -0.02,
            "liquidity": -0.02,
            "momentum": -0.02,
            "volatility": +0.02,
        }
    )
    combined_market_sigma: float = 2.0
    combined_delta_rho: float = 0.20
    combined_eta_adv: float = 0.50
    combined_spread_mult: float = 1.5
    pnl_warn_limit: float = 0.03
    pnl_hard_limit: float = 0.05
    mdd_warn_limit: float = 0.10
    mdd_hard_limit: float = 0.15
    es_warn_limit: float = 0.04
    es_hard_limit: float = 0.06
    pr_warn_limit: float = 0.08
    pr_hard_limit: float = 0.10
    dtl_warn_limit: float = 15.0
    dtl_hard_limit: float = 20.0
    psd_diag_floor: float = 1e-8
    psd_eps: float = 1e-10


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


def _normalize_positions(positions: pd.DataFrame | str | Path, *, calc_date: pd.Timestamp | None, capital: float) -> tuple[pd.Series, pd.Timestamp]:
    df = _to_df(positions)
    if len(df) == 0:
        raise ValueError("positions input is empty")

    sym = "symbol" if "symbol" in df.columns else "asset"
    if sym not in df.columns:
        raise ValueError("positions must include symbol/asset")

    d = df.copy()
    if "date" not in d.columns:
        d["date"] = calc_date if calc_date is not None else pd.Timestamp.utcnow().normalize()
    d["date"] = pd.to_datetime(d["date"])
    d[sym] = d[sym].astype(str)

    if calc_date is not None:
        d = d[d["date"] <= calc_date]
    if len(d) == 0:
        raise ValueError("no positions at or before calc_date")

    snap_date = pd.Timestamp(d["date"].max())
    s = d[d["date"] == snap_date].drop_duplicates(subset=[sym], keep="last").set_index(sym)

    if "weight" in s.columns:
        w = pd.to_numeric(s["weight"], errors="coerce")
    elif "w" in s.columns:
        w = pd.to_numeric(s["w"], errors="coerce")
    elif "w_current" in s.columns:
        w = pd.to_numeric(s["w_current"], errors="coerce")
    elif "notional" in s.columns:
        n = pd.to_numeric(s["notional"], errors="coerce")
        w = n / max(float(capital), 1e-12)
    else:
        raise ValueError("positions require weight/w/w_current/notional")

    w = w.fillna(0.0)
    return w.astype(float), snap_date


def _to_returns_matrix(returns: pd.DataFrame | str | Path) -> pd.DataFrame:
    df = _to_df(returns)
    if len(df) == 0:
        raise ValueError("returns input is empty")

    if {"date", "symbol"}.issubset(df.columns):
        c = next((x for x in ["return", "ret", "r", "return_1d"] if x in df.columns), None)
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


def _normalize_exposures(exposures: pd.DataFrame | str | Path | None, *, calc_date: pd.Timestamp) -> pd.DataFrame:
    df = _to_df(exposures)
    if len(df) == 0:
        return pd.DataFrame()
    if {"date", "factor_name", "exposure_value"}.issubset(df.columns) and (
        "symbol" in df.columns or "asset_id" in df.columns
    ):
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        d = d[d["date"] <= calc_date]
        if len(d) == 0:
            return pd.DataFrame()
        d = d[d["date"] == d["date"].max()]
        if "symbol" not in d.columns and "asset_id" in d.columns:
            d = d.rename(columns={"asset_id": "symbol"})
        d["symbol"] = d["symbol"].astype(str)
        d["factor_name"] = d["factor_name"].astype(str)
        d["exposure_value"] = pd.to_numeric(d["exposure_value"], errors="coerce")
        return d.pivot_table(index="symbol", columns="factor_name", values="exposure_value", aggfunc="last")

    if "date" not in df.columns:
        raise ValueError("exposures must include date")
    sym = "symbol" if "symbol" in df.columns else ("asset" if "asset" in df.columns else "asset_id")
    if sym not in df.columns:
        raise ValueError("exposures must include symbol/asset")
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= calc_date]
    if len(d) == 0:
        return pd.DataFrame()
    d = d[d["date"] == d["date"].max()].drop_duplicates(subset=[sym], keep="last").set_index(sym)
    fac = [c for c in d.columns if c not in {"date", "sector", "group"}]
    for c in fac:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[fac]


def _normalize_liquidity(liquidity: pd.DataFrame | str | Path | None, *, calc_date: pd.Timestamp) -> pd.DataFrame:
    df = _to_df(liquidity)
    if len(df) == 0:
        return pd.DataFrame(columns=["adv_usd", "spread_bps"])
    if "date" not in df.columns:
        raise ValueError("liquidity must include date")
    sym = "symbol" if "symbol" in df.columns else "asset"
    if sym not in df.columns:
        raise ValueError("liquidity must include symbol/asset")

    adv = "adv_usd" if "adv_usd" in df.columns else ("adv" if "adv" in df.columns else "dollar_adv")
    if adv not in df.columns:
        raise ValueError("liquidity must include adv_usd/adv/dollar_adv")

    spread = "spread_bps" if "spread_bps" in df.columns else None
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= calc_date]
    if len(d) == 0:
        return pd.DataFrame(columns=["adv_usd", "spread_bps"])
    d = d[d["date"] == d["date"].max()].drop_duplicates(subset=[sym], keep="last").set_index(sym)
    d[adv] = pd.to_numeric(d[adv], errors="coerce")
    if spread is None:
        d["spread_bps"] = 0.0
    else:
        d["spread_bps"] = pd.to_numeric(d[spread], errors="coerce").fillna(0.0)
    return d[[adv, "spread_bps"]].rename(columns={adv: "adv_usd"})


def _sym(m: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(0.5 * (m.values + m.values.T), index=m.index, columns=m.columns)


def _psd(m: pd.DataFrame, eps: float, d_floor: float) -> pd.DataFrame:
    s = _sym(m)
    ev, vec = np.linalg.eigh(s.values)
    lam_max = float(np.max(ev)) if ev.size else 0.0
    floor = max(float(eps), float(d_floor) * max(lam_max, 1.0))
    ev = np.maximum(ev, floor)
    out = vec @ np.diag(ev) @ vec.T
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, np.maximum(np.diag(out), d_floor))
    return pd.DataFrame(out, index=m.index, columns=m.columns)


def _nearest_corr_psd(corr: pd.DataFrame, eps: float = 1e-10) -> pd.DataFrame:
    c = _sym(corr)
    np.fill_diagonal(c.values, 1.0)
    ev, vec = np.linalg.eigh(c.values)
    ev = np.maximum(ev, eps)
    x = vec @ np.diag(ev) @ vec.T
    d = np.sqrt(np.clip(np.diag(x), eps, None))
    x = x / np.outer(d, d)
    np.fill_diagonal(x, 1.0)
    return pd.DataFrame(x, index=c.index, columns=c.columns)


def _cov_corr_shock(sigma: pd.DataFrame, delta_rho: float, cfg: StressTestsConfig) -> pd.DataFrame:
    s = _psd(sigma, cfg.psd_eps, cfg.psd_diag_floor)
    std = np.sqrt(np.clip(np.diag(s.values), cfg.psd_diag_floor, None))
    corr = s.values / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    raw = corr.copy()
    n = raw.shape[0]
    mask = ~np.eye(n, dtype=bool)
    raw[mask] = np.clip(raw[mask] + float(delta_rho), -1.0, 1.0)
    np.fill_diagonal(raw, 1.0)
    corr_psd = _nearest_corr_psd(pd.DataFrame(raw, index=s.index, columns=s.columns), eps=cfg.psd_eps)
    sig = np.outer(std, std) * corr_psd.values
    return _psd(pd.DataFrame(sig, index=s.index, columns=s.columns), cfg.psd_eps, cfg.psd_diag_floor)

def _base_sigma(covariance: pd.DataFrame | str | Path | None, returns: pd.DataFrame, symbols: pd.Index, cfg: StressTestsConfig) -> pd.DataFrame:
    if covariance is None:
        c = returns.reindex(columns=symbols).cov(min_periods=2)
        c = c.reindex(index=symbols, columns=symbols).fillna(0.0)
        return _psd(c, cfg.psd_eps, cfg.psd_diag_floor)

    df = _to_df(covariance)
    if {"asset_i", "asset_j", "cov_value"}.issubset(df.columns):
        d = df.copy()
        if "date" in d.columns:
            d["date"] = pd.to_datetime(d["date"])
            d = d[d["date"] == d["date"].max()]
        piv = d.pivot_table(index="asset_i", columns="asset_j", values="cov_value", aggfunc="last")
        piv = piv.reindex(index=symbols, columns=symbols).fillna(0.0)
        return _psd(piv, cfg.psd_eps, cfg.psd_diag_floor)

    mat = df.copy()
    if "date" in mat.columns:
        mat = mat.drop(columns=["date"])
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("covariance matrix must be square")
    if isinstance(mat.index, pd.RangeIndex):
        mat.index = symbols[: len(mat)]
    if isinstance(mat.columns, pd.RangeIndex):
        mat.columns = symbols[: len(mat.columns)]
    mat = mat.reindex(index=symbols, columns=symbols)
    mat = mat.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return _psd(mat, cfg.psd_eps, cfg.psd_diag_floor)


def _es_loss(ret: pd.Series, alpha: float) -> float:
    x = pd.to_numeric(ret, errors="coerce").dropna()
    if len(x) == 0:
        return float("nan")
    q = float(np.quantile(x.values, alpha))
    tail = x[x <= q]
    if len(tail) == 0:
        return float("nan")
    return float(-tail.mean())


def _dd_metrics(ret: pd.Series, nav0: float = 1.0) -> tuple[float, float]:
    x = pd.to_numeric(ret, errors="coerce").dropna()
    if len(x) == 0:
        return float("nan"), float("nan")
    nav = nav0 * np.cumprod(1.0 + x.values)
    hwm = np.maximum.accumulate(nav)
    dd = (nav - hwm) / np.clip(hwm, 1e-12, None)
    return float(dd[-1]), float(np.min(dd))


def _liq_damage(
    w: pd.Series,
    liq: pd.DataFrame,
    *,
    eta_adv: float,
    spread_mult: float,
    cfg: StressTestsConfig,
) -> tuple[float, float, float, float, float, float, bool]:
    if len(w) == 0:
        return 0.0, float("nan"), float("nan"), float("nan"), float("nan"), 0.0, True

    q = np.abs(w.values) * cfg.capital * cfg.trade_fraction
    pos = np.abs(w.values) * cfg.capital

    l = liq.reindex(w.index) if len(liq) else pd.DataFrame(index=w.index, data={"adv_usd": np.nan, "spread_bps": 0.0})
    adv = pd.to_numeric(l["adv_usd"], errors="coerce").values if "adv_usd" in l.columns else np.full(len(w), np.nan)
    spread = pd.to_numeric(l["spread_bps"], errors="coerce").fillna(0.0).values if "spread_bps" in l.columns else np.zeros(len(w))

    adv_s = np.clip(eta_adv, 1e-6, None) * adv
    valid = np.isfinite(adv_s) & (adv_s > 0)
    notional_total = float(np.sum(pos)) + 1e-12
    adv_covered = float(np.sum(pos[valid]) / notional_total)

    pr = np.full(len(w), np.nan, dtype=float)
    dtl = np.full(len(w), np.nan, dtype=float)
    pr[valid] = q[valid] / (cfg.phi_adv * adv_s[valid])
    dtl[valid] = pos[valid] / (cfg.phi_adv * adv_s[valid])

    impact = np.zeros(len(w), dtype=float)
    impact[valid] = cfg.c1_liq * pr[valid] + cfg.c2_liq * (pr[valid] ** 2)
    spread_cost = (spread_mult * spread / 1e4) * 0.5

    per_asset_cost = (impact + spread_cost) * np.abs(w.values)
    liq_cost = float(np.nansum(per_asset_cost))

    pr_ok = pr[np.isfinite(pr)]
    dtl_ok = dtl[np.isfinite(dtl)]
    pr_p95 = float(np.percentile(pr_ok, 95)) if pr_ok.size else float("nan")
    pr_max = float(np.max(pr_ok)) if pr_ok.size else float("nan")
    dtl_p95 = float(np.percentile(dtl_ok, 95)) if dtl_ok.size else float("nan")
    dtl_max = float(np.max(dtl_ok)) if dtl_ok.size else float("nan")

    fallback = bool(adv_covered < cfg.coverage_adv_min)
    return liq_cost, pr_p95, pr_max, dtl_p95, dtl_max, adv_covered, fallback


def _default_scenarios(base_ret: pd.Series, base_vol: float, expo: pd.DataFrame, cfg: StressTestsConfig) -> list[dict[str, Any]]:
    scen: list[dict[str, Any]] = []

    if len(base_ret) > 0:
        worst_day = base_ret.idxmin()
        scen.append(
            {
                "scenario_id": "hist_worst_day",
                "scenario_type": "historical",
                "kind": "replay",
                "start_date": str(pd.Timestamp(worst_day).date()),
                "end_date": str(pd.Timestamp(worst_day).date()),
                "severity_value": float(abs(base_ret.min())),
            }
        )

    if len(base_ret) >= 5:
        roll5 = base_ret.rolling(5).sum()
        end = roll5.idxmin()
        loc = base_ret.index.get_loc(end)
        start = base_ret.index[max(0, loc - 4)]
        scen.append(
            {
                "scenario_id": "hist_worst_5d",
                "scenario_type": "historical",
                "kind": "replay",
                "start_date": str(pd.Timestamp(start).date()),
                "end_date": str(pd.Timestamp(end).date()),
                "severity_value": float(abs(roll5.min())),
            }
        )

    for k in cfg.market_sigma_levels:
        scen.append(
            {
                "scenario_id": f"param_market_{str(k).replace('.', '_')}sigma",
                "scenario_type": "parametric",
                "kind": "market",
                "market_shock": float(-abs(k) * max(base_vol, 1e-8)),
                "severity_value": float(abs(k)),
            }
        )

    for kappa in cfg.vol_kappa_levels:
        scen.append(
            {
                "scenario_id": f"param_vol_kappa_{str(kappa).replace('.', '_')}",
                "scenario_type": "parametric",
                "kind": "vol",
                "kappa": float(kappa),
                "severity_value": float(abs(kappa)),
            }
        )

    for dr in cfg.corr_delta_levels:
        scen.append(
            {
                "scenario_id": f"param_corr_drho_{str(dr).replace('.', '_')}",
                "scenario_type": "parametric",
                "kind": "corr",
                "delta_rho": float(dr),
                "severity_value": float(abs(dr)),
            }
        )

    if len(expo):
        for f, sh in cfg.factor_shocks.items():
            if f in expo.columns:
                scen.append(
                    {
                        "scenario_id": f"param_factor_{f}",
                        "scenario_type": "parametric",
                        "kind": "factor",
                        "factor_name": str(f),
                        "factor_shock": float(sh),
                        "severity_value": float(abs(sh)),
                    }
                )

    for eta in cfg.liquidity_eta_levels:
        scen.append(
            {
                "scenario_id": f"op_liq_eta_{str(eta).replace('.', '_')}",
                "scenario_type": "operational",
                "kind": "liquidity",
                "eta_adv": float(eta),
                "spread_mult": 1.0,
                "severity_value": float(abs(1.0 - eta)),
            }
        )

    scen.append(
        {
            "scenario_id": "combined_market_corr_liq",
            "scenario_type": "parametric",
            "kind": "combined",
            "market_shock": float(-abs(cfg.combined_market_sigma) * max(base_vol, 1e-8)),
            "delta_rho": float(cfg.combined_delta_rho),
            "eta_adv": float(cfg.combined_eta_adv),
            "spread_mult": float(cfg.combined_spread_mult),
            "severity_value": float(abs(cfg.combined_market_sigma) + abs(cfg.combined_delta_rho) + abs(1.0 - cfg.combined_eta_adv)),
        }
    )

    return scen


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


def persist_stress_outputs(
    scenario_results: pd.DataFrame,
    breaches: pd.DataFrame,
    summary: Mapping[str, Any],
    scenario_definitions: list[dict[str, Any]],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "scenario_results": _save_df(scenario_results, out / "scenario_results.parquet"),
        "breaches": _save_df(breaches, out / "breaches.parquet"),
        "summary": _save_json(summary, out / "summary.json"),
        "scenario_definitions": _save_json({"scenarios": scenario_definitions}, out / "scenario_definitions.json"),
    }


def run_stress_tests(
    positions: pd.DataFrame | str | Path,
    *,
    returns: pd.DataFrame | str | Path,
    covariance: pd.DataFrame | str | Path | None = None,
    exposures: pd.DataFrame | str | Path | None = None,
    liquidity: pd.DataFrame | str | Path | None = None,
    scenarios: list[dict[str, Any]] | None = None,
    config: StressTestsConfig | None = None,
    calc_date: str | pd.Timestamp | None = None,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or StressTestsConfig()
    rid = run_id or run_id_with_prefix("stresst")
    cd = pd.Timestamp(calc_date) if calc_date is not None else None

    w, snap_date = _normalize_positions(positions, calc_date=cd, capital=cfg.capital)
    rmat = _to_returns_matrix(returns)
    if cd is not None:
        rmat = rmat.loc[rmat.index <= cd]
    if len(rmat) == 0:
        raise ValueError("returns empty after calc_date filter")

    symbols = pd.Index(w.index.astype(str), name="symbol")
    r_aligned = rmat.reindex(columns=symbols).fillna(0.0)
    p_ret = r_aligned.dot(w.reindex(symbols).fillna(0.0))
    p_ret = pd.to_numeric(p_ret, errors="coerce").dropna()

    sigma = _base_sigma(covariance, rmat, symbols, cfg)
    wv = w.reindex(symbols).fillna(0.0).values.astype(float)
    base_vol = float(np.sqrt(max(wv @ sigma.values @ wv, 0.0)))

    expo = _normalize_exposures(exposures, calc_date=snap_date)
    liq = _normalize_liquidity(liquidity, calc_date=snap_date)

    scen_defs = scenarios if scenarios is not None else _default_scenarios(p_ret, base_vol, expo, cfg)

    result_rows: list[dict[str, Any]] = []
    breach_rows: list[dict[str, Any]] = []
    any_fallback = False

    for s in scen_defs:
        sc = dict(s)
        sid = str(sc.get("scenario_id", f"scenario_{len(result_rows)+1}"))
        stype = str(sc.get("scenario_type", "parametric"))
        kind = str(sc.get("kind", "market"))

        r_stress = p_ret.copy()
        sigma_stress = sigma.copy()
        liq_cost = 0.0
        pr_p95 = pr_max = dtl_p95 = dtl_max = float("nan")
        adv_cov = 1.0
        fallback = False

        if stype == "historical":
            sd = pd.Timestamp(sc.get("start_date"))
            ed = pd.Timestamp(sc.get("end_date"))
            r_stress = p_ret.loc[(p_ret.index >= sd) & (p_ret.index <= ed)]
            if len(r_stress) == 0:
                fallback = True
                r_stress = p_ret.tail(1)

        elif kind == "market":
            sh = float(sc.get("market_shock", 0.0))
            r_stress = p_ret + sh

        elif kind == "vol":
            kappa = float(sc.get("kappa", 0.0))
            sigma_stress = _psd((1.0 + kappa) * sigma, cfg.psd_eps, cfg.psd_diag_floor)
            scale = float(np.sqrt(max(1.0 + kappa, 1e-12)))
            r_stress = p_ret * scale

        elif kind == "corr":
            dr = float(sc.get("delta_rho", 0.0))
            sigma_stress = _cov_corr_shock(sigma, dr, cfg)
            v0 = max(float(wv @ sigma.values @ wv), 1e-12)
            v1 = max(float(wv @ sigma_stress.values @ wv), 1e-12)
            scale = float(np.sqrt(v1 / v0))
            r_stress = p_ret * scale

        elif kind == "factor":
            f = str(sc.get("factor_name", ""))
            shock = float(sc.get("factor_shock", 0.0))
            if len(expo) and f in expo.columns:
                b = pd.to_numeric(expo[f].reindex(symbols), errors="coerce").fillna(0.0)
                shift = float(np.dot(wv, b.values * shock))
                r_stress = p_ret + shift
            else:
                fallback = True

        elif kind == "liquidity":
            eta = float(sc.get("eta_adv", 1.0))
            sm = float(sc.get("spread_mult", 1.0))
            liq_cost, pr_p95, pr_max, dtl_p95, dtl_max, adv_cov, liq_fb = _liq_damage(w.reindex(symbols).fillna(0.0), liq, eta_adv=eta, spread_mult=sm, cfg=cfg)
            fallback = fallback or liq_fb
            r_stress = p_ret - liq_cost

        elif kind == "combined":
            sh = float(sc.get("market_shock", 0.0))
            dr = float(sc.get("delta_rho", 0.0))
            eta = float(sc.get("eta_adv", 1.0))
            sm = float(sc.get("spread_mult", 1.0))
            sigma_stress = _cov_corr_shock(sigma, dr, cfg)
            v0 = max(float(wv @ sigma.values @ wv), 1e-12)
            v1 = max(float(wv @ sigma_stress.values @ wv), 1e-12)
            scale = float(np.sqrt(v1 / v0))
            liq_cost, pr_p95, pr_max, dtl_p95, dtl_max, adv_cov, liq_fb = _liq_damage(w.reindex(symbols).fillna(0.0), liq, eta_adv=eta, spread_mult=sm, cfg=cfg)
            fallback = fallback or liq_fb
            r_stress = p_ret * scale + sh - liq_cost

        any_fallback = any_fallback or fallback

        pnl = float(np.nanmin(r_stress.values)) if len(r_stress) else float("nan")
        vol = float(np.nanstd(r_stress.values, ddof=1) * np.sqrt(252)) if len(r_stress) >= 2 else float(np.sqrt(max(wv @ sigma_stress.values @ wv, 0.0)) * np.sqrt(252))
        dd, mdd = _dd_metrics(r_stress)
        es = _es_loss(r_stress, cfg.es_alpha)

        pnl_loss = -pnl
        mdd_loss = -mdd
        hard = 0
        warn = 0

        def add(metric: str, value: float, warn_lim: float, hard_lim: float) -> None:
            nonlocal hard, warn
            if not np.isfinite(value):
                sev = "warn"
            elif value > hard_lim:
                sev = "hard"
            elif value > warn_lim:
                sev = "warn"
            else:
                return
            if sev == "hard":
                hard += 1
            else:
                warn += 1
            breach_rows.append(
                {
                    "scenario_id": sid,
                    "metric_name": metric,
                    "metric_value": float(value),
                    "warn_limit": float(warn_lim),
                    "hard_limit": float(hard_lim),
                    "severity": sev,
                    "notes": "",
                    "run_id": rid,
                }
            )

        add("pnl_loss", pnl_loss, cfg.pnl_warn_limit, cfg.pnl_hard_limit)
        add("mdd_loss", mdd_loss, cfg.mdd_warn_limit, cfg.mdd_hard_limit)
        add("es_loss", es, cfg.es_warn_limit, cfg.es_hard_limit)
        add("pr_max", pr_max if np.isfinite(pr_max) else 0.0, cfg.pr_warn_limit, cfg.pr_hard_limit)
        add("dtl_max", dtl_max if np.isfinite(dtl_max) else 0.0, cfg.dtl_warn_limit, cfg.dtl_hard_limit)
        if adv_cov < cfg.coverage_adv_min:
            add("adv_coverage_gap", 1.0 - adv_cov, 1.0 - cfg.coverage_adv_min, 1.0 - cfg.coverage_adv_min + 0.02)

        decision = "pass"
        if hard > 0:
            decision = "fail"
        elif warn > 0:
            decision = "warn"

        result_rows.append(
            {
                "scenario_id": sid,
                "scenario_type": stype,
                "severity_value": float(sc.get("severity_value", np.nan)),
                "pnl_stress": pnl,
                "vol_stress": vol,
                "dd_stress": dd,
                "mdd_stress": mdd,
                "es_stress": es,
                "liquidity_cost": float(liq_cost),
                "pr_p95": pr_p95,
                "pr_max": pr_max,
                "dtl_p95": dtl_p95,
                "dtl_max": dtl_max,
                "num_breaches": int(hard + warn),
                "num_hard": int(hard),
                "num_warn": int(warn),
                "decision": decision,
                "run_id": rid,
            }
        )

    scenario_results = pd.DataFrame(result_rows)
    breaches = pd.DataFrame(breach_rows) if breach_rows else pd.DataFrame(columns=["scenario_id", "metric_name", "metric_value", "warn_limit", "hard_limit", "severity", "notes", "run_id"])

    if len(scenario_results) == 0:
        raise ValueError("no scenarios executed")

    n_fail = int((scenario_results["decision"] == "fail").sum())
    n_warn = int((scenario_results["decision"] == "warn").sum())
    gdec = "pass"
    if n_fail > 0:
        gdec = "fail"
    elif n_warn > 0:
        gdec = "warn"

    worst_pnl = scenario_results.loc[scenario_results["pnl_stress"].idxmin(), "scenario_id"]
    worst_dd = scenario_results.loc[scenario_results["mdd_stress"].idxmin(), "scenario_id"]
    worst_es = scenario_results.loc[scenario_results["es_stress"].idxmax(), "scenario_id"]

    summary = {
        "calc_date": str(snap_date.date()),
        "num_scenarios": int(len(scenario_results)),
        "worst_scenario_by_pnl": str(worst_pnl),
        "worst_scenario_by_dd": str(worst_dd),
        "worst_scenario_by_es": str(worst_es),
        "num_fail": int(n_fail),
        "num_warn": int(n_warn),
        "global_decision": gdec,
        "fallback_used": bool(any_fallback),
        "config_hash": config_hash(cfg.__dict__, n=24),
        "run_id": rid,
        "execution_timestamp": utc_now_iso(),
    }

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_stress_outputs(scenario_results, breaches, summary, [json_safe(s) for s in scen_defs], output_dir=output_dir)

    return {
        "scenario_results": scenario_results,
        "breaches": breaches,
        "summary": summary,
        "scenario_definitions": scen_defs,
        "artifacts": artifacts,
    }


__all__ = ["StressTestsConfig", "persist_stress_outputs", "run_stress_tests"]
