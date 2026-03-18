
"""
risk/exposure_report.py - Exposure aggregation and gating (allow/warn/block).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExposureReportConfig:
    mode: str = "pre_trade"  # pre_trade | post_trade
    capital: float = 100_000_000.0
    phi_adv: float = 0.2
    tau_warn: float = 0.05
    gross_max: float = 2.0
    net_abs_max: float = 0.8
    hhi_max: float = 0.15
    top1_max: float = 0.15
    top5_max: float = 0.55
    pr_max: float = 0.10
    dtl_max: float = 20.0
    factor_limits: Mapping[str, float] = field(default_factory=dict)
    sector_net_limits: Mapping[str, float] = field(default_factory=dict)
    sector_gross_limits: Mapping[str, float] = field(default_factory=dict)
    coverage_beta_min: float = 0.95
    coverage_sector_min: float = 0.98
    coverage_adv_min: float = 0.95


def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_id_with_prefix(prefix: str) -> str:
    from datetime import datetime, timezone

    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def config_hash(cfg: Mapping[str, Any], n: int = 24) -> str:
    import hashlib
    import json

    blob = json.dumps(dict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:n]


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if (np.isnan(v) or np.isinf(v)) else v
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


def _to_df(obj: pd.DataFrame | str | Path | None) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, (str, Path)):
        p = Path(obj)
        if p.suffix.lower() == ".parquet":
            return pd.read_parquet(p)
        return pd.read_csv(p)
    return obj.copy()


def _normalize_positions(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError("positions must include date")
    sym = "symbol" if "symbol" in df.columns else "asset"
    if sym not in df.columns:
        raise ValueError("positions must include symbol/asset")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out[sym] = out[sym].astype(str)

    if "w_current" not in out.columns:
        for c in ["weight", "w", "position_weight"]:
            if c in out.columns:
                out = out.rename(columns={c: "w_current"})
                break
    if "w_current" not in out.columns:
        raise ValueError("positions require w_current/weight")

    out["w_current"] = pd.to_numeric(out["w_current"], errors="coerce").fillna(0.0)
    return out[["date", sym, "w_current"]].rename(columns={sym: "symbol"})


def _normalize_orders(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=["date", "symbol", "delta_w", "order_notional"])

    if "date" not in df.columns:
        raise ValueError("orders must include date")
    sym = "symbol" if "symbol" in df.columns else "asset"
    if sym not in df.columns:
        raise ValueError("orders must include symbol/asset")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out[sym] = out[sym].astype(str)

    if "delta_w" not in out.columns:
        for c in ["delta_weight", "order_weight", "dw"]:
            if c in out.columns:
                out = out.rename(columns={c: "delta_w"})
                break
    if "delta_w" not in out.columns:
        out["delta_w"] = 0.0

    if "order_notional" not in out.columns:
        out["order_notional"] = np.nan

    out["delta_w"] = pd.to_numeric(out["delta_w"], errors="coerce").fillna(0.0)
    out["order_notional"] = pd.to_numeric(out["order_notional"], errors="coerce")
    return out[["date", sym, "delta_w", "order_notional"]].rename(columns={sym: "symbol"})


def _normalize_exposures(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=["date", "symbol", "factor_name", "exposure_value"]) 

    if {"date", "symbol", "factor_name", "exposure_value"}.issubset(df.columns):
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])
        out["symbol"] = out["symbol"].astype(str)
        out["factor_name"] = out["factor_name"].astype(str)
        out["exposure_value"] = pd.to_numeric(out["exposure_value"], errors="coerce")
        return out[["date", "symbol", "factor_name", "exposure_value"]]

    if "date" not in df.columns:
        raise ValueError("exposures must include date")
    sym = "symbol" if "symbol" in df.columns else "asset"
    if sym not in df.columns:
        raise ValueError("exposures must include symbol/asset")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out[sym] = out[sym].astype(str)
    fac_cols = [c for c in out.columns if c not in {"date", sym, "sector"}]
    long_rows = []
    for c in fac_cols:
        vals = pd.to_numeric(out[c], errors="coerce")
        tmp = pd.DataFrame(
            {
                "date": out["date"],
                "symbol": out[sym],
                "factor_name": c,
                "exposure_value": vals,
            }
        )
        long_rows.append(tmp)
    return pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame(columns=["date", "symbol", "factor_name", "exposure_value"])


def _normalize_sector(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=["date", "symbol", "sector"])
    if "date" not in df.columns:
        raise ValueError("sector panel must include date")
    sym = "symbol" if "symbol" in df.columns else "asset"
    if sym not in df.columns:
        raise ValueError("sector panel must include symbol/asset")

    sec_col = "sector" if "sector" in df.columns else "group"
    if sec_col not in df.columns:
        raise ValueError("sector panel must include sector/group")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out[sym] = out[sym].astype(str)
    out[sec_col] = out[sec_col].fillna("UNKNOWN").astype(str)
    return out[["date", sym, sec_col]].rename(columns={sym: "symbol", sec_col: "sector"})


def _normalize_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=["date", "symbol", "adv_usd"])
    if "date" not in df.columns:
        raise ValueError("liquidity panel must include date")
    sym = "symbol" if "symbol" in df.columns else "asset"
    if sym not in df.columns:
        raise ValueError("liquidity panel must include symbol/asset")

    adv_col = "adv_usd"
    if adv_col not in df.columns:
        for c in ["adv_dollar", "adv", "dollar_adv"]:
            if c in df.columns:
                adv_col = c
                break

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out[sym] = out[sym].astype(str)
    out[adv_col] = pd.to_numeric(out[adv_col], errors="coerce")
    return out[["date", sym, adv_col]].rename(columns={sym: "symbol", adv_col: "adv_usd"})

def _severity(metric_value: float, limit_value: float, tau: float) -> str:
    if not np.isfinite(metric_value) or not np.isfinite(limit_value):
        return "warn"
    if metric_value <= limit_value:
        return "info"
    if metric_value <= limit_value * (1.0 + tau):
        return "warn"
    return "hard"


def _json_write(payload: Mapping[str, Any], path: Path) -> str:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), sort_keys=True, indent=2, default=str), encoding="utf-8")
    return str(path)


def persist_exposure_outputs(daily: pd.DataFrame, breaches: pd.DataFrame, summary: Mapping[str, Any], *, output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dpath = out / "daily.parquet"
    bpath = out / "breaches.parquet"
    spath = out / "summary.json"

    try:
        daily.to_parquet(dpath, index=False)
        dfile = str(dpath)
    except Exception:
        daily.to_csv(dpath.with_suffix(".csv"), index=False)
        dfile = str(dpath.with_suffix(".csv"))

    try:
        breaches.to_parquet(bpath, index=False)
        bfile = str(bpath)
    except Exception:
        breaches.to_csv(bpath.with_suffix(".csv"), index=False)
        bfile = str(bpath.with_suffix(".csv"))

    return {
        "daily": dfile,
        "breaches": bfile,
        "summary": _json_write(summary, spath),
    }


def run_exposure_report(
    positions: pd.DataFrame | str | Path,
    *,
    exposures: pd.DataFrame | str | Path | None = None,
    sectors: pd.DataFrame | str | Path | None = None,
    liquidity: pd.DataFrame | str | Path | None = None,
    orders: pd.DataFrame | str | Path | None = None,
    config: ExposureReportConfig | None = None,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or ExposureReportConfig()
    rid = run_id or run_id_with_prefix("exprep")

    pos = _normalize_positions(_to_df(positions))
    ord_df = _normalize_orders(_to_df(orders))
    exp = _normalize_exposures(_to_df(exposures))
    sec = _normalize_sector(_to_df(sectors))
    liq = _normalize_liquidity(_to_df(liquidity))

    all_dates = sorted(pos["date"].unique())
    daily_rows: list[dict[str, Any]] = []
    breach_rows: list[dict[str, Any]] = []

    for d in all_dates:
        p = pos[pos["date"] == d].copy()
        o = ord_df[ord_df["date"] == d].copy()

        p = p.merge(o[["symbol", "delta_w", "order_notional"]], on="symbol", how="left")
        p["delta_w"] = pd.to_numeric(p["delta_w"], errors="coerce").fillna(0.0)

        if cfg.mode == "pre_trade":
            p["w_post"] = p["w_current"] + p["delta_w"]
        else:
            p["w_post"] = p["w_current"]

        gross = float(np.abs(p["w_post"]).sum())
        net = float(p["w_post"].sum())

        absw = np.abs(p["w_post"].values)
        if gross > 0:
            wtilde = absw / gross
            hhi = float(np.sum(wtilde ** 2))
            top1 = float(np.max(absw))
            top5 = float(np.sort(absw)[::-1][:5].sum())
        else:
            hhi = float("nan")
            top1 = 0.0
            top5 = 0.0

        n_eff = float(1.0 / hhi) if np.isfinite(hhi) and hhi > 0 else float("nan")

        e_d = exp[exp["date"] == d].copy()
        e_wide = e_d.pivot_table(index="symbol", columns="factor_name", values="exposure_value", aggfunc="last") if len(e_d) else pd.DataFrame(index=p["symbol"].unique())
        e_wide = e_wide.reindex(p["symbol"].values)

        p_local = p.set_index("symbol")
        factor_exposures: dict[str, float] = {}
        for f in e_wide.columns:
            beta = pd.to_numeric(e_wide[f], errors="coerce")
            w = pd.to_numeric(p_local["w_post"], errors="coerce")
            factor_exposures[str(f)] = float((w * beta).sum(skipna=True))

        s_d = sec[sec["date"] == d].copy()
        p_sec = p.merge(s_d[["symbol", "sector"]], on="symbol", how="left") if len(s_d) else p.assign(sector="UNKNOWN")
        p_sec["sector"] = p_sec["sector"].fillna("UNKNOWN")
        sector_net = p_sec.groupby("sector")["w_post"].sum().to_dict()
        sector_gross = p_sec.assign(abs_w=np.abs(p_sec["w_post"])).groupby("sector")["abs_w"].sum().to_dict()

        l_d = liq[liq["date"] == d].copy()
        p_liq = p.merge(l_d[["symbol", "adv_usd"]], on="symbol", how="left") if len(l_d) else p.assign(adv_usd=np.nan)

        pos_notional = np.abs(p_liq["w_post"]) * cfg.capital
        if "order_notional" in p_liq.columns and p_liq["order_notional"].notna().any():
            q = np.abs(pd.to_numeric(p_liq["order_notional"], errors="coerce").fillna(0.0))
        else:
            q = np.abs(pd.to_numeric(p_liq["delta_w"], errors="coerce").fillna(0.0)) * cfg.capital

        adv = pd.to_numeric(p_liq["adv_usd"], errors="coerce")
        pr = q / (cfg.phi_adv * adv)
        dtl = pos_notional / (cfg.phi_adv * adv)

        pr_p95 = float(np.nanpercentile(pr.values, 95)) if np.isfinite(pr.values).any() else float("nan")
        pr_max = float(np.nanmax(pr.values)) if np.isfinite(pr.values).any() else float("nan")
        dtl_p95 = float(np.nanpercentile(dtl.values, 95)) if np.isfinite(dtl.values).any() else float("nan")
        dtl_max = float(np.nanmax(dtl.values)) if np.isfinite(dtl.values).any() else float("nan")

        # Coverage stats by material notional.
        notional = np.abs(p_liq["w_post"]) * cfg.capital
        total_notional = float(notional.sum()) + 1e-12

        beta_valid_mask = e_wide.notna().any(axis=1).reindex(p_liq["symbol"].values).fillna(False).values if len(e_wide.columns) else np.zeros(len(p_liq), dtype=bool)
        sector_valid_mask = p_sec["sector"].astype(str).ne("UNKNOWN").values
        adv_valid_mask = pd.to_numeric(p_liq["adv_usd"], errors="coerce").gt(0).fillna(False).values

        coverage = {
            "beta_valid_notional_pct": float(notional[beta_valid_mask].sum() / total_notional),
            "sector_known_notional_pct": float(notional[sector_valid_mask].sum() / total_notional),
            "adv_valid_notional_pct": float(notional[adv_valid_mask].sum() / total_notional),
        }

        # Build breaches.
        def add_breach(metric_name: str, metric_value: float, limit: float, affected: str = "portfolio") -> None:
            sev = _severity(abs(metric_value), abs(limit), cfg.tau_warn)
            if sev == "info":
                return
            breach_rows.append(
                {
                    "date": d,
                    "metric_name": metric_name,
                    "metric_value": float(metric_value),
                    "limit_value": float(limit),
                    "severity": sev,
                    "affected_bucket": affected,
                    "mode pre/post-trade": cfg.mode,
                    "notes": "",
                    "run_id": rid,
                }
            )

        add_breach("gross", gross, cfg.gross_max)
        add_breach("abs_net", abs(net), cfg.net_abs_max)
        add_breach("hhi", hhi, cfg.hhi_max)
        add_breach("top1", top1, cfg.top1_max)
        add_breach("top5", top5, cfg.top5_max)
        add_breach("pr_max", pr_max, cfg.pr_max)
        add_breach("dtl_max", dtl_max, cfg.dtl_max)

        for f, lim in cfg.factor_limits.items():
            if f in factor_exposures:
                add_breach(f"factor::{f}", abs(factor_exposures[f]), float(lim), affected=f)

        for sname, sval in sector_net.items():
            lim = float(cfg.sector_net_limits.get(sname, np.inf))
            if np.isfinite(lim):
                add_breach(f"sector_net::{sname}", abs(float(sval)), lim, affected=sname)

        for sname, sval in sector_gross.items():
            lim = float(cfg.sector_gross_limits.get(sname, np.inf))
            if np.isfinite(lim):
                add_breach(f"sector_gross::{sname}", abs(float(sval)), lim, affected=sname)

        # Coverage-based degradations.
        if coverage["beta_valid_notional_pct"] < cfg.coverage_beta_min:
            add_breach("coverage_beta", 1.0 - coverage["beta_valid_notional_pct"], 1.0 - cfg.coverage_beta_min)
        if coverage["sector_known_notional_pct"] < cfg.coverage_sector_min:
            add_breach("coverage_sector", 1.0 - coverage["sector_known_notional_pct"], 1.0 - cfg.coverage_sector_min)
        if coverage["adv_valid_notional_pct"] < cfg.coverage_adv_min:
            add_breach("coverage_adv", 1.0 - coverage["adv_valid_notional_pct"], 1.0 - cfg.coverage_adv_min)

        b_today = [b for b in breach_rows if pd.Timestamp(b["date"]) == d]
        hard = sum(1 for b in b_today if b["severity"] == "hard")
        warn = sum(1 for b in b_today if b["severity"] == "warn")

        decision = "allow"
        if hard > 0:
            decision = "block"
        elif warn > 0:
            decision = "warn"

        daily_rows.append(
            {
                "date": d,
                "gross": gross,
                "net": net,
                "factor exposures": json_safe(factor_exposures),
                "sector exposures net/gross": json_safe({"net": sector_net, "gross": sector_gross}),
                "HHI": hhi,
                "Top1": top1,
                "Top5": top5,
                "N_eff": n_eff,
                "PR_p95": pr_p95,
                "PR_max": pr_max,
                "DTL_p95": dtl_p95,
                "DTL_max": dtl_max,
                "decision": decision,
                "coverage stats": json_safe(coverage),
                "run_id": rid,
            }
        )

    daily = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    breaches = pd.DataFrame(breach_rows).sort_values(["date", "severity"]).reset_index(drop=True) if breach_rows else pd.DataFrame(
        columns=["date", "metric_name", "metric_value", "limit_value", "severity", "affected_bucket", "mode pre/post-trade", "notes", "run_id"]
    )

    if len(daily):
        latest = daily.iloc[-1].to_dict()
    else:
        latest = {"date": None, "decision": "warn"}

    num_hard = int((breaches["severity"] == "hard").sum()) if len(breaches) else 0
    num_warn = int((breaches["severity"] == "warn").sum()) if len(breaches) else 0
    num_breaches = int(len(breaches))

    most_severe = None
    if len(breaches):
        hard_rows = breaches[breaches["severity"] == "hard"]
        sel = hard_rows if len(hard_rows) else breaches[breaches["severity"] == "warn"]
        if len(sel):
            most_severe = sel.iloc[0]["metric_name"]

    summary = {
        "date": str(latest.get("date")),
        "gross": float(latest.get("gross", np.nan)),
        "net": float(latest.get("net", np.nan)),
        "factor exposures": latest.get("factor exposures", {}),
        "sector exposures": latest.get("sector exposures net/gross", {}),
        "HHI": float(latest.get("HHI", np.nan)),
        "Top1": float(latest.get("Top1", np.nan)),
        "Top5": float(latest.get("Top5", np.nan)),
        "N_eff": float(latest.get("N_eff", np.nan)),
        "num_breaches": num_breaches,
        "num_hard": num_hard,
        "num_warn": num_warn,
        "most_severe_breach": most_severe,
        "decision": latest.get("decision", "warn"),
        "coverage": latest.get("coverage stats", {}),
        "fallback_used": False,
        "config_hash": config_hash(cfg.__dict__, n=24),
        "run_id": rid,
        "execution_timestamp": utc_now_iso(),
    }

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_exposure_outputs(daily, breaches, summary, output_dir=output_dir)

    return {
        "daily": daily,
        "breaches": breaches,
        "summary": summary,
        "artifacts": artifacts,
    }


__all__ = [
    "ExposureReportConfig",
    "persist_exposure_outputs",
    "run_exposure_report",
]
