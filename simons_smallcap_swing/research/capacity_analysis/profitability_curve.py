
"""
research/capacity_analysis/profitability_curve.py - Capacity profitability curve.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from simons_smallcap_swing.research._shared import (
    block_bootstrap_mean,
    config_hash,
    json_safe,
    read_dataframe,
    run_id_with_prefix,
    utc_now_iso,
    write_json_safe,
    write_parquet_safe,
)


@dataclass(frozen=True)
class ProfitabilityCurveConfig:
    base_aum: float = 100_000_000.0
    aum_grid_multipliers: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
    rf_annual: float = 0.02
    sr_min: float = 0.6
    impact_y: float = 0.7
    impact_delta: float = 0.6
    impact_gamma: float = 0.0
    default_participation_cap: float = 0.10
    bootstrap_enabled: bool = True
    n_bootstrap: int = 300
    bootstrap_block_size: int = 20
    sensitivity_enabled: bool = True


def _to_df(obj: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    return read_dataframe(obj)


def _normalize_baseline(panel: pd.DataFrame) -> pd.DataFrame:
    if "date" not in panel.columns:
        raise ValueError("baseline_panel requires date")
    sym = "symbol" if "symbol" in panel.columns else "asset"
    if sym not in panel.columns:
        raise ValueError("baseline_panel requires symbol/asset")

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    out[sym] = out[sym].astype(str)

    if "executed_notional" not in out.columns:
        if "turnover" in out.columns:
            out["executed_notional"] = pd.to_numeric(out["turnover"], errors="coerce")
        else:
            out["executed_notional"] = 0.0

    if "gross_alpha_dollar" not in out.columns:
        if "gross_return" in out.columns:
            out["gross_alpha_dollar"] = pd.to_numeric(out["gross_return"], errors="coerce")
        elif "forecast_alpha_bps" in out.columns:
            out["gross_alpha_dollar"] = pd.to_numeric(out["forecast_alpha_bps"], errors="coerce") / 10_000.0
        else:
            out["gross_alpha_dollar"] = 0.0

    if "spread_bps" not in out.columns:
        out["spread_bps"] = np.nan
    if "volatility" not in out.columns:
        out["volatility"] = np.nan

    out = out[["date", sym, "executed_notional", "gross_alpha_dollar", "spread_bps", "volatility"]]
    return out.rename(columns={sym: "symbol"})


def _normalize_liquidity(liq: pd.DataFrame, default_cap: float) -> pd.DataFrame:
    if "date" not in liq.columns:
        raise ValueError("liquidity_panel requires date")
    sym = "symbol" if "symbol" in liq.columns else "asset"
    if sym not in liq.columns:
        raise ValueError("liquidity_panel requires symbol/asset")

    out = liq.copy()
    out["date"] = pd.to_datetime(out["date"])
    out[sym] = out[sym].astype(str)

    adv_col = "adv_dollar"
    if adv_col not in out.columns:
        for c in ["adv", "dollar_adv", "avg_dollar_volume"]:
            if c in out.columns:
                adv_col = c
                break
        else:
            raise ValueError("liquidity_panel requires adv_dollar/adv")

    if "participation_cap" not in out.columns:
        out["participation_cap"] = default_cap

    if "spread_bps" not in out.columns:
        out["spread_bps"] = np.nan
    if "volatility" not in out.columns:
        out["volatility"] = np.nan

    out = out[["date", sym, adv_col, "participation_cap", "spread_bps", "volatility"]]
    out = out.rename(columns={sym: "symbol", adv_col: "adv_dollar"})
    return out


def _simulate_aum(
    panel: pd.DataFrame,
    *,
    aum: float,
    base_aum: float,
    cfg: ProfitabilityCurveConfig,
    y_scale: float = 1.0,
    delta_shift: float = 0.0,
    spread_scale: float = 1.0,
    cap_scale: float = 1.0,
) -> tuple[dict[str, Any], pd.Series, pd.Series]:
    s = float(aum / max(base_aum, 1e-9))
    df = panel.copy()

    desired = pd.to_numeric(df["executed_notional"], errors="coerce").abs().fillna(0.0) * s
    adv = pd.to_numeric(df["adv_dollar"], errors="coerce").clip(lower=1.0)
    cap = pd.to_numeric(df["participation_cap"], errors="coerce").fillna(cfg.default_participation_cap) * cap_scale
    cap_notional = cap * adv
    realized = np.minimum(desired, cap_notional)

    trunc_frac = 1.0 - (realized / (desired + 1e-12))
    alpha_base = pd.to_numeric(df["gross_alpha_dollar"], errors="coerce").fillna(0.0)
    gross_pnl = alpha_base * s * (1.0 - trunc_frac)

    spread = pd.to_numeric(df["spread_bps"], errors="coerce").fillna(15.0) * spread_scale
    vol = pd.to_numeric(df["volatility"], errors="coerce").fillna(0.02)

    c_spread = 0.5 * spread / 10_000.0
    c_temp = (cfg.impact_y * y_scale) * vol * (realized / (adv + 1e-9)) ** (cfg.impact_delta + delta_shift)
    c_perm = cfg.impact_gamma * (realized / (adv + 1e-9))
    cost_per_dollar = c_spread + c_temp + c_perm
    trading_cost = cost_per_dollar * realized

    net_pnl = gross_pnl - trading_cost

    daily = pd.DataFrame(
        {
            "date": df["date"],
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "realized": realized,
            "adv": adv,
            "trunc_frac": trunc_frac,
        }
    ).groupby("date", as_index=False).sum()

    gross_return = pd.Series(daily["gross_pnl"] / max(aum, 1e-9), index=daily["date"])
    net_return = pd.Series(daily["net_pnl"] / max(aum, 1e-9), index=daily["date"])

    mu_gross = float(gross_return.mean() * 252.0)
    mu_net = float(net_return.mean() * 252.0)
    vol_gross = float(gross_return.std(ddof=0) * np.sqrt(252.0))
    vol_net = float(net_return.std(ddof=0) * np.sqrt(252.0))

    sr_gross = float((mu_gross - cfg.rf_annual) / (vol_gross + 1e-12))
    sr_net = float((mu_net - cfg.rf_annual) / (vol_net + 1e-12))

    avg_participation = float((daily["realized"] / (daily["adv"] + 1e-9)).mean())
    frac_truncated = float((daily["trunc_frac"] > 0).mean())
    alpha_trunc = float(daily["trunc_frac"].mean())

    result = {
        "aum": float(aum),
        "gross_return": mu_gross,
        "net_return": mu_net,
        "gross_pnl": float(daily["gross_pnl"].sum()),
        "net_pnl": float(daily["net_pnl"].sum()),
        "gross_vol": vol_gross,
        "net_vol": vol_net,
        "gross_sharpe": sr_gross,
        "net_sharpe": sr_net,
        "avg_turnover": float((daily["realized"] / max(aum, 1e-9)).mean() * 252.0),
        "avg_participation": avg_participation,
        "frac_truncated_trades": frac_truncated,
        "alpha_truncation_pct": alpha_trunc,
    }
    return result, gross_return, net_return

def _bootstrap_ci(series: pd.Series, cfg: ProfitabilityCurveConfig) -> tuple[float, float, float, float]:
    x = pd.to_numeric(series, errors="coerce").dropna().values
    if x.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    boot_mean = block_bootstrap_mean(
        x,
        n_bootstrap=cfg.n_bootstrap,
        block_size=cfg.bootstrap_block_size,
        seed=42,
    )
    if boot_mean.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    # Approximate Sharpe bootstrap with fixed volatility estimate.
    sigma = float(np.std(x, ddof=0) * np.sqrt(252.0))
    sr_boot = (boot_mean * 252.0 - cfg.rf_annual) / (sigma + 1e-12)

    return (
        float(np.percentile(boot_mean * 252.0, 2.5)),
        float(np.percentile(boot_mean * 252.0, 97.5)),
        float(np.percentile(sr_boot, 2.5)),
        float(np.percentile(sr_boot, 97.5)),
    )


def _sensitivity_summary(panel: pd.DataFrame, aum: float, cfg: ProfitabilityCurveConfig) -> dict[str, Any]:
    scenarios = {
        "Y_minus_20": dict(y_scale=0.8),
        "Y_plus_20": dict(y_scale=1.2),
        "delta_minus_0p1": dict(delta_shift=-0.1),
        "delta_plus_0p1": dict(delta_shift=0.1),
        "cap_minus_10pct": dict(cap_scale=0.9),
        "spread_plus_20pct": dict(spread_scale=1.2),
    }
    out: dict[str, float] = {}
    for name, kw in scenarios.items():
        res, _, _ = _simulate_aum(panel, aum=aum, base_aum=cfg.base_aum, cfg=cfg, **kw)
        out[name] = float(res["net_return"])
    return out


def persist_profitability_curve(curve: pd.DataFrame, manifest: Mapping[str, Any], *, output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "profitability_curve": write_parquet_safe(curve, out / "profitability_curve.parquet"),
        "manifest": write_json_safe(dict(manifest), out / "manifest.json"),
    }


def run_profitability_curve(
    baseline_panel: pd.DataFrame | str | Path,
    liquidity_panel: pd.DataFrame | str | Path,
    *,
    config: ProfitabilityCurveConfig | None = None,
    run_id: str | None = None,
    data_snapshot_id: str = "unknown",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or ProfitabilityCurveConfig()
    rid = run_id or run_id_with_prefix("pcurve")

    baseline = _normalize_baseline(_to_df(baseline_panel))
    liq = _normalize_liquidity(_to_df(liquidity_panel), cfg.default_participation_cap)

    panel = baseline.merge(liq, on=["date", "symbol"], how="left", suffixes=("", "_liq"))
    panel["adv_dollar"] = pd.to_numeric(panel["adv_dollar"], errors="coerce").fillna(1.0)
    panel["participation_cap"] = pd.to_numeric(panel["participation_cap"], errors="coerce").fillna(cfg.default_participation_cap)

    aum_grid = [cfg.base_aum * m for m in cfg.aum_grid_multipliers]

    rows: list[dict[str, Any]] = []
    net_return_by_aum: dict[float, pd.Series] = {}

    for aum in aum_grid:
        res, _, net_ret = _simulate_aum(panel, aum=aum, base_aum=cfg.base_aum, cfg=cfg)

        ci_mu_low = ci_mu_high = ci_sr_low = ci_sr_high = float("nan")
        if cfg.bootstrap_enabled:
            ci_mu_low, ci_mu_high, ci_sr_low, ci_sr_high = _bootstrap_ci(net_ret, cfg)

        sensitivity = {}
        if cfg.sensitivity_enabled:
            sensitivity = _sensitivity_summary(panel, aum, cfg)

        row = {
            **res,
            "capacity_flag": "admissible" if (res["net_return"] > 0 and res["net_sharpe"] >= cfg.sr_min) else "constrained",
            "binding_constraint": "none",
            "marginal_net_pnl": np.nan,
            "ci_mu_net_low": ci_mu_low,
            "ci_mu_net_high": ci_mu_high,
            "ci_sr_net_low": ci_sr_low,
            "ci_sr_net_high": ci_sr_high,
            "sensitivity_summary_json": json_safe(sensitivity),
            "notes": "",
        }

        if res["net_return"] <= 0:
            row["binding_constraint"] = "return_break_even_exceeded"
        elif res["net_sharpe"] < cfg.sr_min:
            row["binding_constraint"] = "sharpe_constraint_binding"
        elif res["frac_truncated_trades"] > 0.20:
            row["binding_constraint"] = "liquidity_constraint_binding"

        rows.append(row)
        net_return_by_aum[aum] = net_ret

    curve = pd.DataFrame(rows).sort_values("aum").reset_index(drop=True)
    curve["marginal_net_pnl"] = curve["net_pnl"].diff()
    curve["run_id"] = rid

    manifest = {
        "run_id": rid,
        "config_hash": config_hash(cfg.__dict__, n=24),
        "data_snapshot_id": data_snapshot_id,
        "base_aum": cfg.base_aum,
        "n_aum_points": int(len(curve)),
        "execution_timestamp": utc_now_iso(),
    }

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_profitability_curve(curve, manifest, output_dir=output_dir)

    return {
        "profitability_curve": curve,
        "manifest": json_safe(manifest),
        "artifacts": artifacts,
    }


__all__ = [
    "ProfitabilityCurveConfig",
    "persist_profitability_curve",
    "run_profitability_curve",
]
