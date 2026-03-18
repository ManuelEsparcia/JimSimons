
"""
research/capacity_analysis/turnover_vs_edge.py - Trading intensity trade-off.
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
class TurnoverVsEdgeConfig:
    throttle_grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    capital: float = 100_000_000.0
    rf_annual: float = 0.02
    sr_min: float = 0.6
    pi_max: float = 0.10
    impact_y: float = 0.7
    impact_delta: float = 0.6
    impact_gamma: float = 0.0
    n_bootstrap: int = 300
    bootstrap_block_size: int = 20
    bootstrap_enabled: bool = True


def _to_df(obj: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    return read_dataframe(obj)


def _normalize_panel(rebalance_panel: pd.DataFrame) -> pd.DataFrame:
    if "date" not in rebalance_panel.columns:
        raise ValueError("rebalance_panel requires date")
    sym = "symbol" if "symbol" in rebalance_panel.columns else "asset"
    if sym not in rebalance_panel.columns:
        raise ValueError("rebalance_panel requires symbol/asset")

    out = rebalance_panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    out[sym] = out[sym].astype(str)

    for c, default in [("w_prev", 0.0), ("w_target", 0.0), ("forecast_alpha_bps", 0.0), ("adv_dollar", 1.0), ("spread_bps", 15.0), ("volatility", 0.02)]:
        if c not in out.columns:
            out[c] = default
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(default)

    if "realized_return" not in out.columns:
        out["realized_return"] = np.nan
    out["realized_return"] = pd.to_numeric(out["realized_return"], errors="coerce")

    return out.rename(columns={sym: "symbol"})


def _simulate_lambda(df: pd.DataFrame, lam: float, cfg: TurnoverVsEdgeConfig) -> tuple[dict[str, Any], pd.Series, pd.Series]:
    x = df.copy()
    delta_w = pd.to_numeric(x["w_target"], errors="coerce") - pd.to_numeric(x["w_prev"], errors="coerce")
    trade_w = lam * delta_w
    w_live = pd.to_numeric(x["w_prev"], errors="coerce") + trade_w

    trade_notional = trade_w.abs() * cfg.capital
    turnover = 0.5 * trade_w.abs()

    spread = pd.to_numeric(x["spread_bps"], errors="coerce").fillna(15.0)
    adv = pd.to_numeric(x["adv_dollar"], errors="coerce").clip(lower=1.0)
    vol = pd.to_numeric(x["volatility"], errors="coerce").fillna(0.02)

    c_spread = 0.5 * spread / 10_000.0
    c_temp = cfg.impact_y * vol * (trade_notional / (adv + 1e-9)) ** cfg.impact_delta
    c_perm = cfg.impact_gamma * (trade_notional / (adv + 1e-9))
    cost_per_dollar = c_spread + c_temp + c_perm
    cost_dollar = cost_per_dollar * trade_notional

    realized = pd.to_numeric(x["realized_return"], errors="coerce")
    if realized.notna().any():
        gross_row_ret = w_live * realized.fillna(0.0)
    else:
        alpha_bps = pd.to_numeric(x["forecast_alpha_bps"], errors="coerce").fillna(0.0)
        gross_row_ret = w_live.abs() * (alpha_bps / 10_000.0)

    net_row_ret = gross_row_ret - (cost_dollar / cfg.capital)

    daily = pd.DataFrame(
        {
            "date": x["date"],
            "gross_ret": gross_row_ret,
            "net_ret": net_row_ret,
            "turnover": turnover,
            "participation": trade_notional / (adv + 1e-9),
        }
    ).groupby("date", as_index=False).sum()

    gross_series = pd.Series(daily["gross_ret"].values, index=daily["date"])
    net_series = pd.Series(daily["net_ret"].values, index=daily["date"])

    mu_gross = float(gross_series.mean() * 252.0)
    mu_net = float(net_series.mean() * 252.0)
    vol_gross = float(gross_series.std(ddof=0) * np.sqrt(252.0))
    vol_net = float(net_series.std(ddof=0) * np.sqrt(252.0))

    sr_gross = float((mu_gross - cfg.rf_annual) / (vol_gross + 1e-12))
    sr_net = float((mu_net - cfg.rf_annual) / (vol_net + 1e-12))

    res = {
        "grid_value": float(lam),
        "turnover_annualized": float(daily["turnover"].mean() * 252.0),
        "gross_edge_bps": float(mu_gross * 10_000.0),
        "cost_drag_bps": float((mu_gross - mu_net) * 10_000.0),
        "net_edge_bps": float(mu_net * 10_000.0),
        "gross_return": mu_gross,
        "net_return": mu_net,
        "gross_sharpe": sr_gross,
        "net_sharpe": sr_net,
        "participation_mean": float(daily["participation"].mean()),
        "participation_p95": float(daily["participation"].quantile(0.95)),
        "capacity_stress": float((daily["participation"] > cfg.pi_max).mean()),
    }

    return res, gross_series, net_series

def _bootstrap_ci(series: pd.Series, cfg: TurnoverVsEdgeConfig) -> tuple[float, float, float, float]:
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

    sigma = float(np.std(x, ddof=0) * np.sqrt(252.0))
    sr_boot = (boot_mean * 252.0 - cfg.rf_annual) / (sigma + 1e-12)

    return (
        float(np.percentile(boot_mean * 252.0, 2.5)),
        float(np.percentile(boot_mean * 252.0, 97.5)),
        float(np.percentile(sr_boot, 2.5)),
        float(np.percentile(sr_boot, 97.5)),
    )


def persist_turnover_curve(curve: pd.DataFrame, manifest: Mapping[str, Any], *, output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "turnover_vs_edge": write_parquet_safe(curve, out / "turnover_vs_edge.parquet"),
        "manifest": write_json_safe(dict(manifest), out / "manifest.json"),
    }


def run_turnover_vs_edge(
    rebalance_panel: pd.DataFrame | str | Path,
    *,
    config: TurnoverVsEdgeConfig | None = None,
    run_id: str | None = None,
    regime: str | None = None,
    data_snapshot_id: str = "unknown",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or TurnoverVsEdgeConfig()
    rid = run_id or run_id_with_prefix("turnedge")

    panel = _normalize_panel(_to_df(rebalance_panel))

    rows: list[dict[str, Any]] = []
    net_by_lambda: dict[float, pd.Series] = {}

    for lam in cfg.throttle_grid:
        res, _, net_series = _simulate_lambda(panel, float(lam), cfg)

        ci_mu_l = ci_mu_h = ci_sr_l = ci_sr_h = float("nan")
        if cfg.bootstrap_enabled:
            ci_mu_l, ci_mu_h, ci_sr_l, ci_sr_h = _bootstrap_ci(net_series, cfg)

        status = "admissible" if (res["net_sharpe"] >= cfg.sr_min and res["participation_mean"] <= cfg.pi_max) else "inadmissible"

        rows.append(
            {
                **res,
                "status": status,
                "marginal_mu_net": np.nan,
                "marginal_sr_net": np.nan,
                "ci_mu_net_low": ci_mu_l,
                "ci_mu_net_high": ci_mu_h,
                "ci_sr_net_low": ci_sr_l,
                "ci_sr_net_high": ci_sr_h,
                "regime": regime,
                "notes": "",
            }
        )
        net_by_lambda[float(lam)] = net_series

    curve = pd.DataFrame(rows).sort_values("grid_value").reset_index(drop=True)
    curve["marginal_mu_net"] = curve["net_return"].diff() / curve["grid_value"].diff().replace(0, np.nan)
    curve["marginal_sr_net"] = curve["net_sharpe"].diff() / curve["grid_value"].diff().replace(0, np.nan)
    curve["run_id"] = rid

    manifest = {
        "run_id": rid,
        "config_hash": config_hash(cfg.__dict__, n=24),
        "data_snapshot_id": data_snapshot_id,
        "n_grid_points": int(len(curve)),
        "execution_timestamp": utc_now_iso(),
    }

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_turnover_curve(curve, manifest, output_dir=output_dir)

    return {
        "turnover_vs_edge": curve,
        "manifest": json_safe(manifest),
        "artifacts": artifacts,
    }


__all__ = [
    "TurnoverVsEdgeConfig",
    "persist_turnover_curve",
    "run_turnover_vs_edge",
]
