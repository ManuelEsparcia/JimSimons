
"""
research/capacity_analysis/crowding_detector.py - Book crowding diagnostics.
"""
from __future__ import annotations

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
class CrowdingConfig:
    lookback: int = 252
    eps: float = 1e-8
    train_fraction: float = 0.7
    warn_quantile: float = 0.90
    hard_quantile: float = 0.97
    w_hhi: float = 0.25
    w_top10: float = 0.25
    w_liquidity_pressure: float = 0.25
    w_signal_overlap: float = 0.25
    winsor_lower: float | None = None
    winsor_upper: float | None = None
    calibration_version: str = "1.0"


def _to_df(obj: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    return read_dataframe(obj)


def _normalize_positions(positions: pd.DataFrame) -> pd.DataFrame:
    if "date" not in positions.columns:
        raise ValueError("positions_history must have date")
    sym_col = "symbol" if "symbol" in positions.columns else "asset"
    if sym_col not in positions.columns:
        raise ValueError("positions_history must have symbol/asset")

    out = positions.copy()
    out["date"] = pd.to_datetime(out["date"])
    out[sym_col] = out[sym_col].astype(str)

    if "dollar_position" not in out.columns:
        if "shares" in out.columns and "close" in out.columns:
            out["dollar_position"] = pd.to_numeric(out["shares"], errors="coerce") * pd.to_numeric(out["close"], errors="coerce")
        elif "gross_weight" in out.columns:
            out["dollar_position"] = pd.to_numeric(out["gross_weight"], errors="coerce")
        else:
            raise ValueError("positions_history requires dollar_position or (shares,close) or gross_weight")

    return out[["date", sym_col, "dollar_position"]].rename(columns={sym_col: "symbol"})


def _normalize_signals(signals: pd.DataFrame) -> pd.DataFrame:
    if "date" not in signals.columns:
        raise ValueError("signal_history must have date")
    sym_col = "symbol" if "symbol" in signals.columns else "asset"
    if sym_col not in signals.columns:
        raise ValueError("signal_history must have symbol/asset")

    out = signals.copy()
    out["date"] = pd.to_datetime(out["date"])
    out[sym_col] = out[sym_col].astype(str)

    signal_col = "signal"
    if signal_col not in out.columns:
        for c in ["score", "alpha_score", "target_weight"]:
            if c in out.columns:
                signal_col = c
                break
        else:
            raise ValueError("signal_history requires signal/score/alpha_score")

    out[signal_col] = pd.to_numeric(out[signal_col], errors="coerce")
    return out[["date", sym_col, signal_col]].rename(columns={sym_col: "symbol", signal_col: "signal"})


def _normalize_liquidity(liq: pd.DataFrame) -> pd.DataFrame:
    if "date" not in liq.columns:
        raise ValueError("liquidity_panel must have date")
    sym_col = "symbol" if "symbol" in liq.columns else "asset"
    if sym_col not in liq.columns:
        raise ValueError("liquidity_panel must have symbol/asset")

    out = liq.copy()
    out["date"] = pd.to_datetime(out["date"])
    out[sym_col] = out[sym_col].astype(str)

    adv_col = "adv_dollar"
    if adv_col not in out.columns:
        for c in ["adv", "dollar_adv", "avg_dollar_volume"]:
            if c in out.columns:
                adv_col = c
                break
        else:
            raise ValueError("liquidity_panel requires adv_dollar/adv")

    out[adv_col] = pd.to_numeric(out[adv_col], errors="coerce")
    return out[["date", sym_col, adv_col]].rename(columns={sym_col: "symbol", adv_col: "adv_dollar"})


def _mad_standardize(s: pd.Series, eps: float) -> pd.Series:
    med = float(s.median())
    mad = float((s - med).abs().median())
    return (s - med) / (mad + eps)


def _compute_daily_components(panel: pd.DataFrame, cfg: CrowdingConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for date, grp in panel.groupby("date", sort=True):
        x = pd.to_numeric(grp["dollar_position"], errors="coerce").fillna(0.0)
        gross = float(x.abs().sum())
        if gross <= cfg.eps:
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "n_names": 0,
                    "hhi": np.nan,
                    "top_5_share": np.nan,
                    "top_10_share": np.nan,
                    "liquidity_pressure": np.nan,
                    "signal_overlap": np.nan,
                }
            )
            continue

        w = x.abs() / (gross + cfg.eps)
        hhi = float((w ** 2).sum())

        ws = w.sort_values(ascending=False)
        top_5_share = float(ws.head(5).sum())
        top_10_share = float(ws.head(10).sum())

        adv = pd.to_numeric(grp["adv_dollar"], errors="coerce")
        liq_use = x.abs() / (adv + cfg.eps)
        lp = float((w * liq_use.fillna(0.0)).sum())

        signal = pd.to_numeric(grp["signal"], errors="coerce")
        s_tilde = _mad_standardize(signal.fillna(signal.median()), cfg.eps)
        w_tilde = _mad_standardize(w, cfg.eps)
        denom = float(np.sqrt((s_tilde ** 2).sum()) * np.sqrt((w_tilde ** 2).sum()) + cfg.eps)
        rho = float((s_tilde * w_tilde).sum() / denom) if denom > 0 else 0.0

        rows.append(
            {
                "date": pd.Timestamp(date),
                "n_names": int((w > 0).sum()),
                "hhi": hhi,
                "top_5_share": top_5_share,
                "top_10_share": top_10_share,
                "liquidity_pressure": lp,
                "signal_overlap": rho,
            }
        )

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out

def _calibrate_alerts(score: pd.Series, cfg: CrowdingConfig) -> tuple[float, float]:
    s = pd.to_numeric(score, errors="coerce").dropna().reset_index(drop=True)
    if len(s) == 0:
        return float("nan"), float("nan")

    cut = int(max(5, round(len(s) * cfg.train_fraction)))
    train = s.iloc[:cut]
    warn_thr = float(train.quantile(cfg.warn_quantile))
    hard_thr = float(train.quantile(cfg.hard_quantile))
    return warn_thr, hard_thr


def persist_crowding_outputs(
    scores: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "crowding_scores": write_parquet_safe(scores, out / "crowding_scores.parquet"),
        "manifest": write_json_safe(dict(manifest), out / "manifest.json"),
    }


def run_crowding_detector(
    positions_history: pd.DataFrame | str | Path,
    signal_history: pd.DataFrame | str | Path,
    liquidity_panel: pd.DataFrame | str | Path,
    *,
    config: CrowdingConfig | None = None,
    run_id: str | None = None,
    data_snapshot_id: str = "unknown",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or CrowdingConfig()
    rid = run_id or run_id_with_prefix("crowd")

    pos = _normalize_positions(_to_df(positions_history))
    sig = _normalize_signals(_to_df(signal_history))
    liq = _normalize_liquidity(_to_df(liquidity_panel))

    panel = pos.merge(sig, on=["date", "symbol"], how="left")
    panel = panel.merge(liq, on=["date", "symbol"], how="left")

    daily = _compute_daily_components(panel, cfg)

    # Shifted rolling z-scores to keep strict PIT semantics.
    daily["z_hhi"] = rolling_zscore_shifted(daily["hhi"], cfg.lookback, eps=cfg.eps)
    daily["z_top_10_share"] = rolling_zscore_shifted(daily["top_10_share"], cfg.lookback, eps=cfg.eps)
    daily["z_liquidity_pressure"] = rolling_zscore_shifted(daily["liquidity_pressure"], cfg.lookback, eps=cfg.eps)

    overlap_pos = pd.to_numeric(daily["signal_overlap"], errors="coerce").clip(lower=0.0)
    daily["z_signal_overlap"] = rolling_zscore_shifted(overlap_pos, cfg.lookback, eps=cfg.eps)

    if cfg.winsor_lower is not None or cfg.winsor_upper is not None:
        low = 0.01 if cfg.winsor_lower is None else cfg.winsor_lower
        high = 0.99 if cfg.winsor_upper is None else cfg.winsor_upper
        for c in ["z_hhi", "z_top_10_share", "z_liquidity_pressure", "z_signal_overlap"]:
            ql = float(pd.to_numeric(daily[c], errors="coerce").quantile(low))
            qh = float(pd.to_numeric(daily[c], errors="coerce").quantile(high))
            daily[c] = pd.to_numeric(daily[c], errors="coerce").clip(lower=ql, upper=qh)

    daily["contribution_hhi"] = cfg.w_hhi * pd.to_numeric(daily["z_hhi"], errors="coerce")
    daily["contribution_top_10"] = cfg.w_top10 * pd.to_numeric(daily["z_top_10_share"], errors="coerce")
    daily["contribution_lp"] = cfg.w_liquidity_pressure * pd.to_numeric(daily["z_liquidity_pressure"], errors="coerce")
    daily["contribution_overlap"] = cfg.w_signal_overlap * pd.to_numeric(daily["z_signal_overlap"], errors="coerce")

    daily["crowding_score"] = (
        daily["contribution_hhi"]
        + daily["contribution_top_10"]
        + daily["contribution_lp"]
        + daily["contribution_overlap"]
    )

    warn_thr, hard_thr = _calibrate_alerts(daily["crowding_score"], cfg)
    alert = np.where(
        pd.to_numeric(daily["crowding_score"], errors="coerce") > hard_thr,
        "hard",
        np.where(pd.to_numeric(daily["crowding_score"], errors="coerce") > warn_thr, "warn", "ok"),
    )

    daily["alert"] = alert
    daily["calibration_version"] = cfg.calibration_version
    daily["notes"] = ""
    daily["run_id"] = rid

    output_cols = [
        "date",
        "n_names",
        "hhi",
        "top_5_share",
        "top_10_share",
        "liquidity_pressure",
        "signal_overlap",
        "z_hhi",
        "z_top_10_share",
        "z_liquidity_pressure",
        "z_signal_overlap",
        "crowding_score",
        "alert",
        "calibration_version",
        "notes",
        "run_id",
        "contribution_hhi",
        "contribution_top_10",
        "contribution_lp",
        "contribution_overlap",
    ]

    crowding_scores = daily[output_cols].copy().sort_values("date").reset_index(drop=True)

    manifest = {
        "run_id": rid,
        "config_hash": config_hash(cfg.__dict__, n=24),
        "data_snapshot_id": data_snapshot_id,
        "warn_threshold": warn_thr,
        "hard_threshold": hard_thr,
        "n_dates": int(len(crowding_scores)),
        "execution_timestamp": utc_now_iso(),
    }

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_crowding_outputs(crowding_scores, manifest, output_dir=output_dir)

    return {
        "crowding_scores": crowding_scores,
        "manifest": json_safe(manifest),
        "artifacts": artifacts,
    }


__all__ = [
    "CrowdingConfig",
    "persist_crowding_outputs",
    "run_crowding_detector",
]
