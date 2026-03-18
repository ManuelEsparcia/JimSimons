"""
research/alpha_discovery/ic_stability.py - Temporal IC robustness auditor.

Audits out-of-sample signal stability using daily cross-sectional IC time series,
with HAC inference, block/regime consistency diagnostics, and explicit gates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from simons_smallcap_swing.research._shared import (
    compute_newey_west_tstat,
    config_hash,
    json_safe,
    read_dataframe,
    run_id_with_prefix,
    utc_now_iso,
    write_json_safe,
    write_parquet_safe,
)


@dataclass(frozen=True)
class ICStabilityConfig:
    rolling_windows: tuple[int, ...] = (21, 63, 126, 252)
    recent_window: int = 63
    block_scheme: str = "yearly"  # yearly | quarterly | monthly
    nw_lags: int = 10
    min_valid_dates: int = 126
    min_names_per_date: int = 25
    min_mean_coverage: float = 0.80
    min_dates_per_regime: int = 30
    min_mean_ic: float = 0.010
    min_hit_rate_positive: float = 0.52
    min_nw_tstat: float = 2.0
    max_recent_decay_abs: float = 0.020
    max_block_std: float = 0.020
    warn_mean_ic: float = 0.015
    warn_nw_tstat: float = 2.5
    warn_hit_rate: float = 0.55
    gate_policy_version: str = "1.0"


def _to_df(obj: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    return read_dataframe(obj)


def _normalize_signals(signals: pd.DataFrame, signal_col: str) -> pd.DataFrame:
    required = {"date", "symbol"}
    if not required.issubset(signals.columns):
        raise ValueError("signals panel must contain date and symbol")

    out = signals.copy()
    if "alpha_id" not in out.columns:
        out["alpha_id"] = "alpha_0"

    if signal_col not in out.columns:
        # fallback aliases
        aliases = ["signal", "score", "alpha_score", "value"]
        for a in aliases:
            if a in out.columns:
                signal_col = a
                break
        else:
            raise ValueError(f"signals panel missing signal column: {signal_col}")

    out = out[["date", "symbol", "alpha_id", signal_col]].rename(columns={signal_col: "score"})
    out["date"] = pd.to_datetime(out["date"])
    out["alpha_id"] = out["alpha_id"].astype(str)
    out["symbol"] = out["symbol"].astype(str)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    return out


def _normalize_labels(labels: pd.DataFrame, label_col: str) -> pd.DataFrame:
    required = {"date", "symbol"}
    if not required.issubset(labels.columns):
        raise ValueError("labels panel must contain date and symbol")

    out = labels.copy()
    if label_col not in out.columns:
        aliases = ["label", "target", "forward_return", "y"]
        for a in aliases:
            if a in out.columns:
                label_col = a
                break
        else:
            raise ValueError(f"labels panel missing label column: {label_col}")

    out = out[["date", "symbol", label_col]].rename(columns={label_col: "label"})
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    return out


def _normalize_universe(universe_mask: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol"}
    if not required.issubset(universe_mask.columns):
        raise ValueError("universe mask must contain date and symbol")

    out = universe_mask.copy()
    eligible_col = "eligible"
    if eligible_col not in out.columns:
        candidates = ["in_universe", "is_eligible", "mask", "active"]
        for c in candidates:
            if c in out.columns:
                eligible_col = c
                break
        else:
            out[eligible_col] = 1
    out = out[["date", "symbol", eligible_col]].rename(columns={eligible_col: "eligible"})
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)
    out["eligible"] = out["eligible"].fillna(0).astype(int).clip(0, 1)
    return out


def _normalize_regimes(regime_labels: pd.DataFrame) -> pd.DataFrame:
    if "date" not in regime_labels.columns:
        raise ValueError("regime labels must contain date")
    out = regime_labels.copy()
    label_col = "regime_label"
    if label_col not in out.columns:
        for c in ["regime", "state", "regime_name"]:
            if c in out.columns:
                label_col = c
                break
        else:
            raise ValueError("regime labels missing regime_label/regime/state column")
    out = out[["date", label_col]].rename(columns={label_col: "regime_label"})
    out["date"] = pd.to_datetime(out["date"])
    out["regime_label"] = out["regime_label"].astype(str)
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out


def _normalize_splits(splits: pd.DataFrame) -> pd.DataFrame:
    if "date" not in splits.columns:
        raise ValueError("splits must contain date")
    out = splits.copy()
    role_col = "split_role"
    if role_col not in out.columns:
        for c in ["role", "fold_role", "partition"]:
            if c in out.columns:
                role_col = c
                break
        else:
            out[role_col] = "oos"
    out = out[["date", role_col]].rename(columns={role_col: "split_role"})
    out["date"] = pd.to_datetime(out["date"])
    out["split_role"] = out["split_role"].astype(str)
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out


def align_panels(
    signals: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    universe_mask: pd.DataFrame | None = None,
    regime_labels: pd.DataFrame | None = None,
    splits: pd.DataFrame | None = None,
) -> pd.DataFrame:
    panel = signals.merge(labels, on=["date", "symbol"], how="left")

    if universe_mask is not None:
        panel = panel.merge(universe_mask, on=["date", "symbol"], how="left")
        panel["eligible"] = panel["eligible"].fillna(0).astype(int).clip(0, 1)
    else:
        panel["eligible"] = 1

    if regime_labels is not None:
        panel = panel.merge(regime_labels, on="date", how="left")
    else:
        panel["regime_label"] = pd.NA

    if splits is not None:
        panel = panel.merge(splits, on="date", how="left")
    else:
        panel["split_role"] = pd.NA

    panel = panel.sort_values(["alpha_id", "date", "symbol"]).reset_index(drop=True)

    dup = panel.duplicated(subset=["alpha_id", "date", "symbol"])
    if bool(dup.any()):
        raise ValueError("duplicate rows detected for (alpha_id, date, symbol)")

    return panel


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 3:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))


def compute_daily_ic(panel: pd.DataFrame, config: ICStabilityConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (alpha_id, date), grp in panel.groupby(["alpha_id", "date"], sort=True):
        eligible = grp["eligible"].fillna(0).astype(int) > 0
        n_eligible = int(eligible.sum())

        valid = eligible & grp["score"].notna() & grp["label"].notna()
        n_valid = int(valid.sum())

        coverage = float(n_valid / n_eligible) if n_eligible > 0 else float("nan")
        ic_value = float("nan")
        if n_valid >= config.min_names_per_date:
            ic_value = _safe_spearman(grp.loc[valid, "score"], grp.loc[valid, "label"])

        regime_label = pd.NA
        if "regime_label" in grp.columns and grp["regime_label"].notna().any():
            regime_label = str(grp["regime_label"].dropna().iloc[0])

        split_role = pd.NA
        if "split_role" in grp.columns and grp["split_role"].notna().any():
            split_role = str(grp["split_role"].dropna().iloc[0])

        rows.append(
            {
                "alpha_id": str(alpha_id),
                "date": pd.Timestamp(date),
                "ic_value": ic_value,
                "n_valid_names": n_valid,
                "n_eligible_names": n_eligible,
                "coverage": coverage,
                "regime_label": regime_label,
                "split_role": split_role,
            }
        )

    return pd.DataFrame(rows).sort_values(["alpha_id", "date"]).reset_index(drop=True)


def _block_label(date: pd.Timestamp, scheme: str) -> str:
    d = pd.Timestamp(date)
    s = scheme.lower()
    if s in {"year", "yearly"}:
        return f"{d.year}"
    if s in {"quarter", "quarterly"}:
        return f"{d.year}Q{d.quarter}"
    if s in {"month", "monthly"}:
        return f"{d.year}-{d.month:02d}"
    return f"{d.year}"


def compute_block_stats(ic_daily: pd.DataFrame, config: ICStabilityConfig) -> pd.DataFrame:
    if len(ic_daily) == 0:
        return pd.DataFrame(
            columns=[
                "alpha_id",
                "block_id",
                "start_date",
                "end_date",
                "n_dates",
                "mean_ic",
                "median_ic",
                "std_ic",
                "hit_rate",
                "coverage_mean",
            ]
        )

    tmp = ic_daily.copy()
    tmp["block_id"] = tmp["date"].map(lambda d: _block_label(pd.Timestamp(d), config.block_scheme))

    rows: list[dict[str, Any]] = []
    for (alpha_id, block_id), grp in tmp.groupby(["alpha_id", "block_id"], sort=True):
        ic = pd.to_numeric(grp["ic_value"], errors="coerce").dropna()
        rows.append(
            {
                "alpha_id": str(alpha_id),
                "block_id": str(block_id),
                "start_date": pd.Timestamp(grp["date"].min()),
                "end_date": pd.Timestamp(grp["date"].max()),
                "n_dates": int(ic.size),
                "mean_ic": float(ic.mean()) if ic.size else float("nan"),
                "median_ic": float(ic.median()) if ic.size else float("nan"),
                "std_ic": float(ic.std(ddof=0)) if ic.size else float("nan"),
                "hit_rate": float((ic > 0).mean()) if ic.size else float("nan"),
                "coverage_mean": float(pd.to_numeric(grp["coverage"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["alpha_id", "block_id"]).reset_index(drop=True)


def compute_regime_stats(ic_daily: pd.DataFrame) -> pd.DataFrame:
    if "regime_label" not in ic_daily.columns:
        return pd.DataFrame(
            columns=[
                "alpha_id",
                "regime_label",
                "n_dates",
                "mean_ic",
                "std_ic",
                "hit_rate",
                "coverage_mean",
            ]
        )

    tmp = ic_daily[ic_daily["regime_label"].notna()].copy()
    if len(tmp) == 0:
        return pd.DataFrame(
            columns=[
                "alpha_id",
                "regime_label",
                "n_dates",
                "mean_ic",
                "std_ic",
                "hit_rate",
                "coverage_mean",
            ]
        )

    rows: list[dict[str, Any]] = []
    for (alpha_id, regime), grp in tmp.groupby(["alpha_id", "regime_label"], sort=True):
        ic = pd.to_numeric(grp["ic_value"], errors="coerce").dropna()
        rows.append(
            {
                "alpha_id": str(alpha_id),
                "regime_label": str(regime),
                "n_dates": int(ic.size),
                "mean_ic": float(ic.mean()) if ic.size else float("nan"),
                "std_ic": float(ic.std(ddof=0)) if ic.size else float("nan"),
                "hit_rate": float((ic > 0).mean()) if ic.size else float("nan"),
                "coverage_mean": float(pd.to_numeric(grp["coverage"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["alpha_id", "regime_label"]).reset_index(drop=True)


def compute_halflife(ic_series: pd.Series) -> float:
    s = pd.to_numeric(ic_series, errors="coerce").dropna().values
    if s.size < 20:
        return float("nan")

    x = s[:-1]
    y = s[1:]
    denom = float(np.dot(x, x))
    if abs(denom) < 1e-12:
        return float("nan")

    phi = float(np.dot(x, y) / denom)
    phi_abs = abs(phi)
    if phi_abs <= 0 or phi_abs >= 1:
        return float("nan")

    return float(-math.log(2.0) / math.log(phi_abs))


def _compute_recent_stats(ic_series: pd.Series, recent_window: int) -> tuple[float, float]:
    s = pd.to_numeric(ic_series, errors="coerce").dropna()
    if len(s) < max(10, recent_window + 5):
        return float("nan"), float("nan")

    recent = s.iloc[-recent_window:]
    prev = s.iloc[:-recent_window]
    if len(prev) < 5:
        return float("nan"), float("nan")

    recent_mean = float(recent.mean())
    prev_mean = float(prev.mean())
    decay = recent_mean - prev_mean
    ratio = recent_mean / (abs(prev_mean) + 1e-12)
    return float(decay), float(ratio)


def _evaluate_gate(
    row: Mapping[str, Any],
    *,
    config: ICStabilityConfig,
    min_regime_dates: int | None,
) -> tuple[bool, bool, bool, str]:
    fails: list[str] = []
    warns: list[str] = []

    mean_ic = float(row.get("mean_ic", np.nan))
    hit_rate = float(row.get("hit_rate_positive", np.nan))
    nw_t = float(row.get("nw_tstat", np.nan))
    coverage_mean = float(row.get("coverage_mean", np.nan))
    n_valid_dates = int(row.get("n_valid_dates", 0))
    recent_decay = float(row.get("recent_decay", np.nan))
    block_std = float(row.get("block_std", np.nan))

    if n_valid_dates < config.min_valid_dates:
        fails.append("n_valid_dates")
    if not np.isfinite(coverage_mean) or coverage_mean < config.min_mean_coverage:
        fails.append("coverage_mean")
    if not np.isfinite(mean_ic) or mean_ic < -max(config.min_mean_ic, 1e-4):
        fails.append("mean_ic_sign")
    if not np.isfinite(nw_t) or nw_t < max(0.5, config.min_nw_tstat * 0.5):
        fails.append("nw_tstat")
    if np.isfinite(recent_decay) and abs(recent_decay) > config.max_recent_decay_abs * 1.5:
        fails.append("recent_decay")
    if np.isfinite(block_std) and block_std > config.max_block_std * 1.5:
        fails.append("block_std")
    if min_regime_dates is not None and min_regime_dates < config.min_dates_per_regime:
        fails.append("min_dates_per_regime")

    if not fails:
        if mean_ic < config.warn_mean_ic:
            warns.append("mean_ic")
        if hit_rate < config.warn_hit_rate:
            warns.append("hit_rate")
        if nw_t < config.warn_nw_tstat:
            warns.append("nw_tstat")
        if np.isfinite(block_std) and block_std > config.max_block_std:
            warns.append("block_std")
        if np.isfinite(recent_decay) and abs(recent_decay) > config.max_recent_decay_abs:
            warns.append("recent_decay")

    if fails:
        return False, False, True, ";".join(fails)
    if warns:
        return False, True, False, ";".join(warns)
    return True, False, False, "pass"


def compute_summary_stats(
    ic_daily: pd.DataFrame,
    ic_blocks: pd.DataFrame,
    ic_regimes: pd.DataFrame,
    *,
    config: ICStabilityConfig,
) -> pd.DataFrame:
    if len(ic_daily) == 0:
        return pd.DataFrame(
            columns=[
                "alpha_id",
                "n_valid_dates",
                "mean_ic",
                "median_ic",
                "std_ic",
                "hit_rate_positive",
                "coverage_mean",
                "coverage_std",
                "nw_tstat",
                "nw_pvalue",
                "ic_halflife",
                "recent_decay",
                "recent_ratio",
                "block_std",
                "regime_dispersion",
                "pass_flag",
                "warn_flag",
                "fail_flag",
                "binding_reason",
            ]
        )

    block_disp = (
        ic_blocks.groupby("alpha_id", as_index=False)["mean_ic"]
        .std(ddof=0)
        .rename(columns={"mean_ic": "block_std"})
    )
    regime_disp = (
        ic_regimes.groupby("alpha_id", as_index=False)["mean_ic"]
        .std(ddof=0)
        .rename(columns={"mean_ic": "regime_dispersion"})
        if len(ic_regimes) > 0
        else pd.DataFrame(columns=["alpha_id", "regime_dispersion"])
    )
    regime_min_dates = (
        ic_regimes.groupby("alpha_id")["n_dates"].min().to_dict()
        if len(ic_regimes) > 0
        else {}
    )

    rows: list[dict[str, Any]] = []
    for alpha_id, grp in ic_daily.groupby("alpha_id", sort=True):
        ic = pd.to_numeric(grp["ic_value"], errors="coerce").dropna()
        coverage = pd.to_numeric(grp["coverage"], errors="coerce")

        nw_t, nw_p = compute_newey_west_tstat(ic.values, lags=config.nw_lags)
        recent_decay, recent_ratio = _compute_recent_stats(ic, config.recent_window)

        row: dict[str, Any] = {
            "alpha_id": str(alpha_id),
            "n_valid_dates": int(ic.size),
            "mean_ic": float(ic.mean()) if ic.size else float("nan"),
            "median_ic": float(ic.median()) if ic.size else float("nan"),
            "std_ic": float(ic.std(ddof=0)) if ic.size else float("nan"),
            "hit_rate_positive": float((ic > 0).mean()) if ic.size else float("nan"),
            "coverage_mean": float(coverage.mean()),
            "coverage_std": float(coverage.std(ddof=0)),
            "nw_tstat": float(nw_t),
            "nw_pvalue": float(nw_p),
            "ic_halflife": compute_halflife(ic),
            "recent_decay": float(recent_decay),
            "recent_ratio": float(recent_ratio),
            "block_std": float("nan"),
            "regime_dispersion": float("nan"),
        }

        block_val = block_disp.loc[block_disp["alpha_id"] == str(alpha_id), "block_std"]
        if len(block_val):
            row["block_std"] = float(block_val.iloc[0])

        regime_val = regime_disp.loc[regime_disp["alpha_id"] == str(alpha_id), "regime_dispersion"]
        if len(regime_val):
            row["regime_dispersion"] = float(regime_val.iloc[0])

        min_reg_dates = regime_min_dates.get(str(alpha_id))
        p, w, f, reason = _evaluate_gate(row, config=config, min_regime_dates=min_reg_dates)
        row["pass_flag"] = bool(p)
        row["warn_flag"] = bool(w)
        row["fail_flag"] = bool(f)
        row["binding_reason"] = reason
        rows.append(row)

    return pd.DataFrame(rows).sort_values("alpha_id").reset_index(drop=True)


def persist_results(
    ic_daily: pd.DataFrame,
    ic_summary: pd.DataFrame,
    ic_blocks: pd.DataFrame,
    ic_regimes: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "ic_daily": write_parquet_safe(ic_daily, out / "ic_daily.parquet"),
        "ic_summary": write_parquet_safe(ic_summary, out / "ic_summary.parquet"),
        "ic_blocks": write_parquet_safe(ic_blocks, out / "ic_blocks.parquet"),
        "ic_regimes": write_parquet_safe(ic_regimes, out / "ic_regimes.parquet"),
        "manifest": write_json_safe(dict(manifest), out / "manifest.json"),
    }


def run_ic_stability(
    signals_panel: pd.DataFrame | str | Path,
    labels_panel: pd.DataFrame | str | Path,
    *,
    universe_mask: pd.DataFrame | str | Path | None = None,
    regime_labels: pd.DataFrame | str | Path | None = None,
    splits: pd.DataFrame | str | Path | None = None,
    signal_col: str = "signal",
    label_col: str = "label",
    config: ICStabilityConfig | None = None,
    run_id: str | None = None,
    data_snapshot_id: str = "unknown",
    signals_version: str = "unknown",
    labels_version: str = "unknown",
    regime_version: str = "none",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or ICStabilityConfig()
    rid = run_id or run_id_with_prefix("icstab")

    signals = _normalize_signals(_to_df(signals_panel), signal_col=signal_col)
    labels = _normalize_labels(_to_df(labels_panel), label_col=label_col)
    univ = _normalize_universe(_to_df(universe_mask)) if universe_mask is not None else None
    regimes = _normalize_regimes(_to_df(regime_labels)) if regime_labels is not None else None
    fold_map = _normalize_splits(_to_df(splits)) if splits is not None else None

    panel = align_panels(
        signals,
        labels,
        universe_mask=univ,
        regime_labels=regimes,
        splits=fold_map,
    )

    ic_daily = compute_daily_ic(panel, cfg)
    ic_daily["run_id"] = rid

    ic_blocks = compute_block_stats(ic_daily, cfg)
    ic_regimes = compute_regime_stats(ic_daily)

    ic_summary = compute_summary_stats(ic_daily, ic_blocks, ic_regimes, config=cfg)
    ic_summary["run_id"] = rid

    if len(ic_blocks):
        ic_blocks["run_id"] = rid
    else:
        ic_blocks = ic_blocks.assign(run_id=pd.Series(dtype=object))

    if len(ic_regimes):
        ic_regimes["run_id"] = rid
    else:
        ic_regimes = ic_regimes.assign(run_id=pd.Series(dtype=object))

    manifest = {
        "run_id": rid,
        "config_hash": config_hash(cfg.__dict__, n=24),
        "data_snapshot_id": data_snapshot_id,
        "labels_version": labels_version,
        "signals_version": signals_version,
        "regime_version": regime_version if regimes is not None else "none",
        "n_alphas": int(ic_daily["alpha_id"].nunique()) if len(ic_daily) else 0,
        "execution_timestamp": utc_now_iso(),
        "gate_policy_version": cfg.gate_policy_version,
    }

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_results(ic_daily, ic_summary, ic_blocks, ic_regimes, manifest, output_dir=output_dir)

    return {
        "ic_daily": ic_daily,
        "ic_summary": ic_summary,
        "ic_blocks": ic_blocks,
        "ic_regimes": ic_regimes,
        "manifest": json_safe(manifest),
        "artifacts": artifacts,
    }


__all__ = [
    "ICStabilityConfig",
    "align_panels",
    "compute_daily_ic",
    "compute_block_stats",
    "compute_regime_stats",
    "compute_halflife",
    "compute_summary_stats",
    "persist_results",
    "run_ic_stability",
]
