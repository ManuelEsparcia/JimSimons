
"""
risk/cov_shrinkage.py - Robust ex-ante covariance with linear shrinkage.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CovShrinkageConfig:
    lookback: int = 252
    min_obs: int | None = None
    target_type: str = "diag"  # diag | identity | const_corr | factor | auto
    shrinkage_method: str = "ledoit_wolf"  # ledoit_wolf | oas | fixed
    fixed_delta: float | None = None
    psd_policy: str = "eigenvalue_floor"
    diag_floor: float = 1e-8
    eps_psd: float = 1e-10
    jitter: float = 1e-8
    max_condition_number: float = 1e7
    nan_policy: str = "filter_assets"
    fallback_policy: str = "conservative"


def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_id_with_prefix(prefix: str) -> str:
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}"


def config_hash(cfg: Mapping[str, Any], n: int = 24) -> str:
    import hashlib
    import json

    blob = json.dumps(dict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:n]


def _to_returns_matrix(returns: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(returns, (str, Path)):
        p = Path(returns)
        if p.suffix.lower() == ".parquet":
            returns = pd.read_parquet(p)
        else:
            returns = pd.read_csv(p)

    df = returns.copy()

    # Long format: (date,symbol,return)
    if {"date", "symbol"}.issubset(df.columns):
        ret_col = "return"
        if ret_col not in df.columns:
            for c in ["ret", "r", "return_1d"]:
                if c in df.columns:
                    ret_col = c
                    break
            else:
                raise ValueError("returns long format requires return column")
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = df["symbol"].astype(str)
        df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")
        wide = df.pivot_table(index="date", columns="symbol", values=ret_col, aggfunc="last")
        wide = wide.sort_index()
        return wide

    # Wide format: index as date or has date column.
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as exc:
            raise ValueError("returns matrix index must be datetime-like") from exc

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_index()


def _min_obs(cfg: CovShrinkageConfig) -> int:
    if cfg.min_obs is not None:
        return int(cfg.min_obs)
    return int(max(30, np.floor(0.6 * cfg.lookback)))


def _sample_cov(x: pd.DataFrame) -> pd.DataFrame:
    s = x.cov(min_periods=2)
    s = s.replace([np.inf, -np.inf], np.nan)
    s = 0.5 * (s + s.T)
    diag = np.diag(np.nan_to_num(np.diag(s.values), nan=0.0))
    s = s.fillna(0.0)
    if np.any(np.diag(s.values) <= 0):
        vals = np.maximum(np.diag(s.values), 1e-8)
        np.fill_diagonal(s.values, vals)
    return s


def _target_cov(sample_cov: pd.DataFrame, target_type: str, factor_target: pd.DataFrame | None = None) -> pd.DataFrame:
    s = sample_cov.copy()
    n = s.shape[0]
    idx = s.index

    tt = target_type.lower()
    if tt in {"diag", "diagonal"}:
        t = np.diag(np.diag(s.values))
        return pd.DataFrame(t, index=idx, columns=idx)

    if tt in {"identity", "id"}:
        v = float(np.mean(np.diag(s.values)))
        t = np.eye(n) * max(v, 1e-8)
        return pd.DataFrame(t, index=idx, columns=idx)

    if tt in {"const_corr", "constant_correlation"}:
        std = np.sqrt(np.maximum(np.diag(s.values), 1e-12))
        corr = s.values / np.outer(std, std)
        np.fill_diagonal(corr, 1.0)
        off = corr[~np.eye(n, dtype=bool)]
        rho_bar = float(np.nanmean(off)) if off.size else 0.0
        rho_bar = float(np.clip(rho_bar, -0.99, 0.99))
        c = np.full((n, n), rho_bar)
        np.fill_diagonal(c, 1.0)
        t = c * np.outer(std, std)
        return pd.DataFrame(t, index=idx, columns=idx)

    if tt == "factor" and factor_target is not None:
        ft = factor_target.copy()
        ft = ft.reindex(index=idx, columns=idx)
        ft = ft.fillna(0.0)
        return 0.5 * (ft + ft.T)

    # Fallback
    t = np.diag(np.diag(s.values))
    return pd.DataFrame(t, index=idx, columns=idx)

def _estimate_delta(
    x: pd.DataFrame,
    sample_cov: pd.DataFrame,
    target_cov: pd.DataFrame,
    cfg: CovShrinkageConfig,
) -> tuple[float, str]:
    method = cfg.shrinkage_method.lower()

    if method == "fixed":
        if cfg.fixed_delta is not None:
            return float(np.clip(cfg.fixed_delta, 0.0, 1.0)), "fixed_delta"
        n = sample_cov.shape[0]
        t = max(x.shape[0], 1)
        d = float(np.clip(n / max(t, 1), 0.05, 0.95))
        return d, "fixed_heuristic"

    x_imp = x.copy()
    for c in x_imp.columns:
        m = float(pd.to_numeric(x_imp[c], errors="coerce").mean())
        x_imp[c] = pd.to_numeric(x_imp[c], errors="coerce").fillna(m)
    x_mat = x_imp.values

    s = sample_cov.values
    t = target_cov.values
    denom = float(np.sum((t - s) ** 2))

    if denom <= 1e-18:
        return 0.0, "degenerate_target"

    if method in {"ledoit_wolf", "oas"}:
        try:
            if method == "ledoit_wolf":
                from sklearn.covariance import LedoitWolf

                est = LedoitWolf().fit(x_mat)
                s_est = est.covariance_
            else:
                from sklearn.covariance import OAS

                est = OAS().fit(x_mat)
                s_est = est.covariance_

            num = float(np.sum((s_est - s) * (t - s)))
            delta = float(np.clip(num / denom, 0.0, 1.0))
            return delta, method
        except Exception:
            n = s.shape[0]
            tt = max(x_mat.shape[0], 1)
            delta = float(np.clip(n / max(tt, 1), 0.05, 0.95))
            return delta, f"{method}_fallback_heuristic"

    # Unknown -> heuristic
    n = s.shape[0]
    tt = max(x_mat.shape[0], 1)
    return float(np.clip(n / max(tt, 1), 0.05, 0.95)), "unknown_method_fallback"


def _symmetrize(m: pd.DataFrame) -> pd.DataFrame:
    out = 0.5 * (m.values + m.values.T)
    return pd.DataFrame(out, index=m.index, columns=m.columns)


def _psd_repair(m: pd.DataFrame, eps_psd: float, diag_floor: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    s = _symmetrize(m)
    vals, vecs = np.linalg.eigh(s.values)
    lam_max = float(np.max(vals)) if vals.size else 0.0
    floor_abs = max(float(eps_psd), float(diag_floor) * max(lam_max, 1.0))

    min_before = float(np.min(vals)) if vals.size else float("nan")
    vals_fixed = np.maximum(vals, floor_abs)
    num_fixed = int(np.sum(vals < floor_abs))

    s_psd = vecs @ np.diag(vals_fixed) @ vecs.T
    s_psd = 0.5 * (s_psd + s_psd.T)

    d = np.diag(s_psd)
    d = np.maximum(d, diag_floor)
    np.fill_diagonal(s_psd, d)

    vals_after = np.linalg.eigvalsh(s_psd)
    min_after = float(np.min(vals_after)) if vals_after.size else float("nan")

    out = pd.DataFrame(s_psd, index=m.index, columns=m.columns)
    return out, {
        "min_eig_before": min_before,
        "min_eig_after": min_after,
        "num_fixed_eigs": num_fixed,
    }


def _condition_number(s: pd.DataFrame) -> float:
    vals = np.linalg.eigvalsh(_symmetrize(s).values)
    vals = np.clip(vals, 1e-15, None)
    return float(np.max(vals) / np.min(vals))


def _effective_rank(s: pd.DataFrame) -> float:
    vals = np.linalg.eigvalsh(_symmetrize(s).values)
    vals = np.clip(vals, 0.0, None)
    total = float(np.sum(vals))
    if total <= 0:
        return float("nan")
    p = vals / total
    p = np.clip(p, 1e-15, 1.0)
    return float(np.exp(-np.sum(p * np.log(p))))


def _apply_jitter(s: pd.DataFrame, jitter: float, max_cond: float) -> tuple[pd.DataFrame, float]:
    out = s.copy()
    used = 0.0
    cond = _condition_number(out)
    scale = jitter
    it = 0
    while np.isfinite(cond) and cond > max_cond and it < 8:
        out = out + np.eye(out.shape[0]) * scale
        used += scale
        scale *= 10.0
        cond = _condition_number(out)
        it += 1
    return out, used


def _save_matrix_with_index(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=True)
        return str(path)
    except Exception:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=True)
        return str(csv_path)


def _save_json(payload: Mapping[str, Any], path: Path) -> str:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, default=str), encoding="utf-8")
    return str(path)


def persist_covariance_artifacts(
    cov_shrunk: pd.DataFrame,
    cov_sample: pd.DataFrame,
    cov_target: pd.DataFrame,
    diagnostics: Mapping[str, Any],
    universe_stats: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    return {
        "cov_shrunk": _save_matrix_with_index(cov_shrunk, out / "cov_shrunk.parquet"),
        "cov_sample": _save_matrix_with_index(cov_sample, out / "cov_sample.parquet"),
        "cov_target": _save_matrix_with_index(cov_target, out / "cov_target.parquet"),
        "diagnostics": _save_json(dict(diagnostics), out / "diagnostics.json"),
        "universe_stats": _save_json(dict(universe_stats), out / "universe_stats.json"),
    }

def run_cov_shrinkage(
    returns: pd.DataFrame | str | Path,
    *,
    config: CovShrinkageConfig | None = None,
    calc_date: str | pd.Timestamp | None = None,
    run_id: str | None = None,
    factor_target: pd.DataFrame | str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or CovShrinkageConfig()
    rid = run_id or run_id_with_prefix("covshr")

    x = _to_returns_matrix(returns)
    if len(x) == 0 or x.shape[1] == 0:
        raise ValueError("empty returns matrix")

    x = x.sort_index()
    if calc_date is not None:
        cd = pd.Timestamp(calc_date)
        x = x.loc[x.index <= cd]
        if len(x) == 0:
            raise ValueError("no returns available at or before calc_date")

    x = x.iloc[-cfg.lookback :].copy()
    n_input = int(x.shape[1])

    obs = x.notna().sum(axis=0)
    min_obs = _min_obs(cfg)
    used_cols = obs[obs >= min_obs].index.tolist()
    excluded_cols = sorted(set(x.columns) - set(used_cols))

    if len(used_cols) < 2:
        # Conservative terminal fallback: use top-coverage assets if possible.
        used_cols = obs.sort_values(ascending=False).head(min(2, len(obs))).index.tolist()

    x_use = x[used_cols].copy()
    if x_use.shape[1] < 2:
        raise ValueError("insufficient assets after coverage filter")

    cov_sample = _sample_cov(x_use)

    ft = None
    if factor_target is not None:
        if isinstance(factor_target, (str, Path)):
            p = Path(factor_target)
            ft = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p, index_col=0)
        else:
            ft = factor_target.copy()

    tt = cfg.target_type.lower()
    if tt == "auto":
        t_effective = int(x_use.shape[0])
        n_used = int(x_use.shape[1])
        if t_effective >= 3 * n_used and float(x_use.notna().mean().mean()) >= 0.85:
            tt = "const_corr"
        else:
            tt = "diag"

    cov_target = _target_cov(cov_sample, tt, factor_target=ft)

    delta, delta_source = _estimate_delta(x_use, cov_sample, cov_target, cfg)
    cov_shrunk = (1.0 - delta) * cov_sample + delta * cov_target
    cov_shrunk = _symmetrize(cov_shrunk)

    cov_shrunk, psd_diag = _psd_repair(cov_shrunk, eps_psd=cfg.eps_psd, diag_floor=cfg.diag_floor)
    cond_pre = _condition_number(cov_shrunk)
    cov_shrunk, jitter_used = _apply_jitter(cov_shrunk, cfg.jitter, cfg.max_condition_number)
    cond_post = _condition_number(cov_shrunk)

    fallback_used = False
    fallback_reason = ""

    if (not np.isfinite(cond_post)) or cond_post > cfg.max_condition_number * 5:
        fallback_used = True
        fallback_reason = "conditioning_failure"
        cov_target_fb = _target_cov(cov_sample, "diag")
        cov_shrunk = 0.2 * cov_sample + 0.8 * cov_target_fb
        cov_shrunk, psd_fb = _psd_repair(cov_shrunk, eps_psd=cfg.eps_psd, diag_floor=cfg.diag_floor)
        psd_diag = psd_fb
        cov_shrunk, jitter_used_fb = _apply_jitter(cov_shrunk, cfg.jitter, cfg.max_condition_number)
        jitter_used += jitter_used_fb
        cond_post = _condition_number(cov_shrunk)

    diagnostics = {
        "run_id": rid,
        "calc_date": str(x_use.index.max().date()),
        "lookback": int(cfg.lookback),
        "N_input": n_input,
        "N_used": int(x_use.shape[1]),
        "T_effective": int(x_use.shape[0]),
        "target_type": tt,
        "shrinkage_method": cfg.shrinkage_method,
        "delta": float(delta),
        "delta_source": delta_source,
        "coverage_mean": float(x_use.notna().mean().mean()),
        "coverage_min": float(x_use.notna().mean().min()),
        "cond_num": float(cond_post),
        "effective_rank": _effective_rank(cov_shrunk),
        "min_eig_before": psd_diag["min_eig_before"],
        "min_eig_after": psd_diag["min_eig_after"],
        "num_fixed_eigs": psd_diag["num_fixed_eigs"],
        "psd_policy": cfg.psd_policy,
        "jitter_used": float(jitter_used),
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "config_hash": config_hash(cfg.__dict__, n=24),
        "execution_timestamp": utc_now_iso(),
    }

    universe_stats = {
        "run_id": rid,
        "N_input": n_input,
        "N_used": int(x_use.shape[1]),
        "coverage_mean": float(x_use.notna().mean().mean()),
        "coverage_min": float(x_use.notna().mean().min()),
        "excluded_symbols": excluded_cols,
        "min_obs": int(min_obs),
    }

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_covariance_artifacts(
            cov_shrunk,
            cov_sample,
            cov_target,
            diagnostics,
            universe_stats,
            output_dir=output_dir,
        )

    return {
        "cov_shrunk": cov_shrunk,
        "cov_sample": cov_sample,
        "cov_target": cov_target,
        "diagnostics": diagnostics,
        "universe_stats": universe_stats,
        "artifacts": artifacts,
    }


__all__ = [
    "CovShrinkageConfig",
    "persist_covariance_artifacts",
    "run_cov_shrinkage",
]
