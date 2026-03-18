"""
Shared helpers for research modules.

This file centralizes deterministic IO, hashing, statistical utilities, and
simple validation routines used across research/* modules.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_id_with_prefix(prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}"


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat() if pd.notna(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(json_safe(v) for v in value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(cfg: Mapping[str, Any], n: int = 16) -> str:
    blob = json.dumps(json_safe(cfg), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:n]


def stable_params_hash(params: Mapping[str, Any], n: int = 16) -> str:
    return config_hash(params, n=n)


def read_dataframe(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.exists():
        raise ValueError(f"Unsupported extension for tabular data: {p}")
    for ext in (".parquet", ".csv"):
        px = p.with_suffix(ext)
        if px.exists():
            return read_dataframe(px)
    raise FileNotFoundError(f"No tabular file found for: {p}")


def write_parquet_safe(df: pd.DataFrame, path: str | Path, *, compression: str = "snappy") -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(p, index=False, compression=compression)
        return str(p)
    except Exception:
        csv_path = p.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return str(csv_path)


def write_json_safe(payload: Mapping[str, Any], path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(json_safe(payload), sort_keys=True, indent=2, default=str),
        encoding="utf-8",
    )
    return str(p)


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(payload), sort_keys=True, default=str))
        f.write("\n")
    return str(p)


def ensure_required_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def bh_fdr(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.maximum(np.arange(1, n + 1), 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def spearman_corr_safe(x: pd.Series, y: pd.Series, min_obs: int = 10) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < min_obs:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))


def compute_daily_ic(
    panel: pd.DataFrame,
    *,
    score_col: str,
    label_col: str,
    date_col: str = "date",
    min_obs: int = 10,
) -> pd.DataFrame:
    ensure_required_columns(panel, [date_col, score_col, label_col], "panel")
    out = []
    for date, grp in panel.groupby(date_col):
        ic = spearman_corr_safe(grp[score_col], grp[label_col], min_obs=min_obs)
        n_valid = int(grp[[score_col, label_col]].dropna().shape[0])
        out.append({"date": pd.Timestamp(date), "ic_value": ic, "n_valid": n_valid})
    return pd.DataFrame(out).sort_values("date").reset_index(drop=True)


def compute_newey_west_tstat(series: Sequence[float], lags: int = 5) -> tuple[float, float]:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float("nan"), float("nan")
    mu = x.mean()
    e = x - mu
    gamma0 = float(np.dot(e, e) / n)
    var_hac = gamma0
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        cov = float(np.dot(e[lag:], e[:-lag]) / n)
        var_hac += 2.0 * w * cov
    var_mean = max(var_hac / n, 1e-15)
    tstat = mu / math.sqrt(var_mean)
    # Large-sample normal approximation
    pval = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(tstat) / math.sqrt(2.0)))))
    return float(tstat), pval


def block_bootstrap_mean(
    values: Sequence[float],
    *,
    n_bootstrap: int = 500,
    block_size: int = 20,
    seed: int = 42,
) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.repeat(x[0], n_bootstrap)
    rng = np.random.RandomState(seed)
    bs = []
    starts = np.arange(n)
    for _ in range(n_bootstrap):
        sample = []
        while len(sample) < n:
            s = int(rng.choice(starts))
            e = min(s + block_size, n)
            sample.extend(x[s:e].tolist())
        arr = np.asarray(sample[:n], dtype=float)
        bs.append(float(np.nanmean(arr)))
    return np.asarray(bs, dtype=float)


def rolling_zscore_shifted(series: pd.Series, lookback: int, eps: float = 1e-8) -> pd.Series:
    s = pd.Series(series, copy=False)
    mu = s.rolling(lookback, min_periods=max(5, lookback // 4)).mean().shift(1)
    sd = s.rolling(lookback, min_periods=max(5, lookback // 4)).std(ddof=0).shift(1)
    return (s - mu) / (sd + eps)
