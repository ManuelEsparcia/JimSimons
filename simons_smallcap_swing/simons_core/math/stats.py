"""
simons_core/math/stats.py — Descriptive statistics + financial metrics.

Descriptive:  nanmean, nanvar, nanstd, sem
Rolling:      rolling_mean, rolling_std, rolling_zscore
Correlation:  pearson_corr, spearman_corr (average ranks), spearman_ic (sign preserved)
Transform:    winsorize
Financial:    sharpe_ratio (√P), sortino_ratio, calmar_ratio, cagr
Drawdown:     drawdown_series (DD≤0), max_drawdown, drawdown_details

Conventions:  NaN-omit default · ddof configurable · Sharpe ×√P not ×P · DD≤0
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

DEFAULT_DDOF = 1
DEFAULT_EPS = 1e-12
DEFAULT_PPY = 252  # periods_per_year


# ── helpers ──────────────────────────────────────────────────────────────────

def _finite_1d(x):
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]

def _pairwise_finite(x, y):
    x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def _average_ranks(x):
    """Proper average ranks for Spearman (handles ties)."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


# ── descriptive (spec §5-6) ─────────────────────────────────────────────────

def nanmean(x):
    xv = _finite_1d(x)
    return float(np.mean(xv)) if xv.size else np.nan

def nanvar(x, ddof=DEFAULT_DDOF):
    xv = _finite_1d(x)
    return float(np.var(xv, ddof=ddof)) if xv.size > ddof else np.nan

def nanstd(x, ddof=DEFAULT_DDOF):
    xv = _finite_1d(x)
    return float(np.std(xv, ddof=ddof)) if xv.size > ddof else np.nan

def sem(x, ddof=DEFAULT_DDOF):
    """Standard error of the mean."""
    xv = _finite_1d(x)
    return float(np.std(xv, ddof=ddof) / np.sqrt(xv.size)) if xv.size > ddof else np.nan


# ── rolling (spec §7) ───────────────────────────────────────────────────────

def rolling_mean(x: pd.Series, window: int, min_periods=None) -> pd.Series:
    if window <= 0: raise ValueError("window must be positive")
    return x.rolling(window=window, min_periods=min_periods or window).mean()

def rolling_std(x: pd.Series, window: int, ddof=DEFAULT_DDOF, min_periods=None) -> pd.Series:
    if window <= 0: raise ValueError("window must be positive")
    return x.rolling(window=window, min_periods=min_periods or window).std(ddof=ddof)

def rolling_zscore(x: pd.Series, window: int, ddof=DEFAULT_DDOF, min_periods=None, eps=DEFAULT_EPS) -> pd.Series:
    """(x − rolling_mean) / rolling_std.  Constant window → 0."""
    mu = rolling_mean(x, window, min_periods)
    sd = rolling_std(x, window, ddof, min_periods)
    return ((x - mu) / sd).where(sd.abs() > eps, 0.0)


# ── correlations (spec §8-9) ────────────────────────────────────────────────

def pearson_corr(x, y):
    xv, yv = _pairwise_finite(x, y)
    if xv.size < 2 or np.std(xv) == 0 or np.std(yv) == 0:
        return np.nan
    return float(np.corrcoef(xv, yv)[0, 1])

def spearman_corr(x, y):
    """Spearman ρ with proper average ranks (handles ties)."""
    xv, yv = _pairwise_finite(x, y)
    if xv.size < 2:
        return np.nan
    rx, ry = _average_ranks(xv), _average_ranks(yv)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

def spearman_ic(scores, fwd_returns):
    """IC = Spearman(scores, fwd_returns).  Sign preserved."""
    return spearman_corr(scores, fwd_returns)


# ── winsorize (spec §10) ────────────────────────────────────────────────────

def winsorize(x, lower_q=0.01, upper_q=0.99):
    if not (0.0 <= lower_q < upper_q <= 1.0):
        raise ValueError(f"Require 0 ≤ lower_q < upper_q ≤ 1")
    x = np.asarray(x, dtype=float).ravel()
    out = x.copy()
    m = np.isfinite(x)
    if not m.any(): return out
    lo, hi = float(np.quantile(x[m], lower_q)), float(np.quantile(x[m], upper_q))
    out[m] = np.clip(x[m], lo, hi)
    return out


# ── financial metrics (spec §11-13) ─────────────────────────────────────────

def sharpe_ratio(returns, rf=0.0, periods_per_year=DEFAULT_PPY, ddof=DEFAULT_DDOF, eps=DEFAULT_EPS):
    """SR_ann = mean(r−rf)/std(r−rf) × √P."""
    r = _finite_1d(returns)
    if r.size <= ddof: return np.nan
    excess = r - rf
    sd = float(np.std(excess, ddof=ddof))
    return np.nan if sd <= eps else float(np.mean(excess)) / sd * np.sqrt(periods_per_year)

def sortino_ratio(returns, target=0.0, periods_per_year=DEFAULT_PPY, eps=DEFAULT_EPS):
    """Sortino_ann = mean(r−target)/downside_dev × √P."""
    r = _finite_1d(returns)
    if r.size == 0: return np.nan
    diff = r - target
    dd_dev = float(np.sqrt(np.mean(np.minimum(diff, 0.0) ** 2)))
    return np.nan if dd_dev <= eps else float(np.mean(diff)) / dd_dev * np.sqrt(periods_per_year)

def drawdown_series(returns):
    """DD_t = W_t / M_t − 1 ≤ 0."""
    r = np.asarray(returns, dtype=float).ravel()
    out = np.full_like(r, np.nan, dtype=float)
    m = np.isfinite(r)
    if not m.any(): return out
    wealth = np.cumprod(1.0 + r[m])
    peaks = np.maximum.accumulate(wealth)
    out[m] = wealth / peaks - 1.0
    return out

def max_drawdown(returns):
    dd = drawdown_series(returns)
    ddv = dd[np.isfinite(dd)]
    return float(np.min(ddv)) if ddv.size else np.nan

@dataclass
class DrawdownDetails:
    max_dd: float
    peak_idx: int
    trough_idx: int
    duration: int
    dd_series: np.ndarray

def drawdown_details(returns):
    dd = drawdown_series(returns)
    ddv = dd[np.isfinite(dd)]
    if ddv.size == 0:
        return DrawdownDetails(np.nan, -1, -1, 0, dd)
    trough = int(np.argmin(ddv))
    r_clean = np.asarray(returns, dtype=float).ravel()
    wealth = np.cumprod(1.0 + r_clean[np.isfinite(r_clean)])
    peak = int(np.argmax(wealth[:trough + 1])) if trough > 0 else 0
    return DrawdownDetails(float(ddv[trough]), peak, trough, trough - peak, dd)

def cagr(returns, periods_per_year=DEFAULT_PPY):
    r = _finite_1d(returns)
    if r.size == 0: return np.nan
    w = float(np.prod(1.0 + r))
    if w <= 0: return np.nan
    yrs = r.size / periods_per_year
    return float(w ** (1.0 / yrs) - 1.0) if yrs > 0 else np.nan

def calmar_ratio(returns, periods_per_year=DEFAULT_PPY, eps=DEFAULT_EPS):
    """annual_mean / |MDD|.  Note: uses mean×P, not CAGR."""
    r = _finite_1d(returns)
    if r.size == 0: return np.nan
    mdd = max_drawdown(r)
    if not np.isfinite(mdd) or abs(mdd) <= eps: return np.nan
    return float(np.mean(r)) * periods_per_year / abs(mdd)
