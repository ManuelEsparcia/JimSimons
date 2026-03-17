from __future__ import annotations

"""
simons_core.math.stats
======================

Institutional descriptive-statistics and core financial-metrics layer for the
research / validation / backtest stack.

Design goals
------------
- Explicit definitions and conventions for every metric.
- Consistent NaN handling (omit, never silent zero-imputation).
- Numerically stable behaviour in degenerate regimes.
- Clear separation between descriptive statistics and inference.
- Coherent financial conventions for annualisation and drawdowns.

Non-goals
---------
This module does *not* implement advanced inference, bootstrap confidence
intervals, multiple-testing correction, deflated Sharpe, Bayesian metrics, or
serial-correlation-adjusted performance metrics in the core path.
"""

from dataclasses import dataclass
import argparse
import json
import math
from typing import Any, Final, Literal, Sequence, TypeAlias

import numpy as np
import pandas as pd


__all__ = [
    "DEFAULT_DDOF",
    "DEFAULT_EPS",
    "DEFAULT_PERIODS_PER_YEAR",
    "StatsError",
    "InvalidInputError",
    "InsufficientDataError",
    "DrawdownDetails",
    "nanmean1d",
    "nanvar1d",
    "nanstd1d",
    "sem1d",
    "rolling_mean",
    "rolling_std",
    "rolling_zscore",
    "pearson_corr",
    "spearman_corr",
    "spearman_ic",
    "winsorize",
    "winsorize_series",
    "sharpe_ratio",
    "sortino_ratio",
    "drawdown_series",
    "max_drawdown",
    "drawdown_details",
    "cagr",
    "calmar_ratio",
]


ArrayLike: TypeAlias = np.ndarray | Sequence[float] | pd.Series
BenchmarkLike: TypeAlias = float | np.ndarray | Sequence[float] | pd.Series
CalmarMethod: TypeAlias = Literal["annual_mean", "cagr"]

DEFAULT_DDOF: Final[int] = 1
DEFAULT_EPS: Final[float] = 1e-12
DEFAULT_PERIODS_PER_YEAR: Final[int] = 252


class StatsError(RuntimeError):
    """Base class for statistics-layer failures."""


class InvalidInputError(StatsError, ValueError):
    """Raised when an input violates the function contract."""


class InsufficientDataError(StatsError, ValueError):
    """Raised when a requested statistic cannot be estimated."""


@dataclass(frozen=True)
class DrawdownDetails:
    """
    Structured drawdown event diagnostics.

    Notes
    -----
    - Indices are integer positions into the finite wealth path induced by the
      finite entries of the input returns series.
    - ``recovery_index`` is ``None`` if the previous peak is not recovered
      within the observed horizon.
    - ``duration`` is the number of finite periods from peak to trough.
    - ``recovery_duration`` is the number of finite periods from trough to
      recovery.
    """

    max_drawdown: float
    peak_index: int | None
    trough_index: int | None
    recovery_index: int | None
    duration: int | None
    recovery_duration: int | None

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "max_drawdown": float(self.max_drawdown),
            "peak_index": self.peak_index,
            "trough_index": self.trough_index,
            "recovery_index": self.recovery_index,
            "duration": self.duration,
            "recovery_duration": self.recovery_duration,
        }


def _as_1d_float_array(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    else:
        arr = arr.reshape(-1)
    return arr


def _validate_ddof(ddof: int) -> int:
    if not isinstance(ddof, int):
        raise InvalidInputError(f"ddof must be an int, got {type(ddof).__name__}.")
    if ddof < 0:
        raise InvalidInputError(f"ddof must be >= 0, got {ddof}.")
    return ddof


def _validate_window(window: int, *, min_periods: int | None = None) -> tuple[int, int]:
    if not isinstance(window, int) or window <= 0:
        raise InvalidInputError(f"window must be a positive integer, got {window!r}.")
    mp = window if min_periods is None else int(min_periods)
    if mp <= 0:
        raise InvalidInputError(f"min_periods must be positive, got {mp}.")
    if mp > window:
        raise InvalidInputError(
            f"min_periods={mp} cannot exceed window={window}."
        )
    return window, mp


def _validate_periods_per_year(periods_per_year: int) -> int:
    if not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise InvalidInputError(
            f"periods_per_year must be a positive integer, got {periods_per_year!r}."
        )
    return periods_per_year


def _finite_1d(x: ArrayLike) -> np.ndarray:
    arr = _as_1d_float_array(x)
    return arr[np.isfinite(arr)]


def _pairwise_finite(x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    xv = _as_1d_float_array(x)
    yv = _as_1d_float_array(y)
    if xv.shape != yv.shape:
        raise InvalidInputError(
            f"x and y must have the same shape, got {xv.shape} vs {yv.shape}."
        )
    mask = np.isfinite(xv) & np.isfinite(yv)
    return xv[mask], yv[mask]


def _as_series(x: pd.Series | ArrayLike, *, name: str | None = None) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.astype(float, copy=False)
    return pd.Series(_as_1d_float_array(x), name=name, dtype=float)


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Stable average ranks for ties, 1-indexed internally then returned as float."""
    xv = np.asarray(x, dtype=float).reshape(-1)
    n = xv.size
    if n == 0:
        return np.empty(0, dtype=float)

    order = np.argsort(xv, kind="mergesort")
    sorted_x = xv[order]
    ranks_sorted = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks_sorted[i : j + 1] = avg_rank
        i = j + 1

    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def _std_or_nan(x: np.ndarray, *, ddof: int = DEFAULT_DDOF) -> float:
    ddof = _validate_ddof(ddof)
    if x.size <= ddof:
        return float("nan")
    return float(np.std(x, ddof=ddof))


def _validate_simple_returns(returns: np.ndarray) -> None:
    bad = returns[np.isfinite(returns)] < -1.0
    if bool(np.any(bad)):
        raise InvalidInputError(
            "Simple returns < -1.0 are invalid for wealth / drawdown calculations."
        )


def _align_benchmark(returns: ArrayLike, benchmark: BenchmarkLike) -> tuple[np.ndarray, np.ndarray]:
    r = _as_1d_float_array(returns)
    if np.isscalar(benchmark):
        b = np.full_like(r, float(benchmark), dtype=float)
    else:
        b = _as_1d_float_array(benchmark)
        if r.shape != b.shape:
            raise InvalidInputError(
                f"Benchmark shape {b.shape} must match returns shape {r.shape}."
            )
    return _pairwise_finite(r, b)


def nanmean1d(x: ArrayLike) -> float:
    """NaN-safe mean over finite observations only."""
    xv = _finite_1d(x)
    return float(np.mean(xv)) if xv.size else float("nan")


def nanvar1d(x: ArrayLike, ddof: int = DEFAULT_DDOF) -> float:
    """NaN-safe variance over finite observations only."""
    ddof = _validate_ddof(ddof)
    xv = _finite_1d(x)
    if xv.size <= ddof:
        return float("nan")
    return float(np.var(xv, ddof=ddof))


def nanstd1d(x: ArrayLike, ddof: int = DEFAULT_DDOF) -> float:
    """NaN-safe standard deviation over finite observations only."""
    ddof = _validate_ddof(ddof)
    xv = _finite_1d(x)
    if xv.size <= ddof:
        return float("nan")
    return float(np.std(xv, ddof=ddof))


def sem1d(x: ArrayLike, ddof: int = DEFAULT_DDOF) -> float:
    """
    Standard error of the mean over finite observations only.

    Returns ``NaN`` when there is insufficient data for the requested ``ddof``.
    """
    ddof = _validate_ddof(ddof)
    xv = _finite_1d(x)
    if xv.size <= ddof:
        return float("nan")
    sd = float(np.std(xv, ddof=ddof))
    return sd / math.sqrt(float(xv.size))


def rolling_mean(
    x: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling mean preserving index and NaN semantics from pandas."""
    window, mp = _validate_window(window, min_periods=min_periods)
    xs = _as_series(x)
    return xs.rolling(window=window, min_periods=mp).mean()


def rolling_std(
    x: pd.Series,
    window: int,
    ddof: int = DEFAULT_DDOF,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling standard deviation preserving index and NaN semantics."""
    ddof = _validate_ddof(ddof)
    window, mp = _validate_window(window, min_periods=min_periods)
    xs = _as_series(x)
    return xs.rolling(window=window, min_periods=mp).std(ddof=ddof)


def rolling_zscore(
    x: pd.Series,
    window: int,
    ddof: int = DEFAULT_DDOF,
    min_periods: int | None = None,
    eps: float = DEFAULT_EPS,
) -> pd.Series:
    """
    Rolling z-score ``(x - rolling_mean) / rolling_std``.

    Conventions
    -----------
    - If insufficient observations are available, result is ``NaN``.
    - If the rolling standard deviation is finite but ``<= eps``, result is set
      to ``0.0`` in that window (constant-window convention).
    """
    if eps <= 0:
        raise InvalidInputError(f"eps must be > 0, got {eps}.")
    xs = _as_series(x)
    mu = rolling_mean(xs, window=window, min_periods=min_periods)
    sd = rolling_std(xs, window=window, ddof=ddof, min_periods=min_periods)
    z = (xs - mu) / sd
    z = z.mask(sd.abs() <= eps, 0.0)
    return z


def pearson_corr(x: ArrayLike, y: ArrayLike, *, ddof: int = DEFAULT_DDOF, eps: float = DEFAULT_EPS) -> float:
    """
    Pairwise-finite Pearson correlation.

    Returns ``NaN`` when fewer than two valid observations remain or when either
    series is effectively constant.
    """
    _validate_ddof(ddof)
    if eps <= 0:
        raise InvalidInputError(f"eps must be > 0, got {eps}.")
    xv, yv = _pairwise_finite(x, y)
    if xv.size < 2:
        return float("nan")
    sx = _std_or_nan(xv, ddof=ddof)
    sy = _std_or_nan(yv, ddof=ddof)
    if not np.isfinite(sx) or not np.isfinite(sy) or sx <= eps or sy <= eps:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def spearman_corr(x: ArrayLike, y: ArrayLike, *, ddof: int = DEFAULT_DDOF, eps: float = DEFAULT_EPS) -> float:
    """
    Pairwise-finite Spearman correlation using stable average ranks for ties.

    This intentionally avoids the incorrect shortcut of applying ``argsort``
    directly as a rank proxy.
    """
    xv, yv = _pairwise_finite(x, y)
    if xv.size < 2:
        return float("nan")
    rx = _average_ranks(xv)
    ry = _average_ranks(yv)
    return pearson_corr(rx, ry, ddof=ddof, eps=eps)


def spearman_ic(scores: ArrayLike, fwd_returns: ArrayLike, *, ddof: int = DEFAULT_DDOF, eps: float = DEFAULT_EPS) -> float:
    """Information coefficient defined as signed Spearman correlation."""
    return spearman_corr(scores, fwd_returns, ddof=ddof, eps=eps)


def winsorize(
    x: ArrayLike,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> np.ndarray:
    """
    Quantile winsorisation over finite observations only.

    NaN / Inf positions are preserved in the output and not used for quantile
    estimation.
    """
    if not (0.0 <= lower_q < upper_q <= 1.0):
        raise InvalidInputError(
            f"Require 0 <= lower_q < upper_q <= 1, got {lower_q} and {upper_q}."
        )

    arr = _as_1d_float_array(x)
    out = arr.copy()
    mask = np.isfinite(arr)
    if not bool(np.any(mask)):
        return out

    finite = arr[mask]
    lo = float(np.quantile(finite, lower_q))
    hi = float(np.quantile(finite, upper_q))
    out[mask] = np.clip(finite, lo, hi)
    return out


def winsorize_series(
    x: pd.Series,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.Series:
    """Series-preserving wrapper around :func:`winsorize`."""
    xs = _as_series(x)
    return pd.Series(
        winsorize(xs.to_numpy(dtype=float), lower_q=lower_q, upper_q=upper_q),
        index=xs.index,
        name=xs.name,
        dtype=float,
    )


def sharpe_ratio(
    returns: ArrayLike,
    rf: BenchmarkLike = 0.0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    ddof: int = DEFAULT_DDOF,
    eps: float = DEFAULT_EPS,
) -> float:
    """
    Annualised Sharpe ratio using ``sqrt(periods_per_year)``.

    Parameters
    ----------
    returns:
        Periodic simple returns.
    rf:
        Risk-free rate in the *same periodicity* as ``returns``. Can be either a
        scalar or an array-like aligned to ``returns``.
    """
    periods_per_year = _validate_periods_per_year(periods_per_year)
    ddof = _validate_ddof(ddof)
    if eps <= 0:
        raise InvalidInputError(f"eps must be > 0, got {eps}.")

    r, rfv = _align_benchmark(returns, rf)
    if r.size <= ddof:
        return float("nan")

    excess = r - rfv
    mu = float(np.mean(excess))
    sd = float(np.std(excess, ddof=ddof))
    if not np.isfinite(sd) or sd <= eps:
        return float("nan")
    return float(mu / sd * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: ArrayLike,
    target: BenchmarkLike = 0.0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    eps: float = DEFAULT_EPS,
) -> float:
    """
    Annualised Sortino ratio using downside deviation relative to ``target``.

    Downside deviation is computed as ``sqrt(mean(min(r-target, 0)^2))`` over
    pairwise-finite observations.
    """
    periods_per_year = _validate_periods_per_year(periods_per_year)
    if eps <= 0:
        raise InvalidInputError(f"eps must be > 0, got {eps}.")

    r, tgt = _align_benchmark(returns, target)
    if r.size == 0:
        return float("nan")

    diff = r - tgt
    downside = np.minimum(diff, 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside))))
    if not np.isfinite(downside_dev) or downside_dev <= eps:
        return float("nan")
    return float(np.mean(diff) / downside_dev * math.sqrt(periods_per_year))


def drawdown_series(returns: ArrayLike) -> np.ndarray:
    """
    Drawdown series for simple returns.

    Convention
    ----------
    ``DD_t = W_t / max_{u<=t}(W_u) - 1``, hence ``DD_t <= 0``.

    Non-finite entries in the input remain ``NaN`` in the output and are not
    used in the wealth recursion.
    """
    r = _as_1d_float_array(returns)
    out = np.full(r.shape, np.nan, dtype=float)

    mask = np.isfinite(r)
    if not bool(np.any(mask)):
        return out

    finite = r[mask]
    _validate_simple_returns(finite)

    wealth = np.cumprod(1.0 + finite)
    peaks = np.maximum.accumulate(wealth)
    dd = wealth / peaks - 1.0
    out[mask] = dd
    return out


def max_drawdown(returns: ArrayLike) -> float:
    """Minimum of the drawdown series, or ``NaN`` if undefined."""
    dd = drawdown_series(returns)
    ddv = dd[np.isfinite(dd)]
    return float(np.min(ddv)) if ddv.size else float("nan")


def drawdown_details(returns: ArrayLike) -> DrawdownDetails:
    """
    Extended diagnostics for the maximum drawdown event.

    This is intentionally an *extension* over the base ``drawdown_series`` /
    ``max_drawdown`` contract and does not alter their semantics.
    """
    r = _as_1d_float_array(returns)
    mask = np.isfinite(r)
    if not bool(np.any(mask)):
        return DrawdownDetails(
            max_drawdown=float("nan"),
            peak_index=None,
            trough_index=None,
            recovery_index=None,
            duration=None,
            recovery_duration=None,
        )

    finite = r[mask]
    _validate_simple_returns(finite)
    wealth = np.cumprod(1.0 + finite)
    peaks = np.maximum.accumulate(wealth)
    dd = wealth / peaks - 1.0
    if dd.size == 0:
        return DrawdownDetails(float("nan"), None, None, None, None, None)

    trough_idx = int(np.argmin(dd))
    mdd = float(dd[trough_idx])
    if not np.isfinite(mdd):
        return DrawdownDetails(float("nan"), None, None, None, None, None)

    peak_value = peaks[trough_idx]
    peak_candidates = np.flatnonzero(wealth[: trough_idx + 1] == peak_value)
    peak_idx = int(peak_candidates[0]) if peak_candidates.size else 0

    recovery_idx: int | None = None
    if trough_idx + 1 < wealth.size:
        recovery_candidates = np.flatnonzero(wealth[trough_idx + 1 :] >= peak_value)
        if recovery_candidates.size:
            recovery_idx = int(trough_idx + 1 + recovery_candidates[0])

    duration = trough_idx - peak_idx if trough_idx >= peak_idx else None
    recovery_duration = (
        recovery_idx - trough_idx if recovery_idx is not None else None
    )

    return DrawdownDetails(
        max_drawdown=mdd,
        peak_index=peak_idx,
        trough_index=trough_idx,
        recovery_index=recovery_idx,
        duration=duration,
        recovery_duration=recovery_duration,
    )


def cagr(
    returns: ArrayLike,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """
    Compound annual growth rate from periodic simple returns.

    Returns ``NaN`` if no finite observations exist. Allows a terminal wealth of
    zero (which yields ``-1.0`` CAGR). Returns less than zero wealth are invalid
    under the simple-return wealth convention and fail closed.
    """
    periods_per_year = _validate_periods_per_year(periods_per_year)
    r = _finite_1d(returns)
    if r.size == 0:
        return float("nan")
    _validate_simple_returns(r)

    wealth_terminal = float(np.prod(1.0 + r))
    if wealth_terminal < 0.0:
        raise InvalidInputError("Terminal wealth became negative; simple-return CAGR undefined.")
    if wealth_terminal == 0.0:
        return -1.0

    years = float(r.size) / float(periods_per_year)
    if years <= 0.0:
        return float("nan")
    return float(wealth_terminal ** (1.0 / years) - 1.0)


def calmar_ratio(
    returns: ArrayLike,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    eps: float = DEFAULT_EPS,
    *,
    method: CalmarMethod = "annual_mean",
) -> float:
    """
    Calmar ratio with an explicit numerator convention.

    Parameters
    ----------
    method:
        - ``'annual_mean'`` (institutional default): ``mean(r) * P / abs(MDD)``
        - ``'cagr'``: ``CAGR(r) / abs(MDD)``

    The default intentionally matches the base contract described in the
    repository specification.
    """
    periods_per_year = _validate_periods_per_year(periods_per_year)
    if eps <= 0:
        raise InvalidInputError(f"eps must be > 0, got {eps}.")
    if method not in {"annual_mean", "cagr"}:
        raise InvalidInputError(
            f"Unsupported Calmar method={method!r}. Expected 'annual_mean' or 'cagr'."
        )

    r = _finite_1d(returns)
    if r.size == 0:
        return float("nan")

    mdd = max_drawdown(r)
    if not np.isfinite(mdd) or abs(mdd) <= eps:
        return float("nan")

    if method == "annual_mean":
        numerator = float(np.mean(r)) * float(periods_per_year)
    else:
        numerator = cagr(r, periods_per_year=periods_per_year)

    return float(numerator / abs(mdd))


def _self_test() -> dict[str, Any]:
    """Deterministic smoke tests for core contracts."""
    # Descriptives with NaN omission.
    x = np.array([1.0, 2.0, np.nan, 4.0])
    assert math.isclose(nanmean1d(x), 7.0 / 3.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(nanvar1d(x, ddof=1), 7.0 / 3.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(nanstd1d(x, ddof=1), math.sqrt(7.0 / 3.0), rel_tol=0.0, abs_tol=1e-12)

    # Rolling z-score on constant window => 0.0 after enough observations.
    s = pd.Series([5.0, 5.0, 5.0, 5.0])
    rz = rolling_zscore(s, window=2)
    assert np.isnan(rz.iloc[0])
    assert math.isclose(float(rz.iloc[1]), 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(rz.iloc[3]), 0.0, rel_tol=0.0, abs_tol=1e-12)

    # Pearson / Spearman with ties.
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 4.0, 6.0, 8.0])
    assert math.isclose(pearson_corr(a, b), 1.0, rel_tol=0.0, abs_tol=1e-12)
    tied_x = np.array([1.0, 1.0, 2.0, 3.0])
    tied_y = np.array([10.0, 10.0, 20.0, 30.0])
    assert math.isclose(spearman_corr(tied_x, tied_y), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(spearman_ic(tied_x, tied_y), 1.0, rel_tol=0.0, abs_tol=1e-12)

    # Winsorization preserves NaNs and shape.
    w = winsorize(np.array([1.0, 2.0, 100.0, np.nan]), lower_q=0.0, upper_q=0.5)
    assert w.shape == (4,)
    assert np.isnan(w[-1])
    assert w[2] <= 2.0

    # Financial metrics.
    r = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    assert np.isfinite(sharpe_ratio(r))
    assert np.isfinite(sortino_ratio(r))

    # Drawdown path with known maximum.
    r2 = np.array([0.10, -0.20, 0.05, -0.10, 0.20])
    dd = drawdown_series(r2)
    assert dd.shape == r2.shape
    assert np.all(dd[np.isfinite(dd)] <= 1e-12)
    mdd = max_drawdown(r2)
    assert math.isclose(mdd, -0.244, rel_tol=0.0, abs_tol=1e-12)
    details = drawdown_details(r2)
    assert details.peak_index == 0
    assert details.trough_index == 3

    # Calmar explicit method contract.
    assert np.isfinite(calmar_ratio(r2, method="annual_mean"))
    assert np.isfinite(calmar_ratio(r2, periods_per_year=5, method="cagr"))

    return {
        "status": "ok",
        "tests": 15,
        "max_drawdown_example": mdd,
        "drawdown_details_example": details.as_dict(),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Self-test utilities for simons_core.math.stats")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic smoke tests and print JSON summary.",
    )
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(_self_test(), indent=2, sort_keys=True))
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()
