"""
backtest/run_walkforward.py — Walk-forward backtest orchestrator.

Executes the full walk-forward loop:
    For each window k:
        1. Build train/test split (rolling or expanding) with purge + embargo
        2. Train model on train window
        3. Generate OOS predictions for test window
        4. Run engine on OOS predictions
        5. Concatenate OOS segments into single non-overlapping equity curve

Supports: rolling windows, expanding windows, custom window schedule.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from .engine import BacktestResult, EngineConfig, run_backtest

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for walk-forward backtest."""
    # Window scheme
    scheme: str = "rolling"           # "rolling" | "expanding"
    train_days: int = 252             # ~1 year train
    test_days: int = 63               # ~1 quarter test (OOS)
    step_days: int = 63               # slide by 1 quarter

    # Purge and embargo
    purge_days: int = 10              # label horizon
    embargo_days: int = 21            # ~1 month safety gap

    # Minimum requirements
    min_train_after_purge: int = 126  # ~6 months minimum train
    min_test_obs: int = 500           # minimum OOS observations

    # Engine
    engine: EngineConfig = field(default_factory=EngineConfig)


@dataclass
class WalkForwardWindow:
    """Definition of a single walk-forward window."""
    window_id: int
    train_start: Any
    train_end: Any
    test_start: Any
    test_end: Any
    purge_start: Any
    purge_end: Any
    n_train_dates: int
    n_test_dates: int
    status: str = "OK"  # "OK" | "SKIPPED" | "INVALID"
    reason: str = ""


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------

def build_windows(
    all_dates: np.ndarray,
    config: WalkForwardConfig,
) -> list[WalkForwardWindow]:
    """Build walk-forward windows with purge and embargo."""
    unique_dates = np.sort(np.unique(all_dates))
    n = len(unique_dates)
    windows = []

    test_start_idx = config.train_days
    win_id = 0

    while test_start_idx + config.test_days <= n:
        test_end_idx = test_start_idx + config.test_days

        # Purge: remove labels whose horizon touches test
        purge_start_idx = max(0, test_start_idx - config.purge_days)

        # Embargo: gap between train end and test start
        train_end_idx = max(0, purge_start_idx - config.embargo_days)

        # Train start
        if config.scheme == "rolling":
            train_start_idx = max(0, train_end_idx - config.train_days)
        else:  # expanding
            train_start_idx = 0

        # Validate
        n_train = train_end_idx - train_start_idx
        n_test = test_end_idx - test_start_idx
        status = "OK"
        reason = ""

        if n_train < config.min_train_after_purge:
            status = "SKIPPED"
            reason = f"train too small after purge: {n_train} < {config.min_train_after_purge}"

        windows.append(WalkForwardWindow(
            window_id=win_id,
            train_start=unique_dates[train_start_idx],
            train_end=unique_dates[min(train_end_idx, n-1)],
            test_start=unique_dates[test_start_idx],
            test_end=unique_dates[min(test_end_idx-1, n-1)],
            purge_start=unique_dates[purge_start_idx],
            purge_end=unique_dates[min(test_start_idx-1, n-1)],
            n_train_dates=n_train,
            n_test_dates=n_test,
            status=status,
            reason=reason,
        ))

        test_start_idx += config.step_days
        win_id += 1

    LOGGER.info(
        "Walk-forward: %d windows (%d OK, %d skipped)",
        len(windows),
        sum(1 for w in windows if w.status == "OK"),
        sum(1 for w in windows if w.status != "OK"),
    )
    return windows


# ---------------------------------------------------------------------------
# Walk-forward execution
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    """Result of a complete walk-forward backtest."""
    windows: list[WalkForwardWindow]
    oos_equity_curve: pd.DataFrame
    oos_summary: dict[str, Any]
    per_window_summaries: list[dict[str, Any]]
    config: WalkForwardConfig


def run_walkforward(
    scores_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    config: WalkForwardConfig | None = None,
    train_predict_fn: Callable | None = None,
    universe_df: pd.DataFrame | None = None,
) -> WalkForwardResult:
    """Run a complete walk-forward backtest.

    Parameters
    ----------
    scores_df : DataFrame with (date, symbol, score)
        Pre-computed scores. If train_predict_fn is provided, scores
        are generated per window instead.
    prices_df : DataFrame with (date, symbol, open, close, volume)
    config : WalkForwardConfig
    train_predict_fn : Callable, optional
        fn(train_df, test_df) → test_scores_df. If provided, trains a
        model per window and generates OOS predictions.
    universe_df : optional

    Returns
    -------
    WalkForwardResult with concatenated OOS equity curve.
    """
    cfg = config or WalkForwardConfig()

    all_dates = np.sort(scores_df["date"].unique())
    windows = build_windows(all_dates, cfg)

    oos_segments: list[pd.DataFrame] = []
    per_window: list[dict[str, Any]] = []

    for win in windows:
        if win.status != "OK":
            per_window.append({"window_id": win.window_id, "status": win.status, "reason": win.reason})
            continue

        # Filter scores to test window
        test_mask = (scores_df["date"] >= win.test_start) & (scores_df["date"] <= win.test_end)
        test_scores = scores_df[test_mask].copy()

        if train_predict_fn is not None:
            train_mask = (scores_df["date"] >= win.train_start) & (scores_df["date"] <= win.train_end)
            train_data = scores_df[train_mask]
            test_data = scores_df[test_mask]
            try:
                test_scores = train_predict_fn(train_data, test_data)
            except Exception as e:
                LOGGER.error("Window %d train_predict failed: %s", win.window_id, e)
                per_window.append({"window_id": win.window_id, "status": "ERROR", "reason": str(e)})
                continue

        if len(test_scores) < cfg.min_test_obs:
            per_window.append({"window_id": win.window_id, "status": "SKIPPED",
                              "reason": f"too few OOS obs: {len(test_scores)}"})
            continue

        # Filter prices to test window
        price_mask = (prices_df["date"] >= win.test_start) & (prices_df["date"] <= win.test_end)
        test_prices = prices_df[price_mask]

        # Run engine on this window
        try:
            bt = run_backtest(test_scores, test_prices, config=cfg.engine, universe_df=universe_df)
            oos_segments.append(bt.equity_curve)
            per_window.append({
                "window_id": win.window_id,
                "status": "OK",
                "test_start": str(win.test_start),
                "test_end": str(win.test_end),
                "n_sessions": len(bt.equity_curve),
                "sharpe_net": bt.summary.get("sharpe_net", 0),
                "total_return_pct": bt.summary.get("total_return_pct", 0),
            })
        except Exception as e:
            LOGGER.error("Window %d engine failed: %s", win.window_id, e)
            per_window.append({"window_id": win.window_id, "status": "ERROR", "reason": str(e)})

    # Concatenate OOS segments
    if oos_segments:
        oos_eq = pd.concat(oos_segments, ignore_index=True).sort_values("date").reset_index(drop=True)
        # Recompute cumulative metrics on concatenated OOS
        oos_eq["cum_return_net_oos"] = (1 + oos_eq["return_net"].fillna(0)).cumprod() - 1
    else:
        oos_eq = pd.DataFrame()

    # OOS summary
    oos_summary: dict[str, Any] = {"n_windows_ok": sum(1 for w in per_window if w.get("status") == "OK")}
    if len(oos_eq) > 0:
        ret = oos_eq["return_net"].dropna().values
        P = cfg.engine.sessions_per_year
        if len(ret) > 1 and np.std(ret) > 0:
            oos_summary["oos_sharpe"] = round(float(np.mean(ret) * np.sqrt(P) / np.std(ret)), 3)
            oos_summary["oos_total_return_pct"] = round(float(np.prod(1 + ret) - 1) * 100, 2)
            oos_summary["oos_n_sessions"] = len(ret)

    LOGGER.info("Walk-forward complete: %d windows, OOS Sharpe=%.3f",
                len(windows), oos_summary.get("oos_sharpe", 0))

    return WalkForwardResult(
        windows=windows,
        oos_equity_curve=oos_eq,
        oos_summary=oos_summary,
        per_window_summaries=per_window,
        config=cfg,
    )
