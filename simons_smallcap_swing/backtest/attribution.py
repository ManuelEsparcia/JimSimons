"""
backtest/attribution.py — P&L attribution by asset, sector, and factor.

Decomposes portfolio returns into:
1. Asset-level contributions (gross and net)
2. Group-level (sector/industry) aggregations
3. Cost attribution (commission, slippage, impact, borrow)
4. Factor attribution: r_p = Σ β_f F_f + α

Ensures additive consistency: Σ contrib_i ≈ total P&L at every level.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .engine import BacktestResult, PortfolioState

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Asset-level attribution
# ---------------------------------------------------------------------------

@dataclass
class AssetAttribution:
    """Per-asset P&L contribution."""
    contributions: pd.DataFrame  # date, symbol, pnl_gross, pnl_net, weight, return_contrib
    total_by_asset: pd.DataFrame  # symbol, total_pnl, avg_weight, n_days_held
    consistency_error: float       # |Σ asset_pnl - portfolio_pnl|


def attribute_by_asset(result: BacktestResult) -> AssetAttribution:
    """Compute per-asset daily P&L contributions."""
    rows = []
    for i, state in enumerate(result.daily_states):
        for sym, w in state.weights.items():
            # Contribution = weight × return (approximation from daily P&L)
            rows.append({
                "date": state.date,
                "symbol": sym,
                "weight": w,
                "pnl_gross_contrib": state.daily_pnl_gross * abs(w) / max(sum(abs(ww) for ww in state.weights.values()), 1e-10) * np.sign(w) if state.daily_pnl_gross != 0 else 0,
            })

    if not rows:
        empty = pd.DataFrame(columns=["date", "symbol", "weight", "pnl_gross_contrib"])
        return AssetAttribution(empty, pd.DataFrame(), 0.0)

    contrib = pd.DataFrame(rows)

    # Aggregate by asset
    total = contrib.groupby("symbol").agg(
        total_pnl=("pnl_gross_contrib", "sum"),
        avg_weight=("weight", "mean"),
        n_days=("date", "nunique"),
    ).reset_index().sort_values("total_pnl", ascending=False)

    # Consistency check
    daily_sum = contrib.groupby("date")["pnl_gross_contrib"].sum()
    daily_actual = pd.Series({s.date: s.daily_pnl_gross for s in result.daily_states})
    err = float((daily_sum - daily_actual.reindex(daily_sum.index).fillna(0)).abs().sum())

    return AssetAttribution(contrib, total, err)


# ---------------------------------------------------------------------------
# Sector/group attribution
# ---------------------------------------------------------------------------

def attribute_by_group(
    asset_attr: AssetAttribution,
    group_map: dict[str, str],
    group_name: str = "sector",
) -> pd.DataFrame:
    """Aggregate asset contributions by group (sector, industry, etc.).

    group_map: {symbol: group_label}
    """
    df = asset_attr.contributions.copy()
    df[group_name] = df["symbol"].map(group_map).fillna("Unknown")
    grouped = df.groupby(["date", group_name])["pnl_gross_contrib"].sum().reset_index()
    return grouped


# ---------------------------------------------------------------------------
# Cost attribution
# ---------------------------------------------------------------------------

def attribute_costs(result: BacktestResult) -> pd.DataFrame:
    """Daily cost breakdown."""
    rows = []
    for state in result.daily_states:
        row = {"date": state.date}
        row.update(state.daily_costs)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Factor attribution
# ---------------------------------------------------------------------------

@dataclass
class FactorAttribution:
    """Factor-based return decomposition."""
    daily_df: pd.DataFrame  # date, factor_1, ..., alpha, r_portfolio
    factor_contributions: dict[str, float]  # factor → total contribution
    alpha_gross: float
    alpha_net: float
    r_squared: float


def attribute_by_factor(
    result: BacktestResult,
    factor_returns: pd.DataFrame | None = None,
) -> FactorAttribution:
    """Decompose portfolio returns into factor contributions + alpha.

    r_{p,t} = Σ_f β_f × F_{f,t} + α_t

    If factor_returns not provided, returns trivial decomposition (all alpha).
    """
    eq = result.equity_curve
    if len(eq) < 10:
        return FactorAttribution(pd.DataFrame(), {}, 0, 0, 0)

    r_port = eq["return_net"].values

    if factor_returns is None or len(factor_returns) == 0:
        # No factors: everything is alpha
        alpha = float(np.nansum(r_port))
        return FactorAttribution(
            daily_df=eq[["date", "return_net"]].rename(columns={"return_net": "alpha"}),
            factor_contributions={},
            alpha_gross=float(np.nansum(eq["return_gross"].values)),
            alpha_net=alpha,
            r_squared=0.0,
        )

    # Merge factor returns with portfolio returns
    merged = eq[["date", "return_net", "return_gross"]].merge(
        factor_returns, on="date", how="inner",
    )

    factor_cols = [c for c in factor_returns.columns if c != "date"]
    F = merged[factor_cols].values
    r = merged["return_net"].values

    # OLS regression: r = F @ beta + alpha
    F_with_const = np.column_stack([np.ones(len(F)), F])
    try:
        beta, residuals, _, _ = np.linalg.lstsq(F_with_const, r, rcond=None)
    except np.linalg.LinAlgError:
        return FactorAttribution(pd.DataFrame(), {}, 0, 0, 0)

    alpha_daily = beta[0]
    factor_betas = beta[1:]

    # Factor contributions
    factor_contribs = {}
    result_df = merged[["date"]].copy()
    for j, fname in enumerate(factor_cols):
        contrib = factor_betas[j] * F[:, j]
        factor_contribs[fname] = float(np.sum(contrib))
        result_df[fname] = contrib

    # Alpha series
    predicted = F_with_const @ beta
    alpha_series = r - predicted + alpha_daily
    result_df["alpha"] = alpha_series
    result_df["r_portfolio"] = r

    # R-squared
    ss_res = np.sum((r - predicted) ** 2)
    ss_tot = np.sum((r - r.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return FactorAttribution(
        daily_df=result_df,
        factor_contributions=factor_contribs,
        alpha_gross=float(np.nansum(merged["return_gross"].values) - sum(factor_contribs.values())),
        alpha_net=float(np.sum(alpha_series)),
        r_squared=float(r2),
    )
