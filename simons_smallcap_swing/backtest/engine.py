"""
backtest/engine.py — Event-driven portfolio simulation engine.

Institutional-grade: session-by-session state machine with explicit
temporal conventions, transaction cost modelling, capacity constraints,
and full P&L decomposition.

Session event sequence (invariant):
1. Load calendar + universe PIT
2. Load prices at decision_cut
3. Construct target weights from scores
4. Generate orders (target - current)
5. Apply constraints (turnover, ADV participation, position limits)
6. Execute fills with cost model
7. Update positions and cash
8. Mark-to-market portfolio
9. Compute P&L (gross and net)
10. Record daily state

P&L decomposition:
    P&L_gross = Σ_i q_{i,t-1} × (P_{i,t} - P_{i,t-1})
    P&L_net   = P&L_gross - costs_total
    costs     = commission + slippage + impact + borrow + financing

Produces: daily NAV, equity curve, trade log, cost breakdown, position history.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    BacktestError, ConfigError, DataContractError, ReconciliationError,
    EngineConfig, CostConfig, CostModel, ExecutionModel,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------

@dataclass
class PortfolioState:
    """Complete portfolio state at end of session t."""
    date: Any
    positions: dict[str, float]     # symbol -> shares
    weights: dict[str, float]       # symbol -> weight (pct of NAV)
    cash: float
    nav: float
    gross_exposure: float
    net_exposure: float
    n_longs: int
    n_shorts: int
    daily_pnl_gross: float
    daily_pnl_net: float
    daily_costs: dict[str, float]
    daily_turnover: float
    daily_trades: int


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    date: Any
    symbol: str
    side: str           # "BUY" | "SELL" | "SHORT" | "COVER"
    shares: float
    price: float
    notional: float
    commission: float
    slippage: float
    impact: float
    total_cost: float


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------

def compute_trade_costs(
    notional: float,
    adv: float,
    config: CostConfig,
) -> dict[str, float]:
    """Compute all costs for a single trade."""
    abs_notional = abs(notional)

    if config.model == CostModel.ZERO.value:
        return {"commission": 0, "slippage": 0, "impact": 0, "total": 0}

    commission = max(abs_notional * config.fixed_commission_bps / 1e4, config.min_commission)
    slippage = abs_notional * config.slippage_bps / 1e4

    if config.model == CostModel.LINEAR_IMPACT.value and adv > 0:
        participation = abs_notional / adv
        impact = config.impact_coeff * np.sqrt(participation) * abs_notional / 1e4
    else:
        impact = 0.0

    return {
        "commission": commission,
        "slippage": slippage,
        "impact": impact,
        "total": commission + slippage + impact,
    }


def compute_holding_costs(
    positions: dict[str, float],
    prices: dict[str, float],
    config: CostConfig,
    sessions_per_year: int = 252,
) -> dict[str, float]:
    """Compute daily borrow and financing costs."""
    borrow_daily_rate = config.borrow_annual_bps / 1e4 / sessions_per_year
    finance_daily_rate = config.financing_annual_bps / 1e4 / sessions_per_year

    borrow_cost = 0.0
    for sym, shares in positions.items():
        if shares < 0:
            px = prices.get(sym, 0)
            borrow_cost += abs(shares * px) * borrow_daily_rate

    # Financing cost on net cash (if negative = margin)
    financing_cost = 0.0  # simplified: only if cash < 0

    return {
        "borrow": borrow_cost,
        "financing": financing_cost,
        "total_holding": borrow_cost + financing_cost,
    }


# ---------------------------------------------------------------------------
# Score → target weights
# ---------------------------------------------------------------------------

def scores_to_weights(
    scores: pd.Series,
    config: EngineConfig,
) -> pd.Series:
    """Convert raw scores to target portfolio weights.

    Uses a long-short quantile approach:
    - Top quantile: equal-weight long
    - Bottom quantile: equal-weight short
    - Middle: zero weight

    Scaled to respect gross/net exposure limits.
    """
    if len(scores.dropna()) < 10:
        return pd.Series(dtype=float)

    ranked = scores.rank(pct=True)
    n = len(ranked.dropna())

    # Long top 20%, short bottom 20%
    long_mask = ranked >= 0.8
    short_mask = ranked <= 0.2
    n_long = long_mask.sum()
    n_short = short_mask.sum()

    weights = pd.Series(0.0, index=scores.index)

    if n_long > 0:
        long_weight = (config.max_gross_exposure / 2) / n_long
        weights[long_mask] = min(long_weight, config.max_position_weight)

    if n_short > 0:
        short_weight = (config.max_gross_exposure / 2) / n_short
        weights[short_mask] = -min(short_weight, config.max_position_weight)

    # Enforce constraints
    gross = weights.abs().sum()
    if gross > config.max_gross_exposure:
        weights *= config.max_gross_exposure / gross

    net = weights.sum()
    if abs(net) > config.max_net_exposure:
        # Tilt to reduce net exposure
        excess = net - np.sign(net) * config.max_net_exposure
        n_positions = (weights != 0).sum()
        if n_positions > 0:
            weights -= excess / n_positions

    return weights


# ---------------------------------------------------------------------------
# Order generation with constraints
# ---------------------------------------------------------------------------

def generate_orders(
    target_weights: pd.Series,
    current_positions: dict[str, float],
    prices: dict[str, float],
    nav: float,
    adv: dict[str, float],
    config: EngineConfig,
) -> list[dict[str, Any]]:
    """Generate constrained orders from target weights."""
    orders = []

    # Target shares from weights
    all_symbols = set(target_weights.index) | set(current_positions.keys())
    total_turnover = 0.0

    for sym in all_symbols:
        target_w = target_weights.get(sym, 0.0)
        if pd.isna(target_w):
            target_w = 0.0
        px = prices.get(sym)
        if px is None or px <= 0:
            continue

        target_shares = (target_w * nav) / px
        current_shares = current_positions.get(sym, 0.0)
        delta_shares = target_shares - current_shares
        delta_notional = abs(delta_shares * px)

        if abs(delta_shares) < 0.5:  # skip dust
            continue

        # ADV participation limit
        sym_adv = adv.get(sym, 0)
        if sym_adv > 0:
            max_notional = sym_adv * config.max_adv_participation
            if delta_notional > max_notional:
                delta_shares = np.sign(delta_shares) * max_notional / px
                delta_notional = abs(delta_shares * px)

        total_turnover += delta_notional

        # Turnover limit (daily)
        if total_turnover / nav > config.max_turnover_daily:
            break

        side = "BUY" if delta_shares > 0 else "SELL"
        if current_shares <= 0 and delta_shares < 0:
            side = "SHORT"
        elif current_shares < 0 and delta_shares > 0:
            side = "COVER"

        orders.append({
            "symbol": sym,
            "side": side,
            "shares": delta_shares,
            "price": px,
            "notional": delta_shares * px,
            "adv": sym_adv,
        })

    return orders


# ---------------------------------------------------------------------------
# MAIN ENGINE
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Complete backtest output."""
    daily_states: list[PortfolioState]
    trade_log: list[TradeRecord]
    equity_curve: pd.DataFrame       # date, nav, pnl_gross, pnl_net, ...
    summary: dict[str, Any]
    config: EngineConfig


def run_backtest(
    scores_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    config: EngineConfig | None = None,
    universe_df: pd.DataFrame | None = None,
    adv_df: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run a full event-driven backtest.

    Parameters
    ----------
    scores_df : DataFrame with (date, symbol, score)
    prices_df : DataFrame with (date, symbol, open, close, volume)
    config : EngineConfig
    universe_df : DataFrame with (date, symbol, eligible) — optional filter
    adv_df : DataFrame with (date, symbol, adv) — for impact model

    Returns
    -------
    BacktestResult with daily states, trade log, equity curve, summary.
    """
    cfg = config or EngineConfig()
    LOGGER.info("Backtest engine: capital=%.0f, cost=%s, execution=%s",
                cfg.initial_capital, cfg.cost.model, cfg.execution_model)

    # Sort and validate
    scores_df = scores_df.copy()
    prices_df = prices_df.copy()
    scores_df["date"] = pd.to_datetime(scores_df["date"])
    prices_df["date"] = pd.to_datetime(prices_df["date"])

    # Get sorted unique dates from scores (trading dates)
    score_dates = sorted(scores_df["date"].unique())
    all_dates = sorted(prices_df["date"].unique())

    # Build price lookups
    price_by_date: dict[Any, dict[str, dict[str, float]]] = {}
    for _, row in prices_df.iterrows():
        d = row["date"]
        sym = row["symbol"]
        if d not in price_by_date:
            price_by_date[d] = {}
        price_by_date[d][sym] = {
            "open": float(row.get("open", row.get("close", 0))),
            "close": float(row.get("close", 0)),
            "volume": float(row.get("volume", 0)),
        }

    # ADV lookup
    adv_lookup: dict[Any, dict[str, float]] = {}
    if adv_df is not None:
        for _, row in adv_df.iterrows():
            d = row["date"]
            if d not in adv_lookup:
                adv_lookup[d] = {}
            adv_lookup[d][row["symbol"]] = float(row["adv"])

    # Universe lookup
    universe_lookup: dict[Any, set[str]] = {}
    if universe_df is not None:
        for _, row in universe_df.iterrows():
            d = row["date"]
            if d not in universe_lookup:
                universe_lookup[d] = set()
            if row.get("eligible", True):
                universe_lookup[d].add(row["symbol"])

    # --- Simulation state ---
    positions: dict[str, float] = {}
    cash = cfg.initial_capital
    nav = cfg.initial_capital
    daily_states: list[PortfolioState] = []
    trade_log: list[TradeRecord] = []
    equity_rows: list[dict[str, Any]] = []

    prev_prices: dict[str, float] = {}

    for t_idx, t in enumerate(score_dates):
        # --- Step 1-2: Load prices ---
        today_prices_raw = price_by_date.get(t, {})

        # Execution prices
        exec_prices: dict[str, float] = {}
        mark_prices: dict[str, float] = {}
        for sym, px_data in today_prices_raw.items():
            if cfg.execution_model == ExecutionModel.OPEN_NEXT.value:
                exec_prices[sym] = px_data["open"]
            elif cfg.execution_model == ExecutionModel.CLOSE_SAME.value:
                exec_prices[sym] = px_data["close"]
            else:
                exec_prices[sym] = (px_data["open"] + px_data["close"]) / 2
            mark_prices[sym] = px_data[cfg.valuation_price]

        # ADV for today
        today_adv: dict[str, float] = {}
        for sym, px_data in today_prices_raw.items():
            if t in adv_lookup and sym in adv_lookup[t]:
                today_adv[sym] = adv_lookup[t][sym]
            else:
                today_adv[sym] = px_data["volume"] * px_data["close"]  # proxy

        # --- Step 3: Scores for today ---
        scores_today = scores_df[scores_df["date"] == t].set_index("symbol")["score"]
        if universe_lookup:
            eligible = universe_lookup.get(t, set())
            if eligible:
                scores_today = scores_today[scores_today.index.isin(eligible)]

        # --- Step 4: Target weights ---
        target_w = scores_to_weights(scores_today, cfg)

        # --- Step 5-6: Orders + fills ---
        orders = generate_orders(target_w, positions, exec_prices, nav, today_adv, cfg)

        day_costs = {"commission": 0.0, "slippage": 0.0, "impact": 0.0,
                     "borrow": 0.0, "financing": 0.0, "total": 0.0}
        day_trades = 0
        day_turnover = 0.0

        for order in orders:
            sym = order["symbol"]
            px = exec_prices.get(sym, order["price"])
            if px <= 0:
                continue

            shares = order["shares"]
            notional = shares * px

            costs = compute_trade_costs(abs(notional), today_adv.get(sym, 0), cfg.cost)

            # Update positions
            positions[sym] = positions.get(sym, 0) + shares
            cash -= notional + costs["total"]

            day_costs["commission"] += costs["commission"]
            day_costs["slippage"] += costs["slippage"]
            day_costs["impact"] += costs["impact"]
            day_costs["total"] += costs["total"]
            day_trades += 1
            day_turnover += abs(notional)

            trade_log.append(TradeRecord(
                date=t, symbol=sym, side=order["side"],
                shares=shares, price=px, notional=notional,
                commission=costs["commission"], slippage=costs["slippage"],
                impact=costs["impact"], total_cost=costs["total"],
            ))

        # --- Step 7: Holding costs ---
        holding_costs = compute_holding_costs(positions, mark_prices, cfg.cost, cfg.sessions_per_year)
        day_costs["borrow"] = holding_costs["borrow"]
        day_costs["financing"] = holding_costs["financing"]
        day_costs["total"] += holding_costs["total_holding"]
        cash -= holding_costs["total_holding"]

        # --- Step 8: Mark-to-market ---
        portfolio_value = 0.0
        pnl_gross = 0.0
        for sym, shares in list(positions.items()):
            px = mark_prices.get(sym, 0)
            if px <= 0 or abs(shares) < 0.01:
                if abs(shares) < 0.01:
                    del positions[sym]
                continue
            portfolio_value += shares * px

            # P&L from price change
            prev_px = prev_prices.get(sym, px)
            pnl_gross += shares * (px - prev_px)

        # Clean zero positions
        positions = {s: q for s, q in positions.items() if abs(q) >= 0.01}

        nav = cash + portfolio_value
        pnl_net = pnl_gross - day_costs["total"]

        # Exposure
        long_val = sum(q * mark_prices.get(s, 0) for s, q in positions.items() if q > 0)
        short_val = sum(abs(q) * mark_prices.get(s, 0) for s, q in positions.items() if q < 0)
        gross_exp = (long_val + short_val) / max(nav, 1)
        net_exp = (long_val - short_val) / max(nav, 1)

        weights = {}
        for sym, shares in positions.items():
            px = mark_prices.get(sym, 0)
            weights[sym] = (shares * px) / max(nav, 1)

        # --- Step 9: Record state ---
        state = PortfolioState(
            date=t, positions=dict(positions), weights=weights,
            cash=cash, nav=nav,
            gross_exposure=gross_exp, net_exposure=net_exp,
            n_longs=sum(1 for q in positions.values() if q > 0),
            n_shorts=sum(1 for q in positions.values() if q < 0),
            daily_pnl_gross=pnl_gross, daily_pnl_net=pnl_net,
            daily_costs=day_costs,
            daily_turnover=day_turnover / max(nav, 1),
            daily_trades=day_trades,
        )
        daily_states.append(state)

        equity_rows.append({
            "date": t, "nav": nav, "cash": cash,
            "pnl_gross": pnl_gross, "pnl_net": pnl_net,
            "costs_total": day_costs["total"],
            "costs_commission": day_costs["commission"],
            "costs_slippage": day_costs["slippage"],
            "costs_impact": day_costs["impact"],
            "costs_borrow": day_costs["borrow"],
            "gross_exposure": gross_exp, "net_exposure": net_exp,
            "n_longs": state.n_longs, "n_shorts": state.n_shorts,
            "turnover": state.daily_turnover, "n_trades": day_trades,
        })

        prev_prices = {s: mark_prices.get(s, 0) for s in positions}

    # --- Build equity curve ---
    eq = pd.DataFrame(equity_rows)
    if len(eq) > 0:
        eq["cum_pnl_gross"] = eq["pnl_gross"].cumsum()
        eq["cum_pnl_net"] = eq["pnl_net"].cumsum()
        eq["return_gross"] = eq["pnl_gross"] / eq["nav"].shift(1).fillna(cfg.initial_capital)
        eq["return_net"] = eq["pnl_net"] / eq["nav"].shift(1).fillna(cfg.initial_capital)
        eq["cum_return_gross"] = (1 + eq["return_gross"]).cumprod() - 1
        eq["cum_return_net"] = (1 + eq["return_net"]).cumprod() - 1

    # --- Summary ---
    summary = _compute_summary(eq, cfg, trade_log)

    LOGGER.info(
        "Backtest complete: %d sessions, NAV=%.0f→%.0f, Sharpe=%.2f, MaxDD=%.1f%%",
        len(daily_states), cfg.initial_capital,
        eq["nav"].iloc[-1] if len(eq) > 0 else cfg.initial_capital,
        summary.get("sharpe_net", 0), summary.get("max_drawdown_pct", 0),
    )

    return BacktestResult(
        daily_states=daily_states,
        trade_log=trade_log,
        equity_curve=eq,
        summary=summary,
        config=cfg,
    )


def _compute_summary(eq: pd.DataFrame, cfg: EngineConfig, trades: list) -> dict[str, Any]:
    """Compute summary statistics from equity curve."""
    if len(eq) < 2:
        return {"error": "insufficient data"}

    P = cfg.sessions_per_year
    nav_0 = cfg.initial_capital
    nav_T = float(eq["nav"].iloc[-1])
    n_days = len(eq)
    T_years = n_days / P

    # Returns
    ret_net = eq["return_net"].dropna()
    ret_gross = eq["return_gross"].dropna()

    total_return = nav_T / nav_0 - 1
    cagr = (nav_T / nav_0) ** (1 / max(T_years, 0.01)) - 1

    # Volatility
    vol_net = float(ret_net.std() * np.sqrt(P)) if len(ret_net) > 1 else 0
    vol_gross = float(ret_gross.std() * np.sqrt(P)) if len(ret_gross) > 1 else 0

    # Sharpe
    mu_net = float(ret_net.mean())
    sharpe_net = mu_net * np.sqrt(P) / ret_net.std() if ret_net.std() > 0 else 0
    mu_gross = float(ret_gross.mean())
    sharpe_gross = mu_gross * np.sqrt(P) / ret_gross.std() if ret_gross.std() > 0 else 0

    # Sortino
    downside = ret_net[ret_net < 0]
    sortino = mu_net * np.sqrt(P) / downside.std() if len(downside) > 1 and downside.std() > 0 else 0

    # Drawdown
    cum_nav = eq["nav"]
    running_max = cum_nav.cummax()
    drawdown = cum_nav / running_max - 1
    max_dd = float(drawdown.min())
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0

    # Time under water
    tuw = float((drawdown < 0).mean())

    # Costs
    total_costs = float(eq["costs_total"].sum())
    cost_bps = total_costs / max(float(eq["nav"].mean() * n_days), 1) * 1e4

    # Trades
    n_trades = len(trades)
    avg_turnover = float(eq["turnover"].mean()) if "turnover" in eq.columns else 0

    # Kelly
    kelly_f = mu_net / (ret_net.var()) if ret_net.var() > 0 else 0

    return {
        "n_sessions": n_days,
        "duration_years": round(T_years, 2),
        "nav_initial": nav_0,
        "nav_final": round(nav_T, 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "annual_vol_net_pct": round(vol_net * 100, 2),
        "annual_vol_gross_pct": round(vol_gross * 100, 2),
        "sharpe_net": round(float(sharpe_net), 3),
        "sharpe_gross": round(float(sharpe_gross), 3),
        "sortino_net": round(float(sortino), 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar": round(float(calmar), 3),
        "time_under_water_pct": round(tuw * 100, 1),
        "total_costs": round(total_costs, 2),
        "cost_annual_bps": round(cost_bps * P, 1),
        "n_trades": n_trades,
        "avg_daily_turnover_pct": round(avg_turnover * 100, 2),
        "kelly_fraction": round(float(kelly_f), 3),
    }
