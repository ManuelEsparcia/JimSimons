"""
backtest — Event-driven portfolio simulation with institutional-grade
P&L accounting, transaction cost modelling, attribution, and diagnostics.

Pipeline: run_walkforward → engine (per session) → attribution → diagnostics → report
"""
from __future__ import annotations
import enum
from dataclasses import dataclass, field
from typing import Any


class BacktestError(RuntimeError):
    """Base error for backtest package."""

class ConfigError(BacktestError, ValueError):
    pass

class DataContractError(BacktestError):
    pass

class ReconciliationError(BacktestError):
    pass


class ExecutionModel(str, enum.Enum):
    """Price at which trades execute."""
    OPEN_NEXT = "open_next"     # execute at open of t+1 (conservative)
    CLOSE_SAME = "close_same"   # execute at close of t (aggressive)
    VWAP_NEXT = "vwap_next"     # VWAP of t+1 (realistic)
    MID = "mid"                 # mid of open/close (approximation)


class CostModel(str, enum.Enum):
    """Transaction cost model."""
    FIXED_BPS = "fixed_bps"
    LINEAR_IMPACT = "linear_impact"   # c * |notional| + k * sqrt(|notional|/ADV)
    ZERO = "zero"


@dataclass
class CostConfig:
    """Transaction cost parameters."""
    model: str = CostModel.FIXED_BPS.value
    fixed_commission_bps: float = 5.0   # 5bps per side
    slippage_bps: float = 10.0          # 10bps per side
    impact_coeff: float = 0.1           # for linear impact model
    borrow_annual_bps: float = 100.0    # 100bps annual for shorts
    financing_annual_bps: float = 50.0  # 50bps annual for cash
    min_commission: float = 0.0

    @property
    def total_one_way_bps(self) -> float:
        return self.fixed_commission_bps + self.slippage_bps


@dataclass
class EngineConfig:
    """Master configuration for the backtest engine."""
    initial_capital: float = 1_000_000.0
    execution_model: str = ExecutionModel.OPEN_NEXT.value
    cost: CostConfig = field(default_factory=CostConfig)

    # Portfolio constraints
    max_gross_exposure: float = 2.0       # 200% gross
    max_net_exposure: float = 0.3         # ±30% net
    max_position_weight: float = 0.05     # 5% per name
    max_turnover_daily: float = 0.5       # 50% daily turnover cap
    max_adv_participation: float = 0.05   # 5% of ADV

    # Rebalance
    rebalance_frequency: str = "daily"    # "daily" | "weekly" | "monthly"
    cash_buffer_pct: float = 0.02         # 2% cash buffer

    # Valuation
    valuation_price: str = "close"        # "close" | "open" | "vwap"

    # Calendar
    sessions_per_year: int = 252
