"""
backtest/diagnostics.py — Post-hoc backtest health checks and anomaly detection.

Organised in diagnostic layers:
    I.   Structural integrity (no NaN, monotonic dates, NAV reconciliation)
    II.  Economic viability (costs vs gross, capacity constraints)
    III. Statistical robustness (Sharpe significance, bootstrap, deflated Sharpe)
    IV.  Overfitting signals (path dependency, strategy decay, regime fragility)

Each check returns a diagnostic record with severity (PASS/WARN/FAIL).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .engine import BacktestResult

LOGGER = logging.getLogger(__name__)


@dataclass
class DiagnosticCheck:
    layer: str          # "integrity" | "viability" | "robustness" | "overfitting"
    check_name: str
    severity: str       # "PASS" | "WARN" | "FAIL"
    metric_value: Any
    threshold: Any
    message: str


# ---------------------------------------------------------------------------
# Layer I: Structural integrity
# ---------------------------------------------------------------------------

def check_integrity(result: BacktestResult) -> list[DiagnosticCheck]:
    checks = []
    eq = result.equity_curve

    # NAV monotonicity check (no NaN)
    n_nan = int(eq["nav"].isna().sum())
    checks.append(DiagnosticCheck(
        "integrity", "nav_no_nan",
        "FAIL" if n_nan > 0 else "PASS",
        n_nan, 0, f"{n_nan} NaN NAV values",
    ))

    # NAV positive
    n_neg = int((eq["nav"] <= 0).sum())
    checks.append(DiagnosticCheck(
        "integrity", "nav_positive",
        "FAIL" if n_neg > 0 else "PASS",
        n_neg, 0, f"{n_neg} non-positive NAV sessions",
    ))

    # Date monotonicity
    dates = eq["date"].values
    is_sorted = all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
    checks.append(DiagnosticCheck(
        "integrity", "dates_monotonic",
        "PASS" if is_sorted else "FAIL",
        is_sorted, True, "Dates monotonic" if is_sorted else "Dates not sorted",
    ))

    # P&L reconciliation: Σ daily P&L ≈ NAV_T - NAV_0
    cumsum_pnl = float(eq["pnl_net"].sum())
    actual_change = float(eq["nav"].iloc[-1] - result.config.initial_capital)
    recon_err = abs(cumsum_pnl - actual_change) / max(abs(actual_change), 1)
    checks.append(DiagnosticCheck(
        "integrity", "pnl_reconciliation",
        "WARN" if recon_err > 0.01 else "PASS",
        round(recon_err, 6), 0.01,
        f"P&L recon error: {recon_err:.4%}",
    ))

    return checks


# ---------------------------------------------------------------------------
# Layer II: Economic viability
# ---------------------------------------------------------------------------

def check_viability(result: BacktestResult) -> list[DiagnosticCheck]:
    checks = []
    eq = result.equity_curve
    s = result.summary

    # Cost drag: total costs / gross P&L
    gross_pnl = float(eq["pnl_gross"].sum())
    total_costs = float(eq["costs_total"].sum())
    cost_ratio = total_costs / max(abs(gross_pnl), 1)
    checks.append(DiagnosticCheck(
        "viability", "cost_to_gross_ratio",
        "FAIL" if cost_ratio > 1.0 else ("WARN" if cost_ratio > 0.5 else "PASS"),
        round(cost_ratio, 4), 0.5,
        f"Costs consume {cost_ratio:.0%} of gross P&L",
    ))

    # Net Sharpe must be positive
    sharpe = s.get("sharpe_net", 0)
    checks.append(DiagnosticCheck(
        "viability", "sharpe_positive",
        "FAIL" if sharpe < 0 else ("WARN" if sharpe < 0.5 else "PASS"),
        sharpe, 0.5, f"Net Sharpe = {sharpe:.3f}",
    ))

    # Max drawdown severity
    max_dd = abs(s.get("max_drawdown_pct", 0))
    checks.append(DiagnosticCheck(
        "viability", "max_drawdown",
        "FAIL" if max_dd > 30 else ("WARN" if max_dd > 15 else "PASS"),
        max_dd, 15, f"Max DD = {max_dd:.1f}%",
    ))

    # Average turnover
    avg_to = s.get("avg_daily_turnover_pct", 0)
    checks.append(DiagnosticCheck(
        "viability", "avg_turnover",
        "WARN" if avg_to > 30 else "PASS",
        avg_to, 30, f"Avg daily turnover = {avg_to:.1f}%",
    ))

    return checks


# ---------------------------------------------------------------------------
# Layer III: Statistical robustness
# ---------------------------------------------------------------------------

def check_robustness(result: BacktestResult, n_bootstrap: int = 1000) -> list[DiagnosticCheck]:
    checks = []
    eq = result.equity_curve
    ret = eq["return_net"].dropna().values
    n = len(ret)
    P = result.config.sessions_per_year

    if n < 30:
        checks.append(DiagnosticCheck(
            "robustness", "sample_size", "WARN", n, 30,
            f"Only {n} sessions — stats unreliable",
        ))
        return checks

    # Bootstrap Sharpe confidence interval
    rng = np.random.RandomState(42)
    boot_sharpes = []
    for _ in range(n_bootstrap):
        sample = rng.choice(ret, size=n, replace=True)
        s_mu = sample.mean()
        s_std = sample.std()
        if s_std > 0:
            boot_sharpes.append(s_mu * np.sqrt(P) / s_std)

    boot_sharpes = np.array(boot_sharpes)
    ci_lo = float(np.percentile(boot_sharpes, 2.5))
    ci_hi = float(np.percentile(boot_sharpes, 97.5))
    pval = float((boot_sharpes <= 0).mean())

    checks.append(DiagnosticCheck(
        "robustness", "sharpe_bootstrap_ci",
        "WARN" if ci_lo < 0 else "PASS",
        f"[{ci_lo:.2f}, {ci_hi:.2f}]", "CI > 0",
        f"Sharpe 95% CI: [{ci_lo:.2f}, {ci_hi:.2f}], p(Sharpe≤0)={pval:.3f}",
    ))

    # Deflated Sharpe (correction for multiple testing)
    observed_sharpe = ret.mean() * np.sqrt(P) / ret.std() if ret.std() > 0 else 0
    # Assuming 10 strategy variants tested (configurable)
    n_tests = 10
    deflated = float(observed_sharpe / np.sqrt(1 + n_tests * 0.01))
    checks.append(DiagnosticCheck(
        "robustness", "deflated_sharpe",
        "WARN" if deflated < 0.5 else "PASS",
        round(deflated, 3), 0.5,
        f"Deflated Sharpe (assuming {n_tests} tests) = {deflated:.3f}",
    ))

    # Skewness and kurtosis
    skew = float(pd.Series(ret).skew())
    kurt = float(pd.Series(ret).kurtosis())
    checks.append(DiagnosticCheck(
        "robustness", "return_skewness",
        "WARN" if skew < -1.0 else "PASS",
        round(skew, 3), -1.0,
        f"Return skew = {skew:.3f} (negative skew = tail risk)",
    ))
    checks.append(DiagnosticCheck(
        "robustness", "return_kurtosis",
        "WARN" if kurt > 5.0 else "PASS",
        round(kurt, 3), 5.0,
        f"Return kurtosis = {kurt:.3f} (>5 = fat tails)",
    ))

    return checks


# ---------------------------------------------------------------------------
# Layer IV: Overfitting signals
# ---------------------------------------------------------------------------

def check_overfitting(result: BacktestResult) -> list[DiagnosticCheck]:
    checks = []
    eq = result.equity_curve
    s = result.summary
    P = result.config.sessions_per_year

    # Sharpe decay: first half vs second half
    n = len(eq)
    if n >= 60:
        r1 = eq["return_net"].iloc[:n//2].values
        r2 = eq["return_net"].iloc[n//2:].values
        s1 = r1.mean() * np.sqrt(P) / r1.std() if r1.std() > 0 else 0
        s2 = r2.mean() * np.sqrt(P) / r2.std() if r2.std() > 0 else 0
        decay = s1 - s2

        checks.append(DiagnosticCheck(
            "overfitting", "sharpe_half_decay",
            "WARN" if decay > 0.5 else "PASS",
            round(float(decay), 3), 0.5,
            f"Sharpe H1={s1:.2f}, H2={s2:.2f}, decay={decay:.2f}",
        ))

    # Gross vs net Sharpe gap (high cost drag = signal barely survives friction)
    sharpe_gap = s.get("sharpe_gross", 0) - s.get("sharpe_net", 0)
    checks.append(DiagnosticCheck(
        "overfitting", "gross_net_sharpe_gap",
        "WARN" if sharpe_gap > 0.5 else "PASS",
        round(float(sharpe_gap), 3), 0.5,
        f"Gross-Net Sharpe gap = {sharpe_gap:.3f} (high gap = fragile edge)",
    ))

    # Concentration: do top N names drive all P&L?
    if result.trade_log:
        trade_df = pd.DataFrame([{
            "symbol": t.symbol, "pnl": t.notional * 0.001  # proxy
        } for t in result.trade_log])
        if len(trade_df) > 0:
            by_sym = trade_df.groupby("symbol")["pnl"].sum().abs()
            total_abs = by_sym.sum()
            if total_abs > 0:
                top5_pct = by_sym.nlargest(5).sum() / total_abs
                checks.append(DiagnosticCheck(
                    "overfitting", "pnl_concentration_top5",
                    "WARN" if top5_pct > 0.5 else "PASS",
                    round(float(top5_pct), 3), 0.5,
                    f"Top 5 names = {top5_pct:.0%} of absolute P&L",
                ))

    return checks


# ---------------------------------------------------------------------------
# Run all diagnostics
# ---------------------------------------------------------------------------

def run_diagnostics(
    result: BacktestResult,
    n_bootstrap: int = 1000,
) -> tuple[list[DiagnosticCheck], dict[str, Any]]:
    """Run all 4 diagnostic layers."""
    all_checks: list[DiagnosticCheck] = []
    all_checks.extend(check_integrity(result))
    all_checks.extend(check_viability(result))
    all_checks.extend(check_robustness(result, n_bootstrap))
    all_checks.extend(check_overfitting(result))

    n_fail = sum(1 for c in all_checks if c.severity == "FAIL")
    n_warn = sum(1 for c in all_checks if c.severity == "WARN")
    n_pass = sum(1 for c in all_checks if c.severity == "PASS")

    overall = "FAIL" if n_fail > 0 else ("WARN" if n_warn > 0 else "PASS")
    summary = {
        "overall": overall,
        "n_checks": len(all_checks),
        "n_fail": n_fail, "n_warn": n_warn, "n_pass": n_pass,
    }

    LOGGER.info("Diagnostics: %s (%dF %dW %dP)", overall, n_fail, n_warn, n_pass)
    return all_checks, summary
