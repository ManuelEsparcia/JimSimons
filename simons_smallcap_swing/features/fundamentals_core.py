"""
features/fundamentals_core.py — PIT fundamental features (PhD-level).

Goes far beyond basic ratios. Implements academically documented anomalies
with strong out-of-sample evidence in small caps:

Valuation:
    ep_ratio             E/P (earnings yield)
    bm_ratio             B/M (book-to-market, Fama-French HML)
    fcf_yield            FCF/MarketCap
    ev_to_ebitda         EV/EBITDA
    sales_yield          Revenue/MarketCap

Profitability:
    roe, roa             Return on equity/assets
    gross_margin         Gross profit / Revenue
    operating_margin     Operating income / Revenue
    net_margin           Net income / Revenue
    gross_profitability  Gross profit / Assets (Novy-Marx 2013, ~4-6% alpha)

Quality composites:
    piotroski_f_score    9-signal composite (Piotroski 2000, ~23% in small value)
    altman_z_score       Distress proxy (Altman 1968, critical for shorts)
    sloan_accruals       (NI - CFO) / Assets (Sloan 1996, ~5-8% anomaly)
    earnings_quality     CFO / NI (high = good quality)

Earnings dynamics:
    sue                  Standardised Unexpected Earnings (PEAD, ~4-6%)
    earnings_surprise    Actual - Estimate (if estimates available)

Growth anomalies:
    asset_growth         (Assets_t - Assets_{t-4Q}) / Assets_{t-4Q} (Cooper 2008, ~7%)
    investment_rate      CapEx / Assets

Leverage & distress:
    debt_to_equity, debt_to_assets, current_ratio, interest_coverage

Size:
    log_mcap, log_assets, log_revenue

All features use PIT-strict data. Staleness tracked and invalidated.

References:
    Piotroski (2000) "Value Investing: The Use of Historical Financial Statement Information"
    Novy-Marx (2013) "The Other Side of Value: The Gross Profitability Premium"
    Sloan (1996) "Do Stock Prices Fully Reflect Information in Accruals and Cash Flows?"
    Cooper, Gulen & Schill (2008) "Asset Growth and the Cross-Section of Stock Returns"
    Altman (1968) "Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    FeatureError, DataContractError, FeatureFamily, FeatureDef,
    DEFAULT_DECISION_LAG, get_logger,
)

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class FundamentalsConfig:
    decision_lag: int = DEFAULT_DECISION_LAG
    max_staleness_days: int = 365
    near_zero_threshold: float = 1e-6
    stale_flag_days: int = 120
    enable_valuation: bool = True
    enable_profitability: bool = True
    enable_quality: bool = True
    enable_earnings_dynamics: bool = True
    enable_growth: bool = True
    enable_leverage: bool = True
    enable_size: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_ratio(num: pd.Series, den: pd.Series, thr: float = 1e-6) -> pd.Series:
    return num / den.where(den.abs() > thr)

def _safe_log(s: pd.Series) -> pd.Series:
    return np.log(s.where(s > 0))

def _resolve(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series | None:
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return df[low[c.lower()]]
    return None


METRIC_ALIASES = {
    "revenue": ("revenue", "total_revenue", "net_revenue", "revenuefromcontractwithcustomerexcludingassessedtax", "revenues"),
    "net_income": ("net_income", "netincomeloss", "netincome", "net_income_loss", "earnings"),
    "total_assets": ("total_assets", "assets", "totalassets"),
    "total_equity": ("total_equity", "stockholdersequity", "equity", "stockholders_equity"),
    "total_debt": ("total_debt", "longtermdebt", "long_term_debt"),
    "total_liabilities": ("total_liabilities", "liabilities", "liabilitiestotal"),
    "current_assets": ("current_assets", "assetscurrent"),
    "current_liabilities": ("current_liabilities", "liabilitiescurrent"),
    "ebitda": ("ebitda", "operating_income", "operatingincomeloss"),
    "gross_profit": ("gross_profit", "grossprofit"),
    "operating_income": ("operating_income", "operatingincomeloss", "operating_profit"),
    "cfo": ("cfo", "operating_cash_flow", "cash_from_operations", "netcashprovidedbyoperatingactivities"),
    "capex": ("capex", "capital_expenditures", "paymentstoacquirepropertyplantandequipment"),
    "fcf": ("free_cash_flow", "fcf"),
    "book_value": ("book_value", "stockholdersequity", "total_equity"),
    "shares_outstanding": ("shares_outstanding", "commonstocksharesoutstanding", "weighted_average_shares", "shares_out"),
    "mcap": ("mcap", "market_cap", "market_capitalization"),
    "enterprise_value": ("enterprise_value", "ev"),
    "interest_expense": ("interest_expense", "interestexpense", "interest_paid"),
    "retained_earnings": ("retained_earnings", "retainedearningsaccumulateddeficit"),
    "working_capital": ("working_capital",),
    "sales": ("sales", "revenue", "total_revenue"),
    "receivables": ("receivables", "accountsreceivablenetcurrent"),
    "inventory": ("inventory", "inventorynet"),
    "depreciation": ("depreciation", "depreciationandamortization", "da"),
}


def _resolve_all(df: pd.DataFrame) -> dict[str, pd.Series | None]:
    return {k: _resolve(df, v) for k, v in METRIC_ALIASES.items()}


# ---------------------------------------------------------------------------
# VALUATION
# ---------------------------------------------------------------------------

def compute_valuation(m: dict, cfg: FundamentalsConfig) -> dict[str, pd.Series]:
    f = {}
    ni, mcap = m["net_income"], m["mcap"]
    bv = m["book_value"] if m["book_value"] is not None else m["total_equity"]
    fcf, ev, ebitda, rev = m["fcf"], m["enterprise_value"], m["ebitda"], m["revenue"]

    if ni is not None and mcap is not None:
        f["ep_ratio"] = _safe_ratio(ni, mcap, cfg.near_zero_threshold)
    if bv is not None and mcap is not None:
        f["bm_ratio"] = _safe_ratio(bv, mcap, cfg.near_zero_threshold)
    if fcf is not None and mcap is not None:
        f["fcf_yield"] = _safe_ratio(fcf, mcap, cfg.near_zero_threshold)
    if ev is not None and ebitda is not None:
        f["ev_to_ebitda"] = _safe_ratio(ev, ebitda, cfg.near_zero_threshold)
    if rev is not None and mcap is not None:
        f["sales_yield"] = _safe_ratio(rev, mcap, cfg.near_zero_threshold)
    return f


# ---------------------------------------------------------------------------
# PROFITABILITY
# ---------------------------------------------------------------------------

def compute_profitability(m: dict, cfg: FundamentalsConfig) -> dict[str, pd.Series]:
    f = {}
    ni, eq, assets = m["net_income"], m["total_equity"], m["total_assets"]
    rev, gp, oi = m["revenue"], m["gross_profit"], m["operating_income"]

    if ni is not None and eq is not None:
        f["roe"] = _safe_ratio(ni, eq, cfg.near_zero_threshold)
    if ni is not None and assets is not None:
        f["roa"] = _safe_ratio(ni, assets, cfg.near_zero_threshold)
    if gp is not None and rev is not None:
        f["gross_margin"] = _safe_ratio(gp, rev, cfg.near_zero_threshold)
    if oi is not None and rev is not None:
        f["operating_margin"] = _safe_ratio(oi, rev, cfg.near_zero_threshold)
    if ni is not None and rev is not None:
        f["net_margin"] = _safe_ratio(ni, rev, cfg.near_zero_threshold)
    # Novy-Marx (2013): Gross profitability premium ~4-6% annually
    if gp is not None and assets is not None:
        f["gross_profitability"] = _safe_ratio(gp, assets, cfg.near_zero_threshold)
    return f


# ---------------------------------------------------------------------------
# QUALITY COMPOSITES
# ---------------------------------------------------------------------------

def compute_piotroski_f_score(m: dict, df: pd.DataFrame, sym_col: str) -> pd.Series:
    """Piotroski (2000) F-Score: 9 binary signals of financial health.

    In small-cap value stocks, high F-score (7-9) outperforms low (0-2)
    by ~23% annually. This is one of the most robust anomalies.

    Signals:
    1. ROA > 0                    (profitability)
    2. CFO > 0                    (cash quality)
    3. ΔROA > 0                   (improving profitability)
    4. CFO > NI (accrual quality) (earnings quality)
    5. ΔLeverage < 0              (deleveraging)
    6. ΔCurrent Ratio > 0         (improving liquidity)
    7. No new equity issuance     (no dilution)
    8. ΔGross Margin > 0          (improving efficiency)
    9. ΔAsset Turnover > 0        (improving efficiency)
    """
    score = pd.Series(0.0, index=df.index)
    valid_count = pd.Series(0, index=df.index)

    ni, cfo, assets = m["net_income"], m["cfo"], m["total_assets"]
    eq, cl, gp, rev = m["total_equity"], m["current_liabilities"], m["gross_profit"], m["revenue"]
    ca, debt = m["current_assets"], m["total_debt"]
    shares = m["shares_outstanding"]

    # Helper: lagged value within group
    def _lag(s, n=1):
        if s is None: return None
        return s.groupby(df[sym_col]).shift(n)

    # 1. ROA > 0
    if ni is not None and assets is not None:
        roa = _safe_ratio(ni, assets)
        s1 = (roa > 0).astype(float)
        score += s1.fillna(0)
        valid_count += s1.notna().astype(int)

    # 2. CFO > 0
    if cfo is not None:
        s2 = (cfo > 0).astype(float)
        score += s2.fillna(0)
        valid_count += s2.notna().astype(int)

    # 3. ΔROA > 0
    if ni is not None and assets is not None:
        roa = _safe_ratio(ni, assets)
        roa_lag = _lag(roa)
        s3 = (roa > roa_lag).astype(float) if roa_lag is not None else None
        if s3 is not None:
            score += s3.fillna(0)
            valid_count += s3.notna().astype(int)

    # 4. CFO > NI (accrual quality)
    if cfo is not None and ni is not None:
        s4 = (cfo > ni).astype(float)
        score += s4.fillna(0)
        valid_count += s4.notna().astype(int)

    # 5. ΔLeverage < 0
    if debt is not None and assets is not None:
        lev = _safe_ratio(debt, assets)
        lev_lag = _lag(lev)
        if lev_lag is not None:
            s5 = (lev < lev_lag).astype(float)
            score += s5.fillna(0)
            valid_count += s5.notna().astype(int)

    # 6. ΔCurrent Ratio > 0
    if ca is not None and cl is not None:
        cr = _safe_ratio(ca, cl)
        cr_lag = _lag(cr)
        if cr_lag is not None:
            s6 = (cr > cr_lag).astype(float)
            score += s6.fillna(0)
            valid_count += s6.notna().astype(int)

    # 7. No dilution (shares not increasing)
    if shares is not None:
        sh_lag = _lag(shares)
        if sh_lag is not None:
            s7 = (shares <= sh_lag).astype(float)
            score += s7.fillna(0)
            valid_count += s7.notna().astype(int)

    # 8. ΔGross Margin > 0
    if gp is not None and rev is not None:
        gm = _safe_ratio(gp, rev)
        gm_lag = _lag(gm)
        if gm_lag is not None:
            s8 = (gm > gm_lag).astype(float)
            score += s8.fillna(0)
            valid_count += s8.notna().astype(int)

    # 9. ΔAsset Turnover > 0
    if rev is not None and assets is not None:
        at = _safe_ratio(rev, assets)
        at_lag = _lag(at)
        if at_lag is not None:
            s9 = (at > at_lag).astype(float)
            score += s9.fillna(0)
            valid_count += s9.notna().astype(int)

    # Require at least 5 of 9 signals to be computable
    score = score.where(valid_count >= 5)
    return score


def compute_altman_z(m: dict, cfg: FundamentalsConfig) -> pd.Series:
    """Altman (1968) Z-Score: bankruptcy prediction.

    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Cap / Total Liabilities
    X5 = Revenue / Total Assets

    Z < 1.81: distress zone (short signal in small caps)
    Z > 2.99: safe zone
    """
    assets = m["total_assets"]
    if assets is None:
        return pd.Series(np.nan, index=assets.index if assets is not None else pd.RangeIndex(0))

    thr = cfg.near_zero_threshold
    wc = m.get("working_capital")
    if wc is None and m["current_assets"] is not None and m["current_liabilities"] is not None:
        wc = m["current_assets"] - m["current_liabilities"]

    x1 = _safe_ratio(wc, assets, thr) if wc is not None else 0.0
    x2 = _safe_ratio(m.get("retained_earnings"), assets, thr) if m.get("retained_earnings") is not None else 0.0
    oi = m.get("operating_income")
    if oi is None:
        oi = m.get("ebitda")
    x3 = _safe_ratio(oi, assets, thr) if oi is not None else 0.0
    x4 = _safe_ratio(m["mcap"], m.get("total_liabilities"), thr) if m["mcap"] is not None and m.get("total_liabilities") is not None else 0.0
    x5 = _safe_ratio(m["revenue"], assets, thr) if m["revenue"] is not None else 0.0

    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
    if isinstance(z, (int, float)):
        return pd.Series(np.nan, index=assets.index)
    return z


def compute_sloan_accruals(m: dict, cfg: FundamentalsConfig) -> pd.Series:
    """Sloan (1996) accruals ratio: (NI - CFO) / Total Assets.

    High accruals (NI >> CFO) predict negative future returns.
    ~5-8% annual alpha from going long low-accruals, short high-accruals.
    """
    ni, cfo, assets = m["net_income"], m["cfo"], m["total_assets"]
    if ni is None or cfo is None or assets is None:
        idx = ni.index if ni is not None else (cfo.index if cfo is not None else pd.RangeIndex(0))
        return pd.Series(np.nan, index=idx)
    accruals = ni - cfo
    return _safe_ratio(accruals, assets, cfg.near_zero_threshold)


def compute_earnings_quality(m: dict) -> pd.Series:
    """CFO / NI: high ratio = high earnings quality (cash-backed)."""
    cfo, ni = m["cfo"], m["net_income"]
    if cfo is None or ni is None:
        idx = cfo.index if cfo is not None else pd.RangeIndex(0)
        return pd.Series(np.nan, index=idx)
    return _safe_ratio(cfo, ni)


# ---------------------------------------------------------------------------
# EARNINGS DYNAMICS
# ---------------------------------------------------------------------------

def compute_sue(
    earnings: pd.Series, group: pd.Series, window: int = 8,
) -> pd.Series:
    """Standardised Unexpected Earnings.

    SUE = (E_t - E_{t-4}) / std(E_t - E_{t-4}, window)

    Post-Earnings Announcement Drift (PEAD) is one of the most robust
    anomalies: ~4-6% annually. SUE is the standard operationalisation.
    """
    if earnings is None:
        return pd.Series(dtype=float)
    e_lag4 = earnings.groupby(group).shift(4)
    surprise = earnings - e_lag4
    vol = surprise.groupby(group).rolling(window, min_periods=4).std().droplevel(0)
    return _safe_ratio(surprise, vol)


# ---------------------------------------------------------------------------
# GROWTH ANOMALIES
# ---------------------------------------------------------------------------

def compute_asset_growth(assets: pd.Series, group: pd.Series) -> pd.Series:
    """Cooper, Gulen & Schill (2008): asset growth anomaly.

    AG = (Assets_t - Assets_{t-4Q}) / Assets_{t-4Q}

    Firms with high asset growth underperform by ~7% annually.
    This is a robust anomaly, especially in small caps.
    """
    if assets is None:
        return pd.Series(dtype=float)
    lag = assets.groupby(group).shift(4)
    return _safe_ratio(assets - lag, lag)


def compute_investment_rate(m: dict, cfg: FundamentalsConfig) -> pd.Series:
    """CapEx / Total Assets."""
    capex, assets = m["capex"], m["total_assets"]
    if capex is None or assets is None:
        idx = assets.index if assets is not None else pd.RangeIndex(0)
        return pd.Series(np.nan, index=idx)
    return _safe_ratio(capex.abs(), assets, cfg.near_zero_threshold)


# ---------------------------------------------------------------------------
# LEVERAGE
# ---------------------------------------------------------------------------

def compute_leverage(m: dict, cfg: FundamentalsConfig) -> dict[str, pd.Series]:
    f = {}
    debt, eq, assets = m["total_debt"], m["total_equity"], m["total_assets"]
    ca, cl = m["current_assets"], m["current_liabilities"]
    oi, interest = m["operating_income"], m["interest_expense"]

    if debt is not None and eq is not None:
        f["debt_to_equity"] = _safe_ratio(debt, eq, cfg.near_zero_threshold)
    if debt is not None and assets is not None:
        f["debt_to_assets"] = _safe_ratio(debt, assets, cfg.near_zero_threshold)
    if ca is not None and cl is not None:
        f["current_ratio"] = _safe_ratio(ca, cl, cfg.near_zero_threshold)
    if oi is not None and interest is not None:
        f["interest_coverage"] = _safe_ratio(oi, interest, cfg.near_zero_threshold)
    return f


# ---------------------------------------------------------------------------
# SIZE
# ---------------------------------------------------------------------------

def compute_size(m: dict) -> dict[str, pd.Series]:
    f = {}
    for name, key in [("log_mcap", "mcap"), ("log_assets", "total_assets"), ("log_revenue", "revenue")]:
        s = m.get(key)
        if s is not None:
            f[name] = _safe_log(s)
    return f


# ---------------------------------------------------------------------------
# STALENESS
# ---------------------------------------------------------------------------

def compute_staleness(df, date_col="date", filed_date_col="filed_date"):
    if filed_date_col not in df.columns:
        return pd.Series(np.nan, index=df.index, name="staleness_days")
    return (pd.to_datetime(df[date_col]) - pd.to_datetime(df[filed_date_col])).dt.days.rename("staleness_days")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def build_fundamental_features(
    fundamentals: pd.DataFrame,
    *,
    config: FundamentalsConfig | None = None,
    price_mcap: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[FeatureDef]]:
    cfg = config or FundamentalsConfig()
    df = fundamentals.copy()

    if price_mcap is not None and "mcap" not in df.columns:
        mcap_cols = [c for c in ("date", "symbol", "mcap", "market_cap") if c in price_mcap.columns]
        if len(mcap_cols) >= 3:
            mc = price_mcap[mcap_cols].copy()
            if "market_cap" in mc.columns and "mcap" not in mc.columns:
                mc = mc.rename(columns={"market_cap": "mcap"})
            df = df.merge(mc[["date", "symbol", "mcap"]], on=["date", "symbol"], how="left")

    m = _resolve_all(df)
    sym_col = "symbol" if "symbol" in df.columns else "ticker"
    all_f: dict[str, pd.Series] = {}

    if cfg.enable_valuation:
        all_f.update(compute_valuation(m, cfg))
    if cfg.enable_profitability:
        all_f.update(compute_profitability(m, cfg))
    if cfg.enable_quality:
        all_f["piotroski_f_score"] = compute_piotroski_f_score(m, df, sym_col)
        all_f["altman_z_score"] = compute_altman_z(m, cfg)
        all_f["sloan_accruals"] = compute_sloan_accruals(m, cfg)
        all_f["earnings_quality"] = compute_earnings_quality(m)
    if cfg.enable_earnings_dynamics:
        if m["net_income"] is not None:
            all_f["sue"] = compute_sue(m["net_income"], df[sym_col])
    if cfg.enable_growth:
        if m["total_assets"] is not None:
            all_f["asset_growth"] = compute_asset_growth(m["total_assets"], df[sym_col])
        all_f["investment_rate"] = compute_investment_rate(m, cfg)
    if cfg.enable_leverage:
        all_f.update(compute_leverage(m, cfg))
    if cfg.enable_size:
        all_f.update(compute_size(m))

    date_col = "date" if "date" in df.columns else "asof_date"
    out = df[[date_col, sym_col]].copy()
    out.columns = ["date", "symbol"]

    for name, series in all_f.items():
        if isinstance(series, pd.Series) and len(series) == len(out):
            out[name] = series.values
        elif isinstance(series, pd.Series):
            out[name] = series.reindex(df.index).values
        # Skip non-Series (e.g. scalar 0.0 from Altman fallback)

    staleness = compute_staleness(df)
    out["staleness_days"] = staleness.values
    out["is_stale"] = staleness > cfg.stale_flag_days

    feat_cols = [c for c in out.columns if c not in ("date", "symbol", "staleness_days", "is_stale")]
    out[feat_cols] = out[feat_cols].replace([np.inf, -np.inf], np.nan)

    if cfg.max_staleness_days > 0:
        too_stale = staleness > cfg.max_staleness_days
        for col in feat_cols:
            out.loc[too_stale, col] = np.nan

    defs = [FeatureDef(c, FeatureFamily.FUNDAMENTAL_LEVEL, c, 0, cfg.decision_lag, "edgar_pit") for c in feat_cols]
    LOGGER.info("Fundamentals: %d features, %d rows", len(feat_cols), len(out))
    return out, defs
