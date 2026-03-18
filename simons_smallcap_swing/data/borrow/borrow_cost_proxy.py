"""
data/borrow/borrow_cost_proxy.py — Causal borrow cost estimation.

3-layer architecture:
    Layer 1: Normalised causal signals (log ADV, log MC, σ, SI, DTC, HTB)
    Layer 2: Latent stress score z = β·x (monotonic: β ≥ 0)
    Layer 3: Deterministic mapping z → B_hat (piecewise), α = sigmoid(-z)

Conservatism principle: in ambiguity, overestimate cost, underestimate availability.
Monotonicity invariant: less liquid + smaller + more volatile + more shorted → higher cost.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from . import BorrowTier, ProxyQuality

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BorrowProxyConfig:
    """All parameters versionable and auditable."""
    # Stress score coefficients (all >= 0 by spec)
    beta_log_adv: float = 0.25       # lower ADV → higher stress
    beta_log_mc: float = 0.20        # lower mcap → higher stress
    beta_log_vol: float = 0.15       # higher vol → higher stress
    beta_si: float = 0.20            # higher short interest → higher stress
    beta_dtc: float = 0.10           # higher days-to-cover → higher stress
    beta_htb: float = 0.10           # HTB flag → higher stress

    # Piecewise fee mapping breakpoints (on z)
    c1: float = -1.0                 # easy/medium boundary
    c2: float = 0.5                  # medium/hard boundary
    c3: float = 2.0                  # hard/extreme boundary
    gamma: float = 0.5              # exponential growth rate above c3

    # Fee levels (annualised fraction)
    b_min: float = 0.005             # 50bps easy
    b_med: float = 0.05              # 5% medium
    b_hard: float = 0.30             # 30% hard
    b_max: float = 5.0               # 500% cap

    # Availability thresholds
    htb_fee_threshold: float = 0.10  # fee > 10% → HTB flag
    htb_alpha_threshold: float = 0.20  # availability < 20% → HTB

    # Tier thresholds (on fee)
    tier_easy_max: float = 0.015
    tier_medium_max: float = 0.10
    tier_hard_max: float = 0.60

    # Smoothing
    ema_lambda: float = 0.3          # EMA smoothing factor
    hysteresis_down_days: int = 3    # days before downgrading tier

    # Winsor percentiles
    winsor_lo: float = 0.01
    winsor_hi: float = 0.99

    # Rolling windows
    adv_window: int = 20
    vol_window: int = 20

    version: str = "1.0"


# ---------------------------------------------------------------------------
# Layer 1: Causal signal extraction
# ---------------------------------------------------------------------------

def extract_signals(
    df: pd.DataFrame,
    cfg: BorrowProxyConfig,
) -> pd.DataFrame:
    """Extract and normalise causal signals per (date, symbol).

    All signals use only information available at close of date t.
    """
    out = df[["date", "symbol"]].copy()
    g = df.groupby("symbol")

    # Log ADV (negated: low ADV → high stress)
    if "close" in df.columns and "volume" in df.columns:
        dv = df["close"] * df["volume"]
        adv = dv.groupby(df["symbol"]).rolling(cfg.adv_window, min_periods=5).mean().droplevel(0)
        out["neg_log_adv"] = -np.log(adv.clip(lower=1).reindex(df.index))
    else:
        out["neg_log_adv"] = 0.0

    # Log market cap (negated)
    if "mcap" in df.columns:
        out["neg_log_mc"] = -np.log(df["mcap"].clip(lower=1))
    elif "close" in df.columns and "shares_outstanding" in df.columns:
        mc = df["close"] * df["shares_outstanding"]
        out["neg_log_mc"] = -np.log(mc.clip(lower=1))
    else:
        out["neg_log_mc"] = 0.0

    # Log volatility
    if "close" in df.columns:
        log_ret = np.log(df["close"] / df.groupby("symbol")["close"].shift(1))
        vol = log_ret.groupby(df["symbol"]).rolling(cfg.vol_window, min_periods=5).std().droplevel(0)
        out["log_vol"] = np.log(1 + vol.fillna(0).clip(lower=0))
    else:
        out["log_vol"] = 0.0

    # Short interest
    if "short_interest_ratio" in df.columns:
        out["log_si"] = np.log(1 + df["short_interest_ratio"].fillna(0).clip(lower=0))
    else:
        out["log_si"] = 0.0

    # Days to cover
    if "days_to_cover" in df.columns:
        out["log_dtc"] = np.log(1 + df["days_to_cover"].fillna(0).clip(lower=0))
    else:
        out["log_dtc"] = 0.0

    # HTB evidence
    if "htb_flag" in df.columns:
        out["htb_signal"] = df["htb_flag"].fillna(0).astype(float)
    else:
        out["htb_signal"] = 0.0

    return out


# ---------------------------------------------------------------------------
# Layer 2: Latent stress score (monotonic)
# ---------------------------------------------------------------------------

def compute_stress_score(signals: pd.DataFrame, cfg: BorrowProxyConfig) -> pd.Series:
    """z = β·x with β ≥ 0. Higher z = more borrowing stress."""
    z = (
        cfg.beta_log_adv * signals["neg_log_adv"]
        + cfg.beta_log_mc * signals["neg_log_mc"]
        + cfg.beta_log_vol * signals["log_vol"]
        + cfg.beta_si * signals["log_si"]
        + cfg.beta_dtc * signals["log_dtc"]
        + cfg.beta_htb * signals["htb_signal"]
    )
    return z


# ---------------------------------------------------------------------------
# Layer 3: Deterministic mapping z → fee, availability, tier
# ---------------------------------------------------------------------------

def map_fee(z: pd.Series, cfg: BorrowProxyConfig) -> pd.Series:
    """Piecewise monotonic mapping: z → B_hat (annual fee)."""
    fee = pd.Series(cfg.b_min, index=z.index)

    # Linear interpolation zones
    mask1 = (z > cfg.c1) & (z <= cfg.c2)
    fee[mask1] = cfg.b_min + (cfg.b_med - cfg.b_min) * (z[mask1] - cfg.c1) / (cfg.c2 - cfg.c1)

    mask2 = (z > cfg.c2) & (z <= cfg.c3)
    fee[mask2] = cfg.b_med + (cfg.b_hard - cfg.b_med) * (z[mask2] - cfg.c2) / (cfg.c3 - cfg.c2)

    # Exponential above c3
    mask3 = z > cfg.c3
    fee[mask3] = (cfg.b_hard * np.exp(cfg.gamma * (z[mask3] - cfg.c3))).clip(upper=cfg.b_max)

    return fee.clip(lower=0, upper=cfg.b_max)


def map_availability(z: pd.Series) -> pd.Series:
    """α = sigmoid(-z): higher stress → lower availability."""
    return 1.0 / (1.0 + np.exp(z))


def classify_tier(fee: pd.Series, alpha: pd.Series, htb: pd.Series, cfg: BorrowProxyConfig) -> pd.Series:
    """Deterministic tier assignment."""
    tier = pd.Series(BorrowTier.EASY.value, index=fee.index)
    tier[(fee > cfg.tier_easy_max) | (alpha < 0.6)] = BorrowTier.MEDIUM.value
    tier[(fee > cfg.tier_medium_max) | (alpha < 0.3) | (htb == 1)] = BorrowTier.HARD.value
    tier[(fee > cfg.tier_hard_max) | (alpha < 0.1)] = BorrowTier.BLOCKED.value
    return tier


def classify_htb(fee: pd.Series, alpha: pd.Series, htb_input: pd.Series, cfg: BorrowProxyConfig) -> pd.Series:
    """HTB flag: fee > threshold OR availability < threshold OR direct HTB."""
    return ((fee >= cfg.htb_fee_threshold) | (alpha <= cfg.htb_alpha_threshold) | (htb_input == 1)).astype(int)


# ---------------------------------------------------------------------------
# Smoothing and hysteresis
# ---------------------------------------------------------------------------

def smooth_fee(fee: pd.Series, group: pd.Series, lam: float) -> pd.Series:
    """EMA smoothing: B_hat^sm = λ·B_hat + (1-λ)·B_hat^sm_{t-1}."""
    return fee.groupby(group).transform(lambda s: s.ewm(alpha=lam, adjust=False).mean())


# ---------------------------------------------------------------------------
# Proxy quality and source
# ---------------------------------------------------------------------------

def classify_quality(df: pd.DataFrame) -> pd.Series:
    """Proxy quality based on input availability."""
    q = pd.Series(ProxyQuality.HIGH.value, index=df.index)

    # Degrade if missing short interest
    has_si = df.get("short_interest_ratio", pd.Series(0, index=df.index)).notna() & (df.get("short_interest_ratio", pd.Series(0, index=df.index)) > 0)
    q[~has_si] = ProxyQuality.MEDIUM.value

    # Degrade further if missing volume/price
    has_vol = df.get("volume", pd.Series(0, index=df.index)).notna() & (df.get("volume", pd.Series(0, index=df.index)) > 0)
    q[~has_vol] = ProxyQuality.LOW.value

    return q


def classify_source(df: pd.DataFrame) -> pd.Series:
    """Dominant source used for proxy inference."""
    src = pd.Series("liquidity_volatility_proxy", index=df.index)
    if "short_interest_ratio" in df.columns:
        has_si = df["short_interest_ratio"].notna() & (df["short_interest_ratio"] > 0)
        src[has_si] = "short_interest_derived"
    if "htb_flag" in df.columns:
        src[df["htb_flag"] == 1] = "explicit_htb_flag"
    return src


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def build_borrow_cost_proxy(
    market_data: pd.DataFrame,
    *,
    config: BorrowProxyConfig | None = None,
) -> pd.DataFrame:
    """Build daily borrow cost proxy for all symbols.

    Parameters
    ----------
    market_data : DataFrame
        Must have: date, symbol, close, volume.
        Optional: mcap, shares_outstanding, short_interest_ratio,
                  days_to_cover, htb_flag.
    config : BorrowProxyConfig

    Returns
    -------
    DataFrame with full output contract (15 columns per spec).
    """
    cfg = config or BorrowProxyConfig()
    df = market_data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Layer 1: Extract signals
    signals = extract_signals(df, cfg)

    # Winsorise and STANDARDISE signals per date (cross-sectional z-score)
    signal_cols = ["neg_log_adv", "neg_log_mc", "log_vol", "log_si", "log_dtc"]
    for col in signal_cols:
        def _standardise(s):
            if len(s) < 10:
                return s - s.median()
            clipped = s.clip(lower=s.quantile(cfg.winsor_lo), upper=s.quantile(cfg.winsor_hi))
            med = clipped.median()
            mad = (clipped - med).abs().median()
            scale = mad * 1.4826 if mad > 1e-10 else clipped.std()
            return (clipped - med) / max(scale, 1e-10)
        signals[col] = signals.groupby("date")[col].transform(_standardise)

    # Layer 2: Stress score
    z = compute_stress_score(signals, cfg)

    # Layer 3: Map to fee, availability, tier
    fee = map_fee(z, cfg)
    fee_smooth = smooth_fee(fee, df["symbol"], cfg.ema_lambda)
    alpha = map_availability(z)

    htb_input = df["htb_flag"].fillna(0).astype(float) if "htb_flag" in df.columns else pd.Series(0, index=df.index)
    htb = classify_htb(fee_smooth, alpha, htb_input, cfg)
    tier = classify_tier(fee_smooth, alpha, htb, cfg)
    quality = classify_quality(df)
    source = classify_source(df)

    # Detect jumps (fee changes > 3× in one day)
    fee_pct_change = fee_smooth.groupby(df["symbol"]).pct_change().abs()
    jump_flag = (fee_pct_change > 3.0).astype(int).fillna(0)

    # Fallback flag
    fallback = (quality == ProxyQuality.LOW.value).astype(int)

    # Stale input flag (no volume for > 5 days)
    stale = (df["volume"].fillna(0) == 0).astype(int) if "volume" in df.columns else pd.Series(0, index=df.index)

    # Build output
    out = pd.DataFrame({
        "symbol": df["symbol"],
        "date": df["date"],
        "borrow_fee_annual": fee_smooth.clip(lower=0, upper=cfg.b_max),
        "borrow_fee_daily": (fee_smooth / 365).clip(lower=0),
        "borrow_availability_score": alpha.clip(0, 1),
        "htb_flag": htb,
        "borrow_tier": tier,
        "proxy_quality": quality,
        "proxy_source": source,
        "stress_score": z,
        "jump_flag": jump_flag.astype(int),
        "stale_input_flag": stale,
        "fallback_flag": fallback,
        "config_version": cfg.version,
        "asof_timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    })

    # Replace any NaN in critical fields
    out["borrow_fee_annual"] = out["borrow_fee_annual"].fillna(cfg.b_med)
    out["borrow_fee_daily"] = out["borrow_fee_daily"].fillna(cfg.b_med / 365)
    out["borrow_availability_score"] = out["borrow_availability_score"].fillna(0.5)

    LOGGER.info(
        "Borrow proxy: %d rows, tier distribution: %s",
        len(out), out["borrow_tier"].value_counts().to_dict(),
    )
    return out
