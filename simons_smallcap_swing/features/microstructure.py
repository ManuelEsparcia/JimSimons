"""
features/microstructure.py — Daily microstructure proxies from OHLCV.

PhD-level: 20+ features covering friction, liquidity, volatility
microstructure, and information asymmetry from daily OHLCV only.

Friction / Spread:
    spread_proxy          (H-L)/C
    roll_spread           Roll (1984) 2*sqrt(-Cov(dr,dr_lag))
    corwin_schultz_spread Corwin-Schultz (2012) from 2-day H/L
    bid_ask_bounce        -autocorr(r, lag=1)

Liquidity:
    amihud               mean(|r|/DV)            Amihud (2002)
    kyle_lambda          mean(|dP|/sqrt(V))      Kyle (1985) proxy
    turnover             mean(V/shares_out)
    dv_zscore            z(log(DV)) vs trailing
    zero_vol_fraction    frac(V=0) in window

Volatility:
    realized_vol         std(log_ret)             close-to-close
    parkinson_vol        sqrt(mean(log(H/L)^2)/(4ln2))  Parkinson(1980)
    garman_klass_vol     GK from OHLC             Garman-Klass(1980)
    vol_of_vol           std(realized_vol)
    noise_ratio          parkinson/realized        intraday vs cc
    overnight_vol        std(log(O_t/C_{t-1}))

Patterns:
    gap_open, hl_range, reversal_1d, vol_shock

Momentum:
    momentum, momentum_vol_adjusted (momentum/vol)

All vectorised. Zero iterrows. PIT via decision_lag.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import (
    FeatureError, DataContractError, FeatureFamily, FeatureDef,
    DEFAULT_DECISION_LAG, resolve_date_col, resolve_symbol_col, get_logger,
)

LOGGER = get_logger(__name__)

LN2 = np.log(2)
FOUR_LN2 = 4.0 * LN2
TWO_LN2_MINUS_1 = 2.0 * LN2 - 1.0
CS_ALPHA_DENOM = 2.0 * (np.sqrt(2.0) - 1.0)


@dataclass(frozen=True)
class MicrostructureConfig:
    momentum_windows: tuple[int, ...] = (5, 10, 21, 63)
    volatility_windows: tuple[int, ...] = (5, 10, 21, 63)
    amihud_window: int = 21
    kyle_window: int = 21
    turnover_window: int = 21
    vol_shock_window: int = 21
    roll_spread_window: int = 21
    corwin_schultz_window: int = 21
    bounce_window: int = 21
    overnight_vol_window: int = 21
    min_price: float = 0.50
    min_dollar_volume: float = 1000.0
    decision_lag: int = DEFAULT_DECISION_LAG
    clip_amihud_upper: float = 1e6
    clip_kyle_upper: float = 1e6
    clip_spread_upper: float = 1.0
    clip_vol_upper: float = 5.0
    enable_friction: bool = True
    enable_liquidity: bool = True
    enable_volatility: bool = True
    enable_patterns: bool = True
    enable_momentum: bool = True
    enable_vol_adjusted_momentum: bool = True
    shares_out_col: str = "shares_outstanding"

    @property
    def max_lookback(self) -> int:
        return max(
            max(self.momentum_windows, default=0),
            max(self.volatility_windows, default=0),
            self.amihud_window, self.kyle_window,
            self.roll_spread_window, self.corwin_schultz_window,
        )


# --- OHLCV resolution ---

def _resolve_ohlcv(df: pd.DataFrame) -> dict[str, str]:
    lowered = {c.lower(): c for c in df.columns}
    aliases = {
        "open": ("open", "adj_open", "open_adj", "px_open"),
        "high": ("high", "adj_high", "high_adj", "px_high"),
        "low": ("low", "adj_low", "low_adj", "px_low"),
        "close": ("close", "adj_close", "adjusted_close", "close_adj", "px_close"),
        "volume": ("volume", "vol", "trade_volume"),
    }
    resolved = {}
    for sem, cands in aliases.items():
        for c in cands:
            if c.lower() in lowered:
                resolved[sem] = lowered[c.lower()]
                break
        if sem not in resolved:
            raise DataContractError(f"Cannot resolve OHLCV: {sem}")
    return resolved


def _validate_ohlcv(df: pd.DataFrame, cols: dict[str, str]) -> pd.Series:
    o, h, l, c, v = (df[cols[k]] for k in ("open", "high", "low", "close", "volume"))
    return (o.notna() & h.notna() & l.notna() & c.notna() & v.notna()
            & (o > 0) & (h > 0) & (l > 0) & (c > 0) & (v >= 0) & (h >= l))


# --- Vectorised building blocks ---

def _returns(close, group):
    return close / close.groupby(group).shift(1) - 1.0

def _log_returns(close, group):
    ratio = close / close.groupby(group).shift(1)
    return np.log(ratio.where(ratio > 0))

def _dollar_volume(close, volume):
    return close * volume

def _rolling(series, group, window, fn, min_pct=0.5):
    min_p = max(2, int(window * min_pct))
    return getattr(series.groupby(group).rolling(window, min_periods=min_p), fn)().droplevel(0)


# --- FRICTION / SPREAD ---

def compute_spread_proxy(H, L, C):
    return (H - L) / C.where(C > 0)

def compute_roll_spread(log_ret, group, window):
    """Roll (1984): S = 2*sqrt(max(0, -Cov(dr_t, dr_{t-1})))."""
    lr_lag = log_ret.groupby(group).shift(1)
    product = log_ret * lr_lag
    min_p = max(5, window // 2)
    mean_prod = _rolling(product, group, window, "mean", 0.5)
    mean_lr = _rolling(log_ret, group, window, "mean", 0.5)
    mean_lag = _rolling(lr_lag, group, window, "mean", 0.5)
    cov = mean_prod - mean_lr * mean_lag
    neg_cov = (-cov).where(cov < 0)
    return 2.0 * np.sqrt(neg_cov)

def compute_corwin_schultz_spread(H, L, group, window):
    """Corwin-Schultz (2012) from 2-day high/low."""
    log_hl_sq = np.log(H / L.where(L > 0)) ** 2
    h2 = H.groupby(group).rolling(2, min_periods=2).max().droplevel(0)
    l2 = L.groupby(group).rolling(2, min_periods=2).min().droplevel(0)
    gamma = np.log(h2 / l2.where(l2 > 0)) ** 2
    beta = _rolling(log_hl_sq, group, window, "mean", 0.5)
    gamma_m = _rolling(gamma.reindex(log_hl_sq.index), group, window, "mean", 0.5)
    sqrt_2b = np.sqrt((2.0 * beta).clip(lower=0))
    sqrt_b = np.sqrt(beta.clip(lower=0))
    denom = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (sqrt_2b - sqrt_b) / CS_ALPHA_DENOM - np.sqrt(gamma_m.clip(lower=0) / max(abs(denom), 1e-10))
    ea = np.exp(alpha.clip(-5, 5))
    return (2.0 * (ea - 1.0) / (1.0 + ea)).clip(lower=0.0)

def compute_bid_ask_bounce(log_ret, group, window):
    """-autocorr(r, lag=1): bid-ask bounce proxy."""
    lr_lag = log_ret.groupby(group).shift(1)
    product = log_ret * lr_lag
    min_p = max(10, window // 2)
    cov = _rolling(product, group, window, "mean", 0.5) - \
          _rolling(log_ret, group, window, "mean", 0.5) * _rolling(lr_lag, group, window, "mean", 0.5)
    var = _rolling(log_ret ** 2, group, window, "mean", 0.5) - _rolling(log_ret, group, window, "mean", 0.5) ** 2
    autocorr = cov / var.where(var > 0)
    return -autocorr


# --- LIQUIDITY ---

def compute_amihud(ret, dv, group, window):
    ratio = ret.abs() / dv.where(dv > 0)
    return _rolling(ratio, group, window, "mean")

def compute_kyle_lambda(close, volume, group, window):
    """Kyle (1985) proxy: mean(|dP|/sqrt(V))."""
    dp = close.diff().abs()
    sv = np.sqrt(volume.where(volume > 0))
    return _rolling(dp / sv.where(sv > 0), group, window, "mean")

def compute_turnover(volume, shares_out, group, window):
    if shares_out is None:
        return pd.Series(np.nan, index=volume.index)
    return _rolling(volume / shares_out.where(shares_out > 0), group, window, "mean")

def compute_dv_zscore(dv, group, window):
    ldv = np.log(dv.where(dv > 0))
    mu = _rolling(ldv, group, window, "mean")
    sig = _rolling(ldv, group, window, "std")
    return (ldv - mu) / sig.where(sig > 0)

def compute_zero_vol_frac(volume, group, window):
    return _rolling((volume == 0).astype(float), group, window, "mean")


# --- VOLATILITY ---

def compute_realized_vol(log_ret, group, window):
    return _rolling(log_ret, group, window, "std")

def compute_parkinson_vol(H, L, group, window):
    """Parkinson (1980): sqrt(mean(log(H/L)^2) / (4*ln2)). ~5x more efficient."""
    sq = np.log(H / L.where(L > 0)) ** 2
    return np.sqrt(_rolling(sq, group, window, "mean") / FOUR_LN2)

def compute_garman_klass_vol(O, H, L, C, group, window):
    """Garman-Klass (1980): uses OHLC. ~8x more efficient than cc."""
    hl = np.log(H / L.where(L > 0)) ** 2
    co = np.log(C / O.where(O > 0)) ** 2
    gk = 0.5 * hl - TWO_LN2_MINUS_1 * co
    return np.sqrt(_rolling(gk, group, window, "mean").clip(lower=0))

def compute_vol_of_vol(rvol, group, window):
    return _rolling(rvol, group, window, "std")

def compute_noise_ratio(parkinson_vol, realized_vol):
    return parkinson_vol / realized_vol.where(realized_vol > 0)

def compute_overnight_vol(O, C, group, window):
    prev_c = C.groupby(group).shift(1)
    oret = np.log(O / prev_c.where(prev_c > 0))
    return _rolling(oret, group, window, "std")

def compute_gap_open(O, C, group):
    prev = C.groupby(group).shift(1)
    return (O - prev) / prev.where(prev > 0)

def compute_hl_range(H, L):
    return np.log(H / L.where(L > 0))

def compute_vol_shock(V, group, window):
    vf = V.astype(float)
    mu = _rolling(vf, group, window, "mean")
    sig = _rolling(vf, group, window, "std")
    return (vf - mu) / sig.where(sig > 0)

def compute_momentum(close, group, window):
    lag = close.groupby(group).shift(window)
    return close / lag.where(lag > 0) - 1.0

def compute_momentum_va(mom, rvol):
    return mom / rvol.where(rvol > 0)


# --- Feature catalog ---

def _build_defs(cfg):
    defs = []
    lag = cfg.decision_lag
    if cfg.enable_friction:
        defs += [
            FeatureDef("spread_proxy", FeatureFamily.MICROSTRUCTURE, "(H-L)/C", 0, lag),
            FeatureDef(f"roll_spread_{cfg.roll_spread_window}d", FeatureFamily.MICROSTRUCTURE, "Roll(1984)", cfg.roll_spread_window, lag),
            FeatureDef(f"cs_spread_{cfg.corwin_schultz_window}d", FeatureFamily.MICROSTRUCTURE, "Corwin-Schultz(2012)", cfg.corwin_schultz_window, lag),
            FeatureDef(f"bounce_{cfg.bounce_window}d", FeatureFamily.MICROSTRUCTURE, "-autocorr(r,1)", cfg.bounce_window, lag),
        ]
    if cfg.enable_liquidity:
        defs += [
            FeatureDef(f"amihud_{cfg.amihud_window}d", FeatureFamily.LIQUIDITY, "Amihud(2002)", cfg.amihud_window, lag),
            FeatureDef(f"kyle_lambda_{cfg.kyle_window}d", FeatureFamily.LIQUIDITY, "Kyle(1985)", cfg.kyle_window, lag),
            FeatureDef(f"turnover_{cfg.turnover_window}d", FeatureFamily.LIQUIDITY, "V/SO", cfg.turnover_window, lag),
            FeatureDef(f"dv_zscore_{cfg.amihud_window}d", FeatureFamily.LIQUIDITY, "z(log(DV))", cfg.amihud_window, lag),
            FeatureDef(f"zero_vol_{cfg.amihud_window}d", FeatureFamily.LIQUIDITY, "frac(V=0)", cfg.amihud_window, lag),
        ]
    if cfg.enable_volatility:
        for w in cfg.volatility_windows:
            defs += [
                FeatureDef(f"realized_vol_{w}d", FeatureFamily.VOLATILITY, f"std(lr,{w})", w, lag),
                FeatureDef(f"parkinson_vol_{w}d", FeatureFamily.VOLATILITY, f"Parkinson({w})", w, lag),
                FeatureDef(f"gk_vol_{w}d", FeatureFamily.VOLATILITY, f"GK({w})", w, lag),
            ]
        defs += [
            FeatureDef(f"vol_of_vol_{cfg.volatility_windows[-1]}d", FeatureFamily.VOLATILITY, "std(rvol)", cfg.volatility_windows[-1]*2, lag),
            FeatureDef(f"noise_ratio_{cfg.volatility_windows[1]}d", FeatureFamily.VOLATILITY, "park/cc", cfg.volatility_windows[1], lag),
            FeatureDef(f"overnight_vol_{cfg.overnight_vol_window}d", FeatureFamily.VOLATILITY, "std(O/C_prev)", cfg.overnight_vol_window, lag),
        ]
    if cfg.enable_patterns:
        defs += [
            FeatureDef("gap_open", FeatureFamily.MICROSTRUCTURE, "(O-Cprev)/Cprev", 1, lag),
            FeatureDef("hl_range", FeatureFamily.MICROSTRUCTURE, "log(H/L)", 0, lag),
            FeatureDef("reversal_1d", FeatureFamily.REVERSAL, "-r", 1, lag),
            FeatureDef(f"vol_shock_{cfg.vol_shock_window}d", FeatureFamily.MICROSTRUCTURE, "(V-mu)/sig", cfg.vol_shock_window, lag),
        ]
    if cfg.enable_momentum:
        for w in cfg.momentum_windows:
            defs.append(FeatureDef(f"momentum_{w}d", FeatureFamily.MOMENTUM, f"C/C_-{w}-1", w, lag))
            if cfg.enable_vol_adjusted_momentum:
                defs.append(FeatureDef(f"momentum_va_{w}d", FeatureFamily.MOMENTUM, f"mom/vol", w, lag))
    return defs


# --- MAIN ---

def build_microstructure_features(
    prices: pd.DataFrame,
    *,
    config: MicrostructureConfig | None = None,
    universe_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, list[FeatureDef]]:
    cfg = config or MicrostructureConfig()
    df = prices.copy()
    date_col = resolve_date_col(df)
    sym_col = resolve_symbol_col(df)
    ohlcv = _resolve_ohlcv(df)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([sym_col, date_col]).reset_index(drop=True)
    group = df[sym_col]

    valid = _validate_ohlcv(df, ohlcv)
    if universe_mask is not None:
        valid = valid & universe_mask.reindex(df.index, fill_value=False)
    dv = _dollar_volume(df[ohlcv["close"]], df[ohlcv["volume"]])
    valid = valid & (df[ohlcv["close"]] >= cfg.min_price) & (dv >= cfg.min_dollar_volume)

    O, H, L, C = df[ohlcv["open"]], df[ohlcv["high"]], df[ohlcv["low"]], df[ohlcv["close"]]
    V = df[ohlcv["volume"]].astype(float)
    ret = _returns(C, group)
    log_ret = _log_returns(C, group)

    shares_out = None
    for cand in (cfg.shares_out_col, "shares_outstanding", "sharesoutstanding"):
        if cand in df.columns:
            shares_out = df[cand].astype(float)
            break

    out = df[[date_col, sym_col]].copy()
    out.columns = ["date", "symbol"]

    # FRICTION
    if cfg.enable_friction:
        out["spread_proxy"] = compute_spread_proxy(H, L, C).where(valid)
        out[f"roll_spread_{cfg.roll_spread_window}d"] = compute_roll_spread(log_ret, group, cfg.roll_spread_window).where(valid)
        out[f"cs_spread_{cfg.corwin_schultz_window}d"] = compute_corwin_schultz_spread(H, L, group, cfg.corwin_schultz_window).where(valid)
        out[f"bounce_{cfg.bounce_window}d"] = compute_bid_ask_bounce(log_ret, group, cfg.bounce_window).where(valid)

    # LIQUIDITY
    if cfg.enable_liquidity:
        out[f"amihud_{cfg.amihud_window}d"] = compute_amihud(ret, dv, group, cfg.amihud_window).clip(upper=cfg.clip_amihud_upper).where(valid)
        out[f"kyle_lambda_{cfg.kyle_window}d"] = compute_kyle_lambda(C, V, group, cfg.kyle_window).clip(upper=cfg.clip_kyle_upper).where(valid)
        out[f"turnover_{cfg.turnover_window}d"] = compute_turnover(V, shares_out, group, cfg.turnover_window).where(valid)
        out[f"dv_zscore_{cfg.amihud_window}d"] = compute_dv_zscore(dv, group, cfg.amihud_window).where(valid)
        out[f"zero_vol_{cfg.amihud_window}d"] = compute_zero_vol_frac(V, group, cfg.amihud_window).where(valid)

    # VOLATILITY
    rvols, pvols = {}, {}
    if cfg.enable_volatility:
        for w in cfg.volatility_windows:
            rv = compute_realized_vol(log_ret, group, w)
            pv = compute_parkinson_vol(H, L, group, w)
            gk = compute_garman_klass_vol(O, H, L, C, group, w)
            rvols[w], pvols[w] = rv, pv
            out[f"realized_vol_{w}d"] = rv.clip(upper=cfg.clip_vol_upper).where(valid)
            out[f"parkinson_vol_{w}d"] = pv.clip(upper=cfg.clip_vol_upper).where(valid)
            out[f"gk_vol_{w}d"] = gk.clip(upper=cfg.clip_vol_upper).where(valid)
        wL = cfg.volatility_windows[-1]
        if wL in rvols:
            out[f"vol_of_vol_{wL}d"] = compute_vol_of_vol(rvols[wL], group, wL).where(valid)
        wM = cfg.volatility_windows[min(1, len(cfg.volatility_windows)-1)]
        if wM in pvols and wM in rvols:
            out[f"noise_ratio_{wM}d"] = compute_noise_ratio(pvols[wM], rvols[wM]).where(valid)
        out[f"overnight_vol_{cfg.overnight_vol_window}d"] = compute_overnight_vol(O, C, group, cfg.overnight_vol_window).where(valid)

    # PATTERNS
    if cfg.enable_patterns:
        out["gap_open"] = compute_gap_open(O, C, group).where(valid)
        out["hl_range"] = compute_hl_range(H, L).where(valid)
        out["reversal_1d"] = (-ret).where(valid)
        out[f"vol_shock_{cfg.vol_shock_window}d"] = compute_vol_shock(V, group, cfg.vol_shock_window).where(valid)

    # MOMENTUM
    if cfg.enable_momentum:
        for w in cfg.momentum_windows:
            mom = compute_momentum(C, group, w)
            out[f"momentum_{w}d"] = mom.where(valid)
            if cfg.enable_vol_adjusted_momentum:
                rv_ref = rvols.get(w)
                if rv_ref is None:
                    rv_ref = rvols.get(cfg.volatility_windows[0], pd.Series(np.nan, index=mom.index))
                out[f"momentum_va_{w}d"] = compute_momentum_va(mom, rv_ref).where(valid)

    # DECISION LAG
    fcols = [c for c in out.columns if c not in ("date", "symbol")]
    if cfg.decision_lag > 0:
        for col in fcols:
            out[col] = out.groupby("symbol")[col].shift(cfg.decision_lag)

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    if cfg.enable_friction:
        for col in [c for c in fcols if "spread" in c or "bounce" in c]:
            if col in out.columns:
                out[col] = out[col].clip(upper=cfg.clip_spread_upper)

    feature_defs = _build_defs(cfg)
    cov = out[fcols].notna().mean().mean() if fcols else 0.0
    LOGGER.info("Microstructure: %d features x %d rows, cov=%.1f%%", len(fcols), len(out), cov*100)
    return out, feature_defs
