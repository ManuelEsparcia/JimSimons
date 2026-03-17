
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


# ======================================================================================
# data/borrow/borrow_cost_proxy.py
# Research-production implementation inspired by the specification provided by the user.
# ======================================================================================


# -----------------------------
# Configuration
# -----------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "config_version": "borrow_proxy_v1",
    "market_timezone": "UTC",
    "output_format": "csv",
    "winsor": {"lower": 0.01, "upper": 0.99},
    "features": {
        "adv_window": 20,
        "vol_window": 20,
        "turnover_window": 20,
        "ret_window": 1,
        "dollar_adv": True,
    },
    "stale_thresholds_days": {
        "short_interest": 30,
        "days_to_cover": 30,
        "utilization": 10,
        "direct_fee": 5,
        "htb_flag": 5,
    },
    "betas": {
        "beta0": 0.10,
        "beta_adv": 1.20,
        "beta_mcap": 0.90,
        "beta_vol": 0.80,
        "beta_si": 1.10,
        "beta_dtc": 0.80,
        "beta_htb": 1.50,
        "beta_util": 0.70,
    },
    "piecewise_mapping": {
        "c1": -0.50,
        "c2": 0.70,
        "c3": 1.70,
        "gamma": 0.55,
        "B_min": 0.005,
        "B_med": 0.050,
        "B_hard": 0.250,
        "B_max": 3.000,
        "B_max_global": 5.000,
    },
    "thresholds": {
        "theta_B": 0.25,
        "theta_alpha_htb": 0.20,
        "tau1": 0.02,
        "tau2": 0.10,
        "tau3": 0.40,
        "a1": 0.75,
        "jump_abs_delta": 0.50,
        "jump_rel_ratio": 3.0,
    },
    "hysteresis": {
        "k_down_hard_to_medium": 3,
        "k_down_medium_to_easy": 5,
    },
    "smoothing": {
        "lambda": 0.50,
        "bypass_on_direct_or_htb": True,
    },
    "fallback_policy": {
        "micro_low_liq_floor_tier": "medium",
        "insufficient_data_tier": "blocked",
        "coarse_low_quality": True,
        "min_confidence_without_si": "medium",
    },
    "tiers": {
        "easy": {"max_fee": 0.02, "min_alpha": 0.75},
        "medium": {"max_fee": 0.10, "min_alpha": 0.45},
        "hard": {"max_fee": 0.40, "min_alpha": 0.15},
    },
    "position_sizing": {
        "base_short_weight": 0.03,
        "eta_fee": 2.0,
        "tier_caps": {"easy": 1.00, "medium": 0.60, "hard": 0.25, "blocked": 0.00},
    },
}

TIER_TO_CODE = {"easy": 0, "medium": 1, "hard": 2, "blocked": 3}
SIZE_BUCKETS = ["mega", "large", "mid", "small", "micro"]
PRESSURE_BUCKETS = ["none", "low", "medium", "high", "extreme"]


@dataclass
class BorrowProxyResult:
    daily: pd.DataFrame
    proposals: pd.DataFrame
    coverage: pd.DataFrame
    summary: Dict[str, Any]
    manifest: Dict[str, Any]


# -----------------------------
# Generic helpers
# -----------------------------
def _utc_now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()


def _deep_merge(base: Dict[str, Any], extra: Mapping[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(base))
    for k, v in extra.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(dict(out[k]), v)
        else:
            out[k] = v
    return out


def load_config(config_path: Optional[str | Path]) -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG
    if config_path is None:
        return json.loads(json.dumps(cfg))
    path = Path(config_path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        user_cfg = json.loads(text)
    else:
        try:
            import yaml  # type: ignore

            user_cfg = yaml.safe_load(text)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("YAML config requested but PyYAML is not installed.") from exc
    return _deep_merge(cfg, user_cfg or {})


def _require_columns(df: pd.DataFrame, required: Iterable[str], where: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing required columns: {missing}")


def _canonicalize_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    as_str = s.astype(str).str.strip().str.lower()
    return as_str.isin(["1", "true", "t", "yes", "y"])


def _canonicalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "ticker": "symbol",
        "dt": "date",
        "close_adj": "price_adj",
        "adj_close": "price_adj",
        "adjclose": "price_adj",
        "close": "price_close",
        "volume_shares": "volume",
        "dollar_volume": "dollar_volume",
        "market_cap": "mcap",
        "marketcap": "mcap",
        "short_interest_ratio": "short_interest",
        "si": "short_interest",
        "days_to_cover_ratio": "days_to_cover",
        "dtc": "days_to_cover",
        "util": "utilization",
        "borrow_fee": "direct_fee_annual",
        "borrow_fee_annualized": "direct_fee_annual",
        "inventory_htb_flag": "explicit_htb_flag",
        "htb": "explicit_htb_flag",
        "hard_to_borrow": "explicit_htb_flag",
        "short_interest_asof": "short_interest_asof_date",
        "days_to_cover_asof": "days_to_cover_asof_date",
        "utilization_asof": "utilization_asof_date",
        "direct_fee_asof": "direct_fee_asof_date",
        "htb_asof": "htb_asof_date",
    }
    ren = {c: aliases[c] for c in df.columns if c in aliases}
    out = df.rename(columns=ren).copy()

    _require_columns(out, ["symbol", "date"], "borrow_cost_proxy input")
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()

    numeric_cols = [
        "price_close",
        "price_adj",
        "volume",
        "dollar_volume",
        "mcap",
        "float_shares",
        "short_interest",
        "days_to_cover",
        "utilization",
        "direct_fee_annual",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    bool_cols = ["explicit_htb_flag", "no_locate_flag", "hard_to_borrow_direct"]
    for col in bool_cols:
        if col in out.columns:
            out[col] = _canonicalize_bool(out[col])

    asof_cols = [
        "short_interest_asof_date",
        "days_to_cover_asof_date",
        "utilization_asof_date",
        "direct_fee_asof_date",
        "htb_asof_date",
    ]
    for col in asof_cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()

    if "price_adj" not in out.columns and "price_close" in out.columns:
        out["price_adj"] = out["price_close"]
    if "dollar_volume" not in out.columns and {"price_close", "volume"}.issubset(out.columns):
        out["dollar_volume"] = out["price_close"] * out["volume"]

    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def _winsorize_by_date(s: pd.Series, date_index: pd.Series, lower: float, upper: float) -> pd.Series:
    out = s.copy()
    for d, idx in date_index.groupby(date_index).groups.items():
        block = s.loc[idx]
        valid = block.dropna()
        if valid.empty:
            continue
        lo = valid.quantile(lower)
        hi = valid.quantile(upper)
        out.loc[idx] = block.clip(lower=lo, upper=hi)
    return out


def _robust_zscore_by_date(s: pd.Series, date_index: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=s.index, dtype=float)
    for _, idx in date_index.groupby(date_index).groups.items():
        block = s.loc[idx]
        valid = block.dropna()
        if valid.empty:
            continue
        med = float(valid.median())
        mad = float(np.median(np.abs(valid - med)))
        scale = 1.4826 * mad
        if scale <= 1e-12:
            std = float(valid.std(ddof=0))
            scale = std if std > 1e-12 else 1.0
        out.loc[idx] = (block - med) / scale
    return out


def _bucket_from_rank(x: pd.Series, labels: List[str]) -> pd.Series:
    valid = x.dropna()
    out = pd.Series(index=x.index, dtype="object")
    if valid.empty:
        return out
    ranks = valid.rank(method="average", pct=True)
    cuts = np.linspace(0, 1, len(labels) + 1)
    assigned = pd.cut(ranks, bins=cuts, labels=labels, include_lowest=True)
    out.loc[assigned.index] = assigned.astype(str)
    return out


def _coalesce(*vals: Any) -> Any:
    for v in vals:
        if pd.notna(v):
            return v
    return np.nan


def _enum_tier_from_fee_alpha_htb(
    fee: float,
    alpha: float,
    htb: bool,
    no_locate: bool,
    cfg: Mapping[str, Any],
) -> str:
    th = cfg["thresholds"]
    tiers = cfg["tiers"]
    if no_locate or fee >= th["tau3"]:
        return "blocked"
    if htb or alpha <= th["theta_alpha_htb"] or fee >= th["tau2"]:
        return "hard"
    if fee >= th["tau1"] or alpha < tiers["easy"]["min_alpha"]:
        return "medium"
    return "easy"


def max_short_weight_for_tier(
    tier: str,
    *,
    alpha: Optional[float] = None,
    fee_annual: Optional[float] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> float:
    cfg = DEFAULT_CONFIG if config is None else config
    base = float(cfg["position_sizing"]["base_short_weight"])
    caps = cfg["position_sizing"]["tier_caps"]
    eta = float(cfg["position_sizing"]["eta_fee"])
    tier_cap = float(caps.get(tier, 0.0))
    a = 1.0 if alpha is None or pd.isna(alpha) else float(alpha)
    f = 0.0 if fee_annual is None or pd.isna(fee_annual) else float(fee_annual)
    return float(base * tier_cap * max(0.0, a) * math.exp(-eta * max(0.0, f)))


def apply_locate_sizing(
    df: pd.DataFrame,
    *,
    desired_weight_col: str = "target_short_weight",
    out_col: str = "target_short_weight_capped",
    tier_col: str = "borrow_tier",
    alpha_col: str = "borrow_availability_score",
    fee_col: str = "borrow_fee_annual",
    config: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    cfg = DEFAULT_CONFIG if config is None else config
    out = df.copy()
    capped = []
    for _, row in out.iterrows():
        cap = max_short_weight_for_tier(
            str(row.get(tier_col, "blocked")),
            alpha=row.get(alpha_col),
            fee_annual=row.get(fee_col),
            config=cfg,
        )
        desired = float(row.get(desired_weight_col, 0.0))
        capped.append(min(desired, cap))
    out[out_col] = capped
    return out


def compute_net_short_return(
    gross_short_return: float | pd.Series,
    borrow_fee_annual: float | pd.Series,
    holding_days: int = 1,
) -> float | pd.Series:
    daily_cost = np.asarray(borrow_fee_annual, dtype=float) / 365.0
    return np.asarray(gross_short_return, dtype=float) - holding_days * daily_cost


# -----------------------------
# Core feature engineering
# -----------------------------
def build_features(df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    cfg = config["features"]
    adv_w = int(cfg["adv_window"])
    vol_w = int(cfg["vol_window"])
    to_w = int(cfg["turnover_window"])

    out = df.copy()
    g = out.groupby("symbol", group_keys=False)

    if "dollar_volume" not in out.columns and {"price_close", "volume"}.issubset(out.columns):
        out["dollar_volume"] = out["price_close"] * out["volume"]

    if "dollar_volume" in out.columns:
        out["adv20"] = g["dollar_volume"].transform(lambda s: s.rolling(adv_w, min_periods=1).mean())
    elif "volume" in out.columns:
        out["adv20"] = g["volume"].transform(lambda s: s.rolling(adv_w, min_periods=1).mean())
    else:
        out["adv20"] = np.nan

    px_col = "price_adj" if "price_adj" in out.columns else "price_close"
    if px_col in out.columns:
        out["ret1"] = g[px_col].transform(lambda s: s.pct_change())
        out["sigma20"] = g["ret1"].transform(lambda s: s.rolling(vol_w, min_periods=5).std(ddof=0))
    else:
        out["ret1"] = np.nan
        out["sigma20"] = np.nan

    if {"volume", "float_shares"}.issubset(out.columns):
        out["turnover20"] = (
            (out["volume"] / out["float_shares"].replace(0.0, np.nan))
            .groupby(out["symbol"])
            .transform(lambda s: s.rolling(to_w, min_periods=1).mean())
        )
    elif {"volume", "mcap", "price_close"}.issubset(out.columns):
        est_float = out["mcap"] / out["price_close"].replace(0.0, np.nan)
        out["turnover20"] = (
            (out["volume"] / est_float.replace(0.0, np.nan))
            .groupby(out["symbol"])
            .transform(lambda s: s.rolling(to_w, min_periods=1).mean())
        )
    else:
        out["turnover20"] = np.nan

    # align asof-dated signals causally
    out["si_effective"] = _apply_asof_gate(out, "short_interest", "short_interest_asof_date")
    out["dtc_effective"] = _apply_asof_gate(out, "days_to_cover", "days_to_cover_asof_date")
    out["util_effective"] = _apply_asof_gate(out, "utilization", "utilization_asof_date")
    out["direct_fee_effective"] = _apply_asof_gate(out, "direct_fee_annual", "direct_fee_asof_date")
    out["htb_effective"] = _apply_asof_gate(
        out,
        "explicit_htb_flag",
        "htb_asof_date",
        default_if_no_asof=True,
        cast_bool=True,
    )

    out["stale_input_flag"] = _compute_staleness_flag(out, config)

    low, high = config["winsor"]["lower"], config["winsor"]["upper"]
    date_ix = out["date"]
    for col in ["adv20", "mcap", "sigma20", "si_effective", "dtc_effective", "util_effective"]:
        if col in out.columns:
            out[col] = _winsorize_by_date(pd.to_numeric(out[col], errors="coerce"), date_ix, low, high)

    # transformed features
    out["f_neg_log_adv"] = _robust_zscore_by_date(-np.log(out["adv20"].replace(0.0, np.nan)), date_ix)
    out["f_neg_log_mcap"] = _robust_zscore_by_date(-np.log(out["mcap"].replace(0.0, np.nan)), date_ix)
    out["f_log_vol"] = _robust_zscore_by_date(np.log1p(out["sigma20"]), date_ix)
    out["f_log_si"] = _robust_zscore_by_date(np.log1p(out["si_effective"]), date_ix)
    out["f_log_dtc"] = _robust_zscore_by_date(np.log1p(out["dtc_effective"]), date_ix)
    out["f_util"] = _robust_zscore_by_date(out["util_effective"], date_ix)

    out["size_bucket"] = _bucket_from_rank(-np.log(out["mcap"].replace(0.0, np.nan)), SIZE_BUCKETS)
    out["liquidity_bucket"] = _bucket_from_rank(np.log(out["adv20"].replace(0.0, np.nan)), ["q1", "q2", "q3", "q4", "q5"])

    pressure_signal = pd.Series(np.nan, index=out.index)
    if "si_effective" in out.columns:
        pressure_signal = pd.to_numeric(out["si_effective"], errors="coerce")
    elif "dtc_effective" in out.columns:
        pressure_signal = pd.to_numeric(out["dtc_effective"], errors="coerce")
    out["pressure_bucket"] = _bucket_from_rank(pressure_signal, PRESSURE_BUCKETS)

    return out


def _apply_asof_gate(
    df: pd.DataFrame,
    value_col: str,
    asof_col: str,
    *,
    default_if_no_asof: bool = False,
    cast_bool: bool = False,
) -> pd.Series:
    if value_col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    values = df[value_col].copy()
    if cast_bool:
        values = values.fillna(False).astype(bool)
    if asof_col not in df.columns:
        return values if default_if_no_asof else pd.Series(np.nan, index=df.index)

    asof = pd.to_datetime(df[asof_col], errors="coerce")
    date = pd.to_datetime(df["date"], errors="coerce")
    ok = asof.notna() & (asof <= date)
    if default_if_no_asof:
        ok = ok | asof.isna()
    out = values.where(ok)
    if cast_bool:
        out = out.fillna(False).astype(bool)
    return out


def _compute_staleness_flag(df: pd.DataFrame, config: Mapping[str, Any]) -> pd.Series:
    th = config["stale_thresholds_days"]
    date = pd.to_datetime(df["date"], errors="coerce")
    stale = pd.Series(False, index=df.index)

    mapping = {
        "short_interest_asof_date": th["short_interest"],
        "days_to_cover_asof_date": th["days_to_cover"],
        "utilization_asof_date": th["utilization"],
        "direct_fee_asof_date": th["direct_fee"],
        "htb_asof_date": th["htb_flag"],
    }
    for col, max_days in mapping.items():
        if col not in df.columns:
            continue
        age = (date - pd.to_datetime(df[col], errors="coerce")).dt.days
        stale = stale | (age > max_days)
    return stale.astype(bool)


# -----------------------------
# Scoring and mapping
# -----------------------------
def compute_stress_score(df: pd.DataFrame, config: Mapping[str, Any]) -> pd.Series:
    b = config["betas"]
    x = (
        float(b["beta0"])
        + float(b["beta_adv"]) * df["f_neg_log_adv"].fillna(0.0)
        + float(b["beta_mcap"]) * df["f_neg_log_mcap"].fillna(0.0)
        + float(b["beta_vol"]) * df["f_log_vol"].fillna(0.0)
        + float(b["beta_si"]) * df["f_log_si"].fillna(0.0)
        + float(b["beta_dtc"]) * df["f_log_dtc"].fillna(0.0)
        + float(b["beta_util"]) * df["f_util"].fillna(0.0)
        + float(b["beta_htb"]) * df["htb_effective"].fillna(False).astype(float)
    )
    return x.astype(float)


def score_to_availability(score: pd.Series) -> pd.Series:
    arr = np.asarray(score, dtype=float)
    return pd.Series(1.0 / (1.0 + np.exp(arr)), index=score.index)


def score_to_fee(score: pd.Series, config: Mapping[str, Any]) -> pd.Series:
    m = config["piecewise_mapping"]
    c1, c2, c3 = float(m["c1"]), float(m["c2"]), float(m["c3"])
    bmin, bmed, bhard, bmax = float(m["B_min"]), float(m["B_med"]), float(m["B_hard"]), float(m["B_max"])
    gamma = float(m["gamma"])

    out = []
    for z in score.astype(float):
        if z <= c1:
            fee = bmin
        elif z <= c2:
            fee = bmin + (bmed - bmin) * (z - c1) / (c2 - c1)
        elif z <= c3:
            fee = bmed + (bhard - bmed) * (z - c2) / (c3 - c2)
        else:
            fee = min(bmax, bhard * math.exp(gamma * (z - c3)))
        out.append(fee)
    out = np.asarray(out, dtype=float)
    out = np.clip(out, 0.0, float(m["B_max_global"]))
    return pd.Series(out, index=score.index)


# -----------------------------
# Source proposals and aggregation
# -----------------------------
def build_source_proposals(df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    out = []
    base_score = compute_stress_score(df, config)
    base_alpha = score_to_availability(base_score)
    base_fee = score_to_fee(base_score, config)

    for idx, row in df.iterrows():
        symbol = row["symbol"]
        date = row["date"]
        proposals = []

        direct_fee = row.get("direct_fee_effective", np.nan)
        if pd.notna(direct_fee):
            proposals.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "proposal_source": "direct_lending_feed",
                    "proposed_fee_annual": float(max(0.0, direct_fee)),
                    "proposed_alpha": float(max(0.0, min(1.0, 1.0 - min(1.0, float(direct_fee) / 1.0)))),
                    "proxy_quality": "high",
                    "priority": 5,
                    "valid": True,
                }
            )

        htb = bool(row.get("htb_effective", False) or row.get("hard_to_borrow_direct", False))
        no_locate = bool(row.get("no_locate_flag", False))
        if htb or no_locate:
            forced_fee = max(float(config["thresholds"]["tau2"]), base_fee.iloc[idx])
            if no_locate:
                forced_fee = max(forced_fee, float(config["thresholds"]["tau3"]))
            proposals.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "proposal_source": "explicit_htb_flag",
                    "proposed_fee_annual": float(forced_fee),
                    "proposed_alpha": float(0.05 if no_locate else 0.15),
                    "proxy_quality": "high" if no_locate else "medium",
                    "priority": 4,
                    "valid": True,
                }
            )

        si_ok = pd.notna(row.get("si_effective", np.nan)) or pd.notna(row.get("dtc_effective", np.nan))
        if si_ok:
            proposals.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "proposal_source": "short_interest_derived",
                    "proposed_fee_annual": float(base_fee.iloc[idx]),
                    "proposed_alpha": float(base_alpha.iloc[idx]),
                    "proxy_quality": "medium" if not row.get("stale_input_flag", False) else "low",
                    "priority": 3,
                    "valid": True,
                }
            )

        liq_ok = pd.notna(row.get("adv20", np.nan)) and pd.notna(row.get("sigma20", np.nan))
        if liq_ok:
            proposals.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "proposal_source": "liquidity_volatility_proxy",
                    "proposed_fee_annual": float(base_fee.iloc[idx]),
                    "proposed_alpha": float(base_alpha.iloc[idx]),
                    "proxy_quality": "medium" if pd.notna(row.get("mcap", np.nan)) else "low",
                    "priority": 2,
                    "valid": True,
                }
            )

        coarse = _coarse_fallback(row, config)
        proposals.append(
            {
                "symbol": symbol,
                "date": date,
                "proposal_source": coarse["source"],
                "proposed_fee_annual": coarse["fee"],
                "proposed_alpha": coarse["alpha"],
                "proxy_quality": coarse["quality"],
                "priority": 1,
                "valid": True,
            }
        )
        out.extend(proposals)

    props = pd.DataFrame(out)
    props["symbol"] = props["symbol"].astype(str)
    props["date"] = pd.to_datetime(props["date"]).dt.normalize()
    return props.sort_values(["symbol", "date", "priority"], ascending=[True, True, False]).reset_index(drop=True)


def _coarse_fallback(row: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = config["fallback_policy"]
    tau1 = float(config["thresholds"]["tau1"])
    tau2 = float(config["thresholds"]["tau2"])
    tau3 = float(config["thresholds"]["tau3"])

    adv = row.get("adv20", np.nan)
    mcap = row.get("mcap", np.nan)
    sigma = row.get("sigma20", np.nan)

    if pd.isna(adv) and pd.isna(mcap):
        return {
            "source": "insufficient_data",
            "fee": tau3,
            "alpha": 0.0,
            "quality": "low",
        }

    micro = False
    illiquid = False
    volatile = False
    if pd.notna(mcap):
        micro = float(mcap) < 50_000_000
    if pd.notna(adv):
        illiquid = float(adv) < 250_000
    if pd.notna(sigma):
        volatile = float(sigma) > 0.06

    if micro and illiquid and volatile:
        floor_tier = cfg["micro_low_liq_floor_tier"]
        if floor_tier == "medium":
            return {"source": "coarse_fallback", "fee": tau1 * 1.25, "alpha": 0.45, "quality": "low"}
        return {"source": "coarse_fallback", "fee": tau2 * 1.1, "alpha": 0.20, "quality": "low"}

    if micro or illiquid:
        return {"source": "coarse_fallback", "fee": tau1 * 1.05, "alpha": 0.55, "quality": "low"}

    return {"source": "coarse_fallback", "fee": 0.010, "alpha": 0.85, "quality": "low"}


def aggregate_proposals(proposals: pd.DataFrame, raw_df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    thresholds = config["thresholds"]

    for (symbol, date), grp in proposals.groupby(["symbol", "date"], sort=True):
        grp = grp.sort_values(["priority", "proposed_fee_annual"], ascending=[False, False]).reset_index(drop=True)
        chosen = grp.iloc[0].copy()

        # conservative aggregation: lower-priority sources cannot make fee more benign
        fee = float(grp["proposed_fee_annual"].max())
        alpha = float(grp["proposed_alpha"].min())
        source = str(chosen["proposal_source"])
        quality = str(chosen["proxy_quality"])

        raw_row = raw_df[(raw_df["symbol"] == symbol) & (raw_df["date"] == date)].iloc[0]
        htb_flag = bool(raw_row.get("htb_effective", False) or raw_row.get("hard_to_borrow_direct", False))
        no_locate = bool(raw_row.get("no_locate_flag", False))
        stale_flag = bool(raw_row.get("stale_input_flag", False))

        if stale_flag and quality == "high" and source != "direct_lending_feed":
            quality = "medium"
        if source in {"coarse_fallback", "insufficient_data"}:
            quality = "low"

        tier = _enum_tier_from_fee_alpha_htb(fee, alpha, htb_flag, no_locate, config)
        fallback_flag = source in {"coarse_fallback", "insufficient_data"}

        rows.append(
            {
                "symbol": symbol,
                "date": pd.Timestamp(date),
                "borrow_fee_annual_raw": fee,
                "borrow_availability_score_raw": alpha,
                "htb_flag": bool(
                    htb_flag or fee >= float(thresholds["theta_B"]) or alpha <= float(thresholds["theta_alpha_htb"])
                ),
                "borrow_tier_raw": tier,
                "proxy_quality": quality,
                "proxy_source": source,
                "fallback_flag": bool(fallback_flag),
                "stale_input_flag": stale_flag,
                "no_locate_flag": no_locate,
            }
        )

    return pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)


# -----------------------------
# Smoothing, jumps, hysteresis
# -----------------------------
def apply_temporal_controls(df: pd.DataFrame, raw_df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()
    lam = float(config["smoothing"]["lambda"])
    bypass = bool(config["smoothing"]["bypass_on_direct_or_htb"])
    jump_abs = float(config["thresholds"]["jump_abs_delta"])
    jump_rel = float(config["thresholds"]["jump_rel_ratio"])
    k_hm = int(config["hysteresis"]["k_down_hard_to_medium"])
    k_me = int(config["hysteresis"]["k_down_medium_to_easy"])

    sm_fees = []
    sm_alpha = []
    jump_flags = []
    final_tiers = []

    state: Dict[str, Dict[str, Any]] = {}

    for _, row in out.iterrows():
        symbol = row["symbol"]
        fee_raw = float(row["borrow_fee_annual_raw"])
        alpha_raw = float(row["borrow_availability_score_raw"])
        tier_raw = str(row["borrow_tier_raw"])
        source = str(row["proxy_source"])
        htb_flag = bool(row["htb_flag"])
        no_locate = bool(row.get("no_locate_flag", False))

        prev = state.get(symbol)
        if prev is None:
            fee_sm = fee_raw
            alpha_sm = alpha_raw
            jump = False
            tier_final = tier_raw
            down_count = 0
        else:
            bypass_now = bypass and (source == "direct_lending_feed" or htb_flag or no_locate)
            if bypass_now:
                fee_sm = fee_raw
                alpha_sm = alpha_raw
            else:
                fee_sm = lam * fee_raw + (1.0 - lam) * prev["fee_sm"]
                alpha_sm = lam * alpha_raw + (1.0 - lam) * prev["alpha_sm"]

            jump = abs(fee_sm - prev["fee_sm"]) >= jump_abs or (
                min(prev["fee_sm"], fee_sm) > 0 and max(prev["fee_sm"], fee_sm) / min(prev["fee_sm"], fee_sm) >= jump_rel
            )

            tier_from_sm = _enum_tier_from_fee_alpha_htb(fee_sm, alpha_sm, htb_flag, no_locate, config)
            tier_final, down_count = _apply_hysteresis(prev["tier_final"], tier_from_sm, prev["down_count"], k_hm, k_me)

        state[symbol] = {
            "fee_sm": fee_sm,
            "alpha_sm": alpha_sm,
            "tier_final": tier_final,
            "down_count": down_count,
        }
        sm_fees.append(float(fee_sm))
        sm_alpha.append(float(alpha_sm))
        jump_flags.append(bool(jump))
        final_tiers.append(tier_final)

    out["borrow_fee_annual"] = np.clip(np.asarray(sm_fees, dtype=float), 0.0, float(config["piecewise_mapping"]["B_max_global"]))
    out["borrow_fee_daily"] = out["borrow_fee_annual"] / 365.0
    out["borrow_availability_score"] = np.clip(np.asarray(sm_alpha, dtype=float), 0.0, 1.0)
    out["borrow_tier"] = final_tiers
    out["jump_flag"] = jump_flags
    out["stress_score"] = np.nan

    score_map = raw_df[["symbol", "date", "stress_score"]].drop_duplicates()
    out = out.merge(score_map, on=["symbol", "date"], how="left", suffixes=("", "_score"))
    out["stress_score"] = out["stress_score"].fillna(out.get("stress_score_score"))
    if "stress_score_score" in out.columns:
        out = out.drop(columns=["stress_score_score"])

    blocked_high_quality_ok = out["borrow_tier"].eq("blocked") & out["proxy_quality"].eq("high") & out["proxy_source"].eq("direct_lending_feed")
    out.loc[out["borrow_tier"].eq("blocked") & ~blocked_high_quality_ok, "proxy_quality"] = out.loc[
        out["borrow_tier"].eq("blocked") & ~blocked_high_quality_ok, "proxy_quality"
    ].replace({"high": "medium"})

    return out


def _apply_hysteresis(prev_tier: str, candidate_tier: str, prev_down_count: int, k_hm: int, k_me: int) -> Tuple[str, int]:
    prev_code = TIER_TO_CODE[prev_tier]
    cand_code = TIER_TO_CODE[candidate_tier]

    # deterioration immediate
    if cand_code > prev_code:
        return candidate_tier, 0

    # no change
    if cand_code == prev_code:
        return prev_tier, 0

    # improvement requires persistence
    down_count = prev_down_count + 1
    if prev_tier == "hard" and candidate_tier in {"medium", "easy"}:
        if down_count >= k_hm:
            return ("medium" if candidate_tier == "easy" else candidate_tier), 0
        return prev_tier, down_count

    if prev_tier == "medium" and candidate_tier == "easy":
        if down_count >= k_me:
            return candidate_tier, 0
        return prev_tier, down_count

    if prev_tier == "blocked":
        # keep blocked unless explicit persistence enough; be conservative
        if down_count >= max(k_hm, 3):
            return candidate_tier, 0
        return prev_tier, down_count

    return candidate_tier, 0


# -----------------------------
# Summary and IO
# -----------------------------
def build_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, grp in df.groupby("date", sort=True):
        rows.append(
            {
                "date": pd.Timestamp(date),
                "n_symbols": int(len(grp)),
                "pct_fallback": float(grp["fallback_flag"].mean()) if len(grp) else np.nan,
                "pct_stale": float(grp["stale_input_flag"].mean()) if len(grp) else np.nan,
                "pct_htb": float(grp["htb_flag"].mean()) if len(grp) else np.nan,
                "pct_blocked": float((grp["borrow_tier"] == "blocked").mean()) if len(grp) else np.nan,
                "pct_hard_or_blocked": float(grp["borrow_tier"].isin(["hard", "blocked"]).mean()) if len(grp) else np.nan,
                "mean_fee": float(grp["borrow_fee_annual"].mean()) if len(grp) else np.nan,
                "median_fee": float(grp["borrow_fee_annual"].median()) if len(grp) else np.nan,
                "mean_alpha": float(grp["borrow_availability_score"].mean()) if len(grp) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_summary(df: pd.DataFrame, coverage: pd.DataFrame, config: Mapping[str, Any]) -> Dict[str, Any]:
    tier_counts = df["borrow_tier"].value_counts(dropna=False).to_dict()
    src_counts = df["proxy_source"].value_counts(dropna=False).to_dict()
    q_counts = df["proxy_quality"].value_counts(dropna=False).to_dict()

    return {
        "config_version": config["config_version"],
        "generated_at_utc": _utc_now_iso(),
        "n_rows": int(len(df)),
        "n_symbols": int(df["symbol"].nunique()),
        "date_min": None if df.empty else str(pd.Timestamp(df["date"].min()).date()),
        "date_max": None if df.empty else str(pd.Timestamp(df["date"].max()).date()),
        "tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
        "proxy_source_counts": {str(k): int(v) for k, v in src_counts.items()},
        "proxy_quality_counts": {str(k): int(v) for k, v in q_counts.items()},
        "mean_borrow_fee_annual": None if df.empty else float(df["borrow_fee_annual"].mean()),
        "median_borrow_fee_annual": None if df.empty else float(df["borrow_fee_annual"].median()),
        "mean_borrow_availability_score": None if df.empty else float(df["borrow_availability_score"].mean()),
        "pct_fallback": None if df.empty else float(df["fallback_flag"].mean()),
        "pct_stale_input": None if df.empty else float(df["stale_input_flag"].mean()),
        "pct_jump": None if df.empty else float(df["jump_flag"].mean()),
        "coverage_rows": int(len(coverage)),
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _write_table(df: pd.DataFrame, path: Path, fmt: str = "csv") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()
    if fmt == "parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Parquet output requested but no parquet engine is available. Install pyarrow or fastparquet."
            ) from exc
    df.to_csv(path.with_suffix(".csv"), index=False)
    return path.with_suffix(".csv")


def write_outputs(result: BorrowProxyResult, output_dir: str | Path, config: Mapping[str, Any]) -> Dict[str, str]:
    root = Path(output_dir)
    fmt = str(config.get("output_format", "csv")).lower()

    files = {}
    files["daily"] = str(_write_table(result.daily, root / "borrow_proxy_daily.parquet", fmt))
    files["proposals"] = str(_write_table(result.proposals, root / "borrow_proxy_proposals.parquet", fmt))
    files["coverage"] = str(_write_table(result.coverage, root / "borrow_proxy_coverage.parquet", fmt))
    summary_path = root / "borrow_proxy_summary.json"
    summary_path.write_text(json.dumps(_to_jsonable(result.summary), indent=2), encoding="utf-8")
    files["summary"] = str(summary_path)
    manifest = dict(result.manifest)
    manifest["files"] = files
    manifest_path = root / "borrow_proxy_manifest.json"
    manifest_path.write_text(json.dumps(_to_jsonable(manifest), indent=2), encoding="utf-8")
    files["manifest"] = str(manifest_path)
    return files


# -----------------------------
# Main orchestration
# -----------------------------
def run_borrow_cost_proxy(
    data: pd.DataFrame,
    *,
    config: Optional[Mapping[str, Any]] = None,
    asof_timestamp: Optional[str] = None,
) -> BorrowProxyResult:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG)) if config is None else _deep_merge(DEFAULT_CONFIG, config)
    raw = _canonicalize_schema(data)
    feats = build_features(raw, cfg)
    feats["stress_score"] = compute_stress_score(feats, cfg)

    proposals = build_source_proposals(feats, cfg)
    aggregated = aggregate_proposals(proposals, feats, cfg)
    final = apply_temporal_controls(aggregated, feats, cfg)

    final["config_version"] = cfg["config_version"]
    final["asof_timestamp"] = asof_timestamp or _utc_now_iso()
    final["borrow_tier_code"] = final["borrow_tier"].map(TIER_TO_CODE).astype(int)

    # enforce output contract
    out_cols = [
        "symbol",
        "date",
        "borrow_fee_annual",
        "borrow_fee_daily",
        "borrow_availability_score",
        "htb_flag",
        "borrow_tier",
        "proxy_quality",
        "proxy_source",
        "stress_score",
        "jump_flag",
        "stale_input_flag",
        "fallback_flag",
        "config_version",
        "asof_timestamp",
        "borrow_tier_code",
        "no_locate_flag",
    ]
    final = final[out_cols].sort_values(["date", "symbol"]).reset_index(drop=True)

    coverage = build_coverage(final)
    summary = build_summary(final, coverage, cfg)
    manifest = {
        "module": "data/borrow/borrow_cost_proxy.py",
        "config_version": cfg["config_version"],
        "generated_at_utc": _utc_now_iso(),
        "asof_timestamp": asof_timestamp or _utc_now_iso(),
        "row_count": int(len(final)),
        "proposal_count": int(len(proposals)),
        "coverage_rows": int(len(coverage)),
    }
    return BorrowProxyResult(
        daily=final,
        proposals=proposals,
        coverage=coverage,
        summary=summary,
        manifest=manifest,
    )


# backwards-friendly alias names
def build_borrow_cost_proxy(data: pd.DataFrame, *, config: Optional[Mapping[str, Any]] = None, asof_timestamp: Optional[str] = None) -> BorrowProxyResult:
    return run_borrow_cost_proxy(data, config=config, asof_timestamp=asof_timestamp)


def borrow_cost_proxy(data: pd.DataFrame, *, config: Optional[Mapping[str, Any]] = None, asof_timestamp: Optional[str] = None) -> BorrowProxyResult:
    return run_borrow_cost_proxy(data, config=config, asof_timestamp=asof_timestamp)


# -----------------------------
# CLI
# -----------------------------
def _read_input(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported input format: {p.suffix}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conservative borrow cost proxy.")
    parser.add_argument("--input-path", required=True, help="Input CSV or parquet with causal market/borrow features.")
    parser.add_argument("--output-dir", required=False, help="Directory where outputs will be materialized.")
    parser.add_argument("--config-path", required=False, help="JSON/YAML config path.")
    parser.add_argument("--asof-timestamp", required=False, help="Optional build timestamp.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config_path)
    df = _read_input(args.input_path)
    result = run_borrow_cost_proxy(df, config=cfg, asof_timestamp=args.asof_timestamp)
    if args.output_dir:
        write_outputs(result, args.output_dir, cfg)
    else:
        print(result.daily.head().to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
