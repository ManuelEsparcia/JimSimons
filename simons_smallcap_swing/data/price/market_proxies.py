from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

MODULE_NAME = "data.price.market_proxies"
CODE_VERSION = "1.0.0"
SEVERITY_ORDER = {"INFO": 1, "WARN": 2, "FAIL": 3}
CORE_PROXIES = [
    "market_ret_1d",
    "market_ret_5d",
    "market_ret_20d",
    "market_realized_vol_20d",
    "breadth_pct_up",
    "advance_decline_ratio",
    "cross_section_dispersion",
    "market_turnover_proxy",
]
EXPLORATORY_PROXIES = [
    "market_spread_proxy",
    "cross_section_corr_proxy",
    "market_ret_cap_weighted",
    "breadth_ma_distance",
    "liquidity_stress_proxy",
]


class MarketProxiesError(RuntimeError):
    pass


@dataclass(frozen=True)
class CoverageThresholds:
    n_min: int = 50
    ratio_min: float = 0.25
    warn_margin_n: int = 10
    warn_margin_ratio: float = 0.05


@dataclass(frozen=True)
class SmoothingConfig:
    enabled: bool = False
    columns: Tuple[str, ...] = ()
    ema_windows: Tuple[int, ...] = ()


@dataclass(frozen=True)
class ZScoreConfig:
    enabled: bool = False
    columns: Tuple[str, ...] = ()
    rolling_windows: Tuple[int, ...] = ()


@dataclass(frozen=True)
class ExploratoryConfig:
    enabled: Tuple[str, ...] = ()
    corr_window: int = 20
    breadth_ma_window: int = 20
    liquidity_stress_threshold: float = 0.5


@dataclass(frozen=True)
class ProxyConfig:
    return_windows: Tuple[int, ...] = (1, 5, 20)
    realized_vol_window: int = 20
    turnover_ref_window: int = 20
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    laplace_lambda: float = 1.0
    turnover_lambda: float = 1.0
    volume_log_transform: bool = False
    spread_log_transform: bool = False
    coverage: CoverageThresholds = field(default_factory=CoverageThresholds)
    exploratory: ExploratoryConfig = field(default_factory=ExploratoryConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    zscore: ZScoreConfig = field(default_factory=ZScoreConfig)
    output_dir: str = "data/price/market_proxies"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_datetime_utc(value: str) -> str:
    try:
        dt = pd.Timestamp(value)
    except Exception as exc:
        raise MarketProxiesError(f"Invalid --as-of-ts-utc: {value!r}") from exc
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")
    return dt.isoformat().replace("+00:00", "Z")


def load_config(path: str | Path) -> Tuple[ProxyConfig, Dict[str, Any], str]:
    config_path = Path(path)
    if not config_path.exists():
        raise MarketProxiesError(f"config_path does not exist: {config_path}")
    raw_text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise MarketProxiesError("PyYAML is required to read YAML configs.")
        raw_cfg = yaml.safe_load(raw_text) or {}
    elif config_path.suffix.lower() == ".json":
        raw_cfg = json.loads(raw_text)
    else:
        if yaml is not None:
            raw_cfg = yaml.safe_load(raw_text) or {}
        else:
            raw_cfg = json.loads(raw_text)
    if not isinstance(raw_cfg, Mapping):
        raise MarketProxiesError("Configuration must be a mapping/object at top level.")

    coverage_cfg = raw_cfg.get("coverage", {}) or {}
    exploratory_cfg = raw_cfg.get("exploratory", {}) or {}
    smoothing_cfg = raw_cfg.get("smoothing", {}) or {}
    zscore_cfg = raw_cfg.get("zscore", {}) or {}

    cfg = ProxyConfig(
        return_windows=tuple(int(x) for x in raw_cfg.get("return_windows", [1, 5, 20])),
        realized_vol_window=int(raw_cfg.get("realized_vol_window", 20)),
        turnover_ref_window=int(raw_cfg.get("turnover_ref_window", 20)),
        winsor_lower=float(raw_cfg.get("winsor_lower", 0.01)),
        winsor_upper=float(raw_cfg.get("winsor_upper", 0.99)),
        laplace_lambda=float(raw_cfg.get("laplace_lambda", 1.0)),
        turnover_lambda=float(raw_cfg.get("turnover_lambda", 1.0)),
        volume_log_transform=bool(raw_cfg.get("volume_log_transform", False)),
        spread_log_transform=bool(raw_cfg.get("spread_log_transform", False)),
        coverage=CoverageThresholds(
            n_min=int(coverage_cfg.get("n_min", 50)),
            ratio_min=float(coverage_cfg.get("ratio_min", 0.25)),
            warn_margin_n=int(coverage_cfg.get("warn_margin_n", 10)),
            warn_margin_ratio=float(coverage_cfg.get("warn_margin_ratio", 0.05)),
        ),
        exploratory=ExploratoryConfig(
            enabled=tuple(str(x) for x in exploratory_cfg.get("enabled", [])),
            corr_window=int(exploratory_cfg.get("corr_window", 20)),
            breadth_ma_window=int(exploratory_cfg.get("breadth_ma_window", 20)),
            liquidity_stress_threshold=float(exploratory_cfg.get("liquidity_stress_threshold", 0.5)),
        ),
        smoothing=SmoothingConfig(
            enabled=bool(smoothing_cfg.get("enabled", False)),
            columns=tuple(str(x) for x in smoothing_cfg.get("columns", [])),
            ema_windows=tuple(int(x) for x in smoothing_cfg.get("ema_windows", [])),
        ),
        zscore=ZScoreConfig(
            enabled=bool(zscore_cfg.get("enabled", False)),
            columns=tuple(str(x) for x in zscore_cfg.get("columns", [])),
            rolling_windows=tuple(int(x) for x in zscore_cfg.get("rolling_windows", [])),
        ),
        output_dir=str(raw_cfg.get("output_dir", "data/price/market_proxies")),
    )

    if cfg.winsor_lower < 0 or cfg.winsor_upper > 1 or cfg.winsor_lower >= cfg.winsor_upper:
        raise MarketProxiesError("winsor_lower/winsor_upper must satisfy 0 <= lower < upper <= 1")
    if 1 not in cfg.return_windows:
        raise MarketProxiesError("return_windows must include 1 because market_realized_vol is based on market_ret_1d")
    cfg_hash = hashlib.sha256(json.dumps(raw_cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return cfg, dict(raw_cfg), cfg_hash


def read_table(path_like: str | Path) -> pd.DataFrame:
    path = Path(path_like)
    if not path.exists():
        raise MarketProxiesError(f"Input path does not exist: {path}")
    if path.is_file():
        return read_single_file(path)
    files: List[Path] = []
    for ext in ("*.parquet", "*.pq", "*.csv", "*.csv.gz", "*.json", "*.jsonl"):
        files.extend(sorted(path.rglob(ext)))
    if not files:
        raise MarketProxiesError(f"No readable data files found under directory: {path}")
    parts = [read_single_file(p) for p in files]
    df = pd.concat(parts, ignore_index=True, sort=False)
    return df


def read_single_file(path: Path) -> pd.DataFrame:
    suffixes = tuple(s.lower() for s in path.suffixes)
    if suffixes[-1:] == (".parquet",) or suffixes[-1:] == (".pq",):
        return pd.read_parquet(path)
    if suffixes[-2:] == (".csv", ".gz") or suffixes[-1:] == (".csv",):
        return pd.read_csv(path)
    if suffixes[-1:] == (".json",):
        return pd.read_json(path)
    if suffixes[-1:] == (".jsonl",):
        return pd.read_json(path, lines=True)
    raise MarketProxiesError(f"Unsupported file format: {path}")


def ensure_parquet_support() -> None:
    try:
        import pyarrow  # noqa: F401  # type: ignore
        return
    except Exception:
        pass
    try:
        import fastparquet  # noqa: F401  # type: ignore
        return
    except Exception:
        pass
    raise MarketProxiesError(
        "Writing parquet requires 'pyarrow' or 'fastparquet'. Install one of them in your runtime."
    )


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "date", "close_adj", "ret_1d_adj"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise MarketProxiesError(f"prices_adjusted is missing required columns: {missing}")

    prices = df.copy()
    prices["symbol"] = prices["symbol"].astype(str).str.strip()
    prices["date"] = pd.to_datetime(prices["date"], utc=False, errors="coerce").dt.normalize()
    if prices["date"].isna().any():
        raise MarketProxiesError("prices_adjusted contains invalid dates")
    dup_mask = prices.duplicated(["symbol", "date"], keep=False)
    if dup_mask.any():
        examples = prices.loc[dup_mask, ["symbol", "date"]].head(10).to_dict("records")
        raise MarketProxiesError(f"Duplicate (symbol, date) rows in prices_adjusted: {examples}")

    numeric_cols = [
        "close_adj",
        "ret_1d_adj",
        "volume_adj",
        "dollar_volume",
        "market_cap",
        "spread_proxy",
        "high_adj",
        "low_adj",
        "high",
        "low",
    ]
    for col in numeric_cols:
        if col in prices.columns:
            prices[col] = pd.to_numeric(prices[col], errors="coerce")
    prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)
    return prices


ELIGIBLE_STATES = {"member", "eligible", "included", "in", "active", "current"}


def normalize_universe(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        raise MarketProxiesError("universe_history is missing required column 'date'")
    if "symbol" not in df.columns:
        if "ticker" in df.columns:
            df = df.rename(columns={"ticker": "symbol"})
        else:
            raise MarketProxiesError("universe_history is missing required column 'symbol'")

    universe = df.copy()
    universe["symbol"] = universe["symbol"].astype(str).str.strip()
    universe["date"] = pd.to_datetime(universe["date"], utc=False, errors="coerce").dt.normalize()
    if universe["date"].isna().any():
        raise MarketProxiesError("universe_history contains invalid dates")

    if "is_eligible" in universe.columns:
        eligible = universe["is_eligible"].astype("boolean")
    elif "membership_state" in universe.columns:
        eligible = universe["membership_state"].astype(str).str.lower().isin(ELIGIBLE_STATES)
    elif "is_member" in universe.columns:
        eligible = universe["is_member"].astype("boolean")
    elif "in_universe" in universe.columns:
        eligible = universe["in_universe"].astype("boolean")
    else:
        eligible = pd.Series(True, index=universe.index, dtype="boolean")

    universe["is_eligible_resolved"] = eligible.fillna(False)
    universe = universe[universe["is_eligible_resolved"]].copy()
    if universe.empty:
        raise MarketProxiesError("Universe history has zero eligible rows after eligibility resolution.")

    snapshot_col = None
    for candidate in ["universe_snapshot_id", "snapshot_id", "run_id"]:
        if candidate in universe.columns:
            snapshot_col = candidate
            break
    if snapshot_col is None:
        universe["universe_snapshot_id"] = "unknown_snapshot"
    else:
        universe["universe_snapshot_id"] = universe[snapshot_col].astype(str)

    dup_mask = universe.duplicated(["date", "symbol"], keep=False)
    if dup_mask.any():
        de = (
            universe.loc[dup_mask, ["date", "symbol", "universe_snapshot_id"]]
            .head(10)
            .to_dict("records")
        )
        raise MarketProxiesError(f"Duplicate eligible universe rows for (date, symbol): {de}")

    return universe[["date", "symbol", "universe_snapshot_id"]].sort_values(["date", "symbol"]).reset_index(drop=True)


def build_universe_panel(prices: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    panel = universe.merge(prices, on=["date", "symbol"], how="left", validate="one_to_one")
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def winsorize_series(s: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    clean = s.dropna()
    if clean.empty:
        return s.astype(float)
    if len(clean) == 1:
        return s.astype(float)
    low = clean.quantile(lower_q)
    high = clean.quantile(upper_q)
    return s.astype(float).clip(lower=low, upper=high)


def robust_mean(s: pd.Series, lower_q: float, upper_q: float) -> Tuple[float, pd.Series]:
    w = winsorize_series(s, lower_q, upper_q)
    return float(w.mean()) if w.notna().any() else np.nan, w


def robust_std(s: pd.Series, lower_q: float, upper_q: float) -> Tuple[float, pd.Series]:
    w = winsorize_series(s, lower_q, upper_q)
    if w.notna().sum() < 2:
        return np.nan, w
    return float(w.std(ddof=1)), w


def median_or_nan(s: pd.Series) -> float:
    return float(s.median()) if s.notna().any() else np.nan


def add_symbol_level_derived_columns(prices: pd.DataFrame, cfg: ProxyConfig) -> pd.DataFrame:
    df = prices.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)

    for window in sorted(set(cfg.return_windows)):
        if window == 1:
            df["ret_1d_roll"] = pd.to_numeric(df["ret_1d_adj"], errors="coerce")
            continue
        col = f"ret_{window}d_roll"
        df[col] = (
            g["ret_1d_adj"]
            .rolling(window=window, min_periods=window)
            .apply(lambda x: float(np.prod(1.0 + x) - 1.0), raw=True)
            .reset_index(level=0, drop=True)
        )

    if "dollar_volume" not in df.columns:
        if "volume_adj" in df.columns:
            df["dollar_volume"] = df["close_adj"] * df["volume_adj"]
        else:
            df["dollar_volume"] = np.nan

    if cfg.volume_log_transform:
        df["dollar_volume_for_proxy"] = np.where(df["dollar_volume"] > 0, np.log1p(df["dollar_volume"]), np.nan)
    else:
        df["dollar_volume_for_proxy"] = df["dollar_volume"]

    df["dollar_volume_ref_20d"] = (
        g["dollar_volume_for_proxy"]
        .rolling(window=cfg.turnover_ref_window, min_periods=1)
        .median()
        .reset_index(level=0, drop=True)
    )
    df["turnover_ratio_raw"] = df["dollar_volume_for_proxy"] / (df["dollar_volume_ref_20d"] + cfg.turnover_lambda)

    if "spread_proxy" not in df.columns:
        hi = None
        lo = None
        for c in ["high_adj", "high"]:
            if c in df.columns:
                hi = c
                break
        for c in ["low_adj", "low"]:
            if c in df.columns:
                lo = c
                break
        if hi is not None and lo is not None:
            df["spread_proxy"] = (df[hi] - df[lo]) / df["close_adj"]
        else:
            df["spread_proxy"] = np.nan
    if cfg.spread_log_transform:
        df["spread_proxy_for_proxy"] = np.where(df["spread_proxy"] > 0, np.log1p(df["spread_proxy"]), np.nan)
    else:
        df["spread_proxy_for_proxy"] = df["spread_proxy"]

    return df


def build_market_return_base(panel: pd.DataFrame, cfg: ProxyConfig) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for date, day in panel.groupby("date", sort=True):
        universe_snapshot_id = infer_snapshot_id(day)
        universe_count = int(day["symbol"].nunique())
        rec: Dict[str, Any] = {
            "date": date,
            "universe_snapshot_id": universe_snapshot_id,
            "universe_count": universe_count,
        }
        for window in sorted(set(cfg.return_windows)):
            src = "ret_1d_adj" if window == 1 else f"ret_{window}d_roll"
            vals = pd.to_numeric(day[src], errors="coerce")
            coverage_count = int(vals.notna().sum())
            coverage_ratio = float(coverage_count / universe_count) if universe_count else np.nan
            agg, _ = robust_mean(vals, cfg.winsor_lower, cfg.winsor_upper)
            rec[f"market_ret_{window}d"] = agg
            rec[f"coverage_count_market_ret_{window}d"] = coverage_count
            rec[f"coverage_ratio_market_ret_{window}d"] = coverage_ratio
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    vol_window = cfg.realized_vol_window
    out[f"market_realized_vol_{vol_window}d"] = (
        out["market_ret_1d"].rolling(window=vol_window, min_periods=vol_window).std(ddof=1)
    )
    out[f"coverage_count_market_realized_vol_{vol_window}d"] = (
        out["market_ret_1d"].rolling(window=vol_window, min_periods=vol_window).count().astype("float")
    )
    out[f"coverage_ratio_market_realized_vol_{vol_window}d"] = np.where(
        out[f"coverage_count_market_realized_vol_{vol_window}d"].notna(), 1.0, np.nan
    )
    return out




def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return np.nan
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return np.nan

def infer_snapshot_id(day: pd.DataFrame) -> str:
    vals = day["universe_snapshot_id"].dropna().astype(str).unique().tolist()
    if not vals:
        return "unknown_snapshot"
    if len(vals) == 1:
        return vals[0]
    vals.sort()
    return f"multi:{'|'.join(vals[:5])}{'|...' if len(vals) > 5 else ''}"


def compute_daily_proxies(panel: pd.DataFrame, market_base: pd.DataFrame, cfg: ProxyConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_map = market_base.set_index("date")
    rows: List[Dict[str, Any]] = []
    warn_rows: List[Dict[str, Any]] = []
    corr_col = None
    if "cross_section_corr_proxy" in cfg.exploratory.enabled:
        corr_col = compute_symbol_market_corr(panel, market_base, cfg.exploratory.corr_window)
        panel[corr_col] = panel[corr_col]

    for date, day in panel.groupby("date", sort=True):
        snapshot = infer_snapshot_id(day)
        universe_count = int(day["symbol"].nunique())
        rec: Dict[str, Any] = {
            "date": date,
            "universe_snapshot_id": snapshot,
            "universe_count": universe_count,
        }
        proxy_status: Dict[str, str] = {}

        # market returns / realized vol come from base table
        base_row = base_map.loc[date]
        for proxy in ["market_ret_1d", "market_ret_5d", "market_ret_20d", f"market_realized_vol_{cfg.realized_vol_window}d"]:
            rec[proxy] = base_row.get(proxy, np.nan)
            c_count = _safe_int(base_row.get(f"coverage_count_{proxy}", 0), 0)
            c_ratio = _safe_float(base_row.get(f"coverage_ratio_{proxy}", np.nan))
            rec[f"coverage_count_{proxy}"] = c_count
            rec[f"coverage_ratio_{proxy}"] = c_ratio
            sev = classify_proxy_severity(proxy, c_count, c_ratio, universe_count, cfg.coverage)
            proxy_status[proxy] = sev
            if sev != "INFO":
                warn_rows.append(make_warning_row(date, proxy, sev, universe_count, c_count, c_ratio, "coverage_or_validity"))

        # breadth
        ret_vals = pd.to_numeric(day["ret_1d_adj"], errors="coerce")
        breadth_count = int(ret_vals.notna().sum())
        breadth_ratio = float(breadth_count / universe_count) if universe_count else np.nan
        if breadth_count > 0:
            n_up = int((ret_vals > 0).sum())
            n_down = int((ret_vals < 0).sum())
            breadth_pct_up = float(n_up / breadth_count)
            adr = float((n_up + cfg.laplace_lambda) / (n_down + cfg.laplace_lambda))
        else:
            n_up = 0
            n_down = 0
            breadth_pct_up = np.nan
            adr = np.nan
        rec["breadth_pct_up"] = breadth_pct_up
        rec["advance_decline_ratio"] = adr
        rec["coverage_count_breadth_pct_up"] = breadth_count
        rec["coverage_ratio_breadth_pct_up"] = breadth_ratio
        rec["coverage_count_advance_decline_ratio"] = breadth_count
        rec["coverage_ratio_advance_decline_ratio"] = breadth_ratio
        for proxy in ["breadth_pct_up", "advance_decline_ratio"]:
            sev = classify_proxy_severity(proxy, breadth_count, breadth_ratio, universe_count, cfg.coverage)
            if proxy == "breadth_pct_up" and pd.notna(breadth_pct_up) and not (0.0 <= breadth_pct_up <= 1.0):
                sev = "FAIL"
                warn_rows.append(make_warning_row(date, proxy, sev, universe_count, breadth_count, breadth_ratio, "breadth_out_of_range"))
            proxy_status[proxy] = sev
            if sev != "INFO" and not any(w["proxy_name"] == proxy and w["date"] == date for w in warn_rows):
                warn_rows.append(make_warning_row(date, proxy, sev, universe_count, breadth_count, breadth_ratio, "coverage_or_validity"))

        # dispersion
        disp_value, wins_ret = robust_std(ret_vals, cfg.winsor_lower, cfg.winsor_upper)
        disp_count = int(wins_ret.notna().sum())
        disp_ratio = float(disp_count / universe_count) if universe_count else np.nan
        rec["cross_section_dispersion"] = disp_value
        rec["coverage_count_cross_section_dispersion"] = disp_count
        rec["coverage_ratio_cross_section_dispersion"] = disp_ratio
        sev = classify_proxy_severity("cross_section_dispersion", disp_count, disp_ratio, universe_count, cfg.coverage)
        proxy_status["cross_section_dispersion"] = sev
        if sev != "INFO":
            warn_rows.append(make_warning_row(date, "cross_section_dispersion", sev, universe_count, disp_count, disp_ratio, "coverage_or_validity"))

        # turnover proxy
        turnover_vals = pd.to_numeric(day["turnover_ratio_raw"], errors="coerce")
        turnover_w = winsorize_series(turnover_vals, cfg.winsor_lower, cfg.winsor_upper)
        turnover_count = int(turnover_w.notna().sum())
        turnover_ratio = float(turnover_count / universe_count) if universe_count else np.nan
        rec["market_turnover_proxy"] = median_or_nan(turnover_w)
        rec["coverage_count_market_turnover_proxy"] = turnover_count
        rec["coverage_ratio_market_turnover_proxy"] = turnover_ratio
        sev = classify_proxy_severity("market_turnover_proxy", turnover_count, turnover_ratio, universe_count, cfg.coverage)
        proxy_status["market_turnover_proxy"] = sev
        if sev != "INFO":
            warn_rows.append(make_warning_row(date, "market_turnover_proxy", sev, universe_count, turnover_count, turnover_ratio, "coverage_or_validity"))

        # Family coverage contract
        rec["coverage_count_ret"] = _safe_int(base_row.get("coverage_count_market_ret_1d", 0), 0)
        rec["coverage_ratio_ret"] = _safe_float(base_row.get("coverage_ratio_market_ret_1d", np.nan))
        rec["coverage_count_breadth"] = breadth_count
        rec["coverage_ratio_breadth"] = breadth_ratio
        rec["coverage_count_liq"] = turnover_count
        rec["coverage_ratio_liq"] = turnover_ratio

        # exploratory
        enabled = set(cfg.exploratory.enabled)
        if "market_spread_proxy" in enabled:
            spread_vals = pd.to_numeric(day["spread_proxy_for_proxy"], errors="coerce")
            spread_w = winsorize_series(spread_vals, cfg.winsor_lower, cfg.winsor_upper)
            c_count = int(spread_w.notna().sum())
            c_ratio = float(c_count / universe_count) if universe_count else np.nan
            rec["market_spread_proxy"] = median_or_nan(spread_w)
            rec["coverage_count_market_spread_proxy"] = c_count
            rec["coverage_ratio_market_spread_proxy"] = c_ratio
            sev = classify_proxy_severity("market_spread_proxy", c_count, c_ratio, universe_count, cfg.coverage)
            proxy_status["market_spread_proxy"] = sev
            if sev != "INFO":
                warn_rows.append(make_warning_row(date, "market_spread_proxy", sev, universe_count, c_count, c_ratio, "coverage_or_validity"))
        else:
            warn_rows.append(make_warning_row(date, "market_spread_proxy", "INFO", universe_count, 0, np.nan, "exploratory_disabled"))

        if "cross_section_corr_proxy" in enabled:
            corr_vals = pd.to_numeric(day[corr_col], errors="coerce") if corr_col is not None else pd.Series(np.nan, index=day.index)
            corr_w = winsorize_series(corr_vals, cfg.winsor_lower, cfg.winsor_upper)
            c_count = int(corr_w.notna().sum())
            c_ratio = float(c_count / universe_count) if universe_count else np.nan
            rec["cross_section_corr_proxy"] = median_or_nan(corr_w)
            rec["coverage_count_cross_section_corr_proxy"] = c_count
            rec["coverage_ratio_cross_section_corr_proxy"] = c_ratio
            sev = classify_proxy_severity("cross_section_corr_proxy", c_count, c_ratio, universe_count, cfg.coverage)
            proxy_status["cross_section_corr_proxy"] = sev
            if sev != "INFO":
                warn_rows.append(make_warning_row(date, "cross_section_corr_proxy", sev, universe_count, c_count, c_ratio, "coverage_or_validity"))
        else:
            warn_rows.append(make_warning_row(date, "cross_section_corr_proxy", "INFO", universe_count, 0, np.nan, "exploratory_disabled"))

        if "market_ret_cap_weighted" in enabled:
            if "market_cap" in day.columns:
                cap = pd.to_numeric(day["market_cap"], errors="coerce")
                ret = pd.to_numeric(day["ret_1d_adj"], errors="coerce")
                valid = cap.notna() & ret.notna() & (cap > 0)
                c_count = int(valid.sum())
                c_ratio = float(c_count / universe_count) if universe_count else np.nan
                if c_count:
                    w = cap[valid] / cap[valid].sum()
                    value = float(np.sum(w * ret[valid]))
                else:
                    value = np.nan
            else:
                c_count = 0
                c_ratio = np.nan
                value = np.nan
            rec["market_ret_cap_weighted"] = value
            rec["coverage_count_market_ret_cap_weighted"] = c_count
            rec["coverage_ratio_market_ret_cap_weighted"] = c_ratio
            sev = classify_proxy_severity("market_ret_cap_weighted", c_count, c_ratio, universe_count, cfg.coverage)
            proxy_status["market_ret_cap_weighted"] = sev
            if sev != "INFO":
                warn_rows.append(make_warning_row(date, "market_ret_cap_weighted", sev, universe_count, c_count, c_ratio, "coverage_or_validity"))
        else:
            warn_rows.append(make_warning_row(date, "market_ret_cap_weighted", "INFO", universe_count, 0, np.nan, "exploratory_disabled"))

        if "breadth_ma_distance" in enabled:
            rec["breadth_ma_distance"] = np.nan  # populated after dataframe build
            rec["coverage_count_breadth_ma_distance"] = breadth_count
            rec["coverage_ratio_breadth_ma_distance"] = breadth_ratio
            proxy_status["breadth_ma_distance"] = classify_proxy_severity("breadth_ma_distance", breadth_count, breadth_ratio, universe_count, cfg.coverage)
        else:
            warn_rows.append(make_warning_row(date, "breadth_ma_distance", "INFO", universe_count, 0, np.nan, "exploratory_disabled"))

        if "liquidity_stress_proxy" in enabled:
            vals = pd.to_numeric(day["turnover_ratio_raw"], errors="coerce")
            valid = vals.notna()
            c_count = int(valid.sum())
            c_ratio = float(c_count / universe_count) if universe_count else np.nan
            if c_count:
                value = float((vals[valid] < cfg.exploratory.liquidity_stress_threshold).mean())
            else:
                value = np.nan
            rec["liquidity_stress_proxy"] = value
            rec["coverage_count_liquidity_stress_proxy"] = c_count
            rec["coverage_ratio_liquidity_stress_proxy"] = c_ratio
            sev = classify_proxy_severity("liquidity_stress_proxy", c_count, c_ratio, universe_count, cfg.coverage)
            proxy_status["liquidity_stress_proxy"] = sev
            if sev != "INFO":
                warn_rows.append(make_warning_row(date, "liquidity_stress_proxy", sev, universe_count, c_count, c_ratio, "coverage_or_validity"))
        else:
            warn_rows.append(make_warning_row(date, "liquidity_stress_proxy", "INFO", universe_count, 0, np.nan, "exploratory_disabled"))

        rec["severity_max"] = aggregate_severity(proxy_status.values())
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    enabled = set(cfg.exploratory.enabled)
    if "breadth_ma_distance" in enabled:
        ma = out["breadth_pct_up"].rolling(cfg.exploratory.breadth_ma_window, min_periods=cfg.exploratory.breadth_ma_window).mean()
        out["breadth_ma_distance"] = out["breadth_pct_up"] - ma
    warn_df = pd.DataFrame(warn_rows).sort_values(["date", "proxy_name", "severity"]).reset_index(drop=True)
    return out, warn_df


def classify_proxy_severity(
    proxy_name: str,
    coverage_count: int,
    coverage_ratio: float,
    universe_count: int,
    thresholds: CoverageThresholds,
) -> str:
    if universe_count <= 0:
        return "FAIL"
    if pd.isna(coverage_ratio) or coverage_count < thresholds.n_min or coverage_ratio < thresholds.ratio_min:
        return "WARN"
    near_n = coverage_count < (thresholds.n_min + thresholds.warn_margin_n)
    near_ratio = coverage_ratio < (thresholds.ratio_min + thresholds.warn_margin_ratio)
    if near_n or near_ratio:
        return "INFO"
    return "INFO"


def aggregate_severity(values: Iterable[str]) -> str:
    max_sev = "INFO"
    max_rank = 1
    for v in values:
        rank = SEVERITY_ORDER.get(v, 1)
        if rank > max_rank:
            max_rank = rank
            max_sev = v
    return max_sev


def make_warning_row(
    date: pd.Timestamp,
    proxy_name: str,
    severity: str,
    universe_count: int,
    coverage_count: int,
    coverage_ratio: float,
    reason: str,
) -> Dict[str, Any]:
    return {
        "date": pd.Timestamp(date),
        "proxy_name": proxy_name,
        "severity": severity,
        "universe_count": int(universe_count),
        "coverage_count": int(coverage_count),
        "coverage_ratio": None if pd.isna(coverage_ratio) else float(coverage_ratio),
        "reason": reason,
    }


def compute_symbol_market_corr(panel: pd.DataFrame, market_base: pd.DataFrame, window: int) -> str:
    col = f"corr_to_market_{window}d"
    merged = panel.merge(market_base[["date", "market_ret_1d"]], on="date", how="left")

    def _corr(g: pd.DataFrame) -> pd.Series:
        return (
            g["ret_1d_adj"]
            .rolling(window=window, min_periods=window)
            .corr(g["market_ret_1d"])
        )

    merged[col] = merged.groupby("symbol", sort=False, group_keys=False).apply(_corr)
    panel[col] = merged[col].values
    return col


def apply_optional_transforms(df: pd.DataFrame, cfg: ProxyConfig) -> pd.DataFrame:
    out = df.copy()
    if cfg.smoothing.enabled:
        for col in cfg.smoothing.columns:
            if col not in out.columns:
                continue
            for w in cfg.smoothing.ema_windows:
                out[f"{col}_ema{w}"] = out[col].ewm(span=w, adjust=False, min_periods=w).mean()
    if cfg.zscore.enabled:
        for col in cfg.zscore.columns:
            if col not in out.columns:
                continue
            for w in cfg.zscore.rolling_windows:
                mean = out[col].rolling(window=w, min_periods=w).mean()
                std = out[col].rolling(window=w, min_periods=w).std(ddof=1)
                out[f"{col}_z{w}"] = (out[col] - mean) / std
    return out


def validate_output(df: pd.DataFrame, warn_df: pd.DataFrame, cfg: ProxyConfig) -> Dict[str, Any]:
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    if df["date"].duplicated().any():
        failures.append({"check": "duplicate_dates", "message": "Output contains duplicate dates"})
    if df["universe_snapshot_id"].isna().any():
        failures.append({"check": "missing_universe_snapshot_id", "message": "Missing universe_snapshot_id in output"})

    if "breadth_pct_up" in df.columns:
        bad = df["breadth_pct_up"].notna() & ~df["breadth_pct_up"].between(0.0, 1.0)
        if bad.any():
            failures.append(
                {
                    "check": "breadth_range",
                    "message": "breadth_pct_up must lie in [0,1]",
                    "rows": int(bad.sum()),
                }
            )

    for proxy in CORE_PROXIES:
        if proxy not in df.columns:
            failures.append({"check": "missing_core_proxy", "message": f"Missing core proxy column: {proxy}"})

    warn_counts = warn_df[warn_df["severity"] == "WARN"].groupby("proxy_name").size().to_dict() if not warn_df.empty else {}
    for proxy, count in warn_counts.items():
        warnings.append({"check": "proxy_warn_days", "proxy_name": proxy, "warn_days": int(count)})

    gate = "PASS"
    if failures:
        gate = "FAIL"
    elif warnings:
        gate = "WARN"

    return {
        "gate_status": gate,
        "failures": failures,
        "warnings": warnings,
        "severity_counts": warn_df["severity"].value_counts(dropna=False).to_dict() if not warn_df.empty else {},
        "proxy_warn_days": warn_counts,
        "n_rows": int(len(df)),
        "date_min": None if df.empty else str(df["date"].min().date()),
        "date_max": None if df.empty else str(df["date"].max().date()),
    }


def build_summary(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    proxy_like = [c for c in numeric_cols if c in CORE_PROXIES or c in EXPLORATORY_PROXIES or "_ema" in c or "_z" in c]
    summary_cols = sorted(set(proxy_like + [c for c in numeric_cols if c.startswith("coverage_")]))
    stats: Dict[str, Any] = {}
    for col in summary_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        stats[col] = {
            "mean": None if s.dropna().empty else float(s.mean()),
            "std": None if s.dropna().empty else float(s.std(ddof=1)),
            "p01": None if s.dropna().empty else float(s.quantile(0.01)),
            "p05": None if s.dropna().empty else float(s.quantile(0.05)),
            "p50": None if s.dropna().empty else float(s.quantile(0.50)),
            "p95": None if s.dropna().empty else float(s.quantile(0.95)),
            "p99": None if s.dropna().empty else float(s.quantile(0.99)),
            "nulls": int(s.isna().sum()),
            "nonnull": int(s.notna().sum()),
        }
    return {
        "n_rows": int(len(df)),
        "date_min": None if df.empty else str(df["date"].min().date()),
        "date_max": None if df.empty else str(df["date"].max().date()),
        "columns": df.columns.tolist(),
        "stats": stats,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if pd.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def persist_outputs(
    df: pd.DataFrame,
    warn_df: pd.DataFrame,
    summary: Mapping[str, Any],
    validation: Mapping[str, Any],
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    run_id: str,
) -> None:
    ensure_parquet_support()
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    payload = df.copy()
    payload["date"] = pd.to_datetime(payload["date"]).dt.strftime("%Y-%m-%d")
    for d, day in payload.groupby("date", sort=True):
        part_dir = outdir / f"date={d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        day.to_parquet(part_dir / "part-00000.parquet", index=False)

    warn_path = outdir / f"validation_rows_{run_id}.parquet"
    warn_df.to_parquet(warn_path, index=False)
    write_json(outdir / f"summary_{run_id}.json", summary)
    write_json(outdir / f"validation_{run_id}.json", validation)
    write_json(outdir / f"manifest_{run_id}.json", manifest)


def run_market_proxies(
    prices_path: str | Path,
    universe_path: str | Path,
    config_path: str | Path,
    run_id: str,
    asof_ts_utc: str,
    output_dir_override: Optional[str | Path] = None,
) -> Dict[str, Any]:
    asof_norm = parse_datetime_utc(asof_ts_utc)
    cfg, raw_cfg, cfg_hash = load_config(config_path)
    output_dir = str(output_dir_override) if output_dir_override else cfg.output_dir

    prices = normalize_prices(read_table(prices_path))
    universe = normalize_universe(read_table(universe_path))

    if prices["date"].min() > universe["date"].max() or prices["date"].max() < universe["date"].min():
        raise MarketProxiesError("Prices and universe have non-overlapping date ranges.")

    derived_prices = add_symbol_level_derived_columns(prices, cfg)
    panel = build_universe_panel(derived_prices, universe)
    market_base = build_market_return_base(panel, cfg)
    proxies, warn_df = compute_daily_proxies(panel, market_base, cfg)
    proxies = apply_optional_transforms(proxies, cfg)
    proxies["run_id"] = run_id
    proxies["asof_ts_utc"] = asof_norm
    front_cols = ["date", "run_id", "asof_ts_utc", "universe_snapshot_id"]
    other_cols = [c for c in proxies.columns if c not in front_cols]
    proxies = proxies[front_cols + other_cols]

    validation = validate_output(proxies, warn_df, cfg)
    summary = build_summary(proxies)
    manifest = {
        "module_name": MODULE_NAME,
        "code_version": CODE_VERSION,
        "run_id": run_id,
        "asof_ts_utc": asof_norm,
        "generated_at_utc": utc_now_iso(),
        "config_hash": cfg_hash,
        "config": raw_cfg,
        "inputs": {
            "prices_adjusted_path": str(prices_path),
            "universe_history_path": str(universe_path),
            "config_path": str(config_path),
        },
        "rows_output": int(len(proxies)),
        "date_min": None if proxies.empty else str(pd.to_datetime(proxies["date"]).min().date()),
        "date_max": None if proxies.empty else str(pd.to_datetime(proxies["date"]).max().date()),
        "core_proxies": CORE_PROXIES,
        "exploratory_enabled": list(cfg.exploratory.enabled),
        "gate_status": validation["gate_status"],
        "artifacts": {
            "partitioned_dataset_dir": str(Path(output_dir)),
            "summary_json": str(Path(output_dir) / f"summary_{run_id}.json"),
            "validation_json": str(Path(output_dir) / f"validation_{run_id}.json"),
            "manifest_json": str(Path(output_dir) / f"manifest_{run_id}.json"),
            "validation_rows_parquet": str(Path(output_dir) / f"validation_rows_{run_id}.parquet"),
        },
    }

    persist_outputs(proxies, warn_df, summary, validation, manifest, output_dir, run_id)
    return {
        "data": proxies,
        "validation_rows": warn_df,
        "summary": summary,
        "validation": validation,
        "manifest": manifest,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build PIT-safe market proxies from adjusted prices and universe history.")
    p.add_argument("--prices-path", required=True, help="Adjusted prices path (file or directory)")
    p.add_argument("--universe-path", required=True, help="Universe history path (file or directory)")
    p.add_argument("--config-path", required=True, help="Config path (YAML or JSON)")
    p.add_argument("--run-id", required=True, help="Run identifier")
    p.add_argument("--as-of-ts-utc", required=True, help="As-of timestamp in UTC")
    p.add_argument("--output-dir", default=None, help="Optional override for output directory")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = run_market_proxies(
            prices_path=args.prices_path,
            universe_path=args.universe_path,
            config_path=args.config_path,
            run_id=args.run_id,
            asof_ts_utc=args.as_of_ts_utc,
            output_dir_override=args.output_dir,
        )
    except MarketProxiesError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover
        print(f"UNEXPECTED ERROR: {exc}", file=sys.stderr)
        return 3

    manifest = result["manifest"]
    print(json.dumps({
        "status": "ok",
        "run_id": manifest["run_id"],
        "gate_status": manifest["gate_status"],
        "rows_output": manifest["rows_output"],
        "output_dir": manifest["artifacts"]["partitioned_dataset_dir"],
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
