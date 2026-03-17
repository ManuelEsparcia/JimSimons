from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# -----------------------------------------------------------------------------
# Errors / constants
# -----------------------------------------------------------------------------


class BorrowQCError(RuntimeError):
    """Raised when borrow QC cannot be evaluated safely."""


STRUCTURAL_SEVERITIES = {"FAIL_STRUCTURAL"}
NUMERIC_SEVERITIES = {"FAIL_NUMERIC"}
SOFT_SEVERITIES = {"WARN"}
INFO_SEVERITIES = {"INFO"}
VALID_TIERS = {"easy", "medium", "hard", "blocked"}
VALID_QUALITIES = {"high", "medium", "low"}
VALID_GATES = {"pass", "warn", "fail"}

BORROW_REQUIRED_COLUMNS = {
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
}

DAILY_REQUIRED_COLUMNS = [
    "date",
    "coverage_rate",
    "coverage_smallcap",
    "jump_share",
    "stale_share",
    "underblocking_rate",
    "overblocking_rate",
    "avg_borrow_fee_daily",
    "p95_borrow_fee_daily",
    "blocked_share",
    "score_coverage",
    "score_stability",
    "score_coherence",
    "score_freshness",
    "score_total",
    "gate",
    "config_version",
]


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class CoverageThresholds:
    pass_min: float = 0.97
    warn_min: float = 0.92
    fail_min: float = 0.85
    smallcap_warn_min: float = 0.90
    short_book_warn_min: float = 0.88


@dataclass
class StabilityThresholds:
    jump_rel_symbol: float = 2.50
    jump_abs_symbol: float = 0.25
    suspicious_jump_warn_share: float = 0.05
    suspicious_jump_fail_share: float = 0.12
    stale_warn_share: float = 0.08
    stale_fail_share: float = 0.20
    unchanged_stale_trading_days: int = 7
    extreme_fee_absolute: float = 5.00
    extreme_fee_htb_absolute: float = 10.00


@dataclass
class CoherenceThresholds:
    theta_B_block: float = 0.60
    theta_alpha_block: float = 0.08
    theta_ub_fail: float = 0.010
    theta_ub_warn: float = 0.003
    theta_ob_warn: float = 0.05
    boundary_fee_band: float = 0.10
    boundary_alpha_band: float = 0.02


@dataclass
class ScoringWeights:
    coverage: float = 0.30
    stability: float = 0.20
    coherence: float = 0.30
    freshness: float = 0.10
    schema: float = 0.10


@dataclass
class ScoreThresholds:
    pass_min: float = 85.0
    warn_min: float = 70.0


@dataclass
class BorrowQCConfig:
    config_version: str = "borrow_qc_v1"
    coverage: CoverageThresholds = field(default_factory=CoverageThresholds)
    stability: StabilityThresholds = field(default_factory=StabilityThresholds)
    coherence: CoherenceThresholds = field(default_factory=CoherenceThresholds)
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    score_thresholds: ScoreThresholds = field(default_factory=ScoreThresholds)
    require_universe_nonempty: bool = True
    fail_on_missing_locate: bool = True
    market_calendar_name: str = "trading_dates_from_universe"


DEFAULT_CONFIG = BorrowQCConfig()


# -----------------------------------------------------------------------------
# Column aliases
# -----------------------------------------------------------------------------


UNIVERSE_ALIASES: Dict[str, Sequence[str]] = {
    "symbol": ["symbol", "ticker", "asset", "sid"],
    "date": ["date", "trade_date", "session_date"],
    "size_bucket": ["size_bucket", "cap_bucket", "market_cap_bucket"],
    "liq_bucket": ["liq_bucket", "liquidity_bucket", "adv_bucket"],
    "short_subuniverse_flag": ["short_subuniverse_flag", "short_book_flag", "short_universe_flag", "is_short_universe"],
    "membership_state": ["membership_state", "universe_state", "state"],
    "is_member": ["is_member", "in_universe", "universe_member_flag"],
}

LOCATE_ALIASES: Dict[str, Sequence[str]] = {
    "symbol": ["symbol", "ticker", "asset", "sid"],
    "date": ["date", "trade_date", "session_date"],
    "locate_allowed": ["locate_allowed", "allowed", "is_allowed", "locate_pass", "can_short", "short_allowed"],
    "locate_blocked": ["locate_blocked", "blocked", "is_blocked", "short_blocked", "locate_fail"],
    "locate_decision": ["locate_decision", "decision", "locate_status", "status"],
    "override_flag": ["override_flag", "locate_override_flag", "documented_override", "expected_exception_flag"],
    "override_reason": ["override_reason", "locate_override_reason", "exception_reason", "decision_reason"],
    "config_version": ["config_version", "locate_config_version", "rule_version"],
}

BOOLEAN_TRUE = {"1", "true", "t", "yes", "y", "allowed", "pass", "ok", "open"}
BOOLEAN_FALSE = {"0", "false", "f", "no", "n", "blocked", "fail", "reject", "closed"}


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def utc_now_iso() -> str:
    ts = pd.Timestamp.utcnow()
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")



def git_revision() -> Optional[str]:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return proc.stdout.strip()
    except Exception:
        return None



def load_yaml_or_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise BorrowQCError("PyYAML is required to load YAML configs but is not installed.")
        return yaml.safe_load(text)
    return json.loads(text)



def deep_update_dataclass(instance: Any, updates: Mapping[str, Any]) -> Any:
    for key, value in updates.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            deep_update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance



def load_config(path: Optional[Path]) -> BorrowQCConfig:
    cfg = BorrowQCConfig()
    if path is None:
        return cfg
    payload = load_yaml_or_json(path)
    if not isinstance(payload, Mapping):
        raise BorrowQCError("QC config must deserialize to a mapping.")
    return deep_update_dataclass(cfg, payload)



def choose_parquet_engine() -> str:
    for engine in ("pyarrow", "fastparquet"):
        try:
            __import__(engine)
            return engine
        except Exception:
            continue
    raise BorrowQCError(
        "Writing parquet requires either 'pyarrow' or 'fastparquet'. Install one of them in your environment."
    )



def write_parquet(df: pd.DataFrame, path: Path) -> None:
    engine = choose_parquet_engine()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine=engine, index=False)



def write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, default=_json_default)



def _json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if pd.isna(obj):
        return None
    return str(obj)



def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise BorrowQCError(f"Unsupported table format: {path}")



def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out



def resolve_alias(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for alias in aliases:
        col = lower_map.get(alias.lower())
        if col is not None:
            return col
    return None



def ensure_columns(df: pd.DataFrame, mapping: Dict[str, Sequence[str]], required: Sequence[str], context: str) -> pd.DataFrame:
    out = normalize_columns(df)
    renamed: Dict[str, str] = {}
    missing: List[str] = []
    for canonical in required:
        col = resolve_alias(out, mapping.get(canonical, [canonical]))
        if col is None:
            missing.append(canonical)
        else:
            renamed[col] = canonical
    if missing:
        raise BorrowQCError(f"{context}: missing required columns {missing}")
    out = out.rename(columns=renamed)
    return out



def to_bool(series: pd.Series, default: Optional[bool] = None) -> pd.Series:
    if series.dtype == bool:
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(float(default) if default is not None else np.nan).astype(float).round().astype("Int64").map({1: True, 0: False})
    vals = series.astype(str).str.strip().str.lower()
    out = vals.map(lambda x: True if x in BOOLEAN_TRUE else False if x in BOOLEAN_FALSE else default)
    return pd.Series(out, index=series.index, dtype="boolean")



def safe_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(pd.to_numeric(series, errors="coerce").mean())



def safe_quantile(series: pd.Series, q: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float(s.quantile(q))



def contract_hash(columns: Iterable[str]) -> str:
    payload = "|".join(sorted(map(str, columns)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()



def add_failure(
    failures: List[Dict[str, Any]],
    *,
    date: Optional[Any],
    symbol: Optional[Any],
    check_type: str,
    severity: str,
    observed_value: Any,
    threshold_violated: Any,
    classification: str,
    config_version: str,
    message: Optional[str] = None,
) -> None:
    failures.append(
        {
            "date": str(pd.Timestamp(date).date()) if date is not None and not pd.isna(date) else None,
            "symbol": None if symbol is None or pd.isna(symbol) else str(symbol),
            "check_type": check_type,
            "severity": severity,
            "observed_value": _json_default(observed_value),
            "threshold_violated": _json_default(threshold_violated),
            "classification": classification,
            "config_version": config_version,
            "message": message,
        }
    )


# -----------------------------------------------------------------------------
# Canonicalization
# -----------------------------------------------------------------------------



def canonicalize_borrow_proxy(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(df)
    missing = [c for c in BORROW_REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise BorrowQCError(f"borrow_proxy: missing required columns {missing}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["symbol"] = out["symbol"].astype(str).str.strip()

    numeric_cols = [
        "borrow_fee_annual",
        "borrow_fee_daily",
        "borrow_availability_score",
        "stress_score",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["htb_flag", "jump_flag", "stale_input_flag", "fallback_flag"]:
        out[col] = to_bool(out[col], default=False).fillna(False).astype(bool)

    out["borrow_tier"] = out["borrow_tier"].astype(str).str.strip().str.lower()
    out["proxy_quality"] = out["proxy_quality"].astype(str).str.strip().str.lower()
    out["proxy_source"] = out["proxy_source"].astype(str).str.strip().str.lower()
    out["config_version"] = out["config_version"].astype(str).str.strip()

    if "asof_timestamp" in out.columns:
        out["asof_timestamp"] = pd.to_datetime(out["asof_timestamp"], errors="coerce", utc=True)
    if "size_bucket" in out.columns:
        out["size_bucket"] = out["size_bucket"].astype(str).str.strip().str.lower()
    if "liq_bucket" in out.columns:
        out["liq_bucket"] = out["liq_bucket"].astype(str).str.strip().str.lower()
    if "short_pressure_bucket" in out.columns:
        out["short_pressure_bucket"] = out["short_pressure_bucket"].astype(str).str.strip().str.lower()

    return out



def canonicalize_locate_filter(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(df)
    out = ensure_columns(out, LOCATE_ALIASES, ["symbol", "date"], "locate_filter")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["symbol"] = out["symbol"].astype(str).str.strip()

    allowed_col = resolve_alias(out, LOCATE_ALIASES["locate_allowed"])
    blocked_col = resolve_alias(out, LOCATE_ALIASES["locate_blocked"])
    decision_col = resolve_alias(out, LOCATE_ALIASES["locate_decision"])
    override_flag_col = resolve_alias(out, LOCATE_ALIASES["override_flag"])
    override_reason_col = resolve_alias(out, LOCATE_ALIASES["override_reason"])
    cfg_col = resolve_alias(out, LOCATE_ALIASES["config_version"])

    if allowed_col is not None:
        out["locate_allowed"] = to_bool(out[allowed_col], default=None)
    elif blocked_col is not None:
        out["locate_allowed"] = ~to_bool(out[blocked_col], default=None)
    elif decision_col is not None:
        vals = out[decision_col].astype(str).str.strip().str.lower()
        allowed = vals.map(
            lambda x: True
            if x in {"allow", "allowed", "pass", "approved", "locatable", "open"}
            else False
            if x in {"block", "blocked", "reject", "denied", "not_locatable", "closed", "fail"}
            else None
        )
        out["locate_allowed"] = pd.Series(allowed, index=out.index, dtype="boolean")
    else:
        raise BorrowQCError(
            "locate_filter: need one of locate_allowed / locate_blocked / locate_decision to infer locate decision."
        )

    out["override_flag"] = (
        to_bool(out[override_flag_col], default=False) if override_flag_col is not None else pd.Series(False, index=out.index, dtype="boolean")
    ).fillna(False).astype(bool)
    out["override_reason"] = out[override_reason_col].astype(str) if override_reason_col is not None else ""
    out["locate_config_version"] = out[cfg_col].astype(str) if cfg_col is not None else "unknown"

    keep = ["symbol", "date", "locate_allowed", "override_flag", "override_reason", "locate_config_version"]
    return out[keep]



def canonicalize_universe(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(df)
    out = ensure_columns(out, UNIVERSE_ALIASES, ["symbol", "date"], "universe")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["symbol"] = out["symbol"].astype(str).str.strip()

    membership_col = resolve_alias(out, UNIVERSE_ALIASES["membership_state"])
    is_member_col = resolve_alias(out, UNIVERSE_ALIASES["is_member"])
    size_col = resolve_alias(out, UNIVERSE_ALIASES["size_bucket"])
    liq_col = resolve_alias(out, UNIVERSE_ALIASES["liq_bucket"])
    short_col = resolve_alias(out, UNIVERSE_ALIASES["short_subuniverse_flag"])

    if membership_col is not None:
        state = out[membership_col].astype(str).str.strip().str.lower()
        out["membership_state"] = state
        out["is_member"] = state.isin({"active", "member", "eligible", "pass", "in", "current"})
    elif is_member_col is not None:
        out["is_member"] = to_bool(out[is_member_col], default=True).fillna(True).astype(bool)
        out["membership_state"] = np.where(out["is_member"], "active", "inactive")
    else:
        out["is_member"] = True
        out["membership_state"] = "active"

    if size_col is not None:
        out["size_bucket"] = out[size_col].astype(str).str.strip().str.lower()
    else:
        out["size_bucket"] = np.nan

    if liq_col is not None:
        out["liq_bucket"] = out[liq_col].astype(str).str.strip().str.lower()
    else:
        out["liq_bucket"] = np.nan

    if short_col is not None:
        out["short_subuniverse_flag"] = to_bool(out[short_col], default=False).fillna(False).astype(bool)
    else:
        out["short_subuniverse_flag"] = False

    out = out[out["is_member"]].copy()
    keep = ["symbol", "date", "membership_state", "is_member", "size_bucket", "liq_bucket", "short_subuniverse_flag"]
    return out[keep]


# -----------------------------------------------------------------------------
# Validation and joins
# -----------------------------------------------------------------------------



def structural_validation(
    universe: pd.DataFrame,
    borrow: pd.DataFrame,
    locate: pd.DataFrame,
    cfg: BorrowQCConfig,
) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []

    def dupes(df: pd.DataFrame, name: str) -> None:
        if df.empty:
            return
        mask = df.duplicated(["symbol", "date"], keep=False)
        if mask.any():
            sample = df.loc[mask, ["date", "symbol"]].drop_duplicates().head(50)
            for row in sample.itertuples(index=False):
                add_failure(
                    failures,
                    date=row.date,
                    symbol=row.symbol,
                    check_type=f"duplicate_key_{name}",
                    severity="FAIL_STRUCTURAL",
                    observed_value="duplicate (symbol,date)",
                    threshold_violated="unique",
                    classification="duplicate_key_failure",
                    config_version=cfg.config_version,
                )

    dupes(universe, "universe")
    dupes(borrow, "borrow_proxy")
    dupes(locate, "locate_filter")

    for name, df in {"universe": universe, "borrow_proxy": borrow, "locate_filter": locate}.items():
        invalid_date = df["date"].isna()
        if invalid_date.any():
            add_failure(
                failures,
                date=None,
                symbol=None,
                check_type=f"invalid_date_{name}",
                severity="FAIL_STRUCTURAL",
                observed_value=int(invalid_date.sum()),
                threshold_violated=0,
                classification="invalid_dates",
                config_version=cfg.config_version,
            )
        blank_symbol = df["symbol"].isna() | (df["symbol"].astype(str).str.strip() == "")
        if blank_symbol.any():
            add_failure(
                failures,
                date=None,
                symbol=None,
                check_type=f"blank_symbol_{name}",
                severity="FAIL_STRUCTURAL",
                observed_value=int(blank_symbol.sum()),
                threshold_violated=0,
                classification="invalid_symbol",
                config_version=cfg.config_version,
            )

    if cfg.require_universe_nonempty and universe.empty:
        add_failure(
            failures,
            date=None,
            symbol=None,
            check_type="empty_universe",
            severity="FAIL_STRUCTURAL",
            observed_value=0,
            threshold_violated=">0 rows",
            classification="empty_input",
            config_version=cfg.config_version,
        )

    return failures



def build_joined_panel(
    universe: pd.DataFrame,
    borrow: pd.DataFrame,
    locate: pd.DataFrame,
    cfg: BorrowQCConfig,
    failures: List[Dict[str, Any]],
) -> pd.DataFrame:
    if universe.empty:
        return pd.DataFrame(columns=["date", "symbol"])

    u = universe.drop_duplicates(["symbol", "date"], keep="last").copy()
    b = borrow.drop_duplicates(["symbol", "date"], keep="last").copy()
    l = locate.drop_duplicates(["symbol", "date"], keep="last").copy()

    joined = u.merge(b, on=["symbol", "date"], how="left", suffixes=("", "_borrow"))
    joined = joined.merge(l, on=["symbol", "date"], how="left", suffixes=("", "_locate"))

    if joined.empty:
        add_failure(
            failures,
            date=None,
            symbol=None,
            check_type="join_empty",
            severity="FAIL_STRUCTURAL",
            observed_value=0,
            threshold_violated=">0 rows",
            classification="join_failure",
            config_version=cfg.config_version,
        )
        return joined

    joined["locate_allowed"] = joined.get("locate_allowed", pd.Series(index=joined.index, dtype="boolean"))
    joined["override_flag"] = joined.get("override_flag", False)
    joined["override_reason"] = joined.get("override_reason", "")
    joined["locate_config_version"] = joined.get("locate_config_version", "unknown")

    return joined


# -----------------------------------------------------------------------------
# Numeric / logical checks at row level
# -----------------------------------------------------------------------------



def numeric_and_logic_checks(panel: pd.DataFrame, cfg: BorrowQCConfig, failures: List[Dict[str, Any]]) -> pd.DataFrame:
    if panel.empty:
        return panel

    out = panel.copy()
    out["missing_join_borrow"] = out["borrow_fee_annual"].isna() & out["borrow_availability_score"].isna()
    out["missing_join_locate"] = out["locate_allowed"].isna()
    out["missing_critical"] = out[["borrow_fee_annual", "borrow_availability_score", "borrow_tier"]].isna().any(axis=1) | out["locate_allowed"].isna()

    neg_fee = pd.to_numeric(out["borrow_fee_annual"], errors="coerce") < 0
    avail_oob = (pd.to_numeric(out["borrow_availability_score"], errors="coerce") < 0) | (
        pd.to_numeric(out["borrow_availability_score"], errors="coerce") > 1
    )
    bad_tier = ~out["borrow_tier"].astype(str).str.lower().isin(VALID_TIERS) & out["borrow_tier"].notna()
    bad_quality = ~out["proxy_quality"].astype(str).str.lower().isin(VALID_QUALITIES) & out["proxy_quality"].notna()

    out["hard_numeric_fail"] = neg_fee | avail_oob | bad_tier | bad_quality | out["missing_critical"]

    extreme_no_htb = (pd.to_numeric(out["borrow_fee_annual"], errors="coerce") > cfg.stability.extreme_fee_absolute) & (~out["htb_flag"].fillna(False).astype(bool))
    extreme_htb = (pd.to_numeric(out["borrow_fee_annual"], errors="coerce") > cfg.stability.extreme_fee_htb_absolute) & out["htb_flag"].fillna(False).astype(bool)

    for idx in np.where(neg_fee)[0][:500]:
        row = out.iloc[idx]
        add_failure(failures, date=row.date, symbol=row.symbol, check_type="negative_fee", severity="FAIL_NUMERIC", observed_value=row.borrow_fee_annual,
                    threshold_violated=0, classification="numeric_hard_fail", config_version=cfg.config_version)
    for idx in np.where(avail_oob)[0][:500]:
        row = out.iloc[idx]
        add_failure(failures, date=row.date, symbol=row.symbol, check_type="availability_out_of_range", severity="FAIL_NUMERIC", observed_value=row.borrow_availability_score,
                    threshold_violated="[0,1]", classification="numeric_hard_fail", config_version=cfg.config_version)
    for idx in np.where(bad_tier)[0][:500]:
        row = out.iloc[idx]
        add_failure(failures, date=row.date, symbol=row.symbol, check_type="invalid_borrow_tier", severity="FAIL_NUMERIC", observed_value=row.borrow_tier,
                    threshold_violated=sorted(VALID_TIERS), classification="catalog_violation", config_version=cfg.config_version)
    for idx in np.where(bad_quality)[0][:500]:
        row = out.iloc[idx]
        add_failure(failures, date=row.date, symbol=row.symbol, check_type="invalid_proxy_quality", severity="FAIL_NUMERIC", observed_value=row.proxy_quality,
                    threshold_violated=sorted(VALID_QUALITIES), classification="catalog_violation", config_version=cfg.config_version)
    for idx in np.where(out["missing_critical"])[0][:500]:
        row = out.iloc[idx]
        classification = "missing_join" if bool(row.missing_join_borrow or row.missing_join_locate) else "missing_critical"
        add_failure(failures, date=row.date, symbol=row.symbol, check_type="missing_critical_field", severity="FAIL_NUMERIC", observed_value="null critical field",
                    threshold_violated="non-null criticals", classification=classification, config_version=cfg.config_version)
    for idx in np.where(extreme_no_htb)[0][:500]:
        row = out.iloc[idx]
        add_failure(failures, date=row.date, symbol=row.symbol, check_type="implausible_extreme_fee", severity="FAIL_NUMERIC", observed_value=row.borrow_fee_annual,
                    threshold_violated=cfg.stability.extreme_fee_absolute, classification="fee_ceiling_violation", config_version=cfg.config_version)
    for idx in np.where(extreme_htb)[0][:500]:
        row = out.iloc[idx]
        add_failure(failures, date=row.date, symbol=row.symbol, check_type="extreme_htb_fee", severity="WARN", observed_value=row.borrow_fee_annual,
                    threshold_violated=cfg.stability.extreme_fee_htb_absolute, classification="contextual_extreme_htb", config_version=cfg.config_version)

    return out


# -----------------------------------------------------------------------------
# Temporal context
# -----------------------------------------------------------------------------



def enrich_temporal_context(panel: pd.DataFrame, cfg: BorrowQCConfig) -> pd.DataFrame:
    if panel.empty:
        return panel

    out = panel.sort_values(["symbol", "date"]).copy()
    unique_dates = pd.Index(sorted(pd.to_datetime(out["date"].dropna().unique())))
    date_pos = {pd.Timestamp(d): i for i, d in enumerate(unique_dates)}
    out["trading_day_pos"] = out["date"].map(date_pos)

    out["prev_fee"] = out.groupby("symbol", sort=False)["borrow_fee_annual"].shift(1)
    out["prev_alpha"] = out.groupby("symbol", sort=False)["borrow_availability_score"].shift(1)
    out["prev_date"] = out.groupby("symbol", sort=False)["date"].shift(1)
    out["prev_tier"] = out.groupby("symbol", sort=False)["borrow_tier"].shift(1)
    out["prev_jump_flag"] = out.groupby("symbol", sort=False)["jump_flag"].shift(1)

    prev_pos = out["prev_date"].map(date_pos)
    out["trading_gap"] = out["trading_day_pos"] - prev_pos
    out["has_history"] = out["prev_fee"].notna() & (out["trading_gap"] == 1)
    denom = np.maximum(pd.to_numeric(out["prev_fee"], errors="coerce").fillna(0.0).abs(), 1e-6)
    out["fee_jump_ratio"] = (pd.to_numeric(out["borrow_fee_annual"], errors="coerce") - pd.to_numeric(out["prev_fee"], errors="coerce")).abs() / denom
    out["fee_jump_abs"] = (pd.to_numeric(out["borrow_fee_annual"], errors="coerce") - pd.to_numeric(out["prev_fee"], errors="coerce")).abs()

    out["computed_jump_flag"] = out["has_history"] & (
        (out["fee_jump_ratio"] > cfg.stability.jump_rel_symbol) | (out["fee_jump_abs"] > cfg.stability.jump_abs_symbol)
    )

    # Unchanged streak in trading days for stale fallback.
    same_value = (
        out["has_history"]
        & np.isclose(pd.to_numeric(out["borrow_fee_annual"], errors="coerce"), pd.to_numeric(out["prev_fee"], errors="coerce"), equal_nan=False)
        & np.isclose(pd.to_numeric(out["borrow_availability_score"], errors="coerce"), pd.to_numeric(out["prev_alpha"], errors="coerce"), equal_nan=False)
    )
    streak = np.zeros(len(out), dtype=int)
    last_symbol: Optional[str] = None
    for i, row in enumerate(out.itertuples(index=False)):
        if row.symbol != last_symbol:
            streak[i] = 0
            last_symbol = row.symbol
            continue
        streak[i] = streak[i - 1] + 1 if bool(same_value.iloc[i]) else 0
    out["unchanged_streak"] = streak
    out["stale_derived_flag"] = out["unchanged_streak"] >= cfg.stability.unchanged_stale_trading_days
    out["stale_effective_flag"] = out["stale_input_flag"].fillna(False).astype(bool) | out["stale_derived_flag"].fillna(False).astype(bool)

    return out


# -----------------------------------------------------------------------------
# Coherence and contextual checks
# -----------------------------------------------------------------------------



def classify_jump_context(group: pd.DataFrame, cfg: BorrowQCConfig) -> pd.Series:
    if group.empty:
        return pd.Series(dtype="object")

    suspicious_candidates = group["computed_jump_flag"].fillna(False) | group["jump_flag"].fillna(False)
    share = float(suspicious_candidates.mean()) if len(group) else 0.0
    results: List[str] = []
    for row in group.itertuples(index=False):
        if not bool(row.has_history):
            results.append("insufficient_history")
        elif not (bool(row.computed_jump_flag) or bool(row.jump_flag)):
            results.append("none")
        elif bool(getattr(row, "config_regime_change", False)):
            results.append("config_induced")
        elif share >= cfg.stability.suspicious_jump_warn_share and str(getattr(row, "proxy_source", "")).lower() in {
            "direct_lending_feed",
            "explicit_htb_flag",
        }:
            results.append("market_plausible")
        elif share >= max(cfg.stability.suspicious_jump_warn_share, 0.03):
            results.append("market_plausible")
        else:
            results.append("pipeline_suspicious")
    return pd.Series(results, index=group.index)



def enrich_coherence(panel: pd.DataFrame, cfg: BorrowQCConfig, failures: List[Dict[str, Any]]) -> pd.DataFrame:
    if panel.empty:
        return panel

    out = panel.copy()
    block_fee = pd.to_numeric(out["borrow_fee_annual"], errors="coerce") > cfg.coherence.theta_B_block
    block_alpha = pd.to_numeric(out["borrow_availability_score"], errors="coerce") < cfg.coherence.theta_alpha_block
    block_htb = out["htb_flag"].fillna(False).astype(bool)
    out["expected_block"] = block_fee | block_alpha | block_htb
    out["locate_blocked"] = ~out["locate_allowed"].fillna(False).astype(bool)

    fee_boundary = (pd.to_numeric(out["borrow_fee_annual"], errors="coerce") - cfg.coherence.theta_B_block).abs() <= cfg.coherence.boundary_fee_band
    alpha_boundary = (pd.to_numeric(out["borrow_availability_score"], errors="coerce") - cfg.coherence.theta_alpha_block).abs() <= cfg.coherence.boundary_alpha_band
    out["boundary_disagreement"] = fee_boundary | alpha_boundary
    out["expected_exception"] = out["override_flag"].fillna(False).astype(bool) | out["override_reason"].fillna("").astype(str).str.strip().ne("")
    out["underblocking_critical"] = out["expected_block"] & (~out["locate_blocked"]) & (~out["boundary_disagreement"]) & (~out["expected_exception"])
    out["overblocking"] = (~out["expected_block"]) & out["locate_blocked"] & (~out["expected_exception"])
    out["incoherent_basic"] = out["expected_block"] != out["locate_blocked"]

    # Required invariant from spec.
    bad_htb = out["htb_flag"].fillna(False).astype(bool) & out["locate_allowed"].fillna(False).astype(bool) & (~out["expected_exception"])
    for idx in np.where(bad_htb)[0][:500]:
        row = out.iloc[idx]
        add_failure(
            failures,
            date=row.date,
            symbol=row.symbol,
            check_type="htb_locate_incoherence",
            severity="WARN" if bool(row.boundary_disagreement) else "FAIL_NUMERIC",
            observed_value={"htb_flag": 1, "locate_allowed": 1},
            threshold_violated="htb should normally imply blocked unless documented override",
            classification="underblocking_critical" if not bool(row.boundary_disagreement) else "boundary_disagreement",
            config_version=cfg.config_version,
        )

    return out


# -----------------------------------------------------------------------------
# Daily aggregation and scoring
# -----------------------------------------------------------------------------



def compute_config_regime_change(panel: pd.DataFrame) -> pd.Series:
    if panel.empty:
        return pd.Series(dtype=bool)
    tmp = panel[["date", "config_version", "locate_config_version"]].drop_duplicates().sort_values("date")
    tmp["prev_proxy_cfg"] = tmp["config_version"].shift(1)
    tmp["prev_locate_cfg"] = tmp["locate_config_version"].shift(1)
    tmp["config_regime_change"] = (
        (tmp["config_version"] != tmp["prev_proxy_cfg"]) | (tmp["locate_config_version"] != tmp["prev_locate_cfg"])
    ) & tmp["prev_proxy_cfg"].notna()
    return panel[["date"]].merge(tmp[["date", "config_regime_change"]], on="date", how="left")["config_regime_change"].fillna(False)



def score_piecewise(value: float, pass_min: float, warn_min: float, fail_min: float) -> float:
    if pd.isna(value):
        return 0.0
    value = float(value)
    if value >= pass_min:
        return 100.0
    if value < fail_min:
        return 0.0
    if value < warn_min:
        return 30.0 * (value - fail_min) / max(warn_min - fail_min, 1e-9)
    return 30.0 + 70.0 * (value - warn_min) / max(pass_min - warn_min, 1e-9)



def score_inverted_rate(rate: float, warn: float, fail: float) -> float:
    if pd.isna(rate):
        return 0.0
    rate = float(rate)
    if rate <= 0:
        return 100.0
    if rate >= fail:
        return 0.0
    if rate <= warn:
        return 100.0 - 30.0 * (rate / max(warn, 1e-9))
    return 70.0 * max(0.0, 1.0 - (rate - warn) / max(fail - warn, 1e-9))



def compute_daily_panel(panel: pd.DataFrame, cfg: BorrowQCConfig, failures: List[Dict[str, Any]]) -> pd.DataFrame:
    if panel.empty:
        daily = pd.DataFrame(columns=DAILY_REQUIRED_COLUMNS)
        return daily

    out = panel.copy()
    out["config_regime_change"] = compute_config_regime_change(out)
    out["jump_context"] = out.groupby("date", group_keys=False).apply(lambda g: classify_jump_context(g, cfg))
    out["pipeline_suspicious_jump"] = out["jump_context"].eq("pipeline_suspicious")

    daily_rows: List[Dict[str, Any]] = []
    for date, g in out.groupby("date", sort=True):
        n = int(len(g))
        n_valid = int((~g["missing_critical"]).sum())
        coverage_rate = (n_valid / n) if n else float("nan")

        smallcap_mask = g["size_bucket"].astype(str).str.lower().eq("small") | g["size_bucket"].astype(str).str.lower().eq("micro")
        n_small = int(smallcap_mask.sum())
        coverage_smallcap = float((~g.loc[smallcap_mask, "missing_critical"]).mean()) if n_small > 0 else np.nan

        short_mask = g.get("short_subuniverse_flag", pd.Series(False, index=g.index)).fillna(False).astype(bool)
        coverage_short = float((~g.loc[short_mask, "missing_critical"]).mean()) if int(short_mask.sum()) > 0 else np.nan

        suspicious_jump_share = float(g.loc[g["has_history"], "pipeline_suspicious_jump"].mean()) if bool(g["has_history"].any()) else 0.0
        stale_share = float(g["stale_effective_flag"].mean()) if n else float("nan")
        underblocking_rate = float(g["underblocking_critical"].mean()) if n else float("nan")
        overblocking_rate = float(g["overblocking"].mean()) if n else float("nan")
        blocked_share = float(g["locate_blocked"].mean()) if n else float("nan")
        avg_fee_daily = safe_mean(g["borrow_fee_daily"])
        p95_fee_daily = safe_quantile(g["borrow_fee_daily"], 0.95)
        fallback_share = float(g["fallback_flag"].fillna(False).astype(bool).mean()) if n else float("nan")
        hard_fail_count = int(g["hard_numeric_fail"].sum())

        score_coverage = score_piecewise(
            min(
                x for x in [coverage_rate, coverage_smallcap if not pd.isna(coverage_smallcap) else 1.0, coverage_short if not pd.isna(coverage_short) else 1.0]
            ),
            pass_min=cfg.coverage.pass_min,
            warn_min=cfg.coverage.warn_min,
            fail_min=cfg.coverage.fail_min,
        )
        score_stability = score_inverted_rate(
            suspicious_jump_share,
            warn=cfg.stability.suspicious_jump_warn_share,
            fail=cfg.stability.suspicious_jump_fail_share,
        )
        score_freshness = score_inverted_rate(
            stale_share,
            warn=cfg.stability.stale_warn_share,
            fail=cfg.stability.stale_fail_share,
        )
        coherence_raw = 0.75 * max(0.0, 1.0 - underblocking_rate / max(cfg.coherence.theta_ub_fail, 1e-9)) + 0.25 * max(
            0.0,
            1.0 - overblocking_rate / max(cfg.coherence.theta_ob_warn, 1e-9),
        )
        score_coherence = float(np.clip(100.0 * coherence_raw, 0.0, 100.0))
        score_schema = 0.0 if hard_fail_count > 0 else 100.0

        w = cfg.scoring
        score_total = (
            w.coverage * score_coverage
            + w.stability * score_stability
            + w.coherence * score_coherence
            + w.freshness * score_freshness
            + w.schema * score_schema
        )

        if hard_fail_count > 0:
            gate = "fail"
        elif underblocking_rate > cfg.coherence.theta_ub_fail:
            gate = "fail"
        elif score_total >= cfg.score_thresholds.pass_min:
            gate = "pass"
        elif score_total >= cfg.score_thresholds.warn_min:
            gate = "warn"
        else:
            gate = "fail"

        row = {
            "date": pd.Timestamp(date),
            "coverage_rate": coverage_rate,
            "coverage_smallcap": coverage_smallcap,
            "coverage_short_book": coverage_short,
            "jump_share": suspicious_jump_share,
            "stale_share": stale_share,
            "underblocking_rate": underblocking_rate,
            "overblocking_rate": overblocking_rate,
            "avg_borrow_fee_daily": avg_fee_daily,
            "p95_borrow_fee_daily": p95_fee_daily,
            "blocked_share": blocked_share,
            "score_coverage": score_coverage,
            "score_stability": score_stability,
            "score_coherence": score_coherence,
            "score_freshness": score_freshness,
            "score_total": score_total,
            "score_schema": score_schema,
            "hard_fail_count": hard_fail_count,
            "fallback_share": fallback_share,
            "gate": gate,
            "config_version": cfg.config_version,
            "borrow_proxy_config_versions": json.dumps(sorted(set(map(str, g["config_version"].dropna().unique())))),
            "locate_config_versions": json.dumps(sorted(set(map(str, g["locate_config_version"].dropna().unique())))),
            "config_regime_change": bool(g["config_regime_change"].any()),
            "n_universe": n,
            "n_valid": n_valid,
            "coverage_liq_worst": float(
                g.groupby("liq_bucket")["missing_critical"].apply(lambda s: 1.0 - float(s.mean())).min()
            )
            if g["liq_bucket"].notna().any()
            else np.nan,
            "coverage_tier_worst": float(
                g.groupby("borrow_tier")["missing_critical"].apply(lambda s: 1.0 - float(s.mean())).min()
            )
            if g["borrow_tier"].notna().any()
            else np.nan,
        }
        daily_rows.append(row)

        if coverage_rate < cfg.coverage.pass_min:
            add_failure(
                failures,
                date=date,
                symbol=None,
                check_type="coverage_rate",
                severity="WARN" if coverage_rate >= cfg.coverage.fail_min else "FAIL_NUMERIC",
                observed_value=coverage_rate,
                threshold_violated={"warn_min": cfg.coverage.warn_min, "fail_min": cfg.coverage.fail_min, "pass_min": cfg.coverage.pass_min},
                classification="coverage_insufficient",
                config_version=cfg.config_version,
            )
        if not pd.isna(coverage_smallcap) and coverage_smallcap < cfg.coverage.smallcap_warn_min:
            add_failure(
                failures,
                date=date,
                symbol=None,
                check_type="coverage_smallcap",
                severity="WARN",
                observed_value=coverage_smallcap,
                threshold_violated=cfg.coverage.smallcap_warn_min,
                classification="coverage_selective_degradation",
                config_version=cfg.config_version,
            )
        if suspicious_jump_share >= cfg.stability.suspicious_jump_warn_share:
            add_failure(
                failures,
                date=date,
                symbol=None,
                check_type="jump_share",
                severity="WARN" if suspicious_jump_share < cfg.stability.suspicious_jump_fail_share else "FAIL_NUMERIC",
                observed_value=suspicious_jump_share,
                threshold_violated=cfg.stability.suspicious_jump_warn_share,
                classification="pipeline_suspicious_jump_share",
                config_version=cfg.config_version,
            )
        if stale_share >= cfg.stability.stale_warn_share:
            add_failure(
                failures,
                date=date,
                symbol=None,
                check_type="stale_share",
                severity="WARN" if stale_share < cfg.stability.stale_fail_share else "FAIL_NUMERIC",
                observed_value=stale_share,
                threshold_violated=cfg.stability.stale_warn_share,
                classification="freshness_degradation",
                config_version=cfg.config_version,
            )
        if underblocking_rate >= cfg.coherence.theta_ub_warn:
            add_failure(
                failures,
                date=date,
                symbol=None,
                check_type="underblocking_rate",
                severity="WARN" if underblocking_rate < cfg.coherence.theta_ub_fail else "FAIL_NUMERIC",
                observed_value=underblocking_rate,
                threshold_violated=cfg.coherence.theta_ub_warn,
                classification="underblocking_critical_rate",
                config_version=cfg.config_version,
            )

    daily = pd.DataFrame(daily_rows)
    for col in DAILY_REQUIRED_COLUMNS:
        if col not in daily.columns:
            daily[col] = np.nan
    ordered = DAILY_REQUIRED_COLUMNS + [c for c in daily.columns if c not in DAILY_REQUIRED_COLUMNS]
    return daily[ordered].sort_values("date").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------



def build_borrow_qc(
    borrow_proxy: pd.DataFrame,
    locate_filter: pd.DataFrame,
    universe: pd.DataFrame,
    cfg: Optional[BorrowQCConfig] = None,
    run_id: str = "manual_run",
    asof_timestamp: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    cfg = cfg or BorrowQCConfig()
    asof_timestamp = asof_timestamp or utc_now_iso()

    failures: List[Dict[str, Any]] = []
    borrow = canonicalize_borrow_proxy(borrow_proxy)
    locate = canonicalize_locate_filter(locate_filter)
    uni = canonicalize_universe(universe)

    failures.extend(structural_validation(uni, borrow, locate, cfg))
    panel = build_joined_panel(uni, borrow, locate, cfg, failures)
    panel = numeric_and_logic_checks(panel, cfg, failures)
    panel = enrich_temporal_context(panel, cfg)
    panel = enrich_coherence(panel, cfg, failures)
    daily = compute_daily_panel(panel, cfg, failures)

    failures_df = pd.DataFrame(failures)
    if not failures_df.empty:
        failures_df = failures_df.sort_values(["date", "severity", "check_type", "symbol"], na_position="last").reset_index(drop=True)

    summary = {
        "module": "data/borrow/borrow_qc.py",
        "run_id": run_id,
        "config_version": cfg.config_version,
        "window_start": str(pd.Timestamp(daily["date"].min()).date()) if not daily.empty else None,
        "window_end": str(pd.Timestamp(daily["date"].max()).date()) if not daily.empty else None,
        "n_dates": int(daily["date"].nunique()) if not daily.empty else 0,
        "n_rows_joined": int(panel.shape[0]),
        "pass_pct": float((daily["gate"] == "pass").mean()) if not daily.empty else 0.0,
        "warn_pct": float((daily["gate"] == "warn").mean()) if not daily.empty else 0.0,
        "fail_pct": float((daily["gate"] == "fail").mean()) if not daily.empty else 0.0,
        "hard_fail_count": int(failures_df["severity"].isin(STRUCTURAL_SEVERITIES | NUMERIC_SEVERITIES).sum()) if not failures_df.empty else 0,
        "avg_coverage_rate": safe_mean(daily["coverage_rate"]) if not daily.empty else float("nan"),
        "avg_jump_share": safe_mean(daily["jump_share"]) if not daily.empty else float("nan"),
        "avg_stale_share": safe_mean(daily["stale_share"]) if not daily.empty else float("nan"),
        "avg_underblocking_rate": safe_mean(daily["underblocking_rate"]) if not daily.empty else float("nan"),
        "avg_overblocking_rate": safe_mean(daily["overblocking_rate"]) if not daily.empty else float("nan"),
        "avg_score_total": safe_mean(daily["score_total"]) if not daily.empty else float("nan"),
        "avg_score_coverage": safe_mean(daily["score_coverage"]) if not daily.empty else float("nan"),
        "avg_score_stability": safe_mean(daily["score_stability"]) if not daily.empty else float("nan"),
        "avg_score_coherence": safe_mean(daily["score_coherence"]) if not daily.empty else float("nan"),
        "avg_score_freshness": safe_mean(daily["score_freshness"]) if not daily.empty else float("nan"),
        "borrow_proxy_versions": sorted(set(map(str, borrow["config_version"].dropna().unique()))) if not borrow.empty else [],
        "locate_versions": sorted(set(map(str, locate["locate_config_version"].dropna().unique()))) if not locate.empty else [],
        "column_contract_hash": contract_hash(BORROW_REQUIRED_COLUMNS.union({"locate_allowed", "symbol", "date"})),
        "market_calendar_name": cfg.market_calendar_name,
        "asof_timestamp": asof_timestamp,
        "validation": {
            "structural_failures": int(failures_df["severity"].isin(STRUCTURAL_SEVERITIES).sum()) if not failures_df.empty else 0,
            "numeric_failures": int(failures_df["severity"].isin(NUMERIC_SEVERITIES).sum()) if not failures_df.empty else 0,
            "warns": int((failures_df["severity"] == "WARN").sum()) if not failures_df.empty else 0,
            "infos": int((failures_df["severity"] == "INFO").sum()) if not failures_df.empty else 0,
        },
    }

    return daily, failures_df, summary, panel


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal QC gate for borrow proxy + locate data used by the short book.")
    parser.add_argument("--borrow-proxy-path", required=True, type=Path, help="Output from borrow_cost_proxy.py")
    parser.add_argument("--locate-filter-path", required=True, type=Path, help="Output from locate_filter.py")
    parser.add_argument("--universe-path", required=True, type=Path, help="Effective causal universe by date")
    parser.add_argument("--qc-config-path", default=None, type=Path, help="Optional YAML/JSON QC config override")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory where QC outputs will be written")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument("--asof-timestamp", default=utc_now_iso(), help="UTC timestamp describing QC execution time")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    cfg = load_config(args.qc_config_path)
    borrow = load_table(args.borrow_proxy_path)
    locate = load_table(args.locate_filter_path)
    universe = load_table(args.universe_path)

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        daily, failures, summary, panel = build_borrow_qc(
            borrow_proxy=borrow,
            locate_filter=locate,
            universe=universe,
            cfg=cfg,
            run_id=args.run_id,
            asof_timestamp=args.asof_timestamp,
        )
    except Exception as exc:
        daily = pd.DataFrame(columns=DAILY_REQUIRED_COLUMNS)
        failures = pd.DataFrame([
            {
                "date": None,
                "symbol": None,
                "check_type": "build_borrow_qc_exception",
                "severity": "FAIL_STRUCTURAL",
                "observed_value": str(exc),
                "threshold_violated": None,
                "classification": "pipeline_exception",
                "config_version": cfg.config_version,
                "message": str(exc),
            }
        ])
        summary = {
            "module": "data/borrow/borrow_qc.py",
            "run_id": args.run_id,
            "config_version": cfg.config_version,
            "window_start": None,
            "window_end": None,
            "n_dates": 0,
            "n_rows_joined": 0,
            "pass_pct": 0.0,
            "warn_pct": 0.0,
            "fail_pct": 1.0,
            "hard_fail_count": 1,
            "avg_coverage_rate": None,
            "avg_jump_share": None,
            "avg_stale_share": None,
            "avg_underblocking_rate": None,
            "avg_overblocking_rate": None,
            "avg_score_total": None,
            "avg_score_coverage": None,
            "avg_score_stability": None,
            "avg_score_coherence": None,
            "avg_score_freshness": None,
            "borrow_proxy_versions": [],
            "locate_versions": [],
            "column_contract_hash": contract_hash(BORROW_REQUIRED_COLUMNS.union({"locate_allowed", "symbol", "date"})),
            "market_calendar_name": cfg.market_calendar_name,
            "asof_timestamp": args.asof_timestamp,
            "validation": {"structural_failures": 1, "numeric_failures": 0, "warns": 0, "infos": 0},
            "exception": str(exc),
        }
        panel = pd.DataFrame()

    manifest = {
        "module": "data/borrow/borrow_qc.py",
        "run_id": args.run_id,
        "config_version": cfg.config_version,
        "asof_timestamp": args.asof_timestamp,
        "inputs": {
            "borrow_proxy_path": str(args.borrow_proxy_path),
            "locate_filter_path": str(args.locate_filter_path),
            "universe_path": str(args.universe_path),
            "qc_config_path": str(args.qc_config_path) if args.qc_config_path else None,
        },
        "outputs": {
            "summary": str(outdir / "borrow_qc_summary.json"),
            "daily": str(outdir / "borrow_qc_daily.parquet"),
            "failures": str(outdir / "borrow_qc_failures.parquet"),
            "manifest": str(outdir / "manifest.json"),
            "joined_panel_debug": str(outdir / "borrow_qc_joined_panel.parquet"),
        },
        "column_contract_hash": contract_hash(BORROW_REQUIRED_COLUMNS.union({"locate_allowed", "symbol", "date"})),
        "borrow_proxy_versions": summary.get("borrow_proxy_versions"),
        "locate_versions": summary.get("locate_versions"),
        "market_calendar_name": cfg.market_calendar_name,
        "python_version": platform.python_version(),
        "git_revision": git_revision(),
        "n_dates": summary.get("n_dates"),
        "n_rows_joined": summary.get("n_rows_joined"),
    }

    try:
        write_parquet(daily, outdir / "borrow_qc_daily.parquet")
        write_parquet(failures, outdir / "borrow_qc_failures.parquet")
        write_parquet(panel, outdir / "borrow_qc_joined_panel.parquet")
    finally:
        write_json(summary, outdir / "borrow_qc_summary.json")
        write_json(manifest, outdir / "manifest.json")


if __name__ == "__main__":
    main()
