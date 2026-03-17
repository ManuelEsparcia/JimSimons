from __future__ import annotations

import argparse
import json
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


class LocateFilterError(RuntimeError):
    """Raised when locate filtering cannot be computed safely."""


VALID_TIERS = ("easy", "medium", "hard", "blocked")
TIER_TO_CODE = {"easy": 0, "medium": 1, "hard": 2, "blocked": 3}
TIER_SEVERITY = TIER_TO_CODE.copy()
PRECEDENCE = [
    "structural_missing",
    "explicit_block_override",
    "availability_critical",
    "fee_critical",
    "htb_severe",
    "proxy_quality_too_low",
    "fallback_severe",
    "tier_mapping_soft",
]

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
    "fallback_flag",
    "stale_input_flag",
    "config_version",
}

OUTPUT_REQUIRED_COLUMNS = [
    "date",
    "symbol",
    "run_id",
    "short_eligible_flag",
    "locate_tier",
    "locate_tier_code",
    "reject_reason",
    "dominant_rule",
    "all_triggered_rules",
    "borrow_fee_daily",
    "borrow_fee_annual",
    "availability_score",
    "htb_flag",
    "proxy_quality",
    "fallback_flag",
    "override_flag",
    "data_quality_flag",
    "locate_config_version",
    "upstream_proxy_version",
    "asof_timestamp",
    "instantaneous_tier",
    "tier_changed_flag",
    "hysteresis_hold_flag",
]


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class Thresholds:
    theta_alpha_easy: float = 0.70
    theta_alpha_medium: float = 0.40
    theta_alpha_hard: float = 0.18
    theta_alpha_block: float = 0.08
    theta_b_easy: float = 0.00020
    theta_b_medium: float = 0.00075
    theta_b_hard: float = 0.00200
    theta_b_block: float = 0.00600


@dataclass
class Policy:
    block_on_htb: bool = False
    block_on_low_quality: bool = True
    block_on_fallback_severe: bool = False
    fallback_severe_sources: Sequence[str] = field(default_factory=lambda: ["coarse_fallback", "insufficient_data"])
    low_quality_values: Sequence[str] = field(default_factory=lambda: ["low"])
    allow_hard_shorts: bool = True


@dataclass
class Hysteresis:
    k_down_blocked_to_hard: int = 3
    k_down_hard_to_medium: int = 3
    k_down_medium_to_easy: int = 5
    k_down_default: int = 3
    reset_streak_on_missing: bool = True


@dataclass
class SizingPolicy:
    base_short_weight: float = 0.02
    omega_easy: float = 1.00
    omega_medium: float = 0.65
    omega_hard: float = 0.25
    omega_blocked: float = 0.00
    aggregate_hard_limit: float = 0.10


@dataclass
class LocateConfig:
    config_version: str = "locate_filter_v1"
    thresholds: Thresholds = field(default_factory=Thresholds)
    policy: Policy = field(default_factory=Policy)
    hysteresis: Hysteresis = field(default_factory=Hysteresis)
    sizing: SizingPolicy = field(default_factory=SizingPolicy)
    manual_blocklist: Sequence[str] = field(default_factory=list)
    allow_override_symbols: Sequence[str] = field(default_factory=list)
    overrides: Sequence[Mapping[str, Any]] = field(default_factory=list)


DEFAULT_CONFIG = LocateConfig()


# -----------------------------------------------------------------------------
# Aliases / helpers
# -----------------------------------------------------------------------------


UNIVERSE_ALIASES: Dict[str, Sequence[str]] = {
    "symbol": ["symbol", "ticker", "asset", "sid"],
    "date": ["date", "trade_date", "session_date"],
    "membership_state": ["membership_state", "universe_state", "state"],
    "is_member": ["is_member", "in_universe", "universe_member_flag"],
    "size_bucket": ["size_bucket", "cap_bucket", "market_cap_bucket"],
    "market_cap": ["market_cap", "mkt_cap", "market_value"],
    "short_subuniverse_flag": ["short_subuniverse_flag", "short_book_flag", "short_universe_flag", "is_short_universe"],
}

BOOLEAN_TRUE = {"1", "true", "t", "yes", "y", "eligible", "member", "in", "active", "allow", "allowed"}
BOOLEAN_FALSE = {"0", "false", "f", "no", "n", "blocked", "out", "inactive", "reject", "rejected"}


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
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise LocateFilterError("PyYAML is required to read YAML configs but is not installed.")
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



def load_config(config_path: Optional[Path]) -> LocateConfig:
    cfg = LocateConfig()
    if config_path is None:
        return cfg
    payload = load_yaml_or_json(config_path)
    if isinstance(payload, Mapping):
        deep_update_dataclass(cfg, payload)
    return cfg



def require_parquet_engine() -> None:
    try:
        import pyarrow  # type: ignore  # noqa: F401

        return
    except Exception:
        pass
    try:
        import fastparquet  # type: ignore  # noqa: F401

        return
    except Exception:
        pass
    raise LocateFilterError(
        "Parquet support is unavailable. Install 'pyarrow' or 'fastparquet' to persist locate outputs."
    )



def write_parquet(df: pd.DataFrame, path: Path) -> None:
    require_parquet_engine()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)



def write_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")



def canonicalize_columns(df: pd.DataFrame, aliases: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    seen: set[str] = set()
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            if candidate in df.columns and candidate not in seen:
                rename_map[candidate] = canonical
                seen.add(candidate)
                break
    return df.rename(columns=rename_map)



def coerce_bool(x: Any) -> Optional[bool]:
    if pd.isna(x):
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (np.bool_,)):  # type: ignore[arg-type]
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    s = str(x).strip().lower()
    if s in BOOLEAN_TRUE:
        return True
    if s in BOOLEAN_FALSE:
        return False
    return None



def ensure_datetime_utc(series: pd.Series, name: str) -> pd.Series:
    out = pd.to_datetime(series, utc=True, errors="coerce")
    if out.isna().all():
        raise LocateFilterError(f"Column '{name}' could not be parsed as datetime.")
    return out



def ensure_date(series: pd.Series, name: str) -> pd.Series:
    out = pd.to_datetime(series, errors="coerce").dt.normalize()
    if out.isna().all():
        raise LocateFilterError(f"Column '{name}' could not be parsed as date.")
    return out



def normalize_symbol(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()



def serialize_rule_list(values: Iterable[str]) -> str:
    vals = [v for v in values if v]
    if not vals:
        return ""
    ordered = sorted(set(vals), key=lambda x: PRECEDENCE.index(x) if x in PRECEDENCE else 999)
    return "|".join(ordered)



def first_nonempty(values: Iterable[Any]) -> Optional[Any]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        if pd.isna(value):
            continue
        return value
    return None


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------



def load_borrow_proxy(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = BORROW_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise LocateFilterError(f"Borrow proxy is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["symbol"] = normalize_symbol(out["symbol"])
    out["date"] = ensure_date(out["date"], "date")
    for col in ["borrow_fee_annual", "borrow_fee_daily", "borrow_availability_score", "stress_score"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["htb_flag", "fallback_flag", "stale_input_flag"]:
        out[col] = out[col].map(coerce_bool)
    out["borrow_tier"] = out["borrow_tier"].astype(str).str.strip().str.lower()
    out["proxy_quality"] = out["proxy_quality"].astype(str).str.strip().str.lower()
    out["proxy_source"] = out["proxy_source"].astype(str).str.strip().str.lower()
    if "asof_timestamp" in out.columns:
        out["asof_timestamp"] = ensure_datetime_utc(out["asof_timestamp"], "asof_timestamp")
    else:
        out["asof_timestamp"] = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow().tz_convert("UTC")
    return out



def universe_membership_mask(df: pd.DataFrame) -> pd.Series:
    if "is_member" in df.columns:
        parsed = df["is_member"].map(coerce_bool)
        if parsed.notna().any():
            return parsed.fillna(False)
    if "membership_state" in df.columns:
        states = df["membership_state"].astype(str).str.strip().str.lower()
        return states.isin({"member", "active", "incumbent", "entered", "re-entered", "reentered", "in"})
    return pd.Series(True, index=df.index)



def load_universe(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = canonicalize_columns(df, UNIVERSE_ALIASES)
    required = {"symbol", "date"}
    missing = required - set(df.columns)
    if missing:
        raise LocateFilterError(f"Universe is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["symbol"] = normalize_symbol(out["symbol"])
    out["date"] = ensure_date(out["date"], "date")
    mask = universe_membership_mask(out)
    out = out.loc[mask].copy()
    if "short_subuniverse_flag" in out.columns:
        parsed = out["short_subuniverse_flag"].map(coerce_bool)
        out = out.loc[parsed.fillna(True)].copy()
    if out.empty:
        raise LocateFilterError("Universe has no effective rows after membership filtering.")

    keep_cols = [c for c in ["date", "symbol", "membership_state", "size_bucket", "market_cap"] if c in out.columns]
    out = out[keep_cols].drop_duplicates(subset=["date", "symbol"], keep="last")
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)



def load_previous_state(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["symbol", "date", "locate_tier", "improvement_target", "improvement_streak"])
    df = pd.read_parquet(path)
    required = {"symbol", "date", "locate_tier"}
    missing = required - set(df.columns)
    if missing:
        raise LocateFilterError(f"Previous locate state missing columns: {sorted(missing)}")
    out = df.copy()
    out["symbol"] = normalize_symbol(out["symbol"])
    out["date"] = ensure_date(out["date"], "date")
    if "improvement_streak" not in out.columns:
        out["improvement_streak"] = 0
    if "improvement_target" not in out.columns:
        out["improvement_target"] = None
    return out.sort_values(["symbol", "date"]).groupby("symbol", as_index=False).tail(1).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Overrides / rules
# -----------------------------------------------------------------------------


@dataclass
class OverrideDecision:
    block: bool = False
    allow: bool = False
    reason: Optional[str] = None
    rule_id: Optional[str] = None



def _date_in_window(date_value: pd.Timestamp, start: Optional[str], end: Optional[str]) -> bool:
    if start is not None:
        start_ts = pd.Timestamp(start).normalize()
        if date_value < start_ts:
            return False
    if end is not None:
        end_ts = pd.Timestamp(end).normalize()
        if date_value > end_ts:
            return False
    return True



def resolve_override(symbol: str, date_value: pd.Timestamp, cfg: LocateConfig) -> OverrideDecision:
    if symbol in {s.upper() for s in cfg.manual_blocklist}:
        return OverrideDecision(block=True, reason="manual_blocklist", rule_id="manual_blocklist")

    for entry in cfg.overrides:
        if not isinstance(entry, Mapping):
            continue
        entry_symbol = str(entry.get("symbol", "")).strip().upper()
        if entry_symbol and entry_symbol != symbol:
            continue
        if not _date_in_window(date_value, entry.get("start_date"), entry.get("end_date")):
            continue
        action = str(entry.get("action", "")).strip().lower()
        reason = str(entry.get("reason", action or "override")).strip() or "override"
        rule_id = str(entry.get("rule_id", reason)).strip() or reason
        if action == "block":
            return OverrideDecision(block=True, reason=reason, rule_id=rule_id)
        if action == "allow":
            return OverrideDecision(allow=True, reason=reason, rule_id=rule_id)

    if symbol in {s.upper() for s in cfg.allow_override_symbols}:
        return OverrideDecision(allow=True, reason="allow_override_symbols", rule_id="allow_override_symbols")
    return OverrideDecision()



def determine_soft_tier(alpha: Optional[float], fee_daily: Optional[float], htb_flag: Optional[bool], cfg: LocateConfig) -> str:
    th = cfg.thresholds
    alpha = float(alpha) if alpha is not None and not pd.isna(alpha) else np.nan
    fee_daily = float(fee_daily) if fee_daily is not None and not pd.isna(fee_daily) else np.nan
    htb = bool(htb_flag) if htb_flag is not None and not pd.isna(htb_flag) else False

    if (
        not np.isnan(alpha)
        and alpha >= th.theta_alpha_easy
        and not np.isnan(fee_daily)
        and fee_daily <= th.theta_b_easy
        and not htb
    ):
        return "easy"

    hard_condition = False
    if not np.isnan(alpha) and alpha < th.theta_alpha_hard:
        hard_condition = True
    if not np.isnan(fee_daily) and fee_daily > th.theta_b_hard:
        hard_condition = True
    if htb:
        hard_condition = True
    if hard_condition:
        return "hard"
    return "medium"



def required_persistence(prev_tier: str, new_tier: str, cfg: LocateConfig) -> int:
    prev_sev = TIER_SEVERITY[prev_tier]
    new_sev = TIER_SEVERITY[new_tier]
    if new_sev >= prev_sev:
        return 0
    if prev_tier == "blocked":
        return int(cfg.hysteresis.k_down_blocked_to_hard)
    if prev_tier == "hard":
        return int(cfg.hysteresis.k_down_hard_to_medium)
    if prev_tier == "medium" and new_tier == "easy":
        return int(cfg.hysteresis.k_down_medium_to_easy)
    return int(cfg.hysteresis.k_down_default)



def apply_hysteresis(
    instantaneous_tier: str,
    prev_tier: Optional[str],
    prev_target: Optional[str],
    prev_streak: int,
    cfg: LocateConfig,
) -> Tuple[str, str, int, bool]:
    if prev_tier is None or prev_tier not in TIER_TO_CODE:
        return instantaneous_tier, instantaneous_tier, 0, False

    inst_sev = TIER_SEVERITY[instantaneous_tier]
    prev_sev = TIER_SEVERITY[prev_tier]

    if inst_sev > prev_sev:
        return instantaneous_tier, instantaneous_tier, 0, False
    if inst_sev == prev_sev:
        return prev_tier, prev_tier, 0, False

    # improvement: hold unless persistent enough
    needed = required_persistence(prev_tier, instantaneous_tier, cfg)
    target = instantaneous_tier
    streak = prev_streak + 1 if prev_target == target else 1
    if streak >= needed:
        return instantaneous_tier, instantaneous_tier, 0, False
    return prev_tier, target, streak, True



def dominant_rule_from(triggered: Sequence[str]) -> Optional[str]:
    for rule in PRECEDENCE:
        if rule in triggered:
            return rule
    return None


# -----------------------------------------------------------------------------
# Core decision engine
# -----------------------------------------------------------------------------



def compute_locate_filter(
    borrow_proxy: pd.DataFrame,
    universe: pd.DataFrame,
    cfg: LocateConfig,
    run_id: str,
    asof_timestamp: str,
    previous_state: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if previous_state is None:
        previous_state = pd.DataFrame(columns=["symbol", "date", "locate_tier", "improvement_target", "improvement_streak"])

    merged = universe.merge(
        borrow_proxy,
        on=["date", "symbol"],
        how="left",
        suffixes=("", "_borrow"),
        validate="one_to_one",
    ).sort_values(["date", "symbol"]).reset_index(drop=True)

    # Normalize optional fields after join.
    for col in ["borrow_fee_daily", "borrow_fee_annual", "borrow_availability_score", "stress_score"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    for col in ["htb_flag", "fallback_flag", "stale_input_flag"]:
        if col in merged.columns:
            merged[col] = merged[col].map(coerce_bool)
    if "proxy_quality" in merged.columns:
        merged["proxy_quality"] = merged["proxy_quality"].astype(str).str.strip().str.lower().replace({"nan": np.nan})
    if "proxy_source" in merged.columns:
        merged["proxy_source"] = merged["proxy_source"].astype(str).str.strip().str.lower().replace({"nan": np.nan})

    prev_map = {
        row.symbol: {
            "tier": row.locate_tier,
            "target": first_nonempty([getattr(row, "improvement_target", None)]),
            "streak": int(first_nonempty([getattr(row, "improvement_streak", 0)]) or 0),
        }
        for row in previous_state.itertuples(index=False)
    }

    results: List[Dict[str, Any]] = []
    dates = list(pd.Index(merged["date"]).sort_values().unique())
    for date_value in dates:
        day = merged.loc[merged["date"] == date_value].sort_values("symbol")
        for row in day.itertuples(index=False):
            rec = row._asdict()
            symbol = rec["symbol"]
            override = resolve_override(symbol, pd.Timestamp(date_value), cfg)
            triggered: List[str] = []
            data_quality_flags: List[str] = []

            critical_missing = []
            for col in ["borrow_fee_daily", "borrow_availability_score", "htb_flag", "proxy_quality"]:
                if col not in rec or pd.isna(rec.get(col)):
                    critical_missing.append(col)
            if critical_missing:
                triggered.append("structural_missing")
                data_quality_flags.append("missing_borrow_join")

            alpha = rec.get("borrow_availability_score")
            fee_daily = rec.get("borrow_fee_daily")
            fee_annual = rec.get("borrow_fee_annual")
            htb_flag = rec.get("htb_flag")
            proxy_quality = rec.get("proxy_quality")
            proxy_source = rec.get("proxy_source")
            fallback_flag = rec.get("fallback_flag")
            stale_input_flag = rec.get("stale_input_flag")
            upstream_proxy_version = rec.get("config_version")

            if override.block:
                triggered.append("explicit_block_override")
            if not pd.isna(alpha) and float(alpha) < cfg.thresholds.theta_alpha_block:
                triggered.append("availability_critical")
            if not pd.isna(fee_daily) and float(fee_daily) > cfg.thresholds.theta_b_block:
                triggered.append("fee_critical")
            if bool(htb_flag) and cfg.policy.block_on_htb:
                triggered.append("htb_severe")
            if str(proxy_quality).lower() in {x.lower() for x in cfg.policy.low_quality_values} and cfg.policy.block_on_low_quality:
                triggered.append("proxy_quality_too_low")
            if bool(fallback_flag) and cfg.policy.block_on_fallback_severe and str(proxy_source).lower() in {
                x.lower() for x in cfg.policy.fallback_severe_sources
            }:
                triggered.append("fallback_severe")
            if bool(stale_input_flag):
                data_quality_flags.append("stale_borrow_proxy")
            if bool(fallback_flag):
                data_quality_flags.append("fallback_proxy")
            if str(proxy_quality).lower() == "low":
                data_quality_flags.append("low_quality_proxy")

            hard_block = dominant_rule_from(triggered) is not None and dominant_rule_from(triggered) != "tier_mapping_soft"
            instantaneous_tier = "blocked" if hard_block else determine_soft_tier(alpha, fee_daily, htb_flag, cfg)

            # Allow override can unblock some rules, but never structural_missing or availability_critical.
            override_flag = False
            if override.allow:
                if not any(rule in triggered for rule in ["structural_missing", "availability_critical"]):
                    triggered = [r for r in triggered if r not in {"explicit_block_override", "fee_critical", "htb_severe", "proxy_quality_too_low", "fallback_severe"}]
                    instantaneous_tier = determine_soft_tier(alpha, fee_daily, htb_flag, cfg)
                    override_flag = True
                else:
                    # keep blocked; just record documented override attempt
                    override_flag = True
            elif override.block:
                override_flag = True

            if instantaneous_tier != "blocked" and instantaneous_tier == "hard" and not cfg.policy.allow_hard_shorts:
                instantaneous_tier = "blocked"
                if "explicit_block_override" not in triggered:
                    triggered.append("tier_mapping_soft")

            if instantaneous_tier != "blocked":
                triggered.append("tier_mapping_soft")

            state = prev_map.get(symbol, {"tier": None, "target": None, "streak": 0})
            final_tier, new_target, new_streak, hold_flag = apply_hysteresis(
                instantaneous_tier=instantaneous_tier,
                prev_tier=state.get("tier"),
                prev_target=state.get("target"),
                prev_streak=int(state.get("streak", 0)),
                cfg=cfg,
            )

            prev_tier = state.get("tier")
            tier_changed = prev_tier is not None and final_tier != prev_tier
            short_eligible = int(final_tier != "blocked")
            dominant_rule = dominant_rule_from(triggered)
            reject_reason: Optional[str]
            if final_tier == "blocked":
                if hold_flag and instantaneous_tier != "blocked":
                    reject_reason = "hysteresis_hold"
                else:
                    reject_reason = dominant_rule or "blocked_unspecified"
            elif final_tier in {"medium", "hard"}:
                reject_reason = "tier_mapping_soft"
            else:
                reject_reason = None

            result = {
                "date": pd.Timestamp(date_value),
                "symbol": symbol,
                "run_id": run_id,
                "short_eligible_flag": short_eligible,
                "locate_tier": final_tier,
                "locate_tier_code": TIER_TO_CODE[final_tier],
                "reject_reason": reject_reason,
                "dominant_rule": dominant_rule,
                "all_triggered_rules": serialize_rule_list(triggered),
                "borrow_fee_daily": fee_daily,
                "borrow_fee_annual": fee_annual,
                "availability_score": alpha,
                "htb_flag": bool(htb_flag) if htb_flag is not None and not pd.isna(htb_flag) else False,
                "proxy_quality": proxy_quality if not pd.isna(proxy_quality) else None,
                "fallback_flag": bool(fallback_flag) if fallback_flag is not None and not pd.isna(fallback_flag) else False,
                "override_flag": bool(override_flag),
                "override_reason": override.reason,
                "override_rule_id": override.rule_id,
                "data_quality_flag": serialize_rule_list(data_quality_flags),
                "locate_config_version": cfg.config_version,
                "upstream_proxy_version": upstream_proxy_version,
                "asof_timestamp": asof_timestamp,
                "instantaneous_tier": instantaneous_tier,
                "tier_changed_flag": bool(tier_changed),
                "hysteresis_hold_flag": bool(hold_flag),
                "improvement_target": None if new_target == final_tier else new_target,
                "improvement_streak": int(new_streak),
                "size_bucket": rec.get("size_bucket"),
                "market_cap": rec.get("market_cap"),
                "proxy_source": proxy_source,
                "stress_score": rec.get("stress_score"),
            }
            results.append(result)
            prev_map[symbol] = {"tier": final_tier, "target": result["improvement_target"], "streak": result["improvement_streak"]}

    locate = pd.DataFrame(results)
    if locate.empty:
        raise LocateFilterError("Locate filter produced no rows.")

    # Contract validations
    invalid_blocked = locate[(locate["locate_tier"] == "blocked") & (locate["short_eligible_flag"] != 0)]
    if not invalid_blocked.empty:
        raise LocateFilterError("Contract violation: blocked names cannot be short-eligible.")
    invalid_reject = locate[(locate["short_eligible_flag"] == 0) & (locate["reject_reason"].isna())]
    if not invalid_reject.empty:
        raise LocateFilterError("Contract violation: blocked names must have reject_reason.")

    daily_summary = build_daily_summary(locate)
    global_summary = build_global_summary(locate, daily_summary, cfg)
    return locate, daily_summary, global_summary


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------



def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() == 0:
        return 0.0
    return float(x.mean())



def build_daily_summary(locate: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for date_value, day in locate.groupby("date", sort=True):
        universe_size = int(len(day))
        eligible_share = float(day["short_eligible_flag"].mean()) if universe_size else 0.0
        tier_dist = day["locate_tier"].value_counts(normalize=True)
        reason_dist = day.loc[day["locate_tier"] == "blocked", "reject_reason"].value_counts(normalize=True)
        blocked_small_micro_share = np.nan
        if "size_bucket" in day.columns and day["size_bucket"].notna().any():
            blocked = day.loc[day["locate_tier"] == "blocked"]
            if len(blocked) > 0:
                blocked_small_micro_share = float(
                    blocked["size_bucket"].astype(str).str.lower().isin({"micro", "small"}).mean()
                )
        rows.append(
            {
                "date": pd.Timestamp(date_value),
                "universe_size": universe_size,
                "short_eligible_share": eligible_share,
                "tier_easy_share": float(tier_dist.get("easy", 0.0)),
                "tier_medium_share": float(tier_dist.get("medium", 0.0)),
                "tier_hard_share": float(tier_dist.get("hard", 0.0)),
                "tier_blocked_share": float(tier_dist.get("blocked", 0.0)),
                "blocked_structural_missing_share": float(reason_dist.get("structural_missing", 0.0)),
                "blocked_override_share": float(reason_dist.get("explicit_block_override", 0.0)),
                "blocked_availability_critical_share": float(reason_dist.get("availability_critical", 0.0)),
                "blocked_fee_critical_share": float(reason_dist.get("fee_critical", 0.0)),
                "blocked_htb_severe_share": float(reason_dist.get("htb_severe", 0.0)),
                "blocked_proxy_quality_low_share": float(reason_dist.get("proxy_quality_too_low", 0.0)),
                "blocked_fallback_severe_share": float(reason_dist.get("fallback_severe", 0.0)),
                "override_share": float(day["override_flag"].astype(bool).mean()),
                "low_quality_share": float(day["proxy_quality"].astype(str).str.lower().eq("low").mean()),
                "fallback_share": float(day["fallback_flag"].astype(bool).mean()),
                "hard_concentration_share": float((day["locate_tier"] == "hard").mean()),
                "avg_borrow_fee_daily": _safe_mean(day["borrow_fee_daily"]),
                "avg_availability_score": _safe_mean(day["availability_score"]),
                "tier_changed_share": float(day["tier_changed_flag"].astype(bool).mean()),
                "hysteresis_hold_share": float(day["hysteresis_hold_flag"].astype(bool).mean()),
                "blocked_small_micro_share": blocked_small_micro_share,
                "locate_config_version": first_nonempty(day["locate_config_version"]) or "",
            }
        )
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if out.empty:
        return out
    share_cols = [c for c in out.columns if c.endswith("_share")]
    prev = out[share_cols].shift(1)
    delta = out[share_cols].subtract(prev)
    delta.columns = [f"delta_{c}" for c in delta.columns]
    out = pd.concat([out, delta], axis=1)
    return out



def build_global_summary(locate: pd.DataFrame, daily_summary: pd.DataFrame, cfg: LocateConfig) -> Dict[str, Any]:
    blocked = locate.loc[locate["locate_tier"] == "blocked"]
    summary = {
        "n_rows": int(len(locate)),
        "n_dates": int(locate["date"].nunique()),
        "n_symbols": int(locate["symbol"].nunique()),
        "mean_short_eligible_share": _safe_mean(daily_summary.get("short_eligible_share", pd.Series(dtype=float))),
        "mean_blocked_share": _safe_mean(daily_summary.get("tier_blocked_share", pd.Series(dtype=float))),
        "mean_hard_share": _safe_mean(daily_summary.get("tier_hard_share", pd.Series(dtype=float))),
        "blocked_reason_distribution": {
            str(k): float(v)
            for k, v in blocked["reject_reason"].value_counts(normalize=True, dropna=True).to_dict().items()
        },
        "tier_distribution": {
            str(k): float(v)
            for k, v in locate["locate_tier"].value_counts(normalize=True, dropna=True).to_dict().items()
        },
        "override_share": float(locate["override_flag"].astype(bool).mean()),
        "low_quality_share": float(locate["proxy_quality"].astype(str).str.lower().eq("low").mean()),
        "fallback_share": float(locate["fallback_flag"].astype(bool).mean()),
        "hysteresis_hold_share": float(locate["hysteresis_hold_flag"].astype(bool).mean()),
        "config_version": cfg.config_version,
    }
    return summary


# -----------------------------------------------------------------------------
# Portfolio / backtest helpers
# -----------------------------------------------------------------------------



def max_short_weight_for_tier(tier: str, cfg: LocateConfig) -> float:
    if tier not in TIER_TO_CODE:
        raise LocateFilterError(f"Unknown locate tier: {tier}")
    mult = {
        "easy": cfg.sizing.omega_easy,
        "medium": cfg.sizing.omega_medium,
        "hard": cfg.sizing.omega_hard,
        "blocked": cfg.sizing.omega_blocked,
    }[tier]
    return float(cfg.sizing.base_short_weight * mult)



def apply_locate_sizing(df: pd.DataFrame, tier_col: str = "locate_tier", cfg: LocateConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    out = df.copy()
    out["max_short_weight"] = out[tier_col].astype(str).map(lambda x: max_short_weight_for_tier(str(x), cfg))
    return out



def compute_net_short_return(raw_short_return: pd.Series, borrow_fee_daily: pd.Series, short_position_flag: pd.Series) -> pd.Series:
    raw = pd.to_numeric(raw_short_return, errors="coerce")
    fee = pd.to_numeric(borrow_fee_daily, errors="coerce").fillna(0.0)
    pos = short_position_flag.map(coerce_bool).fillna(False).astype(bool)
    return raw - fee * pos.astype(float)


# -----------------------------------------------------------------------------
# Persistence / CLI orchestration
# -----------------------------------------------------------------------------



def manifest_payload(
    *,
    borrow_proxy_path: Path,
    universe_path: Path,
    previous_state_path: Optional[Path],
    config_path: Optional[Path],
    run_id: str,
    asof_timestamp: str,
    cfg: LocateConfig,
    locate: pd.DataFrame,
    daily_summary: pd.DataFrame,
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "module": "data.borrow.locate_filter",
        "run_id": run_id,
        "asof_timestamp": asof_timestamp,
        "config_version": cfg.config_version,
        "git_revision": git_revision(),
        "python_version": platform.python_version(),
        "inputs": {
            "borrow_proxy_path": str(borrow_proxy_path),
            "universe_path": str(universe_path),
            "previous_state_path": str(previous_state_path) if previous_state_path else None,
            "config_path": str(config_path) if config_path else None,
        },
        "outputs": {
            "locate_daily_rows": int(len(locate)),
            "daily_summary_rows": int(len(daily_summary)),
            "n_dates": int(locate["date"].nunique()) if not locate.empty else 0,
            "n_symbols": int(locate["symbol"].nunique()) if not locate.empty else 0,
            "required_columns_present": all(col in locate.columns for col in OUTPUT_REQUIRED_COLUMNS),
        },
        "summary": dict(summary),
        "config": asdict(cfg),
    }



def persist_outputs(
    locate: pd.DataFrame,
    daily_summary: pd.DataFrame,
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    output_dir: Path,
    run_id: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    locate_path = output_dir / f"locate_filter_daily_{run_id}.parquet"
    daily_path = output_dir / f"locate_filter_summary_daily_{run_id}.parquet"
    summary_path = output_dir / f"locate_filter_summary_{run_id}.json"
    manifest_path = output_dir / f"locate_filter_manifest_{run_id}.json"

    write_parquet(locate, locate_path)
    write_parquet(daily_summary, daily_path)
    write_json(dict(summary), summary_path)
    write_json(dict(manifest), manifest_path)



def run_locate_filter(
    *,
    borrow_proxy_path: Path,
    universe_path: Path,
    output_dir: Path,
    run_id: str,
    config_path: Optional[Path] = None,
    previous_state_path: Optional[Path] = None,
    asof_timestamp: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    cfg = load_config(config_path)
    borrow_proxy = load_borrow_proxy(borrow_proxy_path)
    universe = load_universe(universe_path)
    previous_state = load_previous_state(previous_state_path)
    asof_ts = asof_timestamp or utc_now_iso()
    locate, daily_summary, summary = compute_locate_filter(
        borrow_proxy=borrow_proxy,
        universe=universe,
        cfg=cfg,
        run_id=run_id,
        asof_timestamp=asof_ts,
        previous_state=previous_state,
    )
    manifest = manifest_payload(
        borrow_proxy_path=borrow_proxy_path,
        universe_path=universe_path,
        previous_state_path=previous_state_path,
        config_path=config_path,
        run_id=run_id,
        asof_timestamp=asof_ts,
        cfg=cfg,
        locate=locate,
        daily_summary=daily_summary,
        summary=summary,
    )
    persist_outputs(locate, daily_summary, summary, manifest, output_dir, run_id)
    return locate, daily_summary, summary, manifest



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build locate eligibility / tier decisions from borrow proxy inputs.")
    parser.add_argument("--borrow-proxy-path", required=True, type=Path)
    parser.add_argument("--universe-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True, type=str)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--previous-state-path", type=Path, default=None)
    parser.add_argument("--asof-timestamp", type=str, default=None)
    return parser



def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_locate_filter(
        borrow_proxy_path=args.borrow_proxy_path,
        universe_path=args.universe_path,
        output_dir=args.output_dir,
        run_id=args.run_id,
        config_path=args.config_path,
        previous_state_path=args.previous_state_path,
        asof_timestamp=args.asof_timestamp,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
