from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

LOGGER = logging.getLogger(__name__)

CANONICAL_EVENT_TYPES: tuple[str, ...] = (
    "EARNINGS",
    "CORP_NEWS",
    "OPEN_GAP",
    "TECH_BREAK",
    "VOL_SHOCK",
    "MICROSTRUCTURE_SHOCK",
)
EVENT_ALIASES: dict[str, str] = {
    "earnings": "EARNINGS",
    "results": "EARNINGS",
    "guidance": "EARNINGS",
    "corp_news": "CORP_NEWS",
    "corporate_news": "CORP_NEWS",
    "news": "CORP_NEWS",
    "ma": "CORP_NEWS",
    "m&a": "CORP_NEWS",
    "financing": "CORP_NEWS",
    "open_gap": "OPEN_GAP",
    "gap": "OPEN_GAP",
    "gap_up": "OPEN_GAP",
    "gap_down": "OPEN_GAP",
    "tech_break": "TECH_BREAK",
    "breakout": "TECH_BREAK",
    "breakdown": "TECH_BREAK",
    "reversal": "TECH_BREAK",
    "vol_shock": "VOL_SHOCK",
    "volume_shock": "VOL_SHOCK",
    "volatility_shock": "VOL_SHOCK",
    "microstructure_shock": "MICROSTRUCTURE_SHOCK",
    "halt": "MICROSTRUCTURE_SHOCK",
    "spread_shock": "MICROSTRUCTURE_SHOCK",
    "liquidity_shock": "MICROSTRUCTURE_SHOCK",
}
DEFAULT_HORIZONS: dict[str, tuple[int, ...]] = {
    "EARNINGS": (1, 3, 5, 10),
    "CORP_NEWS": (1, 3, 5, 10),
    "OPEN_GAP": (1, 2, 5),
    "TECH_BREAK": (3, 5, 10),
    "VOL_SHOCK": (1, 3, 5),
    "MICROSTRUCTURE_SHOCK": (1, 2, 3),
}
EXCLUSION_PRIORITY: tuple[str, ...] = (
    "UNKNOWN_EVENT_TYPE",
    "TIMESTAMP_AMBIGUOUS",
    "NOT_IN_UNIVERSE",
    "MISSING_ENTRY_PRICE",
    "MISSING_EXIT_PRICE",
    "INCOMPLETE_OUTCOME_WINDOW",
    "OVERLAP_REJECTED",
    "TRADING_HALT_OR_SUSPENSION",
    "CORPORATE_ACTION_UNSAFE",
    "DELIST_AMBIGUOUS",
    "MISSING_COST_INPUT",
    "BENCHMARK_UNAVAILABLE",
    "QC_REJECTED",
)
ENTRY_EXIT_CONVENTIONS = {
    "open_to_close",
    "open_to_open",
    "close_to_close",
    "event_to_next_close",
    "event_to_next_open",
}
TIMING_HINTS = {
    "pre_market",
    "before_open",
    "at_open",
    "intraday",
    "after_close",
    "post_close",
    "at_close",
    "unknown",
}
DATE_CANDIDATES = ("date", "trade_date", "session_date", "asof_date")
SYMBOL_CANDIDATES = ("symbol", "ticker", "asset", "instrument_id")
EVENT_ID_CANDIDATES = ("event_id", "id", "episode_id")
EVENT_TYPE_CANDIDATES = ("event_type", "event_family", "event_name", "type")
EVENT_SOURCE_CANDIDATES = ("event_source", "source", "provider", "vendor")
EVENT_OCC_CANDIDATES = ("event_timestamp_occ", "timestamp_occ", "occ_ts", "occurrence_ts", "occurred_at")
EVENT_PUB_CANDIDATES = ("event_timestamp_pub", "timestamp_pub", "pub_ts", "published_at", "timestamp")
EVENT_KNOWN_CANDIDATES = ("event_timestamp_known", "timestamp_known", "known_ts", "known_at")
EVENT_TRADABLE_CANDIDATES = (
    "event_timestamp_tradable",
    "timestamp_tradable",
    "tradable_ts",
    "tradeable_ts",
    "t_trad",
)
TIMING_HINT_CANDIDATES = ("timing_hint", "publication_timing", "event_timing", "time_bucket")
PRICE_FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "open": ("adj_open", "open_adj", "open", "px_open", "open_price"),
    "close": ("adj_close", "close_adj", "close", "px_close", "close_price", "price"),
}
UNIVERSE_FLAG_CANDIDATES = (
    "is_eligible",
    "eligible",
    "eligible_flag",
    "in_universe",
    "is_in_universe",
    "universe_member",
)
SECTOR_CANDIDATES = ("sector", "sector_name", "gics_sector", "industry_group")
SIZE_CANDIDATES = ("size_bucket", "mcap_bucket", "market_cap_bucket")
BENCHMARK_MODE_NONE = "none"
SEVERITY_FAIL = "FAIL"
SEVERITY_WARN = "WARN"
SEVERITY_INFO = "INFO"


class EventLabelError(RuntimeError):
    """Base error for event label construction."""


class ConfigError(EventLabelError):
    """Raised when the event label config is invalid."""


class DataContractError(EventLabelError):
    """Raised when an input table violates the expected contract."""


class QCFailure(EventLabelError):
    """Raised when a FAIL severity QC check is triggered."""


@dataclass(frozen=True)
class ClassificationPolicy:
    enabled: bool = True
    mode: str = "ternary_tail"
    within_family: bool = True
    top_quantile: float = 0.8
    bottom_quantile: float = 0.2
    positive_threshold: float = 0.0
    emit_binary: bool = True
    emit_ternary: bool = True


@dataclass(frozen=True)
class CostPolicy:
    net_of_costs: bool = True
    missing_cost_policy: str = "strict_invalidate"
    fallback_entry_bps: float = 0.0
    fallback_exit_bps: float = 0.0
    fallback_annual_borrow_rate: float = 0.0
    fallback_annual_carry_rate: float = 0.0


@dataclass(frozen=True)
class EventLabelConfig:
    canonical_event_types: tuple[str, ...] = CANONICAL_EVENT_TYPES
    event_to_horizons_map: dict[str, tuple[int, ...]] = field(
        default_factory=lambda: {k: tuple(v) for k, v in DEFAULT_HORIZONS.items()}
    )
    timestamp_policy: str = "strict"
    entry_exit_convention: str = "open_to_open"
    decision_lag: int = 0
    overlap_policy: str = "first_event_wins_with_cooldown"
    cooldown_policy: str = "max_horizon"
    cooldown_by_event: dict[str, int] = field(default_factory=dict)
    net_of_costs: bool = True
    abnormal_mode: str = "auto"
    classification_policy: ClassificationPolicy = field(default_factory=ClassificationPolicy)
    benchmark_policy: str = "strict"
    delisting_policy: str = "strict"
    missing_cost_policy: str = "strict_invalidate"
    max_event_staleness_days: int = 30
    min_family_sample: int = 10
    policy_version: str = "1.0.0"
    allow_intraday_immediate: bool = False
    intraday_entry_convention: str = "next_open"
    event_target_base: str = "net_return"
    benchmark_mode_default_corporate: str = "sector_market_adjusted"
    benchmark_mode_default_technical: str = "none"
    strict_unknown_taxonomy: bool = False
    output_dir: str | None = None
    non_canonical_flag: bool = False
    taxonomy_version: str = "1.0.0"
    price_entry_field_override: str | None = None
    price_exit_field_override: str | None = None

    @staticmethod
    def from_mapping(mapping: Mapping[str, Any]) -> "EventLabelConfig":
        raw = dict(mapping)
        canonical_event_types = raw.get("canonical_event_types", CANONICAL_EVENT_TYPES)
        if not isinstance(canonical_event_types, Sequence) or isinstance(canonical_event_types, (str, bytes)):
            raise ConfigError("canonical_event_types must be a sequence")
        canonical_tuple = tuple(str(x).upper() for x in canonical_event_types)
        event_to_horizons_raw = raw.get("event_to_horizons_map", DEFAULT_HORIZONS)
        if not isinstance(event_to_horizons_raw, Mapping):
            raise ConfigError("event_to_horizons_map must be a mapping")
        event_to_horizons_map: dict[str, tuple[int, ...]] = {}
        for key, value in event_to_horizons_raw.items():
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ConfigError(f"Horizons for event family {key!r} must be a sequence")
            horizons = tuple(sorted({int(v) for v in value}))
            if not horizons or min(horizons) <= 0:
                raise ConfigError(f"Horizons for event family {key!r} must be positive integers")
            event_to_horizons_map[str(key).upper()] = horizons
        entry_exit_convention = str(raw.get("entry_exit_convention", "open_to_open"))
        if entry_exit_convention not in ENTRY_EXIT_CONVENTIONS:
            raise ConfigError(
                "entry_exit_convention must be one of " + ", ".join(sorted(ENTRY_EXIT_CONVENTIONS))
            )
        decision_lag = int(raw.get("decision_lag", 0))
        if decision_lag < 0:
            raise ConfigError("decision_lag must be >= 0")
        overlap_policy = str(raw.get("overlap_policy", "first_event_wins_with_cooldown"))
        if overlap_policy not in {"first_event_wins_with_cooldown", "merge_same_episode"}:
            raise ConfigError("Unsupported overlap_policy")
        classification_raw = raw.get("classification_policy", {})
        if classification_raw is None:
            classification_raw = {}
        if not isinstance(classification_raw, Mapping):
            raise ConfigError("classification_policy must be a mapping when provided")
        classification_policy = ClassificationPolicy(
            enabled=bool(classification_raw.get("enabled", True)),
            mode=str(classification_raw.get("mode", "ternary_tail")),
            within_family=bool(classification_raw.get("within_family", True)),
            top_quantile=float(classification_raw.get("top_quantile", 0.8)),
            bottom_quantile=float(classification_raw.get("bottom_quantile", 0.2)),
            positive_threshold=float(classification_raw.get("positive_threshold", 0.0)),
            emit_binary=bool(classification_raw.get("emit_binary", True)),
            emit_ternary=bool(classification_raw.get("emit_ternary", True)),
        )
        if not 0.0 <= classification_policy.bottom_quantile < classification_policy.top_quantile <= 1.0:
            raise ConfigError("Classification quantiles must satisfy 0 <= bottom < top <= 1")
        cooldown_raw = raw.get("cooldown_policy", "max_horizon")
        cooldown_by_event_raw = raw.get("cooldown_by_event", {})
        if not isinstance(cooldown_by_event_raw, Mapping):
            raise ConfigError("cooldown_by_event must be a mapping")
        cooldown_by_event = {str(k).upper(): int(v) for k, v in cooldown_by_event_raw.items()}
        abnormal_mode = str(raw.get("abnormal_mode", "auto"))
        if abnormal_mode not in {
            "auto",
            "none",
            "market_adjusted",
            "sector_adjusted",
            "sector_market_adjusted",
            "matched_control",
            "factor_adjusted",
        }:
            raise ConfigError("Unsupported abnormal_mode")
        return EventLabelConfig(
            canonical_event_types=canonical_tuple,
            event_to_horizons_map=event_to_horizons_map,
            timestamp_policy=str(raw.get("timestamp_policy", "strict")),
            entry_exit_convention=entry_exit_convention,
            decision_lag=decision_lag,
            overlap_policy=overlap_policy,
            cooldown_policy=str(cooldown_raw),
            cooldown_by_event=cooldown_by_event,
            net_of_costs=bool(raw.get("net_of_costs", True)),
            abnormal_mode=abnormal_mode,
            classification_policy=classification_policy,
            benchmark_policy=str(raw.get("benchmark_policy", "strict")),
            delisting_policy=str(raw.get("delisting_policy", "strict")),
            missing_cost_policy=str(raw.get("missing_cost_policy", "strict_invalidate")),
            max_event_staleness_days=int(raw.get("max_event_staleness", raw.get("max_event_staleness_days", 30))),
            min_family_sample=int(raw.get("min_family_sample", 10)),
            policy_version=str(raw.get("policy_version", "1.0.0")),
            allow_intraday_immediate=bool(raw.get("allow_intraday_immediate", False)),
            intraday_entry_convention=str(raw.get("intraday_entry_convention", "next_open")),
            event_target_base=str(raw.get("event_target_base", "net_return")),
            benchmark_mode_default_corporate=str(
                raw.get("benchmark_mode_default_corporate", "sector_market_adjusted")
            ),
            benchmark_mode_default_technical=str(raw.get("benchmark_mode_default_technical", "none")),
            strict_unknown_taxonomy=bool(raw.get("strict_unknown_taxonomy", False)),
            output_dir=_none_if_blank(raw.get("output_dir")),
            non_canonical_flag=bool(raw.get("non_canonical_flag", False)),
            taxonomy_version=str(raw.get("taxonomy_version", "1.0.0")),
            price_entry_field_override=_none_if_blank(raw.get("price_entry_field_override")),
            price_exit_field_override=_none_if_blank(raw.get("price_exit_field_override")),
        )


@dataclass(frozen=True)
class QCRecord:
    check_name: str
    severity: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)


def _none_if_blank(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def _load_mapping(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config not found: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    suffix = file_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ConfigError("PyYAML is required to read YAML config files")
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        if yaml is not None:
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError("Config file must parse into a mapping")
    return dict(data)


def _read_table(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(file_path)
    if suffix in {".json", ".jsonl"}:
        try:
            return pd.read_json(file_path, lines=suffix == ".jsonl")
        except ValueError:
            return pd.read_json(file_path)
    if suffix == ".feather":
        return pd.read_feather(file_path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(file_path)
    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def _write_table(df: pd.DataFrame, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix == ".parquet":
        try:
            df.to_parquet(out, index=False)
            return out
        except Exception:
            fallback = out.with_suffix(".csv")
            df.to_csv(fallback, index=False)
            return fallback
    if suffix == ".csv":
        df.to_csv(out, index=False)
        return out
    if suffix == ".json":
        df.to_json(out, orient="records", date_format="iso")
        return out
    raise ValueError(f"Unsupported output suffix: {suffix}")


def _write_json(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_stable_json_dumps(payload), encoding="utf-8")
    return out


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def _find_first_column(df: pd.DataFrame, candidates: Iterable[str], *, required: bool = True) -> str | None:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    if required:
        raise DataContractError(f"Missing required column among candidates: {tuple(candidates)}")
    return None


def _to_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def _normalize_symbol(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _series_has_time_info(series: pd.Series) -> pd.Series:
    ser = _to_timestamp(series)
    return ser.notna() & (
        (ser.dt.hour != 0) | (ser.dt.minute != 0) | (ser.dt.second != 0) | (ser.dt.microsecond != 0)
    )


@dataclass(frozen=True)
class TradingCalendar:
    session_dates: pd.DatetimeIndex
    session_open: time = time(9, 30)
    session_close: time = time(16, 0)

    @staticmethod
    def from_sources(prices: pd.DataFrame, trading_calendar_path: str | Path | None = None) -> "TradingCalendar":
        if trading_calendar_path is not None:
            cal_df = _normalize_columns(_read_table(trading_calendar_path))
            date_col = _find_first_column(cal_df, DATE_CANDIDATES)
            session_dates = pd.to_datetime(cal_df[date_col], errors="coerce").dt.normalize().dropna().drop_duplicates().sort_values()
            if session_dates.empty:
                raise DataContractError("trading_calendar_path resolved to an empty session set")
            open_candidates = ("market_open", "open_time", "session_open")
            close_candidates = ("market_close", "close_time", "session_close")
            open_col = _find_first_column(cal_df, open_candidates, required=False)
            close_col = _find_first_column(cal_df, close_candidates, required=False)
            open_time_value = time(9, 30)
            close_time_value = time(16, 0)
            if open_col is not None and cal_df[open_col].notna().any():
                open_time_value = _coerce_time(cal_df[open_col].dropna().iloc[0])
            if close_col is not None and cal_df[close_col].notna().any():
                close_time_value = _coerce_time(cal_df[close_col].dropna().iloc[0])
            return TradingCalendar(pd.DatetimeIndex(session_dates), open_time_value, close_time_value)
        prices = _normalize_columns(prices)
        date_col = _find_first_column(prices, DATE_CANDIDATES)
        session_dates = pd.to_datetime(prices[date_col], errors="coerce").dt.normalize().dropna().drop_duplicates().sort_values()
        if session_dates.empty:
            raise DataContractError("Could not derive trading sessions from prices data")
        return TradingCalendar(pd.DatetimeIndex(session_dates), time(9, 30), time(16, 0))

    def has_session(self, value: pd.Timestamp) -> bool:
        ts = pd.Timestamp(value).normalize()
        return bool(ts in self.session_dates)

    def next_session(self, value: pd.Timestamp, offset: int = 0) -> pd.Timestamp | pd.NaT:
        ts = pd.Timestamp(value).normalize()
        pos = self.session_dates.searchsorted(ts, side="left")
        if pos >= len(self.session_dates):
            return pd.NaT
        if self.session_dates[pos] < ts:
            pos += 1
        pos += offset
        if pos < 0 or pos >= len(self.session_dates):
            return pd.NaT
        return self.session_dates[pos]

    def same_or_next_session(self, value: pd.Timestamp) -> pd.Timestamp | pd.NaT:
        ts = pd.Timestamp(value).normalize()
        pos = self.session_dates.searchsorted(ts, side="left")
        if pos >= len(self.session_dates):
            return pd.NaT
        return self.session_dates[pos]

    def prev_session(self, value: pd.Timestamp, offset: int = 1) -> pd.Timestamp | pd.NaT:
        ts = pd.Timestamp(value).normalize()
        pos = self.session_dates.searchsorted(ts, side="left") - offset
        if pos < 0 or pos >= len(self.session_dates):
            return pd.NaT
        return self.session_dates[pos]

    def shift_session(self, session_date: pd.Timestamp, periods: int) -> pd.Timestamp | pd.NaT:
        if pd.isna(session_date):
            return pd.NaT
        ts = pd.Timestamp(session_date).normalize()
        pos = self.session_dates.get_indexer([ts])[0]
        if pos < 0:
            return self.same_or_next_session(ts)
        new_pos = pos + periods
        if new_pos < 0 or new_pos >= len(self.session_dates):
            return pd.NaT
        return self.session_dates[new_pos]

    def session_timestamp(self, session_date: pd.Timestamp, which: str) -> pd.Timestamp | pd.NaT:
        if pd.isna(session_date):
            return pd.NaT
        session_date = pd.Timestamp(session_date).normalize()
        if which == "open":
            return session_date + pd.Timedelta(hours=self.session_open.hour, minutes=self.session_open.minute)
        if which == "close":
            return session_date + pd.Timedelta(hours=self.session_close.hour, minutes=self.session_close.minute)
        raise ValueError(f"Unsupported session timestamp kind: {which}")


def _coerce_time(value: Any) -> time:
    if isinstance(value, time):
        return value
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return time(9, 30)
    return time(int(ts.hour), int(ts.minute), int(ts.second))


def load_event_label_config(path: str | Path) -> EventLabelConfig:
    return EventLabelConfig.from_mapping(_load_mapping(path))


def load_prices_panel(path: str | Path) -> tuple[pd.DataFrame, dict[str, str]]:
    prices = _normalize_columns(_read_table(path))
    symbol_col = _find_first_column(prices, SYMBOL_CANDIDATES)
    date_col = _find_first_column(prices, DATE_CANDIDATES)
    prices[symbol_col] = _normalize_symbol(prices[symbol_col])
    prices[date_col] = pd.to_datetime(prices[date_col], errors="coerce").dt.normalize()
    if prices[date_col].isna().any():
        raise DataContractError("prices_pit_path contains invalid dates")
    price_fields: dict[str, str] = {}
    for logical_name, candidates in PRICE_FIELD_CANDIDATES.items():
        col = _find_first_column(prices, candidates, required=logical_name == "close")
        if col is None and logical_name == "open":
            col = price_fields.get("close")
        if col is None:
            raise DataContractError(f"Missing price field for {logical_name}")
        prices[col] = pd.to_numeric(prices[col], errors="coerce")
        price_fields[logical_name] = col
    prices = prices.sort_values([symbol_col, date_col]).drop_duplicates([symbol_col, date_col], keep="last")
    return prices, {"symbol": symbol_col, "date": date_col, **price_fields}


def load_universe_history(path: str | Path) -> pd.DataFrame:
    df = _normalize_columns(_read_table(path))
    symbol_col = _find_first_column(df, SYMBOL_CANDIDATES)
    df[symbol_col] = _normalize_symbol(df[symbol_col])
    return df


def load_event_source(path: str | Path) -> pd.DataFrame:
    events = _normalize_columns(_read_table(path))
    if events.empty:
        raise DataContractError("event_source_path resolved to an empty table")
    return events


def canonicalize_event_types(events: pd.DataFrame, config: EventLabelConfig) -> pd.DataFrame:
    df = events.copy()
    event_type_col = _find_first_column(df, EVENT_TYPE_CANDIDATES)
    event_id_col = _find_first_column(df, EVENT_ID_CANDIDATES, required=False)
    if event_id_col is None:
        df["event_id"] = [f"event_{i:08d}" for i in range(len(df))]
        event_id_col = "event_id"
    else:
        df[event_id_col] = df[event_id_col].astype(str)
    symbol_col = _find_first_column(df, SYMBOL_CANDIDATES)
    df[symbol_col] = _normalize_symbol(df[symbol_col])
    raw_type = df[event_type_col].astype(str).str.strip()
    canonical = raw_type.str.lower().map(EVENT_ALIASES).fillna(raw_type.str.upper())
    df["event_id"] = df[event_id_col].astype(str)
    df["symbol"] = df[symbol_col]
    df["event_type_raw"] = raw_type
    df["event_type"] = canonical
    df["event_family_status"] = np.where(
        df["event_type"].isin(config.canonical_event_types), "official", "experimental"
    )
    unknown_mask = ~df["event_type"].isin(config.canonical_event_types)
    df["event_source"] = (
        df[_find_first_column(df, EVENT_SOURCE_CANDIDATES, required=False)]
        if _find_first_column(df, EVENT_SOURCE_CANDIDATES, required=False) is not None
        else "unknown"
    )
    df["event_source"] = df["event_source"].astype(str)
    df["event_exclusion_reason"] = np.where(unknown_mask, "UNKNOWN_EVENT_TYPE", pd.NA)
    if config.strict_unknown_taxonomy and unknown_mask.any():
        raise DataContractError(
            f"Encountered non-canonical event types: {sorted(df.loc[unknown_mask, 'event_type'].unique())}"
        )
    return df


def resolve_event_timestamps(
    events: pd.DataFrame,
    calendar: TradingCalendar,
    config: EventLabelConfig,
) -> pd.DataFrame:
    df = events.copy()
    occ_col = _find_first_column(df, EVENT_OCC_CANDIDATES, required=False)
    pub_col = _find_first_column(df, EVENT_PUB_CANDIDATES, required=False)
    known_col = _find_first_column(df, EVENT_KNOWN_CANDIDATES, required=False)
    tradable_col = _find_first_column(df, EVENT_TRADABLE_CANDIDATES, required=False)
    timing_hint_col = _find_first_column(df, TIMING_HINT_CANDIDATES, required=False)

    if occ_col is None and pub_col is None and known_col is None:
        date_col = _find_first_column(df, DATE_CANDIDATES, required=False)
        if date_col is None:
            raise DataContractError("Event source lacks timestamp/date information")
        df["event_timestamp_occ"] = _to_timestamp(df[date_col])
    else:
        df["event_timestamp_occ"] = _to_timestamp(df[occ_col]) if occ_col is not None else pd.NaT
    df["event_timestamp_pub"] = (
        _to_timestamp(df[pub_col]) if pub_col is not None else df["event_timestamp_occ"]
    )
    df["event_timestamp_known"] = (
        _to_timestamp(df[known_col]) if known_col is not None else df["event_timestamp_pub"]
    )
    if occ_col is None:
        df["event_timestamp_occ"] = df["event_timestamp_pub"]
    if tradable_col is not None:
        df["event_timestamp_tradable"] = _to_timestamp(df[tradable_col])
    else:
        df["event_timestamp_tradable"] = pd.NaT
    timing_hint = (
        df[timing_hint_col].astype(str).str.strip().str.lower() if timing_hint_col is not None else pd.Series("", index=df.index)
    )
    timing_hint = timing_hint.where(timing_hint.isin(TIMING_HINTS), "")
    df["timing_hint"] = timing_hint.replace("", pd.NA)

    has_known_time = _series_has_time_info(df["event_timestamp_known"])
    inferred_tradable = []
    inferred_tradable_session = []
    tradable_reason = []
    exclusion_reason = df.get("event_exclusion_reason", pd.Series(pd.NA, index=df.index)).copy()

    for idx, row in df.iterrows():
        current_reason = exclusion_reason.loc[idx]
        known_ts = row["event_timestamp_known"]
        explicit_tradable = row["event_timestamp_tradable"]
        hint = row["timing_hint"] if pd.notna(row["timing_hint"]) else None
        if pd.notna(explicit_tradable):
            tradable_ts = explicit_tradable
            tradable_session = calendar.same_or_next_session(explicit_tradable)
            reason = "explicit"
        else:
            tradable_ts = pd.NaT
            tradable_session = pd.NaT
            reason = "derived"
            if pd.isna(known_ts):
                current_reason = _pick_reason(current_reason, "TIMESTAMP_AMBIGUOUS")
            else:
                normalized_date = pd.Timestamp(known_ts).normalize()
                same_day_open = calendar.session_timestamp(normalized_date, "open")
                next_day_open = calendar.session_timestamp(calendar.next_session(normalized_date, 1), "open")
                same_day_close = calendar.session_timestamp(normalized_date, "close")
                if hint in {"pre_market", "before_open", "at_open"}:
                    tradable_ts = same_day_open
                    tradable_session = normalized_date if calendar.has_session(normalized_date) else calendar.same_or_next_session(normalized_date)
                    reason = hint
                elif hint in {"after_close", "post_close", "at_close"}:
                    tradable_session = calendar.next_session(normalized_date, 1)
                    tradable_ts = calendar.session_timestamp(tradable_session, "open")
                    reason = hint
                elif hint == "intraday":
                    if config.allow_intraday_immediate and config.intraday_entry_convention == "immediate":
                        tradable_ts = known_ts
                        tradable_session = calendar.same_or_next_session(normalized_date)
                        reason = "intraday_immediate"
                    else:
                        tradable_session = calendar.next_session(normalized_date, 1)
                        tradable_ts = calendar.session_timestamp(tradable_session, "open")
                        reason = "intraday_next_open"
                elif bool(has_known_time.loc[idx]):
                    session_open_ts = same_day_open
                    session_close_ts = same_day_close
                    if pd.isna(session_open_ts) or pd.isna(session_close_ts):
                        current_reason = _pick_reason(current_reason, "TIMESTAMP_AMBIGUOUS")
                    elif known_ts <= session_open_ts:
                        tradable_session = normalized_date if calendar.has_session(normalized_date) else calendar.same_or_next_session(normalized_date)
                        tradable_ts = calendar.session_timestamp(tradable_session, "open")
                        reason = "known_before_open"
                    elif known_ts >= session_close_ts:
                        tradable_session = calendar.next_session(normalized_date, 1)
                        tradable_ts = calendar.session_timestamp(tradable_session, "open")
                        reason = "known_after_close"
                    else:
                        if config.allow_intraday_immediate and config.intraday_entry_convention == "immediate":
                            tradable_ts = known_ts
                            tradable_session = normalized_date if calendar.has_session(normalized_date) else calendar.same_or_next_session(normalized_date)
                            reason = "known_intraday_immediate"
                        else:
                            tradable_session = calendar.next_session(normalized_date, 1)
                            tradable_ts = calendar.session_timestamp(tradable_session, "open")
                            reason = "known_intraday_next_open"
                else:
                    if config.timestamp_policy == "strict":
                        current_reason = _pick_reason(current_reason, "TIMESTAMP_AMBIGUOUS")
                    else:
                        event_type = str(row["event_type"])
                        if event_type in {"TECH_BREAK", "VOL_SHOCK", "MICROSTRUCTURE_SHOCK"}:
                            tradable_session = calendar.next_session(normalized_date, 1)
                            tradable_ts = calendar.session_timestamp(tradable_session, "open")
                            reason = "date_only_next_open"
                        elif event_type == "OPEN_GAP":
                            tradable_session = normalized_date if calendar.has_session(normalized_date) else calendar.same_or_next_session(normalized_date)
                            tradable_ts = calendar.session_timestamp(tradable_session, "open")
                            reason = "date_only_gap_same_open"
                        else:
                            current_reason = _pick_reason(current_reason, "TIMESTAMP_AMBIGUOUS")
        if pd.notna(tradable_ts) and pd.notna(known_ts) and tradable_ts < known_ts:
            current_reason = _pick_reason(current_reason, "TIMESTAMP_AMBIGUOUS")
        if pd.notna(tradable_session) and config.decision_lag > 0:
            tradable_session = calendar.shift_session(tradable_session, config.decision_lag)
            tradable_ts = calendar.session_timestamp(tradable_session, "open")
            reason = f"{reason}_decision_lag_{config.decision_lag}"
        inferred_tradable.append(tradable_ts)
        inferred_tradable_session.append(tradable_session)
        tradable_reason.append(reason)
        exclusion_reason.loc[idx] = current_reason

    df["event_timestamp_tradable"] = inferred_tradable
    df["event_tradable_session"] = pd.to_datetime(pd.Series(inferred_tradable_session, index=df.index), errors="coerce").dt.normalize()
    df["tradability_resolution"] = tradable_reason
    df["event_exclusion_reason"] = exclusion_reason
    return df


def filter_by_universe(events: pd.DataFrame, universe_history: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    universe = universe_history.copy()
    symbol_col = _find_first_column(universe, SYMBOL_CANDIDATES)
    universe[symbol_col] = _normalize_symbol(universe[symbol_col])
    reason = df["event_exclusion_reason"].copy()

    date_col = _find_first_column(universe, DATE_CANDIDATES, required=False)
    start_col = _find_first_column(universe, ("start_date", "effective_from", "from_date"), required=False)
    end_col = _find_first_column(universe, ("end_date", "effective_to", "to_date"), required=False)
    flag_col = _find_first_column(universe, UNIVERSE_FLAG_CANDIDATES, required=False)

    eligible_map: dict[tuple[str, pd.Timestamp], bool] = {}
    if date_col is not None:
        universe[date_col] = pd.to_datetime(universe[date_col], errors="coerce").dt.normalize()
        if flag_col is not None:
            eligible_values = universe[flag_col].astype(bool)
        else:
            eligible_values = pd.Series(True, index=universe.index)
        for sym, dt, flag in zip(universe[symbol_col], universe[date_col], eligible_values):
            if pd.notna(dt):
                eligible_map[(sym, pd.Timestamp(dt))] = bool(flag)
        in_universe = []
        for _, row in df.iterrows():
            session_dt = pd.Timestamp(row["event_tradable_session"]).normalize() if pd.notna(row["event_tradable_session"]) else pd.NaT
            in_universe.append(bool(eligible_map.get((row["symbol"], session_dt), False)))
        df["in_universe"] = in_universe
    elif start_col is not None and end_col is not None:
        universe[start_col] = pd.to_datetime(universe[start_col], errors="coerce").dt.normalize()
        universe[end_col] = pd.to_datetime(universe[end_col], errors="coerce").dt.normalize()
        in_universe = []
        for _, row in df.iterrows():
            session_dt = pd.Timestamp(row["event_tradable_session"]).normalize() if pd.notna(row["event_tradable_session"]) else pd.NaT
            mask = (
                (universe[symbol_col] == row["symbol"]) &
                (universe[start_col] <= session_dt) &
                (universe[end_col] >= session_dt)
            )
            if flag_col is not None:
                mask &= universe[flag_col].astype(bool)
            in_universe.append(bool(mask.any()))
        df["in_universe"] = in_universe
    else:
        universe_symbols = set(universe[symbol_col].unique())
        df["in_universe"] = df["symbol"].isin(universe_symbols)

    df.loc[~df["in_universe"], "event_exclusion_reason"] = [
        _pick_reason(cur, "NOT_IN_UNIVERSE") for cur in df.loc[~df["in_universe"], "event_exclusion_reason"]
    ]
    return df


def attach_event_horizons(events: pd.DataFrame, config: EventLabelConfig) -> pd.DataFrame:
    df = events.copy()
    df["event_horizons"] = df["event_type"].map(config.event_to_horizons_map)
    missing_horizons = df["event_horizons"].isna()
    if missing_horizons.any():
        df.loc[missing_horizons, "event_horizons"] = [tuple()] * int(missing_horizons.sum())
        df.loc[missing_horizons, "event_family_status"] = "experimental"
        df.loc[missing_horizons, "event_exclusion_reason"] = [
            _pick_reason(cur, "UNKNOWN_EVENT_TYPE") for cur in df.loc[missing_horizons, "event_exclusion_reason"]
        ]
    df["cooldown_days"] = [
        _resolve_cooldown_days(event_type, horizons, config) for event_type, horizons in zip(df["event_type"], df["event_horizons"])
    ]
    return df


def _resolve_cooldown_days(event_type: str, horizons: Sequence[int], config: EventLabelConfig) -> int:
    event_type = str(event_type).upper()
    if event_type in config.cooldown_by_event:
        return max(0, int(config.cooldown_by_event[event_type]))
    if config.cooldown_policy == "max_horizon":
        return int(max(horizons) if horizons else 0)
    if config.cooldown_policy == "min_horizon":
        return int(min(horizons) if horizons else 0)
    if config.cooldown_policy.isdigit():
        return int(config.cooldown_policy)
    return int(max(horizons) if horizons else 0)


def resolve_overlaps(events: pd.DataFrame, calendar: TradingCalendar, config: EventLabelConfig) -> pd.DataFrame:
    df = events.copy().sort_values(["symbol", "event_timestamp_tradable", "event_id"], kind="stable")
    episode_group_ids: list[str] = []
    rejection_flags: list[bool] = []
    active_until: dict[str, pd.Timestamp] = {}
    active_group: dict[str, str] = {}
    rejection_reason = df["event_exclusion_reason"].copy()

    for _, row in df.iterrows():
        symbol = row["symbol"]
        tradable_session = row["event_tradable_session"]
        group_id = row["event_id"]
        reject = False
        if pd.notna(tradable_session) and symbol in active_until:
            if pd.Timestamp(tradable_session) <= active_until[symbol]:
                if config.overlap_policy == "merge_same_episode":
                    group_id = active_group[symbol]
                else:
                    reject = True
        if not reject and pd.notna(tradable_session):
            cooldown_end = calendar.shift_session(tradable_session, int(row["cooldown_days"]))
            if pd.notna(cooldown_end):
                active_until[symbol] = pd.Timestamp(cooldown_end)
                active_group[symbol] = group_id
        if reject:
            rejection_reason.loc[row.name] = _pick_reason(rejection_reason.loc[row.name], "OVERLAP_REJECTED")
        episode_group_ids.append(group_id)
        rejection_flags.append(reject)

    df["episode_group_id"] = episode_group_ids
    df["overlap_rejected"] = rejection_flags
    df["event_exclusion_reason"] = rejection_reason
    return df


def build_price_lookup(prices: pd.DataFrame, schema: Mapping[str, str]) -> tuple[pd.DataFrame, dict[tuple[str, pd.Timestamp], dict[str, float]]]:
    symbol_col = schema["symbol"]
    date_col = schema["date"]
    price_lookup: dict[tuple[str, pd.Timestamp], dict[str, float]] = {}
    for _, row in prices.iterrows():
        key = (row[symbol_col], pd.Timestamp(row[date_col]).normalize())
        price_lookup[key] = {
            "open": float(row[schema["open"]]) if pd.notna(row[schema["open"]]) else np.nan,
            "close": float(row[schema["close"]]) if pd.notna(row[schema["close"]]) else np.nan,
        }
    return prices, price_lookup


def compute_event_returns(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    price_schema: Mapping[str, str],
    calendar: TradingCalendar,
    config: EventLabelConfig,
    *,
    execution_costs: pd.DataFrame | None = None,
    borrow_costs: pd.DataFrame | None = None,
    carry_costs: pd.DataFrame | None = None,
    delisting_returns: pd.DataFrame | None = None,
    corporate_actions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    _, price_lookup = build_price_lookup(prices, price_schema)
    execution_lookup = _build_cost_lookup(execution_costs) if execution_costs is not None else {}
    borrow_lookup = _build_rate_lookup(borrow_costs) if borrow_costs is not None else {}
    carry_lookup = _build_rate_lookup(carry_costs) if carry_costs is not None else {}
    delisting_lookup = _build_delisting_lookup(delisting_returns) if delisting_returns is not None else {}
    corp_actions_lookup = _build_corporate_action_lookup(corporate_actions) if corporate_actions is not None else {}

    rows: list[dict[str, Any]] = []
    for _, row in events.iterrows():
        horizons = tuple(int(h) for h in row["event_horizons"])
        if not horizons:
            rows.append(_base_episode_row(row, horizon=np.nan))
            continue
        for horizon in horizons:
            episode = _base_episode_row(row, horizon=horizon)
            exclusion_reason = episode["event_exclusion_reason"]
            if pd.isna(row["event_tradable_session"]):
                episode["event_exclusion_reason"] = _pick_reason(exclusion_reason, "TIMESTAMP_AMBIGUOUS")
                rows.append(episode)
                continue
            entry_session, exit_session, entry_field, exit_field = _resolve_entry_exit_sessions(
                pd.Timestamp(row["event_tradable_session"]),
                int(horizon),
                calendar,
                config,
                str(row["event_type"]),
            )
            episode["entry_session"] = entry_session
            episode["exit_session"] = exit_session
            episode["entry_convention"] = entry_field
            episode["exit_convention"] = exit_field
            episode["event_start"] = calendar.session_timestamp(entry_session, "open") if pd.notna(entry_session) else pd.NaT
            episode["event_end"] = (
                calendar.session_timestamp(exit_session, "close" if exit_field == "close" else "open")
                if pd.notna(exit_session)
                else pd.NaT
            )
            if pd.isna(entry_session) or pd.isna(exit_session):
                episode["event_exclusion_reason"] = _pick_reason(exclusion_reason, "INCOMPLETE_OUTCOME_WINDOW")
                rows.append(episode)
                continue
            entry_px = _lookup_price(price_lookup, row["symbol"], entry_session, entry_field)
            exit_px = _lookup_price(price_lookup, row["symbol"], exit_session, exit_field)
            episode["entry_px"] = entry_px
            episode["exit_px"] = exit_px
            if np.isnan(entry_px):
                episode["event_exclusion_reason"] = _pick_reason(exclusion_reason, "MISSING_ENTRY_PRICE")
                rows.append(episode)
                continue
            if np.isnan(exit_px):
                delisting_ret = _lookup_delisting_return(delisting_lookup, row["symbol"], exit_session)
                if delisting_ret is not None:
                    episode["delisting_return_used"] = True
                    episode["event_ret_gross"] = float(delisting_ret)
                else:
                    episode["event_exclusion_reason"] = _pick_reason(exclusion_reason, "MISSING_EXIT_PRICE")
                    rows.append(episode)
                    continue
            if pd.notna(corp_actions_lookup.get((row["symbol"], pd.Timestamp(entry_session).normalize()), np.nan)):
                if bool(corp_actions_lookup[(row["symbol"], pd.Timestamp(entry_session).normalize())]):
                    episode["event_exclusion_reason"] = _pick_reason(exclusion_reason, "CORPORATE_ACTION_UNSAFE")
            if pd.isna(episode["event_ret_gross"]):
                episode["event_ret_gross"] = float(exit_px / entry_px - 1.0)
            entry_cost, exit_cost = _lookup_execution_costs(execution_lookup, row["symbol"], entry_session, exit_session)
            borrow_rate = _lookup_rate(borrow_lookup, row["symbol"], entry_session, default=0.0)
            carry_rate = _lookup_rate(carry_lookup, row["symbol"], entry_session, default=0.0)
            if execution_costs is None:
                entry_cost = config_missing_entry_cost(config)
                exit_cost = config_missing_exit_cost(config)
            if borrow_costs is None:
                borrow_rate = 0.0
            if carry_costs is None:
                carry_rate = 0.0
            episode["cost_entry"] = entry_cost
            episode["cost_exit"] = exit_cost
            episode["cost_carry"] = _carry_cost_for_horizon(borrow_rate, carry_rate, int(horizon))
            if config.net_of_costs:
                missing_components = [
                    np.isnan(entry_cost),
                    np.isnan(exit_cost),
                    np.isnan(episode["cost_carry"]),
                ]
                if any(missing_components) and config.missing_cost_policy == "strict_invalidate":
                    episode["event_exclusion_reason"] = _pick_reason(episode["event_exclusion_reason"], "MISSING_COST_INPUT")
                else:
                    entry_cost = 0.0 if np.isnan(entry_cost) else entry_cost
                    exit_cost = 0.0 if np.isnan(exit_cost) else exit_cost
                    carry_cost = 0.0 if np.isnan(episode["cost_carry"] ) else episode["cost_carry"]
                    episode["event_ret_net"] = float(episode["event_ret_gross"] - entry_cost - exit_cost - carry_cost)
            else:
                episode["event_ret_net"] = float(episode["event_ret_gross"])
            rows.append(episode)
    return pd.DataFrame(rows)


def config_missing_entry_cost(config: EventLabelConfig) -> float:
    return 0.0


def config_missing_exit_cost(config: EventLabelConfig) -> float:
    return 0.0


def _base_episode_row(row: pd.Series, horizon: float | int) -> dict[str, Any]:
    return {
        "event_id": row["event_id"],
        "episode_group_id": row.get("episode_group_id", row["event_id"]),
        "symbol": row["symbol"],
        "event_type": row["event_type"],
        "event_family_status": row.get("event_family_status", "official"),
        "event_source": row.get("event_source", "unknown"),
        "event_timestamp_occ": row.get("event_timestamp_occ", pd.NaT),
        "event_timestamp_pub": row.get("event_timestamp_pub", pd.NaT),
        "event_timestamp_known": row.get("event_timestamp_known", pd.NaT),
        "event_timestamp_tradable": row.get("event_timestamp_tradable", pd.NaT),
        "event_tradable_session": row.get("event_tradable_session", pd.NaT),
        "timing_hint": row.get("timing_hint", pd.NA),
        "tradability_resolution": row.get("tradability_resolution", pd.NA),
        "horizon_days": horizon,
        "entry_px": np.nan,
        "exit_px": np.nan,
        "event_ret_gross": np.nan,
        "event_ret_net": np.nan,
        "event_alpha": np.nan,
        "event_zscore": np.nan,
        "label_event_cont": np.nan,
        "label_event_bin": np.nan,
        "label_event_tri": np.nan,
        "cost_entry": np.nan,
        "cost_exit": np.nan,
        "cost_carry": np.nan,
        "delisting_return_used": False,
        "event_exclusion_reason": row.get("event_exclusion_reason", pd.NA),
        "event_label_valid_flag": False,
        "event_start": pd.NaT,
        "event_end": pd.NaT,
        "run_id": pd.NA,
    }


def _resolve_entry_exit_sessions(
    tradable_session: pd.Timestamp,
    horizon: int,
    calendar: TradingCalendar,
    config: EventLabelConfig,
    event_type: str,
) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT, str, str]:
    convention = config.entry_exit_convention
    if convention == "open_to_close":
        entry_session = tradable_session
        exit_session = calendar.shift_session(tradable_session, max(horizon - 1, 0))
        return entry_session, exit_session, "open", "close"
    if convention == "open_to_open":
        entry_session = tradable_session
        exit_session = calendar.shift_session(tradable_session, horizon)
        return entry_session, exit_session, "open", "open"
    if convention == "close_to_close":
        entry_session = tradable_session
        exit_session = calendar.shift_session(tradable_session, horizon)
        return entry_session, exit_session, "close", "close"
    if convention == "event_to_next_close":
        entry_session = tradable_session
        exit_session = calendar.shift_session(tradable_session, max(horizon - 1, 0))
        return entry_session, exit_session, "open", "close"
    if convention == "event_to_next_open":
        entry_session = tradable_session
        exit_session = calendar.shift_session(tradable_session, horizon)
        return entry_session, exit_session, "open", "open"
    raise ConfigError(f"Unsupported entry_exit_convention: {convention}")


def _lookup_price(
    price_lookup: Mapping[tuple[str, pd.Timestamp], Mapping[str, float]],
    symbol: str,
    session_date: pd.Timestamp,
    field: str,
) -> float:
    rec = price_lookup.get((str(symbol), pd.Timestamp(session_date).normalize()))
    if rec is None:
        return np.nan
    value = rec.get(field)
    if value is None:
        return np.nan
    return float(value)


def _build_cost_lookup(costs: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], tuple[float, float]]:
    df = _normalize_columns(costs)
    symbol_col = _find_first_column(df, SYMBOL_CANDIDATES)
    date_col = _find_first_column(df, DATE_CANDIDATES)
    df[symbol_col] = _normalize_symbol(df[symbol_col])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    entry_col = _find_first_column(df, ("entry_cost", "entry_cost_ret", "entry_cost_pct", "entry_cost_frac", "entry_cost_bps"), required=False)
    exit_col = _find_first_column(df, ("exit_cost", "exit_cost_ret", "exit_cost_pct", "exit_cost_frac", "exit_cost_bps"), required=False)
    total_col = _find_first_column(df, ("total_cost", "cost_total", "turnover_cost", "total_cost_bps"), required=False)
    lookup: dict[tuple[str, pd.Timestamp], tuple[float, float]] = {}
    for _, row in df.iterrows():
        entry = np.nan
        exit = np.nan
        if entry_col is not None:
            entry = _cost_value_to_return_fraction(float(row[entry_col]), entry_col)
        if exit_col is not None:
            exit = _cost_value_to_return_fraction(float(row[exit_col]), exit_col)
        if total_col is not None and (np.isnan(entry) or np.isnan(exit)):
            total = _cost_value_to_return_fraction(float(row[total_col]), total_col)
            if np.isnan(entry):
                entry = total / 2.0
            if np.isnan(exit):
                exit = total / 2.0
        lookup[(row[symbol_col], pd.Timestamp(row[date_col]).normalize())] = (entry, exit)
    return lookup


def _cost_value_to_return_fraction(value: float, colname: str) -> float:
    if np.isnan(value):
        return np.nan
    if "bps" in colname.lower():
        return float(value) / 10000.0
    if value > 1.0:
        return float(value) / 10000.0
    return float(value)


def _lookup_execution_costs(
    lookup: Mapping[tuple[str, pd.Timestamp], tuple[float, float]],
    symbol: str,
    entry_session: pd.Timestamp,
    exit_session: pd.Timestamp,
) -> tuple[float, float]:
    entry = lookup.get((str(symbol), pd.Timestamp(entry_session).normalize()))
    exit = lookup.get((str(symbol), pd.Timestamp(exit_session).normalize()))
    entry_cost = entry[0] if entry is not None else np.nan
    exit_cost = exit[1] if exit is not None else np.nan
    return entry_cost, exit_cost


def _build_rate_lookup(costs: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], float]:
    df = _normalize_columns(costs)
    symbol_col = _find_first_column(df, SYMBOL_CANDIDATES)
    date_col = _find_first_column(df, DATE_CANDIDATES)
    rate_col = _find_first_column(
        df,
        (
            "annual_borrow_rate",
            "annual_carry_rate",
            "borrow_rate",
            "carry_rate",
            "daily_borrow_rate",
            "daily_carry_rate",
            "rate",
        ),
    )
    df[symbol_col] = _normalize_symbol(df[symbol_col])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    lookup: dict[tuple[str, pd.Timestamp], float] = {}
    for _, row in df.iterrows():
        value = float(row[rate_col])
        if np.isnan(value):
            continue
        if value > 1.0:
            value = value / 100.0
        lookup[(row[symbol_col], pd.Timestamp(row[date_col]).normalize())] = value
    return lookup


def _lookup_rate(
    lookup: Mapping[tuple[str, pd.Timestamp], float],
    symbol: str,
    session: pd.Timestamp,
    *,
    default: float,
) -> float:
    return float(lookup.get((str(symbol), pd.Timestamp(session).normalize()), default))


def _carry_cost_for_horizon(borrow_rate: float, carry_rate: float, horizon_days: int) -> float:
    if np.isnan(borrow_rate) or np.isnan(carry_rate):
        return np.nan
    return float((borrow_rate + carry_rate) * float(horizon_days) / 252.0)


def _build_delisting_lookup(df: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], float]:
    data = _normalize_columns(df)
    symbol_col = _find_first_column(data, SYMBOL_CANDIDATES)
    date_col = _find_first_column(data, DATE_CANDIDATES)
    ret_col = _find_first_column(data, ("delisting_return", "return", "ret", "event_ret_gross"))
    data[symbol_col] = _normalize_symbol(data[symbol_col])
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce").dt.normalize()
    return {
        (row[symbol_col], pd.Timestamp(row[date_col]).normalize()): float(row[ret_col])
        for _, row in data.iterrows()
        if pd.notna(row[ret_col])
    }


def _lookup_delisting_return(
    lookup: Mapping[tuple[str, pd.Timestamp], float], symbol: str, exit_session: pd.Timestamp
) -> float | None:
    key = (str(symbol), pd.Timestamp(exit_session).normalize())
    value = lookup.get(key)
    return None if value is None else float(value)


def _build_corporate_action_lookup(df: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], bool]:
    data = _normalize_columns(df)
    symbol_col = _find_first_column(data, SYMBOL_CANDIDATES)
    date_col = _find_first_column(data, DATE_CANDIDATES)
    unsafe_col = _find_first_column(data, ("unsafe_flag", "corporate_action_unsafe", "unsafe", "flag"), required=False)
    action_type_col = _find_first_column(data, ("action_type", "corporate_action_type", "event_type"), required=False)
    data[symbol_col] = _normalize_symbol(data[symbol_col])
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce").dt.normalize()
    lookup: dict[tuple[str, pd.Timestamp], bool] = {}
    for _, row in data.iterrows():
        unsafe = False
        if unsafe_col is not None:
            unsafe = bool(row[unsafe_col])
        elif action_type_col is not None:
            unsafe = str(row[action_type_col]).strip().lower() in {"split", "merge", "spinoff", "tender_offer"}
        lookup[(row[symbol_col], pd.Timestamp(row[date_col]).normalize())] = unsafe
    return lookup


def attach_benchmark_and_abnormal_returns(
    episodes: pd.DataFrame,
    config: EventLabelConfig,
    *,
    benchmark_returns: pd.DataFrame | None = None,
    sector_mapping: pd.DataFrame | None = None,
    factor_exposures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = episodes.copy()
    df["sector"] = pd.NA
    if sector_mapping is not None:
        sector_df = _normalize_columns(sector_mapping)
        symbol_col = _find_first_column(sector_df, SYMBOL_CANDIDATES)
        sector_col = _find_first_column(sector_df, SECTOR_CANDIDATES)
        sector_df[symbol_col] = _normalize_symbol(sector_df[symbol_col])
        sector_lookup = dict(zip(sector_df[symbol_col], sector_df[sector_col].astype(str)))
        df["sector"] = df["symbol"].map(sector_lookup)
    benchmark_long = _prepare_benchmark_long(benchmark_returns) if benchmark_returns is not None else pd.DataFrame()
    benchmark_index = _index_benchmark_long(benchmark_long) if not benchmark_long.empty else {}

    alphas = []
    reasons = df["event_exclusion_reason"].copy()
    benchmark_modes = []
    for idx, row in df.iterrows():
        mode = _resolve_abnormal_mode(str(row["event_type"]), config)
        benchmark_modes.append(mode)
        alpha = np.nan
        if mode == BENCHMARK_MODE_NONE:
            alphas.append(alpha)
            continue
        base_ret = row["event_ret_net"] if pd.notna(row["event_ret_net"]) else row["event_ret_gross"]
        if pd.isna(base_ret):
            alphas.append(alpha)
            continue
        if mode == "market_adjusted":
            bench = benchmark_index.get((pd.Timestamp(row["event_tradable_session"]).normalize(), int(row["horizon_days"]), "market", None))
            if bench is None:
                reasons.loc[idx] = _pick_reason(reasons.loc[idx], "BENCHMARK_UNAVAILABLE")
            else:
                alpha = float(base_ret - bench)
        elif mode == "sector_adjusted":
            sector = None if pd.isna(row["sector"]) else str(row["sector"])
            bench = benchmark_index.get((pd.Timestamp(row["event_tradable_session"]).normalize(), int(row["horizon_days"]), "sector", sector))
            if bench is None:
                reasons.loc[idx] = _pick_reason(reasons.loc[idx], "BENCHMARK_UNAVAILABLE")
            else:
                alpha = float(base_ret - bench)
        elif mode == "sector_market_adjusted":
            sector = None if pd.isna(row["sector"]) else str(row["sector"])
            market = benchmark_index.get((pd.Timestamp(row["event_tradable_session"]).normalize(), int(row["horizon_days"]), "market", None))
            sec = benchmark_index.get((pd.Timestamp(row["event_tradable_session"]).normalize(), int(row["horizon_days"]), "sector", sector))
            if market is None and sec is None:
                reasons.loc[idx] = _pick_reason(reasons.loc[idx], "BENCHMARK_UNAVAILABLE")
            else:
                adj = 0.0
                if market is not None:
                    adj += market
                if sec is not None:
                    adj += sec
                alpha = float(base_ret - adj)
        elif mode == "factor_adjusted":
            factor_alpha = _factor_adjusted_alpha(row, benchmark_long, factor_exposures)
            if factor_alpha is None:
                reasons.loc[idx] = _pick_reason(reasons.loc[idx], "BENCHMARK_UNAVAILABLE")
            else:
                alpha = factor_alpha
        else:
            reasons.loc[idx] = _pick_reason(reasons.loc[idx], "BENCHMARK_UNAVAILABLE")
        alphas.append(alpha)
    df["abnormal_mode_used"] = benchmark_modes
    df["event_alpha"] = alphas
    df["event_exclusion_reason"] = reasons
    return df


def _resolve_abnormal_mode(event_type: str, config: EventLabelConfig) -> str:
    requested = config.abnormal_mode
    if requested != "auto":
        return requested
    if event_type in {"EARNINGS", "CORP_NEWS"}:
        return config.benchmark_mode_default_corporate
    return config.benchmark_mode_default_technical


def _prepare_benchmark_long(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    bench = _normalize_columns(df)
    date_col = _find_first_column(bench, DATE_CANDIDATES)
    bench[date_col] = pd.to_datetime(bench[date_col], errors="coerce").dt.normalize()
    horizon_col = _find_first_column(bench, ("horizon_days", "horizon", "days"), required=False)
    if horizon_col is None:
        value_cols = [col for col in bench.columns if col.lower().startswith(("market_ret_", "sector_ret_", "benchmark_ret_"))]
        rows = []
        for _, row in bench.iterrows():
            sector_val = None
            sector_col = _find_first_column(bench, SECTOR_CANDIDATES, required=False)
            for col in value_cols:
                parts = col.lower().split("_")
                horizon = int(parts[-1].replace("d", ""))
                bench_type = "sector" if col.lower().startswith("sector") else "market"
                rows.append(
                    {
                        "date": row[date_col],
                        "horizon_days": horizon,
                        "benchmark_type": bench_type,
                        "sector": row[sector_col] if sector_col is not None and bench_type == "sector" else None,
                        "benchmark_return": row[col],
                    }
                )
        return pd.DataFrame(rows)
    bench[horizon_col] = pd.to_numeric(bench[horizon_col], errors="coerce").astype("Int64")
    bench_type_col = _find_first_column(bench, ("benchmark_type", "type", "bucket"), required=False)
    sector_col = _find_first_column(bench, SECTOR_CANDIDATES, required=False)
    return_col = _find_first_column(bench, ("benchmark_return", "return", "ret", "value"))
    out = pd.DataFrame(
        {
            "date": bench[date_col],
            "horizon_days": bench[horizon_col].astype(float),
            "benchmark_type": bench[bench_type_col].astype(str).str.lower() if bench_type_col is not None else "market",
            "sector": bench[sector_col].astype(str) if sector_col is not None else None,
            "benchmark_return": pd.to_numeric(bench[return_col], errors="coerce"),
        }
    )
    return out.dropna(subset=["date", "horizon_days", "benchmark_return"])


def _index_benchmark_long(df: pd.DataFrame) -> dict[tuple[pd.Timestamp, int, str, str | None], float]:
    out: dict[tuple[pd.Timestamp, int, str, str | None], float] = {}
    if df.empty:
        return out
    for _, row in df.iterrows():
        key = (
            pd.Timestamp(row["date"]).normalize(),
            int(row["horizon_days"]),
            str(row["benchmark_type"]).lower(),
            None if row.get("sector") is None or pd.isna(row.get("sector")) else str(row.get("sector")),
        )
        out[key] = float(row["benchmark_return"])
    return out


def _factor_adjusted_alpha(
    row: pd.Series,
    benchmark_long: pd.DataFrame,
    factor_exposures: pd.DataFrame | None,
) -> float | None:
    if factor_exposures is None or benchmark_long.empty:
        return None
    exposures = _normalize_columns(factor_exposures)
    symbol_col = _find_first_column(exposures, SYMBOL_CANDIDATES)
    date_col = _find_first_column(exposures, DATE_CANDIDATES)
    exposures[symbol_col] = _normalize_symbol(exposures[symbol_col])
    exposures[date_col] = pd.to_datetime(exposures[date_col], errors="coerce").dt.normalize()
    match = exposures[
        (exposures[symbol_col] == row["symbol"]) &
        (exposures[date_col] == pd.Timestamp(row["event_tradable_session"]).normalize())
    ]
    if match.empty:
        return None
    factor_cols = [c for c in exposures.columns if c not in {symbol_col, date_col} and pd.api.types.is_numeric_dtype(exposures[c])]
    if not factor_cols:
        return None
    factor_returns = benchmark_long[
        (benchmark_long["date"] == pd.Timestamp(row["event_tradable_session"]).normalize()) &
        (benchmark_long["horizon_days"] == int(row["horizon_days"])) &
        (benchmark_long["benchmark_type"] == "factor")
    ]
    if factor_returns.empty:
        return None
    fr_map = {str(r.get("sector") or r.get("factor") or r.get("name")): float(r["benchmark_return"]) for _, r in factor_returns.iterrows()}
    prediction = 0.0
    for col in factor_cols:
        prediction += float(match.iloc[0][col]) * float(fr_map.get(col, 0.0))
    base_ret = row["event_ret_net"] if pd.notna(row["event_ret_net"]) else row["event_ret_gross"]
    return float(base_ret - prediction)


def derive_continuous_labels(episodes: pd.DataFrame, config: EventLabelConfig) -> pd.DataFrame:
    df = episodes.copy()
    if config.event_target_base == "net_return":
        base = df["event_ret_net"].where(df["event_ret_net"].notna(), df["event_ret_gross"])
    elif config.event_target_base == "gross_return":
        base = df["event_ret_gross"]
    elif config.event_target_base == "alpha":
        base = df["event_alpha"]
    else:
        base = df["event_alpha"].where(df["event_alpha"].notna(), df["event_ret_net"].where(df["event_ret_net"].notna(), df["event_ret_gross"]))
    df["label_event_cont"] = base
    zscores = pd.Series(np.nan, index=df.index, dtype=float)
    valid = df["label_event_cont"].notna()
    for (_, horizon), grp in df.loc[valid].groupby(["event_type", "horizon_days"]):
        mu = grp["label_event_cont"].mean()
        sigma = grp["label_event_cont"].std(ddof=0)
        sigma = sigma if pd.notna(sigma) and sigma > 1e-12 else np.nan
        if np.isnan(sigma):
            zscores.loc[grp.index] = 0.0
        else:
            zscores.loc[grp.index] = (grp["label_event_cont"] - mu) / sigma
    df["event_zscore"] = zscores
    return df


def derive_rank_or_class_labels(episodes: pd.DataFrame, config: EventLabelConfig) -> pd.DataFrame:
    df = episodes.copy()
    df["label_event_bin"] = np.nan
    df["label_event_tri"] = np.nan
    if not config.classification_policy.enabled:
        return df
    group_cols = ["event_type", "horizon_days"] if config.classification_policy.within_family else ["horizon_days"]
    valid = df["label_event_cont"].notna()
    for _, grp in df.loc[valid].groupby(group_cols):
        values = grp["label_event_cont"]
        q_bot = values.quantile(config.classification_policy.bottom_quantile)
        q_top = values.quantile(config.classification_policy.top_quantile)
        if config.classification_policy.emit_binary:
            df.loc[grp.index, "label_event_bin"] = (values > config.classification_policy.positive_threshold).astype(int)
        if config.classification_policy.emit_ternary:
            tri = np.where(values >= q_top, 1, np.where(values <= q_bot, -1, 0))
            df.loc[grp.index, "label_event_tri"] = tri
    return df


def apply_event_validity_hierarchy(episodes: pd.DataFrame) -> pd.DataFrame:
    df = episodes.copy()
    ordered_reason = []
    valid_flag = []
    for _, row in df.iterrows():
        reason = row.get("event_exclusion_reason", pd.NA)
        normalized = _normalize_reason(reason)
        if normalized is None:
            ordered_reason.append(pd.NA)
            valid_flag.append(True)
            continue
        winner = normalized
        if isinstance(normalized, str) and "|" in normalized:
            components = [part for part in normalized.split("|") if part]
            winner = _pick_highest_priority_reason(components)
        ordered_reason.append(winner)
        valid_flag.append(False)
    df["event_exclusion_reason"] = ordered_reason
    df["event_label_valid_flag"] = valid_flag
    return df


def run_event_qc(episodes: pd.DataFrame, events: pd.DataFrame, config: EventLabelConfig) -> tuple[list[QCRecord], pd.DataFrame]:
    records: list[QCRecord] = []
    coverage_rows: list[dict[str, Any]] = []
    duplicate_event_ids = int(events["event_id"].duplicated().sum())
    records.append(
        QCRecord(
            check_name="unique_event_id",
            severity=SEVERITY_FAIL,
            status="PASS" if duplicate_event_ids == 0 else "FAIL",
            details={"duplicate_event_ids": duplicate_event_ids},
        )
    )
    monotonic_violations = int(
        (
            (events["event_timestamp_pub"].notna() & events["event_timestamp_occ"].notna() & (events["event_timestamp_pub"] < events["event_timestamp_occ"])) |
            (events["event_timestamp_known"].notna() & events["event_timestamp_pub"].notna() & (events["event_timestamp_known"] < events["event_timestamp_pub"])) |
            (events["event_timestamp_tradable"].notna() & events["event_timestamp_known"].notna() & (events["event_timestamp_tradable"] < events["event_timestamp_known"]))
        ).sum()
    )
    records.append(
        QCRecord(
            check_name="timestamp_monotonicity",
            severity=SEVERITY_FAIL,
            status="PASS" if monotonic_violations == 0 else "FAIL",
            details={"violations": monotonic_violations},
        )
    )
    for (event_type, horizon), grp in episodes.groupby(["event_type", "horizon_days"], dropna=False):
        n_total = int(len(grp))
        n_valid = int(grp["event_label_valid_flag"].sum())
        exclusion_ratio = float(1.0 - n_valid / n_total) if n_total else np.nan
        variance = float(grp.loc[grp["event_label_valid_flag"], "label_event_cont"].var(ddof=0)) if n_valid else np.nan
        tri_unique = int(grp.loc[grp["event_label_valid_flag"], "label_event_tri"].dropna().nunique())
        overlap_reject_ratio = float((grp["event_exclusion_reason"] == "OVERLAP_REJECTED").mean()) if n_total else np.nan
        coverage_rows.append(
            {
                "event_type": event_type,
                "horizon_days": horizon,
                "n_events_total": n_total,
                "n_events_valid": n_valid,
                "exclusion_ratio": exclusion_ratio,
                "label_variance": variance,
                "tri_unique_values": tri_unique,
                "overlap_reject_ratio": overlap_reject_ratio,
            }
        )
        if pd.notna(event_type) and event_type in config.canonical_event_types:
            records.append(
                QCRecord(
                    check_name=f"min_family_sample::{event_type}::{horizon}",
                    severity=SEVERITY_WARN,
                    status="PASS" if n_valid >= config.min_family_sample else "WARN",
                    details={"n_events_valid": n_valid, "required": config.min_family_sample},
                )
            )
        records.append(
            QCRecord(
                check_name=f"variance_positive::{event_type}::{horizon}",
                severity=SEVERITY_WARN,
                status="PASS" if pd.notna(variance) and variance > 0 else "WARN",
                details={"label_variance": variance},
            )
        )
        if config.classification_policy.enabled:
            records.append(
                QCRecord(
                    check_name=f"class_collapse::{event_type}::{horizon}",
                    severity=SEVERITY_WARN,
                    status="PASS" if tri_unique >= 2 else "WARN",
                    details={"tri_unique_values": tri_unique},
                )
            )
    fail_count = sum(1 for rec in records if rec.severity == SEVERITY_FAIL and rec.status == "FAIL")
    records.append(
        QCRecord(
            check_name="qc_rollup",
            severity=SEVERITY_FAIL,
            status="PASS" if fail_count == 0 else "FAIL",
            details={"fail_count": fail_count},
        )
    )
    coverage = pd.DataFrame(coverage_rows)
    return records, coverage


def persist_event_outputs(
    episodes: pd.DataFrame,
    coverage: pd.DataFrame,
    qc_records: Sequence[QCRecord],
    *,
    output_dir: str | Path,
    run_id: str,
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    episodes_path = _write_table(episodes, outdir / f"event_labels_{run_id}.parquet")
    coverage_path = _write_table(coverage, outdir / f"event_coverage_{run_id}.parquet")
    qc_payload = {
        "run_id": run_id,
        "generated_at_utc": _utc_now_iso(),
        "records": [asdict(rec) for rec in qc_records],
    }
    qc_path = _write_json(qc_payload, outdir / f"event_qc_{run_id}.json")
    summary_payload = build_event_summary(episodes, coverage, qc_records, run_id)
    summary_path = _write_json(summary_payload, outdir / f"event_labels_summary_{run_id}.json")
    windows_payload = {
        "run_id": run_id,
        "windows": [
            {
                "event_id": row["event_id"],
                "episode_group_id": row["episode_group_id"],
                "symbol": row["symbol"],
                "horizon_days": int(row["horizon_days"]),
                "event_start": _to_iso_or_none(row["event_start"]),
                "event_end": _to_iso_or_none(row["event_end"]),
            }
            for _, row in episodes.loc[episodes["event_label_valid_flag"]].iterrows()
            if pd.notna(row["horizon_days"])
        ],
    }
    windows_path = _write_json(windows_payload, outdir / f"event_windows_for_splits_{run_id}.json")
    manifest_path = _write_json(dict(manifest), outdir / f"event_manifest_{run_id}.json")
    return {
        "episodes": str(episodes_path),
        "coverage": str(coverage_path),
        "qc": str(qc_path),
        "summary": str(summary_path),
        "windows_for_splits": str(windows_path),
        "manifest": str(manifest_path),
    }


def build_event_summary(
    episodes: pd.DataFrame,
    coverage: pd.DataFrame,
    qc_records: Sequence[QCRecord],
    run_id: str,
) -> dict[str, Any]:
    valid = episodes[episodes["event_label_valid_flag"]]
    return {
        "run_id": run_id,
        "generated_at_utc": _utc_now_iso(),
        "n_event_rows": int(len(episodes)),
        "n_valid_event_rows": int(valid.shape[0]),
        "n_unique_events": int(episodes["event_id"].nunique()),
        "n_valid_unique_events": int(valid["event_id"].nunique()),
        "event_types": sorted(str(x) for x in episodes["event_type"].dropna().unique()),
        "coverage_records": coverage.to_dict(orient="records"),
        "exclusion_reason_counts": {
            str(k): int(v)
            for k, v in episodes["event_exclusion_reason"].fillna("VALID").value_counts().to_dict().items()
        },
        "qc_fail_count": int(sum(1 for rec in qc_records if rec.severity == SEVERITY_FAIL and rec.status == "FAIL")),
        "qc_warn_count": int(sum(1 for rec in qc_records if rec.status == "WARN")),
    }


def _to_iso_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def build_event_manifest(
    *,
    run_id: str,
    config: EventLabelConfig,
    config_path: str | Path,
    event_source_path: str | Path,
    prices_pit_path: str | Path,
    universe_history_path: str | Path,
    execution_costs_path: str | Path | None,
    benchmark_returns_path: str | Path | None,
) -> dict[str, Any]:
    config_hash = _sha256_text(_stable_json_dumps(asdict(config)))
    non_canonical_flag = bool(config.non_canonical_flag)
    if config.abnormal_mode != "auto":
        requested_mode = config.abnormal_mode
    else:
        requested_mode = "auto"
    if config.entry_exit_convention != "open_to_open":
        non_canonical_flag = True
    if config.overlap_policy != "first_event_wins_with_cooldown":
        non_canonical_flag = True
    if config.timestamp_policy != "strict":
        non_canonical_flag = True
    return {
        "run_id": run_id,
        "generated_at_utc": _utc_now_iso(),
        "config_hash": config_hash,
        "config_file_hash": _sha256_file(config_path),
        "event_source_hash": _sha256_file(event_source_path),
        "prices_version": _sha256_file(prices_pit_path),
        "universe_history_version": _sha256_file(universe_history_path),
        "cost_model_version": _sha256_file(execution_costs_path),
        "benchmark_version": _sha256_file(benchmark_returns_path),
        "taxonomy_version": config.taxonomy_version,
        "timestamp_policy": config.timestamp_policy,
        "overlap_policy": config.overlap_policy,
        "classification_default": {
            "mode": config.classification_policy.mode,
            "top_quantile": config.classification_policy.top_quantile,
            "bottom_quantile": config.classification_policy.bottom_quantile,
            "within_family": config.classification_policy.within_family,
        },
        "abnormal_mode": requested_mode,
        "policy_version": config.policy_version,
        "non_canonical_flag": non_canonical_flag,
    }


def build_event_labels(
    *,
    event_source_path: str | Path,
    prices_pit_path: str | Path,
    universe_history_path: str | Path,
    trading_calendar_path: str | Path | None,
    event_label_config_path: str | Path,
    run_id: str,
    output_dir: str | Path | None = None,
    execution_costs_path: str | Path | None = None,
    borrow_costs_path: str | Path | None = None,
    carry_costs_path: str | Path | None = None,
    benchmark_returns_path: str | Path | None = None,
    sector_mapping_path: str | Path | None = None,
    factor_exposures_path: str | Path | None = None,
    delisting_returns_path: str | Path | None = None,
    corporate_actions_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_event_label_config(event_label_config_path)
    prices, price_schema = load_prices_panel(prices_pit_path)
    calendar = TradingCalendar.from_sources(prices, trading_calendar_path)
    events = load_event_source(event_source_path)
    universe = load_universe_history(universe_history_path)
    execution_costs = _read_table(execution_costs_path) if execution_costs_path is not None else None
    borrow_costs = _read_table(borrow_costs_path) if borrow_costs_path is not None else None
    carry_costs = _read_table(carry_costs_path) if carry_costs_path is not None else None
    benchmark_returns = _read_table(benchmark_returns_path) if benchmark_returns_path is not None else None
    sector_mapping = _read_table(sector_mapping_path) if sector_mapping_path is not None else None
    factor_exposures = _read_table(factor_exposures_path) if factor_exposures_path is not None else None
    delisting_returns = _read_table(delisting_returns_path) if delisting_returns_path is not None else None
    corporate_actions = _read_table(corporate_actions_path) if corporate_actions_path is not None else None

    events = canonicalize_event_types(events, config)
    events = resolve_event_timestamps(events, calendar, config)
    events = filter_by_universe(events, universe)
    events = attach_event_horizons(events, config)
    events = resolve_overlaps(events, calendar, config)

    episodes = compute_event_returns(
        events,
        prices,
        price_schema,
        calendar,
        config,
        execution_costs=execution_costs,
        borrow_costs=borrow_costs,
        carry_costs=carry_costs,
        delisting_returns=delisting_returns,
        corporate_actions=corporate_actions,
    )
    episodes = attach_benchmark_and_abnormal_returns(
        episodes,
        config,
        benchmark_returns=benchmark_returns,
        sector_mapping=sector_mapping,
        factor_exposures=factor_exposures,
    )
    episodes = derive_continuous_labels(episodes, config)
    episodes = derive_rank_or_class_labels(episodes, config)
    episodes = apply_event_validity_hierarchy(episodes)
    episodes["run_id"] = run_id

    qc_records, coverage = run_event_qc(episodes, events, config)
    if any(rec.severity == SEVERITY_FAIL and rec.status == "FAIL" for rec in qc_records):
        failing = [rec.check_name for rec in qc_records if rec.severity == SEVERITY_FAIL and rec.status == "FAIL"]
        raise QCFailure(f"Event label QC failed: {', '.join(failing)}")

    manifest = build_event_manifest(
        run_id=run_id,
        config=config,
        config_path=event_label_config_path,
        event_source_path=event_source_path,
        prices_pit_path=prices_pit_path,
        universe_history_path=universe_history_path,
        execution_costs_path=execution_costs_path,
        benchmark_returns_path=benchmark_returns_path,
    )
    final_output_dir = Path(output_dir or config.output_dir or Path(event_source_path).resolve().parent / "event_labels_output")
    artifacts = persist_event_outputs(
        episodes,
        coverage,
        qc_records,
        output_dir=final_output_dir,
        run_id=run_id,
        manifest=manifest,
    )
    return {
        "episodes": episodes,
        "coverage": coverage,
        "qc_records": [asdict(rec) for rec in qc_records],
        "manifest": manifest,
        "artifacts": artifacts,
    }


def _normalize_reason(reason: Any) -> str | None:
    if reason is None or pd.isna(reason):
        return None
    text = str(reason).strip()
    return text or None


def _pick_reason(existing: Any, new_reason: str) -> str:
    normalized = _normalize_reason(existing)
    if normalized is None:
        return new_reason
    components = [part for part in normalized.split("|") if part]
    if new_reason not in components:
        components.append(new_reason)
    return _pick_highest_priority_reason(components)


def _pick_highest_priority_reason(reasons: Sequence[str]) -> str:
    priority_map = {reason: i for i, reason in enumerate(EXCLUSION_PRIORITY)}
    reasons = [str(r) for r in reasons if r]
    if not reasons:
        return ""
    return sorted(reasons, key=lambda x: priority_map.get(x, 10_000))[0]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical event-driven labels")
    parser.add_argument("--event-source-path", required=True)
    parser.add_argument("--prices-path", "--prices-pit-path", dest="prices_pit_path", required=True)
    parser.add_argument("--universe-history-path", required=True)
    parser.add_argument("--trading-calendar-path", default=None)
    parser.add_argument("--config-path", "--event-label-config-path", dest="event_label_config_path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--execution-costs-path", "--costs-path", dest="execution_costs_path", default=None)
    parser.add_argument("--borrow-costs-path", default=None)
    parser.add_argument("--carry-costs-path", default=None)
    parser.add_argument("--benchmark-path", "--benchmark-returns-path", dest="benchmark_returns_path", default=None)
    parser.add_argument("--sector-mapping-path", default=None)
    parser.add_argument("--factor-exposures-path", default=None)
    parser.add_argument("--delisting-returns-path", default=None)
    parser.add_argument("--corporate-actions-path", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args(argv)
    result = build_event_labels(
        event_source_path=args.event_source_path,
        prices_pit_path=args.prices_pit_path,
        universe_history_path=args.universe_history_path,
        trading_calendar_path=args.trading_calendar_path,
        event_label_config_path=args.event_label_config_path,
        run_id=args.run_id,
        output_dir=args.output_dir,
        execution_costs_path=args.execution_costs_path,
        borrow_costs_path=args.borrow_costs_path,
        carry_costs_path=args.carry_costs_path,
        benchmark_returns_path=args.benchmark_returns_path,
        sector_mapping_path=args.sector_mapping_path,
        factor_exposures_path=args.factor_exposures_path,
        delisting_returns_path=args.delisting_returns_path,
        corporate_actions_path=args.corporate_actions_path,
    )
    LOGGER.info("event_labels completed", extra={"artifacts": result["artifacts"]})
    print(json.dumps({"artifacts": result["artifacts"], "manifest": result["manifest"]}, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
