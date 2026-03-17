from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

LOGGER = logging.getLogger(__name__)

DATE_CANDIDATES = ("date", "asof_date", "trade_date", "decision_date")
SYMBOL_CANDIDATES = ("symbol", "ticker", "asset", "instrument_id")
UNIVERSE_FLAG_CANDIDATES = (
    "is_eligible",
    "eligible",
    "in_universe",
    "universe_member",
    "eligible_flag",
    "is_in_universe",
)
PRICE_FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "open": ("adj_open", "open_adj", "open", "px_open", "open_price"),
    "close": ("adj_close", "adjusted_close", "close_adj", "close", "px_close", "close_price"),
    "vwap": ("adj_vwap", "vwap_adj", "vwap", "px_vwap"),
}
ENTRY_EXIT_MODES = {
    "close_to_close": ("close", "close"),
    "close_to_open": ("close", "open"),
    "open_to_close": ("open", "close"),
    "open_to_open": ("open", "open"),
}
EXCLUSION_PRIORITY = [
    "OUT_OF_SAMPLE_DATE",
    "NOT_IN_UNIVERSE",
    "MISSING_ENTRY_PRICE",
    "MISSING_EXIT_PRICE",
    "INCOMPLETE_FORWARD_WINDOW",
    "CORPORATE_ACTION_UNSAFE",
    "TRADING_HALT_OR_SUSPENSION",
    "DELIST_AMBIGUOUS",
    "MISSING_COST_INPUT",
    "MISSING_EXPOSURES",
    "NEUTRALIZATION_FAILED",
    "QC_REJECTED",
]
SEVERITY_FAIL = "FAIL"
SEVERITY_WARN = "WARN"
SEVERITY_INFO = "INFO"
CANONICAL_DEFAULTS = {
    "horizons": [5, 10, 20],
    "primary_target": "y_fwd_ret_net_10d",
    "return_mode": "close_to_close",
    "decision_lag": 1,
    "net_of_costs": True,
    "missing_cost_policy": "strict_invalidate",
    "neutralization_mode": "sector_beta_size_liquidity",
    "classification_default": "tail_20_20_on_net_10d",
    "label_mode": "fixed_horizon",
}


class BuildLabelsError(RuntimeError):
    """Base error for label construction."""


class ConfigError(BuildLabelsError):
    """Raised when the provided configuration is invalid."""


class DataContractError(BuildLabelsError):
    """Raised when an input table violates the expected contract."""


class QCFailure(BuildLabelsError):
    """Raised when a FAIL-level QC check trips."""


@dataclass(frozen=True)
class RankingPolicy:
    enabled: bool = True
    target_base: str = "auto"
    method: str = "average"


@dataclass(frozen=True)
class ClassificationPolicy:
    enabled: bool = True
    kind: str = "tail"
    target_base: str = "auto"
    top_quantile: float = 0.8
    bottom_quantile: float = 0.2
    fixed_threshold: float = 0.0
    horizon: int | None = 10
    emit_all_horizons: bool = True


@dataclass(frozen=True)
class NeutralizationPolicy:
    enabled: bool = True
    mode: str = "sector_beta_size_liquidity"
    estimator: str = "weighted_ridge"
    fixed_lambda: float = 1.0
    lambda_grid: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    lambda_selection: str = "fixed"
    n_min_neut: int = 25
    max_missing_exposure_ratio: float = 0.20
    condition_number_threshold: float = 1e8
    residual_var_floor: float = 1e-12
    failure_policy: str = "invalidate_derived_target"
    weight_col: str | None = None


@dataclass(frozen=True)
class CostPolicy:
    net_of_costs: bool = True
    missing_cost_policy: str = "strict_invalidate"
    fallback_total_cost_bps: float = 0.0
    annual_borrow_rate: float = 0.0
    annual_carry_rate: float = 0.0
    ci_zscore: float = 1.96


@dataclass(frozen=True)
class LabelConfig:
    horizons: tuple[int, ...] = (5, 10, 20)
    return_mode: str = "close_to_close"
    decision_lag: int = 1
    net_of_costs: bool = True
    primary_target: str = "y_fwd_ret_net_10d"
    missing_cost_policy: str = "strict_invalidate"
    neutralization_mode: str = "sector_beta_size_liquidity"
    classification_policy: ClassificationPolicy = field(default_factory=ClassificationPolicy)
    ranking_policy: RankingPolicy = field(default_factory=RankingPolicy)
    delisting_policy: str = "strict"
    max_forward_gap_days: int = 5
    min_valid_cross_section: int = 20
    calendar_mode: str = "trading_sessions"
    policy_version: str = "1.0.0"
    label_mode: str = "fixed_horizon"
    start_date: str | None = None
    end_date: str | None = None
    output_dir: str | None = None
    cost_policy: CostPolicy = field(default_factory=CostPolicy)
    neutralization_policy: NeutralizationPolicy = field(default_factory=NeutralizationPolicy)
    non_canonical_flag: bool = False

    @staticmethod
    def from_mapping(mapping: Mapping[str, Any]) -> "LabelConfig":
        mapping = dict(mapping)
        horizons = mapping.get("horizons", CANONICAL_DEFAULTS["horizons"])
        if not isinstance(horizons, Sequence) or isinstance(horizons, (str, bytes)):
            raise ConfigError("horizons must be a sequence of positive integers")
        horizons_tuple = tuple(sorted({int(x) for x in horizons}))
        if not horizons_tuple or min(horizons_tuple) <= 0:
            raise ConfigError("horizons must contain positive integers")
        return_mode = str(mapping.get("return_mode", "close_to_close"))
        if return_mode not in ENTRY_EXIT_MODES:
            raise ConfigError(
                "return_mode must be one of "
                + ", ".join(sorted(ENTRY_EXIT_MODES))
            )
        decision_lag = int(mapping.get("decision_lag", 1))
        if decision_lag < 0:
            raise ConfigError("decision_lag must be >= 0")

        ranking_raw = mapping.get("ranking_policy", {})
        ranking_policy = RankingPolicy(
            enabled=bool(getattr(ranking_raw, "get", lambda *_: True)("enabled", True)),
            target_base=str(getattr(ranking_raw, "get", lambda *_: "auto")("target_base", "auto")),
            method=str(getattr(ranking_raw, "get", lambda *_: "average")("method", "average")),
        )

        classification_raw = mapping.get("classification_policy")
        classification_default = str(
            mapping.get("classification_default", CANONICAL_DEFAULTS["classification_default"])
        )
        if classification_raw is None:
            classification_policy = _classification_policy_from_default(classification_default)
        elif isinstance(classification_raw, Mapping):
            classification_policy = ClassificationPolicy(
                enabled=bool(classification_raw.get("enabled", True)),
                kind=str(classification_raw.get("kind", "tail")),
                target_base=str(classification_raw.get("target_base", "auto")),
                top_quantile=float(classification_raw.get("top_quantile", 0.8)),
                bottom_quantile=float(classification_raw.get("bottom_quantile", 0.2)),
                fixed_threshold=float(classification_raw.get("fixed_threshold", 0.0)),
                horizon=_optional_int(classification_raw.get("horizon", 10)),
                emit_all_horizons=bool(classification_raw.get("emit_all_horizons", True)),
            )
        else:
            raise ConfigError("classification_policy must be a mapping when provided")

        neutralization_raw = mapping.get("neutralization_policy", {})
        neutralization_mode = str(
            mapping.get("neutralization_mode", CANONICAL_DEFAULTS["neutralization_mode"])
        )
        neutralization_enabled = neutralization_mode.lower() not in {"none", "off", "disabled", "false"}
        lambda_grid_raw = neutralization_raw.get("lambda_grid", (1e-3, 1e-2, 1e-1, 1.0, 10.0))
        if isinstance(lambda_grid_raw, Sequence) and not isinstance(lambda_grid_raw, (str, bytes)):
            lambda_grid = tuple(float(x) for x in lambda_grid_raw)
        else:
            lambda_grid = (float(lambda_grid_raw),)
        neutralization_policy = NeutralizationPolicy(
            enabled=bool(neutralization_raw.get("enabled", neutralization_enabled)),
            mode=neutralization_mode,
            estimator=str(neutralization_raw.get("estimator", "weighted_ridge")),
            fixed_lambda=float(neutralization_raw.get("fixed_lambda", 1.0)),
            lambda_grid=lambda_grid,
            lambda_selection=str(neutralization_raw.get("lambda_selection", "fixed")),
            n_min_neut=int(neutralization_raw.get("n_min_neut", 25)),
            max_missing_exposure_ratio=float(neutralization_raw.get("max_missing_exposure_ratio", 0.20)),
            condition_number_threshold=float(neutralization_raw.get("condition_number_threshold", 1e8)),
            residual_var_floor=float(neutralization_raw.get("residual_var_floor", 1e-12)),
            failure_policy=str(
                neutralization_raw.get("failure_policy", "invalidate_derived_target")
            ),
            weight_col=_optional_str(neutralization_raw.get("weight_col")),
        )

        cost_raw = mapping.get("cost_policy", {})
        net_of_costs = bool(mapping.get("net_of_costs", True))
        missing_cost_policy = str(mapping.get("missing_cost_policy", "strict_invalidate"))
        cost_policy = CostPolicy(
            net_of_costs=net_of_costs,
            missing_cost_policy=missing_cost_policy,
            fallback_total_cost_bps=float(cost_raw.get("fallback_total_cost_bps", 0.0)),
            annual_borrow_rate=float(cost_raw.get("annual_borrow_rate", 0.0)),
            annual_carry_rate=float(cost_raw.get("annual_carry_rate", 0.0)),
            ci_zscore=float(cost_raw.get("ci_zscore", 1.96)),
        )

        primary_target = str(mapping.get("primary_target", CANONICAL_DEFAULTS["primary_target"]))
        config = LabelConfig(
            horizons=horizons_tuple,
            return_mode=return_mode,
            decision_lag=decision_lag,
            net_of_costs=net_of_costs,
            primary_target=primary_target,
            missing_cost_policy=missing_cost_policy,
            neutralization_mode=neutralization_mode,
            classification_policy=classification_policy,
            ranking_policy=ranking_policy,
            delisting_policy=str(mapping.get("delisting_policy", "strict")),
            max_forward_gap_days=int(mapping.get("max_forward_gap_days", 5)),
            min_valid_cross_section=int(mapping.get("min_valid_cross_section", 20)),
            calendar_mode=str(mapping.get("calendar_mode", "trading_sessions")),
            policy_version=str(mapping.get("policy_version", "1.0.0")),
            label_mode=str(mapping.get("label_mode", CANONICAL_DEFAULTS["label_mode"])),
            start_date=_optional_str(mapping.get("start_date")),
            end_date=_optional_str(mapping.get("end_date")),
            output_dir=_optional_str(mapping.get("output_dir")),
            cost_policy=cost_policy,
            neutralization_policy=neutralization_policy,
        )
        object.__setattr__(config, "non_canonical_flag", _detect_non_canonical(config, classification_default))
        _validate_config(config)
        return config

    def to_serializable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["horizons"] = list(self.horizons)
        payload["classification_policy"] = asdict(self.classification_policy)
        payload["ranking_policy"] = asdict(self.ranking_policy)
        payload["neutralization_policy"] = asdict(self.neutralization_policy)
        payload["cost_policy"] = asdict(self.cost_policy)
        return payload


@dataclass(frozen=True)
class InputPaths:
    adjusted_prices_pit_path: str
    universe_history_path: str
    trading_calendar_path: str
    label_config_path: str
    execution_costs_path: str | None = None
    borrow_costs_path: str | None = None
    carry_costs_path: str | None = None
    neutralization_exposures_path: str | None = None
    delisting_returns_path: str | None = None
    corporate_actions_path: str | None = None
    output_dir: str | None = None


@dataclass(frozen=True)
class ResolvedColumns:
    date_col: str
    symbol_col: str
    universe_flag_col: str | None
    entry_price_field: str
    exit_price_field: str


@dataclass
class QCRecord:
    check_name: str
    severity: str
    status: str
    metric_value: Any
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_name": self.check_name,
            "severity": self.severity,
            "status": self.status,
            "metric_value": _json_safe(self.metric_value),
            "message": self.message,
        }


@dataclass
class NeutralizationResult:
    residuals: pd.Series
    lambda_used: float | None
    fit_status: str
    missing_ratio: float
    condition_number: float | None


def _classification_policy_from_default(classification_default: str) -> ClassificationPolicy:
    token = classification_default.lower()
    if token == "tail_20_20_on_net_10d":
        return ClassificationPolicy(
            enabled=True,
            kind="tail",
            target_base="net",
            top_quantile=0.8,
            bottom_quantile=0.2,
            horizon=10,
            emit_all_horizons=True,
        )
    return ClassificationPolicy(enabled=True)


def _detect_non_canonical(config: LabelConfig, classification_default: str) -> bool:
    if list(config.horizons) != CANONICAL_DEFAULTS["horizons"]:
        return True
    if config.primary_target != CANONICAL_DEFAULTS["primary_target"]:
        return True
    if config.return_mode != CANONICAL_DEFAULTS["return_mode"]:
        return True
    if config.decision_lag != CANONICAL_DEFAULTS["decision_lag"]:
        return True
    if config.net_of_costs != CANONICAL_DEFAULTS["net_of_costs"]:
        return True
    if config.missing_cost_policy != CANONICAL_DEFAULTS["missing_cost_policy"]:
        return True
    if config.neutralization_mode != CANONICAL_DEFAULTS["neutralization_mode"]:
        return True
    if classification_default != CANONICAL_DEFAULTS["classification_default"]:
        return True
    if config.label_mode != CANONICAL_DEFAULTS["label_mode"]:
        return True
    return False


def _validate_config(config: LabelConfig) -> None:
    if config.decision_lag < 0:
        raise ConfigError("decision_lag must be >= 0")
    if config.max_forward_gap_days < 0:
        raise ConfigError("max_forward_gap_days must be >= 0")
    if config.min_valid_cross_section <= 0:
        raise ConfigError("min_valid_cross_section must be > 0")
    if config.return_mode not in ENTRY_EXIT_MODES:
        raise ConfigError("unsupported return_mode")
    if config.label_mode not in {"fixed_horizon", "triple_barrier"}:
        raise ConfigError("label_mode must be fixed_horizon or triple_barrier")
    if config.label_mode == "triple_barrier":
        LOGGER.warning("triple_barrier is marked non-canonical in the current implementation")
    if config.primary_target:
        valid_prefixes = ("y_fwd_ret_gross_", "y_fwd_ret_net_", "y_neut_")
        if not config.primary_target.startswith(valid_prefixes):
            raise ConfigError("primary_target must start with y_fwd_ret_gross_, y_fwd_ret_net_, or y_neut_")
    if config.classification_policy.kind not in {"tail", "binary_up"}:
        raise ConfigError("classification_policy.kind must be tail or binary_up")
    if not 0 <= config.classification_policy.bottom_quantile <= 1:
        raise ConfigError("classification_policy.bottom_quantile must be in [0, 1]")
    if not 0 <= config.classification_policy.top_quantile <= 1:
        raise ConfigError("classification_policy.top_quantile must be in [0, 1]")
    if config.classification_policy.bottom_quantile > config.classification_policy.top_quantile:
        raise ConfigError("classification_policy.bottom_quantile cannot exceed top_quantile")
    if config.cost_policy.missing_cost_policy not in {
        "strict_invalidate",
        "fallback_proxy",
        "partial_costs",
        "zero_imputation",
    }:
        raise ConfigError("unsupported missing_cost_policy")
    if config.neutralization_policy.lambda_selection not in {"fixed", "grid_search"}:
        raise ConfigError("neutralization_policy.lambda_selection must be fixed or grid_search")
    if config.neutralization_policy.estimator != "weighted_ridge":
        raise ConfigError("only weighted_ridge is implemented in this version")


def load_label_config(path: str | Path) -> LabelConfig:
    payload = _load_mapping(path)
    return LabelConfig.from_mapping(payload)


def _load_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8")
    if suffix in {".json"}:
        return json.loads(raw)
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ConfigError("pyyaml is required to load YAML configuration files")
        loaded = yaml.safe_load(raw)
        if not isinstance(loaded, Mapping):
            raise ConfigError("configuration file must contain a mapping/object at top level")
        return dict(loaded)
    raise ConfigError(f"unsupported config extension: {path.suffix}")


def load_table(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    if candidate.is_dir():
        parquet_parts = sorted(candidate.rglob("*.parquet"))
        csv_parts = sorted(candidate.rglob("*.csv"))
        if parquet_parts:
            frames = [_read_parquet(p) for p in parquet_parts]
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if csv_parts:
            frames = [pd.read_csv(p) for p in csv_parts]
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        raise DataContractError(f"directory has no parquet/csv files: {candidate}")
    suffix = candidate.suffix.lower()
    if suffix == ".parquet":
        return _read_parquet(candidate)
    if suffix == ".csv":
        return pd.read_csv(candidate)
    if suffix in {".feather", ".ft"}:
        return pd.read_feather(candidate)
    raise DataContractError(f"unsupported table extension: {candidate.suffix}")


def _fingerprint_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    if candidate.is_file():
        stat = candidate.stat()
        payload = f"{candidate.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    parts = []
    for p in sorted(x for x in candidate.rglob("*") if x.is_file()):
        st = p.stat()
        parts.append(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()




def _read_parquet(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover
        raise DataContractError(
            "Reading parquet requires pyarrow or fastparquet in the runtime environment"
        ) from exc


def _write_frame(df: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    try:
        df.to_parquet(target, index=False)
        return target
    except Exception:  # pragma: no cover
        fallback = target.with_suffix('.csv')
        LOGGER.warning(
            "Parquet engine not available; persisting fallback CSV at %s", fallback
        )
        df.to_csv(fallback, index=False)
        return fallback


def _hash_payload(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(x) for x in value]
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _to_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=False)
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.normalize()


def _find_column(columns: Sequence[str], candidates: Sequence[str], *, required: bool = True) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    if required:
        raise DataContractError(f"could not resolve any of columns {list(candidates)}")
    return None


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int) != 0
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "1": True, "yes": True, "y": True, "false": False, "0": False, "no": False, "n": False})
    )
    return mapped.fillna(False)


def _resolve_price_field(columns: Sequence[str], semantic_field: str) -> str:
    for candidate in PRICE_FIELD_CANDIDATES[semantic_field]:
        for col in columns:
            if col.lower() == candidate.lower():
                return col
    raise DataContractError(f"could not resolve price field for semantic '{semantic_field}'")


def _prepare_prices(df: pd.DataFrame, config: LabelConfig) -> tuple[pd.DataFrame, str, str, str, str]:
    date_col = _find_column(df.columns, DATE_CANDIDATES)
    symbol_col = _find_column(df.columns, SYMBOL_CANDIDATES)
    entry_semantic, exit_semantic = ENTRY_EXIT_MODES[config.return_mode]
    entry_field = _resolve_price_field(df.columns, entry_semantic)
    exit_field = _resolve_price_field(df.columns, exit_semantic)
    working = df.copy()
    working[date_col] = _to_datetime(working[date_col])
    working[symbol_col] = working[symbol_col].astype(str)
    working = working.dropna(subset=[date_col, symbol_col]).drop_duplicates([date_col, symbol_col], keep="last")
    return working, date_col, symbol_col, entry_field, exit_field


def _prepare_universe(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str, str | None]:
    date_col = _find_column(df.columns, DATE_CANDIDATES)
    symbol_col = _find_column(df.columns, SYMBOL_CANDIDATES)
    universe_flag_col = _find_column(df.columns, UNIVERSE_FLAG_CANDIDATES, required=False)
    working = df.copy()
    working[date_col] = _to_datetime(working[date_col])
    working[symbol_col] = working[symbol_col].astype(str)
    if universe_flag_col is None:
        working["__universe_flag__"] = True
        universe_flag_col = "__universe_flag__"
    else:
        working[universe_flag_col] = _coerce_bool(working[universe_flag_col])
    working = working.dropna(subset=[date_col, symbol_col]).drop_duplicates([date_col, symbol_col], keep="last")
    return working, date_col, symbol_col, universe_flag_col


def _prepare_calendar(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if df is None or df.empty:
        raise DataContractError("trading calendar is required and cannot be empty")
    date_col = _find_column(df.columns, DATE_CANDIDATES)
    working = df.copy()
    working[date_col] = _to_datetime(working[date_col])
    working = working.dropna(subset=[date_col]).drop_duplicates([date_col]).sort_values(date_col).reset_index(drop=True)
    return working, date_col


def _build_calendar_index(calendar_df: pd.DataFrame, date_col: str) -> tuple[np.ndarray, dict[pd.Timestamp, int]]:
    dates = calendar_df[date_col].dropna().sort_values().unique()
    date_array = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    mapping: dict[pd.Timestamp, int] = {}
    for idx, raw in enumerate(pd.to_datetime(date_array)):
        mapping[pd.Timestamp(raw).normalize()] = idx
    return date_array, mapping


def _date_offsets(base_dates: pd.Series, date_to_index: Mapping[pd.Timestamp, int], offset: int, calendar_dates: np.ndarray) -> pd.Series:
    indices = []
    for dt in pd.to_datetime(base_dates):
        key = pd.Timestamp(dt).normalize()
        indices.append(date_to_index.get(key, -10**9))
    idx = np.asarray(indices, dtype=int)
    out = idx + int(offset)
    valid = (idx >= 0) & (out >= 0) & (out < len(calendar_dates))
    result = np.full(len(base_dates), np.datetime64("NaT"), dtype="datetime64[ns]")
    if valid.any():
        result[valid] = calendar_dates[out[valid]]
    return pd.Series(pd.to_datetime(result), index=base_dates.index)


def _date_index_series(dates: pd.Series, date_to_index: Mapping[pd.Timestamp, int]) -> pd.Series:
    values = []
    for dt in pd.to_datetime(dates):
        if pd.isna(dt):
            values.append(np.nan)
        else:
            values.append(date_to_index.get(pd.Timestamp(dt).normalize(), np.nan))
    return pd.Series(values, index=dates.index, dtype="float64")


def _filter_date_range(df: pd.DataFrame, date_col: str, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    working = df
    if start_date is not None:
        start = pd.Timestamp(start_date).normalize()
        working = working.loc[working[date_col] >= start]
    if end_date is not None:
        end = pd.Timestamp(end_date).normalize()
        working = working.loc[working[date_col] <= end]
    return working.copy()


def _prepare_base_observations(
    universe_df: pd.DataFrame,
    date_col: str,
    symbol_col: str,
    universe_flag_col: str,
    config: LabelConfig,
) -> pd.DataFrame:
    base = universe_df[[date_col, symbol_col, universe_flag_col]].copy()
    base = _filter_date_range(base, date_col, config.start_date, config.end_date)
    base = base.sort_values([date_col, symbol_col]).reset_index(drop=True)
    if base.duplicated([date_col, symbol_col]).any():
        raise DataContractError("universe_history contains duplicate (date, symbol) rows after filtering")
    return base.rename(columns={date_col: "date", symbol_col: "symbol", universe_flag_col: "in_universe"})


def _merge_price_for_event(
    base: pd.DataFrame,
    prices_df: pd.DataFrame,
    date_col: str,
    symbol_col: str,
    event_date_col: str,
    price_col: str,
    output_col: str,
) -> pd.DataFrame:
    lookup = prices_df[[date_col, symbol_col, price_col]].rename(
        columns={date_col: event_date_col, symbol_col: "symbol", price_col: output_col}
    )
    return base.merge(lookup, on=[event_date_col, "symbol"], how="left")


def _prepare_costs_table(costs_df: pd.DataFrame | None, date_col: str | None = None, symbol_col: str | None = None) -> pd.DataFrame | None:
    if costs_df is None or costs_df.empty:
        return None
    c_date = date_col or _find_column(costs_df.columns, DATE_CANDIDATES)
    c_symbol = symbol_col or _find_column(costs_df.columns, SYMBOL_CANDIDATES)
    working = costs_df.copy()
    working[c_date] = _to_datetime(working[c_date])
    working[c_symbol] = working[c_symbol].astype(str)
    horizon_col = _find_column(working.columns, ("horizon_days", "horizon", "forward_horizon_days"), required=False)
    rename_map = {c_date: "date", c_symbol: "symbol"}
    if horizon_col is not None:
        rename_map[horizon_col] = "horizon_days"
    working = working.rename(columns=rename_map)
    if "horizon_days" in working.columns:
        working["horizon_days"] = working["horizon_days"].astype(int)

    def pick(*names: str) -> str | None:
        for name in names:
            for col in working.columns:
                if col.lower() == name.lower():
                    return col
        return None

    total_col = pick("total_cost", "total_cost_return", "cost_total", "all_in_cost", "total_cost_pct")
    entry_col = pick("entry_cost", "cost_entry", "entry_cost_return")
    exit_col = pick("exit_cost", "cost_exit", "exit_cost_return")
    carry_col = pick("carry_cost", "borrow_cost", "financing_cost", "cost_carry")
    mean_col = pick("cost_mean", "total_cost_mean")
    var_col = pick("cost_var", "total_cost_var")

    kept = ["date", "symbol"]
    if "horizon_days" in working.columns:
        kept.append("horizon_days")
    for candidate, std_name in [
        (total_col, "total_cost"),
        (entry_col, "entry_cost"),
        (exit_col, "exit_cost"),
        (carry_col, "carry_cost"),
        (mean_col, "cost_mean"),
        (var_col, "cost_var"),
    ]:
        if candidate is not None:
            working = working.rename(columns={candidate: std_name})
            kept.append(std_name)
    if len(kept) <= 2:
        raise DataContractError("execution_costs_path does not contain recognizable cost columns")
    working = working[kept].copy()
    working = working.drop_duplicates([c for c in kept if c in {"date", "symbol", "horizon_days"}], keep="last")
    return working


def _prepare_component_cost_table(df: pd.DataFrame | None, component_name: str) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    date_col = _find_column(df.columns, DATE_CANDIDATES)
    symbol_col = _find_column(df.columns, SYMBOL_CANDIDATES)
    working = df.copy()
    working[date_col] = _to_datetime(working[date_col])
    working[symbol_col] = working[symbol_col].astype(str)
    horizon_col = _find_column(working.columns, ("horizon_days", "horizon", "forward_horizon_days"), required=False)
    value_candidates = {
        "borrow_cost": ("borrow_cost", "borrow_rate_cost", "short_borrow_cost", "cost_borrow"),
        "carry_cost": ("carry_cost", "financing_cost", "funding_cost", "cost_carry"),
    }[component_name]
    value_col = _find_column(working.columns, value_candidates)
    rename_map = {date_col: "date", symbol_col: "symbol", value_col: component_name}
    if horizon_col is not None:
        rename_map[horizon_col] = "horizon_days"
    working = working.rename(columns=rename_map)
    keep = ["date", "symbol", component_name]
    if "horizon_days" in working.columns:
        keep.insert(2, "horizon_days")
        working["horizon_days"] = working["horizon_days"].astype(int)
    working = working[keep].drop_duplicates([c for c in keep if c in {"date", "symbol", "horizon_days"}], keep="last")
    return working


def _prepare_delistings(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    symbol_col = _find_column(df.columns, SYMBOL_CANDIDATES)
    date_col = _find_column(df.columns, ("delisting_date", "date", "effective_date", "event_date"))
    return_col = _find_column(df.columns, ("delisting_return", "dlret", "return", "delist_ret"), required=False)
    monetizable_col = _find_column(df.columns, ("monetizable", "is_monetizable", "cash_settlement_available"), required=False)
    working = df.copy()
    working[symbol_col] = working[symbol_col].astype(str)
    working[date_col] = _to_datetime(working[date_col])
    working = working.rename(columns={symbol_col: "symbol", date_col: "delisting_date"})
    if return_col is not None:
        working = working.rename(columns={return_col: "delisting_return"})
    else:
        working["delisting_return"] = np.nan
    if monetizable_col is not None:
        working = working.rename(columns={monetizable_col: "monetizable"})
        working["monetizable"] = _coerce_bool(working["monetizable"])
    else:
        working["monetizable"] = False
    working = working[["symbol", "delisting_date", "delisting_return", "monetizable"]]
    working = working.drop_duplicates(["symbol", "delisting_date"], keep="last")
    return working


def _prepare_corporate_actions(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    symbol_col = _find_column(df.columns, SYMBOL_CANDIDATES)
    date_col = _find_column(df.columns, ("date", "event_date", "action_date"))
    unsafe_col = _find_column(df.columns, ("unsafe_flag", "is_unsafe", "corporate_action_unsafe"), required=False)
    halt_col = _find_column(df.columns, ("halt_flag", "trading_halt", "suspension_flag", "is_halted"), required=False)
    action_type_col = _find_column(df.columns, ("action_type", "corporate_action_type", "event_type"), required=False)
    working = df.copy()
    working[symbol_col] = working[symbol_col].astype(str)
    working[date_col] = _to_datetime(working[date_col])
    working = working.rename(columns={symbol_col: "symbol", date_col: "event_date"})
    if unsafe_col is not None:
        working = working.rename(columns={unsafe_col: "unsafe_flag"})
        working["unsafe_flag"] = _coerce_bool(working["unsafe_flag"])
    else:
        types = working[action_type_col].astype(str).str.lower() if action_type_col else pd.Series("", index=working.index)
        working["unsafe_flag"] = types.str.contains("merge|spin|tender|reverse|split|liquidation|bankrupt", regex=True)
    if halt_col is not None:
        working = working.rename(columns={halt_col: "halt_flag"})
        working["halt_flag"] = _coerce_bool(working["halt_flag"])
    else:
        types = working[action_type_col].astype(str).str.lower() if action_type_col else pd.Series("", index=working.index)
        working["halt_flag"] = types.str.contains("halt|suspension", regex=True)
    working = working[["symbol", "event_date", "unsafe_flag", "halt_flag"]].drop_duplicates(["symbol", "event_date"], keep="last")
    return working


def _prepare_exposures(df: pd.DataFrame | None) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    if df is None or df.empty:
        return None, [], []
    date_col = _find_column(df.columns, DATE_CANDIDATES)
    symbol_col = _find_column(df.columns, SYMBOL_CANDIDATES)
    working = df.copy()
    working[date_col] = _to_datetime(working[date_col])
    working[symbol_col] = working[symbol_col].astype(str)
    working = working.rename(columns={date_col: "date", symbol_col: "symbol"})
    numeric_candidates = []
    categorical_candidates = []
    for col in working.columns:
        if col in {"date", "symbol"}:
            continue
        lower = col.lower()
        if any(key in lower for key in ("sector", "industry", "group")):
            categorical_candidates.append(col)
        elif any(
            key in lower
            for key in (
                "beta",
                "mkt_cap",
                "market_cap",
                "size",
                "liq",
                "adv",
                "dollar_volume",
                "turnover",
                "vol",
                "volatility",
                "weight",
            )
        ):
            numeric_candidates.append(col)
        elif pd.api.types.is_numeric_dtype(working[col]):
            numeric_candidates.append(col)
    working = working.drop_duplicates(["date", "symbol"], keep="last")
    return working, numeric_candidates, categorical_candidates


def _build_event_flags(
    base: pd.DataFrame,
    actions_df: pd.DataFrame | None,
    delist_df: pd.DataFrame | None,
    date_to_index: Mapping[pd.Timestamp, int],
    horizons: Sequence[int],
) -> pd.DataFrame:
    working = base.copy()
    base_entry_idx = _date_index_series(working["event_start_primary"], date_to_index)
    working["__entry_idx_primary"] = base_entry_idx
    if actions_df is not None and not actions_df.empty:
        act = actions_df.copy()
        act["event_idx"] = _date_index_series(act["event_date"], date_to_index)
        unsafe_map: dict[str, np.ndarray] = {}
        halt_map: dict[str, np.ndarray] = {}
        for sym, group in act.groupby("symbol"):
            unsafe_map[str(sym)] = group.loc[group["unsafe_flag"], "event_idx"].dropna().astype(int).to_numpy()
            halt_map[str(sym)] = group.loc[group["halt_flag"], "event_idx"].dropna().astype(int).to_numpy()
    else:
        unsafe_map = {}
        halt_map = {}

    if delist_df is not None and not delist_df.empty:
        dl = delist_df.copy()
        dl["delist_idx"] = _date_index_series(dl["delisting_date"], date_to_index)
        dl_map = {
            str(sym): grp.sort_values("delist_idx")
            for sym, grp in dl.groupby("symbol")
        }
    else:
        dl_map = {}

    for h in horizons:
        start_idx = _date_index_series(working[f"event_start_{h}d"], date_to_index)
        end_idx = _date_index_series(working[f"event_end_{h}d"], date_to_index)
        corp_unsafe = np.zeros(len(working), dtype=bool)
        halted = np.zeros(len(working), dtype=bool)
        delist_ambiguous = np.zeros(len(working), dtype=bool)
        delisting_return = np.full(len(working), np.nan)
        delisting_event_date = np.array([np.datetime64("NaT")] * len(working), dtype="datetime64[ns]")
        monetizable = np.zeros(len(working), dtype=bool)
        for idx, row in enumerate(working[["symbol"]].itertuples(index=False)):
            sym = str(row.symbol)
            s_idx = start_idx.iat[idx]
            e_idx = end_idx.iat[idx]
            if np.isnan(s_idx) or np.isnan(e_idx):
                continue
            s = int(s_idx)
            e = int(e_idx)
            unsafe_events = unsafe_map.get(sym)
            if unsafe_events is not None and unsafe_events.size:
                pos = np.searchsorted(unsafe_events, s, side="left")
                if pos < unsafe_events.size and unsafe_events[pos] <= e:
                    corp_unsafe[idx] = True
            halt_events = halt_map.get(sym)
            if halt_events is not None and halt_events.size:
                pos = np.searchsorted(halt_events, s, side="left")
                if pos < halt_events.size and halt_events[pos] <= e:
                    halted[idx] = True
            dl_grp = dl_map.get(sym)
            if dl_grp is not None and not dl_grp.empty:
                eligible = dl_grp.loc[(dl_grp["delist_idx"] >= s) & (dl_grp["delist_idx"] <= e)]
                if not eligible.empty:
                    first = eligible.iloc[0]
                    delisting_event_date[idx] = np.datetime64(first["delisting_date"])
                    monetizable[idx] = bool(first.get("monetizable", False))
                    if pd.notna(first.get("delisting_return")):
                        delisting_return[idx] = float(first["delisting_return"])
                    elif not monetizable[idx]:
                        delist_ambiguous[idx] = True
        working[f"corporate_action_unsafe_{h}d"] = corp_unsafe
        working[f"trading_halt_or_suspension_{h}d"] = halted
        working[f"delist_ambiguous_{h}d"] = delist_ambiguous
        working[f"delisting_return_{h}d"] = delisting_return
        working[f"delisting_event_date_{h}d"] = pd.to_datetime(delisting_event_date)
        working[f"delisting_monetizable_{h}d"] = monetizable
    working = working.drop(columns=["__entry_idx_primary"])
    return working


def _attach_cost_components(
    df: pd.DataFrame,
    config: LabelConfig,
    costs_df: pd.DataFrame | None,
    borrow_df: pd.DataFrame | None,
    carry_df: pd.DataFrame | None,
) -> pd.DataFrame:
    working = df.copy()
    for h in config.horizons:
        base_cols = []
        if costs_df is not None:
            cost_slice = costs_df.copy()
            if "horizon_days" in cost_slice.columns:
                cost_slice = cost_slice.loc[cost_slice["horizon_days"] == h]
            rename_map = {col: f"{col}_{h}d" for col in cost_slice.columns if col not in {"date", "symbol", "horizon_days"}}
            cost_slice = cost_slice.rename(columns=rename_map)
            keep_cols = ["date", "symbol", *rename_map.values()]
            working = working.merge(cost_slice[keep_cols], on=["date", "symbol"], how="left")
            base_cols.extend(rename_map.values())
        if borrow_df is not None:
            borrow_slice = borrow_df.copy()
            if "horizon_days" in borrow_slice.columns:
                borrow_slice = borrow_slice.loc[borrow_slice["horizon_days"] == h]
            borrow_slice = borrow_slice.rename(columns={"borrow_cost": f"borrow_cost_{h}d"})
            keep = ["date", "symbol", f"borrow_cost_{h}d"]
            working = working.merge(borrow_slice[keep], on=["date", "symbol"], how="left")
            base_cols.append(f"borrow_cost_{h}d")
        if carry_df is not None:
            carry_slice = carry_df.copy()
            if "horizon_days" in carry_slice.columns:
                carry_slice = carry_slice.loc[carry_slice["horizon_days"] == h]
            carry_slice = carry_slice.rename(columns={"carry_cost": f"carry_cost_component_{h}d"})
            keep = ["date", "symbol", f"carry_cost_component_{h}d"]
            working = working.merge(carry_slice[keep], on=["date", "symbol"], how="left")
            base_cols.append(f"carry_cost_component_{h}d")

        entry_col = f"entry_cost_{h}d"
        exit_col = f"exit_cost_{h}d"
        carry_col = f"carry_cost_{h}d"
        mean_col = f"cost_mean_{h}d"
        var_col = f"cost_var_{h}d"
        total_cost_col = f"total_cost_{h}d"

        if total_cost_col not in working.columns:
            total = pd.Series(np.nan, index=working.index, dtype=float)
        else:
            total = working[total_cost_col].astype(float)

        if entry_col not in working.columns:
            if total_cost_col in working.columns:
                working[entry_col] = total / 2.0
            else:
                working[entry_col] = np.nan
        if exit_col not in working.columns:
            if total_cost_col in working.columns:
                working[exit_col] = total / 2.0
            else:
                working[exit_col] = np.nan
        if carry_col not in working.columns:
            base_carry = pd.Series(0.0, index=working.index)
            if f"borrow_cost_{h}d" in working.columns:
                base_carry = base_carry.add(working[f"borrow_cost_{h}d"].fillna(0.0), fill_value=0.0)
            if f"carry_cost_component_{h}d" in working.columns:
                base_carry = base_carry.add(working[f"carry_cost_component_{h}d"].fillna(0.0), fill_value=0.0)
            working[carry_col] = base_carry.replace(0.0, np.nan)
        if mean_col not in working.columns and total_cost_col in working.columns:
            working[mean_col] = working[total_cost_col]
        if var_col not in working.columns:
            working[var_col] = np.nan

        if config.cost_policy.missing_cost_policy == "fallback_proxy":
            missing_all = working[[entry_col, exit_col, carry_col]].isna().all(axis=1)
            proxy_total = config.cost_policy.fallback_total_cost_bps / 10000.0
            proxy_carry = (
                (config.cost_policy.annual_borrow_rate + config.cost_policy.annual_carry_rate)
                * h
                / 252.0
            )
            working.loc[missing_all, entry_col] = proxy_total / 2.0
            working.loc[missing_all, exit_col] = proxy_total / 2.0
            working.loc[missing_all, carry_col] = proxy_carry
        elif config.cost_policy.missing_cost_policy == "zero_imputation":
            working[[entry_col, exit_col, carry_col]] = working[[entry_col, exit_col, carry_col]].fillna(0.0)
        elif config.cost_policy.missing_cost_policy == "partial_costs":
            working[entry_col] = working[entry_col].fillna(0.0)
            working[exit_col] = working[exit_col].fillna(0.0)
            working[carry_col] = working[carry_col].fillna(0.0)

    return working


def _build_labels_core(
    prices_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    config: LabelConfig,
    costs_df: pd.DataFrame | None = None,
    borrow_df: pd.DataFrame | None = None,
    carry_df: pd.DataFrame | None = None,
    exposures_df: pd.DataFrame | None = None,
    delist_df: pd.DataFrame | None = None,
    actions_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[QCRecord], dict[str, Any], pd.DataFrame]:
    prices, price_date_col, price_symbol_col, entry_field, exit_field = _prepare_prices(prices_df, config)
    universe, uni_date_col, uni_symbol_col, universe_flag_col = _prepare_universe(universe_df)
    calendar, cal_date_col = _prepare_calendar(calendar_df)
    calendar_dates, date_to_index = _build_calendar_index(calendar, cal_date_col)

    base = _prepare_base_observations(universe, uni_date_col, uni_symbol_col, universe_flag_col, config)
    base["date_idx"] = _date_index_series(base["date"], date_to_index)
    base["out_of_sample_flag"] = False
    if config.start_date is not None:
        base.loc[base["date"] < pd.Timestamp(config.start_date), "out_of_sample_flag"] = True
    if config.end_date is not None:
        base.loc[base["date"] > pd.Timestamp(config.end_date), "out_of_sample_flag"] = True

    entry_date_primary = _date_offsets(base["date"], date_to_index, config.decision_lag, calendar_dates)
    base["event_start_primary"] = entry_date_primary
    base = _attach_cost_components(base, config, costs_df, borrow_df, carry_df)

    for h in config.horizons:
        base[f"event_start_{h}d"] = entry_date_primary
        base[f"event_end_{h}d"] = _date_offsets(
            base["date"], date_to_index, config.decision_lag + h, calendar_dates
        )
        base[f"effective_horizon_days_{h}d"] = h
        base[f"entry_price_field_{h}d"] = entry_field
        base[f"exit_price_field_{h}d"] = exit_field
        base[f"entry_calendar_rule_{h}d"] = f"date + decision_lag({config.decision_lag})"
        base[f"exit_calendar_rule_{h}d"] = f"date + decision_lag({config.decision_lag}) + horizon({h})"
        base = _merge_price_for_event(
            base,
            prices,
            price_date_col,
            price_symbol_col,
            f"event_start_{h}d",
            entry_field,
            f"entry_px_{h}d",
        )
        base = _merge_price_for_event(
            base,
            prices,
            price_date_col,
            price_symbol_col,
            f"event_end_{h}d",
            exit_field,
            f"exit_px_{h}d",
        )

    base = _build_event_flags(base, actions_df, delist_df, date_to_index, config.horizons)

    for h in config.horizons:
        entry_px_col = f"entry_px_{h}d"
        exit_px_col = f"exit_px_{h}d"
        gross_col = f"y_fwd_ret_gross_{h}d"
        net_col = f"y_fwd_ret_net_{h}d"
        mean_col = f"y_net_mean_{h}d"
        se_col = f"y_net_se_{h}d"
        ci_low_col = f"y_net_ci_low_{h}d"
        ci_high_col = f"y_net_ci_high_{h}d"
        entry_cost_col = f"entry_cost_{h}d"
        exit_cost_col = f"exit_cost_{h}d"
        carry_cost_col = f"carry_cost_{h}d"
        cost_mean_col = f"cost_mean_{h}d"
        cost_var_col = f"cost_var_{h}d"

        incomplete_window = base[f"event_end_{h}d"].isna()
        missing_entry_price = base[f"event_start_{h}d"].notna() & base[entry_px_col].isna()
        missing_exit_price = base[f"event_end_{h}d"].notna() & base[exit_px_col].isna()
        base[f"incomplete_forward_window_{h}d"] = incomplete_window
        base[f"missing_entry_price_{h}d"] = missing_entry_price
        base[f"missing_exit_price_{h}d"] = missing_exit_price

        gross = (base[exit_px_col] / base[entry_px_col]) - 1.0
        dl_return_col = f"delisting_return_{h}d"
        dl_amb_col = f"delist_ambiguous_{h}d"
        resolved_dlist = base[dl_return_col].notna() & ~base[dl_amb_col]
        gross = gross.where(~resolved_dlist, base[dl_return_col])
        gross = gross.where(base[entry_px_col].notna())
        base[gross_col] = gross

        if config.net_of_costs:
            material_missing_costs = base[[entry_cost_col, exit_cost_col, carry_cost_col]].isna().any(axis=1)
            base[f"missing_cost_input_{h}d"] = material_missing_costs
            if config.cost_policy.missing_cost_policy == "strict_invalidate":
                base[net_col] = gross - base[entry_cost_col].fillna(0.0) - base[exit_cost_col].fillna(0.0) - base[carry_cost_col].fillna(0.0)
                base.loc[material_missing_costs, net_col] = np.nan
            else:
                base[net_col] = gross - base[entry_cost_col].fillna(0.0) - base[exit_cost_col].fillna(0.0) - base[carry_cost_col].fillna(0.0)
            if cost_mean_col in base.columns:
                base[mean_col] = gross - base[cost_mean_col].fillna(
                    base[entry_cost_col].fillna(0.0) + base[exit_cost_col].fillna(0.0) + base[carry_cost_col].fillna(0.0)
                )
            else:
                base[mean_col] = np.nan
            if cost_var_col in base.columns:
                base[se_col] = np.sqrt(base[cost_var_col].clip(lower=0.0))
                z = config.cost_policy.ci_zscore
                base[ci_low_col] = base[mean_col] - z * base[se_col]
                base[ci_high_col] = base[mean_col] + z * base[se_col]
            else:
                base[se_col] = np.nan
                base[ci_low_col] = np.nan
                base[ci_high_col] = np.nan
        else:
            base[f"missing_cost_input_{h}d"] = False
            base[net_col] = np.nan
            base[mean_col] = np.nan
            base[se_col] = np.nan
            base[ci_low_col] = np.nan
            base[ci_high_col] = np.nan

    exposures_work, numeric_exposures, categorical_exposures = _prepare_exposures(exposures_df)
    if config.neutralization_policy.enabled:
        if exposures_work is None:
            raise ConfigError("neutralization requested but neutralization_exposures_path is missing or empty")
        base = base.merge(exposures_work, on=["date", "symbol"], how="left", suffixes=("", "_exp"))
    else:
        exposures_work = None
        numeric_exposures = []
        categorical_exposures = []

    coverage_rows: list[dict[str, Any]] = []
    qc_records: list[QCRecord] = []
    dictionary: dict[str, Any] = {
        "entry_price_field": entry_field,
        "exit_price_field": exit_field,
        "entry_calendar_rule": f"date + decision_lag({config.decision_lag})",
        "exit_calendar_rules": {str(h): f"date + decision_lag({config.decision_lag}) + horizon({h})" for h in config.horizons},
        "return_mode": config.return_mode,
        "policy_version": config.policy_version,
        "primary_target": config.primary_target,
        "label_mode": config.label_mode,
        "classification_policy": asdict(config.classification_policy),
        "ranking_policy": asdict(config.ranking_policy),
        "neutralization_policy": asdict(config.neutralization_policy),
    }

    for h in config.horizons:
        gross_col = f"y_fwd_ret_gross_{h}d"
        net_col = f"y_fwd_ret_net_{h}d"
        base_target_col = net_col if config.net_of_costs else gross_col
        primary_reason_col = f"label_exclusion_reason_{h}d"
        primary_valid_col = f"label_valid_flag_{h}d"

        reason_masks: dict[str, pd.Series] = {
            "OUT_OF_SAMPLE_DATE": base["out_of_sample_flag"],
            "NOT_IN_UNIVERSE": ~base["in_universe"].fillna(False),
            "MISSING_ENTRY_PRICE": base[f"missing_entry_price_{h}d"],
            "MISSING_EXIT_PRICE": base[f"missing_exit_price_{h}d"],
            "INCOMPLETE_FORWARD_WINDOW": base[f"incomplete_forward_window_{h}d"],
            "CORPORATE_ACTION_UNSAFE": base[f"corporate_action_unsafe_{h}d"],
            "TRADING_HALT_OR_SUSPENSION": base[f"trading_halt_or_suspension_{h}d"],
            "DELIST_AMBIGUOUS": base[f"delist_ambiguous_{h}d"],
            "MISSING_COST_INPUT": base[f"missing_cost_input_{h}d"] if config.net_of_costs else pd.Series(False, index=base.index),
            "MISSING_EXPOSURES": pd.Series(False, index=base.index),
            "NEUTRALIZATION_FAILED": pd.Series(False, index=base.index),
            "QC_REJECTED": pd.Series(False, index=base.index),
        }

        base[primary_reason_col] = None
        for reason in EXCLUSION_PRIORITY:
            mask = reason_masks[reason].fillna(False)
            base.loc[base[primary_reason_col].isna() & mask, primary_reason_col] = reason
        base[primary_valid_col] = base[primary_reason_col].isna() & base[base_target_col].notna()

        valid_target = base.loc[base[primary_valid_col], ["date", base_target_col]].copy()
        counts_by_date = valid_target.groupby("date").size().rename("n_valid_cross_section")
        base[f"n_valid_cross_section_{h}d"] = base["date"].map(counts_by_date).fillna(0).astype(int)
        qc_rejected = base[f"n_valid_cross_section_{h}d"] < config.min_valid_cross_section
        reason_masks["QC_REJECTED"] = qc_rejected
        base.loc[qc_rejected & base[primary_reason_col].isna(), primary_reason_col] = "QC_REJECTED"
        base[primary_valid_col] = base[primary_reason_col].isna() & base[base_target_col].notna()
        valid_target = base.loc[base[primary_valid_col], ["date", base_target_col]].copy()
        counts_by_date = valid_target.groupby("date").size().rename("n_valid_cross_section")
        base[f"n_valid_cross_section_{h}d"] = base["date"].map(counts_by_date).fillna(0).astype(int)

        if config.ranking_policy.enabled:
            rank_col = f"y_rank_{h}d"
            base[rank_col] = np.nan
            subset = base.loc[base[primary_valid_col], ["date", base_target_col]].copy()
            if not subset.empty:
                subset[rank_col] = subset.groupby("date")[base_target_col].rank(
                    method=config.ranking_policy.method,
                    pct=True,
                )
                base.loc[subset.index, rank_col] = subset[rank_col]
        else:
            base[f"y_rank_{h}d"] = np.nan

        if config.classification_policy.enabled:
            cls_kind = config.classification_policy.kind
            if cls_kind == "tail":
                cls_col = f"y_cls_tail_{h}d"
                base[cls_col] = np.nan
                valid_idx = base.index[base[primary_valid_col]]
                if len(valid_idx):
                    grouped = base.loc[valid_idx, ["date", base_target_col]].groupby("date")[base_target_col]
                    top = grouped.transform(lambda s: s.quantile(config.classification_policy.top_quantile))
                    bot = grouped.transform(lambda s: s.quantile(config.classification_policy.bottom_quantile))
                    vals = base.loc[valid_idx, base_target_col]
                    labels = np.where(vals >= top, 1, np.where(vals <= bot, -1, 0))
                    base.loc[valid_idx, cls_col] = labels
            else:
                cls_col = f"y_cls_up_{h}d"
                base[cls_col] = np.nan
                valid_idx = base.index[base[primary_valid_col]]
                if len(valid_idx):
                    thr = config.classification_policy.fixed_threshold
                    base.loc[valid_idx, cls_col] = (base.loc[valid_idx, base_target_col] > thr).astype(int)
        else:
            base[f"y_cls_tail_{h}d"] = np.nan
            base[f"y_cls_up_{h}d"] = np.nan

        if config.neutralization_policy.enabled:
            neut_col = f"y_neut_{h}d"
            base[neut_col] = np.nan
            base[f"ridge_lambda_used_{h}d"] = np.nan
            base[f"neutralization_fit_status_{h}d"] = "not_run"
            base[f"missing_exposures_{h}d"] = False
            base[f"neutralization_failed_{h}d"] = False
            for dt, group_idx in base.groupby("date").groups.items():
                group_idx = list(group_idx)
                valid_idx = [idx for idx in group_idx if bool(base.at[idx, primary_valid_col])]
                if not valid_idx:
                    continue
                y = base.loc[valid_idx, base_target_col].astype(float)
                X, missing_ratio = _design_matrix_for_date(
                    base.loc[valid_idx],
                    numeric_exposures,
                    categorical_exposures,
                    config.neutralization_policy.weight_col,
                )
                base.loc[valid_idx, f"missing_exposures_{h}d"] = missing_ratio > config.neutralization_policy.max_missing_exposure_ratio
                if missing_ratio > config.neutralization_policy.max_missing_exposure_ratio:
                    base.loc[valid_idx, f"neutralization_fit_status_{h}d"] = "missing_exposures"
                    base.loc[valid_idx, f"neutralization_failed_{h}d"] = True
                    continue
                if len(y) < config.neutralization_policy.n_min_neut:
                    base.loc[valid_idx, f"neutralization_fit_status_{h}d"] = "insufficient_cross_section"
                    base.loc[valid_idx, f"neutralization_failed_{h}d"] = True
                    continue
                try:
                    result = _weighted_ridge_residuals(
                        y,
                        X,
                        weights=None,
                        policy=config.neutralization_policy,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.exception("neutralization failed on %s horizon %sd", dt, h)
                    base.loc[valid_idx, f"neutralization_fit_status_{h}d"] = f"exception:{type(exc).__name__}"
                    base.loc[valid_idx, f"neutralization_failed_{h}d"] = True
                    continue
                base.loc[valid_idx, neut_col] = result.residuals.to_numpy()
                base.loc[valid_idx, f"ridge_lambda_used_{h}d"] = result.lambda_used
                base.loc[valid_idx, f"neutralization_fit_status_{h}d"] = result.fit_status
                fail_mask = (
                    result.fit_status != "ok"
                    or result.missing_ratio > config.neutralization_policy.max_missing_exposure_ratio
                    or (result.condition_number is not None and result.condition_number > config.neutralization_policy.condition_number_threshold)
                    or (result.residuals.var(ddof=1) if len(result.residuals) > 1 else 0.0) < config.neutralization_policy.residual_var_floor
                )
                if fail_mask:
                    base.loc[valid_idx, neut_col] = np.nan
                    base.loc[valid_idx, f"neutralization_failed_{h}d"] = True
            base.loc[base[f"missing_exposures_{h}d"] & base[f"label_exclusion_reason_{h}d"].isna(), f"label_exclusion_reason_{h}d"] = "MISSING_EXPOSURES"
            if config.primary_target == neut_col:
                base.loc[base[f"neutralization_failed_{h}d"] & base[f"label_exclusion_reason_{h}d"].isna(), f"label_exclusion_reason_{h}d"] = "NEUTRALIZATION_FAILED"
                base[primary_valid_col] = base[f"label_exclusion_reason_{h}d"].isna() & base[neut_col].notna()
        else:
            base[f"y_neut_{h}d"] = np.nan
            base[f"ridge_lambda_used_{h}d"] = np.nan
            base[f"neutralization_fit_status_{h}d"] = "disabled"
            base[f"missing_exposures_{h}d"] = False
            base[f"neutralization_failed_{h}d"] = False

        coverage_rows.extend(
            _coverage_rows_for_horizon(base, h, config)
        )

    primary_h = _parse_primary_horizon(config.primary_target, config.horizons)
    primary_reason_col = f"label_exclusion_reason_{primary_h}d"
    primary_valid_col = f"label_valid_flag_{primary_h}d"
    base["label_exclusion_reason"] = base[primary_reason_col]
    base["label_valid_flag"] = base[primary_valid_col]
    base["n_valid_cross_section"] = base[f"n_valid_cross_section_{primary_h}d"]
    base["run_id"] = ""

    qc_records.extend(_run_label_qc(base, config, primary_h))
    coverage_df = pd.DataFrame(coverage_rows)
    split_compatibility = {
        "event_windows": {
            f"{h}d": {
                "event_start_col": f"event_start_{h}d",
                "event_end_col": f"event_end_{h}d",
                "decision_lag": config.decision_lag,
                "horizon_days": h,
            }
            for h in config.horizons
        },
        "primary_target": config.primary_target,
        "primary_horizon": primary_h,
    }
    return base, qc_records, dictionary | split_compatibility, coverage_df


def _coverage_rows_for_horizon(df: pd.DataFrame, horizon: int, config: LabelConfig) -> list[dict[str, Any]]:
    reason_col = f"label_exclusion_reason_{horizon}d"
    valid_col = f"label_valid_flag_{horizon}d"
    base_target_col = f"y_fwd_ret_net_{horizon}d" if config.net_of_costs else f"y_fwd_ret_gross_{horizon}d"
    rows = []
    grouped = df.groupby("date", sort=True)
    for dt, grp in grouped:
        valid_grp = grp.loc[grp[valid_col] & grp[base_target_col].notna()]
        row = {
            "date": dt,
            "horizon_days": horizon,
            "n_rows": int(len(grp)),
            "n_valid": int(len(valid_grp)),
            "coverage": float(len(valid_grp) / len(grp)) if len(grp) else np.nan,
            "target_mean": float(valid_grp[base_target_col].mean()) if len(valid_grp) else np.nan,
            "target_var": float(valid_grp[base_target_col].var(ddof=1)) if len(valid_grp) > 1 else np.nan,
            "n_qc_rejected": int((grp[reason_col] == "QC_REJECTED").sum()),
            "n_missing_cost": int((grp[reason_col] == "MISSING_COST_INPUT").sum()),
            "n_missing_exposures": int((grp[reason_col] == "MISSING_EXPOSURES").sum()),
            "n_neutralization_failed": int((grp[reason_col] == "NEUTRALIZATION_FAILED").sum()),
        }
        tail_col = f"y_cls_tail_{horizon}d"
        if tail_col in grp.columns:
            valid_cls = grp.loc[grp[valid_col], tail_col].dropna()
            row["n_cls_pos"] = int((valid_cls == 1).sum())
            row["n_cls_neg"] = int((valid_cls == -1).sum())
            row["n_cls_zero"] = int((valid_cls == 0).sum())
        rows.append(row)
    return rows


def _design_matrix_for_date(
    group: pd.DataFrame,
    numeric_exposures: Sequence[str],
    categorical_exposures: Sequence[str],
    weight_col: str | None,
) -> tuple[pd.DataFrame, float]:
    df = group.copy()
    design_parts = [pd.Series(1.0, index=df.index, name="intercept")]
    missing_tracker = pd.DataFrame(index=df.index)
    for col in numeric_exposures:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        missing_tracker[col] = series.isna()
        filled = series.fillna(series.median())
        std = float(filled.std(ddof=0))
        if std > 0:
            filled = (filled - float(filled.mean())) / std
        else:
            filled = filled * 0.0
        design_parts.append(filled.rename(col))
    for col in categorical_exposures:
        if col not in df.columns:
            continue
        cat = df[col].astype(str).replace({"nan": "missing", "None": "missing"}).fillna("missing")
        dummies = pd.get_dummies(cat, prefix=col, drop_first=True, dtype=float)
        if not dummies.empty:
            design_parts.append(dummies)
    design = pd.concat(design_parts, axis=1)
    if weight_col is not None and weight_col in df.columns:
        weights = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0)
        design["__weight__"] = weights
    missing_ratio = float(missing_tracker.any(axis=1).mean()) if not missing_tracker.empty else 0.0
    return design, missing_ratio


def _weighted_ridge_residuals(
    y: pd.Series,
    X: pd.DataFrame,
    weights: pd.Series | None,
    policy: NeutralizationPolicy,
) -> NeutralizationResult:
    working = X.copy()
    if "__weight__" in working.columns:
        w = pd.to_numeric(working.pop("__weight__"), errors="coerce").fillna(1.0).to_numpy(dtype=float)
    elif weights is not None:
        w = pd.to_numeric(weights, errors="coerce").fillna(1.0).to_numpy(dtype=float)
    else:
        w = np.ones(len(working), dtype=float)
    X_mat = working.to_numpy(dtype=float)
    y_vec = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y_vec) & np.all(np.isfinite(X_mat), axis=1) & np.isfinite(w)
    if valid.sum() < policy.n_min_neut:
        return NeutralizationResult(
            residuals=pd.Series(np.nan, index=y.index),
            lambda_used=None,
            fit_status="insufficient_valid_rows",
            missing_ratio=1.0 - float(valid.mean()) if len(valid) else 1.0,
            condition_number=None,
        )
    Xv = X_mat[valid]
    yv = y_vec[valid]
    wv = np.clip(w[valid], 1e-12, None)
    sqrt_w = np.sqrt(wv)
    Xw = Xv * sqrt_w[:, None]
    yw = yv * sqrt_w

    def fit_for_lambda(lmbda: float) -> tuple[np.ndarray, float]:
        gram = Xw.T @ Xw
        penalty = np.eye(gram.shape[0]) * float(lmbda)
        penalty[0, 0] = 0.0
        lhs = gram + penalty
        rhs = Xw.T @ yw
        beta = np.linalg.solve(lhs, rhs)
        cond = float(np.linalg.cond(lhs))
        return beta, cond

    lambda_used: float | None = None
    best_beta: np.ndarray | None = None
    best_cond: float | None = None
    if policy.lambda_selection == "grid_search":
        best_score = math.inf
        for lmbda in policy.lambda_grid:
            try:
                beta, cond = fit_for_lambda(lmbda)
            except np.linalg.LinAlgError:
                continue
            resid = yv - Xv @ beta
            score = float(np.average(resid**2, weights=wv))
            if score < best_score:
                best_score = score
                best_beta = beta
                best_cond = cond
                lambda_used = float(lmbda)
    else:
        lambda_used = float(policy.fixed_lambda)
        beta, cond = fit_for_lambda(lambda_used)
        best_beta = beta
        best_cond = cond

    if best_beta is None:
        return NeutralizationResult(
            residuals=pd.Series(np.nan, index=y.index),
            lambda_used=lambda_used,
            fit_status="solve_failed",
            missing_ratio=1.0 - float(valid.mean()),
            condition_number=best_cond,
        )

    if best_cond is not None and best_cond > policy.condition_number_threshold:
        return NeutralizationResult(
            residuals=pd.Series(np.nan, index=y.index),
            lambda_used=lambda_used,
            fit_status="ill_conditioned",
            missing_ratio=1.0 - float(valid.mean()),
            condition_number=best_cond,
        )

    fitted = Xv @ best_beta
    resid_valid = yv - fitted
    if len(resid_valid) > 1 and float(np.var(resid_valid, ddof=1)) < policy.residual_var_floor:
        return NeutralizationResult(
            residuals=pd.Series(np.nan, index=y.index),
            lambda_used=lambda_used,
            fit_status="degenerate_residual_variance",
            missing_ratio=1.0 - float(valid.mean()),
            condition_number=best_cond,
        )

    residuals = pd.Series(np.nan, index=y.index, dtype=float)
    residuals.loc[y.index[valid]] = resid_valid
    return NeutralizationResult(
        residuals=residuals,
        lambda_used=lambda_used,
        fit_status="ok",
        missing_ratio=1.0 - float(valid.mean()),
        condition_number=best_cond,
    )


def _parse_primary_horizon(primary_target: str, horizons: Sequence[int]) -> int:
    for h in sorted(horizons, reverse=True):
        if f"_{h}d" in primary_target:
            return h
    raise ConfigError(f"could not infer primary horizon from primary_target='{primary_target}'")


def _run_label_qc(df: pd.DataFrame, config: LabelConfig, primary_horizon: int) -> list[QCRecord]:
    records: list[QCRecord] = []
    dup_count = int(df.duplicated(["date", "symbol"]).sum())
    records.append(
        QCRecord(
            check_name="structure.no_duplicate_date_symbol",
            severity=SEVERITY_FAIL,
            status="PASS" if dup_count == 0 else "FAIL",
            metric_value=dup_count,
            message="No duplicate (date, symbol) rows in output labels",
        )
    )
    out_of_universe = int((~df["in_universe"]).sum())
    records.append(
        QCRecord(
            check_name="universe.labels_within_universe_contract",
            severity=SEVERITY_FAIL,
            status="PASS" if out_of_universe == 0 else "FAIL",
            metric_value=out_of_universe,
            message="Rows not eligible in universe_history must not remain valid in final labels",
        )
    )
    for h in config.horizons:
        valid_col = f"label_valid_flag_{h}d"
        target_col = f"y_fwd_ret_net_{h}d" if config.net_of_costs else f"y_fwd_ret_gross_{h}d"
        coverage = float(df[valid_col].mean()) if len(df) else np.nan
        records.append(
            QCRecord(
                check_name=f"coverage.h{h}",
                severity=SEVERITY_WARN,
                status="PASS" if (np.isnan(coverage) or coverage >= 0.05) else "WARN",
                metric_value=coverage,
                message=f"Coverage for horizon {h}d should not collapse structurally",
            )
        )
        valid_target = df.loc[df[valid_col], ["date", target_col]]
        per_date_var = valid_target.groupby("date")[target_col].var(ddof=1)
        collapsed = int((per_date_var.fillna(0.0) <= 0.0).sum())
        records.append(
            QCRecord(
                check_name=f"variance.cross_section_positive.h{h}",
                severity=SEVERITY_WARN if h != primary_horizon else SEVERITY_FAIL,
                status="PASS" if collapsed == 0 else ("FAIL" if h == primary_horizon else "WARN"),
                metric_value=collapsed,
                message=f"Cross-sectional variance for horizon {h}d should be positive on valid dates",
            )
        )
        tail_col = f"y_cls_tail_{h}d"
        if tail_col in df.columns:
            by_date = df.loc[df[valid_col]].groupby("date")[tail_col]
            collapsed_cls = 0
            for _, series in by_date:
                observed = set(pd.Series(series).dropna().astype(int).tolist())
                if len(observed) <= 1:
                    collapsed_cls += 1
            records.append(
                QCRecord(
                    check_name=f"classification.non_collapsed.h{h}",
                    severity=SEVERITY_WARN,
                    status="PASS" if collapsed_cls == 0 else "WARN",
                    metric_value=collapsed_cls,
                    message=f"Tail classification for horizon {h}d should not collapse to a single class",
                )
            )
        if config.net_of_costs:
            bad = int(df.loc[df[valid_col], f"missing_cost_input_{h}d"].sum())
            records.append(
                QCRecord(
                    check_name=f"costs.net_target_requires_costs.h{h}",
                    severity=SEVERITY_FAIL,
                    status="PASS" if bad == 0 else "FAIL",
                    metric_value=bad,
                    message=f"No valid net targets may exist with missing material cost inputs for horizon {h}d",
                )
            )
        if config.neutralization_policy.enabled:
            failed = int(df[f"neutralization_failed_{h}d"].sum())
            records.append(
                QCRecord(
                    check_name=f"neutralization.fit_status.h{h}",
                    severity=SEVERITY_INFO,
                    status="INFO",
                    metric_value=failed,
                    message=f"Count of rows whose neutralized target failed for horizon {h}d",
                )
            )
    return records


def _enforce_qc(records: Sequence[QCRecord]) -> None:
    fails = [r for r in records if r.severity == SEVERITY_FAIL and r.status == "FAIL"]
    if fails:
        messages = "; ".join(f"{r.check_name}: {r.metric_value}" for r in fails)
        raise QCFailure(messages)


def _persist_partitioned_labels(df: pd.DataFrame, output_dir: Path, run_id: str) -> Path:
    store_dir = output_dir / "store"
    store_dir.mkdir(parents=True, exist_ok=True)
    for dt, grp in df.groupby("date", sort=True):
        date_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
        part_dir = store_dir / f"date={date_str}"
        part_dir.mkdir(parents=True, exist_ok=True)
        _write_frame(grp, part_dir / f"part-{run_id}.parquet")
    return store_dir


def _manifest_payload(
    run_id: str,
    config: LabelConfig,
    input_paths: InputPaths,
    dictionary: Mapping[str, Any],
) -> dict[str, Any]:
    config_payload = config.to_serializable()
    return {
        "run_id": run_id,
        "config_hash": _hash_payload(config_payload),
        "calendar_version": _fingerprint_path(input_paths.trading_calendar_path),
        "universe_history_version": _fingerprint_path(input_paths.universe_history_path),
        "adjusted_prices_version": _fingerprint_path(input_paths.adjusted_prices_pit_path),
        "cost_model_version": _fingerprint_path(input_paths.execution_costs_path),
        "exposures_version": _fingerprint_path(input_paths.neutralization_exposures_path),
        "delisting_returns_version": _fingerprint_path(input_paths.delisting_returns_path),
        "corporate_actions_version": _fingerprint_path(input_paths.corporate_actions_path),
        "policy_version": config.policy_version,
        "primary_target": config.primary_target,
        "defaults_canonicos": CANONICAL_DEFAULTS,
        "overrides": config_payload,
        "label_mode": config.label_mode,
        "non_canonical_flag": config.non_canonical_flag,
        "entry_price_field": dictionary.get("entry_price_field"),
        "exit_price_field": dictionary.get("exit_price_field"),
        "entry_calendar_rule": dictionary.get("entry_calendar_rule"),
        "exit_calendar_rules": dictionary.get("exit_calendar_rules"),
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def build_labels(
    *,
    input_paths: InputPaths,
    run_id: str,
) -> dict[str, Any]:
    config = load_label_config(input_paths.label_config_path)
    if input_paths.output_dir is not None and config.output_dir is None:
        config_payload = config.to_serializable()
        config_payload["output_dir"] = input_paths.output_dir
        config = LabelConfig.from_mapping(config_payload)

    prices_df = load_table(input_paths.adjusted_prices_pit_path)
    universe_df = load_table(input_paths.universe_history_path)
    calendar_df = load_table(input_paths.trading_calendar_path)
    if prices_df is None or universe_df is None or calendar_df is None:
        raise DataContractError("required inputs could not be loaded")
    costs_df = _prepare_costs_table(load_table(input_paths.execution_costs_path))
    borrow_df = _prepare_component_cost_table(load_table(input_paths.borrow_costs_path), "borrow_cost")
    carry_df = _prepare_component_cost_table(load_table(input_paths.carry_costs_path), "carry_cost")
    exposures_df = load_table(input_paths.neutralization_exposures_path)
    delist_df = _prepare_delistings(load_table(input_paths.delisting_returns_path))
    actions_df = _prepare_corporate_actions(load_table(input_paths.corporate_actions_path))

    labels_df, qc_records, dictionary, coverage_df = _build_labels_core(
        prices_df=prices_df,
        universe_df=universe_df,
        calendar_df=calendar_df,
        config=config,
        costs_df=costs_df,
        borrow_df=borrow_df,
        carry_df=carry_df,
        exposures_df=exposures_df,
        delist_df=delist_df,
        actions_df=actions_df,
    )
    labels_df["run_id"] = run_id
    _enforce_qc(qc_records)

    output_dir = Path(config.output_dir or input_paths.output_dir or f"build_labels_{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = _persist_partitioned_labels(labels_df, output_dir, run_id)
    all_labels_path = output_dir / f"labels_{run_id}.parquet"
    all_labels_path = _write_frame(labels_df, all_labels_path)
    coverage_path = output_dir / f"label_coverage_{run_id}.parquet"
    coverage_path = _write_frame(coverage_df, coverage_path)

    dictionary_path = output_dir / f"label_dictionary_{run_id}.json"
    dictionary_path.write_text(json.dumps(_json_safe(dictionary), indent=2, sort_keys=True), encoding="utf-8")

    manifest = _manifest_payload(run_id, config, input_paths, dictionary)
    manifest_path = output_dir / f"label_manifest_{run_id}.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")

    split_compatibility = {
        "run_id": run_id,
        "primary_target": config.primary_target,
        "decision_lag": config.decision_lag,
        "horizons": list(config.horizons),
        "event_start_cols": {f"{h}d": f"event_start_{h}d" for h in config.horizons},
        "event_end_cols": {f"{h}d": f"event_end_{h}d" for h in config.horizons},
        "label_mode": config.label_mode,
    }
    split_compatibility_path = output_dir / f"split_compatibility_{run_id}.json"
    split_compatibility_path.write_text(
        json.dumps(_json_safe(split_compatibility), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    qc_payload = {
        "run_id": run_id,
        "checks": [record.to_dict() for record in qc_records],
    }
    qc_path = output_dir / f"label_qc_{run_id}.json"
    qc_path.write_text(json.dumps(_json_safe(qc_payload), indent=2, sort_keys=True), encoding="utf-8")

    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "labels_store": str(labels_path),
        "labels_parquet": str(all_labels_path),
        "coverage_parquet": str(coverage_path),
        "dictionary_json": str(dictionary_path),
        "manifest_json": str(manifest_path),
        "split_compatibility_json": str(split_compatibility_path),
        "qc_json": str(qc_path),
        "n_rows": int(len(labels_df)),
    }


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build PIT forward labels with costs, neutralization, and QC")
    parser.add_argument("--prices-adjusted-path", required=True)
    parser.add_argument("--universe-history-path", required=True)
    parser.add_argument("--trading-calendar-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--costs-path", default=None)
    parser.add_argument("--borrow-costs-path", default=None)
    parser.add_argument("--carry-costs-path", default=None)
    parser.add_argument("--exposures-path", default=None)
    parser.add_argument("--delisting-returns-path", default=None)
    parser.add_argument("--corporate-actions-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _cli_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    input_paths = InputPaths(
        adjusted_prices_pit_path=args.prices_adjusted_path,
        universe_history_path=args.universe_history_path,
        trading_calendar_path=args.trading_calendar_path,
        label_config_path=args.config_path,
        execution_costs_path=args.costs_path,
        borrow_costs_path=args.borrow_costs_path,
        carry_costs_path=args.carry_costs_path,
        neutralization_exposures_path=args.exposures_path,
        delisting_returns_path=args.delisting_returns_path,
        corporate_actions_path=args.corporate_actions_path,
        output_dir=args.output_dir,
    )
    result = build_labels(input_paths=input_paths, run_id=args.run_id)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
