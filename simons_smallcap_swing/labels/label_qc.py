from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
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

DATE_CANDIDATES = (
    "date",
    "asof_date",
    "trade_date",
    "decision_date",
    "session_date",
)
SYMBOL_CANDIDATES = (
    "symbol",
    "ticker",
    "asset",
    "instrument_id",
    "sid",
)
HORIZON_CANDIDATES = (
    "horizon",
    "horizon_days",
    "label_horizon_days",
    "forward_horizon_days",
)
TARGET_LONG_CANDIDATES = (
    "y",
    "target",
    "label_value",
    "value",
    "label",
)
VALID_FLAG_GENERIC_CANDIDATES = (
    "label_valid_flag",
    "is_label_valid",
    "valid_label",
    "valid_flag",
)
EXCLUSION_REASON_GENERIC_CANDIDATES = (
    "label_exclusion_reason",
    "exclusion_reason",
    "invalid_reason",
    "reason",
)
EVENT_START_GENERIC_CANDIDATES = (
    "event_start",
    "label_window_start",
    "window_start",
    "entry_date",
    "event_start_primary",
)
EVENT_END_GENERIC_CANDIDATES = (
    "event_end",
    "label_window_end",
    "window_end",
    "exit_date",
    "event_end_primary",
)
RUN_ID_CANDIDATES = (
    "run_id",
    "label_run_id",
)
N_VALID_CS_CANDIDATES = (
    "n_valid_cross_section",
    "n_valid_cs",
)
FEATURE_VALID_CANDIDATES = (
    "feature_valid_flag",
    "valid_feature_flag",
    "is_feature_valid",
)
FEATURE_TS_CANDIDATES = (
    "feature_timestamp",
    "asof_timestamp",
    "timestamp",
    "feature_ts",
    "known_timestamp",
)
UNIVERSE_FLAG_CANDIDATES = (
    "is_eligible",
    "eligible",
    "in_universe",
    "universe_member",
    "eligible_flag",
    "is_in_universe",
)
PRICE_FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "close": ("adj_close", "adjusted_close", "close_adj", "close", "px_close", "close_price"),
    "open": ("adj_open", "open_adj", "open", "px_open", "open_price"),
}

SEVERITY_FAIL = "FAIL"
SEVERITY_WARN = "WARN"
SEVERITY_INFO = "INFO"
GATE_PASS = "PASS"
GATE_WARN = "WARN"
GATE_FAIL = "FAIL"

CANONICAL_DEFAULTS = {
    "alignment_policy": "exact_on_eligible_rows",
    "duplicate_policy": "fail",
    "nan_policy": "fail_if_valid",
    "inf_policy": "fail_if_valid",
    "leakage_policy": "fail",
    "outlier_policy": "robust_plus_economic_sanity",
    "drift_policy": "cusum_on_coverage",
    "gate_aggregation_policy": "fail_dominates",
    "policy_version": "1.0.0",
}

TARGET_PRIORITY_PREFIXES = (
    "y_neut_",
    "y_fwd_ret_net_",
    "y_fwd_ret_gross_",
    "y_event_abn_ret_",
    "y_event_ret_net_",
    "y_event_ret_gross_",
    "y_cls_",
    "y_rank_",
    "y_",
)
TARGET_HORIZON_RE = re.compile(r"^(?P<family>.+?)_(?P<horizon>\d+)d$")


class LabelQCError(RuntimeError):
    """Base error for label QC."""


class ConfigError(LabelQCError):
    """Raised when configuration is invalid."""


class DataContractError(LabelQCError):
    """Raised when input data violate the expected contract."""


@dataclass(frozen=True)
class LabelStoreSpec:
    label_format: str
    date_col: str
    symbol_col: str
    horizon_col: str | None
    selected_target_cols: dict[int, str]
    selected_target_families: dict[int, str]
    valid_flag_cols: dict[int, str]
    reason_cols: dict[int, str]
    event_start_cols: dict[int, str | None]
    event_end_cols: dict[int, str | None]
    n_valid_cs_cols: dict[int, str | None]
    run_id_col: str | None = None
    ignored_target_cols: dict[int, list[str]] = field(default_factory=dict)
    feature_timestamp_col: str | None = None


@dataclass(frozen=True)
class FeatureIndexSpec:
    date_col: str
    symbol_col: str
    valid_flag_col: str | None = None
    timestamp_col: str | None = None


@dataclass(frozen=True)
class PriceSpec:
    date_col: str
    symbol_col: str
    close_col: str | None = None
    open_col: str | None = None


@dataclass(frozen=True)
class QCConfig:
    labels_store_path: str
    features_index_path: str
    run_id: str
    output_dir: str
    coverage_fail_by_horizon: dict[int, float] = field(default_factory=lambda: {5: 0.80, 10: 0.75, 20: 0.70})
    coverage_warn_by_horizon: dict[int, float] = field(default_factory=lambda: {5: 0.90, 10: 0.85, 20: 0.80})
    var_min_by_horizon: dict[int, float] = field(default_factory=lambda: {5: 1e-8, 10: 1e-8, 20: 1e-8})
    class_balance_warn_by_horizon: dict[int, float] = field(default_factory=lambda: {5: 0.10, 10: 0.10, 20: 0.10})
    class_balance_fail_by_horizon: dict[int, float] = field(default_factory=lambda: {5: 0.03, 10: 0.03, 20: 0.03})
    outlier_method: str = "mad"
    outlier_thresholds: dict[str, Any] = field(default_factory=lambda: {
        "mad_k": 10.0,
        "iqr_k": 6.0,
        "warn_rate": 0.01,
        "fail_rate": 0.05,
        "economic_abs_diff_fail": 0.50,
        "economic_abs_return_cap": 10.0,
    })
    leakage_policy: str = CANONICAL_DEFAULTS["leakage_policy"]
    alignment_policy: str = CANONICAL_DEFAULTS["alignment_policy"]
    drift_detection_policy: str = CANONICAL_DEFAULTS["drift_policy"]
    qc_window: int = 63
    gate_aggregation_policy: str = CANONICAL_DEFAULTS["gate_aggregation_policy"]
    duplicate_policy: str = CANONICAL_DEFAULTS["duplicate_policy"]
    nan_policy: str = CANONICAL_DEFAULTS["nan_policy"]
    inf_policy: str = CANONICAL_DEFAULTS["inf_policy"]
    outlier_policy: str = CANONICAL_DEFAULTS["outlier_policy"]
    policy_version: str = CANONICAL_DEFAULTS["policy_version"]
    exploratory_mode: bool = False
    decision_lag: int | None = None
    primary_target_preference: tuple[str, ...] = TARGET_PRIORITY_PREFIXES
    prices_pit_path: str | None = None
    universe_history_path: str | None = None
    trading_calendar_path: str | None = None
    costs_path: str | None = None
    split_metadata_path: str | None = None
    label_manifest_path: str | None = None
    persist_timeseries: bool = True
    fail_on_missing_event_window: bool = False
    coverage_reference: str = "features_index"
    min_cross_sectional_var_fraction: float = 0.70
    drift_k: float = 0.01
    drift_h: float = 0.25
    drift_fail_if_coverage_breach: bool = True
    non_canonical_flag: bool = False

    @staticmethod
    def from_mapping(mapping: Mapping[str, Any]) -> "QCConfig":
        raw = dict(mapping)
        labels_store_path = _required_str(raw.get("labels_store_path"), "labels_store_path")
        features_index_path = _required_str(raw.get("features_index_path"), "features_index_path")
        run_id = _required_str(raw.get("run_id"), "run_id")
        output_dir = str(raw.get("output_dir") or Path(labels_store_path).resolve().parent / f"label_qc_{run_id}")

        coverage_fail = _coerce_horizon_map(raw.get("coverage_fail_by_horizon"), {5: 0.80, 10: 0.75, 20: 0.70})
        coverage_warn = _coerce_horizon_map(raw.get("coverage_warn_by_horizon"), {5: 0.90, 10: 0.85, 20: 0.80})
        var_min = _coerce_horizon_map(raw.get("var_min_by_horizon"), {5: 1e-8, 10: 1e-8, 20: 1e-8})
        class_warn = _coerce_horizon_map(raw.get("class_balance_warn_by_horizon"), {5: 0.10, 10: 0.10, 20: 0.10})
        class_fail = _coerce_horizon_map(raw.get("class_balance_fail_by_horizon"), {5: 0.03, 10: 0.03, 20: 0.03})
        outlier_thresholds = dict(raw.get("outlier_thresholds", {}))
        merged_outlier_thresholds = {
            "mad_k": 10.0,
            "iqr_k": 6.0,
            "warn_rate": 0.01,
            "fail_rate": 0.05,
            "economic_abs_diff_fail": 0.50,
            "economic_abs_return_cap": 10.0,
            **outlier_thresholds,
        }
        preference_raw = raw.get("primary_target_preference", TARGET_PRIORITY_PREFIXES)
        if isinstance(preference_raw, Sequence) and not isinstance(preference_raw, (str, bytes)):
            preference = tuple(str(x) for x in preference_raw)
        else:
            raise ConfigError("primary_target_preference must be a sequence of strings")
        config = QCConfig(
            labels_store_path=labels_store_path,
            features_index_path=features_index_path,
            run_id=run_id,
            output_dir=output_dir,
            coverage_fail_by_horizon=coverage_fail,
            coverage_warn_by_horizon=coverage_warn,
            var_min_by_horizon=var_min,
            class_balance_warn_by_horizon=class_warn,
            class_balance_fail_by_horizon=class_fail,
            outlier_method=str(raw.get("outlier_method", "mad")).lower(),
            outlier_thresholds=merged_outlier_thresholds,
            leakage_policy=str(raw.get("leakage_policy", CANONICAL_DEFAULTS["leakage_policy"])),
            alignment_policy=str(raw.get("alignment_policy", CANONICAL_DEFAULTS["alignment_policy"])),
            drift_detection_policy=str(raw.get("drift_detection_policy", CANONICAL_DEFAULTS["drift_policy"])),
            qc_window=int(raw.get("qc_window", 63)),
            gate_aggregation_policy=str(raw.get("gate_aggregation_policy", CANONICAL_DEFAULTS["gate_aggregation_policy"])),
            duplicate_policy=str(raw.get("duplicate_policy", CANONICAL_DEFAULTS["duplicate_policy"])),
            nan_policy=str(raw.get("nan_policy", CANONICAL_DEFAULTS["nan_policy"])),
            inf_policy=str(raw.get("inf_policy", CANONICAL_DEFAULTS["inf_policy"])),
            outlier_policy=str(raw.get("outlier_policy", CANONICAL_DEFAULTS["outlier_policy"])),
            policy_version=str(raw.get("policy_version", CANONICAL_DEFAULTS["policy_version"])),
            exploratory_mode=bool(raw.get("exploratory_mode", False)),
            decision_lag=_optional_int(raw.get("decision_lag")),
            primary_target_preference=preference,
            prices_pit_path=_optional_str(raw.get("prices_pit_path")),
            universe_history_path=_optional_str(raw.get("universe_history_path")),
            trading_calendar_path=_optional_str(raw.get("trading_calendar_path")),
            costs_path=_optional_str(raw.get("costs_path")),
            split_metadata_path=_optional_str(raw.get("split_metadata_path")),
            label_manifest_path=_optional_str(raw.get("label_manifest_path")),
            persist_timeseries=bool(raw.get("persist_timeseries", True)),
            fail_on_missing_event_window=bool(raw.get("fail_on_missing_event_window", False)),
            coverage_reference=str(raw.get("coverage_reference", "features_index")),
            min_cross_sectional_var_fraction=float(raw.get("min_cross_sectional_var_fraction", 0.70)),
            drift_k=float(raw.get("drift_k", 0.01)),
            drift_h=float(raw.get("drift_h", 0.25)),
            drift_fail_if_coverage_breach=bool(raw.get("drift_fail_if_coverage_breach", True)),
            non_canonical_flag=bool(raw.get("non_canonical_flag", False)),
        )
        _validate_config(config)
        return _mark_non_canonical(config)

    def to_serializable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["primary_target_preference"] = list(self.primary_target_preference)
        return payload


@dataclass(frozen=True)
class QCArtifacts:
    normalized_labels: pd.DataFrame
    features_index: pd.DataFrame
    qc_by_horizon: pd.DataFrame
    qc_failures: pd.DataFrame
    qc_timeseries: pd.DataFrame
    qc_summary: dict[str, Any]
    manifest: dict[str, Any]
    output_dir: str
    output_paths: dict[str, str]


def _required_str(value: Any, name: str) -> str:
    text = _optional_str(value)
    if text is None:
        raise ConfigError(f"{name} is required")
    return text


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _coerce_horizon_map(raw: Any, default: Mapping[int, float]) -> dict[int, float]:
    if raw is None:
        return {int(k): float(v) for k, v in default.items()}
    if not isinstance(raw, Mapping):
        raise ConfigError("horizon threshold configuration must be a mapping")
    out: dict[int, float] = {}
    for key, value in raw.items():
        horizon = int(str(key).rstrip("dD"))
        out[horizon] = float(value)
    return out


def _validate_config(config: QCConfig) -> None:
    if config.outlier_method not in {"mad", "iqr"}:
        raise ConfigError("outlier_method must be 'mad' or 'iqr'")
    if config.qc_window <= 0:
        raise ConfigError("qc_window must be > 0")
    if config.min_cross_sectional_var_fraction < 0 or config.min_cross_sectional_var_fraction > 1:
        raise ConfigError("min_cross_sectional_var_fraction must be in [0,1]")
    for h, fail_thr in config.coverage_fail_by_horizon.items():
        warn_thr = config.coverage_warn_by_horizon.get(h, fail_thr)
        if fail_thr < 0 or warn_thr > 1 or fail_thr > warn_thr:
            raise ConfigError(f"coverage thresholds invalid for horizon {h}")
    for h, fail_thr in config.class_balance_fail_by_horizon.items():
        warn_thr = config.class_balance_warn_by_horizon.get(h, fail_thr)
        if fail_thr < 0 or warn_thr > 0.5 or fail_thr > warn_thr:
            raise ConfigError(f"class balance thresholds invalid for horizon {h}")


def _mark_non_canonical(config: QCConfig) -> QCConfig:
    non_canonical = config.non_canonical_flag
    canonical_fields = {
        "alignment_policy": CANONICAL_DEFAULTS["alignment_policy"],
        "duplicate_policy": CANONICAL_DEFAULTS["duplicate_policy"],
        "nan_policy": CANONICAL_DEFAULTS["nan_policy"],
        "inf_policy": CANONICAL_DEFAULTS["inf_policy"],
        "leakage_policy": CANONICAL_DEFAULTS["leakage_policy"],
        "outlier_policy": CANONICAL_DEFAULTS["outlier_policy"],
        "drift_detection_policy": CANONICAL_DEFAULTS["drift_policy"],
        "gate_aggregation_policy": CANONICAL_DEFAULTS["gate_aggregation_policy"],
    }
    for field_name, canonical_value in canonical_fields.items():
        if getattr(config, field_name) != canonical_value:
            non_canonical = True
            break
    if config.policy_version != CANONICAL_DEFAULTS["policy_version"]:
        non_canonical = True
    if non_canonical == config.non_canonical_flag:
        return config
    payload = config.to_serializable()
    payload["non_canonical_flag"] = non_canonical
    return QCConfig.from_mapping(payload)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        if yaml is None:  # pragma: no cover
            raise ConfigError("PyYAML is required to read YAML configs")
        payload = yaml.safe_load(text)
    elif suffix == ".json":
        payload = json.loads(text)
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            if yaml is None:  # pragma: no cover
                raise ConfigError("Config must be JSON, or install PyYAML for YAML support")
            payload = yaml.safe_load(text)
    if not isinstance(payload, Mapping):
        raise ConfigError("Configuration root must be a mapping")
    return dict(payload)


def _load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataContractError(f"input path does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover
            raise DataContractError(
                f"failed to read parquet at {path}; install pyarrow or fastparquet"
            ) from exc
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    raise DataContractError(f"unsupported input file format for {path}")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _write_table(df: pd.DataFrame, path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".parquet":
        try:
            df.to_parquet(target, index=False)
            return str(target)
        except Exception:  # pragma: no cover
            fallback = target.with_suffix(".csv")
            LOGGER.warning("Parquet engine unavailable; falling back to CSV at %s", fallback)
            df.to_csv(fallback, index=False)
            return str(fallback)
    if target.suffix.lower() == ".csv":
        df.to_csv(target, index=False)
        return str(target)
    raise ValueError(f"unsupported output format: {target}")


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _safe_hash_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    if candidate.is_file():
        st = candidate.stat()
        blob = f"{candidate.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
    parts = []
    for p in sorted(x for x in candidate.rglob("*") if x.is_file()):
        st = p.stat()
        parts.append(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _hash_payload(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(x) for x in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_datetime(series: pd.Series, col_name: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=False)
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    parsed = parsed.dt.normalize()
    if parsed.isna().any():
        raise DataContractError(f"column '{col_name}' contains unparseable datetimes")
    return parsed


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0.0
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "1": True,
            "yes": True,
            "y": True,
            "false": False,
            "0": False,
            "no": False,
            "n": False,
            "nan": False,
            "none": False,
            "": False,
        })
    )
    return mapped.fillna(False).astype(bool)


def _find_column(columns: Sequence[str], candidates: Sequence[str], *, required: bool = True) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    if required:
        raise DataContractError(f"could not resolve any column from candidates: {list(candidates)}")
    return None


def _maybe_find_pattern_col(columns: Sequence[str], exact: str, generic_candidates: Sequence[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    if exact.lower() in lowered:
        return lowered[exact.lower()]
    for candidate in generic_candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _normalize_reason(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(index=pd.RangeIndex(0), dtype="object")
    cleaned = series.astype("object")
    cleaned = cleaned.where(~cleaned.isna(), None)
    cleaned = cleaned.map(lambda x: None if x is None or str(x).strip() == "" or str(x).strip().lower() in {"nan", "none", "null"} else str(x).strip())
    return cleaned


def _family_priority_index(family: str, preference: Sequence[str]) -> tuple[int, int]:
    for idx, prefix in enumerate(preference):
        if family.startswith(prefix):
            return idx, len(family)
    return len(preference) + 1, len(family)


def _resolve_feature_index_spec(df: pd.DataFrame) -> FeatureIndexSpec:
    date_col = _find_column(df.columns, DATE_CANDIDATES, required=True)
    symbol_col = _find_column(df.columns, SYMBOL_CANDIDATES, required=True)
    valid_flag_col = _find_column(df.columns, FEATURE_VALID_CANDIDATES, required=False)
    timestamp_col = _find_column(df.columns, FEATURE_TS_CANDIDATES, required=False)
    return FeatureIndexSpec(
        date_col=date_col,
        symbol_col=symbol_col,
        valid_flag_col=valid_flag_col,
        timestamp_col=timestamp_col,
    )


def _normalize_features_index(df: pd.DataFrame, spec: FeatureIndexSpec) -> pd.DataFrame:
    working = df.copy()
    working[spec.date_col] = _to_datetime(working[spec.date_col], spec.date_col)
    working[spec.symbol_col] = working[spec.symbol_col].astype(str).str.strip()
    if spec.valid_flag_col is not None:
        valid_mask = _coerce_bool(working[spec.valid_flag_col])
        working = working.loc[valid_mask].copy()
    for col in UNIVERSE_FLAG_CANDIDATES:
        if col in working.columns:
            working = working.loc[_coerce_bool(working[col])].copy()
            break
    out = pd.DataFrame({
        "date": working[spec.date_col],
        "symbol": working[spec.symbol_col],
    })
    if spec.timestamp_col is not None:
        ts = pd.to_datetime(working[spec.timestamp_col], errors="coerce", utc=False)
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_localize(None)
        out["feature_timestamp"] = ts
    duplicated = out.duplicated(subset=["date", "symbol"], keep=False)
    if duplicated.any():
        dup_count = int(duplicated.sum())
        raise DataContractError(f"features_index contains {dup_count} duplicated (date, symbol) rows")
    out = out.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)
    return out


def _resolve_price_spec(df: pd.DataFrame) -> PriceSpec:
    date_col = _find_column(df.columns, DATE_CANDIDATES, required=True)
    symbol_col = _find_column(df.columns, SYMBOL_CANDIDATES, required=True)
    close_col = _find_column(df.columns, PRICE_FIELD_CANDIDATES["close"], required=False)
    open_col = _find_column(df.columns, PRICE_FIELD_CANDIDATES["open"], required=False)
    return PriceSpec(date_col=date_col, symbol_col=symbol_col, close_col=close_col, open_col=open_col)


def _normalize_prices(df: pd.DataFrame, spec: PriceSpec) -> pd.DataFrame:
    working = df.copy()
    working[spec.date_col] = _to_datetime(working[spec.date_col], spec.date_col)
    working[spec.symbol_col] = working[spec.symbol_col].astype(str).str.strip()
    cols = {
        "date": working[spec.date_col],
        "symbol": working[spec.symbol_col],
    }
    if spec.close_col is not None:
        cols["close"] = pd.to_numeric(working[spec.close_col], errors="coerce")
    if spec.open_col is not None:
        cols["open"] = pd.to_numeric(working[spec.open_col], errors="coerce")
    out = pd.DataFrame(cols).drop_duplicates(subset=["date", "symbol"], keep="last")
    return out


def _resolve_label_store_spec(df: pd.DataFrame, config: QCConfig) -> LabelStoreSpec:
    date_col = _find_column(df.columns, DATE_CANDIDATES, required=True)
    symbol_col = _find_column(df.columns, SYMBOL_CANDIDATES, required=True)
    run_id_col = _find_column(df.columns, RUN_ID_CANDIDATES, required=False)
    horizon_col = _find_column(df.columns, HORIZON_CANDIDATES, required=False)

    if horizon_col is not None:
        target_col = _resolve_long_target_col(df)
        valid_col = _find_column(df.columns, VALID_FLAG_GENERIC_CANDIDATES, required=False)
        reason_col = _find_column(df.columns, EXCLUSION_REASON_GENERIC_CANDIDATES, required=False)
        if valid_col is None:
            raise DataContractError("long label store must contain a label_valid_flag-equivalent column")
        if reason_col is None:
            raise DataContractError("long label store must contain a label_exclusion_reason-equivalent column")
        event_start_col = _find_column(df.columns, EVENT_START_GENERIC_CANDIDATES, required=False)
        event_end_col = _find_column(df.columns, EVENT_END_GENERIC_CANDIDATES, required=False)
        n_valid_col = _find_column(df.columns, N_VALID_CS_CANDIDATES, required=False)
        feature_ts_col = _find_column(df.columns, FEATURE_TS_CANDIDATES, required=False)
        unique_horizons = sorted({int(h) for h in pd.to_numeric(df[horizon_col], errors="coerce").dropna().astype(int).tolist()})
        if not unique_horizons:
            raise DataContractError("no valid horizons found in long label store")
        target_family = _infer_family_from_col(target_col)
        return LabelStoreSpec(
            label_format="long",
            date_col=date_col,
            symbol_col=symbol_col,
            horizon_col=horizon_col,
            selected_target_cols={h: target_col for h in unique_horizons},
            selected_target_families={h: target_family for h in unique_horizons},
            valid_flag_cols={h: valid_col for h in unique_horizons},
            reason_cols={h: reason_col for h in unique_horizons},
            event_start_cols={h: event_start_col for h in unique_horizons},
            event_end_cols={h: event_end_col for h in unique_horizons},
            n_valid_cs_cols={h: n_valid_col for h in unique_horizons},
            run_id_col=run_id_col,
            ignored_target_cols={},
            feature_timestamp_col=feature_ts_col,
        )

    target_map: dict[int, list[tuple[str, str]]] = {}
    for col in df.columns:
        match = TARGET_HORIZON_RE.match(str(col))
        if not match:
            continue
        family = match.group("family")
        horizon = int(match.group("horizon"))
        if not family.startswith("y_"):
            continue
        target_map.setdefault(horizon, []).append((col, family))
    if not target_map:
        raise DataContractError("could not resolve any horizon-specific target column in wide label store")

    selected_target_cols: dict[int, str] = {}
    selected_target_families: dict[int, str] = {}
    ignored_target_cols: dict[int, list[str]] = {}
    valid_cols: dict[int, str] = {}
    reason_cols: dict[int, str] = {}
    event_start_cols: dict[int, str | None] = {}
    event_end_cols: dict[int, str | None] = {}
    n_valid_cols: dict[int, str | None] = {}
    for horizon, candidates in sorted(target_map.items()):
        ordered = sorted(candidates, key=lambda x: _family_priority_index(x[1], config.primary_target_preference))
        selected_col, selected_family = ordered[0]
        ignored = [c for c, _ in ordered[1:]]
        valid_col = _maybe_find_pattern_col(df.columns, f"label_valid_flag_{horizon}d", VALID_FLAG_GENERIC_CANDIDATES)
        reason_col = _maybe_find_pattern_col(df.columns, f"label_exclusion_reason_{horizon}d", EXCLUSION_REASON_GENERIC_CANDIDATES)
        if valid_col is None:
            raise DataContractError(f"missing valid flag column for horizon {horizon}d")
        if reason_col is None:
            raise DataContractError(f"missing exclusion reason column for horizon {horizon}d")
        selected_target_cols[horizon] = selected_col
        selected_target_families[horizon] = selected_family
        ignored_target_cols[horizon] = ignored
        valid_cols[horizon] = valid_col
        reason_cols[horizon] = reason_col
        event_start_cols[horizon] = _maybe_find_pattern_col(df.columns, f"event_start_{horizon}d", EVENT_START_GENERIC_CANDIDATES)
        event_end_cols[horizon] = _maybe_find_pattern_col(df.columns, f"event_end_{horizon}d", EVENT_END_GENERIC_CANDIDATES)
        n_valid_cols[horizon] = _maybe_find_pattern_col(df.columns, f"n_valid_cross_section_{horizon}d", N_VALID_CS_CANDIDATES)
    feature_ts_col = _find_column(df.columns, FEATURE_TS_CANDIDATES, required=False)
    return LabelStoreSpec(
        label_format="wide",
        date_col=date_col,
        symbol_col=symbol_col,
        horizon_col=None,
        selected_target_cols=selected_target_cols,
        selected_target_families=selected_target_families,
        valid_flag_cols=valid_cols,
        reason_cols=reason_cols,
        event_start_cols=event_start_cols,
        event_end_cols=event_end_cols,
        n_valid_cs_cols=n_valid_cols,
        run_id_col=run_id_col,
        ignored_target_cols=ignored_target_cols,
        feature_timestamp_col=feature_ts_col,
    )


def _resolve_long_target_col(df: pd.DataFrame) -> str:
    for candidate in TARGET_LONG_CANDIDATES:
        if candidate in df.columns:
            return candidate
    candidates = [c for c in df.columns if str(c).startswith("y_")]
    if len(candidates) == 1:
        return candidates[0]
    numeric_candidates = [
        c
        for c in df.columns
        if c not in set(DATE_CANDIDATES) | set(SYMBOL_CANDIDATES) | set(HORIZON_CANDIDATES) | set(VALID_FLAG_GENERIC_CANDIDATES) | set(EXCLUSION_REASON_GENERIC_CANDIDATES)
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(numeric_candidates) == 1:
        return numeric_candidates[0]
    raise DataContractError("could not resolve target column in long label store")


def _infer_family_from_col(col: str) -> str:
    match = TARGET_HORIZON_RE.match(col)
    if match:
        return match.group("family")
    return col


def _normalize_labels_store(df: pd.DataFrame, spec: LabelStoreSpec, config: QCConfig) -> pd.DataFrame:
    working = df.copy()
    working[spec.date_col] = _to_datetime(working[spec.date_col], spec.date_col)
    working[spec.symbol_col] = working[spec.symbol_col].astype(str).str.strip()

    if spec.run_id_col is not None:
        working[spec.run_id_col] = working[spec.run_id_col].astype(str)

    pieces: list[pd.DataFrame] = []
    if spec.label_format == "long":
        horizon_col = spec.horizon_col
        assert horizon_col is not None
        piece = pd.DataFrame({
            "date": working[spec.date_col],
            "symbol": working[spec.symbol_col],
            "horizon": pd.to_numeric(working[horizon_col], errors="coerce").astype("Int64"),
            "target_family": working[spec.selected_target_cols[next(iter(spec.selected_target_cols))]].name,
            "y": pd.to_numeric(working[spec.selected_target_cols[next(iter(spec.selected_target_cols))]], errors="coerce"),
            "label_valid_flag": _coerce_bool(working[spec.valid_flag_cols[next(iter(spec.valid_flag_cols))]]),
            "label_exclusion_reason": _normalize_reason(working[spec.reason_cols[next(iter(spec.reason_cols))]]),
            "run_id": config.run_id,
        })
        if spec.run_id_col is not None:
            piece["run_id"] = working[spec.run_id_col].astype(str)
        if spec.event_start_cols[next(iter(spec.event_start_cols))] is not None:
            col = spec.event_start_cols[next(iter(spec.event_start_cols))]
            assert col is not None
            piece["event_start"] = pd.to_datetime(working[col], errors="coerce", utc=False)
            if getattr(piece["event_start"].dt, "tz", None) is not None:
                piece["event_start"] = piece["event_start"].dt.tz_localize(None)
            piece["event_start"] = piece["event_start"].dt.normalize()
        else:
            piece["event_start"] = pd.NaT
        if spec.event_end_cols[next(iter(spec.event_end_cols))] is not None:
            col = spec.event_end_cols[next(iter(spec.event_end_cols))]
            assert col is not None
            piece["event_end"] = pd.to_datetime(working[col], errors="coerce", utc=False)
            if getattr(piece["event_end"].dt, "tz", None) is not None:
                piece["event_end"] = piece["event_end"].dt.tz_localize(None)
            piece["event_end"] = piece["event_end"].dt.normalize()
        else:
            piece["event_end"] = pd.NaT
        if spec.n_valid_cs_cols[next(iter(spec.n_valid_cs_cols))] is not None:
            col = spec.n_valid_cs_cols[next(iter(spec.n_valid_cs_cols))]
            assert col is not None
            piece["n_valid_cross_section"] = pd.to_numeric(working[col], errors="coerce")
        else:
            piece["n_valid_cross_section"] = np.nan
        if spec.feature_timestamp_col is not None:
            ts = pd.to_datetime(working[spec.feature_timestamp_col], errors="coerce", utc=False)
            if getattr(ts.dt, "tz", None) is not None:
                ts = ts.dt.tz_localize(None)
            piece["feature_timestamp"] = ts
        else:
            piece["feature_timestamp"] = pd.NaT
        pieces.append(piece)
    else:
        for horizon in sorted(spec.selected_target_cols):
            target_col = spec.selected_target_cols[horizon]
            valid_col = spec.valid_flag_cols[horizon]
            reason_col = spec.reason_cols[horizon]
            piece = pd.DataFrame({
                "date": working[spec.date_col],
                "symbol": working[spec.symbol_col],
                "horizon": horizon,
                "target_family": spec.selected_target_families[horizon],
                "target_col": target_col,
                "y": pd.to_numeric(working[target_col], errors="coerce"),
                "label_valid_flag": _coerce_bool(working[valid_col]),
                "label_exclusion_reason": _normalize_reason(working[reason_col]),
                "run_id": config.run_id,
            })
            if spec.run_id_col is not None:
                piece["run_id"] = working[spec.run_id_col].astype(str)
            if spec.event_start_cols[horizon] is not None:
                col = spec.event_start_cols[horizon]
                assert col is not None
                ts = pd.to_datetime(working[col], errors="coerce", utc=False)
                if getattr(ts.dt, "tz", None) is not None:
                    ts = ts.dt.tz_localize(None)
                piece["event_start"] = ts.dt.normalize()
            else:
                piece["event_start"] = pd.NaT
            if spec.event_end_cols[horizon] is not None:
                col = spec.event_end_cols[horizon]
                assert col is not None
                ts = pd.to_datetime(working[col], errors="coerce", utc=False)
                if getattr(ts.dt, "tz", None) is not None:
                    ts = ts.dt.tz_localize(None)
                piece["event_end"] = ts.dt.normalize()
            else:
                piece["event_end"] = pd.NaT
            if spec.n_valid_cs_cols[horizon] is not None:
                col = spec.n_valid_cs_cols[horizon]
                assert col is not None
                piece["n_valid_cross_section"] = pd.to_numeric(working[col], errors="coerce")
            else:
                piece["n_valid_cross_section"] = np.nan
            if spec.feature_timestamp_col is not None:
                ts = pd.to_datetime(working[spec.feature_timestamp_col], errors="coerce", utc=False)
                if getattr(ts.dt, "tz", None) is not None:
                    ts = ts.dt.tz_localize(None)
                piece["feature_timestamp"] = ts
            else:
                piece["feature_timestamp"] = pd.NaT
            pieces.append(piece)
    out = pd.concat(pieces, axis=0, ignore_index=True)
    out["horizon"] = pd.to_numeric(out["horizon"], errors="coerce").astype("Int64")
    if out["horizon"].isna().any():
        raise DataContractError("normalized labels contain non-parsable horizons")
    out["run_id"] = out["run_id"].astype(str)
    duplicated = out.duplicated(subset=["date", "symbol", "horizon"], keep=False)
    if duplicated.any():
        dup_count = int(duplicated.sum())
        raise DataContractError(f"labels store contains {dup_count} duplicated (date, symbol, horizon) rows")
    out = out.sort_values(["date", "symbol", "horizon"], kind="stable").reset_index(drop=True)
    return out


def _threshold_for(horizon: int, mapping: Mapping[int, float]) -> float:
    if horizon in mapping:
        return float(mapping[horizon])
    if mapping:
        default_h = sorted(mapping)[0]
        return float(mapping[default_h])
    return float("nan")


def _issue(
    run_id: str,
    horizon: int | str,
    severity: str,
    check_name: str,
    object_scope: str,
    offending_count: int,
    details: Mapping[str, Any] | str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "horizon": None if horizon == "GLOBAL" else int(horizon),
        "severity": severity,
        "check_name": check_name,
        "object_scope": object_scope,
        "offending_count": int(offending_count),
        "details": json.dumps(_json_safe(details), ensure_ascii=False) if isinstance(details, Mapping) else str(details),
    }
    return payload


def _severity_from_metric(value: float, fail_thr: float, warn_thr: float, *, lower_is_worse: bool = True) -> str:
    if math.isnan(value):
        return SEVERITY_FAIL
    if lower_is_worse:
        if value < fail_thr:
            return SEVERITY_FAIL
        if value < warn_thr:
            return SEVERITY_WARN
    else:
        if value > fail_thr:
            return SEVERITY_FAIL
        if value > warn_thr:
            return SEVERITY_WARN
    return SEVERITY_INFO


def _cusum_flags(series: pd.Series, k: float, h: float) -> pd.DataFrame:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    values = numeric.to_numpy()
    if values.size == 0:
        return pd.DataFrame({"cusum_pos": [], "cusum_neg": [], "drift_breach": []})
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return pd.DataFrame({"cusum_pos": np.zeros_like(values, dtype=float), "cusum_neg": np.zeros_like(values, dtype=float), "drift_breach": np.zeros_like(values, dtype=bool)})
    mu = float(np.nanmean(finite))
    s_pos = np.zeros_like(values, dtype=float)
    s_neg = np.zeros_like(values, dtype=float)
    breach = np.zeros_like(values, dtype=bool)
    for i, x in enumerate(values):
        prev_pos = s_pos[i - 1] if i > 0 else 0.0
        prev_neg = s_neg[i - 1] if i > 0 else 0.0
        if not np.isfinite(x):
            s_pos[i] = prev_pos
            s_neg[i] = prev_neg
            breach[i] = False
            continue
        s_pos[i] = max(0.0, prev_pos + (x - mu - k))
        s_neg[i] = min(0.0, prev_neg + (x - mu + k))
        breach[i] = bool(s_pos[i] > h or abs(s_neg[i]) > h)
    return pd.DataFrame({"cusum_pos": s_pos, "cusum_neg": s_neg, "drift_breach": breach})


def _detect_discrete_family(values: pd.Series) -> str:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return "unknown"
    uniq = set(np.unique(finite.to_numpy()))
    if uniq.issubset({0.0, 1.0}):
        return "binary"
    if uniq.issubset({-1.0, 0.0, 1.0}):
        return "trinomial"
    return "continuous"


def _outlier_mask(values: pd.Series, method: str, thresholds: Mapping[str, Any]) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    finite = x[np.isfinite(x)]
    mask = pd.Series(False, index=x.index)
    if finite.empty:
        return mask
    med = float(np.median(finite))
    if method == "mad":
        mad = float(np.median(np.abs(finite - med)))
        if mad <= 0:
            return mask
        k = float(thresholds.get("mad_k", 10.0))
        score = np.abs(x - med) / mad
        mask = score > k
        return mask.fillna(False)
    iqr = float(np.percentile(finite, 75) - np.percentile(finite, 25))
    if iqr <= 0:
        return mask
    k = float(thresholds.get("iqr_k", 6.0))
    score = np.abs(x - med) / iqr
    mask = score > k
    return mask.fillna(False)


def _economic_sanity_outliers(valid_df: pd.DataFrame, prices_df: pd.DataFrame | None, thresholds: Mapping[str, Any]) -> pd.Series:
    if prices_df is None or valid_df.empty:
        return pd.Series(False, index=valid_df.index)
    if "event_start" not in valid_df.columns or "event_end" not in valid_df.columns:
        return pd.Series(False, index=valid_df.index)
    if not {"close"}.issubset(prices_df.columns):
        return pd.Series(False, index=valid_df.index)
    px = prices_df[["date", "symbol", "close"]].rename(columns={"close": "close_px"})
    start_px = px.rename(columns={"date": "event_start", "close_px": "entry_px"})
    end_px = px.rename(columns={"date": "event_end", "close_px": "exit_px"})
    merged = valid_df[["symbol", "event_start", "event_end", "y"]].merge(start_px, on=["symbol", "event_start"], how="left")
    merged = merged.merge(end_px, on=["symbol", "event_end"], how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_ret = (merged["exit_px"] - merged["entry_px"]) / merged["entry_px"]
    diff = (pd.to_numeric(merged["y"], errors="coerce") - raw_ret).abs()
    incoherent = (
        merged["entry_px"].le(0)
        | merged["exit_px"].le(0)
        | raw_ret.abs().gt(float(thresholds.get("economic_abs_return_cap", 10.0)))
        | diff.gt(float(thresholds.get("economic_abs_diff_fail", 0.50)))
    )
    incoherent = incoherent.fillna(False)
    incoherent.index = valid_df.index
    return incoherent


def _parse_label_manifest(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    candidate = Path(path)
    if not candidate.exists():
        return {}
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _build_eligible_index(features: pd.DataFrame, universe_history_df: pd.DataFrame | None) -> pd.DataFrame:
    eligible = features[["date", "symbol"]].drop_duplicates().copy()
    if universe_history_df is None or universe_history_df.empty:
        return eligible
    try:
        date_col = _find_column(universe_history_df.columns, DATE_CANDIDATES, required=True)
        symbol_col = _find_column(universe_history_df.columns, SYMBOL_CANDIDATES, required=True)
    except DataContractError:
        return eligible
    uni = universe_history_df.copy()
    uni[date_col] = _to_datetime(uni[date_col], date_col)
    uni[symbol_col] = uni[symbol_col].astype(str).str.strip()
    flag_col = _find_column(uni.columns, UNIVERSE_FLAG_CANDIDATES, required=False)
    if flag_col is not None:
        uni = uni.loc[_coerce_bool(uni[flag_col])].copy()
    universe_idx = pd.DataFrame({"date": uni[date_col], "symbol": uni[symbol_col]}).drop_duplicates()
    eligible = eligible.merge(universe_idx, on=["date", "symbol"], how="inner")
    return eligible


def _check_schema_and_integrity(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    config: QCConfig,
    issues: list[dict[str, Any]],
) -> None:
    required_label_cols = ["date", "symbol", "horizon", "y", "label_valid_flag", "label_exclusion_reason", "run_id"]
    missing = [c for c in required_label_cols if c not in labels.columns]
    if missing:
        issues.append(_issue(config.run_id, "GLOBAL", SEVERITY_FAIL, "schema_required_columns", "labels_store", len(missing), {"missing": missing}))
    if not pd.api.types.is_datetime64_any_dtype(labels["date"]):
        issues.append(_issue(config.run_id, "GLOBAL", SEVERITY_FAIL, "schema_date_dtype", "labels_store", len(labels), "date column is not datetime64"))
    if labels["symbol"].dtype.kind not in {"O", "U", "S"}:
        issues.append(_issue(config.run_id, "GLOBAL", SEVERITY_FAIL, "schema_symbol_dtype", "labels_store", len(labels), "symbol column must be string-like"))
    duplicated = labels.duplicated(subset=["date", "symbol", "horizon"], keep=False)
    if duplicated.any():
        issues.append(_issue(config.run_id, "GLOBAL", SEVERITY_FAIL, "key_uniqueness", "labels_store", int(duplicated.sum()), "duplicated (date, symbol, horizon) in normalized store"))
    dup_feat = features.duplicated(subset=["date", "symbol"], keep=False)
    if dup_feat.any():
        issues.append(_issue(config.run_id, "GLOBAL", SEVERITY_FAIL, "feature_index_uniqueness", "features_index", int(dup_feat.sum()), "duplicated (date, symbol) in features index"))

    consistency_bad = (
        (labels["label_valid_flag"] & labels["label_exclusion_reason"].notna())
        | (~labels["label_valid_flag"] & labels["label_exclusion_reason"].isna())
    )
    if consistency_bad.any():
        severity = SEVERITY_WARN if config.exploratory_mode and int(consistency_bad.sum()) == 1 else SEVERITY_FAIL
        issues.append(
            _issue(
                config.run_id,
                "GLOBAL",
                severity,
                "valid_flag_reason_consistency",
                "labels_store",
                int(consistency_bad.sum()),
                "valid rows must have null reason; invalid rows must have non-null reason",
            )
        )

    valid_rows = labels.loc[labels["label_valid_flag"]].copy()
    invalid_nan = valid_rows["y"].isna()
    invalid_inf = ~np.isfinite(pd.to_numeric(valid_rows["y"], errors="coerce").fillna(np.nan)) & valid_rows["y"].notna()
    if invalid_nan.any() and config.nan_policy == "fail_if_valid":
        issues.append(_issue(config.run_id, "GLOBAL", SEVERITY_FAIL, "nan_in_valid_rows", "labels_store", int(invalid_nan.sum()), "NaN found in valid label rows"))
    if invalid_inf.any() and config.inf_policy == "fail_if_valid":
        issues.append(_issue(config.run_id, "GLOBAL", SEVERITY_FAIL, "inf_in_valid_rows", "labels_store", int(invalid_inf.sum()), "Inf found in valid label rows"))


def _compute_horizon_metrics(
    labels: pd.DataFrame,
    eligible_index: pd.DataFrame,
    features: pd.DataFrame,
    prices: pd.DataFrame | None,
    config: QCConfig,
    label_spec: LabelStoreSpec,
    issues: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qc_rows: list[dict[str, Any]] = []
    ts_rows: list[pd.DataFrame] = []

    feature_ts_lookup = None
    if "feature_timestamp" in features.columns:
        feature_ts_lookup = features[["date", "symbol", "feature_timestamp"]].copy()

    for horizon in sorted({int(h) for h in labels["horizon"].dropna().astype(int).tolist()}):
        lbl = labels.loc[labels["horizon"] == horizon].copy()
        family = str(lbl["target_family"].dropna().iloc[0]) if not lbl.empty else "unknown"
        aligned = eligible_index.merge(lbl, on=["date", "symbol"], how="left", indicator=True)
        missing_label_rows = aligned["_merge"].eq("left_only")
        structural_ineligible = aligned["label_exclusion_reason"].isin(["INCOMPLETE_FORWARD_WINDOW", "OUT_OF_SAMPLE_DATE"]).fillna(False)
        eligible_mask = ~structural_ineligible
        valid_mask = aligned["label_valid_flag"].fillna(False) & eligible_mask
        eligible_count = int(eligible_mask.sum())
        valid_count = int(valid_mask.sum())
        coverage = float(valid_count / eligible_count) if eligible_count > 0 else float("nan")
        coverage_fail = _threshold_for(horizon, config.coverage_fail_by_horizon)
        coverage_warn = _threshold_for(horizon, config.coverage_warn_by_horizon)
        coverage_severity = _severity_from_metric(coverage, coverage_fail, coverage_warn, lower_is_worse=True)
        if coverage_severity != SEVERITY_INFO:
            issues.append(_issue(config.run_id, horizon, coverage_severity, "coverage_global", "horizon", max(0, eligible_count - valid_count), {"coverage": coverage, "coverage_fail": coverage_fail, "coverage_warn": coverage_warn}))

        orphan_labels = lbl.merge(eligible_index, on=["date", "symbol"], how="left", indicator=True)
        orphan_count = int(orphan_labels["_merge"].eq("left_only").sum())
        alignment_mismatch = int(missing_label_rows.sum()) + orphan_count
        if alignment_mismatch > 0 and config.alignment_policy == "exact_on_eligible_rows":
            issues.append(
                _issue(
                    config.run_id,
                    horizon,
                    SEVERITY_FAIL,
                    "feature_label_alignment",
                    "eligible_index",
                    alignment_mismatch,
                    {
                        "missing_label_rows": int(missing_label_rows.sum()),
                        "orphan_label_rows": orphan_count,
                    },
                )
            )

        finite_valid = aligned.loc[valid_mask, "y"]
        finite_valid = pd.to_numeric(finite_valid, errors="coerce")
        finite_valid = finite_valid[np.isfinite(finite_valid)]
        variance = float(finite_valid.var(ddof=0)) if not finite_valid.empty else float("nan")
        var_min = _threshold_for(horizon, config.var_min_by_horizon)
        frac_pos_var = float("nan")
        cs_var = aligned.loc[valid_mask, ["date", "y"]].copy()
        cs_var["y"] = pd.to_numeric(cs_var["y"], errors="coerce")
        if not cs_var.empty:
            per_date_var = cs_var.groupby("date", observed=True)["y"].var(ddof=0)
            frac_pos_var = float((per_date_var > var_min).mean()) if len(per_date_var) else float("nan")
        var_severity = SEVERITY_INFO
        if math.isnan(variance) or variance < var_min or (not math.isnan(frac_pos_var) and frac_pos_var < config.min_cross_sectional_var_fraction):
            var_severity = SEVERITY_FAIL if (math.isnan(variance) or variance < var_min) else SEVERITY_WARN
            issues.append(
                _issue(
                    config.run_id,
                    horizon,
                    var_severity,
                    "variance_degeneracy",
                    "horizon",
                    int(valid_count),
                    {
                        "variance": variance,
                        "var_min": var_min,
                        "frac_dates_positive_var": frac_pos_var,
                        "frac_threshold": config.min_cross_sectional_var_fraction,
                    },
                )
            )

        discrete_kind = _detect_discrete_family(finite_valid)
        class_balance = float("nan")
        min_class_share = float("nan")
        class_counts: dict[str, int] = {}
        if discrete_kind in {"binary", "trinomial"} and not finite_valid.empty:
            counts = pd.Series(finite_valid).value_counts(dropna=False)
            total = int(counts.sum())
            if discrete_kind == "binary":
                pos_share = float(counts.get(1.0, 0) / total) if total > 0 else float("nan")
                class_balance = pos_share
                min_class_share = float(min(pos_share, 1.0 - pos_share)) if not math.isnan(pos_share) else float("nan")
                class_counts = {"0": int(counts.get(0.0, 0)), "1": int(counts.get(1.0, 0))}
            else:
                shares = [float(counts.get(v, 0) / total) for v in (-1.0, 0.0, 1.0)]
                class_balance = float(counts.get(1.0, 0) / total) if total > 0 else float("nan")
                min_class_share = float(min(shares)) if shares else float("nan")
                class_counts = {"-1": int(counts.get(-1.0, 0)), "0": int(counts.get(0.0, 0)), "1": int(counts.get(1.0, 0))}
            fail_thr = _threshold_for(horizon, config.class_balance_fail_by_horizon)
            warn_thr = _threshold_for(horizon, config.class_balance_warn_by_horizon)
            class_severity = _severity_from_metric(min_class_share, fail_thr, warn_thr, lower_is_worse=True)
            if class_severity != SEVERITY_INFO:
                issues.append(
                    _issue(
                        config.run_id,
                        horizon,
                        class_severity,
                        "class_balance",
                        "horizon",
                        int(valid_count),
                        {
                            "class_kind": discrete_kind,
                            "class_counts": class_counts,
                            "min_class_share": min_class_share,
                            "warn_threshold": warn_thr,
                            "fail_threshold": fail_thr,
                        },
                    )
                )

        valid_numeric = aligned.loc[valid_mask].copy()
        valid_numeric["y"] = pd.to_numeric(valid_numeric["y"], errors="coerce")
        valid_numeric = valid_numeric[np.isfinite(valid_numeric["y"])].copy()
        stat_outlier_mask = _outlier_mask(valid_numeric["y"], config.outlier_method, config.outlier_thresholds)
        outlier_rate = float(stat_outlier_mask.mean()) if len(stat_outlier_mask) else 0.0
        warn_rate = float(config.outlier_thresholds.get("warn_rate", 0.01))
        fail_rate = float(config.outlier_thresholds.get("fail_rate", 0.05))
        if outlier_rate > 0:
            severity = _severity_from_metric(outlier_rate, fail_rate, warn_rate, lower_is_worse=False)
            if severity != SEVERITY_INFO:
                issues.append(
                    _issue(
                        config.run_id,
                        horizon,
                        severity,
                        "outlier_rate",
                        "horizon",
                        int(stat_outlier_mask.sum()),
                        {"outlier_rate": outlier_rate, "warn_rate": warn_rate, "fail_rate": fail_rate, "method": config.outlier_method},
                    )
                )
        econ_outlier_mask = _economic_sanity_outliers(valid_numeric.loc[stat_outlier_mask], prices, config.outlier_thresholds)
        if len(econ_outlier_mask) and econ_outlier_mask.any() and config.outlier_policy == "robust_plus_economic_sanity":
            issues.append(
                _issue(
                    config.run_id,
                    horizon,
                    SEVERITY_FAIL,
                    "economic_sanity_outliers",
                    "rows",
                    int(econ_outlier_mask.sum()),
                    "extreme statistical outliers also fail simple economic sanity checks",
                )
            )

        merged_ts = aligned[["date", "symbol", "label_valid_flag", "label_exclusion_reason", "y", "event_start", "event_end"]].copy()
        merged_ts["y"] = pd.to_numeric(merged_ts["y"], errors="coerce")
        merged_ts["eligible_row"] = ~merged_ts["label_exclusion_reason"].isin(["INCOMPLETE_FORWARD_WINDOW", "OUT_OF_SAMPLE_DATE"]).fillna(False)
        coverage_ts = merged_ts.groupby("date", observed=True).agg(
            eligible_count=("eligible_row", lambda s: int(pd.Series(s).fillna(False).sum())),
            valid_count=("label_valid_flag", lambda s: int((pd.Series(s).fillna(False) & merged_ts.loc[s.index, "eligible_row"]).sum())),
        )
        coverage_ts["coverage_t"] = np.where(coverage_ts["eligible_count"] > 0, coverage_ts["valid_count"] / coverage_ts["eligible_count"], np.nan)
        valid_by_date = merged_ts.loc[merged_ts["label_valid_flag"].fillna(False)].groupby("date", observed=True)["y"].agg(["var"])
        coverage_ts = coverage_ts.join(valid_by_date.rename(columns={"var": "variance_t"}), how="left")
        if discrete_kind == "binary":
            pos_share_t = merged_ts.loc[merged_ts["label_valid_flag"].fillna(False)].groupby("date", observed=True)["y"].apply(lambda s: float((s == 1).mean()) if len(s) else np.nan)
            coverage_ts["class_positive_share_t"] = pos_share_t
        elif discrete_kind == "trinomial":
            tmp = merged_ts.loc[merged_ts["label_valid_flag"].fillna(False)].groupby("date", observed=True)["y"]
            coverage_ts["class_positive_share_t"] = tmp.apply(lambda s: float((s == 1).mean()) if len(s) else np.nan)
            coverage_ts["class_neutral_share_t"] = tmp.apply(lambda s: float((s == 0).mean()) if len(s) else np.nan)
            coverage_ts["class_negative_share_t"] = tmp.apply(lambda s: float((s == -1).mean()) if len(s) else np.nan)
        drift = _cusum_flags(coverage_ts["coverage_t"], config.drift_k, config.drift_h)
        coverage_ts = pd.concat([coverage_ts.reset_index(), drift], axis=1)
        coverage_ts["horizon"] = horizon
        coverage_ts["target_family"] = family
        drift_flag = bool(coverage_ts["drift_breach"].any()) if not coverage_ts.empty else False
        min_coverage_t = float(coverage_ts["coverage_t"].min()) if not coverage_ts.empty else float("nan")
        if drift_flag:
            drift_severity = SEVERITY_WARN
            if config.drift_fail_if_coverage_breach and not math.isnan(min_coverage_t) and min_coverage_t < coverage_fail:
                drift_severity = SEVERITY_FAIL
            issues.append(
                _issue(
                    config.run_id,
                    horizon,
                    drift_severity,
                    "coverage_drift_cusum",
                    "timeseries",
                    int(coverage_ts["drift_breach"].sum()),
                    {"min_coverage_t": min_coverage_t, "coverage_fail": coverage_fail},
                )
            )

        leak_fail_count = 0
        valid_rows = aligned.loc[valid_mask].copy()
        if valid_rows.empty:
            if config.leakage_policy == "fail":
                issues.append(_issue(config.run_id, horizon, SEVERITY_FAIL, "no_valid_rows", "horizon", 0, "no valid rows remain after normalization"))
        else:
            if feature_ts_lookup is not None and "feature_timestamp" not in valid_rows.columns:
                valid_rows = valid_rows.merge(feature_ts_lookup, on=["date", "symbol"], how="left")
            declared_lag = config.decision_lag
            manifest_lag = None
            if config.label_manifest_path:
                manifest_payload = _parse_label_manifest(config.label_manifest_path)
                manifest_lag = _optional_int(manifest_payload.get("decision_lag"))
            if declared_lag is None:
                declared_lag = manifest_lag
            if valid_rows["event_start"].notna().any():
                if declared_lag is None or declared_lag == 0:
                    bad_start = valid_rows["event_start"] < valid_rows["date"]
                else:
                    bad_start = valid_rows["event_start"] <= valid_rows["date"]
                leak_fail_count += int(bad_start.sum())
                if bad_start.any():
                    issues.append(_issue(config.run_id, horizon, SEVERITY_FAIL, "temporal_window_start", "valid_rows", int(bad_start.sum()), {"decision_lag": declared_lag}))
            elif config.fail_on_missing_event_window:
                issues.append(_issue(config.run_id, horizon, SEVERITY_FAIL, "missing_event_start", "horizon", int(valid_count), "event_start not available for leakage checks"))
            else:
                issues.append(_issue(config.run_id, horizon, SEVERITY_WARN, "missing_event_start", "horizon", int(valid_count), "event_start not available for leakage checks"))
            if valid_rows["event_end"].notna().any():
                bad_end = valid_rows["event_end"].isna() | ((valid_rows["event_start"].notna()) & (valid_rows["event_end"] < valid_rows["event_start"]))
                leak_fail_count += int(bad_end.sum())
                if bad_end.any():
                    issues.append(_issue(config.run_id, horizon, SEVERITY_FAIL, "temporal_window_end", "valid_rows", int(bad_end.sum()), "event_end missing or precedes event_start on valid rows"))
            elif config.fail_on_missing_event_window:
                issues.append(_issue(config.run_id, horizon, SEVERITY_FAIL, "missing_event_end", "horizon", int(valid_count), "event_end not available for leakage checks"))
            else:
                issues.append(_issue(config.run_id, horizon, SEVERITY_WARN, "missing_event_end", "horizon", int(valid_count), "event_end not available for leakage checks"))
            if "feature_timestamp" in valid_rows.columns and valid_rows["feature_timestamp"].notna().any() and valid_rows["event_start"].notna().any():
                feature_date = pd.to_datetime(valid_rows["feature_timestamp"], errors="coerce", utc=False)
                if getattr(feature_date.dt, "tz", None) is not None:
                    feature_date = feature_date.dt.tz_localize(None)
                feature_date = feature_date.dt.normalize()
                bad_ts = valid_rows["event_start"] < feature_date
                leak_fail_count += int(bad_ts.sum())
                if bad_ts.any():
                    issues.append(_issue(config.run_id, horizon, SEVERITY_FAIL, "feature_timestamp_alignment", "valid_rows", int(bad_ts.sum()), "event_start predates feature timestamp"))

        qc_rows.append({
            "horizon": horizon,
            "target_family": family,
            "coverage": coverage,
            "variance": variance,
            "class_balance": class_balance,
            "min_class_share": min_class_share,
            "outlier_rate": outlier_rate,
            "alignment_mismatch": alignment_mismatch,
            "drift_flag": drift_flag,
            "valid_count": valid_count,
            "eligible_count": eligible_count,
            "frac_dates_positive_variance": frac_pos_var,
            "discrete_kind": discrete_kind,
            "ignored_target_cols": json.dumps(label_spec.ignored_target_cols.get(horizon, []), ensure_ascii=False),
        })
        ts_rows.append(coverage_ts)

    qc_by_horizon = pd.DataFrame(qc_rows)
    qc_timeseries = pd.concat(ts_rows, axis=0, ignore_index=True) if ts_rows else pd.DataFrame(columns=["date", "horizon"])
    return qc_by_horizon, qc_timeseries


def _attach_gates(qc_by_horizon: pd.DataFrame, issues: list[dict[str, Any]]) -> pd.DataFrame:
    if qc_by_horizon.empty:
        return qc_by_horizon.assign(gate=GATE_FAIL, n_failures=0, n_warnings=0)
    issues_df = pd.DataFrame(issues)
    if issues_df.empty:
        qc_by_horizon = qc_by_horizon.copy()
        qc_by_horizon["gate"] = GATE_PASS
        qc_by_horizon["n_failures"] = 0
        qc_by_horizon["n_warnings"] = 0
        return qc_by_horizon
    horizon_stats = issues_df.dropna(subset=["horizon"]).groupby("horizon", observed=True)["severity"].agg(
        n_failures=lambda s: int((s == SEVERITY_FAIL).sum()),
        n_warnings=lambda s: int((s == SEVERITY_WARN).sum()),
    )
    out = qc_by_horizon.merge(horizon_stats, left_on="horizon", right_index=True, how="left")
    out["n_failures"] = out["n_failures"].fillna(0).astype(int)
    out["n_warnings"] = out["n_warnings"].fillna(0).astype(int)
    out["gate"] = np.select(
        [out["n_failures"] > 0, out["n_warnings"] > 0],
        [GATE_FAIL, GATE_WARN],
        default=GATE_PASS,
    )
    return out


def _top_failure_reasons(issues_df: pd.DataFrame) -> list[dict[str, Any]]:
    if issues_df.empty:
        return []
    counts = issues_df.loc[issues_df["severity"] == SEVERITY_FAIL, "check_name"].value_counts()
    return [{"check_name": str(k), "count": int(v)} for k, v in counts.head(10).items()]


def _build_summary(config: QCConfig, qc_by_horizon: pd.DataFrame, issues_df: pd.DataFrame, input_shapes: Mapping[str, Any]) -> dict[str, Any]:
    if qc_by_horizon.empty:
        global_gate = GATE_FAIL
        gates_by_horizon: dict[str, str] = {}
    else:
        if (qc_by_horizon["gate"] == GATE_FAIL).any():
            global_gate = GATE_FAIL
        elif (qc_by_horizon["gate"] == GATE_WARN).any():
            global_gate = GATE_WARN
        else:
            global_gate = GATE_PASS
        gates_by_horizon = {f"{int(h)}d": gate for h, gate in zip(qc_by_horizon["horizon"], qc_by_horizon["gate"])}
    return {
        "run_id": config.run_id,
        "global_gate": global_gate,
        "gates_by_horizon": gates_by_horizon,
        "n_failures": int((issues_df["severity"] == SEVERITY_FAIL).sum()) if not issues_df.empty else 0,
        "n_warnings": int((issues_df["severity"] == SEVERITY_WARN).sum()) if not issues_df.empty else 0,
        "top_failure_reasons": _top_failure_reasons(issues_df),
        "config_hash": _hash_payload(config.to_serializable()),
        "policy_version": config.policy_version,
        "non_canonical_flag": config.non_canonical_flag,
        "input_shapes": dict(input_shapes),
        "created_at_utc": _utc_now_iso(),
    }


def _build_manifest(
    config: QCConfig,
    label_spec: LabelStoreSpec,
    feature_spec: FeatureIndexSpec,
    summary: Mapping[str, Any],
    outputs: Mapping[str, Any],
    input_shapes: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "execution_timestamp": _utc_now_iso(),
        "labels_version": _safe_hash_file(config.labels_store_path),
        "features_index_version": _safe_hash_file(config.features_index_path),
        "config_hash": _hash_payload(config.to_serializable()),
        "policy_version": config.policy_version,
        "non_canonical_flag": config.non_canonical_flag,
        "config": config.to_serializable(),
        "label_store_spec": asdict(label_spec),
        "feature_index_spec": asdict(feature_spec),
        "input_files": {
            "labels_store_path": config.labels_store_path,
            "features_index_path": config.features_index_path,
            "prices_pit_path": config.prices_pit_path,
            "universe_history_path": config.universe_history_path,
            "trading_calendar_path": config.trading_calendar_path,
            "costs_path": config.costs_path,
            "split_metadata_path": config.split_metadata_path,
            "label_manifest_path": config.label_manifest_path,
            "labels_store_sha256": _safe_hash_file(config.labels_store_path),
            "features_index_sha256": _safe_hash_file(config.features_index_path),
            "prices_pit_sha256": _safe_hash_file(config.prices_pit_path),
            "universe_history_sha256": _safe_hash_file(config.universe_history_path),
            "label_manifest_sha256": _safe_hash_file(config.label_manifest_path),
        },
        "input_shapes": dict(input_shapes),
        "summary": dict(summary),
        "outputs": dict(outputs),
    }


def run_label_qc(config: QCConfig) -> QCArtifacts:
    labels_raw = _load_table(config.labels_store_path)
    features_raw = _load_table(config.features_index_path)
    universe_raw = _load_table(config.universe_history_path) if config.universe_history_path else None
    prices_raw = _load_table(config.prices_pit_path) if config.prices_pit_path else None

    label_spec = _resolve_label_store_spec(labels_raw, config)
    feature_spec = _resolve_feature_index_spec(features_raw)
    labels = _normalize_labels_store(labels_raw, label_spec, config)
    features = _normalize_features_index(features_raw, feature_spec)
    prices = None
    if prices_raw is not None:
        prices = _normalize_prices(prices_raw, _resolve_price_spec(prices_raw))
    eligible_index = _build_eligible_index(features, universe_raw)

    issues: list[dict[str, Any]] = []
    _check_schema_and_integrity(labels, features, config, issues)
    qc_by_horizon, qc_timeseries = _compute_horizon_metrics(labels, eligible_index, features, prices, config, label_spec, issues)
    qc_by_horizon = _attach_gates(qc_by_horizon, issues)
    issues_df = pd.DataFrame(issues, columns=["run_id", "horizon", "severity", "check_name", "object_scope", "offending_count", "details"])
    if issues_df.empty:
        issues_df = pd.DataFrame(columns=["run_id", "horizon", "severity", "check_name", "object_scope", "offending_count", "details"])
    summary = _build_summary(
        config,
        qc_by_horizon,
        issues_df,
        input_shapes={
            "labels_rows": int(len(labels_raw)),
            "labels_cols": int(labels_raw.shape[1]),
            "normalized_labels_rows": int(len(labels)),
            "features_rows": int(len(features_raw)),
            "features_cols": int(features_raw.shape[1]),
            "eligible_index_rows": int(len(eligible_index)),
        },
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_summary_path = output_dir / f"qc_summary_{config.run_id}.json"
    qc_by_horizon_path = output_dir / f"qc_by_horizon_{config.run_id}.parquet"
    qc_failures_path = output_dir / f"qc_failures_{config.run_id}.parquet"
    qc_timeseries_path = output_dir / f"qc_timeseries_{config.run_id}.parquet"
    normalized_labels_path = output_dir / f"normalized_labels_{config.run_id}.parquet"

    _write_json(qc_summary_path, summary)
    output_paths: dict[str, str] = {
        "qc_summary": str(qc_summary_path),
        "qc_by_horizon": _write_table(qc_by_horizon, qc_by_horizon_path),
        "qc_failures": _write_table(issues_df, qc_failures_path),
        "normalized_labels": _write_table(labels, normalized_labels_path),
    }
    if config.persist_timeseries:
        output_paths["qc_timeseries"] = _write_table(qc_timeseries, qc_timeseries_path)

    manifest = _build_manifest(
        config=config,
        label_spec=label_spec,
        feature_spec=feature_spec,
        summary=summary,
        outputs=output_paths,
        input_shapes=summary["input_shapes"],
    )
    manifest_path = output_dir / f"manifest_{config.run_id}.json"
    _write_json(manifest_path, manifest)
    output_paths["manifest"] = str(manifest_path)

    return QCArtifacts(
        normalized_labels=labels,
        features_index=features,
        qc_by_horizon=qc_by_horizon,
        qc_failures=issues_df,
        qc_timeseries=qc_timeseries,
        qc_summary=summary,
        manifest=manifest,
        output_dir=str(output_dir),
        output_paths=output_paths,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quality-control and gate PIT labels before model usage")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON QC config")
    parser.add_argument("--labels-store-path", help="Override labels_store_path")
    parser.add_argument("--features-index-path", help="Override features_index_path")
    parser.add_argument("--run-id", help="Override run_id")
    parser.add_argument("--output-dir", help="Override output_dir")
    parser.add_argument("--prices-pit-path", help="Optional prices PIT path")
    parser.add_argument("--label-manifest-path", help="Optional label manifest path")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    payload = _load_mapping(args.config)
    if args.labels_store_path:
        payload["labels_store_path"] = args.labels_store_path
    if args.features_index_path:
        payload["features_index_path"] = args.features_index_path
    if args.run_id:
        payload["run_id"] = args.run_id
    if args.output_dir:
        payload["output_dir"] = args.output_dir
    if args.prices_pit_path:
        payload["prices_pit_path"] = args.prices_pit_path
    if args.label_manifest_path:
        payload["label_manifest_path"] = args.label_manifest_path
    config = QCConfig.from_mapping(payload)
    outputs = run_label_qc(config)
    LOGGER.info("label QC outputs written to %s", outputs.output_dir)
    LOGGER.info("summary: %s", outputs.output_paths.get("qc_summary"))
    LOGGER.info("global gate: %s", outputs.qc_summary["global_gate"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
