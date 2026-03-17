from __future__ import annotations

import argparse
import hashlib
import itertools
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

DATE_CANDIDATES = ("date", "decision_date", "asof_date", "trade_date")
SYMBOL_CANDIDATES = ("symbol", "ticker", "asset", "instrument_id")
HORIZON_CANDIDATES = (
    "horizon_days",
    "label_horizon_days",
    "forward_horizon_days",
    "horizon",
    "target_horizon_days",
)
SAMPLE_INDEX_CANDIDATES = ("sample_index", "row_id", "panel_index", "obs_id")
LABEL_VALID_FLAG_CANDIDATES = ("label_valid_flag", "is_label_valid", "valid_label")
FEATURE_MEMORY_CANDIDATES = (
    "primary_feature_memory",
    "feature_memory_days",
    "max_feature_memory_days",
    "feature_lookback_days",
)
PRIMARY_HORIZON_CANDIDATES = (
    "primary_horizon",
    "primary_horizon_days",
    "horizon_days",
)
HORIZONS_CANDIDATES = ("horizons", "horizon_days_list", "forward_horizons", "label_horizons")
DECISION_LAG_CANDIDATES = ("decision_lag", "decision_lag_days", "lag_days")


class PurgedSplitsError(RuntimeError):
    """Base error for temporal split construction."""


class ConfigError(PurgedSplitsError):
    """Raised when the split configuration is invalid."""


class MetadataError(PurgedSplitsError):
    """Raised when metadata is inconsistent or insufficient."""


class LeakageError(PurgedSplitsError):
    """Raised when leakage is detected under fail-fast policy."""


@dataclass(frozen=True)
class WalkForwardSpec:
    test_size_dates: int
    step_size_dates: int | None = None
    valid_size_dates: int = 0
    train_size_dates: int | None = None
    min_train_size_dates: int | None = None
    expanding_train: bool = True
    max_folds: int | None = None
    start_date: str | None = None
    end_date: str | None = None

    def normalized(self) -> "WalkForwardSpec":
        step = self.step_size_dates or self.test_size_dates
        min_train = self.min_train_size_dates
        if not self.expanding_train and self.train_size_dates is None:
            raise ConfigError("rolling walkforward requires train_size_dates")
        if self.test_size_dates <= 0:
            raise ConfigError("walkforward.test_size_dates must be > 0")
        if step <= 0:
            raise ConfigError("walkforward.step_size_dates must be > 0")
        if self.valid_size_dates < 0:
            raise ConfigError("walkforward.valid_size_dates cannot be negative")
        if self.train_size_dates is not None and self.train_size_dates <= 0:
            raise ConfigError("walkforward.train_size_dates must be > 0 when provided")
        if min_train is not None and min_train <= 0:
            raise ConfigError("walkforward.min_train_size_dates must be > 0 when provided")
        return WalkForwardSpec(
            test_size_dates=self.test_size_dates,
            step_size_dates=step,
            valid_size_dates=self.valid_size_dates,
            train_size_dates=self.train_size_dates,
            min_train_size_dates=min_train,
            expanding_train=self.expanding_train,
            max_folds=self.max_folds,
            start_date=self.start_date,
            end_date=self.end_date,
        )


@dataclass(frozen=True)
class CPCVSpec:
    n_blocks: int
    n_test_blocks: int = 2
    max_combinations: int | None = None

    def normalized(self) -> "CPCVSpec":
        if self.n_blocks < 2:
            raise ConfigError("cpcv.n_blocks must be >= 2")
        if self.n_test_blocks < 1:
            raise ConfigError("cpcv.n_test_blocks must be >= 1")
        if self.n_test_blocks >= self.n_blocks:
            raise ConfigError("cpcv.n_test_blocks must be < cpcv.n_blocks")
        return self


@dataclass(frozen=True)
class SplitConfig:
    split_mode: str = "purged_kfold"
    k_folds: int = 5
    fold_construction: str = "contiguous_date_blocks"
    decision_lag: int | None = None
    horizon_policy: str = "max"
    primary_horizon: int | None = None
    horizon_col: str | None = None
    embargo_policy: str = "feature_memory_aware"
    fixed_embargo_days: int | None = None
    pct_embargo: float | None = None
    primary_feature_memory: int | None = None
    min_train_obs: int = 1
    min_test_obs: int = 1
    leakage_policy: str = "fail"
    versioning_policy: str = "strict"
    policy_version: str = "1.0.0"
    date_col: str | None = None
    symbol_col: str | None = None
    sample_index_col: str | None = None
    label_valid_flag_col: str | None = None
    require_label_valid_flag: bool = True
    features_index_inclusion_col: str = "inclusion_flag"
    labels_path: str | None = None
    calendar_path: str | None = None
    labels_manifest_path: str | None = None
    feature_manifest_path: str | None = None
    features_index_path: str | None = None
    universe_history_path: str | None = None
    output_dir: str | None = None
    purge_warn_threshold: float = 0.35
    embargo_warn_threshold: float = 0.20
    low_density_warn_p10: float = 3.0
    imbalance_warn_ratio: float = 2.5
    walkforward: WalkForwardSpec | None = None
    cpcv: CPCVSpec | None = None

    @staticmethod
    def from_mapping(mapping: Mapping[str, Any]) -> "SplitConfig":
        mapping = dict(mapping)
        walkforward = mapping.get("walkforward")
        cpcv = mapping.get("cpcv")
        return SplitConfig(
            split_mode=str(mapping.get("split_mode", "purged_kfold")),
            k_folds=int(mapping.get("k_folds", 5)),
            fold_construction=str(mapping.get("fold_construction", "contiguous_date_blocks")),
            decision_lag=_optional_int(mapping.get("decision_lag")),
            horizon_policy=str(mapping.get("horizon_policy", "max")),
            primary_horizon=_optional_int(
                mapping.get("primary_horizon", mapping.get("primary_horizon_days"))
            ),
            horizon_col=_optional_str(mapping.get("horizon_col")),
            embargo_policy=str(mapping.get("embargo_policy", "feature_memory_aware")),
            fixed_embargo_days=_optional_int(mapping.get("fixed_embargo_days")),
            pct_embargo=_optional_float(mapping.get("pct_embargo")),
            primary_feature_memory=_optional_int(
                mapping.get("primary_feature_memory", mapping.get("feature_memory_days"))
            ),
            min_train_obs=int(mapping.get("min_train_obs", 1)),
            min_test_obs=int(mapping.get("min_test_obs", 1)),
            leakage_policy=str(mapping.get("leakage_policy", "fail")),
            versioning_policy=str(mapping.get("versioning_policy", "strict")),
            policy_version=str(mapping.get("policy_version", "1.0.0")),
            date_col=_optional_str(mapping.get("date_col")),
            symbol_col=_optional_str(mapping.get("symbol_col")),
            sample_index_col=_optional_str(mapping.get("sample_index_col")),
            label_valid_flag_col=_optional_str(mapping.get("label_valid_flag_col")),
            require_label_valid_flag=bool(mapping.get("require_label_valid_flag", True)),
            features_index_inclusion_col=str(
                mapping.get("features_index_inclusion_col", "inclusion_flag")
            ),
            labels_path=_optional_str(mapping.get("labels_path")),
            calendar_path=_optional_str(mapping.get("calendar_path")),
            labels_manifest_path=_optional_str(mapping.get("labels_manifest_path")),
            feature_manifest_path=_optional_str(mapping.get("feature_manifest_path")),
            features_index_path=_optional_str(mapping.get("features_index_path")),
            universe_history_path=_optional_str(mapping.get("universe_history_path")),
            output_dir=_optional_str(mapping.get("output_dir")),
            purge_warn_threshold=float(mapping.get("purge_warn_threshold", 0.35)),
            embargo_warn_threshold=float(mapping.get("embargo_warn_threshold", 0.20)),
            low_density_warn_p10=float(mapping.get("low_density_warn_p10", 3.0)),
            imbalance_warn_ratio=float(mapping.get("imbalance_warn_ratio", 2.5)),
            walkforward=WalkForwardSpec(**walkforward).normalized() if isinstance(walkforward, Mapping) else None,
            cpcv=CPCVSpec(**cpcv).normalized() if isinstance(cpcv, Mapping) else None,
        )

    def to_serializable(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.walkforward is not None:
            payload["walkforward"] = asdict(self.walkforward)
        if self.cpcv is not None:
            payload["cpcv"] = asdict(self.cpcv)
        return payload


@dataclass(frozen=True)
class ResolvedFields:
    date_col: str
    symbol_col: str
    sample_index_col: str | None
    horizon_col: str | None
    label_valid_flag_col: str | None


@dataclass(frozen=True)
class HorizonResolution:
    policy: str
    decision_lag: int
    effective_horizon_days: int
    all_horizons: tuple[int, ...]
    per_row_horizon_col: str | None
    labels_version: str | None


@dataclass(frozen=True)
class EmbargoResolution:
    policy: str
    effective_embargo_days: int
    primary_feature_memory: int | None
    non_canonical_flag: bool


@dataclass(frozen=True)
class FoldDefinition:
    fold_id: str
    split_mode: str
    train_dates: tuple[pd.Timestamp, ...]
    valid_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BuiltSplits:
    splits: pd.DataFrame
    fold_summary: pd.DataFrame
    leakage_validation: pd.DataFrame
    manifest: dict[str, Any]


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)



def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)



def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None



def _coerce_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()



def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()



def _read_json_or_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ConfigError("PyYAML is required to load YAML config files")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, Mapping):
        raise ConfigError(f"Expected mapping in config/manifest file: {path}")
    return dict(data)



def _resolve_existing_data_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    if not p.is_dir():
        raise FileNotFoundError(f"Unsupported path: {p}")
    candidates: list[Path] = []
    for pattern in ("*.parquet", "*.csv", "*.feather"):
        candidates.extend(sorted(p.glob(pattern)))
    if not candidates:
        raise FileNotFoundError(f"No supported data files found inside directory: {p}")
    if len(candidates) > 1:
        LOGGER.warning("Multiple candidate files found in %s; using %s", p, candidates[0])
    return candidates[0]



def _read_table(path: str | Path) -> pd.DataFrame:
    file_path = _resolve_existing_data_path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".feather":
        return pd.read_feather(file_path)
    raise ConfigError(f"Unsupported file format: {file_path}")



def load_split_config(config_path: str | Path | None, overrides: Mapping[str, Any] | None = None) -> SplitConfig:
    payload: dict[str, Any] = {}
    if config_path is not None:
        payload.update(_read_json_or_yaml(Path(config_path)))
    if overrides:
        payload.update(dict(overrides))
    config = SplitConfig.from_mapping(payload)
    if config.fold_construction != "contiguous_date_blocks":
        raise ConfigError("Only fold_construction='contiguous_date_blocks' is supported")
    if config.split_mode not in {"purged_kfold", "walkforward", "cpcv"}:
        raise ConfigError("split_mode must be one of: purged_kfold, walkforward, cpcv")
    if config.horizon_policy not in {"primary", "max", "per_label"}:
        raise ConfigError("horizon_policy must be one of: primary, max, per_label")
    if config.embargo_policy not in {"fixed_days", "pct_of_test", "feature_memory_aware"}:
        raise ConfigError(
            "embargo_policy must be one of: fixed_days, pct_of_test, feature_memory_aware"
        )
    if config.k_folds < 2 and config.split_mode == "purged_kfold":
        raise ConfigError("k_folds must be >= 2 for purged_kfold")
    return config



def _flatten_mapping(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_mapping(value, prefix=name))
        else:
            flat[name] = value
    return flat



def _extract_first(mapping: Mapping[str, Any], candidates: Sequence[str]) -> Any:
    flat = _flatten_mapping(mapping)
    lowered = {key.lower(): value for key, value in flat.items()}
    for candidate in candidates:
        candidate_lower = candidate.lower()
        if candidate_lower in lowered:
            return lowered[candidate_lower]
    for key, value in lowered.items():
        tail = key.split(".")[-1]
        if tail in {c.lower() for c in candidates}:
            return value
    return None



def _normalize_list_of_ints(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        chunks = [chunk.strip() for chunk in value.replace(";", ",").split(",") if chunk.strip()]
        return [int(x) for x in chunks]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [int(x) for x in value]
    return [int(value)]



def load_manifest(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest file not found: {p}")
    return _read_json_or_yaml(p)



def resolve_columns(labels_df: pd.DataFrame, config: SplitConfig) -> ResolvedFields:
    columns = list(labels_df.columns)
    lower_to_original = {col.lower(): col for col in columns}

    def pick(configured: str | None, candidates: Sequence[str], *, required: bool) -> str | None:
        if configured is not None:
            if configured not in labels_df.columns:
                raise ConfigError(f"Configured column '{configured}' not found in labels dataframe")
            return configured
        for candidate in candidates:
            match = lower_to_original.get(candidate.lower())
            if match is not None:
                return match
        if required:
            raise ConfigError(f"Could not resolve required column from candidates: {candidates}")
        return None

    date_col = pick(config.date_col, DATE_CANDIDATES, required=True)
    symbol_col = pick(config.symbol_col, SYMBOL_CANDIDATES, required=True)
    sample_index_col = pick(config.sample_index_col, SAMPLE_INDEX_CANDIDATES, required=False)
    horizon_col = pick(config.horizon_col, HORIZON_CANDIDATES, required=False)
    label_valid_flag_col = pick(
        config.label_valid_flag_col, LABEL_VALID_FLAG_CANDIDATES, required=False
    )

    return ResolvedFields(
        date_col=date_col,
        symbol_col=symbol_col,
        sample_index_col=sample_index_col,
        horizon_col=horizon_col,
        label_valid_flag_col=label_valid_flag_col,
    )



def load_labels(
    labels_df: pd.DataFrame | None = None,
    labels_path: str | Path | None = None,
) -> pd.DataFrame:
    if labels_df is None and labels_path is None:
        raise ConfigError("Either labels_df or labels_path must be provided")
    frame = labels_df.copy() if labels_df is not None else _read_table(labels_path)
    if frame.empty:
        raise ConfigError("Labels dataframe is empty")
    return frame



def load_calendar(calendar_df: pd.DataFrame | None = None, calendar_path: str | Path | None = None) -> pd.DataFrame:
    if calendar_df is None and calendar_path is None:
        raise ConfigError("Either calendar_df or calendar_path must be provided")
    frame = calendar_df.copy() if calendar_df is not None else _read_table(calendar_path)
    if frame.empty:
        raise ConfigError("Calendar dataframe is empty")
    date_col = None
    for candidate in DATE_CANDIDATES:
        if candidate in frame.columns:
            date_col = candidate
            break
    if date_col is None and len(frame.columns) == 1:
        date_col = frame.columns[0]
    if date_col is None:
        raise ConfigError("Could not infer calendar date column")
    out = pd.DataFrame({"date": pd.to_datetime(frame[date_col]).dt.normalize()})
    out = out.drop_duplicates().sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ConfigError("Calendar has no usable dates after normalization")
    out["calendar_idx"] = np.arange(len(out), dtype=int)
    return out



def filter_valid_labels(
    labels: pd.DataFrame,
    fields: ResolvedFields,
    config: SplitConfig,
) -> pd.DataFrame:
    frame = labels.copy()
    frame[fields.date_col] = pd.to_datetime(frame[fields.date_col]).dt.normalize()
    if fields.label_valid_flag_col is None:
        if config.require_label_valid_flag:
            raise ConfigError(
                "Labels dataframe is missing label_valid_flag column and require_label_valid_flag=True"
            )
        frame["__label_valid_flag"] = True
        label_valid_col = "__label_valid_flag"
    else:
        label_valid_col = fields.label_valid_flag_col
    frame = frame.loc[frame[label_valid_col].fillna(False).astype(bool)].copy()
    if frame.empty:
        raise ConfigError("No valid labels remain after applying label_valid_flag filter")
    frame = frame.sort_values(
        [fields.date_col, fields.symbol_col]
        + ([fields.sample_index_col] if fields.sample_index_col is not None else [])
    ).reset_index(drop=True)
    return frame



def reconcile_with_features_index(
    labels: pd.DataFrame,
    fields: ResolvedFields,
    config: SplitConfig,
    features_index_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if features_index_df is None:
        return labels
    frame = features_index_df.copy()
    feature_date_col = None
    feature_symbol_col = None
    for candidate in DATE_CANDIDATES:
        if candidate in frame.columns:
            feature_date_col = candidate
            break
    for candidate in SYMBOL_CANDIDATES:
        if candidate in frame.columns:
            feature_symbol_col = candidate
            break
    if feature_date_col is None or feature_symbol_col is None:
        raise ConfigError("features_index is missing date/symbol columns")
    frame[feature_date_col] = pd.to_datetime(frame[feature_date_col]).dt.normalize()
    inclusion_col = config.features_index_inclusion_col
    if inclusion_col in frame.columns:
        frame = frame.loc[frame[inclusion_col].fillna(False).astype(bool)].copy()
    join_cols_labels = [fields.date_col, fields.symbol_col]
    join_cols_features = [feature_date_col, feature_symbol_col]
    if fields.sample_index_col is not None and fields.sample_index_col in frame.columns:
        join_cols_labels.append(fields.sample_index_col)
        join_cols_features.append(fields.sample_index_col)
    right = frame[join_cols_features].drop_duplicates().copy()
    rename_map = dict(zip(join_cols_features, join_cols_labels))
    right = right.rename(columns=rename_map)
    merged = labels.merge(right, on=join_cols_labels, how="inner")
    if merged.empty:
        raise ConfigError("Reconciliation with features_index removed all labels")
    return merged



def validate_calendar_consistency(
    labels: pd.DataFrame,
    calendar: pd.DataFrame,
    date_col: str,
) -> None:
    observed = set(pd.to_datetime(labels[date_col]).dt.normalize().unique())
    available = set(calendar["date"].tolist())
    missing = sorted(observed - available)
    if missing:
        preview = ", ".join(str(x.date()) for x in missing[:10])
        raise MetadataError(
            f"Calendar is inconsistent with labels; missing observed dates: {preview}"
        )



def resolve_effective_horizon(
    labels: pd.DataFrame,
    fields: ResolvedFields,
    config: SplitConfig,
    labels_metadata: Mapping[str, Any],
) -> HorizonResolution:
    metadata_decision_lag = _extract_first(labels_metadata, DECISION_LAG_CANDIDATES)
    decision_lag_candidates = [x for x in [config.decision_lag, metadata_decision_lag] if x is not None]
    if not decision_lag_candidates:
        raise MetadataError("decision_lag is not resolvable from config or labels metadata")
    decision_lag = int(decision_lag_candidates[0])
    if any(int(x) != decision_lag for x in decision_lag_candidates):
        raise MetadataError("decision_lag conflict between config and labels metadata")
    if decision_lag < 0:
        raise MetadataError("decision_lag cannot be negative")

    labels_version = _optional_str(_extract_first(labels_metadata, ("labels_version", "version")))

    horizons_from_metadata = _normalize_list_of_ints(_extract_first(labels_metadata, HORIZONS_CANDIDATES))
    primary_horizon_meta = _optional_int(_extract_first(labels_metadata, PRIMARY_HORIZON_CANDIDATES))
    horizons_from_column: list[int] = []
    if fields.horizon_col is not None:
        horizons_from_column = sorted(
            {int(x) for x in pd.Series(labels[fields.horizon_col]).dropna().astype(int).tolist()}
        )

    all_horizons = sorted(
        {
            *horizons_from_metadata,
            *horizons_from_column,
            *( [primary_horizon_meta] if primary_horizon_meta is not None else [] ),
            *( [config.primary_horizon] if config.primary_horizon is not None else [] ),
        }
    )
    if not all_horizons:
        raise MetadataError("Could not resolve horizon information from config, labels, or metadata")
    if any(h <= 0 for h in all_horizons):
        raise MetadataError(f"All horizons must be positive, got: {all_horizons}")

    if config.horizon_policy == "primary":
        primary_candidates = [x for x in [config.primary_horizon, primary_horizon_meta] if x is not None]
        if not primary_candidates:
            raise MetadataError("primary horizon policy requires primary_horizon in config or metadata")
        effective_horizon = int(primary_candidates[0])
        if any(int(x) != effective_horizon for x in primary_candidates):
            raise MetadataError("primary_horizon conflict between config and labels metadata")
        per_row_horizon_col = None
    elif config.horizon_policy == "max":
        effective_horizon = int(max(all_horizons))
        per_row_horizon_col = None
    else:
        if fields.horizon_col is None:
            raise MetadataError(
                "horizon_policy='per_label' requires a per-row horizon column in labels dataframe"
            )
        effective_horizon = int(max(all_horizons))
        per_row_horizon_col = fields.horizon_col

    return HorizonResolution(
        policy=config.horizon_policy,
        decision_lag=decision_lag,
        effective_horizon_days=effective_horizon,
        all_horizons=tuple(int(x) for x in all_horizons),
        per_row_horizon_col=per_row_horizon_col,
        labels_version=labels_version,
    )



def resolve_effective_embargo(
    config: SplitConfig,
    feature_metadata: Mapping[str, Any],
    effective_horizon_days: int,
    test_block_len: int,
) -> EmbargoResolution:
    metadata_feature_memory = _optional_int(_extract_first(feature_metadata, FEATURE_MEMORY_CANDIDATES))
    primary_feature_memory_candidates = [
        x for x in [config.primary_feature_memory, metadata_feature_memory] if x is not None
    ]
    non_canonical_flag = False

    if config.embargo_policy == "fixed_days":
        if config.fixed_embargo_days is None:
            raise MetadataError("fixed_days embargo policy requires fixed_embargo_days")
        b_eff = int(config.fixed_embargo_days)
    elif config.embargo_policy == "pct_of_test":
        if config.pct_embargo is None:
            raise MetadataError("pct_of_test embargo policy requires pct_embargo")
        if config.pct_embargo < 0:
            raise MetadataError("pct_embargo cannot be negative")
        b_eff = int(math.ceil(config.pct_embargo * int(test_block_len)))
    else:
        if primary_feature_memory_candidates:
            b_eff = int(max(int(x) for x in primary_feature_memory_candidates + [effective_horizon_days]))
        elif config.fixed_embargo_days is not None:
            b_eff = int(config.fixed_embargo_days)
            non_canonical_flag = True
        else:
            raise MetadataError(
                "feature_memory_aware embargo requires feature memory metadata or fixed_embargo_days fallback"
            )

    if b_eff < 0:
        raise MetadataError("Resolved embargo cannot be negative")

    return EmbargoResolution(
        policy=config.embargo_policy,
        effective_embargo_days=b_eff,
        primary_feature_memory=max(primary_feature_memory_candidates) if primary_feature_memory_candidates else None,
        non_canonical_flag=non_canonical_flag,
    )



def build_event_intervals(
    labels: pd.DataFrame,
    calendar: pd.DataFrame,
    fields: ResolvedFields,
    horizon_resolution: HorizonResolution,
) -> pd.DataFrame:
    frame = labels.copy()
    frame[fields.date_col] = pd.to_datetime(frame[fields.date_col]).dt.normalize()
    calendar_index = calendar.set_index("date")["calendar_idx"]
    try:
        frame["__decision_idx"] = frame[fields.date_col].map(calendar_index).astype(int)
    except Exception as exc:
        raise MetadataError("Could not map label dates onto master calendar") from exc

    if horizon_resolution.per_row_horizon_col is not None:
        row_horizon = pd.to_numeric(frame[horizon_resolution.per_row_horizon_col], errors="raise").astype(int)
    else:
        row_horizon = pd.Series(
            horizon_resolution.effective_horizon_days, index=frame.index, dtype="int64"
        )
    frame["effective_horizon_days"] = row_horizon
    frame["effective_event_start_idx"] = frame["__decision_idx"] + horizon_resolution.decision_lag
    frame["effective_event_end_idx"] = frame["effective_event_start_idx"] + frame["effective_horizon_days"]

    max_calendar_idx = int(calendar["calendar_idx"].max())
    invalid = frame.loc[
        (frame["effective_event_start_idx"] > max_calendar_idx)
        | (frame["effective_event_end_idx"] > max_calendar_idx)
    ]
    if not invalid.empty:
        preview = invalid[[fields.date_col, fields.symbol_col]].head(5).to_dict("records")
        raise MetadataError(
            f"Labels contain intervals beyond available master calendar coverage; examples: {preview}"
        )

    calendar_dates = calendar.set_index("calendar_idx")["date"]
    frame["effective_event_start"] = frame["effective_event_start_idx"].map(calendar_dates)
    frame["effective_event_end"] = frame["effective_event_end_idx"].map(calendar_dates)
    return frame



def construct_time_blocks(dates: Sequence[pd.Timestamp], n_blocks: int) -> list[tuple[pd.Timestamp, ...]]:
    if n_blocks < 1:
        raise ConfigError("n_blocks must be >= 1")
    if len(dates) < n_blocks:
        raise ConfigError(
            f"Not enough unique dates ({len(dates)}) to build {n_blocks} contiguous date blocks"
        )
    chunks = np.array_split(np.array(list(dates), dtype="datetime64[ns]"), n_blocks)
    blocks: list[tuple[pd.Timestamp, ...]] = []
    for block in chunks:
        if len(block) == 0:
            raise ConfigError("Encountered empty time block during construction")
        blocks.append(tuple(pd.Timestamp(x).normalize() for x in block.tolist()))
    return blocks



def build_purged_kfold_folds(unique_dates: Sequence[pd.Timestamp], config: SplitConfig) -> list[FoldDefinition]:
    blocks = construct_time_blocks(unique_dates, config.k_folds)
    folds: list[FoldDefinition] = []
    for idx, test_dates in enumerate(blocks, start=1):
        test_set = set(test_dates)
        train_dates = tuple(date for date in unique_dates if date not in test_set)
        folds.append(
            FoldDefinition(
                fold_id=f"fold_{idx:02d}",
                split_mode="purged_kfold",
                train_dates=train_dates,
                valid_dates=tuple(),
                test_dates=test_dates,
                metadata={"test_block_index": idx - 1, "n_blocks": config.k_folds},
            )
        )
    return folds



def _slice_dates_by_optional_bounds(
    dates: Sequence[pd.Timestamp], start_date: str | None, end_date: str | None
) -> list[pd.Timestamp]:
    start_ts = _coerce_timestamp(start_date) if start_date else None
    end_ts = _coerce_timestamp(end_date) if end_date else None
    out: list[pd.Timestamp] = []
    for date in dates:
        if start_ts is not None and date < start_ts:
            continue
        if end_ts is not None and date > end_ts:
            continue
        out.append(date)
    return out



def build_walkforward_folds(
    unique_dates: Sequence[pd.Timestamp],
    config: SplitConfig,
) -> list[FoldDefinition]:
    if config.walkforward is None:
        raise ConfigError("walkforward split_mode requires a walkforward configuration block")
    spec = config.walkforward.normalized()
    dates = _slice_dates_by_optional_bounds(unique_dates, spec.start_date, spec.end_date)
    if len(dates) < spec.test_size_dates + spec.valid_size_dates + 1:
        raise ConfigError("Not enough dates for requested walkforward specification")

    min_train = spec.min_train_size_dates
    if min_train is None:
        min_train = spec.train_size_dates if spec.train_size_dates is not None else max(1, spec.test_size_dates)
    cursor = int(min_train)
    folds: list[FoldDefinition] = []
    fold_number = 1

    while cursor + spec.valid_size_dates + spec.test_size_dates <= len(dates):
        if spec.expanding_train:
            train_start_idx = 0
            train_end_idx = cursor
            if spec.train_size_dates is not None:
                train_start_idx = max(0, train_end_idx - spec.train_size_dates)
        else:
            assert spec.train_size_dates is not None
            train_end_idx = cursor
            train_start_idx = max(0, train_end_idx - spec.train_size_dates)

        valid_start_idx = cursor
        valid_end_idx = valid_start_idx + spec.valid_size_dates
        test_start_idx = valid_end_idx
        test_end_idx = test_start_idx + spec.test_size_dates

        train_dates = tuple(dates[train_start_idx:train_end_idx])
        valid_dates = tuple(dates[valid_start_idx:valid_end_idx])
        test_dates = tuple(dates[test_start_idx:test_end_idx])

        if len(train_dates) == 0 or len(test_dates) == 0:
            break

        folds.append(
            FoldDefinition(
                fold_id=f"fold_{fold_number:02d}",
                split_mode="walkforward",
                train_dates=train_dates,
                valid_dates=valid_dates,
                test_dates=test_dates,
                metadata={
                    "expanding_train": spec.expanding_train,
                    "train_size_dates": spec.train_size_dates,
                    "valid_size_dates": spec.valid_size_dates,
                    "test_size_dates": spec.test_size_dates,
                    "step_size_dates": spec.step_size_dates,
                },
            )
        )

        fold_number += 1
        cursor += int(spec.step_size_dates or spec.test_size_dates)
        if spec.max_folds is not None and len(folds) >= spec.max_folds:
            break

    if not folds:
        raise ConfigError("walkforward specification did not generate any valid folds")
    return folds



def build_cpcv_folds(unique_dates: Sequence[pd.Timestamp], config: SplitConfig) -> list[FoldDefinition]:
    if config.cpcv is None:
        raise ConfigError("cpcv split_mode requires a cpcv configuration block")
    spec = config.cpcv.normalized()
    blocks = construct_time_blocks(unique_dates, spec.n_blocks)
    combinations_iter = itertools.combinations(range(spec.n_blocks), spec.n_test_blocks)
    folds: list[FoldDefinition] = []
    for fold_number, combo in enumerate(combinations_iter, start=1):
        test_set = {date for block_idx in combo for date in blocks[block_idx]}
        train_dates = tuple(date for date in unique_dates if date not in test_set)
        test_dates = tuple(date for date in unique_dates if date in test_set)
        folds.append(
            FoldDefinition(
                fold_id=f"fold_{fold_number:02d}",
                split_mode="cpcv",
                train_dates=train_dates,
                valid_dates=tuple(),
                test_dates=test_dates,
                metadata={
                    "base_n_blocks": spec.n_blocks,
                    "n_test_blocks": spec.n_test_blocks,
                    "selected_test_block_indices": list(combo),
                    "multiplicity": math.comb(spec.n_blocks, spec.n_test_blocks),
                },
            )
        )
        if spec.max_combinations is not None and len(folds) >= spec.max_combinations:
            break
    if not folds:
        raise ConfigError("cpcv configuration did not generate any folds")
    return folds



def build_fold_definitions(unique_dates: Sequence[pd.Timestamp], config: SplitConfig) -> list[FoldDefinition]:
    if config.split_mode == "purged_kfold":
        return build_purged_kfold_folds(unique_dates, config)
    if config.split_mode == "walkforward":
        return build_walkforward_folds(unique_dates, config)
    if config.split_mode == "cpcv":
        return build_cpcv_folds(unique_dates, config)
    raise ConfigError(f"Unsupported split_mode: {config.split_mode}")



def _merge_intervals(starts: Sequence[int], ends: Sequence[int]) -> list[tuple[int, int]]:
    if len(starts) == 0:
        return []
    order = np.argsort(np.asarray(starts, dtype=int), kind="mergesort")
    sorted_starts = np.asarray(starts, dtype=int)[order]
    sorted_ends = np.asarray(ends, dtype=int)[order]
    merged: list[tuple[int, int]] = []
    current_start = int(sorted_starts[0])
    current_end = int(sorted_ends[0])
    for start, end in zip(sorted_starts[1:], sorted_ends[1:]):
        s = int(start)
        e = int(end)
        if s <= current_end + 1:
            current_end = max(current_end, e)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = s, e
    merged.append((current_start, current_end))
    return merged



def _interval_overlaps_any(
    starts: np.ndarray,
    ends: np.ndarray,
    merged_intervals: Sequence[tuple[int, int]],
) -> np.ndarray:
    if len(merged_intervals) == 0:
        return np.zeros(len(starts), dtype=bool)
    interval_starts = np.asarray([s for s, _ in merged_intervals], dtype=int)
    interval_ends = np.asarray([e for _, e in merged_intervals], dtype=int)
    candidate_idx = np.searchsorted(interval_ends, starts, side="left")
    valid = candidate_idx < len(interval_starts)
    overlap = np.zeros(len(starts), dtype=bool)
    overlap[valid] = interval_starts[candidate_idx[valid]] <= ends[valid]
    return overlap



def _next_calendar_idx_with_cap(last_idx: int, step: int, max_idx: int) -> int:
    return min(last_idx + step, max_idx)



def _build_embargo_dates(calendar: pd.DataFrame, last_test_date: pd.Timestamp, b_eff: int) -> set[pd.Timestamp]:
    if b_eff <= 0:
        return set()
    calendar_index = calendar.set_index("date")["calendar_idx"]
    last_idx = int(calendar_index[last_test_date])
    max_idx = int(calendar["calendar_idx"].max())
    embargo_end_idx = _next_calendar_idx_with_cap(last_idx, b_eff, max_idx)
    embargo_mask = (calendar["calendar_idx"] > last_idx) & (calendar["calendar_idx"] <= embargo_end_idx)
    return set(calendar.loc[embargo_mask, "date"].tolist())



def _date_stats(frame: pd.DataFrame, date_col: str, symbol_col: str) -> tuple[float, float, float]:
    if frame.empty:
        return 0.0, 0.0, 0.0
    by_date = frame.groupby(date_col)[symbol_col].nunique().astype(float)
    return float(by_date.mean()), float(by_date.quantile(0.10)), float(by_date.quantile(0.90))



def _build_fold_rows(
    intervals: pd.DataFrame,
    fields: ResolvedFields,
    fold: FoldDefinition,
    calendar: pd.DataFrame,
    config: SplitConfig,
    horizon_resolution: HorizonResolution,
    feature_metadata: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    date_col = fields.date_col
    symbol_col = fields.symbol_col

    frame = intervals.copy()
    train_dates_set = set(fold.train_dates)
    valid_dates_set = set(fold.valid_dates)
    test_dates_set = set(fold.test_dates)
    if not test_dates_set:
        raise ConfigError(f"Fold {fold.fold_id} has no test dates")

    frame["split_role"] = np.where(
        frame[date_col].isin(test_dates_set),
        "test",
        np.where(frame[date_col].isin(valid_dates_set), "valid", "train"),
    )
    frame["is_in_raw_train"] = frame[date_col].isin(train_dates_set)

    eval_frame = frame.loc[frame["split_role"].isin(["valid", "test"])].copy()
    test_frame = frame.loc[frame["split_role"] == "test"].copy()
    train_raw = frame.loc[frame["is_in_raw_train"]].copy()

    if test_frame.empty:
        raise ConfigError(f"Fold {fold.fold_id} has zero test observations")

    eval_intervals = _merge_intervals(
        eval_frame["effective_event_start_idx"].to_numpy(dtype=int),
        eval_frame["effective_event_end_idx"].to_numpy(dtype=int),
    )
    purge_mask = _interval_overlaps_any(
        train_raw["effective_event_start_idx"].to_numpy(dtype=int),
        train_raw["effective_event_end_idx"].to_numpy(dtype=int),
        eval_intervals,
    )

    test_last_date = pd.Timestamp(max(test_dates_set)).normalize()
    embargo_resolution = resolve_effective_embargo(
        config=config,
        feature_metadata=feature_metadata,
        effective_horizon_days=horizon_resolution.effective_horizon_days,
        test_block_len=len(test_dates_set),
    )
    embargo_dates = _build_embargo_dates(calendar, test_last_date, embargo_resolution.effective_embargo_days)
    embargo_mask = train_raw[date_col].isin(embargo_dates).to_numpy(dtype=bool)

    train_raw = train_raw.assign(is_purged=purge_mask, is_embargoed=embargo_mask)
    train_raw["inclusion_flag"] = ~(train_raw["is_purged"] | train_raw["is_embargoed"])

    eval_frame = eval_frame.assign(is_purged=False, is_embargoed=False, inclusion_flag=True)
    fold_rows = pd.concat([train_raw, eval_frame], axis=0, ignore_index=True)

    if fields.sample_index_col is None:
        fold_rows["sample_index"] = pd.NA
    else:
        fold_rows["sample_index"] = fold_rows[fields.sample_index_col]

    fold_rows["fold_id"] = fold.fold_id
    fold_rows["run_id"] = None
    fold_rows["effective_embargo_days"] = embargo_resolution.effective_embargo_days

    included_train = fold_rows.loc[(fold_rows["split_role"] == "train") & (fold_rows["inclusion_flag"])]
    included_eval = fold_rows.loc[fold_rows["split_role"].isin(["valid", "test"])]
    leakage_rows: list[dict[str, Any]] = []

    if not included_train.empty and not included_eval.empty:
        eval_merged = _merge_intervals(
            included_eval["effective_event_start_idx"].to_numpy(dtype=int),
            included_eval["effective_event_end_idx"].to_numpy(dtype=int),
        )
        violating_overlap = _interval_overlaps_any(
            included_train["effective_event_start_idx"].to_numpy(dtype=int),
            included_train["effective_event_end_idx"].to_numpy(dtype=int),
            eval_merged,
        )
        if np.any(violating_overlap):
            bad = included_train.loc[violating_overlap, [date_col, symbol_col, "sample_index"]]
            for row in bad.itertuples(index=False):
                leakage_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "check_name": "train_eval_interval_overlap",
                        "status": "FAIL",
                        "date": getattr(row, date_col),
                        "symbol": getattr(row, symbol_col),
                        "sample_index": getattr(row, "sample_index"),
                        "details": "Included train sample overlaps evaluation event window",
                    }
                )

    if not included_train.empty and embargo_dates:
        bad_embargo = included_train.loc[included_train[date_col].isin(embargo_dates), [date_col, symbol_col, "sample_index"]]
        for row in bad_embargo.itertuples(index=False):
            leakage_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "check_name": "train_in_embargo",
                    "status": "FAIL",
                    "date": getattr(row, date_col),
                    "symbol": getattr(row, symbol_col),
                    "sample_index": getattr(row, "sample_index"),
                    "details": "Included train sample falls inside embargo window",
                }
            )

    leakage_validation = pd.DataFrame(leakage_rows)
    if leakage_validation.empty:
        leakage_validation = pd.DataFrame(
            [
                {
                    "fold_id": fold.fold_id,
                    "check_name": "validate_zero_leakage",
                    "status": "PASS",
                    "date": pd.NaT,
                    "symbol": pd.NA,
                    "sample_index": pd.NA,
                    "details": "No overlap or embargo violations detected",
                }
            ]
        )

    n_train_raw = int((fold_rows["split_role"] == "train").sum())
    n_train_after_purge = int(
        ((fold_rows["split_role"] == "train") & (~fold_rows["is_purged"])).sum()
    )
    n_train_purged = n_train_after_purge
    n_train_final = int(((fold_rows["split_role"] == "train") & (fold_rows["inclusion_flag"])).sum())
    n_valid = int((fold_rows["split_role"] == "valid").sum())
    n_test = int((fold_rows["split_role"] == "test").sum())
    if n_train_final < config.min_train_obs:
        raise ConfigError(
            f"Fold {fold.fold_id} violates min_train_obs: {n_train_final} < {config.min_train_obs}"
        )
    if n_test < config.min_test_obs:
        raise ConfigError(
            f"Fold {fold.fold_id} violates min_test_obs: {n_test} < {config.min_test_obs}"
        )

    density_mean, density_p10, density_p90 = _date_stats(
        fold_rows.loc[fold_rows["inclusion_flag"]], date_col, symbol_col
    )

    warnings_list: list[str] = []
    removed_purge = int(((fold_rows["split_role"] == "train") & (fold_rows["is_purged"])).sum())
    removed_embargo = int(((fold_rows["split_role"] == "train") & (~fold_rows["is_purged"]) & (fold_rows["is_embargoed"])).sum())
    pct_removed_by_purge = removed_purge / n_train_raw if n_train_raw else 0.0
    pct_removed_by_embargo = removed_embargo / n_train_raw if n_train_raw else 0.0
    if pct_removed_by_purge > config.purge_warn_threshold:
        warnings_list.append("high_pct_removed_by_purge")
    if pct_removed_by_embargo > config.embargo_warn_threshold:
        warnings_list.append("high_pct_removed_by_embargo")
    if density_p10 < config.low_density_warn_p10:
        warnings_list.append("low_cross_sectional_density")
    if test_frame[date_col].nunique() <= max(1, horizon_resolution.effective_horizon_days // 2):
        warnings_list.append("narrow_test_temporal_coverage")
    if embargo_resolution.non_canonical_flag:
        warnings_list.append("non_canonical_embargo_fallback")

    summary = {
        "fold_id": fold.fold_id,
        "split_mode": fold.split_mode,
        "n_train_raw": n_train_raw,
        "n_train_purged": n_train_purged,
        "n_train_final": n_train_final,
        "n_valid": n_valid,
        "n_test": n_test,
        "pct_removed_by_purge": pct_removed_by_purge,
        "pct_removed_by_embargo": pct_removed_by_embargo,
        "leakage_violations_count": int((leakage_validation["status"] == "FAIL").sum()),
        "effective_horizon_days": horizon_resolution.effective_horizon_days,
        "effective_embargo_days": embargo_resolution.effective_embargo_days,
        "first_test_date": pd.Timestamp(min(test_dates_set)).normalize(),
        "last_test_date": test_last_date,
        "symbols_per_date_mean": density_mean,
        "symbols_per_date_p10": density_p10,
        "symbols_per_date_p90": density_p90,
        "warnings": json.dumps(warnings_list, ensure_ascii=False),
        "fold_metadata": json.dumps(fold.metadata, ensure_ascii=False, default=str),
    }

    output_cols = [
        "fold_id",
        "split_role",
        date_col,
        symbol_col,
        "sample_index",
        "inclusion_flag",
        "is_purged",
        "is_embargoed",
        "effective_event_start",
        "effective_event_end",
        "effective_horizon_days",
        "effective_embargo_days",
        "run_id",
    ]
    fold_rows = fold_rows.rename(columns={date_col: "date", symbol_col: "symbol"})
    output_cols = [col if col not in {date_col, symbol_col} else {date_col: "date", symbol_col: "symbol"}[col] for col in output_cols]
    fold_rows = fold_rows[output_cols].copy()
    return fold_rows, summary, leakage_validation



def compute_fold_diagnostics(
    fold_summary: pd.DataFrame,
    config: SplitConfig,
) -> pd.DataFrame:
    if fold_summary.empty:
        return fold_summary
    ratio = (
        fold_summary["n_train_final"].max() / max(float(fold_summary["n_train_final"].min()), 1.0)
    )
    if ratio > config.imbalance_warn_ratio:
        fold_summary = fold_summary.copy()
        fold_summary["warnings"] = fold_summary["warnings"].apply(
            lambda raw: json.dumps(sorted(set(json.loads(raw)) | {"strong_fold_imbalance"}), ensure_ascii=False)
        )
    return fold_summary



def emit_manifest(
    config: SplitConfig,
    config_hash: str,
    horizon_resolution: HorizonResolution,
    embargo_resolution: EmbargoResolution,
    calendar: pd.DataFrame,
    run_id: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "config_hash": config_hash,
        "labels_version": horizon_resolution.labels_version,
        "calendar_version": None,
        "split_mode": config.split_mode,
        "horizon_policy": config.horizon_policy,
        "h_eff": horizon_resolution.effective_horizon_days,
        "embargo_policy": embargo_resolution.policy,
        "b_eff": embargo_resolution.effective_embargo_days,
        "decision_lag": horizon_resolution.decision_lag,
        "all_horizons": list(horizon_resolution.all_horizons),
        "non_canonical_flag": embargo_resolution.non_canonical_flag,
        "policy_version": config.policy_version,
        "fold_construction": config.fold_construction,
        "calendar_start": str(calendar["date"].min().date()),
        "calendar_end": str(calendar["date"].max().date()),
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
    }



def persist_outputs(result: BuiltSplits, output_dir: str | Path) -> dict[str, str]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    splits_path = outdir / "splits.parquet"
    fold_summary_path = outdir / "fold_summary.parquet"
    leakage_validation_path = outdir / "leakage_validation.parquet"
    manifest_path = outdir / "manifest.json"

    result.splits.to_parquet(splits_path, index=False)
    result.fold_summary.to_parquet(fold_summary_path, index=False)
    result.leakage_validation.to_parquet(leakage_validation_path, index=False)
    manifest_path.write_text(json.dumps(result.manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "splits": str(splits_path),
        "fold_summary": str(fold_summary_path),
        "leakage_validation": str(leakage_validation_path),
        "manifest": str(manifest_path),
    }



def build_purged_splits(
    *,
    labels_df: pd.DataFrame | None = None,
    labels_path: str | Path | None = None,
    calendar_df: pd.DataFrame | None = None,
    calendar_path: str | Path | None = None,
    config: SplitConfig | None = None,
    config_path: str | Path | None = None,
    run_id: str,
    labels_manifest: Mapping[str, Any] | None = None,
    labels_manifest_path: str | Path | None = None,
    feature_manifest: Mapping[str, Any] | None = None,
    feature_manifest_path: str | Path | None = None,
    features_index_df: pd.DataFrame | None = None,
    features_index_path: str | Path | None = None,
) -> BuiltSplits:
    cfg = config or load_split_config(config_path)
    labels_meta = dict(labels_manifest or {})
    if labels_manifest_path is not None:
        labels_meta.update(load_manifest(labels_manifest_path))
    elif cfg.labels_manifest_path:
        labels_meta.update(load_manifest(cfg.labels_manifest_path))

    feature_meta = dict(feature_manifest or {})
    if feature_manifest_path is not None:
        feature_meta.update(load_manifest(feature_manifest_path))
    elif cfg.feature_manifest_path:
        feature_meta.update(load_manifest(cfg.feature_manifest_path))

    labels = load_labels(labels_df=labels_df, labels_path=labels_path or cfg.labels_path)
    calendar = load_calendar(calendar_df=calendar_df, calendar_path=calendar_path or cfg.calendar_path)
    fields = resolve_columns(labels, cfg)
    labels = filter_valid_labels(labels, fields, cfg)

    if features_index_df is None and (features_index_path or cfg.features_index_path):
        features_index_df = _read_table(features_index_path or cfg.features_index_path)  # type: ignore[arg-type]
    labels = reconcile_with_features_index(labels, fields, cfg, features_index_df)
    validate_calendar_consistency(labels, calendar, fields.date_col)

    horizon_resolution = resolve_effective_horizon(labels, fields, cfg, labels_meta)
    intervals = build_event_intervals(labels, calendar, fields, horizon_resolution)
    unique_dates = [pd.Timestamp(x).normalize() for x in sorted(intervals[fields.date_col].unique())]
    folds = build_fold_definitions(unique_dates, cfg)
    if not folds:
        raise ConfigError("No folds were generated")

    config_payload = cfg.to_serializable()
    config_payload["labels_metadata"] = labels_meta
    config_payload["feature_metadata"] = feature_meta
    config_payload["run_id"] = run_id
    config_hash = _stable_hash(config_payload)

    all_splits: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []
    leakage_frames: list[pd.DataFrame] = []
    resolved_embargos: list[int] = []
    non_canonical_any = False

    for fold in folds:
        fold_rows, summary, leakage_validation = _build_fold_rows(
            intervals=intervals,
            fields=fields,
            fold=fold,
            calendar=calendar,
            config=cfg,
            horizon_resolution=horizon_resolution,
            feature_metadata=feature_meta,
        )
        # re-resolve embargo using actual test block length for manifest-wide conservative summary
        embargo_resolution = resolve_effective_embargo(
            config=cfg,
            feature_metadata=feature_meta,
            effective_horizon_days=horizon_resolution.effective_horizon_days,
            test_block_len=len(fold.test_dates),
        )
        fold_rows["effective_embargo_days"] = embargo_resolution.effective_embargo_days
        fold_rows["run_id"] = run_id
        summary["effective_embargo_days"] = embargo_resolution.effective_embargo_days
        if embargo_resolution.non_canonical_flag:
            warnings = set(json.loads(summary["warnings"]))
            warnings.add("non_canonical_embargo_fallback")
            summary["warnings"] = json.dumps(sorted(warnings), ensure_ascii=False)
            non_canonical_any = True
        resolved_embargos.append(embargo_resolution.effective_embargo_days)
        all_splits.append(fold_rows)
        fold_summaries.append(summary)
        leakage_frames.append(leakage_validation)

    splits = pd.concat(all_splits, axis=0, ignore_index=True)
    fold_summary = pd.DataFrame(fold_summaries)
    fold_summary = compute_fold_diagnostics(fold_summary, cfg)
    leakage_validation = pd.concat(leakage_frames, axis=0, ignore_index=True)

    fail_rows = leakage_validation.loc[leakage_validation["status"] == "FAIL"]
    if not fail_rows.empty and cfg.leakage_policy == "fail":
        preview = fail_rows.head(10).to_dict("records")
        raise LeakageError(f"Leakage detected in generated splits; examples: {preview}")

    embargo_resolution_manifest = EmbargoResolution(
        policy=cfg.embargo_policy,
        effective_embargo_days=max(resolved_embargos) if resolved_embargos else 0,
        primary_feature_memory=cfg.primary_feature_memory,
        non_canonical_flag=non_canonical_any,
    )
    manifest = emit_manifest(
        config=cfg,
        config_hash=config_hash,
        horizon_resolution=horizon_resolution,
        embargo_resolution=embargo_resolution_manifest,
        calendar=calendar,
        run_id=run_id,
    )

    return BuiltSplits(
        splits=splits.sort_values(["fold_id", "date", "symbol", "split_role"]).reset_index(drop=True),
        fold_summary=fold_summary.sort_values("fold_id").reset_index(drop=True),
        leakage_validation=leakage_validation.sort_values(["fold_id", "status", "check_name"]).reset_index(drop=True),
        manifest=manifest,
    )



def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build purged temporal splits for labels panels")
    parser.add_argument("--labels-path", required=True, help="Path to labels parquet/csv/feather or directory")
    parser.add_argument("--calendar-path", required=True, help="Path to master calendar parquet/csv/feather")
    parser.add_argument("--config-path", required=True, help="Path to purged_splits YAML/JSON config")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument("--labels-manifest-path", default=None)
    parser.add_argument("--feature-manifest-path", default=None)
    parser.add_argument("--features-index-path", default=None)
    parser.add_argument("--output-dir", default=None, help="Directory where artifacts will be written")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args(argv)



def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )



def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_logging(args.log_level)
    config = load_split_config(args.config_path)
    result = build_purged_splits(
        labels_path=args.labels_path,
        calendar_path=args.calendar_path,
        config=config,
        run_id=args.run_id,
        labels_manifest_path=args.labels_manifest_path,
        feature_manifest_path=args.feature_manifest_path,
        features_index_path=args.features_index_path,
    )
    output_dir = args.output_dir or config.output_dir
    if output_dir is None:
        raise ConfigError("output_dir must be provided either in CLI or config")
    artifact_paths = persist_outputs(result, output_dir)
    LOGGER.info("Purged splits built successfully: %s", json.dumps(artifact_paths, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
