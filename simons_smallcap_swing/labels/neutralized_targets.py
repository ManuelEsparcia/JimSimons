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
    "decision_date",
    "trade_date",
    "session_date",
)
SYMBOL_CANDIDATES = (
    "symbol",
    "ticker",
    "asset",
    "instrument_id",
)
HORIZON_CANDIDATES = (
    "horizon",
    "horizon_days",
    "label_horizon_days",
    "forward_horizon_days",
)
LABEL_VALID_FLAG_CANDIDATES = (
    "label_valid_flag",
    "is_label_valid",
    "valid_label",
)
RUN_ID_CANDIDATES = ("run_id", "label_run_id")

SECTOR_CANDIDATES = (
    "sector",
    "gics_sector",
    "sector_name",
    "industry_sector",
    "sector_l1",
)
INDUSTRY_CANDIDATES = (
    "industry",
    "gics_industry",
    "industry_group",
    "subindustry",
)
BETA_CANDIDATES = (
    "beta_mkt",
    "market_beta",
    "beta",
    "beta_252d",
    "beta_126d",
)
SIZE_CANDIDATES = (
    "log_mktcap",
    "log_market_cap",
    "mktcap_log",
    "ln_mktcap",
    "market_cap",
    "mktcap",
    "market_cap_usd",
)
LIQUIDITY_CANDIDATES = (
    "liq",
    "liquidity",
    "adv20",
    "adv_usd_20d",
    "avg_dollar_volume_20d",
    "dollar_volume_20d",
    "turnover_20d",
)
VOLATILITY_CANDIDATES = (
    "volatility",
    "vol_20d",
    "realized_vol_20d",
    "realized_volatility_20d",
    "idio_vol_20d",
)
WEIGHT_CANDIDATES = (
    "neutralization_weight",
    "obs_weight",
    "quality_weight",
    "weight",
)

FAIL_N_OBS_TOO_SMALL = "N_OBS_TOO_SMALL"
FAIL_MISSING_ESSENTIAL = "MISSING_ESSENTIAL_EXPOSURES"
FAIL_DESIGN_SINGULAR = "DESIGN_SINGULAR"
FAIL_CONDITION_NUMBER = "CONDITION_NUMBER_TOO_HIGH"
FAIL_LOW_SECTOR_SUPPORT = "LOW_SECTOR_SUPPORT"
FAIL_ESTIMATOR_FAILED = "ESTIMATOR_FAILED"
FAIL_RESIDUAL_COLLAPSED = "RESIDUAL_VARIANCE_COLLAPSED"
FAIL_ROW_DROP_TOO_HIGH = "ROW_DROP_FRACTION_TOO_HIGH"
FAIL_MISSING_TARGET = "MISSING_TARGET_INPUT"

CANONICAL_DEFAULTS = {
    "neutralization_input": "y_fwd_ret_net",
    "neutralization_mode": "sector_beta_size_liquidity",
    "neutralization_estimator": "weighted_ridge",
    "lambda_default": 1.0,
    "min_obs_per_date": 50,
    "cond_max": 1000.0,
    "residual_var_min": 1e-10,
    "missing_exposure_policy": "row_drop_if_inessential_material_fail_if_essential",
    "failure_policy": "invalidate_derived_target",
    "canonical_flag_policy": "strict",
    "policy_version": "1.0.0",
}


class NeutralizedTargetsError(RuntimeError):
    """Base error for target neutralization."""


class ConfigError(NeutralizedTargetsError):
    """Raised when the configuration is invalid."""


class DataContractError(NeutralizedTargetsError):
    """Raised when input tables violate the data contract."""


@dataclass(frozen=True)
class ResolvedColumns:
    label_date_col: str
    label_symbol_col: str
    exposure_date_col: str
    exposure_symbol_col: str
    label_valid_flag_col: str | None = None
    label_run_id_col: str | None = None


@dataclass(frozen=True)
class DesignSpec:
    numeric_factors: tuple[str, ...]
    categorical_factors: tuple[str, ...]
    essential_factors: tuple[str, ...]
    weight_col: str | None = None


@dataclass(frozen=True)
class FitResult:
    success: bool
    failure_reason: str | None
    model_name: str
    lambda_used: float | None
    residuals: pd.Series
    fitted: pd.Series
    r2: float | None
    residual_std: float | None
    residual_var: float | None
    cond_number: float | None
    effective_rank: int | None
    n_obs_candidate: int
    n_obs_used: int
    dropped_row_frac: float
    missing_essential_frac: float
    near_constant_frac: float
    min_sector_support: int | None
    beta_norm: float | None
    effective_df: float | None
    corr_before: dict[str, float]
    corr_after: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NeutralizationConfig:
    labels_path: str
    factor_exposures_path: str
    run_id: str
    output_dir: str
    neutralization_input: str = "y_fwd_ret_net"
    neutralization_mode: str = "sector_beta_size_liquidity"
    neutralization_estimator: str = "weighted_ridge"
    lambda_default: float = 1.0
    lambda_selection: str = "fixed"
    lambda_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    cv_min_obs_per_date: int = 120
    cv_folds: int = 3
    min_obs_per_date: int = 50
    cond_max: float = 1000.0
    residual_var_min: float = 1e-10
    missing_exposure_policy: str = "row_drop_if_inessential_material_fail_if_essential"
    essential_factors_list: tuple[str, ...] = ("sector", "beta_mkt", "log_mktcap", "liq")
    max_row_drop_frac: float = 0.10
    max_missing_essential_frac: float = 0.05
    failure_policy: str = "invalidate_derived_target"
    canonical_flag_policy: str = "strict"
    policy_version: str = "1.0.0"
    min_sector_support: int = 3
    low_support_sector_policy: str = "group_to_other"
    group_to_other_label: str = "OTHER"
    include_volatility_if_available: bool = False
    exposure_weight_col: str | None = None
    date_col: str | None = None
    symbol_col: str | None = None
    exposure_date_col: str | None = None
    exposure_symbol_col: str | None = None
    label_valid_flag_col: str | None = None
    horizon_col: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    factor_manifest_path: str | None = None
    label_manifest_path: str | None = None
    prices_pit_path: str | None = None
    universe_history_path: str | None = None
    emit_date_level_stats: bool = True
    persist_partitioned_output: bool = True
    non_canonical_flag: bool = False
    extra_numeric_factors: tuple[str, ...] = ()
    extra_categorical_factors: tuple[str, ...] = ()
    fit_hc3_se: bool = False

    @staticmethod
    def from_mapping(mapping: Mapping[str, Any]) -> "NeutralizationConfig":
        data = dict(mapping)
        labels_path = _required_str(data.get("labels_path"), "labels_path")
        factor_exposures_path = _required_str(
            data.get("factor_exposures_path"),
            "factor_exposures_path",
        )
        run_id = _required_str(data.get("run_id"), "run_id")
        output_dir = _required_str(data.get("output_dir"), "output_dir")

        lambda_grid_raw = data.get("lambda_grid", (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0))
        if isinstance(lambda_grid_raw, Sequence) and not isinstance(lambda_grid_raw, (str, bytes)):
            lambda_grid = tuple(float(x) for x in lambda_grid_raw)
        else:
            lambda_grid = (float(lambda_grid_raw),)
        if not lambda_grid:
            raise ConfigError("lambda_grid cannot be empty")

        essential_raw = data.get("essential_factors_list", ("sector", "beta_mkt", "log_mktcap", "liq"))
        if not isinstance(essential_raw, Sequence) or isinstance(essential_raw, (str, bytes)):
            raise ConfigError("essential_factors_list must be a sequence of factor names")
        essential = tuple(str(x) for x in essential_raw)

        extra_num_raw = data.get("extra_numeric_factors", ())
        extra_cat_raw = data.get("extra_categorical_factors", ())
        extra_numeric = tuple(str(x) for x in extra_num_raw) if isinstance(extra_num_raw, Sequence) and not isinstance(extra_num_raw, (str, bytes)) else ()
        extra_categorical = tuple(str(x) for x in extra_cat_raw) if isinstance(extra_cat_raw, Sequence) and not isinstance(extra_cat_raw, (str, bytes)) else ()

        config = NeutralizationConfig(
            labels_path=labels_path,
            factor_exposures_path=factor_exposures_path,
            run_id=run_id,
            output_dir=output_dir,
            neutralization_input=str(data.get("neutralization_input", CANONICAL_DEFAULTS["neutralization_input"])),
            neutralization_mode=str(data.get("neutralization_mode", CANONICAL_DEFAULTS["neutralization_mode"])),
            neutralization_estimator=str(data.get("neutralization_estimator", CANONICAL_DEFAULTS["neutralization_estimator"])),
            lambda_default=float(data.get("lambda_default", CANONICAL_DEFAULTS["lambda_default"])),
            lambda_selection=str(data.get("lambda_selection", "fixed")),
            lambda_grid=lambda_grid,
            cv_min_obs_per_date=int(data.get("cv_min_obs_per_date", 120)),
            cv_folds=int(data.get("cv_folds", 3)),
            min_obs_per_date=int(data.get("min_obs_per_date", CANONICAL_DEFAULTS["min_obs_per_date"])),
            cond_max=float(data.get("cond_max", CANONICAL_DEFAULTS["cond_max"])),
            residual_var_min=float(data.get("residual_var_min", CANONICAL_DEFAULTS["residual_var_min"])),
            missing_exposure_policy=str(data.get("missing_exposure_policy", CANONICAL_DEFAULTS["missing_exposure_policy"])),
            essential_factors_list=essential,
            max_row_drop_frac=float(data.get("max_row_drop_frac", 0.10)),
            max_missing_essential_frac=float(data.get("max_missing_essential_frac", 0.05)),
            failure_policy=str(data.get("failure_policy", CANONICAL_DEFAULTS["failure_policy"])),
            canonical_flag_policy=str(data.get("canonical_flag_policy", CANONICAL_DEFAULTS["canonical_flag_policy"])),
            policy_version=str(data.get("policy_version", CANONICAL_DEFAULTS["policy_version"])),
            min_sector_support=int(data.get("min_sector_support", 3)),
            low_support_sector_policy=str(data.get("low_support_sector_policy", "group_to_other")),
            group_to_other_label=str(data.get("group_to_other_label", "OTHER")),
            include_volatility_if_available=bool(data.get("include_volatility_if_available", False)),
            exposure_weight_col=_optional_str(data.get("exposure_weight_col")),
            date_col=_optional_str(data.get("date_col")),
            symbol_col=_optional_str(data.get("symbol_col")),
            exposure_date_col=_optional_str(data.get("exposure_date_col")),
            exposure_symbol_col=_optional_str(data.get("exposure_symbol_col")),
            label_valid_flag_col=_optional_str(data.get("label_valid_flag_col")),
            horizon_col=_optional_str(data.get("horizon_col")),
            start_date=_optional_str(data.get("start_date")),
            end_date=_optional_str(data.get("end_date")),
            factor_manifest_path=_optional_str(data.get("factor_manifest_path")),
            label_manifest_path=_optional_str(data.get("label_manifest_path")),
            prices_pit_path=_optional_str(data.get("prices_pit_path")),
            universe_history_path=_optional_str(data.get("universe_history_path")),
            emit_date_level_stats=bool(data.get("emit_date_level_stats", True)),
            persist_partitioned_output=bool(data.get("persist_partitioned_output", True)),
            extra_numeric_factors=extra_numeric,
            extra_categorical_factors=extra_categorical,
            fit_hc3_se=bool(data.get("fit_hc3_se", False)),
        )
        _validate_config(config)
        object.__setattr__(config, "non_canonical_flag", _detect_non_canonical(config))
        return config

    def to_serializable(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def _validate_config(config: NeutralizationConfig) -> None:
    if config.lambda_default < 0:
        raise ConfigError("lambda_default must be >= 0")
    if config.lambda_selection not in {"fixed", "cv", "grid_search_cv"}:
        raise ConfigError("lambda_selection must be one of fixed, cv, grid_search_cv")
    if config.cv_folds < 2:
        raise ConfigError("cv_folds must be >= 2")
    if config.min_obs_per_date <= 0:
        raise ConfigError("min_obs_per_date must be > 0")
    if config.cond_max <= 0:
        raise ConfigError("cond_max must be > 0")
    if config.residual_var_min < 0:
        raise ConfigError("residual_var_min must be >= 0")
    if not 0 <= config.max_row_drop_frac <= 1:
        raise ConfigError("max_row_drop_frac must be in [0, 1]")
    if not 0 <= config.max_missing_essential_frac <= 1:
        raise ConfigError("max_missing_essential_frac must be in [0, 1]")
    if config.failure_policy != "invalidate_derived_target":
        raise ConfigError("Only failure_policy='invalidate_derived_target' is supported")
    if config.missing_exposure_policy not in {
        "row_drop_if_inessential_material_fail_if_essential",
        "material_missing_fails_date",
    }:
        raise ConfigError(
            "missing_exposure_policy must be one of "
            "row_drop_if_inessential_material_fail_if_essential, material_missing_fails_date"
        )
    if config.neutralization_estimator not in {
        "ols",
        "ridge",
        "weighted_ridge",
        "huber",
        "sector_demean_plus_factor_residual",
    }:
        raise ConfigError(
            "neutralization_estimator must be one of "
            "ols, ridge, weighted_ridge, huber, sector_demean_plus_factor_residual"
        )
    if config.neutralization_mode not in {
        "sector_beta_size_liquidity",
        "sector_beta_size_liquidity_volatility",
        "custom",
    }:
        raise ConfigError(
            "neutralization_mode must be sector_beta_size_liquidity, "
            "sector_beta_size_liquidity_volatility, or custom"
        )


def _detect_non_canonical(config: NeutralizationConfig) -> bool:
    return any(
        [
            config.neutralization_input != CANONICAL_DEFAULTS["neutralization_input"],
            config.neutralization_mode != CANONICAL_DEFAULTS["neutralization_mode"],
            config.neutralization_estimator != CANONICAL_DEFAULTS["neutralization_estimator"],
            not math.isclose(config.lambda_default, CANONICAL_DEFAULTS["lambda_default"]),
            config.min_obs_per_date != CANONICAL_DEFAULTS["min_obs_per_date"],
            not math.isclose(config.cond_max, CANONICAL_DEFAULTS["cond_max"]),
            not math.isclose(config.residual_var_min, CANONICAL_DEFAULTS["residual_var_min"]),
            config.missing_exposure_policy != CANONICAL_DEFAULTS["missing_exposure_policy"],
            config.failure_policy != CANONICAL_DEFAULTS["failure_policy"],
            config.canonical_flag_policy != CANONICAL_DEFAULTS["canonical_flag_policy"],
            config.policy_version != CANONICAL_DEFAULTS["policy_version"],
            config.lambda_selection != "fixed",
            config.include_volatility_if_available,
            bool(config.extra_numeric_factors),
            bool(config.extra_categorical_factors),
            config.fit_hc3_se,
        ]
    )


def _required_str(value: Any, field_name: str) -> str:
    text = _optional_str(value)
    if text is None:
        raise ConfigError(f"{field_name} is required")
    return text


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _load_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("pyyaml is required to read YAML config files")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return dict(payload or {})
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return dict(payload or {})
    raise ValueError(f"Unsupported config format: {path}")


def _load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    if suffix == ".pkl":
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported table format: {path}")


def _safe_hash_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _standardize_datetime(series: pd.Series, colname: str) -> pd.Series:
    out = pd.to_datetime(series, errors="coerce")
    if out.isna().any():
        bad = int(out.isna().sum())
        raise DataContractError(f"Column '{colname}' has {bad} invalid datetime values")
    return out.dt.normalize()


def _resolve_column(df: pd.DataFrame, explicit: str | None, candidates: Sequence[str], field_name: str) -> str:
    if explicit is not None:
        if explicit not in df.columns:
            raise DataContractError(f"Explicit {field_name}='{explicit}' not found")
        return explicit
    for col in candidates:
        if col in df.columns:
            return col
    raise DataContractError(
        f"Could not resolve {field_name}. Tried explicit value and candidates {list(candidates)}"
    )


def _resolve_optional_column(df: pd.DataFrame, explicit: str | None, candidates: Sequence[str]) -> str | None:
    if explicit is not None:
        if explicit not in df.columns:
            raise DataContractError(f"Explicit optional column '{explicit}' not found")
        return explicit
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _ensure_unique_pk(df: pd.DataFrame, date_col: str, symbol_col: str, table_name: str) -> None:
    dup = int(df.duplicated([date_col, symbol_col]).sum())
    if dup:
        raise DataContractError(
            f"{table_name} has {dup} duplicate rows on primary key ({date_col}, {symbol_col})"
        )


def _resolve_columns(
    labels_df: pd.DataFrame,
    exposures_df: pd.DataFrame,
    config: NeutralizationConfig,
) -> ResolvedColumns:
    label_date_col = _resolve_column(labels_df, config.date_col, DATE_CANDIDATES, "label date column")
    label_symbol_col = _resolve_column(labels_df, config.symbol_col, SYMBOL_CANDIDATES, "label symbol column")
    exposure_date_col = _resolve_column(
        exposures_df,
        config.exposure_date_col,
        DATE_CANDIDATES,
        "exposure date column",
    )
    exposure_symbol_col = _resolve_column(
        exposures_df,
        config.exposure_symbol_col,
        SYMBOL_CANDIDATES,
        "exposure symbol column",
    )
    label_valid_flag_col = _resolve_optional_column(
        labels_df,
        config.label_valid_flag_col,
        LABEL_VALID_FLAG_CANDIDATES,
    )
    label_run_id_col = _resolve_optional_column(labels_df, None, RUN_ID_CANDIDATES)
    return ResolvedColumns(
        label_date_col=label_date_col,
        label_symbol_col=label_symbol_col,
        exposure_date_col=exposure_date_col,
        exposure_symbol_col=exposure_symbol_col,
        label_valid_flag_col=label_valid_flag_col,
        label_run_id_col=label_run_id_col,
    )


def _parse_horizon_from_colname(col: str, prefix: str) -> int | None:
    patt = re.compile(rf"^{re.escape(prefix)}_(\d+)d$")
    m = patt.match(col)
    if m:
        return int(m.group(1))
    return None


def _candidate_target_columns(labels_df: pd.DataFrame, neutralization_input: str) -> dict[int, str]:
    candidates: dict[int, str] = {}
    for col in labels_df.columns:
        horizon = _parse_horizon_from_colname(col, neutralization_input)
        if horizon is not None:
            candidates[horizon] = col
    return dict(sorted(candidates.items()))


def _prepare_labels_long(
    labels_df: pd.DataFrame,
    resolved: ResolvedColumns,
    config: NeutralizationConfig,
) -> tuple[pd.DataFrame, list[int]]:
    labels = labels_df.copy()
    labels[resolved.label_date_col] = _standardize_datetime(labels[resolved.label_date_col], resolved.label_date_col)
    labels[resolved.label_symbol_col] = labels[resolved.label_symbol_col].astype(str)

    if config.start_date is not None:
        start = pd.Timestamp(config.start_date).normalize()
        labels = labels.loc[labels[resolved.label_date_col] >= start].copy()
    if config.end_date is not None:
        end = pd.Timestamp(config.end_date).normalize()
        labels = labels.loc[labels[resolved.label_date_col] <= end].copy()

    if labels.empty:
        raise DataContractError("Labels table is empty after date filtering")

    horizon_col = config.horizon_col if config.horizon_col in labels.columns else None
    if horizon_col is None:
        horizon_col = _resolve_optional_column(labels, None, HORIZON_CANDIDATES)

    if horizon_col and config.neutralization_input in labels.columns:
        long_df = labels.copy()
        long_df["horizon"] = pd.to_numeric(long_df[horizon_col], errors="coerce")
        if long_df["horizon"].isna().any():
            raise DataContractError(f"Horizon column '{horizon_col}' contains non-numeric values")
        long_df["horizon"] = long_df["horizon"].astype(int)
        long_df["y_base"] = pd.to_numeric(long_df[config.neutralization_input], errors="coerce")
        long_df["source_target_col"] = config.neutralization_input
        if resolved.label_valid_flag_col is not None:
            long_df["label_valid_flag_source"] = long_df[resolved.label_valid_flag_col].astype(bool)
        else:
            long_df["label_valid_flag_source"] = long_df["y_base"].notna()
        horizons = sorted(long_df["horizon"].dropna().astype(int).unique().tolist())
        return long_df, horizons

    target_cols = _candidate_target_columns(labels, config.neutralization_input)
    if not target_cols:
        available = [c for c in labels.columns if c.startswith(config.neutralization_input)]
        if config.neutralization_input in labels.columns:
            raise DataContractError(
                f"Found base column '{config.neutralization_input}' but no horizon column. "
                "Provide horizon_col in config or use wide y_*_Xd labels."
            )
        raise DataContractError(
            f"Could not find any columns for neutralization_input='{config.neutralization_input}'. "
            f"Available prefixed columns: {available[:10]}"
        )

    id_vars = [c for c in labels.columns if c not in target_cols.values()]
    pieces: list[pd.DataFrame] = []
    for horizon, target_col in target_cols.items():
        cols = list(id_vars)
        wide_slice = labels.loc[:, cols].copy()
        wide_slice["horizon"] = int(horizon)
        wide_slice["y_base"] = pd.to_numeric(labels[target_col], errors="coerce")
        wide_slice["source_target_col"] = target_col
        if f"label_valid_flag_{horizon}d" in labels.columns:
            wide_slice["label_valid_flag_source"] = labels[f"label_valid_flag_{horizon}d"].astype(bool)
        elif resolved.label_valid_flag_col is not None:
            wide_slice["label_valid_flag_source"] = labels[resolved.label_valid_flag_col].astype(bool)
        else:
            wide_slice["label_valid_flag_source"] = wide_slice["y_base"].notna()
        wide_slice["event_start"] = labels[f"event_start_{horizon}d"] if f"event_start_{horizon}d" in labels.columns else pd.NaT
        wide_slice["event_end"] = labels[f"event_end_{horizon}d"] if f"event_end_{horizon}d" in labels.columns else pd.NaT
        pieces.append(wide_slice)
    long_df = pd.concat(pieces, axis=0, ignore_index=True)
    horizons = sorted(target_cols)
    return long_df, horizons


def _prepare_exposures(
    exposures_df: pd.DataFrame,
    resolved: ResolvedColumns,
    config: NeutralizationConfig,
) -> pd.DataFrame:
    exposures = exposures_df.copy()
    exposures[resolved.exposure_date_col] = _standardize_datetime(
        exposures[resolved.exposure_date_col],
        resolved.exposure_date_col,
    )
    exposures[resolved.exposure_symbol_col] = exposures[resolved.exposure_symbol_col].astype(str)
    exposures = exposures.sort_values([resolved.exposure_symbol_col, resolved.exposure_date_col]).reset_index(drop=True)
    _ensure_unique_pk(exposures, resolved.exposure_date_col, resolved.exposure_symbol_col, "factor_exposures")

    rename_map: dict[str, str] = {}
    sector_col = _resolve_optional_column(exposures, None, SECTOR_CANDIDATES)
    industry_col = _resolve_optional_column(exposures, None, INDUSTRY_CANDIDATES)
    beta_col = _resolve_optional_column(exposures, None, BETA_CANDIDATES)
    size_col = _resolve_optional_column(exposures, None, SIZE_CANDIDATES)
    liq_col = _resolve_optional_column(exposures, None, LIQUIDITY_CANDIDATES)
    vol_col = _resolve_optional_column(exposures, None, VOLATILITY_CANDIDATES)
    weight_col = _resolve_optional_column(exposures, config.exposure_weight_col, WEIGHT_CANDIDATES)

    if sector_col is not None:
        rename_map[sector_col] = "sector"
    elif industry_col is not None:
        rename_map[industry_col] = "sector"
    if beta_col is not None:
        rename_map[beta_col] = "beta_mkt"
    if size_col is not None:
        rename_map[size_col] = "size_raw"
    if liq_col is not None:
        rename_map[liq_col] = "liq_raw"
    if vol_col is not None:
        rename_map[vol_col] = "volatility"
    if weight_col is not None:
        rename_map[weight_col] = "neutralization_weight"

    exposures = exposures.rename(columns=rename_map)
    if "size_raw" in exposures.columns:
        size_series = pd.to_numeric(exposures["size_raw"], errors="coerce")
        if "log_mktcap" in exposures.columns:
            exposures["log_mktcap"] = pd.to_numeric(exposures["log_mktcap"], errors="coerce")
        else:
            size_positive = size_series.where(size_series > 0)
            exposures["log_mktcap"] = np.log(size_positive)
    if "liq_raw" in exposures.columns:
        liq_series = pd.to_numeric(exposures["liq_raw"], errors="coerce")
        if "liq" not in exposures.columns:
            exposures["liq"] = np.log1p(liq_series.where(liq_series >= 0))
        else:
            exposures["liq"] = pd.to_numeric(exposures["liq"], errors="coerce")
    if "beta_mkt" in exposures.columns:
        exposures["beta_mkt"] = pd.to_numeric(exposures["beta_mkt"], errors="coerce")
    if "volatility" in exposures.columns:
        exposures["volatility"] = pd.to_numeric(exposures["volatility"], errors="coerce")
    if "neutralization_weight" in exposures.columns:
        exposures["neutralization_weight"] = pd.to_numeric(
            exposures["neutralization_weight"], errors="coerce"
        )
    return exposures


def _align_labels_and_exposures(
    labels_long: pd.DataFrame,
    exposures: pd.DataFrame,
    resolved: ResolvedColumns,
) -> pd.DataFrame:
    labels = labels_long.copy()
    labels = labels.sort_values([resolved.label_symbol_col, resolved.label_date_col]).reset_index(drop=True)
    exposures = exposures.sort_values([resolved.exposure_symbol_col, resolved.exposure_date_col]).reset_index(drop=True)

    pieces: list[pd.DataFrame] = []
    exp_groups = {
        str(symbol): grp.sort_values(resolved.exposure_date_col).reset_index(drop=True)
        for symbol, grp in exposures.groupby(resolved.exposure_symbol_col, sort=False)
    }
    for symbol, lbl_grp in labels.groupby(resolved.label_symbol_col, sort=False):
        rhs = exp_groups.get(str(symbol))
        if rhs is None or rhs.empty:
            tmp = lbl_grp.copy()
            for col in exposures.columns:
                if col not in tmp.columns:
                    tmp[col] = np.nan
            pieces.append(tmp)
            continue
        lhs = lbl_grp.sort_values(resolved.label_date_col).reset_index(drop=True)
        merged_symbol = pd.merge_asof(
            lhs,
            rhs,
            left_on=resolved.label_date_col,
            right_on=resolved.exposure_date_col,
            direction="backward",
            allow_exact_matches=True,
        )
        left_symbol_x = f"{resolved.label_symbol_col}_x"
        right_symbol_y = f"{resolved.exposure_symbol_col}_y"
        if resolved.label_symbol_col not in merged_symbol.columns and left_symbol_x in merged_symbol.columns:
            merged_symbol = merged_symbol.rename(columns={left_symbol_x: resolved.label_symbol_col})
        if right_symbol_y in merged_symbol.columns and resolved.exposure_symbol_col != resolved.label_symbol_col:
            merged_symbol = merged_symbol.rename(columns={right_symbol_y: resolved.exposure_symbol_col})
        elif right_symbol_y in merged_symbol.columns:
            merged_symbol = merged_symbol.drop(columns=[right_symbol_y])
        pieces.append(merged_symbol)
    merged = pd.concat(pieces, axis=0, ignore_index=True) if pieces else pd.DataFrame()
    if merged.empty:
        raise DataContractError("Aligned labels/exposures table is empty")
    merged["exposure_staleness_days"] = (
        merged[resolved.label_date_col] - merged[resolved.exposure_date_col]
    ).dt.days
    return merged


def _build_design_spec(df: pd.DataFrame, config: NeutralizationConfig) -> DesignSpec:
    numeric: list[str] = []
    categorical: list[str] = []
    essential = list(config.essential_factors_list)

    if config.neutralization_mode in {"sector_beta_size_liquidity", "sector_beta_size_liquidity_volatility"}:
        if "sector" in df.columns:
            categorical.append("sector")
        if "beta_mkt" in df.columns:
            numeric.append("beta_mkt")
        if "log_mktcap" in df.columns:
            numeric.append("log_mktcap")
        if "liq" in df.columns:
            numeric.append("liq")
        if config.neutralization_mode == "sector_beta_size_liquidity_volatility" or config.include_volatility_if_available:
            if "volatility" in df.columns:
                numeric.append("volatility")
                if config.neutralization_mode == "sector_beta_size_liquidity_volatility":
                    if "volatility" not in essential:
                        essential.append("volatility")
    elif config.neutralization_mode == "custom":
        numeric.extend(config.extra_numeric_factors)
        categorical.extend(config.extra_categorical_factors)

    for col in config.extra_numeric_factors:
        if col not in numeric and col in df.columns:
            numeric.append(col)
    for col in config.extra_categorical_factors:
        if col not in categorical and col in df.columns:
            categorical.append(col)

    weight_col = "neutralization_weight" if "neutralization_weight" in df.columns else None
    if config.exposure_weight_col and config.exposure_weight_col in df.columns:
        weight_col = config.exposure_weight_col

    if not numeric and not categorical:
        raise DataContractError(
            "No usable exposures found for the requested neutralization_mode. "
            "Check factor_exposures_path or configure custom factors explicitly."
        )
    return DesignSpec(
        numeric_factors=tuple(numeric),
        categorical_factors=tuple(categorical),
        essential_factors=tuple(essential),
        weight_col=weight_col,
    )


def _canonicalize_sector(series: pd.Series, min_support: int, other_label: str) -> tuple[pd.Series, int]:
    clean = (
        series.astype("string")
        .fillna("MISSING")
        .replace({"": "MISSING", "nan": "MISSING", "<NA>": "MISSING", "None": "MISSING"})
    )
    counts = clean.value_counts(dropna=False)
    low_support = counts[counts < min_support].index
    out = clean.where(~clean.isin(low_support), other_label)
    min_count = int(out.value_counts(dropna=False).min()) if not out.empty else 0
    return out.astype(str), min_count


@dataclass
class DesignBundle:
    X: pd.DataFrame
    y: pd.Series
    weights: pd.Series | None
    row_mask: pd.Series
    sector_support_min: int | None
    dropped_row_frac: float
    missing_essential_frac: float
    near_constant_frac: float
    diagnostics: dict[str, Any]


def _prepare_design_matrix(
    group: pd.DataFrame,
    design_spec: DesignSpec,
    config: NeutralizationConfig,
) -> tuple[DesignBundle | None, str | None]:
    df = group.copy()
    candidate_count = len(df)
    if candidate_count < config.min_obs_per_date:
        return None, FAIL_N_OBS_TOO_SMALL

    y = pd.to_numeric(df["y_base"], errors="coerce")
    valid_base = y.notna() & df["label_valid_flag_source"].fillna(False).astype(bool)
    if valid_base.sum() < config.min_obs_per_date:
        return None, FAIL_MISSING_TARGET if valid_base.sum() == 0 else FAIL_N_OBS_TOO_SMALL
    df = df.loc[valid_base].copy()
    y = y.loc[valid_base].astype(float)

    diagnostics: dict[str, Any] = {}
    missing_essential_frames: list[pd.Series] = []
    missing_any_frames: list[pd.Series] = []
    numeric_parts: list[pd.Series] = []
    categorical_parts: list[pd.DataFrame] = []
    near_constant_cols = 0
    total_design_cols = 0
    sector_support_min: int | None = None

    for factor in design_spec.numeric_factors:
        if factor not in df.columns:
            if factor in design_spec.essential_factors:
                return None, FAIL_MISSING_ESSENTIAL
            continue
        raw = pd.to_numeric(df[factor], errors="coerce")
        missing = raw.isna()
        missing_any_frames.append(missing)
        if factor in design_spec.essential_factors:
            missing_essential_frames.append(missing)
        total_design_cols += 1
        non_missing = raw.dropna()
        if non_missing.empty:
            if factor in design_spec.essential_factors:
                return None, FAIL_MISSING_ESSENTIAL
            continue
        median = float(non_missing.median())
        filled = raw.fillna(median)
        std = float(filled.std(ddof=0))
        if std <= 1e-12:
            near_constant_cols += 1
            filled = filled * 0.0
        else:
            filled = (filled - float(filled.mean())) / std
        numeric_parts.append(filled.rename(factor))

    for factor in design_spec.categorical_factors:
        if factor not in df.columns:
            if factor in design_spec.essential_factors:
                return None, FAIL_MISSING_ESSENTIAL
            continue
        raw = df[factor]
        missing = raw.isna()
        missing_any_frames.append(missing)
        if factor in design_spec.essential_factors:
            missing_essential_frames.append(missing)
        total_design_cols += 1
        if factor == "sector":
            if config.low_support_sector_policy == "group_to_other":
                sector, min_support = _canonicalize_sector(
                    raw,
                    config.min_sector_support,
                    config.group_to_other_label,
                )
                sector_support_min = min_support
                if min_support < config.min_sector_support:
                    return None, FAIL_LOW_SECTOR_SUPPORT
                dummies = pd.get_dummies(sector, prefix="sector", drop_first=True, dtype=float)
            else:
                sector = raw.astype("string").fillna("MISSING").astype(str)
                counts = sector.value_counts(dropna=False)
                min_count = int(counts.min()) if not counts.empty else 0
                sector_support_min = min_count
                if min_count < config.min_sector_support:
                    return None, FAIL_LOW_SECTOR_SUPPORT
                dummies = pd.get_dummies(sector, prefix="sector", drop_first=True, dtype=float)
        else:
            cat = raw.astype("string").fillna("MISSING").astype(str)
            dummies = pd.get_dummies(cat, prefix=factor, drop_first=True, dtype=float)
        categorical_parts.append(dummies)

    if missing_any_frames:
        missing_any = pd.concat(missing_any_frames, axis=1).any(axis=1)
    else:
        missing_any = pd.Series(False, index=df.index)
    if missing_essential_frames:
        missing_essential = pd.concat(missing_essential_frames, axis=1).any(axis=1)
    else:
        missing_essential = pd.Series(False, index=df.index)

    missing_essential_frac = float(missing_essential.mean()) if len(missing_essential) else 0.0
    if missing_essential_frac > config.max_missing_essential_frac:
        return None, FAIL_MISSING_ESSENTIAL

    keep_mask = ~missing_any
    dropped_row_frac = 1.0 - float(keep_mask.mean()) if len(keep_mask) else 1.0
    if dropped_row_frac > config.max_row_drop_frac:
        return None, FAIL_ROW_DROP_TOO_HIGH

    design_parts: list[pd.DataFrame | pd.Series] = [pd.Series(1.0, index=df.index, name="intercept")]
    design_parts.extend(numeric_parts)
    design_parts.extend(categorical_parts)
    X = pd.concat(design_parts, axis=1)
    X = X.loc[keep_mask].copy()
    y_keep = y.loc[keep_mask].copy()
    if len(y_keep) < config.min_obs_per_date:
        return None, FAIL_N_OBS_TOO_SMALL

    weights: pd.Series | None = None
    if design_spec.weight_col is not None and design_spec.weight_col in df.columns:
        weights = pd.to_numeric(df[design_spec.weight_col], errors="coerce").fillna(1.0).clip(lower=1e-12)
        weights = weights.loc[keep_mask]

    X = X.astype(float)
    if not np.isfinite(X.to_numpy(dtype=float)).all() or not np.isfinite(y_keep.to_numpy(dtype=float)).all():
        return None, FAIL_ESTIMATOR_FAILED

    rank = int(np.linalg.matrix_rank(X.to_numpy(dtype=float)))
    if rank < X.shape[1]:
        return None, FAIL_DESIGN_SINGULAR

    gram = X.to_numpy(dtype=float).T @ X.to_numpy(dtype=float)
    cond = float(np.linalg.cond(gram)) if gram.size else math.nan
    diagnostics["cond_number_pre_fit"] = cond
    if np.isfinite(cond) and cond > config.cond_max and config.neutralization_estimator == "ols":
        return None, FAIL_CONDITION_NUMBER

    near_constant_frac = float(near_constant_cols / total_design_cols) if total_design_cols else 0.0
    diagnostics["rank_pre_fit"] = rank
    diagnostics["n_columns"] = int(X.shape[1])

    return DesignBundle(
        X=X,
        y=y_keep,
        weights=weights,
        row_mask=keep_mask.reindex(group.index, fill_value=False),
        sector_support_min=sector_support_min,
        dropped_row_frac=dropped_row_frac,
        missing_essential_frac=missing_essential_frac,
        near_constant_frac=near_constant_frac,
        diagnostics=diagnostics,
    ), None


def _stable_fold_ids(index: pd.Index, n_folds: int) -> np.ndarray:
    ids = np.empty(len(index), dtype=int)
    for i, idx in enumerate(index):
        h = hashlib.sha256(str(idx).encode("utf-8")).hexdigest()
        ids[i] = int(h[:8], 16) % n_folds
    return ids


def _weighted_ridge_beta(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    lmbda: float,
    penalize_intercept: bool = False,
) -> tuple[np.ndarray, float]:
    w = np.clip(weights, 1e-12, None)
    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    gram = Xw.T @ Xw
    pen = np.eye(gram.shape[0]) * float(lmbda)
    if not penalize_intercept and gram.shape[0] > 0:
        pen[0, 0] = 0.0
    lhs = gram + pen
    rhs = Xw.T @ yw
    beta = np.linalg.solve(lhs, rhs)
    cond = float(np.linalg.cond(lhs)) if lhs.size else math.nan
    return beta, cond


def _weighted_ols_beta(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float]:
    return _weighted_ridge_beta(X, y, weights, lmbda=0.0)


def _choose_lambda_cv(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    grid: Sequence[float],
    n_folds: int,
) -> float:
    if len(y) < max(2 * n_folds, 12):
        return float(grid[len(grid) // 2])
    fold_ids = _stable_fold_ids(pd.Index(range(len(y))), n_folds)
    best_lambda = float(grid[0])
    best_score = math.inf
    for lmbda in grid:
        fold_scores: list[float] = []
        for fold in range(n_folds):
            train = fold_ids != fold
            test = fold_ids == fold
            if train.sum() <= X.shape[1] or test.sum() == 0:
                continue
            try:
                beta, _ = _weighted_ridge_beta(X[train], y[train], weights[train], lmbda)
            except np.linalg.LinAlgError:
                continue
            pred = X[test] @ beta
            err = (y[test] - pred) ** 2
            fold_scores.append(float(np.average(err, weights=weights[test])))
        if not fold_scores:
            continue
        score = float(np.mean(fold_scores))
        if score < best_score:
            best_score = score
            best_lambda = float(lmbda)
    return best_lambda


def _fit_huber(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    lmbda: float,
    max_iter: int = 25,
    c: float = 1.345,
) -> tuple[np.ndarray, float]:
    beta, cond = _weighted_ridge_beta(X, y, weights, lmbda)
    for _ in range(max_iter):
        resid = y - X @ beta
        scale = float(np.median(np.abs(resid - np.median(resid))) / 0.6745) if len(resid) else 0.0
        if not np.isfinite(scale) or scale <= 1e-12:
            break
        scaled = np.abs(resid) / scale
        psi_over_r = np.where(scaled <= c, 1.0, c / np.maximum(scaled, 1e-12))
        eff_weights = weights * psi_over_r
        new_beta, cond = _weighted_ridge_beta(X, y, eff_weights, lmbda)
        if np.max(np.abs(new_beta - beta)) < 1e-8:
            beta = new_beta
            break
        beta = new_beta
    return beta, cond


def _effective_df_ridge(X: np.ndarray, weights: np.ndarray, lmbda: float) -> float | None:
    try:
        sqrt_w = np.sqrt(np.clip(weights, 1e-12, None))
        Xw = X * sqrt_w[:, None]
        gram = Xw.T @ Xw
        pen = np.eye(gram.shape[0]) * float(lmbda)
        if pen.shape[0] > 0:
            pen[0, 0] = 0.0
        inv = np.linalg.inv(gram + pen)
        hat = Xw @ inv @ Xw.T
        return float(np.trace(hat))
    except Exception:
        return None


def _safe_corr(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    x = pd.Series(a, dtype=float)
    y = pd.Series(b, dtype=float)
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return math.nan
    xs = x.loc[mask]
    ys = y.loc[mask]
    if float(xs.std(ddof=0)) <= 1e-12 or float(ys.std(ddof=0)) <= 1e-12:
        return math.nan
    return float(xs.corr(ys))


def _fit_one_group(
    group: pd.DataFrame,
    design_spec: DesignSpec,
    config: NeutralizationConfig,
) -> FitResult:
    prepared, failure = _prepare_design_matrix(group, design_spec, config)
    y_full = pd.to_numeric(group["y_base"], errors="coerce")
    empty_series = pd.Series(np.nan, index=group.index, dtype=float)
    if prepared is None:
        return FitResult(
            success=False,
            failure_reason=failure,
            model_name=config.neutralization_estimator,
            lambda_used=None,
            residuals=empty_series.copy(),
            fitted=empty_series.copy(),
            r2=None,
            residual_std=None,
            residual_var=None,
            cond_number=None,
            effective_rank=None,
            n_obs_candidate=int(len(group)),
            n_obs_used=0,
            dropped_row_frac=1.0,
            missing_essential_frac=math.nan,
            near_constant_frac=math.nan,
            min_sector_support=None,
            beta_norm=None,
            effective_df=None,
            corr_before={},
            corr_after={},
            diagnostics={},
        )

    Xdf = prepared.X
    y = prepared.y
    w = prepared.weights if prepared.weights is not None else pd.Series(1.0, index=Xdf.index)
    X = Xdf.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    weights = pd.to_numeric(w, errors="coerce").fillna(1.0).clip(lower=1e-12).to_numpy(dtype=float)

    lambda_used: float
    if config.neutralization_estimator in {"ridge", "weighted_ridge", "huber", "sector_demean_plus_factor_residual"}:
        if config.lambda_selection in {"cv", "grid_search_cv"} and len(yv) >= config.cv_min_obs_per_date:
            lambda_used = _choose_lambda_cv(X, yv, weights, config.lambda_grid, config.cv_folds)
        else:
            lambda_used = float(config.lambda_default)
    else:
        lambda_used = 0.0

    try:
        if config.neutralization_estimator == "ols":
            beta, cond = _weighted_ols_beta(X, yv, np.ones_like(weights))
            effective_df = None
        elif config.neutralization_estimator == "ridge":
            beta, cond = _weighted_ridge_beta(X, yv, np.ones_like(weights), lambda_used)
            effective_df = _effective_df_ridge(X, np.ones_like(weights), lambda_used)
        elif config.neutralization_estimator == "weighted_ridge":
            beta, cond = _weighted_ridge_beta(X, yv, weights, lambda_used)
            effective_df = _effective_df_ridge(X, weights, lambda_used)
        elif config.neutralization_estimator == "huber":
            beta, cond = _fit_huber(X, yv, weights, lambda_used)
            effective_df = _effective_df_ridge(X, weights, lambda_used)
        elif config.neutralization_estimator == "sector_demean_plus_factor_residual":
            demeaned = y.copy()
            sector_cols = [c for c in Xdf.columns if c.startswith("sector_")]
            if "sector" in group.columns:
                sector_series = group.loc[Xdf.index, "sector"].astype(str)
                demeaned = demeaned - demeaned.groupby(sector_series).transform("mean")
            factor_cols = [c for c in Xdf.columns if c not in ["intercept", *sector_cols]]
            if not factor_cols:
                beta = np.array([float(demeaned.mean())])
                cond = 1.0
                pred = np.full(len(demeaned), float(demeaned.mean()))
                resid_valid = demeaned.to_numpy(dtype=float) - pred
                residuals = empty_series.copy()
                fitted = empty_series.copy()
                residuals.loc[Xdf.index] = resid_valid
                fitted.loc[Xdf.index] = pred
                r2 = 1.0 - float(np.sum(resid_valid ** 2)) / float(np.sum((demeaned - float(demeaned.mean())) ** 2)) if len(demeaned) > 1 and float(np.sum((demeaned - float(demeaned.mean())) ** 2)) > 0 else math.nan
                res_var = float(np.var(resid_valid, ddof=1)) if len(resid_valid) > 1 else 0.0
                return FitResult(
                    success=res_var >= config.residual_var_min,
                    failure_reason=None if res_var >= config.residual_var_min else FAIL_RESIDUAL_COLLAPSED,
                    model_name=config.neutralization_estimator,
                    lambda_used=lambda_used,
                    residuals=residuals,
                    fitted=fitted,
                    r2=r2,
                    residual_std=math.sqrt(res_var) if res_var >= 0 else math.nan,
                    residual_var=res_var,
                    cond_number=cond,
                    effective_rank=1,
                    n_obs_candidate=int(len(group)),
                    n_obs_used=int(len(Xdf)),
                    dropped_row_frac=prepared.dropped_row_frac,
                    missing_essential_frac=prepared.missing_essential_frac,
                    near_constant_frac=prepared.near_constant_frac,
                    min_sector_support=prepared.sector_support_min,
                    beta_norm=float(np.linalg.norm(beta)),
                    effective_df=None,
                    corr_before={},
                    corr_after={},
                    diagnostics=prepared.diagnostics,
                )
            X_num = Xdf.loc[:, ["intercept", *factor_cols]].to_numpy(dtype=float)
            beta, cond = _weighted_ridge_beta(X_num, demeaned.to_numpy(dtype=float), weights, lambda_used)
            effective_df = _effective_df_ridge(X_num, weights, lambda_used)
            X = X_num
            yv = demeaned.to_numpy(dtype=float)
        else:
            raise ConfigError(f"Unsupported estimator: {config.neutralization_estimator}")
    except np.linalg.LinAlgError:
        return FitResult(
            success=False,
            failure_reason=FAIL_ESTIMATOR_FAILED,
            model_name=config.neutralization_estimator,
            lambda_used=lambda_used,
            residuals=empty_series.copy(),
            fitted=empty_series.copy(),
            r2=None,
            residual_std=None,
            residual_var=None,
            cond_number=None,
            effective_rank=None,
            n_obs_candidate=int(len(group)),
            n_obs_used=int(len(Xdf)),
            dropped_row_frac=prepared.dropped_row_frac,
            missing_essential_frac=prepared.missing_essential_frac,
            near_constant_frac=prepared.near_constant_frac,
            min_sector_support=prepared.sector_support_min,
            beta_norm=None,
            effective_df=None,
            corr_before={},
            corr_after={},
            diagnostics=prepared.diagnostics,
        )

    rank = int(np.linalg.matrix_rank(X))
    if rank < X.shape[1]:
        return FitResult(
            success=False,
            failure_reason=FAIL_DESIGN_SINGULAR,
            model_name=config.neutralization_estimator,
            lambda_used=lambda_used,
            residuals=empty_series.copy(),
            fitted=empty_series.copy(),
            r2=None,
            residual_std=None,
            residual_var=None,
            cond_number=cond,
            effective_rank=rank,
            n_obs_candidate=int(len(group)),
            n_obs_used=int(len(Xdf)),
            dropped_row_frac=prepared.dropped_row_frac,
            missing_essential_frac=prepared.missing_essential_frac,
            near_constant_frac=prepared.near_constant_frac,
            min_sector_support=prepared.sector_support_min,
            beta_norm=None,
            effective_df=effective_df,
            corr_before={},
            corr_after={},
            diagnostics=prepared.diagnostics,
        )

    if np.isfinite(cond) and cond > config.cond_max:
        return FitResult(
            success=False,
            failure_reason=FAIL_CONDITION_NUMBER,
            model_name=config.neutralization_estimator,
            lambda_used=lambda_used,
            residuals=empty_series.copy(),
            fitted=empty_series.copy(),
            r2=None,
            residual_std=None,
            residual_var=None,
            cond_number=cond,
            effective_rank=rank,
            n_obs_candidate=int(len(group)),
            n_obs_used=int(len(Xdf)),
            dropped_row_frac=prepared.dropped_row_frac,
            missing_essential_frac=prepared.missing_essential_frac,
            near_constant_frac=prepared.near_constant_frac,
            min_sector_support=prepared.sector_support_min,
            beta_norm=float(np.linalg.norm(beta)),
            effective_df=effective_df,
            corr_before={},
            corr_after={},
            diagnostics=prepared.diagnostics,
        )

    fitted_valid = X @ beta
    resid_valid = yv - fitted_valid
    residual_var = float(np.var(resid_valid, ddof=1)) if len(resid_valid) > 1 else 0.0
    if residual_var < config.residual_var_min:
        return FitResult(
            success=False,
            failure_reason=FAIL_RESIDUAL_COLLAPSED,
            model_name=config.neutralization_estimator,
            lambda_used=lambda_used,
            residuals=empty_series.copy(),
            fitted=empty_series.copy(),
            r2=None,
            residual_std=math.sqrt(max(residual_var, 0.0)),
            residual_var=residual_var,
            cond_number=cond,
            effective_rank=rank,
            n_obs_candidate=int(len(group)),
            n_obs_used=int(len(Xdf)),
            dropped_row_frac=prepared.dropped_row_frac,
            missing_essential_frac=prepared.missing_essential_frac,
            near_constant_frac=prepared.near_constant_frac,
            min_sector_support=prepared.sector_support_min,
            beta_norm=float(np.linalg.norm(beta)),
            effective_df=effective_df,
            corr_before={},
            corr_after={},
            diagnostics=prepared.diagnostics,
        )

    tss = float(np.sum((yv - float(np.mean(yv))) ** 2))
    rss = float(np.sum(resid_valid**2))
    r2 = 1.0 - rss / tss if tss > 0 else math.nan

    residuals = empty_series.copy()
    fitted = empty_series.copy()
    residuals.loc[Xdf.index] = resid_valid
    fitted.loc[Xdf.index] = fitted_valid

    corr_before: dict[str, float] = {}
    corr_after: dict[str, float] = {}
    for factor in design_spec.numeric_factors:
        if factor in group.columns:
            factor_series = pd.to_numeric(group.loc[Xdf.index, factor], errors="coerce")
            corr_before[factor] = _safe_corr(y.loc[Xdf.index], factor_series)
            corr_after[factor] = _safe_corr(residuals.loc[Xdf.index], factor_series)

    diagnostics = dict(prepared.diagnostics)
    diagnostics["hc3_supported"] = bool(config.fit_hc3_se)

    return FitResult(
        success=True,
        failure_reason=None,
        model_name=config.neutralization_estimator,
        lambda_used=lambda_used,
        residuals=residuals,
        fitted=fitted,
        r2=r2,
        residual_std=math.sqrt(residual_var),
        residual_var=residual_var,
        cond_number=cond,
        effective_rank=rank,
        n_obs_candidate=int(len(group)),
        n_obs_used=int(len(Xdf)),
        dropped_row_frac=prepared.dropped_row_frac,
        missing_essential_frac=prepared.missing_essential_frac,
        near_constant_frac=prepared.near_constant_frac,
        min_sector_support=prepared.sector_support_min,
        beta_norm=float(np.linalg.norm(beta)),
        effective_df=effective_df,
        corr_before=corr_before,
        corr_after=corr_after,
        diagnostics=diagnostics,
    )


def _apply_group_fit(
    group: pd.DataFrame,
    fit: FitResult,
    run_id: str,
) -> pd.DataFrame:
    out = group.copy()
    out["y_neut"] = fit.residuals.reindex(out.index)
    out["neutralization_valid_flag"] = bool(fit.success)
    out["neutralization_failure_reason"] = fit.failure_reason
    out["neutralization_model"] = fit.model_name
    out["lambda_used"] = fit.lambda_used
    out["r2_cross_sectional"] = fit.r2
    out["residual_std"] = fit.residual_std
    out["cond_number"] = fit.cond_number
    out["n_obs_date"] = fit.n_obs_candidate
    out["n_obs_used_date"] = fit.n_obs_used
    out["dropped_row_frac"] = fit.dropped_row_frac
    out["missing_essential_frac"] = fit.missing_essential_frac
    out["near_constant_frac"] = fit.near_constant_frac
    out["beta_norm"] = fit.beta_norm
    out["effective_rank"] = fit.effective_rank
    out["effective_df"] = fit.effective_df
    out["min_sector_support"] = fit.min_sector_support
    out["run_id"] = run_id
    return out


def _compute_before_after_stats(
    result_df: pd.DataFrame,
    design_spec: DesignSpec,
    emit_date_level_stats: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def build_row(level: str, grp_key: tuple[Any, ...], grp: pd.DataFrame) -> dict[str, Any]:
        row: dict[str, Any] = {
            "stat_level": level,
            "horizon": int(grp_key[-1]) if level in {"date_horizon", "horizon"} else None,
            "date": grp_key[0] if level == "date_horizon" else pd.NaT,
            "n_rows": int(len(grp)),
            "n_valid_neut": int(grp["neutralization_valid_flag"].fillna(False).sum()),
            "invalid_date_fraction": float((~grp["neutralization_valid_flag"].fillna(False)).mean()),
            "mean_y_base": float(pd.to_numeric(grp["y_base"], errors="coerce").mean()),
            "mean_y_neut": float(pd.to_numeric(grp["y_neut"], errors="coerce").mean()),
            "var_y_base": float(pd.to_numeric(grp["y_base"], errors="coerce").var(ddof=1)) if len(grp) > 1 else math.nan,
            "var_y_neut": float(pd.to_numeric(grp["y_neut"], errors="coerce").var(ddof=1)) if len(grp) > 1 else math.nan,
            "median_r2": float(pd.to_numeric(grp["r2_cross_sectional"], errors="coerce").median()),
            "median_residual_std": float(pd.to_numeric(grp["residual_std"], errors="coerce").median()),
        }
        for factor in ["beta_mkt", "log_mktcap", "liq"]:
            if factor in grp.columns:
                row[f"corr_y_base_{factor}"] = _safe_corr(grp["y_base"], grp[factor])
                row[f"corr_y_neut_{factor}"] = _safe_corr(grp["y_neut"], grp[factor])
        return row

    if emit_date_level_stats:
        for (date_value, horizon), grp in result_df.groupby(["date", "horizon"], sort=True):
            rows.append(build_row("date_horizon", (date_value, horizon), grp))
    for horizon, grp in result_df.groupby("horizon", sort=True):
        rows.append(build_row("horizon", (horizon,), grp))
    return pd.DataFrame(rows)


def _build_failed_dates(result_df: pd.DataFrame) -> pd.DataFrame:
    failed = result_df.loc[~result_df["neutralization_valid_flag"].fillna(False)].copy()
    if failed.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "horizon",
                "failure_reason",
                "n_rows",
                "n_obs_date",
                "n_obs_used_date",
                "cond_number",
                "residual_std",
                "run_id",
            ]
        )
    grouped = (
        failed.groupby(["date", "horizon", "neutralization_failure_reason"], dropna=False)
        .agg(
            n_rows=("symbol", "size"),
            n_obs_date=("n_obs_date", "max"),
            n_obs_used_date=("n_obs_used_date", "max"),
            cond_number=("cond_number", "max"),
            residual_std=("residual_std", "max"),
            run_id=("run_id", "last"),
        )
        .reset_index()
        .rename(columns={"neutralization_failure_reason": "failure_reason"})
    )
    return grouped


def _build_summary(
    result_df: pd.DataFrame,
    before_after_df: pd.DataFrame,
    failed_dates_df: pd.DataFrame,
    config: NeutralizationConfig,
    design_spec: DesignSpec,
) -> dict[str, Any]:
    horizon_summary: dict[str, Any] = {}
    for horizon, grp in result_df.groupby("horizon", sort=True):
        valid_mask = grp["neutralization_valid_flag"].fillna(False)
        base = pd.to_numeric(grp["y_base"], errors="coerce")
        neut = pd.to_numeric(grp["y_neut"], errors="coerce")
        horizon_summary[str(int(horizon))] = {
            "n_rows": int(len(grp)),
            "n_valid_rows": int(valid_mask.sum()),
            "valid_fraction": float(valid_mask.mean()),
            "mean_y_base": float(base.mean()),
            "mean_y_neut": float(neut.mean()),
            "var_y_base": float(base.var(ddof=1)) if len(grp) > 1 else math.nan,
            "var_y_neut": float(neut.var(ddof=1)) if len(grp) > 1 else math.nan,
            "median_r2": float(pd.to_numeric(grp["r2_cross_sectional"], errors="coerce").median()),
            "median_residual_std": float(pd.to_numeric(grp["residual_std"], errors="coerce").median()),
            "failure_reasons": grp.loc[~valid_mask, "neutralization_failure_reason"].value_counts(dropna=False).to_dict(),
        }
    summary = {
        "run_id": config.run_id,
        "created_at_utc": _utc_now_iso(),
        "policy_version": config.policy_version,
        "non_canonical_flag": config.non_canonical_flag,
        "neutralization_input": config.neutralization_input,
        "neutralization_mode": config.neutralization_mode,
        "neutralization_estimator": config.neutralization_estimator,
        "lambda_selection": config.lambda_selection,
        "lambda_default": config.lambda_default,
        "n_rows": int(len(result_df)),
        "n_valid_rows": int(result_df["neutralization_valid_flag"].fillna(False).sum()),
        "valid_fraction": float(result_df["neutralization_valid_flag"].fillna(False).mean()),
        "n_failed_date_horizon": int(len(failed_dates_df)),
        "design_spec": asdict(design_spec),
        "horizon_summary": horizon_summary,
        "before_after_rows": int(len(before_after_df)),
    }
    return summary


def _manifest_payload(
    config: NeutralizationConfig,
    resolved: ResolvedColumns,
    design_spec: DesignSpec,
    summary: dict[str, Any],
    input_shapes: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "module": "labels/neutralized_targets.py",
        "run_id": config.run_id,
        "created_at_utc": _utc_now_iso(),
        "config": config.to_serializable(),
        "resolved_columns": asdict(resolved),
        "design_spec": asdict(design_spec),
        "input_files": {
            "labels_path": config.labels_path,
            "factor_exposures_path": config.factor_exposures_path,
            "label_manifest_path": config.label_manifest_path,
            "factor_manifest_path": config.factor_manifest_path,
            "prices_pit_path": config.prices_pit_path,
            "universe_history_path": config.universe_history_path,
            "labels_sha256": _safe_hash_file(config.labels_path),
            "factor_exposures_sha256": _safe_hash_file(config.factor_exposures_path),
            "label_manifest_sha256": _safe_hash_file(config.label_manifest_path),
            "factor_manifest_sha256": _safe_hash_file(config.factor_manifest_path),
        },
        "input_shapes": dict(input_shapes),
        "summary": summary,
        "outputs": dict(outputs),
    }


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_table(df: pd.DataFrame, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return str(path)
        except Exception:
            fallback = path.with_suffix(".csv")
            df.to_csv(fallback, index=False)
            return str(fallback)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return str(path)
    raise ValueError(f"Unsupported output format: {path}")


def _persist_main_output(df: pd.DataFrame, output_dir: Path, config: NeutralizationConfig) -> list[str]:
    files: list[str] = []
    if config.persist_partitioned_output:
        base_dir = output_dir / "labels" / "neutralized"
        for date_value, grp in df.groupby("date", sort=True):
            date_dir = base_dir / f"date={pd.Timestamp(date_value).date().isoformat()}"
            files.append(_write_table(grp.reset_index(drop=True), date_dir / f"part-{config.run_id}.parquet"))
    else:
        files.append(_write_table(df.reset_index(drop=True), output_dir / f"neutralized_targets_{config.run_id}.parquet"))
    return files


def run_neutralized_targets(config: NeutralizationConfig) -> dict[str, Any]:
    labels_df = _load_table(config.labels_path)
    exposures_df = _load_table(config.factor_exposures_path)
    resolved = _resolve_columns(labels_df, exposures_df, config)
    labels_long, horizons = _prepare_labels_long(labels_df, resolved, config)
    exposures = _prepare_exposures(exposures_df, resolved, config)
    aligned = _align_labels_and_exposures(labels_long, exposures, resolved)
    design_spec = _build_design_spec(aligned, config)

    result_pieces: list[pd.DataFrame] = []
    for (date_value, horizon), group in aligned.groupby([resolved.label_date_col, "horizon"], sort=True):
        grp = group.copy()
        grp = grp.rename(
            columns={
                resolved.label_date_col: "date",
                resolved.label_symbol_col: "symbol",
            }
        )
        fit = _fit_one_group(grp, design_spec, config)
        applied = _apply_group_fit(grp, fit, config.run_id)
        result_pieces.append(applied)
        LOGGER.info(
            "neutralized date=%s horizon=%s n=%s success=%s reason=%s",
            pd.Timestamp(date_value).date().isoformat(),
            horizon,
            len(group),
            fit.success,
            fit.failure_reason,
        )

    result_df = pd.concat(result_pieces, axis=0, ignore_index=True)
    result_df["date"] = _standardize_datetime(result_df["date"], "date")
    result_df["symbol"] = result_df["symbol"].astype(str)
    result_df["horizon"] = pd.to_numeric(result_df["horizon"], errors="coerce").astype("Int64")

    ordered_cols = [
        "date",
        "symbol",
        "horizon",
        "y_base",
        "y_neut",
        "neutralization_valid_flag",
        "neutralization_failure_reason",
        "neutralization_model",
        "lambda_used",
        "r2_cross_sectional",
        "residual_std",
        "cond_number",
        "n_obs_date",
        "n_obs_used_date",
        "run_id",
    ]
    head_cols = [c for c in ordered_cols if c in result_df.columns]
    tail_cols = [c for c in result_df.columns if c not in head_cols]
    result_df = result_df.loc[:, [*head_cols, *tail_cols]]

    before_after_df = _compute_before_after_stats(result_df, design_spec, config.emit_date_level_stats)
    failed_dates_df = _build_failed_dates(result_df)
    summary = _build_summary(result_df, before_after_df, failed_dates_df, config, design_spec)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_files = _persist_main_output(result_df, output_dir, config)
    before_after_path = _write_table(before_after_df, output_dir / f"before_after_stats_{config.run_id}.parquet")
    failed_dates_path = _write_table(failed_dates_df, output_dir / f"failed_dates_{config.run_id}.parquet")
    summary_path = output_dir / f"neutralization_summary_{config.run_id}.json"
    _write_json(summary_path, summary)

    manifest = _manifest_payload(
        config=config,
        resolved=resolved,
        design_spec=design_spec,
        summary=summary,
        input_shapes={
            "labels_rows": int(len(labels_df)),
            "labels_cols": int(labels_df.shape[1]),
            "labels_long_rows": int(len(labels_long)),
            "factor_rows": int(len(exposures_df)),
            "factor_cols": int(exposures_df.shape[1]),
            "aligned_rows": int(len(aligned)),
            "horizons": [int(h) for h in horizons],
        },
        outputs={
            "main_output_files": main_files,
            "before_after_stats": before_after_path,
            "failed_dates": failed_dates_path,
            "summary_json": str(summary_path),
        },
    )
    manifest_path = output_dir / f"manifest_{config.run_id}.json"
    _write_json(manifest_path, manifest)

    return {
        "result_df": result_df,
        "before_after_df": before_after_df,
        "failed_dates_df": failed_dates_df,
        "summary": summary,
        "manifest": manifest,
        "output_dir": str(output_dir),
        "main_output_files": main_files,
        "before_after_path": before_after_path,
        "failed_dates_path": failed_dates_path,
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Neutralize PIT forward targets against systematic exposures")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    parser.add_argument("--labels-path", help="Override labels_path")
    parser.add_argument("--factor-exposures-path", help="Override factor_exposures_path")
    parser.add_argument("--run-id", help="Override run_id")
    parser.add_argument("--output-dir", help="Override output_dir")
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
    if args.labels_path:
        payload["labels_path"] = args.labels_path
    if args.factor_exposures_path:
        payload["factor_exposures_path"] = args.factor_exposures_path
    if args.run_id:
        payload["run_id"] = args.run_id
    if args.output_dir:
        payload["output_dir"] = args.output_dir
    config = NeutralizationConfig.from_mapping(payload)
    outputs = run_neutralized_targets(config)
    LOGGER.info("neutralized targets written to %s", outputs["output_dir"])
    LOGGER.info("summary: %s", outputs["summary_path"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
