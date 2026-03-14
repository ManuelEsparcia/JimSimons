from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Allow direct script execution:
# `python simons_smallcap_swing/models/baselines/cross_validated_baselines.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "cross_validated_baselines_mvp_v1"
MODE_RIDGE_CV = "ridge_cv"
MODE_LOGISTIC_CV = "logistic_cv"
MODE_DUMMY_REGRESSOR_CV = "dummy_regressor_cv"
MODE_DUMMY_CLASSIFIER_CV = "dummy_classifier_cv"
SUPPORTED_MODES: tuple[str, ...] = (
    MODE_RIDGE_CV,
    MODE_LOGISTIC_CV,
    MODE_DUMMY_REGRESSOR_CV,
    MODE_DUMMY_CLASSIFIER_CV,
)

REGRESSION_MODES: tuple[str, ...] = (MODE_RIDGE_CV, MODE_DUMMY_REGRESSOR_CV)
CLASSIFICATION_MODES: tuple[str, ...] = (MODE_LOGISTIC_CV, MODE_DUMMY_CLASSIFIER_CV)
DUMMY_MODES: tuple[str, ...] = (MODE_DUMMY_REGRESSOR_CV, MODE_DUMMY_CLASSIFIER_CV)

DEFAULT_LABEL_BY_MODE = {
    MODE_RIDGE_CV: "fwd_ret_5d",
    MODE_LOGISTIC_CV: "fwd_dir_up_5d",
    MODE_DUMMY_REGRESSOR_CV: "fwd_ret_5d",
    MODE_DUMMY_CLASSIFIER_CV: "fwd_dir_up_5d",
}
TARGET_TYPE_BY_MODE = {
    MODE_RIDGE_CV: "continuous_forward_return",
    MODE_LOGISTIC_CV: "binary_direction",
    MODE_DUMMY_REGRESSOR_CV: "continuous_forward_return",
    MODE_DUMMY_CLASSIFIER_CV: "binary_direction",
}
PRIMARY_METRIC_BY_MODE = {
    MODE_RIDGE_CV: "mse",
    MODE_LOGISTIC_CV: "log_loss",
    MODE_DUMMY_REGRESSOR_CV: "mse",
    MODE_DUMMY_CLASSIFIER_CV: "log_loss",
}
DEFAULT_ALPHA_GRID: tuple[float, ...] = (0.1, 1.0, 10.0)
DEFAULT_C_GRID: tuple[float, ...] = (0.1, 1.0, 10.0)
DEFAULT_DUMMY_STRATEGY_BY_MODE: dict[str, str] = {
    MODE_DUMMY_REGRESSOR_CV: "mean",
    MODE_DUMMY_CLASSIFIER_CV: "prior",
}
ALLOWED_DUMMY_STRATEGIES_BY_MODE: dict[str, tuple[str, ...]] = {
    MODE_DUMMY_REGRESSOR_CV: ("mean", "median"),
    MODE_DUMMY_CLASSIFIER_CV: ("prior", "majority"),
}

TRAIN_SPLIT_ROLE = "train"
VALID_SPLIT_ROLE = "valid"
IGNORED_ROLES: tuple[str, ...] = ("dropped_by_purge", "dropped_by_embargo")
ALLOWED_CV_ROLES: tuple[str, ...] = (TRAIN_SPLIT_ROLE, VALID_SPLIT_ROLE, *IGNORED_ROLES)

MODEL_DATASET_SCHEMA = DataSchema(
    name="cv_baseline_model_dataset_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("horizon_days", "int64", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("target_value", "number", nullable=False),
        ColumnSpec("target_type", "string", nullable=False),
    ),
    primary_key=("date", "instrument_id", "horizon_days", "label_name"),
    allow_extra_columns=True,
)

PURGED_CV_SCHEMA = DataSchema(
    name="cv_baseline_purged_cv_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("fold_id", "int64", nullable=False),
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("horizon_days", "int64", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("split_role", "string", nullable=False),
        ColumnSpec("entry_date", "datetime64", nullable=False),
        ColumnSpec("exit_date", "datetime64", nullable=False),
    ),
    primary_key=("fold_id", "date", "instrument_id", "horizon_days", "label_name"),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class CrossValidatedBaselineResult:
    fold_metrics_path: Path
    summary_path: Path
    predictions_path: Path | None
    mode: str
    folds_completed: int
    n_folds: int
    primary_metric: str
    config_hash: str


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains invalid dates.")
    return parsed.dt.normalize()


def _normalize_grid(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    normalized = sorted({float(v) for v in values})
    if not normalized:
        raise ValueError(f"{name} cannot be empty.")
    if any(v <= 0 for v in normalized):
        raise ValueError(f"{name} must contain positive values. Received: {normalized}")
    return tuple(normalized)


def _parse_csv_floats(values: str | None) -> tuple[float, ...]:
    if not values:
        return ()
    return tuple(float(item.strip()) for item in values.split(",") if item.strip())


def _parse_csv_ints(values: str | None) -> tuple[int, ...]:
    if not values:
        return ()
    return tuple(int(item.strip()) for item in values.split(",") if item.strip())


def _config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _normalize_dummy_strategy(mode: str, dummy_strategy: str | None) -> str | None:
    if mode not in DUMMY_MODES:
        if dummy_strategy is not None:
            raise ValueError(f"dummy_strategy is only valid for dummy modes. mode='{mode}'")
        return None
    default = DEFAULT_DUMMY_STRATEGY_BY_MODE[mode]
    strategy = str(dummy_strategy or default).strip().lower()
    allowed = ALLOWED_DUMMY_STRATEGIES_BY_MODE[mode]
    if strategy not in allowed:
        raise ValueError(
            f"Unsupported dummy_strategy '{strategy}' for mode '{mode}'. Allowed: {allowed}"
        )
    return strategy


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return _safe_corr(x_rank, y_rank)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    mse = float(np.mean(np.square(residual)))
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum(np.square(residual)))
    mean_true = float(np.mean(y_true))
    ss_tot = float(np.sum(np.square(y_true - mean_true)))
    r2 = float("nan") if ss_tot == 0 else float(1.0 - ss_res / ss_tot)
    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "pearson_ic": _safe_corr(y_true, y_pred),
        "spearman_ic": _spearman_corr(y_true, y_pred),
    }


def _safe_roc_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, proba))
    except ValueError:
        return float("nan")


def _safe_average_precision(y_true: np.ndarray, proba: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, proba))
    except ValueError:
        return float("nan")


def _classification_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    pred_class: np.ndarray,
) -> dict[str, float]:
    return {
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, pred_class)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred_class)),
        "precision": float(precision_score(y_true, pred_class, zero_division=0)),
        "recall": float(recall_score(y_true, pred_class, zero_division=0)),
        "f1": float(f1_score(y_true, pred_class, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, proba),
        "average_precision": _safe_average_precision(y_true, proba),
    }


def _class_balance(y: np.ndarray) -> dict[str, float]:
    y_int = y.astype(int)
    n = int(len(y_int))
    n_pos = int(np.sum(y_int == 1))
    n_neg = int(np.sum(y_int == 0))
    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "positive_rate": float(n_pos / n) if n > 0 else float("nan"),
    }


def _solve_ridge(x_train: np.ndarray, y_train: np.ndarray, *, alpha: float) -> tuple[float, np.ndarray]:
    n, p = x_train.shape
    x_aug = np.column_stack([np.ones(n), x_train])
    penalty = np.eye(p + 1, dtype=float)
    penalty[0, 0] = 0.0
    lhs = x_aug.T @ x_aug + float(alpha) * penalty
    rhs = x_aug.T @ y_train
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs
    intercept = float(beta[0])
    weights = beta[1:].astype(float)
    return intercept, weights


def _predict_linear(x: np.ndarray, intercept: float, weights: np.ndarray) -> np.ndarray:
    return intercept + x @ weights


def _load_cv_method_hint(purged_cv_folds_path: Path) -> str | None:
    summary_path = purged_cv_folds_path.with_suffix(".summary.json")
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get("cv_method")
    return str(value) if value is not None else None


def _prepare_inputs(
    *,
    mode: str,
    model_dataset_path: str | Path | None,
    purged_cv_folds_path: str | Path | None,
    label_name: str,
    horizon_days: int | None,
    split_name: str | None,
    cv_method: str | None,
    fold_ids: Iterable[int] | None,
) -> tuple[pd.DataFrame, Path, Path, str | None]:
    base = Path(__file__).resolve().parents[2] / "data"
    dataset_source = (
        Path(model_dataset_path).expanduser().resolve()
        if model_dataset_path
        else base / "datasets" / "model_dataset.parquet"
    )
    folds_source = (
        Path(purged_cv_folds_path).expanduser().resolve()
        if purged_cv_folds_path
        else base / "labels" / "purged_cv_folds.parquet"
    )

    dataset = read_parquet(dataset_source)
    folds = read_parquet(folds_source)
    assert_schema(dataset, MODEL_DATASET_SCHEMA)
    assert_schema(folds, PURGED_CV_SCHEMA)

    dataset = dataset.copy()
    folds = folds.copy()
    dataset["date"] = _normalize_date(dataset["date"], column="date")
    dataset["instrument_id"] = dataset["instrument_id"].astype(str)
    dataset["horizon_days"] = pd.to_numeric(dataset["horizon_days"], errors="coerce").astype("int64")
    dataset["label_name"] = dataset["label_name"].astype(str)
    dataset["target_type"] = dataset["target_type"].astype(str)
    dataset["target_value"] = pd.to_numeric(dataset["target_value"], errors="coerce")
    if dataset["target_value"].isna().any():
        raise ValueError("model_dataset has invalid target_value values.")
    if dataset.duplicated(["date", "instrument_id", "horizon_days", "label_name"]).any():
        raise ValueError("model_dataset has duplicate PK rows.")

    folds["fold_id"] = pd.to_numeric(folds["fold_id"], errors="coerce").astype("int64")
    folds["date"] = _normalize_date(folds["date"], column="date")
    folds["instrument_id"] = folds["instrument_id"].astype(str)
    folds["horizon_days"] = pd.to_numeric(folds["horizon_days"], errors="coerce").astype("int64")
    folds["label_name"] = folds["label_name"].astype(str)
    folds["split_role"] = folds["split_role"].astype(str)
    folds["entry_date"] = _normalize_date(folds["entry_date"], column="entry_date")
    folds["exit_date"] = _normalize_date(folds["exit_date"], column="exit_date")
    invalid_roles = sorted(set(folds["split_role"].tolist()) - set(ALLOWED_CV_ROLES))
    if invalid_roles:
        raise ValueError(f"purged_cv_folds has invalid split_role values: {invalid_roles}")
    if folds.duplicated(["fold_id", "date", "instrument_id", "horizon_days", "label_name"]).any():
        raise ValueError("purged_cv_folds has duplicate PK rows.")

    dataset = dataset[dataset["label_name"] == label_name].copy()
    folds = folds[folds["label_name"] == label_name].copy()
    if horizon_days is not None:
        dataset = dataset[dataset["horizon_days"] == int(horizon_days)].copy()
        folds = folds[folds["horizon_days"] == int(horizon_days)].copy()

    selected_split_name: str | None = None
    if split_name is not None:
        if "split_name" not in dataset.columns:
            raise ValueError("split_name filter requested but model_dataset has no split_name column.")
        dataset = dataset[dataset["split_name"].astype(str) == str(split_name)].copy()
        selected_split_name = str(split_name)
    elif "split_name" in dataset.columns:
        unique_splits = sorted(dataset["split_name"].astype(str).unique().tolist())
        if len(unique_splits) == 1:
            selected_split_name = unique_splits[0]

    if fold_ids is not None:
        selected_folds = sorted({int(fold_id) for fold_id in fold_ids})
        if not selected_folds:
            raise ValueError("fold_ids filter provided but empty after normalization.")
        folds = folds[folds["fold_id"].isin(selected_folds)].copy()

    if dataset.empty:
        raise ValueError("No model_dataset rows left after filters.")
    if folds.empty:
        raise ValueError("No purged_cv_folds rows left after filters.")

    expected_target_type = TARGET_TYPE_BY_MODE[mode]
    observed_target_types = sorted(set(dataset["target_type"].tolist()))
    if observed_target_types != [expected_target_type]:
        raise ValueError(
            f"Target type mismatch for mode '{mode}'. Expected '{expected_target_type}', "
            f"observed {observed_target_types}."
        )

    if cv_method is not None:
        cv_method_hint = _load_cv_method_hint(folds_source)
        if cv_method_hint is None:
            raise ValueError(
                f"cv_method='{cv_method}' requested but summary hint file not found for {folds_source}."
            )
        if str(cv_method_hint) != str(cv_method):
            raise ValueError(
                f"cv_method mismatch. Requested '{cv_method}', folds summary has '{cv_method_hint}'."
            )

    join_key = ["date", "instrument_id", "horizon_days", "label_name"]
    merged = folds.merge(
        dataset,
        on=join_key,
        how="inner",
        suffixes=("_fold", ""),
    )
    if merged.empty:
        raise ValueError("No rows after joining model_dataset with purged_cv_folds on PK.")

    for col in ("split_role", "entry_date", "exit_date"):
        fold_col = f"{col}_fold"
        if fold_col in merged.columns:
            merged[col] = merged[fold_col]
            merged = merged.drop(columns=[fold_col])
    merged["split_role"] = merged["split_role"].astype(str)
    return merged, dataset_source, folds_source, selected_split_name


def _detect_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    excluded = {
        "fold_id",
        "date",
        "instrument_id",
        "ticker",
        "horizon_days",
        "label_name",
        "split_role",
        "entry_date",
        "exit_date",
        "target_value",
        "target_type",
        "split_name",
        "run_id",
        "config_hash",
        "built_ts_utc",
    }
    features = [
        col
        for col in frame.columns
        if col not in excluded and is_numeric_dtype(frame[col])
    ]
    if not features:
        raise ValueError("No numeric features detected in model_dataset for CV baseline training.")
    return tuple(sorted(features))


def _prepare_fold_features(
    fold_frame: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...],
    use_standardization: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    train_mask = fold_frame["split_role"] == TRAIN_SPLIT_ROLE
    valid_mask = fold_frame["split_role"] == VALID_SPLIT_ROLE

    train_df = fold_frame.loc[train_mask, list(feature_cols)].copy()
    valid_df = fold_frame.loc[valid_mask, list(feature_cols)].copy()
    y_train = fold_frame.loc[train_mask, "target_value"].to_numpy(dtype=float)
    y_valid = fold_frame.loc[valid_mask, "target_value"].to_numpy(dtype=float)

    train_median = train_df.median(axis=0, skipna=True)
    dropped_all_nan = sorted([col for col in feature_cols if pd.isna(train_median[col])])
    used_features = [col for col in feature_cols if col not in dropped_all_nan]
    if not used_features:
        raise ValueError("All features are all-NaN in train for this fold.")

    train_df = train_df[used_features]
    valid_df = valid_df[used_features]
    train_median = train_median[used_features]

    train_imp = train_df.fillna(train_median)
    valid_imp = valid_df.fillna(train_median)

    train_mean = train_imp.mean(axis=0)
    train_std = train_imp.std(axis=0, ddof=0).replace(0.0, 1.0)
    if use_standardization:
        train_proc = (train_imp - train_mean) / train_std
        valid_proc = (valid_imp - train_mean) / train_std
    else:
        train_proc = train_imp.copy()
        valid_proc = valid_imp.copy()

    x_train = train_proc.to_numpy(dtype=float)
    x_valid = valid_proc.to_numpy(dtype=float)
    if not np.isfinite(x_train).all() or not np.isfinite(x_valid).all():
        raise ValueError("Non-finite values detected after fold preprocessing.")

    meta = {
        "features_used": used_features,
        "features_dropped_all_nan_train": dropped_all_nan,
        "imputer_median_train_only": {k: float(v) for k, v in train_median.to_dict().items()},
    }
    return x_train, x_valid, y_train, y_valid, meta


def _fit_ridge_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    alphas: tuple[float, ...],
) -> tuple[float, np.ndarray, np.ndarray]:
    best_alpha = alphas[0]
    best_valid_pred: np.ndarray | None = None
    best_score = float("inf")
    best_intercept = 0.0
    best_weights: np.ndarray | None = None

    for alpha in alphas:
        intercept, weights = _solve_ridge(x_train, y_train, alpha=alpha)
        valid_pred = _predict_linear(x_valid, intercept, weights)
        mse = float(np.mean(np.square(valid_pred - y_valid)))
        if mse < best_score:
            best_score = mse
            best_alpha = float(alpha)
            best_valid_pred = valid_pred
            best_intercept = intercept
            best_weights = weights

    assert best_weights is not None
    assert best_valid_pred is not None
    train_pred = _predict_linear(x_train, best_intercept, best_weights)
    return best_alpha, train_pred, best_valid_pred


def _fit_logistic_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    c_values: tuple[float, ...],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not np.isin(y_train, [0.0, 1.0]).all():
        raise ValueError("Logistic mode expects binary target values in train set.")
    if not np.isin(y_valid, [0.0, 1.0]).all():
        raise ValueError("Logistic mode expects binary target values in valid set.")

    y_train_int = y_train.astype(int)
    y_valid_int = y_valid.astype(int)
    unique_classes = sorted(set(y_train_int.tolist()))
    if unique_classes != [0, 1]:
        raise ValueError(
            "Logistic mode requires both classes {0,1} in train for each fold. "
            f"Observed {unique_classes}."
        )

    best_c = c_values[0]
    best_model: LogisticRegression | None = None
    best_score = float("inf")
    for c_val in c_values:
        clf = LogisticRegression(
            C=float(c_val),
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=0,
        )
        clf.fit(x_train, y_train_int)
        valid_proba = clf.predict_proba(x_valid)[:, 1]
        score = float(log_loss(y_valid_int, valid_proba, labels=[0, 1]))
        if score < best_score:
            best_score = score
            best_c = float(c_val)
            best_model = clf

    assert best_model is not None
    train_proba = best_model.predict_proba(x_train)[:, 1]
    valid_proba = best_model.predict_proba(x_valid)[:, 1]
    train_class = (train_proba >= 0.5).astype(int)
    valid_class = (valid_proba >= 0.5).astype(int)
    return best_c, train_proba, train_class, valid_proba, valid_class


def _safe_float(value: float | int | np.floating | None) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if pd.isna(parsed):
        return None
    return parsed


def _fit_dummy_regressor_fold(
    y_train: np.ndarray,
    y_valid: np.ndarray,
    *,
    strategy: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    if strategy == "mean":
        statistic = float(np.mean(y_train))
    elif strategy == "median":
        statistic = float(np.median(y_train))
    else:
        raise ValueError(f"Unsupported dummy regressor strategy '{strategy}'.")

    pred_train = np.full(shape=y_train.shape, fill_value=statistic, dtype=float)
    pred_valid = np.full(shape=y_valid.shape, fill_value=statistic, dtype=float)
    return statistic, pred_train, pred_valid


def _clip_proba(proba: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(proba, eps, 1.0 - eps)


def _fit_dummy_classifier_fold(
    y_train: np.ndarray,
    y_valid: np.ndarray,
    *,
    strategy: str,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not np.isin(y_train, [0.0, 1.0]).all():
        raise ValueError("dummy_classifier_cv expects binary target values in train set.")
    if not np.isin(y_valid, [0.0, 1.0]).all():
        raise ValueError("dummy_classifier_cv expects binary target values in valid set.")

    y_train_bin = y_train.astype(int)
    positive_rate_train = float(np.mean(y_train_bin))

    if strategy == "prior":
        proba_value = positive_rate_train
        pred_class_value = int(proba_value >= 0.5)
    elif strategy == "majority":
        pred_class_value = int(positive_rate_train >= 0.5)
        proba_value = float(pred_class_value)
    else:
        raise ValueError(f"Unsupported dummy classifier strategy '{strategy}'.")

    train_proba = _clip_proba(np.full(shape=y_train.shape, fill_value=proba_value, dtype=float))
    valid_proba = _clip_proba(np.full(shape=y_valid.shape, fill_value=proba_value, dtype=float))
    train_class = np.full(shape=y_train.shape, fill_value=pred_class_value, dtype=int)
    valid_class = np.full(shape=y_valid.shape, fill_value=pred_class_value, dtype=int)
    return positive_rate_train, train_proba, train_class, valid_proba, valid_class


def run_cross_validated_baseline(
    *,
    mode: str,
    model_dataset_path: str | Path | None = None,
    purged_cv_folds_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    label_name: str | None = None,
    horizon_days: int | None = None,
    split_name: str | None = None,
    cv_method: str | None = None,
    fold_ids: Iterable[int] | None = None,
    alpha_grid: Iterable[float] = DEFAULT_ALPHA_GRID,
    c_grid: Iterable[float] = DEFAULT_C_GRID,
    dummy_strategy: str | None = None,
    use_standardization: bool = True,
    fail_on_invalid_fold: bool = False,
    write_predictions: bool = True,
    run_id: str = MODULE_VERSION,
) -> CrossValidatedBaselineResult:
    logger = get_logger("models.baselines.cross_validated_baselines")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of {SUPPORTED_MODES}")

    selected_label = str(label_name or DEFAULT_LABEL_BY_MODE[mode])
    alphas = _normalize_grid(alpha_grid, name="alpha_grid")
    c_values = _normalize_grid(c_grid, name="c_grid")
    selected_dummy_strategy = _normalize_dummy_strategy(mode, dummy_strategy)
    target_type = TARGET_TYPE_BY_MODE[mode]
    primary_metric = PRIMARY_METRIC_BY_MODE[mode]

    merged, dataset_source, folds_source, selected_split_name = _prepare_inputs(
        mode=mode,
        model_dataset_path=model_dataset_path,
        purged_cv_folds_path=purged_cv_folds_path,
        label_name=selected_label,
        horizon_days=horizon_days,
        split_name=split_name,
        cv_method=cv_method,
        fold_ids=fold_ids,
    )
    feature_cols: tuple[str, ...] = ()
    if mode not in DUMMY_MODES:
        feature_cols = _detect_feature_columns(merged)
    all_fold_ids = sorted({int(x) for x in merged["fold_id"].tolist()})

    config_payload = {
        "module_version": MODULE_VERSION,
        "mode": mode,
        "label_name": selected_label,
        "horizon_days": None if horizon_days is None else int(horizon_days),
        "split_name": selected_split_name if selected_split_name is not None else split_name,
        "cv_method": cv_method,
        "fold_ids": all_fold_ids,
        "use_standardization": bool(use_standardization),
        "alpha_grid": list(alphas),
        "c_grid": list(c_values),
        "dummy_strategy": selected_dummy_strategy,
        "fail_on_invalid_fold": bool(fail_on_invalid_fold),
        "write_predictions": bool(write_predictions),
        "run_id": run_id,
    }
    cfg_hash = _config_hash(config_payload)
    built_ts = datetime.now(UTC).isoformat()

    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    skipped_folds: list[dict[str, Any]] = []

    for fold_id in all_fold_ids:
        fold_full = merged[merged["fold_id"].astype(int) == int(fold_id)].copy()
        modelable = fold_full[fold_full["split_role"].isin({TRAIN_SPLIT_ROLE, VALID_SPLIT_ROLE})].copy()
        role_counts_all = {
            str(k): int(v)
            for k, v in fold_full["split_role"].value_counts(dropna=False).to_dict().items()
        }

        n_train = int((modelable["split_role"] == TRAIN_SPLIT_ROLE).sum())
        n_valid = int((modelable["split_role"] == VALID_SPLIT_ROLE).sum())
        n_dropped_purge = int((fold_full["split_role"] == "dropped_by_purge").sum())
        n_dropped_embargo = int((fold_full["split_role"] == "dropped_by_embargo").sum())

        row_common = {
            "model_name": mode,
            "fold_id": int(fold_id),
            "label_name": selected_label,
            "horizon_days": None if horizon_days is None else int(horizon_days),
            "target_type": target_type,
            "n_train": n_train,
            "n_valid": n_valid,
            "n_dropped_by_purge": n_dropped_purge,
            "n_dropped_by_embargo": n_dropped_embargo,
            "primary_metric": primary_metric,
            "run_id": run_id,
            "config_hash": cfg_hash,
            "built_ts_utc": built_ts,
            "split_counts_all_roles_json": json.dumps(role_counts_all, sort_keys=True),
        }

        if n_train == 0 or n_valid == 0:
            reason = f"missing train/valid rows (n_train={n_train}, n_valid={n_valid})"
            if fail_on_invalid_fold:
                raise ValueError(f"Fold {fold_id}: {reason}")
            skipped_folds.append({"fold_id": int(fold_id), "reason": reason})
            fold_rows.append(
                {
                    **row_common,
                    "status": "skipped",
                    "skip_reason": reason,
                    "n_features_used": 0,
                    "valid_primary_metric": np.nan,
                }
            )
            continue

        if mode in DUMMY_MODES:
            y_train = modelable.loc[modelable["split_role"] == TRAIN_SPLIT_ROLE, "target_value"].to_numpy(dtype=float)
            y_valid = modelable.loc[modelable["split_role"] == VALID_SPLIT_ROLE, "target_value"].to_numpy(dtype=float)
            preprocess_meta = {
                "features_used": [],
                "features_dropped_all_nan_train": [],
                "imputer_median_train_only": {},
            }
        else:
            try:
                x_train, x_valid, y_train, y_valid, preprocess_meta = _prepare_fold_features(
                    modelable,
                    feature_cols=feature_cols,
                    use_standardization=use_standardization,
                )
            except Exception as exc:
                reason = f"fold preprocessing failed: {exc}"
                if fail_on_invalid_fold:
                    raise ValueError(f"Fold {fold_id}: {reason}") from exc
                skipped_folds.append({"fold_id": int(fold_id), "reason": reason})
                fold_rows.append(
                    {
                        **row_common,
                        "status": "skipped",
                        "skip_reason": reason,
                        "n_features_used": 0,
                        "valid_primary_metric": np.nan,
                    }
                )
                continue

        row_common = {
            **row_common,
            "status": "completed",
            "skip_reason": None,
            "n_features_used": int(len(preprocess_meta["features_used"])),
            "features_used_json": json.dumps(preprocess_meta["features_used"], sort_keys=True),
            "features_dropped_all_nan_train_json": json.dumps(
                preprocess_meta["features_dropped_all_nan_train"], sort_keys=True
            ),
            "imputer_median_train_only_json": json.dumps(
                preprocess_meta["imputer_median_train_only"], sort_keys=True
            ),
        }

        fold_train = modelable[modelable["split_role"] == TRAIN_SPLIT_ROLE].copy()
        fold_valid = modelable[modelable["split_role"] == VALID_SPLIT_ROLE].copy()

        if mode == MODE_RIDGE_CV:
            alpha_selected, pred_train, pred_valid = _fit_ridge_fold(
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                alphas=alphas,
            )
            train_metrics = _regression_metrics(y_train, pred_train)
            valid_metrics = _regression_metrics(y_valid, pred_valid)

            fold_row = {
                **row_common,
                "alpha_selected": float(alpha_selected),
                "valid_primary_metric": valid_metrics["mse"],
            }
            for metric_name, metric_value in train_metrics.items():
                fold_row[f"train_{metric_name}"] = float(metric_value)
            for metric_name, metric_value in valid_metrics.items():
                fold_row[f"valid_{metric_name}"] = float(metric_value)
            fold_rows.append(fold_row)

            if write_predictions:
                pred_train_df = fold_train[["date", "instrument_id", "ticker", "horizon_days", "label_name"]].copy()
                pred_train_df["fold_id"] = int(fold_id)
                pred_train_df["split_role"] = TRAIN_SPLIT_ROLE
                pred_train_df["target_value"] = y_train
                pred_train_df["prediction"] = pred_train
                pred_train_df["residual"] = y_train - pred_train

                pred_valid_df = fold_valid[["date", "instrument_id", "ticker", "horizon_days", "label_name"]].copy()
                pred_valid_df["fold_id"] = int(fold_id)
                pred_valid_df["split_role"] = VALID_SPLIT_ROLE
                pred_valid_df["target_value"] = y_valid
                pred_valid_df["prediction"] = pred_valid
                pred_valid_df["residual"] = y_valid - pred_valid
                prediction_frames.append(pd.concat([pred_train_df, pred_valid_df], ignore_index=True))

        elif mode == MODE_DUMMY_REGRESSOR_CV:
            assert selected_dummy_strategy is not None
            train_stat, pred_train, pred_valid = _fit_dummy_regressor_fold(
                y_train=y_train,
                y_valid=y_valid,
                strategy=selected_dummy_strategy,
            )
            train_metrics = _regression_metrics(y_train, pred_train)
            valid_metrics = _regression_metrics(y_valid, pred_valid)

            fold_row = {
                **row_common,
                "dummy_strategy": selected_dummy_strategy,
                "dummy_stat_name": "train_target_stat",
                "dummy_stat_train": float(train_stat),
                "valid_primary_metric": valid_metrics["mse"],
            }
            for metric_name, metric_value in train_metrics.items():
                fold_row[f"train_{metric_name}"] = float(metric_value)
            for metric_name, metric_value in valid_metrics.items():
                fold_row[f"valid_{metric_name}"] = float(metric_value)
            fold_rows.append(fold_row)

            if write_predictions:
                pred_train_df = fold_train[["date", "instrument_id", "ticker", "horizon_days", "label_name"]].copy()
                pred_train_df["fold_id"] = int(fold_id)
                pred_train_df["split_role"] = TRAIN_SPLIT_ROLE
                pred_train_df["target_value"] = y_train
                pred_train_df["prediction"] = pred_train
                pred_train_df["residual"] = y_train - pred_train

                pred_valid_df = fold_valid[["date", "instrument_id", "ticker", "horizon_days", "label_name"]].copy()
                pred_valid_df["fold_id"] = int(fold_id)
                pred_valid_df["split_role"] = VALID_SPLIT_ROLE
                pred_valid_df["target_value"] = y_valid
                pred_valid_df["prediction"] = pred_valid
                pred_valid_df["residual"] = y_valid - pred_valid
                prediction_frames.append(pd.concat([pred_train_df, pred_valid_df], ignore_index=True))

        elif mode == MODE_LOGISTIC_CV:
            try:
                c_selected, proba_train, class_train, proba_valid, class_valid = _fit_logistic_fold(
                    x_train=x_train,
                    y_train=y_train,
                    x_valid=x_valid,
                    y_valid=y_valid,
                    c_values=c_values,
                )
            except Exception as exc:
                reason = f"logistic fold fit failed: {exc}"
                if fail_on_invalid_fold:
                    raise ValueError(f"Fold {fold_id}: {reason}") from exc
                skipped_folds.append({"fold_id": int(fold_id), "reason": reason})
                fold_rows.append(
                    {
                        **row_common,
                        "status": "skipped",
                        "skip_reason": reason,
                        "valid_primary_metric": np.nan,
                    }
                )
                continue

            y_train_bin = y_train.astype(int)
            y_valid_bin = y_valid.astype(int)
            train_metrics = _classification_metrics(y_train_bin, proba_train, class_train)
            valid_metrics = _classification_metrics(y_valid_bin, proba_valid, class_valid)

            fold_row = {
                **row_common,
                "c_selected": float(c_selected),
                "valid_primary_metric": valid_metrics["log_loss"],
                "train_class_balance_json": json.dumps(_class_balance(y_train_bin), sort_keys=True),
                "valid_class_balance_json": json.dumps(_class_balance(y_valid_bin), sort_keys=True),
            }
            for metric_name, metric_value in train_metrics.items():
                fold_row[f"train_{metric_name}"] = float(metric_value)
            for metric_name, metric_value in valid_metrics.items():
                fold_row[f"valid_{metric_name}"] = float(metric_value)
            fold_rows.append(fold_row)

            if write_predictions:
                pred_train_df = fold_train[["date", "instrument_id", "ticker", "horizon_days", "label_name"]].copy()
                pred_train_df["fold_id"] = int(fold_id)
                pred_train_df["split_role"] = TRAIN_SPLIT_ROLE
                pred_train_df["target_value"] = y_train_bin
                pred_train_df["pred_proba"] = proba_train
                pred_train_df["pred_class"] = class_train
                pred_train_df["residual_like"] = y_train_bin - proba_train

                pred_valid_df = fold_valid[["date", "instrument_id", "ticker", "horizon_days", "label_name"]].copy()
                pred_valid_df["fold_id"] = int(fold_id)
                pred_valid_df["split_role"] = VALID_SPLIT_ROLE
                pred_valid_df["target_value"] = y_valid_bin
                pred_valid_df["pred_proba"] = proba_valid
                pred_valid_df["pred_class"] = class_valid
                pred_valid_df["residual_like"] = y_valid_bin - proba_valid
                prediction_frames.append(pd.concat([pred_train_df, pred_valid_df], ignore_index=True))

        elif mode == MODE_DUMMY_CLASSIFIER_CV:
            assert selected_dummy_strategy is not None
            try:
                (
                    positive_rate_train,
                    proba_train,
                    class_train,
                    proba_valid,
                    class_valid,
                ) = _fit_dummy_classifier_fold(
                    y_train=y_train,
                    y_valid=y_valid,
                    strategy=selected_dummy_strategy,
                )
            except Exception as exc:
                reason = f"dummy classifier fold fit failed: {exc}"
                if fail_on_invalid_fold:
                    raise ValueError(f"Fold {fold_id}: {reason}") from exc
                skipped_folds.append({"fold_id": int(fold_id), "reason": reason})
                fold_rows.append(
                    {
                        **row_common,
                        "status": "skipped",
                        "skip_reason": reason,
                        "valid_primary_metric": np.nan,
                    }
                )
                continue

            y_train_bin = y_train.astype(int)
            y_valid_bin = y_valid.astype(int)
            train_metrics = _classification_metrics(y_train_bin, proba_train, class_train)
            valid_metrics = _classification_metrics(y_valid_bin, proba_valid, class_valid)

            fold_row = {
                **row_common,
                "dummy_strategy": selected_dummy_strategy,
                "dummy_stat_name": "positive_rate_train",
                "dummy_stat_train": float(positive_rate_train),
                "valid_primary_metric": valid_metrics["log_loss"],
                "train_class_balance_json": json.dumps(_class_balance(y_train_bin), sort_keys=True),
                "valid_class_balance_json": json.dumps(_class_balance(y_valid_bin), sort_keys=True),
            }
            for metric_name, metric_value in train_metrics.items():
                fold_row[f"train_{metric_name}"] = float(metric_value)
            for metric_name, metric_value in valid_metrics.items():
                fold_row[f"valid_{metric_name}"] = float(metric_value)
            fold_rows.append(fold_row)

            if write_predictions:
                pred_train_df = fold_train[["date", "instrument_id", "ticker", "horizon_days", "label_name"]].copy()
                pred_train_df["fold_id"] = int(fold_id)
                pred_train_df["split_role"] = TRAIN_SPLIT_ROLE
                pred_train_df["target_value"] = y_train_bin
                pred_train_df["pred_proba"] = proba_train
                pred_train_df["pred_class"] = class_train
                pred_train_df["residual_like"] = y_train_bin - proba_train

                pred_valid_df = fold_valid[["date", "instrument_id", "ticker", "horizon_days", "label_name"]].copy()
                pred_valid_df["fold_id"] = int(fold_id)
                pred_valid_df["split_role"] = VALID_SPLIT_ROLE
                pred_valid_df["target_value"] = y_valid_bin
                pred_valid_df["pred_proba"] = proba_valid
                pred_valid_df["pred_class"] = class_valid
                pred_valid_df["residual_like"] = y_valid_bin - proba_valid
                prediction_frames.append(pd.concat([pred_train_df, pred_valid_df], ignore_index=True))

        else:
            raise ValueError(f"Unsupported mode '{mode}'.")

    fold_metrics = pd.DataFrame(fold_rows).sort_values(["fold_id"]).reset_index(drop=True)
    output_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else (Path(__file__).resolve().parents[1] / "artifacts")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    fold_metrics_path = write_parquet(
        fold_metrics,
        output_root / "cv_baseline_fold_metrics.parquet",
        schema_name="cv_baseline_fold_metrics_mvp",
        run_id=run_id,
    )

    predictions_path: Path | None = None
    if write_predictions and prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions["target_type"] = target_type
        predictions["run_id"] = run_id
        predictions["config_hash"] = cfg_hash
        predictions["built_ts_utc"] = built_ts
        predictions_path = write_parquet(
            predictions,
            output_root / "cv_baseline_predictions.parquet",
            schema_name="cv_baseline_predictions_mvp",
            run_id=run_id,
        )

    completed = fold_metrics[fold_metrics["status"].astype(str) == "completed"].copy()
    if completed.empty:
        raise ValueError("No completed folds. All folds were skipped.")

    valid_primary = pd.to_numeric(completed["valid_primary_metric"], errors="coerce")
    n_features_used = pd.to_numeric(completed["n_features_used"], errors="coerce")
    notes = [
        "Only split_role in {train, valid} are used for fit/eval.",
        "dropped_by_purge and dropped_by_embargo are excluded from fit/eval.",
    ]
    if mode in DUMMY_MODES:
        notes.append("Dummy CV modes ignore feature matrix and learn target-only statistics per fold train set.")

    summary = {
        "module_version": MODULE_VERSION,
        "model_name": mode,
        "label_name": selected_label,
        "target_type": target_type,
        "horizon_days": None if horizon_days is None else int(horizon_days),
        "dummy_strategy": selected_dummy_strategy,
        "split_name": selected_split_name if selected_split_name is not None else split_name,
        "cv_method": cv_method or _load_cv_method_hint(folds_source),
        "n_folds": len(all_fold_ids),
        "folds_completed": int(len(completed)),
        "folds_skipped": int(len(fold_metrics) - len(completed)),
        "completed_fold_ids": sorted(completed["fold_id"].astype(int).tolist()),
        "skipped_folds": skipped_folds,
        "metric_aggregation_policy": (
            f"primary_metric_per_fold={primary_metric}; "
            "global_summary=mean/median/std over completed folds"
        ),
        "primary_metric": primary_metric,
        "mean_valid_primary_metric": _safe_float(valid_primary.mean(skipna=True)),
        "median_valid_primary_metric": _safe_float(valid_primary.median(skipna=True)),
        "std_valid_primary_metric": _safe_float(valid_primary.std(skipna=True, ddof=0)),
        "mean_n_train": _safe_float(pd.to_numeric(completed["n_train"], errors="coerce").mean(skipna=True)),
        "mean_n_valid": _safe_float(pd.to_numeric(completed["n_valid"], errors="coerce").mean(skipna=True)),
        "n_features_used_summary": {
            "min": _safe_float(n_features_used.min(skipna=True)),
            "median": _safe_float(n_features_used.median(skipna=True)),
            "max": _safe_float(n_features_used.max(skipna=True)),
            "mean": _safe_float(n_features_used.mean(skipna=True)),
        },
        "source_paths": {
            "model_dataset": str(dataset_source),
            "purged_cv_folds": str(folds_source),
        },
        "used_feature_columns": list(feature_cols),
        "notes": notes,
        "run_id": run_id,
        "config_hash": cfg_hash,
        "built_ts_utc": built_ts,
    }
    summary_path = output_root / "cv_baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "cv_baseline_completed",
        mode=mode,
        folds_completed=int(len(completed)),
        n_folds=len(all_fold_ids),
        summary_path=str(summary_path),
    )
    return CrossValidatedBaselineResult(
        fold_metrics_path=fold_metrics_path,
        summary_path=summary_path,
        predictions_path=predictions_path,
        mode=mode,
        folds_completed=int(len(completed)),
        n_folds=len(all_fold_ids),
        primary_metric=primary_metric,
        config_hash=cfg_hash,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run unified purged-CV baseline training/evaluation (ridge/logistic/dummy)."
    )
    parser.add_argument("--mode", required=True, choices=SUPPORTED_MODES)
    parser.add_argument("--model-dataset-path", default=None)
    parser.add_argument("--purged-cv-folds-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--label-name", default=None)
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument("--split-name", default=None)
    parser.add_argument("--cv-method", default=None)
    parser.add_argument("--fold-ids", default=None, help="Comma-separated fold ids.")
    parser.add_argument("--alphas", default="0.1,1.0,10.0", help="Comma-separated alpha grid for ridge.")
    parser.add_argument("--cs", default="0.1,1.0,10.0", help="Comma-separated C grid for logistic.")
    parser.add_argument(
        "--dummy-strategy",
        default=None,
        help="Dummy strategy by mode: regressor {mean,median}, classifier {prior,majority}.",
    )
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--fail-on-invalid-fold", action="store_true")
    parser.add_argument("--no-predictions", action="store_true")
    parser.add_argument("--run-id", default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_cross_validated_baseline(
        mode=args.mode,
        model_dataset_path=args.model_dataset_path,
        purged_cv_folds_path=args.purged_cv_folds_path,
        output_dir=args.output_dir,
        label_name=args.label_name,
        horizon_days=args.horizon_days,
        split_name=args.split_name,
        cv_method=args.cv_method,
        fold_ids=_parse_csv_ints(args.fold_ids) or None,
        alpha_grid=_parse_csv_floats(args.alphas) or DEFAULT_ALPHA_GRID,
        c_grid=_parse_csv_floats(args.cs) or DEFAULT_C_GRID,
        dummy_strategy=args.dummy_strategy,
        use_standardization=not args.no_standardize,
        fail_on_invalid_fold=bool(args.fail_on_invalid_fold),
        write_predictions=not args.no_predictions,
        run_id=args.run_id,
    )
    print(
        json.dumps(
            {
                "mode": result.mode,
                "n_folds": result.n_folds,
                "folds_completed": result.folds_completed,
                "primary_metric": result.primary_metric,
                "fold_metrics_path": str(result.fold_metrics_path),
                "summary_path": str(result.summary_path),
                "predictions_path": str(result.predictions_path) if result.predictions_path else None,
                "config_hash": result.config_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
