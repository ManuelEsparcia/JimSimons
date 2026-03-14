from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
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

# Allow direct script execution: `python simons_smallcap_swing/models/baselines/train_dummy_baselines.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "dummy_baselines_mvp_v1"

MODE_DUMMY_REGRESSOR = "dummy_regressor"
MODE_DUMMY_CLASSIFIER = "dummy_classifier"
SUPPORTED_MODES: tuple[str, ...] = (MODE_DUMMY_REGRESSOR, MODE_DUMMY_CLASSIFIER)

DEFAULT_LABEL_NAME_BY_MODE: dict[str, str] = {
    MODE_DUMMY_REGRESSOR: "fwd_ret_5d",
    MODE_DUMMY_CLASSIFIER: "fwd_dir_up_5d",
}
DEFAULT_STRATEGY_BY_MODE: dict[str, str] = {
    MODE_DUMMY_REGRESSOR: "mean",
    MODE_DUMMY_CLASSIFIER: "prior",
}
SUPPORTED_STRATEGIES_BY_MODE: dict[str, tuple[str, ...]] = {
    MODE_DUMMY_REGRESSOR: ("mean", "median"),
    MODE_DUMMY_CLASSIFIER: ("prior", "majority"),
}
TARGET_TYPE_BY_MODE: dict[str, str] = {
    MODE_DUMMY_REGRESSOR: "continuous_forward_return",
    MODE_DUMMY_CLASSIFIER: "binary_direction",
}
TRAINABLE_SPLIT_ROLES: tuple[str, ...] = ("train", "valid", "test")
DROPPED_SPLIT_ROLES: tuple[str, ...] = ("dropped_by_purge", "dropped_by_embargo")


MODEL_DATASET_INPUT_SCHEMA = DataSchema(
    name="dummy_baseline_model_dataset_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("horizon_days", "int64", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("split_name", "string", nullable=False),
        ColumnSpec("split_role", "string", nullable=False),
        ColumnSpec("target_value", "number", nullable=False),
        ColumnSpec("target_type", "string", nullable=False),
    ),
    primary_key=("date", "instrument_id", "horizon_days", "label_name"),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class TrainDummyBaselineResult:
    mode: str
    strategy: str
    metrics_path: Path
    predictions_path: Path
    learned_statistic: float
    n_train: int
    n_valid: int
    n_test: int
    config_hash: str


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains invalid dates.")
    return parsed.dt.normalize()


def _config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


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
    r2 = float("nan") if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
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


def _classification_metrics(y_true: np.ndarray, proba: np.ndarray, pred_class: np.ndarray) -> dict[str, float]:
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


def _validate_mode(mode: str) -> str:
    clean = str(mode).strip()
    if clean not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode '{clean}'. Allowed: {SUPPORTED_MODES}")
    return clean


def _resolve_strategy(mode: str, strategy: str | None) -> str:
    default = DEFAULT_STRATEGY_BY_MODE[mode]
    clean = default if strategy is None else str(strategy).strip().lower()
    supported = SUPPORTED_STRATEGIES_BY_MODE[mode]
    if clean not in supported:
        raise ValueError(
            f"Unsupported dummy strategy '{clean}' for mode '{mode}'. "
            f"Allowed: {supported}"
        )
    return clean


def _load_and_filter_dataset(
    *,
    mode: str,
    model_dataset_path: str | Path | None,
    label_name: str,
    horizon_days: int | None,
    split_name: str | None,
) -> tuple[pd.DataFrame, Path, str]:
    default_source = Path(__file__).resolve().parents[2] / "data" / "datasets" / "model_dataset.parquet"
    source = Path(model_dataset_path).expanduser().resolve() if model_dataset_path else default_source

    frame = read_parquet(source)
    assert_schema(frame, MODEL_DATASET_INPUT_SCHEMA)

    frame = frame.copy()
    frame["date"] = _normalize_date(frame["date"], column="date")
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["horizon_days"] = pd.to_numeric(frame["horizon_days"], errors="coerce").astype("int64")
    frame["label_name"] = frame["label_name"].astype(str)
    frame["split_name"] = frame["split_name"].astype(str)
    frame["split_role"] = frame["split_role"].astype(str)
    frame["target_type"] = frame["target_type"].astype(str)
    frame["target_value"] = pd.to_numeric(frame["target_value"], errors="coerce")

    if frame["target_value"].isna().any():
        raise ValueError("model_dataset contains null/non-numeric target_value.")
    if frame.duplicated(["date", "instrument_id", "horizon_days", "label_name"]).any():
        raise ValueError(
            "model_dataset has duplicate (date, instrument_id, horizon_days, label_name) rows."
        )

    valid_roles = set(TRAINABLE_SPLIT_ROLES) | set(DROPPED_SPLIT_ROLES)
    invalid_roles = sorted(set(frame["split_role"].tolist()) - valid_roles)
    if invalid_roles:
        raise ValueError(f"model_dataset has unsupported split_role values: {invalid_roles}")

    filtered = frame[frame["label_name"] == str(label_name)].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for label_name='{label_name}'.")

    if horizon_days is not None:
        filtered = filtered[filtered["horizon_days"] == int(horizon_days)].copy()
    if filtered.empty:
        raise ValueError(
            f"No rows found for label_name='{label_name}', horizon_days={int(horizon_days)}."
        )

    selected_split_name = split_name
    unique_splits = sorted(filtered["split_name"].unique().tolist())
    if selected_split_name is None:
        if len(unique_splits) == 1:
            selected_split_name = unique_splits[0]
        else:
            raise ValueError(
                "Multiple split_name values found. Please provide --split-name. "
                f"Available: {unique_splits}"
            )
    filtered = filtered[filtered["split_name"] == selected_split_name].copy()
    if filtered.empty:
        raise ValueError(f"No rows left for split_name='{selected_split_name}'.")

    expected_target_type = TARGET_TYPE_BY_MODE[mode]
    observed_target_types = sorted(filtered["target_type"].dropna().astype(str).unique().tolist())
    if observed_target_types != [expected_target_type]:
        raise ValueError(
            f"Target type mismatch for mode '{mode}'. Expected '{expected_target_type}', "
            f"observed {observed_target_types}. Rebuild model_dataset with coherent target_type."
        )

    model_rows = filtered[filtered["split_role"].isin(TRAINABLE_SPLIT_ROLES)].copy()
    if model_rows.empty:
        raise ValueError("No train/valid/test rows available after filtering.")

    n_train = int((model_rows["split_role"] == "train").sum())
    n_valid = int((model_rows["split_role"] == "valid").sum())
    n_test = int((model_rows["split_role"] == "test").sum())
    if n_train == 0 or n_valid == 0 or n_test == 0:
        raise ValueError(
            "Expected non-empty train, valid and test rows. "
            f"Found n_train={n_train}, n_valid={n_valid}, n_test={n_test}."
        )

    return (
        filtered.sort_values(["date", "instrument_id", "horizon_days", "label_name"]).reset_index(drop=True),
        source,
        str(selected_split_name),
    )


def train_dummy_baseline(
    *,
    mode: str,
    model_dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    label_name: str | None = None,
    horizon_days: int | None = None,
    split_name: str | None = None,
    dummy_strategy: str | None = None,
    run_id: str = MODULE_VERSION,
) -> TrainDummyBaselineResult:
    clean_mode = _validate_mode(mode)
    strategy = _resolve_strategy(clean_mode, dummy_strategy)
    selected_label_name = (
        str(label_name).strip() if label_name is not None else DEFAULT_LABEL_NAME_BY_MODE[clean_mode]
    )
    if not selected_label_name:
        raise ValueError("label_name cannot be empty.")

    logger = get_logger("models.baselines.train_dummy_baselines")
    dataset_all_roles, dataset_source, selected_split_name = _load_and_filter_dataset(
        mode=clean_mode,
        model_dataset_path=model_dataset_path,
        label_name=selected_label_name,
        horizon_days=horizon_days,
        split_name=split_name,
    )

    role_counts_all = dataset_all_roles["split_role"].value_counts().to_dict()
    model_rows = dataset_all_roles[dataset_all_roles["split_role"].isin(TRAINABLE_SPLIT_ROLES)].copy()

    train_mask = model_rows["split_role"] == "train"
    valid_mask = model_rows["split_role"] == "valid"
    test_mask = model_rows["split_role"] == "test"
    n_train = int(train_mask.sum())
    n_valid = int(valid_mask.sum())
    n_test = int(test_mask.sum())

    y_train = model_rows.loc[train_mask, "target_value"].to_numpy(dtype=float)
    y_valid = model_rows.loc[valid_mask, "target_value"].to_numpy(dtype=float)
    y_test = model_rows.loc[test_mask, "target_value"].to_numpy(dtype=float)

    learned_statistic: float
    predictions = model_rows[
        [
            "date",
            "instrument_id",
            "ticker",
            "split_name",
            "split_role",
            "horizon_days",
            "label_name",
            "target_value",
        ]
    ].copy()

    metrics_train: dict[str, float]
    metrics_valid: dict[str, float]
    metrics_test: dict[str, float]
    class_balance_payload: dict[str, dict[str, float]] | None = None

    if clean_mode == MODE_DUMMY_REGRESSOR:
        if strategy == "mean":
            learned_statistic = float(np.mean(y_train))
        elif strategy == "median":
            learned_statistic = float(np.median(y_train))
        else:
            raise ValueError(f"Unsupported strategy '{strategy}' for mode '{clean_mode}'.")

        pred_train = np.full(shape=n_train, fill_value=learned_statistic, dtype=float)
        pred_valid = np.full(shape=n_valid, fill_value=learned_statistic, dtype=float)
        pred_test = np.full(shape=n_test, fill_value=learned_statistic, dtype=float)

        pred_series = pd.Series(index=model_rows.index, dtype=float)
        pred_series.loc[train_mask] = pred_train
        pred_series.loc[valid_mask] = pred_valid
        pred_series.loc[test_mask] = pred_test

        predictions["prediction"] = pred_series.to_numpy(dtype=float)
        predictions["residual"] = (
            predictions["target_value"].to_numpy(dtype=float)
            - predictions["prediction"].to_numpy(dtype=float)
        )
        predictions["run_id"] = run_id

        metrics_train = _regression_metrics(y_train, pred_train)
        metrics_valid = _regression_metrics(y_valid, pred_valid)
        metrics_test = _regression_metrics(y_test, pred_test)

        artifact_prefix = "dummy_regressor"
        model_name = "dummy_regressor_baseline"
        target_type = TARGET_TYPE_BY_MODE[clean_mode]
        learned_stat_field = {"train_target_statistic_name": strategy, "train_target_statistic_value": learned_statistic}

    elif clean_mode == MODE_DUMMY_CLASSIFIER:
        unique_y = sorted(set(model_rows["target_value"].astype(float).unique().tolist()))
        if any(item not in {0.0, 1.0} for item in unique_y):
            raise ValueError(
                f"Expected binary target values in {{0,1}} for dummy_classifier. Found: {unique_y}"
            )

        train_positive_rate = float(np.mean(y_train))
        if strategy == "prior":
            learned_statistic = train_positive_rate
            pred_class_value = 1 if train_positive_rate >= 0.5 else 0
            pred_proba_train = np.full(shape=n_train, fill_value=train_positive_rate, dtype=float)
            pred_proba_valid = np.full(shape=n_valid, fill_value=train_positive_rate, dtype=float)
            pred_proba_test = np.full(shape=n_test, fill_value=train_positive_rate, dtype=float)
            pred_class_train = np.full(shape=n_train, fill_value=pred_class_value, dtype=int)
            pred_class_valid = np.full(shape=n_valid, fill_value=pred_class_value, dtype=int)
            pred_class_test = np.full(shape=n_test, fill_value=pred_class_value, dtype=int)
        elif strategy == "majority":
            majority_class = 1 if train_positive_rate >= 0.5 else 0
            learned_statistic = float(majority_class)
            pred_proba_train = np.full(shape=n_train, fill_value=float(majority_class), dtype=float)
            pred_proba_valid = np.full(shape=n_valid, fill_value=float(majority_class), dtype=float)
            pred_proba_test = np.full(shape=n_test, fill_value=float(majority_class), dtype=float)
            pred_class_train = np.full(shape=n_train, fill_value=majority_class, dtype=int)
            pred_class_valid = np.full(shape=n_valid, fill_value=majority_class, dtype=int)
            pred_class_test = np.full(shape=n_test, fill_value=majority_class, dtype=int)
        else:
            raise ValueError(f"Unsupported strategy '{strategy}' for mode '{clean_mode}'.")

        proba_series = pd.Series(index=model_rows.index, dtype=float)
        class_series = pd.Series(index=model_rows.index, dtype="int64")
        proba_series.loc[train_mask] = pred_proba_train
        proba_series.loc[valid_mask] = pred_proba_valid
        proba_series.loc[test_mask] = pred_proba_test
        class_series.loc[train_mask] = pred_class_train
        class_series.loc[valid_mask] = pred_class_valid
        class_series.loc[test_mask] = pred_class_test

        predictions["pred_proba"] = proba_series.to_numpy(dtype=float)
        predictions["pred_class"] = class_series.to_numpy(dtype=int)
        predictions["prob_residual"] = (
            predictions["target_value"].to_numpy(dtype=float)
            - predictions["pred_proba"].to_numpy(dtype=float)
        )
        predictions["error_flag"] = (
            predictions["pred_class"].to_numpy(dtype=int)
            != predictions["target_value"].to_numpy(dtype=int)
        )
        predictions["run_id"] = run_id

        metrics_train = _classification_metrics(y_train.astype(int), pred_proba_train, pred_class_train)
        metrics_valid = _classification_metrics(y_valid.astype(int), pred_proba_valid, pred_class_valid)
        metrics_test = _classification_metrics(y_test.astype(int), pred_proba_test, pred_class_test)

        class_balance_payload = {
            "train": _class_balance(y_train.astype(int)),
            "valid": _class_balance(y_valid.astype(int)),
            "test": _class_balance(y_test.astype(int)),
        }

        artifact_prefix = "dummy_classifier"
        model_name = "dummy_classifier_baseline"
        target_type = TARGET_TYPE_BY_MODE[clean_mode]
        learned_stat_field = {
            "train_target_statistic_name": "positive_rate_train" if strategy == "prior" else "majority_class_train",
            "train_target_statistic_value": learned_statistic,
        }

    else:
        raise ValueError(f"Unsupported mode '{clean_mode}'.")

    predictions = predictions.sort_values(["date", "instrument_id"]).reset_index(drop=True)

    config_hash = _config_hash(
        {
            "version": MODULE_VERSION,
            "mode": clean_mode,
            "dummy_strategy": strategy,
            "label_name": selected_label_name,
            "horizon_days": None if horizon_days is None else int(horizon_days),
            "split_name": selected_split_name,
            "target_type": target_type,
            "dataset_path": str(dataset_source),
            "learned_statistic": learned_statistic,
        }
    )
    built_ts = datetime.now(UTC).isoformat()

    target_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else (Path(__file__).resolve().parents[1] / "artifacts")
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = write_parquet(
        predictions.assign(config_hash=config_hash, built_ts_utc=built_ts),
        target_dir / f"{artifact_prefix}_predictions.parquet",
        schema_name=f"{artifact_prefix}_predictions_mvp",
        run_id=run_id,
    )

    metrics_payload: dict[str, Any] = {
        "created_at_utc": built_ts,
        "run_id": run_id,
        "config_hash": config_hash,
        "model_name": model_name,
        "mode": clean_mode,
        "dummy_strategy": strategy,
        "label_name": selected_label_name,
        "horizon_days": None if horizon_days is None else int(horizon_days),
        "split_name": selected_split_name,
        "target_type": target_type,
        **learned_stat_field,
        "split_counts_all_roles": {k: int(v) for k, v in role_counts_all.items()},
        "split_counts_modelable_roles": {
            "train": n_train,
            "valid": n_valid,
            "test": n_test,
        },
        "metrics": {
            "train": {"n": n_train, **metrics_train},
            "valid": {"n": n_valid, **metrics_valid},
            "test": {"n": n_test, **metrics_test},
        },
        "preprocessing": {
            "uses_features": False,
            "imputation": "none",
            "standardization": "none",
            "train_only_fit": True,
        },
        "artifacts": {
            "predictions_path": str(predictions_path),
            "dataset_path": str(dataset_source),
        },
    }
    if class_balance_payload is not None:
        metrics_payload["class_balance"] = class_balance_payload

    metrics_path = target_dir / f"{artifact_prefix}_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "dummy_baseline_trained",
        run_id=run_id,
        mode=clean_mode,
        dummy_strategy=strategy,
        learned_statistic=learned_statistic,
        n_train=n_train,
        n_valid=n_valid,
        n_test=n_test,
        metrics_path=str(metrics_path),
    )

    return TrainDummyBaselineResult(
        mode=clean_mode,
        strategy=strategy,
        metrics_path=metrics_path,
        predictions_path=Path(predictions_path),
        learned_statistic=float(learned_statistic),
        n_train=n_train,
        n_valid=n_valid,
        n_test=n_test,
        config_hash=config_hash,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train dummy baseline regressors/classifiers on model_dataset.")
    parser.add_argument("--mode", type=str, choices=SUPPORTED_MODES, required=True)
    parser.add_argument("--model-dataset-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--label-name", type=str, default=None)
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument("--split-name", type=str, default=None)
    parser.add_argument("--dummy-strategy", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = train_dummy_baseline(
        mode=args.mode,
        model_dataset_path=args.model_dataset_path,
        output_dir=args.output_dir,
        label_name=args.label_name,
        horizon_days=args.horizon_days,
        split_name=args.split_name,
        dummy_strategy=args.dummy_strategy,
        run_id=args.run_id,
    )
    print("Dummy baseline trained:")
    print(f"- mode: {result.mode}")
    print(f"- strategy: {result.strategy}")
    print(f"- metrics: {result.metrics_path}")
    print(f"- predictions: {result.predictions_path}")
    print(f"- learned_statistic: {result.learned_statistic}")
    print(f"- split_sizes: train={result.n_train}, valid={result.n_valid}, test={result.n_test}")


if __name__ == "__main__":
    main()
