from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import pickle
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

# Allow direct script execution: `python simons_smallcap_swing/models/baselines/train_logistic.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "logistic_baseline_mvp_v1"
DEFAULT_C_GRID: tuple[float, ...] = (0.1, 1.0, 10.0)
DEFAULT_LABEL_NAME = "fwd_dir_up_5d"
DEFAULT_TARGET_TYPE = "binary_direction"
TRAINABLE_SPLIT_ROLES: tuple[str, ...] = ("train", "valid", "test")
DROPPED_SPLIT_ROLES: tuple[str, ...] = ("dropped_by_purge", "dropped_by_embargo")

MODEL_DATASET_INPUT_SCHEMA = DataSchema(
    name="logistic_model_dataset_input_mvp",
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
    ),
    primary_key=("date", "instrument_id", "horizon_days", "label_name"),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class TrainLogisticResult:
    metrics_path: Path
    predictions_path: Path
    model_path: Path
    feature_stats_path: Path
    c_selected: float
    n_features_used: int
    n_train: int
    n_valid: int
    n_test: int
    config_hash: str


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains invalid dates.")
    return parsed.dt.normalize()


def _normalize_c_grid(c_grid: Iterable[float]) -> tuple[float, ...]:
    values = sorted({float(v) for v in c_grid})
    if not values:
        raise ValueError("c_grid cannot be empty.")
    if any(v <= 0 for v in values):
        raise ValueError(f"All C values must be > 0. Received: {values}")
    return tuple(values)


def _config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


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


def _load_and_filter_dataset(
    *,
    model_dataset_path: str | Path | None,
    label_name: str,
    horizon_days: int | None,
    split_name: str | None,
) -> tuple[pd.DataFrame, Path, str]:
    default_source = Path(__file__).resolve().parents[2] / "data" / "datasets" / "model_dataset.parquet"
    source = (
        Path(model_dataset_path).expanduser().resolve()
        if model_dataset_path
        else default_source
    )
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
        if str(label_name) == DEFAULT_LABEL_NAME:
            raise ValueError(
                "Label 'fwd_dir_up_5d' not found in model_dataset. "
                "Regenerate labels with binary direction enabled (build_labels --include-binary-direction) "
                "and rebuild model_dataset selecting label_name=fwd_dir_up_5d."
            )
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

    return (
        filtered.sort_values(["date", "instrument_id", "horizon_days", "label_name"]).reset_index(drop=True),
        source,
        str(selected_split_name),
    )


def _detect_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    excluded = {
        "date",
        "instrument_id",
        "ticker",
        "horizon_days",
        "label_name",
        "split_name",
        "split_role",
        "entry_date",
        "exit_date",
        "target_value",
        "target_type",
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
        raise ValueError("No numeric feature columns detected for logistic baseline.")
    return tuple(sorted(features))


def train_logistic_baseline(
    *,
    model_dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    label_name: str = DEFAULT_LABEL_NAME,
    horizon_days: int | None = None,
    split_name: str | None = None,
    c_grid: Iterable[float] = DEFAULT_C_GRID,
    use_standardization: bool = True,
    run_id: str = MODULE_VERSION,
) -> TrainLogisticResult:
    logger = get_logger("models.baselines.train_logistic")
    c_values = _normalize_c_grid(c_grid)

    dataset, dataset_source, selected_split_name = _load_and_filter_dataset(
        model_dataset_path=model_dataset_path,
        label_name=label_name,
        horizon_days=horizon_days,
        split_name=split_name,
    )

    feature_cols = _detect_feature_columns(dataset)
    role_counts_all = dataset["split_role"].value_counts().to_dict()
    model_rows = dataset[dataset["split_role"].isin(TRAINABLE_SPLIT_ROLES)].copy()
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

    y_all = pd.to_numeric(model_rows["target_value"], errors="coerce")
    if y_all.isna().any():
        raise ValueError("target_value contains invalid values in modelable rows.")
    unique_labels = sorted(set(y_all.astype(float).unique().tolist()))
    if any(item not in {0.0, 1.0} for item in unique_labels):
        raise ValueError(
            f"Expected binary target values in {{0,1}} for logistic baseline. Found: {unique_labels}"
        )

    x_all = model_rows.loc[:, feature_cols].copy()
    train_mask = model_rows["split_role"] == "train"
    valid_mask = model_rows["split_role"] == "valid"
    test_mask = model_rows["split_role"] == "test"

    x_train = x_all.loc[train_mask].copy()
    x_valid = x_all.loc[valid_mask].copy()
    x_test = x_all.loc[test_mask].copy()
    y_train = y_all.loc[train_mask].to_numpy(dtype=int)
    y_valid = y_all.loc[valid_mask].to_numpy(dtype=int)
    y_test = y_all.loc[test_mask].to_numpy(dtype=int)

    if len(np.unique(y_train)) < 2:
        raise ValueError("Train split must contain both binary classes for logistic regression.")

    missingness_before = {col: float(x_all[col].isna().mean()) for col in feature_cols}
    train_median = x_train.median(axis=0, skipna=True)
    all_nan_features = sorted([col for col in feature_cols if pd.isna(train_median[col])])
    feature_cols_used = [col for col in feature_cols if col not in all_nan_features]
    if not feature_cols_used:
        raise ValueError("All candidate features are all-NaN in train split.")

    x_train = x_train[feature_cols_used].copy()
    x_valid = x_valid[feature_cols_used].copy()
    x_test = x_test[feature_cols_used].copy()
    train_median = train_median[feature_cols_used]

    x_train_imp = x_train.fillna(train_median)
    x_valid_imp = x_valid.fillna(train_median)
    x_test_imp = x_test.fillna(train_median)

    train_mean = x_train_imp.mean(axis=0)
    train_std = x_train_imp.std(axis=0, ddof=0).replace(0.0, 1.0)
    if use_standardization:
        x_train_proc = (x_train_imp - train_mean) / train_std
        x_valid_proc = (x_valid_imp - train_mean) / train_std
        x_test_proc = (x_test_imp - train_mean) / train_std
    else:
        x_train_proc = x_train_imp.copy()
        x_valid_proc = x_valid_imp.copy()
        x_test_proc = x_test_imp.copy()
        train_mean = pd.Series(0.0, index=feature_cols_used, dtype=float)
        train_std = pd.Series(1.0, index=feature_cols_used, dtype=float)

    x_train_np = x_train_proc.to_numpy(dtype=float)
    x_valid_np = x_valid_proc.to_numpy(dtype=float)
    x_test_np = x_test_proc.to_numpy(dtype=float)
    if not np.isfinite(x_train_np).all() or not np.isfinite(x_valid_np).all() or not np.isfinite(x_test_np).all():
        raise ValueError("Non-finite values detected after preprocessing.")

    c_results: list[dict[str, float]] = []
    best_c = float(c_values[0])
    best_valid_log_loss = float("inf")
    best_model: LogisticRegression | None = None

    for c_value in c_values:
        model = LogisticRegression(
            C=float(c_value),
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            fit_intercept=True,
            random_state=42,
        )
        model.fit(x_train_np, y_train)
        valid_proba = model.predict_proba(x_valid_np)[:, 1]
        valid_log_loss = float(log_loss(y_valid, valid_proba, labels=[0, 1]))
        c_results.append({"C": float(c_value), "valid_log_loss": valid_log_loss})
        if valid_log_loss < best_valid_log_loss:
            best_valid_log_loss = valid_log_loss
            best_c = float(c_value)
            best_model = model

    if best_model is None:
        raise RuntimeError("Failed to fit logistic baseline.")

    train_proba = best_model.predict_proba(x_train_np)[:, 1]
    valid_proba = best_model.predict_proba(x_valid_np)[:, 1]
    test_proba = best_model.predict_proba(x_test_np)[:, 1]
    train_pred = (train_proba >= 0.5).astype(int)
    valid_pred = (valid_proba >= 0.5).astype(int)
    test_pred = (test_proba >= 0.5).astype(int)

    metrics_train = _classification_metrics(y_train, train_proba, train_pred)
    metrics_valid = _classification_metrics(y_valid, valid_proba, valid_pred)
    metrics_test = _classification_metrics(y_test, test_proba, test_pred)

    proba_series = pd.Series(index=model_rows.index, dtype=float)
    class_series = pd.Series(index=model_rows.index, dtype="int64")
    proba_series.loc[train_mask] = train_proba
    proba_series.loc[valid_mask] = valid_proba
    proba_series.loc[test_mask] = test_proba
    class_series.loc[train_mask] = train_pred
    class_series.loc[valid_mask] = valid_pred
    class_series.loc[test_mask] = test_pred

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
    predictions["pred_proba"] = proba_series.to_numpy(dtype=float)
    predictions["pred_class"] = class_series.to_numpy(dtype=int)
    predictions["prob_residual"] = predictions["target_value"].to_numpy(dtype=float) - predictions["pred_proba"].to_numpy(dtype=float)
    predictions["error_flag"] = (
        predictions["pred_class"].to_numpy(dtype=int)
        != predictions["target_value"].to_numpy(dtype=int)
    )
    predictions["run_id"] = run_id
    predictions = predictions.sort_values(["date", "instrument_id"]).reset_index(drop=True)

    config_hash = _config_hash(
        {
            "version": MODULE_VERSION,
            "label_name": str(label_name),
            "horizon_days": None if horizon_days is None else int(horizon_days),
            "split_name": selected_split_name,
            "c_grid": list(c_values),
            "c_selected": best_c,
            "use_standardization": bool(use_standardization),
            "features_used": feature_cols_used,
            "dataset_path": str(dataset_source),
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
        target_dir / "logistic_baseline_predictions.parquet",
        schema_name="logistic_baseline_predictions_mvp",
        run_id=run_id,
    )

    feature_stats_payload = {
        "created_at_utc": built_ts,
        "run_id": run_id,
        "config_hash": config_hash,
        "features_candidate": list(feature_cols),
        "features_used": list(feature_cols_used),
        "features_dropped_all_nan_train": all_nan_features,
        "missingness_before_imputation": missingness_before,
        "imputer_median_train_only": {k: float(v) for k, v in train_median.to_dict().items()},
        "scaler_mean_train_only": {k: float(v) for k, v in train_mean.to_dict().items()},
        "scaler_std_train_only": {k: float(v) for k, v in train_std.to_dict().items()},
        "use_standardization": bool(use_standardization),
        "dataset_path": str(dataset_source),
    }
    feature_stats_path = target_dir / "logistic_baseline_feature_stats.json"
    feature_stats_path.write_text(json.dumps(feature_stats_payload, indent=2, sort_keys=True), encoding="utf-8")

    model_payload = {
        "version": MODULE_VERSION,
        "run_id": run_id,
        "config_hash": config_hash,
        "label_name": str(label_name),
        "horizon_days": None if horizon_days is None else int(horizon_days),
        "split_name": selected_split_name,
        "c_selected": best_c,
        "intercept": best_model.intercept_.astype(float).tolist(),
        "coef": best_model.coef_.astype(float).tolist(),
        "classes": best_model.classes_.astype(int).tolist(),
        "feature_names": list(feature_cols_used),
        "imputer_median": {k: float(v) for k, v in train_median.to_dict().items()},
        "scaler_mean": {k: float(v) for k, v in train_mean.to_dict().items()},
        "scaler_std": {k: float(v) for k, v in train_std.to_dict().items()},
        "use_standardization": bool(use_standardization),
    }
    model_path = target_dir / "logistic_baseline_model.pkl"
    with model_path.open("wb") as fh:
        pickle.dump(model_payload, fh)

    metrics_payload = {
        "created_at_utc": built_ts,
        "run_id": run_id,
        "config_hash": config_hash,
        "model_name": "logistic_baseline",
        "label_name": str(label_name),
        "horizon_days": None if horizon_days is None else int(horizon_days),
        "split_name": selected_split_name,
        "target_type": DEFAULT_TARGET_TYPE,
        "c_grid": list(c_values),
        "c_selected": best_c,
        "n_features_used": int(len(feature_cols_used)),
        "feature_names_used": list(feature_cols_used),
        "split_counts_all_roles": {k: int(v) for k, v in role_counts_all.items()},
        "split_counts_modelable_roles": {
            "train": n_train,
            "valid": n_valid,
            "test": n_test,
        },
        "class_balance": {
            "train": _class_balance(y_train),
            "valid": _class_balance(y_valid),
            "test": _class_balance(y_test),
        },
        "metrics": {
            "train": {"n": n_train, **metrics_train},
            "valid": {"n": n_valid, **metrics_valid},
            "test": {"n": n_test, **metrics_test},
        },
        "preprocessing": {
            "imputation": "median_train_only",
            "standardization": "zscore_train_only" if use_standardization else "none",
            "features_dropped_all_nan_train": all_nan_features,
        },
        "artifacts": {
            "predictions_path": str(predictions_path),
            "model_path": str(model_path),
            "feature_stats_path": str(feature_stats_path),
            "dataset_path": str(dataset_source),
        },
    }
    metrics_path = target_dir / "logistic_baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "logistic_baseline_trained",
        run_id=run_id,
        c_selected=best_c,
        n_features_used=int(len(feature_cols_used)),
        n_train=n_train,
        n_valid=n_valid,
        n_test=n_test,
        metrics_path=str(metrics_path),
    )

    return TrainLogisticResult(
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        model_path=model_path,
        feature_stats_path=feature_stats_path,
        c_selected=float(best_c),
        n_features_used=int(len(feature_cols_used)),
        n_train=n_train,
        n_valid=n_valid,
        n_test=n_test,
        config_hash=config_hash,
    )


def _parse_cs(text: str) -> tuple[float, ...]:
    items = [part.strip() for part in text.split(",") if part.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one C value.")
    try:
        return _normalize_c_grid(float(item) for item in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate Logistic baseline on model_dataset.")
    parser.add_argument("--model-dataset-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--label-name", type=str, default=DEFAULT_LABEL_NAME)
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument("--split-name", type=str, default=None)
    parser.add_argument("--cs", type=_parse_cs, default=DEFAULT_C_GRID)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = train_logistic_baseline(
        model_dataset_path=args.model_dataset_path,
        output_dir=args.output_dir,
        label_name=args.label_name,
        horizon_days=args.horizon_days,
        split_name=args.split_name,
        c_grid=args.cs,
        use_standardization=not args.no_standardize,
        run_id=args.run_id,
    )
    print("Logistic baseline trained:")
    print(f"- metrics: {result.metrics_path}")
    print(f"- predictions: {result.predictions_path}")
    print(f"- model: {result.model_path}")
    print(f"- feature_stats: {result.feature_stats_path}")
    print(f"- c_selected: {result.c_selected}")
    print(f"- n_features_used: {result.n_features_used}")
    print(f"- split_sizes: train={result.n_train}, valid={result.n_valid}, test={result.n_test}")


if __name__ == "__main__":
    main()
