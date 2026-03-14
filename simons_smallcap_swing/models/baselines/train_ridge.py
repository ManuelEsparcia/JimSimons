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

# Allow direct script execution: `python simons_smallcap_swing/models/baselines/train_ridge.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "ridge_baseline_mvp_v1"
DEFAULT_ALPHA_GRID: tuple[float, ...] = (0.1, 1.0, 10.0)
DEFAULT_LABEL_NAME = "fwd_ret_5d"
DEFAULT_TARGET_TYPE = "continuous_forward_return"
TRAINABLE_SPLIT_ROLES: tuple[str, ...] = ("train", "valid", "test")
DROPPED_SPLIT_ROLES: tuple[str, ...] = ("dropped_by_purge", "dropped_by_embargo")


MODEL_DATASET_INPUT_SCHEMA = DataSchema(
    name="ridge_model_dataset_input_mvp",
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
class TrainRidgeResult:
    metrics_path: Path
    predictions_path: Path
    model_path: Path
    feature_stats_path: Path
    alpha_selected: float
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


def _normalize_alpha_grid(alpha_grid: Iterable[float]) -> tuple[float, ...]:
    vals = sorted({float(v) for v in alpha_grid})
    if not vals:
        raise ValueError("alpha_grid cannot be empty.")
    if any(v <= 0 for v in vals):
        raise ValueError(f"alpha_grid must contain positive values. Received: {vals}")
    return tuple(vals)


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
    r2 = float("nan") if ss_tot == 0 else float(1.0 - ss_res / ss_tot)
    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "pearson_ic": _safe_corr(y_true, y_pred),
        "spearman_ic": _spearman_corr(y_true, y_pred),
    }


def _solve_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> tuple[float, np.ndarray]:
    n, p = x_train.shape
    x_aug = np.column_stack([np.ones(n), x_train])
    penalty = np.eye(p + 1, dtype=float)
    penalty[0, 0] = 0.0  # do not regularize intercept
    lhs = x_aug.T @ x_aug + float(alpha) * penalty
    rhs = x_aug.T @ y_train
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs
    intercept = float(beta[0])
    weights = beta[1:].astype(float)
    return intercept, weights


def _predict(x: np.ndarray, intercept: float, weights: np.ndarray) -> np.ndarray:
    return intercept + x @ weights


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
    if horizon_days is not None:
        filtered = filtered[filtered["horizon_days"] == int(horizon_days)].copy()
    if filtered.empty:
        raise ValueError(
            f"No rows found for label_name='{label_name}'"
            + (f", horizon_days={int(horizon_days)}." if horizon_days is not None else ".")
        )

    split_name_selected = split_name
    unique_splits = sorted(filtered["split_name"].unique().tolist())
    if split_name_selected is None:
        if len(unique_splits) == 1:
            split_name_selected = unique_splits[0]
        else:
            raise ValueError(
                "Multiple split_name values found. Please select one explicitly with --split-name. "
                f"Available: {unique_splits}"
            )
    filtered = filtered[filtered["split_name"] == split_name_selected].copy()
    if filtered.empty:
        raise ValueError(f"No rows left for split_name='{split_name_selected}'.")

    return (
        filtered.sort_values(["date", "instrument_id", "horizon_days", "label_name"]).reset_index(drop=True),
        source,
        str(split_name_selected),
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
        raise ValueError("No numeric feature columns detected for ridge training.")
    return tuple(sorted(features))


def train_ridge_baseline(
    *,
    model_dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    label_name: str = DEFAULT_LABEL_NAME,
    horizon_days: int | None = None,
    split_name: str | None = None,
    alpha_grid: Iterable[float] = DEFAULT_ALPHA_GRID,
    use_standardization: bool = True,
    run_id: str = MODULE_VERSION,
) -> TrainRidgeResult:
    logger = get_logger("models.baselines.train_ridge")
    alphas = _normalize_alpha_grid(alpha_grid)
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

    x_all = model_rows.loc[:, feature_cols].copy()
    y_all = pd.to_numeric(model_rows["target_value"], errors="coerce")
    if y_all.isna().any():
        raise ValueError("target_value contains invalid values in modelable rows.")

    train_mask = model_rows["split_role"] == "train"
    valid_mask = model_rows["split_role"] == "valid"
    test_mask = model_rows["split_role"] == "test"

    x_train = x_all.loc[train_mask].copy()
    x_valid = x_all.loc[valid_mask].copy()
    x_test = x_all.loc[test_mask].copy()
    y_train = y_all.loc[train_mask].to_numpy(dtype=float)
    y_valid = y_all.loc[valid_mask].to_numpy(dtype=float)
    y_test = y_all.loc[test_mask].to_numpy(dtype=float)

    missingness_before = {
        col: float(x_all[col].isna().mean()) for col in feature_cols
    }
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

    # Select alpha by valid MSE using train-only fitted preprocessing.
    alpha_results: list[dict[str, float]] = []
    best_alpha = float(alphas[0])
    best_valid_mse = float("inf")
    best_intercept = 0.0
    best_weights = np.zeros(len(feature_cols_used), dtype=float)

    for alpha in alphas:
        intercept, weights = _solve_ridge(x_train_np, y_train, alpha=float(alpha))
        pred_valid = _predict(x_valid_np, intercept, weights)
        valid_mse = float(np.mean(np.square(y_valid - pred_valid)))
        alpha_results.append({"alpha": float(alpha), "valid_mse": valid_mse})
        if valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            best_alpha = float(alpha)
            best_intercept = float(intercept)
            best_weights = weights.astype(float)

    pred_train = _predict(x_train_np, best_intercept, best_weights)
    pred_valid = _predict(x_valid_np, best_intercept, best_weights)
    pred_test = _predict(x_test_np, best_intercept, best_weights)

    metrics_train = _regression_metrics(y_train, pred_train)
    metrics_valid = _regression_metrics(y_valid, pred_valid)
    metrics_test = _regression_metrics(y_test, pred_test)

    pred_series = pd.Series(index=model_rows.index, dtype=float)
    pred_series.loc[train_mask] = pred_train
    pred_series.loc[valid_mask] = pred_valid
    pred_series.loc[test_mask] = pred_test

    preds = model_rows[
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
    preds["prediction"] = pred_series.to_numpy(dtype=float)
    preds["residual"] = preds["target_value"].to_numpy(dtype=float) - preds["prediction"].to_numpy(dtype=float)
    preds["run_id"] = run_id
    preds = preds.sort_values(["date", "instrument_id"]).reset_index(drop=True)

    config_hash = _config_hash(
        {
            "version": MODULE_VERSION,
            "label_name": str(label_name),
            "horizon_days": None if horizon_days is None else int(horizon_days),
            "split_name": selected_split_name,
            "alpha_grid": list(alphas),
            "alpha_selected": best_alpha,
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
        preds.assign(config_hash=config_hash, built_ts_utc=built_ts),
        target_dir / "ridge_baseline_predictions.parquet",
        schema_name="ridge_baseline_predictions_mvp",
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
    feature_stats_path = target_dir / "ridge_baseline_feature_stats.json"
    feature_stats_path.write_text(json.dumps(feature_stats_payload, indent=2, sort_keys=True), encoding="utf-8")

    model_payload = {
        "version": MODULE_VERSION,
        "run_id": run_id,
        "config_hash": config_hash,
        "label_name": str(label_name),
        "horizon_days": None if horizon_days is None else int(horizon_days),
        "split_name": selected_split_name,
        "alpha_selected": float(best_alpha),
        "intercept": float(best_intercept),
        "weights": best_weights.astype(float).tolist(),
        "feature_names": list(feature_cols_used),
        "imputer_median": {k: float(v) for k, v in train_median.to_dict().items()},
        "scaler_mean": {k: float(v) for k, v in train_mean.to_dict().items()},
        "scaler_std": {k: float(v) for k, v in train_std.to_dict().items()},
        "use_standardization": bool(use_standardization),
    }
    model_path = target_dir / "ridge_baseline_model.pkl"
    with model_path.open("wb") as fh:
        pickle.dump(model_payload, fh)

    metrics_payload = {
        "created_at_utc": built_ts,
        "run_id": run_id,
        "config_hash": config_hash,
        "model_name": "ridge_baseline",
        "label_name": str(label_name),
        "horizon_days": None if horizon_days is None else int(horizon_days),
        "split_name": selected_split_name,
        "target_type": DEFAULT_TARGET_TYPE,
        "alpha_grid": list(alphas),
        "alpha_selected": float(best_alpha),
        "n_features_used": int(len(feature_cols_used)),
        "feature_names_used": list(feature_cols_used),
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
    metrics_path = target_dir / "ridge_baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "ridge_baseline_trained",
        run_id=run_id,
        alpha_selected=best_alpha,
        n_features_used=int(len(feature_cols_used)),
        n_train=n_train,
        n_valid=n_valid,
        n_test=n_test,
        metrics_path=str(metrics_path),
    )

    return TrainRidgeResult(
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        model_path=model_path,
        feature_stats_path=feature_stats_path,
        alpha_selected=float(best_alpha),
        n_features_used=int(len(feature_cols_used)),
        n_train=n_train,
        n_valid=n_valid,
        n_test=n_test,
        config_hash=config_hash,
    )


def _parse_alphas(text: str) -> tuple[float, ...]:
    items = [part.strip() for part in text.split(",") if part.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one alpha value.")
    try:
        return _normalize_alpha_grid(float(item) for item in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate Ridge baseline on model_dataset.")
    parser.add_argument("--model-dataset-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--label-name", type=str, default=DEFAULT_LABEL_NAME)
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument("--split-name", type=str, default=None)
    parser.add_argument("--alphas", type=_parse_alphas, default=DEFAULT_ALPHA_GRID)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = train_ridge_baseline(
        model_dataset_path=args.model_dataset_path,
        output_dir=args.output_dir,
        label_name=args.label_name,
        horizon_days=args.horizon_days,
        split_name=args.split_name,
        alpha_grid=args.alphas,
        use_standardization=not args.no_standardize,
        run_id=args.run_id,
    )
    print("Ridge baseline trained:")
    print(f"- metrics: {result.metrics_path}")
    print(f"- predictions: {result.predictions_path}")
    print(f"- model: {result.model_path}")
    print(f"- feature_stats: {result.feature_stats_path}")
    print(f"- alpha_selected: {result.alpha_selected}")
    print(f"- n_features_used: {result.n_features_used}")
    print(f"- split_sizes: train={result.n_train}, valid={result.n_valid}, test={result.n_test}")


if __name__ == "__main__":
    main()
