from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

# Allow direct script execution: `python simons_smallcap_swing/models/baselines/model_comparison.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simons_core.io.parquet_store import write_parquet
from simons_core.logging import get_logger


MODULE_VERSION = "model_comparison_mvp_v1"
REGRESSION_TARGET_TYPE = "continuous_forward_return"
CLASSIFICATION_TARGET_TYPE = "binary_direction"
TRAINABLE_SPLIT_ROLES: tuple[str, ...] = ("train", "valid", "test")

METRIC_CATALOG: dict[str, tuple[str, ...]] = {
    REGRESSION_TARGET_TYPE: ("mse", "mae", "r2", "pearson_ic", "spearman_ic"),
    CLASSIFICATION_TARGET_TYPE: (
        "log_loss",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ),
}
PRIMARY_METRIC: dict[str, str] = {
    REGRESSION_TARGET_TYPE: "mse",
    CLASSIFICATION_TARGET_TYPE: "log_loss",
}
METRIC_DIRECTION: dict[str, dict[str, str]] = {
    REGRESSION_TARGET_TYPE: {
        "mse": "min",
        "mae": "min",
        "r2": "max",
        "pearson_ic": "max",
        "spearman_ic": "max",
    },
    CLASSIFICATION_TARGET_TYPE: {
        "log_loss": "min",
        "accuracy": "max",
        "balanced_accuracy": "max",
        "precision": "max",
        "recall": "max",
        "f1": "max",
        "roc_auc": "max",
        "average_precision": "max",
    },
}


@dataclass(frozen=True)
class ModelComparisonResult:
    summary_path: Path
    table_path: Path
    comparability_status: str
    winner_valid: str
    winner_test: str
    primary_metric_used: str | None


def _read_metrics(path: str | Path) -> dict[str, Any]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {file_path}")
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in metrics file: {file_path}") from exc
    required = {"model_name", "label_name", "split_name", "target_type", "metrics"}
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise ValueError(f"Metrics file {file_path} missing required keys: {missing}")
    if not isinstance(payload["metrics"], dict):
        raise ValueError(f"Metrics file {file_path} has invalid 'metrics' section.")
    payload["_path"] = str(file_path)
    return payload


def _split_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts_raw = payload.get("split_counts_modelable_roles", {})
    counts: dict[str, int] = {}
    if isinstance(counts_raw, dict):
        for role in TRAINABLE_SPLIT_ROLES:
            value = counts_raw.get(role)
            if value is not None:
                counts[role] = int(value)
    if len(counts) == len(TRAINABLE_SPLIT_ROLES):
        return counts
    metrics = payload.get("metrics", {})
    for role in TRAINABLE_SPLIT_ROLES:
        role_payload = metrics.get(role, {})
        n_value = role_payload.get("n")
        if n_value is None:
            continue
        counts[role] = int(n_value)
    return counts


def _normalize_horizon(value: Any) -> int | None:
    if value is None:
        return None
    if value == "":
        return None
    return int(value)


def _to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _metric_value(payload: dict[str, Any], split_role: str, metric_name: str) -> float:
    metrics = payload.get("metrics", {})
    split_payload = metrics.get(split_role, {})
    if not isinstance(split_payload, dict):
        return float("nan")
    value = split_payload.get(metric_name)
    try:
        return _to_float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_nan(value: float) -> bool:
    return pd.isna(value)


def _comparability_checks(a: dict[str, Any], b: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if str(a["label_name"]) != str(b["label_name"]):
        reasons.append("label_name mismatch")
    if str(a["split_name"]) != str(b["split_name"]):
        reasons.append("split_name mismatch")

    horizon_a = _normalize_horizon(a.get("horizon_days"))
    horizon_b = _normalize_horizon(b.get("horizon_days"))
    if horizon_a != horizon_b:
        reasons.append("horizon_days mismatch")

    target_a = str(a["target_type"])
    target_b = str(b["target_type"])
    if target_a != target_b:
        reasons.append("target_type mismatch")
    elif target_a not in METRIC_CATALOG:
        reasons.append(f"unsupported target_type '{target_a}'")

    counts_a = _split_counts(a)
    counts_b = _split_counts(b)
    if len(counts_a) == len(TRAINABLE_SPLIT_ROLES) and len(counts_b) == len(TRAINABLE_SPLIT_ROLES):
        for role in TRAINABLE_SPLIT_ROLES:
            left = int(counts_a[role])
            right = int(counts_b[role])
            tolerance = max(1, int(round(max(left, right) * 0.05)))
            if abs(left - right) > tolerance:
                reasons.append(f"split count mismatch on role '{role}' ({left} vs {right})")
    else:
        reasons.append("missing split counts for modelable roles")

    status = "comparable" if not reasons else "non_comparable"
    return status, reasons


def _winner(
    *,
    value_a: float,
    value_b: float,
    direction: str,
    tie_tolerance: float,
) -> str:
    if _is_nan(value_a) or _is_nan(value_b):
        return "non_comparable"
    if abs(value_a - value_b) <= tie_tolerance:
        return "tie"
    if direction == "min":
        return "model_a" if value_a < value_b else "model_b"
    if direction == "max":
        return "model_a" if value_a > value_b else "model_b"
    raise ValueError(f"Unsupported metric direction: {direction}")


def _winner_model_name(
    winner: str,
    model_name_a: str,
    model_name_b: str,
) -> str | None:
    if winner == "model_a":
        return model_name_a
    if winner == "model_b":
        return model_name_b
    return None


def _build_metric_table(
    *,
    model_a: dict[str, Any],
    model_b: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_slot, payload in (("model_a", model_a), ("model_b", model_b)):
        target_type = str(payload["target_type"])
        metrics_to_use = METRIC_CATALOG.get(target_type, ())
        directions = METRIC_DIRECTION.get(target_type, {})
        if not metrics_to_use:
            # Keep an auditable table even when the target type is not supported.
            rows.append(
                {
                    "model_slot": model_slot,
                    "model_name": str(payload["model_name"]),
                    "label_name": str(payload["label_name"]),
                    "target_type": target_type,
                    "split_name": str(payload["split_name"]),
                    "horizon_days": _normalize_horizon(payload.get("horizon_days")),
                    "split_role": "valid",
                    "metric_name": "unsupported_target_type",
                    "metric_value": float("nan"),
                    "metric_direction": None,
                }
            )
            continue
        for split_role in TRAINABLE_SPLIT_ROLES:
            for metric_name in metrics_to_use:
                rows.append(
                    {
                        "model_slot": model_slot,
                        "model_name": str(payload["model_name"]),
                        "label_name": str(payload["label_name"]),
                        "target_type": str(payload["target_type"]),
                        "split_name": str(payload["split_name"]),
                        "horizon_days": _normalize_horizon(payload.get("horizon_days")),
                        "split_role": split_role,
                        "metric_name": metric_name,
                        "metric_value": _metric_value(payload, split_role, metric_name),
                        "metric_direction": directions.get(metric_name),
                    }
                )
    return pd.DataFrame(rows)


def compare_baseline_metrics(
    *,
    model_a_metrics_path: str | Path,
    model_b_metrics_path: str | Path,
    output_dir: str | Path | None = None,
    run_id: str = MODULE_VERSION,
    tie_tolerance: float = 1e-12,
) -> ModelComparisonResult:
    logger = get_logger("models.baselines.model_comparison")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be >= 0.")

    model_a = _read_metrics(model_a_metrics_path)
    model_b = _read_metrics(model_b_metrics_path)

    comparability_status, reasons = _comparability_checks(model_a, model_b)
    target_type = str(model_a["target_type"]) if comparability_status == "comparable" else None
    primary_metric = PRIMARY_METRIC.get(target_type) if target_type is not None else None

    winner_valid = "non_comparable"
    winner_test = "non_comparable"
    primary_metric_values: dict[str, dict[str, float | None]] = {
        "valid": {"model_a": None, "model_b": None},
        "test": {"model_a": None, "model_b": None},
    }

    if comparability_status == "comparable" and primary_metric is not None:
        direction = METRIC_DIRECTION[target_type][primary_metric]
        valid_a = _metric_value(model_a, "valid", primary_metric)
        valid_b = _metric_value(model_b, "valid", primary_metric)
        test_a = _metric_value(model_a, "test", primary_metric)
        test_b = _metric_value(model_b, "test", primary_metric)

        primary_metric_values = {
            "valid": {"model_a": valid_a, "model_b": valid_b},
            "test": {"model_a": test_a, "model_b": test_b},
        }
        winner_valid = _winner(
            value_a=valid_a,
            value_b=valid_b,
            direction=direction,
            tie_tolerance=tie_tolerance,
        )
        winner_test = _winner(
            value_a=test_a,
            value_b=test_b,
            direction=direction,
            tie_tolerance=tie_tolerance,
        )
        if winner_valid == "non_comparable" or winner_test == "non_comparable":
            comparability_status = "non_comparable"
            reasons.append("primary metric missing/invalid on valid or test split")

    output_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else (Path(__file__).resolve().parents[1] / "artifacts")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    metric_table = _build_metric_table(model_a=model_a, model_b=model_b)
    built_ts = datetime.now(UTC).isoformat()
    table_path = write_parquet(
        metric_table.assign(run_id=run_id, built_ts_utc=built_ts),
        output_root / "model_comparison_table.parquet",
        schema_name="model_comparison_table_mvp",
        run_id=run_id,
    )

    summary_payload: dict[str, Any] = {
        "created_at_utc": built_ts,
        "run_id": run_id,
        "model_names": {
            "model_a": str(model_a["model_name"]),
            "model_b": str(model_b["model_name"]),
        },
        "label_names": {
            "model_a": str(model_a["label_name"]),
            "model_b": str(model_b["label_name"]),
        },
        "target_types": {
            "model_a": str(model_a["target_type"]),
            "model_b": str(model_b["target_type"]),
        },
        "split_names": {
            "model_a": str(model_a["split_name"]),
            "model_b": str(model_b["split_name"]),
        },
        "horizon_days": {
            "model_a": _normalize_horizon(model_a.get("horizon_days")),
            "model_b": _normalize_horizon(model_b.get("horizon_days")),
        },
        "split_counts_modelable_roles": {
            "model_a": _split_counts(model_a),
            "model_b": _split_counts(model_b),
        },
        "metrics_paths": {
            "model_a": str(model_a["_path"]),
            "model_b": str(model_b["_path"]),
        },
        "comparability_status": comparability_status,
        "comparison_status": comparability_status,
        "notes": reasons,
        "primary_metric_used": primary_metric,
        "primary_metric_values": primary_metric_values,
        "winner_valid": winner_valid,
        "winner_test": winner_test,
        "winner_valid_model_name": _winner_model_name(
            winner=winner_valid,
            model_name_a=str(model_a["model_name"]),
            model_name_b=str(model_b["model_name"]),
        ),
        "winner_test_model_name": _winner_model_name(
            winner=winner_test,
            model_name_a=str(model_a["model_name"]),
            model_name_b=str(model_b["model_name"]),
        ),
        "metric_catalog_compared": list(METRIC_CATALOG.get(target_type, ())),
        "comparison_policy": {
            "winner_values": ["model_a", "model_b", "tie", "non_comparable"],
            "tie_tolerance": float(tie_tolerance),
            "primary_metric_by_target_type": PRIMARY_METRIC,
            "direction_by_metric": METRIC_DIRECTION,
        },
        "artifacts": {
            "model_comparison_table_path": str(table_path),
        },
    }

    summary_path = output_root / "model_comparison_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "model_comparison_completed",
        run_id=run_id,
        comparability_status=comparability_status,
        winner_valid=winner_valid,
        winner_test=winner_test,
        summary_path=str(summary_path),
    )

    return ModelComparisonResult(
        summary_path=summary_path,
        table_path=Path(table_path),
        comparability_status=comparability_status,
        winner_valid=winner_valid,
        winner_test=winner_test,
        primary_metric_used=primary_metric,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare baseline model metric artifacts.")
    parser.add_argument("--model-a-metrics-path", type=str, required=True)
    parser.add_argument("--model-b-metrics-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = compare_baseline_metrics(
        model_a_metrics_path=args.model_a_metrics_path,
        model_b_metrics_path=args.model_b_metrics_path,
        output_dir=args.output_dir,
        run_id=args.run_id,
        tie_tolerance=float(args.tie_tolerance),
    )
    print("Model comparison completed:")
    print(f"- summary: {result.summary_path}")
    print(f"- table: {result.table_path}")
    print(f"- comparability_status: {result.comparability_status}")
    print(f"- winner_valid: {result.winner_valid}")
    print(f"- winner_test: {result.winner_test}")
    print(f"- primary_metric_used: {result.primary_metric_used}")


if __name__ == "__main__":
    main()
