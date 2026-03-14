from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from itertools import combinations
import json
from math import log
from pathlib import Path
import random
import sys
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "pbo_cscv_mvp_v1"
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
STATUS_RANK = {PASS: 0, WARN: 1, FAIL: 2}

TASK_CONFIG = {
    "regression_candidates": {
        "target_type": "continuous_forward_return",
        "primary_metric": "mse",
        "models": ("ridge_cv", "dummy_regressor_cv"),
    },
    "classification_candidates": {
        "target_type": "binary_direction",
        "primary_metric": "log_loss",
        "models": ("logistic_cv", "dummy_classifier_cv"),
    },
}

INPUT_REQUIRED_COLUMNS = {
    "model_name",
    "fold_id",
    "label_name",
    "horizon_days",
    "target_type",
    "primary_metric",
    "valid_primary_metric",
}

RESULTS_SCHEMA = DataSchema(
    name="pbo_cscv_results_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("task_name", "string", nullable=False),
        ColumnSpec("partition_id", "int64", nullable=False),
        ColumnSpec("candidate_name", "string", nullable=False),
        ColumnSpec("target_type", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("horizon_days", "int64", nullable=False),
        ColumnSpec("primary_metric", "string", nullable=False),
        ColumnSpec("in_sample_metric", "float64", nullable=False),
        ColumnSpec("out_of_sample_metric", "float64", nullable=False),
        ColumnSpec("is_in_sample_winner", "bool", nullable=False),
        ColumnSpec("oos_rank", "int64", nullable=False),
        ColumnSpec("oos_percentile", "float64", nullable=False),
        ColumnSpec("overfit_flag", "int64", nullable=True),
    ),
    primary_key=(),
    allow_extra_columns=True,
)

PARTITIONS_SCHEMA = DataSchema(
    name="pbo_cscv_partitions_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("task_name", "string", nullable=False),
        ColumnSpec("partition_id", "int64", nullable=False),
        ColumnSpec("is_fold_ids_json", "string", nullable=False),
        ColumnSpec("oos_fold_ids_json", "string", nullable=False),
        ColumnSpec("winner_candidate_name", "string", nullable=False),
        ColumnSpec("winner_oos_rank", "int64", nullable=False),
        ColumnSpec("winner_oos_percentile", "float64", nullable=False),
        ColumnSpec("winner_logit", "float64", nullable=False),
        ColumnSpec("overfit_flag", "int64", nullable=False),
        ColumnSpec("n_candidates", "int64", nullable=False),
    ),
    primary_key=(),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class PboCscvResult:
    results_path: Path
    summary_path: Path
    partitions_path: Path
    overall_status: str
    config_hash: str


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def _cfg_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _worst_status(values: list[str]) -> str:
    if not values:
        return WARN
    return max(values, key=lambda v: STATUS_RANK.get(v, -1))


def _load_candidate_frame(path: Path, expected_model_name: str) -> tuple[pd.DataFrame | None, str | None]:
    if not path.exists():
        return None, f"missing_artifact:{path}"

    frame = read_parquet(path)
    missing_cols = sorted(INPUT_REQUIRED_COLUMNS - set(frame.columns))
    if missing_cols:
        return None, f"missing_columns:{expected_model_name}:{missing_cols}"

    out = frame.copy()
    out["model_name"] = out["model_name"].astype(str)
    if expected_model_name not in out["model_name"].unique().tolist():
        if len(out["model_name"].unique().tolist()) == 1:
            # Allow aliasing when file is single-model and path-level name is trusted.
            out["model_name"] = expected_model_name
        else:
            return None, f"model_name_mismatch:{expected_model_name}"
    else:
        out = out[out["model_name"] == expected_model_name].copy()

    if "status" in out.columns:
        out = out[out["status"].astype(str) == "completed"].copy()
    if out.empty:
        return None, f"no_completed_rows:{expected_model_name}"

    out["fold_id"] = pd.to_numeric(out["fold_id"], errors="coerce").astype("Int64")
    out["horizon_days"] = pd.to_numeric(out["horizon_days"], errors="coerce").astype("Int64")
    out["valid_primary_metric"] = pd.to_numeric(out["valid_primary_metric"], errors="coerce")
    if out["fold_id"].isna().any():
        return None, f"invalid_fold_id:{expected_model_name}"
    if out["valid_primary_metric"].isna().any():
        return None, f"invalid_valid_primary_metric:{expected_model_name}"
    return out.reset_index(drop=True), None


def _build_partitions(folds: list[int], *, max_partitions: int, seed: int) -> list[tuple[list[int], list[int]]]:
    n_folds = len(folds)
    is_size = n_folds // 2
    if is_size < 1 or (n_folds - is_size) < 1:
        return []
    all_is = [tuple(sorted(part)) for part in combinations(folds, is_size)]
    if len(all_is) > max_partitions:
        rng = random.Random(seed)
        all_is = sorted(rng.sample(all_is, max_partitions))
    partitions: list[tuple[list[int], list[int]]] = []
    fold_set = set(folds)
    for is_tuple in all_is:
        is_folds = sorted(is_tuple)
        oos_folds = sorted(fold_set - set(is_folds))
        if is_folds and oos_folds:
            partitions.append((is_folds, oos_folds))
    return partitions


def _evaluate_task(
    *,
    task_name: str,
    config: dict[str, Any],
    candidates: dict[str, pd.DataFrame],
    max_partitions: int,
    seed: int,
    run_id: str,
    built_ts_utc: str,
    config_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[str]]:
    notes: list[str] = []
    target_type = str(config["target_type"])
    primary_metric = str(config["primary_metric"])
    model_names = list(config["models"])

    task_frames: dict[str, pd.DataFrame] = {}
    for model_name in model_names:
        frame = candidates.get(model_name)
        if frame is None:
            notes.append(f"{task_name}:missing_candidate:{model_name}")
            continue
        frame = frame.copy()
        obs_target = sorted(frame["target_type"].astype(str).dropna().unique().tolist())
        if obs_target != [target_type]:
            notes.append(f"{task_name}:target_type_mismatch:{model_name}:{obs_target}")
            continue
        obs_metric = sorted(frame["primary_metric"].astype(str).dropna().unique().tolist())
        if obs_metric != [primary_metric]:
            notes.append(f"{task_name}:primary_metric_mismatch:{model_name}:{obs_metric}")
            continue
        if frame["label_name"].astype(str).nunique() != 1:
            notes.append(f"{task_name}:label_name_not_unique:{model_name}")
            continue
        if frame["horizon_days"].nunique() != 1:
            notes.append(f"{task_name}:horizon_not_unique:{model_name}")
            continue
        task_frames[model_name] = frame

    if len(task_frames) < 2:
        task_status = WARN
        summary = {
            "task_status": task_status,
            "n_candidates": int(len(task_frames)),
            "n_partitions": 0,
            "pbo_estimate": None,
            "overfit_rate": None,
            "median_oos_percentile_of_is_winner": None,
            "target_type": target_type,
            "primary_metric": primary_metric,
            "label_name": None,
            "horizon_days": None,
        }
        return pd.DataFrame(), pd.DataFrame(), summary, notes

    common_folds = None
    for frame in task_frames.values():
        folds = set(frame["fold_id"].astype(int).tolist())
        common_folds = folds if common_folds is None else (common_folds & folds)
    fold_ids = sorted(common_folds or [])
    if len(fold_ids) < 2:
        notes.append(f"{task_name}:insufficient_common_folds:{len(fold_ids)}")
        summary = {
            "task_status": WARN,
            "n_candidates": int(len(task_frames)),
            "n_partitions": 0,
            "pbo_estimate": None,
            "overfit_rate": None,
            "median_oos_percentile_of_is_winner": None,
            "target_type": target_type,
            "primary_metric": primary_metric,
            "label_name": None,
            "horizon_days": None,
        }
        return pd.DataFrame(), pd.DataFrame(), summary, notes

    norm_frames: dict[str, pd.DataFrame] = {}
    labels = set()
    horizons = set()
    for model_name, frame in task_frames.items():
        subset = frame[frame["fold_id"].astype(int).isin(fold_ids)].copy()
        subset["fold_id"] = subset["fold_id"].astype(int)
        subset = subset.sort_values("fold_id").reset_index(drop=True)
        norm_frames[model_name] = subset
        labels.add(str(subset["label_name"].iloc[0]))
        horizons.add(int(subset["horizon_days"].iloc[0]))
    label_name = sorted(labels)[0] if len(labels) == 1 else None
    horizon_days = sorted(horizons)[0] if len(horizons) == 1 else None
    if label_name is None:
        notes.append(f"{task_name}:label_mismatch_across_candidates")
    if horizon_days is None:
        notes.append(f"{task_name}:horizon_mismatch_across_candidates")
    if label_name is None or horizon_days is None:
        summary = {
            "task_status": WARN,
            "n_candidates": int(len(norm_frames)),
            "n_partitions": 0,
            "pbo_estimate": None,
            "overfit_rate": None,
            "median_oos_percentile_of_is_winner": None,
            "target_type": target_type,
            "primary_metric": primary_metric,
            "label_name": label_name,
            "horizon_days": horizon_days,
        }
        return pd.DataFrame(), pd.DataFrame(), summary, notes

    partitions = _build_partitions(fold_ids, max_partitions=max_partitions, seed=seed)
    if len(partitions) < 1:
        notes.append(f"{task_name}:no_valid_partitions")
        summary = {
            "task_status": WARN,
            "n_candidates": int(len(norm_frames)),
            "n_partitions": 0,
            "pbo_estimate": None,
            "overfit_rate": None,
            "median_oos_percentile_of_is_winner": None,
            "target_type": target_type,
            "primary_metric": primary_metric,
            "label_name": label_name,
            "horizon_days": horizon_days,
        }
        return pd.DataFrame(), pd.DataFrame(), summary, notes

    results_rows: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    winner_rows: list[dict[str, Any]] = []

    candidate_names = sorted(norm_frames.keys())
    n_candidates = len(candidate_names)

    for partition_id, (is_folds, oos_folds) in enumerate(partitions, start=1):
        scores: list[dict[str, Any]] = []
        for candidate_name in candidate_names:
            frame = norm_frames[candidate_name]
            is_metric = float(frame.loc[frame["fold_id"].isin(is_folds), "valid_primary_metric"].mean())
            oos_metric = float(frame.loc[frame["fold_id"].isin(oos_folds), "valid_primary_metric"].mean())
            scores.append(
                {
                    "candidate_name": candidate_name,
                    "in_sample_metric": is_metric,
                    "out_of_sample_metric": oos_metric,
                }
            )

        scores_sorted_is = sorted(scores, key=lambda row: (row["in_sample_metric"], row["candidate_name"]))
        winner_name = str(scores_sorted_is[0]["candidate_name"])

        scores_sorted_oos = sorted(scores, key=lambda row: (row["out_of_sample_metric"], row["candidate_name"]))
        rank_best = {row["candidate_name"]: idx + 1 for idx, row in enumerate(scores_sorted_oos)}
        rank_worst_to_best = {name: int(n_candidates - best + 1) for name, best in rank_best.items()}
        omega = {name: float(rank_worst_to_best[name] / float(n_candidates + 1)) for name in candidate_names}
        winner_omega = omega[winner_name]
        winner_logit = float(log(winner_omega / (1.0 - winner_omega)))
        winner_overfit = int(winner_logit < 0.0)

        for row in scores:
            candidate_name = str(row["candidate_name"])
            is_winner = candidate_name == winner_name
            results_rows.append(
                {
                    "task_name": task_name,
                    "partition_id": int(partition_id),
                    "candidate_name": candidate_name,
                    "target_type": target_type,
                    "label_name": label_name,
                    "horizon_days": int(horizon_days),
                    "primary_metric": primary_metric,
                    "in_sample_metric": float(row["in_sample_metric"]),
                    "out_of_sample_metric": float(row["out_of_sample_metric"]),
                    "is_in_sample_winner": bool(is_winner),
                    "oos_rank": int(rank_worst_to_best[candidate_name]),
                    "oos_percentile": float(omega[candidate_name]),
                    "overfit_flag": int(winner_overfit) if is_winner else None,
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "built_ts_utc": built_ts_utc,
                }
            )

        winner_rows.append(
            {
                "partition_id": int(partition_id),
                "winner_name": winner_name,
                "winner_oos_rank": int(rank_worst_to_best[winner_name]),
                "winner_oos_percentile": float(winner_omega),
                "winner_logit": float(winner_logit),
                "overfit_flag": int(winner_overfit),
            }
        )
        partition_rows.append(
            {
                "task_name": task_name,
                "partition_id": int(partition_id),
                "is_fold_ids_json": json.dumps(is_folds),
                "oos_fold_ids_json": json.dumps(oos_folds),
                "winner_candidate_name": winner_name,
                "winner_oos_rank": int(rank_worst_to_best[winner_name]),
                "winner_oos_percentile": float(winner_omega),
                "winner_logit": float(winner_logit),
                "overfit_flag": int(winner_overfit),
                "n_candidates": int(n_candidates),
                "run_id": run_id,
                "config_hash": config_hash,
                "built_ts_utc": built_ts_utc,
            }
        )

    winners_df = pd.DataFrame(winner_rows)
    pbo_estimate = float(winners_df["overfit_flag"].mean()) if not winners_df.empty else None
    median_oos_percentile = (
        float(winners_df["winner_oos_percentile"].median()) if not winners_df.empty else None
    )
    overfit_rate = pbo_estimate

    if n_candidates < 2 or len(partitions) < 3 or len(fold_ids) < 4:
        task_status = WARN
    elif pbo_estimate is None:
        task_status = WARN
    elif pbo_estimate >= 0.60:
        task_status = FAIL
    elif pbo_estimate >= 0.40:
        task_status = WARN
    else:
        task_status = PASS

    summary = {
        "task_status": task_status,
        "n_candidates": int(n_candidates),
        "n_partitions": int(len(partitions)),
        "pbo_estimate": pbo_estimate,
        "overfit_rate": overfit_rate,
        "median_oos_percentile_of_is_winner": median_oos_percentile,
        "target_type": target_type,
        "primary_metric": primary_metric,
        "label_name": label_name,
        "horizon_days": int(horizon_days),
    }

    return pd.DataFrame(results_rows), pd.DataFrame(partition_rows), summary, notes


def run_pbo_cscv(
    *,
    ridge_cv_path: str | Path | None = None,
    dummy_regressor_cv_path: str | Path | None = None,
    logistic_cv_path: str | Path | None = None,
    dummy_classifier_cv_path: str | Path | None = None,
    max_partitions: int = 64,
    seed: int = 42,
    output_dir: str | Path | None = None,
    run_id: str = MODULE_VERSION,
) -> PboCscvResult:
    logger = get_logger("validation.pbo_cscv")
    base = data_dir()
    artifacts_root = base / "models" / "artifacts"

    input_paths = {
        "ridge_cv": Path(ridge_cv_path).expanduser().resolve() if ridge_cv_path else (artifacts_root / "ridge_cv" / "cv_baseline_fold_metrics.parquet"),
        "dummy_regressor_cv": Path(dummy_regressor_cv_path).expanduser().resolve() if dummy_regressor_cv_path else (artifacts_root / "dummy_regressor_cv" / "cv_baseline_fold_metrics.parquet"),
        "logistic_cv": Path(logistic_cv_path).expanduser().resolve() if logistic_cv_path else (artifacts_root / "logistic_cv" / "cv_baseline_fold_metrics.parquet"),
        "dummy_classifier_cv": Path(dummy_classifier_cv_path).expanduser().resolve() if dummy_classifier_cv_path else (artifacts_root / "dummy_classifier_cv" / "cv_baseline_fold_metrics.parquet"),
    }

    loaded: dict[str, pd.DataFrame] = {}
    notes: list[str] = []
    for model_name, path in input_paths.items():
        frame, err = _load_candidate_frame(path, expected_model_name=model_name)
        if err is not None:
            notes.append(f"{model_name}:{err}")
            continue
        loaded[model_name] = frame

    built_ts_utc = datetime.now(UTC).isoformat()
    cfg_hash = _cfg_hash(
        {
            "module_version": MODULE_VERSION,
            "input_paths": {key: str(value) for key, value in input_paths.items()},
            "max_partitions": int(max_partitions),
            "seed": int(seed),
            "run_id": run_id,
        }
    )

    all_results: list[pd.DataFrame] = []
    all_partitions: list[pd.DataFrame] = []
    task_summary: dict[str, dict[str, Any]] = {}
    task_statuses: list[str] = []

    for task_name, cfg in TASK_CONFIG.items():
        result_df, partitions_df, summary, task_notes = _evaluate_task(
            task_name=task_name,
            config=cfg,
            candidates=loaded,
            max_partitions=max_partitions,
            seed=seed,
            run_id=run_id,
            built_ts_utc=built_ts_utc,
            config_hash=cfg_hash,
        )
        notes.extend(task_notes)
        task_summary[task_name] = summary
        task_statuses.append(str(summary["task_status"]))
        if not result_df.empty:
            all_results.append(result_df)
        if not partitions_df.empty:
            all_partitions.append(partitions_df)

    if not all_results:
        raise ValueError(
            "pbo_cscv could not evaluate any task. "
            "Check candidate fold metric inputs, comparability, and required columns."
        )

    results_df = pd.concat(all_results, ignore_index=True).sort_values(
        ["task_name", "partition_id", "candidate_name"]
    )
    partitions_df = (
        pd.concat(all_partitions, ignore_index=True).sort_values(["task_name", "partition_id"])
        if all_partitions
        else pd.DataFrame(
            columns=[
                "task_name",
                "partition_id",
                "is_fold_ids_json",
                "oos_fold_ids_json",
                "winner_candidate_name",
                "winner_oos_rank",
                "winner_oos_percentile",
                "winner_logit",
                "overfit_flag",
                "n_candidates",
            ]
        )
    )

    # Type normalization before schema checks.
    for col in (
        "task_name",
        "candidate_name",
        "target_type",
        "label_name",
        "primary_metric",
    ):
        results_df[col] = results_df[col].astype("string")
    results_df["partition_id"] = pd.to_numeric(results_df["partition_id"], errors="coerce").astype("int64")
    results_df["horizon_days"] = pd.to_numeric(results_df["horizon_days"], errors="coerce").astype("int64")
    results_df["oos_rank"] = pd.to_numeric(results_df["oos_rank"], errors="coerce").astype("int64")
    results_df["is_in_sample_winner"] = results_df["is_in_sample_winner"].astype(bool)
    for col in ("in_sample_metric", "out_of_sample_metric", "oos_percentile"):
        results_df[col] = pd.to_numeric(results_df[col], errors="coerce")
    results_df["overfit_flag"] = pd.to_numeric(results_df["overfit_flag"], errors="coerce").astype("Int64")

    if results_df.duplicated(
        ["task_name", "partition_id", "candidate_name"], keep=False
    ).any():
        raise ValueError("Duplicate rows detected in pbo_cscv results logical PK.")
    if not results_df["oos_percentile"].between(0.0, 1.0).all():
        raise ValueError("oos_percentile must be in [0,1].")

    if not partitions_df.empty:
        for col in ("task_name", "is_fold_ids_json", "oos_fold_ids_json", "winner_candidate_name"):
            partitions_df[col] = partitions_df[col].astype("string")
        partitions_df["partition_id"] = pd.to_numeric(partitions_df["partition_id"], errors="coerce").astype("int64")
        partitions_df["winner_oos_rank"] = pd.to_numeric(partitions_df["winner_oos_rank"], errors="coerce").astype("int64")
        partitions_df["n_candidates"] = pd.to_numeric(partitions_df["n_candidates"], errors="coerce").astype("int64")
        for col in ("winner_oos_percentile", "winner_logit"):
            partitions_df[col] = pd.to_numeric(partitions_df[col], errors="coerce")
        partitions_df["overfit_flag"] = pd.to_numeric(partitions_df["overfit_flag"], errors="coerce").astype("int64")
        if not partitions_df["winner_oos_percentile"].between(0.0, 1.0).all():
            raise ValueError("winner_oos_percentile must be in [0,1].")

    assert_schema(results_df, RESULTS_SCHEMA)
    if not partitions_df.empty:
        assert_schema(partitions_df, PARTITIONS_SCHEMA)

    overall_status = _worst_status(task_statuses)
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (base / "validation")
    target_dir.mkdir(parents=True, exist_ok=True)

    results_path = write_parquet(
        results_df,
        target_dir / "pbo_cscv_results.parquet",
        schema_name=RESULTS_SCHEMA.name,
        run_id=run_id,
    )

    if partitions_df.empty:
        # Keep artifact always present and non-empty by writing winner rows from results.
        partitions_fallback = (
            results_df[results_df["is_in_sample_winner"]]
            .assign(
                is_fold_ids_json="[]",
                oos_fold_ids_json="[]",
                winner_candidate_name=lambda x: x["candidate_name"].astype(str),
                winner_oos_rank=lambda x: x["oos_rank"].astype(int),
                winner_oos_percentile=lambda x: x["oos_percentile"].astype(float),
                winner_logit=0.0,
                overfit_flag=lambda x: x["overfit_flag"].fillna(0).astype(int),
                n_candidates=1,
            )[
                [
                    "task_name",
                    "partition_id",
                    "is_fold_ids_json",
                    "oos_fold_ids_json",
                    "winner_candidate_name",
                    "winner_oos_rank",
                    "winner_oos_percentile",
                    "winner_logit",
                    "overfit_flag",
                    "n_candidates",
                ]
            ]
        )
        partitions_path = write_parquet(
            partitions_fallback,
            target_dir / "pbo_cscv_partitions.parquet",
            schema_name=PARTITIONS_SCHEMA.name,
            run_id=run_id,
        )
    else:
        partitions_path = write_parquet(
            partitions_df,
            target_dir / "pbo_cscv_partitions.parquet",
            schema_name=PARTITIONS_SCHEMA.name,
            run_id=run_id,
        )

    summary_payload = {
        "built_ts_utc": built_ts_utc,
        "run_id": run_id,
        "module_version": MODULE_VERSION,
        "config_hash": cfg_hash,
        "overall_status": overall_status,
        "tasks_evaluated": [task for task, info in task_summary.items() if info["n_partitions"] > 0],
        "task_status_by_task": {task: info["task_status"] for task, info in task_summary.items()},
        "n_candidates_by_task": {task: info["n_candidates"] for task, info in task_summary.items()},
        "n_partitions": {task: info["n_partitions"] for task, info in task_summary.items()},
        "primary_metric_by_task": {task: info["primary_metric"] for task, info in task_summary.items()},
        "pbo_estimate_by_task": {task: info["pbo_estimate"] for task, info in task_summary.items()},
        "median_oos_percentile_of_is_winner": {
            task: info["median_oos_percentile_of_is_winner"] for task, info in task_summary.items()
        },
        "overfit_rate_by_task": {task: info["overfit_rate"] for task, info in task_summary.items()},
        "notes": notes,
        "input_paths": {key: str(value) for key, value in input_paths.items()},
        "output_paths": {
            "pbo_cscv_results": str(results_path),
            "pbo_cscv_partitions": str(partitions_path),
        },
    }
    summary_path = target_dir / "pbo_cscv_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "pbo_cscv_completed",
        run_id=run_id,
        overall_status=overall_status,
        results_path=str(results_path),
        partitions_path=str(partitions_path),
        summary_path=str(summary_path),
    )
    return PboCscvResult(
        results_path=results_path,
        summary_path=summary_path,
        partitions_path=partitions_path,
        overall_status=overall_status,
        config_hash=cfg_hash,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MVP CSCV/PBO over baseline CV fold metrics by task."
    )
    parser.add_argument("--ridge-cv-path", type=str, default=None)
    parser.add_argument("--dummy-regressor-cv-path", type=str, default=None)
    parser.add_argument("--logistic-cv-path", type=str, default=None)
    parser.add_argument("--dummy-classifier-cv-path", type=str, default=None)
    parser.add_argument("--max-partitions", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_pbo_cscv(
        ridge_cv_path=args.ridge_cv_path,
        dummy_regressor_cv_path=args.dummy_regressor_cv_path,
        logistic_cv_path=args.logistic_cv_path,
        dummy_classifier_cv_path=args.dummy_classifier_cv_path,
        max_partitions=args.max_partitions,
        seed=args.seed,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print("PBO/CSCV completed:")
    print(f"- results: {result.results_path}")
    print(f"- partitions: {result.partitions_path}")
    print(f"- summary: {result.summary_path}")
    print(f"- overall_status: {result.overall_status}")


if __name__ == "__main__":
    main()
