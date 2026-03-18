"""
experiment_tracker/sweep_launcher.py - Governed hyperparameter sweep launcher.

Converts a declared search space into reproducible trial execution with explicit
budgeting, trial-state persistence, deterministic proposal indexing, and candidate
export artifacts.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from simons_smallcap_swing.research._shared import (
    config_hash,
    json_safe,
    run_id_with_prefix,
    stable_params_hash,
    utc_now_iso,
    write_json_safe,
    write_parquet_safe,
)


TRIAL_STATES = {"completed", "failed", "timeout", "cancelled", "invalid"}


@dataclass(frozen=True)
class SweepConfig:
    objective_metric: str
    objective_direction: str = "max"  # max | min
    proposal_policy: str = "grid"  # grid | random
    max_trials: int = 100
    max_parallel_trials: int = 1
    timeout_seconds: int = 120
    max_retries: int = 1
    random_seed: int = 42
    top_k: int = 10
    allow_duplicate_params: bool = False


def _enumerate_grid(search_space: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    keys = sorted(search_space.keys())
    values = [list(search_space[k]) for k in keys]
    proposals = []
    for combo in itertools.product(*values):
        proposals.append({k: v for k, v in zip(keys, combo)})
    return proposals


def _enumerate_random(search_space: Mapping[str, Sequence[Any]], n: int, seed: int) -> list[dict[str, Any]]:
    keys = sorted(search_space.keys())
    values = [list(search_space[k]) for k in keys]
    rng = np.random.RandomState(seed)
    proposals = []
    for _ in range(n):
        proposals.append({k: values[i][int(rng.randint(0, len(values[i])))] for i, k in enumerate(keys)})
    return proposals


def _sort_leaderboard(df: pd.DataFrame, objective_metric: str, direction: str) -> pd.DataFrame:
    asc = direction.lower() == "min"
    return df.sort_values([objective_metric, "trial_id"], ascending=[asc, True]).reset_index(drop=True)


def launch_sweep(
    search_space: Mapping[str, Sequence[Any]],
    trial_runner: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *,
    config: SweepConfig,
    output_dir: str | Path,
    sweep_id: str | None = None,
    parent_run_id: str | None = None,
    constraints_fn: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """
    Execute a governed sweep and persist canonical outputs:
    - trials.parquet
    - leaderboard.parquet
    - best_candidates.json
    - sweep_summary.json
    """
    if config.objective_direction.lower() not in {"max", "min"}:
        raise ValueError("objective_direction must be 'max' or 'min'")
    if config.max_trials <= 0:
        raise ValueError("max_trials must be > 0")

    sid = sweep_id or run_id_with_prefix("sweep")
    parent = parent_run_id or sid
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if config.proposal_policy == "grid":
        proposals = _enumerate_grid(search_space)
    elif config.proposal_policy == "random":
        proposals = _enumerate_random(search_space, config.max_trials, config.random_seed)
    else:
        raise ValueError(f"unsupported proposal_policy: {config.proposal_policy}")

    proposals = proposals[: config.max_trials]
    trial_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    for idx, params in enumerate(proposals):
        trial_id = f"{sid}_trial_{idx:04d}"
        child_run_id = trial_id
        started = utc_now_iso()
        t0 = time.perf_counter()
        retries = 0

        params_json = json_safe(params)
        params_hash = stable_params_hash(params_json, n=24)
        key = (sid, params_hash)

        if not config.allow_duplicate_params and key in seen_keys:
            elapsed = time.perf_counter() - t0
            trial_rows.append(
                {
                    "trial_id": trial_id,
                    "proposal_index": idx,
                    "parent_run_id": parent,
                    "child_run_id": child_run_id,
                    "params_json": params_json,
                    "params_hash": params_hash,
                    "seed_child": config.random_seed + idx,
                    "status": "invalid",
                    "objective_metric": config.objective_metric,
                    "objective_value": float("nan"),
                    "elapsed_seconds": round(float(elapsed), 4),
                    "retry_count": retries,
                    "failure_reason": "duplicate_params_hash",
                    "started_at": started,
                    "finished_at": utc_now_iso(),
                    "batch_id": idx // max(config.max_parallel_trials, 1),
                }
            )
            continue
        seen_keys.add(key)

        if constraints_fn is not None and not bool(constraints_fn(params)):
            elapsed = time.perf_counter() - t0
            trial_rows.append(
                {
                    "trial_id": trial_id,
                    "proposal_index": idx,
                    "parent_run_id": parent,
                    "child_run_id": child_run_id,
                    "params_json": params_json,
                    "params_hash": params_hash,
                    "seed_child": config.random_seed + idx,
                    "status": "invalid",
                    "objective_metric": config.objective_metric,
                    "objective_value": float("nan"),
                    "elapsed_seconds": round(float(elapsed), 4),
                    "retry_count": retries,
                    "failure_reason": "constraints_reject",
                    "started_at": started,
                    "finished_at": utc_now_iso(),
                    "batch_id": idx // max(config.max_parallel_trials, 1),
                }
            )
            continue

        status = "failed"
        objective_value = float("nan")
        failure_reason = ""
        result_payload: Mapping[str, Any] = {}

        for attempt in range(config.max_retries + 1):
            retries = attempt
            try:
                result_payload = trial_runner(params)
                if config.objective_metric not in result_payload:
                    raise ValueError(f"trial result missing objective metric '{config.objective_metric}'")
                objective_value = float(result_payload[config.objective_metric])
                if not math.isfinite(objective_value):
                    raise ValueError("objective metric is not finite")
                status = "completed"
                failure_reason = ""
                break
            except Exception as exc:
                failure_reason = str(exc)
                status = "failed"
                if attempt >= config.max_retries:
                    break

        elapsed = float(time.perf_counter() - t0)
        if status == "completed" and elapsed > config.timeout_seconds:
            status = "timeout"
            failure_reason = "trial exceeded timeout budget"

        if status not in TRIAL_STATES:
            status = "failed"
            failure_reason = f"invalid_status:{status}"

        trial_rows.append(
            {
                "trial_id": trial_id,
                "proposal_index": idx,
                "parent_run_id": parent,
                "child_run_id": child_run_id,
                "params_json": params_json,
                "params_hash": params_hash,
                "seed_child": config.random_seed + idx,
                "status": status,
                "objective_metric": config.objective_metric,
                "objective_value": objective_value,
                "elapsed_seconds": round(elapsed, 4),
                "retry_count": retries,
                "failure_reason": failure_reason,
                "started_at": started,
                "finished_at": utc_now_iso(),
                "batch_id": idx // max(config.max_parallel_trials, 1),
                "result_payload": json_safe(result_payload),
            }
        )

    trials = pd.DataFrame(trial_rows)
    completed = trials[trials["status"] == "completed"].copy()
    leaderboard = _sort_leaderboard(completed, config.objective_metric, config.objective_direction)
    top = leaderboard.head(max(config.top_k, 1)).copy()

    best_candidates = []
    for _, row in top.iterrows():
        best_candidates.append(
            {
                "trial_id": row["trial_id"],
                "proposal_index": int(row["proposal_index"]),
                "objective_metric": config.objective_metric,
                "objective_value": float(row["objective_value"]),
                "params_json": row["params_json"],
                "params_hash": row["params_hash"],
                "status": row["status"],
            }
        )

    terminated_by = "max_trials"
    if len(trials) < config.max_trials:
        terminated_by = "space_exhausted"
    if len(leaderboard) == 0:
        best_value = None
    else:
        best_value = float(leaderboard.iloc[0]["objective_value"])

    summary = {
        "sweep_id": sid,
        "objective_metric": config.objective_metric,
        "objective_direction": config.objective_direction,
        "n_proposed": int(len(trials)),
        "n_completed": int((trials["status"] == "completed").sum()),
        "n_failed": int((trials["status"] == "failed").sum()),
        "n_timeouts": int((trials["status"] == "timeout").sum()),
        "n_invalid": int((trials["status"] == "invalid").sum()),
        "best_objective_value": best_value,
        "terminated_by": terminated_by,
        "config_hash": config_hash(config.__dict__),
        "generated_at": utc_now_iso(),
    }

    trials_path = Path(write_parquet_safe(trials, out / "trials.parquet"))
    leaderboard_path = Path(write_parquet_safe(leaderboard, out / "leaderboard.parquet"))
    best_path = Path(write_json_safe({"sweep_id": sid, "best_candidates": best_candidates}, out / "best_candidates.json"))
    summary_path = Path(write_json_safe(summary, out / "sweep_summary.json"))

    return {
        "trials": trials,
        "leaderboard": leaderboard,
        "best_candidates": best_candidates,
        "summary": summary,
        "artifacts": {
            "trials_path": str(trials_path),
            "leaderboard_path": str(leaderboard_path),
            "best_candidates_path": str(best_path),
            "summary_path": str(summary_path),
        },
    }


__all__ = ["SweepConfig", "launch_sweep"]
