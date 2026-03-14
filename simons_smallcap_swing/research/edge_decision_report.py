from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

# Allow direct script execution:
# `python simons_smallcap_swing/research/edge_decision_report.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "edge_decision_report_mvp_v1"
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
STATUS_RANK = {PASS: 0, WARN: 1, FAIL: 2}

RECOMMEND_IMPROVE = "improve_features_or_labels"
RECOMMEND_RICHER = "try_slightly_richer_model"
RECOMMEND_PAUSE = "pause_and_rethink"

CANDIDATES_SCHEMA = DataSchema(
    name="edge_decision_candidates_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("candidate_id", "string", nullable=False),
        ColumnSpec("feature_family", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("horizon_days", "int64", nullable=False),
        ColumnSpec("target_type", "string", nullable=False),
        ColumnSpec("primary_metric", "string", nullable=False),
        ColumnSpec("improvement_vs_dummy", "float64", nullable=False),
        ColumnSpec("winner_vs_dummy", "string", nullable=False),
        ColumnSpec("validation_status_if_available", "string", nullable=True),
        ColumnSpec("pbo_estimate_if_available", "float64", nullable=True),
        ColumnSpec("multiple_testing_status_if_available", "string", nullable=True),
        ColumnSpec("promoted_flag", "bool", nullable=False),
        ColumnSpec("rejection_reason", "string", nullable=True),
    ),
    primary_key=("candidate_id",),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class EdgeDecisionReportResult:
    candidates_path: Path
    report_path: Path
    summary_md_path: Path
    overall_research_status: str
    recommendation_next_step: str
    config_hash: str


def _cfg_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _load_json_optional(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(payload, dict):
        return None, "invalid_json_object"
    return payload, None


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], *, name: str) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _task_from_target_type(target_type: str) -> str:
    value = str(target_type).strip().lower()
    if value == "continuous_forward_return":
        return "regression"
    if value == "binary_direction":
        return "classification"
    return "unknown"


def _task_aliases(task_name: str) -> tuple[str, ...]:
    base = str(task_name).strip()
    if base in {"regression", "classification"}:
        return (
            base,
            f"{base}_candidates",
            f"{base}_cv_baselines",
            f"{base}_baselines",
        )
    return (base,)


def _lookup_by_alias(mapping: dict[str, Any], task_name: str) -> Any:
    for alias in _task_aliases(task_name):
        if alias in mapping:
            return mapping[alias]
    return None


def _status_worst(values: list[str]) -> str:
    if not values:
        return WARN
    return max(values, key=lambda x: STATUS_RANK.get(x, -1))


def _build_candidate_id(
    *,
    task_name: str,
    feature_family: str,
    label_name: str,
    horizon_days: int,
    model_name: str,
) -> str:
    return (
        f"{task_name}|{feature_family}|{label_name}|h{int(horizon_days)}|{model_name}"
    )


def _resolve_recommendation(
    *,
    validation_blocker: bool,
    promoted_count: int,
    beating_count: int,
) -> tuple[str, str]:
    if validation_blocker:
        return (
            RECOMMEND_PAUSE,
            "validation_suite indicates FAIL (or leakage FAIL), so research credibility is blocked.",
        )
    if promoted_count > 0:
        return (
            RECOMMEND_RICHER,
            "At least one candidate beats dummy and passes minimum credibility gates.",
        )
    if beating_count > 0:
        return (
            RECOMMEND_IMPROVE,
            "Some candidates beat dummy but fail promotion gates; improve labels/features before richer models.",
        )
    return (
        RECOMMEND_PAUSE,
        "No candidate shows robust edge vs dummy under current evidence.",
    )


def run_edge_decision_report(
    *,
    feature_ablation_results_path: str | Path | None = None,
    feature_ablation_summary_path: str | Path | None = None,
    label_horizon_ablation_results_path: str | Path | None = None,
    label_horizon_ablation_summary_path: str | Path | None = None,
    validation_suite_summary_path: str | Path | None = None,
    pbo_cscv_summary_path: str | Path | None = None,
    multiple_testing_summary_path: str | Path | None = None,
    backtest_diagnostics_summary_path: str | Path | None = None,
    cv_model_comparison_summary_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    min_improvement_for_positive: float = 0.0,
    max_pbo_for_promotion: float = 0.50,
    run_id: str = MODULE_VERSION,
) -> EdgeDecisionReportResult:
    logger = get_logger("research.edge_decision_report")
    base = data_dir()
    repo_root = Path(__file__).resolve().parents[1]

    feature_results_source = (
        Path(feature_ablation_results_path).expanduser().resolve()
        if feature_ablation_results_path
        else (base / "research" / "feature_ablation_results.parquet")
    )
    feature_summary_source = (
        Path(feature_ablation_summary_path).expanduser().resolve()
        if feature_ablation_summary_path
        else (base / "research" / "feature_ablation_summary.json")
    )
    label_results_source = (
        Path(label_horizon_ablation_results_path).expanduser().resolve()
        if label_horizon_ablation_results_path
        else (base / "research" / "label_horizon_ablation_results.parquet")
    )
    label_summary_source = (
        Path(label_horizon_ablation_summary_path).expanduser().resolve()
        if label_horizon_ablation_summary_path
        else (base / "research" / "label_horizon_ablation_summary.json")
    )
    validation_source = (
        Path(validation_suite_summary_path).expanduser().resolve()
        if validation_suite_summary_path
        else (base / "validation" / "validation_suite_summary.json")
    )
    pbo_source = (
        Path(pbo_cscv_summary_path).expanduser().resolve()
        if pbo_cscv_summary_path
        else (base / "validation" / "pbo_cscv_summary.json")
    )
    mt_source = (
        Path(multiple_testing_summary_path).expanduser().resolve()
        if multiple_testing_summary_path
        else (base / "validation" / "multiple_testing_summary.json")
    )
    backtest_source = (
        Path(backtest_diagnostics_summary_path).expanduser().resolve()
        if backtest_diagnostics_summary_path
        else (base / "backtest" / "backtest_diagnostics_summary.json")
    )
    cv_source = (
        Path(cv_model_comparison_summary_path).expanduser().resolve()
        if cv_model_comparison_summary_path
        else (
            repo_root / "models" / "artifacts" / "cv_model_comparison_summary.json"
            if (repo_root / "models" / "artifacts" / "cv_model_comparison_summary.json").exists()
            else (base / "models" / "artifacts" / "cv_model_comparison_summary.json")
        )
    )
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else (base / "research")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not feature_results_source.exists():
        raise FileNotFoundError(f"feature_ablation_results not found: {feature_results_source}")
    if not label_results_source.exists():
        raise FileNotFoundError(f"label_horizon_ablation_results not found: {label_results_source}")
    if not validation_source.exists():
        raise FileNotFoundError(f"validation_suite_summary not found: {validation_source}")

    feature_results = read_parquet(feature_results_source).copy()
    label_results = read_parquet(label_results_source).copy()

    _require_columns(
        feature_results,
        (
            "task_name",
            "feature_family",
            "target_type",
            "mean_valid_primary_metric",
            "improvement_vs_dummy",
            "winner_vs_dummy",
        ),
        name="feature_ablation_results",
    )
    _require_columns(
        label_results,
        (
            "task_name",
            "label_name",
            "target_type",
            "horizon_days",
            "feature_family",
            "model_name",
            "primary_metric",
            "mean_valid_primary_metric",
            "improvement_vs_dummy",
            "winner_vs_dummy",
        ),
        name="label_horizon_ablation_results",
    )

    for text_col in ("task_name", "feature_family", "target_type", "winner_vs_dummy"):
        feature_results[text_col] = feature_results[text_col].astype(str)
    feature_results["mean_valid_primary_metric"] = pd.to_numeric(
        feature_results["mean_valid_primary_metric"], errors="coerce"
    )
    feature_results["improvement_vs_dummy"] = pd.to_numeric(
        feature_results["improvement_vs_dummy"], errors="coerce"
    )

    for text_col in (
        "task_name",
        "label_name",
        "target_type",
        "feature_family",
        "model_name",
        "primary_metric",
        "winner_vs_dummy",
    ):
        label_results[text_col] = label_results[text_col].astype(str)
    label_results["horizon_days"] = pd.to_numeric(
        label_results["horizon_days"], errors="coerce"
    ).astype("Int64")
    label_results["mean_valid_primary_metric"] = pd.to_numeric(
        label_results["mean_valid_primary_metric"], errors="coerce"
    )
    label_results["improvement_vs_dummy"] = pd.to_numeric(
        label_results["improvement_vs_dummy"], errors="coerce"
    )
    if label_results["horizon_days"].isna().any():
        raise ValueError("label_horizon_ablation_results has invalid horizon_days.")
    label_results["horizon_days"] = label_results["horizon_days"].astype("int64")

    validation_summary, validation_err = _load_json_optional(validation_source)
    if validation_err is not None or validation_summary is None:
        raise ValueError(
            f"validation_suite_summary is not consumable: {validation_source} ({validation_err})"
        )
    validation_overall = str(validation_summary.get("overall_status", WARN)).upper()
    leakage_status = str(validation_summary.get("leakage_status", validation_overall)).upper()

    missing_inputs: list[str] = []

    feature_summary, feature_summary_err = _load_json_optional(feature_summary_source)
    if feature_summary_err is not None:
        missing_inputs.append(f"feature_ablation_summary:{feature_summary_err}")

    label_summary, label_summary_err = _load_json_optional(label_summary_source)
    if label_summary_err is not None:
        missing_inputs.append(f"label_horizon_ablation_summary:{label_summary_err}")

    pbo_summary, pbo_err = _load_json_optional(pbo_source)
    if pbo_err is not None:
        missing_inputs.append(f"pbo_cscv_summary:{pbo_err}")
    pbo_estimate_by_task = (
        pbo_summary.get("pbo_estimate_by_task", {})
        if isinstance(pbo_summary, dict)
        else {}
    )
    pbo_overall_status = (
        str(pbo_summary.get("overall_status", WARN)).upper()
        if isinstance(pbo_summary, dict)
        else None
    )

    mt_summary, mt_err = _load_json_optional(mt_source)
    if mt_err is not None:
        missing_inputs.append(f"multiple_testing_summary:{mt_err}")
    mt_task_status = (
        mt_summary.get("task_status_by_task", {})
        if isinstance(mt_summary, dict)
        else {}
    )
    mt_overall_status = (
        str(mt_summary.get("overall_status", WARN)).upper()
        if isinstance(mt_summary, dict)
        else None
    )

    backtest_summary, backtest_err = _load_json_optional(backtest_source)
    if backtest_err is not None:
        missing_inputs.append(f"backtest_diagnostics_summary:{backtest_err}")

    cv_summary, cv_err = _load_json_optional(cv_source)
    if cv_err is not None:
        missing_inputs.append(f"cv_model_comparison_summary:{cv_err}")

    built_ts = datetime.now(UTC).isoformat()
    config_hash = _cfg_hash(
        {
            "module_version": MODULE_VERSION,
            "feature_ablation_results_path": str(feature_results_source),
            "feature_ablation_summary_path": str(feature_summary_source),
            "label_horizon_ablation_results_path": str(label_results_source),
            "label_horizon_ablation_summary_path": str(label_summary_source),
            "validation_suite_summary_path": str(validation_source),
            "pbo_cscv_summary_path": str(pbo_source),
            "multiple_testing_summary_path": str(mt_source),
            "backtest_diagnostics_summary_path": str(backtest_source),
            "cv_model_comparison_summary_path": str(cv_source),
            "min_improvement_for_positive": float(min_improvement_for_positive),
            "max_pbo_for_promotion": float(max_pbo_for_promotion),
            "run_id": run_id,
        }
    )

    rows: list[dict[str, Any]] = []
    for row in label_results.itertuples(index=False):
        task_name = str(row.task_name)
        target_type = str(row.target_type)
        inferred_task = _task_from_target_type(target_type)
        if inferred_task != "unknown" and task_name != inferred_task:
            task_name = inferred_task

        pbo_value = _to_float(_lookup_by_alias(pbo_estimate_by_task, task_name))
        mt_status = _lookup_by_alias(mt_task_status, task_name)
        mt_status_norm = str(mt_status).upper() if mt_status is not None else None

        improvement = float(row.improvement_vs_dummy)
        winner = str(row.winner_vs_dummy)
        positive_edge = bool(
            (winner == "model") and (improvement > float(min_improvement_for_positive))
        )
        validation_blocker = validation_overall == FAIL or leakage_status == FAIL
        pbo_blocker = pbo_value is not None and pbo_value > float(max_pbo_for_promotion)
        mt_blocker = mt_status_norm in {WARN, FAIL}

        rejection_reasons: list[str] = []
        if not positive_edge:
            rejection_reasons.append("not_beating_dummy")
        if validation_blocker:
            rejection_reasons.append("validation_fail")
        if pbo_blocker:
            rejection_reasons.append("pbo_fragile")
        if mt_blocker:
            rejection_reasons.append(f"multiple_testing_{mt_status_norm.lower()}")
        promoted = len(rejection_reasons) == 0

        candidate_id = _build_candidate_id(
            task_name=task_name,
            feature_family=str(row.feature_family),
            label_name=str(row.label_name),
            horizon_days=int(row.horizon_days),
            model_name=str(row.model_name),
        )

        rows.append(
            {
                "candidate_id": candidate_id,
                "task_name": task_name,
                "model_name": str(row.model_name),
                "feature_family": str(row.feature_family),
                "label_name": str(row.label_name),
                "horizon_days": int(row.horizon_days),
                "target_type": target_type,
                "primary_metric": str(row.primary_metric),
                "mean_valid_primary_metric": float(row.mean_valid_primary_metric),
                "improvement_vs_dummy": improvement,
                "winner_vs_dummy": winner,
                "validation_status_if_available": validation_overall,
                "pbo_estimate_if_available": pbo_value,
                "multiple_testing_status_if_available": mt_status_norm,
                "promoted_flag": bool(promoted),
                "fragile_flag": bool(pbo_blocker or mt_blocker or (validation_overall == WARN)),
                "rejection_reason": ";".join(rejection_reasons) if rejection_reasons else None,
                "run_id": run_id,
                "config_hash": config_hash,
                "built_ts_utc": built_ts,
            }
        )

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ValueError("No candidates generated from label_horizon_ablation_results.")
    if candidates.duplicated(["candidate_id"]).any():
        raise ValueError("Duplicate candidate_id generated.")

    # Stable ranking: promoted first, then larger improvement, then lower metric.
    candidates = candidates.sort_values(
        ["promoted_flag", "improvement_vs_dummy", "mean_valid_primary_metric", "candidate_id"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    candidates["candidate_rank"] = (candidates.index + 1).astype("int64")

    # Normalize for schema assertion.
    for col in (
        "candidate_id",
        "feature_family",
        "label_name",
        "target_type",
        "primary_metric",
        "winner_vs_dummy",
        "validation_status_if_available",
        "multiple_testing_status_if_available",
        "rejection_reason",
    ):
        candidates[col] = candidates[col].astype("string")
    candidates["horizon_days"] = pd.to_numeric(candidates["horizon_days"], errors="coerce").astype("int64")
    candidates["improvement_vs_dummy"] = pd.to_numeric(
        candidates["improvement_vs_dummy"], errors="coerce"
    )
    candidates["pbo_estimate_if_available"] = pd.to_numeric(
        candidates["pbo_estimate_if_available"], errors="coerce"
    )
    candidates["promoted_flag"] = candidates["promoted_flag"].astype(bool)
    assert_schema(candidates, CANDIDATES_SCHEMA)

    best_candidate_row = candidates.iloc[0]
    promoted_count = int(candidates["promoted_flag"].sum())
    beating_count = int(
        (
            (candidates["winner_vs_dummy"].astype(str) == "model")
            & (pd.to_numeric(candidates["improvement_vs_dummy"], errors="coerce") > float(min_improvement_for_positive))
        ).sum()
    )

    validation_blocker = validation_overall == FAIL or leakage_status == FAIL
    recommendation_next_step, recommendation_rationale = _resolve_recommendation(
        validation_blocker=validation_blocker,
        promoted_count=promoted_count,
        beating_count=beating_count,
    )

    if validation_blocker:
        overall_research_status = FAIL
    elif promoted_count > 0:
        overall_research_status = PASS
    elif beating_count > 0:
        overall_research_status = WARN
    else:
        overall_research_status = FAIL

    if missing_inputs and overall_research_status == PASS:
        overall_research_status = WARN

    # Additional downgrade from optional anti-overfitting artifacts when available.
    if (pbo_overall_status == FAIL or mt_overall_status == FAIL) and overall_research_status == PASS:
        overall_research_status = WARN

    # Summary blocks.
    best_label_by_task: dict[str, str] = {}
    for task_name, group in candidates.groupby("task_name", sort=True):
        best = group.sort_values(
            ["promoted_flag", "improvement_vs_dummy", "mean_valid_primary_metric", "candidate_id"],
            ascending=[False, False, True, True],
        ).iloc[0]
        best_label_by_task[str(task_name)] = str(best["label_name"])

    best_feature_family = str(best_candidate_row["feature_family"])

    candidates_beating_dummy = sorted(
        candidates.loc[
            (candidates["winner_vs_dummy"].astype(str) == "model")
            & (pd.to_numeric(candidates["improvement_vs_dummy"], errors="coerce") > float(min_improvement_for_positive)),
            "candidate_id",
        ]
        .astype(str)
        .tolist()
    )
    candidates_failing_validation = sorted(
        candidates.loc[
            candidates["rejection_reason"].fillna("").astype(str).str.contains("validation_fail"),
            "candidate_id",
        ]
        .astype(str)
        .tolist()
    )
    candidates_flagged_fragile = sorted(
        candidates.loc[candidates["fragile_flag"].astype(bool), "candidate_id"]
        .astype(str)
        .tolist()
    )

    report_payload: dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "run_id": run_id,
        "config_hash": config_hash,
        "built_ts_utc": built_ts,
        "best_candidate_overall": {
            "candidate_id": str(best_candidate_row["candidate_id"]),
            "task_name": str(best_candidate_row["task_name"]),
            "feature_family": str(best_candidate_row["feature_family"]),
            "label_name": str(best_candidate_row["label_name"]),
            "horizon_days": int(best_candidate_row["horizon_days"]),
            "target_type": str(best_candidate_row["target_type"]),
            "improvement_vs_dummy": _to_float(best_candidate_row["improvement_vs_dummy"]),
            "winner_vs_dummy": str(best_candidate_row["winner_vs_dummy"]),
            "promoted_flag": bool(best_candidate_row["promoted_flag"]),
        },
        "best_feature_family": best_feature_family,
        "best_label_by_task": best_label_by_task,
        "candidates_beating_dummy": candidates_beating_dummy,
        "candidates_failing_validation": candidates_failing_validation,
        "candidates_flagged_fragile": candidates_flagged_fragile,
        "recommendation_next_step": recommendation_next_step,
        "recommendation_rationale": recommendation_rationale,
        "missing_inputs": missing_inputs,
        "overall_research_status": overall_research_status,
        "credibility_context": {
            "validation_suite_overall_status": validation_overall,
            "leakage_status": leakage_status,
            "pbo_overall_status": pbo_overall_status,
            "multiple_testing_overall_status": mt_overall_status,
            "backtest_diagnostics_available": backtest_err is None,
            "cv_model_comparison_available": cv_err is None,
        },
        "source_paths": {
            "feature_ablation_results": str(feature_results_source),
            "feature_ablation_summary": str(feature_summary_source),
            "label_horizon_ablation_results": str(label_results_source),
            "label_horizon_ablation_summary": str(label_summary_source),
            "validation_suite_summary": str(validation_source),
            "pbo_cscv_summary": str(pbo_source),
            "multiple_testing_summary": str(mt_source),
            "backtest_diagnostics_summary": str(backtest_source),
            "cv_model_comparison_summary": str(cv_source),
        },
    }

    # Optional references to upstream summary signals.
    if isinstance(feature_summary, dict):
        report_payload["feature_ablation_context"] = {
            "best_family_by_task": feature_summary.get("best_family_by_task"),
            "best_model_by_task": feature_summary.get("best_model_by_task"),
        }
    if isinstance(label_summary, dict):
        report_payload["label_horizon_context"] = {
            "best_label_by_task": label_summary.get("best_label_by_task"),
            "best_horizon_by_task": label_summary.get("best_horizon_by_task"),
            "missing_label_combinations": label_summary.get("missing_label_combinations"),
        }
    if isinstance(backtest_summary, dict):
        report_payload["economic_context"] = {
            "best_mode_by_cumulative_net_return": backtest_summary.get(
                "best_mode_by_cumulative_net_return"
            ),
            "max_drawdown_net_all_modes": backtest_summary.get(
                "max_drawdown_net_all_modes"
            ),
            "mean_cost_drag": backtest_summary.get("mean_cost_drag"),
        }
    if isinstance(cv_summary, dict):
        report_payload["cv_comparison_context"] = {
            "regression": cv_summary.get("regression"),
            "classification": cv_summary.get("classification"),
        }

    candidates_path = write_parquet(
        candidates,
        out_dir / "edge_decision_candidates.parquet",
        schema_name=CANDIDATES_SCHEMA.name,
        run_id=run_id,
    )
    report_path = out_dir / "edge_decision_report.json"
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")

    summary_md_path = out_dir / "edge_decision_summary.md"
    summary_md = [
        "# Edge Decision Report (MVP)",
        "",
        f"- `overall_research_status`: **{overall_research_status}**",
        f"- `recommendation_next_step`: **{recommendation_next_step}**",
        f"- `recommendation_rationale`: {recommendation_rationale}",
        f"- `best_candidate_overall`: `{best_candidate_row['candidate_id']}`",
        "",
        "## Credibility Context",
        f"- validation_suite: `{validation_overall}` (leakage: `{leakage_status}`)",
        f"- pbo_cscv: `{pbo_overall_status}`",
        f"- multiple_testing: `{mt_overall_status}`",
        f"- missing_inputs: {', '.join(missing_inputs) if missing_inputs else 'none'}",
        "",
        "## Candidate Counts",
        f"- total_candidates: {int(len(candidates))}",
        f"- promoted_candidates: {promoted_count}",
        f"- candidates_beating_dummy: {len(candidates_beating_dummy)}",
        f"- candidates_flagged_fragile: {len(candidates_flagged_fragile)}",
        "",
    ]
    summary_md_path.write_text("\n".join(summary_md), encoding="utf-8")

    logger.info(
        "edge_decision_report_completed",
        run_id=run_id,
        overall_research_status=overall_research_status,
        recommendation_next_step=recommendation_next_step,
        candidates_path=str(candidates_path),
        report_path=str(report_path),
    )
    return EdgeDecisionReportResult(
        candidates_path=candidates_path,
        report_path=report_path,
        summary_md_path=summary_md_path,
        overall_research_status=overall_research_status,
        recommendation_next_step=recommendation_next_step,
        config_hash=config_hash,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an MVP edge decision report from ablations + validation artifacts."
    )
    parser.add_argument("--feature-ablation-results-path", default=None)
    parser.add_argument("--feature-ablation-summary-path", default=None)
    parser.add_argument("--label-horizon-ablation-results-path", default=None)
    parser.add_argument("--label-horizon-ablation-summary-path", default=None)
    parser.add_argument("--validation-suite-summary-path", default=None)
    parser.add_argument("--pbo-cscv-summary-path", default=None)
    parser.add_argument("--multiple-testing-summary-path", default=None)
    parser.add_argument("--backtest-diagnostics-summary-path", default=None)
    parser.add_argument("--cv-model-comparison-summary-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-improvement-for-positive", type=float, default=0.0)
    parser.add_argument("--max-pbo-for-promotion", type=float, default=0.50)
    parser.add_argument("--run-id", default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_edge_decision_report(
        feature_ablation_results_path=args.feature_ablation_results_path,
        feature_ablation_summary_path=args.feature_ablation_summary_path,
        label_horizon_ablation_results_path=args.label_horizon_ablation_results_path,
        label_horizon_ablation_summary_path=args.label_horizon_ablation_summary_path,
        validation_suite_summary_path=args.validation_suite_summary_path,
        pbo_cscv_summary_path=args.pbo_cscv_summary_path,
        multiple_testing_summary_path=args.multiple_testing_summary_path,
        backtest_diagnostics_summary_path=args.backtest_diagnostics_summary_path,
        cv_model_comparison_summary_path=args.cv_model_comparison_summary_path,
        output_dir=args.output_dir,
        min_improvement_for_positive=float(args.min_improvement_for_positive),
        max_pbo_for_promotion=float(args.max_pbo_for_promotion),
        run_id=args.run_id,
    )
    print(
        json.dumps(
            {
                "candidates_path": str(result.candidates_path),
                "report_path": str(result.report_path),
                "summary_md_path": str(result.summary_md_path),
                "overall_research_status": result.overall_research_status,
                "recommendation_next_step": result.recommendation_next_step,
                "config_hash": result.config_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
