
"""
research/research_pipeline.py - Research governance orchestrator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from simons_smallcap_swing.research._shared import (
    config_hash,
    json_safe,
    run_id_with_prefix,
    sha256_text,
    utc_now_iso,
    write_json_safe,
    write_parquet_safe,
)


STAGES = [
    "stage_00_intake",
    "stage_01_data_lineage",
    "stage_02_label_construction",
    "stage_03_feature_qc",
    "stage_04_model_spec",
    "stage_05_in_sample",
    "stage_06_walkforward",
    "stage_07_robustness",
    "stage_08_decision",
]


@dataclass(frozen=True)
class PipelineConfig:
    tau_psr: float = 0.70
    tau_fdr: float = 0.10
    tau_capacity: float = 0.60
    iterate_psr_floor: float = 0.50
    iterate_fdr_ceiling: float = 0.20
    stop_on_fail: bool = True
    seed_bundle: Mapping[str, Any] = field(default_factory=lambda: {"global_seed": 42})


@dataclass
class StageResult:
    run_id: str
    stage_name: str
    status: str
    gate_state: str
    input_artifacts: list[str]
    output_artifacts: list[str]
    metrics: Mapping[str, Any]
    thresholds: Mapping[str, Any]
    seed_bundle: Mapping[str, Any]
    started_at: str
    finished_at: str
    error_class: str = ""
    error_message: str = ""


REQUIRED_HYPOTHESIS_FIELDS = [
    "hypothesis_id",
    "thesis",
    "null_hypothesis",
    "alternative_hypothesis",
    "economic_mechanism",
    "primary_metric",
    "secondary_metrics",
    "holding_period",
    "rebalance_rule",
    "universe",
    "known_risks",
    "trial_family_id",
    "max_trials_allowed",
    "kill_criteria",
    "promotion_target",
    "falsification_conditions",
]


def _load_mapping(obj: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(obj, Mapping):
        return dict(obj)
    p = Path(obj)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {p}")
    if p.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("yaml parsing requires pyyaml") from exc
        return dict(yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    import json

    return dict(json.loads(p.read_text(encoding="utf-8")))


def _stage_gate(pass_cond: bool, borderline_cond: bool) -> str:
    if pass_cond:
        return "Pass"
    if borderline_cond:
        return "Borderline"
    return "Fail"


def _run_default_stage(stage: str, ctx: dict[str, Any], cfg: PipelineConfig) -> tuple[dict[str, Any], dict[str, Any], str]:
    h = ctx["hypothesis"]

    if stage == "stage_00_intake":
        missing = [f for f in REQUIRED_HYPOTHESIS_FIELDS if f not in h or h[f] in (None, "", [])]
        metrics = {
            "n_required_fields": len(REQUIRED_HYPOTHESIS_FIELDS),
            "n_missing_fields": len(missing),
            "missing_fields": missing,
        }
        thresholds = {"n_missing_fields": 0}
        gate = "Pass" if len(missing) == 0 else "Fail"
        return metrics, thresholds, gate

    if stage == "stage_01_data_lineage":
        m = ctx.get("inputs", {}).get("data_lineage", {})
        pit = bool(m.get("pit_integrity", True))
        coverage = float(m.get("coverage_ratio", 1.0))
        metrics = {"pit_integrity": pit, "coverage_ratio": coverage}
        thresholds = {"pit_integrity": True, "coverage_ratio": 0.80}
        gate = _stage_gate(pit and coverage >= 0.80, pit and coverage >= 0.60)
        return metrics, thresholds, gate

    if stage == "stage_02_label_construction":
        m = ctx.get("inputs", {}).get("label_quality", {})
        alignment = bool(m.get("alignment_ok", True))
        leakage = bool(m.get("leakage_flag", False))
        metrics = {"alignment_ok": alignment, "leakage_flag": leakage}
        thresholds = {"alignment_ok": True, "leakage_flag": False}
        gate = "Pass" if alignment and not leakage else "Fail"
        return metrics, thresholds, gate

    if stage == "stage_03_feature_qc":
        m = ctx.get("inputs", {}).get("feature_qc", {})
        coverage = float(m.get("coverage_stability", 0.85))
        validity = bool(m.get("feature_validity", True))
        metrics = {"coverage_stability": coverage, "feature_validity": validity}
        thresholds = {"coverage_stability": 0.80, "feature_validity": True}
        gate = _stage_gate(validity and coverage >= 0.80, validity and coverage >= 0.65)
        return metrics, thresholds, gate

    if stage == "stage_04_model_spec":
        max_trials = int(h.get("max_trials_allowed", 0))
        declared = bool(h.get("trial_family_id"))
        metrics = {"max_trials_allowed": max_trials, "trial_family_declared": declared}
        thresholds = {"max_trials_allowed_min": 1, "trial_family_declared": True}
        gate = "Pass" if declared and max_trials >= 1 else "Fail"
        return metrics, thresholds, gate

    if stage == "stage_05_in_sample":
        m = ctx.get("inputs", {}).get("in_sample", {})
        n_trials = int(m.get("n_trials", h.get("max_trials_allowed", 0)))
        best_is = float(m.get("best_is_metric", 0.0))
        metrics = {"n_trials": n_trials, "best_is_metric": best_is}
        thresholds = {"n_trials": 1}
        gate = "Pass" if n_trials >= 1 else "Fail"
        return metrics, thresholds, gate

    if stage == "stage_06_walkforward":
        m = ctx.get("inputs", {}).get("walkforward", {})
        oos = float(m.get("oos_performance", 0.0))
        psr = float(m.get("psr", 0.0))
        fdr = float(m.get("fdr", 1.0))
        metrics = {"oos_performance": oos, "psr": psr, "fdr": fdr}
        thresholds = {"oos_performance": 0.0, "psr": cfg.iterate_psr_floor, "fdr": cfg.iterate_fdr_ceiling}
        gate = _stage_gate(oos > 0 and psr >= cfg.iterate_psr_floor and fdr <= cfg.iterate_fdr_ceiling, oos >= -0.001)
        return metrics, thresholds, gate

    if stage == "stage_07_robustness":
        m = ctx.get("inputs", {}).get("robustness", {})
        robust = bool(m.get("robust_survival", True))
        capacity = float(m.get("capacity_score", 0.0))
        metrics = {"robust_survival": robust, "capacity_score": capacity}
        thresholds = {"robust_survival": True, "capacity_score": cfg.tau_capacity}
        gate = _stage_gate(robust and capacity >= cfg.tau_capacity, robust and capacity >= max(0.4, cfg.tau_capacity - 0.1))
        return metrics, thresholds, gate

    if stage == "stage_08_decision":
        metrics = {}
        thresholds = {}
        return metrics, thresholds, "Pass"

    raise ValueError(f"unknown stage: {stage}")

def _build_evidence(run_id: str, hypothesis: Mapping[str, Any], stages: list[StageResult], cfg: PipelineConfig) -> dict[str, Any]:
    stage_map = {s.stage_name: s for s in stages}

    wf = stage_map.get("stage_06_walkforward")
    rb = stage_map.get("stage_07_robustness")

    psr = float(wf.metrics.get("psr", 0.0)) if wf else 0.0
    fdr = float(wf.metrics.get("fdr", 1.0)) if wf else 1.0
    oos = float(wf.metrics.get("oos_performance", 0.0)) if wf else 0.0
    capacity_score = float(rb.metrics.get("capacity_score", 0.0)) if rb else 0.0
    robustness_state = rb.gate_state if rb else "Fail"

    return {
        "run_id": run_id,
        "hypothesis_id": hypothesis.get("hypothesis_id", "unknown"),
        "stage_results": [asdict(s) for s in stages],
        "primary_metrics": {"oos_performance": oos},
        "secondary_metrics": {"capacity_score": capacity_score},
        "psr": psr,
        "fdr": fdr,
        "n_trials": int(stage_map.get("stage_05_in_sample").metrics.get("n_trials", 0)) if "stage_05_in_sample" in stage_map else 0,
        "n_effective_trials": int(stage_map.get("stage_05_in_sample").metrics.get("n_trials", 0)) if "stage_05_in_sample" in stage_map else 0,
        "pvalues": None,
        "best_trial_id": stage_map.get("stage_05_in_sample").metrics.get("best_trial_id", "unknown") if "stage_05_in_sample" in stage_map else "unknown",
        "trial_family_id": hypothesis.get("trial_family_id", "unknown"),
        "capacity_score": capacity_score,
        "robustness_state": robustness_state,
        "oos_summary": {"oos_performance": oos, "psr": psr, "fdr": fdr},
        "fold_stability_summary": {"status": wf.gate_state if wf else "Fail"},
        "lineage_references": {"config_hash": config_hash(cfg.__dict__, n=24)},
    }


def _final_decision(evidence: Mapping[str, Any], stages: list[StageResult], cfg: PipelineConfig) -> tuple[str, list[str]]:
    reasons: list[str] = []

    critical_pass = True

    # Explicit critical gate check (excluding final stage).
    for s in stages:
        if s.stage_name == "stage_08_decision":
            continue
        if s.gate_state == "Fail":
            critical_pass = False
            reasons.append(f"{s.stage_name}=Fail")

    psr = float(evidence.get("psr", 0.0))
    fdr = float(evidence.get("fdr", 1.0))
    capacity = float(evidence.get("capacity_score", 0.0))
    robust_ok = str(evidence.get("robustness_state", "Fail")) == "Pass"

    if critical_pass and psr >= cfg.tau_psr and fdr <= cfg.tau_fdr and capacity >= cfg.tau_capacity and robust_ok:
        return "Promote", reasons or ["all_critical_gates_pass"]

    borderline = (
        psr >= cfg.iterate_psr_floor
        and fdr <= cfg.iterate_fdr_ceiling
        and capacity >= max(0.4, cfg.tau_capacity - 0.1)
    )
    if borderline:
        if psr < cfg.tau_psr:
            reasons.append("psr_marginal")
        if fdr > cfg.tau_fdr:
            reasons.append("fdr_marginal")
        if capacity < cfg.tau_capacity:
            reasons.append("capacity_marginal")
        if not robust_ok:
            reasons.append("robustness_not_pass")
        return "Iterate", reasons

    if not reasons:
        reasons.append("insufficient_joint_evidence")
    return "Reject", reasons


def persist_pipeline_artifacts(
    stage_results: list[StageResult],
    evidence_bundle: Mapping[str, Any],
    decision_card: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stage_df = pd.DataFrame([asdict(s) for s in stage_results])
    return {
        "stage_results": write_parquet_safe(stage_df, out / "stage_results.parquet"),
        "evidence_bundle": write_json_safe(dict(evidence_bundle), out / "evidence_bundle.json"),
        "decision_card": write_json_safe(dict(decision_card), out / "decision_card.json"),
        "manifest": write_json_safe(dict(manifest), out / "manifest.json"),
    }


def run_research_pipeline(
    conf_path: Mapping[str, Any] | str | Path,
    hypothesis_card: Mapping[str, Any] | str | Path,
    *,
    mode: str = "full",
    run_id: str | None = None,
    stage_only: str | None = None,
    stage_handlers: Mapping[str, Callable[[dict[str, Any]], Mapping[str, Any]]] | None = None,
    input_artifacts: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg_map = _load_mapping(conf_path)
    hyp = _load_mapping(hypothesis_card)

    cfg = PipelineConfig(**{k: v for k, v in cfg_map.items() if k in PipelineConfig.__dataclass_fields__})
    rid = run_id or run_id_with_prefix("rpipe")

    mode = str(mode).lower()
    if mode not in {"full", "resume", "stage_only", "dry_run"}:
        raise ValueError("mode must be one of full/resume/stage_only/dry_run")

    ctx: dict[str, Any] = {
        "run_id": rid,
        "config": cfg,
        "hypothesis": hyp,
        "inputs": dict(input_artifacts or {}),
    }

    handlers = dict(stage_handlers or {})

    stages_to_run = STAGES
    if mode == "stage_only" and stage_only:
        if stage_only not in STAGES:
            raise ValueError(f"unknown stage_only: {stage_only}")
        stages_to_run = [stage_only]

    stage_results: list[StageResult] = []
    stopped_early = False

    for stage in stages_to_run:
        started = utc_now_iso()
        status = "ok"
        err_class = ""
        err_msg = ""
        metrics: dict[str, Any] = {}
        thresholds: dict[str, Any] = {}
        gate = "Fail"

        try:
            if stage in handlers:
                payload = dict(handlers[stage](ctx) or {})
                metrics = dict(payload.get("metrics", {}))
                thresholds = dict(payload.get("thresholds", {}))
                gate = str(payload.get("gate_state", "Pass"))
            else:
                metrics, thresholds, gate = _run_default_stage(stage, ctx, cfg)
        except Exception as exc:
            status = "error"
            err_class = exc.__class__.__name__
            err_msg = str(exc)
            gate = "Fail"

        finished = utc_now_iso()
        sr = StageResult(
            run_id=rid,
            stage_name=stage,
            status=status,
            gate_state=gate,
            input_artifacts=[f"{k}:{type(v).__name__}" for k, v in ctx.get("inputs", {}).items()],
            output_artifacts=[],
            metrics=json_safe(metrics),
            thresholds=json_safe(thresholds),
            seed_bundle=json_safe(cfg.seed_bundle),
            started_at=started,
            finished_at=finished,
            error_class=err_class,
            error_message=err_msg,
        )
        stage_results.append(sr)

        if gate == "Fail" and mode != "stage_only" and cfg.stop_on_fail and mode != "dry_run":
            stopped_early = True
            break

    evidence = _build_evidence(rid, hyp, stage_results, cfg)
    decision, reasons = _final_decision(evidence, stage_results, cfg)

    decision_card = {
        "run_id": rid,
        "hypothesis_id": hyp.get("hypothesis_id", "unknown"),
        "decision": decision,
        "reasons": reasons,
        "gates": {s.stage_name: s.gate_state for s in stage_results},
        "primary_metrics": evidence.get("primary_metrics", {}),
        "secondary_metrics": evidence.get("secondary_metrics", {}),
        "psr": evidence.get("psr"),
        "fdr": evidence.get("fdr"),
        "trial_family_id": hyp.get("trial_family_id", "unknown"),
        "next_actions": [
            "prepare promote package" if decision == "Promote" else "iterate hypothesis" if decision == "Iterate" else "archive and reject"
        ],
    }

    manifest = {
        "run_id": rid,
        "mode": mode,
        "stage_sequence": stages_to_run,
        "stopped_early": stopped_early,
        "config_hash": config_hash(cfg.__dict__, n=24),
        "hypothesis_hash": sha256_text(str(sorted(hyp.items())))[:24],
        "execution_timestamp": utc_now_iso(),
    }

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_pipeline_artifacts(stage_results, evidence, decision_card, manifest, output_dir=output_dir)

    return {
        "decision": decision,
        "stage_results": [asdict(s) for s in stage_results],
        "evidence_bundle": json_safe(evidence),
        "decision_card": json_safe(decision_card),
        "manifest": json_safe(manifest),
        "artifacts": artifacts,
    }


__all__ = [
    "PipelineConfig",
    "StageResult",
    "persist_pipeline_artifacts",
    "run_research_pipeline",
]
