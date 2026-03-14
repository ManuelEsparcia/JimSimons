from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable

import pandas as pd

# Allow direct script execution: `python simons_smallcap_swing/run_week8_research_refinement.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from research.edge_decision_report import (
    RECOMMEND_IMPROVE,
    RECOMMEND_PAUSE,
    RECOMMEND_RICHER,
    run_edge_decision_report,
)
from research.feature_ablation import run_feature_ablation
from research.h20_regime_conditioned_refinement import run_h20_regime_conditioned_refinement
from research.h20_regime_diagnostics import run_h20_regime_diagnostics
from research.improve_best_candidate import run_improve_best_candidate
from research.label_horizon_ablation import run_label_horizon_ablation
from research.refine_h20_features import run_refine_h20_features
from research.refine_h20_target import run_refine_h20_target
from simons_core.io.paths import data_dir


@dataclass(frozen=True)
class Week8RunResult:
    run_prefix: str
    data_root: Path
    manifest_path: Path
    artifacts: dict[str, Path]
    statuses: dict[str, str]
    final_recommendation: str
    should_try_richer_model_now: bool


def _run_id(prefix: str, step: str) -> str:
    return f"{prefix}_{step}"


def _run_step(
    idx: int,
    total: int,
    label: str,
    func: Callable[..., object],
    **kwargs: object,
) -> object:
    t0 = time.perf_counter()
    print(f"[{idx}/{total}] {label} ...")
    try:
        out = func(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"Step failed [{idx}/{total}] {label}: {exc}") from exc
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    print(f"[{idx}/{total}] {label} done ({elapsed_ms} ms)")
    return out


def _resolve_path(value: str | Path | None, default_path: Path) -> Path:
    return Path(value).expanduser().resolve() if value is not None else default_path.resolve()


def _resolve_optional_existing(value: str | Path | None, default_path: Path) -> Path | None:
    if value is not None:
        return Path(value).expanduser().resolve()
    resolved = default_path.resolve()
    return resolved if resolved.exists() else None


def _parse_csv_text(values: str | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(item.strip() for item in values.split(",") if item.strip())


def _parse_csv_int(values: str | None) -> tuple[int, ...]:
    if not values:
        return ()
    return tuple(int(item.strip()) for item in values.split(",") if item.strip())


def _ensure_required(required_paths: dict[str, Path]) -> dict[str, Path]:
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Week 8 runner requires existing Week 3/7 research prerequisites. "
            f"Missing: {missing}. Paths checked: "
            + ", ".join(f"{name}={path}" for name, path in required_paths.items())
        )
    return required_paths


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON object at {path}")
    return payload


def _best_regime_candidate(results_path: Path) -> dict[str, Any] | None:
    frame = pd.read_parquet(results_path).copy()
    required = {
        "candidate_variant",
        "regime_family",
        "regime_name",
        "n_obs",
        "n_folds_used",
        "improvement_vs_dummy",
        "winner_vs_dummy",
        "winner_vs_comparison_variant",
    }
    if not required.issubset(set(frame.columns)):
        return None
    for col in ("improvement_vs_dummy",):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in ("n_obs", "n_folds_used"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["improvement_vs_dummy", "n_obs", "n_folds_used"]).copy()
    if frame.empty:
        return None
    frame = frame.sort_values(
        ["improvement_vs_dummy", "n_folds_used", "n_obs"],
        ascending=[False, False, False],
    )
    row = frame.iloc[0]
    return {
        "candidate_variant": str(row["candidate_variant"]),
        "regime_family": str(row["regime_family"]),
        "regime_name": str(row["regime_name"]),
        "improvement_vs_dummy": float(row["improvement_vs_dummy"]),
        "n_folds_used": int(row["n_folds_used"]),
        "n_obs": int(row["n_obs"]),
        "winner_vs_dummy": str(row["winner_vs_dummy"]),
        "winner_vs_comparison_variant": str(row["winner_vs_comparison_variant"]),
    }


def _robust_regime_niche_count(results_path: Path, *, min_folds: int) -> int:
    frame = pd.read_parquet(results_path).copy()
    required = {"winner_vs_dummy", "winner_vs_comparison_variant", "n_folds_used"}
    if not required.issubset(set(frame.columns)):
        return 0
    frame["n_folds_used"] = pd.to_numeric(frame["n_folds_used"], errors="coerce").fillna(0).astype(int)
    mask = (
        (frame["winner_vs_dummy"].astype(str) == "variant")
        & (frame["winner_vs_comparison_variant"].astype(str) == "variant")
        & (frame["n_folds_used"] >= int(min_folds))
    )
    return int(mask.sum())


def run_week8_research_refinement(
    *,
    run_prefix: str = "week8_research_refinement",
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_dataset_path: str | Path | None = None,
    h20_model_dataset_path: str | Path | None = None,
    labels_forward_path: str | Path | None = None,
    features_matrix_path: str | Path | None = None,
    purged_cv_folds_path: str | Path | None = None,
    validation_suite_summary_path: str | Path | None = None,
    pbo_cscv_summary_path: str | Path | None = None,
    multiple_testing_summary_path: str | Path | None = None,
    backtest_diagnostics_summary_path: str | Path | None = None,
    cv_model_comparison_summary_path: str | Path | None = None,
    feature_ablation_horizon: int = 5,
    label_horizon_set: Iterable[int] = (1, 5, 20),
    h20_label_name: str = "fwd_ret_20d",
    h20_target_type: str = "continuous_forward_return",
    h20_horizon_days: int = 20,
    candidate_variants: Iterable[str] = ("baseline_all_features", "stable_sign_plus_low_collinearity"),
    target_regimes: Iterable[str] = (
        "time_split_early_vs_late:late",
        "high_vs_low_market_breadth:low",
        "high_vs_low_cross_sectional_dispersion:low",
        "high_vs_low_liquidity:low",
    ),
    should_try_min_regime_folds: int = 3,
) -> Week8RunResult:
    base_data = Path(data_root).expanduser().resolve() if data_root else data_dir()
    research_root = Path(output_dir).expanduser().resolve() if output_dir else (base_data / "research")
    research_root.mkdir(parents=True, exist_ok=True)

    required = _ensure_required(
        {
            "model_dataset": _resolve_path(model_dataset_path, base_data / "datasets" / "model_dataset.parquet"),
            "model_dataset_h20": _resolve_path(
                h20_model_dataset_path,
                base_data / "datasets" / "regression_h20" / "model_dataset.parquet",
            ),
            "labels_forward": _resolve_path(labels_forward_path, base_data / "labels" / "labels_forward.parquet"),
            "features_matrix": _resolve_path(
                features_matrix_path, base_data / "features" / "features_matrix.parquet"
            ),
            "purged_cv_folds": _resolve_path(
                purged_cv_folds_path, base_data / "labels" / "purged_cv_folds.parquet"
            ),
            "validation_suite_summary": _resolve_path(
                validation_suite_summary_path,
                base_data / "validation" / "validation_suite_summary.json",
            ),
        }
    )

    optional_inputs = {
        "pbo_cscv_summary": _resolve_optional_existing(
            pbo_cscv_summary_path, base_data / "validation" / "pbo_cscv_summary.json"
        ),
        "multiple_testing_summary": _resolve_optional_existing(
            multiple_testing_summary_path, base_data / "validation" / "multiple_testing_summary.json"
        ),
        "backtest_diagnostics_summary": _resolve_optional_existing(
            backtest_diagnostics_summary_path,
            base_data / "backtest" / "backtest_diagnostics_summary.json",
        ),
        "cv_model_comparison_summary": _resolve_optional_existing(
            cv_model_comparison_summary_path,
            base_data / "models" / "artifacts" / "cv_model_comparison_summary.json",
        ),
    }

    selected_horizons = tuple(int(v) for v in label_horizon_set)
    if not selected_horizons:
        raise ValueError("label_horizon_set cannot be empty.")
    selected_variants = tuple(str(v).strip() for v in candidate_variants if str(v).strip())
    if len(selected_variants) < 2:
        raise ValueError("candidate_variants must include at least two variants.")
    selected_regimes = tuple(str(v).strip() for v in target_regimes if str(v).strip())
    if not selected_regimes:
        raise ValueError("target_regimes cannot be empty.")

    total_steps = 8
    step = 1
    statuses: dict[str, str] = {}

    feature_ablation = _run_step(
        step,
        total_steps,
        "run feature ablation",
        run_feature_ablation,
        model_dataset_path=required["model_dataset"],
        purged_cv_folds_path=required["purged_cv_folds"],
        features_matrix_path=required["features_matrix"],
        output_dir=research_root,
        horizon_days=int(feature_ablation_horizon),
        run_id=_run_id(run_prefix, "feature_ablation"),
    )
    statuses["feature_ablation"] = "DONE"
    step += 1

    label_horizon = _run_step(
        step,
        total_steps,
        "run label/horizon ablation",
        run_label_horizon_ablation,
        labels_forward_path=required["labels_forward"],
        features_matrix_path=required["features_matrix"],
        purged_cv_folds_path=required["purged_cv_folds"],
        output_dir=research_root,
        horizons=selected_horizons,
        run_id=_run_id(run_prefix, "label_horizon_ablation"),
    )
    statuses["label_horizon_ablation"] = "DONE"
    step += 1

    edge = _run_step(
        step,
        total_steps,
        "run edge decision report",
        run_edge_decision_report,
        feature_ablation_results_path=feature_ablation.results_path,
        feature_ablation_summary_path=feature_ablation.summary_path,
        label_horizon_ablation_results_path=label_horizon.results_path,
        label_horizon_ablation_summary_path=label_horizon.summary_path,
        validation_suite_summary_path=required["validation_suite_summary"],
        pbo_cscv_summary_path=optional_inputs["pbo_cscv_summary"],
        multiple_testing_summary_path=optional_inputs["multiple_testing_summary"],
        backtest_diagnostics_summary_path=optional_inputs["backtest_diagnostics_summary"],
        cv_model_comparison_summary_path=optional_inputs["cv_model_comparison_summary"],
        output_dir=research_root,
        run_id=_run_id(run_prefix, "edge_decision_report"),
    )
    statuses["edge_decision_report"] = "DONE"
    step += 1

    improve = _run_step(
        step,
        total_steps,
        "run improve best candidate",
        run_improve_best_candidate,
        model_dataset_path=required["model_dataset_h20"],
        purged_cv_folds_path=required["purged_cv_folds"],
        feature_ablation_summary_path=feature_ablation.summary_path,
        output_dir=research_root,
        label_name=h20_label_name,
        target_type=h20_target_type,
        horizon_days=int(h20_horizon_days),
        run_id=_run_id(run_prefix, "improve_best_candidate"),
    )
    statuses["improve_best_candidate"] = "DONE"
    step += 1

    refined_features = _run_step(
        step,
        total_steps,
        "run refine h20 features",
        run_refine_h20_features,
        model_dataset_path=required["model_dataset_h20"],
        purged_cv_folds_path=required["purged_cv_folds"],
        output_dir=research_root,
        label_name=h20_label_name,
        target_type=h20_target_type,
        horizon_days=int(h20_horizon_days),
        run_id=_run_id(run_prefix, "refine_h20_features"),
    )
    statuses["refine_h20_features"] = "DONE"
    step += 1

    refined_target = _run_step(
        step,
        total_steps,
        "run refine h20 target",
        run_refine_h20_target,
        model_dataset_path=required["model_dataset_h20"],
        purged_cv_folds_path=required["purged_cv_folds"],
        output_dir=research_root,
        label_name=h20_label_name,
        target_type=h20_target_type,
        horizon_days=int(h20_horizon_days),
        run_id=_run_id(run_prefix, "refine_h20_target"),
    )
    statuses["refine_h20_target"] = "DONE"
    step += 1

    regime_diag = _run_step(
        step,
        total_steps,
        "run h20 regime diagnostics",
        run_h20_regime_diagnostics,
        model_dataset_path=required["model_dataset_h20"],
        purged_cv_folds_path=required["purged_cv_folds"],
        output_dir=research_root,
        label_name=h20_label_name,
        target_type=h20_target_type,
        horizon_days=int(h20_horizon_days),
        candidate_variants=selected_variants,
        run_id=_run_id(run_prefix, "h20_regime_diagnostics"),
    )
    statuses["h20_regime_diagnostics"] = "DONE"
    step += 1

    regime_conditioned = _run_step(
        step,
        total_steps,
        "run h20 regime-conditioned refinement",
        run_h20_regime_conditioned_refinement,
        model_dataset_path=required["model_dataset_h20"],
        purged_cv_folds_path=required["purged_cv_folds"],
        output_dir=research_root,
        label_name=h20_label_name,
        target_type=h20_target_type,
        horizon_days=int(h20_horizon_days),
        candidate_variants=selected_variants,
        target_regimes=selected_regimes,
        run_id=_run_id(run_prefix, "h20_regime_conditioned_refinement"),
    )
    statuses["h20_regime_conditioned_refinement"] = "DONE"

    artifacts: dict[str, Path] = {
        "feature_ablation_results": feature_ablation.results_path,
        "feature_ablation_summary": feature_ablation.summary_path,
        "feature_ablation_fold_metrics": feature_ablation.fold_metrics_path,
        "label_horizon_ablation_results": label_horizon.results_path,
        "label_horizon_ablation_summary": label_horizon.summary_path,
        "label_horizon_ablation_fold_metrics": label_horizon.fold_metrics_path,
        "edge_decision_candidates": edge.candidates_path,
        "edge_decision_report": edge.report_path,
        "edge_decision_summary_md": edge.summary_md_path,
        "improve_best_candidate_results": improve.results_path,
        "improve_best_candidate_summary": improve.summary_path,
        "improve_best_candidate_fold_metrics": improve.fold_metrics_path,
        "refine_h20_features_results": refined_features.results_path,
        "refine_h20_features_summary": refined_features.summary_path,
        "refine_h20_features_fold_metrics": refined_features.fold_metrics_path,
        "refine_h20_target_results": refined_target.results_path,
        "refine_h20_target_summary": refined_target.summary_path,
        "refine_h20_target_fold_metrics": refined_target.fold_metrics_path,
        "h20_regime_diagnostics_results": regime_diag.results_path,
        "h20_regime_diagnostics_summary": regime_diag.summary_path,
        "h20_regime_diagnostics_fold_metrics": regime_diag.fold_metrics_path,
        "h20_regime_conditioned_results": regime_conditioned.results_path,
        "h20_regime_conditioned_summary": regime_conditioned.summary_path,
        "h20_regime_conditioned_fold_metrics": regime_conditioned.fold_metrics_path,
    }

    edge_report = _load_json(edge.report_path)
    conditioned_summary = _load_json(regime_conditioned.summary_path)

    best_global_candidate = edge_report.get("best_candidate_overall")
    best_regime_candidate = _best_regime_candidate(regime_conditioned.results_path)

    robust_niche_count = _robust_regime_niche_count(
        regime_conditioned.results_path,
        min_folds=int(should_try_min_regime_folds),
    )
    edge_next_step = str(edge_report.get("recommendation_next_step", RECOMMEND_IMPROVE))
    global_overall_status = str(edge_report.get("overall_research_status", "WARN")).upper()
    regime_recommendation = str(conditioned_summary.get("recommendation", "unknown"))

    should_try_richer_model_now = bool(
        edge_next_step == RECOMMEND_RICHER
        and global_overall_status == "PASS"
        and robust_niche_count >= 2
    )
    if should_try_richer_model_now:
        week8_final_recommendation = RECOMMEND_RICHER
    elif edge_next_step == RECOMMEND_PAUSE and robust_niche_count == 0:
        week8_final_recommendation = RECOMMEND_PAUSE
    else:
        week8_final_recommendation = RECOMMEND_IMPROVE

    if edge_next_step == RECOMMEND_RICHER and not should_try_richer_model_now:
        global_h20_status = "global_candidate_positive_but_not_yet_promotable"
    elif edge_next_step == RECOMMEND_IMPROVE:
        global_h20_status = "global_edge_fragile"
    elif edge_next_step == RECOMMEND_PAUSE:
        global_h20_status = "global_edge_not_defendable"
    else:
        global_h20_status = "global_status_unknown"

    if robust_niche_count > 0:
        regime_conditioned_h20_status = "regime_niche_detected_but_fragile"
    else:
        regime_conditioned_h20_status = "no_regime_niche_defendable"

    manifest_payload = {
        "run_prefix": run_prefix,
        "built_ts_utc": datetime.now(UTC).isoformat(),
        "input_paths": {
            "model_dataset": str(required["model_dataset"]),
            "model_dataset_h20": str(required["model_dataset_h20"]),
            "labels_forward": str(required["labels_forward"]),
            "features_matrix": str(required["features_matrix"]),
            "purged_cv_folds": str(required["purged_cv_folds"]),
            "validation_suite_summary": str(required["validation_suite_summary"]),
            "pbo_cscv_summary": str(optional_inputs["pbo_cscv_summary"]) if optional_inputs["pbo_cscv_summary"] else None,
            "multiple_testing_summary": str(optional_inputs["multiple_testing_summary"]) if optional_inputs["multiple_testing_summary"] else None,
            "backtest_diagnostics_summary": str(optional_inputs["backtest_diagnostics_summary"]) if optional_inputs["backtest_diagnostics_summary"] else None,
            "cv_model_comparison_summary": str(optional_inputs["cv_model_comparison_summary"]) if optional_inputs["cv_model_comparison_summary"] else None,
        },
        "steps_run": [
            "feature_ablation",
            "label_horizon_ablation",
            "edge_decision_report",
            "improve_best_candidate",
            "refine_h20_features",
            "refine_h20_target",
            "h20_regime_diagnostics",
            "h20_regime_conditioned_refinement",
        ],
        "step_statuses": statuses,
        "key_outputs": {key: str(path) for key, path in artifacts.items()},
        "final_recommendation": week8_final_recommendation,
        "week8_final_recommendation": week8_final_recommendation,
        "global_h20_status": global_h20_status,
        "regime_conditioned_h20_status": regime_conditioned_h20_status,
        "should_try_richer_model_now": bool(should_try_richer_model_now),
        "best_global_candidate_if_any": best_global_candidate,
        "best_regime_candidate_if_any": best_regime_candidate,
        "notes": [
            f"edge_decision_recommendation_next_step={edge_next_step}",
            f"edge_decision_overall_research_status={global_overall_status}",
            f"regime_conditioned_recommendation={regime_recommendation}",
            f"robust_regime_niche_count(min_folds={int(should_try_min_regime_folds)})={robust_niche_count}",
            "Conservative gate applied: do not promote richer model unless global PASS and >=2 robust regime niches.",
        ],
    }
    manifest_path = base_data / f"week8_research_refinement_manifest_{run_prefix}.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] Week 8 research refinement pipeline completed. Manifest: {manifest_path}")

    return Week8RunResult(
        run_prefix=run_prefix,
        data_root=base_data,
        manifest_path=manifest_path,
        artifacts=artifacts,
        statuses=statuses,
        final_recommendation=week8_final_recommendation,
        should_try_richer_model_now=bool(should_try_richer_model_now),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Week 8 research refinement closure pipeline."
    )
    parser.add_argument("--run-prefix", type=str, default="week8_research_refinement")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-dataset-path", type=str, default=None)
    parser.add_argument("--h20-model-dataset-path", type=str, default=None)
    parser.add_argument("--labels-forward-path", type=str, default=None)
    parser.add_argument("--features-matrix-path", type=str, default=None)
    parser.add_argument("--purged-cv-folds-path", type=str, default=None)
    parser.add_argument("--validation-suite-summary-path", type=str, default=None)
    parser.add_argument("--pbo-cscv-summary-path", type=str, default=None)
    parser.add_argument("--multiple-testing-summary-path", type=str, default=None)
    parser.add_argument("--backtest-diagnostics-summary-path", type=str, default=None)
    parser.add_argument("--cv-model-comparison-summary-path", type=str, default=None)
    parser.add_argument("--feature-ablation-horizon", type=int, default=5)
    parser.add_argument("--label-horizon-set", type=str, default="1,5,20")
    parser.add_argument("--h20-label-name", type=str, default="fwd_ret_20d")
    parser.add_argument("--h20-target-type", type=str, default="continuous_forward_return")
    parser.add_argument("--h20-horizon-days", type=int, default=20)
    parser.add_argument(
        "--candidate-variants",
        type=str,
        default="baseline_all_features,stable_sign_plus_low_collinearity",
    )
    parser.add_argument(
        "--target-regimes",
        type=str,
        default=(
            "time_split_early_vs_late:late,"
            "high_vs_low_market_breadth:low,"
            "high_vs_low_cross_sectional_dispersion:low,"
            "high_vs_low_liquidity:low"
        ),
    )
    parser.add_argument("--should-try-min-regime-folds", type=int, default=3)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_week8_research_refinement(
        run_prefix=args.run_prefix,
        data_root=args.data_root,
        output_dir=args.output_dir,
        model_dataset_path=args.model_dataset_path,
        h20_model_dataset_path=args.h20_model_dataset_path,
        labels_forward_path=args.labels_forward_path,
        features_matrix_path=args.features_matrix_path,
        purged_cv_folds_path=args.purged_cv_folds_path,
        validation_suite_summary_path=args.validation_suite_summary_path,
        pbo_cscv_summary_path=args.pbo_cscv_summary_path,
        multiple_testing_summary_path=args.multiple_testing_summary_path,
        backtest_diagnostics_summary_path=args.backtest_diagnostics_summary_path,
        cv_model_comparison_summary_path=args.cv_model_comparison_summary_path,
        feature_ablation_horizon=int(args.feature_ablation_horizon),
        label_horizon_set=_parse_csv_int(args.label_horizon_set) or (1, 5, 20),
        h20_label_name=args.h20_label_name,
        h20_target_type=args.h20_target_type,
        h20_horizon_days=int(args.h20_horizon_days),
        candidate_variants=_parse_csv_text(args.candidate_variants)
        or ("baseline_all_features", "stable_sign_plus_low_collinearity"),
        target_regimes=_parse_csv_text(args.target_regimes)
        or (
            "time_split_early_vs_late:late",
            "high_vs_low_market_breadth:low",
            "high_vs_low_cross_sectional_dispersion:low",
            "high_vs_low_liquidity:low",
        ),
        should_try_min_regime_folds=int(args.should_try_min_regime_folds),
    )
    print("Week 8 statuses:")
    for key, value in result.statuses.items():
        print(f"- {key}: {value}")
    print(f"Week 8 final recommendation: {result.final_recommendation}")
    print(f"should_try_richer_model_now: {result.should_try_richer_model_now}")


if __name__ == "__main__":
    main()
