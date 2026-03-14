from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time
from typing import Callable

# Allow direct script execution: `python simons_smallcap_swing/run_week7_validation.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from simons_core.io.paths import data_dir
from validation.leakage_audit import run_leakage_audit
from validation.multiple_testing import run_multiple_testing
from validation.pbo_cscv import run_pbo_cscv
from validation.validation_suite import run_validation_suite


@dataclass(frozen=True)
class Week7RunResult:
    run_prefix: str
    data_root: Path
    manifest_path: Path
    artifacts: dict[str, Path]
    statuses: dict[str, str]


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


def _ensure_week7_prerequisites(required_paths: dict[str, Path]) -> dict[str, Path]:
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Week 7 runner requires existing Week 3-6 artifacts for leakage checks. "
            f"Missing: {missing}. Paths checked: "
            + ", ".join(f"{name}={path}" for name, path in required_paths.items())
        )
    return required_paths


def run_week7_validation(
    *,
    run_prefix: str = "week7_validation",
    data_root: str | Path | None = None,
    labels_path: str | Path | None = None,
    features_path: str | Path | None = None,
    model_dataset_path: str | Path | None = None,
    trading_calendar_path: str | Path | None = None,
    purged_splits_path: str | Path | None = None,
    purged_cv_folds_path: str | Path | None = None,
    fundamentals_pit_path: str | Path | None = None,
    backtest_diagnostics_summary_path: str | Path | None = None,
    cv_model_comparison_summary_path: str | Path | None = None,
    decile_analysis_summary_path: str | Path | None = None,
    paper_portfolio_summary_path: str | Path | None = None,
    ridge_cv_path: str | Path | None = None,
    dummy_regressor_cv_path: str | Path | None = None,
    logistic_cv_path: str | Path | None = None,
    dummy_classifier_cv_path: str | Path | None = None,
    candidate_tests_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_partitions: int = 64,
    seed: int = 42,
    alpha: float = 0.05,
) -> Week7RunResult:
    base_data = Path(data_root).expanduser().resolve() if data_root else data_dir()
    validation_root = Path(output_dir).expanduser().resolve() if output_dir else (base_data / "validation")
    validation_root.mkdir(parents=True, exist_ok=True)

    required = _ensure_week7_prerequisites(
        {
            "labels_forward": _resolve_path(labels_path, base_data / "labels" / "labels_forward.parquet"),
            "features_matrix": _resolve_path(features_path, base_data / "features" / "features_matrix.parquet"),
            "model_dataset": _resolve_path(model_dataset_path, base_data / "datasets" / "model_dataset.parquet"),
            "trading_calendar": _resolve_path(
                trading_calendar_path, base_data / "reference" / "trading_calendar.parquet"
            ),
        }
    )

    optional_sources = {
        "purged_splits": _resolve_optional_existing(
            purged_splits_path, base_data / "labels" / "purged_splits.parquet"
        ),
        "purged_cv_folds": _resolve_optional_existing(
            purged_cv_folds_path, base_data / "labels" / "purged_cv_folds.parquet"
        ),
        "fundamentals_pit": _resolve_optional_existing(
            fundamentals_pit_path, base_data / "edgar" / "fundamentals_pit.parquet"
        ),
        "backtest_diagnostics_summary": _resolve_path(
            backtest_diagnostics_summary_path,
            base_data / "backtest" / "backtest_diagnostics_summary.json",
        ),
        "cv_model_comparison_summary": _resolve_path(
            cv_model_comparison_summary_path,
            base_data / "models" / "artifacts" / "cv_model_comparison_summary.json",
        ),
        "cv_model_comparison_table": _resolve_path(
            None, base_data / "models" / "artifacts" / "cv_model_comparison_table.parquet"
        ),
        "decile_analysis_summary": _resolve_path(
            decile_analysis_summary_path, base_data / "signals" / "decile_analysis_summary.json"
        ),
        "paper_portfolio_summary": _resolve_path(
            paper_portfolio_summary_path, base_data / "signals" / "paper_portfolio_summary.json"
        ),
        "ridge_cv": _resolve_path(
            ridge_cv_path,
            base_data / "models" / "artifacts" / "ridge_cv" / "cv_baseline_fold_metrics.parquet",
        ),
        "dummy_regressor_cv": _resolve_path(
            dummy_regressor_cv_path,
            base_data / "models" / "artifacts" / "dummy_regressor_cv" / "cv_baseline_fold_metrics.parquet",
        ),
        "logistic_cv": _resolve_path(
            logistic_cv_path,
            base_data / "models" / "artifacts" / "logistic_cv" / "cv_baseline_fold_metrics.parquet",
        ),
        "dummy_classifier_cv": _resolve_path(
            dummy_classifier_cv_path,
            base_data / "models" / "artifacts" / "dummy_classifier_cv" / "cv_baseline_fold_metrics.parquet",
        ),
        "candidate_tests": _resolve_optional_existing(
            candidate_tests_path, base_data / "validation" / "candidate_tests.parquet"
        ),
    }

    total_steps = 4
    step = 1
    statuses: dict[str, str] = {}

    leakage = _run_step(
        step,
        total_steps,
        "run leakage audit",
        run_leakage_audit,
        labels_path=required["labels_forward"],
        features_path=required["features_matrix"],
        model_dataset_path=required["model_dataset"],
        trading_calendar_path=required["trading_calendar"],
        purged_splits_path=optional_sources["purged_splits"],
        purged_cv_folds_path=optional_sources["purged_cv_folds"],
        fundamentals_pit_path=optional_sources["fundamentals_pit"],
        output_dir=validation_root,
        run_id=_run_id(run_prefix, "leakage_audit"),
    )
    statuses["leakage_audit"] = "DONE"
    step += 1

    suite = _run_step(
        step,
        total_steps,
        "run validation suite",
        run_validation_suite,
        leakage_summary_path=leakage.summary_path,
        backtest_diagnostics_summary_path=optional_sources["backtest_diagnostics_summary"],
        cv_model_comparison_summary_path=optional_sources["cv_model_comparison_summary"],
        decile_analysis_summary_path=optional_sources["decile_analysis_summary"],
        paper_portfolio_summary_path=optional_sources["paper_portfolio_summary"],
        output_dir=validation_root,
        run_id=_run_id(run_prefix, "validation_suite"),
    )
    statuses["validation_suite"] = "DONE"
    step += 1

    pbo = _run_step(
        step,
        total_steps,
        "run pbo/cscv",
        run_pbo_cscv,
        ridge_cv_path=optional_sources["ridge_cv"],
        dummy_regressor_cv_path=optional_sources["dummy_regressor_cv"],
        logistic_cv_path=optional_sources["logistic_cv"],
        dummy_classifier_cv_path=optional_sources["dummy_classifier_cv"],
        max_partitions=int(max_partitions),
        seed=int(seed),
        output_dir=validation_root,
        run_id=_run_id(run_prefix, "pbo_cscv"),
    )
    statuses["pbo_cscv"] = "DONE"
    step += 1

    multiple = _run_step(
        step,
        total_steps,
        "run multiple testing",
        run_multiple_testing,
        candidate_tests_path=optional_sources["candidate_tests"],
        cv_model_comparison_table_path=optional_sources["cv_model_comparison_table"],
        pbo_cscv_results_path=pbo.results_path,
        pbo_cscv_summary_path=pbo.summary_path,
        validation_suite_summary_path=suite.summary_path,
        alpha=float(alpha),
        output_dir=validation_root,
        run_id=_run_id(run_prefix, "multiple_testing"),
    )
    statuses["multiple_testing"] = "DONE"

    artifacts: dict[str, Path] = {
        "leakage_audit_findings": leakage.findings_path,
        "leakage_audit_metrics": leakage.metrics_path,
        "leakage_audit_summary": leakage.summary_path,
        "validation_suite_findings": suite.findings_path,
        "validation_suite_metrics": suite.metrics_path,
        "validation_suite_summary": suite.summary_path,
        "pbo_cscv_results": pbo.results_path,
        "pbo_cscv_partitions": pbo.partitions_path,
        "pbo_cscv_summary": pbo.summary_path,
        "multiple_testing_results": multiple.results_path,
        "multiple_testing_metrics": multiple.metrics_path,
        "multiple_testing_summary": multiple.summary_path,
    }

    manifest_payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_prefix": run_prefix,
        "data_root": str(base_data),
        "steps_total": total_steps,
        "flags": {
            "max_partitions": int(max_partitions),
            "seed": int(seed),
            "alpha": float(alpha),
        },
        "statuses": statuses,
        "status_summary": {
            "leakage_overall_status": leakage.overall_status,
            "validation_suite_overall_status": suite.overall_status,
            "pbo_cscv_overall_status": pbo.overall_status,
            "multiple_testing_overall_status": multiple.overall_status,
        },
        "prerequisites": {key: str(path) for key, path in required.items()},
        "optional_inputs": {
            key: (str(path) if path is not None else None)
            for key, path in optional_sources.items()
        },
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }
    manifest_path = base_data / f"week7_validation_manifest_{run_prefix}.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] Week 7 validation pipeline completed. Manifest: {manifest_path}")

    return Week7RunResult(
        run_prefix=run_prefix,
        data_root=base_data,
        manifest_path=manifest_path,
        artifacts=artifacts,
        statuses=statuses,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Week 7 strong-validation pipeline (leakage + suite + PBO/CSCV + multiple testing)."
    )
    parser.add_argument("--run-prefix", type=str, default="week7_validation")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--labels-path", type=str, default=None)
    parser.add_argument("--features-path", type=str, default=None)
    parser.add_argument("--model-dataset-path", type=str, default=None)
    parser.add_argument("--trading-calendar-path", type=str, default=None)
    parser.add_argument("--purged-splits-path", type=str, default=None)
    parser.add_argument("--purged-cv-folds-path", type=str, default=None)
    parser.add_argument("--fundamentals-pit-path", type=str, default=None)
    parser.add_argument("--backtest-diagnostics-summary-path", type=str, default=None)
    parser.add_argument("--cv-model-comparison-summary-path", type=str, default=None)
    parser.add_argument("--decile-analysis-summary-path", type=str, default=None)
    parser.add_argument("--paper-portfolio-summary-path", type=str, default=None)
    parser.add_argument("--ridge-cv-path", type=str, default=None)
    parser.add_argument("--dummy-regressor-cv-path", type=str, default=None)
    parser.add_argument("--logistic-cv-path", type=str, default=None)
    parser.add_argument("--dummy-classifier-cv-path", type=str, default=None)
    parser.add_argument("--candidate-tests-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-partitions", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_week7_validation(
        run_prefix=args.run_prefix,
        data_root=args.data_root,
        labels_path=args.labels_path,
        features_path=args.features_path,
        model_dataset_path=args.model_dataset_path,
        trading_calendar_path=args.trading_calendar_path,
        purged_splits_path=args.purged_splits_path,
        purged_cv_folds_path=args.purged_cv_folds_path,
        fundamentals_pit_path=args.fundamentals_pit_path,
        backtest_diagnostics_summary_path=args.backtest_diagnostics_summary_path,
        cv_model_comparison_summary_path=args.cv_model_comparison_summary_path,
        decile_analysis_summary_path=args.decile_analysis_summary_path,
        paper_portfolio_summary_path=args.paper_portfolio_summary_path,
        ridge_cv_path=args.ridge_cv_path,
        dummy_regressor_cv_path=args.dummy_regressor_cv_path,
        logistic_cv_path=args.logistic_cv_path,
        dummy_classifier_cv_path=args.dummy_classifier_cv_path,
        candidate_tests_path=args.candidate_tests_path,
        output_dir=args.output_dir,
        max_partitions=args.max_partitions,
        seed=args.seed,
        alpha=args.alpha,
    )
    print("Week 7 statuses:")
    for key, value in result.statuses.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
