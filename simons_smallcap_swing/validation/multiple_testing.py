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

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "multiple_testing_mvp_v1"
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
STATUS_RANK = {PASS: 0, WARN: 1, FAIL: 2}

RESULTS_SCHEMA = DataSchema(
    name="multiple_testing_results_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("task_name", "string", nullable=False),
        ColumnSpec("candidate_name", "string", nullable=False),
        ColumnSpec("metric_name", "string", nullable=False),
        ColumnSpec("raw_score", "float64", nullable=True),
        ColumnSpec("raw_rank", "int64", nullable=True),
        ColumnSpec("raw_pvalue", "float64", nullable=True),
        ColumnSpec("adjusted_pvalue_bonferroni", "float64", nullable=True),
        ColumnSpec("adjusted_pvalue_bh", "float64", nullable=True),
        ColumnSpec("testing_status", "string", nullable=False),
        ColumnSpec("message", "string", nullable=False),
        ColumnSpec("source_artifact", "string", nullable=True),
    ),
    primary_key=(),
    allow_extra_columns=True,
)

METRICS_SCHEMA = DataSchema(
    name="multiple_testing_metrics_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("task_name", "string", nullable=False),
        ColumnSpec("task_status", "string", nullable=False),
        ColumnSpec("n_tests_effective", "int64", nullable=False),
        ColumnSpec("n_with_pvalues", "int64", nullable=False),
        ColumnSpec("n_without_pvalues", "int64", nullable=False),
        ColumnSpec("n_surviving_bonferroni", "int64", nullable=False),
        ColumnSpec("n_surviving_bh", "int64", nullable=False),
    ),
    primary_key=("task_name",),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class MultipleTestingResult:
    results_path: Path
    summary_path: Path
    metrics_path: Path
    overall_status: str
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


def _worst_status(values: list[str]) -> str:
    if not values:
        return WARN
    return max(values, key=lambda x: STATUS_RANK.get(x, -1))


def _bh_adjust(pvals: pd.Series, m_total: int) -> pd.Series:
    if pvals.empty:
        return pd.Series(dtype=float)
    order = pvals.sort_values().index.tolist()
    sorted_p = pvals.loc[order].to_numpy(dtype=float)
    m = float(max(m_total, 1))
    ranked = np.array([(sorted_p[idx] * m) / float(idx + 1) for idx in range(len(sorted_p))], dtype=float)
    ranked = np.minimum(ranked, 1.0)
    # Enforce monotonicity from tail to head.
    for idx in range(len(ranked) - 2, -1, -1):
        ranked[idx] = min(ranked[idx], ranked[idx + 1])
    out = pd.Series(index=order, data=ranked, dtype=float)
    return out.reindex(pvals.index)


def _bonferroni_adjust(pvals: pd.Series, m_total: int) -> pd.Series:
    if pvals.empty:
        return pd.Series(dtype=float)
    m = float(max(m_total, 1))
    return (pvals.astype(float) * m).clip(lower=0.0, upper=1.0)


def _build_tests_from_candidate_table(path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = read_parquet(path)
    notes: list[str] = []
    required_base = {"task_name", "metric_name"}
    if not required_base.issubset(frame.columns):
        missing = sorted(required_base - set(frame.columns))
        raise ValueError(f"candidate_tests table missing required columns: {missing}")

    out = frame.copy()
    if "candidate_name" not in out.columns:
        if "comparison_name" in out.columns:
            out["candidate_name"] = out["comparison_name"].astype(str)
            notes.append("candidate_name derived from comparison_name")
        else:
            raise ValueError("candidate_tests table requires candidate_name or comparison_name")

    if "raw_score" not in out.columns:
        out["raw_score"] = np.nan
        notes.append("raw_score missing in candidate table; filled as NaN")
    if "raw_pvalue" not in out.columns:
        out["raw_pvalue"] = np.nan
        notes.append("raw_pvalue missing in candidate table; heuristic mode for affected rows")
    if "source_artifact" not in out.columns:
        out["source_artifact"] = str(path)

    keep_cols = ["task_name", "candidate_name", "metric_name", "raw_score", "raw_pvalue", "source_artifact"]
    out = out[keep_cols].copy()
    out["task_name"] = out["task_name"].astype(str)
    out["candidate_name"] = out["candidate_name"].astype(str)
    out["metric_name"] = out["metric_name"].astype(str)
    out["raw_score"] = pd.to_numeric(out["raw_score"], errors="coerce")
    out["raw_pvalue"] = pd.to_numeric(out["raw_pvalue"], errors="coerce")
    out["source_artifact"] = out["source_artifact"].astype(str)
    return out, notes


def _build_tests_from_cv_table(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"cv_model_comparison_table not found: {path}")
    frame = read_parquet(path)
    required = {"task_name", "model_a_name", "model_b_name", "primary_metric", "mean_delta"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"cv_model_comparison_table missing required columns: {missing}")

    out = frame.copy()
    out["task_name"] = out["task_name"].astype(str)
    out["candidate_name"] = (
        out["model_a_name"].astype(str) + "_vs_" + out["model_b_name"].astype(str)
    )
    out["metric_name"] = out["primary_metric"].astype(str)
    out["raw_score"] = pd.to_numeric(out["mean_delta"], errors="coerce")
    out["raw_pvalue"] = pd.to_numeric(out.get("raw_pvalue"), errors="coerce")
    out["source_artifact"] = str(path)
    return out[["task_name", "candidate_name", "metric_name", "raw_score", "raw_pvalue", "source_artifact"]].copy(), [
        "built test family from cv_model_comparison_table"
    ]


def _build_tests_from_pbo_results(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"pbo_cscv_results not found: {path}")
    frame = read_parquet(path)
    required = {"task_name", "candidate_name", "primary_metric", "out_of_sample_metric"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"pbo_cscv_results missing required columns: {missing}")

    grouped = (
        frame.groupby(["task_name", "candidate_name", "primary_metric"], dropna=False, as_index=False)[
            "out_of_sample_metric"
        ]
        .mean()
        .rename(columns={"primary_metric": "metric_name", "out_of_sample_metric": "raw_score"})
    )
    grouped["raw_pvalue"] = np.nan
    grouped["source_artifact"] = str(path)
    return grouped[["task_name", "candidate_name", "metric_name", "raw_score", "raw_pvalue", "source_artifact"]].copy(), [
        "built heuristic family from pbo_cscv_results (no raw p-values)"
    ]


def _prepare_tests_frame(
    *,
    candidate_tests_path: Path | None,
    cv_table_path: Path,
    pbo_results_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    if candidate_tests_path is not None:
        tests, local_notes = _build_tests_from_candidate_table(candidate_tests_path)
        notes.extend(local_notes)
        return tests, notes

    if cv_table_path.exists():
        tests, local_notes = _build_tests_from_cv_table(cv_table_path)
        notes.extend(local_notes)
        return tests, notes

    if pbo_results_path.exists():
        tests, local_notes = _build_tests_from_pbo_results(pbo_results_path)
        notes.extend(local_notes)
        return tests, notes

    raise FileNotFoundError(
        "No usable input found for multiple_testing. Provide --candidate-tests-path, "
        "or ensure cv_model_comparison_table / pbo_cscv_results artifacts exist."
    )


def run_multiple_testing(
    *,
    candidate_tests_path: str | Path | None = None,
    cv_model_comparison_table_path: str | Path | None = None,
    pbo_cscv_results_path: str | Path | None = None,
    pbo_cscv_summary_path: str | Path | None = None,
    validation_suite_summary_path: str | Path | None = None,
    alpha: float = 0.05,
    output_dir: str | Path | None = None,
    run_id: str = MODULE_VERSION,
) -> MultipleTestingResult:
    logger = get_logger("validation.multiple_testing")
    base = data_dir()

    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0,1).")

    candidate_path = Path(candidate_tests_path).expanduser().resolve() if candidate_tests_path else None
    cv_path = (
        Path(cv_model_comparison_table_path).expanduser().resolve()
        if cv_model_comparison_table_path
        else (base / "models" / "artifacts" / "cv_model_comparison_table.parquet")
    )
    pbo_results_path = (
        Path(pbo_cscv_results_path).expanduser().resolve()
        if pbo_cscv_results_path
        else (base / "validation" / "pbo_cscv_results.parquet")
    )
    pbo_summary_path = (
        Path(pbo_cscv_summary_path).expanduser().resolve()
        if pbo_cscv_summary_path
        else (base / "validation" / "pbo_cscv_summary.json")
    )
    suite_summary_path = (
        Path(validation_suite_summary_path).expanduser().resolve()
        if validation_suite_summary_path
        else (base / "validation" / "validation_suite_summary.json")
    )

    tests, notes = _prepare_tests_frame(
        candidate_tests_path=candidate_path,
        cv_table_path=cv_path,
        pbo_results_path=pbo_results_path,
    )

    if tests.empty:
        raise ValueError("multiple_testing received an empty tests table.")

    required_cols = {"task_name", "candidate_name", "metric_name", "raw_score", "raw_pvalue", "source_artifact"}
    missing = sorted(required_cols - set(tests.columns))
    if missing:
        raise ValueError(f"multiple_testing tests table missing required columns: {missing}")

    tests = tests.copy()
    tests["task_name"] = tests["task_name"].astype(str)
    tests["candidate_name"] = tests["candidate_name"].astype(str)
    tests["metric_name"] = tests["metric_name"].astype(str)
    tests["raw_score"] = pd.to_numeric(tests["raw_score"], errors="coerce")
    tests["raw_pvalue"] = pd.to_numeric(tests["raw_pvalue"], errors="coerce")
    tests["source_artifact"] = tests["source_artifact"].astype(str)

    if tests["task_name"].str.strip().eq("").any():
        raise ValueError("task_name contains empty values.")
    if tests["candidate_name"].str.strip().eq("").any():
        raise ValueError("candidate_name contains empty values.")
    if tests["metric_name"].str.strip().eq("").any():
        raise ValueError("metric_name contains empty values.")

    if tests.duplicated(["task_name", "candidate_name", "metric_name"], keep=False).any():
        raise ValueError("Duplicate logical rows found by (task_name, candidate_name, metric_name).")

    invalid_p = tests["raw_pvalue"].notna() & ~tests["raw_pvalue"].between(0.0, 1.0)
    if invalid_p.any():
        bad = tests.loc[invalid_p, ["task_name", "candidate_name", "raw_pvalue"]].head(5).to_dict("records")
        raise ValueError(f"raw_pvalue must be in [0,1]. Examples: {bad}")

    built_ts_utc = datetime.now(UTC).isoformat()
    cfg_hash = _cfg_hash(
        {
            "module_version": MODULE_VERSION,
            "candidate_tests_path": str(candidate_path) if candidate_path else None,
            "cv_model_comparison_table_path": str(cv_path),
            "pbo_cscv_results_path": str(pbo_results_path),
            "alpha": float(alpha),
            "run_id": run_id,
        }
    )

    result_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    task_statuses: list[str] = []
    missing_pvalue_blocks: list[str] = []
    candidates_surviving_adjustment: dict[str, list[str]] = {}
    candidates_flagged_fragile: dict[str, list[str]] = {}
    n_tests_effective_by_task: dict[str, int] = {}

    for task_name, group in tests.groupby("task_name", sort=True):
        block = group.copy().reset_index(drop=True)
        n_tests = int(len(block))
        n_tests_effective_by_task[task_name] = n_tests

        block["raw_rank"] = pd.Series([None] * len(block), dtype="Int64")
        block["adjusted_pvalue_bonferroni"] = np.nan
        block["adjusted_pvalue_bh"] = np.nan
        block["testing_status"] = ""
        block["message"] = ""

        with_p = block["raw_pvalue"].notna()
        n_with = int(with_p.sum())
        n_without = int((~with_p).sum())

        if n_with == 0:
            task_status = WARN
            missing_pvalue_blocks.append(task_name)
            block["raw_rank"] = (
                block["raw_score"].abs().rank(method="dense", ascending=False).astype("Int64")
                if block["raw_score"].notna().any()
                else pd.Series([pd.NA] * len(block), dtype="Int64")
            )
            block["testing_status"] = "heuristic_only_no_pvalue"
            block["message"] = "No raw p-values available; adjustments skipped (heuristic-only mode)."
            surviving = []
            fragile = sorted(block["candidate_name"].astype(str).tolist())
            n_survive_bonf = 0
            n_survive_bh = 0
        else:
            if n_without > 0:
                task_status = WARN
                missing_pvalue_blocks.append(task_name)
            else:
                task_status = PASS

            available = block.loc[with_p, "raw_pvalue"].astype(float)
            bonf = _bonferroni_adjust(available, m_total=n_tests)
            bh = _bh_adjust(available, m_total=n_tests)

            block.loc[with_p, "adjusted_pvalue_bonferroni"] = bonf.values
            block.loc[with_p, "adjusted_pvalue_bh"] = bh.values
            block.loc[with_p, "raw_rank"] = available.rank(method="dense", ascending=True).astype("Int64")
            if n_without > 0:
                block.loc[~with_p, "raw_rank"] = (
                    block.loc[~with_p, "raw_score"].abs().rank(method="dense", ascending=False).astype("Int64")
                    if block.loc[~with_p, "raw_score"].notna().any()
                    else pd.Series([pd.NA] * n_without, index=block.index[~with_p], dtype="Int64")
                )

            survives_bonf = with_p & (block["adjusted_pvalue_bonferroni"] <= alpha)
            survives_bh = with_p & (block["adjusted_pvalue_bh"] <= alpha)
            survives_any = survives_bonf | survives_bh
            n_survive_bonf = int(survives_bonf.sum())
            n_survive_bh = int(survives_bh.sum())

            block.loc[with_p, "testing_status"] = np.where(
                survives_any.loc[with_p],
                "survives_adjustment",
                "fails_adjustment",
            )
            block.loc[~with_p, "testing_status"] = "missing_raw_pvalue"
            block.loc[with_p, "message"] = "Adjusted with Bonferroni and BH."
            block.loc[~with_p, "message"] = (
                "Raw p-value missing; candidate not tested with multiplicity corrections."
            )

            surviving = sorted(block.loc[survives_any, "candidate_name"].astype(str).tolist())
            fragile = sorted(
                block.loc[~survives_any, "candidate_name"].astype(str).tolist()
            )
            if n_survive_bh == 0 and n_survive_bonf == 0:
                task_status = WARN

        candidates_surviving_adjustment[task_name] = surviving
        candidates_flagged_fragile[task_name] = fragile
        task_statuses.append(task_status)

        metrics_rows.append(
            {
                "task_name": task_name,
                "task_status": task_status,
                "n_tests_effective": int(n_tests),
                "n_with_pvalues": int(n_with),
                "n_without_pvalues": int(n_without),
                "n_surviving_bonferroni": int(n_survive_bonf),
                "n_surviving_bh": int(n_survive_bh),
                "run_id": run_id,
                "config_hash": cfg_hash,
                "built_ts_utc": built_ts_utc,
            }
        )

        for row in block.itertuples(index=False):
            result_rows.append(
                {
                    "task_name": str(row.task_name),
                    "candidate_name": str(row.candidate_name),
                    "metric_name": str(row.metric_name),
                    "raw_score": _to_float(row.raw_score),
                    "raw_rank": int(row.raw_rank) if not pd.isna(row.raw_rank) else None,
                    "raw_pvalue": _to_float(row.raw_pvalue),
                    "adjusted_pvalue_bonferroni": _to_float(row.adjusted_pvalue_bonferroni),
                    "adjusted_pvalue_bh": _to_float(row.adjusted_pvalue_bh),
                    "testing_status": str(row.testing_status),
                    "message": str(row.message),
                    "source_artifact": str(row.source_artifact),
                    "run_id": run_id,
                    "config_hash": cfg_hash,
                    "built_ts_utc": built_ts_utc,
                }
            )

    results_df = pd.DataFrame(result_rows).sort_values(["task_name", "candidate_name"]).reset_index(drop=True)
    metrics_df = pd.DataFrame(metrics_rows).sort_values(["task_name"]).reset_index(drop=True)

    for col in ("task_name", "candidate_name", "metric_name", "testing_status", "message", "source_artifact"):
        results_df[col] = results_df[col].astype("string")
    for col in ("raw_score", "raw_pvalue", "adjusted_pvalue_bonferroni", "adjusted_pvalue_bh"):
        results_df[col] = pd.to_numeric(results_df[col], errors="coerce")
    results_df["raw_rank"] = pd.to_numeric(results_df["raw_rank"], errors="coerce").astype("Int64")

    for col in ("task_name", "task_status"):
        metrics_df[col] = metrics_df[col].astype("string")
    for col in (
        "n_tests_effective",
        "n_with_pvalues",
        "n_without_pvalues",
        "n_surviving_bonferroni",
        "n_surviving_bh",
    ):
        metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce").fillna(0).astype("int64")

    assert_schema(results_df, RESULTS_SCHEMA)
    assert_schema(metrics_df, METRICS_SCHEMA)

    overall_status = _worst_status(task_statuses)
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (base / "validation")
    target_dir.mkdir(parents=True, exist_ok=True)

    results_path = write_parquet(
        results_df,
        target_dir / "multiple_testing_results.parquet",
        schema_name=RESULTS_SCHEMA.name,
        run_id=run_id,
    )
    metrics_path = write_parquet(
        metrics_df,
        target_dir / "multiple_testing_metrics.parquet",
        schema_name=METRICS_SCHEMA.name,
        run_id=run_id,
    )

    # Optional context from pbo / validation suite.
    pbo_context = None
    if pbo_summary_path.exists():
        try:
            pbo_context = json.loads(pbo_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            notes.append("pbo_cscv_summary.json exists but is invalid JSON")
    suite_context = None
    if suite_summary_path.exists():
        try:
            suite_context = json.loads(suite_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            notes.append("validation_suite_summary.json exists but is invalid JSON")

    summary_payload = {
        "built_ts_utc": built_ts_utc,
        "run_id": run_id,
        "module_version": MODULE_VERSION,
        "config_hash": cfg_hash,
        "overall_status": overall_status,
        "tasks_evaluated": sorted(metrics_df["task_name"].astype(str).unique().tolist()),
        "task_status_by_task": {
            row.task_name: row.task_status for row in metrics_df[["task_name", "task_status"]].itertuples(index=False)
        },
        "n_tests_effective_by_task": n_tests_effective_by_task,
        "correction_methods_used": ["bonferroni", "benjamini_hochberg"],
        "candidates_flagged_fragile": candidates_flagged_fragile,
        "candidates_surviving_adjustment": candidates_surviving_adjustment,
        "missing_pvalue_blocks": sorted(set(missing_pvalue_blocks)),
        "notes": notes,
        "input_artifacts": {
            "candidate_tests_path": str(candidate_path) if candidate_path else None,
            "cv_model_comparison_table": str(cv_path),
            "pbo_cscv_results": str(pbo_results_path),
            "pbo_cscv_summary": str(pbo_summary_path),
            "validation_suite_summary": str(suite_summary_path),
        },
        "context": {
            "pbo_overall_status": pbo_context.get("overall_status") if isinstance(pbo_context, dict) else None,
            "validation_suite_overall_status": suite_context.get("overall_status")
            if isinstance(suite_context, dict)
            else None,
        },
        "output_paths": {
            "multiple_testing_results": str(results_path),
            "multiple_testing_metrics": str(metrics_path),
        },
    }
    summary_path = target_dir / "multiple_testing_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "multiple_testing_completed",
        run_id=run_id,
        overall_status=overall_status,
        results_path=str(results_path),
        summary_path=str(summary_path),
    )
    return MultipleTestingResult(
        results_path=results_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        overall_status=overall_status,
        config_hash=cfg_hash,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MVP multiple-testing control over candidate comparison families."
    )
    parser.add_argument("--candidate-tests-path", type=str, default=None)
    parser.add_argument("--cv-model-comparison-table-path", type=str, default=None)
    parser.add_argument("--pbo-cscv-results-path", type=str, default=None)
    parser.add_argument("--pbo-cscv-summary-path", type=str, default=None)
    parser.add_argument("--validation-suite-summary-path", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_multiple_testing(
        candidate_tests_path=args.candidate_tests_path,
        cv_model_comparison_table_path=args.cv_model_comparison_table_path,
        pbo_cscv_results_path=args.pbo_cscv_results_path,
        pbo_cscv_summary_path=args.pbo_cscv_summary_path,
        validation_suite_summary_path=args.validation_suite_summary_path,
        alpha=args.alpha,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print("Multiple testing audit completed:")
    print(f"- results: {result.results_path}")
    print(f"- metrics: {result.metrics_path}")
    print(f"- summary: {result.summary_path}")
    print(f"- overall_status: {result.overall_status}")


if __name__ == "__main__":
    main()
