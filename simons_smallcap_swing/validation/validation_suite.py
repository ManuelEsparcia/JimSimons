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

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simons_core.io.parquet_store import write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "validation_suite_mvp_v1"
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
RANK = {PASS: 0, WARN: 1, FAIL: 2}

BLOCKS = (
    "leakage_integrity",
    "cv_comparison_robustness",
    "signal_quality",
    "portfolio_backtest_sanity",
)

FINDINGS_SCHEMA = DataSchema(
    name="validation_suite_findings_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("validation_block", "string", nullable=False),
        ColumnSpec("metric_name", "string", nullable=False),
        ColumnSpec("severity", "string", nullable=False),
        ColumnSpec("status", "string", nullable=False),
        ColumnSpec("observed_value", "string", nullable=True),
        ColumnSpec("threshold_or_rule", "string", nullable=False),
        ColumnSpec("message", "string", nullable=False),
        ColumnSpec("source_artifact", "string", nullable=True),
    ),
    primary_key=(),
    allow_extra_columns=True,
)

METRICS_SCHEMA = DataSchema(
    name="validation_suite_metrics_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("validation_block", "string", nullable=False),
        ColumnSpec("block_status", "string", nullable=False),
        ColumnSpec("n_findings", "int64", nullable=False),
        ColumnSpec("n_pass", "int64", nullable=False),
        ColumnSpec("n_warn", "int64", nullable=False),
        ColumnSpec("n_fail", "int64", nullable=False),
        ColumnSpec("n_missing", "int64", nullable=False),
        ColumnSpec("source_artifact", "string", nullable=True),
    ),
    primary_key=("validation_block",),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class ValidationSuiteResult:
    findings_path: Path
    summary_path: Path
    metrics_path: Path
    overall_status: str
    config_hash: str


def _max_status(items: list[str]) -> str:
    if not items:
        return WARN
    return max(items, key=lambda v: RANK.get(v, -1))


def _cfg_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _to_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def _load_json_optional(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "artifact_missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(payload, dict):
        return None, "invalid_json_object"
    return payload, None


def _emit(
    rows: list[dict[str, Any]],
    *,
    block: str,
    metric_name: str,
    severity: str,
    status: str,
    observed: Any,
    rule: str,
    message: str,
    source: Path,
) -> None:
    rows.append(
        {
            "validation_block": block,
            "metric_name": metric_name,
            "severity": severity,
            "status": status,
            "observed_value": _to_text(observed),
            "threshold_or_rule": rule,
            "message": message,
            "source_artifact": str(source),
        }
    )


def _check_leakage(rows: list[dict[str, Any]], path: Path) -> str:
    payload, err = _load_json_optional(path)
    if err:
        _emit(rows, block=BLOCKS[0], metric_name="leakage_summary_available", severity=WARN, status="missing_input", observed=err, rule="Leakage summary should exist and be parseable.", message="Leakage summary unavailable.", source=path)
        return WARN
    status = str(payload.get("overall_status", "UNKNOWN")).upper()
    sev = status if status in {PASS, WARN, FAIL} else WARN
    _emit(rows, block=BLOCKS[0], metric_name="leakage_overall_status", severity=sev, status="evaluated", observed=status, rule="Leakage FAIL is structural blocker.", message=f"leakage overall_status={status}", source=path)
    return sev


def _check_cv(rows: list[dict[str, Any]], path: Path) -> str:
    payload, err = _load_json_optional(path)
    if err:
        _emit(rows, block=BLOCKS[1], metric_name="cv_summary_available", severity=WARN, status="missing_input", observed=err, rule="CV summary should exist and be parseable.", message="CV comparison summary unavailable.", source=path)
        return WARN
    sev_items = [PASS]
    for task in ("regression", "classification"):
        block = payload.get(task, {})
        if not isinstance(block, dict):
            _emit(rows, block=BLOCKS[1], metric_name=f"{task}_block_present", severity=WARN, status="missing_input", observed=block, rule=f"{task} block required.", message=f"{task} block missing.", source=path)
            sev_items.append(WARN)
            continue
        comp = str(block.get("comparability_status", "unknown")).lower()
        comp_sev = PASS if comp == "comparable" else WARN
        _emit(rows, block=BLOCKS[1], metric_name=f"{task}_comparability", severity=comp_sev, status="evaluated", observed=comp, rule="comparability_status should be comparable.", message=f"{task} comparability={comp}", source=path)
        sev_items.append(comp_sev)
        winner = str(block.get("winner_global", "unknown")).lower()
        winner_sev = PASS if (comp == "comparable" and winner == "model_a") else WARN
        _emit(rows, block=BLOCKS[1], metric_name=f"{task}_winner_global", severity=winner_sev, status="evaluated", observed=winner, rule="winner_global should favor model_a over dummy.", message=f"{task} winner={winner}", source=path)
        sev_items.append(winner_sev)
    return _max_status(sev_items)


def _check_signal(rows: list[dict[str, Any]], path: Path) -> str:
    payload, err = _load_json_optional(path)
    if err:
        _emit(rows, block=BLOCKS[2], metric_name="signal_summary_available", severity=WARN, status="missing_input", observed=err, rule="Decile summary should exist and be parseable.", message="Decile summary unavailable.", source=path)
        return WARN
    sev_items = [PASS]
    spread = _to_float(payload.get("mean_top_minus_bottom_spread"))
    sev = PASS if spread is not None and spread > 0 else WARN
    _emit(rows, block=BLOCKS[2], metric_name="mean_top_minus_bottom_spread", severity=sev, status="evaluated", observed=spread, rule="Spread should be > 0.", message="Signal spread check.", source=path)
    sev_items.append(sev)
    pos_rate = _to_float(payload.get("positive_spread_rate"))
    if pos_rate is None:
        sev = WARN
    elif pos_rate < 0.40:
        sev = FAIL
    elif pos_rate < 0.50:
        sev = WARN
    else:
        sev = PASS
    _emit(rows, block=BLOCKS[2], metric_name="positive_spread_rate", severity=sev, status="evaluated", observed=pos_rate, rule=">=0.50 preferred; <0.40 critical.", message="Signal positive spread rate check.", source=path)
    sev_items.append(sev)
    mono = _to_float(payload.get("monotonicity_score"))
    sev = PASS if mono is not None and mono > 0 else WARN
    _emit(rows, block=BLOCKS[2], metric_name="monotonicity_score", severity=sev, status="evaluated", observed=mono, rule="Monotonicity should be > 0.", message="Signal monotonicity check.", source=path)
    sev_items.append(sev)
    return _max_status(sev_items)


def _check_portfolio(rows: list[dict[str, Any]], backtest_path: Path, paper_path: Path) -> str:
    back_payload, back_err = _load_json_optional(backtest_path)
    paper_payload, paper_err = _load_json_optional(paper_path)
    sev_items: list[str] = []
    if back_err and paper_err:
        _emit(rows, block=BLOCKS[3], metric_name="portfolio_summary_available", severity=WARN, status="missing_input", observed={"backtest": back_err, "paper": paper_err}, rule="At least backtest or paper summary should exist.", message="Both backtest and paper summaries unavailable.", source=backtest_path)
        return WARN
    if not back_err:
        drawdown = _to_float(back_payload.get("max_drawdown_net_all_modes"))
        if drawdown is None:
            sev = WARN
        elif drawdown > 0 or drawdown <= -0.50:
            sev = FAIL
        elif drawdown <= -0.25:
            sev = WARN
        else:
            sev = PASS
        _emit(rows, block=BLOCKS[3], metric_name="max_drawdown_net_all_modes", severity=sev, status="evaluated", observed=drawdown, rule="range [-0.25,0] preferred; <=-0.5 critical.", message="Backtest drawdown check.", source=backtest_path)
        sev_items.append(sev)
        cost_drag = _to_float(back_payload.get("mean_cost_drag"))
        if cost_drag is None:
            sev = WARN
        elif cost_drag < 0 or cost_drag > 0.01:
            sev = FAIL
        elif cost_drag > 0.003:
            sev = WARN
        else:
            sev = PASS
        _emit(rows, block=BLOCKS[3], metric_name="mean_cost_drag", severity=sev, status="evaluated", observed=cost_drag, rule="0<=cost_drag<=0.003 preferred; >0.01 critical.", message="Backtest cost drag check.", source=backtest_path)
        sev_items.append(sev)
    if not paper_err:
        mode_summaries = paper_payload.get("mode_summaries")
        if isinstance(mode_summaries, list) and mode_summaries:
            best_hit = max((_to_float(item.get("positive_return_rate")) or float("-inf")) for item in mode_summaries)
            if best_hit >= 0.50:
                sev = PASS
            elif best_hit >= 0.40:
                sev = WARN
            else:
                sev = FAIL
            _emit(rows, block=BLOCKS[3], metric_name="paper_best_positive_return_rate", severity=sev, status="evaluated", observed=best_hit, rule=">=0.50 preferred; <0.40 critical.", message="Paper portfolio hit-rate check.", source=paper_path)
            sev_items.append(sev)
        else:
            _emit(rows, block=BLOCKS[3], metric_name="paper_mode_summaries_present", severity=WARN, status="missing_input", observed=mode_summaries, rule="mode_summaries should be non-empty.", message="Paper portfolio summary has no mode_summaries.", source=paper_path)
            sev_items.append(WARN)
    return _max_status(sev_items or [WARN])


def run_validation_suite(
    *,
    leakage_summary_path: str | Path | None = None,
    backtest_diagnostics_summary_path: str | Path | None = None,
    cv_model_comparison_summary_path: str | Path | None = None,
    decile_analysis_summary_path: str | Path | None = None,
    paper_portfolio_summary_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    run_id: str = MODULE_VERSION,
) -> ValidationSuiteResult:
    base = data_dir()
    leakage_path = Path(leakage_summary_path).expanduser().resolve() if leakage_summary_path else (base / "validation" / "leakage_audit_summary.json")
    backtest_path = Path(backtest_diagnostics_summary_path).expanduser().resolve() if backtest_diagnostics_summary_path else (base / "backtest" / "backtest_diagnostics_summary.json")
    cv_path = Path(cv_model_comparison_summary_path).expanduser().resolve() if cv_model_comparison_summary_path else (base / "models" / "artifacts" / "cv_model_comparison_summary.json")
    decile_path = Path(decile_analysis_summary_path).expanduser().resolve() if decile_analysis_summary_path else (base / "signals" / "decile_analysis_summary.json")
    paper_path = Path(paper_portfolio_summary_path).expanduser().resolve() if paper_portfolio_summary_path else (base / "signals" / "paper_portfolio_summary.json")

    findings_rows: list[dict[str, Any]] = []
    block_status = {
        BLOCKS[0]: _check_leakage(findings_rows, leakage_path),
        BLOCKS[1]: _check_cv(findings_rows, cv_path),
        BLOCKS[2]: _check_signal(findings_rows, decile_path),
        BLOCKS[3]: _check_portfolio(findings_rows, backtest_path, paper_path),
    }

    if block_status[BLOCKS[0]] == FAIL:
        overall = FAIL
    elif any(value == FAIL for value in block_status.values()):
        overall = FAIL
    elif any(value == WARN for value in block_status.values()):
        overall = WARN
    else:
        overall = PASS

    built_ts_utc = datetime.now(UTC).isoformat()
    cfg_hash = _cfg_hash(
        {
            "module_version": MODULE_VERSION,
            "leakage_summary_path": str(leakage_path),
            "backtest_diagnostics_summary_path": str(backtest_path),
            "cv_model_comparison_summary_path": str(cv_path),
            "decile_analysis_summary_path": str(decile_path),
            "paper_portfolio_summary_path": str(paper_path),
            "block_status": block_status,
            "overall_status": overall,
        }
    )

    findings_df = pd.DataFrame(findings_rows)
    for col in ("validation_block", "metric_name", "severity", "status", "observed_value", "threshold_or_rule", "message", "source_artifact"):
        findings_df[col] = findings_df[col].astype("string")
    findings_df["run_id"] = run_id
    findings_df["config_hash"] = cfg_hash
    findings_df["built_ts_utc"] = built_ts_utc
    assert_schema(findings_df, FINDINGS_SCHEMA)

    metrics_rows: list[dict[str, Any]] = []
    for block in BLOCKS:
        subset = findings_df[findings_df["validation_block"] == block]
        metrics_rows.append(
            {
                "validation_block": block,
                "block_status": block_status[block],
                "n_findings": int(len(subset)),
                "n_pass": int((subset["severity"] == PASS).sum()),
                "n_warn": int((subset["severity"] == WARN).sum()),
                "n_fail": int((subset["severity"] == FAIL).sum()),
                "n_missing": int((subset["status"] == "missing_input").sum()),
                "source_artifact": ", ".join(sorted(subset["source_artifact"].dropna().astype(str).unique().tolist())) or None,
            }
        )
    metrics_df = pd.DataFrame(metrics_rows)
    for col in ("validation_block", "block_status", "source_artifact"):
        metrics_df[col] = metrics_df[col].astype("string")
    for col in ("n_findings", "n_pass", "n_warn", "n_fail", "n_missing"):
        metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce").fillna(0).astype("int64")
    metrics_df["run_id"] = run_id
    metrics_df["config_hash"] = cfg_hash
    metrics_df["built_ts_utc"] = built_ts_utc
    assert_schema(metrics_df, METRICS_SCHEMA)

    target = Path(output_dir).expanduser().resolve() if output_dir else (base / "validation")
    target.mkdir(parents=True, exist_ok=True)
    findings_path = write_parquet(findings_df, target / "validation_suite_findings.parquet", schema_name=FINDINGS_SCHEMA.name, run_id=run_id)
    metrics_path = write_parquet(metrics_df, target / "validation_suite_metrics.parquet", schema_name=METRICS_SCHEMA.name, run_id=run_id)

    failed_blocks = [k for k, v in block_status.items() if v == FAIL]
    warning_blocks = [k for k, v in block_status.items() if v == WARN]
    summary = {
        "built_ts_utc": built_ts_utc,
        "run_id": run_id,
        "config_hash": cfg_hash,
        "module_version": MODULE_VERSION,
        "overall_status": overall,
        "n_blocks_evaluated": int(len(BLOCKS)),
        "leakage_status": block_status[BLOCKS[0]],
        "cv_comparison_status": block_status[BLOCKS[1]],
        "signal_quality_status": block_status[BLOCKS[2]],
        "portfolio_backtest_status": block_status[BLOCKS[3]],
        "failed_blocks": failed_blocks,
        "warning_blocks": warning_blocks,
        "key_findings": findings_df.loc[findings_df["severity"].isin([WARN, FAIL]), "message"].astype(str).head(10).tolist(),
        "input_artifacts": {
            "leakage_audit_summary": {"path": str(leakage_path), "exists": bool(leakage_path.exists())},
            "backtest_diagnostics_summary": {"path": str(backtest_path), "exists": bool(backtest_path.exists())},
            "cv_model_comparison_summary": {"path": str(cv_path), "exists": bool(cv_path.exists())},
            "decile_analysis_summary": {"path": str(decile_path), "exists": bool(decile_path.exists())},
            "paper_portfolio_summary": {"path": str(paper_path), "exists": bool(paper_path.exists())},
        },
        "output_paths": {
            "validation_suite_findings": str(findings_path),
            "validation_suite_metrics": str(metrics_path),
        },
    }
    summary_path = target / "validation_suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    get_logger("validation.validation_suite").info("validation_suite_completed", overall_status=overall, summary_path=str(summary_path))
    return ValidationSuiteResult(findings_path=findings_path, summary_path=summary_path, metrics_path=metrics_path, overall_status=overall, config_hash=cfg_hash)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run unified validation suite over leakage + CV + signal + portfolio/backtest summaries.")
    parser.add_argument("--leakage-summary-path", type=str, default=None)
    parser.add_argument("--backtest-diagnostics-summary-path", type=str, default=None)
    parser.add_argument("--cv-model-comparison-summary-path", type=str, default=None)
    parser.add_argument("--decile-analysis-summary-path", type=str, default=None)
    parser.add_argument("--paper-portfolio-summary-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_validation_suite(
        leakage_summary_path=args.leakage_summary_path,
        backtest_diagnostics_summary_path=args.backtest_diagnostics_summary_path,
        cv_model_comparison_summary_path=args.cv_model_comparison_summary_path,
        decile_analysis_summary_path=args.decile_analysis_summary_path,
        paper_portfolio_summary_path=args.paper_portfolio_summary_path,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print("Validation suite completed:")
    print(f"- findings: {result.findings_path}")
    print(f"- metrics: {result.metrics_path}")
    print(f"- summary: {result.summary_path}")
    print(f"- overall_status: {result.overall_status}")


if __name__ == "__main__":
    main()
