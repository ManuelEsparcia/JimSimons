from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

# Allow direct script execution: `python simons_smallcap_swing/data/edgar/edgar_qc.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, validate_schema

ALLOWED_METRICS: tuple[str, ...] = (
    "revenue",
    "net_income",
    "total_assets",
    "shares_outstanding",
)
SEVERITY_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}
NULL_STRINGS = {"", "none", "nan", "null", "nat"}

FUNDAMENTALS_SCHEMA = DataSchema(
    name="edgar_fundamentals_pit_v2",
    version="2.0.0",
    columns=(
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("cik", "string", nullable=False),
        ColumnSpec("asof_date", "datetime64[ns, UTC]", nullable=False),
        ColumnSpec("acceptance_ts", "datetime64[ns, UTC]", nullable=False),
        ColumnSpec("filing_date", "datetime64", nullable=False),
        ColumnSpec("fact_end_date", "datetime64", nullable=True),
        ColumnSpec("metric_name", "string", nullable=False),
        ColumnSpec("metric_value", "number", nullable=False),
        ColumnSpec("metric_unit", "string", nullable=False),
        ColumnSpec("accession_number", "string", nullable=True),
        ColumnSpec("source_type", "string", nullable=False),
        ColumnSpec("data_quality", "string", nullable=False),
        ColumnSpec("visibility_rule", "string", nullable=False),
    ),
    primary_key=("instrument_id", "asof_date", "metric_name"),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class EdgarQCResult:
    gate_status: str
    n_fail: int
    n_warn: int
    summary_path: Path
    row_level_path: Path
    failures_path: Path
    metrics_path: Path
    manifest_path: Path


def _max_severity(values: pd.Series) -> str:
    if values.empty:
        return "PASS"
    max_order = int(values.map(SEVERITY_ORDER).max())
    for severity, order in SEVERITY_ORDER.items():
        if order == max_order:
            return severity
    return "PASS"


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' has invalid dates.")
    return parsed.dt.normalize()


def _normalize_string(value: object) -> str:
    text = str(value).strip()
    if text.lower() in NULL_STRINGS:
        return ""
    return text


def _record_issue(
    issues: list[dict[str, Any]],
    *,
    check_name: str,
    severity: str,
    message: str,
    asof_date: object = None,
    instrument_id: object = None,
    ticker: object = None,
    cik: object = None,
    metric_name: object = None,
    observed_value: object = None,
    threshold: object = None,
) -> None:
    if asof_date is None:
        normalized_asof = pd.NaT
    else:
        ts = pd.Timestamp(asof_date)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        normalized_asof = ts

    issues.append(
        {
            "asof_date": normalized_asof,
            "instrument_id": instrument_id,
            "ticker": ticker,
            "cik": cik,
            "metric_name": metric_name,
            "check_name": check_name,
            "severity": severity,
            "observed_value": str(observed_value) if observed_value is not None else "",
            "threshold": str(threshold) if threshold is not None else "",
            "message": message,
        }
    )


def _build_metrics_summary(
    pit: pd.DataFrame,
    row_level: pd.DataFrame,
) -> pd.DataFrame:
    if pit.empty:
        return pd.DataFrame(
            [
                {
                    "metric_name": "__NONE__",
                    "row_count": 0,
                    "n_instruments": 0,
                    "n_sessions": 0,
                    "min_metric_value": 0.0,
                    "max_metric_value": 0.0,
                    "n_fail": int((row_level["severity"] == "FAIL").sum()) if not row_level.empty else 0,
                    "n_warn": int((row_level["severity"] == "WARN").sum()) if not row_level.empty else 0,
                }
            ]
        )

    summary = (
        pit.groupby("metric_name", as_index=False)
        .agg(
            row_count=("instrument_id", "size"),
            n_instruments=("instrument_id", "nunique"),
            n_sessions=("asof_date", "nunique"),
            min_metric_value=("metric_value", "min"),
            max_metric_value=("metric_value", "max"),
        )
        .sort_values("metric_name")
        .reset_index(drop=True)
    )

    fail_count = int((row_level["severity"] == "FAIL").sum())
    warn_count = int((row_level["severity"] == "WARN").sum())
    summary["n_fail"] = fail_count
    summary["n_warn"] = warn_count
    return summary


def run_edgar_qc(
    *,
    fundamentals_pit_path: str | Path | None = None,
    ticker_cik_map_path: str | Path | None = None,
    submissions_raw_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    run_id: str = "edgar_qc_v2",
) -> EdgarQCResult:
    logger = get_logger("data.edgar.edgar_qc")
    edgar_base = data_dir() / "edgar"

    pit_source = (
        Path(fundamentals_pit_path).expanduser().resolve()
        if fundamentals_pit_path
        else (edgar_base / "fundamentals_pit.parquet")
    )
    map_source = (
        Path(ticker_cik_map_path).expanduser().resolve()
        if ticker_cik_map_path
        else (edgar_base / "ticker_cik_map.parquet")
    )
    submissions_source = (
        Path(submissions_raw_path).expanduser().resolve()
        if submissions_raw_path
        else (edgar_base / "submissions_raw.parquet")
    )

    pit = read_parquet(pit_source)
    ticker_cik_map = read_parquet(map_source)
    submissions_raw = read_parquet(submissions_source) if submissions_source.exists() else pd.DataFrame()

    issues: list[dict[str, Any]] = []

    schema_result = validate_schema(pit, FUNDAMENTALS_SCHEMA)
    if not schema_result.ok:
        for issue in schema_result.issues:
            _record_issue(
                issues,
                check_name="schema",
                severity="FAIL",
                message=issue.message,
                observed_value=issue.code,
            )

    if pit.empty:
        _record_issue(
            issues,
            check_name="coverage",
            severity="FAIL",
            message="fundamentals_pit is empty.",
        )

    pit = pit.copy()
    pit["instrument_id"] = pit["instrument_id"].astype(str)
    pit["ticker"] = pit["ticker"].astype(str)
    pit["cik"] = pit["cik"].astype(str)
    pit["metric_name"] = pit["metric_name"].astype(str)
    pit["source_type"] = pit["source_type"].astype(str)
    pit["visibility_rule"] = pit["visibility_rule"].astype(str)
    pit["accession_number"] = pit["accession_number"].map(_normalize_string)
    pit["metric_value"] = pd.to_numeric(pit["metric_value"], errors="coerce")
    pit["asof_date"] = pd.to_datetime(pit["asof_date"], utc=True, errors="coerce")
    pit["acceptance_ts"] = pd.to_datetime(pit["acceptance_ts"], utc=True, errors="coerce")
    pit["filing_date"] = pd.to_datetime(pit["filing_date"], errors="coerce").dt.normalize()
    pit["fact_end_date"] = pd.to_datetime(pit["fact_end_date"], errors="coerce").dt.normalize()

    critical_cols = [
        "instrument_id",
        "ticker",
        "cik",
        "asof_date",
        "acceptance_ts",
        "filing_date",
        "metric_name",
        "metric_value",
        "metric_unit",
        "source_type",
        "visibility_rule",
    ]
    for col in critical_cols:
        n_null = int(pit[col].isna().sum())
        if n_null > 0:
            _record_issue(
                issues,
                check_name="critical_nulls",
                severity="FAIL",
                message=f"Column '{col}' has null values.",
                observed_value=n_null,
                threshold=0,
            )

    lookahead = pit["acceptance_ts"] > pit["asof_date"]
    for row in pit.loc[
        lookahead,
        ["instrument_id", "ticker", "cik", "metric_name", "asof_date", "acceptance_ts"],
    ].head(300).itertuples(index=False):
        _record_issue(
            issues,
            check_name="pit_acceptance",
            severity="FAIL",
            message="acceptance_ts is after asof_date (look-ahead leakage).",
            asof_date=row.asof_date,
            instrument_id=row.instrument_id,
            ticker=row.ticker,
            cik=row.cik,
            metric_name=row.metric_name,
            observed_value=f"acceptance_ts={row.acceptance_ts}",
            threshold="<= asof_date",
        )

    invalid_metric = ~pit["metric_name"].isin(ALLOWED_METRICS)
    if invalid_metric.any():
        bad_metrics = sorted(pit.loc[invalid_metric, "metric_name"].unique().tolist())
        _record_issue(
            issues,
            check_name="metric_catalog",
            severity="FAIL",
            message="Found metrics outside allowed canonical catalog.",
            observed_value=bad_metrics,
            threshold=list(ALLOWED_METRICS),
        )

    non_numeric = pit["metric_value"].isna()
    if non_numeric.any():
        _record_issue(
            issues,
            check_name="metric_numeric",
            severity="FAIL",
            message="Found non-numeric metric_value rows.",
            observed_value=int(non_numeric.sum()),
            threshold=0,
        )

    limits = {
        "revenue": (0.0, 1_000_000_000_000.0),
        "net_income": (-500_000_000_000.0, 500_000_000_000.0),
        "total_assets": (0.0, 10_000_000_000_000.0),
        "shares_outstanding": (0.0, 3_000_000_000_000.0),
    }
    for metric, (lo, hi) in limits.items():
        mask = (pit["metric_name"] == metric) & (
            (pit["metric_value"] < lo) | (pit["metric_value"] > hi)
        )
        for row in pit.loc[
            mask, ["instrument_id", "ticker", "cik", "metric_name", "asof_date", "metric_value"]
        ].head(100).itertuples(index=False):
            _record_issue(
                issues,
                check_name="metric_reasonable_range",
                severity="FAIL",
                message="metric_value outside reasonable range for metric.",
                asof_date=row.asof_date,
                instrument_id=row.instrument_id,
                ticker=row.ticker,
                cik=row.cik,
                metric_name=row.metric_name,
                observed_value=row.metric_value,
                threshold=f"[{lo}, {hi}]",
            )

    dup_key = pit.duplicated(["instrument_id", "asof_date", "metric_name"], keep=False)
    for row in pit.loc[
        dup_key, ["instrument_id", "ticker", "cik", "metric_name", "asof_date"]
    ].head(200).itertuples(index=False):
        _record_issue(
            issues,
            check_name="duplicate_pit_key",
            severity="FAIL",
            message="Duplicate (instrument_id, asof_date, metric_name) found.",
            asof_date=row.asof_date,
            instrument_id=row.instrument_id,
            ticker=row.ticker,
            cik=row.cik,
            metric_name=row.metric_name,
        )

    non_synthetic = ~pit["source_type"].str.contains("synthetic", case=False, na=False)
    missing_lineage = non_synthetic & (
        (pit["accession_number"] == "")
        | pit["source_type"].str.strip().eq("")
        | pit["visibility_rule"].str.strip().eq("")
    )
    for row in pit.loc[
        missing_lineage,
        ["instrument_id", "ticker", "cik", "metric_name", "asof_date", "source_type", "accession_number"],
    ].head(200).itertuples(index=False):
        _record_issue(
            issues,
            check_name="lineage_fields",
            severity="FAIL",
            message="Missing accession_number/source_type/visibility_rule for non-synthetic row.",
            asof_date=row.asof_date,
            instrument_id=row.instrument_id,
            ticker=row.ticker,
            cik=row.cik,
            metric_name=row.metric_name,
            observed_value=f"source_type={row.source_type}, accession={row.accession_number}",
            threshold="non-empty lineage fields",
        )

    mapping = ticker_cik_map.copy()
    required_map = {"instrument_id", "ticker", "cik", "start_date", "end_date"}
    missing_map = sorted(required_map - set(mapping.columns))
    if missing_map:
        _record_issue(
            issues,
            check_name="ticker_cik_schema",
            severity="FAIL",
            message=f"ticker_cik_map missing columns: {missing_map}",
        )
    else:
        mapping["instrument_id"] = mapping["instrument_id"].astype(str)
        mapping["ticker"] = mapping["ticker"].astype(str)
        mapping["cik"] = mapping["cik"].astype(str)
        mapping["start_date"] = _normalize_date(mapping["start_date"], column="start_date")
        mapping["end_date"] = pd.to_datetime(mapping["end_date"], errors="coerce").dt.normalize()
        pit_check = pit.reset_index(drop=True).copy()
        pit_check["__row_id"] = pit_check.index
        pit_check["asof_session_date"] = (
            pit_check["asof_date"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
        )
        merged = pit_check.merge(
            mapping[["instrument_id", "ticker", "cik", "start_date", "end_date"]],
            on=["instrument_id", "ticker", "cik"],
            how="left",
        )
        merged["valid_map"] = (
            merged["start_date"].notna()
            & (merged["asof_session_date"] >= merged["start_date"])
            & (merged["end_date"].isna() | (merged["asof_session_date"] <= merged["end_date"]))
        )
        valid_by_row = merged.groupby("__row_id", as_index=True)["valid_map"].any()
        invalid_row_ids = valid_by_row.index[~valid_by_row]
        invalid_rows = pit_check.loc[pit_check["__row_id"].isin(invalid_row_ids)]
        for row in invalid_rows.head(300).itertuples(index=False):
            _record_issue(
                issues,
                check_name="cik_interval_consistency",
                severity="FAIL",
                message="instrument_id/ticker/cik not valid for asof_date under ticker_cik_map intervals.",
                asof_date=row.asof_date,
                instrument_id=row.instrument_id,
                ticker=row.ticker,
                cik=row.cik,
                metric_name=row.metric_name,
            )

    if not submissions_raw.empty and {"cik", "accession_number"}.issubset(submissions_raw.columns):
        lookup = submissions_raw[["cik", "accession_number"]].copy()
        lookup["cik"] = lookup["cik"].astype(str)
        lookup["accession_number"] = lookup["accession_number"].map(_normalize_string)
        lookup = lookup[(lookup["accession_number"] != "")].drop_duplicates()
        check_rows = pit[non_synthetic & (pit["accession_number"] != "")].copy()
        merged = check_rows.merge(
            lookup,
            on=["cik", "accession_number"],
            how="left",
            indicator=True,
        )
        missing_subm = merged["_merge"] != "both"
        for row in merged.loc[
            missing_subm,
            ["instrument_id", "ticker", "cik", "metric_name", "asof_date", "accession_number"],
        ].head(200).itertuples(index=False):
            _record_issue(
                issues,
                check_name="submission_lineage",
                severity="WARN",
                message="No matching submissions_raw row for accession_number lineage.",
                asof_date=row.asof_date,
                instrument_id=row.instrument_id,
                ticker=row.ticker,
                cik=row.cik,
                metric_name=row.metric_name,
                observed_value=row.accession_number,
            )

    source_type_counts = pit["source_type"].value_counts(dropna=False).to_dict()

    missing_metric_coverage = sorted(set(ALLOWED_METRICS) - set(pit["metric_name"].unique().tolist()))
    if missing_metric_coverage:
        _record_issue(
            issues,
            check_name="metric_coverage",
            severity="WARN",
            message="Some canonical metrics are missing from fundamentals_pit.",
            observed_value=missing_metric_coverage,
            threshold=list(ALLOWED_METRICS),
        )

    if not issues:
        _record_issue(
            issues,
            check_name="qc_summary",
            severity="PASS",
            message="No WARN/FAIL issues detected.",
        )

    row_level = pd.DataFrame(issues)
    if row_level.empty:
        row_level = pd.DataFrame(
            [
                {
                    "asof_date": pd.NaT,
                    "instrument_id": "__NONE__",
                    "ticker": "__NONE__",
                    "cik": "__NONE__",
                    "metric_name": "__NONE__",
                    "check_name": "qc_summary",
                    "severity": "PASS",
                    "observed_value": "",
                    "threshold": "",
                    "message": "No issues.",
                }
            ]
        )

    failures = row_level[row_level["severity"].isin(["FAIL", "WARN"])].copy()
    if failures.empty:
        failures = row_level.head(1).copy()

    metrics = _build_metrics_summary(pit, row_level)
    gate_status = _max_severity(row_level["severity"])
    counts = Counter(row_level["severity"].tolist())
    n_fail = int(counts.get("FAIL", 0))
    n_warn = int(counts.get("WARN", 0))

    qc_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else data_dir() / "edgar" / "qc" / run_id
    )
    qc_root.mkdir(parents=True, exist_ok=True)

    row_level_path = write_parquet(
        row_level,
        qc_root / "edgar_qc_row_level.parquet",
        schema_name="edgar_qc_row_level_v2",
        run_id=run_id,
    )
    failures_path = write_parquet(
        failures,
        qc_root / "edgar_qc_failures.parquet",
        schema_name="edgar_qc_failures_v2",
        run_id=run_id,
    )
    metrics_path = write_parquet(
        metrics,
        qc_root / "edgar_qc_metrics.parquet",
        schema_name="edgar_qc_metrics_v2",
        run_id=run_id,
    )

    summary_payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "gate_status": gate_status,
        "n_rows": int(len(pit)),
        "n_instruments": int(pit["instrument_id"].nunique()) if not pit.empty else 0,
        "n_sessions": int(pit["asof_date"].nunique()) if not pit.empty else 0,
        "n_metrics": int(pit["metric_name"].nunique()) if not pit.empty else 0,
        "n_fail": n_fail,
        "n_warn": n_warn,
        "source_type_counts": source_type_counts,
        "min_asof_date": str(pit["asof_date"].min()) if not pit.empty else "",
        "max_asof_date": str(pit["asof_date"].max()) if not pit.empty else "",
        "input_paths": {
            "fundamentals_pit": str(pit_source),
            "ticker_cik_map": str(map_source),
            "submissions_raw": str(submissions_source) if submissions_source.exists() else "",
        },
    }
    summary_path = qc_root / "edgar_qc_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    manifest_payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "gate_status": gate_status,
        "summary_path": str(summary_path),
        "row_level_path": str(row_level_path),
        "failures_path": str(failures_path),
        "metrics_path": str(metrics_path),
    }
    manifest_path = qc_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "edgar_qc_completed",
        run_id=run_id,
        gate_status=gate_status,
        n_fail=n_fail,
        n_warn=n_warn,
        output_dir=str(qc_root),
    )

    return EdgarQCResult(
        gate_status=gate_status,
        n_fail=n_fail,
        n_warn=n_warn,
        summary_path=summary_path,
        row_level_path=row_level_path,
        failures_path=failures_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EDGAR fundamentals PIT QC v2.")
    parser.add_argument("--fundamentals-pit-path", type=str, default=None)
    parser.add_argument("--ticker-cik-map-path", type=str, default=None)
    parser.add_argument("--submissions-raw-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default="edgar_qc_v2")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_edgar_qc(
        fundamentals_pit_path=args.fundamentals_pit_path,
        ticker_cik_map_path=args.ticker_cik_map_path,
        submissions_raw_path=args.submissions_raw_path,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print("EDGAR QC completed:")
    print(f"- gate_status: {result.gate_status}")
    print(f"- summary: {result.summary_path}")
    print(f"- row_level: {result.row_level_path}")
    print(f"- failures: {result.failures_path}")
    print(f"- metrics: {result.metrics_path}")
    print(f"- manifest: {result.manifest_path}")


if __name__ == "__main__":
    main()
