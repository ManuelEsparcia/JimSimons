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

# Allow direct script execution: `python simons_smallcap_swing/data/universe/universe_qc.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir, reference_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, validate_schema

SEVERITY_ORDER: dict[str, int] = {"PASS": 0, "WARN": 1, "FAIL": 2}

UNIVERSE_HISTORY_SCHEMA = DataSchema(
    name="universe_history_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("is_eligible", "bool", nullable=False),
        ColumnSpec("exchange", "string", nullable=False),
        ColumnSpec("asset_type", "string", nullable=False),
        ColumnSpec("currency", "string", nullable=False),
        ColumnSpec("sector", "string", nullable=True),
        ColumnSpec("industry", "string", nullable=True),
        ColumnSpec("run_id", "string", nullable=False),
        ColumnSpec("config_hash", "string", nullable=False),
        ColumnSpec("built_ts_utc", "string", nullable=False),
    ),
    primary_key=("date", "instrument_id"),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class UniverseQCResult:
    gate_status: str
    n_fail: int
    n_warn: int
    summary_path: Path
    daily_path: Path
    failures_path: Path
    manifest_path: Path


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains invalid dates.")
    return parsed.dt.normalize()


def _max_severity(severities: pd.Series) -> str:
    if severities.empty:
        return "PASS"
    idx = int(severities.map(SEVERITY_ORDER).max())
    for key, value in SEVERITY_ORDER.items():
        if value == idx:
            return key
    return "PASS"


def _record_issue(
    issues: list[dict[str, Any]],
    *,
    check_type: str,
    severity: str,
    flag_reason: str,
    evidence_summary: str,
    date: object = None,
    instrument_id: object = None,
    ticker: object = None,
) -> None:
    issues.append(
        {
            "date": pd.Timestamp(date).normalize() if date is not None else pd.NaT,
            "instrument_id": instrument_id,
            "ticker": ticker,
            "check_type": check_type,
            "severity": severity,
            "flag_reason": flag_reason,
            "evidence_summary": evidence_summary,
        }
    )


def _compute_daily_metrics(universe_history: pd.DataFrame) -> pd.DataFrame:
    ordered = universe_history.sort_values(["date", "instrument_id"]).reset_index(drop=True)
    daily = (
        ordered.groupby("date", as_index=False)
        .agg(
            n_rows=("instrument_id", "size"),
            n_instruments=("instrument_id", "nunique"),
            n_eligible=("is_eligible", lambda series: int(series.astype(bool).sum())),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["n_ineligible"] = daily["n_rows"] - daily["n_eligible"]

    turnover_values: list[float] = []
    prev_members: set[str] | None = None
    for date in daily["date"]:
        members = set(
            ordered.loc[(ordered["date"] == date) & (ordered["is_eligible"]), "instrument_id"]
            .astype(str)
            .tolist()
        )
        if prev_members is None:
            turnover_values.append(0.0)
        else:
            union = prev_members | members
            if not union:
                turnover_values.append(0.0)
            else:
                turnover_values.append(float(len(prev_members ^ members) / len(union)))
        prev_members = members
    daily["turnover"] = turnover_values
    return daily


def run_universe_qc(
    *,
    universe_history_path: str | Path | None = None,
    universe_current_path: str | Path | None = None,
    ticker_history_map_path: str | Path | None = None,
    trading_calendar_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    run_id: str = "universe_qc_mvp_v1",
) -> UniverseQCResult:
    logger = get_logger("data.universe.universe_qc")
    base_data = data_dir() / "universe"
    base_ref = reference_dir()

    history_source = (
        Path(universe_history_path).expanduser().resolve()
        if universe_history_path
        else (base_data / "universe_history.parquet")
    )
    current_source = (
        Path(universe_current_path).expanduser().resolve()
        if universe_current_path
        else (base_data / "universe_current.parquet")
    )
    ticker_map_source = (
        Path(ticker_history_map_path).expanduser().resolve()
        if ticker_history_map_path
        else (base_ref / "ticker_history_map.parquet")
    )
    calendar_source = (
        Path(trading_calendar_path).expanduser().resolve()
        if trading_calendar_path
        else (base_ref / "trading_calendar.parquet")
    )

    history = read_parquet(history_source)
    current = read_parquet(current_source)
    ticker_map = read_parquet(ticker_map_source)
    calendar = read_parquet(calendar_source)

    history["date"] = _normalize_date(history["date"], column="date")
    current["date"] = _normalize_date(current["date"], column="date")
    ticker_map["start_date"] = _normalize_date(ticker_map["start_date"], column="start_date")
    ticker_map["end_date"] = pd.to_datetime(ticker_map["end_date"], errors="coerce").dt.normalize()
    sessions = _normalize_date(
        calendar.loc[calendar["is_session"].astype(bool), "date"],
        column="date",
    )

    issues: list[dict[str, Any]] = []

    # 1) Schema and structural checks.
    for dataset_name, frame in (("history", history), ("current", current)):
        result = validate_schema(frame, UNIVERSE_HISTORY_SCHEMA)
        if not result.ok:
            for issue in result.issues:
                _record_issue(
                    issues,
                    check_type=f"schema_{dataset_name}",
                    severity="FAIL",
                    flag_reason=issue.code,
                    evidence_summary=issue.message,
                )

    if history.empty:
        _record_issue(
            issues,
            check_type="coverage",
            severity="FAIL",
            flag_reason="empty_history",
            evidence_summary="Universe history is empty.",
        )
    if current.empty:
        _record_issue(
            issues,
            check_type="coverage",
            severity="FAIL",
            flag_reason="empty_current",
            evidence_summary="Universe current snapshot is empty.",
        )

    duplicated_pk = history.duplicated(["date", "instrument_id"], keep=False)
    if duplicated_pk.any():
        duplicated_rows = history.loc[duplicated_pk, ["date", "instrument_id", "ticker"]].head(100)
        for row in duplicated_rows.itertuples(index=False):
            _record_issue(
                issues,
                check_type="pk_uniqueness",
                severity="FAIL",
                flag_reason="duplicate_date_instrument",
                evidence_summary="Duplicate (date, instrument_id) in universe_history.",
                date=row.date,
                instrument_id=row.instrument_id,
                ticker=row.ticker,
            )

    critical_columns = ["date", "instrument_id", "ticker", "is_eligible", "exchange", "asset_type"]
    for column in critical_columns:
        n_null = int(history[column].isna().sum())
        if n_null > 0:
            _record_issue(
                issues,
                check_type="non_null_critical",
                severity="FAIL",
                flag_reason=f"null_{column}",
                evidence_summary=f"Column '{column}' has {n_null} null rows.",
            )

    invalid_is_eligible = ~history["is_eligible"].isin([0, 1, True, False])
    if invalid_is_eligible.any():
        _record_issue(
            issues,
            check_type="eligibility_domain",
            severity="FAIL",
            flag_reason="invalid_is_eligible_domain",
            evidence_summary=f"{int(invalid_is_eligible.sum())} rows have invalid is_eligible values.",
        )
    history["is_eligible"] = history["is_eligible"].astype(bool)
    current["is_eligible"] = current["is_eligible"].astype(bool)

    # 2) Calendar consistency.
    valid_sessions = set(sessions.tolist())
    invalid_dates = history.loc[~history["date"].isin(valid_sessions), "date"].drop_duplicates().head(50)
    for bad_date in invalid_dates.tolist():
        _record_issue(
            issues,
            check_type="calendar_membership",
            severity="FAIL",
            flag_reason="date_not_in_trading_calendar",
            evidence_summary="Universe date is outside trading calendar sessions.",
            date=bad_date,
        )

    # 3) PIT ticker interval consistency.
    invalid_intervals = ticker_map["end_date"].notna() & (ticker_map["end_date"] < ticker_map["start_date"])
    if invalid_intervals.any():
        _record_issue(
            issues,
            check_type="ticker_map_interval",
            severity="FAIL",
            flag_reason="invalid_ticker_interval",
            evidence_summary=f"{int(invalid_intervals.sum())} ticker_history_map rows have end_date < start_date.",
        )

    history_for_check = history.reset_index(drop=True).copy()
    history_for_check["__row_id"] = history_for_check.index
    merged = history_for_check.merge(
        ticker_map[["instrument_id", "ticker", "start_date", "end_date"]],
        on=["instrument_id", "ticker"],
        how="left",
    )
    merged["is_valid_interval"] = (
        merged["start_date"].notna()
        & (merged["date"] >= merged["start_date"])
        & (merged["end_date"].isna() | (merged["date"] <= merged["end_date"]))
    )
    valid_by_row = merged.groupby("__row_id", as_index=True)["is_valid_interval"].any()
    invalid_row_ids = valid_by_row.index[~valid_by_row]
    if len(invalid_row_ids) > 0:
        invalid_rows = history_for_check.loc[
            history_for_check["__row_id"].isin(invalid_row_ids),
            ["date", "instrument_id", "ticker"],
        ].head(100)
        for row in invalid_rows.itertuples(index=False):
            _record_issue(
                issues,
                check_type="ticker_pit_consistency",
                severity="FAIL",
                flag_reason="ticker_not_valid_for_date",
                evidence_summary="Ticker is not valid for instrument_id on this date per ticker_history_map.",
                date=row.date,
                instrument_id=row.instrument_id,
                ticker=row.ticker,
            )

    # 4) universe_current vs last date in history.
    if not history.empty and not current.empty:
        last_date = history["date"].max()
        current_dates = current["date"].dropna().unique()
        if len(current_dates) != 1 or pd.Timestamp(current_dates[0]).normalize() != last_date:
            _record_issue(
                issues,
                check_type="current_snapshot",
                severity="FAIL",
                flag_reason="current_not_last_date",
                evidence_summary=(
                    "universe_current must contain only the last session from universe_history."
                ),
            )

        expected_current = history[(history["date"] == last_date) & (history["is_eligible"])][
            ["instrument_id", "ticker"]
        ]
        observed_current = current[["instrument_id", "ticker"]]
        expected_keys = set(expected_current.itertuples(index=False, name=None))
        observed_keys = set(observed_current.itertuples(index=False, name=None))
        if expected_keys != observed_keys:
            _record_issue(
                issues,
                check_type="current_snapshot",
                severity="FAIL",
                flag_reason="current_mismatch",
                evidence_summary=(
                    "universe_current constituents do not match eligible rows on last history date."
                ),
            )

    # 5) Coverage and stability warnings.
    if history["instrument_id"].nunique() == 0:
        _record_issue(
            issues,
            check_type="coverage",
            severity="FAIL",
            flag_reason="no_instruments",
            evidence_summary="Universe history has zero distinct instruments.",
        )

    daily = _compute_daily_metrics(history) if not history.empty else pd.DataFrame()
    if not daily.empty:
        high_turnover_days = int((daily["turnover"] > 0.35).sum())
        if high_turnover_days > 0:
            _record_issue(
                issues,
                check_type="temporal_stability",
                severity="WARN",
                flag_reason="high_turnover_days",
                evidence_summary=f"{high_turnover_days} days with turnover > 0.35.",
            )

    if not issues:
        _record_issue(
            issues,
            check_type="qc_summary",
            severity="PASS",
            flag_reason="no_issues",
            evidence_summary="No WARN/FAIL issues detected.",
        )

    failures = pd.DataFrame(issues)
    gate_status = _max_severity(failures["severity"])
    counts = Counter(failures["severity"].tolist())
    n_fail = int(counts.get("FAIL", 0))
    n_warn = int(counts.get("WARN", 0))

    if daily.empty:
        daily = pd.DataFrame(
            [
                {
                    "date": pd.NaT,
                    "n_rows": 0,
                    "n_instruments": 0,
                    "n_eligible": 0,
                    "n_ineligible": 0,
                    "turnover": 0.0,
                }
            ]
        )

    if output_dir:
        qc_root = Path(output_dir).expanduser().resolve()
    else:
        qc_root = data_dir() / "universe" / "qc" / run_id
    qc_root.mkdir(parents=True, exist_ok=True)

    daily_path = write_parquet(
        daily,
        qc_root / "universe_qc_daily.parquet",
        schema_name="universe_qc_daily_mvp",
        run_id=run_id,
    )
    failures_path = write_parquet(
        failures,
        qc_root / "universe_qc_failures.parquet",
        schema_name="universe_qc_failures_mvp",
        run_id=run_id,
    )

    top_failure_types = (
        failures.loc[failures["severity"].isin(["WARN", "FAIL"]), "check_type"]
        .value_counts()
        .head(5)
        .to_dict()
    )

    summary_payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "gate_status": gate_status,
        "n_rows_processed": int(len(history)),
        "n_distinct_instruments": int(history["instrument_id"].nunique()) if not history.empty else 0,
        "n_fail": n_fail,
        "n_warn": n_warn,
        "pct_days_high_turnover": float((daily["turnover"] > 0.35).mean()),
        "pct_days_extreme_turnover": float((daily["turnover"] > 0.60).mean()),
        "top_failure_types": top_failure_types,
        "input_paths": {
            "universe_history": str(history_source),
            "universe_current": str(current_source),
            "ticker_history_map": str(ticker_map_source),
            "trading_calendar": str(calendar_source),
        },
    }
    summary_path = qc_root / "universe_qc_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    manifest_payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "gate_status": gate_status,
        "summary_path": str(summary_path),
        "daily_path": str(daily_path),
        "failures_path": str(failures_path),
    }
    manifest_path = qc_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "universe_qc_completed",
        run_id=run_id,
        gate_status=gate_status,
        n_fail=n_fail,
        n_warn=n_warn,
        output_dir=str(qc_root),
    )

    return UniverseQCResult(
        gate_status=gate_status,
        n_fail=n_fail,
        n_warn=n_warn,
        summary_path=summary_path,
        daily_path=daily_path,
        failures_path=failures_path,
        manifest_path=manifest_path,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MVP QC checks for historical PIT universe.")
    parser.add_argument("--universe-history-path", type=str, default=None)
    parser.add_argument("--universe-current-path", type=str, default=None)
    parser.add_argument("--ticker-history-map-path", type=str, default=None)
    parser.add_argument("--trading-calendar-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default="universe_qc_mvp_v1")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_universe_qc(
        universe_history_path=args.universe_history_path,
        universe_current_path=args.universe_current_path,
        ticker_history_map_path=args.ticker_history_map_path,
        trading_calendar_path=args.trading_calendar_path,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print("Universe QC completed:")
    print(f"- gate_status: {result.gate_status}")
    print(f"- summary: {result.summary_path}")
    print(f"- daily: {result.daily_path}")
    print(f"- failures: {result.failures_path}")
    print(f"- manifest: {result.manifest_path}")


if __name__ == "__main__":
    main()
