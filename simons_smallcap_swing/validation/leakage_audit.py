from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "leakage_audit_mvp_v1"
SEVERITY_PASS = "PASS"
SEVERITY_WARN = "WARN"
SEVERITY_FAIL = "FAIL"
SEVERITY_RANK = {SEVERITY_PASS: 0, SEVERITY_WARN: 1, SEVERITY_FAIL: 2}
ALLOWED_DATASET_SPLIT_ROLES = {
    "train",
    "valid",
    "test",
    "dropped_by_purge",
    "dropped_by_embargo",
}
ALLOWED_CV_SPLIT_ROLES = {
    "train",
    "valid",
    "dropped_by_purge",
    "dropped_by_embargo",
}
LABEL_NAME_PATTERN = re.compile(r"^(fwd_ret|fwd_dir_up)_(\d+)d$")
SUSPICIOUS_FEATURE_PATTERN = re.compile(
    r"(future|forward|target|label|leak|tplus|next|exit|entry)", re.IGNORECASE
)
MIN_CORR_OBS = 30
PERFECT_CORR_THRESHOLD = 0.999

FINDINGS_SCHEMA = DataSchema(
    name="leakage_audit_findings_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("check_name", "string", nullable=False),
        ColumnSpec("severity", "string", nullable=False),
        ColumnSpec("object_type", "string", nullable=False),
        ColumnSpec("date", "datetime64", nullable=True),
        ColumnSpec("instrument_id", "string", nullable=True),
        ColumnSpec("label_name", "string", nullable=True),
        ColumnSpec("feature_name", "string", nullable=True),
        ColumnSpec("message", "string", nullable=False),
        ColumnSpec("observed_value", "string", nullable=True),
        ColumnSpec("expected_rule", "string", nullable=True),
    ),
    primary_key=(),
    allow_extra_columns=True,
)

METRICS_SCHEMA = DataSchema(
    name="leakage_audit_metrics_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("check_name", "string", nullable=False),
        ColumnSpec("severity", "string", nullable=False),
        ColumnSpec("object_type", "string", nullable=False),
        ColumnSpec("n_rows_evaluated", "int64", nullable=False),
        ColumnSpec("n_violations", "int64", nullable=False),
        ColumnSpec("violation_rate", "float64", nullable=True),
        ColumnSpec("message", "string", nullable=False),
    ),
    primary_key=("check_name",),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class LeakageAuditResult:
    findings_path: Path
    summary_path: Path
    metrics_path: Path
    overall_status: str
    n_fail: int
    n_warn: int
    n_pass: int
    config_hash: str


def _normalize_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def _config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _to_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return str(value)


def _resolve_optional_path(
    provided: str | Path | None,
    *,
    default_path: Path,
) -> Path | None:
    if provided is not None:
        candidate = Path(provided).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Optional input path does not exist: {candidate}")
        return candidate
    if default_path.exists():
        return default_path
    return None


def _first_violation_row(frame: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    if frame.empty or mask.empty or not bool(mask.any()):
        return {}
    first = frame.loc[mask].head(1)
    if first.empty:
        return {}
    row = first.iloc[0]
    return {
        "date": row.get("date", pd.NaT),
        "instrument_id": row.get("instrument_id"),
        "label_name": row.get("label_name"),
        "feature_name": row.get("feature_name"),
    }


def _emit_rule(
    *,
    findings: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    check_name: str,
    object_type: str,
    n_rows_evaluated: int,
    n_violations: int,
    expected_rule: str,
    pass_message: str,
    violation_message: str,
    severity_if_violation: str = SEVERITY_FAIL,
    sample: dict[str, Any] | None = None,
    observed_value: str | None = None,
) -> None:
    if n_violations > 0:
        severity = severity_if_violation
        message = f"{violation_message} (n_violations={n_violations})"
    else:
        severity = SEVERITY_PASS
        message = pass_message

    violation_rate: float | None
    if n_rows_evaluated > 0:
        violation_rate = float(n_violations) / float(n_rows_evaluated)
    else:
        violation_rate = None

    if observed_value is None:
        observed_value = f"evaluated={n_rows_evaluated};violations={n_violations}"

    findings.append(
        {
            "check_name": check_name,
            "severity": severity,
            "object_type": object_type,
            "date": (sample or {}).get("date", pd.NaT),
            "instrument_id": _to_string((sample or {}).get("instrument_id")),
            "label_name": _to_string((sample or {}).get("label_name")),
            "feature_name": _to_string((sample or {}).get("feature_name")),
            "message": message,
            "observed_value": observed_value,
            "expected_rule": expected_rule,
        }
    )
    metrics.append(
        {
            "check_name": check_name,
            "severity": severity,
            "object_type": object_type,
            "n_rows_evaluated": int(n_rows_evaluated),
            "n_violations": int(n_violations),
            "violation_rate": violation_rate,
            "message": message,
        }
    )


def _emit_missing_columns(
    *,
    findings: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    check_name: str,
    object_type: str,
    missing_columns: list[str],
    expected_columns: Iterable[str],
) -> None:
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name=check_name,
        object_type=object_type,
        n_rows_evaluated=0,
        n_violations=len(missing_columns),
        expected_rule=f"Required columns present: {sorted(expected_columns)}",
        pass_message="All required columns are present.",
        violation_message=f"Missing required columns: {sorted(missing_columns)}",
        severity_if_violation=SEVERITY_FAIL,
        observed_value=f"missing_columns={sorted(missing_columns)}",
    )


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals)
    merged: list[tuple[int, int]] = []
    cur_start, cur_end = sorted_intervals[0]
    for start, end in sorted_intervals[1:]:
        if start <= cur_end + 1:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def _interval_overlaps_any(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    for int_start, int_end in intervals:
        if int_start > end:
            return False
        if int_end < start:
            continue
        return True
    return False


def _label_horizon_from_name(value: object) -> int | None:
    match = LABEL_NAME_PATTERN.match(str(value).strip())
    if not match:
        return None
    try:
        return int(match.group(2))
    except ValueError:
        return None


def _check_labels(
    *,
    labels: pd.DataFrame,
    session_pos_map: dict[pd.Timestamp, int],
    max_session_pos: int,
    findings: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
) -> pd.DataFrame:
    required_cols = ["date", "instrument_id", "horizon_days", "entry_date", "exit_date", "label_name"]
    missing = [col for col in required_cols if col not in labels.columns]
    if missing:
        _emit_missing_columns(
            findings=findings,
            metrics=metrics,
            check_name="labels_required_columns",
            object_type="label",
            missing_columns=missing,
            expected_columns=required_cols,
        )
        return labels.copy()

    frame = labels.copy()
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["label_name"] = frame["label_name"].astype(str)
    frame["date"] = _normalize_date(frame["date"])
    frame["entry_date"] = _normalize_date(frame["entry_date"])
    frame["exit_date"] = _normalize_date(frame["exit_date"])
    frame["horizon_days"] = pd.to_numeric(frame["horizon_days"], errors="coerce")

    n_rows = int(len(frame))
    invalid_ts_mask = frame[["date", "entry_date", "exit_date"]].isna().any(axis=1)
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="labels_dates_parseable",
        object_type="label",
        n_rows_evaluated=n_rows,
        n_violations=int(invalid_ts_mask.sum()),
        expected_rule="date, entry_date and exit_date must be valid trading dates.",
        pass_message="All label timestamps are parseable.",
        violation_message="Invalid timestamps detected in labels.",
        sample=_first_violation_row(frame, invalid_ts_mask),
    )

    dup_mask = frame.duplicated(["date", "instrument_id", "horizon_days", "label_name"], keep=False)
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="labels_logical_pk_unique",
        object_type="label",
        n_rows_evaluated=n_rows,
        n_violations=int(dup_mask.sum()),
        expected_rule="(date, instrument_id, horizon_days, label_name) must be unique in labels_forward.",
        pass_message="Labels logical PK has no duplicates.",
        violation_message="Duplicate labels logical PK rows detected.",
        sample=_first_violation_row(frame, dup_mask),
    )

    entry_leak_mask = frame["entry_date"].notna() & frame["date"].notna() & (frame["entry_date"] <= frame["date"])
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="labels_entry_after_decision",
        object_type="label",
        n_rows_evaluated=n_rows,
        n_violations=int(entry_leak_mask.sum()),
        expected_rule="entry_date must be strictly greater than date.",
        pass_message="All labels satisfy entry_date > date.",
        violation_message="Labels with entry_date <= date found (temporal leakage).",
        sample=_first_violation_row(frame, entry_leak_mask),
    )

    exit_before_entry_mask = (
        frame["exit_date"].notna()
        & frame["entry_date"].notna()
        & (frame["exit_date"] < frame["entry_date"])
    )
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="labels_exit_after_entry",
        object_type="label",
        n_rows_evaluated=n_rows,
        n_violations=int(exit_before_entry_mask.sum()),
        expected_rule="exit_date must be >= entry_date.",
        pass_message="All labels satisfy exit_date >= entry_date.",
        violation_message="Labels with exit_date < entry_date found.",
        sample=_first_violation_row(frame, exit_before_entry_mask),
    )

    horizon_bad_mask = frame["horizon_days"].isna() | (frame["horizon_days"] <= 0)
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="labels_horizon_positive",
        object_type="label",
        n_rows_evaluated=n_rows,
        n_violations=int(horizon_bad_mask.sum()),
        expected_rule="horizon_days must be numeric and > 0.",
        pass_message="All labels have valid positive horizon_days.",
        violation_message="Invalid horizon_days detected in labels.",
        sample=_first_violation_row(frame, horizon_bad_mask),
    )

    valid_sessions = set(session_pos_map.keys())
    out_of_calendar_mask = (
        ~frame["date"].isin(valid_sessions)
        | ~frame["entry_date"].isin(valid_sessions)
        | ~frame["exit_date"].isin(valid_sessions)
    )
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="labels_dates_in_calendar",
        object_type="label",
        n_rows_evaluated=n_rows,
        n_violations=int(out_of_calendar_mask.sum()),
        expected_rule="date, entry_date and exit_date must exist in trading calendar sessions.",
        pass_message="All label dates are present in trading calendar sessions.",
        violation_message="Labels with dates outside trading calendar sessions detected.",
        sample=_first_violation_row(frame, out_of_calendar_mask),
    )

    frame["date_pos"] = frame["date"].map(session_pos_map)
    frame["entry_pos"] = frame["entry_date"].map(session_pos_map)
    frame["exit_pos"] = frame["exit_date"].map(session_pos_map)
    with_pos_mask = (
        frame["date_pos"].notna()
        & frame["entry_pos"].notna()
        & frame["exit_pos"].notna()
        & frame["horizon_days"].notna()
    )
    horizon_delta_mask = with_pos_mask & (
        (frame["exit_pos"] - frame["entry_pos"]) != frame["horizon_days"]
    )
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="labels_horizon_matches_entry_exit_delta",
        object_type="label",
        n_rows_evaluated=int(with_pos_mask.sum()),
        n_violations=int(horizon_delta_mask.sum()),
        expected_rule="exit_pos - entry_pos must equal horizon_days.",
        pass_message="Label horizon is consistent with entry/exit delta.",
        violation_message="Label horizon inconsistent with entry/exit delta.",
        sample=_first_violation_row(frame, horizon_delta_mask),
    )

    decision_delta_mask = with_pos_mask & (
        (frame["exit_pos"] - frame["date_pos"]) != (frame["horizon_days"] + 1.0)
    )
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="labels_horizon_matches_decision_exit_delta",
        object_type="label",
        n_rows_evaluated=int(with_pos_mask.sum()),
        n_violations=int(decision_delta_mask.sum()),
        expected_rule="exit_pos - date_pos must equal horizon_days + 1 (decision lag 1 convention).",
        pass_message="Decision-to-exit delta is consistent with configured label semantics.",
        violation_message="Decision-to-exit delta inconsistent with label horizon semantics.",
        sample=_first_violation_row(frame, decision_delta_mask),
    )

    parsed_horizon = frame["label_name"].map(_label_horizon_from_name)
    unrecognized_name_mask = parsed_horizon.isna()
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="labels_name_pattern",
        object_type="label",
        n_rows_evaluated=n_rows,
        n_violations=int(unrecognized_name_mask.sum()),
        expected_rule="label_name should match fwd_ret_{h}d or fwd_dir_up_{h}d naming.",
        pass_message="All labels match expected naming conventions.",
        violation_message="Unrecognized label_name pattern detected.",
        severity_if_violation=SEVERITY_WARN,
        sample=_first_violation_row(frame, unrecognized_name_mask),
    )

    parsed_numeric = pd.to_numeric(parsed_horizon, errors="coerce")
    name_horizon_mismatch_mask = (
        parsed_numeric.notna()
        & frame["horizon_days"].notna()
        & (parsed_numeric != frame["horizon_days"])
    )
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="labels_name_horizon_consistency",
        object_type="label",
        n_rows_evaluated=int(parsed_numeric.notna().sum()),
        n_violations=int(name_horizon_mismatch_mask.sum()),
        expected_rule="label_name horizon suffix must match horizon_days column.",
        pass_message="Label names are consistent with horizon_days.",
        violation_message="label_name horizon suffix mismatches horizon_days.",
        sample=_first_violation_row(frame, name_horizon_mismatch_mask),
    )

    complete_window_mask = with_pos_mask & ((frame["date_pos"] + frame["horizon_days"] + 1.0) > max_session_pos)
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="labels_complete_forward_window",
        object_type="label",
        n_rows_evaluated=int(with_pos_mask.sum()),
        n_violations=int(complete_window_mask.sum()),
        expected_rule="Forward label window must be fully inside available trading sessions.",
        pass_message="No incomplete forward windows detected in labels.",
        violation_message="Labels with incomplete forward windows detected.",
        sample=_first_violation_row(frame, complete_window_mask),
    )

    return frame


def _check_features(
    *,
    features: pd.DataFrame,
    findings: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
) -> pd.DataFrame:
    required_cols = ["date", "instrument_id", "ticker"]
    missing = [col for col in required_cols if col not in features.columns]
    if missing:
        _emit_missing_columns(
            findings=findings,
            metrics=metrics,
            check_name="features_required_columns",
            object_type="feature",
            missing_columns=missing,
            expected_columns=required_cols,
        )
        return features.copy()

    frame = features.copy()
    frame["date"] = _normalize_date(frame["date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str)
    n_rows = int(len(frame))

    invalid_date_mask = frame["date"].isna()
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="features_dates_parseable",
        object_type="feature",
        n_rows_evaluated=n_rows,
        n_violations=int(invalid_date_mask.sum()),
        expected_rule="Feature date must be parseable timestamp.",
        pass_message="All feature dates are parseable.",
        violation_message="Invalid feature dates detected.",
        sample=_first_violation_row(frame, invalid_date_mask),
    )

    dup_mask = frame.duplicated(["date", "instrument_id"], keep=False)
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="features_logical_pk_unique",
        object_type="feature",
        n_rows_evaluated=n_rows,
        n_violations=int(dup_mask.sum()),
        expected_rule="(date, instrument_id) must be unique in features_matrix.",
        pass_message="Features logical PK has no duplicates.",
        violation_message="Duplicate feature PK rows detected.",
        sample=_first_violation_row(frame, dup_mask),
    )

    metadata_cols = {"date", "instrument_id", "ticker", "run_id", "config_hash", "built_ts_utc"}
    feature_cols = [col for col in frame.columns if col not in metadata_cols]
    numeric_feature_cols = [col for col in feature_cols if is_numeric_dtype(frame[col])]
    forbidden_cols = [col for col in numeric_feature_cols if col in {"target_value", "label_value"}]
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="features_forbidden_target_columns",
        object_type="feature",
        n_rows_evaluated=len(numeric_feature_cols),
        n_violations=len(forbidden_cols),
        expected_rule="features_matrix must not include target_value/label_value columns.",
        pass_message="No forbidden target columns found in features.",
        violation_message=f"Forbidden target-like columns found in features: {forbidden_cols}",
        severity_if_violation=SEVERITY_FAIL,
        sample={"feature_name": forbidden_cols[0]} if forbidden_cols else None,
        observed_value=f"forbidden_columns={forbidden_cols}",
    )

    suspicious_cols = [col for col in numeric_feature_cols if SUSPICIOUS_FEATURE_PATTERN.search(col)]
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="features_suspicious_name_red_flags",
        object_type="feature",
        n_rows_evaluated=len(numeric_feature_cols),
        n_violations=len(suspicious_cols),
        expected_rule="Feature names should avoid forward/target/label semantic red flags.",
        pass_message="No suspicious leakage-like feature names detected.",
        violation_message=f"Suspicious feature names detected: {suspicious_cols}",
        severity_if_violation=SEVERITY_WARN,
        sample={"feature_name": suspicious_cols[0]} if suspicious_cols else None,
        observed_value=f"suspicious_columns={suspicious_cols}",
    )

    unlagged_market_cols = [
        col
        for col in numeric_feature_cols
        if col.startswith("mkt_") and not col.endswith("_lag1")
    ]
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="features_market_columns_lagged",
        object_type="feature",
        n_rows_evaluated=len([c for c in numeric_feature_cols if c.startswith("mkt_")]),
        n_violations=len(unlagged_market_cols),
        expected_rule="Market context features should be lagged (_lag1) in MVP.",
        pass_message="Market context features are properly lagged.",
        violation_message=f"Unlagged market context features detected: {unlagged_market_cols}",
        severity_if_violation=SEVERITY_WARN,
        sample={"feature_name": unlagged_market_cols[0]} if unlagged_market_cols else None,
        observed_value=f"unlagged_market_columns={unlagged_market_cols}",
    )

    return frame


def _check_dataset(
    *,
    dataset: pd.DataFrame,
    labels: pd.DataFrame,
    features: pd.DataFrame,
    findings: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
) -> pd.DataFrame:
    required_cols = [
        "date",
        "instrument_id",
        "horizon_days",
        "label_name",
        "entry_date",
        "exit_date",
        "target_value",
        "split_role",
    ]
    missing = [col for col in required_cols if col not in dataset.columns]
    if missing:
        _emit_missing_columns(
            findings=findings,
            metrics=metrics,
            check_name="dataset_required_columns",
            object_type="dataset",
            missing_columns=missing,
            expected_columns=required_cols,
        )
        return dataset.copy()

    frame = dataset.copy()
    frame["date"] = _normalize_date(frame["date"])
    frame["entry_date"] = _normalize_date(frame["entry_date"])
    frame["exit_date"] = _normalize_date(frame["exit_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["label_name"] = frame["label_name"].astype(str)
    frame["split_role"] = frame["split_role"].astype(str)
    frame["horizon_days"] = pd.to_numeric(frame["horizon_days"], errors="coerce")
    frame["target_value"] = pd.to_numeric(frame["target_value"], errors="coerce")
    n_rows = int(len(frame))

    dup_mask = frame.duplicated(["date", "instrument_id", "horizon_days", "label_name"], keep=False)
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="dataset_logical_pk_unique",
        object_type="dataset",
        n_rows_evaluated=n_rows,
        n_violations=int(dup_mask.sum()),
        expected_rule="(date, instrument_id, horizon_days, label_name) must be unique in model_dataset.",
        pass_message="Dataset logical PK has no duplicates.",
        violation_message="Duplicate dataset logical PK rows detected.",
        sample=_first_violation_row(frame, dup_mask),
    )

    target_bad_mask = frame["target_value"].isna()
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="dataset_target_numeric_non_null",
        object_type="dataset",
        n_rows_evaluated=n_rows,
        n_violations=int(target_bad_mask.sum()),
        expected_rule="target_value must be numeric and non-null.",
        pass_message="Dataset target_value is numeric and non-null.",
        violation_message="Null/non-numeric target_value detected in dataset.",
        sample=_first_violation_row(frame, target_bad_mask),
    )

    split_bad_mask = ~frame["split_role"].isin(ALLOWED_DATASET_SPLIT_ROLES)
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="dataset_split_role_allowed",
        object_type="dataset",
        n_rows_evaluated=n_rows,
        n_violations=int(split_bad_mask.sum()),
        expected_rule=f"split_role in {sorted(ALLOWED_DATASET_SPLIT_ROLES)}.",
        pass_message="All dataset split_role values are allowed.",
        violation_message="Invalid split_role values detected in dataset.",
        sample=_first_violation_row(frame, split_bad_mask),
    )

    entry_leak_mask = frame["entry_date"].notna() & frame["date"].notna() & (frame["entry_date"] <= frame["date"])
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="dataset_entry_after_decision",
        object_type="dataset",
        n_rows_evaluated=n_rows,
        n_violations=int(entry_leak_mask.sum()),
        expected_rule="entry_date must be > date for model_dataset rows.",
        pass_message="Dataset rows satisfy entry_date > date.",
        violation_message="Dataset rows with entry_date <= date detected.",
        sample=_first_violation_row(frame, entry_leak_mask),
    )

    exit_before_entry_mask = (
        frame["exit_date"].notna()
        & frame["entry_date"].notna()
        & (frame["exit_date"] < frame["entry_date"])
    )
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="dataset_exit_after_entry",
        object_type="dataset",
        n_rows_evaluated=n_rows,
        n_violations=int(exit_before_entry_mask.sum()),
        expected_rule="exit_date must be >= entry_date.",
        pass_message="Dataset rows satisfy exit_date >= entry_date.",
        violation_message="Dataset rows with exit_date < entry_date detected.",
        sample=_first_violation_row(frame, exit_before_entry_mask),
    )

    parsed_horizon = frame["label_name"].map(_label_horizon_from_name)
    parsed_numeric = pd.to_numeric(parsed_horizon, errors="coerce")
    horizon_consistency_mask = (
        parsed_numeric.notna()
        & frame["horizon_days"].notna()
        & (parsed_numeric != frame["horizon_days"])
    )
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="dataset_label_name_horizon_consistency",
        object_type="dataset",
        n_rows_evaluated=n_rows,
        n_violations=int(horizon_consistency_mask.sum()),
        expected_rule="label_name horizon suffix must match dataset horizon_days.",
        pass_message="Dataset label_name horizon suffix matches horizon_days.",
        violation_message="Dataset label_name horizon mismatch detected.",
        sample=_first_violation_row(frame, horizon_consistency_mask),
    )

    if {"date", "instrument_id", "horizon_days", "label_name", "label_value"}.issubset(labels.columns):
        labels_ref = labels.copy()
        labels_ref["date"] = _normalize_date(labels_ref["date"])
        labels_ref["instrument_id"] = labels_ref["instrument_id"].astype(str)
        labels_ref["label_name"] = labels_ref["label_name"].astype(str)
        labels_ref["horizon_days"] = pd.to_numeric(labels_ref["horizon_days"], errors="coerce")
        labels_ref["label_value"] = pd.to_numeric(labels_ref["label_value"], errors="coerce")
        labels_ref = labels_ref[
            ["date", "instrument_id", "horizon_days", "label_name", "label_value"]
        ].rename(columns={"label_value": "label_value_ref"})

        merged = frame.merge(
            labels_ref,
            on=["date", "instrument_id", "horizon_days", "label_name"],
            how="left",
            indicator=True,
        )
        missing_in_labels_mask = merged["_merge"] != "both"
        _emit_rule(
            findings=findings,
            metrics=metrics,
            check_name="dataset_rows_present_in_labels",
            object_type="dataset",
            n_rows_evaluated=n_rows,
            n_violations=int(missing_in_labels_mask.sum()),
            expected_rule="Every model_dataset PK must exist in labels_forward.",
            pass_message="All dataset rows are present in labels_forward.",
            violation_message="Dataset rows missing in labels_forward.",
            sample=_first_violation_row(merged, missing_in_labels_mask),
        )

        merged_both = merged[merged["_merge"] == "both"].copy()
        if merged_both.empty:
            mismatch_count = 0
            mismatch_mask = pd.Series([], dtype=bool)
        else:
            mismatch_mask = ~np.isclose(
                merged_both["target_value"].astype(float),
                merged_both["label_value_ref"].astype(float),
                atol=1e-12,
                rtol=0.0,
            )
            mismatch_count = int(mismatch_mask.sum())
        _emit_rule(
            findings=findings,
            metrics=metrics,
            check_name="dataset_target_matches_labels",
            object_type="dataset",
            n_rows_evaluated=int(len(merged_both)),
            n_violations=mismatch_count,
            expected_rule="target_value in model_dataset must match labels_forward label_value for same PK.",
            pass_message="Dataset target_value matches labels_forward.",
            violation_message="Dataset target_value mismatch against labels_forward.",
            sample=_first_violation_row(merged_both, mismatch_mask) if mismatch_count > 0 else None,
        )
    else:
        _emit_missing_columns(
            findings=findings,
            metrics=metrics,
            check_name="dataset_rows_present_in_labels",
            object_type="dataset",
            missing_columns=[
                col
                for col in ["date", "instrument_id", "horizon_days", "label_name", "label_value"]
                if col not in labels.columns
            ],
            expected_columns=["date", "instrument_id", "horizon_days", "label_name", "label_value"],
        )

    if {"date", "instrument_id"}.issubset(features.columns):
        features_ref = features.copy()
        features_ref["date"] = _normalize_date(features_ref["date"])
        features_ref["instrument_id"] = features_ref["instrument_id"].astype(str)
        merged_features = frame.merge(
            features_ref[["date", "instrument_id"]].drop_duplicates(),
            on=["date", "instrument_id"],
            how="left",
            indicator=True,
        )
        missing_in_features_mask = merged_features["_merge"] != "both"
        _emit_rule(
            findings=findings,
            metrics=metrics,
            check_name="dataset_rows_present_in_features",
            object_type="dataset",
            n_rows_evaluated=n_rows,
            n_violations=int(missing_in_features_mask.sum()),
            expected_rule="Every model_dataset (date, instrument_id) must exist in features_matrix.",
            pass_message="All dataset rows are present in features_matrix.",
            violation_message="Dataset rows missing in features_matrix.",
            sample=_first_violation_row(merged_features, missing_in_features_mask),
        )
    else:
        _emit_missing_columns(
            findings=findings,
            metrics=metrics,
            check_name="dataset_rows_present_in_features",
            object_type="dataset",
            missing_columns=[col for col in ["date", "instrument_id"] if col not in features.columns],
            expected_columns=["date", "instrument_id"],
        )

    metadata_cols = {
        "date",
        "instrument_id",
        "ticker",
        "horizon_days",
        "label_name",
        "split_name",
        "split_role",
        "entry_date",
        "exit_date",
        "target_value",
        "target_type",
        "run_id",
        "config_hash",
        "built_ts_utc",
    }
    numeric_features = [col for col in frame.columns if col not in metadata_cols and is_numeric_dtype(frame[col])]
    suspicious_corr_features: list[str] = []
    for col in numeric_features:
        pair = frame[[col, "target_value"]].dropna()
        if len(pair) < MIN_CORR_OBS:
            continue
        corr = pair[col].corr(pair["target_value"])
        if corr is not None and np.isfinite(corr) and abs(float(corr)) >= PERFECT_CORR_THRESHOLD:
            suspicious_corr_features.append(col)
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="dataset_feature_target_perfect_corr_red_flag",
        object_type="dataset",
        n_rows_evaluated=len(numeric_features),
        n_violations=len(suspicious_corr_features),
        expected_rule="No feature should have near-perfect corr with target in OOS-style dataset without explanation.",
        pass_message="No near-perfect feature-target correlations detected.",
        violation_message=(
            "Near-perfect feature-target correlations detected (heuristic red flag): "
            f"{suspicious_corr_features}"
        ),
        severity_if_violation=SEVERITY_WARN,
        sample={"feature_name": suspicious_corr_features[0]} if suspicious_corr_features else None,
        observed_value=f"suspicious_features={suspicious_corr_features}",
    )

    return frame


def _check_splits(
    *,
    splits: pd.DataFrame | None,
    session_pos_map: dict[pd.Timestamp, int],
    findings: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    embargo_sessions: int | None,
) -> None:
    if splits is None:
        _emit_rule(
            findings=findings,
            metrics=metrics,
            check_name="splits_file_available",
            object_type="split",
            n_rows_evaluated=0,
            n_violations=1,
            expected_rule="purged_splits.parquet should be available for holdout leakage audit.",
            pass_message="purged_splits.parquet is available.",
            violation_message="purged_splits.parquet is missing; split leakage checks are unverified.",
            severity_if_violation=SEVERITY_WARN,
            observed_value="file_missing",
        )
        return

    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="splits_file_available",
        object_type="split",
        n_rows_evaluated=1,
        n_violations=0,
        expected_rule="purged_splits.parquet should be available for holdout leakage audit.",
        pass_message="purged_splits.parquet is available.",
        violation_message="purged_splits.parquet missing.",
    )

    required = ["date", "instrument_id", "horizon_days", "label_name", "split_name", "split_role", "entry_date", "exit_date"]
    missing = [col for col in required if col not in splits.columns]
    if missing:
        _emit_missing_columns(
            findings=findings,
            metrics=metrics,
            check_name="splits_required_columns",
            object_type="split",
            missing_columns=missing,
            expected_columns=required,
        )
        return

    frame = splits.copy()
    frame["date"] = _normalize_date(frame["date"])
    frame["entry_date"] = _normalize_date(frame["entry_date"])
    frame["exit_date"] = _normalize_date(frame["exit_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["label_name"] = frame["label_name"].astype(str)
    frame["split_name"] = frame["split_name"].astype(str)
    frame["split_role"] = frame["split_role"].astype(str)
    frame["horizon_days"] = pd.to_numeric(frame["horizon_days"], errors="coerce")
    n_rows = int(len(frame))

    dup_mask = frame.duplicated(["date", "instrument_id", "horizon_days", "label_name"], keep=False)
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="splits_logical_pk_unique",
        object_type="split",
        n_rows_evaluated=n_rows,
        n_violations=int(dup_mask.sum()),
        expected_rule="(date, instrument_id, horizon_days, label_name) must be unique in purged_splits.",
        pass_message="purged_splits PK is unique.",
        violation_message="Duplicate PK rows detected in purged_splits.",
        sample=_first_violation_row(frame, dup_mask),
    )

    invalid_role_mask = ~frame["split_role"].isin(ALLOWED_DATASET_SPLIT_ROLES)
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="splits_split_role_allowed",
        object_type="split",
        n_rows_evaluated=n_rows,
        n_violations=int(invalid_role_mask.sum()),
        expected_rule=f"split_role in {sorted(ALLOWED_DATASET_SPLIT_ROLES)}.",
        pass_message="All split roles are valid in purged_splits.",
        violation_message="Invalid split roles detected in purged_splits.",
        sample=_first_violation_row(frame, invalid_role_mask),
    )

    valid_sessions = set(session_pos_map.keys())
    out_of_calendar_mask = (
        ~frame["date"].isin(valid_sessions)
        | ~frame["entry_date"].isin(valid_sessions)
        | ~frame["exit_date"].isin(valid_sessions)
    )
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="splits_dates_in_calendar",
        object_type="split",
        n_rows_evaluated=n_rows,
        n_violations=int(out_of_calendar_mask.sum()),
        expected_rule="split date/entry_date/exit_date must be in trading calendar sessions.",
        pass_message="All purged_splits timestamps are in trading calendar sessions.",
        violation_message="purged_splits contains timestamps outside trading sessions.",
        sample=_first_violation_row(frame, out_of_calendar_mask),
    )

    frame["date_pos"] = frame["date"].map(session_pos_map)
    frame["entry_pos"] = frame["entry_date"].map(session_pos_map)
    frame["exit_pos"] = frame["exit_date"].map(session_pos_map)

    overlap_violations = 0
    dropped_purge_non_overlap = 0
    train_embargo_violations = 0
    dropped_embargo_outside_window = 0
    overlap_sample: dict[str, Any] | None = None
    dropped_sample: dict[str, Any] | None = None
    embargo_sample: dict[str, Any] | None = None
    dropped_embargo_sample: dict[str, Any] | None = None

    group_cols = ["split_name", "label_name", "horizon_days"]
    for _, group in frame.groupby(group_cols, dropna=False):
        eval_rows = group[group["split_role"].isin(["valid", "test"])].copy()
        if eval_rows.empty:
            continue
        eval_intervals = _merge_intervals(
            list(zip(eval_rows["entry_pos"].astype(int).tolist(), eval_rows["exit_pos"].astype(int).tolist()))
        )
        eval_end_pos = int(eval_rows["date_pos"].astype(int).max())
        embargo_upper = eval_end_pos + int(embargo_sessions) if embargo_sessions is not None else None

        for row in group[group["split_role"] == "train"].itertuples(index=False):
            overlaps = _interval_overlaps_any(int(row.entry_pos), int(row.exit_pos), eval_intervals)
            if overlaps:
                overlap_violations += 1
                if overlap_sample is None:
                    overlap_sample = {
                        "date": row.date,
                        "instrument_id": row.instrument_id,
                        "label_name": row.label_name,
                    }
            if embargo_upper is not None and eval_end_pos < int(row.date_pos) <= embargo_upper:
                train_embargo_violations += 1
                if embargo_sample is None:
                    embargo_sample = {
                        "date": row.date,
                        "instrument_id": row.instrument_id,
                        "label_name": row.label_name,
                    }

        for row in group[group["split_role"] == "dropped_by_purge"].itertuples(index=False):
            overlaps = _interval_overlaps_any(int(row.entry_pos), int(row.exit_pos), eval_intervals)
            if not overlaps:
                dropped_purge_non_overlap += 1
                if dropped_sample is None:
                    dropped_sample = {
                        "date": row.date,
                        "instrument_id": row.instrument_id,
                        "label_name": row.label_name,
                    }

        if embargo_upper is not None:
            for row in group[group["split_role"] == "dropped_by_embargo"].itertuples(index=False):
                in_window = eval_end_pos < int(row.date_pos) <= embargo_upper
                if not in_window:
                    dropped_embargo_outside_window += 1
                    if dropped_embargo_sample is None:
                        dropped_embargo_sample = {
                            "date": row.date,
                            "instrument_id": row.instrument_id,
                            "label_name": row.label_name,
                        }

    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="splits_train_eval_no_overlap",
        object_type="split",
        n_rows_evaluated=n_rows,
        n_violations=int(overlap_violations),
        expected_rule="Train rows must not overlap economically with valid/test windows.",
        pass_message="No train/eval economic overlap found in purged_splits.",
        violation_message="Train/eval overlap detected in purged_splits.",
        sample=overlap_sample,
    )

    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="splits_dropped_by_purge_overlap_expected",
        object_type="split",
        n_rows_evaluated=int((frame["split_role"] == "dropped_by_purge").sum()),
        n_violations=int(dropped_purge_non_overlap),
        expected_rule="Rows marked dropped_by_purge should overlap valid/test economic windows.",
        pass_message="dropped_by_purge rows are consistent with overlap policy.",
        violation_message="Rows marked dropped_by_purge without overlap were found.",
        severity_if_violation=SEVERITY_WARN,
        sample=dropped_sample,
    )

    if embargo_sessions is None:
        _emit_rule(
            findings=findings,
            metrics=metrics,
            check_name="splits_embargo_train_exclusion",
            object_type="split",
            n_rows_evaluated=0,
            n_violations=1,
            expected_rule="Embargo checks require embargo_sessions metadata from purged_splits.summary.json.",
            pass_message="Embargo metadata available.",
            violation_message="Embargo metadata unavailable; embargo checks unverified for purged_splits.",
            severity_if_violation=SEVERITY_WARN,
            observed_value="embargo_sessions_unavailable",
        )
    else:
        _emit_rule(
            findings=findings,
            metrics=metrics,
            check_name="splits_embargo_train_exclusion",
            object_type="split",
            n_rows_evaluated=n_rows,
            n_violations=int(train_embargo_violations),
            expected_rule=f"Train rows must be excluded from embargo window after eval block (embargo={embargo_sessions}).",
            pass_message="No train rows found inside embargo window in purged_splits.",
            violation_message="Train rows found inside embargo window in purged_splits.",
            sample=embargo_sample,
        )
        _emit_rule(
            findings=findings,
            metrics=metrics,
            check_name="splits_dropped_by_embargo_window_expected",
            object_type="split",
            n_rows_evaluated=int((frame["split_role"] == "dropped_by_embargo").sum()),
            n_violations=int(dropped_embargo_outside_window),
            expected_rule="Rows marked dropped_by_embargo should be inside embargo window.",
            pass_message="dropped_by_embargo rows are consistent with embargo window.",
            violation_message="Rows marked dropped_by_embargo outside embargo window detected.",
            severity_if_violation=SEVERITY_WARN,
            sample=dropped_embargo_sample,
        )


def _check_cv_folds(
    *,
    cv_folds: pd.DataFrame | None,
    session_pos_map: dict[pd.Timestamp, int],
    findings: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    embargo_sessions: int | None,
) -> None:
    if cv_folds is None:
        _emit_rule(
            findings=findings,
            metrics=metrics,
            check_name="cv_folds_file_available",
            object_type="cv_fold",
            n_rows_evaluated=0,
            n_violations=1,
            expected_rule="purged_cv_folds.parquet should be available for CV leakage audit.",
            pass_message="purged_cv_folds.parquet is available.",
            violation_message="purged_cv_folds.parquet missing; CV leakage checks unverified.",
            severity_if_violation=SEVERITY_WARN,
            observed_value="file_missing",
        )
        return

    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="cv_folds_file_available",
        object_type="cv_fold",
        n_rows_evaluated=1,
        n_violations=0,
        expected_rule="purged_cv_folds.parquet should be available for CV leakage audit.",
        pass_message="purged_cv_folds.parquet is available.",
        violation_message="purged_cv_folds.parquet missing.",
    )

    required = ["fold_id", "date", "instrument_id", "horizon_days", "label_name", "split_role", "entry_date", "exit_date"]
    missing = [col for col in required if col not in cv_folds.columns]
    if missing:
        _emit_missing_columns(
            findings=findings,
            metrics=metrics,
            check_name="cv_required_columns",
            object_type="cv_fold",
            missing_columns=missing,
            expected_columns=required,
        )
        return

    frame = cv_folds.copy()
    frame["fold_id"] = pd.to_numeric(frame["fold_id"], errors="coerce")
    frame["date"] = _normalize_date(frame["date"])
    frame["entry_date"] = _normalize_date(frame["entry_date"])
    frame["exit_date"] = _normalize_date(frame["exit_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["label_name"] = frame["label_name"].astype(str)
    frame["split_role"] = frame["split_role"].astype(str)
    frame["horizon_days"] = pd.to_numeric(frame["horizon_days"], errors="coerce")
    n_rows = int(len(frame))

    invalid_fold_id_mask = frame["fold_id"].isna()
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="cv_fold_id_parseable",
        object_type="cv_fold",
        n_rows_evaluated=n_rows,
        n_violations=int(invalid_fold_id_mask.sum()),
        expected_rule="fold_id must be numeric/non-null.",
        pass_message="All fold_id values are parseable.",
        violation_message="Invalid fold_id values detected.",
        sample=_first_violation_row(frame, invalid_fold_id_mask),
    )

    dup_mask = frame.duplicated(["fold_id", "date", "instrument_id", "horizon_days", "label_name"], keep=False)
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="cv_logical_pk_unique",
        object_type="cv_fold",
        n_rows_evaluated=n_rows,
        n_violations=int(dup_mask.sum()),
        expected_rule="(fold_id, date, instrument_id, horizon_days, label_name) must be unique.",
        pass_message="CV folds logical PK has no duplicates.",
        violation_message="Duplicate CV fold PK rows detected.",
        sample=_first_violation_row(frame, dup_mask),
    )

    invalid_role_mask = ~frame["split_role"].isin(ALLOWED_CV_SPLIT_ROLES)
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="cv_split_role_allowed",
        object_type="cv_fold",
        n_rows_evaluated=n_rows,
        n_violations=int(invalid_role_mask.sum()),
        expected_rule=f"split_role in {sorted(ALLOWED_CV_SPLIT_ROLES)}.",
        pass_message="All CV split_role values are allowed.",
        violation_message="Invalid CV split_role values detected.",
        sample=_first_violation_row(frame, invalid_role_mask),
    )

    valid_sessions = set(session_pos_map.keys())
    out_of_calendar_mask = (
        ~frame["date"].isin(valid_sessions)
        | ~frame["entry_date"].isin(valid_sessions)
        | ~frame["exit_date"].isin(valid_sessions)
    )
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="cv_dates_in_calendar",
        object_type="cv_fold",
        n_rows_evaluated=n_rows,
        n_violations=int(out_of_calendar_mask.sum()),
        expected_rule="CV fold timestamps must exist in trading calendar sessions.",
        pass_message="All CV fold timestamps are in trading sessions.",
        violation_message="CV fold timestamps outside trading sessions detected.",
        sample=_first_violation_row(frame, out_of_calendar_mask),
    )

    frame["date_pos"] = frame["date"].map(session_pos_map)
    frame["entry_pos"] = frame["entry_date"].map(session_pos_map)
    frame["exit_pos"] = frame["exit_date"].map(session_pos_map)

    missing_valid_blocks = 0
    train_overlap_count = 0
    dropped_non_overlap = 0
    embargo_train_violations = 0
    sample_overlap: dict[str, Any] | None = None
    sample_dropped: dict[str, Any] | None = None
    sample_embargo: dict[str, Any] | None = None

    group_cols = ["fold_id", "label_name", "horizon_days"]
    for _, group in frame.groupby(group_cols, dropna=False):
        valid_rows = group[group["split_role"] == "valid"].copy()
        if valid_rows.empty:
            missing_valid_blocks += 1
            continue

        valid_intervals = _merge_intervals(
            list(zip(valid_rows["entry_pos"].astype(int).tolist(), valid_rows["exit_pos"].astype(int).tolist()))
        )
        valid_end_pos = int(valid_rows["date_pos"].astype(int).max())
        embargo_upper = valid_end_pos + int(embargo_sessions) if embargo_sessions is not None else None

        for row in group[group["split_role"] == "train"].itertuples(index=False):
            overlaps = _interval_overlaps_any(int(row.entry_pos), int(row.exit_pos), valid_intervals)
            if overlaps:
                train_overlap_count += 1
                if sample_overlap is None:
                    sample_overlap = {
                        "date": row.date,
                        "instrument_id": row.instrument_id,
                        "label_name": row.label_name,
                    }
            if embargo_upper is not None and valid_end_pos < int(row.date_pos) <= embargo_upper:
                embargo_train_violations += 1
                if sample_embargo is None:
                    sample_embargo = {
                        "date": row.date,
                        "instrument_id": row.instrument_id,
                        "label_name": row.label_name,
                    }

        for row in group[group["split_role"] == "dropped_by_purge"].itertuples(index=False):
            overlaps = _interval_overlaps_any(int(row.entry_pos), int(row.exit_pos), valid_intervals)
            if not overlaps:
                dropped_non_overlap += 1
                if sample_dropped is None:
                    sample_dropped = {
                        "date": row.date,
                        "instrument_id": row.instrument_id,
                        "label_name": row.label_name,
                    }

    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="cv_valid_block_present",
        object_type="cv_fold",
        n_rows_evaluated=len(frame.groupby(group_cols)),
        n_violations=int(missing_valid_blocks),
        expected_rule="Each (fold_id, label_name, horizon_days) block must contain valid rows.",
        pass_message="All CV fold blocks contain valid rows.",
        violation_message="CV fold blocks without valid rows detected.",
        # Some long-horizon label blocks can legitimately lose the valid segment
        # after horizon truncation/purge at the end of sample. This is a
        # coverage warning, not temporal leakage by itself.
        severity_if_violation=SEVERITY_WARN,
    )

    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="cv_train_eval_no_overlap",
        object_type="cv_fold",
        n_rows_evaluated=n_rows,
        n_violations=int(train_overlap_count),
        expected_rule="CV train rows must not overlap economically with valid rows.",
        pass_message="No CV train/valid overlap detected.",
        violation_message="CV train/valid overlap detected.",
        sample=sample_overlap,
    )

    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="cv_dropped_by_purge_overlap_expected",
        object_type="cv_fold",
        n_rows_evaluated=int((frame["split_role"] == "dropped_by_purge").sum()),
        n_violations=int(dropped_non_overlap),
        expected_rule="Rows marked dropped_by_purge should overlap valid economic window.",
        pass_message="CV dropped_by_purge rows are consistent with overlap policy.",
        violation_message="CV dropped_by_purge rows without overlap detected.",
        severity_if_violation=SEVERITY_WARN,
        sample=sample_dropped,
    )

    if embargo_sessions is None:
        _emit_rule(
            findings=findings,
            metrics=metrics,
            check_name="cv_embargo_train_exclusion",
            object_type="cv_fold",
            n_rows_evaluated=0,
            n_violations=1,
            expected_rule="Embargo checks require embargo_sessions metadata from purged_cv_folds.summary.json.",
            pass_message="CV embargo metadata available.",
            violation_message="CV embargo metadata unavailable; embargo checks unverified.",
            severity_if_violation=SEVERITY_WARN,
            observed_value="embargo_sessions_unavailable",
        )
    else:
        _emit_rule(
            findings=findings,
            metrics=metrics,
            check_name="cv_embargo_train_exclusion",
            object_type="cv_fold",
            n_rows_evaluated=n_rows,
            n_violations=int(embargo_train_violations),
            expected_rule=f"No CV train rows inside embargo window after valid block (embargo={embargo_sessions}).",
            pass_message="No CV train rows found inside embargo window.",
            violation_message="CV train rows found inside embargo window.",
            sample=sample_embargo,
        )


def _check_fundamentals_visibility(
    *,
    fundamentals: pd.DataFrame | None,
    findings: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
) -> None:
    if fundamentals is None:
        _emit_rule(
            findings=findings,
            metrics=metrics,
            check_name="fundamentals_file_available",
            object_type="feature",
            n_rows_evaluated=0,
            n_violations=1,
            expected_rule="fundamentals_pit.parquet should be available for PIT acceptance checks.",
            pass_message="fundamentals_pit.parquet is available.",
            violation_message="fundamentals_pit.parquet missing; fundamental PIT checks unverified.",
            severity_if_violation=SEVERITY_WARN,
            observed_value="file_missing",
        )
        return

    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="fundamentals_file_available",
        object_type="feature",
        n_rows_evaluated=1,
        n_violations=0,
        expected_rule="fundamentals_pit.parquet should be available for PIT acceptance checks.",
        pass_message="fundamentals_pit.parquet is available.",
        violation_message="fundamentals_pit.parquet missing.",
    )

    required = ["instrument_id", "asof_date", "metric_name", "metric_value"]
    missing = [col for col in required if col not in fundamentals.columns]
    if missing:
        _emit_missing_columns(
            findings=findings,
            metrics=metrics,
            check_name="fundamentals_required_columns",
            object_type="feature",
            missing_columns=missing,
            expected_columns=required,
        )
        return

    frame = fundamentals.copy()
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["asof_date"] = pd.to_datetime(frame["asof_date"], utc=True, errors="coerce")
    n_rows = int(len(frame))

    invalid_asof_mask = frame["asof_date"].isna()
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="fundamentals_asof_parseable",
        object_type="feature",
        n_rows_evaluated=n_rows,
        n_violations=int(invalid_asof_mask.sum()),
        expected_rule="asof_date must be parseable UTC timestamp.",
        pass_message="All fundamentals asof_date values are parseable.",
        violation_message="Invalid asof_date values in fundamentals_pit.",
    )

    if "acceptance_ts" not in frame.columns:
        _emit_rule(
            findings=findings,
            metrics=metrics,
            check_name="fundamentals_acceptance_column_available",
            object_type="feature",
            n_rows_evaluated=n_rows,
            n_violations=1,
            expected_rule="acceptance_ts column should be present for strict PIT verification.",
            pass_message="acceptance_ts column is present in fundamentals_pit.",
            violation_message="acceptance_ts missing in fundamentals_pit; strict PIT check is unverified.",
            severity_if_violation=SEVERITY_WARN,
        )
        return

    frame["acceptance_ts"] = pd.to_datetime(frame["acceptance_ts"], utc=True, errors="coerce")
    invalid_acc_mask = frame["acceptance_ts"].isna()
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="fundamentals_acceptance_parseable",
        object_type="feature",
        n_rows_evaluated=n_rows,
        n_violations=int(invalid_acc_mask.sum()),
        expected_rule="acceptance_ts must be parseable UTC timestamp.",
        pass_message="All fundamentals acceptance_ts values are parseable.",
        violation_message="Invalid acceptance_ts values in fundamentals_pit.",
    )

    acceptance_after_asof_mask = (
        frame["acceptance_ts"].notna()
        & frame["asof_date"].notna()
        & (frame["acceptance_ts"] > frame["asof_date"])
    )
    sample: dict[str, Any] | None = None
    if acceptance_after_asof_mask.any():
        first = frame.loc[acceptance_after_asof_mask].head(1).iloc[0]
        sample = {
            "instrument_id": first.get("instrument_id"),
            "label_name": _to_string(first.get("metric_name")),
        }
    _emit_rule(
        findings=findings,
        metrics=metrics,
        check_name="fundamentals_acceptance_not_after_asof",
        object_type="feature",
        n_rows_evaluated=n_rows,
        n_violations=int(acceptance_after_asof_mask.sum()),
        expected_rule="acceptance_ts must be <= asof_date in fundamentals_pit visibility semantics.",
        pass_message="fundamentals_pit satisfies acceptance_ts <= asof_date.",
        violation_message="fundamentals_pit rows with acceptance_ts > asof_date detected.",
        sample=sample,
    )


def run_leakage_audit(
    *,
    labels_path: str | Path | None = None,
    features_path: str | Path | None = None,
    model_dataset_path: str | Path | None = None,
    trading_calendar_path: str | Path | None = None,
    purged_splits_path: str | Path | None = None,
    purged_cv_folds_path: str | Path | None = None,
    fundamentals_pit_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    run_id: str = MODULE_VERSION,
) -> LeakageAuditResult:
    logger = get_logger("validation.leakage_audit")
    base = data_dir()

    labels_source = Path(labels_path).expanduser().resolve() if labels_path else (base / "labels" / "labels_forward.parquet")
    features_source = Path(features_path).expanduser().resolve() if features_path else (base / "features" / "features_matrix.parquet")
    dataset_source = (
        Path(model_dataset_path).expanduser().resolve()
        if model_dataset_path
        else (base / "datasets" / "model_dataset.parquet")
    )
    calendar_source = (
        Path(trading_calendar_path).expanduser().resolve()
        if trading_calendar_path
        else (base / "reference" / "trading_calendar.parquet")
    )
    splits_source = _resolve_optional_path(
        purged_splits_path,
        default_path=base / "labels" / "purged_splits.parquet",
    )
    cv_source = _resolve_optional_path(
        purged_cv_folds_path,
        default_path=base / "labels" / "purged_cv_folds.parquet",
    )
    fundamentals_source = _resolve_optional_path(
        fundamentals_pit_path,
        default_path=base / "edgar" / "fundamentals_pit.parquet",
    )

    labels = read_parquet(labels_source)
    features = read_parquet(features_source)
    dataset = read_parquet(dataset_source)
    calendar = read_parquet(calendar_source)
    splits = read_parquet(splits_source) if splits_source is not None else None
    cv_folds = read_parquet(cv_source) if cv_source is not None else None
    fundamentals = read_parquet(fundamentals_source) if fundamentals_source is not None else None

    if "date" not in calendar.columns or "is_session" not in calendar.columns:
        raise ValueError("trading_calendar must contain 'date' and 'is_session' columns.")
    calendar = calendar.copy()
    calendar["date"] = _normalize_date(calendar["date"])
    calendar["is_session"] = calendar["is_session"].astype(bool)
    sessions = pd.DatetimeIndex(sorted(calendar.loc[calendar["is_session"], "date"].dropna().unique()))
    if sessions.empty:
        raise ValueError("trading_calendar has no active sessions for leakage audit.")
    session_pos_map = {pd.Timestamp(item): int(idx) for idx, item in enumerate(sessions)}
    max_session_pos = int(len(sessions) - 1)

    splits_embargo: int | None = None
    if splits_source is not None:
        splits_summary_path = splits_source.with_name("purged_splits.summary.json")
        if splits_summary_path.exists():
            payload = json.loads(splits_summary_path.read_text(encoding="utf-8"))
            if payload.get("embargo_sessions") is not None:
                splits_embargo = int(payload["embargo_sessions"])

    cv_embargo: int | None = None
    if cv_source is not None:
        cv_summary_path = cv_source.with_name("purged_cv_folds.summary.json")
        if cv_summary_path.exists():
            payload = json.loads(cv_summary_path.read_text(encoding="utf-8"))
            if payload.get("embargo_sessions") is not None:
                cv_embargo = int(payload["embargo_sessions"])

    findings_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []

    labels_norm = _check_labels(
        labels=labels,
        session_pos_map=session_pos_map,
        max_session_pos=max_session_pos,
        findings=findings_rows,
        metrics=metrics_rows,
    )
    features_norm = _check_features(
        features=features,
        findings=findings_rows,
        metrics=metrics_rows,
    )
    _check_dataset(
        dataset=dataset,
        labels=labels_norm,
        features=features_norm,
        findings=findings_rows,
        metrics=metrics_rows,
    )
    _check_splits(
        splits=splits,
        session_pos_map=session_pos_map,
        findings=findings_rows,
        metrics=metrics_rows,
        embargo_sessions=splits_embargo,
    )
    _check_cv_folds(
        cv_folds=cv_folds,
        session_pos_map=session_pos_map,
        findings=findings_rows,
        metrics=metrics_rows,
        embargo_sessions=cv_embargo,
    )
    _check_fundamentals_visibility(
        fundamentals=fundamentals,
        findings=findings_rows,
        metrics=metrics_rows,
    )

    if not findings_rows:
        raise ValueError("Leakage audit produced no findings rows.")
    if not metrics_rows:
        raise ValueError("Leakage audit produced no metrics rows.")

    findings_df = pd.DataFrame(findings_rows)
    metrics_df = pd.DataFrame(metrics_rows)

    for col in [
        "check_name",
        "severity",
        "object_type",
        "message",
        "observed_value",
        "expected_rule",
        "instrument_id",
        "label_name",
        "feature_name",
    ]:
        if col in findings_df.columns:
            findings_df[col] = findings_df[col].astype("string")
    findings_df["date"] = pd.to_datetime(findings_df.get("date", pd.NaT), errors="coerce").dt.normalize()

    for col in ["check_name", "severity", "object_type", "message"]:
        metrics_df[col] = metrics_df[col].astype("string")
    metrics_df["n_rows_evaluated"] = pd.to_numeric(metrics_df["n_rows_evaluated"], errors="coerce").fillna(0).astype("int64")
    metrics_df["n_violations"] = pd.to_numeric(metrics_df["n_violations"], errors="coerce").fillna(0).astype("int64")
    metrics_df["violation_rate"] = pd.to_numeric(metrics_df["violation_rate"], errors="coerce")

    metrics_df = metrics_df.drop_duplicates(subset=["check_name"], keep="last").sort_values("check_name").reset_index(drop=True)

    worst_by_check = (
        metrics_df.groupby("check_name")["severity"]
        .agg(lambda s: max((SEVERITY_RANK.get(str(v), -1) for v in s), default=-1))
    )
    n_fail = int((worst_by_check == SEVERITY_RANK[SEVERITY_FAIL]).sum())
    n_warn = int((worst_by_check == SEVERITY_RANK[SEVERITY_WARN]).sum())
    n_pass = int((worst_by_check == SEVERITY_RANK[SEVERITY_PASS]).sum())
    if n_fail > 0:
        overall_status = SEVERITY_FAIL
    elif n_warn > 0:
        overall_status = SEVERITY_WARN
    else:
        overall_status = SEVERITY_PASS

    failed_checks = metrics_df.loc[metrics_df["severity"] == SEVERITY_FAIL, "check_name"].astype(str).tolist()
    warning_checks = metrics_df.loc[metrics_df["severity"] == SEVERITY_WARN, "check_name"].astype(str).tolist()

    non_pass = findings_df[findings_df["severity"].isin([SEVERITY_FAIL, SEVERITY_WARN])].copy()
    non_pass["severity_rank"] = non_pass["severity"].map({SEVERITY_FAIL: 0, SEVERITY_WARN: 1}).fillna(2)
    non_pass = non_pass.sort_values(["severity_rank", "check_name"]).drop(columns=["severity_rank"])
    key_findings = non_pass["message"].astype(str).head(10).tolist()

    built_ts_utc = datetime.now(UTC).isoformat()
    config_hash = _config_hash(
        {
            "version": MODULE_VERSION,
            "labels_path": str(labels_source),
            "features_path": str(features_source),
            "model_dataset_path": str(dataset_source),
            "trading_calendar_path": str(calendar_source),
            "purged_splits_path": str(splits_source) if splits_source else None,
            "purged_cv_folds_path": str(cv_source) if cv_source else None,
            "fundamentals_pit_path": str(fundamentals_source) if fundamentals_source else None,
            "n_findings": int(len(findings_df)),
            "n_checks_run": int(len(metrics_df)),
        }
    )

    findings_df["run_id"] = run_id
    findings_df["config_hash"] = config_hash
    findings_df["built_ts_utc"] = built_ts_utc
    metrics_df["run_id"] = run_id
    metrics_df["config_hash"] = config_hash
    metrics_df["built_ts_utc"] = built_ts_utc

    assert_schema(findings_df, FINDINGS_SCHEMA)
    assert_schema(metrics_df, METRICS_SCHEMA)

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (base / "validation")
    target_dir.mkdir(parents=True, exist_ok=True)

    findings_path = write_parquet(
        findings_df,
        target_dir / "leakage_audit_findings.parquet",
        schema_name=FINDINGS_SCHEMA.name,
        run_id=run_id,
    )
    metrics_path = write_parquet(
        metrics_df,
        target_dir / "leakage_audit_metrics.parquet",
        schema_name=METRICS_SCHEMA.name,
        run_id=run_id,
    )

    summary_payload = {
        "built_ts_utc": built_ts_utc,
        "run_id": run_id,
        "config_hash": config_hash,
        "module_version": MODULE_VERSION,
        "overall_status": overall_status,
        "n_checks_run": int(len(metrics_df)),
        "n_fail": n_fail,
        "n_warn": n_warn,
        "n_pass": n_pass,
        "failed_checks": failed_checks,
        "warning_checks": warning_checks,
        "key_findings": key_findings,
        "input_paths": {
            "labels_forward": str(labels_source),
            "features_matrix": str(features_source),
            "model_dataset": str(dataset_source),
            "trading_calendar": str(calendar_source),
            "purged_splits": str(splits_source) if splits_source else None,
            "purged_cv_folds": str(cv_source) if cv_source else None,
            "fundamentals_pit": str(fundamentals_source) if fundamentals_source else None,
        },
        "output_paths": {
            "leakage_audit_findings": str(findings_path),
            "leakage_audit_metrics": str(metrics_path),
        },
    }
    summary_path = target_dir / "leakage_audit_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "leakage_audit_completed",
        run_id=run_id,
        overall_status=overall_status,
        n_checks_run=int(len(metrics_df)),
        n_fail=n_fail,
        n_warn=n_warn,
        n_pass=n_pass,
        findings_path=str(findings_path),
        summary_path=str(summary_path),
    )

    return LeakageAuditResult(
        findings_path=findings_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        overall_status=overall_status,
        n_fail=n_fail,
        n_warn=n_warn,
        n_pass=n_pass,
        config_hash=config_hash,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MVP temporal/causal leakage audit across labels, features, dataset and splits."
    )
    parser.add_argument("--labels-path", type=str, default=None)
    parser.add_argument("--features-path", type=str, default=None)
    parser.add_argument("--model-dataset-path", type=str, default=None)
    parser.add_argument("--trading-calendar-path", type=str, default=None)
    parser.add_argument("--purged-splits-path", type=str, default=None)
    parser.add_argument("--purged-cv-folds-path", type=str, default=None)
    parser.add_argument("--fundamentals-pit-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_leakage_audit(
        labels_path=args.labels_path,
        features_path=args.features_path,
        model_dataset_path=args.model_dataset_path,
        trading_calendar_path=args.trading_calendar_path,
        purged_splits_path=args.purged_splits_path,
        purged_cv_folds_path=args.purged_cv_folds_path,
        fundamentals_pit_path=args.fundamentals_pit_path,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print("Leakage audit completed:")
    print(f"- findings: {result.findings_path}")
    print(f"- metrics: {result.metrics_path}")
    print(f"- summary: {result.summary_path}")
    print(f"- overall_status: {result.overall_status}")
    print(f"- n_fail: {result.n_fail}")
    print(f"- n_warn: {result.n_warn}")
    print(f"- n_pass: {result.n_pass}")


if __name__ == "__main__":
    main()
