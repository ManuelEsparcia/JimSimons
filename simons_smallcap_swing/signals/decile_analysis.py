from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

# Allow direct script execution: `python simons_smallcap_swing/signals/decile_analysis.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "decile_analysis_mvp_v1"
DEFAULT_SPLIT_ROLES: tuple[str, ...] = ("valid", "test")

SIGNALS_INPUT_SCHEMA = DataSchema(
    name="decile_analysis_signals_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("split_role", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("bucket", "int64", nullable=True),
        ColumnSpec("is_top", "bool", nullable=False),
        ColumnSpec("is_bottom", "bool", nullable=False),
    ),
    primary_key=("date", "instrument_id", "model_name", "label_name"),
    allow_extra_columns=True,
)

LABELS_INPUT_SCHEMA = DataSchema(
    name="decile_analysis_labels_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("horizon_days", "int64", nullable=False),
        ColumnSpec("label_value", "number", nullable=False),
    ),
    primary_key=("date", "instrument_id", "horizon_days", "label_name"),
    allow_extra_columns=True,
)

DECILE_DAILY_SCHEMA = DataSchema(
    name="decile_daily_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("bucket", "int64", nullable=False),
        ColumnSpec("n_names_bucket", "int64", nullable=False),
        ColumnSpec("bucket_mean_target", "float64", nullable=False),
        ColumnSpec("bucket_median_target", "float64", nullable=False),
        ColumnSpec("top_bucket_mean_target", "float64", nullable=True),
        ColumnSpec("bottom_bucket_mean_target", "float64", nullable=True),
        ColumnSpec("top_minus_bottom_spread", "float64", nullable=True),
        ColumnSpec("n_names_ranked", "int64", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
    ),
    primary_key=("date", "bucket", "model_name", "label_name"),
    allow_extra_columns=True,
)

DECILE_SUMMARY_SCHEMA = DataSchema(
    name="decile_summary_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("bucket", "int64", nullable=False),
        ColumnSpec("n_obs_bucket", "int64", nullable=False),
        ColumnSpec("n_dates_bucket", "int64", nullable=False),
        ColumnSpec("bucket_mean_target", "float64", nullable=False),
        ColumnSpec("bucket_median_target", "float64", nullable=False),
        ColumnSpec("bucket_std_target", "float64", nullable=True),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
    ),
    primary_key=("bucket", "model_name", "label_name"),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class DecileAnalysisResult:
    decile_daily_path: Path
    decile_summary_path: Path
    summary_json_path: Path
    row_count_daily: int
    row_count_summary: int
    model_name: str
    label_name: str
    config_hash: str


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains invalid dates.")
    return parsed.dt.normalize()


def _normalize_split_roles(split_roles: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(sorted({str(item).strip() for item in split_roles if str(item).strip()}))
    if not normalized:
        raise ValueError("At least one split_role is required.")
    return normalized


def _config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return float(parsed)


def _prepare_signals(source: Path) -> pd.DataFrame:
    signals = read_parquet(source)
    assert_schema(signals, SIGNALS_INPUT_SCHEMA)
    signals = signals.copy()
    signals["date"] = _normalize_date(signals["date"], column="date")
    signals["instrument_id"] = signals["instrument_id"].astype(str)
    signals["split_role"] = signals["split_role"].astype(str)
    signals["label_name"] = signals["label_name"].astype(str)
    signals["model_name"] = signals["model_name"].astype(str)
    signals["is_top"] = signals["is_top"].astype(bool)
    signals["is_bottom"] = signals["is_bottom"].astype(bool)
    signals["bucket"] = pd.to_numeric(signals["bucket"], errors="coerce").astype("Int64")
    if "n_names_ranked" in signals.columns:
        signals["n_names_ranked"] = pd.to_numeric(signals["n_names_ranked"], errors="coerce").astype("Int64")
    else:
        signals["n_names_ranked"] = pd.Series([pd.NA] * len(signals), dtype="Int64")

    if signals.duplicated(["date", "instrument_id", "model_name", "label_name"], keep=False).any():
        raise ValueError("signals_daily contains duplicate (date, instrument_id, model_name, label_name) rows.")
    return signals


def _prepare_labels(source: Path) -> pd.DataFrame:
    labels = read_parquet(source)
    assert_schema(labels, LABELS_INPUT_SCHEMA)
    labels = labels.copy()
    labels["date"] = _normalize_date(labels["date"], column="date")
    labels["instrument_id"] = labels["instrument_id"].astype(str)
    labels["label_name"] = labels["label_name"].astype(str)
    labels["horizon_days"] = pd.to_numeric(labels["horizon_days"], errors="coerce").astype("Int64")
    labels["label_value"] = pd.to_numeric(labels["label_value"], errors="coerce")

    if labels["label_value"].isna().any():
        raise ValueError("labels_forward has null/non-numeric label_value values.")
    if labels.duplicated(["date", "instrument_id", "horizon_days", "label_name"], keep=False).any():
        raise ValueError("labels_forward contains duplicate (date, instrument_id, horizon_days, label_name) rows.")
    return labels


def _select_unique_value(frame: pd.DataFrame, column: str, *, provided: str | None) -> str:
    if provided is not None:
        selected = str(provided).strip()
        if not selected:
            raise ValueError(f"Provided {column} is empty.")
        frame_filtered = frame[frame[column].astype(str) == selected]
        if frame_filtered.empty:
            raise ValueError(f"No rows left after filtering {column}='{selected}'.")
        return selected

    unique_vals = sorted(frame[column].astype(str).unique().tolist())
    if len(unique_vals) != 1:
        raise ValueError(
            f"Expected exactly one {column} per run. "
            f"Observed {column} values: {unique_vals}. "
            f"Pass --{column.replace('_', '-')} explicitly."
        )
    return str(unique_vals[0])


def _resolve_join_keys(
    *,
    signals: pd.DataFrame,
    labels: pd.DataFrame,
    horizon_days: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], int | None]:
    join_keys = ["date", "instrument_id", "label_name"]

    signals_has_h = "horizon_days" in signals.columns
    labels_has_h = "horizon_days" in labels.columns
    selected_horizon: int | None = None

    if signals_has_h:
        signals = signals.copy()
        signals["horizon_days"] = pd.to_numeric(signals["horizon_days"], errors="coerce").astype("Int64")
        if horizon_days is not None:
            signals = signals[signals["horizon_days"] == int(horizon_days)].copy()
        if signals.empty:
            raise ValueError("No signal rows left after horizon_days filtering.")
        unique_signal_h = sorted(signals["horizon_days"].dropna().astype(int).unique().tolist())
        if len(unique_signal_h) != 1:
            raise ValueError(
                f"signals_daily has multiple horizon_days values: {unique_signal_h}. "
                "Pass --horizon-days explicitly or pre-filter input."
            )
        selected_horizon = unique_signal_h[0]

    if labels_has_h:
        labels = labels.copy()
        labels["horizon_days"] = pd.to_numeric(labels["horizon_days"], errors="coerce").astype("Int64")
        if horizon_days is not None:
            labels = labels[labels["horizon_days"] == int(horizon_days)].copy()
        if labels.empty:
            raise ValueError("No label rows left after horizon_days filtering.")

    if signals_has_h and labels_has_h:
        if selected_horizon is None:
            unique_signal_h = sorted(signals["horizon_days"].dropna().astype(int).unique().tolist())
            selected_horizon = unique_signal_h[0]
        labels = labels[labels["horizon_days"] == int(selected_horizon)].copy()
        if labels.empty:
            raise ValueError(
                f"No labels for selected horizon_days={selected_horizon} after filtering."
            )
        join_keys.append("horizon_days")
    elif not signals_has_h and labels_has_h:
        unique_label_h = sorted(labels["horizon_days"].dropna().astype(int).unique().tolist())
        if len(unique_label_h) != 1:
            raise ValueError(
                "signals_daily has no horizon_days column but labels_forward has multiple horizons. "
                f"Observed label horizons: {unique_label_h}. Pass --horizon-days explicitly."
            )
        selected_horizon = unique_label_h[0]
        labels = labels[labels["horizon_days"] == int(selected_horizon)].copy()

    if labels.duplicated(join_keys, keep=False).any():
        raise ValueError(
            f"labels_forward has duplicate rows for join keys {join_keys}; join would be ambiguous."
        )
    if signals.duplicated(join_keys + ["model_name"], keep=False).any():
        raise ValueError(
            f"signals_daily has duplicate rows for logical keys {join_keys + ['model_name']}."
        )

    return signals, labels, join_keys, selected_horizon


def _compute_monotonicity(decile_summary: pd.DataFrame) -> float | None:
    if len(decile_summary) < 2:
        return None
    x = pd.to_numeric(decile_summary["bucket"], errors="coerce")
    y = pd.to_numeric(decile_summary["bucket_mean_target"], errors="coerce")
    corr = x.corr(y, method="pearson")
    if pd.isna(corr):
        return None
    return float(corr)


def run_decile_analysis(
    *,
    signals_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_name: str | None = None,
    label_name: str | None = None,
    horizon_days: int | None = None,
    split_roles: Iterable[str] = DEFAULT_SPLIT_ROLES,
    expected_n_buckets: int | None = None,
    run_id: str = MODULE_VERSION,
) -> DecileAnalysisResult:
    logger = get_logger("signals.decile_analysis")

    signals_source = (
        Path(signals_path).expanduser().resolve()
        if signals_path
        else (data_dir() / "signals" / "signals_daily.parquet")
    )
    labels_source = (
        Path(labels_path).expanduser().resolve()
        if labels_path
        else (data_dir() / "labels" / "labels_forward.parquet")
    )

    signals = _prepare_signals(signals_source)
    labels = _prepare_labels(labels_source)

    selected_split_roles = _normalize_split_roles(split_roles)
    signals = signals[signals["split_role"].isin(set(selected_split_roles))].copy()
    if signals.empty:
        raise ValueError(
            f"No signal rows left after split_role filter: {list(selected_split_roles)}"
        )

    selected_model_name = _select_unique_value(signals, "model_name", provided=model_name)
    signals = signals[signals["model_name"] == selected_model_name].copy()

    selected_label_name = _select_unique_value(signals, "label_name", provided=label_name)
    signals = signals[signals["label_name"] == selected_label_name].copy()
    labels = labels[labels["label_name"] == selected_label_name].copy()
    if labels.empty:
        raise ValueError(f"No labels found for label_name='{selected_label_name}'.")

    signals, labels, join_keys, selected_horizon = _resolve_join_keys(
        signals=signals,
        labels=labels,
        horizon_days=horizon_days,
    )

    merged = signals.merge(
        labels[join_keys + ["label_value"]],
        on=join_keys,
        how="inner",
    )
    if merged.empty:
        raise ValueError("Join between signals_daily and labels_forward produced no rows.")

    merged["target_value"] = pd.to_numeric(merged["label_value"], errors="coerce")
    if merged["target_value"].isna().any():
        raise ValueError("Joined target_value has null/non-numeric values.")

    ranked = merged[merged["bucket"].notna()].copy()
    if ranked.empty:
        raise ValueError("No ranked rows found (bucket is null for all merged rows).")

    ranked["bucket"] = pd.to_numeric(ranked["bucket"], errors="coerce").astype("Int64")
    if expected_n_buckets is not None:
        if int(expected_n_buckets) < 2:
            raise ValueError("expected_n_buckets must be >= 2.")
        invalid_bucket = ranked["bucket"].dropna().astype(int)
        if (invalid_bucket < 1).any() or (invalid_bucket > int(expected_n_buckets)).any():
            raise ValueError(
                f"Bucket values outside [1, {int(expected_n_buckets)}] detected in ranked rows."
            )

    daily_bucket = (
        ranked.groupby(["date", "bucket"], as_index=False)
        .agg(
            n_names_bucket=("instrument_id", "nunique"),
            bucket_mean_target=("target_value", "mean"),
            bucket_median_target=("target_value", "median"),
        )
        .sort_values(["date", "bucket"])
        .reset_index(drop=True)
    )

    daily_extremes = (
        ranked.groupby("date", as_index=False)
        .agg(
            top_bucket_mean_target=("target_value", lambda series: float(series[ranked.loc[series.index, "is_top"]].mean()) if ranked.loc[series.index, "is_top"].any() else np.nan),
            bottom_bucket_mean_target=("target_value", lambda series: float(series[ranked.loc[series.index, "is_bottom"]].mean()) if ranked.loc[series.index, "is_bottom"].any() else np.nan),
            n_names_ranked=("instrument_id", "nunique"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily_extremes["top_minus_bottom_spread"] = (
        daily_extremes["top_bucket_mean_target"] - daily_extremes["bottom_bucket_mean_target"]
    )

    decile_daily = daily_bucket.merge(daily_extremes, on="date", how="left")
    decile_daily["model_name"] = selected_model_name
    decile_daily["label_name"] = selected_label_name
    if selected_horizon is not None:
        decile_daily["horizon_days"] = int(selected_horizon)

    decile_daily = decile_daily[
        [
            "date",
            "bucket",
            "n_names_bucket",
            "bucket_mean_target",
            "bucket_median_target",
            "top_bucket_mean_target",
            "bottom_bucket_mean_target",
            "top_minus_bottom_spread",
            "n_names_ranked",
            *(("horizon_days",) if selected_horizon is not None else ()),
            "model_name",
            "label_name",
        ]
    ].sort_values(["date", "bucket"]).reset_index(drop=True)
    assert_schema(decile_daily, DECILE_DAILY_SCHEMA)

    decile_summary = (
        decile_daily.groupby("bucket", as_index=False)
        .agg(
            n_obs_bucket=("date", "count"),
            n_dates_bucket=("date", "nunique"),
            bucket_mean_target=("bucket_mean_target", "mean"),
            bucket_median_target=("bucket_median_target", "median"),
            bucket_std_target=("bucket_mean_target", "std"),
            avg_n_names_bucket=("n_names_bucket", "mean"),
        )
        .sort_values("bucket")
        .reset_index(drop=True)
    )
    decile_summary["model_name"] = selected_model_name
    decile_summary["label_name"] = selected_label_name
    if selected_horizon is not None:
        decile_summary["horizon_days"] = int(selected_horizon)
    assert_schema(decile_summary, DECILE_SUMMARY_SCHEMA)

    spread_daily = (
        daily_extremes[["date", "top_minus_bottom_spread"]]
        .dropna(subset=["top_minus_bottom_spread"])
        .copy()
    )
    monotonicity_score = _compute_monotonicity(decile_summary)
    mean_top = _to_float_or_none(daily_extremes["top_bucket_mean_target"].mean(skipna=True))
    mean_bottom = _to_float_or_none(daily_extremes["bottom_bucket_mean_target"].mean(skipna=True))
    mean_spread = _to_float_or_none(spread_daily["top_minus_bottom_spread"].mean(skipna=True))
    median_spread = _to_float_or_none(spread_daily["top_minus_bottom_spread"].median(skipna=True))
    std_spread = _to_float_or_none(spread_daily["top_minus_bottom_spread"].std(skipna=True, ddof=0))
    positive_spread_rate = (
        float((spread_daily["top_minus_bottom_spread"] > 0).mean()) if not spread_daily.empty else None
    )

    n_buckets_observed = int(pd.to_numeric(decile_summary["bucket"], errors="coerce").max())
    if expected_n_buckets is not None:
        n_buckets_observed = int(expected_n_buckets)

    config_hash = _config_hash(
        {
            "version": MODULE_VERSION,
            "signals_path": str(signals_source),
            "labels_path": str(labels_source),
            "model_name": selected_model_name,
            "label_name": selected_label_name,
            "horizon_days": selected_horizon,
            "split_roles": list(selected_split_roles),
            "expected_n_buckets": int(expected_n_buckets) if expected_n_buckets is not None else None,
        }
    )

    built_ts_utc = datetime.now(UTC).isoformat()
    decile_daily["run_id"] = run_id
    decile_daily["config_hash"] = config_hash
    decile_daily["built_ts_utc"] = built_ts_utc
    decile_summary["run_id"] = run_id
    decile_summary["config_hash"] = config_hash
    decile_summary["built_ts_utc"] = built_ts_utc

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (data_dir() / "signals")
    target_dir.mkdir(parents=True, exist_ok=True)

    decile_daily_path = write_parquet(
        decile_daily,
        target_dir / "decile_daily.parquet",
        schema_name=DECILE_DAILY_SCHEMA.name,
        run_id=run_id,
    )
    decile_summary_path = write_parquet(
        decile_summary,
        target_dir / "decile_summary.parquet",
        schema_name=DECILE_SUMMARY_SCHEMA.name,
        run_id=run_id,
    )

    summary_payload = {
        "created_at_utc": built_ts_utc,
        "run_id": run_id,
        "config_hash": config_hash,
        "module_version": MODULE_VERSION,
        "signals_path": str(signals_source),
        "labels_path": str(labels_source),
        "label_name": selected_label_name,
        "model_name": selected_model_name,
        "horizon_days": selected_horizon,
        "n_buckets": n_buckets_observed,
        "n_dates": int(decile_daily["date"].nunique()),
        "mean_top_bucket_target": mean_top,
        "mean_bottom_bucket_target": mean_bottom,
        "mean_top_minus_bottom_spread": mean_spread,
        "median_top_minus_bottom_spread": median_spread,
        "std_top_minus_bottom_spread": std_spread,
        "positive_spread_rate": positive_spread_rate,
        "monotonicity_score": monotonicity_score,
        "avg_n_names_ranked": _to_float_or_none(decile_daily["n_names_ranked"].mean(skipna=True)),
        "worst_bucket_coverage": int(pd.to_numeric(decile_daily["n_names_bucket"], errors="coerce").min()),
        "split_roles_included": list(selected_split_roles),
        "join_stats": {
            "n_rows_signals_filtered": int(len(signals)),
            "n_rows_labels_filtered": int(len(labels)),
            "n_rows_joined": int(len(merged)),
            "n_rows_ranked": int(len(ranked)),
            "n_rows_without_bucket": int(len(merged) - len(ranked)),
        },
        "output_paths": {
            "decile_daily": str(decile_daily_path),
            "decile_summary": str(decile_summary_path),
        },
    }
    summary_json_path = target_dir / "decile_analysis_summary.json"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "decile_analysis_built",
        run_id=run_id,
        model_name=selected_model_name,
        label_name=selected_label_name,
        n_dates=int(decile_daily["date"].nunique()),
        mean_spread=mean_spread,
        output_path=str(decile_daily_path),
    )

    return DecileAnalysisResult(
        decile_daily_path=decile_daily_path,
        decile_summary_path=decile_summary_path,
        summary_json_path=summary_json_path,
        row_count_daily=int(len(decile_daily)),
        row_count_summary=int(len(decile_summary)),
        model_name=selected_model_name,
        label_name=selected_label_name,
        config_hash=config_hash,
    )


def _parse_csv_strings(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return values


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run signal decile/bucket analysis on realized labels.")
    parser.add_argument("--signals-path", type=str, default=None)
    parser.add_argument("--labels-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--label-name", type=str, default=None)
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument("--split-roles", type=_parse_csv_strings, default=DEFAULT_SPLIT_ROLES)
    parser.add_argument("--expected-n-buckets", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_decile_analysis(
        signals_path=args.signals_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        label_name=args.label_name,
        horizon_days=args.horizon_days,
        split_roles=args.split_roles,
        expected_n_buckets=args.expected_n_buckets,
        run_id=args.run_id,
    )
    print("Decile analysis built:")
    print(f"- decile_daily: {result.decile_daily_path}")
    print(f"- decile_summary: {result.decile_summary_path}")
    print(f"- summary_json: {result.summary_json_path}")
    print(f"- row_count_daily: {result.row_count_daily}")
    print(f"- row_count_summary: {result.row_count_summary}")
    print(f"- model_name: {result.model_name}")
    print(f"- label_name: {result.label_name}")


if __name__ == "__main__":
    main()
