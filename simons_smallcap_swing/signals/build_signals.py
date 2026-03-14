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

# Allow direct script execution: `python simons_smallcap_swing/signals/build_signals.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import app_root, data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "signals_daily_mvp_v1"
DEFAULT_SPLIT_ROLES: tuple[str, ...] = ("valid", "test")
DEFAULT_N_BUCKETS = 10
DEFAULT_TOP_BUCKETS = 1
DEFAULT_BOTTOM_BUCKETS = 1

PREDICTIONS_INPUT_SCHEMA = DataSchema(
    name="signals_predictions_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("split_role", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
    ),
    primary_key=("date", "instrument_id", "split_role", "label_name"),
    allow_extra_columns=True,
)

UNIVERSE_INPUT_SCHEMA = DataSchema(
    name="signals_universe_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("is_eligible", "bool", nullable=False),
    ),
    primary_key=("date", "instrument_id"),
    allow_extra_columns=True,
)

SIGNALS_OUTPUT_SCHEMA = DataSchema(
    name="signals_daily_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("split_role", "string", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("model_name", "string", nullable=False),
        ColumnSpec("signal_source_type", "string", nullable=False),
        ColumnSpec("raw_score", "float64", nullable=True),
        ColumnSpec("rank_pct", "float64", nullable=True),
        ColumnSpec("bucket", "int64", nullable=True),
        ColumnSpec("signal_side", "int64", nullable=False),
        ColumnSpec("is_top", "bool", nullable=False),
        ColumnSpec("is_bottom", "bool", nullable=False),
        ColumnSpec("n_names_ranked", "int64", nullable=False),
        ColumnSpec("bucket_scheme", "string", nullable=False),
    ),
    primary_key=("date", "instrument_id", "model_name", "label_name"),
    allow_extra_columns=True,
)


@dataclass(frozen=True)
class BuildSignalsResult:
    signals_path: Path
    summary_path: Path
    row_count: int
    n_dates: int
    model_name: str
    label_name: str
    signal_source_type: str
    config_hash: str


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains invalid dates.")
    return parsed.dt.normalize()


def _config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _normalize_split_roles(split_roles: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(sorted({str(item).strip() for item in split_roles if str(item).strip()}))
    if not normalized:
        raise ValueError("At least one split_role is required.")
    return normalized


def _resolve_predictions_path(
    predictions_path: str | Path | None,
    *,
    model_name: str | None,
) -> Path:
    if predictions_path:
        source = Path(predictions_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"predictions file not found: {source}")
        return source

    model_alias = str(model_name or "").strip().lower()
    if model_alias in {"ridge", "ridge_baseline"}:
        names = ("ridge_baseline_predictions.parquet",)
    elif model_alias in {"logistic", "logistic_baseline"}:
        names = ("logistic_baseline_predictions.parquet",)
    else:
        names = (
            "ridge_baseline_predictions.parquet",
            "logistic_baseline_predictions.parquet",
        )

    candidates: list[Path] = []
    for root in (
        data_dir() / "models" / "artifacts",
        app_root() / "models" / "artifacts",
    ):
        for name in names:
            candidates.append(root / name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not resolve predictions file. Pass --predictions-path explicitly. "
        f"Checked: {[str(path) for path in candidates]}"
    )


def _infer_model_name(predictions_source: Path, provided_model_name: str | None) -> str:
    if provided_model_name and str(provided_model_name).strip():
        return str(provided_model_name).strip()
    name = predictions_source.name.lower()
    if "ridge" in name:
        return "ridge_baseline"
    if "logistic" in name:
        return "logistic_baseline"
    if "dummy_regressor" in name:
        return "dummy_regressor"
    if "dummy_classifier" in name:
        return "dummy_classifier"
    return predictions_source.stem.replace("_predictions", "")


def _load_predictions(source: Path) -> pd.DataFrame:
    frame = read_parquet(source)
    assert_schema(frame, PREDICTIONS_INPUT_SCHEMA)

    frame = frame.copy()
    frame["date"] = _normalize_date(frame["date"], column="date")
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["split_role"] = frame["split_role"].astype(str)
    frame["label_name"] = frame["label_name"].astype(str)

    if frame.duplicated(["date", "instrument_id", "split_role", "label_name"], keep=False).any():
        raise ValueError(
            "predictions contain duplicate (date, instrument_id, split_role, label_name) rows."
        )
    return frame


def _resolve_signal_source(
    frame: pd.DataFrame,
    *,
    score_column: str | None,
    center_classification_proba: bool,
) -> tuple[pd.Series, str, str]:
    selected_score_col = None
    if score_column:
        candidate = str(score_column).strip()
        if candidate not in frame.columns:
            raise ValueError(
                f"score_column='{candidate}' is not present in predictions. "
                f"Available columns: {sorted(frame.columns)}"
            )
        selected_score_col = candidate
    elif "prediction" in frame.columns:
        selected_score_col = "prediction"
    elif "pred_proba" in frame.columns:
        selected_score_col = "pred_proba"

    if selected_score_col is None:
        raise ValueError(
            "Could not infer score column. Provide --score-column or include "
            "'prediction' (regression) or 'pred_proba' (classification) in predictions."
        )

    raw = pd.to_numeric(frame[selected_score_col], errors="coerce")
    if selected_score_col == "pred_proba":
        if ((raw.dropna() < 0.0) | (raw.dropna() > 1.0)).any():
            raise ValueError("pred_proba must be within [0, 1].")
        if center_classification_proba:
            raw = raw - 0.5
            source_type = "classification_probability_centered"
        else:
            source_type = "classification_probability"
    else:
        source_type = "regression_prediction"

    return raw.astype(float), source_type, selected_score_col


def _load_eligible_universe(universe_history_path: str | Path | None) -> tuple[pd.DataFrame, Path]:
    source = (
        Path(universe_history_path).expanduser().resolve()
        if universe_history_path
        else (data_dir() / "universe" / "universe_history.parquet")
    )
    frame = read_parquet(source)
    assert_schema(frame, UNIVERSE_INPUT_SCHEMA)

    frame = frame.copy()
    frame["date"] = _normalize_date(frame["date"], column="date")
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["is_eligible"] = frame["is_eligible"].astype(bool)

    if frame.duplicated(["date", "instrument_id"], keep=False).any():
        raise ValueError("universe_history has duplicate (date, instrument_id) rows.")

    eligible = frame.loc[frame["is_eligible"], ["date", "instrument_id"]].copy()
    return eligible, source


def _assign_cross_sectional_ranks(
    frame: pd.DataFrame,
    *,
    n_buckets: int,
) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["rank_pct"] = np.nan
    ranked["bucket"] = pd.Series([pd.NA] * len(ranked), dtype="Int64")
    ranked["n_names_ranked"] = 0

    for date, index in ranked.groupby("date", sort=True).groups.items():
        _ = date
        group = ranked.loc[index].copy()
        scored = group[group["raw_score"].notna()].copy()
        n_scored = int(len(scored))
        ranked.loc[index, "n_names_ranked"] = n_scored
        if n_scored == 0:
            continue

        # Deterministic ties: score desc, instrument_id asc.
        scored = scored.sort_values(["raw_score", "instrument_id"], ascending=[False, True]).copy()
        scored["position"] = np.arange(1, n_scored + 1)

        if n_scored == 1:
            scored["rank_pct"] = 1.0
        else:
            scored["rank_pct"] = 1.0 - ((scored["position"] - 1) / float(n_scored - 1))

        scored["bucket"] = (
            n_buckets
            - np.floor(((scored["position"] - 1) * float(n_buckets)) / float(n_scored)).astype(int)
        )
        scored["bucket"] = scored["bucket"].clip(lower=1, upper=n_buckets).astype("Int64")

        ranked.loc[scored.index, "rank_pct"] = scored["rank_pct"].astype(float)
        ranked.loc[scored.index, "bucket"] = scored["bucket"]

    return ranked


def build_signals(
    *,
    predictions_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_name: str | None = None,
    label_name: str | None = None,
    split_name: str | None = None,
    horizon_days: int | None = None,
    split_roles: Iterable[str] = DEFAULT_SPLIT_ROLES,
    score_column: str | None = None,
    n_buckets: int = DEFAULT_N_BUCKETS,
    top_buckets: int = DEFAULT_TOP_BUCKETS,
    bottom_buckets: int = DEFAULT_BOTTOM_BUCKETS,
    center_classification_proba: bool = False,
    universe_history_path: str | Path | None = None,
    use_universe_filter: bool = True,
    run_id: str = MODULE_VERSION,
) -> BuildSignalsResult:
    logger = get_logger("signals.build_signals")

    n_buckets_int = int(n_buckets)
    top_buckets_int = int(top_buckets)
    bottom_buckets_int = int(bottom_buckets)
    if n_buckets_int < 2:
        raise ValueError("n_buckets must be >= 2.")
    if top_buckets_int < 1 or bottom_buckets_int < 1:
        raise ValueError("top_buckets and bottom_buckets must be >= 1.")
    if top_buckets_int + bottom_buckets_int > n_buckets_int:
        raise ValueError("top_buckets + bottom_buckets cannot exceed n_buckets.")

    selected_split_roles = _normalize_split_roles(split_roles)
    predictions_source = _resolve_predictions_path(predictions_path, model_name=model_name)
    frame = _load_predictions(predictions_source)
    inferred_model_name = _infer_model_name(predictions_source, model_name)
    score_series, signal_source_type, selected_score_col = _resolve_signal_source(
        frame,
        score_column=score_column,
        center_classification_proba=bool(center_classification_proba),
    )
    frame["raw_score"] = score_series

    filter_stats: dict[str, int] = {
        "n_rows_input": int(len(frame)),
    }

    frame = frame[frame["split_role"].isin(set(selected_split_roles))].copy()
    filter_stats["n_rows_after_split_role_filter"] = int(len(frame))
    filter_stats["n_rows_excluded_by_split_role"] = (
        filter_stats["n_rows_input"] - filter_stats["n_rows_after_split_role_filter"]
    )
    if frame.empty:
        raise ValueError(
            f"No prediction rows left after split_role filter: {list(selected_split_roles)}"
        )

    if label_name:
        frame = frame[frame["label_name"] == str(label_name)].copy()
    unique_labels = sorted(frame["label_name"].unique().tolist())
    if not unique_labels:
        raise ValueError("No rows left after label_name filtering.")
    if len(unique_labels) > 1:
        raise ValueError(
            "predictions include multiple label_name values after filtering. "
            "Pass --label-name explicitly. "
            f"Observed labels: {unique_labels}"
        )
    selected_label_name = str(unique_labels[0])

    if "split_name" in frame.columns:
        frame["split_name"] = frame["split_name"].astype(str)
        if split_name:
            frame = frame[frame["split_name"] == str(split_name)].copy()
        unique_split_names = sorted(frame["split_name"].unique().tolist())
        if not unique_split_names:
            raise ValueError("No rows left after split_name filtering.")
        if len(unique_split_names) > 1:
            raise ValueError(
                "predictions include multiple split_name values. Pass --split-name explicitly. "
                f"Observed split_name values: {unique_split_names}"
            )

    if "horizon_days" in frame.columns:
        frame["horizon_days"] = pd.to_numeric(frame["horizon_days"], errors="coerce").astype("Int64")
        if horizon_days is not None:
            frame = frame[frame["horizon_days"] == int(horizon_days)].copy()
        unique_horizons = sorted(frame["horizon_days"].dropna().astype(int).unique().tolist())
        if not unique_horizons:
            raise ValueError("No rows left after horizon_days filtering.")
        if len(unique_horizons) > 1:
            raise ValueError(
                "predictions include multiple horizon_days values. Pass --horizon-days explicitly. "
                f"Observed horizon_days: {unique_horizons}"
            )

    universe_source: Path | None = None
    filter_stats["n_rows_excluded_by_universe_filter"] = 0
    if use_universe_filter:
        eligible_universe, universe_source = _load_eligible_universe(universe_history_path)
        before = len(frame)
        frame = frame.merge(
            eligible_universe,
            on=["date", "instrument_id"],
            how="inner",
        )
        filter_stats["n_rows_excluded_by_universe_filter"] = int(before - len(frame))
        if frame.empty:
            raise ValueError(
                "No rows left after applying eligible universe filter (date, instrument_id)."
            )

    if frame.duplicated(["date", "instrument_id", "label_name", "split_role"], keep=False).any():
        raise ValueError(
            "Filtered predictions have duplicate (date, instrument_id, label_name, split_role) rows."
        )

    ranked = _assign_cross_sectional_ranks(frame, n_buckets=n_buckets_int)
    n_ranked = int(ranked["raw_score"].notna().sum())
    if n_ranked == 0:
        raise ValueError("No rankable rows: raw_score is null for all filtered rows.")

    top_threshold = n_buckets_int - top_buckets_int + 1
    ranked["is_top"] = ranked["bucket"].notna() & (ranked["bucket"] >= top_threshold)
    ranked["is_bottom"] = ranked["bucket"].notna() & (ranked["bucket"] <= bottom_buckets_int)

    ranked["signal_side"] = 0
    ranked.loc[ranked["is_top"], "signal_side"] = 1
    ranked.loc[ranked["is_bottom"], "signal_side"] = -1
    ranked["signal_side"] = ranked["signal_side"].astype(int)
    ranked["model_name"] = inferred_model_name
    ranked["signal_source_type"] = signal_source_type
    ranked["bucket_scheme"] = f"quantile_{n_buckets_int}_top{top_buckets_int}_bottom{bottom_buckets_int}"

    if ranked[ranked["rank_pct"].notna()]["rank_pct"].lt(0).any() or ranked[ranked["rank_pct"].notna()]["rank_pct"].gt(1).any():
        raise ValueError("rank_pct must be within [0, 1] for ranked rows.")
    if ranked[ranked["bucket"].notna()]["bucket"].lt(1).any() or ranked[ranked["bucket"].notna()]["bucket"].gt(n_buckets_int).any():
        raise ValueError(f"bucket values must be within [1, {n_buckets_int}] for ranked rows.")
    if not set(ranked["signal_side"].astype(int).unique().tolist()).issubset({-1, 0, 1}):
        raise ValueError("signal_side contains values outside {-1, 0, 1}.")

    if ranked.duplicated(["date", "instrument_id", "model_name", "label_name"], keep=False).any():
        raise ValueError("signals output has duplicate logical PK rows.")

    output_cols = [
        "date",
        "instrument_id",
        "ticker",
        "split_role",
        "label_name",
        "model_name",
        "signal_source_type",
        "raw_score",
        "rank_pct",
        "bucket",
        "signal_side",
        "is_top",
        "is_bottom",
        "n_names_ranked",
        "bucket_scheme",
    ]
    if "split_name" in ranked.columns:
        output_cols.insert(4, "split_name")
    if "horizon_days" in ranked.columns:
        insert_idx = output_cols.index("label_name")
        output_cols.insert(insert_idx, "horizon_days")

    signals = ranked[output_cols].sort_values(["date", "instrument_id"]).reset_index(drop=True)
    assert_schema(signals, SIGNALS_OUTPUT_SCHEMA)
    if signals.empty:
        raise ValueError("signals output is empty.")

    config_hash = _config_hash(
        {
            "version": MODULE_VERSION,
            "predictions_path": str(predictions_source),
            "universe_history_path": str(universe_source) if universe_source else None,
            "model_name": inferred_model_name,
            "label_name": selected_label_name,
            "split_roles": list(selected_split_roles),
            "score_column": selected_score_col,
            "signal_source_type": signal_source_type,
            "n_buckets": n_buckets_int,
            "top_buckets": top_buckets_int,
            "bottom_buckets": bottom_buckets_int,
            "center_classification_proba": bool(center_classification_proba),
            "use_universe_filter": bool(use_universe_filter),
        }
    )
    built_ts_utc = datetime.now(UTC).isoformat()
    signals["run_id"] = run_id
    signals["config_hash"] = config_hash
    signals["built_ts_utc"] = built_ts_utc

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (data_dir() / "signals")
    target_dir.mkdir(parents=True, exist_ok=True)
    signals_path = write_parquet(
        signals,
        target_dir / "signals_daily.parquet",
        schema_name=SIGNALS_OUTPUT_SCHEMA.name,
        run_id=run_id,
    )

    summary_payload = {
        "created_at_utc": built_ts_utc,
        "run_id": run_id,
        "config_hash": config_hash,
        "module_version": MODULE_VERSION,
        "predictions_path": str(predictions_source),
        "universe_history_path": str(universe_source) if universe_source else None,
        "model_name": inferred_model_name,
        "label_name": selected_label_name,
        "signal_source_type": signal_source_type,
        "score_column": selected_score_col,
        "split_roles_included": list(selected_split_roles),
        "bucket_scheme": f"quantile_{n_buckets_int}_top{top_buckets_int}_bottom{bottom_buckets_int}",
        "n_buckets": n_buckets_int,
        "top_buckets": top_buckets_int,
        "bottom_buckets": bottom_buckets_int,
        "n_rows_output": int(len(signals)),
        "n_dates": int(signals["date"].nunique()),
        "n_rows_ranked": n_ranked,
        "n_rows_nan_score": int(signals["raw_score"].isna().sum()),
        "n_top_signals": int(signals["is_top"].sum()),
        "n_bottom_signals": int(signals["is_bottom"].sum()),
        "avg_n_names_ranked_per_date": float(signals["n_names_ranked"].mean()),
        "filter_stats": filter_stats,
        "output_path": str(signals_path),
    }
    summary_path = target_dir / "signals_daily.summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "signals_built",
        run_id=run_id,
        model_name=inferred_model_name,
        label_name=selected_label_name,
        signal_source_type=signal_source_type,
        row_count=int(len(signals)),
        n_dates=int(signals["date"].nunique()),
        output_path=str(signals_path),
    )

    return BuildSignalsResult(
        signals_path=signals_path,
        summary_path=summary_path,
        row_count=int(len(signals)),
        n_dates=int(signals["date"].nunique()),
        model_name=inferred_model_name,
        label_name=selected_label_name,
        signal_source_type=signal_source_type,
        config_hash=config_hash,
    )


def _parse_csv_strings(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return values


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build daily cross-sectional signals from model predictions.")
    parser.add_argument("--predictions-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--label-name", type=str, default=None)
    parser.add_argument("--split-name", type=str, default=None)
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument("--split-roles", type=_parse_csv_strings, default=DEFAULT_SPLIT_ROLES)
    parser.add_argument("--score-column", type=str, default=None)
    parser.add_argument("--n-buckets", type=int, default=DEFAULT_N_BUCKETS)
    parser.add_argument("--top-buckets", type=int, default=DEFAULT_TOP_BUCKETS)
    parser.add_argument("--bottom-buckets", type=int, default=DEFAULT_BOTTOM_BUCKETS)
    parser.add_argument("--center-classification-proba", action="store_true")
    parser.add_argument("--universe-history-path", type=str, default=None)
    parser.add_argument("--disable-universe-filter", action="store_true")
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = build_signals(
        predictions_path=args.predictions_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        label_name=args.label_name,
        split_name=args.split_name,
        horizon_days=args.horizon_days,
        split_roles=args.split_roles,
        score_column=args.score_column,
        n_buckets=args.n_buckets,
        top_buckets=args.top_buckets,
        bottom_buckets=args.bottom_buckets,
        center_classification_proba=bool(args.center_classification_proba),
        universe_history_path=args.universe_history_path,
        use_universe_filter=not bool(args.disable_universe_filter),
        run_id=args.run_id,
    )
    print("Signals built:")
    print(f"- path: {result.signals_path}")
    print(f"- summary: {result.summary_path}")
    print(f"- row_count: {result.row_count}")
    print(f"- n_dates: {result.n_dates}")
    print(f"- model_name: {result.model_name}")
    print(f"- label_name: {result.label_name}")
    print(f"- signal_source_type: {result.signal_source_type}")


if __name__ == "__main__":
    main()
