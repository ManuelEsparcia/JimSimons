from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.edgar.point_in_time import build_fundamentals_pit
from data.edgar.ticker_cik import build_ticker_cik_map
from data.price.adjust_prices import adjust_prices
from data.price.fetch_prices import fetch_prices
from data.price.market_proxies import build_market_proxies
from data.reference.build_reference import build_reference_data
from data.universe.build_universe import build_universe
from data.universe.corporate_actions import build_corporate_actions
from datasets.build_model_dataset import build_model_dataset
from features.build_features import build_features
from labels.build_labels import build_labels
from labels.purged_splits import build_purged_splits
from simons_core.io.parquet_store import read_parquet
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODEL_DATASET_MIN_SCHEMA = DataSchema(
    name="model_dataset_min_test",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("horizon_days", "int64", nullable=False),
        ColumnSpec("label_name", "string", nullable=False),
        ColumnSpec("split_name", "string", nullable=False),
        ColumnSpec("split_role", "string", nullable=False),
        ColumnSpec("entry_date", "datetime64", nullable=False),
        ColumnSpec("exit_date", "datetime64", nullable=False),
        ColumnSpec("target_value", "float64", nullable=False),
        ColumnSpec("target_type", "string", nullable=False),
    ),
    primary_key=("date", "instrument_id", "horizon_days", "label_name"),
    allow_extra_columns=True,
)


def _build_model_dataset_inputs(tmp_workspace: dict[str, Path]) -> dict[str, Path]:
    reference_root = tmp_workspace["data"] / "reference"
    universe_root = tmp_workspace["data"] / "universe"
    price_root = tmp_workspace["data"] / "price"
    edgar_root = tmp_workspace["data"] / "edgar"
    labels_root = tmp_workspace["data"] / "labels"
    features_root = tmp_workspace["data"] / "features"
    datasets_root = tmp_workspace["data"] / "datasets"

    build_reference_data(output_dir=reference_root, run_id="test_reference_model_dataset")
    universe_result = build_universe(
        reference_root=reference_root,
        output_dir=universe_root,
        run_id="test_universe_model_dataset",
    )
    corporate_actions_result = build_corporate_actions(
        reference_root=reference_root,
        universe_history_path=universe_result.universe_history,
        output_dir=universe_root,
        run_id="test_corporate_actions_model_dataset",
    )
    raw_result = fetch_prices(
        reference_root=reference_root,
        universe_history_path=universe_result.universe_history,
        output_dir=price_root,
        run_id="test_fetch_model_dataset",
    )
    adjusted_result = adjust_prices(
        raw_prices_path=raw_result.raw_prices_path,
        corporate_actions_path=corporate_actions_result.corporate_actions_path,
        output_dir=price_root,
        run_id="test_adjust_model_dataset",
    )
    labels_result = build_labels(
        adjusted_prices_path=adjusted_result.adjusted_prices_path,
        universe_history_path=universe_result.universe_history,
        trading_calendar_path=reference_root / "trading_calendar.parquet",
        output_dir=labels_root,
        horizons=(1, 5, 20),
        decision_lag=1,
        run_id="test_labels_model_dataset",
    )
    splits_result = build_purged_splits(
        labels_path=labels_result.labels_path,
        trading_calendar_path=reference_root / "trading_calendar.parquet",
        output_dir=labels_root,
        valid_fraction=0.2,
        test_fraction=0.2,
        embargo_sessions=1,
        run_id="test_splits_model_dataset",
    )
    market_result = build_market_proxies(
        adjusted_prices_path=adjusted_result.adjusted_prices_path,
        universe_history_path=universe_result.universe_history,
        trading_calendar_path=reference_root / "trading_calendar.parquet",
        output_dir=price_root,
        run_id="test_market_model_dataset",
    )
    ticker_cik_result = build_ticker_cik_map(
        reference_root=reference_root,
        universe_history_path=universe_result.universe_history,
        output_dir=edgar_root,
        run_id="test_ticker_cik_model_dataset",
    )
    fundamentals_result = build_fundamentals_pit(
        ticker_cik_map_path=ticker_cik_result.ticker_cik_map_path,
        universe_history_path=universe_result.universe_history,
        submissions_raw_path=edgar_root / "submissions_raw.parquet",
        companyfacts_raw_path=edgar_root / "companyfacts_raw.parquet",
        output_dir=edgar_root,
        run_id="test_fundamentals_model_dataset",
    )
    features_result = build_features(
        adjusted_prices_path=adjusted_result.adjusted_prices_path,
        universe_history_path=universe_result.universe_history,
        market_proxies_path=market_result.market_proxies_path,
        fundamentals_pit_path=fundamentals_result.fundamentals_pit_path,
        trading_calendar_path=reference_root / "trading_calendar.parquet",
        output_dir=features_root,
        run_id="test_features_model_dataset",
    )

    return {
        "features": features_result.features_path,
        "labels": labels_result.labels_path,
        "splits": splits_result.splits_path,
        "datasets_root": datasets_root,
    }


def test_build_model_dataset_generates_non_empty_schema_valid_output(
    tmp_workspace: dict[str, Path],
) -> None:
    paths = _build_model_dataset_inputs(tmp_workspace)
    result = build_model_dataset(
        features_path=paths["features"],
        labels_path=paths["labels"],
        purged_splits_path=paths["splits"],
        output_dir=paths["datasets_root"],
        label_names=("fwd_ret_5d",),
        run_id="test_build_model_dataset_mvp",
    )

    assert result.dataset_path.exists()
    assert result.summary_path.exists()
    assert (result.dataset_path.with_suffix(".parquet.manifest.json")).exists()
    assert result.row_count > 0
    assert result.n_features > 0
    assert "fwd_ret_5d" in result.selected_label_names

    dataset = read_parquet(result.dataset_path)
    assert_schema(dataset, MODEL_DATASET_MIN_SCHEMA)
    assert len(dataset) > 0
    assert not dataset.duplicated(["date", "instrument_id", "horizon_days", "label_name"]).any()
    assert set(dataset["split_role"].astype(str).unique().tolist()).issubset(
        {"train", "valid", "test", "dropped_by_purge", "dropped_by_embargo"}
    )

    labels = read_parquet(paths["labels"]).copy()
    labels = labels[labels["label_name"].astype(str) == "fwd_ret_5d"].copy()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    labels["instrument_id"] = labels["instrument_id"].astype(str)
    labels["horizon_days"] = pd.to_numeric(labels["horizon_days"], errors="coerce").astype("int64")
    labels["label_name"] = labels["label_name"].astype(str)
    labels["label_value"] = pd.to_numeric(labels["label_value"], errors="coerce")

    compare = dataset.merge(
        labels[["date", "instrument_id", "horizon_days", "label_name", "label_value"]],
        on=["date", "instrument_id", "horizon_days", "label_name"],
        how="left",
    )
    assert compare["label_value"].notna().all()
    assert np.allclose(compare["target_value"], compare["label_value"])

    expected_feature_cols = {
        "ret_1d_lag1",
        "ret_5d_lag1",
        "ret_20d_lag1",
        "vol_20d",
        "mkt_breadth_up_lag1",
        "log_total_assets",
    }
    assert expected_feature_cols.issubset(set(dataset.columns))

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    required_summary = {
        "label_name_selected",
        "horizon_days_present",
        "n_rows_total",
        "n_train",
        "n_valid",
        "n_test",
        "n_dropped_by_purge",
        "n_dropped_by_embargo",
        "n_features",
        "feature_names",
        "pct_missing_by_feature",
        "target_missing_rate",
        "join_drop_counts",
    }
    assert required_summary.issubset(summary.keys())
    assert summary["label_name_selected"] == ["fwd_ret_5d"]
    assert summary["n_rows_total"] == len(dataset)


def test_build_model_dataset_small_case_join_and_split_role_propagation(
    tmp_workspace: dict[str, Path],
) -> None:
    base = tmp_workspace["data"] / "model_dataset_small_case"
    features_path = base / "features_matrix.parquet"
    labels_path = base / "labels_forward.parquet"
    splits_path = base / "purged_splits.parquet"
    output_dir = base / "output"
    base.mkdir(parents=True, exist_ok=True)

    features = pd.DataFrame(
        [
            {"date": "2026-01-05", "instrument_id": "SIMA", "ticker": "AAA", "ret_1d_lag1": 0.01, "vol_5d": 0.02},
            {"date": "2026-01-06", "instrument_id": "SIMA", "ticker": "AAA", "ret_1d_lag1": 0.02, "vol_5d": 0.03},
            {"date": "2026-01-07", "instrument_id": "SIMA", "ticker": "AAA", "ret_1d_lag1": 0.03, "vol_5d": np.nan},
        ]
    )
    features["date"] = pd.to_datetime(features["date"])
    features.to_parquet(features_path, index=False)

    labels = pd.DataFrame(
        [
            {
                "date": "2026-01-05",
                "instrument_id": "SIMA",
                "ticker": "AAA",
                "horizon_days": 5,
                "entry_date": "2026-01-06",
                "exit_date": "2026-01-13",
                "label_name": "fwd_ret_5d",
                "label_value": 0.11,
                "price_entry": 100.0,
                "price_exit": 111.0,
                "source_price_field": "close_adj",
            },
            {
                "date": "2026-01-06",
                "instrument_id": "SIMA",
                "ticker": "AAA",
                "horizon_days": 5,
                "entry_date": "2026-01-07",
                "exit_date": "2026-01-14",
                "label_name": "fwd_ret_5d",
                "label_value": -0.05,
                "price_entry": 100.0,
                "price_exit": 95.0,
                "source_price_field": "close_adj",
            },
            {
                "date": "2026-01-07",
                "instrument_id": "SIMA",
                "ticker": "AAA",
                "horizon_days": 5,
                "entry_date": "2026-01-08",
                "exit_date": "2026-01-15",
                "label_name": "fwd_ret_5d",
                "label_value": 0.02,
                "price_entry": 100.0,
                "price_exit": 102.0,
                "source_price_field": "close_adj",
            },
            {
                # Intentionally no feature row for this date to force join drop count.
                "date": "2026-01-08",
                "instrument_id": "SIMA",
                "ticker": "AAA",
                "horizon_days": 5,
                "entry_date": "2026-01-09",
                "exit_date": "2026-01-16",
                "label_name": "fwd_ret_5d",
                "label_value": 0.03,
                "price_entry": 100.0,
                "price_exit": 103.0,
                "source_price_field": "close_adj",
            },
        ]
    )
    for col in ("date", "entry_date", "exit_date"):
        labels[col] = pd.to_datetime(labels[col])
    labels.to_parquet(labels_path, index=False)

    splits = pd.DataFrame(
        [
            {
                "date": "2026-01-05",
                "instrument_id": "SIMA",
                "horizon_days": 5,
                "label_name": "fwd_ret_5d",
                "split_name": "holdout_temporal_purged",
                "split_role": "train",
                "entry_date": "2026-01-06",
                "exit_date": "2026-01-13",
            },
            {
                "date": "2026-01-06",
                "instrument_id": "SIMA",
                "horizon_days": 5,
                "label_name": "fwd_ret_5d",
                "split_name": "holdout_temporal_purged",
                "split_role": "dropped_by_purge",
                "entry_date": "2026-01-07",
                "exit_date": "2026-01-14",
            },
            {
                "date": "2026-01-07",
                "instrument_id": "SIMA",
                "horizon_days": 5,
                "label_name": "fwd_ret_5d",
                "split_name": "holdout_temporal_purged",
                "split_role": "test",
                "entry_date": "2026-01-08",
                "exit_date": "2026-01-15",
            },
            {
                "date": "2026-01-08",
                "instrument_id": "SIMA",
                "horizon_days": 5,
                "label_name": "fwd_ret_5d",
                "split_name": "holdout_temporal_purged",
                "split_role": "dropped_by_embargo",
                "entry_date": "2026-01-09",
                "exit_date": "2026-01-16",
            },
        ]
    )
    for col in ("date", "entry_date", "exit_date"):
        splits[col] = pd.to_datetime(splits[col])
    splits.to_parquet(splits_path, index=False)

    result = build_model_dataset(
        features_path=features_path,
        labels_path=labels_path,
        purged_splits_path=splits_path,
        output_dir=output_dir,
        label_names=("fwd_ret_5d",),
        run_id="test_build_model_dataset_small_case",
    )
    dataset = read_parquet(result.dataset_path).sort_values(["date"]).reset_index(drop=True)
    assert len(dataset) == 3  # one row dropped by join due missing features
    assert set(dataset["split_role"].tolist()) == {"train", "dropped_by_purge", "test"}

    row_train = dataset.loc[dataset["split_role"] == "train"].iloc[0]
    assert np.isclose(float(row_train["target_value"]), 0.11)
    assert np.isclose(float(row_train["ret_1d_lag1"]), 0.01)
    assert np.isclose(float(row_train["vol_5d"]), 0.02)

    row_drop = dataset.loc[dataset["split_role"] == "dropped_by_purge"].iloc[0]
    assert np.isclose(float(row_drop["target_value"]), -0.05)
    assert np.isclose(float(row_drop["ret_1d_lag1"]), 0.02)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["join_drop_counts"]["label_split_without_features"] == 1
    assert summary["n_dropped_by_purge"] == 1
    assert summary["n_dropped_by_embargo"] == 0  # embargo row existed but was dropped by join

