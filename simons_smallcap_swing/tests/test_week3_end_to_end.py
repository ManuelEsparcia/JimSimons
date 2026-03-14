from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_week3_modeling_baselines import run_week3_modeling_baselines
from simons_core.io.parquet_store import read_parquet


def _seed_week3_prerequisites(data_root: Path) -> dict[str, Path]:
    reference_root = data_root / "reference"
    universe_root = data_root / "universe"
    price_root = data_root / "price"
    edgar_root = data_root / "edgar"
    for directory in (reference_root, universe_root, price_root, edgar_root):
        directory.mkdir(parents=True, exist_ok=True)

    sessions = pd.bdate_range("2026-01-05", "2026-04-30", freq="B")
    calendar = pd.DataFrame({"date": sessions, "is_session": True})
    trading_calendar_path = reference_root / "trading_calendar.parquet"
    calendar.to_parquet(trading_calendar_path, index=False)

    instruments = [
        ("SIMA", "AAA"),
        ("SIMB", "BBB"),
    ]

    universe_rows: list[dict[str, object]] = []
    price_rows: list[dict[str, object]] = []
    market_rows: list[dict[str, object]] = []
    for i, dt in enumerate(sessions):
        market_rows.append(
            {
                "date": dt,
                "breadth_up": 0.60 if i % 2 == 0 else 0.40,
                "equal_weight_return": 0.0010 if i % 2 == 0 else -0.0005,
                "cross_sectional_vol": 0.020 + (0.0001 * (i % 5)),
                "coverage_ratio": 1.0,
                "n_names": 2,
                "n_names_with_prices": 2,
            }
        )
        for instrument_id, ticker in instruments:
            universe_rows.append(
                {
                    "date": dt,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "is_eligible": True,
                }
            )
            if instrument_id == "SIMA":
                close = 100.0 + (0.8 * i)
                volume = 1_000_000 + (1_000 * i)
            else:
                close = 180.0 - (0.5 * i)
                volume = 900_000 + (500 * i)

            price_rows.append(
                {
                    "date": dt,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "open_adj": close * 0.995,
                    "high_adj": close * 1.010,
                    "low_adj": close * 0.990,
                    "close_adj": close,
                    "volume_adj": float(volume),
                }
            )

    universe_history_path = universe_root / "universe_history.parquet"
    adjusted_prices_path = price_root / "adjusted_prices.parquet"
    market_proxies_path = price_root / "market_proxies.parquet"
    pd.DataFrame(universe_rows).to_parquet(universe_history_path, index=False)
    pd.DataFrame(price_rows).to_parquet(adjusted_prices_path, index=False)
    pd.DataFrame(market_rows).to_parquet(market_proxies_path, index=False)

    fundamentals_rows: list[dict[str, object]] = []
    asof_dates = [sessions[0], sessions[25]]
    metric_templates = [
        ("Revenues", [120_000_000.0, 130_000_000.0]),
        ("NetIncomeLoss", [11_500_000.0, 12_800_000.0]),
        ("Assets", [260_000_000.0, 275_000_000.0]),
        ("EntityCommonStockSharesOutstanding", [25_000_000.0, 25_800_000.0]),
    ]
    for instrument_id, _ticker in instruments:
        multiplier = 1.0 if instrument_id == "SIMA" else 0.9
        for asof_idx, asof_date in enumerate(asof_dates):
            for metric_name, values in metric_templates:
                fundamentals_rows.append(
                    {
                        "instrument_id": instrument_id,
                        "asof_date": asof_date,
                        "metric_name": metric_name,
                        "metric_value": float(values[asof_idx] * multiplier),
                    }
                )

    fundamentals_pit_path = edgar_root / "fundamentals_pit.parquet"
    pd.DataFrame(fundamentals_rows).to_parquet(fundamentals_pit_path, index=False)

    return {
        "trading_calendar": trading_calendar_path,
        "universe_history": universe_history_path,
        "adjusted_prices": adjusted_prices_path,
        "market_proxies": market_proxies_path,
        "fundamentals_pit": fundamentals_pit_path,
    }


def test_week3_runner_end_to_end_smoke(tmp_workspace: dict[str, Path]) -> None:
    prereq = _seed_week3_prerequisites(tmp_workspace["data"])

    result = run_week3_modeling_baselines(
        run_prefix="test_week3_e2e",
        data_root=tmp_workspace["data"],
    )

    expected_artifacts = {
        "labels_forward": result.artifacts["labels_forward"],
        "features_matrix": result.artifacts["features_matrix"],
        "purged_splits": result.artifacts["purged_splits"],
        "model_dataset_regression": result.artifacts["model_dataset_regression"],
        "model_dataset_classification": result.artifacts["model_dataset_classification"],
        "ridge_metrics": result.artifacts["ridge_metrics"],
        "ridge_predictions": result.artifacts["ridge_predictions"],
        "logistic_metrics": result.artifacts["logistic_metrics"],
        "logistic_predictions": result.artifacts["logistic_predictions"],
        "dummy_regressor_metrics": result.artifacts["dummy_regressor_metrics"],
        "dummy_regressor_predictions": result.artifacts["dummy_regressor_predictions"],
        "dummy_classifier_metrics": result.artifacts["dummy_classifier_metrics"],
        "dummy_classifier_predictions": result.artifacts["dummy_classifier_predictions"],
        "baseline_benchmark_summary": result.artifacts["baseline_benchmark_summary"],
        "baseline_benchmark_table": result.artifacts["baseline_benchmark_table"],
    }
    for path in expected_artifacts.values():
        assert path.exists()
        assert path.stat().st_size > 0

    labels = read_parquet(expected_artifacts["labels_forward"])
    features = read_parquet(expected_artifacts["features_matrix"])
    splits = read_parquet(expected_artifacts["purged_splits"])
    dataset_reg = read_parquet(expected_artifacts["model_dataset_regression"])
    dataset_cls = read_parquet(expected_artifacts["model_dataset_classification"])

    assert len(labels) > 0
    assert len(features) > 0
    assert len(splits) > 0
    assert len(dataset_reg) > 0
    assert len(dataset_cls) > 0

    for prediction_key in (
        "ridge_predictions",
        "logistic_predictions",
        "dummy_regressor_predictions",
        "dummy_classifier_predictions",
    ):
        prediction = read_parquet(expected_artifacts[prediction_key])
        roles = set(prediction["split_role"].astype(str).unique().tolist())
        assert roles.issubset({"train", "valid", "test"})
        assert "dropped_by_purge" not in roles
        assert "dropped_by_embargo" not in roles

    benchmark_summary = json.loads(
        expected_artifacts["baseline_benchmark_summary"].read_text(encoding="utf-8")
    )
    assert set(benchmark_summary["tasks_compared"]) == {
        "regression_baselines",
        "classification_baselines",
    }
    assert benchmark_summary["regression"]["comparability_status"] in {
        "comparable",
        "non_comparable",
    }
    assert benchmark_summary["classification"]["comparability_status"] in {
        "comparable",
        "non_comparable",
    }

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_prefix"] == "test_week3_e2e"
    assert manifest["statuses"]["labels"] == "DONE"
    assert manifest["statuses"]["features"] == "DONE"
    assert manifest["statuses"]["purged_splits"] == "DONE"
    assert manifest["statuses"]["train_ridge"] == "DONE"
    assert manifest["statuses"]["train_dummy_regressor"] == "DONE"
    assert manifest["statuses"]["baseline_benchmark"] == "DONE"

    # Compatibility check with Week 1/2 artifact expectations.
    for key, path in prereq.items():
        assert Path(manifest["prerequisites"][key]) == path
        assert path.exists()

