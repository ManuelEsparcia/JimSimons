from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_week5_signals_paper import run_week5_signals_paper
from simons_core.io.parquet_store import read_parquet


def _seed_week5_prerequisites(data_root: Path) -> dict[str, Path]:
    models_root = data_root / "models" / "artifacts"
    labels_root = data_root / "labels"
    universe_root = data_root / "universe"
    reference_root = data_root / "reference"
    price_root = data_root / "price"
    edgar_root = data_root / "edgar"
    features_root = data_root / "features"
    datasets_root = data_root / "datasets"
    for directory in (
        models_root,
        labels_root,
        universe_root,
        reference_root,
        price_root,
        edgar_root,
        features_root,
        datasets_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    d1 = pd.Timestamp("2026-04-06")
    d2 = pd.Timestamp("2026-04-07")
    sessions = pd.bdate_range("2026-04-01", periods=10, freq="B")

    trading_calendar_path = reference_root / "trading_calendar.parquet"
    pd.DataFrame({"date": sessions, "is_session": True}).to_parquet(trading_calendar_path, index=False)

    names = [
        ("SIMA", "AAA"),
        ("SIMB", "BBB"),
        ("SIMC", "CCC"),
        ("SIMD", "DDD"),
        ("SIME", "EEE"),
    ]
    universe_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    per_date_values = {
        d1: {
            "pred": [0.90, 0.55, 0.15, -0.10, -0.35],
            "ret": [0.05, 0.03, 0.00, -0.02, -0.04],
        },
        d2: {
            "pred": [0.80, 0.45, 0.10, -0.15, -0.30],
            "ret": [0.04, 0.02, -0.01, -0.03, -0.05],
        },
    }

    for date in (d1, d2):
        values = per_date_values[date]
        for idx, (instrument_id, ticker) in enumerate(names):
            universe_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "is_eligible": True,
                }
            )
            label_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "horizon_days": 5,
                    "entry_date": date + pd.Timedelta(days=1),
                    "exit_date": date + pd.Timedelta(days=6),
                    "label_name": "fwd_ret_5d",
                    "label_value": float(values["ret"][idx]),
                    "price_entry": 100.0,
                    "price_exit": 101.0,
                    "source_price_field": "close_adj",
                }
            )
            prediction_rows.append(
                {
                    "date": date,
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "split_name": "holdout_temporal_purged",
                    "split_role": "test",
                    "horizon_days": 5,
                    "label_name": "fwd_ret_5d",
                    "prediction": float(values["pred"][idx]),
                }
            )

    # Extra row in train role to verify split-role filtering.
    prediction_rows.append(
        {
            "date": d1,
            "instrument_id": "SIMZ",
            "ticker": "ZZZ",
            "split_name": "holdout_temporal_purged",
            "split_role": "train",
            "horizon_days": 5,
            "label_name": "fwd_ret_5d",
            "prediction": 0.99,
        }
    )

    predictions_path = models_root / "ridge_baseline_predictions.parquet"
    labels_path = labels_root / "labels_forward.parquet"
    universe_path = universe_root / "universe_history.parquet"
    pd.DataFrame(prediction_rows).to_parquet(predictions_path, index=False)
    pd.DataFrame(label_rows).to_parquet(labels_path, index=False)
    pd.DataFrame(universe_rows).to_parquet(universe_path, index=False)

    # Compatibility placeholders from Week 1-4.
    adjusted_prices_path = price_root / "adjusted_prices.parquet"
    fundamentals_pit_path = edgar_root / "fundamentals_pit.parquet"
    features_matrix_path = features_root / "features_matrix.parquet"
    model_dataset_path = datasets_root / "model_dataset.parquet"
    purged_cv_folds_path = labels_root / "purged_cv_folds.parquet"
    pd.DataFrame(
        {
            "date": [d1],
            "instrument_id": ["SIMA"],
            "ticker": ["AAA"],
            "close_adj": [100.0],
            "volume_adj": [1_000_000.0],
        }
    ).to_parquet(adjusted_prices_path, index=False)
    pd.DataFrame(
        {
            "instrument_id": ["SIMA"],
            "asof_date": [d1],
            "metric_name": ["Revenues"],
            "metric_value": [100_000_000.0],
        }
    ).to_parquet(fundamentals_pit_path, index=False)
    pd.DataFrame(
        {
            "date": [d1],
            "instrument_id": ["SIMA"],
            "ticker": ["AAA"],
            "ret_1d_lag1": [0.01],
        }
    ).to_parquet(features_matrix_path, index=False)
    pd.DataFrame(
        {
            "date": [d1],
            "instrument_id": ["SIMA"],
            "horizon_days": [5],
            "label_name": ["fwd_ret_5d"],
            "split_role": ["train"],
            "target_value": [0.01],
            "f1": [0.1],
        }
    ).to_parquet(model_dataset_path, index=False)
    pd.DataFrame(
        {
            "fold_id": [0],
            "date": [d1],
            "instrument_id": ["SIMA"],
            "horizon_days": [5],
            "label_name": ["fwd_ret_5d"],
            "split_role": ["train"],
            "entry_date": [d1 + pd.Timedelta(days=1)],
            "exit_date": [d1 + pd.Timedelta(days=6)],
        }
    ).to_parquet(purged_cv_folds_path, index=False)

    return {
        "predictions": predictions_path,
        "labels_forward": labels_path,
        "universe_history": universe_path,
        "trading_calendar": trading_calendar_path,
        "adjusted_prices": adjusted_prices_path,
        "fundamentals_pit": fundamentals_pit_path,
        "features_matrix": features_matrix_path,
        "model_dataset": model_dataset_path,
        "purged_cv_folds": purged_cv_folds_path,
    }


def test_week5_runner_end_to_end_smoke(tmp_workspace: dict[str, Path]) -> None:
    prereq = _seed_week5_prerequisites(tmp_workspace["data"])
    compatibility_counts_before = {
        key: int(len(read_parquet(path)))
        for key, path in prereq.items()
        if key in {"adjusted_prices", "fundamentals_pit", "features_matrix", "model_dataset", "purged_cv_folds"}
    }

    result = run_week5_signals_paper(
        run_prefix="test_week5_e2e",
        data_root=tmp_workspace["data"],
        model_name="ridge_baseline",
        label_name="fwd_ret_5d",
        split_name="holdout_temporal_purged",
        horizon_days=5,
        split_roles=("test",),
        n_buckets=5,
        top_buckets=1,
        bottom_buckets=1,
        portfolio_modes=("long_only_top", "long_short_top_bottom"),
    )

    expected_artifacts = {
        "signals_daily": result.artifacts["signals_daily"],
        "signals_summary": result.artifacts["signals_summary"],
        "decile_daily": result.artifacts["decile_daily"],
        "decile_summary": result.artifacts["decile_summary"],
        "decile_analysis_summary": result.artifacts["decile_analysis_summary"],
        "paper_daily": result.artifacts["paper_portfolio_daily"],
        "paper_positions": result.artifacts["paper_portfolio_positions"],
        "paper_summary": result.artifacts["paper_portfolio_summary"],
    }
    for path in expected_artifacts.values():
        assert path.exists()
        assert path.stat().st_size > 0

    signals = read_parquet(expected_artifacts["signals_daily"])
    assert len(signals) > 0
    assert set(signals["split_role"].astype(str).unique().tolist()) == {"test"}
    assert signals[signals["rank_pct"].notna()]["rank_pct"].between(0.0, 1.0).all()
    assert signals[signals["bucket"].notna()]["bucket"].between(1, 5).all()

    decile_daily = read_parquet(expected_artifacts["decile_daily"])
    decile_summary = read_parquet(expected_artifacts["decile_summary"])
    decile_json = json.loads(expected_artifacts["decile_analysis_summary"].read_text(encoding="utf-8"))
    assert len(decile_daily) > 0
    assert len(decile_summary) > 0
    assert "top_minus_bottom_spread" in decile_daily.columns
    assert decile_daily["top_minus_bottom_spread"].notna().any()
    assert decile_json["mean_top_minus_bottom_spread"] is not None

    paper_daily = read_parquet(expected_artifacts["paper_daily"])
    paper_positions = read_parquet(expected_artifacts["paper_positions"])
    paper_json = json.loads(expected_artifacts["paper_summary"].read_text(encoding="utf-8"))
    assert len(paper_daily) > 0
    assert len(paper_positions) > 0
    assert set(paper_positions["split_role"].astype(str).unique().tolist()) == {"test"}
    assert set(paper_daily["portfolio_mode"].astype(str).unique().tolist()) == {
        "long_only_top",
        "long_short_top_bottom",
    }

    executable = paper_daily[paper_daily["is_executable"]].copy()
    assert len(executable) > 0
    contribution_sum = (
        paper_positions.groupby(["date", "portfolio_mode"], as_index=False)["contribution"]
        .sum()
        .rename(columns={"contribution": "contrib_sum"})
    )
    check = executable.merge(contribution_sum, on=["date", "portfolio_mode"], how="left")
    assert np.allclose(
        check["contrib_sum"].to_numpy(dtype=float),
        check["gross_portfolio_return"].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
    )

    weights = (
        paper_positions.groupby(["date", "portfolio_mode", "side"], as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "weight_sum"})
    )
    for row in executable.itertuples(index=False):
        day = weights[(weights["date"] == row.date) & (weights["portfolio_mode"] == row.portfolio_mode)]
        long_sum = float(day.loc[day["side"] == "long", "weight_sum"].sum())
        short_sum = float(day.loc[day["side"] == "short", "weight_sum"].sum())
        if row.portfolio_mode == "long_only_top":
            assert np.isclose(long_sum, 1.0, atol=1e-12)
            assert np.isclose(short_sum, 0.0, atol=1e-12)
        elif row.portfolio_mode == "long_short_top_bottom":
            assert np.isclose(long_sum, 1.0, atol=1e-12)
            assert np.isclose(short_sum, -1.0, atol=1e-12)

    assert len(paper_json["mode_summaries"]) == 2
    mode_names = {item["portfolio_mode"] for item in paper_json["mode_summaries"]}
    assert mode_names == {"long_only_top", "long_short_top_bottom"}

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_prefix"] == "test_week5_e2e"
    assert manifest["statuses"]["build_signals"] == "DONE"
    assert manifest["statuses"]["decile_analysis"] == "DONE"
    assert manifest["statuses"]["paper_portfolio"] == "DONE"

    compatibility_counts_after = {
        key: int(len(read_parquet(path)))
        for key, path in prereq.items()
        if key in compatibility_counts_before
    }
    assert compatibility_counts_after == compatibility_counts_before
