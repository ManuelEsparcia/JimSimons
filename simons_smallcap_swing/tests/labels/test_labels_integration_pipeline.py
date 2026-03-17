from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(module_name: str):
    candidates = [
        f"simons_smallcap_swing.labels.{module_name}",
        f"labels.{module_name}",
        module_name,
    ]
    last_exc = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:  # pragma: no cover - import fallback path
            last_exc = exc
    raise last_exc  # type: ignore[misc]


bl = _load_module("build_labels")
nt = _load_module("neutralized_targets")
lq = _load_module("label_qc")
ps = _load_module("purged_splits")

InputPaths = bl.InputPaths
build_labels = bl.build_labels
load_table = bl.load_table

NeutralizationConfig = nt.NeutralizationConfig
run_neutralized_targets = nt.run_neutralized_targets

QCConfig = lq.QCConfig
run_label_qc = lq.run_label_qc

SplitConfig = ps.SplitConfig
build_purged_splits = ps.build_purged_splits


def _write_csv(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def _write_json(payload: dict[str, object], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _make_market_bundle(
    *,
    n_dates: int = 36,
    n_symbols: int = 30,
    horizons: tuple[int, ...] = (5, 10),
    extra_calendar_dates: int = 12,
) -> dict[str, pd.DataFrame]:
    """Synthetic PIT market bundle aligned with the real build_labels contract.

    We intentionally give the trading calendar some extra tail coverage so
    `purged_splits` can validate event windows against a master calendar even
    near the end of the label panel.
    """
    label_dates = pd.bdate_range("2024-01-02", periods=n_dates)
    full_calendar = pd.bdate_range("2024-01-02", periods=n_dates + extra_calendar_dates)
    symbols = [f"S{i:03d}" for i in range(n_symbols)]

    price_rows: list[dict[str, object]] = []
    universe_rows: list[dict[str, object]] = []
    exposure_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []

    for sym_idx, symbol in enumerate(symbols):
        beta = 0.60 + 0.03 * sym_idx
        market_cap = 100_000_000.0 + 2_000_000.0 * sym_idx
        liquidity = 1_000_000.0 + 25_000.0 * sym_idx
        sector = ("tech", "health", "industrial")[sym_idx % 3]

        close = 10.0 + 0.15 * sym_idx
        for t_idx, date in enumerate(label_dates):
            daily_ret = -0.0012 + 0.00012 * sym_idx + 0.00004 * ((t_idx % 4) - 1.5)
            close = close * (1.0 + daily_ret)
            open_px = close * (1.0 - 0.00035)
            vwap_px = (2.0 * open_px + close) / 3.0

            price_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "adj_open": open_px,
                    "adj_close": close,
                    "adj_vwap": vwap_px,
                }
            )
            universe_rows.append({"date": date, "symbol": symbol, "is_eligible": True})
            exposure_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "sector": sector,
                    "market_beta": beta,
                    "market_cap": market_cap,
                    "liquidity_proxy": liquidity,
                }
            )
            for h in horizons:
                cost_rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "horizon_days": h,
                        "entry_cost": 0.0004,
                        "exit_cost": 0.0004,
                        "carry_cost": 0.00001 * h,
                        "cost_mean": 0.0008 + 0.00001 * h,
                        "cost_var": 1e-8,
                    }
                )

    return {
        "calendar": pd.DataFrame({"date": full_calendar}),
        "prices": pd.DataFrame(price_rows),
        "universe": pd.DataFrame(universe_rows),
        "exposures": pd.DataFrame(exposure_rows),
        "costs": pd.DataFrame(cost_rows),
    }


def _config_mapping(
    *,
    horizons: tuple[int, ...] = (5, 10),
    min_valid_cross_section: int = 10,
) -> dict[str, object]:
    max_h = max(horizons)
    return {
        "horizons": list(horizons),
        "return_mode": "close_to_close",
        "decision_lag": 1,
        "net_of_costs": True,
        "primary_target": f"y_fwd_ret_net_{max_h}d",
        "missing_cost_policy": "strict_invalidate",
        "neutralization_mode": "sector_beta_size_liquidity",
        "min_valid_cross_section": min_valid_cross_section,
        "neutralization_policy": {
            "n_min_neut": min_valid_cross_section,
            "fixed_lambda": 1.0,
            "lambda_selection": "fixed",
            "residual_var_floor": 1e-14,
        },
        "cost_policy": {
            "fallback_total_cost_bps": 8.0,
            "annual_borrow_rate": 0.0,
            "annual_carry_rate": 0.0,
            "ci_zscore": 1.96,
        },
    }


def _make_features_index(labels_df: pd.DataFrame) -> pd.DataFrame:
    features = labels_df[["date", "symbol"]].copy()
    features["feature_timestamp"] = pd.to_datetime(features["date"])
    features["feature_valid_flag"] = True
    features["inclusion_flag"] = True
    return features


def _run_pipeline(tmp_path: Path) -> dict[str, object]:
    bundle = _make_market_bundle()

    input_dir = tmp_path / "inputs"
    build_dir = tmp_path / "build_out"
    qc_dir = tmp_path / "qc_out"
    neut_dir = tmp_path / "neut_out"

    prices_path = _write_csv(bundle["prices"], input_dir / "prices.csv")
    universe_path = _write_csv(bundle["universe"], input_dir / "universe.csv")
    calendar_path = _write_csv(bundle["calendar"], input_dir / "calendar.csv")
    costs_path = _write_csv(bundle["costs"], input_dir / "costs.csv")
    exposures_path = _write_csv(bundle["exposures"], input_dir / "exposures.csv")
    config_path = _write_json(_config_mapping(), input_dir / "config.json")

    build_result = build_labels(
        input_paths=InputPaths(
            adjusted_prices_pit_path=prices_path,
            universe_history_path=universe_path,
            trading_calendar_path=calendar_path,
            label_config_path=config_path,
            execution_costs_path=costs_path,
            neutralization_exposures_path=exposures_path,
            output_dir=str(build_dir),
        ),
        run_id="integration_build",
    )
    labels_df = load_table(build_result["labels_parquet"])
    assert labels_df is not None

    features_df = _make_features_index(labels_df)
    features_path = _write_csv(features_df, tmp_path / "features.csv")

    # Relax coverage thresholds slightly for this synthetic integration panel.
    # The point here is to validate pipeline compatibility, not to re-litigate
    # the stricter QC thresholds already tested in test_label_qc.py.
    qc_artifacts = run_label_qc(
        QCConfig(
            labels_store_path=build_result["labels_parquet"],
            features_index_path=features_path,
            run_id="integration_qc",
            output_dir=str(qc_dir),
            decision_lag=1,
            coverage_fail_by_horizon={5: 0.75, 10: 0.60},
            coverage_warn_by_horizon={5: 0.82, 10: 0.68},
            drift_fail_if_coverage_breach=False,
        )
    )

    neut_result = run_neutralized_targets(
        NeutralizationConfig.from_mapping(
            {
                "labels_path": build_result["labels_parquet"],
                "factor_exposures_path": exposures_path,
                "run_id": "integration_neut",
                "output_dir": str(neut_dir),
                "neutralization_input": "y_fwd_ret_net",
                "neutralization_mode": "sector_beta_size_liquidity",
                "neutralization_estimator": "weighted_ridge",
                "lambda_selection": "fixed",
                "lambda_default": 1.0,
                "min_obs_per_date": 12,
                "cv_min_obs_per_date": 1000,
                "min_sector_support": 3,
                "max_row_drop_frac": 0.20,
                "max_missing_essential_frac": 0.10,
                "include_volatility_if_available": False,
                "emit_date_level_stats": True,
                "persist_partitioned_output": True,
            }
        )
    )

    split_result = build_purged_splits(
        labels_path=build_result["labels_parquet"],
        calendar_path=calendar_path,
        config=SplitConfig(
            split_mode="purged_kfold",
            k_folds=3,
            decision_lag=1,
            horizon_policy="primary",
            primary_horizon=5,
            label_valid_flag_col="label_valid_flag_5d",
            embargo_policy="fixed_days",
            fixed_embargo_days=1,
            min_train_obs=1,
            min_test_obs=1,
        ),
        run_id="integration_split",
        labels_manifest_path=build_result["manifest_json"],
        features_index_path=features_path,
    )

    return {
        "bundle": bundle,
        "build_result": build_result,
        "labels_df": labels_df,
        "features_df": features_df,
        "features_path": features_path,
        "qc_artifacts": qc_artifacts,
        "neut_result": neut_result,
        "split_result": split_result,
    }


def test_labels_pipeline_build_qc_neutralize_and_split_end_to_end(tmp_path: Path) -> None:
    artifacts = _run_pipeline(tmp_path)

    build_result = artifacts["build_result"]
    labels_df = artifacts["labels_df"]
    qc_artifacts = artifacts["qc_artifacts"]
    neut_result = artifacts["neut_result"]
    split_result = artifacts["split_result"]

    assert build_result["run_id"] == "integration_build"
    assert len(labels_df) == build_result["n_rows"]
    assert {"y_fwd_ret_net_5d", "y_fwd_ret_net_10d", "event_start_5d", "event_end_10d"}.issubset(labels_df.columns)
    assert int(labels_df["label_valid_flag_5d"].sum()) > int(labels_df["label_valid_flag_10d"].sum()) > 0

    # QC should not hard fail on a clean synthetic store once thresholds are
    # adapted to the finite-sample characteristics of this integration fixture.
    assert qc_artifacts.qc_summary["global_gate"] in {"PASS", "WARN"}
    assert qc_artifacts.qc_summary["n_failures"] == 0
    if not qc_artifacts.qc_failures.empty:
        assert set(qc_artifacts.qc_failures["severity"].unique()) <= {"WARN"}
    for path in qc_artifacts.output_paths.values():
        assert Path(path).exists(), f"missing QC artifact: {path}"

    result_df = neut_result["result_df"]
    before_after_df = neut_result["before_after_df"]
    assert len(result_df) == len(labels_df) * 2
    assert set(result_df["horizon"].dropna().astype(int).unique()) == {5, 10}
    assert neut_result["summary"]["valid_fraction"] > 0.70
    assert set(result_df["run_id"].unique()) == {"integration_neut"}

    horizon_stats = before_after_df.loc[before_after_df["stat_level"] == "horizon"].sort_values("horizon")
    assert not horizon_stats.empty
    assert (horizon_stats["corr_y_neut_beta_mkt"].abs() < horizon_stats["corr_y_base_beta_mkt"].abs()).all()
    assert (horizon_stats["corr_y_neut_log_mktcap"].abs() < horizon_stats["corr_y_base_log_mktcap"].abs()).all()

    assert len(split_result.fold_summary) == 3
    assert (split_result.fold_summary["n_train_final"] > 0).all()
    assert (split_result.fold_summary["n_test"] > 0).all()
    assert set(split_result.leakage_validation["status"].unique()) == {"PASS"}
    assert split_result.manifest["run_id"] == "integration_split"
    assert split_result.manifest["h_eff"] == 5
    assert split_result.manifest["b_eff"] == 1


def test_external_neutralization_preserves_build_label_event_windows_and_valid_support(tmp_path: Path) -> None:
    artifacts = _run_pipeline(tmp_path)

    labels_df = artifacts["labels_df"]
    neut_df = artifacts["neut_result"]["result_df"].copy()

    expected_long_parts: list[pd.DataFrame] = []
    for horizon in (5, 10):
        piece = labels_df[
            [
                "date",
                "symbol",
                f"event_start_{horizon}d",
                f"event_end_{horizon}d",
                f"label_valid_flag_{horizon}d",
            ]
        ].copy()
        piece["horizon"] = horizon
        piece = piece.rename(
            columns={
                f"event_start_{horizon}d": "event_start_expected",
                f"event_end_{horizon}d": "event_end_expected",
                f"label_valid_flag_{horizon}d": "label_valid_expected",
            }
        )
        expected_long_parts.append(piece)
    expected_long = pd.concat(expected_long_parts, axis=0, ignore_index=True)
    expected_long["date"] = pd.to_datetime(expected_long["date"])
    expected_long["event_start_expected"] = pd.to_datetime(expected_long["event_start_expected"])
    expected_long["event_end_expected"] = pd.to_datetime(expected_long["event_end_expected"])

    neut_df["date"] = pd.to_datetime(neut_df["date"])
    neut_df["event_start"] = pd.to_datetime(neut_df["event_start"])
    neut_df["event_end"] = pd.to_datetime(neut_df["event_end"])

    merged = neut_df.merge(expected_long, on=["date", "symbol", "horizon"], how="left", validate="one_to_one")

    assert len(merged) == len(neut_df)
    assert merged["event_start"].equals(merged["event_start_expected"])
    assert merged["event_end"].equals(merged["event_end_expected"])

    valid_by_horizon = merged.groupby("horizon")["neutralization_valid_flag"].sum().to_dict()
    expected_by_horizon = expected_long.groupby("horizon")["label_valid_expected"].sum().to_dict()
    assert {int(k): int(v) for k, v in valid_by_horizon.items()} == {int(k): int(v) for k, v in expected_by_horizon.items()}
