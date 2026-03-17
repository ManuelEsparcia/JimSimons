from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module():
    candidates = [
        "simons_smallcap_swing.labels.neutralized_targets",
        "labels.neutralized_targets",
        "neutralized_targets",
    ]
    last_exc = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:  # pragma: no cover - import fallback path
            last_exc = exc
    raise last_exc  # type: ignore[misc]


nt = _load_module()

NeutralizationConfig = nt.NeutralizationConfig
run_neutralized_targets = nt.run_neutralized_targets
DataContractError = nt.DataContractError
FAIL_N_OBS_TOO_SMALL = nt.FAIL_N_OBS_TOO_SMALL
FAIL_MISSING_ESSENTIAL = nt.FAIL_MISSING_ESSENTIAL


RNG = np.random.default_rng(12345)


def _make_wide_bundle(
    *,
    n_dates: int = 3,
    n_symbols: int = 30,
    horizons: tuple[int, ...] = (5, 10),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    symbols = [f"S{i:03d}" for i in range(n_symbols)]
    sectors = ("tech", "health", "industrial", "energy", "utilities")
    sector_effects = {
        "tech": 0.030,
        "health": -0.015,
        "industrial": 0.000,
        "energy": 0.020,
        "utilities": -0.010,
    }

    label_rows: list[dict[str, object]] = []
    exposure_rows: list[dict[str, object]] = []

    for date_idx, date in enumerate(dates):
        for sym_idx, symbol in enumerate(symbols):
            sector = sectors[sym_idx % len(sectors)]
            beta = float(RNG.normal(1.0 + 0.03 * date_idx, 0.18))
            log_mktcap = float(10.0 + RNG.normal(0.0, 0.85))
            liq_log = float(12.0 + RNG.normal(0.0, 0.60))
            exposure_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "sector": sector,
                    "market_beta": beta,
                    "market_cap": float(np.exp(log_mktcap)),
                    "liquidity": float(np.expm1(liq_log)),
                    "quality_weight": 1.0 + 0.05 * (sym_idx % 7),
                }
            )

            base = (
                0.18 * beta
                + 0.05 * log_mktcap
                + 0.035 * liq_log
                + sector_effects[sector]
                + float(RNG.normal(0.0, 0.01))
            )
            row: dict[str, object] = {
                "date": date,
                "symbol": symbol,
                "label_valid_flag": True,
            }
            for horizon in horizons:
                row[f"y_fwd_ret_net_{horizon}d"] = float(base * (1.0 + 0.01 * horizon) + RNG.normal(0.0, 0.008))
            label_rows.append(row)

    return pd.DataFrame(label_rows), pd.DataFrame(exposure_rows)


def _make_long_from_wide(labels_wide: pd.DataFrame, horizons: tuple[int, ...] = (5, 10)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, rec in labels_wide.iterrows():
        for horizon in horizons:
            rows.append(
                {
                    "date": rec["date"],
                    "symbol": rec["symbol"],
                    "horizon": horizon,
                    "label_valid_flag": rec["label_valid_flag"],
                    "y_fwd_ret_net": rec[f"y_fwd_ret_net_{horizon}d"],
                }
            )
    return pd.DataFrame(rows)


def _write_csv(df: pd.DataFrame, path: Path) -> str:
    df.to_csv(path, index=False)
    return str(path)


def _base_mapping(labels_path: str, exposures_path: str, output_dir: Path) -> dict[str, object]:
    return {
        "labels_path": labels_path,
        "factor_exposures_path": exposures_path,
        "run_id": "unit_neut",
        "output_dir": str(output_dir),
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
        "exposure_weight_col": "quality_weight",
    }


@pytest.fixture()
def wide_bundle() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _make_wide_bundle()


@pytest.fixture()
def wide_run_inputs(tmp_path: Path, wide_bundle: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[dict[str, object], Path]:
    labels_df, exposures_df = wide_bundle
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    labels_path = _write_csv(labels_df, inputs_dir / "labels.csv")
    exposures_path = _write_csv(exposures_df, inputs_dir / "exposures.csv")
    output_dir = tmp_path / "out"
    return _base_mapping(labels_path, exposures_path, output_dir), output_dir


def test_neutralized_targets_end_to_end_persists_outputs_and_expected_schema(
    wide_run_inputs: tuple[dict[str, object], Path],
    wide_bundle: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    mapping, output_dir = wide_run_inputs
    mapping["run_id"] = "unit_neut_e2e"
    config = NeutralizationConfig.from_mapping(mapping)

    result = run_neutralized_targets(config)

    labels_df, _ = wide_bundle
    expected_rows = len(labels_df) * 2
    result_df = result["result_df"]
    before_after_df = result["before_after_df"]

    assert output_dir.exists()
    assert len(result_df) == expected_rows
    assert set(result_df["horizon"].dropna().astype(int).unique()) == {5, 10}
    assert result_df["neutralization_valid_flag"].all()
    assert set(result_df["run_id"].unique()) == {"unit_neut_e2e"}

    required_cols = {
        "date",
        "symbol",
        "horizon",
        "y_base",
        "y_neut",
        "neutralization_valid_flag",
        "neutralization_failure_reason",
        "neutralization_model",
        "lambda_used",
        "r2_cross_sectional",
        "n_obs_date",
        "n_obs_used_date",
        "beta_mkt",
        "log_mktcap",
        "liq",
        "sector",
    }
    assert required_cols.issubset(result_df.columns)
    assert not before_after_df.empty

    for key in ("summary_path", "manifest_path", "before_after_path", "failed_dates_path"):
        assert Path(result[key]).exists(), f"missing artifact for {key}"
    for path in result["main_output_files"]:
        assert Path(path).exists(), f"missing main output artifact: {path}"

    summary_payload = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    manifest_payload = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

    assert summary_payload["run_id"] == "unit_neut_e2e"
    assert summary_payload["valid_fraction"] == pytest.approx(1.0)
    assert summary_payload["horizon_summary"]["5"]["n_rows"] == len(labels_df)
    assert summary_payload["horizon_summary"]["10"]["n_rows"] == len(labels_df)
    assert manifest_payload["module"] == "labels/neutralized_targets.py"
    assert manifest_payload["summary"]["valid_fraction"] == pytest.approx(1.0)


def test_neutralization_reduces_factor_exposure_in_before_after_stats(
    wide_run_inputs: tuple[dict[str, object], Path],
) -> None:
    mapping, _ = wide_run_inputs
    mapping["run_id"] = "unit_neut_exposure"
    config = NeutralizationConfig.from_mapping(mapping)

    result = run_neutralized_targets(config)
    horizon_stats = result["before_after_df"].loc[result["before_after_df"]["stat_level"] == "horizon"].copy()
    horizon_stats = horizon_stats.sort_values("horizon").reset_index(drop=True)

    assert not horizon_stats.empty
    for factor in ("beta_mkt", "log_mktcap", "liq"):
        base_col = f"corr_y_base_{factor}"
        neut_col = f"corr_y_neut_{factor}"
        assert base_col in horizon_stats.columns
        assert neut_col in horizon_stats.columns
        assert (horizon_stats[base_col].abs() > 0.20).all(), f"synthetic dataset too weak for factor {factor}"
        assert (horizon_stats[neut_col].abs() < horizon_stats[base_col].abs()).all(), factor
        assert (horizon_stats[neut_col].abs() < 0.15).all(), factor


def test_invalidates_all_groups_when_cross_section_too_small(tmp_path: Path) -> None:
    labels_df, exposures_df = _make_wide_bundle(n_dates=2, n_symbols=8, horizons=(5,))
    inputs_dir = tmp_path / "inputs_small"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    labels_path = _write_csv(labels_df, inputs_dir / "labels.csv")
    exposures_path = _write_csv(exposures_df, inputs_dir / "exposures.csv")

    mapping = _base_mapping(labels_path, exposures_path, tmp_path / "out_small")
    mapping["run_id"] = "unit_neut_small"
    mapping["min_obs_per_date"] = 12
    config = NeutralizationConfig.from_mapping(mapping)

    result = run_neutralized_targets(config)
    result_df = result["result_df"]
    failed_dates = result["failed_dates_df"]

    assert not result_df["neutralization_valid_flag"].any()
    assert set(result_df["neutralization_failure_reason"].dropna().unique()) == {FAIL_N_OBS_TOO_SMALL}
    assert len(failed_dates) == 2
    assert set(failed_dates["failure_reason"].unique()) == {FAIL_N_OBS_TOO_SMALL}
    assert result["summary"]["valid_fraction"] == pytest.approx(0.0)



def test_material_missing_essential_exposures_fail_only_affected_date(tmp_path: Path) -> None:
    labels_df, exposures_df = _make_wide_bundle(n_dates=2, n_symbols=24, horizons=(5,))
    target_date = pd.Timestamp(labels_df["date"].min())
    target_symbols = sorted(exposures_df.loc[exposures_df["date"] == target_date, "symbol"].unique())[:5]
    missing_mask = (exposures_df["date"] == target_date) & (exposures_df["symbol"].isin(target_symbols))
    exposures_df.loc[missing_mask, "market_beta"] = np.nan

    inputs_dir = tmp_path / "inputs_missing"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    labels_path = _write_csv(labels_df, inputs_dir / "labels.csv")
    exposures_path = _write_csv(exposures_df, inputs_dir / "exposures.csv")

    mapping = _base_mapping(labels_path, exposures_path, tmp_path / "out_missing")
    mapping["run_id"] = "unit_neut_missing"
    mapping["max_missing_essential_frac"] = 0.10
    config = NeutralizationConfig.from_mapping(mapping)

    result = run_neutralized_targets(config)
    result_df = result["result_df"]
    failed_dates = result["failed_dates_df"]

    first_date_rows = result_df.loc[result_df["date"] == target_date]
    second_date_rows = result_df.loc[result_df["date"] != target_date]

    assert not first_date_rows["neutralization_valid_flag"].any()
    assert second_date_rows["neutralization_valid_flag"].all()
    assert set(first_date_rows["neutralization_failure_reason"].dropna().unique()) == {FAIL_MISSING_ESSENTIAL}
    assert len(failed_dates) == 1
    assert failed_dates.iloc[0]["failure_reason"] == FAIL_MISSING_ESSENTIAL



def test_supports_long_input_with_explicit_horizon_column(tmp_path: Path, wide_bundle: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    labels_wide, exposures_df = wide_bundle
    labels_long = _make_long_from_wide(labels_wide)

    inputs_dir = tmp_path / "inputs_long"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    labels_path = _write_csv(labels_long, inputs_dir / "labels_long.csv")
    exposures_path = _write_csv(exposures_df, inputs_dir / "exposures.csv")

    mapping = _base_mapping(labels_path, exposures_path, tmp_path / "out_long")
    mapping["run_id"] = "unit_neut_long"
    mapping["horizon_col"] = "horizon"
    config = NeutralizationConfig.from_mapping(mapping)

    result = run_neutralized_targets(config)
    result_df = result["result_df"]

    assert len(result_df) == len(labels_long)
    assert result_df["neutralization_valid_flag"].all()
    assert set(result_df["horizon"].dropna().astype(int).unique()) == {5, 10}
    assert set(result_df["source_target_col"].dropna().unique()) == {"y_fwd_ret_net"}



def test_raises_on_duplicate_exposure_primary_key(tmp_path: Path, wide_bundle: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    labels_df, exposures_df = wide_bundle
    duplicated_exposures = pd.concat([exposures_df, exposures_df.iloc[[0]]], ignore_index=True)

    inputs_dir = tmp_path / "inputs_dup"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    labels_path = _write_csv(labels_df, inputs_dir / "labels.csv")
    exposures_path = _write_csv(duplicated_exposures, inputs_dir / "exposures.csv")

    mapping = _base_mapping(labels_path, exposures_path, tmp_path / "out_dup")
    mapping["run_id"] = "unit_neut_dup"
    config = NeutralizationConfig.from_mapping(mapping)

    with pytest.raises(DataContractError, match="factor_exposures has 1 duplicate rows on primary key"):
        run_neutralized_targets(config)
