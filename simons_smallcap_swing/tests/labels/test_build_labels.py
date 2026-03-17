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
        "simons_smallcap_swing.labels.build_labels",
        "labels.build_labels",
        "build_labels",
    ]
    last_exc = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:  # pragma: no cover - import fallback path
            last_exc = exc
    raise last_exc  # type: ignore[misc]


bl = _load_module()

InputPaths = bl.InputPaths
LabelConfig = bl.LabelConfig
build_labels = bl.build_labels
_build_labels_core = bl._build_labels_core
load_table = bl.load_table


def _make_market_bundle(
    *,
    n_dates: int = 36,
    n_symbols: int = 30,
    horizons: tuple[int, ...] = (5, 10),
) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
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
        for t_idx, date in enumerate(dates):
            # Cross-sectional return structure intentionally tied to exposures
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
            universe_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "is_eligible": True,
                }
            )
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
        "calendar": pd.DataFrame({"date": dates}),
        "prices": pd.DataFrame(price_rows),
        "universe": pd.DataFrame(universe_rows),
        "exposures": pd.DataFrame(exposure_rows),
        "costs": pd.DataFrame(cost_rows),
    }


def _config_mapping(
    *,
    horizons: tuple[int, ...] = (5, 10),
    primary_target: str | None = None,
    missing_cost_policy: str = "strict_invalidate",
    neutralization_mode: str = "sector_beta_size_liquidity",
    min_valid_cross_section: int = 10,
    fallback_total_cost_bps: float = 8.0,
    annual_borrow_rate: float = 0.0,
    annual_carry_rate: float = 0.0,
) -> dict[str, object]:
    max_h = max(horizons)
    return {
        "horizons": list(horizons),
        "return_mode": "close_to_close",
        "decision_lag": 1,
        "net_of_costs": True,
        "primary_target": primary_target or f"y_fwd_ret_net_{max_h}d",
        "missing_cost_policy": missing_cost_policy,
        "neutralization_mode": neutralization_mode,
        "min_valid_cross_section": min_valid_cross_section,
        "neutralization_policy": {
            "n_min_neut": min_valid_cross_section,
            "fixed_lambda": 1.0,
            "lambda_selection": "fixed",
            "residual_var_floor": 1e-14,
        },
        "cost_policy": {
            "fallback_total_cost_bps": fallback_total_cost_bps,
            "annual_borrow_rate": annual_borrow_rate,
            "annual_carry_rate": annual_carry_rate,
            "ci_zscore": 1.96,
        },
    }


def _write_csv(df: pd.DataFrame, path: Path) -> str:
    df.to_csv(path, index=False)
    return str(path)


def _write_json(payload: dict[str, object], path: Path) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


@pytest.fixture()
def market_bundle() -> dict[str, pd.DataFrame]:
    return _make_market_bundle()


@pytest.fixture()
def base_config() -> dict[str, object]:
    return _config_mapping()


@pytest.fixture()
def public_run_inputs(
    tmp_path: Path,
    market_bundle: dict[str, pd.DataFrame],
    base_config: dict[str, object],
) -> tuple[InputPaths, Path]:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    prices_path = _write_csv(market_bundle["prices"], input_dir / "prices.csv")
    universe_path = _write_csv(market_bundle["universe"], input_dir / "universe.csv")
    calendar_path = _write_csv(market_bundle["calendar"], input_dir / "calendar.csv")
    costs_path = _write_csv(market_bundle["costs"], input_dir / "costs.csv")
    exposures_path = _write_csv(market_bundle["exposures"], input_dir / "exposures.csv")
    config_path = _write_json(base_config, input_dir / "config.json")

    output_dir = tmp_path / "out"
    paths = InputPaths(
        adjusted_prices_pit_path=prices_path,
        universe_history_path=universe_path,
        trading_calendar_path=calendar_path,
        label_config_path=config_path,
        execution_costs_path=costs_path,
        neutralization_exposures_path=exposures_path,
        output_dir=str(output_dir),
    )
    return paths, output_dir


def test_build_labels_end_to_end_persists_outputs_and_expected_schema(
    public_run_inputs: tuple[InputPaths, Path],
    market_bundle: dict[str, pd.DataFrame],
) -> None:
    input_paths, output_dir = public_run_inputs

    result = build_labels(input_paths=input_paths, run_id="unit_build")

    assert result["run_id"] == "unit_build"
    assert result["n_rows"] == len(market_bundle["universe"])
    assert output_dir.exists()

    labels_path = Path(result["labels_parquet"])
    coverage_path = Path(result["coverage_parquet"])
    manifest_path = Path(result["manifest_json"])
    split_path = Path(result["split_compatibility_json"])
    qc_path = Path(result["qc_json"])
    dictionary_path = Path(result["dictionary_json"])

    for path in (labels_path, coverage_path, manifest_path, split_path, qc_path, dictionary_path):
        assert path.exists(), f"missing output artifact: {path}"

    labels_df = load_table(labels_path)
    assert labels_df is not None
    required_cols = {
        "date",
        "symbol",
        "run_id",
        "event_start_10d",
        "event_end_10d",
        "y_fwd_ret_net_10d",
        "y_rank_10d",
        "y_cls_tail_10d",
        "y_neut_10d",
        "label_valid_flag",
    }
    assert required_cols.issubset(labels_df.columns)
    assert set(labels_df["run_id"].unique()) == {"unit_build"}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "unit_build"
    assert manifest["primary_target"] == "y_fwd_ret_net_10d"
    assert manifest["entry_price_field"] == "adj_close"
    assert manifest["exit_price_field"] == "adj_close"

    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    assert split_payload["decision_lag"] == 1
    assert split_payload["event_start_cols"]["10d"] == "event_start_10d"
    assert split_payload["event_end_cols"]["10d"] == "event_end_10d"

    qc_payload = json.loads(qc_path.read_text(encoding="utf-8"))
    fail_checks = [c for c in qc_payload["checks"] if c["severity"] == "FAIL" and c["status"] == "FAIL"]
    assert fail_checks == []


def test_build_labels_net_target_matches_gross_minus_costs_on_known_row(
    market_bundle: dict[str, pd.DataFrame],
) -> None:
    config = LabelConfig.from_mapping(
        _config_mapping(horizons=(5,), primary_target="y_fwd_ret_net_5d", min_valid_cross_section=5)
    )

    labels_df, _, _, _ = _build_labels_core(
        prices_df=market_bundle["prices"],
        universe_df=market_bundle["universe"],
        calendar_df=market_bundle["calendar"],
        config=config,
        costs_df=market_bundle["costs"],
        exposures_df=market_bundle["exposures"],
    )

    row = labels_df.loc[(labels_df["date"] == labels_df["date"].min()) & (labels_df["symbol"] == "S000")].iloc[0]
    expected_gross = row["exit_px_5d"] / row["entry_px_5d"] - 1.0
    expected_net = expected_gross - row["entry_cost_5d"] - row["exit_cost_5d"] - row["carry_cost_5d"]

    assert row["y_fwd_ret_gross_5d"] == pytest.approx(expected_gross)
    assert row["y_fwd_ret_net_5d"] == pytest.approx(expected_net)
    assert row["label_valid_flag_5d"] is True or row["label_valid_flag_5d"] == True


def test_build_labels_marks_incomplete_forward_window_on_tail_rows(
    market_bundle: dict[str, pd.DataFrame],
) -> None:
    config = LabelConfig.from_mapping(
        _config_mapping(horizons=(10,), primary_target="y_fwd_ret_net_10d", min_valid_cross_section=5)
    )

    labels_df, _, _, _ = _build_labels_core(
        prices_df=market_bundle["prices"],
        universe_df=market_bundle["universe"],
        calendar_df=market_bundle["calendar"],
        config=config,
        costs_df=market_bundle["costs"],
        exposures_df=market_bundle["exposures"],
    )

    tail = labels_df.loc[labels_df["event_end_10d"].isna()].copy()
    assert not tail.empty
    assert tail["incomplete_forward_window_10d"].all()
    assert (~tail["label_valid_flag_10d"]).all()
    assert set(tail["label_exclusion_reason_10d"].dropna().unique()) == {"INCOMPLETE_FORWARD_WINDOW"}

    head = labels_df.loc[labels_df["event_end_10d"].notna()].head(20)
    assert (~head["incomplete_forward_window_10d"]).all()


def test_build_labels_strict_missing_cost_invalidate_sets_reason(
    market_bundle: dict[str, pd.DataFrame],
) -> None:
    costs = market_bundle["costs"].copy()
    mask = (
        (costs["date"] == costs["date"].min())
        & (costs["symbol"] == "S000")
        & (costs["horizon_days"] == 5)
    )
    costs = costs.loc[~mask].reset_index(drop=True)

    config = LabelConfig.from_mapping(
        _config_mapping(horizons=(5,), primary_target="y_fwd_ret_net_5d", min_valid_cross_section=5)
    )

    labels_df, _, _, _ = _build_labels_core(
        prices_df=market_bundle["prices"],
        universe_df=market_bundle["universe"],
        calendar_df=market_bundle["calendar"],
        config=config,
        costs_df=costs,
        exposures_df=market_bundle["exposures"],
    )

    row = labels_df.loc[(labels_df["date"] == labels_df["date"].min()) & (labels_df["symbol"] == "S000")].iloc[0]
    assert row["missing_cost_input_5d"] is True or row["missing_cost_input_5d"] == True
    assert pd.isna(row["y_fwd_ret_net_5d"])
    assert row["label_exclusion_reason_5d"] == "MISSING_COST_INPUT"
    assert row["label_valid_flag_5d"] is False or row["label_valid_flag_5d"] == False


def test_build_labels_fallback_proxy_keeps_rows_valid_without_cost_table(
    market_bundle: dict[str, pd.DataFrame],
) -> None:
    config = LabelConfig.from_mapping(
        _config_mapping(
            horizons=(5,),
            primary_target="y_fwd_ret_net_5d",
            min_valid_cross_section=5,
            missing_cost_policy="fallback_proxy",
            neutralization_mode="none",
            fallback_total_cost_bps=8.0,
            annual_borrow_rate=0.0,
            annual_carry_rate=0.0,
        )
    )

    labels_df, _, _, _ = _build_labels_core(
        prices_df=market_bundle["prices"],
        universe_df=market_bundle["universe"],
        calendar_df=market_bundle["calendar"],
        config=config,
        costs_df=None,
        exposures_df=None,
    )

    row = labels_df.loc[(labels_df["date"] == labels_df["date"].min()) & (labels_df["symbol"] == "S000")].iloc[0]
    expected_net = row["y_fwd_ret_gross_5d"] - 0.0008
    assert row["missing_cost_input_5d"] is False or row["missing_cost_input_5d"] == False
    assert row["y_fwd_ret_net_5d"] == pytest.approx(expected_net)
    assert row["label_valid_flag_5d"] is True or row["label_valid_flag_5d"] == True


def test_build_labels_neutralization_reduces_beta_correlation(
    market_bundle: dict[str, pd.DataFrame],
) -> None:
    config = LabelConfig.from_mapping(
        _config_mapping(horizons=(10,), primary_target="y_fwd_ret_net_10d", min_valid_cross_section=10)
    )

    labels_df, _, _, _ = _build_labels_core(
        prices_df=market_bundle["prices"],
        universe_df=market_bundle["universe"],
        calendar_df=market_bundle["calendar"],
        config=config,
        costs_df=market_bundle["costs"],
        exposures_df=market_bundle["exposures"],
    )

    first_valid_date = (
        labels_df.loc[labels_df["label_valid_flag_10d"] & labels_df["y_neut_10d"].notna(), "date"]
        .sort_values()
        .iloc[0]
    )
    cross_section = labels_df.loc[
        (labels_df["date"] == first_valid_date)
        & labels_df["label_valid_flag_10d"]
        & labels_df["y_neut_10d"].notna(),
        ["y_fwd_ret_net_10d", "y_neut_10d", "market_beta"],
    ].copy()

    raw_corr = float(cross_section["y_fwd_ret_net_10d"].corr(cross_section["market_beta"]))
    neut_corr = float(cross_section["y_neut_10d"].corr(cross_section["market_beta"]))
    raw_std = float(cross_section["y_fwd_ret_net_10d"].std(ddof=1))
    neut_std = float(cross_section["y_neut_10d"].std(ddof=1))
    fit_statuses = set(
        labels_df.loc[
            (labels_df["date"] == first_valid_date) & labels_df["label_valid_flag_10d"],
            "neutralization_fit_status_10d",
        ].astype(str)
    )

    assert abs(raw_corr) > 0.50
    assert abs(neut_corr) < abs(raw_corr)
    assert abs(float(cross_section["y_neut_10d"].mean())) < 1e-10
    assert neut_std < 0.05 * raw_std
    assert fit_statuses == {"ok"}
