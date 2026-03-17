from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# Import adapter
# -----------------------------------------------------------------------------


def _load_market_proxies_module():
    module_names = [
        "simons_smallcap_swing.data.price.market_proxies",
        "data.price.market_proxies",
        "market_proxies",
    ]
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue

    here = Path(__file__).resolve()
    candidate_paths = [
        here.parents[3] / "simons_smallcap_swing" / "data" / "price" / "market_proxies.py",
        here.parents[2] / "data" / "price" / "market_proxies.py",
        here.parents[1] / "market_proxies.py",
        Path("/mnt/data/market_proxies.py"),
    ]
    for path in candidate_paths:
        if path.exists():
            module_name = f"_market_proxies_test_{abs(hash(str(path)))}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module

    raise ModuleNotFoundError(
        "Could not import market_proxies.py. Expected it at "
        "simons_smallcap_swing.data.price.market_proxies or in a nearby repo path."
    )


@pytest.fixture(scope="session")
def market_proxies_module():
    return _load_market_proxies_module()


# -----------------------------------------------------------------------------
# Helpers / fixtures
# -----------------------------------------------------------------------------


def _coverage(module, **overrides: Any):
    base = dataclasses.asdict(module.CoverageThresholds())
    base.update(overrides)
    return module.CoverageThresholds(**base)



def _exploratory(module, **overrides: Any):
    base = dataclasses.asdict(module.ExploratoryConfig())
    base.update(overrides)
    return module.ExploratoryConfig(**base)



def _smoothing(module, **overrides: Any):
    base = dataclasses.asdict(module.SmoothingConfig())
    base.update(overrides)
    return module.SmoothingConfig(**base)



def _zscore(module, **overrides: Any):
    base = dataclasses.asdict(module.ZScoreConfig())
    base.update(overrides)
    return module.ZScoreConfig(**base)



def _cfg(module, **overrides: Any):
    base = dataclasses.asdict(module.ProxyConfig())
    base.update(
        {
            "return_windows": (1, 2),
            "realized_vol_window": 2,
            "turnover_ref_window": 2,
            "winsor_lower": 0.05,
            "winsor_upper": 0.95,
            "coverage": _coverage(module, n_min=1, ratio_min=0.50, warn_margin_n=0, warn_margin_ratio=0.0),
            "exploratory": _exploratory(module, enabled=()),
            "smoothing": _smoothing(module, enabled=False),
            "zscore": _zscore(module, enabled=False),
        }
    )
    base.update(overrides)
    return module.ProxyConfig(**base)



def _prices_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df



def _universe_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df



def _normalize_inputs(module, prices: pd.DataFrame, universe: pd.DataFrame):
    p = module.normalize_prices(prices)
    u = module.normalize_universe(universe)
    return p, u



def _build_stack(module, prices: pd.DataFrame, universe: pd.DataFrame, cfg):
    p, u = _normalize_inputs(module, prices, universe)
    derived = module.add_symbol_level_derived_columns(p, cfg)
    panel = module.build_universe_panel(derived, u)
    base = module.build_market_return_base(panel, cfg)
    proxies, warn_df = module.compute_daily_proxies(panel, base, cfg)
    return derived, panel, base, proxies, warn_df



def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path



def _write_config_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@pytest.fixture()
def base_prices_df():
    return _prices_df(
        [
            {"symbol": "AAA", "date": "2024-01-02", "close_adj": 100.0, "ret_1d_adj": 0.00, "volume_adj": 100.0, "high_adj": 101.0, "low_adj": 99.0},
            {"symbol": "AAA", "date": "2024-01-03", "close_adj": 110.0, "ret_1d_adj": 0.10, "volume_adj": 100.0, "high_adj": 111.0, "low_adj": 109.0},
            {"symbol": "AAA", "date": "2024-01-04", "close_adj": 121.0, "ret_1d_adj": 0.10, "volume_adj": 100.0, "high_adj": 122.0, "low_adj": 120.0},
            {"symbol": "BBB", "date": "2024-01-02", "close_adj": 50.0, "ret_1d_adj": 0.00, "volume_adj": 200.0, "high_adj": 50.5, "low_adj": 49.5},
            {"symbol": "BBB", "date": "2024-01-03", "close_adj": 45.0, "ret_1d_adj": -0.10, "volume_adj": 200.0, "high_adj": 45.5, "low_adj": 44.5},
            {"symbol": "BBB", "date": "2024-01-04", "close_adj": 49.5, "ret_1d_adj": 0.10, "volume_adj": 200.0, "high_adj": 50.0, "low_adj": 49.0},
        ]
    )


@pytest.fixture()
def base_universe_df():
    return _universe_df(
        [
            {"symbol": "AAA", "date": "2024-01-02", "is_eligible": True, "run_id": "snapA"},
            {"symbol": "AAA", "date": "2024-01-03", "is_eligible": True, "run_id": "snapA"},
            {"symbol": "AAA", "date": "2024-01-04", "is_eligible": True, "run_id": "snapA"},
            {"symbol": "BBB", "date": "2024-01-02", "is_eligible": True, "run_id": "snapA"},
            {"symbol": "BBB", "date": "2024-01-03", "is_eligible": True, "run_id": "snapA"},
            {"symbol": "BBB", "date": "2024-01-04", "is_eligible": True, "run_id": "snapA"},
        ]
    )


# -----------------------------------------------------------------------------
# Unit tests
# -----------------------------------------------------------------------------


def test_build_market_return_base_on_fixed_universe(market_proxies_module, base_prices_df, base_universe_df):
    cfg = _cfg(market_proxies_module)
    _, panel, base, _, _ = _build_stack(market_proxies_module, base_prices_df, base_universe_df, cfg)

    assert list(base["date"]) == list(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    assert panel["symbol"].nunique() == 2

    day2 = base.loc[base["date"] == pd.Timestamp("2024-01-03")].iloc[0]
    day3 = base.loc[base["date"] == pd.Timestamp("2024-01-04")].iloc[0]

    assert day2["market_ret_1d"] == pytest.approx(0.0)
    assert day3["market_ret_1d"] == pytest.approx(0.1)
    assert day3["market_ret_2d"] == pytest.approx(0.10, abs=1e-12)
    assert day3["market_realized_vol_2d"] == pytest.approx(np.std([0.0, 0.1], ddof=1))



def test_pit_universe_change_changes_aggregation_correctly(market_proxies_module, base_prices_df, base_universe_df):
    cfg = _cfg(market_proxies_module)
    universe = base_universe_df.copy()
    universe.loc[(universe["symbol"] == "BBB") & (universe["date"] == pd.Timestamp("2024-01-04")), "is_eligible"] = False

    _, _, _, proxies, _ = _build_stack(market_proxies_module, base_prices_df, universe, cfg)
    day3 = proxies.loc[proxies["date"] == pd.Timestamp("2024-01-04")].iloc[0]

    assert int(day3["universe_count"]) == 1
    assert day3["market_ret_1d"] == pytest.approx(0.10)
    assert day3["breadth_pct_up"] == pytest.approx(1.0)
    assert day3["advance_decline_ratio"] == pytest.approx(2.0)



def test_insufficient_coverage_flags_warn_for_specific_proxy(market_proxies_module, base_prices_df, base_universe_df):
    cfg = _cfg(
        market_proxies_module,
        coverage=_coverage(market_proxies_module, n_min=3, ratio_min=0.95, warn_margin_n=0, warn_margin_ratio=0.0),
    )
    prices = base_prices_df[~((base_prices_df["symbol"] == "BBB") & (base_prices_df["date"] == pd.Timestamp("2024-01-03")))].copy()

    _, _, _, proxies, warn_df = _build_stack(market_proxies_module, prices, base_universe_df, cfg)
    day2 = proxies.loc[proxies["date"] == pd.Timestamp("2024-01-03")].iloc[0]
    market_ret_warns = warn_df[(warn_df["date"] == pd.Timestamp("2024-01-03")) & (warn_df["proxy_name"] == "market_ret_1d")]

    assert int(day2["coverage_count_market_ret_1d"]) == 1
    assert day2["coverage_ratio_market_ret_1d"] == pytest.approx(0.5)
    assert not market_ret_warns.empty
    assert set(market_ret_warns["severity"].astype(str)) == {"WARN"}



def test_winsorization_limits_outlier_in_cross_section_dispersion(market_proxies_module):
    cfg = _cfg(market_proxies_module, winsor_lower=0.10, winsor_upper=0.80)
    prices = _prices_df(
        [
            {"symbol": "AAA", "date": "2024-01-02", "close_adj": 100.0, "ret_1d_adj": 0.01, "volume_adj": 100.0},
            {"symbol": "BBB", "date": "2024-01-02", "close_adj": 100.0, "ret_1d_adj": 0.02, "volume_adj": 100.0},
            {"symbol": "CCC", "date": "2024-01-02", "close_adj": 100.0, "ret_1d_adj": 5.00, "volume_adj": 100.0},
        ]
    )
    universe = _universe_df(
        [
            {"symbol": "AAA", "date": "2024-01-02", "is_eligible": True, "run_id": "snapA"},
            {"symbol": "BBB", "date": "2024-01-02", "is_eligible": True, "run_id": "snapA"},
            {"symbol": "CCC", "date": "2024-01-02", "is_eligible": True, "run_id": "snapA"},
        ]
    )

    _, panel, base, proxies, _ = _build_stack(market_proxies_module, prices, universe, cfg)
    day = proxies.iloc[0]
    raw_std = float(panel.loc[panel["date"] == pd.Timestamp("2024-01-02"), "ret_1d_adj"].std(ddof=1))

    assert day["cross_section_dispersion"] < raw_std
    assert base.iloc[0]["market_ret_1d"] < float(panel["ret_1d_adj"].mean())



def test_breadth_pct_up_and_advance_decline_ratio_with_no_decliners(market_proxies_module):
    cfg = _cfg(market_proxies_module)
    prices = _prices_df(
        [
            {"symbol": "AAA", "date": "2024-01-02", "close_adj": 100.0, "ret_1d_adj": 0.03, "volume_adj": 100.0},
            {"symbol": "BBB", "date": "2024-01-02", "close_adj": 50.0, "ret_1d_adj": 0.02, "volume_adj": 150.0},
        ]
    )
    universe = _universe_df(
        [
            {"symbol": "AAA", "date": "2024-01-02", "is_eligible": True, "run_id": "snapA"},
            {"symbol": "BBB", "date": "2024-01-02", "is_eligible": True, "run_id": "snapA"},
        ]
    )

    _, _, _, proxies, _ = _build_stack(market_proxies_module, prices, universe, cfg)
    day = proxies.iloc[0]

    assert day["breadth_pct_up"] == pytest.approx(1.0)
    assert day["advance_decline_ratio"] == pytest.approx((2 + cfg.laplace_lambda) / (0 + cfg.laplace_lambda))



def test_market_turnover_proxy_is_robust_to_single_volume_outlier(market_proxies_module):
    cfg = _cfg(market_proxies_module)
    prices = _prices_df(
        [
            {"symbol": "AAA", "date": "2024-01-02", "close_adj": 10.0, "ret_1d_adj": 0.00, "volume_adj": 100.0},
            {"symbol": "BBB", "date": "2024-01-02", "close_adj": 10.0, "ret_1d_adj": 0.00, "volume_adj": 100.0},
            {"symbol": "CCC", "date": "2024-01-02", "close_adj": 10.0, "ret_1d_adj": 0.00, "volume_adj": 100.0},
            {"symbol": "AAA", "date": "2024-01-03", "close_adj": 10.0, "ret_1d_adj": 0.01, "volume_adj": 100.0},
            {"symbol": "BBB", "date": "2024-01-03", "close_adj": 10.0, "ret_1d_adj": 0.01, "volume_adj": 100.0},
            {"symbol": "CCC", "date": "2024-01-03", "close_adj": 10.0, "ret_1d_adj": 0.01, "volume_adj": 100000.0},
        ]
    )
    universe = _universe_df(
        [
            {"symbol": s, "date": d, "is_eligible": True, "run_id": "snapA"}
            for s in ["AAA", "BBB", "CCC"]
            for d in ["2024-01-02", "2024-01-03"]
        ]
    )

    derived, _, _, proxies, _ = _build_stack(market_proxies_module, prices, universe, cfg)
    day2 = proxies.loc[proxies["date"] == pd.Timestamp("2024-01-03")].iloc[0]
    ccc_ratio = float(derived.loc[(derived["symbol"] == "CCC") & (derived["date"] == pd.Timestamp("2024-01-03")), "turnover_ratio_raw"].iloc[0])

    assert ccc_ratio > 1.5
    assert day2["market_turnover_proxy"] < ccc_ratio
    assert day2["market_turnover_proxy"] == pytest.approx(1.0, rel=0.25)



def test_cross_section_corr_proxy_reaches_one_for_perfect_alignment(market_proxies_module):
    cfg = _cfg(
        market_proxies_module,
        exploratory=_exploratory(market_proxies_module, enabled=("cross_section_corr_proxy",), corr_window=2),
    )
    prices = _prices_df(
        [
            {"symbol": "AAA", "date": "2024-01-01", "close_adj": 100.0, "ret_1d_adj": 0.00, "volume_adj": 100.0},
            {"symbol": "AAA", "date": "2024-01-02", "close_adj": 101.0, "ret_1d_adj": 0.01, "volume_adj": 100.0},
            {"symbol": "AAA", "date": "2024-01-03", "close_adj": 103.0, "ret_1d_adj": 0.02, "volume_adj": 100.0},
            {"symbol": "AAA", "date": "2024-01-04", "close_adj": 106.0, "ret_1d_adj": 0.03, "volume_adj": 100.0},
            {"symbol": "BBB", "date": "2024-01-01", "close_adj": 50.0, "ret_1d_adj": 0.00, "volume_adj": 200.0},
            {"symbol": "BBB", "date": "2024-01-02", "close_adj": 50.5, "ret_1d_adj": 0.01, "volume_adj": 200.0},
            {"symbol": "BBB", "date": "2024-01-03", "close_adj": 51.5, "ret_1d_adj": 0.02, "volume_adj": 200.0},
            {"symbol": "BBB", "date": "2024-01-04", "close_adj": 53.0, "ret_1d_adj": 0.03, "volume_adj": 200.0},
        ]
    )
    universe = _universe_df(
        [
            {"symbol": s, "date": d, "is_eligible": True, "run_id": "snapA"}
            for s in ["AAA", "BBB"]
            for d in ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
        ]
    )

    _, _, _, proxies, _ = _build_stack(market_proxies_module, prices, universe, cfg)
    last = proxies.loc[proxies["date"] == pd.Timestamp("2024-01-04")].iloc[0]

    assert "cross_section_corr_proxy" in proxies.columns
    assert last["cross_section_corr_proxy"] == pytest.approx(1.0, abs=1e-12)



def test_exploratory_disabled_records_info_rows_and_core_columns_present(market_proxies_module, base_prices_df, base_universe_df):
    cfg = _cfg(market_proxies_module, exploratory=_exploratory(market_proxies_module, enabled=()))
    _, _, _, proxies, warn_df = _build_stack(market_proxies_module, base_prices_df, base_universe_df, cfg)

    expected_core = [
        "market_ret_1d",
        "market_ret_5d",
        "market_ret_20d",
        f"market_realized_vol_{cfg.realized_vol_window}d",
        "breadth_pct_up",
        "advance_decline_ratio",
        "cross_section_dispersion",
        "market_turnover_proxy",
    ]
    for col in expected_core:
        assert col in proxies.columns

    info_rows = warn_df[(warn_df["reason"] == "exploratory_disabled") & (warn_df["proxy_name"] == "cross_section_corr_proxy")]
    assert not info_rows.empty
    assert set(info_rows["severity"].astype(str)) == {"INFO"}
    assert "cross_section_corr_proxy" not in proxies.columns



def test_apply_optional_transforms_adds_ema_and_zscore_columns(market_proxies_module):
    cfg = _cfg(
        market_proxies_module,
        smoothing=_smoothing(market_proxies_module, enabled=True, columns=("market_ret_1d",), ema_windows=(2,)),
        zscore=_zscore(market_proxies_module, enabled=True, columns=("breadth_pct_up",), rolling_windows=(2,)),
    )
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "market_ret_1d": [0.00, 0.10, 0.20],
            "breadth_pct_up": [0.25, 0.75, 0.50],
        }
    )

    out = market_proxies_module.apply_optional_transforms(df, cfg)

    assert "market_ret_1d_ema2" in out.columns
    assert "breadth_pct_up_z2" in out.columns
    assert pd.isna(out.loc[0, "market_ret_1d_ema2"])
    assert out.loc[2, "market_ret_1d_ema2"] == pytest.approx(0.1555555556, rel=1e-6)
    assert out.loc[2, "breadth_pct_up_z2"] == pytest.approx(-0.70710678, rel=1e-6)



def test_run_market_proxies_is_reproducible_under_same_input(market_proxies_module, tmp_path: Path, base_prices_df, base_universe_df, monkeypatch):
    prices_path = _write_csv(base_prices_df, tmp_path / "prices.csv")
    universe_path = _write_csv(base_universe_df, tmp_path / "universe.csv")
    config = {
        "return_windows": [1, 2],
        "realized_vol_window": 2,
        "turnover_ref_window": 2,
        "winsor_lower": 0.05,
        "winsor_upper": 0.95,
        "coverage": {"n_min": 1, "ratio_min": 0.50, "warn_margin_n": 0, "warn_margin_ratio": 0.0},
        "exploratory": {"enabled": []},
        "output_dir": str(tmp_path / "out"),
    }
    config_path = _write_config_json(tmp_path / "config.json", config)

    monkeypatch.setattr(market_proxies_module, "persist_outputs", lambda *args, **kwargs: None)

    run1 = market_proxies_module.run_market_proxies(
        prices_path=prices_path,
        universe_path=universe_path,
        config_path=config_path,
        run_id="run_same",
        asof_ts_utc="2024-01-05T00:00:00Z",
    )
    run2 = market_proxies_module.run_market_proxies(
        prices_path=prices_path,
        universe_path=universe_path,
        config_path=config_path,
        run_id="run_same",
        asof_ts_utc="2024-01-05T00:00:00Z",
    )

    pd.testing.assert_frame_equal(run1["data"], run2["data"], check_dtype=False, check_like=False)
    pd.testing.assert_frame_equal(run1["validation_rows"], run2["validation_rows"], check_dtype=False, check_like=False)
    assert run1["summary"] == run2["summary"]
    assert run1["validation"] == run2["validation"]

    m1 = dict(run1["manifest"])
    m2 = dict(run2["manifest"])
    m1.pop("generated_at_utc", None)
    m2.pop("generated_at_utc", None)
    assert m1 == m2
