
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    path = Path("/mnt/data/market_proxies.py")
    spec = importlib.util.spec_from_file_location("market_proxies_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["market_proxies_under_test"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mp():
    return _load_module()


@pytest.fixture()
def parquet_shim(monkeypatch):
    import pandas as _pd

    monkeypatch.setattr(_pd.DataFrame, "to_parquet", lambda self, path, index=False, **kwargs: self.to_csv(path, index=index))
    monkeypatch.setattr(_pd, "read_parquet", lambda path, *args, **kwargs: _pd.read_csv(path))
    yield


@pytest.fixture()
def sample_paths(tmp_path, mp, parquet_shim, monkeypatch):
    monkeypatch.setattr(mp, "ensure_parquet_support", lambda: None)

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    rows = []
    bases = {"AAA": 100.0, "BBB": 60.0, "CCC": 80.0}
    drifts = {"AAA": 0.020, "BBB": -0.010, "CCC": 0.005}
    for symbol, base in bases.items():
        prev_close = None
        for i, d in enumerate(dates):
            close = base * (1.0 + drifts[symbol] + 0.01 * i)
            ret = 0.0 if prev_close is None else close / prev_close - 1.0
            rows.append(
                {
                    "symbol": symbol,
                    "date": d.strftime("%Y-%m-%d"),
                    "close_adj": close,
                    "ret_1d_adj": ret,
                    "volume_adj": 1000 + 50 * i + (25 if symbol == "AAA" else 0),
                    "dollar_volume": close * (1000 + 50 * i + (25 if symbol == "AAA" else 0)),
                    "market_cap": close * 1_000_000,
                    "spread_proxy": 0.01 + 0.001 * i,
                    "high_adj": close * 1.01,
                    "low_adj": close * 0.99,
                }
            )
            prev_close = close

    prices = pd.DataFrame(rows)
    universe = prices[["symbol", "date"]].copy()
    universe["is_eligible"] = True
    universe["universe_snapshot_id"] = "snap_001"

    prices_path = tmp_path / "prices.csv"
    universe_path = tmp_path / "universe.csv"
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "out"

    prices.to_csv(prices_path, index=False)
    universe.to_csv(universe_path, index=False)
    config = {
        "return_windows": [1, 5, 20],
        "realized_vol_window": 20,
        "coverage": {"n_min": 2, "ratio_min": 0.50, "warn_margin_n": 1, "warn_margin_ratio": 0.10},
        "exploratory": {"enabled": ["cross_section_corr_proxy"], "corr_window": 3},
        "smoothing": {"enabled": False, "columns": [], "ema_windows": []},
        "zscore": {"enabled": False, "columns": [], "rolling_windows": []},
        "output_dir": str(output_dir),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = mp.run_market_proxies(
        prices_path=prices_path,
        universe_path=universe_path,
        config_path=config_path,
        run_id="run_price_contract",
        asof_ts_utc="2024-01-10T00:00:00Z",
    )
    return {
        "prices_path": prices_path,
        "universe_path": universe_path,
        "config_path": config_path,
        "output_dir": output_dir,
        "result": result,
    }


def test_proxy_dataset_contains_required_columns(sample_paths, mp):
    df = sample_paths["result"]["data"]
    required = {
        "date",
        "run_id",
        "asof_ts_utc",
        "universe_snapshot_id",
        "market_ret_1d",
        "market_ret_5d",
        "market_ret_20d",
        "market_realized_vol_20d",
        "breadth_pct_up",
        "advance_decline_ratio",
        "cross_section_dispersion",
        "market_turnover_proxy",
        "severity_max",
    }
    assert required.issubset(df.columns)

    for core in mp.CORE_PROXIES:
        assert core in df.columns


def test_summary_contains_required_statistics(sample_paths):
    summary = sample_paths["result"]["summary"]
    assert summary["n_rows"] == 5
    assert summary["date_min"] == "2024-01-01"
    assert summary["date_max"] == "2024-01-05"
    assert "stats" in summary and isinstance(summary["stats"], dict)

    ret_stats = summary["stats"]["market_ret_1d"]
    for key in ["mean", "std", "p01", "p05", "p50", "p95", "p99", "nulls", "nonnull"]:
        assert key in ret_stats


def test_validation_contains_coverage_and_severity_fields(sample_paths):
    warn_df = sample_paths["result"]["validation_rows"]
    assert set(["date", "proxy_name", "severity", "universe_count", "coverage_count", "coverage_ratio", "reason"]).issubset(
        warn_df.columns
    )
    assert set(warn_df["severity"].dropna().unique()).issubset({"INFO", "WARN", "FAIL"})
    assert warn_df["proxy_name"].notna().all()


def test_manifest_contains_required_metadata(sample_paths, mp):
    manifest = sample_paths["result"]["manifest"]
    assert manifest["module_name"] == mp.MODULE_NAME
    assert manifest["code_version"] == mp.CODE_VERSION
    assert manifest["run_id"] == "run_price_contract"
    assert manifest["asof_ts_utc"] == "2024-01-10T00:00:00Z"
    assert "config_hash" in manifest and isinstance(manifest["config_hash"], str) and manifest["config_hash"]
    assert manifest["gate_status"] in {"PASS", "WARN", "FAIL"}
    assert set(["prices_adjusted_path", "universe_history_path", "config_path"]).issubset(manifest["inputs"].keys())
    assert set(["partitioned_dataset_dir", "summary_json", "validation_json", "manifest_json", "validation_rows_parquet"]).issubset(
        manifest["artifacts"].keys()
    )


def test_coverage_count_and_severity_max_are_persisted_correctly(sample_paths):
    df = sample_paths["result"]["data"]
    coverage_cols = [c for c in df.columns if c.startswith("coverage_count_") or c.startswith("coverage_ratio_")]
    assert coverage_cols, "Expected coverage columns in market proxy dataset"
    assert df["severity_max"].notna().all()
    assert set(df["severity_max"].astype(str).unique()).issubset({"INFO", "WARN", "FAIL"})


def test_outputs_are_sorted_deterministically(sample_paths):
    df = sample_paths["result"]["data"]
    expected = df.sort_values(["date", "universe_snapshot_id"], kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected)

    warn_df = sample_paths["result"]["validation_rows"]
    expected_warn = warn_df.sort_values(["date", "proxy_name", "severity"], kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(warn_df.reset_index(drop=True), expected_warn)


def test_core_proxies_are_always_present_even_if_exploratories_are_disabled(tmp_path, mp, parquet_shim, monkeypatch):
    monkeypatch.setattr(mp, "ensure_parquet_support", lambda: None)

    prices = pd.DataFrame(
        [
            {"symbol": "AAA", "date": "2024-01-01", "close_adj": 100.0, "ret_1d_adj": 0.0, "volume_adj": 1000, "dollar_volume": 100000, "market_cap": 1e8, "spread_proxy": 0.01, "high_adj": 101.0, "low_adj": 99.0},
            {"symbol": "BBB", "date": "2024-01-01", "close_adj": 50.0, "ret_1d_adj": 0.0, "volume_adj": 900, "dollar_volume": 45000, "market_cap": 5e7, "spread_proxy": 0.015, "high_adj": 50.5, "low_adj": 49.5},
            {"symbol": "AAA", "date": "2024-01-02", "close_adj": 101.0, "ret_1d_adj": 0.01, "volume_adj": 1100, "dollar_volume": 111100, "market_cap": 1.01e8, "spread_proxy": 0.011, "high_adj": 102.0, "low_adj": 100.0},
            {"symbol": "BBB", "date": "2024-01-02", "close_adj": 49.0, "ret_1d_adj": -0.02, "volume_adj": 950, "dollar_volume": 46550, "market_cap": 4.9e7, "spread_proxy": 0.016, "high_adj": 49.5, "low_adj": 48.5},
        ]
    )
    universe = prices[["symbol", "date"]].copy()
    universe["is_eligible"] = True
    universe["universe_snapshot_id"] = "snap_no_exp"

    prices_path = tmp_path / "prices.csv"
    universe_path = tmp_path / "universe.csv"
    prices.to_csv(prices_path, index=False)
    universe.to_csv(universe_path, index=False)

    config = {
        "coverage": {"n_min": 2, "ratio_min": 0.50},
        "exploratory": {"enabled": []},
        "output_dir": str(tmp_path / "out"),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = mp.run_market_proxies(prices_path, universe_path, config_path, "run_no_exp", "2024-01-10T00:00:00Z")
    df = result["data"]
    for core in mp.CORE_PROXIES:
        assert core in df.columns

    warn_df = result["validation_rows"]
    disabled = warn_df.loc[warn_df["reason"] == "exploratory_disabled", "proxy_name"].tolist()
    assert "cross_section_corr_proxy" in disabled


def test_validation_json_and_manifest_json_roundtrip(sample_paths):
    manifest_path = Path(sample_paths["result"]["manifest"]["artifacts"]["manifest_json"])
    validation_path = Path(sample_paths["result"]["manifest"]["artifacts"]["validation_json"])
    summary_path = Path(sample_paths["result"]["manifest"]["artifacts"]["summary_json"])

    manifest_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_disk = json.loads(validation_path.read_text(encoding="utf-8"))
    summary_disk = json.loads(summary_path.read_text(encoding="utf-8"))

    assert manifest_disk["run_id"] == sample_paths["result"]["manifest"]["run_id"]
    assert validation_disk["gate_status"] == sample_paths["result"]["validation"]["gate_status"]
    assert summary_disk["n_rows"] == sample_paths["result"]["summary"]["n_rows"]


def test_validation_rows_artifact_exists_and_matches_semantically(sample_paths):
    warn_path = Path(sample_paths["result"]["manifest"]["artifacts"]["validation_rows_parquet"])
    assert warn_path.exists()

    warn_disk = pd.read_csv(warn_path)
    warn_return = sample_paths["result"]["validation_rows"].copy()

    warn_disk["date"] = pd.to_datetime(warn_disk["date"]).dt.strftime("%Y-%m-%d")
    warn_return["date"] = pd.to_datetime(warn_return["date"]).dt.strftime("%Y-%m-%d")

    warn_disk = warn_disk.sort_values(["date", "proxy_name", "severity", "reason"], kind="mergesort").reset_index(drop=True)
    warn_return = warn_return.sort_values(["date", "proxy_name", "severity", "reason"], kind="mergesort").reset_index(drop=True)

    pd.testing.assert_frame_equal(
        warn_disk[warn_return.columns],
        warn_return,
        check_dtype=False,
    )


def test_partitioned_dataset_exists_for_each_output_date(sample_paths):
    outdir = Path(sample_paths["output_dir"])
    dates = pd.to_datetime(sample_paths["result"]["data"]["date"]).dt.strftime("%Y-%m-%d").tolist()
    for d in dates:
        part = outdir / f"date={d}" / "part-00000.parquet"
        assert part.exists(), f"Missing partition artifact for {d}"
