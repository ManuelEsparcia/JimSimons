from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# Import helpers
# -----------------------------------------------------------------------------


def _nearby_repo_candidates(filename: str) -> list[Path]:
    here = Path(__file__).resolve()
    candidates: list[Path] = [Path(f"/mnt/data/{filename}")]
    for parent in [here.parent, *list(here.parents)]:
        candidates.extend(
            [
                parent / filename,
                parent / "data" / "price" / filename,
                parent / "simons_smallcap_swing" / "data" / "price" / filename,
            ]
        )
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered



def _import_module(candidates: Iterable[str], file_candidates: Iterable[Path], label: str):
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception:
            pass

    for path in file_candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location(f"{label}_cli_smoke_under_test", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module

    raise ModuleNotFoundError(f"Could not import {label} from expected repo paths.")


@pytest.fixture(scope="session")
def fetch_prices_module():
    return _import_module(
        candidates=[
            "simons_smallcap_swing.data.price.fetch_prices",
            "data.price.fetch_prices",
            "fetch_prices",
        ],
        file_candidates=_nearby_repo_candidates("fetch_prices.py"),
        label="fetch_prices",
    )


@pytest.fixture(scope="session")
def adjust_prices_module():
    return _import_module(
        candidates=[
            "simons_smallcap_swing.data.price.adjust_prices",
            "data.price.adjust_prices",
            "adjust_prices",
        ],
        file_candidates=_nearby_repo_candidates("adjust_prices.py"),
        label="adjust_prices",
    )


@pytest.fixture(scope="session")
def qc_prices_module():
    return _import_module(
        candidates=[
            "simons_smallcap_swing.data.price.qc_prices",
            "data.price.qc_prices",
            "qc_prices",
        ],
        file_candidates=_nearby_repo_candidates("qc_prices.py"),
        label="qc_prices",
    )


@pytest.fixture(scope="session")
def market_proxies_module():
    return _import_module(
        candidates=[
            "simons_smallcap_swing.data.price.market_proxies",
            "data.price.market_proxies",
            "market_proxies",
        ],
        file_candidates=_nearby_repo_candidates("market_proxies.py"),
        label="market_proxies",
    )


# -----------------------------------------------------------------------------
# Runtime-safe persistence helpers
# -----------------------------------------------------------------------------


def _fake_to_parquet(self: pd.DataFrame, path: str | Path, *args, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    self.to_csv(path, index=False)



def _fake_write_parquet_deduped(path: Path, frame: pd.DataFrame, dedupe_keys=None, overwrite=False, compression="snappy") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    if dedupe_keys:
        keys = list(dedupe_keys)
        out = out.sort_values(keys, kind="stable").drop_duplicates(subset=keys, keep="last")
    out.to_csv(path, index=False)



def _fake_write_parquet(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)



def _patch_dataframe_to_parquet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet, raising=True)



def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# Small synthetic inputs
# -----------------------------------------------------------------------------


def _calendar_df() -> pd.DataFrame:
    return pd.DataFrame({"date": ["2024-01-02", "2024-01-03", "2024-01-04"], "is_session": [True, True, True]})



def _universe_snapshot_df() -> pd.DataFrame:
    return pd.DataFrame({"symbol": ["AAA", "BBB"], "run_id": ["u_001", "u_001"]})



def _fetch_provider_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)



def _raw_prices_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "AAA", "date": "2024-01-02", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0, "source_provider": "demo"},
            {"symbol": "AAA", "date": "2024-01-03", "open": 50.0, "high": 51.0, "low": 49.0, "close": 50.0, "volume": 2000.0, "source_provider": "demo"},
            {"symbol": "AAA", "date": "2024-01-04", "open": 55.0, "high": 56.0, "low": 54.0, "close": 55.0, "volume": 2100.0, "source_provider": "demo"},
            {"symbol": "BBB", "date": "2024-01-02", "open": 20.0, "high": 20.5, "low": 19.5, "close": 20.0, "volume": 1500.0, "source_provider": "demo"},
            {"symbol": "BBB", "date": "2024-01-03", "open": 21.0, "high": 21.5, "low": 20.5, "close": 21.0, "volume": 1550.0, "source_provider": "demo"},
            {"symbol": "BBB", "date": "2024-01-04", "open": 22.0, "high": 22.5, "low": 21.5, "close": 22.0, "volume": 1600.0, "source_provider": "demo"},
        ]
    )



def _corporate_actions_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "event_type": "split",
                "effective_date": "2024-01-03",
                "announcement_ts": "2024-01-02T21:00:00Z",
                "effective_ts_utc": "2024-01-03T00:00:00Z",
                "split_ratio": 0.5,
                "source_provider": "corp_actions_primary",
                "provider_priority": 0,
                "event_id": "AAA_SPLIT_20240103",
            }
        ]
    )



def _adjusted_prices_for_market_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "AAA", "date": "2024-01-02", "close_adj": 50.0, "ret_1d_adj": 0.0, "volume_adj": 2000.0, "high_adj": 50.5, "low_adj": 49.5},
            {"symbol": "AAA", "date": "2024-01-03", "close_adj": 50.0, "ret_1d_adj": 0.0, "volume_adj": 4000.0, "high_adj": 51.0, "low_adj": 49.0},
            {"symbol": "AAA", "date": "2024-01-04", "close_adj": 55.0, "ret_1d_adj": 0.1, "volume_adj": 4200.0, "high_adj": 56.0, "low_adj": 54.0},
            {"symbol": "BBB", "date": "2024-01-02", "close_adj": 20.0, "ret_1d_adj": 0.0, "volume_adj": 1500.0, "high_adj": 20.5, "low_adj": 19.5},
            {"symbol": "BBB", "date": "2024-01-03", "close_adj": 21.0, "ret_1d_adj": 0.05, "volume_adj": 1550.0, "high_adj": 21.5, "low_adj": 20.5},
            {"symbol": "BBB", "date": "2024-01-04", "close_adj": 22.0, "ret_1d_adj": 0.0476190476, "volume_adj": 1600.0, "high_adj": 22.5, "low_adj": 21.5},
        ]
    )



def _universe_history_df() -> pd.DataFrame:
    rows = []
    for d in ["2024-01-02", "2024-01-03", "2024-01-04"]:
        for symbol in ["AAA", "BBB"]:
            rows.append({"date": d, "symbol": symbol, "is_eligible": True, "run_id": "u_001"})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# CLI smoke tests
# -----------------------------------------------------------------------------


def test_fetch_prices_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fetch_prices_module: Any) -> None:
    _patch_dataframe_to_parquet(monkeypatch)
    monkeypatch.setattr(fetch_prices_module, "write_parquet_deduped", _fake_write_parquet_deduped)
    monkeypatch.setattr(fetch_prices_module, "require_parquet_engine", lambda: None)
    monkeypatch.setattr(fetch_prices_module, "git_code_version", lambda: "test-version")

    primary_root = tmp_path / "providers" / "primary"
    fallback_root = tmp_path / "providers" / "fallback"
    primary_root.mkdir(parents=True)
    fallback_root.mkdir(parents=True)

    _fetch_provider_df(
        [
            {"timestamp": "2024-01-02T21:00:00Z", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0, "vwap": 100.0, "trades_count": 10, "currency": "USD"},
            {"timestamp": "2024-01-03T21:00:00Z", "open": 50.0, "high": 51.0, "low": 49.0, "close": 50.0, "volume": 2000.0, "vwap": 50.0, "trades_count": 15, "currency": "USD"},
            {"timestamp": "2024-01-04T21:00:00Z", "open": 55.0, "high": 56.0, "low": 54.0, "close": 55.0, "volume": 2100.0, "vwap": 55.0, "trades_count": 16, "currency": "USD"},
        ]
    ).to_csv(primary_root / "AAA.csv", index=False)
    _fetch_provider_df(
        [
            {"timestamp": "2024-01-02T21:00:00Z", "open": 20.0, "high": 20.5, "low": 19.5, "close": 20.0, "volume": 1500.0, "vwap": 20.0, "trades_count": 8, "currency": "USD"},
            {"timestamp": "2024-01-03T21:00:00Z", "open": 21.0, "high": 21.5, "low": 20.5, "close": 21.0, "volume": 1550.0, "vwap": 21.0, "trades_count": 9, "currency": "USD"},
            {"timestamp": "2024-01-04T21:00:00Z", "open": 22.0, "high": 22.5, "low": 21.5, "close": 22.0, "volume": 1600.0, "vwap": 22.0, "trades_count": 9, "currency": "USD"},
        ]
    ).to_csv(fallback_root / "BBB.csv", index=False)

    universe_path = tmp_path / "universe.csv"
    calendar_path = tmp_path / "calendar.csv"
    provider_config_path = tmp_path / "providers.json"
    output_root = tmp_path / "fetch_out"

    _universe_snapshot_df().to_csv(universe_path, index=False)
    _calendar_df().to_csv(calendar_path, index=False)
    provider_config_path.write_text(
        json.dumps(
            {
                "fetch": {"mode": "incremental", "provider_priority": ["primary", "fallback"], "allow_fallback": True},
                "persistence": {
                    "output_root": str(output_root),
                    "persist_canonical": True,
                    "persist_raw": True,
                    "canonical_relpath": "canonical/canonical_prices.parquet",
                },
                "providers": {
                    "primary": {
                        "kind": "csv_directory",
                        "name": "primary_csv",
                        "dataset_id": "PRIMARY_DS",
                        "params": {"root_dir": str(primary_root), "file_pattern": "{symbol}.csv", "format": "csv"},
                    },
                    "fallback": {
                        "kind": "csv_directory",
                        "name": "fallback_csv",
                        "dataset_id": "FALLBACK_DS",
                        "params": {"root_dir": str(fallback_root), "file_pattern": "{symbol}.csv", "format": "csv"},
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    argv = [
        "fetch_prices.py",
        "--universe-snapshot-path", str(universe_path),
        "--calendar-path", str(calendar_path),
        "--provider-config-path", str(provider_config_path),
        "--start-date", "2024-01-02",
        "--end-date", "2024-01-04",
        "--fetch-mode", "incremental",
        "--run-id", "fetch_cli_001",
        "--as-of-ts-utc", "2024-01-04T22:00:00Z",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    fetch_prices_module.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["run_id"] == "fetch_cli_001"
    assert payload["rows_ingested"] == 6
    assert (output_root / "reports" / "summary_fetch_cli_001.json").exists()
    assert (output_root / "reports" / "manifest_fetch_cli_001.json").exists()



def test_adjust_prices_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adjust_prices_module: Any) -> None:
    _patch_dataframe_to_parquet(monkeypatch)
    monkeypatch.setattr(adjust_prices_module, "ensure_parquet_support", lambda: None)

    prices_path = tmp_path / "prices_raw.csv"
    events_path = tmp_path / "corporate_actions.csv"
    outdir = tmp_path / "adjust_out"

    _raw_prices_df().to_csv(prices_path, index=False)
    _corporate_actions_df().to_csv(events_path, index=False)

    rc = adjust_prices_module.main(
        [
            "--prices-raw-path", str(prices_path),
            "--corporate-actions-path", str(events_path),
            "--outdir", str(outdir),
            "--run-id", "adjust_cli_001",
            "--asof-ts-utc", "2024-01-04T22:00:00Z",
            "--start-date", "2024-01-02",
            "--end-date", "2024-01-04",
            "--adjust-mode", "dual_output",
        ]
    )

    assert rc == 0
    summary = _read_json(outdir / "adjustment_summary_adjust_cli_001.json")
    manifest = _read_json(outdir / "adjustment_manifest_adjust_cli_001.json")
    assert summary["row_count"] == 6
    assert manifest["run_id"] == "adjust_cli_001"
    assert (outdir / "adjusted_prices.parquet").exists()
    assert (outdir / "adjustment_factors.parquet").exists()
    assert (outdir / "adjustment_events_applied.parquet").exists()
    assert (outdir / "adjustment_conflicts.parquet").exists()



def test_qc_prices_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qc_prices_module: Any) -> None:
    monkeypatch.setattr(qc_prices_module, "write_parquet", _fake_write_parquet)

    raw_path = tmp_path / "prices_raw.csv"
    adjusted_path = tmp_path / "prices_adjusted.csv"
    calendar_path = tmp_path / "calendar.csv"
    config_path = tmp_path / "qc_config.json"
    output_dir = tmp_path / "qc_out"

    _raw_prices_df().to_csv(raw_path, index=False)
    adjusted = _raw_prices_df().rename(columns={"date": "date", "close": "close_adj_split"})[["symbol", "date", "close_adj_split"]].copy()
    adjusted["close_adj_total"] = adjusted["close_adj_split"]
    adjusted["ret_1d_adj_split"] = [None, -0.5, 0.1, None, 0.05, 0.0476190476]
    adjusted["ret_1d_adj_total"] = adjusted["ret_1d_adj_split"]
    adjusted.to_csv(adjusted_path, index=False)
    _calendar_df().to_csv(calendar_path, index=False)
    config_path.write_text(json.dumps(dataclasses.asdict(qc_prices_module.QCConfig()), indent=2), encoding="utf-8")

    rc = qc_prices_module.main(
        [
            "--prices-raw-path", str(raw_path),
            "--prices-adjusted-path", str(adjusted_path),
            "--calendar-path", str(calendar_path),
            "--config-path", str(config_path),
            "--output-dir", str(output_dir),
            "--run-id", "qc_cli_001",
            "--as-of-ts-utc", "2024-01-04T22:00:00Z",
        ]
    )

    assert rc == 0
    summary = _read_json(output_dir / "qc_summary.json")
    manifest = _read_json(output_dir / "manifest.json")
    assert summary["run_id"] == "qc_cli_001"
    assert manifest["run_id"] == "qc_cli_001"
    assert summary["gate"] in {"pass", "warn", "fail"}
    assert (output_dir / "qc_row_level.parquet").exists()
    assert (output_dir / "qc_symbol_level.parquet").exists()



def test_market_proxies_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], market_proxies_module: Any) -> None:
    _patch_dataframe_to_parquet(monkeypatch)
    monkeypatch.setattr(market_proxies_module, "ensure_parquet_support", lambda: None)

    prices_path = tmp_path / "adjusted_prices.csv"
    universe_path = tmp_path / "universe_history.csv"
    config_path = tmp_path / "market_proxies_config.json"
    output_dir = tmp_path / "market_out"

    _adjusted_prices_for_market_df().to_csv(prices_path, index=False)
    _universe_history_df().to_csv(universe_path, index=False)
    config = dataclasses.asdict(market_proxies_module.ProxyConfig())
    config.update(
        {
            "return_windows": [1, 2],
            "realized_vol_window": 2,
            "turnover_ref_window": 2,
            "exploratory": {"enabled": []},
            "smoothing": {"enabled": False},
            "zscore": {"enabled": False},
            "coverage": {"n_min": 1, "ratio_min": 0.5, "warn_margin_n": 0, "warn_margin_ratio": 0.0},
        }
    )
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    rc = market_proxies_module.main(
        [
            "--prices-path", str(prices_path),
            "--universe-path", str(universe_path),
            "--config-path", str(config_path),
            "--run-id", "mp_cli_001",
            "--as-of-ts-utc", "2024-01-04T22:00:00Z",
            "--output-dir", str(output_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["status"] == "ok"
    assert payload["run_id"] == "mp_cli_001"
    assert (output_dir / "summary_mp_cli_001.json").exists()
    assert (output_dir / "validation_mp_cli_001.json").exists()
    assert (output_dir / "manifest_mp_cli_001.json").exists()
