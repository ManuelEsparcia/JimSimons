from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# Import adapter
# -----------------------------------------------------------------------------


def _load_fetch_prices_module():
    module_names = [
        "simons_smallcap_swing.data.price.fetch_prices",
        "data.price.fetch_prices",
        "fetch_prices",
    ]
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue

    here = Path(__file__).resolve()
    candidate_paths = [
        here.parents[3] / "simons_smallcap_swing" / "data" / "price" / "fetch_prices.py",
        here.parents[2] / "data" / "price" / "fetch_prices.py",
        here.parents[1] / "fetch_prices.py",
        Path("/mnt/data/fetch_prices.py"),
    ]
    for path in candidate_paths:
        if path.exists():
            spec = importlib.util.spec_from_file_location("fetch_prices", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module

    raise ModuleNotFoundError("Could not import fetch_prices.py from repo paths or /mnt/data.")


@pytest.fixture(scope="session")
def fetch_prices_module():
    return _load_fetch_prices_module()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


class _Client:
    def __init__(self, alias: str, responses: List[Any], dataset_id: Optional[str] = None):
        self.provider_alias = alias
        self.provider_name = alias.upper()
        self.dataset_id = dataset_id or f"ds_{alias}"
        self._responses = list(responses)
        self.calls: List[tuple[str, pd.Timestamp, pd.Timestamp, str]] = []

    def fetch_daily_bars(self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp, frequency: str) -> pd.DataFrame:
        self.calls.append((symbol, start_date, end_date, frequency))
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_normalize_symbol_trims_and_uppercases(fetch_prices_module):
    module = fetch_prices_module

    assert module.normalize_symbol(" aapl ") == "AAPL"
    assert module.normalize_symbol("msFt") == "MSFT"
    assert module.normalize_symbol(" brk.b ") == "BRK.B"



def test_get_fetch_window_incremental_skips_already_fetched_range(fetch_prices_module):
    module = fetch_prices_module

    window = module.UniverseWindow(
        symbol="AAPL",
        start_date=pd.Timestamp("2024-01-02"),
        end_date=pd.Timestamp("2024-01-05"),
        expected_sessions=[pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")],
    )
    existing = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "source_provider": ["primary", "primary"],
        }
    )

    fetch_window = module.get_fetch_window(window, module.FetchMode.INCREMENTAL.value, existing)

    assert fetch_window is not None
    start, end, expected, _ = fetch_window
    assert start == pd.Timestamp("2024-01-04")
    assert end == pd.Timestamp("2024-01-05")
    assert expected == [pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")]



def test_get_fetch_window_backfill_returns_missing_span_only(fetch_prices_module):
    module = fetch_prices_module

    window = module.UniverseWindow(
        symbol="AAPL",
        start_date=pd.Timestamp("2024-01-02"),
        end_date=pd.Timestamp("2024-01-05"),
        expected_sessions=[pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")],
    )
    existing = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "source_provider": ["primary", "primary"],
        }
    )

    fetch_window = module.get_fetch_window(window, module.FetchMode.BACKFILL.value, existing)

    assert fetch_window is not None
    start, end, expected, _ = fetch_window
    assert start == pd.Timestamp("2024-01-03")
    assert end == pd.Timestamp("2024-01-05")
    assert expected == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-05")]



def test_fetch_symbol_with_fallback_retries_then_uses_fallback(fetch_prices_module, monkeypatch):
    module = fetch_prices_module

    primary = _Client(
        "primary",
        responses=[
            module.RetriableProviderError("timeout"),
            module.RetriableProviderError("503 upstream"),
            module.PermanentProviderError("invalid symbol on primary"),
        ],
    )
    fallback = _Client(
        "fallback",
        responses=[
            pd.DataFrame(
                {
                    "date": ["2024-01-03"],
                    "open": [10.0],
                    "high": [11.0],
                    "low": [9.5],
                    "close": [10.5],
                    "volume": [1000],
                }
            )
        ],
    )

    sleep_calls: List[int] = []
    monkeypatch.setattr(module, "backoff_sleep", lambda policy, retry_idx: sleep_calls.append(retry_idx))

    result = module.fetch_symbol_with_fallback(
        clients=[primary, fallback],
        window=module.UniverseWindow(
            symbol="AAPL",
            start_date=pd.Timestamp("2024-01-03"),
            end_date=pd.Timestamp("2024-01-03"),
            expected_sessions=[pd.Timestamp("2024-01-03")],
        ),
        frequency="1d",
        retry_policy=module.RetryPolicy(max_retries=2, base_delay_sec=0.001, cap_delay_sec=0.001),
        rate_limiter=module.RateLimiter(0.0),
    )

    assert result.frame is not None
    assert result.provider_alias == "fallback"
    assert bool(result.fallback_used) is True
    assert sleep_calls == [0, 1]
    assert len(primary.calls) == 3
    assert len(fallback.calls) == 1



def test_map_trade_dates_uses_market_timezone_by_default(fetch_prices_module):
    module = fetch_prices_module
    cfg = module.FetchPricesConfig()

    frame = pd.DataFrame({"timestamp": ["2024-01-03T01:30:00Z", "2024-01-03T21:00:00Z"]})
    trade_date, provider_ts = module.map_trade_dates(frame, cfg, provider_timezone=None)

    assert list(trade_date.dt.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03"]
    assert str(provider_ts.dt.tz) == "UTC"



def test_normalize_provider_frame_maps_aliases_and_metadata(fetch_prices_module):
    module = fetch_prices_module
    cfg = module.FetchPricesConfig()
    attempt = module.FetchAttemptResult(
        frame=None,
        provider_alias="fallback",
        provider_name="FALLBACK",
        dataset_id="dataset_v1",
        retry_count=1,
        error_code=None,
        error_class=None,
        message=None,
        fallback_used=True,
    )

    raw = pd.DataFrame(
        {
            "ticker": [" aapl ", "msft"],
            "timestamp": ["2024-01-03T21:00:00Z", "2024-01-03T21:00:00Z"],
            "o": [10.0, 20.0],
            "h": [11.0, 21.0],
            "l": [9.5, 19.5],
            "c": [10.5, 20.5],
            "v": [1000, 2000],
            "vw": [10.4, 20.4],
            "n": [12, 22],
        }
    )

    out = module.normalize_provider_frame(
        raw_frame=raw,
        symbol="AAPL",
        attempt=attempt,
        cfg=cfg,
        config_hash="cfg_123",
        universe_snapshot_id="universe_123",
        fetch_mode=module.FetchMode.FULL_REFRESH.value,
        ingest_ts_utc="2026-03-16T10:00:00+00:00",
        provider_timezone=None,
    )

    assert list(out.columns) == module.REQUIRED_CANONICAL_COLUMNS
    assert len(out) == 1
    row = out.iloc[0]
    assert row["symbol"] == "AAPL"
    assert row["source_provider"] == "fallback"
    assert row["provider_dataset_id"] == "dataset_v1"
    assert row["fallback_used"] in {True, 1}
    assert pd.Timestamp(row["trade_date"]) == pd.Timestamp("2024-01-03")
    assert float(row["vwap"]) == pytest.approx(10.4)
    assert float(row["trades_count"]) == pytest.approx(12.0)



def test_validate_ingestion_frame_drops_invalid_rows_and_keeps_valid_row(fetch_prices_module):
    module = fetch_prices_module
    cfg = module.FetchPricesConfig()

    frame = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "trade_date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
            "open": [10.0, 10.0, 10.0],
            "high": [11.0, 9.0, 11.0],
            "low": [9.5, 9.5, 9.5],
            "close": [10.5, 10.2, 10.1],
            "volume": [1000, 1000, -1],
            "vwap": [10.4, 10.1, 10.0],
            "trades_count": [10, 10, 10],
            "currency": ["USD", "USD", "USD"],
            "source_provider": ["primary", "primary", "primary"],
            "provider_dataset_id": ["ds", "ds", "ds"],
            "provider_row_id": ["1", "2", "3"],
            "provider_timestamp_utc": pd.to_datetime(["2024-01-03T21:00:00Z"] * 3, utc=True),
            "ingest_ts_utc": ["2026-03-16T10:00:00+00:00"] * 3,
            "run_id": ["tmp"] * 3,
            "config_hash": ["cfg"] * 3,
            "universe_snapshot_id": ["u"] * 3,
            "fetch_mode": [module.FetchMode.FULL_REFRESH.value] * 3,
            "fallback_used": [False] * 3,
        }
    )

    out = module.validate_ingestion_frame(
        frame=frame,
        window=module.UniverseWindow(
            symbol="AAPL",
            start_date=pd.Timestamp("2024-01-03"),
            end_date=pd.Timestamp("2024-01-05"),
            expected_sessions=[pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")],
        ),
        run_id="pytest_run",
        cfg=cfg,
    )

    assert len(out.clean) == 1
    assert pd.Timestamp(out.clean.iloc[0]["trade_date"]) == pd.Timestamp("2024-01-03")
    codes = {row["error_code"] for row in out.failures}
    assert module.FailureCode.INVALID_PRICE_GEOMETRY.value in codes
    assert module.FailureCode.INVALID_VOLUME_VALUE.value in codes



def test_build_revisions_report_detects_changed_field(fetch_prices_module):
    module = fetch_prices_module

    existing = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "trade_date": pd.to_datetime(["2024-01-03"]),
            "source_provider": ["primary"],
            "run_id": ["old_run"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.5],
            "close": [10.5],
            "volume": [1000],
            "vwap": [10.4],
            "trades_count": [10],
            "currency": ["USD"],
            "provider_dataset_id": ["ds"],
        }
    )
    new = existing.copy()
    new["close"] = [10.8]
    new["run_id"] = ["new_run"]

    revisions = module.build_revisions_report(existing, new, run_id="new_run")

    assert len(revisions) == 1
    row = revisions.iloc[0]
    assert row["field_name"] == "close"
    assert float(row["old_value"]) == pytest.approx(10.5)
    assert float(row["new_value"]) == pytest.approx(10.8)
    assert row["old_run_id"] == "old_run"
    assert row["new_run_id"] == "new_run"



def test_merge_with_existing_snapshot_incremental_does_not_overwrite_existing_range(fetch_prices_module):
    module = fetch_prices_module
    cfg = module.FetchPricesConfig(
        persistence=dataclasses.replace(module.PersistencePolicy(), incremental_update_existing_range=False)
    )

    existing = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "trade_date": pd.to_datetime(["2024-01-03"]),
            "open": [10.0],
            "high": [11.0],
            "low": [9.5],
            "close": [10.5],
            "volume": [1000],
            "vwap": [10.4],
            "trades_count": [10],
            "currency": ["USD"],
            "source_provider": ["primary"],
            "provider_dataset_id": ["ds"],
            "provider_row_id": ["1"],
            "provider_timestamp_utc": pd.to_datetime(["2024-01-03T21:00:00Z"], utc=True),
            "ingest_ts_utc": ["2026-03-16T10:00:00+00:00"],
            "run_id": ["old_run"],
            "config_hash": ["cfg"],
            "universe_snapshot_id": ["u"],
            "fetch_mode": [module.FetchMode.INCREMENTAL.value],
            "fallback_used": [False],
        }
    )
    new = existing.copy()
    new["close"] = [10.8]
    new["run_id"] = ["new_run"]

    canonical, revisions = module.merge_with_existing_snapshot(existing, new, module.FetchMode.INCREMENTAL.value, cfg)

    assert len(canonical) == 1
    assert float(canonical.iloc[0]["close"]) == pytest.approx(10.5)
    assert len(revisions) == 1
    assert bool(revisions.iloc[0]["update_applied"]) is False



def test_run_fetch_prices_full_refresh_uses_fallback_and_does_not_abort_on_primary_failure(fetch_prices_module, monkeypatch, tmp_path):
    module = fetch_prices_module

    persistence = dataclasses.replace(
        module.PersistencePolicy(),
        output_root=str(tmp_path / "out"),
        persist_raw=False,
        persist_canonical=False,
    )
    cfg = module.FetchPricesConfig(
        persistence=persistence,
        fetch=dataclasses.replace(module.FetchPolicy(), mode=module.FetchMode.FULL_REFRESH.value, provider_priority=["primary", "fallback"]),
        rate_limit=module.RateLimitPolicy(min_interval_sec=0.0),
    )

    providers = {
        "primary": module.ProviderSpec(alias="primary", kind="custom", name="PrimaryProvider", params={"class_path": "dummy.Primary"}, enabled=True),
        "fallback": module.ProviderSpec(alias="fallback", kind="custom", name="FallbackProvider", params={"class_path": "dummy.Fallback"}, enabled=True),
    }

    monkeypatch.setattr(module, "load_config", lambda config_path, provider_config_path: (cfg, providers, {"providers": ["primary", "fallback"]}, "cfg_hash_123"))
    monkeypatch.setattr(
        module,
        "load_universe_snapshot",
        lambda path, start_date, end_date, cfg: (
            pd.DataFrame({"symbol": ["AAPL", "MSFT"], "date": [pd.NaT, pd.NaT], "is_eligible": [True, True]}),
            "universe_snapshot_123",
        ),
    )
    monkeypatch.setattr(
        module,
        "load_market_calendar",
        lambda path, start_date, end_date: pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03"])),
    )
    monkeypatch.setattr(module, "git_code_version", lambda: "git:test")

    primary_client = _Client(
        "primary",
        responses=[
            pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03"],
                    "open": [10.0, 10.5],
                    "high": [11.0, 11.2],
                    "low": [9.5, 10.2],
                    "close": [10.6, 11.0],
                    "volume": [1000, 1200],
                }
            ),
            module.PermanentProviderError("invalid symbol on primary"),
        ],
    )
    fallback_client = _Client(
        "fallback",
        responses=[
            pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03"],
                    "open": [20.0, 20.5],
                    "high": [21.0, 21.2],
                    "low": [19.5, 20.2],
                    "close": [20.6, 21.0],
                    "volume": [2000, 2200],
                }
            )
        ],
    )

    def fake_build_provider_client(spec):
        return primary_client if spec.alias == "primary" else fallback_client

    monkeypatch.setattr(module, "build_provider_client", fake_build_provider_client)
    monkeypatch.setattr(module, "load_existing_canonical", lambda path: pd.DataFrame(columns=module.REQUIRED_CANONICAL_COLUMNS))
    monkeypatch.setattr(module, "write_parquet_deduped", lambda path, frame, **kwargs: Path(path).parent.mkdir(parents=True, exist_ok=True) or frame.to_csv(path, index=False))
    monkeypatch.setattr(module, "require_parquet_engine", lambda: None)

    original_to_parquet = pd.DataFrame.to_parquet
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, *args, **kwargs: self.to_csv(path, index=False))
    try:
        artifacts = module.run_fetch_prices(
            universe_snapshot_path=tmp_path / "universe.csv",
            calendar_path=tmp_path / "calendar.csv",
            provider_config_path=tmp_path / "providers.yaml",
            start_date="2024-01-02",
            end_date="2024-01-03",
            frequency="1d",
            fetch_mode=module.FetchMode.FULL_REFRESH.value,
            run_id="pytest_run",
            asof_ts_utc="2026-03-16T10:00:00Z",
            config_path=None,
        )
    finally:
        monkeypatch.setattr(pd.DataFrame, "to_parquet", original_to_parquet)

    assert len(artifacts.raw) == 4
    assert set(artifacts.raw["symbol"]) == {"AAPL", "MSFT"}
    assert artifacts.summary["tickers_ok"] == 2
    assert artifacts.summary["tickers_failed"] == 0
    assert artifacts.summary["fallback_usage_rate"] == pytest.approx(0.5)
    assert {"AAPL", "MSFT"} == set(artifacts.coverage["symbol"])
    assert (artifacts.failures["error_code"] == module.FailureCode.FALLBACK_USED.value).any()
    assert artifacts.manifest["provider_priority"] == ["primary", "fallback"]

    summary_path = tmp_path / "out" / "reports" / "summary_pytest_run.json"
    manifest_path = tmp_path / "out" / "reports" / "manifest_pytest_run.json"
    assert summary_path.exists()
    assert manifest_path.exists()
    assert json.loads(summary_path.read_text())["tickers_ok"] == 2

