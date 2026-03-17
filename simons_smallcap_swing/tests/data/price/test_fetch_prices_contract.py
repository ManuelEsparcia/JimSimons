from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


REQUIRED_RAW_COLUMNS = [
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "trades_count",
    "currency",
    "source_provider",
    "provider_dataset_id",
    "provider_row_id",
    "provider_timestamp_utc",
    "ingest_ts_utc",
    "run_id",
    "config_hash",
    "universe_snapshot_id",
    "fetch_mode",
    "fallback_used",
]

REQUIRED_COVERAGE_COLUMNS = [
    "symbol",
    "expected_sessions",
    "observed_sessions",
    "pct_missing_sessions",
    "first_date",
    "last_date",
    "provider_used",
    "gap_count",
    "missing_dates",
    "severity_max",
    "fallback_used",
]

REQUIRED_FAILURE_COLUMNS = [
    "symbol",
    "trade_date",
    "error_code",
    "error_class",
    "message",
    "retry_count",
    "provider",
    "final_status",
    "severity",
]

REQUIRED_REVISION_COLUMNS = [
    "symbol",
    "trade_date",
    "field_name",
    "old_value",
    "new_value",
    "old_run_id",
    "new_run_id",
    "provider",
    "update_applied",
]


def _load_module() -> Any:
    candidates = [Path("simons_smallcap_swing/data/price/fetch_prices.py"), Path("/mnt/data/fetch_prices.py")]
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "simons_smallcap_swing" / "data" / "price" / "fetch_prices.py")
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("fetch_prices_under_test", path)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Could not locate fetch_prices.py")


@pytest.fixture(scope="module")
def fetch_prices_module() -> Any:
    return _load_module()


@pytest.fixture()
def contract_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fetch_prices_module: Any) -> dict[str, Any]:
    fp = fetch_prices_module

    providers_root = tmp_path / "providers"
    primary_root = providers_root / "primary"
    fallback_root = providers_root / "fallback"
    primary_root.mkdir(parents=True)
    fallback_root.mkdir(parents=True)

    # Primary has AAPL only; fallback has MSFT only, forcing a real fallback path.
    aapl = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-02T21:00:00Z",
                "2024-01-03T21:00:00Z",
                "2024-01-04T21:00:00Z",
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.5, 100.5, 101.5],
            "close": [100.5, 101.5, 102.5],
            "volume": [1_000, 1_100, 1_200],
            "vwap": [100.3, 101.2, 102.1],
            "trades_count": [10, 11, 12],
            "currency": ["USD", "USD", "USD"],
        }
    )
    msft = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-02T21:00:00Z",
                "2024-01-03T21:00:00Z",
                "2024-01-04T21:00:00Z",
            ],
            "open": [200.0, 201.0, 202.0],
            "high": [201.0, 202.0, 203.0],
            "low": [199.0, 200.0, 201.0],
            "close": [200.5, 201.5, 202.5],
            "volume": [2_000, 2_100, 2_200],
            "vwap": [200.2, 201.2, 202.2],
            "trades_count": [20, 21, 22],
            "currency": ["USD", "USD", "USD"],
        }
    )
    aapl.to_csv(primary_root / "AAPL.csv", index=False)
    msft.to_csv(fallback_root / "MSFT.csv", index=False)

    universe_path = tmp_path / "universe.csv"
    pd.DataFrame({"symbol": ["AAPL", "MSFT"]}).to_csv(universe_path, index=False)

    calendar_path = tmp_path / "calendar.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "is_session": [True, True, True],
        }
    ).to_csv(calendar_path, index=False)

    output_root = tmp_path / "out"
    provider_config_path = tmp_path / "provider_config.json"
    provider_config_path.write_text(
        json.dumps(
            {
                "fetch": {
                    "mode": "incremental",
                    "provider_priority": ["primary", "fallback"],
                    "allow_fallback": True,
                },
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
                        "params": {
                            "root_dir": str(primary_root),
                            "file_pattern": "{symbol}.csv",
                            "format": "csv",
                        },
                    },
                    "fallback": {
                        "kind": "csv_directory",
                        "name": "fallback_csv",
                        "dataset_id": "FALLBACK_DS",
                        "params": {
                            "root_dir": str(fallback_root),
                            "file_pattern": "{symbol}.csv",
                            "format": "csv",
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_write_parquet_deduped(path: Path, frame: pd.DataFrame, dedupe_keys=None, overwrite=False, compression="snappy"):
        path.parent.mkdir(parents=True, exist_ok=True)
        out = frame.copy()
        if dedupe_keys:
            out = out.sort_values(list(dedupe_keys), kind="stable").drop_duplicates(subset=list(dedupe_keys), keep="last")
        out.to_csv(path, index=False)

    def fake_to_parquet(self: pd.DataFrame, path: str | Path, *args, **kwargs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_csv(path, index=False)

    monkeypatch.setattr(fp, "write_parquet_deduped", fake_write_parquet_deduped)
    monkeypatch.setattr(fp, "require_parquet_engine", lambda: None)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet, raising=True)
    monkeypatch.setattr(fp, "git_code_version", lambda: "test-version")

    artifacts = fp.run_fetch_prices(
        universe_snapshot_path=universe_path,
        calendar_path=calendar_path,
        provider_config_path=provider_config_path,
        start_date="2024-01-02",
        end_date="2024-01-04",
        frequency="1d",
        fetch_mode="incremental",
        run_id="run_price_contract",
        asof_ts_utc="2024-01-05T00:00:00Z",
        config_path=None,
    )

    return {
        "module": fp,
        "artifacts": artifacts,
        "output_root": output_root,
        "run_id": "run_price_contract",
    }


def _read_csvish(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise AssertionError(f"Expected artifact does not exist: {path}")
    return pd.read_csv(path, parse_dates=parse_dates or [])


def _assert_has_columns(frame: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in frame.columns]
    assert not missing, f"Missing required columns: {missing}"


def test_raw_output_contains_required_columns(contract_run: dict[str, Any]) -> None:
    raw = contract_run["artifacts"].raw
    _assert_has_columns(raw, REQUIRED_RAW_COLUMNS)


def test_pk_raw_is_unique_by_symbol_trade_date_provider_run(contract_run: dict[str, Any]) -> None:
    raw = contract_run["artifacts"].raw
    dup = raw.duplicated(subset=["symbol", "trade_date", "source_provider", "run_id"], keep=False)
    assert not dup.any(), raw.loc[dup]


def test_trade_date_and_ingest_ts_are_both_present_and_parseable(contract_run: dict[str, Any]) -> None:
    raw = contract_run["artifacts"].raw.copy()
    trade = pd.to_datetime(raw["trade_date"], errors="coerce")
    ingest = pd.to_datetime(raw["ingest_ts_utc"], errors="coerce", utc=True)
    provider_ts = pd.to_datetime(raw["provider_timestamp_utc"], errors="coerce", utc=True)
    assert trade.notna().all()
    assert ingest.notna().all()
    assert provider_ts.notna().all()


def test_coverage_report_contains_required_fields(contract_run: dict[str, Any]) -> None:
    coverage = contract_run["artifacts"].coverage
    _assert_has_columns(coverage, REQUIRED_COVERAGE_COLUMNS)
    assert set(coverage["severity_max"].astype(str)).issubset({"INFO", "WARN", "FAIL"})


def test_failures_report_contains_required_fields(contract_run: dict[str, Any]) -> None:
    failures = contract_run["artifacts"].failures
    _assert_has_columns(failures, REQUIRED_FAILURE_COLUMNS)
    assert not failures.empty
    assert "FALLBACK_USED" in set(failures["error_code"].astype(str))


def test_revisions_report_contains_required_fields_even_when_empty(contract_run: dict[str, Any]) -> None:
    revisions = contract_run["artifacts"].revisions
    _assert_has_columns(revisions, REQUIRED_REVISION_COLUMNS)


def test_manifest_contains_required_metadata_and_counts(contract_run: dict[str, Any]) -> None:
    artifacts = contract_run["artifacts"]
    manifest = artifacts.manifest
    required_keys = {
        "run_id",
        "provider_priority",
        "fetch_mode",
        "start_date",
        "end_date",
        "frequency",
        "config_hash",
        "universe_snapshot_id",
        "universe_snapshot_path",
        "calendar_path",
        "provider_config_path",
        "code_version",
        "tickers_requested",
        "tickers_ok",
        "tickers_failed",
        "rows_ingested",
        "asof_ts_utc",
        "created_ts_utc",
        "output_root",
    }
    assert required_keys.issubset(manifest.keys())
    assert manifest["run_id"] == contract_run["run_id"]
    assert manifest["rows_ingested"] == len(artifacts.raw)
    assert manifest["tickers_requested"] == 2
    assert manifest["tickers_ok"] == 2
    assert manifest["tickers_failed"] == 0


def test_raw_and_canonical_layers_are_not_confused(contract_run: dict[str, Any]) -> None:
    artifacts = contract_run["artifacts"]
    raw = artifacts.raw
    canonical = artifacts.canonical
    assert list(canonical.columns) == REQUIRED_RAW_COLUMNS
    assert set(raw["run_id"].astype(str)) == {contract_run["run_id"]}
    assert set(canonical["run_id"].astype(str)) == {contract_run["run_id"]}
    assert canonical[["symbol", "trade_date", "source_provider"]].duplicated().sum() == 0


def test_outputs_are_sorted_deterministically(contract_run: dict[str, Any]) -> None:
    artifacts = contract_run["artifacts"]
    raw = artifacts.raw.reset_index(drop=True)
    expected_raw = raw.sort_values(["symbol", "trade_date"], kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(raw, expected_raw)

    coverage = artifacts.coverage.reset_index(drop=True)
    expected_cov = coverage.sort_values(["symbol"], kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(coverage, expected_cov)

    failures = artifacts.failures.reset_index(drop=True)
    expected_fail = failures.sort_values(["symbol", "trade_date", "severity", "error_code"], kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(failures, expected_fail)


def test_persisted_artifacts_roundtrip_and_paths(contract_run: dict[str, Any]) -> None:
    artifacts = contract_run["artifacts"]
    output_root = contract_run["output_root"]
    run_id = contract_run["run_id"]

    raw_path = output_root / f"raw/provider=primary/freq=1d/run_id={run_id}/prices.parquet"
    canonical_path = output_root / "canonical/canonical_prices.parquet"
    coverage_path = output_root / f"reports/coverage_{run_id}.parquet"
    failures_path = output_root / f"reports/failures_{run_id}.parquet"
    revisions_path = output_root / f"reports/revisions_{run_id}.parquet"
    summary_path = output_root / f"reports/summary_{run_id}.json"
    manifest_path = output_root / f"reports/manifest_{run_id}.json"

    raw_disk = _read_csvish(raw_path, parse_dates=["trade_date", "provider_timestamp_utc", "ingest_ts_utc"])
    canonical_disk = _read_csvish(canonical_path, parse_dates=["trade_date", "provider_timestamp_utc", "ingest_ts_utc"])
    coverage_disk = _read_csvish(coverage_path, parse_dates=["first_date", "last_date"])
    failures_disk = _read_csvish(failures_path, parse_dates=["trade_date"])
    revisions_disk = _read_csvish(revisions_path, parse_dates=["trade_date"])

    # Semantic comparisons, not byte-for-byte dtype equality.
    assert len(raw_disk) == len(artifacts.raw)
    assert len(canonical_disk) == len(artifacts.canonical)
    assert len(coverage_disk) == len(artifacts.coverage)
    assert len(failures_disk) == len(artifacts.failures)
    assert len(revisions_disk) == len(artifacts.revisions)

    summary_disk = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert summary_disk["run_id"] == run_id
    assert manifest_disk["run_id"] == run_id
    assert manifest_disk["rows_ingested"] == len(artifacts.raw)
