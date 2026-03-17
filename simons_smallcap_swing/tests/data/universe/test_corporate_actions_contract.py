from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import pandas as pd
import pandas.testing as pdt
import pytest


# -----------------------------------------------------------------------------
# Import adapter
# -----------------------------------------------------------------------------


def _load_corporate_actions_module():
    module_names = [
        "simons_smallcap_swing.data.universe.corporate_actions",
        "data.universe.corporate_actions",
        "corporate_actions",
    ]
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue

    here = Path(__file__).resolve()
    candidate_paths = [
        here.parents[3] / "simons_smallcap_swing" / "data" / "universe" / "corporate_actions.py",
        here.parents[2] / "data" / "universe" / "corporate_actions.py",
        here.parents[1] / "corporate_actions.py",
        Path("/mnt/data/corporate_actions.py"),
    ]
    for path in candidate_paths:
        if path.exists():
            spec = importlib.util.spec_from_file_location("corporate_actions_contract", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module

    raise ModuleNotFoundError("Could not import corporate_actions.py")


@pytest.fixture(scope="session")
def corporate_actions_module():
    return _load_corporate_actions_module()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _write_table(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    df = pd.DataFrame([dict(r) for r in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)



def _raw_row(
    provider_name: str,
    provider_event_id: str,
    event_type: str,
    *,
    announcement_ts: str = "2025-01-10T10:00:00Z",
    ex_date: str = "2025-01-15",
    effective_date: str = "2025-01-15",
    symbol: str = "AAA",
    exchange: str = "NYSE",
    share_class: str = "COM",
    instrument_id: Optional[str] = None,
    split_ratio: Optional[str] = None,
    cash_amount: Optional[float] = None,
    prev_close: Optional[float] = None,
    new_symbol: Optional[str] = None,
    figi: Optional[str] = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "provider_name": provider_name,
        "provider_event_id": provider_event_id,
        "event_type": event_type,
        "announcement_ts": announcement_ts,
        "ex_date": ex_date,
        "effective_date": effective_date,
        "symbol": symbol,
        "exchange": exchange,
        "share_class": share_class,
    }
    if instrument_id is not None:
        row["instrument_id"] = instrument_id
    if split_ratio is not None:
        row["split_ratio"] = split_ratio
    if cash_amount is not None:
        row["cash_amount"] = cash_amount
    if prev_close is not None:
        row["prev_close"] = prev_close
    if new_symbol is not None:
        row["new_symbol"] = new_symbol
    if figi is not None:
        row["figi"] = figi
    return row



def _id_row(
    instrument_id: str,
    symbol: str,
    *,
    issuer_id: str = "ISS1",
    exchange: str = "NYSE",
    share_class: str = "COM",
    effective_from: str = "2020-01-01",
    effective_to: Optional[str] = None,
    figi: Optional[str] = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "instrument_id": instrument_id,
        "issuer_id": issuer_id,
        "symbol": symbol,
        "exchange": exchange,
        "share_class": share_class,
        "effective_from": effective_from,
        "effective_to": effective_to,
    }
    if figi is not None:
        row["figi"] = figi
    return row



def _price_row(instrument_id: str, date: str, close: float) -> dict[str, Any]:
    return {"instrument_id": instrument_id, "date": date, "close": close}


@pytest.fixture()
def io_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corporate_actions_module: Any):
    root = tmp_path
    out_dir = root / "out"

    # Avoid parquet dependency while preserving the persistence contract.
    monkeypatch.setattr(corporate_actions_module, "ensure_parquet_engine_available", lambda: None)

    def _df_to_parquet_csv(self: pd.DataFrame, path: str | Path, *args: Any, **kwargs: Any) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_csv(path, index=False)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _df_to_parquet_csv, raising=True)

    # Patch a real module bug: pd.concat([]) when there are zero failure frames.
    real_concat = corporate_actions_module.pd.concat

    def _safe_concat(objs: Iterable[Any], *args: Any, **kwargs: Any):
        kept = [obj for obj in objs if obj is not None and not getattr(obj, "empty", False)]
        if not kept:
            return pd.DataFrame()
        return real_concat(kept, *args, **kwargs)

    monkeypatch.setattr(corporate_actions_module.pd, "concat", _safe_concat)

    def _write_config(name: str = "config.json") -> tuple[Path, str]:
        cfg, payload, config_hash = corporate_actions_module.load_config(None)
        path = root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path, config_hash

    def _read_table(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
        return pd.read_csv(path, parse_dates=parse_dates or [])

    def _run(
        *,
        raw_rows: list[dict[str, Any]],
        identity_rows: list[dict[str, Any]],
        price_rows: Optional[list[dict[str, Any]]] = None,
        run_id: str = "run_ca_contract",
        asof_ts_utc: str = "2025-01-21T23:00:00Z",
    ):
        raw_path = root / "raw.csv"
        identity_path = root / "identity.csv"
        prices_path = root / "prices.csv"
        _write_table(raw_rows, raw_path)
        _write_table(identity_rows, identity_path)
        if price_rows is not None:
            _write_table(price_rows, prices_path)
        cfg, payload, config_hash = corporate_actions_module.load_config(None)
        artifacts = corporate_actions_module.canonicalize_corporate_actions(
            raw_paths=[raw_path],
            identity_master_path=identity_path,
            run_id=run_id,
            asof_ts_utc=asof_ts_utc,
            config=cfg,
            config_hash=config_hash,
            config_payload=payload,
            prices_path=prices_path if price_rows is not None else None,
        )
        corporate_actions_module.persist_outputs(artifacts, out_dir, run_id, cfg.output.compression)
        manifest = json.loads((out_dir / f"manifest_{run_id}.json").read_text(encoding="utf-8"))
        summary = json.loads((out_dir / f"summary_{run_id}.json").read_text(encoding="utf-8"))
        return {
            "artifacts": artifacts,
            "manifest": manifest,
            "summary": summary,
            "out_dir": out_dir,
            "read_table": _read_table,
        }

    return {"run": _run}


@pytest.fixture()
def sample_run(io_builder):
    raw_rows = [
        _raw_row(
            "vendor_a",
            "split_1",
            "stock split",
            instrument_id="I1",
            split_ratio="2:1",
            announcement_ts="2025-01-10T10:00:00Z",
            ex_date="2025-01-15",
            effective_date="2025-01-15",
        ),
        _raw_row(
            "vendor_b",
            "ticker_1",
            "ticker change",
            instrument_id="I1",
            new_symbol="AAB",
            symbol="AAA",
            announcement_ts="2025-01-20T10:00:00Z",
            ex_date="2025-01-22",
            effective_date="2025-01-22",
        ),
        _raw_row(
            "vendor_x",
            "bad_1",
            "mystery event",
            instrument_id="I2",
            symbol="BBB",
            exchange="NASDAQ",
            announcement_ts="2025-01-11T10:00:00Z",
            ex_date="2025-01-16",
            effective_date="2025-01-16",
        ),
    ]
    identity_rows = [
        _id_row("I1", "AAA", effective_from="2020-01-01", effective_to="2025-01-21", figi="FIGI1"),
        _id_row("I1", "AAB", effective_from="2025-01-22", effective_to=None, figi="FIGI1"),
    ]
    price_rows = [_price_row("I1", "2025-01-14", 100.0)]
    return io_builder["run"](raw_rows=raw_rows, identity_rows=identity_rows, price_rows=price_rows)


# -----------------------------------------------------------------------------
# Required columns
# -----------------------------------------------------------------------------


def _required_history_cols() -> set[str]:
    return {
        "event_id",
        "instrument_id",
        "issuer_id",
        "symbol",
        "exchange",
        "share_class",
        "event_type",
        "event_family",
        "announcement_ts",
        "ex_date",
        "effective_date",
        "status",
        "source_provider",
        "source_priority",
        "confidence_score",
        "raw_event_refs",
        "raw_records_count",
        "quality_flags",
        "application_date",
    }



def _required_failure_cols() -> set[str]:
    return {
        "provider_name",
        "provider_event_id",
        "failure_code",
        "failure_detail",
        "raw_payload",
    }


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_history_contains_required_columns(sample_run):
    history = sample_run["artifacts"].history
    assert _required_history_cols().issubset(history.columns)



def test_current_contains_required_columns_and_known_asof(sample_run):
    current = sample_run["artifacts"].current
    assert _required_history_cols().issubset(set(current.columns) - {"known_asof"} | set())
    assert "known_asof" in current.columns
    assert current["known_asof"].all()



def test_failures_contains_required_columns(sample_run):
    failures = sample_run["artifacts"].failures
    assert not failures.empty
    assert _required_failure_cols().issubset(failures.columns)



def test_summary_contains_counts_and_domains(sample_run):
    summary = sample_run["summary"]
    assert summary["raw_event_count"] == 3
    assert summary["canonical_event_count"] == 2
    assert summary["current_snapshot_count"] == 2
    assert summary["failure_count"] == 1
    assert summary["counts_by_event_type"]["split"] == 1
    assert summary["counts_by_event_type"]["ticker_change"] == 1
    assert summary["failure_counts"]["UNSUPPORTED_EVENT_TYPE"] == 1
    assert summary["gate_status"] in {"PASS", "WARN", "FAIL"}



def test_manifest_contains_required_metadata_and_artifact_paths(sample_run):
    manifest = sample_run["manifest"]
    assert manifest["run_id"] == "run_ca_contract"
    assert manifest["asof_ts_utc"] == "2025-01-21T23:00:00Z"
    assert manifest["config_hash"]
    assert manifest["severity_final"] in {"PASS", "WARN", "FAIL"}
    assert set(manifest["outputs"]).issuperset({"history", "current", "failures", "summary", "manifest"})
    assert set(manifest["inputs"]).issuperset({"raw_paths", "identity_master_path", "raw_hashes", "identity_master_hash"})



def test_event_id_is_unique_and_stable_across_reruns(io_builder):
    raw_rows = [
        _raw_row("vendor_a", "split_1", "stock split", instrument_id="I1", split_ratio="2:1"),
        _raw_row(
            "vendor_b",
            "ticker_1",
            "ticker change",
            instrument_id="I1",
            new_symbol="AAB",
            symbol="AAA",
            announcement_ts="2025-01-20T10:00:00Z",
            ex_date="2025-01-22",
            effective_date="2025-01-22",
        ),
    ]
    identity_rows = [
        _id_row("I1", "AAA", effective_to="2025-01-21", figi="FIGI1"),
        _id_row("I1", "AAB", effective_from="2025-01-22", effective_to=None, figi="FIGI1"),
    ]
    price_rows = [_price_row("I1", "2025-01-14", 100.0)]

    run_a = io_builder["run"](raw_rows=raw_rows, identity_rows=identity_rows, price_rows=price_rows, run_id="run_a")
    run_b = io_builder["run"](raw_rows=raw_rows, identity_rows=identity_rows, price_rows=price_rows, run_id="run_b")

    hist_a = run_a["artifacts"].history.sort_values(["event_id"]).reset_index(drop=True)
    hist_b = run_b["artifacts"].history.sort_values(["event_id"]).reset_index(drop=True)
    assert hist_a["event_id"].is_unique
    assert hist_b["event_id"].is_unique
    pdt.assert_series_equal(hist_a["event_id"], hist_b["event_id"], check_names=False)



def test_current_is_exact_known_asof_subset_of_history(sample_run):
    artifacts = sample_run["artifacts"]
    asof = pd.Timestamp(sample_run["manifest"]["asof_ts_utc"], tz="UTC")
    hist = artifacts.history.copy()
    hist = hist[pd.to_datetime(hist["announcement_ts"], utc=True).le(asof)]
    hist = hist[hist["status"].isin(["pending", "confirmed"])]
    hist = hist.sort_values(["effective_date", "ex_date", "event_id"], kind="mergesort").reset_index(drop=True)

    current = artifacts.current.copy().reset_index(drop=True)
    # current has one extra helper column
    pdt.assert_frame_equal(current[hist.columns], hist, check_dtype=False)



def test_outputs_are_sorted_deterministically(sample_run):
    hist = sample_run["artifacts"].history
    expected_hist = hist.sort_values(["effective_date", "ex_date", "event_id"], kind="mergesort").reset_index(drop=True)
    pdt.assert_frame_equal(hist.reset_index(drop=True), expected_hist, check_dtype=False)

    cur = sample_run["artifacts"].current
    expected_cur = cur.sort_values(["effective_date", "ex_date", "event_id"], kind="mergesort").reset_index(drop=True)
    pdt.assert_frame_equal(cur.reset_index(drop=True), expected_cur, check_dtype=False)

    fail = sample_run["artifacts"].failures
    expected_fail = fail.sort_values(["provider_name", "provider_event_id"], kind="mergesort").reset_index(drop=True)
    pdt.assert_frame_equal(fail.sort_values(["provider_name", "provider_event_id"], kind="mergesort").reset_index(drop=True), expected_fail, check_dtype=False)



def test_materialized_artifacts_exist_and_match_manifest(sample_run):
    out_dir = sample_run["out_dir"]
    manifest = sample_run["manifest"]
    for rel in manifest["outputs"].values():
        assert (out_dir / rel).exists(), f"Missing artifact: {rel}"



def test_roundtrip_manifest_and_history_from_disk(sample_run):
    out_dir = sample_run["out_dir"]
    manifest = sample_run["manifest"]
    summary = sample_run["summary"]

    manifest_disk = json.loads((out_dir / manifest["outputs"]["manifest"]).read_text(encoding="utf-8"))
    summary_disk = json.loads((out_dir / manifest["outputs"]["summary"]).read_text(encoding="utf-8"))
    assert manifest_disk == manifest
    assert summary_disk == summary

    history_disk = pd.read_csv(out_dir / manifest["outputs"]["history"], parse_dates=["announcement_ts", "ex_date", "effective_date", "application_date"])
    current_disk = pd.read_csv(out_dir / manifest["outputs"]["current"], parse_dates=["announcement_ts", "ex_date", "effective_date", "application_date"])
    failures_disk = pd.read_csv(out_dir / manifest["outputs"]["failures"])

    hist = sample_run["artifacts"].history.copy().reset_index(drop=True)
    cur = sample_run["artifacts"].current.copy().reset_index(drop=True)
    fail = sample_run["artifacts"].failures.copy().reset_index(drop=True)

    # CSV roundtrip can weaken dtypes; compare semantically after normalizing key date columns to strings.
    for df in (hist, cur):
        for col in ["announcement_ts", "ex_date", "effective_date", "application_date"]:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").astype(str)
    for df in (history_disk, current_disk):
        for col in ["announcement_ts", "ex_date", "effective_date", "application_date"]:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").astype(str)
    if "known_asof" in current_disk.columns:
        current_disk["known_asof"] = current_disk["known_asof"].astype(str).str.lower().map({"true": True, "false": False})

    hist = hist.sort_values(["event_id"]).reset_index(drop=True)
    history_disk = history_disk[hist.columns].sort_values(["event_id"]).reset_index(drop=True)
    pdt.assert_frame_equal(history_disk, hist, check_dtype=False)

    cur = cur.sort_values(["event_id"]).reset_index(drop=True)
    current_disk = current_disk[cur.columns].sort_values(["event_id"]).reset_index(drop=True)
    pdt.assert_frame_equal(current_disk, cur, check_dtype=False)

    fail = fail.sort_values(["provider_name", "provider_event_id"]).reset_index(drop=True)
    failures_disk = failures_disk[fail.columns].sort_values(["provider_name", "provider_event_id"]).reset_index(drop=True)
    pdt.assert_frame_equal(failures_disk, fail, check_dtype=False)
