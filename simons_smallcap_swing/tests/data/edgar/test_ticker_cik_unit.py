from __future__ import annotations

import copy
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd
import pandas.testing as pdt
import pytest


HISTORY_REQUIRED_COLUMNS = {
    "symbol",
    "cik",
    "source",
    "source_priority",
    "confidence_score",
    "metadata_consistency",
    "temporal_consistency",
    "effective_from",
    "effective_to",
    "resolution_status",
    "heuristic_flag",
    "issuer_name",
    "exchange",
    "share_class",
    "evidence_hash",
    "mapping_version",
    "lineage_count",
    "corroboration_count",
}

CURRENT_REQUIRED_COLUMNS = HISTORY_REQUIRED_COLUMNS | {"current_candidate_flag", "is_active"}
CONFLICT_REQUIRED_COLUMNS = {
    "symbol",
    "cik",
    "effective_from",
    "effective_to",
    "conflict_class",
    "conflict_status",
    "source",
    "details",
}


# -----------------------------------------------------------------------------
# Import adapter
# -----------------------------------------------------------------------------


def _load_ticker_cik_module():
    module_names = [
        "simons_smallcap_swing.data.edgar.ticker_cik",
        "data.edgar.ticker_cik",
        "ticker_cik",
    ]
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue

    here = Path(__file__).resolve()
    candidate_paths = [
        here.parents[3] / "simons_smallcap_swing" / "data" / "edgar" / "ticker_cik.py",
        here.parents[2] / "data" / "edgar" / "ticker_cik.py",
        here.parents[1] / "ticker_cik.py",
        Path("/mnt/data/ticker_cik.py"),
    ]
    for path in candidate_paths:
        if path.exists():
            spec = importlib.util.spec_from_file_location("ticker_cik", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module

    raise ModuleNotFoundError(
        "Could not import ticker_cik.py. Expected it at "
        "simons_smallcap_swing.data.edgar.ticker_cik or in a nearby repo path."
    )


@pytest.fixture(scope="session")
def ticker_cik_module():
    return _load_ticker_cik_module()


# -----------------------------------------------------------------------------
# Synthetic builders
# -----------------------------------------------------------------------------


def _sec_row(
    symbol: str,
    cik: str | int,
    *,
    issuer_name: str = "ACME CORP",
    exchange: str = "NASDAQ",
    share_class: str = "COMMON",
    effective_from: str = "2020-01-01",
    effective_to: Optional[str] = None,
    confidence_score: Optional[float] = None,
    ingest_ts_utc: str = "2025-03-10T12:00:00Z",
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "ticker": symbol,
        "cik_str": cik,
        "company_name": issuer_name,
        "primary_exchange": exchange,
        "security_class": share_class,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "ingest_ts_utc": ingest_ts_utc,
    }
    if confidence_score is not None:
        row["confidence_score"] = confidence_score
    return row



def _internal_row(
    symbol: str,
    cik: str | int,
    *,
    issuer_name: str = "ACME CORP",
    exchange: str = "NASDAQ",
    share_class: str = "COMMON",
    effective_from: str = "2020-01-01",
    effective_to: Optional[str] = None,
    confidence_score: Optional[float] = None,
    ingest_ts_utc: str = "2025-03-10T12:00:00Z",
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "symbol": symbol,
        "cik": cik,
        "issuer_name": issuer_name,
        "exchange": exchange,
        "share_class": share_class,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "ingest_ts_utc": ingest_ts_utc,
    }
    if confidence_score is not None:
        row["confidence_score"] = confidence_score
    return row


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _default_config(module: Any) -> Dict[str, Any]:
    return copy.deepcopy(getattr(module, "DEFAULT_CONFIG"))



def _write_table(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    df = pd.DataFrame([dict(r) for r in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)



def _run_pipeline(
    module: Any,
    tmp_path: Path,
    *,
    sec_rows: Iterable[Mapping[str, Any]],
    internal_rows: Optional[Iterable[Mapping[str, Any]]] = None,
    config_patch: Optional[Mapping[str, Any]] = None,
    asof: str = "2025-03-10",
    run_id: str = "pytest_ticker_cik",
):
    sec_path = tmp_path / "sec_source.csv"
    _write_table(sec_rows, sec_path)

    internal_path: Optional[Path] = None
    if internal_rows is not None:
        internal_path = tmp_path / "internal_master.csv"
        _write_table(internal_rows, internal_path)

    cfg = _default_config(module)
    cfg["storage"]["output_root"] = str(tmp_path / "out")
    cfg["storage"]["allow_csv_fallback"] = True
    cfg["identity"]["strict_abort_on_unresolved_conflict"] = False
    cfg["identity"]["strict_abort_on_schema_failure"] = True
    if config_patch:
        # use module's own deep merge if available to stay aligned with implementation
        deep_merge = getattr(module, "_deep_merge", None)
        if callable(deep_merge):
            cfg = deep_merge(cfg, dict(config_patch))
        else:
            for k, v in config_patch.items():
                cfg[k] = v

    config_path = tmp_path / "ticker_cik_config.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")

    result = module.run_ticker_cik(
        sec_source_path=str(sec_path),
        internal_master_path=str(internal_path) if internal_path is not None else None,
        config_path=str(config_path),
        run_id=run_id,
        asof=asof,
    )

    history = result["history"].copy()
    current = result["current"].copy()
    conflicts = result["conflicts"].copy()
    metrics = result["metrics"].copy()
    for df in [history, current, conflicts]:
        for col in ["effective_from", "effective_to"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return result, history, current, conflicts, metrics, cfg



def _sort_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sort_values(["symbol", "effective_from", "cik", "source"], kind="mergesort").reset_index(drop=True)



def _sort_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    cols = [c for c in ["symbol", "effective_from", "conflict_class", "conflict_status", "source"] if c in df.columns]
    return df.sort_values(cols, kind="mergesort").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_prepare_raw_input_canonizes_symbol_and_cik(ticker_cik_module):
    module = ticker_cik_module
    df = pd.DataFrame(
        [
            {
                "ticker": " aapl ",
                "cik_str": "320193",
                "company_name": " Apple Inc ",
                "primary_exchange": " nasdaq ",
                "security_class": " common ",
            }
        ]
    )
    failures: List[Dict[str, Any]] = []
    out = module._prepare_raw_input(df, "official_SEC_mapping", _default_config(module), failures)

    assert failures == []
    assert len(out) == 1
    row = out.iloc[0]
    assert row["symbol"] == "AAPL"
    assert row["cik"] == "0000320193"
    assert row["issuer_name"] == "APPLE INC"
    assert row["exchange"] == "NASDAQ"
    assert row["share_class"] == "COMMON"
    assert pd.Timestamp(row["effective_from"]) == pd.Timestamp("1900-01-01")
    assert row["resolution_status"] == "exact"
    assert bool(row["is_active"]) is True



def test_official_sec_precedence_over_internal_master_on_simple_conflict(ticker_cik_module, tmp_path: Path):
    module = ticker_cik_module
    sec_rows = [
        _sec_row("ABC", "1111111111", issuer_name="ABC INC", effective_from="2020-01-01"),
    ]
    internal_rows = [
        _internal_row("ABC", "2222222222", issuer_name="ABC INC", effective_from="2020-01-01"),
    ]

    _, history, current, conflicts, _, _ = _run_pipeline(
        module,
        tmp_path,
        sec_rows=sec_rows,
        internal_rows=internal_rows,
        asof="2025-03-10",
    )

    assert HISTORY_REQUIRED_COLUMNS.issubset(set(history.columns))
    assert CURRENT_REQUIRED_COLUMNS.issubset(set(current.columns))
    assert CONFLICT_REQUIRED_COLUMNS.issubset(set(conflicts.columns))

    assert len(current) == 1
    row = current.iloc[0]
    assert row["symbol"] == "ABC"
    assert row["cik"] == "1111111111"
    assert row["source"] == "official_SEC_mapping"
    assert float(row["confidence_score"]) >= 0.99 - 1e-12

    assert not conflicts.empty
    source_conflicts = conflicts[conflicts["conflict_class"] == "source_conflict"]
    assert not source_conflicts.empty
    assert set(source_conflicts["conflict_status"].astype(str)) == {"resolved_by_precedence"}



def test_unresolved_identity_conflict_is_detected_and_excluded_from_current(ticker_cik_module, tmp_path: Path):
    module = ticker_cik_module
    sec_rows = [
        _sec_row("XYZ", "3333333333", issuer_name="XYZ INC", effective_from="2020-01-01", ingest_ts_utc="2025-03-10T10:00:00Z"),
        _sec_row("XYZ", "4444444444", issuer_name="XYZ INC", effective_from="2020-01-01", ingest_ts_utc="2025-03-10T10:00:00Z"),
    ]

    _, history, current, conflicts, _, _ = _run_pipeline(module, tmp_path, sec_rows=sec_rows, asof="2025-03-10")

    assert history.empty
    assert current.empty
    assert not conflicts.empty
    assert "unresolved_identity_conflict" in set(conflicts["conflict_class"].astype(str))
    unresolved = conflicts[conflicts["conflict_class"] == "unresolved_identity_conflict"]
    assert set(unresolved["conflict_status"].astype(str)) == {"unresolved"}



def test_ticker_change_closes_old_window_and_activates_new_symbol(ticker_cik_module, tmp_path: Path):
    module = ticker_cik_module
    sec_rows = [
        _sec_row("OLD", "5555555555", issuer_name="OMEGA CORP", share_class="COMMON", effective_from="2020-01-01", effective_to=None),
        _sec_row("NEW", "5555555555", issuer_name="OMEGA CORP", share_class="COMMON", effective_from="2022-01-03", effective_to=None),
    ]

    _, history, current, conflicts, _, _ = _run_pipeline(module, tmp_path, sec_rows=sec_rows, asof="2025-03-10")

    history = _sort_history(history)
    assert set(history["symbol"].tolist()) == {"OLD", "NEW"}

    old_row = history[history["symbol"] == "OLD"].iloc[0]
    new_row = history[history["symbol"] == "NEW"].iloc[0]
    assert pd.Timestamp(old_row["effective_from"]) == pd.Timestamp("2020-01-01")
    assert pd.Timestamp(old_row["effective_to"]) == pd.Timestamp("2022-01-03")
    assert pd.Timestamp(new_row["effective_from"]) == pd.Timestamp("2022-01-03")
    assert pd.isna(new_row["effective_to"])

    assert len(current) == 1
    assert current.iloc[0]["symbol"] == "NEW"
    assert current.iloc[0]["cik"] == "5555555555"

    ticker_change_conflicts = conflicts[conflicts["conflict_class"] == "ticker_change"]
    assert not ticker_change_conflicts.empty
    assert set(ticker_change_conflicts["conflict_status"].astype(str)) == {"resolved_by_historification"}



def test_resolve_pit_identity_returns_single_active_mapping_for_given_date(ticker_cik_module, tmp_path: Path):
    module = ticker_cik_module
    sec_rows = [
        _sec_row("OLD", "5555555555", issuer_name="OMEGA CORP", share_class="COMMON", effective_from="2020-01-01", effective_to=None),
        _sec_row("NEW", "5555555555", issuer_name="OMEGA CORP", share_class="COMMON", effective_from="2022-01-03", effective_to=None),
    ]

    _, history, _, _, _, _ = _run_pipeline(module, tmp_path, sec_rows=sec_rows, asof="2025-03-10")

    old_active = module.resolve_pit_identity(history, "OLD", "2021-06-01")
    assert len(old_active) == 1
    assert old_active.iloc[0]["symbol"] == "OLD"
    assert old_active.iloc[0]["cik"] == "5555555555"

    old_after_change = module.resolve_pit_identity(history, "OLD", "2023-03-10")
    assert old_after_change.empty

    new_active = module.resolve_pit_identity(history, "NEW", "2023-03-10")
    assert len(new_active) == 1
    assert new_active.iloc[0]["symbol"] == "NEW"
    assert new_active.iloc[0]["cik"] == "5555555555"



def test_build_current_excludes_only_the_unresolved_symbol_not_all_symbols(ticker_cik_module, tmp_path: Path):
    module = ticker_cik_module
    sec_rows = [
        _sec_row("BAD", "6666666666", issuer_name="BAD INC", effective_from="2020-01-01", ingest_ts_utc="2025-03-10T10:00:00Z"),
        _sec_row("BAD", "7777777777", issuer_name="BAD INC", effective_from="2020-01-01", ingest_ts_utc="2025-03-10T10:00:00Z"),
        _sec_row("GOOD", "8888888888", issuer_name="GOOD INC", effective_from="2020-01-01"),
    ]

    _, history, current, conflicts, _, _ = _run_pipeline(module, tmp_path, sec_rows=sec_rows, asof="2025-03-10")

    assert set(current["symbol"].tolist()) == {"GOOD"}
    assert set(history["symbol"].tolist()) == {"GOOD"}
    assert "BAD" in set(conflicts["symbol"].astype(str))



def test_conflict_class_and_status_are_assigned_for_resolved_source_conflict(ticker_cik_module, tmp_path: Path):
    module = ticker_cik_module
    sec_rows = [
        _sec_row("MNO", "9999999999", issuer_name="MNO PLC", effective_from="2020-01-01"),
    ]
    internal_rows = [
        _internal_row("MNO", "1212121212", issuer_name="MNO PLC", effective_from="2020-01-01"),
    ]

    _, _, _, conflicts, _, _ = _run_pipeline(module, tmp_path, sec_rows=sec_rows, internal_rows=internal_rows, asof="2025-03-10")

    assert not conflicts.empty
    row = conflicts[conflicts["conflict_class"] == "source_conflict"].iloc[0]
    assert row["conflict_status"] == "resolved_by_precedence"
    assert row["symbol"] == "MNO"
    assert int(row["n_candidates"]) >= 2
    assert int(row["n_active_cik"]) >= 2
    assert "winner=" in str(row["details"])



def test_same_input_same_config_produces_stable_history_current_and_conflicts(ticker_cik_module, tmp_path: Path):
    module = ticker_cik_module
    sec_rows = [
        _sec_row("AAA", "1010101010", issuer_name="AAA INC", effective_from="2020-01-01"),
        _sec_row("BBB", "2020202020", issuer_name="BBB INC", effective_from="2021-05-03"),
    ]
    internal_rows = [
        _internal_row("AAA", "1010101010", issuer_name="AAA INC", effective_from="2020-01-01"),
        _internal_row("BBB", "2020202020", issuer_name="BBB INC", effective_from="2021-05-03"),
    ]

    _, h1, c1, x1, m1, cfg1 = _run_pipeline(
        module,
        tmp_path / "run1",
        sec_rows=sec_rows,
        internal_rows=internal_rows,
        asof="2025-03-10",
        run_id="run1",
    )
    _, h2, c2, x2, m2, cfg2 = _run_pipeline(
        module,
        tmp_path / "run2",
        sec_rows=sec_rows,
        internal_rows=internal_rows,
        asof="2025-03-10",
        run_id="run2",
    )

    pdt.assert_frame_equal(_sort_history(h1), _sort_history(h2), check_like=False)
    pdt.assert_frame_equal(
        c1.sort_values(["symbol"]).reset_index(drop=True),
        c2.sort_values(["symbol"]).reset_index(drop=True),
        check_like=False,
    )
    pdt.assert_frame_equal(_sort_conflicts(x1), _sort_conflicts(x2), check_like=False)
    pdt.assert_frame_equal(
        m1.sort_values(["metric_name"]).reset_index(drop=True),
        m2.sort_values(["metric_name"]).reset_index(drop=True),
        check_like=False,
    )

    cfg1_cmp = copy.deepcopy(cfg1)
    cfg2_cmp = copy.deepcopy(cfg2)
    cfg1_cmp["storage"]["output_root"] = "<normalized>"
    cfg2_cmp["storage"]["output_root"] = "<normalized>"
    assert cfg1_cmp == cfg2_cmp
