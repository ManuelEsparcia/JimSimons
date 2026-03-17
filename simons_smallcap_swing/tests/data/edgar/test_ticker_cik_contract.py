from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

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
    "is_active",
    "resolution_status",
    "heuristic_flag",
    "issuer_name",
    "exchange",
    "share_class",
    "evidence_hash",
    "mapping_version",
    "lineage_count",
    "corroboration_count",
    "previous_symbol_cik",
    "_history_row_id",
    "current_candidate_flag",
}

CURRENT_REQUIRED_COLUMNS = HISTORY_REQUIRED_COLUMNS
CONFLICT_REQUIRED_COLUMNS = {
    "symbol",
    "cik",
    "effective_from",
    "effective_to",
    "conflict_class",
    "conflict_status",
    "source",
    "details",
    "n_candidates",
    "n_active_cik",
    "severity_rank",
    "candidate_evidence_hashes",
}

MANIFEST_REQUIRED_KEYS = {
    "run_id",
    "logical_asof_date",
    "config_version",
    "config_hash",
    "sources",
    "n_input_rows",
    "n_resolved_history_rows",
    "n_current_rows",
    "n_unique_history_symbols",
    "n_unique_current_symbols",
    "n_unique_current_cik",
    "conflicts",
    "n_historical_changes_detected",
    "started_at_utc",
    "finished_at_utc",
    "interval_convention",
}

METRIC_MINIMUM_NAMES = {
    "n_input_rows",
    "n_unique_input_symbols",
    "n_unique_input_cik",
    "n_history_rows",
    "n_current_rows",
    "n_current_symbols",
    "n_current_cik",
    "n_conflicts_total",
    "n_conflicts_unresolved",
    "n_ticker_changes_detected",
    "coverage_over_target_symbols",
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
            spec = importlib.util.spec_from_file_location("ticker_cik_contract_under_test", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
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



def _deep_merge(module: Any, a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    deep_merge = getattr(module, "_deep_merge", None)
    if callable(deep_merge):
        return deep_merge(a, b)
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(module, out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out



def _run_pipeline(
    module: Any,
    tmp_path: Path,
    *,
    sec_rows: Iterable[Mapping[str, Any]],
    internal_rows: Optional[Iterable[Mapping[str, Any]]] = None,
    config_patch: Optional[Mapping[str, Any]] = None,
    asof: str = "2025-03-10",
    run_id: str = "pytest_ticker_cik_contract",
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
        cfg = _deep_merge(module, cfg, dict(config_patch))

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



def _metric_map(metrics: pd.DataFrame) -> Dict[str, float]:
    if metrics.empty:
        return {}
    out: Dict[str, float] = {}
    for _, row in metrics.iterrows():
        out[str(row["metric_name"])] = float(row["metric_value"])
    return out



def _assert_no_symbol_overlap(history: pd.DataFrame) -> None:
    if history.empty:
        return
    hist = history.sort_values(["symbol", "effective_from", "effective_to"], kind="mergesort").reset_index(drop=True)
    for symbol, group in hist.groupby("symbol", sort=False):
        g = group.reset_index(drop=True)
        for i in range(len(g) - 1):
            a = g.iloc[i]
            b = g.iloc[i + 1]
            a_end = a["effective_to"]
            if pd.notna(a_end):
                assert a_end <= b["effective_from"], (
                    f"History contains overlap for symbol={symbol}: "
                    f"{a['effective_from']} -> {a_end} overlaps {b['effective_from']}"
                )


@pytest.fixture()
def clean_case(ticker_cik_module, tmp_path):
    result, history, current, conflicts, metrics, cfg = _run_pipeline(
        ticker_cik_module,
        tmp_path,
        sec_rows=[
            _sec_row("AAA", 1, issuer_name="AAA CORP", exchange="NYSE", effective_from="2020-01-01"),
            _sec_row("BBB", 2, issuer_name="BBB CORP", exchange="NASDAQ", effective_from="2021-01-01"),
            _sec_row("OLD", 3, issuer_name="CCC CORP", exchange="NASDAQ", effective_from="2020-01-01", effective_to="2022-01-01"),
            _sec_row("NEW", 3, issuer_name="CCC CORP", exchange="NASDAQ", effective_from="2022-01-01"),
        ],
        internal_rows=[
            _internal_row("AAA", 1, issuer_name="AAA CORP", exchange="NYSE", effective_from="2020-01-01"),
            _internal_row("BBB", 2, issuer_name="BBB CORP", exchange="NASDAQ", effective_from="2021-01-01"),
        ],
    )
    return {
        "result": result,
        "history": history,
        "current": current,
        "conflicts": conflicts,
        "metrics": metrics,
        "cfg": cfg,
        "tmp_path": tmp_path,
    }


@pytest.fixture()
def unresolved_conflict_case(ticker_cik_module, tmp_path):
    result, history, current, conflicts, metrics, cfg = _run_pipeline(
        ticker_cik_module,
        tmp_path,
        sec_rows=[
            _sec_row("AAA", 1, issuer_name="AAA CORP", exchange="NYSE", effective_from="2020-01-01"),
            _sec_row("CLSH", 111, issuer_name="CLASH CORP", exchange="NASDAQ", effective_from="2020-01-01", confidence_score=0.91, ingest_ts_utc="2025-03-10T12:00:00Z"),
            _sec_row("CLSH", 222, issuer_name="CLASH CORP", exchange="NASDAQ", effective_from="2020-01-01", confidence_score=0.91, ingest_ts_utc="2025-03-10T12:00:00Z"),
        ],
        internal_rows=None,
    )
    return {
        "result": result,
        "history": history,
        "current": current,
        "conflicts": conflicts,
        "metrics": metrics,
        "cfg": cfg,
        "tmp_path": tmp_path,
    }


# -----------------------------------------------------------------------------
# Contract tests
# -----------------------------------------------------------------------------


def test_history_output_contains_required_columns(clean_case):
    history = clean_case["history"]
    missing = HISTORY_REQUIRED_COLUMNS - set(history.columns)
    assert not missing, f"Missing history columns: {sorted(missing)}"



def test_current_output_contains_required_columns(clean_case):
    current = clean_case["current"]
    missing = CURRENT_REQUIRED_COLUMNS - set(current.columns)
    assert not missing, f"Missing current columns: {sorted(missing)}"



def test_conflicts_output_contains_required_columns_even_when_empty(clean_case):
    conflicts = clean_case["conflicts"]
    missing = CONFLICT_REQUIRED_COLUMNS - set(conflicts.columns)
    assert not missing, f"Missing conflicts columns: {sorted(missing)}"



def test_current_contains_only_active_resolved_identities(clean_case):
    current = clean_case["current"]
    asof = pd.Timestamp(clean_case["result"]["manifest"]["logical_asof_date"]).normalize()

    assert current["symbol"].is_unique, "Current snapshot must contain at most one row per symbol."
    assert bool(current["current_candidate_flag"].all())
    assert bool(current["is_active"].all())
    assert bool((current["confidence_score"] >= 0.0).all())
    assert bool((current["effective_from"] <= asof).all())
    assert bool((current["effective_to"].isna() | (current["effective_to"] > asof)).all())



def test_history_contains_no_illegitimate_symbol_overlap(clean_case):
    _assert_no_symbol_overlap(clean_case["history"])



def test_unresolved_conflict_symbol_is_excluded_from_current_snapshot(unresolved_conflict_case):
    current = unresolved_conflict_case["current"]
    conflicts = unresolved_conflict_case["conflicts"]
    metrics = _metric_map(unresolved_conflict_case["metrics"])

    assert "CLSH" not in set(current["symbol"].tolist())
    unresolved = conflicts[
        (conflicts["symbol"] == "CLSH")
        & (conflicts["conflict_status"] == "unresolved")
    ]
    assert not unresolved.empty
    assert metrics.get("n_conflicts_unresolved", 0.0) >= 1.0



def test_manifest_contains_required_keys_and_count_consistency(clean_case, ticker_cik_module):
    result = clean_case["result"]
    manifest = result["manifest"]
    history = clean_case["history"]
    current = clean_case["current"]
    conflicts = clean_case["conflicts"]
    cfg = clean_case["cfg"]

    missing = MANIFEST_REQUIRED_KEYS - set(manifest.keys())
    assert not missing, f"Missing manifest keys: {sorted(missing)}"
    assert manifest["n_resolved_history_rows"] == len(history)
    assert manifest["n_current_rows"] == len(current)
    assert manifest["n_unique_history_symbols"] == int(history["symbol"].nunique())
    assert manifest["n_unique_current_symbols"] == int(current["symbol"].nunique())
    assert manifest["n_unique_current_cik"] == int(current["cik"].nunique())

    expected_hash = ticker_cik_module._config_hash(cfg)
    assert manifest["config_hash"] == expected_hash
    assert manifest["interval_convention"]["effective_from"] == "inclusive"
    assert manifest["interval_convention"]["effective_to"] == "exclusive"

    by_class = (
        conflicts.groupby(["conflict_class", "conflict_status"]).size().reset_index(name="n")
        if not conflicts.empty
        else pd.DataFrame(columns=["conflict_class", "conflict_status", "n"])
    )
    expected_conflicts = {
        f"{row['conflict_class']}::{row['conflict_status']}": int(row["n"])
        for _, row in by_class.iterrows()
    }
    assert manifest["conflicts"] == expected_conflicts



def test_metrics_output_contains_required_metrics_and_matches_outputs(clean_case):
    history = clean_case["history"]
    current = clean_case["current"]
    conflicts = clean_case["conflicts"]
    metric_map = _metric_map(clean_case["metrics"])

    missing = METRIC_MINIMUM_NAMES - set(metric_map.keys())
    assert not missing, f"Missing core metrics: {sorted(missing)}"

    assert metric_map["n_history_rows"] == float(len(history))
    assert metric_map["n_current_rows"] == float(len(current))
    assert metric_map["n_current_symbols"] == float(current["symbol"].nunique())
    assert metric_map["n_current_cik"] == float(current["cik"].nunique())
    assert metric_map["n_conflicts_total"] == float(len(conflicts))
    assert metric_map["n_conflicts_unresolved"] == 0.0
    assert metric_map["n_ticker_changes_detected"] >= 1.0



def test_artifact_paths_exist_and_manifest_file_matches_return_value(clean_case):
    result = clean_case["result"]
    manifest = result["manifest"]
    artifacts = result["artifacts"]

    for key in ["history_path", "current_path", "conflicts_path", "metrics_path", "manifest_path"]:
        path = Path(artifacts[key])
        assert path.exists(), f"Expected artifact path to exist: {path}"

    manifest_path = Path(artifacts["manifest_path"])
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk == manifest



def test_outputs_are_deterministic_under_same_input_and_config(ticker_cik_module, tmp_path):
    sec_rows = [
        _sec_row("AAA", 1, issuer_name="AAA CORP", exchange="NYSE", effective_from="2020-01-01"),
        _sec_row("BBB", 2, issuer_name="BBB CORP", exchange="NASDAQ", effective_from="2021-01-01"),
        _sec_row("OLD", 3, issuer_name="CCC CORP", exchange="NASDAQ", effective_from="2020-01-01", effective_to="2022-01-01"),
        _sec_row("NEW", 3, issuer_name="CCC CORP", exchange="NASDAQ", effective_from="2022-01-01"),
    ]
    internal_rows = [
        _internal_row("AAA", 1, issuer_name="AAA CORP", exchange="NYSE", effective_from="2020-01-01"),
        _internal_row("BBB", 2, issuer_name="BBB CORP", exchange="NASDAQ", effective_from="2021-01-01"),
    ]

    common_patch = {"storage": {"output_root": str(tmp_path / "shared_out"), "allow_csv_fallback": True}}
    r1, h1, c1, x1, m1, _ = _run_pipeline(
        ticker_cik_module,
        tmp_path / "run1",
        sec_rows=sec_rows,
        internal_rows=internal_rows,
        config_patch=common_patch,
        run_id="r1",
    )
    r2, h2, c2, x2, m2, _ = _run_pipeline(
        ticker_cik_module,
        tmp_path / "run2",
        sec_rows=sec_rows,
        internal_rows=internal_rows,
        config_patch=common_patch,
        run_id="r2",
    )

    h1 = h1.drop(columns=[c for c in ["_history_row_id"] if c in h1.columns]).sort_values(["symbol", "effective_from", "cik", "source"], kind="mergesort").reset_index(drop=True)
    h2 = h2.drop(columns=[c for c in ["_history_row_id"] if c in h2.columns]).sort_values(["symbol", "effective_from", "cik", "source"], kind="mergesort").reset_index(drop=True)
    c1 = c1.drop(columns=[c for c in ["_history_row_id"] if c in c1.columns]).sort_values(["symbol", "effective_from", "cik", "source"], kind="mergesort").reset_index(drop=True)
    c2 = c2.drop(columns=[c for c in ["_history_row_id"] if c in c2.columns]).sort_values(["symbol", "effective_from", "cik", "source"], kind="mergesort").reset_index(drop=True)
    x1 = x1.sort_values(["symbol", "effective_from", "conflict_class", "conflict_status", "source"], kind="mergesort").reset_index(drop=True)
    x2 = x2.sort_values(["symbol", "effective_from", "conflict_class", "conflict_status", "source"], kind="mergesort").reset_index(drop=True)
    m1 = m1.sort_values(["metric_name"], kind="mergesort").reset_index(drop=True)
    m2 = m2.sort_values(["metric_name"], kind="mergesort").reset_index(drop=True)

    pdt.assert_frame_equal(h1, h2, check_like=False)
    pdt.assert_frame_equal(c1, c2, check_like=False)
    pdt.assert_frame_equal(x1, x2, check_like=False)
    pdt.assert_frame_equal(m1, m2, check_like=False)
    assert r1["manifest"]["config_hash"] == r2["manifest"]["config_hash"]
    assert r1["manifest"]["logical_asof_date"] == r2["manifest"]["logical_asof_date"]

