from __future__ import annotations

import copy
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
            spec = importlib.util.spec_from_file_location("corporate_actions", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            # Python 3.13 + dataclasses + spec loader can require registration first.
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module

    raise ModuleNotFoundError(
        "Could not import corporate_actions.py. Expected it at "
        "simons_smallcap_swing.data.universe.corporate_actions or in a nearby repo path."
    )


@pytest.fixture(scope="session")
def corporate_actions_module():
    return _load_corporate_actions_module()


@pytest.fixture()
def config(corporate_actions_module: Any):
    cfg, payload, config_hash = corporate_actions_module.load_config(None)
    return {
        "config": cfg,
        "payload": payload,
        "hash": config_hash,
    }


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _write_table(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    df = pd.DataFrame([dict(r) for r in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)



def _normalize_rows(module: Any, tmp_path: Path, config: Any, rows: Iterable[Mapping[str, Any]]):
    raw_path = tmp_path / "raw.csv"
    _write_table(rows, raw_path)
    raw_df = module.load_raw_corporate_actions([raw_path])
    return module.normalize_raw_events(raw_df, config)



def _identity_master(module: Any, tmp_path: Path, matching: Any, rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    path = tmp_path / "identity.csv"
    _write_table(rows, path)
    return module.load_identity_master(path, matching)



def _resolve(module: Any, normalized: pd.DataFrame, identity_df: pd.DataFrame, matching: Any):
    return module.resolve_identity(normalized, identity_df, matching)



def _failure_codes(df: pd.DataFrame) -> list[str]:
    if df.empty or "failure_code" not in df.columns:
        return []
    return df["failure_code"].astype(str).tolist()


# -----------------------------------------------------------------------------
# Synthetic row builders
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_normalize_raw_events_maps_ticker_change_and_identity_fields(corporate_actions_module: Any, tmp_path: Path, config: dict[str, Any]):
    normalized, failures = _normalize_rows(
        corporate_actions_module,
        tmp_path,
        config["config"],
        [
            _raw_row(
                "vendor_a",
                "evt_1",
                "ticker change",
                instrument_id="I1",
                new_symbol="AAB",
            )
        ],
    )

    assert failures.empty
    assert len(normalized) == 1
    row = normalized.iloc[0]
    assert row["canonical_event_type"] == corporate_actions_module.CanonicalEventType.TICKER_CHANGE.value
    assert row["event_family"] == corporate_actions_module.EventFamily.IDENTITY.value
    assert row["instrument_id_input"] == "I1"
    assert row["new_symbol"] == "AAB"
    assert pd.Timestamp(row["canonical_date"]).normalize() == pd.Timestamp("2025-01-15")



def test_special_cash_dividend_auto_reclassified_when_large(corporate_actions_module: Any, tmp_path: Path, config: dict[str, Any]):
    normalized, failures = _normalize_rows(
        corporate_actions_module,
        tmp_path,
        config["config"],
        [
            _raw_row(
                "company",
                "div_1",
                "cash dividend",
                cash_amount=15.0,
                prev_close=100.0,
            )
        ],
    )

    assert failures.empty
    assert len(normalized) == 1
    row = normalized.iloc[0]
    assert row["canonical_event_type"] == corporate_actions_module.CanonicalEventType.SPECIAL_CASH_DIVIDEND.value
    assert "AUTO_RECLASSIFIED_SPECIAL_DIVIDEND" in json.loads(row["quality_flags"])



def test_unsupported_event_type_goes_to_failure(corporate_actions_module: Any, tmp_path: Path, config: dict[str, Any]):
    normalized, failures = _normalize_rows(
        corporate_actions_module,
        tmp_path,
        config["config"],
        [_raw_row("vendor_a", "evt_bad", "mystery event")],
    )

    assert normalized.empty
    assert _failure_codes(failures) == [corporate_actions_module.FailureCode.UNSUPPORTED_EVENT_TYPE.value]



def test_resolve_identity_prefers_explicit_instrument_id_over_symbol(corporate_actions_module: Any, tmp_path: Path, config: dict[str, Any]):
    normalized, failures = _normalize_rows(
        corporate_actions_module,
        tmp_path,
        config["config"],
        [
            _raw_row(
                "official",
                "evt_1",
                "split",
                instrument_id="I1",
                symbol="WRONG",
                split_ratio="2:1",
            )
        ],
    )
    assert failures.empty

    identity_df = _identity_master(
        corporate_actions_module,
        tmp_path,
        config["config"].matching,
        [
            _id_row("I1", "AAB", issuer_id="ISS1"),
            _id_row("I2", "WRONG", issuer_id="ISS2"),
        ],
    )
    resolved, failures = _resolve(corporate_actions_module, normalized, identity_df, config["config"].matching)

    assert failures.empty
    assert len(resolved) == 1
    row = resolved.iloc[0]
    assert row["instrument_id"] == "I1"
    assert row["symbol"] == "AAB"
    assert row["linkage_method"] == "explicit_instrument_id"



def test_resolve_identity_ambiguous_symbol_goes_to_failure(corporate_actions_module: Any, tmp_path: Path, config: dict[str, Any]):
    normalized, failures = _normalize_rows(
        corporate_actions_module,
        tmp_path,
        config["config"],
        [
            _raw_row(
                "vendor_a",
                "evt_2",
                "cash dividend",
                symbol="AAA",
                exchange="NYSE",
                share_class="COM",
                cash_amount=0.5,
            )
        ],
    )
    assert failures.empty
    normalized.loc[:, "exchange"] = pd.NA

    identity_df = _identity_master(
        corporate_actions_module,
        tmp_path,
        config["config"].matching,
        [
            _id_row("I1", "AAA", issuer_id="ISS1", exchange="NYSE"),
            _id_row("I2", "AAA", issuer_id="ISS2", exchange="NASDAQ"),
        ],
    )
    resolved, failures = _resolve(corporate_actions_module, normalized, identity_df, config["config"].matching)

    assert resolved.empty
    assert _failure_codes(failures) == [corporate_actions_module.FailureCode.AMBIGUOUS_IDENTITY.value]
    assert "ambiguous_symbol_match" in str(failures.iloc[0]["failure_detail"])



def test_resolve_identity_low_confidence_symbol_match_rejected(corporate_actions_module: Any, tmp_path: Path, config: dict[str, Any]):
    normalized, failures = _normalize_rows(
        corporate_actions_module,
        tmp_path,
        config["config"],
        [
            _raw_row(
                "vendor_a",
                "evt_3",
                "cash dividend",
                symbol="AAA",
                exchange="NYSE",
                share_class="COM",
                cash_amount=0.5,
            )
        ],
    )
    assert failures.empty
    normalized.loc[:, "exchange"] = pd.NA  # force pit_symbol => confidence 0.80

    strict_matching = copy.deepcopy(config["config"].matching)
    strict_matching = corporate_actions_module.MatchingConfig(
        min_confidence_score=0.85,
        relisting_resolution=strict_matching.relisting_resolution,
        prefer_explicit_instrument_id=strict_matching.prefer_explicit_instrument_id,
        external_id_columns=strict_matching.external_id_columns,
        active_window_start_columns=strict_matching.active_window_start_columns,
        active_window_end_columns=strict_matching.active_window_end_columns,
    )
    identity_df = _identity_master(
        corporate_actions_module,
        tmp_path,
        strict_matching,
        [_id_row("I1", "AAA", issuer_id="ISS1", exchange="NYSE")],
    )
    resolved, failures = _resolve(corporate_actions_module, normalized, identity_df, strict_matching)

    assert resolved.empty
    assert _failure_codes(failures) == [corporate_actions_module.FailureCode.LOW_CONFIDENCE_IDENTITY.value]



def test_canonicalize_events_same_day_provider_resolution_is_deterministic(corporate_actions_module: Any, tmp_path: Path, config: dict[str, Any]):
    normalized, failures = _normalize_rows(
        corporate_actions_module,
        tmp_path,
        config["config"],
        [
            _raw_row("vendor_b", "evt_v", "split", instrument_id="I1", split_ratio="2:1"),
            _raw_row("official", "evt_o", "split", instrument_id="I1", split_ratio="2:1"),
        ],
    )
    assert failures.empty
    identity_df = _identity_master(
        corporate_actions_module,
        tmp_path,
        config["config"].matching,
        [_id_row("I1", "AAA")],
    )
    resolved, failures = _resolve(corporate_actions_module, normalized, identity_df, config["config"].matching)
    assert failures.empty

    canonical, failures = corporate_actions_module.canonicalize_events(
        resolved, config["config"], "2025-01-20T00:00:00Z"
    )

    assert failures.empty
    assert len(canonical) == 1
    row = canonical.iloc[0]
    assert row["source_provider"] == "official"
    assert row["raw_records_count"] == 2
    assert row["event_type"] == corporate_actions_module.CanonicalEventType.SPLIT.value
    assert row["same_day_precedence_rank"] == 5
    lineage = json.loads(row["raw_event_refs"])
    assert [x["provider_name"] for x in lineage] == ["official", "vendor_b"]



def test_canonicalize_events_material_conflict_promoted_to_failure(corporate_actions_module: Any, tmp_path: Path, config: dict[str, Any]):
    normalized, failures = _normalize_rows(
        corporate_actions_module,
        tmp_path,
        config["config"],
        [
            _raw_row("vendor_b", "evt_v", "cash dividend", instrument_id="I1", cash_amount=0.50),
            _raw_row("official", "evt_o", "cash dividend", instrument_id="I1", cash_amount=0.80),
        ],
    )
    assert failures.empty
    identity_df = _identity_master(
        corporate_actions_module,
        tmp_path,
        config["config"].matching,
        [_id_row("I1", "AAA")],
    )
    resolved, failures = _resolve(corporate_actions_module, normalized, identity_df, config["config"].matching)
    assert failures.empty

    canonical, failures = corporate_actions_module.canonicalize_events(
        resolved, config["config"], "2025-01-20T00:00:00Z"
    )

    assert canonical.empty
    assert _failure_codes(failures) == [corporate_actions_module.FailureCode.MATERIAL_CONFLICT.value]
    assert "cash_amount" in str(failures.iloc[0]["failure_detail"])



def test_build_current_snapshot_respects_announcement_ts_not_only_effective_date(corporate_actions_module: Any):
    history = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "announcement_ts": pd.Timestamp("2025-01-10T00:00:00Z"),
                "effective_date": pd.Timestamp("2025-01-15"),
                "ex_date": pd.Timestamp("2025-01-15"),
                "status": corporate_actions_module.EventStatus.CONFIRMED.value,
            },
            {
                "event_id": "e2",
                "announcement_ts": pd.Timestamp("2025-01-20T00:00:00Z"),
                "effective_date": pd.Timestamp("2025-01-15"),
                "ex_date": pd.Timestamp("2025-01-15"),
                "status": corporate_actions_module.EventStatus.CONFIRMED.value,
            },
            {
                "event_id": "e3",
                "announcement_ts": pd.Timestamp("2025-01-05T00:00:00Z"),
                "effective_date": pd.Timestamp("2025-01-25"),
                "ex_date": pd.Timestamp("2025-01-25"),
                "status": corporate_actions_module.EventStatus.PENDING.value,
            },
        ]
    )

    current = corporate_actions_module.build_current_snapshot(history, "2025-01-18T00:00:00Z")
    assert current["event_id"].tolist() == ["e1", "e3"]
    assert current["status"].tolist() == ["confirmed", "pending"]



def test_canonicalize_events_is_reproducible_under_same_inputs(corporate_actions_module: Any, tmp_path: Path, config: dict[str, Any]):
    normalized, failures = _normalize_rows(
        corporate_actions_module,
        tmp_path,
        config["config"],
        [
            _raw_row("official", "evt_1", "split", instrument_id="I1", split_ratio="2:1"),
            _raw_row("vendor_b", "evt_2", "ticker change", instrument_id="I1", new_symbol="AAB"),
        ],
    )
    assert failures.empty
    identity_df = _identity_master(
        corporate_actions_module,
        tmp_path,
        config["config"].matching,
        [_id_row("I1", "AAA")],
    )
    resolved, failures = _resolve(corporate_actions_module, normalized, identity_df, config["config"].matching)
    assert failures.empty

    c1, f1 = corporate_actions_module.canonicalize_events(resolved.copy(), config["config"], "2025-01-20T00:00:00Z")
    c2, f2 = corporate_actions_module.canonicalize_events(resolved.copy(), config["config"], "2025-01-20T00:00:00Z")

    pdt.assert_frame_equal(c1.reset_index(drop=True), c2.reset_index(drop=True), check_like=False)
    pdt.assert_frame_equal(f1.reset_index(drop=True), f2.reset_index(drop=True), check_like=False)
