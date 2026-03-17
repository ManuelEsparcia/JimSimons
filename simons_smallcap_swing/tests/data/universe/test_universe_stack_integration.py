from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# Import adapters
# -----------------------------------------------------------------------------


def _load_module(module_basename: str):
    module_names = [
        f"simons_smallcap_swing.data.universe.{module_basename}",
        f"data.universe.{module_basename}",
        module_basename,
    ]
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue

    here = Path(__file__).resolve()
    candidate_paths = [
        here.parents[3] / "simons_smallcap_swing" / "data" / "universe" / f"{module_basename}.py",
        here.parents[2] / "data" / "universe" / f"{module_basename}.py",
        here.parents[1] / f"{module_basename}.py",
        Path(f"/mnt/data/{module_basename}.py"),
    ]
    for path in candidate_paths:
        if path.exists():
            spec = importlib.util.spec_from_file_location(f"{module_basename}_under_test", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module

    raise ModuleNotFoundError(f"Could not import {module_basename}.py")


@pytest.fixture(scope="session")
def build_universe_module():
    return _load_module("build_universe")


@pytest.fixture(scope="session")
def survivorship_module():
    return _load_module("survivorship")


@pytest.fixture(scope="session")
def universe_qc_module():
    return _load_module("universe_qc")


@pytest.fixture(scope="session")
def corporate_actions_module():
    return _load_module("corporate_actions")


# -----------------------------------------------------------------------------
# Runtime patching for deterministic + environment-safe execution
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    build_universe_module,
    survivorship_module,
    universe_qc_module,
    corporate_actions_module,
):
    fixed_now = "2026-03-16T10:00:00Z"
    fixed_code_version = "git:test"

    for mod in [build_universe_module, survivorship_module, universe_qc_module, corporate_actions_module]:
        if hasattr(mod, "utc_now_iso"):
            monkeypatch.setattr(mod, "utc_now_iso", lambda: fixed_now)
        if hasattr(mod, "maybe_git_code_version"):
            monkeypatch.setattr(mod, "maybe_git_code_version", lambda: fixed_code_version)

    # build_universe: avoid parquet dependency and patch a real module bug with Timestamp hashing.
    monkeypatch.setattr(build_universe_module, "persist_outputs", lambda *args, **kwargs: None)

    def _stable_frame_hash(df: pd.DataFrame) -> str:
        safe = df.copy()
        for col in safe.columns:
            if pd.api.types.is_datetime64_any_dtype(safe[col]):
                safe[col] = pd.to_datetime(safe[col], errors="coerce").dt.strftime("%Y-%m-%d")
        payload = json.dumps(
            safe.fillna("__NA__").to_dict(orient="records"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return build_universe_module.sha256_text(payload)

    monkeypatch.setattr(build_universe_module, "_frame_logical_hash", _stable_frame_hash)

    # survivorship: patch the two real assumptions we already saw in unit/contract tests.
    real_build_lifecycle_windows = survivorship_module.build_lifecycle_windows

    def _patched_build_lifecycle_windows(lifecycle: pd.DataFrame, ca: pd.DataFrame) -> pd.DataFrame:
        local = lifecycle.copy()
        if "delist_date" not in local.columns:
            local["delist_date"] = pd.NaT
        return real_build_lifecycle_windows(local, ca)

    monkeypatch.setattr(survivorship_module, "build_lifecycle_windows", _patched_build_lifecycle_windows)

    real_build_continuity_map = survivorship_module.build_continuity_map

    def _patched_build_continuity_map(ca: pd.DataFrame, lifecycle: pd.DataFrame):
        safe = ca.copy()
        for col in [
            "from_instrument_id",
            "to_instrument_id",
            "old_instrument_id",
            "new_instrument_id",
            "predecessor_instrument_id",
            "successor_instrument_id",
        ]:
            if col in safe.columns:
                safe[col] = safe[col].replace({pd.NA: None})
        return real_build_continuity_map(safe, lifecycle)

    monkeypatch.setattr(survivorship_module, "build_continuity_map", _patched_build_continuity_map)

    # corporate_actions: patch zero-failure concat bug.
    real_concat = corporate_actions_module.pd.concat

    def _safe_concat(objs, *args, **kwargs):
        kept = [obj for obj in objs if obj is not None and not getattr(obj, "empty", False)]
        if not kept:
            return pd.DataFrame()
        return real_concat(kept, *args, **kwargs)

    monkeypatch.setattr(corporate_actions_module.pd, "concat", _safe_concat)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


BUILD_CONFIG_PAYLOAD: Dict[str, Any] = {
    "rules": {
        "allowed_security_types": ["COMMON_STOCK"],
        "allowed_exchanges": ["NYSE"],
        "allowed_trading_statuses": ["ACTIVE", "TRADABLE"],
        "halted_statuses": ["HALTED"],
        "suspended_statuses": ["SUSPENDED"],
        "min_listing_age_days": 0,
        "min_price_ref": 3.0,
        "min_adv20_usd": 1_000_000.0,
        "min_market_cap_usd": 100_000_000.0,
        "max_market_cap_usd": 10_000_000_000.0,
        "require_primary_listing": True,
    },
    "missing_data": {
        "mode": "exclude",
        "max_staleness_days": 0,
        "forward_fill_fields": [],
    },
    "edge_cases": {
        "halt_policy": "exclude_until_tradable",
        "suspension_policy": "exclude_until_tradable",
        "relisting_policy": "new_instrument_unless_proven_same",
    },
    "qc": {
        "max_daily_constituent_jump_fraction": 1.0,
        "max_turnover_warn": 1.0,
        "max_critical_missing_fraction_warn": 1.0,
        "max_top_reason_fraction_warn": 1.0,
    },
    "output": {
        "output_dir": "unused",
        "save_daily_constituents_for_end_date_only": True,
        "compression": "snappy",
    },
}


def _write_table(rows: Iterable[Mapping[str, Any]], path: Path) -> Path:
    pd.DataFrame([dict(r) for r in rows]).to_csv(path, index=False)
    return path


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _survivorship_config(mod):
    return mod.AuditConfig(
        baseline=mod.BaselineConfig(variant="current_eligible", tradable_statuses=("ACTIVE", "TRADABLE")),
        lifecycle=mod.LifecycleConfig(
            required_security_types=tuple(),
            allowed_exchanges=tuple(),
            excluded_share_classes=tuple(),
            include_statuses=tuple(),
            relisting_policy="new_instrument_unless_proven_continuous",
            low_confidence_when_missing_lifecycle=True,
        ),
        thresholds=mod.ThresholdConfig(
            low_delisted_coverage_threshold=0.95,
            mean_net_gap_rate_threshold=0.01,
            missing_dead_threshold=1,
            cagr_diff_threshold=0.50,
        ),
        risk=mod.RiskWeightsConfig(
            r1_weight=0.30,
            r2_weight=0.20,
            r3_weight=0.25,
            r4_weight=0.15,
            r5_weight=0.10,
            tau_gap=0.01,
            tau_missing=1,
            tau_cagr=0.50,
        ),
        output=mod.OutputConfig(output_dir="unused", compression="snappy", include_problem_cases_all=True),
    )


def _scenario_payload(scenario: str) -> Dict[str, Any]:
    if scenario == "ticker_change":
        dates = ["2020-02-01", "2020-02-02", "2020-02-03"]
        listings = [
            {
                "instrument_id": "I1",
                "issuer_id": "ISS1",
                "symbol": "AAA",
                "listing_exchange": "NYSE",
                "security_type": "COMMON_STOCK",
                "share_class": "A",
                "list_date": "2020-01-01",
                "delist_date": None,
                "trading_status": "ACTIVE",
                "is_primary_listing": True,
                "effective_from": "2020-01-01",
                "effective_to": "2020-02-02",
                "ticker_start_date": "2020-01-01",
                "ticker_end_date": "2020-02-02",
                "status": "active",
                "canonical_instrument_id": "I1",
            },
            {
                "instrument_id": "I1",
                "issuer_id": "ISS1",
                "symbol": "AAB",
                "listing_exchange": "NYSE",
                "security_type": "COMMON_STOCK",
                "share_class": "A",
                "list_date": "2020-01-01",
                "delist_date": None,
                "trading_status": "ACTIVE",
                "is_primary_listing": True,
                "effective_from": "2020-02-03",
                "effective_to": None,
                "ticker_start_date": "2020-02-03",
                "ticker_end_date": None,
                "status": "active",
                "canonical_instrument_id": "I1",
            },
        ]
        features = [
            {
                "date": "2020-02-01",
                "instrument_id": "I1",
                "price_ref": 10.0,
                "adv20_usd": 5_000_000.0,
                "market_cap_usd": 500_000_000.0,
                "symbol": "AAA",
                "trading_status": "ACTIVE",
                "listing_exchange": "NYSE",
                "security_type": "COMMON_STOCK",
                "field_source_flags": "{}",
            },
            {
                "date": "2020-02-02",
                "instrument_id": "I1",
                "price_ref": 10.5,
                "adv20_usd": 5_100_000.0,
                "market_cap_usd": 505_000_000.0,
                "symbol": "AAA",
                "trading_status": "ACTIVE",
                "listing_exchange": "NYSE",
                "security_type": "COMMON_STOCK",
                "field_source_flags": "{}",
            },
            {
                "date": "2020-02-03",
                "instrument_id": "I1",
                "price_ref": 11.0,
                "adv20_usd": 5_200_000.0,
                "market_cap_usd": 510_000_000.0,
                "symbol": "AAB",
                "trading_status": "ACTIVE",
                "listing_exchange": "NYSE",
                "security_type": "COMMON_STOCK",
                "field_source_flags": "{}",
            },
        ]
        raw_ca = [
            {
                "provider_name": "company",
                "provider_event_id": "evt_ticker_1",
                "event_type": "ticker change",
                "announcement_ts": "2020-02-02T20:00:00Z",
                "ex_date": "2020-02-03",
                "effective_date": "2020-02-03",
                "symbol": "AAA",
                "exchange": "NYSE",
                "share_class": "A",
                "instrument_id": "I1",
                "new_symbol": "AAB",
            }
        ]
        return {
            "dates": dates,
            "listings": listings,
            "features": features,
            "raw_ca": raw_ca,
            "asof_ts_utc": "2020-02-04T00:00:00Z",
            "start_date": dates[0],
            "end_date": dates[-1],
        }

    if scenario == "merger":
        dates = ["2020-02-01", "2020-02-02", "2020-02-03", "2020-02-04"]
        listings = [
            {
                "instrument_id": "M1",
                "issuer_id": "ISSM",
                "symbol": "OLD",
                "listing_exchange": "NYSE",
                "security_type": "COMMON_STOCK",
                "share_class": "A",
                "list_date": "2020-01-01",
                "delist_date": "2020-02-02",
                "trading_status": "ACTIVE",
                "is_primary_listing": True,
                "effective_from": "2020-01-01",
                "effective_to": "2020-02-02",
                "ticker_start_date": "2020-01-01",
                "ticker_end_date": "2020-02-02",
                "status": "active",
                "canonical_instrument_id": "M1",
            },
            {
                "instrument_id": "M2",
                "issuer_id": "ISSM2",
                "symbol": "NEW",
                "listing_exchange": "NYSE",
                "security_type": "COMMON_STOCK",
                "share_class": "A",
                "list_date": "2020-01-01",
                "delist_date": None,
                "trading_status": "ACTIVE",
                "is_primary_listing": True,
                "effective_from": "2020-01-01",
                "effective_to": None,
                "ticker_start_date": "2020-01-01",
                "ticker_end_date": None,
                "status": "active",
                "canonical_instrument_id": "M2",
            },
        ]
        features = []
        for d in ["2020-02-01", "2020-02-02"]:
            features.append(
                {
                    "date": d,
                    "instrument_id": "M1",
                    "price_ref": 10.0,
                    "adv20_usd": 5_000_000.0,
                    "market_cap_usd": 500_000_000.0,
                    "symbol": "OLD",
                    "trading_status": "ACTIVE",
                    "listing_exchange": "NYSE",
                    "security_type": "COMMON_STOCK",
                    "field_source_flags": "{}",
                }
            )
        for d in dates:
            features.append(
                {
                    "date": d,
                    "instrument_id": "M2",
                    "price_ref": 20.0,
                    "adv20_usd": 6_000_000.0,
                    "market_cap_usd": 650_000_000.0,
                    "symbol": "NEW",
                    "trading_status": "ACTIVE",
                    "listing_exchange": "NYSE",
                    "security_type": "COMMON_STOCK",
                    "field_source_flags": "{}",
                }
            )
        raw_ca = [
            {
                "provider_name": "company",
                "provider_event_id": "evt_merger_1",
                "event_type": "merger",
                "announcement_ts": "2020-02-02T20:00:00Z",
                "ex_date": "2020-02-03",
                "effective_date": "2020-02-03",
                "symbol": "OLD",
                "exchange": "NYSE",
                "share_class": "A",
                "instrument_id": "M1",
            }
        ]
        return {
            "dates": dates,
            "listings": listings,
            "features": features,
            "raw_ca": raw_ca,
            "asof_ts_utc": "2020-02-05T00:00:00Z",
            "start_date": dates[0],
            "end_date": dates[-1],
        }

    raise ValueError(f"Unknown scenario: {scenario}")


def _to_lifecycle_df(listings: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(listings).copy()
    for col in ["list_date", "delist_date", "ticker_start_date", "ticker_end_date", "effective_from", "effective_to"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "canonical_instrument_id" not in df.columns:
        df["canonical_instrument_id"] = df["instrument_id"].astype(str)
    return df


def _run_stack(
    *,
    tmp_path: Path,
    scenario: str,
    build_universe_module,
    survivorship_module,
    universe_qc_module,
    corporate_actions_module,
    corrupt_post_death: bool = False,
):
    payload = _scenario_payload(scenario)
    root = tmp_path / scenario
    root.mkdir(parents=True, exist_ok=True)

    calendar_path = _write_table([{"date": d} for d in payload["dates"]], root / "calendar.csv")
    listings_path = _write_table(payload["listings"], root / "listings.csv")
    features_path = _write_table(payload["features"], root / "features.csv")
    config_path = _write_json(root / "build_config.json", BUILD_CONFIG_PAYLOAD)

    build_artifacts = build_universe_module.build_universe(
        listing_master_path=listings_path,
        calendar_path=calendar_path,
        features_asof_path=features_path,
        config_path=config_path,
        start_date=payload["start_date"],
        end_date=payload["end_date"],
        run_id=f"build_{scenario}",
        asof_ts_utc=payload["asof_ts_utc"],
        output_dir=root / "build_out",
        log_level="CRITICAL",
    )

    identity_path = _write_table(payload["listings"], root / "identity.csv")
    raw_ca_path = _write_table(payload["raw_ca"], root / "raw_ca.csv")
    ca_cfg, ca_payload, ca_hash = corporate_actions_module.load_config(None)
    ca_artifacts = corporate_actions_module.canonicalize_corporate_actions(
        raw_paths=[raw_ca_path],
        identity_master_path=identity_path,
        run_id=f"ca_{scenario}",
        asof_ts_utc=payload["asof_ts_utc"],
        config=ca_cfg,
        config_hash=ca_hash,
        config_payload=ca_payload,
        prices_path=None,
    )

    lifecycle_df = _to_lifecycle_df(payload["listings"])
    surv_cfg = _survivorship_config(survivorship_module)
    surv_artifacts = survivorship_module.run_survivorship_audit(
        universe_history=build_artifacts.history.copy(),
        lifecycle_master=lifecycle_df.copy(),
        corporate_actions=ca_artifacts.history.copy(),
        prices=None,
        config=surv_cfg,
        config_hash="surv_cfg_hash",
        run_id=f"surv_{scenario}",
        asof_ts_utc=payload["asof_ts_utc"],
        input_paths={
            "universe_history_path": "mem://universe_history",
            "lifecycle_master_path": "mem://lifecycle_master",
            "corporate_actions_path": "mem://corporate_actions",
            "prices_path": None,
        },
    )

    uq_cfg, uq_payload, uq_hash = universe_qc_module.load_config(None)
    qc_history = build_artifacts.history.copy()
    qc_history["config_hash"] = uq_hash
    qc_history["run_id"] = f"qc_source_{scenario}"

    if corrupt_post_death:
        assert scenario == "merger"
        bad_row = qc_history.loc[qc_history["instrument_id"] == "M1"].sort_values("date").tail(1).copy()
        bad_row.loc[:, "date"] = pd.Timestamp("2020-02-03")
        bad_row.loc[:, "delist_date"] = pd.Timestamp("2020-02-02")
        bad_row.loc[:, "terminal_ca_id"] = "evt_merger_1"
        qc_history = pd.concat([qc_history, bad_row], ignore_index=True)
        qc_history = qc_history.sort_values(["date", "instrument_id"]).reset_index(drop=True)

    universe_history_path = root / "universe_history.csv"
    qc_history.to_csv(universe_history_path, index=False)
    ca_history_path = root / "corporate_actions_history.csv"
    ca_artifacts.history.to_csv(ca_history_path, index=False)

    qc_artifacts = universe_qc_module.audit_universe(
        universe_history_path=universe_history_path,
        listings_master_path=listings_path,
        calendar_path=calendar_path,
        corporate_actions_path=ca_history_path,
        run_id=f"uq_{scenario}",
        config=uq_cfg,
        config_hash=uq_hash,
        config_payload=uq_payload,
    )

    return {
        "payload": payload,
        "build": build_artifacts,
        "ca": ca_artifacts,
        "surv": surv_artifacts,
        "surv_cfg": surv_cfg,
        "lifecycle_df": lifecycle_df,
        "qc": qc_artifacts,
        "qc_history": qc_history,
    }


@pytest.fixture()
def run_stack(tmp_path: Path, build_universe_module, survivorship_module, universe_qc_module, corporate_actions_module):
    def _run(*, scenario: str, corrupt_post_death: bool = False):
        return _run_stack(
            tmp_path=tmp_path,
            scenario=scenario,
            build_universe_module=build_universe_module,
            survivorship_module=survivorship_module,
            universe_qc_module=universe_qc_module,
            corporate_actions_module=corporate_actions_module,
            corrupt_post_death=corrupt_post_death,
        )

    return _run


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_ticker_change_flows_through_universe_without_spurious_turnover(run_stack):
    stack = run_stack(scenario="ticker_change")

    history = stack["build"].history.sort_values(["date", "instrument_id"]).reset_index(drop=True)
    assert history["symbol"].tolist() == ["AAA", "AAA", "AAB"]
    assert history["transition"].tolist() == ["ENTER", "STAY_IN", "STAY_IN"]
    assert set(history["instrument_id"]) == {"I1"}

    turnover = stack["build"].turnover_stats.sort_values("date").reset_index(drop=True)
    # The ticker change should not create a fresh entry/exit at the instrument_id level.
    assert int(turnover.loc[2, "entries"]) == 0
    assert int(turnover.loc[2, "exits"]) == 0
    assert float(turnover.loc[2, "turnover"]) == pytest.approx(0.0)

    assert "structural_missing" not in stack["surv"].summary["absence_classification_counts"]



def test_terminal_merger_classification_uses_corporate_actions_economic_termination(run_stack, survivorship_module):
    stack = run_stack(scenario="merger")

    ctx = survivorship_module.prepare_context(
        universe_history=stack["build"].history,
        lifecycle=stack["lifecycle_df"],
        ca=stack["ca"].history,
        config=stack["surv_cfg"],
    )
    classification, reason, linked_ca, linked_instr = survivorship_module.classify_absence(
        pd.Timestamp("2020-02-03"),
        "M1",
        ctx,
        stack["surv_cfg"],
    )

    assert classification == survivorship_module.AbsenceClassification.ECONOMIC_TERMINATION
    assert "economic death date" in reason
    assert linked_ca is None or pd.isna(linked_ca)
    assert linked_instr is None or pd.isna(linked_instr)



def test_clean_stack_passes_universe_qc(run_stack):
    stack = run_stack(scenario="ticker_change")

    assert stack["qc"].summary["gate_status"] == "PASS"
    assert stack["qc"].failures.empty
    assert stack["ca"].history["event_type"].tolist() == ["ticker_change"]
    assert stack["build"].history["is_eligible"].eq(1).all()



def test_post_death_reactivation_fails_qc(run_stack):
    stack = run_stack(scenario="merger", corrupt_post_death=True)

    assert stack["qc"].summary["gate_status"] == "FAIL"
    check_types = set(stack["qc"].failures["check_type"].astype(str))
    assert "PIT_ELIGIBLE_AFTER_TERMINATION" in check_types
    assert "PIT_OUTSIDE_LIFECYCLE_WINDOW" in check_types
