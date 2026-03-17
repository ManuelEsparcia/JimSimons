from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# Import adapter
# -----------------------------------------------------------------------------


def _load_adjust_prices_module():
    module_names = [
        "simons_smallcap_swing.data.price.adjust_prices",
        "data.price.adjust_prices",
        "adjust_prices",
    ]
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue

    here = Path(__file__).resolve()
    candidate_paths = [
        here.parents[3] / "simons_smallcap_swing" / "data" / "price" / "adjust_prices.py",
        here.parents[2] / "data" / "price" / "adjust_prices.py",
        here.parents[1] / "adjust_prices.py",
        Path("/mnt/data/adjust_prices.py"),
    ]
    for path in candidate_paths:
        if path.exists():
            module_name = f"_adjust_prices_contract_{abs(hash(str(path)))}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module

    raise ModuleNotFoundError("Could not import adjust_prices.py from repo or /mnt/data.")


@pytest.fixture(scope="session")
def adjust_prices_module():
    return _load_adjust_prices_module()


# -----------------------------------------------------------------------------
# Helpers / fixtures
# -----------------------------------------------------------------------------


def _cfg(module, **adjustment_overrides: Any):
    adj_defaults = {
        "mode": module.AdjustmentMode.DUAL_OUTPUT.value,
        "include_special_cash_dividends": False,
        "allow_empty_corporate_actions": True,
        "unsupported_event_severity": module.Severity.WARN.value,
        "missing_prev_close_severity": module.Severity.WARN.value,
        "invalid_dividend_severity": module.Severity.WARN.value,
        "abort_on_fail_symbol": False,
    }
    adj_defaults.update(adjustment_overrides)
    return module.AdjustPricesConfig(
        reverse_split=module.ReverseSplitPolicy(extreme_threshold=20.0),
        conflict=module.ConflictPolicy(),
        adjustment=module.AdjustmentPolicy(**adj_defaults),
        output=module.OutputPolicy(compression="snappy"),
    )


@pytest.fixture()
def sample_prices() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "AAA", "instrument_id": "101", "date": "2026-01-01", "open": 100.0, "high": 110.0, "low": 90.0, "close": 100.0, "volume": 1000.0},
            {"symbol": "AAA", "instrument_id": "101", "date": "2026-01-02", "open": 50.0, "high": 55.0, "low": 45.0, "close": 50.0, "volume": 2000.0},
            {"symbol": "AAA", "instrument_id": "101", "date": "2026-01-03", "open": 48.0, "high": 49.0, "low": 47.0, "close": 48.0, "volume": 2200.0},
            {"symbol": "BBB", "instrument_id": "202", "date": "2026-01-01", "open": 20.0, "high": 21.0, "low": 19.0, "close": 20.0, "volume": 4000.0},
            {"symbol": "BBB", "instrument_id": "202", "date": "2026-01-02", "open": 21.0, "high": 22.0, "low": 20.0, "close": 21.0, "volume": 4100.0},
            {"symbol": "BBB", "instrument_id": "202", "date": "2026-01-03", "open": 22.0, "high": 23.0, "low": 21.0, "close": 22.0, "volume": 4200.0},
        ]
    )


@pytest.fixture()
def sample_events(adjust_prices_module) -> pd.DataFrame:
    module = adjust_prices_module
    return pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "instrument_id": "101",
                "event_type": module.CanonicalEventType.SPLIT.value,
                "ex_date": "2026-01-02",
                "split_ratio": 0.5,
                "source_provider": "primary_provider",
                "provider_event_id": "split_aaa_1",
            },
            {
                "symbol": "AAA",
                "instrument_id": "101",
                "event_type": module.CanonicalEventType.CASH_DIVIDEND.value,
                "ex_date": "2026-01-03",
                "cash_amount": 5.0,
                "source_provider": "primary_provider",
                "provider_event_id": "div_aaa_1",
            },
        ]
    )


@pytest.fixture()
def run_adjust(adjust_prices_module):
    module = adjust_prices_module

    def _run(
        prices: pd.DataFrame,
        events: pd.DataFrame,
        *,
        mode: str | None = None,
        run_id: str = "contract_test_run",
        asof_ts_utc: str = "2026-01-04T00:00:00Z",
        start_date: str = "2026-01-01",
        end_date: str = "2026-01-03",
    ):
        config = _cfg(module, mode=mode or module.AdjustmentMode.DUAL_OUTPUT.value)
        artifacts = module.adjust_prices(
            prices_raw=prices,
            corporate_actions=events,
            run_id=run_id,
            asof_ts_utc=asof_ts_utc,
            start_date=start_date,
            end_date=end_date,
            config=config,
        )
        return artifacts, config

    return _run


@pytest.fixture()
def fake_parquet(monkeypatch):
    def _fake_to_parquet(self, path, *args, **kwargs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_csv(path, index=False)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet, raising=True)



def _build_manifest(module, artifacts, config, *, run_id: str, prices_path: Path, events_path: Path):
    return module.build_manifest(
        run_id=run_id,
        asof_ts_utc="2026-01-04T00:00:00Z",
        mode=config.adjustment.mode,
        prices_path=prices_path,
        events_path=events_path,
        config=config,
        adjusted=artifacts.adjusted,
        events_applied=artifacts.events_applied,
        conflicts=artifacts.conflicts,
        summary=artifacts.summary,
    )


# -----------------------------------------------------------------------------
# Contract tests
# -----------------------------------------------------------------------------


def test_adjusted_output_contains_required_columns_for_dual_mode(run_adjust, sample_prices, sample_events):
    artifacts, _ = run_adjust(sample_prices, sample_events, mode="dual_output")

    required = {
        "symbol",
        "date",
        "run_id",
        "adjustment_mode",
        "asof_ts_utc",
        "open_raw",
        "high_raw",
        "low_raw",
        "close_raw",
        "volume_raw",
        "split_factor_cum",
        "dividend_factor_cum",
        "volume_adj",
        "event_count_applied",
        "has_unsupported_events",
        "severity_max",
        "instrument_id",
        "open_adj_split",
        "high_adj_split",
        "low_adj_split",
        "close_adj_split",
        "open_adj_total",
        "high_adj_total",
        "low_adj_total",
        "close_adj_total",
        "ret_1d_raw",
        "ret_1d_adj_split",
        "ret_1d_adj_total",
    }
    assert required.issubset(set(artifacts.adjusted.columns))
    assert artifacts.adjusted["adjustment_mode"].eq("dual_output").all()


@pytest.mark.parametrize(
    ("mode", "must_have", "semantic_assertion"),
    [
        ("split_only", {"close_adj_split", "ret_1d_adj_split"}, "no_total_columns"),
        ("total_return_like", {"close_adj_total", "ret_1d_adj_total"}, "total_columns_drive_mode"),
    ],
)
def test_mode_specific_output_semantics_are_unambiguous(run_adjust, sample_prices, sample_events, mode, must_have, semantic_assertion):
    artifacts, _ = run_adjust(sample_prices, sample_events, mode=mode)
    cols = set(artifacts.adjusted.columns)
    assert must_have.issubset(cols)
    assert artifacts.adjusted["adjustment_mode"].eq(mode).all()

    if semantic_assertion == "no_total_columns":
        assert "close_adj_total" not in cols
        assert "ret_1d_adj_total" not in cols
    elif semantic_assertion == "total_columns_drive_mode":
        assert "close_adj_total" in cols
        assert "ret_1d_adj_total" in cols
        # In the current implementation, split columns remain present in total_return_like
        # for traceability; the contract we enforce is that total columns exist and the mode is explicit.
        assert artifacts.adjusted["adjustment_mode"].eq("total_return_like").all()
    else:  # pragma: no cover
        raise AssertionError(f"Unhandled semantic assertion: {semantic_assertion}")



def test_factors_output_contains_required_columns(run_adjust, sample_prices, sample_events):
    artifacts, _ = run_adjust(sample_prices, sample_events)
    required = {
        "symbol",
        "date",
        "instrument_id",
        "split_factor_cum",
        "dividend_factor_cum",
        "combined_factor_cum",
        "volume_factor_cum",
        "event_count_applied",
        "split_event_count_applied",
        "dividend_event_count_applied",
        "severity_max",
        "has_unsupported_events",
    }
    assert required.issubset(set(artifacts.factors.columns))
    assert artifacts.factors.duplicated(["symbol", "date"]).sum() == 0



def test_events_applied_output_contains_required_columns(run_adjust, sample_prices, sample_events):
    artifacts, _ = run_adjust(sample_prices, sample_events)
    required = {
        "symbol",
        "instrument_id",
        "event_type",
        "application_date",
        "source_provider",
        "outcome",
        "severity",
        "failure_code",
        "split_factor_elementary",
        "dividend_factor_elementary",
        "include_in_split_chain",
        "include_in_dividend_chain",
    }
    assert required.issubset(set(artifacts.events_applied.columns))
    allowed_outcomes = {
        "APPLIED",
        "SKIPPED_UNSUPPORTED",
        "SKIPPED_SPECIAL_DIVIDEND_EXCLUDED",
        "SKIPPED_INVALID_DIVIDEND_FACTOR",
        "SKIPPED_MISSING_PREV_CLOSE",
        "SKIPPED_OUTSIDE_RANGE",
        "SKIPPED_NOT_VISIBLE_PIT",
        "SKIPPED_MISSING_FIELDS",
    }
    assert set(artifacts.events_applied["outcome"].astype(str).unique()).issubset(allowed_outcomes)



def test_conflicts_output_contract_is_honored_at_persistence_boundary(adjust_prices_module, run_adjust, sample_prices, sample_events, tmp_path, fake_parquet):
    module = adjust_prices_module
    artifacts, config = run_adjust(sample_prices, sample_events, run_id="conflict_contract")
    prices_path = tmp_path / "prices.csv"
    events_path = tmp_path / "events.csv"
    sample_prices.to_csv(prices_path, index=False)
    sample_events.to_csv(events_path, index=False)
    artifacts.manifest = _build_manifest(module, artifacts, config, run_id="conflict_contract", prices_path=prices_path, events_path=events_path)

    module.ensure_parquet_support = lambda: None
    module.write_outputs(tmp_path / "out", artifacts, "conflict_contract", config)

    conflicts_disk = pd.read_csv(tmp_path / "out" / "adjustment_conflicts.parquet")
    expected = {
        "symbol_key",
        "application_date",
        "event_type",
        "conflict_code",
        "severity",
        "chosen_provider",
        "providers_present",
        "details",
        "logical_event_key",
    }
    assert expected.issubset(set(conflicts_disk.columns))



def test_manifest_contains_required_metadata_and_artifacts(adjust_prices_module, run_adjust, sample_prices, sample_events, tmp_path):
    module = adjust_prices_module
    artifacts, config = run_adjust(sample_prices, sample_events, run_id="manifest_case")
    prices_path = tmp_path / "prices.csv"
    events_path = tmp_path / "events.csv"
    sample_prices.to_csv(prices_path, index=False)
    sample_events.to_csv(events_path, index=False)

    manifest = _build_manifest(module, artifacts, config, run_id="manifest_case", prices_path=prices_path, events_path=events_path)

    assert manifest["run_id"] == "manifest_case"
    assert manifest["module"] == "data/price/adjust_prices.py"
    assert manifest["adjustment_mode"] == config.adjustment.mode
    assert manifest["input_prices_snapshot"] == str(prices_path)
    assert manifest["input_corporate_actions_snapshot"] == str(events_path)
    assert manifest["symbol_count"] == int(artifacts.adjusted["symbol"].nunique())
    assert manifest["row_count"] == int(len(artifacts.adjusted))
    assert set(manifest["artifacts"].keys()) == {"adjusted", "factors", "events_applied", "conflicts", "manifest", "summary"}
    assert manifest["artifacts"]["manifest"].startswith("adjustment_manifest_")
    assert manifest["artifacts"]["summary"].startswith("adjustment_summary_")



def test_summary_contains_expected_aggregates(run_adjust, sample_prices, sample_events):
    artifacts, config = run_adjust(sample_prices, sample_events, mode="dual_output")
    summary = artifacts.summary
    assert summary["mode"] == config.adjustment.mode
    assert summary["symbol_count"] == int(artifacts.adjusted["symbol"].nunique())
    assert summary["row_count"] == int(len(artifacts.adjusted))
    assert summary["event_rows_total"] == int(len(artifacts.events_applied))
    assert summary["event_rows_applied"] == int((artifacts.events_applied["outcome"] == "APPLIED").sum())
    assert summary["coverage_by_symbol"]["AAA"] == 3
    assert "split" in summary["events_by_type"]



def test_outputs_are_sorted_deterministically(run_adjust, sample_prices, sample_events):
    artifacts, _ = run_adjust(sample_prices, sample_events)

    expected_adjusted = artifacts.adjusted.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    expected_factors = artifacts.factors.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    expected_events = artifacts.events_applied.sort_values(
        ["symbol", "application_date", "same_day_precedence_rank", "same_day_sequence"], kind="mergesort"
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(artifacts.adjusted.reset_index(drop=True), expected_adjusted, check_like=False)
    pd.testing.assert_frame_equal(artifacts.factors.reset_index(drop=True), expected_factors, check_like=False)
    pd.testing.assert_frame_equal(artifacts.events_applied.reset_index(drop=True), expected_events, check_like=False)



def test_same_input_same_snapshot_is_semantically_stable(run_adjust, sample_prices, sample_events):
    artifacts_a, _ = run_adjust(sample_prices, sample_events, run_id="stable_run")
    artifacts_b, _ = run_adjust(sample_prices, sample_events, run_id="stable_run")

    pd.testing.assert_frame_equal(artifacts_a.adjusted, artifacts_b.adjusted, check_dtype=False)
    pd.testing.assert_frame_equal(artifacts_a.factors, artifacts_b.factors, check_dtype=False)
    pd.testing.assert_frame_equal(artifacts_a.events_applied, artifacts_b.events_applied, check_dtype=False)
    pd.testing.assert_frame_equal(artifacts_a.conflicts, artifacts_b.conflicts, check_dtype=False)
    assert artifacts_a.summary == artifacts_b.summary



def test_write_outputs_materializes_all_expected_artifacts(adjust_prices_module, run_adjust, sample_prices, sample_events, tmp_path, fake_parquet):
    module = adjust_prices_module
    outdir = tmp_path / "out"
    prices_path = tmp_path / "prices.csv"
    events_path = tmp_path / "events.csv"
    sample_prices.to_csv(prices_path, index=False)
    sample_events.to_csv(events_path, index=False)

    artifacts, config = run_adjust(sample_prices, sample_events, run_id="persist_case")
    artifacts.manifest = _build_manifest(module, artifacts, config, run_id="persist_case", prices_path=prices_path, events_path=events_path)

    module.ensure_parquet_support = lambda: None
    module.write_outputs(outdir, artifacts, "persist_case", config)

    expected_files = {
        "adjusted_prices.parquet",
        "adjustment_factors.parquet",
        "adjustment_events_applied.parquet",
        "adjustment_conflicts.parquet",
        "adjustment_summary_persist_case.json",
        "adjustment_manifest_persist_case.json",
    }
    assert expected_files.issubset({p.name for p in outdir.iterdir()})

    adjusted_disk = pd.read_csv(outdir / "adjusted_prices.parquet")
    factors_disk = pd.read_csv(outdir / "adjustment_factors.parquet")
    events_disk = pd.read_csv(outdir / "adjustment_events_applied.parquet")
    conflicts_disk = pd.read_csv(outdir / "adjustment_conflicts.parquet")
    manifest_disk = json.loads((outdir / "adjustment_manifest_persist_case.json").read_text(encoding="utf-8"))
    summary_disk = json.loads((outdir / "adjustment_summary_persist_case.json").read_text(encoding="utf-8"))

    assert len(adjusted_disk) == len(artifacts.adjusted)
    assert len(factors_disk) == len(artifacts.factors)
    assert len(events_disk) == len(artifacts.events_applied)
    assert set(conflicts_disk.columns) == {
        "symbol_key",
        "application_date",
        "event_type",
        "conflict_code",
        "severity",
        "chosen_provider",
        "providers_present",
        "details",
        "logical_event_key",
    }
    assert manifest_disk["run_id"] == "persist_case"
    assert manifest_disk["artifacts"]["adjusted"] == "adjusted_prices.parquet"
    assert summary_disk["row_count"] == len(artifacts.adjusted)
