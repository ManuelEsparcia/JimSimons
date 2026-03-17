from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module():
    candidates = [
        "simons_smallcap_swing.labels.event_labels",
        "labels.event_labels",
        "event_labels",
    ]
    last_exc = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:  # pragma: no cover - import fallback path
            last_exc = exc
    raise last_exc  # type: ignore[misc]


ev = _load_module()

EventLabelConfig = ev.EventLabelConfig
TradingCalendar = ev.TradingCalendar
QCFailure = ev.QCFailure
build_event_labels = ev.build_event_labels
canonicalize_event_types = ev.canonicalize_event_types
resolve_event_timestamps = ev.resolve_event_timestamps
attach_event_horizons = ev.attach_event_horizons
resolve_overlaps = ev.resolve_overlaps
compute_event_returns = ev.compute_event_returns
attach_benchmark_and_abnormal_returns = ev.attach_benchmark_and_abnormal_returns


BASE_DATES = pd.bdate_range("2026-01-05", periods=15)


def _write_csv(df: pd.DataFrame, path: Path) -> str:
    df.to_csv(path, index=False)
    return str(path)


def _write_json(payload: dict[str, object], path: Path) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _base_config_mapping() -> dict[str, object]:
    return {
        "event_to_horizons_map": {
            "EARNINGS": [1, 3],
            "CORP_NEWS": [1, 3],
            "OPEN_GAP": [1, 2],
            "TECH_BREAK": [3],
            "VOL_SHOCK": [1],
            "MICROSTRUCTURE_SHOCK": [1],
        },
        "entry_exit_convention": "open_to_open",
        "timestamp_policy": "strict",
        "overlap_policy": "first_event_wins_with_cooldown",
        "cooldown_policy": "max_horizon",
        "min_family_sample": 1,
        "classification_policy": {
            "enabled": True,
            "top_quantile": 0.75,
            "bottom_quantile": 0.25,
            "within_family": True,
            "emit_binary": True,
            "emit_ternary": True,
        },
        "abnormal_mode": "auto",
        "benchmark_mode_default_corporate": "sector_market_adjusted",
        "benchmark_mode_default_technical": "none",
        "missing_cost_policy": "strict_invalidate",
        "net_of_costs": True,
        "policy_version": "unit-test",
        "taxonomy_version": "unit-test",
    }


def _make_prices() -> pd.DataFrame:
    symbols = {
        "AAA": 10.0,
        "BBB": 20.0,
        "CCC": 30.0,
    }
    rows: list[dict[str, object]] = []
    for symbol, start_open in symbols.items():
        step = {"AAA": 0.2, "BBB": 0.3, "CCC": 0.4}[symbol]
        for idx, date in enumerate(BASE_DATES):
            open_px = start_open + step * idx
            close_px = open_px * (1.0 + 0.01 + 0.001 * idx)
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "adj_open": round(open_px, 6),
                    "adj_close": round(close_px, 6),
                }
            )
    return pd.DataFrame(rows)


def _make_universe() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol in ("AAA", "BBB", "CCC"):
        for date in BASE_DATES:
            rows.append({"date": date, "symbol": symbol, "is_eligible": True})
    return pd.DataFrame(rows)


def _make_costs() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol in ("AAA", "BBB", "CCC"):
        for date in BASE_DATES:
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "entry_cost": 0.0005,
                    "exit_cost": 0.0005,
                }
            )
    return pd.DataFrame(rows)


def _make_sector_mapping() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "sector": ["Tech", "Health", "Industrial"],
        }
    )


def _make_benchmark_returns() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    benchmark_spec = {
        (pd.Timestamp("2026-01-05"), 1): {"market": 0.005, "sector": {"Tech": 0.004}},
        (pd.Timestamp("2026-01-05"), 3): {"market": 0.012, "sector": {"Tech": 0.006}},
        (pd.Timestamp("2026-01-06"), 1): {"market": 0.006, "sector": {"Health": 0.002}},
        (pd.Timestamp("2026-01-06"), 3): {"market": 0.014, "sector": {"Health": 0.005}},
    }
    for (date, horizon), payload in benchmark_spec.items():
        rows.append(
            {
                "date": date,
                "horizon_days": horizon,
                "benchmark_type": "market",
                "benchmark_return": payload["market"],
            }
        )
        for sector, value in payload["sector"].items():
            rows.append(
                {
                    "date": date,
                    "horizon_days": horizon,
                    "benchmark_type": "sector",
                    "sector": sector,
                    "benchmark_return": value,
                }
            )
    return pd.DataFrame(rows)


def _make_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "earn_pre",
                "symbol": "AAA",
                "event_type": "earnings",
                "event_source": "newswire",
                "event_timestamp_known": "2026-01-05 07:00:00",
                "timing_hint": "pre_market",
            },
            {
                "event_id": "news_after",
                "symbol": "BBB",
                "event_type": "corp_news",
                "event_source": "newswire",
                "event_timestamp_known": "2026-01-05 16:30:00",
                "timing_hint": "after_close",
            },
            {
                "event_id": "gap_open",
                "symbol": "CCC",
                "event_type": "open_gap",
                "event_source": "scanner",
                "event_timestamp_known": "2026-01-06 09:20:00",
                "timing_hint": "at_open",
            },
        ]
    )


@pytest.fixture()
def event_bundle() -> dict[str, pd.DataFrame]:
    return {
        "prices": _make_prices(),
        "universe": _make_universe(),
        "costs": _make_costs(),
        "sectors": _make_sector_mapping(),
        "benchmarks": _make_benchmark_returns(),
        "events": _make_events(),
        "calendar": pd.DataFrame({"date": BASE_DATES}),
    }


@pytest.fixture()
def event_config() -> dict[str, object]:
    return _base_config_mapping()


@pytest.fixture()
def tmp_input_paths(
    tmp_path: Path,
    event_bundle: dict[str, pd.DataFrame],
    event_config: dict[str, object],
) -> dict[str, str]:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    return {
        "events": _write_csv(event_bundle["events"], input_dir / "events.csv"),
        "prices": _write_csv(event_bundle["prices"], input_dir / "prices.csv"),
        "universe": _write_csv(event_bundle["universe"], input_dir / "universe.csv"),
        "calendar": _write_csv(event_bundle["calendar"], input_dir / "calendar.csv"),
        "costs": _write_csv(event_bundle["costs"], input_dir / "costs.csv"),
        "benchmarks": _write_csv(event_bundle["benchmarks"], input_dir / "benchmarks.csv"),
        "sectors": _write_csv(event_bundle["sectors"], input_dir / "sectors.csv"),
        "config": _write_json(event_config, input_dir / "config.json"),
        "output_dir": str(tmp_path / "out"),
    }


def test_build_event_labels_end_to_end_persists_outputs_and_expected_schema(
    tmp_input_paths: dict[str, str],
) -> None:
    result = build_event_labels(
        event_source_path=tmp_input_paths["events"],
        prices_pit_path=tmp_input_paths["prices"],
        universe_history_path=tmp_input_paths["universe"],
        trading_calendar_path=tmp_input_paths["calendar"],
        event_label_config_path=tmp_input_paths["config"],
        execution_costs_path=tmp_input_paths["costs"],
        benchmark_returns_path=tmp_input_paths["benchmarks"],
        sector_mapping_path=tmp_input_paths["sectors"],
        run_id="unit_event",
        output_dir=tmp_input_paths["output_dir"],
    )

    episodes = result["episodes"]
    artifacts = result["artifacts"]

    assert not episodes.empty
    required_cols = {
        "event_id",
        "symbol",
        "event_type",
        "horizon_days",
        "event_ret_gross",
        "event_ret_net",
        "event_alpha",
        "label_event_cont",
        "event_label_valid_flag",
        "event_start",
        "event_end",
        "run_id",
    }
    assert required_cols.issubset(episodes.columns)
    assert episodes["run_id"].eq("unit_event").all()

    for path in artifacts.values():
        assert Path(path).exists(), f"missing output artifact: {path}"

    earn_pre_1d = episodes.loc[(episodes["event_id"] == "earn_pre") & (episodes["horizon_days"] == 1)].iloc[0]
    assert bool(earn_pre_1d["event_label_valid_flag"])
    assert math.isclose(float(earn_pre_1d["event_ret_gross"]), 0.02, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(earn_pre_1d["event_ret_net"]), 0.019, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(earn_pre_1d["event_alpha"]), 0.015, rel_tol=0.0, abs_tol=1e-12)

    windows_payload = json.loads(Path(artifacts["windows_for_splits"]).read_text(encoding="utf-8"))
    assert windows_payload["run_id"] == "unit_event"
    assert any(w["event_id"] == "earn_pre" and w["horizon_days"] == 1 for w in windows_payload["windows"])


def test_resolve_event_timestamps_maps_pre_market_after_close_and_intraday_to_expected_sessions(
    event_bundle: dict[str, pd.DataFrame],
    event_config: dict[str, object],
) -> None:
    prices = event_bundle["prices"]
    calendar = TradingCalendar.from_sources(prices)
    config = EventLabelConfig.from_mapping(event_config)

    raw = pd.DataFrame(
        [
            {
                "event_id": "pre",
                "symbol": "AAA",
                "event_type": "earnings",
                "event_timestamp_known": "2026-01-05 07:00:00",
                "timing_hint": "pre_market",
            },
            {
                "event_id": "post",
                "symbol": "AAA",
                "event_type": "earnings",
                "event_timestamp_known": "2026-01-05 16:30:00",
                "timing_hint": "after_close",
            },
            {
                "event_id": "intra",
                "symbol": "AAA",
                "event_type": "earnings",
                "event_timestamp_known": "2026-01-05 12:30:00",
                "timing_hint": "intraday",
            },
        ]
    )

    canonical = canonicalize_event_types(raw, config)
    resolved = resolve_event_timestamps(canonical, calendar, config)
    by_id = resolved.set_index("event_id")

    assert pd.Timestamp(by_id.loc["pre", "event_tradable_session"]) == pd.Timestamp("2026-01-05")
    assert by_id.loc["pre", "tradability_resolution"] == "pre_market"

    assert pd.Timestamp(by_id.loc["post", "event_tradable_session"]) == pd.Timestamp("2026-01-06")
    assert by_id.loc["post", "tradability_resolution"] == "after_close"

    assert pd.Timestamp(by_id.loc["intra", "event_tradable_session"]) == pd.Timestamp("2026-01-06")
    assert by_id.loc["intra", "tradability_resolution"] == "intraday_next_open"


def test_resolve_overlaps_rejects_second_event_inside_cooldown(
    event_bundle: dict[str, pd.DataFrame],
    event_config: dict[str, object],
) -> None:
    prices = event_bundle["prices"]
    calendar = TradingCalendar.from_sources(prices)
    config = EventLabelConfig.from_mapping(event_config)

    raw = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "symbol": "AAA",
                "event_type": "earnings",
                "event_timestamp_known": "2026-01-05 07:00:00",
                "timing_hint": "pre_market",
            },
            {
                "event_id": "e2",
                "symbol": "AAA",
                "event_type": "earnings",
                "event_timestamp_known": "2026-01-06 07:00:00",
                "timing_hint": "pre_market",
            },
        ]
    )

    canonical = canonicalize_event_types(raw, config)
    resolved = resolve_event_timestamps(canonical, calendar, config)
    with_horizons = attach_event_horizons(resolved, config)
    overlapped = resolve_overlaps(with_horizons, calendar, config).set_index("event_id")

    assert bool(overlapped.loc["e1", "overlap_rejected"]) is False
    assert pd.isna(overlapped.loc["e1", "event_exclusion_reason"])
    assert bool(overlapped.loc["e2", "overlap_rejected"]) is True
    assert overlapped.loc["e2", "event_exclusion_reason"] == "OVERLAP_REJECTED"


def test_compute_event_returns_uses_delisting_return_when_exit_price_missing() -> None:
    config = EventLabelConfig.from_mapping(
        {
            **_base_config_mapping(),
            "event_to_horizons_map": {"EARNINGS": [1]},
        }
    )

    prices = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-05"), "symbol": "AAA", "adj_open": 10.0, "adj_close": 10.1},
            {"date": pd.Timestamp("2026-01-05"), "symbol": "BBB", "adj_open": 20.0, "adj_close": 20.1},
            {"date": pd.Timestamp("2026-01-06"), "symbol": "BBB", "adj_open": 20.2, "adj_close": 20.3},
        ]
    )
    price_schema = {"date": "date", "symbol": "symbol", "open": "adj_open", "close": "adj_close"}
    calendar = TradingCalendar.from_sources(prices)

    raw = pd.DataFrame(
        [
            {
                "event_id": "delist_evt",
                "symbol": "AAA",
                "event_type": "earnings",
                "event_timestamp_known": "2026-01-05 07:00:00",
                "timing_hint": "pre_market",
            }
        ]
    )
    canonical = canonicalize_event_types(raw, config)
    resolved = resolve_event_timestamps(canonical, calendar, config)
    with_horizons = attach_event_horizons(resolved, config)
    overlapped = resolve_overlaps(with_horizons, calendar, config)

    delisting = pd.DataFrame(
        [{"date": pd.Timestamp("2026-01-06"), "symbol": "AAA", "delisting_return": -0.4}]
    )

    episodes = compute_event_returns(
        overlapped,
        prices,
        price_schema,
        calendar,
        config,
        delisting_returns=delisting,
    )
    row = episodes.iloc[0]

    assert bool(row["delisting_return_used"]) is True
    assert math.isclose(float(row["event_ret_gross"]), -0.4, rel_tol=0.0, abs_tol=1e-12)
    assert pd.isna(row["event_exclusion_reason"])


def test_attach_benchmark_and_abnormal_returns_sector_market_adjusted() -> None:
    config = EventLabelConfig.from_mapping(_base_config_mapping())
    episodes = pd.DataFrame(
        [
            {
                "event_id": "evt1",
                "symbol": "AAA",
                "event_type": "EARNINGS",
                "event_tradable_session": pd.Timestamp("2026-01-05"),
                "horizon_days": 3,
                "event_ret_gross": 0.051,
                "event_ret_net": 0.05,
                "event_exclusion_reason": pd.NA,
            }
        ]
    )
    sector_mapping = pd.DataFrame({"symbol": ["AAA"], "sector": ["Tech"]})
    benchmark = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-05"),
                "horizon_days": 3,
                "benchmark_type": "market",
                "benchmark_return": 0.01,
            },
            {
                "date": pd.Timestamp("2026-01-05"),
                "horizon_days": 3,
                "benchmark_type": "sector",
                "sector": "Tech",
                "benchmark_return": 0.02,
            },
        ]
    )

    out = attach_benchmark_and_abnormal_returns(
        episodes,
        config,
        benchmark_returns=benchmark,
        sector_mapping=sector_mapping,
    )
    row = out.iloc[0]

    assert row["abnormal_mode_used"] == "sector_market_adjusted"
    assert row["sector"] == "Tech"
    assert math.isclose(float(row["event_alpha"]), 0.03, rel_tol=0.0, abs_tol=1e-12)
    assert pd.isna(row["event_exclusion_reason"])


def test_build_event_labels_raises_qcfailure_on_duplicate_event_ids(
    tmp_path: Path,
    event_bundle: dict[str, pd.DataFrame],
    event_config: dict[str, object],
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    duplicate_events = event_bundle["events"].copy()
    duplicate_events.loc[1, "event_id"] = duplicate_events.loc[0, "event_id"]

    with pytest.raises(QCFailure, match="unique_event_id"):
        build_event_labels(
            event_source_path=_write_csv(duplicate_events, input_dir / "events.csv"),
            prices_pit_path=_write_csv(event_bundle["prices"], input_dir / "prices.csv"),
            universe_history_path=_write_csv(event_bundle["universe"], input_dir / "universe.csv"),
            trading_calendar_path=_write_csv(event_bundle["calendar"], input_dir / "calendar.csv"),
            event_label_config_path=_write_json(event_config, input_dir / "config.json"),
            execution_costs_path=_write_csv(event_bundle["costs"], input_dir / "costs.csv"),
            benchmark_returns_path=_write_csv(event_bundle["benchmarks"], input_dir / "benchmarks.csv"),
            sector_mapping_path=_write_csv(event_bundle["sectors"], input_dir / "sectors.csv"),
            run_id="dup_ids",
            output_dir=str(tmp_path / "out"),
        )
