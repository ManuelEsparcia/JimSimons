from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.reference.build_reference import build_reference_data
from data.universe.build_universe import build_universe
from data.universe.survivorship import run_survivorship_analysis
from simons_core.io.parquet_store import read_parquet


def _build_universe_inputs(tmp_workspace: dict[str, Path]) -> dict[str, Path]:
    reference_root = tmp_workspace["data"] / "reference"
    universe_root = tmp_workspace["data"] / "universe"
    build_reference_data(output_dir=reference_root, run_id="test_reference_survivorship")
    result = build_universe(
        reference_root=reference_root,
        output_dir=universe_root,
        run_id="test_universe_survivorship",
    )
    return {
        "reference_root": reference_root,
        "universe_root": universe_root,
        "history": result.universe_history,
        "current": result.universe_current,
        "ticker_map": reference_root / "ticker_history_map.parquet",
        "calendar": reference_root / "trading_calendar.parquet",
    }


def test_survivorship_mvp_generates_artifacts_and_summary(
    tmp_workspace: dict[str, Path],
) -> None:
    paths = _build_universe_inputs(tmp_workspace)
    output_dir = tmp_workspace["artifacts"] / "survivorship_pass_case"

    result = run_survivorship_analysis(
        universe_history_path=paths["history"],
        universe_current_path=paths["current"],
        ticker_history_map_path=paths["ticker_map"],
        trading_calendar_path=paths["calendar"],
        baseline_mode="current_eligible",
        output_dir=output_dir,
        run_id="test_survivorship_pass_case",
    )

    assert result.summary_path.exists()
    assert result.daily_path.exists()
    assert result.membership_diff_path.exists()
    assert result.symbol_level_path.exists()
    assert result.manifest_path.exists()

    daily = read_parquet(result.daily_path)
    membership = read_parquet(result.membership_diff_path)
    symbol_level = read_parquet(result.symbol_level_path)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert len(daily) > 0
    assert len(membership) > 0
    assert len(symbol_level) > 0

    expected_daily = {
        "date",
        "n_names_pit",
        "n_names_naive",
        "overlap_count",
        "pit_only_count",
        "naive_only_count",
        "jaccard_membership",
        "pct_pit_retained_by_naive",
        "pct_naive_ex_post_names",
    }
    expected_membership = {
        "date",
        "instrument_id",
        "diff_type",
        "absence_classification",
    }
    expected_symbol = {
        "instrument_id",
        "first_seen_pit",
        "last_seen_pit",
        "appears_in_naive",
        "naive_backfilled_flag",
        "terminal_only_flag",
    }
    assert expected_daily.issubset(set(daily.columns))
    assert expected_membership.issubset(set(membership.columns))
    assert expected_symbol.issubset(set(symbol_level.columns))
    assert daily["jaccard_membership"].between(0.0, 1.0).all()

    assert summary["baseline_mode"] == "current_eligible"
    assert "severity_score" in summary
    assert "bias_level" in summary
    assert "recommended_action" in summary


def test_survivorship_detects_bias_when_current_drops_historical_name(
    tmp_workspace: dict[str, Path],
) -> None:
    paths = _build_universe_inputs(tmp_workspace)
    current = read_parquet(paths["current"]).copy()
    current = current[current["instrument_id"] != "SIM0005"].copy()
    degraded_current_path = tmp_workspace["data"] / "universe" / "universe_current_degraded.parquet"
    current.to_parquet(degraded_current_path, index=False)

    result = run_survivorship_analysis(
        universe_history_path=paths["history"],
        universe_current_path=degraded_current_path,
        ticker_history_map_path=paths["ticker_map"],
        trading_calendar_path=paths["calendar"],
        baseline_mode="current_eligible",
        output_dir=tmp_workspace["artifacts"] / "survivorship_missing_name",
        run_id="test_survivorship_missing_name",
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    daily = read_parquet(result.daily_path)
    symbol_level = read_parquet(result.symbol_level_path)

    assert summary["n_pit_only_total"] > 0
    assert summary["mean_pct_pit_retained_by_naive"] < 1.0
    assert (daily["pit_only_count"] > 0).any()

    sim0005 = symbol_level.loc[symbol_level["instrument_id"] == "SIM0005"]
    assert len(sim0005) == 1
    assert not bool(sim0005.iloc[0]["appears_in_naive"])


def test_survivorship_flags_naive_backfill_in_late_listing_case(
    tmp_workspace: dict[str, Path],
) -> None:
    paths = _build_universe_inputs(tmp_workspace)
    history = read_parquet(paths["history"]).copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.normalize()

    # Fabricate a late listing by dropping early SIM0005 rows from PIT history.
    cutoff = pd.Timestamp("2026-02-17")
    keep_mask = (history["instrument_id"] != "SIM0005") | (history["date"] >= cutoff)
    late_history = history.loc[keep_mask].copy()
    late_history_path = tmp_workspace["data"] / "universe" / "universe_history_late_listing.parquet"
    late_history.to_parquet(late_history_path, index=False)

    result = run_survivorship_analysis(
        universe_history_path=late_history_path,
        universe_current_path=paths["current"],
        ticker_history_map_path=paths["ticker_map"],
        trading_calendar_path=paths["calendar"],
        baseline_mode="current_eligible",
        output_dir=tmp_workspace["artifacts"] / "survivorship_backfill_case",
        run_id="test_survivorship_backfill_case",
    )

    daily = read_parquet(result.daily_path)
    membership = read_parquet(result.membership_diff_path)
    symbol_level = read_parquet(result.symbol_level_path)

    sim0005 = symbol_level.loc[symbol_level["instrument_id"] == "SIM0005"]
    assert len(sim0005) == 1
    assert bool(sim0005.iloc[0]["naive_backfilled_flag"])
    assert (daily["naive_only_count"] > 0).any()
    assert daily["jaccard_membership"].between(0.0, 1.0).all()
    assert (membership["absence_classification"] == "naive_backfill").any()
