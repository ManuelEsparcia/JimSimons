from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from data.reference.build_reference import build_reference_data
from data.universe.build_universe import build_universe
from data.universe.corporate_actions import ALLOWED_EVENT_TYPES, build_corporate_actions
from simons_core.io.parquet_store import read_parquet


def _prepare_reference_and_universe(tmp_workspace: dict[str, Path]) -> tuple[Path, Path, Path]:
    reference_root = tmp_workspace["data"] / "reference"
    universe_root = tmp_workspace["data"] / "universe"
    build_reference_data(output_dir=reference_root, run_id="test_reference_corporate_actions")
    universe_result = build_universe(
        reference_root=reference_root,
        output_dir=universe_root,
        run_id="test_universe_corporate_actions",
    )
    return reference_root, universe_root, universe_result.universe_history


def _latest_interval_terminal_instruments(ticker_map: pd.DataFrame) -> set[str]:
    frame = ticker_map.copy()
    frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce").dt.normalize()
    frame["end_date"] = pd.to_datetime(frame["end_date"], errors="coerce").dt.normalize()

    terminal: set[str] = set()
    for instrument_id, sub in frame.groupby("instrument_id", sort=True):
        latest = sub.sort_values("start_date").iloc[-1]
        if pd.notna(latest["end_date"]):
            terminal.add(str(instrument_id))
    return terminal


def test_corporate_actions_mvp_builds_non_empty_with_canonical_schema(
    tmp_workspace: dict[str, Path],
) -> None:
    reference_root, universe_root, universe_history_path = _prepare_reference_and_universe(tmp_workspace)
    result = build_corporate_actions(
        reference_root=reference_root,
        universe_history_path=universe_history_path,
        output_dir=universe_root,
        run_id="test_corporate_actions_mvp",
    )

    assert result.corporate_actions_path.exists()
    assert result.summary_path.exists()
    actions = read_parquet(result.corporate_actions_path)
    assert len(actions) > 0

    required_columns = {
        "event_id",
        "instrument_id",
        "event_type",
        "effective_date",
        "announced_date",
        "source_start_date",
        "source_end_date",
        "old_ticker",
        "new_ticker",
        "split_factor",
        "event_value",
        "event_unit",
        "source_mode",
        "source_ref",
        "is_terminal",
        "run_id",
        "config_hash",
        "built_ts_utc",
    }
    assert required_columns.issubset(actions.columns)
    assert actions["instrument_id"].notna().all()
    assert set(actions["event_type"].unique()).issubset(set(ALLOWED_EVENT_TYPES))

    assert not actions.duplicated(["event_id"]).any()
    dedup_key = actions[
        ["instrument_id", "event_type", "effective_date", "old_ticker", "new_ticker"]
    ].copy()
    dedup_key["old_ticker"] = dedup_key["old_ticker"].fillna("__NULL__")
    dedup_key["new_ticker"] = dedup_key["new_ticker"].fillna("__NULL__")
    assert not dedup_key.duplicated(
        ["instrument_id", "event_type", "effective_date", "old_ticker", "new_ticker"]
    ).any()

    ticker_map = read_parquet(reference_root / "ticker_history_map.parquet")
    implied_ticker_change_instruments = set(
        ticker_map.groupby("instrument_id")["ticker"].nunique().loc[lambda series: series > 1].index.astype(str)
    )
    observed_ticker_change_instruments = set(
        actions.loc[actions["event_type"] == "ticker_change", "instrument_id"].astype(str).unique().tolist()
    )
    assert implied_ticker_change_instruments.issubset(observed_ticker_change_instruments)

    expected_terminal_instruments = _latest_interval_terminal_instruments(ticker_map)
    terminal_rows = actions[actions["event_type"].isin(["delisting", "listing_end"])]
    observed_terminal_instruments = set(terminal_rows["instrument_id"].astype(str).unique().tolist())
    assert expected_terminal_instruments.issubset(observed_terminal_instruments)
    assert terminal_rows["is_terminal"].all()

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["row_count"] == len(actions)
    assert summary["event_type_counts"] == actions["event_type"].value_counts().sort_index().to_dict()


def test_corporate_actions_mvp_accepts_optional_local_split_source(
    tmp_workspace: dict[str, Path],
) -> None:
    reference_root, universe_root, universe_history_path = _prepare_reference_and_universe(tmp_workspace)
    split_source = universe_root / "split_source.csv"
    pd.DataFrame(
        [
            {"instrument_id": "SIM0001", "effective_date": "2026-02-03", "split_factor": 0.5},
            {"instrument_id": "SIM0003", "effective_date": "2026-02-10", "split_factor": 2.0},
        ]
    ).to_csv(split_source, index=False)

    result = build_corporate_actions(
        reference_root=reference_root,
        universe_history_path=universe_history_path,
        split_source_path=split_source,
        output_dir=universe_root,
        run_id="test_corporate_actions_with_splits",
    )
    actions = read_parquet(result.corporate_actions_path)

    split_rows = actions[actions["event_type"].isin(["split", "reverse_split"])].copy()
    assert len(split_rows) == 2
    assert set(split_rows["event_type"].tolist()) == {"split", "reverse_split"}
    assert (split_rows["split_factor"] > 0).all()
    assert (split_rows["event_unit"] == "ratio").all()
    assert (split_rows["source_mode"] == "local_split_file").all()


def test_corporate_actions_mvp_fails_fast_on_unknown_split_instrument(
    tmp_workspace: dict[str, Path],
) -> None:
    reference_root, universe_root, universe_history_path = _prepare_reference_and_universe(tmp_workspace)
    split_source = universe_root / "split_source_invalid.csv"
    pd.DataFrame(
        [
            {"instrument_id": "UNKNOWN_1", "effective_date": "2026-02-03", "split_factor": 0.5},
        ]
    ).to_csv(split_source, index=False)

    with pytest.raises(ValueError, match="unknown instrument_id"):
        build_corporate_actions(
            reference_root=reference_root,
            universe_history_path=universe_history_path,
            split_source_path=split_source,
            output_dir=universe_root,
            run_id="test_corporate_actions_invalid_split_source",
        )
