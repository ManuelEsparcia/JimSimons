from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.price.adjust_prices import adjust_prices
from data.price.fetch_prices import fetch_prices
from data.price.qc_prices import run_price_qc
from data.reference.build_reference import build_reference_data
from data.universe.build_universe import build_universe
from data.universe.corporate_actions import build_corporate_actions
from simons_core.io.parquet_store import read_parquet


def _prepare_price_pipeline_with_split_events(
    tmp_workspace: dict[str, Path],
) -> tuple[Path, Path, Path, Path, Path]:
    reference_root = tmp_workspace["data"] / "reference"
    universe_root = tmp_workspace["data"] / "universe"
    price_root = tmp_workspace["data"] / "price"

    build_reference_data(output_dir=reference_root, run_id="test_reference_adjust_split_only")
    universe_result = build_universe(
        reference_root=reference_root,
        output_dir=universe_root,
        run_id="test_universe_adjust_split_only",
    )
    raw_result = fetch_prices(
        reference_root=reference_root,
        universe_history_path=universe_result.universe_history,
        output_dir=price_root,
        run_id="test_fetch_adjust_split_only",
    )

    split_source = universe_root / "split_events_for_adjust.csv"
    pd.DataFrame(
        [
            {"instrument_id": "SIM0001", "effective_date": "2026-02-03", "split_factor": 0.5},
            {"instrument_id": "SIM0002", "effective_date": "2026-02-10", "split_factor": 2.0},
        ]
    ).to_csv(split_source, index=False)

    corporate_result = build_corporate_actions(
        reference_root=reference_root,
        universe_history_path=universe_result.universe_history,
        split_source_path=split_source,
        output_dir=universe_root,
        run_id="test_corporate_actions_for_adjust_split_only",
    )
    adjusted_result = adjust_prices(
        raw_prices_path=raw_result.raw_prices_path,
        corporate_actions_path=corporate_result.corporate_actions_path,
        output_dir=price_root,
        run_id="test_adjust_split_only_v2",
    )
    return (
        raw_result.raw_prices_path,
        adjusted_result.adjusted_prices_path,
        adjusted_result.adjustment_report_path,
        corporate_result.corporate_actions_path,
        reference_root,
    )


def _expected_split_factor(
    *,
    instrument_id: str,
    date: pd.Timestamp,
    split_events: pd.DataFrame,
) -> tuple[float, int]:
    inst_events = split_events.loc[
        (split_events["instrument_id"] == instrument_id) & (split_events["effective_date"] > date)
    ]
    if inst_events.empty:
        return 1.0, 0
    factor = float(inst_events["split_factor"].prod())
    count_unique_dates = int(inst_events["effective_date"].nunique())
    return factor, count_unique_dates


def test_adjust_prices_split_only_applies_expected_factors_and_traces_sources(
    tmp_workspace: dict[str, Path],
) -> None:
    raw_path, adjusted_path, report_path, corporate_path, _ = _prepare_price_pipeline_with_split_events(
        tmp_workspace
    )
    raw = read_parquet(raw_path).copy()
    adjusted = read_parquet(adjusted_path).copy()
    corporate = read_parquet(corporate_path).copy()

    assert adjusted_path.exists()
    assert report_path.exists()
    assert len(adjusted) > 0
    assert not adjusted.duplicated(["date", "instrument_id"]).any()

    required_columns = {
        "date",
        "instrument_id",
        "ticker",
        "open_adj",
        "high_adj",
        "low_adj",
        "close_adj",
        "volume_adj",
        "adjustment_mode",
        "cumulative_split_factor",
        "applied_split_events_count",
        "source_raw_path",
        "source_corporate_actions_path",
    }
    assert required_columns.issubset(adjusted.columns)
    assert (adjusted["adjustment_mode"] == "split_only").all()

    corporate["event_type"] = corporate["event_type"].astype(str).str.lower()
    split_events = corporate.loc[
        corporate["event_type"].isin(["split", "reverse_split"]),
        ["instrument_id", "effective_date", "split_factor"],
    ].copy()
    split_events["instrument_id"] = split_events["instrument_id"].astype(str)
    split_events["effective_date"] = pd.to_datetime(
        split_events["effective_date"], errors="coerce"
    ).dt.normalize()
    split_events["split_factor"] = pd.to_numeric(split_events["split_factor"], errors="coerce")

    assert len(split_events) > 0
    assert len(corporate.loc[~corporate["event_type"].isin(["split", "reverse_split"])]) > 0

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    adjusted["date"] = pd.to_datetime(adjusted["date"], errors="coerce").dt.normalize()
    joined = raw.merge(adjusted, on=["date", "instrument_id", "ticker"], how="inner")
    assert len(joined) == len(raw)

    expected_factors: list[float] = []
    expected_counts: list[int] = []
    for row in joined.itertuples(index=False):
        factor, count = _expected_split_factor(
            instrument_id=str(row.instrument_id),
            date=pd.Timestamp(row.date).normalize(),
            split_events=split_events,
        )
        expected_factors.append(factor)
        expected_counts.append(count)

    expected_factors_arr = np.array(expected_factors, dtype=float)
    expected_counts_arr = np.array(expected_counts, dtype=int)

    assert np.isclose(joined["cumulative_split_factor"].to_numpy(dtype=float), expected_factors_arr).all()
    assert (joined["applied_split_events_count"].to_numpy(dtype=int) == expected_counts_arr).all()

    assert np.isclose(
        joined["open_adj"].to_numpy(dtype=float),
        joined["open"].to_numpy(dtype=float) * expected_factors_arr,
    ).all()
    assert np.isclose(
        joined["high_adj"].to_numpy(dtype=float),
        joined["high"].to_numpy(dtype=float) * expected_factors_arr,
    ).all()
    assert np.isclose(
        joined["low_adj"].to_numpy(dtype=float),
        joined["low"].to_numpy(dtype=float) * expected_factors_arr,
    ).all()
    assert np.isclose(
        joined["close_adj"].to_numpy(dtype=float),
        joined["close"].to_numpy(dtype=float) * expected_factors_arr,
    ).all()
    assert np.isclose(
        joined["volume_adj"].to_numpy(dtype=float),
        joined["volume"].to_numpy(dtype=float) / expected_factors_arr,
    ).all()

    # Instrument with no split event must stay at factor 1.
    sim0005 = joined.loc[joined["instrument_id"] == "SIM0005"]
    assert len(sim0005) > 0
    assert np.isclose(sim0005["cumulative_split_factor"].to_numpy(dtype=float), 1.0).all()
    assert np.isclose(sim0005["close_adj"].to_numpy(dtype=float), sim0005["close"].to_numpy(dtype=float)).all()

    # There must be at least some adjusted rows with factor != 1.
    assert (~np.isclose(joined["cumulative_split_factor"].to_numpy(dtype=float), 1.0)).any()


def test_adjust_prices_split_only_remains_compatible_with_price_qc(
    tmp_workspace: dict[str, Path],
) -> None:
    raw_path, adjusted_path, _report_path, _corporate_path, reference_root = (
        _prepare_price_pipeline_with_split_events(tmp_workspace)
    )
    qc_root = tmp_workspace["artifacts"] / "price_qc_split_only"

    result = run_price_qc(
        raw_prices_path=raw_path,
        adjusted_prices_path=adjusted_path,
        trading_calendar_path=reference_root / "trading_calendar.parquet",
        ticker_history_map_path=reference_root / "ticker_history_map.parquet",
        output_dir=qc_root,
        run_id="test_price_qc_split_only_v2",
    )

    assert result.gate_status == "PASS"
    assert result.summary_path.exists()
    assert result.row_level_path.exists()
    assert result.symbol_level_path.exists()
    assert result.failures_path.exists()
    assert result.manifest_path.exists()
