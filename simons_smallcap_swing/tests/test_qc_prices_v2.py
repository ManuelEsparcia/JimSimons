from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.price.adjust_prices import adjust_prices
from data.price.fetch_prices import fetch_prices
from data.price.qc_prices import run_price_qc
from data.reference.build_reference import build_reference_data
from data.universe.build_universe import build_universe
from data.universe.corporate_actions import build_corporate_actions
from simons_core.io.parquet_store import read_parquet


def _build_healthy_qc_inputs(tmp_workspace: dict[str, Path]) -> dict[str, Path]:
    reference_root = tmp_workspace["data"] / "reference"
    universe_root = tmp_workspace["data"] / "universe"
    price_root = tmp_workspace["data"] / "price"

    build_reference_data(output_dir=reference_root, run_id="test_reference_qc_v2")
    universe_result = build_universe(
        reference_root=reference_root,
        output_dir=universe_root,
        run_id="test_universe_qc_v2",
    )
    raw_result = fetch_prices(
        reference_root=reference_root,
        universe_history_path=universe_result.universe_history,
        output_dir=price_root,
        run_id="test_fetch_qc_v2",
    )

    split_source = universe_root / "splits_qc_v2.csv"
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
        run_id="test_corporate_qc_v2",
    )
    adjusted_result = adjust_prices(
        raw_prices_path=raw_result.raw_prices_path,
        corporate_actions_path=corporate_result.corporate_actions_path,
        output_dir=price_root,
        run_id="test_adjust_qc_v2",
    )

    return {
        "raw": raw_result.raw_prices_path,
        "adjusted": adjusted_result.adjusted_prices_path,
        "calendar": reference_root / "trading_calendar.parquet",
        "ticker_map": reference_root / "ticker_history_map.parquet",
        "corporate_actions": corporate_result.corporate_actions_path,
    }


def _run_qc(
    paths: dict[str, Path],
    tmp_workspace: dict[str, Path],
    *,
    raw_override: Path | None = None,
    adjusted_override: Path | None = None,
    corporate_override: Path | None = None,
    run_id: str,
):
    output_dir = tmp_workspace["artifacts"] / run_id
    return run_price_qc(
        raw_prices_path=raw_override or paths["raw"],
        adjusted_prices_path=adjusted_override or paths["adjusted"],
        trading_calendar_path=paths["calendar"],
        ticker_history_map_path=paths["ticker_map"],
        corporate_actions_path=corporate_override or paths["corporate_actions"],
        output_dir=output_dir,
        run_id=run_id,
    )


def test_qc_prices_v2_passes_on_canonical_case_and_emits_artifacts(
    tmp_workspace: dict[str, Path],
) -> None:
    paths = _build_healthy_qc_inputs(tmp_workspace)
    result = _run_qc(paths, tmp_workspace, run_id="test_qc_v2_pass")

    assert result.gate_status == "PASS"
    assert result.summary_path.exists()
    assert result.row_level_path.exists()
    assert result.symbol_level_path.exists()
    assert result.session_level_path.exists()
    assert result.corporate_actions_consistency_path.exists()
    assert result.failures_path.exists()
    assert result.manifest_path.exists()

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    expected_keys = {
        "gate_status",
        "n_rows_raw",
        "n_rows_adjusted",
        "n_symbols",
        "n_sessions",
        "n_fail_rows",
        "n_warn_rows",
        "worst_symbol",
        "worst_session",
        "min_symbol_coverage",
        "median_symbol_coverage",
        "n_extreme_return_flags",
        "n_split_consistency_failures",
        "adjustment_mode_detected",
        "source_modes_detected",
    }
    assert expected_keys.issubset(summary.keys())


def test_qc_prices_v2_detects_raw_adjusted_inconsistency(
    tmp_workspace: dict[str, Path],
) -> None:
    paths = _build_healthy_qc_inputs(tmp_workspace)
    adjusted = read_parquet(paths["adjusted"]).copy()
    adjusted.loc[0, "close_adj"] = float(adjusted.loc[0, "close_adj"]) * 1.25
    broken_adjusted_path = tmp_workspace["data"] / "price" / "adjusted_broken.parquet"
    adjusted.to_parquet(broken_adjusted_path, index=False)

    result = _run_qc(
        paths,
        tmp_workspace,
        adjusted_override=broken_adjusted_path,
        run_id="test_qc_v2_broken_adjusted",
    )
    assert result.gate_status == "FAIL"
    row_level = read_parquet(result.row_level_path)
    assert (row_level["check_name"] == "adjusted_split_only_consistency").any()


def test_qc_prices_v2_detects_off_calendar_dates(
    tmp_workspace: dict[str, Path],
) -> None:
    paths = _build_healthy_qc_inputs(tmp_workspace)
    raw = read_parquet(paths["raw"]).copy()
    adjusted = read_parquet(paths["adjusted"]).copy()

    raw.loc[0, "date"] = pd.Timestamp("2026-01-04")  # Sunday
    adjusted.loc[0, "date"] = pd.Timestamp("2026-01-04")
    raw_path = tmp_workspace["data"] / "price" / "raw_off_calendar.parquet"
    adjusted_path = tmp_workspace["data"] / "price" / "adjusted_off_calendar.parquet"
    raw.to_parquet(raw_path, index=False)
    adjusted.to_parquet(adjusted_path, index=False)

    result = _run_qc(
        paths,
        tmp_workspace,
        raw_override=raw_path,
        adjusted_override=adjusted_path,
        run_id="test_qc_v2_off_calendar",
    )
    assert result.gate_status == "FAIL"
    row_level = read_parquet(result.row_level_path)
    assert (row_level["check_name"] == "calendar_membership_raw").any()


def test_qc_prices_v2_detects_duplicate_primary_key(
    tmp_workspace: dict[str, Path],
) -> None:
    paths = _build_healthy_qc_inputs(tmp_workspace)
    raw = read_parquet(paths["raw"]).copy()
    adjusted = read_parquet(paths["adjusted"]).copy()
    raw_dup = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    adjusted_dup = pd.concat([adjusted, adjusted.iloc[[0]]], ignore_index=True)

    raw_path = tmp_workspace["data"] / "price" / "raw_dup.parquet"
    adjusted_path = tmp_workspace["data"] / "price" / "adjusted_dup.parquet"
    raw_dup.to_parquet(raw_path, index=False)
    adjusted_dup.to_parquet(adjusted_path, index=False)

    result = _run_qc(
        paths,
        tmp_workspace,
        raw_override=raw_path,
        adjusted_override=adjusted_path,
        run_id="test_qc_v2_dup_pk",
    )
    assert result.gate_status == "FAIL"
    row_level = read_parquet(result.row_level_path)
    assert (row_level["check_name"] == "primary_key_raw").any()


def test_qc_prices_v2_flags_low_symbol_coverage(
    tmp_workspace: dict[str, Path],
) -> None:
    paths = _build_healthy_qc_inputs(tmp_workspace)
    raw = read_parquet(paths["raw"]).copy()
    adjusted = read_parquet(paths["adjusted"]).copy()

    sim1 = raw["instrument_id"] == "SIM0001"
    keep_sim1 = raw["date"].dt.day % 7 == 0
    keep_mask = (~sim1) | keep_sim1
    raw_sparse = raw[keep_mask].copy()
    adjusted_sparse = adjusted.merge(
        raw_sparse[["date", "instrument_id", "ticker"]],
        on=["date", "instrument_id", "ticker"],
        how="inner",
    )

    raw_path = tmp_workspace["data"] / "price" / "raw_sparse.parquet"
    adjusted_path = tmp_workspace["data"] / "price" / "adjusted_sparse.parquet"
    raw_sparse.to_parquet(raw_path, index=False)
    adjusted_sparse.to_parquet(adjusted_path, index=False)

    result = _run_qc(
        paths,
        tmp_workspace,
        raw_override=raw_path,
        adjusted_override=adjusted_path,
        run_id="test_qc_v2_low_coverage",
    )
    row_level = read_parquet(result.row_level_path)
    assert (row_level["check_name"] == "symbol_coverage_low").any()
    assert result.gate_status in {"WARN", "FAIL"}


def test_qc_prices_v2_detects_corporate_actions_adjusted_mismatch(
    tmp_workspace: dict[str, Path],
) -> None:
    paths = _build_healthy_qc_inputs(tmp_workspace)
    corporate = read_parquet(paths["corporate_actions"]).copy()
    corporate_extra = pd.DataFrame(
        [
            {
                "event_id": "ca_test_extra_split_sim0003",
                "instrument_id": "SIM0003",
                "event_type": "split",
                "effective_date": pd.Timestamp("2026-03-01"),
                "announced_date": pd.Timestamp("2026-03-01"),
                "source_start_date": pd.Timestamp("2026-03-01"),
                "source_end_date": pd.Timestamp("2026-03-01"),
                "old_ticker": None,
                "new_ticker": None,
                "split_factor": 0.5,
                "event_value": 0.5,
                "event_unit": "ratio",
                "source_mode": "test_override",
                "source_ref": "test_qc_prices_v2",
                "is_terminal": False,
                "run_id": "test_qc_v2_override",
                "config_hash": "test_qc_v2_override",
                "built_ts_utc": pd.Timestamp("2026-03-14T00:00:00Z").isoformat(),
            }
        ]
    )
    corporate_bad = pd.concat([corporate, corporate_extra], ignore_index=True)
    corporate_bad_path = tmp_workspace["data"] / "universe" / "corporate_actions_bad.parquet"
    corporate_bad.to_parquet(corporate_bad_path, index=False)

    result = _run_qc(
        paths,
        tmp_workspace,
        corporate_override=corporate_bad_path,
        run_id="test_qc_v2_corporate_mismatch",
    )
    assert result.gate_status == "FAIL"
    row_level = read_parquet(result.row_level_path)
    assert (row_level["check_name"] == "corporate_actions_split_factor_consistency").any()
