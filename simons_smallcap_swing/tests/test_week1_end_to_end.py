from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_week1_mvp import run_week1_mvp
from simons_core.io.parquet_store import read_parquet


def test_week1_runner_end_to_end_smoke(tmp_workspace: dict[str, Path]) -> None:
    result = run_week1_mvp(
        run_prefix="test_week1_e2e",
        data_root=tmp_workspace["data"],
    )

    expected_paths = {
        "universe_history": result.artifacts["universe_history"],
        "raw_prices": result.artifacts["raw_prices"],
        "fundamentals_pit": result.artifacts["fundamentals_pit"],
        "universe_qc_summary": result.artifacts["universe_qc_summary"],
        "price_qc_summary": result.artifacts["price_qc_summary"],
        "edgar_qc_summary": result.artifacts["edgar_qc_summary"],
    }

    for _, path in expected_paths.items():
        assert path.exists()
        assert path.stat().st_size > 0

    universe_history = read_parquet(expected_paths["universe_history"])
    raw_prices = read_parquet(expected_paths["raw_prices"])
    fundamentals_pit = read_parquet(expected_paths["fundamentals_pit"])

    assert len(universe_history) > 0
    assert len(raw_prices) > 0
    assert len(fundamentals_pit) > 0

    # PIT semantics check.
    assert (
        pd.to_datetime(fundamentals_pit["acceptance_ts"], utc=True)
        <= pd.to_datetime(fundamentals_pit["asof_date"], utc=True)
    ).all()

    # QC gates exist and are consumable.
    for gate_name in ("universe_qc", "price_qc", "edgar_qc"):
        assert gate_name in result.gates
        assert result.gates[gate_name] in {"PASS", "WARN"}

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_prefix"] == "test_week1_e2e"
    assert set(manifest["gates"]) == {"universe_qc", "price_qc", "edgar_qc"}
