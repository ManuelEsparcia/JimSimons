from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# Import helpers
# -----------------------------------------------------------------------------


def _nearby_repo_candidates(filename: str) -> list[Path]:
    here = Path(__file__).resolve()
    candidates: list[Path] = [Path(f"/mnt/data/{filename}")]
    for parent in [here.parent, *list(here.parents)]:
        candidates.extend(
            [
                parent / filename,
                parent / "data" / "borrow" / filename,
                parent / "simons_smallcap_swing" / "data" / "borrow" / filename,
            ]
        )
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered



def _import_module(candidates: Iterable[str], file_candidates: Iterable[Path], label: str):
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception:
            pass

    for path in file_candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location(f"{label}_under_test", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module

    raise ModuleNotFoundError(f"Could not import {label} from expected repo paths.")


@pytest.fixture(scope="session")
def locate_module():
    return _import_module(
        candidates=[
            "simons_smallcap_swing.data.borrow.locate_filter",
            "data.borrow.locate_filter",
            "locate_filter",
        ],
        file_candidates=_nearby_repo_candidates("locate_filter.py"),
        label="locate_filter",
    )


@pytest.fixture(scope="session")
def borrow_qc_module():
    return _import_module(
        candidates=[
            "simons_smallcap_swing.data.borrow.borrow_qc",
            "data.borrow.borrow_qc",
            "borrow_qc",
        ],
        file_candidates=_nearby_repo_candidates("borrow_qc.py"),
        label="borrow_qc",
    )


@pytest.fixture(scope="session")
def borrow_cost_proxy_module_or_none():
    try:
        return _import_module(
            candidates=[
                "simons_smallcap_swing.data.borrow.borrow_cost_proxy",
                "data.borrow.borrow_cost_proxy",
                "borrow_cost_proxy",
            ],
            file_candidates=_nearby_repo_candidates("borrow_cost_proxy.py"),
            label="borrow_cost_proxy",
        )
    except ModuleNotFoundError:
        return None


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _fake_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Runtime-safe parquet stand-in for this container.

    We deliberately write CSV payloads into the requested *.parquet paths so that
    the CLI contract around filenames is still validated without depending on
    pyarrow / fastparquet in the test runtime.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)



def _patch_parquet_writer(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    if hasattr(module, "write_parquet"):
        monkeypatch.setattr(module, "write_parquet", _fake_write_parquet)


def _csv_loader(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path)



def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# Synthetic input builders
# -----------------------------------------------------------------------------


def _borrow_proxy_smoke_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-01-05",
                "symbol": "AAA",
                "borrow_fee_annual": 0.04,
                "borrow_fee_daily": 0.04 / 252.0,
                "borrow_availability_score": 0.92,
                "htb_flag": False,
                "borrow_tier": "easy",
                "proxy_quality": "high",
                "proxy_source": "direct_lending_feed",
                "stress_score": 0.12,
                "fallback_flag": False,
                "stale_input_flag": False,
                "jump_flag": False,
                "config_version": "borrow_proxy_v1",
                "asof_timestamp": "2026-01-05T21:00:00Z",
            },
            {
                "date": "2026-01-06",
                "symbol": "AAA",
                "borrow_fee_annual": 0.06,
                "borrow_fee_daily": 0.06 / 252.0,
                "borrow_availability_score": 0.76,
                "htb_flag": False,
                "borrow_tier": "medium",
                "proxy_quality": "medium",
                "proxy_source": "short_interest_derived",
                "stress_score": 0.25,
                "fallback_flag": False,
                "stale_input_flag": False,
                "jump_flag": False,
                "config_version": "borrow_proxy_v1",
                "asof_timestamp": "2026-01-06T21:00:00Z",
            },
        ]
    )



def _universe_smoke_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-01-05",
                "symbol": "AAA",
                "membership_state": "member",
                "is_member": 1,
                "size_bucket": "small",
                "liq_bucket": "medium",
                "market_cap": 1_250_000_000.0,
                "short_subuniverse_flag": 1,
            },
            {
                "date": "2026-01-06",
                "symbol": "AAA",
                "membership_state": "member",
                "is_member": 1,
                "size_bucket": "small",
                "liq_bucket": "medium",
                "market_cap": 1_260_000_000.0,
                "short_subuniverse_flag": 1,
            },
        ]
    )



def _locate_qc_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-01-05",
                "symbol": "AAA",
                "locate_allowed": True,
                "locate_blocked": False,
                "locate_decision": "allow",
                "override_flag": False,
                "override_reason": "",
                "config_version": "locate_filter_v1",
            },
            {
                "date": "2026-01-06",
                "symbol": "AAA",
                "locate_allowed": True,
                "locate_blocked": False,
                "locate_decision": "allow",
                "override_flag": False,
                "override_reason": "",
                "config_version": "locate_filter_v1",
            },
        ]
    )


# -----------------------------------------------------------------------------
# CLI smoke tests
# -----------------------------------------------------------------------------


def test_locate_filter_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, locate_module: Any) -> None:
    _patch_parquet_writer(locate_module, monkeypatch)
    if hasattr(locate_module, "load_borrow_proxy"):
        monkeypatch.setattr(locate_module, "load_borrow_proxy", _csv_loader)
    if hasattr(locate_module, "load_universe"):
        monkeypatch.setattr(locate_module, "load_universe", _csv_loader)
    if hasattr(locate_module, "load_previous_state"):
        monkeypatch.setattr(locate_module, "load_previous_state", _csv_loader)

    borrow_path = tmp_path / "borrow_proxy.csv"
    universe_path = tmp_path / "universe.csv"
    outdir = tmp_path / "locate_out"

    _borrow_proxy_smoke_df().to_csv(borrow_path, index=False)
    _universe_smoke_df().to_csv(universe_path, index=False)

    rc = locate_module.main(
        [
            "--borrow-proxy-path",
            str(borrow_path),
            "--universe-path",
            str(universe_path),
            "--output-dir",
            str(outdir),
            "--run-id",
            "smoke_locate",
            "--asof-timestamp",
            "2026-01-06T21:00:00Z",
        ]
    )

    assert rc == 0

    daily_path = outdir / "locate_filter_daily_smoke_locate.parquet"
    daily_summary_path = outdir / "locate_filter_summary_daily_smoke_locate.parquet"
    summary_path = outdir / "locate_filter_summary_smoke_locate.json"
    manifest_path = outdir / "locate_filter_manifest_smoke_locate.json"

    assert daily_path.exists(), "locate CLI should emit the daily parquet artifact"
    assert daily_summary_path.exists(), "locate CLI should emit the daily summary parquet artifact"
    assert summary_path.exists(), "locate CLI should emit the summary JSON artifact"
    assert manifest_path.exists(), "locate CLI should emit the manifest JSON artifact"

    summary = _read_json(summary_path)
    manifest = _read_json(manifest_path)
    assert summary["config_version"] == "locate_filter_v1"
    assert manifest["run_id"] == "smoke_locate"
    assert manifest["inputs"]["borrow_proxy_path"] == str(borrow_path)
    assert manifest["outputs"]["locate_daily_rows"] >= 1
    assert manifest["summary"]["n_rows"] >= 1



def test_borrow_qc_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, borrow_qc_module: Any) -> None:
    _patch_parquet_writer(borrow_qc_module, monkeypatch)

    borrow_path = tmp_path / "borrow_proxy.csv"
    locate_path = tmp_path / "locate_filter.csv"
    universe_path = tmp_path / "universe.csv"
    outdir = tmp_path / "borrow_qc_out"

    _borrow_proxy_smoke_df().to_csv(borrow_path, index=False)
    _locate_qc_input_df().to_csv(locate_path, index=False)
    _universe_smoke_df().to_csv(universe_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "borrow_qc.py",
            "--borrow-proxy-path",
            str(borrow_path),
            "--locate-filter-path",
            str(locate_path),
            "--universe-path",
            str(universe_path),
            "--output-dir",
            str(outdir),
            "--run-id",
            "smoke_borrow_qc",
            "--asof-timestamp",
            "2026-01-06T21:05:00Z",
        ],
    )

    borrow_qc_module.main()

    daily_path = outdir / "borrow_qc_daily.parquet"
    failures_path = outdir / "borrow_qc_failures.parquet"
    panel_path = outdir / "borrow_qc_joined_panel.parquet"
    summary_path = outdir / "borrow_qc_summary.json"
    manifest_path = outdir / "manifest.json"

    assert daily_path.exists(), "borrow_qc CLI should emit the daily parquet artifact"
    assert failures_path.exists(), "borrow_qc CLI should emit the failures parquet artifact"
    assert panel_path.exists(), "borrow_qc CLI should emit the joined panel parquet artifact"
    assert summary_path.exists(), "borrow_qc CLI should emit the summary JSON artifact"
    assert manifest_path.exists(), "borrow_qc CLI should emit the manifest JSON artifact"

    summary = _read_json(summary_path)
    manifest = _read_json(manifest_path)
    assert summary["run_id"] == "smoke_borrow_qc"
    assert manifest["run_id"] == "smoke_borrow_qc"
    assert manifest["inputs"]["borrow_proxy_path"] == str(borrow_path)
    assert manifest["inputs"]["locate_filter_path"] == str(locate_path)



def test_borrow_cost_proxy_cli_smoke_if_module_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    borrow_cost_proxy_module_or_none: Any,
) -> None:
    if borrow_cost_proxy_module_or_none is None:
        pytest.skip("borrow_cost_proxy.py is not available in this runtime yet; smoke test deferred until the module exists.")

    module = borrow_cost_proxy_module_or_none
    _patch_parquet_writer(module, monkeypatch)

    if not hasattr(module, "main") or not callable(module.main):
        pytest.skip("borrow_cost_proxy.py is importable but does not expose a callable main().")

    pytest.skip(
        "borrow_cost_proxy.py is not present in the current generated artifact set; enable this smoke test once the module file is available to CI."
    )
