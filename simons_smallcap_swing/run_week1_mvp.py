from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time
from typing import Callable

# Allow direct script execution: `python simons_smallcap_swing/run_week1_mvp.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.edgar.edgar_qc import run_edgar_qc
from data.edgar.point_in_time import build_fundamentals_pit
from data.edgar.ticker_cik import build_ticker_cik_map
from data.price.adjust_prices import adjust_prices
from data.price.fetch_prices import fetch_prices
from data.price.qc_prices import run_price_qc
from data.reference.build_reference import build_reference_data
from data.universe.build_universe import build_universe
from data.universe.universe_qc import run_universe_qc
from simons_core.io.paths import data_dir


@dataclass(frozen=True)
class Week1RunResult:
    run_prefix: str
    data_root: Path
    manifest_path: Path
    artifacts: dict[str, Path]
    gates: dict[str, str]


def _run_id(prefix: str, step: str) -> str:
    return f"{prefix}_{step}"


def _run_step(
    idx: int,
    total: int,
    label: str,
    func: Callable[..., object],
    **kwargs: object,
) -> object:
    t0 = time.perf_counter()
    print(f"[{idx}/{total}] {label} ...")
    try:
        out = func(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"Step failed [{idx}/{total}] {label}: {exc}") from exc
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    print(f"[{idx}/{total}] {label} done ({elapsed_ms} ms)")
    return out


def run_week1_mvp(
    *,
    run_prefix: str = "week1_mvp",
    data_root: str | Path | None = None,
    allow_fail_gates: bool = False,
) -> Week1RunResult:
    base_data = Path(data_root).expanduser().resolve() if data_root else data_dir()
    reference_root = base_data / "reference"
    universe_root = base_data / "universe"
    price_root = base_data / "price"
    edgar_root = base_data / "edgar"
    for directory in (reference_root, universe_root, price_root, edgar_root):
        directory.mkdir(parents=True, exist_ok=True)

    total_steps = 9
    step = 1

    reference = _run_step(
        step,
        total_steps,
        "build reference",
        build_reference_data,
        output_dir=reference_root,
        run_id=_run_id(run_prefix, "reference"),
    )
    step += 1

    universe = _run_step(
        step,
        total_steps,
        "build universe",
        build_universe,
        reference_root=reference_root,
        output_dir=universe_root,
        run_id=_run_id(run_prefix, "universe"),
    )
    step += 1

    universe_qc_run_id = _run_id(run_prefix, "universe_qc")
    universe_qc = _run_step(
        step,
        total_steps,
        "run universe QC",
        run_universe_qc,
        universe_history_path=universe.universe_history,
        universe_current_path=universe.universe_current,
        ticker_history_map_path=reference.ticker_history_map,
        trading_calendar_path=reference.trading_calendar,
        output_dir=universe_root / "qc" / universe_qc_run_id,
        run_id=universe_qc_run_id,
    )
    step += 1

    raw_prices = _run_step(
        step,
        total_steps,
        "build raw prices",
        fetch_prices,
        reference_root=reference_root,
        universe_history_path=universe.universe_history,
        output_dir=price_root,
        run_id=_run_id(run_prefix, "price_fetch"),
    )
    step += 1

    adjusted_prices = _run_step(
        step,
        total_steps,
        "build adjusted prices",
        adjust_prices,
        raw_prices_path=raw_prices.raw_prices_path,
        output_dir=price_root,
        run_id=_run_id(run_prefix, "price_adjust"),
    )
    step += 1

    price_qc_run_id = _run_id(run_prefix, "price_qc")
    price_qc = _run_step(
        step,
        total_steps,
        "run price QC",
        run_price_qc,
        raw_prices_path=raw_prices.raw_prices_path,
        adjusted_prices_path=adjusted_prices.adjusted_prices_path,
        trading_calendar_path=reference.trading_calendar,
        ticker_history_map_path=reference.ticker_history_map,
        output_dir=price_root / "qc" / price_qc_run_id,
        run_id=price_qc_run_id,
    )
    step += 1

    ticker_cik = _run_step(
        step,
        total_steps,
        "build ticker_cik",
        build_ticker_cik_map,
        reference_root=reference_root,
        universe_history_path=universe.universe_history,
        output_dir=edgar_root,
        run_id=_run_id(run_prefix, "ticker_cik"),
    )
    step += 1

    fundamentals_pit = _run_step(
        step,
        total_steps,
        "build fundamentals PIT",
        build_fundamentals_pit,
        ticker_cik_map_path=ticker_cik.ticker_cik_map_path,
        universe_history_path=universe.universe_history,
        output_dir=edgar_root,
        run_id=_run_id(run_prefix, "fundamentals_pit"),
    )
    step += 1

    edgar_qc_run_id = _run_id(run_prefix, "edgar_qc")
    edgar_qc = _run_step(
        step,
        total_steps,
        "run EDGAR QC",
        run_edgar_qc,
        fundamentals_pit_path=fundamentals_pit.fundamentals_pit_path,
        ticker_cik_map_path=ticker_cik.ticker_cik_map_path,
        output_dir=edgar_root / "qc" / edgar_qc_run_id,
        run_id=edgar_qc_run_id,
    )

    gates = {
        "universe_qc": universe_qc.gate_status,
        "price_qc": price_qc.gate_status,
        "edgar_qc": edgar_qc.gate_status,
    }
    if not allow_fail_gates:
        failed = [name for name, status in gates.items() if status == "FAIL"]
        if failed:
            raise RuntimeError(f"One or more QC gates failed: {failed}. Full gates: {gates}")

    artifacts = {
        "trading_calendar": reference.trading_calendar,
        "ticker_history_map": reference.ticker_history_map,
        "symbols_metadata": reference.symbols_metadata,
        "sector_industry_map": reference.sector_industry_map,
        "universe_history": universe.universe_history,
        "universe_current": universe.universe_current,
        "raw_prices": raw_prices.raw_prices_path,
        "adjusted_prices": adjusted_prices.adjusted_prices_path,
        "ticker_cik_map": ticker_cik.ticker_cik_map_path,
        "fundamentals_events": fundamentals_pit.fundamentals_events_path,
        "fundamentals_pit": fundamentals_pit.fundamentals_pit_path,
        "universe_qc_summary": universe_qc.summary_path,
        "price_qc_summary": price_qc.summary_path,
        "edgar_qc_summary": edgar_qc.summary_path,
    }

    manifest_payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_prefix": run_prefix,
        "data_root": str(base_data),
        "gates": gates,
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }
    manifest_path = base_data / f"week1_mvp_manifest_{run_prefix}.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] Week 1 MVP pipeline completed. Manifest: {manifest_path}")

    return Week1RunResult(
        run_prefix=run_prefix,
        data_root=base_data,
        manifest_path=manifest_path,
        artifacts=artifacts,
        gates=gates,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full Week 1 MVP pipeline end-to-end.")
    parser.add_argument("--run-prefix", type=str, default="week1_mvp")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument(
        "--allow-fail-gates",
        action="store_true",
        help="Do not raise error if any QC gate is FAIL.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_week1_mvp(
        run_prefix=args.run_prefix,
        data_root=args.data_root,
        allow_fail_gates=args.allow_fail_gates,
    )
    print("QC gates:")
    for gate_name, gate_status in result.gates.items():
        print(f"- {gate_name}: {gate_status}")


if __name__ == "__main__":
    main()
