from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time
from typing import Callable

# Allow direct script execution: `python simons_smallcap_swing/run_week2_data_upgrade.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.edgar.edgar_qc import run_edgar_qc
from data.edgar.fetch_companyfacts import fetch_companyfacts
from data.edgar.fetch_submissions import fetch_submissions
from data.edgar.point_in_time import build_fundamentals_pit
from data.price.adjust_prices import adjust_prices
from data.price.fetch_prices import fetch_prices
from data.price.market_proxies import build_market_proxies
from data.price.qc_prices import run_price_qc
from data.universe.corporate_actions import build_corporate_actions
from data.universe.survivorship import run_survivorship_analysis
from simons_core.io.paths import data_dir


@dataclass(frozen=True)
class Week2RunResult:
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


def _resolve_local_source(
    *,
    explicit_path: str | Path | None,
    preferred_path: Path,
    fallback_path: Path | None,
) -> Path:
    if explicit_path is not None:
        return Path(explicit_path).expanduser().resolve()
    if preferred_path.exists():
        return preferred_path.resolve()
    if fallback_path is not None and fallback_path.exists():
        return fallback_path.resolve()
    return preferred_path.resolve()


def _ensure_week1_prerequisites(base_data: Path) -> dict[str, Path]:
    expected = {
        "trading_calendar": base_data / "reference" / "trading_calendar.parquet",
        "ticker_history_map": base_data / "reference" / "ticker_history_map.parquet",
        "universe_history": base_data / "universe" / "universe_history.parquet",
        "universe_current": base_data / "universe" / "universe_current.parquet",
        "ticker_cik_map": base_data / "edgar" / "ticker_cik_map.parquet",
    }
    missing = [name for name, path in expected.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Week 2 runner requires Week 1 baseline artifacts. "
            f"Missing: {missing}. Run run_week1_mvp.py first for this data_root: {base_data}"
        )
    return expected


def run_week2_data_upgrade(
    *,
    run_prefix: str = "week2_data_upgrade",
    data_root: str | Path | None = None,
    price_ingestion_mode: str = "auto",
    price_local_source_path: str | Path | None = None,
    price_allow_synthetic_fallback: bool = True,
    submissions_ingestion_mode: str = "auto",
    submissions_local_source: str | Path | None = None,
    companyfacts_ingestion_mode: str = "auto",
    companyfacts_local_source: str | Path | None = None,
    allow_edgar_remote: bool = False,
    force_rebuild_edgar: bool = False,
    allow_fail_gates: bool = False,
    enforce_survivorship_gate: bool = False,
) -> Week2RunResult:
    base_data = Path(data_root).expanduser().resolve() if data_root else data_dir()
    reference_root = base_data / "reference"
    universe_root = base_data / "universe"
    price_root = base_data / "price"
    edgar_root = base_data / "edgar"
    for directory in (reference_root, universe_root, price_root, edgar_root):
        directory.mkdir(parents=True, exist_ok=True)

    prereq = _ensure_week1_prerequisites(base_data)
    universe_history_path = prereq["universe_history"]
    universe_current_path = prereq["universe_current"]
    trading_calendar_path = prereq["trading_calendar"]
    ticker_history_map_path = prereq["ticker_history_map"]
    ticker_cik_map_path = prereq["ticker_cik_map"]

    repo_data = data_dir()
    submissions_local = _resolve_local_source(
        explicit_path=submissions_local_source,
        preferred_path=base_data / "edgar" / "source" / "submissions",
        fallback_path=(repo_data / "edgar" / "source" / "submissions"),
    )
    companyfacts_local = _resolve_local_source(
        explicit_path=companyfacts_local_source,
        preferred_path=base_data / "edgar" / "source" / "companyfacts",
        fallback_path=(repo_data / "edgar" / "source" / "companyfacts"),
    )
    price_local = _resolve_local_source(
        explicit_path=price_local_source_path,
        preferred_path=base_data / "price" / "source",
        fallback_path=(repo_data / "price" / "source"),
    )

    total_steps = 10
    step = 1

    raw_prices = _run_step(
        step,
        total_steps,
        "fetch prices v2",
        fetch_prices,
        reference_root=reference_root,
        universe_history_path=universe_history_path,
        output_dir=price_root,
        run_id=_run_id(run_prefix, "price_fetch_v2"),
        ingestion_mode=price_ingestion_mode,
        local_source_path=price_local,
        allow_synthetic_fallback=price_allow_synthetic_fallback,
    )
    step += 1

    corporate_actions = _run_step(
        step,
        total_steps,
        "build corporate actions",
        build_corporate_actions,
        reference_root=reference_root,
        universe_history_path=universe_history_path,
        output_dir=universe_root,
        run_id=_run_id(run_prefix, "corporate_actions"),
    )
    step += 1

    adjusted_prices = _run_step(
        step,
        total_steps,
        "adjust prices v2 split_only",
        adjust_prices,
        raw_prices_path=raw_prices.raw_prices_path,
        corporate_actions_path=corporate_actions.corporate_actions_path,
        output_dir=price_root,
        run_id=_run_id(run_prefix, "price_adjust_v2"),
    )
    step += 1

    price_qc_run_id = _run_id(run_prefix, "price_qc_v2")
    price_qc = _run_step(
        step,
        total_steps,
        "run price QC v2",
        run_price_qc,
        raw_prices_path=raw_prices.raw_prices_path,
        adjusted_prices_path=adjusted_prices.adjusted_prices_path,
        trading_calendar_path=trading_calendar_path,
        ticker_history_map_path=ticker_history_map_path,
        corporate_actions_path=corporate_actions.corporate_actions_path,
        output_dir=price_root / "qc" / price_qc_run_id,
        run_id=price_qc_run_id,
    )
    step += 1

    survivorship_run_id = _run_id(run_prefix, "survivorship")
    survivorship = _run_step(
        step,
        total_steps,
        "run survivorship analysis",
        run_survivorship_analysis,
        universe_history_path=universe_history_path,
        universe_current_path=universe_current_path,
        ticker_history_map_path=ticker_history_map_path,
        trading_calendar_path=trading_calendar_path,
        adjusted_prices_path=adjusted_prices.adjusted_prices_path,
        output_dir=universe_root / "audit" / survivorship_run_id,
        run_id=survivorship_run_id,
    )
    step += 1

    market_proxies = _run_step(
        step,
        total_steps,
        "build market proxies",
        build_market_proxies,
        adjusted_prices_path=adjusted_prices.adjusted_prices_path,
        universe_history_path=universe_history_path,
        trading_calendar_path=trading_calendar_path,
        output_dir=price_root,
        run_id=_run_id(run_prefix, "market_proxies"),
    )
    step += 1

    submissions = _run_step(
        step,
        total_steps,
        "fetch submissions",
        fetch_submissions,
        ticker_cik_map_path=ticker_cik_map_path,
        universe_history_path=universe_history_path,
        local_source_path=submissions_local,
        output_dir=edgar_root,
        ingestion_mode=submissions_ingestion_mode,
        include_universe_filter=True,
        allow_remote=allow_edgar_remote,
        force_rebuild=force_rebuild_edgar,
        run_id=_run_id(run_prefix, "fetch_submissions"),
    )
    step += 1

    companyfacts = _run_step(
        step,
        total_steps,
        "fetch companyfacts",
        fetch_companyfacts,
        ticker_cik_map_path=ticker_cik_map_path,
        universe_history_path=universe_history_path,
        submissions_raw_path=submissions.submissions_raw_path,
        local_source_path=companyfacts_local,
        output_dir=edgar_root,
        ingestion_mode=companyfacts_ingestion_mode,
        include_universe_filter=True,
        allow_remote=allow_edgar_remote,
        force_rebuild=force_rebuild_edgar,
        run_id=_run_id(run_prefix, "fetch_companyfacts"),
    )
    step += 1

    fundamentals = _run_step(
        step,
        total_steps,
        "build fundamentals PIT v2",
        build_fundamentals_pit,
        ticker_cik_map_path=ticker_cik_map_path,
        universe_history_path=universe_history_path,
        submissions_raw_path=submissions.submissions_raw_path,
        companyfacts_raw_path=companyfacts.companyfacts_raw_path,
        output_dir=edgar_root,
        run_id=_run_id(run_prefix, "fundamentals_pit_v2"),
    )
    step += 1

    edgar_qc_run_id = _run_id(run_prefix, "edgar_qc_v2")
    edgar_qc = _run_step(
        step,
        total_steps,
        "run EDGAR QC v2",
        run_edgar_qc,
        fundamentals_pit_path=fundamentals.fundamentals_pit_path,
        ticker_cik_map_path=ticker_cik_map_path,
        submissions_raw_path=submissions.submissions_raw_path,
        output_dir=edgar_root / "qc" / edgar_qc_run_id,
        run_id=edgar_qc_run_id,
    )

    gates = {
        "price_qc": price_qc.gate_status,
        "survivorship": survivorship.gate_status,
        "edgar_qc": edgar_qc.gate_status,
    }
    if not allow_fail_gates:
        blocking = [gate for gate in ("price_qc", "edgar_qc") if gates[gate] == "FAIL"]
        if enforce_survivorship_gate and gates["survivorship"] == "FAIL":
            blocking.append("survivorship")
        if blocking:
            raise RuntimeError(f"One or more blocking gates failed: {blocking}. Full gates: {gates}")

    artifacts = {
        "week1_universe_history": universe_history_path,
        "week1_universe_current": universe_current_path,
        "week1_ticker_cik_map": ticker_cik_map_path,
        "raw_prices": raw_prices.raw_prices_path,
        "raw_prices_report": raw_prices.ingestion_report_path,
        "corporate_actions": corporate_actions.corporate_actions_path,
        "corporate_actions_summary": corporate_actions.summary_path,
        "adjusted_prices": adjusted_prices.adjusted_prices_path,
        "adjusted_prices_report": adjusted_prices.adjustment_report_path,
        "price_qc_summary": price_qc.summary_path,
        "price_qc_manifest": price_qc.manifest_path,
        "survivorship_summary": survivorship.summary_path,
        "survivorship_daily": survivorship.daily_path,
        "survivorship_membership_diff": survivorship.membership_diff_path,
        "survivorship_manifest": survivorship.manifest_path,
        "market_proxies": market_proxies.market_proxies_path,
        "market_proxies_summary": market_proxies.summary_path,
        "submissions_raw": submissions.submissions_raw_path,
        "submissions_report": submissions.ingestion_report_path,
        "companyfacts_raw": companyfacts.companyfacts_raw_path,
        "companyfacts_report": companyfacts.ingestion_report_path,
        "fundamentals_events": fundamentals.fundamentals_events_path,
        "fundamentals_pit": fundamentals.fundamentals_pit_path,
        "edgar_qc_summary": edgar_qc.summary_path,
        "edgar_qc_row_level": edgar_qc.row_level_path,
        "edgar_qc_failures": edgar_qc.failures_path,
        "edgar_qc_manifest": edgar_qc.manifest_path,
    }

    manifest_payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_prefix": run_prefix,
        "data_root": str(base_data),
        "steps_total": total_steps,
        "modes": {
            "price_ingestion_mode": price_ingestion_mode,
            "submissions_ingestion_mode": submissions_ingestion_mode,
            "companyfacts_ingestion_mode": companyfacts_ingestion_mode,
            "allow_edgar_remote": allow_edgar_remote,
            "force_rebuild_edgar": force_rebuild_edgar,
        },
        "resolved_sources": {
            "price_local_source": str(price_local),
            "submissions_local_source": str(submissions_local),
            "companyfacts_local_source": str(companyfacts_local),
        },
        "gates": gates,
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }
    manifest_path = base_data / f"week2_data_upgrade_manifest_{run_prefix}.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] Week 2 data upgrade pipeline completed. Manifest: {manifest_path}")

    return Week2RunResult(
        run_prefix=run_prefix,
        data_root=base_data,
        manifest_path=manifest_path,
        artifacts=artifacts,
        gates=gates,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Week 2 data-upgrade pipeline end-to-end.")
    parser.add_argument("--run-prefix", type=str, default="week2_data_upgrade")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument(
        "--price-ingestion-mode",
        type=str,
        default="auto",
        choices=("auto", "local_file", "synthetic_fallback", "provider_stub"),
    )
    parser.add_argument("--price-local-source-path", type=str, default=None)
    parser.add_argument(
        "--disable-price-synthetic-fallback",
        action="store_true",
        help="Disable synthetic fallback when fetch_prices cannot use local data.",
    )
    parser.add_argument(
        "--submissions-ingestion-mode",
        type=str,
        default="auto",
        choices=("auto", "local_file", "remote_optional"),
    )
    parser.add_argument("--submissions-local-source", type=str, default=None)
    parser.add_argument(
        "--companyfacts-ingestion-mode",
        type=str,
        default="auto",
        choices=("auto", "local_file", "remote_optional"),
    )
    parser.add_argument("--companyfacts-local-source", type=str, default=None)
    parser.add_argument("--allow-edgar-remote", action="store_true")
    parser.add_argument("--force-rebuild-edgar", action="store_true")
    parser.add_argument("--allow-fail-gates", action="store_true")
    parser.add_argument(
        "--enforce-survivorship-gate",
        action="store_true",
        help="Treat survivorship FAIL as a blocking gate.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_week2_data_upgrade(
        run_prefix=args.run_prefix,
        data_root=args.data_root,
        price_ingestion_mode=args.price_ingestion_mode,
        price_local_source_path=args.price_local_source_path,
        price_allow_synthetic_fallback=not args.disable_price_synthetic_fallback,
        submissions_ingestion_mode=args.submissions_ingestion_mode,
        submissions_local_source=args.submissions_local_source,
        companyfacts_ingestion_mode=args.companyfacts_ingestion_mode,
        companyfacts_local_source=args.companyfacts_local_source,
        allow_edgar_remote=args.allow_edgar_remote,
        force_rebuild_edgar=args.force_rebuild_edgar,
        allow_fail_gates=args.allow_fail_gates,
        enforce_survivorship_gate=args.enforce_survivorship_gate,
    )
    print("QC gates:")
    for gate_name, gate_status in result.gates.items():
        print(f"- {gate_name}: {gate_status}")


if __name__ == "__main__":
    main()
