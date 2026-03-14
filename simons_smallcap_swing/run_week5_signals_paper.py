from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time
from typing import Callable, Iterable

# Allow direct script execution: `python simons_smallcap_swing/run_week5_signals_paper.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from signals.build_signals import build_signals
from signals.decile_analysis import run_decile_analysis
from signals.paper_portfolio import run_paper_portfolio
from simons_core.io.paths import data_dir


DEFAULT_SPLIT_ROLES: tuple[str, ...] = ("valid", "test")
DEFAULT_PORTFOLIO_MODES: tuple[str, ...] = ("long_only_top", "long_short_top_bottom")


@dataclass(frozen=True)
class Week5RunResult:
    run_prefix: str
    data_root: Path
    manifest_path: Path
    artifacts: dict[str, Path]
    statuses: dict[str, str]


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


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    items = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not items:
        raise ValueError("Expected at least one comma-separated value.")
    return items


def _normalize_list(values: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(sorted({str(item).strip() for item in values if str(item).strip()}))
    if not normalized:
        raise ValueError("Expected at least one value.")
    return normalized


def _resolve_predictions_path(
    *,
    data_root: Path,
    model_name: str,
    explicit_predictions_path: str | Path | None,
) -> Path:
    if explicit_predictions_path is not None:
        return Path(explicit_predictions_path).expanduser().resolve()

    model_alias = str(model_name).strip().lower()
    if model_alias in {"ridge", "ridge_baseline"}:
        candidates = ("ridge_baseline_predictions.parquet",)
    elif model_alias in {"logistic", "logistic_baseline"}:
        candidates = ("logistic_baseline_predictions.parquet",)
    else:
        candidates = (
            f"{model_alias}_predictions.parquet",
            "ridge_baseline_predictions.parquet",
            "logistic_baseline_predictions.parquet",
        )

    artifacts_root = data_root / "models" / "artifacts"
    for name in candidates:
        candidate = artifacts_root / name
        if candidate.exists():
            return candidate.resolve()
    return (artifacts_root / candidates[0]).resolve()


def _ensure_week5_prerequisites(expected: dict[str, Path]) -> dict[str, Path]:
    missing = [name for name, path in expected.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Week 5 runner requires existing Week 1-4 artifacts. "
            f"Missing: {missing}. Paths checked: "
            + ", ".join(f"{name}={path}" for name, path in expected.items())
        )
    return expected


def run_week5_signals_paper(
    *,
    run_prefix: str = "week5_signals_paper",
    data_root: str | Path | None = None,
    predictions_path: str | Path | None = None,
    universe_history_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_name: str = "ridge_baseline",
    label_name: str = "fwd_ret_5d",
    split_name: str | None = None,
    horizon_days: int | None = 5,
    split_roles: Iterable[str] = DEFAULT_SPLIT_ROLES,
    n_buckets: int = 10,
    top_buckets: int = 1,
    bottom_buckets: int = 1,
    center_classification_proba: bool = False,
    use_universe_filter: bool = True,
    portfolio_modes: Iterable[str] = DEFAULT_PORTFOLIO_MODES,
) -> Week5RunResult:
    base_data = Path(data_root).expanduser().resolve() if data_root else data_dir()
    signals_root = Path(output_dir).expanduser().resolve() if output_dir else (base_data / "signals")
    signals_root.mkdir(parents=True, exist_ok=True)

    predictions_source = _resolve_predictions_path(
        data_root=base_data,
        model_name=model_name,
        explicit_predictions_path=predictions_path,
    )
    universe_source = (
        Path(universe_history_path).expanduser().resolve()
        if universe_history_path
        else (base_data / "universe" / "universe_history.parquet").resolve()
    )
    labels_source = (
        Path(labels_path).expanduser().resolve()
        if labels_path
        else (base_data / "labels" / "labels_forward.parquet").resolve()
    )

    prereq = _ensure_week5_prerequisites(
        {
            "predictions": predictions_source,
            "universe_history": universe_source,
            "labels_forward": labels_source,
        }
    )

    selected_split_roles = _normalize_list(split_roles)
    selected_portfolio_modes = _normalize_list(portfolio_modes)

    total_steps = 3
    step = 1
    statuses: dict[str, str] = {}

    signals = _run_step(
        step,
        total_steps,
        "build signals",
        build_signals,
        predictions_path=prereq["predictions"],
        output_dir=signals_root,
        model_name=model_name,
        label_name=label_name,
        split_name=split_name,
        horizon_days=horizon_days,
        split_roles=selected_split_roles,
        n_buckets=int(n_buckets),
        top_buckets=int(top_buckets),
        bottom_buckets=int(bottom_buckets),
        center_classification_proba=bool(center_classification_proba),
        universe_history_path=prereq["universe_history"],
        use_universe_filter=bool(use_universe_filter),
        run_id=_run_id(run_prefix, "build_signals"),
    )
    statuses["build_signals"] = "DONE"
    step += 1

    deciles = _run_step(
        step,
        total_steps,
        "run decile analysis",
        run_decile_analysis,
        signals_path=signals.signals_path,
        labels_path=prereq["labels_forward"],
        output_dir=signals_root,
        model_name=model_name,
        label_name=label_name,
        horizon_days=horizon_days,
        split_roles=selected_split_roles,
        expected_n_buckets=int(n_buckets),
        run_id=_run_id(run_prefix, "decile_analysis"),
    )
    statuses["decile_analysis"] = "DONE"
    step += 1

    paper = _run_step(
        step,
        total_steps,
        "build paper portfolio",
        run_paper_portfolio,
        signals_path=signals.signals_path,
        labels_path=prereq["labels_forward"],
        output_dir=signals_root,
        model_name=model_name,
        label_name=label_name,
        horizon_days=horizon_days,
        split_roles=selected_split_roles,
        portfolio_modes=selected_portfolio_modes,
        run_id=_run_id(run_prefix, "paper_portfolio"),
    )
    statuses["paper_portfolio"] = "DONE"

    artifacts: dict[str, Path] = {
        "signals_daily": signals.signals_path,
        "signals_summary": signals.summary_path,
        "decile_daily": deciles.decile_daily_path,
        "decile_summary": deciles.decile_summary_path,
        "decile_analysis_summary": deciles.summary_json_path,
        "paper_portfolio_daily": paper.daily_path,
        "paper_portfolio_positions": paper.positions_path,
        "paper_portfolio_summary": paper.summary_path,
    }

    manifest_payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_prefix": run_prefix,
        "data_root": str(base_data),
        "steps_total": total_steps,
        "flags": {
            "model_name": model_name,
            "label_name": label_name,
            "split_name": split_name,
            "horizon_days": horizon_days,
            "split_roles": list(selected_split_roles),
            "n_buckets": int(n_buckets),
            "top_buckets": int(top_buckets),
            "bottom_buckets": int(bottom_buckets),
            "center_classification_proba": bool(center_classification_proba),
            "use_universe_filter": bool(use_universe_filter),
            "portfolio_modes": list(selected_portfolio_modes),
        },
        "statuses": statuses,
        "prerequisites": {key: str(path) for key, path in prereq.items()},
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }
    manifest_path = base_data / f"week5_signals_paper_manifest_{run_prefix}.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] Week 5 signals/paper pipeline completed. Manifest: {manifest_path}")

    return Week5RunResult(
        run_prefix=run_prefix,
        data_root=base_data,
        manifest_path=manifest_path,
        artifacts=artifacts,
        statuses=statuses,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Week 5 signals + paper-portfolio pipeline.")
    parser.add_argument("--run-prefix", type=str, default="week5_signals_paper")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--predictions-path", type=str, default=None)
    parser.add_argument("--universe-history-path", type=str, default=None)
    parser.add_argument("--labels-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="ridge_baseline")
    parser.add_argument("--label-name", type=str, default="fwd_ret_5d")
    parser.add_argument("--split-name", type=str, default=None)
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--split-roles", type=_parse_csv_strings, default=DEFAULT_SPLIT_ROLES)
    parser.add_argument("--n-buckets", type=int, default=10)
    parser.add_argument("--top-buckets", type=int, default=1)
    parser.add_argument("--bottom-buckets", type=int, default=1)
    parser.add_argument("--center-classification-proba", action="store_true")
    parser.add_argument("--disable-universe-filter", action="store_true")
    parser.add_argument("--portfolio-modes", type=_parse_csv_strings, default=DEFAULT_PORTFOLIO_MODES)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_week5_signals_paper(
        run_prefix=args.run_prefix,
        data_root=args.data_root,
        predictions_path=args.predictions_path,
        universe_history_path=args.universe_history_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        label_name=args.label_name,
        split_name=args.split_name,
        horizon_days=args.horizon_days,
        split_roles=args.split_roles,
        n_buckets=args.n_buckets,
        top_buckets=args.top_buckets,
        bottom_buckets=args.bottom_buckets,
        center_classification_proba=bool(args.center_classification_proba),
        use_universe_filter=not bool(args.disable_universe_filter),
        portfolio_modes=args.portfolio_modes,
    )
    print("Week 5 statuses:")
    for key, value in result.statuses.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
