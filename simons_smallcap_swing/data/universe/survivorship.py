from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

# Allow direct script execution: `python simons_smallcap_swing/data/universe/survivorship.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir, reference_dir
from simons_core.logging import get_logger

ALLOWED_BASELINE_MODES: tuple[str, ...] = (
    "current_survivors",
    "current_eligible",
    "current_tradable",
)


@dataclass(frozen=True)
class SurvivorshipResult:
    summary_path: Path
    daily_path: Path
    membership_diff_path: Path
    symbol_level_path: Path
    manifest_path: Path
    bias_level: str
    severity_score: float
    gate_status: str


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains invalid dates.")
    return parsed.dt.normalize()


def _ensure_non_empty_frame(df: pd.DataFrame, placeholder: dict[str, Any]) -> pd.DataFrame:
    if not df.empty:
        return df
    return pd.DataFrame([placeholder])


def _build_config_hash(
    *,
    baseline_mode: str,
    universe_history_path: Path,
    universe_current_path: Path,
    ticker_history_map_path: Path,
    trading_calendar_path: Path,
    adjusted_prices_path: Path | None,
) -> str:
    payload = {
        "version": "survivorship_mvp_v1",
        "baseline_mode": baseline_mode,
        "paths": {
            "universe_history": str(universe_history_path),
            "universe_current": str(universe_current_path),
            "ticker_history_map": str(ticker_history_map_path),
            "trading_calendar": str(trading_calendar_path),
            "adjusted_prices": str(adjusted_prices_path) if adjusted_prices_path else "",
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _resolve_naive_ids(
    *,
    baseline_mode: str,
    universe_current: pd.DataFrame,
    ticker_history_map: pd.DataFrame,
    adjusted_prices: pd.DataFrame | None,
    history_last_date: pd.Timestamp,
) -> tuple[set[str], dict[str, Any]]:
    current_eligible = universe_current[universe_current["is_eligible"].astype(bool)].copy()
    current_eligible_ids = set(current_eligible["instrument_id"].astype(str).tolist())
    meta: dict[str, Any] = {
        "mode": baseline_mode,
        "current_eligible_ids_count": int(len(current_eligible_ids)),
        "resolved_via": baseline_mode,
        "notes": [],
    }

    if baseline_mode == "current_eligible":
        return current_eligible_ids, meta

    if baseline_mode == "current_survivors":
        map_frame = ticker_history_map.copy()
        map_frame["instrument_id"] = map_frame["instrument_id"].astype(str)
        map_frame["is_active"] = map_frame["is_active"].astype(bool)
        map_frame["end_date"] = pd.to_datetime(map_frame["end_date"], errors="coerce").dt.normalize()
        survivor_ids = set(
            map_frame.loc[
                map_frame["is_active"] | map_frame["end_date"].isna(),
                "instrument_id",
            ]
            .astype(str)
            .tolist()
        )
        resolved = current_eligible_ids & survivor_ids
        meta["survivor_ids_from_ticker_map_count"] = int(len(survivor_ids))
        meta["resolved_ids_count"] = int(len(resolved))
        return resolved, meta

    # baseline_mode == "current_tradable"
    resolved_ids = set(current_eligible_ids)
    if adjusted_prices is None or adjusted_prices.empty:
        meta["notes"].append(
            "adjusted_prices_unavailable_or_empty_fallback_to_current_eligible"
        )
        meta["resolved_ids_count"] = int(len(resolved_ids))
        return resolved_ids, meta

    adjusted = adjusted_prices.copy()
    adjusted["date"] = _normalize_date(adjusted["date"], column="date")
    adjusted["instrument_id"] = adjusted["instrument_id"].astype(str)
    tradable_ids = set(
        adjusted.loc[adjusted["date"] == history_last_date, "instrument_id"]
        .astype(str)
        .tolist()
    )
    resolved_ids = resolved_ids & tradable_ids
    if not resolved_ids:
        resolved_ids = set(current_eligible_ids)
        meta["notes"].append(
            "current_tradable_empty_fallback_to_current_eligible"
        )
    meta["tradable_ids_last_date_count"] = int(len(tradable_ids))
    meta["resolved_ids_count"] = int(len(resolved_ids))
    return resolved_ids, meta


def _score_bias(
    *,
    mean_pct_pit_retained_by_naive: float,
    mean_pct_naive_ex_post_names: float,
    mean_jaccard_membership: float,
    terminal_only_ratio: float,
) -> tuple[float, str, str]:
    c_missing = max(0.0, min(1.0, 1.0 - mean_pct_pit_retained_by_naive))
    c_ex_post = max(0.0, min(1.0, mean_pct_naive_ex_post_names))
    c_jaccard = max(0.0, min(1.0, 1.0 - mean_jaccard_membership))
    c_terminal = max(0.0, min(1.0, terminal_only_ratio))

    severity_score = (
        0.45 * c_missing
        + 0.25 * c_ex_post
        + 0.20 * c_jaccard
        + 0.10 * c_terminal
    )
    severity_score = float(max(0.0, min(1.0, severity_score)))

    if severity_score >= 0.60:
        bias_level = "HIGH"
        gate_status = "FAIL"
        recommended_action = (
            "Do not use naive universe in research; enforce PIT universe as canonical."
        )
    elif severity_score >= 0.30:
        bias_level = "MEDIUM"
        gate_status = "WARN"
        recommended_action = (
            "Use PIT universe for research and monitor survivorship deltas each run."
        )
    else:
        bias_level = "LOW"
        gate_status = "PASS"
        recommended_action = (
            "Survivorship distortion is limited in this window; keep PIT as canonical source."
        )
    return severity_score, bias_level, recommended_action if gate_status != "FAIL" else recommended_action


def run_survivorship_analysis(
    *,
    universe_history_path: str | Path | None = None,
    universe_current_path: str | Path | None = None,
    ticker_history_map_path: str | Path | None = None,
    trading_calendar_path: str | Path | None = None,
    adjusted_prices_path: str | Path | None = None,
    baseline_mode: str = "current_eligible",
    output_dir: str | Path | None = None,
    run_id: str = "survivorship_mvp_v1",
) -> SurvivorshipResult:
    logger = get_logger("data.universe.survivorship")

    if baseline_mode not in ALLOWED_BASELINE_MODES:
        raise ValueError(
            f"Unsupported baseline_mode '{baseline_mode}'. "
            f"Allowed: {sorted(ALLOWED_BASELINE_MODES)}"
        )

    universe_base = data_dir() / "universe"
    ref_base = reference_dir()

    history_source = (
        Path(universe_history_path).expanduser().resolve()
        if universe_history_path
        else (universe_base / "universe_history.parquet")
    )
    current_source = (
        Path(universe_current_path).expanduser().resolve()
        if universe_current_path
        else (universe_base / "universe_current.parquet")
    )
    ticker_map_source = (
        Path(ticker_history_map_path).expanduser().resolve()
        if ticker_history_map_path
        else (ref_base / "ticker_history_map.parquet")
    )
    calendar_source = (
        Path(trading_calendar_path).expanduser().resolve()
        if trading_calendar_path
        else (ref_base / "trading_calendar.parquet")
    )
    adjusted_source = (
        Path(adjusted_prices_path).expanduser().resolve()
        if adjusted_prices_path
        else (data_dir() / "price" / "adjusted_prices.parquet")
    )

    history = read_parquet(history_source).copy()
    current = read_parquet(current_source).copy()
    ticker_map = read_parquet(ticker_map_source).copy()
    calendar = read_parquet(calendar_source).copy()

    adjusted: pd.DataFrame | None = None
    if adjusted_source.exists():
        adjusted = read_parquet(adjusted_source).copy()

    required_history = {"date", "instrument_id", "ticker", "is_eligible"}
    required_current = {"date", "instrument_id", "ticker", "is_eligible"}
    required_map = {"instrument_id", "is_active", "end_date"}
    required_calendar = {"date", "is_session"}
    missing_history = sorted(required_history - set(history.columns))
    missing_current = sorted(required_current - set(current.columns))
    missing_map = sorted(required_map - set(ticker_map.columns))
    missing_calendar = sorted(required_calendar - set(calendar.columns))
    if missing_history:
        raise ValueError(f"universe_history missing required columns: {missing_history}")
    if missing_current:
        raise ValueError(f"universe_current missing required columns: {missing_current}")
    if missing_map:
        raise ValueError(f"ticker_history_map missing required columns: {missing_map}")
    if missing_calendar:
        raise ValueError(f"trading_calendar missing required columns: {missing_calendar}")

    history["date"] = _normalize_date(history["date"], column="date")
    current["date"] = _normalize_date(current["date"], column="date")
    history["instrument_id"] = history["instrument_id"].astype(str)
    current["instrument_id"] = current["instrument_id"].astype(str)
    history["ticker"] = history["ticker"].astype(str).str.upper().str.strip()
    current["ticker"] = current["ticker"].astype(str).str.upper().str.strip()
    history["is_eligible"] = history["is_eligible"].astype(bool)
    current["is_eligible"] = current["is_eligible"].astype(bool)

    if history.empty:
        raise ValueError("universe_history is empty.")
    if current.empty:
        raise ValueError("universe_current is empty.")

    history_eligible = history[history["is_eligible"]].copy()
    if history_eligible.empty:
        raise ValueError("universe_history has no eligible rows (is_eligible=True).")

    min_date = history_eligible["date"].min()
    max_date = history_eligible["date"].max()
    sessions = _normalize_date(
        calendar.loc[calendar["is_session"].astype(bool), "date"],
        column="date",
    )
    sessions = pd.DatetimeIndex(
        sorted(sessions[(sessions >= min_date) & (sessions <= max_date)].unique())
    )
    if sessions.empty:
        raise ValueError("No trading sessions in requested universe history range.")

    valid_session_set = set(sessions.tolist())
    invalid_history_dates = history_eligible.loc[
        ~history_eligible["date"].isin(valid_session_set), "date"
    ].drop_duplicates()
    if not invalid_history_dates.empty:
        bad = [str(pd.Timestamp(item).date()) for item in invalid_history_dates.tolist()]
        raise ValueError(
            "universe_history contains eligible rows outside trading calendar sessions: "
            f"{bad[:10]}"
        )

    naive_ids, naive_meta = _resolve_naive_ids(
        baseline_mode=baseline_mode,
        universe_current=current,
        ticker_history_map=ticker_map,
        adjusted_prices=adjusted,
        history_last_date=max_date,
    )

    pit_by_date = {
        date: set(frame["instrument_id"].astype(str).tolist())
        for date, frame in history_eligible.groupby("date", sort=True)
    }
    pit_ticker_lookup = {
        (row.date, row.instrument_id): row.ticker
        for row in history_eligible[["date", "instrument_id", "ticker"]]
        .drop_duplicates(subset=["date", "instrument_id"])
        .itertuples(index=False)
    }

    current_ticker_lookup = {
        str(row.instrument_id): str(row.ticker)
        for row in current[["instrument_id", "ticker"]]
        .drop_duplicates(subset=["instrument_id"])
        .itertuples(index=False)
    }

    pit_first_seen = (
        history_eligible.groupby("instrument_id")["date"].min().to_dict()
    )
    pit_last_seen = (
        history_eligible.groupby("instrument_id")["date"].max().to_dict()
    )
    pit_days = (
        history_eligible.groupby("instrument_id")["date"].nunique().to_dict()
    )
    pit_last_ticker = (
        history_eligible.sort_values(["instrument_id", "date"])
        .drop_duplicates(subset=["instrument_id"], keep="last")
        .set_index("instrument_id")["ticker"]
        .astype(str)
        .to_dict()
    )

    price_by_date: dict[pd.Timestamp, set[str]] = {}
    if adjusted is not None and not adjusted.empty and {"date", "instrument_id"}.issubset(adjusted.columns):
        adjusted["date"] = _normalize_date(adjusted["date"], column="date")
        adjusted["instrument_id"] = adjusted["instrument_id"].astype(str)
        close_col = "close_adj" if "close_adj" in adjusted.columns else None
        if close_col is not None:
            adjusted[close_col] = pd.to_numeric(adjusted[close_col], errors="coerce")
            adjusted = adjusted[adjusted[close_col].notna()]
        for date, frame in adjusted.groupby("date", sort=True):
            price_by_date[pd.Timestamp(date).normalize()] = set(
                frame["instrument_id"].astype(str).tolist()
            )

    daily_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    for session in sessions:
        session_ts = pd.Timestamp(session).normalize()
        pit_set = pit_by_date.get(session_ts, set())
        naive_set = set(naive_ids)
        overlap = pit_set & naive_set
        pit_only = pit_set - naive_set
        naive_only = naive_set - pit_set
        union = pit_set | naive_set

        jaccard = 1.0 if not union else float(len(overlap) / len(union))
        pct_pit_retained = 1.0 if not pit_set else float(len(overlap) / len(pit_set))
        pct_naive_ex_post = 0.0 if not naive_set else float(len(naive_only) / len(naive_set))

        row_payload: dict[str, Any] = {
            "date": session_ts,
            "n_names_pit": int(len(pit_set)),
            "n_names_naive": int(len(naive_set)),
            "overlap_count": int(len(overlap)),
            "pit_only_count": int(len(pit_only)),
            "naive_only_count": int(len(naive_only)),
            "jaccard_membership": jaccard,
            "pct_pit_retained_by_naive": pct_pit_retained,
            "pct_naive_ex_post_names": pct_naive_ex_post,
        }

        price_set = price_by_date.get(session_ts, set())
        if price_set:
            row_payload["pit_price_coverage"] = (
                1.0 if not pit_set else float(len(pit_set & price_set) / len(pit_set))
            )
            row_payload["naive_price_coverage"] = (
                1.0 if not naive_set else float(len(naive_set & price_set) / len(naive_set))
            )

        daily_rows.append(row_payload)

        for instrument_id in sorted(pit_only):
            last_seen = pit_last_seen.get(instrument_id)
            classification = (
                "economic_termination_or_terminal_history"
                if pd.notna(last_seen) and pd.Timestamp(last_seen).normalize() < sessions[-1]
                else "structural_missing"
            )
            membership_rows.append(
                {
                    "date": session_ts,
                    "instrument_id": instrument_id,
                    "ticker_pit": pit_ticker_lookup.get((session_ts, instrument_id), ""),
                    "ticker_naive": current_ticker_lookup.get(instrument_id, ""),
                    "pit_member": True,
                    "naive_member": False,
                    "diff_type": "pit_only",
                    "absence_classification": classification,
                }
            )

        for instrument_id in sorted(naive_only):
            first_seen = pit_first_seen.get(instrument_id)
            if first_seen is None:
                classification = "hindsight_only_name"
            elif session_ts < pd.Timestamp(first_seen).normalize():
                classification = "naive_backfill"
            else:
                classification = "naive_only_non_pit"
            membership_rows.append(
                {
                    "date": session_ts,
                    "instrument_id": instrument_id,
                    "ticker_pit": pit_ticker_lookup.get((session_ts, instrument_id), ""),
                    "ticker_naive": current_ticker_lookup.get(instrument_id, ""),
                    "pit_member": False,
                    "naive_member": True,
                    "diff_type": "naive_only",
                    "absence_classification": classification,
                }
            )

    daily = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    membership_diff = pd.DataFrame(membership_rows)
    if not membership_diff.empty:
        membership_diff = membership_diff.sort_values(
            ["date", "diff_type", "instrument_id"]
        ).reset_index(drop=True)

    all_instruments = sorted(
        set(history_eligible["instrument_id"].astype(str).tolist()) | set(naive_ids)
    )
    sessions_count = int(len(sessions))
    session_min = sessions.min()
    session_max = sessions.max()

    symbol_rows: list[dict[str, Any]] = []
    for instrument_id in all_instruments:
        first_seen = pit_first_seen.get(instrument_id, pd.NaT)
        last_seen = pit_last_seen.get(instrument_id, pd.NaT)
        n_days_pit = int(pit_days.get(instrument_id, 0))
        appears_in_naive = instrument_id in naive_ids
        n_days_naive = sessions_count if appears_in_naive else 0
        n_days_overlap = n_days_pit if appears_in_naive else 0
        n_days_pit_only = n_days_pit - n_days_overlap
        n_days_naive_only = n_days_naive - n_days_overlap

        naive_backfilled_flag = bool(
            appears_in_naive
            and pd.notna(first_seen)
            and pd.Timestamp(first_seen).normalize() > session_min
            and n_days_naive_only > 0
        )
        terminal_only_flag = bool(
            (not appears_in_naive)
            and pd.notna(last_seen)
            and pd.Timestamp(last_seen).normalize() < session_max
        )
        naive_hindsight_only_flag = bool(appears_in_naive and n_days_pit == 0)
        pct_pit_covered = (
            1.0 if n_days_pit == 0 else float(n_days_overlap / n_days_pit)
        )

        symbol_rows.append(
            {
                "instrument_id": instrument_id,
                "ticker": pit_last_ticker.get(
                    instrument_id,
                    current_ticker_lookup.get(instrument_id, ""),
                ),
                "first_seen_pit": first_seen if pd.notna(first_seen) else pd.NaT,
                "last_seen_pit": last_seen if pd.notna(last_seen) else pd.NaT,
                "appears_in_naive": appears_in_naive,
                "n_days_pit": n_days_pit,
                "n_days_naive": n_days_naive,
                "n_days_overlap": n_days_overlap,
                "n_days_pit_only": n_days_pit_only,
                "n_days_naive_only": n_days_naive_only,
                "pct_pit_covered_by_naive": pct_pit_covered,
                "naive_backfilled_flag": naive_backfilled_flag,
                "terminal_only_flag": terminal_only_flag,
                "naive_hindsight_only_flag": naive_hindsight_only_flag,
            }
        )

    symbol_level = pd.DataFrame(symbol_rows).sort_values("instrument_id").reset_index(drop=True)

    mean_jaccard_membership = float(daily["jaccard_membership"].mean())
    mean_pct_pit_retained = float(daily["pct_pit_retained_by_naive"].mean())
    mean_pct_naive_ex_post = float(daily["pct_naive_ex_post_names"].mean())
    avg_overlap_count = float(daily["overlap_count"].mean())
    worst_idx = int(daily["jaccard_membership"].idxmin())
    worst_date = pd.Timestamp(daily.loc[worst_idx, "date"]).strftime("%Y-%m-%d")

    n_naive_symbols = max(1, int(len(naive_ids)))
    proportion_hindsight_only = float(
        symbol_level["naive_hindsight_only_flag"].sum() / n_naive_symbols
    )
    terminal_only_ratio = (
        float(symbol_level["terminal_only_flag"].sum() / len(symbol_level))
        if len(symbol_level) > 0
        else 0.0
    )

    severity_score, bias_level, recommended_action = _score_bias(
        mean_pct_pit_retained_by_naive=mean_pct_pit_retained,
        mean_pct_naive_ex_post_names=mean_pct_naive_ex_post,
        mean_jaccard_membership=mean_jaccard_membership,
        terminal_only_ratio=terminal_only_ratio,
    )
    gate_status = (
        "FAIL" if severity_score >= 0.60 else "WARN" if severity_score >= 0.30 else "PASS"
    )

    if output_dir:
        target_root = Path(output_dir).expanduser().resolve()
    else:
        target_root = data_dir() / "universe" / "audit" / run_id
    target_root.mkdir(parents=True, exist_ok=True)

    daily_path = write_parquet(
        daily,
        target_root / "survivorship_daily.parquet",
        schema_name="survivorship_daily_mvp_v1",
        run_id=run_id,
    )
    membership_diff_path = write_parquet(
        _ensure_non_empty_frame(
            membership_diff,
            {
                "date": pd.NaT,
                "instrument_id": "__NONE__",
                "ticker_pit": "",
                "ticker_naive": "",
                "pit_member": False,
                "naive_member": False,
                "diff_type": "none",
                "absence_classification": "none",
            },
        ),
        target_root / "survivorship_membership_diff.parquet",
        schema_name="survivorship_membership_diff_mvp_v1",
        run_id=run_id,
    )
    symbol_level_path = write_parquet(
        _ensure_non_empty_frame(
            symbol_level,
            {
                "instrument_id": "__NONE__",
                "ticker": "",
                "first_seen_pit": pd.NaT,
                "last_seen_pit": pd.NaT,
                "appears_in_naive": False,
                "n_days_pit": 0,
                "n_days_naive": 0,
                "n_days_overlap": 0,
                "n_days_pit_only": 0,
                "n_days_naive_only": 0,
                "pct_pit_covered_by_naive": 1.0,
                "naive_backfilled_flag": False,
                "terminal_only_flag": False,
                "naive_hindsight_only_flag": False,
            },
        ),
        target_root / "survivorship_symbol_level.parquet",
        schema_name="survivorship_symbol_level_mvp_v1",
        run_id=run_id,
    )

    config_hash = _build_config_hash(
        baseline_mode=baseline_mode,
        universe_history_path=history_source,
        universe_current_path=current_source,
        ticker_history_map_path=ticker_map_source,
        trading_calendar_path=calendar_source,
        adjusted_prices_path=adjusted_source if adjusted is not None else None,
    )

    summary_payload: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "config_hash": config_hash,
        "baseline_mode": baseline_mode,
        "baseline_definition": (
            "Project current baseline instrument_ids across all historical sessions "
            "(hindsight-biased naive universe)."
        ),
        "gate_status": gate_status,
        "severity_score": severity_score,
        "bias_level": bias_level,
        "recommended_action": recommended_action,
        "n_sessions": sessions_count,
        "n_names_pit_unique": int(history_eligible["instrument_id"].nunique()),
        "n_names_naive_unique": int(len(naive_ids)),
        "avg_overlap_count": avg_overlap_count,
        "mean_jaccard_membership": mean_jaccard_membership,
        "mean_pct_pit_retained_by_naive": mean_pct_pit_retained,
        "mean_pct_naive_ex_post_names": mean_pct_naive_ex_post,
        "worst_date_by_divergence": worst_date,
        "n_pit_only_total": int(daily["pit_only_count"].sum()),
        "n_naive_only_total": int(daily["naive_only_count"].sum()),
        "n_membership_diff_rows": int(len(membership_diff)),
        "n_naive_backfilled_symbols": int(symbol_level["naive_backfilled_flag"].sum()),
        "n_terminal_only_symbols": int(symbol_level["terminal_only_flag"].sum()),
        "proportion_hindsight_only_names": proportion_hindsight_only,
        "naive_resolution_meta": naive_meta,
        "input_paths": {
            "universe_history": str(history_source),
            "universe_current": str(current_source),
            "ticker_history_map": str(ticker_map_source),
            "trading_calendar": str(calendar_source),
            "adjusted_prices": str(adjusted_source) if adjusted is not None else "",
        },
    }
    if "pit_price_coverage" in daily.columns and "naive_price_coverage" in daily.columns:
        summary_payload["mean_pit_price_coverage"] = float(daily["pit_price_coverage"].mean())
        summary_payload["mean_naive_price_coverage"] = float(daily["naive_price_coverage"].mean())

    summary_path = target_root / "survivorship_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manifest_payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "config_hash": config_hash,
        "baseline_mode": baseline_mode,
        "gate_status": gate_status,
        "severity_score": severity_score,
        "daily_path": str(daily_path),
        "membership_diff_path": str(membership_diff_path),
        "symbol_level_path": str(symbol_level_path),
        "summary_path": str(summary_path),
        "output_dir": str(target_root),
    }
    manifest_path = target_root / "survivorship_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger.info(
        "survivorship_analysis_completed",
        run_id=run_id,
        baseline_mode=baseline_mode,
        gate_status=gate_status,
        severity_score=round(severity_score, 6),
        bias_level=bias_level,
        n_sessions=sessions_count,
        n_membership_diff_rows=int(len(membership_diff)),
        output_dir=str(target_root),
    )

    return SurvivorshipResult(
        summary_path=summary_path,
        daily_path=daily_path,
        membership_diff_path=membership_diff_path,
        symbol_level_path=symbol_level_path,
        manifest_path=manifest_path,
        bias_level=bias_level,
        severity_score=severity_score,
        gate_status=gate_status,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MVP survivorship analysis comparing PIT universe vs naive hindsight baseline."
    )
    parser.add_argument("--universe-history-path", type=str, default=None)
    parser.add_argument("--universe-current-path", type=str, default=None)
    parser.add_argument("--ticker-history-map-path", type=str, default=None)
    parser.add_argument("--trading-calendar-path", type=str, default=None)
    parser.add_argument("--adjusted-prices-path", type=str, default=None)
    parser.add_argument(
        "--baseline-mode",
        type=str,
        default="current_eligible",
        choices=sorted(ALLOWED_BASELINE_MODES),
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default="survivorship_mvp_v1")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_survivorship_analysis(
        universe_history_path=args.universe_history_path,
        universe_current_path=args.universe_current_path,
        ticker_history_map_path=args.ticker_history_map_path,
        trading_calendar_path=args.trading_calendar_path,
        adjusted_prices_path=args.adjusted_prices_path,
        baseline_mode=args.baseline_mode,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print("Survivorship analysis completed:")
    print(f"- gate_status: {result.gate_status}")
    print(f"- bias_level: {result.bias_level}")
    print(f"- severity_score: {result.severity_score:.6f}")
    print(f"- summary: {result.summary_path}")
    print(f"- daily: {result.daily_path}")
    print(f"- membership_diff: {result.membership_diff_path}")
    print(f"- symbol_level: {result.symbol_level_path}")
    print(f"- manifest: {result.manifest_path}")


if __name__ == "__main__":
    main()
