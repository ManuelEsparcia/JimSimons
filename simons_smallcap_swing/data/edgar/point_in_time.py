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

# Allow direct script execution: `python simons_smallcap_swing/data/edgar/point_in_time.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import assert_pit_no_lookahead

ALLOWED_METRICS: tuple[str, ...] = (
    "revenue",
    "net_income",
    "total_assets",
    "shares_outstanding",
)

# Minimal, explicit metric map for week-2 PIT v2.
METRIC_TAG_MAP: dict[str, tuple[tuple[str, str], ...]] = {
    "revenue": (
        ("us-gaap", "revenues"),
        ("us-gaap", "revenuefromcontractwithcustomerexcludingassessedtax"),
        ("us-gaap", "salesrevenuenet"),
    ),
    "net_income": (
        ("us-gaap", "netincomeloss"),
        ("us-gaap", "profitloss"),
    ),
    "total_assets": (
        ("us-gaap", "assets"),
    ),
    "shares_outstanding": (
        ("us-gaap", "commonstocksharesoutstanding"),
        ("dei", "entitycommonstocksharesoutstanding"),
    ),
}

FORM_PRIORITY = {
    "10-K": 3,
    "10-Q": 2,
    "8-K": 1,
}

MAX_STALENESS_DAYS: dict[str, int] = {
    "revenue": 550,
    "net_income": 550,
    "total_assets": 650,
    "shares_outstanding": 700,
}

NULL_STRINGS = {"", "none", "nan", "null", "nat"}

SYNTHETIC_PERIOD_DEFS: tuple[dict[str, str], ...] = (
    {
        "fact_end_date": "2025-09-30",
        "filing_date": "2025-11-12",
        "acceptance_ts": "2025-11-12T21:30:00Z",
        "accession_number": "SYN-2025Q3",
        "fiscal_period": "Q3",
    },
    {
        "fact_end_date": "2025-12-31",
        "filing_date": "2026-02-18",
        "acceptance_ts": "2026-02-18T21:30:00Z",
        "accession_number": "SYN-2025FY",
        "fiscal_period": "FY",
    },
)


@dataclass(frozen=True)
class FundamentalsPITResult:
    fundamentals_events_path: Path
    fundamentals_pit_path: Path
    event_row_count: int
    pit_row_count: int
    n_instruments: int
    config_hash: str


def _normalize_token(value: object) -> str:
    text = str(value).strip().lower()
    if text in NULL_STRINGS:
        return ""
    return "".join(ch for ch in text if ch.isalnum())


def _normalize_string_series(values: pd.Series, *, uppercase: bool = False) -> pd.Series:
    out = values.astype(str).str.strip()
    out = out.mask(out.str.lower().isin(NULL_STRINGS), "")
    if uppercase:
        out = out.str.upper()
    return out


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains invalid dates.")
    return parsed.dt.normalize()


def _normalize_cik(value: object) -> str:
    text = "".join(ch for ch in str(value) if ch.isdigit())
    if not text:
        return ""
    if len(text) > 10:
        text = text[-10:]
    return text.zfill(10)


def _stable_unit(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    value = int(digest, 16)
    return value / float(16**16 - 1)


def _config_hash(*, source_mode: str) -> str:
    payload = {
        "version": "fundamentals_pit_v2",
        "source_mode": source_mode,
        "allowed_metrics": list(ALLOWED_METRICS),
        "metric_tag_map": {
            metric: list(pairs) for metric, pairs in METRIC_TAG_MAP.items()
        },
        "max_staleness_days": MAX_STALENESS_DAYS,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_inputs(
    *,
    ticker_cik_map_path: str | Path | None,
    universe_history_path: str | Path | None,
    companyfacts_raw_path: str | Path | None,
    submissions_raw_path: str | Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    edgar_root = data_dir() / "edgar"
    ticker_source = (
        Path(ticker_cik_map_path).expanduser().resolve()
        if ticker_cik_map_path
        else edgar_root / "ticker_cik_map.parquet"
    )
    universe_source = (
        Path(universe_history_path).expanduser().resolve()
        if universe_history_path
        else data_dir() / "universe" / "universe_history.parquet"
    )
    companyfacts_source = (
        Path(companyfacts_raw_path).expanduser().resolve()
        if companyfacts_raw_path
        else edgar_root / "companyfacts_raw.parquet"
    )
    submissions_source = (
        Path(submissions_raw_path).expanduser().resolve()
        if submissions_raw_path
        else edgar_root / "submissions_raw.parquet"
    )

    ticker_cik_map = read_parquet(ticker_source)
    universe_history = read_parquet(universe_source)
    companyfacts_raw = read_parquet(companyfacts_source) if companyfacts_source.exists() else None
    submissions_raw = read_parquet(submissions_source) if submissions_source.exists() else None

    required_map = {"instrument_id", "ticker", "cik", "start_date", "end_date"}
    missing_map = sorted(required_map - set(ticker_cik_map.columns))
    if missing_map:
        raise ValueError(f"ticker_cik_map missing required columns: {missing_map}")

    required_universe = {"date", "instrument_id", "ticker"}
    missing_universe = sorted(required_universe - set(universe_history.columns))
    if missing_universe:
        raise ValueError(f"universe_history missing required columns: {missing_universe}")

    return ticker_cik_map, universe_history, companyfacts_raw, submissions_raw


def _resolve_panel_with_cik(
    universe_history: pd.DataFrame,
    ticker_cik_map: pd.DataFrame,
) -> pd.DataFrame:
    panel = universe_history[["date", "instrument_id", "ticker"]].copy()
    panel["date"] = _normalize_date(panel["date"], column="date")
    panel["instrument_id"] = panel["instrument_id"].astype(str)
    panel["ticker"] = panel["ticker"].astype(str)
    panel = panel.drop_duplicates(subset=["date", "instrument_id"]).sort_values(
        ["instrument_id", "date"]
    )

    mapping = ticker_cik_map.copy()
    mapping["instrument_id"] = mapping["instrument_id"].astype(str)
    mapping["ticker"] = mapping["ticker"].astype(str)
    mapping["cik"] = mapping["cik"].map(_normalize_cik)
    mapping["start_date"] = _normalize_date(mapping["start_date"], column="start_date")
    mapping["end_date"] = pd.to_datetime(mapping["end_date"], errors="coerce").dt.normalize()

    invalid_interval = mapping["end_date"].notna() & (mapping["end_date"] < mapping["start_date"])
    if invalid_interval.any():
        raise ValueError("ticker_cik_map has invalid intervals: end_date < start_date.")

    probe = panel.reset_index(drop=True).copy()
    probe["__row_id"] = probe.index
    merged = probe.merge(
        mapping[["instrument_id", "ticker", "cik", "start_date", "end_date"]],
        on=["instrument_id", "ticker"],
        how="left",
    )
    merged["valid"] = (
        merged["start_date"].notna()
        & (merged["date"] >= merged["start_date"])
        & (merged["end_date"].isna() | (merged["date"] <= merged["end_date"]))
    )

    valid_rows = merged[merged["valid"]].copy()
    if valid_rows.empty:
        raise ValueError("No universe rows could be resolved to a valid CIK interval.")

    dup_valid = valid_rows.duplicated(["__row_id"], keep=False)
    if dup_valid.any():
        sample = valid_rows.loc[
            dup_valid, ["date", "instrument_id", "ticker", "cik"]
        ].head(10)
        raise ValueError(
            "Ambiguous CIK mapping: multiple valid rows for same universe row. "
            f"Sample:\n{sample.to_string(index=False)}"
        )

    mapped = probe.merge(
        valid_rows[["__row_id", "cik"]],
        on="__row_id",
        how="left",
    )
    unresolved = mapped["cik"].isna()
    if unresolved.any():
        sample = mapped.loc[unresolved, ["date", "instrument_id", "ticker"]].head(10)
        raise ValueError(
            "Universe rows unresolved to CIK under PIT intervals. "
            f"Sample:\n{sample.to_string(index=False)}"
        )

    mapped = mapped.drop(columns=["__row_id"])
    mapped["asof_date"] = (
        pd.to_datetime(mapped["date"], utc=True)
        + pd.Timedelta(hours=23, minutes=59, seconds=59)
    )
    mapped = mapped.sort_values(["instrument_id", "asof_date"]).reset_index(drop=True)
    return mapped


def _prepare_submissions_lookup(submissions_raw: pd.DataFrame | None) -> pd.DataFrame:
    if submissions_raw is None or submissions_raw.empty:
        return pd.DataFrame(
            columns=[
                "cik",
                "accession_number",
                "acceptance_ts_sub",
                "filing_date_sub",
                "form_type_sub",
            ]
        )

    required = {"cik", "accession_number", "acceptance_ts", "filing_date", "form_type"}
    if not required.issubset(submissions_raw.columns):
        return pd.DataFrame(
            columns=[
                "cik",
                "accession_number",
                "acceptance_ts_sub",
                "filing_date_sub",
                "form_type_sub",
            ]
        )

    frame = submissions_raw[list(required)].copy()
    frame["cik"] = frame["cik"].map(_normalize_cik)
    frame["accession_number"] = _normalize_string_series(frame["accession_number"])
    frame["acceptance_ts_sub"] = pd.to_datetime(frame["acceptance_ts"], utc=True, errors="coerce")
    frame["filing_date_sub"] = pd.to_datetime(frame["filing_date"], errors="coerce").dt.normalize()
    frame["form_type_sub"] = _normalize_string_series(frame["form_type"], uppercase=True)
    frame = frame[
        (frame["cik"] != "")
        & (frame["accession_number"] != "")
        & frame["acceptance_ts_sub"].notna()
        & frame["filing_date_sub"].notna()
    ].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "cik",
                "accession_number",
                "acceptance_ts_sub",
                "filing_date_sub",
                "form_type_sub",
            ]
        )

    frame = frame.sort_values(
        ["cik", "accession_number", "acceptance_ts_sub", "filing_date_sub"]
    )
    frame = frame.drop_duplicates(["cik", "accession_number"], keep="last")
    return frame[
        ["cik", "accession_number", "acceptance_ts_sub", "filing_date_sub", "form_type_sub"]
    ].reset_index(drop=True)


def _metric_lookup() -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for metric_name, pairs in METRIC_TAG_MAP.items():
        for taxonomy, tag in pairs:
            lookup[(_normalize_token(taxonomy), _normalize_token(tag))] = metric_name
    return lookup


def _canonicalize_events_from_companyfacts(
    companyfacts_raw: pd.DataFrame,
    submissions_lookup: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "cik",
        "taxonomy",
        "tag",
        "unit",
        "fact_value",
        "fact_start_date",
        "fact_end_date",
        "filing_date",
        "acceptance_ts",
        "fiscal_year",
        "fiscal_period",
        "form_type",
        "accession_number",
        "source_ref",
    }
    missing = sorted(required - set(companyfacts_raw.columns))
    if missing:
        raise ValueError(f"companyfacts_raw missing required columns: {missing}")

    facts = companyfacts_raw.copy()
    facts["cik"] = facts["cik"].map(_normalize_cik)
    facts["taxonomy"] = _normalize_string_series(facts["taxonomy"])
    facts["tag"] = _normalize_string_series(facts["tag"])
    facts["unit"] = _normalize_string_series(facts["unit"], uppercase=True)
    facts["fiscal_period"] = _normalize_string_series(facts["fiscal_period"], uppercase=True)
    facts["form_type"] = _normalize_string_series(facts["form_type"], uppercase=True)
    facts["accession_number"] = _normalize_string_series(facts["accession_number"])
    facts["source_ref"] = _normalize_string_series(facts["source_ref"])
    facts["fact_value"] = pd.to_numeric(facts["fact_value"], errors="coerce")
    facts["fiscal_year"] = pd.to_numeric(facts["fiscal_year"], errors="coerce").astype("Int64")
    facts["fact_start_date"] = pd.to_datetime(facts["fact_start_date"], errors="coerce").dt.normalize()
    facts["fact_end_date"] = pd.to_datetime(facts["fact_end_date"], errors="coerce").dt.normalize()
    facts["filing_date"] = pd.to_datetime(facts["filing_date"], errors="coerce").dt.normalize()
    facts["acceptance_ts"] = pd.to_datetime(facts["acceptance_ts"], utc=True, errors="coerce")

    metric_map = _metric_lookup()
    facts["taxonomy_norm"] = facts["taxonomy"].map(_normalize_token)
    facts["tag_norm"] = facts["tag"].map(_normalize_token)
    facts["metric_name"] = [
        metric_map.get((tax, tag), "")
        for tax, tag in zip(facts["taxonomy_norm"], facts["tag_norm"], strict=False)
    ]
    facts = facts[facts["metric_name"].isin(ALLOWED_METRICS)].copy()
    if facts.empty:
        return pd.DataFrame()

    if not submissions_lookup.empty:
        facts = facts.merge(
            submissions_lookup,
            on=["cik", "accession_number"],
            how="left",
        )
    else:
        facts["acceptance_ts_sub"] = pd.NaT
        facts["filing_date_sub"] = pd.NaT
        facts["form_type_sub"] = ""

    facts["acceptance_ts"] = facts["acceptance_ts"].fillna(facts["acceptance_ts_sub"])
    facts["filing_date"] = facts["filing_date"].fillna(facts["filing_date_sub"])
    facts["form_type"] = facts["form_type"].mask(facts["form_type"] == "", facts["form_type_sub"])

    fill_acceptance = facts["acceptance_ts"].isna() & facts["filing_date"].notna()
    if fill_acceptance.any():
        facts.loc[fill_acceptance, "acceptance_ts"] = pd.to_datetime(
            facts.loc[fill_acceptance, "filing_date"],
            utc=True,
            errors="coerce",
        )

    facts["metric_unit"] = facts["unit"]
    facts["source_type"] = "companyfacts_raw_v2"
    facts["data_quality"] = "ok"
    facts["form_priority"] = facts["form_type"].map(FORM_PRIORITY).fillna(0).astype(int)
    facts["source_record_id"] = [
        hashlib.sha256(
            "|".join(
                [
                    str(cik),
                    str(metric),
                    str(acc),
                    str(end_date),
                    str(filing),
                    str(value),
                    str(src),
                ]
            ).encode("utf-8")
        ).hexdigest()
        for cik, metric, acc, end_date, filing, value, src in zip(
            facts["cik"],
            facts["metric_name"],
            facts["accession_number"],
            facts["fact_end_date"],
            facts["filing_date"],
            facts["fact_value"],
            facts["source_ref"],
            strict=False,
        )
    ]

    critical_mask = (
        (facts["cik"] != "")
        & (facts["metric_name"] != "")
        & facts["fact_value"].notna()
        & facts["filing_date"].notna()
        & facts["acceptance_ts"].notna()
    )
    facts = facts[critical_mask].copy()
    if facts.empty:
        return pd.DataFrame()

    acceptance_date = (
        facts["acceptance_ts"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    )
    temporal_ok = (
        (acceptance_date >= facts["filing_date"])
        & (
            facts["fact_end_date"].isna()
            | (facts["fact_end_date"] <= facts["filing_date"])
        )
        & (
            facts["fact_start_date"].isna()
            | facts["fact_end_date"].isna()
            | (facts["fact_start_date"] <= facts["fact_end_date"])
        )
    )
    facts = facts[temporal_ok].copy()
    if facts.empty:
        return pd.DataFrame()

    dedup_key = [
        "cik",
        "metric_name",
        "acceptance_ts",
        "filing_date",
        "fact_end_date",
        "accession_number",
        "fact_value",
        "source_record_id",
    ]
    facts = facts.sort_values(
        ["cik", "metric_name", "acceptance_ts", "filing_date", "form_priority", "accession_number", "source_record_id"]
    )
    facts = facts.drop_duplicates(dedup_key, keep="last")

    out = facts[
        [
            "cik",
            "taxonomy",
            "tag",
            "metric_name",
            "fact_value",
            "metric_unit",
            "fact_start_date",
            "fact_end_date",
            "filing_date",
            "acceptance_ts",
            "fiscal_year",
            "fiscal_period",
            "form_type",
            "form_priority",
            "accession_number",
            "source_type",
            "data_quality",
            "source_ref",
            "source_record_id",
        ]
    ].copy()
    return out.reset_index(drop=True)


def _build_synthetic_events(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in (
        panel[["instrument_id", "cik"]]
        .drop_duplicates(subset=["instrument_id", "cik"])
        .sort_values(["instrument_id"])
        .itertuples(index=False)
    ):
        instrument_id = str(row.instrument_id)
        cik = str(row.cik)
        scale = 0.75 + 0.85 * _stable_unit(f"{instrument_id}:scale")
        base_values = {
            "revenue": 90_000_000.0,
            "net_income": 9_000_000.0,
            "total_assets": 250_000_000.0,
            "shares_outstanding": 24_000_000.0,
        }

        for period_idx, period in enumerate(SYNTHETIC_PERIOD_DEFS):
            fact_end = pd.Timestamp(period["fact_end_date"]).normalize()
            filing_date = pd.Timestamp(period["filing_date"]).normalize()
            acceptance_ts = pd.Timestamp(period["acceptance_ts"], tz="UTC")
            for metric_name in ALLOWED_METRICS:
                growth = 1.0 + 0.04 * period_idx
                noise = 0.98 + 0.04 * _stable_unit(f"{instrument_id}:{metric_name}:{period_idx}")
                value = base_values[metric_name] * scale * growth * noise
                if metric_name == "shares_outstanding":
                    value = round(value, 0)
                    unit = "SHARES"
                else:
                    value = round(value, 2)
                    unit = "USD"
                accession = f"{period['accession_number']}:{instrument_id}:{metric_name}"
                rows.append(
                    {
                        "cik": cik,
                        "taxonomy": "synthetic",
                        "tag": metric_name,
                        "metric_name": metric_name,
                        "fact_value": float(value),
                        "metric_unit": unit,
                        "fact_start_date": pd.NaT,
                        "fact_end_date": fact_end,
                        "filing_date": filing_date,
                        "acceptance_ts": acceptance_ts,
                        "fiscal_year": int(fact_end.year),
                        "fiscal_period": period["fiscal_period"],
                        "form_type": "SYNTH",
                        "form_priority": 0,
                        "accession_number": accession,
                        "source_type": "synthetic_fallback_v2",
                        "data_quality": "synthetic",
                        "source_ref": "synthetic://point_in_time_v2",
                        "source_record_id": hashlib.sha256(accession.encode("utf-8")).hexdigest(),
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["cik", "metric_name", "acceptance_ts", "filing_date", "accession_number"]
    ).reset_index(drop=True)


def _materialize_fundamentals_pit(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    run_id: str,
    config_hash: str,
) -> pd.DataFrame:
    pit_rows: list[pd.DataFrame] = []
    panel = panel.sort_values(["instrument_id", "asof_date"]).reset_index(drop=True)
    events = events.sort_values(
        ["cik", "metric_name", "acceptance_ts", "filing_date", "form_priority", "accession_number", "source_record_id"]
    ).reset_index(drop=True)

    for panel_row in (
        panel[["instrument_id", "ticker", "cik"]]
        .drop_duplicates(subset=["instrument_id", "ticker", "cik"])
        .itertuples(index=False)
    ):
        instrument_id = str(panel_row.instrument_id)
        ticker = str(panel_row.ticker)
        cik = str(panel_row.cik)

        asof_frame = panel[
            (panel["instrument_id"] == instrument_id)
            & (panel["ticker"] == ticker)
            & (panel["cik"] == cik)
        ][["asof_date", "date"]].sort_values("asof_date")
        if asof_frame.empty:
            continue

        cik_events = events[events["cik"] == cik]
        if cik_events.empty:
            continue

        for metric_name in ALLOWED_METRICS:
            metric_events = cik_events[cik_events["metric_name"] == metric_name]
            if metric_events.empty:
                continue

            right = metric_events[
                [
                    "acceptance_ts",
                    "filing_date",
                    "fact_start_date",
                    "fact_end_date",
                    "fiscal_year",
                    "fiscal_period",
                    "form_type",
                    "taxonomy",
                    "tag",
                    "metric_name",
                    "fact_value",
                    "metric_unit",
                    "accession_number",
                    "source_type",
                    "data_quality",
                    "source_ref",
                    "source_record_id",
                    "form_priority",
                ]
            ].sort_values(
                ["acceptance_ts", "filing_date", "form_priority", "accession_number", "source_record_id"]
            )

            merged = pd.merge_asof(
                asof_frame[["asof_date"]].sort_values("asof_date"),
                right,
                left_on="asof_date",
                right_on="acceptance_ts",
                direction="backward",
                allow_exact_matches=True,
            )
            merged = merged[merged["acceptance_ts"].notna()].copy()
            if merged.empty:
                continue

            merged = merged.merge(
                asof_frame[["asof_date", "date"]],
                on="asof_date",
                how="left",
            )
            merged["instrument_id"] = instrument_id
            merged["ticker"] = ticker
            merged["cik"] = cik
            merged["asof_session_date"] = pd.to_datetime(merged["date"], errors="coerce").dt.normalize()
            merged["staleness_days"] = (
                merged["asof_date"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
                - merged["acceptance_ts"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
            ).dt.days.astype(int)
            stale_limit = MAX_STALENESS_DAYS.get(metric_name, 600)
            merged.loc[merged["staleness_days"] > stale_limit, "data_quality"] = "stale"
            pit_rows.append(merged)

    if not pit_rows:
        raise ValueError("No visible fundamentals rows after PIT materialization.")

    pit = pd.concat(pit_rows, ignore_index=True)
    pit["fiscal_period_end"] = pit["fact_end_date"]
    pit["metric_value"] = pd.to_numeric(pit["fact_value"], errors="coerce")
    pit["visibility_rule"] = (
        "acceptance_ts<=asof_date|max_acceptance_then_filing_then_form_priority_then_accession"
    )
    pit["run_id"] = run_id
    pit["config_hash"] = config_hash
    pit["built_ts_utc"] = datetime.now(UTC).isoformat()

    pit = pit[
        [
            "instrument_id",
            "ticker",
            "cik",
            "asof_date",
            "asof_session_date",
            "acceptance_ts",
            "filing_date",
            "fact_start_date",
            "fact_end_date",
            "fiscal_period_end",
            "fiscal_year",
            "fiscal_period",
            "form_type",
            "taxonomy",
            "tag",
            "metric_name",
            "metric_value",
            "metric_unit",
            "accession_number",
            "source_type",
            "data_quality",
            "visibility_rule",
            "source_ref",
            "source_record_id",
            "staleness_days",
            "run_id",
            "config_hash",
            "built_ts_utc",
        ]
    ].sort_values(["instrument_id", "asof_date", "metric_name"]).reset_index(drop=True)

    assert_pit_no_lookahead(pit, decision_col="asof_date", available_col="acceptance_ts")

    dup = pit.duplicated(["instrument_id", "asof_date", "metric_name"], keep=False)
    if dup.any():
        raise ValueError(
            "fundamentals_pit contains duplicate (instrument_id, asof_date, metric_name)."
        )
    return pit


def build_fundamentals_pit(
    *,
    ticker_cik_map_path: str | Path | None = None,
    universe_history_path: str | Path | None = None,
    submissions_raw_path: str | Path | None = None,
    companyfacts_raw_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    run_id: str = "fundamentals_pit_v2",
) -> FundamentalsPITResult:
    logger = get_logger("data.edgar.point_in_time")
    ticker_cik_map, universe_history, companyfacts_raw, submissions_raw = _load_inputs(
        ticker_cik_map_path=ticker_cik_map_path,
        universe_history_path=universe_history_path,
        companyfacts_raw_path=companyfacts_raw_path,
        submissions_raw_path=submissions_raw_path,
    )
    panel = _resolve_panel_with_cik(universe_history, ticker_cik_map)
    submissions_lookup = _prepare_submissions_lookup(submissions_raw)

    source_mode = "companyfacts_raw"
    events = pd.DataFrame()
    if companyfacts_raw is not None and not companyfacts_raw.empty:
        events = _canonicalize_events_from_companyfacts(companyfacts_raw, submissions_lookup)

    if events.empty:
        # Compatibility fallback so Week-1 runner/tests stay operable.
        source_mode = "synthetic_fallback"
        events = _build_synthetic_events(panel)
        logger.warning(
            "fundamentals_pit_v2_using_synthetic_fallback",
            run_id=run_id,
            reason="companyfacts_raw missing or produced no canonical events",
        )

    config_hash = _config_hash(source_mode=source_mode)
    pit = _materialize_fundamentals_pit(
        panel,
        events,
        run_id=run_id,
        config_hash=config_hash,
    )

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (data_dir() / "edgar")
    target_dir.mkdir(parents=True, exist_ok=True)

    events_with_build = events.copy()
    events_with_build["run_id"] = run_id
    events_with_build["config_hash"] = config_hash
    events_with_build["built_ts_utc"] = datetime.now(UTC).isoformat()
    events_path = write_parquet(
        events_with_build,
        target_dir / "fundamentals_events.parquet",
        schema_name="edgar_fundamentals_events_v2",
        run_id=run_id,
    )
    pit_path = write_parquet(
        pit,
        target_dir / "fundamentals_pit.parquet",
        schema_name="edgar_fundamentals_pit_v2",
        run_id=run_id,
    )

    logger.info(
        "fundamentals_pit_built",
        run_id=run_id,
        source_mode=source_mode,
        event_rows=int(len(events_with_build)),
        pit_rows=int(len(pit)),
        n_instruments=int(pit["instrument_id"].nunique()),
        output_events=str(events_path),
        output_pit=str(pit_path),
    )

    return FundamentalsPITResult(
        fundamentals_events_path=events_path,
        fundamentals_pit_path=pit_path,
        event_row_count=int(len(events_with_build)),
        pit_row_count=int(len(pit)),
        n_instruments=int(pit["instrument_id"].nunique()),
        config_hash=config_hash,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build EDGAR fundamentals PIT v2 dataset.")
    parser.add_argument("--ticker-cik-map-path", type=str, default=None)
    parser.add_argument("--universe-history-path", type=str, default=None)
    parser.add_argument("--submissions-raw-path", type=str, default=None)
    parser.add_argument("--companyfacts-raw-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default="fundamentals_pit_v2")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = build_fundamentals_pit(
        ticker_cik_map_path=args.ticker_cik_map_path,
        universe_history_path=args.universe_history_path,
        submissions_raw_path=args.submissions_raw_path,
        companyfacts_raw_path=args.companyfacts_raw_path,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print("Fundamentals PIT built:")
    print(f"- events: {result.fundamentals_events_path}")
    print(f"- pit: {result.fundamentals_pit_path}")
    print(f"- event rows: {result.event_row_count}")
    print(f"- pit rows: {result.pit_row_count}")


if __name__ == "__main__":
    main()
