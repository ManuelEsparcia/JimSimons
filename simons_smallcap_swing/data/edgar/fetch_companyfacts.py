from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import pandas as pd

# Allow direct script execution: `python simons_smallcap_swing/data/edgar/fetch_companyfacts.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger

ALLOWED_INGESTION_MODES: tuple[str, ...] = ("auto", "local_file", "remote_optional")
DEFAULT_REMOTE_TIMEOUT_SEC = 14
DEFAULT_USER_AGENT = (
    "JimSimons-Research-MVP/1.0 "
    "(contact: local-test@example.com)"
)
NULL_STRINGS = {"", "none", "nan", "null", "nat"}

CANONICAL_COLUMNS: tuple[str, ...] = (
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
    "frame",
    "accession_number",
    "source_mode",
    "source_ref",
    "run_id",
    "config_hash",
    "built_ts_utc",
)

ALIAS_CANDIDATES: dict[str, tuple[str, ...]] = {
    "cik": ("cik", "cik_str", "ciknumber"),
    "taxonomy": ("taxonomy", "tax", "tax_name"),
    "tag": ("tag", "concept", "metric_name"),
    "unit": ("unit", "units", "uom"),
    "fact_value": ("fact_value", "value", "val", "factval"),
    "fact_start_date": ("fact_start_date", "fact_start", "start_date", "start"),
    "fact_end_date": ("fact_end_date", "fact_end", "end_date", "end"),
    "filing_date": ("filing_date", "filed_date", "filed", "filingdate"),
    "acceptance_ts": (
        "acceptance_ts",
        "acceptance_datetime",
        "acceptancedatetime",
        "acceptance_date_time",
        "acceptance",
        "accepted",
        "acceptanceDateTime",
    ),
    "fiscal_year": ("fiscal_year", "fy", "fyear"),
    "fiscal_period": ("fiscal_period", "fp", "fperiod"),
    "form_type": ("form_type", "form", "formtype"),
    "frame": ("frame", "context_frame", "context"),
    "accession_number": (
        "accession_number",
        "accessionnumber",
        "accession_no",
        "accession",
        "accn",
    ),
}


@dataclass(frozen=True)
class FetchCompanyFactsResult:
    companyfacts_raw_path: Path
    ingestion_report_path: Path
    row_count: int
    ciks_requested: int
    ciks_covered: int
    source_mode: str
    config_hash: str
    reused_cache: bool


def _normalize_col_name(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def _normalize_cik(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = "".join(ch for ch in str(value) if ch.isdigit())
    if not text:
        return None
    if len(text) > 10:
        text = text[-10:]
    return text.zfill(10)


def _normalize_string(values: pd.Series, *, uppercase: bool = False) -> pd.Series:
    normalized = values.astype(str).str.strip()
    lowered = normalized.str.lower()
    normalized = normalized.mask(lowered.isin(NULL_STRINGS), "")
    if uppercase:
        normalized = normalized.str.upper()
    return normalized


def _pick_col(columns: set[str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in columns:
            return alias
    return None


def _resolve_target_ciks(
    *,
    ticker_cik_map: pd.DataFrame,
    universe_history: pd.DataFrame | None,
) -> list[str]:
    required_map = {"instrument_id", "cik"}
    missing_map = sorted(required_map - set(ticker_cik_map.columns))
    if missing_map:
        raise ValueError(f"ticker_cik_map missing required columns: {missing_map}")

    mapping = ticker_cik_map.copy()
    mapping["instrument_id"] = mapping["instrument_id"].astype(str)
    mapping["cik"] = mapping["cik"].map(_normalize_cik)
    mapping = mapping[mapping["cik"].notna()].copy()
    if mapping.empty:
        raise ValueError("ticker_cik_map has no valid CIK values.")

    if universe_history is not None:
        required_universe = {"instrument_id"}
        missing_universe = sorted(required_universe - set(universe_history.columns))
        if missing_universe:
            raise ValueError(f"universe_history missing required columns: {missing_universe}")
        universe = universe_history.copy()
        universe["instrument_id"] = universe["instrument_id"].astype(str)
        if "is_eligible" in universe.columns:
            universe = universe[universe["is_eligible"].astype(bool)]
        instrument_set = set(universe["instrument_id"].tolist())
        mapping = mapping[mapping["instrument_id"].isin(instrument_set)].copy()

    ciks = sorted(set(mapping["cik"].astype(str).tolist()))
    if not ciks:
        raise ValueError("No target CIKs resolved from ticker_cik_map and universe filter.")
    return ciks


def _discover_local_files(local_source_path: Path) -> list[Path]:
    if not local_source_path.exists():
        return []
    if local_source_path.is_file():
        return [local_source_path.resolve()]
    if not local_source_path.is_dir():
        return []
    files: list[Path] = []
    for pattern in ("*.json", "*.csv", "*.parquet"):
        files.extend(sorted(local_source_path.glob(pattern)))
    return [path.resolve() for path in files]


def _records_from_sec_companyfacts_payload(payload: dict[str, Any], source_ref: str) -> list[dict[str, Any]]:
    facts = payload.get("facts")
    if not isinstance(facts, dict):
        return []

    cik_value = _normalize_cik(payload.get("cik") or payload.get("cik_str"))
    rows: list[dict[str, Any]] = []
    for taxonomy, concepts in facts.items():
        if not isinstance(concepts, dict):
            continue
        for tag, tag_payload in concepts.items():
            if not isinstance(tag_payload, dict):
                continue
            units = tag_payload.get("units")
            if not isinstance(units, dict):
                continue
            for unit, observations in units.items():
                if not isinstance(observations, list):
                    continue
                for obs in observations:
                    if not isinstance(obs, dict):
                        continue
                    rows.append(
                        {
                            "cik": cik_value,
                            "taxonomy": taxonomy,
                            "tag": tag,
                            "unit": unit,
                            "fact_value": obs.get("val"),
                            "fact_start_date": obs.get("start"),
                            "fact_end_date": obs.get("end"),
                            "filing_date": obs.get("filed"),
                            "acceptance_ts": (
                                obs.get("acceptance")
                                or obs.get("acceptance_datetime")
                                or obs.get("acceptanceDateTime")
                            ),
                            "fiscal_year": obs.get("fy") or obs.get("fiscalYear"),
                            "fiscal_period": obs.get("fp") or obs.get("fiscalPeriod"),
                            "form_type": obs.get("form"),
                            "frame": obs.get("frame"),
                            "accession_number": (
                                obs.get("accn")
                                or obs.get("accessionNumber")
                                or obs.get("accession_number")
                            ),
                            "source_ref": source_ref,
                        }
                    )
    return rows


def _records_from_generic_json(payload: Any, source_ref: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        sec_rows = _records_from_sec_companyfacts_payload(payload, source_ref)
        if sec_rows:
            return sec_rows
        if "records" in payload and isinstance(payload["records"], list):
            rows = [dict(item) for item in payload["records"] if isinstance(item, dict)]
        elif any(
            key in payload
            for key in ("cik", "taxonomy", "tag", "unit", "fact_value", "value", "val")
        ):
            rows = [dict(payload)]
    elif isinstance(payload, list):
        rows = [dict(item) for item in payload if isinstance(item, dict)]

    for row in rows:
        row["source_ref"] = source_ref
    return rows


def _table_to_records(table: pd.DataFrame, source_ref: str) -> list[dict[str, Any]]:
    if table.empty:
        return []
    normalized = table.copy()
    normalized.columns = [_normalize_col_name(col) for col in normalized.columns]
    rows = normalized.to_dict(orient="records")
    for row in rows:
        row["source_ref"] = source_ref
    return rows


def _load_local_records(local_files: list[Path]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {
        "files_scanned": int(len(local_files)),
        "json_files": 0,
        "csv_files": 0,
        "parquet_files": 0,
    }
    all_records: list[dict[str, Any]] = []
    for file_path in local_files:
        suffix = file_path.suffix.lower()
        source_ref = str(file_path)
        if suffix == ".json":
            stats["json_files"] += 1
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            all_records.extend(_records_from_generic_json(payload, source_ref))
        elif suffix == ".csv":
            stats["csv_files"] += 1
            frame = pd.read_csv(file_path)
            all_records.extend(_table_to_records(frame, source_ref))
        elif suffix == ".parquet":
            stats["parquet_files"] += 1
            frame = pd.read_parquet(file_path)
            all_records.extend(_table_to_records(frame, source_ref))
    return all_records, stats


def _fetch_remote_payload(cik: str, *, user_agent: str, timeout_sec: int) -> dict[str, Any]:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    req = urlrequest.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        },
        method="GET",
    )
    with urlrequest.urlopen(req, timeout=timeout_sec) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Remote SEC companyfacts payload is not a JSON object.")
    return payload


def _fetch_remote_records(
    target_ciks: list[str],
    *,
    user_agent: str,
    timeout_sec: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    success = 0
    for cik in target_ciks:
        try:
            payload = _fetch_remote_payload(
                cik,
                user_agent=user_agent,
                timeout_sec=timeout_sec,
            )
            rows = _records_from_sec_companyfacts_payload(payload, f"sec_companyfacts://{cik}")
            records.extend(rows)
            success += 1
        except (urlerror.URLError, urlerror.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            failures.append({"cik": cik, "error": str(exc)})
    stats = {
        "requested_ciks": int(len(target_ciks)),
        "successful_ciks": int(success),
        "failed_ciks": int(len(failures)),
        "failures": failures,
    }
    return records, stats


def _load_submissions_lookup(
    submissions_raw_path: Path | None,
) -> tuple[dict[str, pd.Timestamp], int]:
    if submissions_raw_path is None or not submissions_raw_path.exists():
        return {}, 0

    submissions = read_parquet(submissions_raw_path)
    required = {"cik", "accession_number", "acceptance_ts"}
    if not required.issubset(submissions.columns):
        return {}, 0

    frame = submissions[list(required)].copy()
    frame["cik"] = frame["cik"].map(_normalize_cik)
    frame["accession_number"] = _normalize_string(frame["accession_number"])
    frame["acceptance_ts"] = pd.to_datetime(frame["acceptance_ts"], utc=True, errors="coerce")
    frame = frame[
        frame["cik"].notna()
        & (frame["accession_number"] != "")
        & frame["acceptance_ts"].notna()
    ].copy()
    if frame.empty:
        return {}, 0

    frame.sort_values(["cik", "accession_number", "acceptance_ts"], inplace=True)
    frame = frame.drop_duplicates(["cik", "accession_number"], keep="last")
    lookup = {
        f"{row.cik}|{row.accession_number}": row.acceptance_ts
        for row in frame.itertuples(index=False)
    }
    return lookup, int(len(frame))


def _normalize_records_to_frame(
    records: list[dict[str, Any]],
    *,
    target_ciks: set[str],
    submissions_lookup: dict[str, pd.Timestamp],
) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {
        "input_rows": int(len(records)),
        "discarded_missing_critical": 0,
        "discarded_unknown_cik": 0,
        "discarded_temporal_inconsistency": 0,
        "duplicate_rows_dropped": 0,
        "duplicate_conflict_groups": 0,
        "acceptance_filled_from_submissions": 0,
        "acceptance_filled_from_filing_date": 0,
    }
    if not records:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS)), stats

    frame = pd.DataFrame(records)
    frame.columns = [_normalize_col_name(col) for col in frame.columns]
    available = set(frame.columns)

    canonical: dict[str, pd.Series] = {}
    for name, aliases in ALIAS_CANDIDATES.items():
        col = _pick_col(available, aliases)
        canonical[name] = frame[col] if col is not None else pd.Series([None] * len(frame))

    if "source_ref" in frame.columns:
        canonical["source_ref"] = frame["source_ref"]
    else:
        canonical["source_ref"] = pd.Series([""] * len(frame))

    out = pd.DataFrame(canonical)
    out["cik"] = out["cik"].map(_normalize_cik)
    out["taxonomy"] = _normalize_string(out["taxonomy"])
    out["tag"] = _normalize_string(out["tag"])
    out["unit"] = _normalize_string(out["unit"])
    out["fiscal_period"] = _normalize_string(out["fiscal_period"], uppercase=True)
    out["form_type"] = _normalize_string(out["form_type"], uppercase=True)
    out["frame"] = _normalize_string(out["frame"])
    out["accession_number"] = _normalize_string(out["accession_number"])
    out["source_ref"] = _normalize_string(out["source_ref"])

    out["fact_value"] = pd.to_numeric(out["fact_value"], errors="coerce")
    out["fiscal_year"] = pd.to_numeric(out["fiscal_year"], errors="coerce").astype("Int64")

    out["fact_start_date"] = pd.to_datetime(out["fact_start_date"], errors="coerce").dt.normalize()
    out["fact_end_date"] = pd.to_datetime(out["fact_end_date"], errors="coerce").dt.normalize()
    out["filing_date"] = pd.to_datetime(out["filing_date"], errors="coerce").dt.normalize()
    out["acceptance_ts"] = pd.to_datetime(out["acceptance_ts"], utc=True, errors="coerce")

    if submissions_lookup:
        composite = out["cik"].fillna("").astype(str) + "|" + out["accession_number"].astype(str)
        from_submissions = composite.map(submissions_lookup)
        fill_mask = out["acceptance_ts"].isna() & from_submissions.notna()
        out.loc[fill_mask, "acceptance_ts"] = from_submissions[fill_mask]
        stats["acceptance_filled_from_submissions"] = int(fill_mask.sum())

    fill_from_filing = out["acceptance_ts"].isna() & out["filing_date"].notna()
    if fill_from_filing.any():
        out.loc[fill_from_filing, "acceptance_ts"] = pd.to_datetime(
            out.loc[fill_from_filing, "filing_date"],
            utc=True,
            errors="coerce",
        )
        stats["acceptance_filled_from_filing_date"] = int(fill_from_filing.sum())

    missing_critical = (
        out["cik"].isna()
        | (out["taxonomy"] == "")
        | (out["tag"] == "")
        | (out["unit"] == "")
        | out["fact_value"].isna()
        | out["filing_date"].isna()
        | out["acceptance_ts"].isna()
    )
    if missing_critical.any():
        stats["discarded_missing_critical"] = int(missing_critical.sum())
    out = out[~missing_critical].copy()
    if out.empty:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS)), stats

    unknown_cik = ~out["cik"].isin(target_ciks)
    if unknown_cik.any():
        stats["discarded_unknown_cik"] = int(unknown_cik.sum())
    out = out[~unknown_cik].copy()
    if out.empty:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS)), stats

    acceptance_date = (
        out["acceptance_ts"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    )
    temporal_invalid = (
        (
            out["fact_start_date"].notna()
            & out["fact_end_date"].notna()
            & (out["fact_start_date"] > out["fact_end_date"])
        )
        | (out["fact_end_date"].notna() & (out["fact_end_date"] > out["filing_date"]))
        | (acceptance_date < out["filing_date"])
    )
    if temporal_invalid.any():
        stats["discarded_temporal_inconsistency"] = int(temporal_invalid.sum())
    out = out[~temporal_invalid].copy()
    if out.empty:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS)), stats

    dedup_key = [
        "cik",
        "taxonomy",
        "tag",
        "unit",
        "filing_date",
        "fact_end_date",
        "frame",
        "accession_number",
        "source_ref",
    ]
    out.sort_values(
        dedup_key + ["acceptance_ts", "fact_value"],
        inplace=True,
    )
    dup_mask = out.duplicated(dedup_key, keep=False)
    if dup_mask.any():
        grouped = out.loc[dup_mask].groupby(dedup_key, dropna=False, as_index=False)
        conflict_groups = 0
        for _, grp in grouped:
            if grp["fact_value"].nunique(dropna=False) > 1:
                conflict_groups += 1
        stats["duplicate_conflict_groups"] = int(conflict_groups)
    before_dedup = len(out)
    out = out.drop_duplicates(dedup_key, keep="last").copy()
    stats["duplicate_rows_dropped"] = int(before_dedup - len(out))

    return out.reset_index(drop=True), stats


def _build_config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _validate_required_output_columns(frame: pd.DataFrame) -> None:
    missing = [col for col in CANONICAL_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"companyfacts_raw output missing required columns: {missing}")
    if frame.empty:
        raise ValueError("companyfacts_raw output is empty.")

    critical = ["cik", "taxonomy", "tag", "unit", "fact_value", "filing_date", "acceptance_ts"]
    if frame[critical].isna().any().any():
        raise ValueError("companyfacts_raw output has null values in critical columns.")

    duplicates = frame.duplicated(
        [
            "cik",
            "taxonomy",
            "tag",
            "unit",
            "filing_date",
            "fact_end_date",
            "frame",
            "accession_number",
            "source_ref",
        ],
        keep=False,
    )
    if duplicates.any():
        raise ValueError("companyfacts_raw output has duplicate logical rows.")


def fetch_companyfacts(
    *,
    ticker_cik_map_path: str | Path | None = None,
    universe_history_path: str | Path | None = None,
    submissions_raw_path: str | Path | None = None,
    local_source_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    ingestion_mode: str = "auto",
    include_universe_filter: bool = True,
    allow_remote: bool = False,
    force_rebuild: bool = False,
    reuse_cache: bool = True,
    remote_timeout_sec: int = DEFAULT_REMOTE_TIMEOUT_SEC,
    user_agent: str = DEFAULT_USER_AGENT,
    run_id: str = "fetch_companyfacts_mvp_v1",
) -> FetchCompanyFactsResult:
    logger = get_logger("data.edgar.fetch_companyfacts")
    if ingestion_mode not in ALLOWED_INGESTION_MODES:
        raise ValueError(
            f"Unsupported ingestion_mode '{ingestion_mode}'. "
            f"Allowed: {sorted(ALLOWED_INGESTION_MODES)}"
        )

    edgar_base = data_dir() / "edgar"
    ticker_source = (
        Path(ticker_cik_map_path).expanduser().resolve()
        if ticker_cik_map_path
        else (edgar_base / "ticker_cik_map.parquet")
    )
    universe_source = (
        Path(universe_history_path).expanduser().resolve()
        if universe_history_path
        else (data_dir() / "universe" / "universe_history.parquet")
    )
    submissions_source = (
        Path(submissions_raw_path).expanduser().resolve()
        if submissions_raw_path
        else (edgar_base / "submissions_raw.parquet")
    )
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else edgar_base
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_output_path = out_dir / "companyfacts_raw.parquet"
    report_path = out_dir / "companyfacts_ingestion_report.json"

    ticker_cik_map = read_parquet(ticker_source)
    universe_history = read_parquet(universe_source) if include_universe_filter else None
    target_ciks = _resolve_target_ciks(
        ticker_cik_map=ticker_cik_map,
        universe_history=universe_history,
    )

    config_hash = _build_config_hash(
        {
            "version": "fetch_companyfacts_mvp_v1",
            "ingestion_mode": ingestion_mode,
            "include_universe_filter": include_universe_filter,
            "allow_remote": allow_remote,
            "remote_timeout_sec": remote_timeout_sec,
            "user_agent": user_agent,
            "target_ciks": target_ciks,
            "ticker_cik_map_path": str(ticker_source),
            "universe_history_path": str(universe_source),
            "submissions_raw_path": str(submissions_source),
            "local_source_path": str(local_source_path) if local_source_path else "",
        }
    )

    if raw_output_path.exists() and reuse_cache and not force_rebuild:
        cached = read_parquet(raw_output_path)
        _validate_required_output_columns(cached)
        resolved_mode = "cache_reuse"
        report_payload = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "config_hash": config_hash,
            "requested_mode": ingestion_mode,
            "resolved_mode": resolved_mode,
            "cache_status": resolved_mode,
            "reused_cache": True,
            "force_rebuild": force_rebuild,
            "ciks_requested": int(len(target_ciks)),
            "ciks_covered": int(cached["cik"].nunique()),
            "n_raw_records": int(len(cached)),
            "n_records_output": int(len(cached)),
            "n_records_dropped": 0,
            "source_local_path": str(local_source_path)
            if local_source_path
            else str(edgar_base / "source" / "companyfacts"),
            "output_path": str(raw_output_path),
        }
        report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")
        return FetchCompanyFactsResult(
            companyfacts_raw_path=raw_output_path,
            ingestion_report_path=report_path,
            row_count=int(len(cached)),
            ciks_requested=int(len(target_ciks)),
            ciks_covered=int(cached["cik"].nunique()),
            source_mode=resolved_mode,
            config_hash=config_hash,
            reused_cache=True,
        )

    local_base = (
        Path(local_source_path).expanduser().resolve()
        if local_source_path
        else (edgar_base / "source" / "companyfacts")
    )
    local_files = _discover_local_files(local_base)
    local_records: list[dict[str, Any]] = []
    local_scan_stats: dict[str, int] = {
        "files_scanned": 0,
        "json_files": 0,
        "csv_files": 0,
        "parquet_files": 0,
    }
    if local_files:
        local_records, local_scan_stats = _load_local_records(local_files)

    resolved_mode = ingestion_mode
    remote_stats: dict[str, Any] = {
        "requested_ciks": 0,
        "successful_ciks": 0,
        "failed_ciks": 0,
        "failures": [],
    }
    raw_records: list[dict[str, Any]] = []

    if ingestion_mode == "local_file":
        if not local_files:
            raise FileNotFoundError(
                f"local_file mode requested but no source files found at: {local_base}"
            )
        raw_records = local_records
        resolved_mode = "local_file"
    elif ingestion_mode == "remote_optional":
        if not allow_remote:
            raise ValueError(
                "remote_optional mode requested but allow_remote=False. "
                "Enable --allow-remote or use local_file mode."
            )
        raw_records, remote_stats = _fetch_remote_records(
            target_ciks,
            user_agent=user_agent,
            timeout_sec=remote_timeout_sec,
        )
        resolved_mode = "remote_optional"
    else:  # auto
        if local_files:
            raw_records = local_records
            resolved_mode = "local_file"
        else:
            if allow_remote:
                raw_records, remote_stats = _fetch_remote_records(
                    target_ciks,
                    user_agent=user_agent,
                    timeout_sec=remote_timeout_sec,
                )
                resolved_mode = "remote_optional"
            else:
                raise FileNotFoundError(
                    "auto mode could not find local companyfacts sources and remote is disabled. "
                    "Provide --local-source-path or enable --allow-remote."
                )

    submissions_lookup, submissions_lookup_rows = _load_submissions_lookup(submissions_source)
    normalized, normalize_stats = _normalize_records_to_frame(
        raw_records,
        target_ciks=set(target_ciks),
        submissions_lookup=submissions_lookup,
    )
    if normalized.empty:
        raise ValueError(
            "No valid companyfacts rows after normalization/filtering. "
            "Check source files, target CIK coverage, and critical fields."
        )

    built_ts_utc = datetime.now(UTC).isoformat()
    normalized["source_mode"] = resolved_mode
    normalized["source_ref"] = normalized["source_ref"].astype(str)
    normalized["run_id"] = run_id
    normalized["config_hash"] = config_hash
    normalized["built_ts_utc"] = built_ts_utc
    normalized = normalized[list(CANONICAL_COLUMNS)].sort_values(
        ["cik", "taxonomy", "tag", "filing_date", "fact_end_date", "acceptance_ts"],
    ).reset_index(drop=True)

    _validate_required_output_columns(normalized)
    output_path = write_parquet(
        normalized,
        raw_output_path,
        schema_name="edgar_companyfacts_raw_mvp",
        run_id=run_id,
    )

    discarded_total = int(
        normalize_stats["discarded_missing_critical"]
        + normalize_stats["discarded_unknown_cik"]
        + normalize_stats["discarded_temporal_inconsistency"]
        + normalize_stats["duplicate_rows_dropped"]
    )

    report_payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "config_hash": config_hash,
        "requested_mode": ingestion_mode,
        "resolved_mode": resolved_mode,
        "cache_status": "rebuilt",
        "reused_cache": False,
        "force_rebuild": force_rebuild,
        "ciks_requested": int(len(target_ciks)),
        "ciks_covered": int(normalized["cik"].nunique()),
        "n_raw_records": int(normalize_stats["input_rows"]),
        "n_records_output": int(len(normalized)),
        "n_records_dropped": discarded_total,
        "discarded_missing_critical": int(normalize_stats["discarded_missing_critical"]),
        "discarded_unknown_cik": int(normalize_stats["discarded_unknown_cik"]),
        "discarded_temporal_inconsistency": int(normalize_stats["discarded_temporal_inconsistency"]),
        "duplicate_rows_dropped": int(normalize_stats["duplicate_rows_dropped"]),
        "duplicate_conflict_groups": int(normalize_stats["duplicate_conflict_groups"]),
        "acceptance_filled_from_submissions": int(normalize_stats["acceptance_filled_from_submissions"]),
        "acceptance_filled_from_filing_date": int(normalize_stats["acceptance_filled_from_filing_date"]),
        "submissions_lookup_rows": int(submissions_lookup_rows),
        "local_scan_stats": local_scan_stats,
        "local_files": [str(path) for path in local_files],
        "remote_stats": remote_stats,
        "ticker_cik_map_path": str(ticker_source),
        "universe_history_path": str(universe_source) if include_universe_filter else "",
        "submissions_raw_path": str(submissions_source),
        "output_path": str(output_path),
    }
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "companyfacts_raw_built",
        run_id=run_id,
        requested_mode=ingestion_mode,
        resolved_mode=resolved_mode,
        ciks_requested=int(len(target_ciks)),
        ciks_covered=int(normalized["cik"].nunique()),
        row_count=int(len(normalized)),
        discarded_rows=discarded_total,
        output_path=str(output_path),
    )

    return FetchCompanyFactsResult(
        companyfacts_raw_path=output_path,
        ingestion_report_path=report_path,
        row_count=int(len(normalized)),
        ciks_requested=int(len(target_ciks)),
        ciks_covered=int(normalized["cik"].nunique()),
        source_mode=resolved_mode,
        config_hash=config_hash,
        reused_cache=False,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch EDGAR companyfacts raw (MVP local-first, cacheable)."
    )
    parser.add_argument("--ticker-cik-map-path", type=str, default=None)
    parser.add_argument("--universe-history-path", type=str, default=None)
    parser.add_argument("--submissions-raw-path", type=str, default=None)
    parser.add_argument("--local-source-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--ingestion-mode",
        type=str,
        default="auto",
        choices=sorted(ALLOWED_INGESTION_MODES),
    )
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--disable-cache-reuse", action="store_true")
    parser.add_argument("--remote-timeout-sec", type=int, default=DEFAULT_REMOTE_TIMEOUT_SEC)
    parser.add_argument("--user-agent", type=str, default=DEFAULT_USER_AGENT)
    parser.add_argument("--run-id", type=str, default="fetch_companyfacts_mvp_v1")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = fetch_companyfacts(
        ticker_cik_map_path=args.ticker_cik_map_path,
        universe_history_path=args.universe_history_path,
        submissions_raw_path=args.submissions_raw_path,
        local_source_path=args.local_source_path,
        output_dir=args.output_dir,
        ingestion_mode=args.ingestion_mode,
        allow_remote=args.allow_remote,
        force_rebuild=args.force_rebuild,
        reuse_cache=not args.disable_cache_reuse,
        remote_timeout_sec=args.remote_timeout_sec,
        user_agent=args.user_agent,
        run_id=args.run_id,
    )
    print("Companyfacts raw built:")
    print(f"- path: {result.companyfacts_raw_path}")
    print(f"- report: {result.ingestion_report_path}")
    print(f"- rows: {result.row_count}")
    print(f"- ciks_requested: {result.ciks_requested}")
    print(f"- ciks_covered: {result.ciks_covered}")
    print(f"- source_mode: {result.source_mode}")
    print(f"- reused_cache: {result.reused_cache}")


if __name__ == "__main__":
    main()
