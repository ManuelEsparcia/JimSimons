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

# Allow direct script execution: `python simons_smallcap_swing/data/edgar/fetch_submissions.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger

ALLOWED_INGESTION_MODES: tuple[str, ...] = ("auto", "local_file", "remote_optional")
DEFAULT_REMOTE_TIMEOUT_SEC = 12
DEFAULT_USER_AGENT = (
    "JimSimons-Research-MVP/1.0 "
    "(contact: local-test@example.com)"
)

CANONICAL_COLUMNS: tuple[str, ...] = (
    "cik",
    "accession_number",
    "filing_date",
    "acceptance_ts",
    "form_type",
    "filing_primary_document",
    "filing_primary_doc_description",
    "report_date",
    "is_xbrl",
    "is_inline_xbrl",
    "source_mode",
    "source_ref",
    "run_id",
    "config_hash",
    "built_ts_utc",
)

ALIAS_CANDIDATES: dict[str, tuple[str, ...]] = {
    "cik": ("cik", "cik_str", "ciknumber"),
    "accession_number": (
        "accession_number",
        "accessionnumber",
        "accession_no",
        "accessionnumber",
    ),
    "filing_date": ("filing_date", "filingdate"),
    "acceptance_ts": (
        "acceptance_ts",
        "acceptance_datetime",
        "acceptancedatetime",
        "acceptance_date_time",
    ),
    "form_type": ("form_type", "form", "formtype"),
    "filing_primary_document": (
        "filing_primary_document",
        "primary_document",
        "primarydocument",
    ),
    "filing_primary_doc_description": (
        "filing_primary_doc_description",
        "primary_doc_description",
        "primarydocdescription",
        "filing_primary_doc_desc",
    ),
    "report_date": ("report_date", "reportdate"),
    "is_xbrl": ("is_xbrl", "isxbrl"),
    "is_inline_xbrl": ("is_inline_xbrl", "isinlinexbrl"),
}


@dataclass(frozen=True)
class FetchSubmissionsResult:
    submissions_raw_path: Path
    ingestion_report_path: Path
    row_count: int
    ciks_requested: int
    ciks_covered: int
    source_mode: str
    config_hash: str
    reused_cache: bool


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains invalid dates.")
    return parsed.dt.normalize()


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


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", ""}:
        return False
    try:
        return bool(int(float(text)))
    except Exception:
        return False


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


def _records_from_sec_recent_payload(payload: dict[str, Any], source_ref: str) -> list[dict[str, Any]]:
    filings = payload.get("filings", {})
    recent = filings.get("recent", {}) if isinstance(filings, dict) else {}
    if not isinstance(recent, dict) or not recent:
        return []

    candidate_keys = (
        "accessionNumber",
        "filingDate",
        "acceptanceDateTime",
        "form",
    )
    lengths = []
    for key in candidate_keys:
        values = recent.get(key)
        if isinstance(values, list):
            lengths.append(len(values))
    if not lengths:
        return []
    n_rows = max(lengths)
    cik_value = _normalize_cik(payload.get("cik") or payload.get("cik_str"))

    rows: list[dict[str, Any]] = []
    for idx in range(n_rows):
        row: dict[str, Any] = {
            "cik": cik_value,
            "accession_number": recent.get("accessionNumber", [None] * n_rows)[idx]
            if idx < len(recent.get("accessionNumber", []))
            else None,
            "filing_date": recent.get("filingDate", [None] * n_rows)[idx]
            if idx < len(recent.get("filingDate", []))
            else None,
            "acceptance_ts": recent.get("acceptanceDateTime", [None] * n_rows)[idx]
            if idx < len(recent.get("acceptanceDateTime", []))
            else None,
            "form_type": recent.get("form", [None] * n_rows)[idx]
            if idx < len(recent.get("form", []))
            else None,
            "filing_primary_document": recent.get("primaryDocument", [None] * n_rows)[idx]
            if idx < len(recent.get("primaryDocument", []))
            else None,
            "filing_primary_doc_description": recent.get("primaryDocDescription", [None] * n_rows)[idx]
            if idx < len(recent.get("primaryDocDescription", []))
            else None,
            "report_date": recent.get("reportDate", [None] * n_rows)[idx]
            if idx < len(recent.get("reportDate", []))
            else None,
            "is_xbrl": recent.get("isXBRL", [None] * n_rows)[idx]
            if idx < len(recent.get("isXBRL", []))
            else None,
            "is_inline_xbrl": recent.get("isInlineXBRL", [None] * n_rows)[idx]
            if idx < len(recent.get("isInlineXBRL", []))
            else None,
            "source_ref": source_ref,
        }
        rows.append(row)
    return rows


def _records_from_generic_json(payload: Any, source_ref: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        sec_rows = _records_from_sec_recent_payload(payload, source_ref)
        if sec_rows:
            return sec_rows

        if "records" in payload and isinstance(payload["records"], list):
            return [dict(item) for item in payload["records"] if isinstance(item, dict)]

        # Single dict record shape.
        if any(key in payload for key in ("accession_number", "accessionNumber", "form", "form_type")):
            return [dict(payload)]

        return []

    if isinstance(payload, list):
        rows: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict):
                rows.append(dict(item))
        return rows

    return []


def _table_to_records(table: pd.DataFrame, source_ref: str) -> list[dict[str, Any]]:
    if table.empty:
        return []
    normalized = table.copy()
    normalized.columns = [_normalize_col_name(col) for col in normalized.columns]
    rows = normalized.to_dict(orient="records")
    for row in rows:
        row["source_ref"] = source_ref
    return rows


def _normalize_records_to_frame(
    records: list[dict[str, Any]],
    *,
    target_ciks: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {
        "input_rows": int(len(records)),
        "discarded_missing_critical": 0,
        "discarded_invalid_dates": 0,
        "discarded_temporal_inconsistency": 0,
        "discarded_unknown_cik": 0,
        "duplicate_rows_dropped": 0,
        "duplicate_conflict_groups": 0,
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
    out["accession_number"] = out["accession_number"].astype(str).str.strip()
    out["form_type"] = out["form_type"].astype(str).str.strip().str.upper()
    out["filing_primary_document"] = (
        out["filing_primary_document"].astype(str).str.strip().replace({"None": ""})
    )
    out["filing_primary_doc_description"] = (
        out["filing_primary_doc_description"].astype(str).str.strip().replace({"None": ""})
    )

    out["filing_date"] = pd.to_datetime(out["filing_date"], errors="coerce").dt.normalize()
    out["acceptance_ts"] = pd.to_datetime(out["acceptance_ts"], utc=True, errors="coerce")
    out["report_date"] = pd.to_datetime(out["report_date"], errors="coerce").dt.normalize()
    out["is_xbrl"] = out["is_xbrl"].map(_to_bool)
    out["is_inline_xbrl"] = out["is_inline_xbrl"].map(_to_bool)

    missing_critical = (
        out["cik"].isna()
        | out["accession_number"].isin({"", "NONE", "NAN", "NULL", "none", "nan", "null"})
        | out["filing_date"].isna()
        | out["acceptance_ts"].isna()
        | out["form_type"].isin({"", "NONE", "NAN", "NULL", "none", "nan", "null"})
    )
    if missing_critical.any():
        stats["discarded_missing_critical"] = int(missing_critical.sum())
    out = out[~missing_critical].copy()

    if out.empty:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS)), stats

    invalid_cik = ~out["cik"].isin(target_ciks)
    if invalid_cik.any():
        stats["discarded_unknown_cik"] = int(invalid_cik.sum())
    out = out[~invalid_cik].copy()
    if out.empty:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS)), stats

    acceptance_date = (
        out["acceptance_ts"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    )
    temporal_invalid = out["filing_date"] > acceptance_date
    if temporal_invalid.any():
        stats["discarded_temporal_inconsistency"] = int(temporal_invalid.sum())
    out = out[~temporal_invalid].copy()
    if out.empty:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS)), stats

    # Deduplicate by (cik, accession_number), keeping the latest acceptance record.
    out.sort_values(
        ["cik", "accession_number", "acceptance_ts", "filing_date", "form_type"],
        inplace=True,
    )
    dup_mask = out.duplicated(["cik", "accession_number"], keep=False)
    if dup_mask.any():
        grouped = out.loc[dup_mask].groupby(["cik", "accession_number"], as_index=False)
        conflict_groups = 0
        for _, grp in grouped:
            key_cols = [
                "filing_date",
                "acceptance_ts",
                "form_type",
                "filing_primary_document",
                "filing_primary_doc_description",
                "report_date",
                "is_xbrl",
                "is_inline_xbrl",
            ]
            if any(grp[col].nunique(dropna=False) > 1 for col in key_cols):
                conflict_groups += 1
        stats["duplicate_conflict_groups"] = int(conflict_groups)
    before_dedup = len(out)
    out = out.drop_duplicates(["cik", "accession_number"], keep="last").copy()
    stats["duplicate_rows_dropped"] = int(before_dedup - len(out))

    return out.reset_index(drop=True), stats


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
            rows = _records_from_generic_json(payload, source_ref)
            all_records.extend(rows)
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
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
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
        raise ValueError("Remote SEC submissions payload is not a JSON object.")
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
            rows = _records_from_sec_recent_payload(payload, f"sec_submissions://{cik}")
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


def _build_config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _validate_required_output_columns(frame: pd.DataFrame) -> None:
    missing = [col for col in CANONICAL_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"submissions_raw output missing required columns: {missing}")
    if frame.empty:
        raise ValueError("submissions_raw output is empty.")
    if frame[["cik", "accession_number", "filing_date", "acceptance_ts", "form_type"]].isna().any().any():
        raise ValueError("submissions_raw output has null values in critical columns.")
    dup = frame.duplicated(["cik", "accession_number"], keep=False)
    if dup.any():
        raise ValueError("submissions_raw output has duplicate (cik, accession_number).")


def fetch_submissions(
    *,
    ticker_cik_map_path: str | Path | None = None,
    universe_history_path: str | Path | None = None,
    local_source_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    ingestion_mode: str = "auto",
    include_universe_filter: bool = True,
    allow_remote: bool = False,
    force_rebuild: bool = False,
    reuse_cache: bool = True,
    remote_timeout_sec: int = DEFAULT_REMOTE_TIMEOUT_SEC,
    user_agent: str = DEFAULT_USER_AGENT,
    run_id: str = "fetch_submissions_mvp_v1",
) -> FetchSubmissionsResult:
    logger = get_logger("data.edgar.fetch_submissions")

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
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else edgar_base
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_output_path = out_dir / "submissions_raw.parquet"
    report_path = out_dir / "submissions_ingestion_report.json"

    ticker_cik_map = read_parquet(ticker_source)
    universe_history = read_parquet(universe_source) if include_universe_filter else None
    target_ciks = _resolve_target_ciks(
        ticker_cik_map=ticker_cik_map,
        universe_history=universe_history,
    )

    config_hash = _build_config_hash(
        {
            "version": "fetch_submissions_mvp_v1",
            "ingestion_mode": ingestion_mode,
            "include_universe_filter": include_universe_filter,
            "allow_remote": allow_remote,
            "remote_timeout_sec": remote_timeout_sec,
            "user_agent": user_agent,
            "target_ciks": target_ciks,
            "ticker_cik_map_path": str(ticker_source),
            "universe_history_path": str(universe_source),
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
            "reused_cache": True,
            "force_rebuild": force_rebuild,
            "ciks_requested": int(len(target_ciks)),
            "ciks_covered": int(cached["cik"].nunique()),
            "rows_output": int(len(cached)),
            "source_local_path": str(local_source_path) if local_source_path else str(edgar_base / "source" / "submissions"),
            "output_path": str(raw_output_path),
        }
        report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")
        return FetchSubmissionsResult(
            submissions_raw_path=raw_output_path,
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
        else (edgar_base / "source" / "submissions")
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
                    "auto mode could not find local submissions sources and remote is disabled. "
                    "Provide --local-source-path or enable --allow-remote."
                )

    normalized, normalize_stats = _normalize_records_to_frame(
        raw_records,
        target_ciks=set(target_ciks),
    )
    if normalized.empty:
        raise ValueError(
            "No valid submissions rows after normalization/filtering. "
            "Check source files, target CIK coverage, and critical fields."
        )

    built_ts_utc = datetime.now(UTC).isoformat()
    normalized["source_mode"] = resolved_mode
    normalized["source_ref"] = normalized["source_ref"].astype(str)
    normalized["run_id"] = run_id
    normalized["config_hash"] = config_hash
    normalized["built_ts_utc"] = built_ts_utc
    normalized = normalized[
        [
            "cik",
            "accession_number",
            "filing_date",
            "acceptance_ts",
            "form_type",
            "filing_primary_document",
            "filing_primary_doc_description",
            "report_date",
            "is_xbrl",
            "is_inline_xbrl",
            "source_mode",
            "source_ref",
            "run_id",
            "config_hash",
            "built_ts_utc",
        ]
    ].sort_values(["cik", "filing_date", "acceptance_ts", "accession_number"]).reset_index(drop=True)

    _validate_required_output_columns(normalized)

    submissions_path = write_parquet(
        normalized,
        raw_output_path,
        schema_name="edgar_submissions_raw_mvp",
        run_id=run_id,
    )

    discarded_total = int(
        normalize_stats["discarded_missing_critical"]
        + normalize_stats["discarded_invalid_dates"]
        + normalize_stats["discarded_temporal_inconsistency"]
        + normalize_stats["discarded_unknown_cik"]
        + normalize_stats["duplicate_rows_dropped"]
    )

    report_payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "config_hash": config_hash,
        "requested_mode": ingestion_mode,
        "resolved_mode": resolved_mode,
        "reused_cache": False,
        "force_rebuild": force_rebuild,
        "ciks_requested": int(len(target_ciks)),
        "ciks_covered": int(normalized["cik"].nunique()),
        "rows_output": int(len(normalized)),
        "rows_input_raw": int(normalize_stats["input_rows"]),
        "rows_discarded_total": discarded_total,
        "discarded_missing_critical": int(normalize_stats["discarded_missing_critical"]),
        "discarded_invalid_dates": int(normalize_stats["discarded_invalid_dates"]),
        "discarded_temporal_inconsistency": int(normalize_stats["discarded_temporal_inconsistency"]),
        "discarded_unknown_cik": int(normalize_stats["discarded_unknown_cik"]),
        "duplicate_rows_dropped": int(normalize_stats["duplicate_rows_dropped"]),
        "duplicate_conflict_groups": int(normalize_stats["duplicate_conflict_groups"]),
        "local_scan_stats": local_scan_stats,
        "local_files": [str(path) for path in local_files],
        "remote_stats": remote_stats,
        "ticker_cik_map_path": str(ticker_source),
        "universe_history_path": str(universe_source) if include_universe_filter else "",
        "output_path": str(submissions_path),
    }
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "submissions_raw_built",
        run_id=run_id,
        requested_mode=ingestion_mode,
        resolved_mode=resolved_mode,
        ciks_requested=int(len(target_ciks)),
        ciks_covered=int(normalized["cik"].nunique()),
        row_count=int(len(normalized)),
        discarded_rows=discarded_total,
        output_path=str(submissions_path),
    )

    return FetchSubmissionsResult(
        submissions_raw_path=submissions_path,
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
        description="Fetch EDGAR submissions metadata (MVP local-first, cacheable)."
    )
    parser.add_argument("--ticker-cik-map-path", type=str, default=None)
    parser.add_argument("--universe-history-path", type=str, default=None)
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
    parser.add_argument("--run-id", type=str, default="fetch_submissions_mvp_v1")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = fetch_submissions(
        ticker_cik_map_path=args.ticker_cik_map_path,
        universe_history_path=args.universe_history_path,
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
    print("Submissions raw built:")
    print(f"- path: {result.submissions_raw_path}")
    print(f"- report: {result.ingestion_report_path}")
    print(f"- rows: {result.row_count}")
    print(f"- ciks_requested: {result.ciks_requested}")
    print(f"- ciks_covered: {result.ciks_covered}")
    print(f"- source_mode: {result.source_mode}")
    print(f"- reused_cache: {result.reused_cache}")


if __name__ == "__main__":
    main()
