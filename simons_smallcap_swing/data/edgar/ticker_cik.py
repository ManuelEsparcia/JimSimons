from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "ticker_cik_config_version": "1.0.0",
    "storage": {
        "output_root": "data/edgar/mappings",
        "allow_csv_fallback": True,
        "compression": "snappy",
    },
    "identity": {
        "default_effective_from": "1900-01-01",
        "input_effective_to_is_inclusive": False,
        "allow_empty_internal_master": True,
        "allow_heuristic_source": True,
        "open_interval_end": None,
        "strict_abort_on_schema_failure": True,
        "strict_abort_on_unresolved_conflict": False,
        "prefer_temporal_continuity": True,
        "allow_multi_symbol_per_cik_when_share_class_distinct": True,
        "share_class_blank_value": "COMMON",
        "max_conflict_current_overlap_days": 0,
    },
    "sources": {
        "priority": {
            "official_SEC_mapping": 100,
            "trusted_internal_master": 80,
            "heuristic_inferred_mapping": 40,
        },
        "base_confidence": {
            "official_SEC_mapping": 0.99,
            "trusted_internal_master": 0.90,
            "heuristic_inferred_mapping": 0.60,
        },
    },
    "resolution": {
        "min_confidence_for_current": 0.50,
        "current_requires_non_heuristic": False,
        "tie_tolerance": 1e-12,
        "metadata_weights": {
            "issuer_name": 0.35,
            "exchange": 0.15,
            "share_class": 0.15,
            "cross_source_corroboration": 0.35,
        },
        "status_by_source": {
            "official_SEC_mapping": "exact",
            "trusted_internal_master": "preferred",
            "heuristic_inferred_mapping": "heuristic",
        },
    },
    "pit": {
        "default_asof_time": "23:59:59",
        "timezone": "UTC",
    },
}

COMMON_ALIASES: Dict[str, List[str]] = {
    "symbol": ["symbol", "ticker", "tickers", "trading_symbol", "display_symbol", "canonical_trading_symbol"],
    "cik": ["cik", "cik_str", "issuer_cik", "sec_cik"],
    "issuer_name": ["issuer_name", "company_name", "title", "name", "issuer"],
    "exchange": ["exchange", "primary_exchange", "listing_exchange"],
    "share_class": ["share_class", "class", "security_class", "class_code"],
    "effective_from": ["effective_from", "valid_from", "start_date", "effective_date"],
    "effective_to": ["effective_to", "valid_to", "end_date", "termination_date"],
    "confidence_score": ["confidence_score", "confidence", "score"],
    "source": ["source", "source_name"],
    "resolution_status": ["resolution_status", "status"],
    "heuristic_flag": ["heuristic_flag", "is_heuristic", "heuristic"],
    "is_active": ["is_active", "active_flag", "active"],
    "mapping_version": ["mapping_version", "version"],
    "ingest_ts_utc": ["ingest_ts_utc", "ingest_ts", "loaded_ts", "timestamp"],
}

CONFLICT_PRECEDENCE = {
    "schema_or_config_failure": 100,
    "unresolved_identity_conflict": 90,
    "temporal_overlap_conflict": 80,
    "coverage_gap": 50,
    "row_level_normalization_issue": 10,
}


@dataclass(frozen=True)
class SegmentDecision:
    symbol: str
    cik: str
    source: str
    source_priority: int
    confidence_score: float
    metadata_consistency: float
    temporal_consistency: int
    effective_from: pd.Timestamp
    effective_to: pd.Timestamp | pd.NaT
    resolution_status: str
    heuristic_flag: bool
    issuer_name: str
    exchange: str
    share_class: str
    evidence_hash: str
    mapping_version: str
    lineage_count: int
    corroboration_count: int


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_convert(None)


def _deep_merge(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(a)
    for k, v in b.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def _config_hash(cfg: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if not path:
        return cfg
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config path not found: {path}")
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML config files.")
        with p.open("r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
    elif p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            user_cfg = json.load(f)
    else:
        raise ValueError("Config path must be YAML/YML or JSON.")
    return _deep_merge(cfg, user_cfg)


def _to_cik(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = "".join(ch for ch in str(x).strip() if ch.isdigit())
    return s.zfill(10) if s else ""


def _to_symbol(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def _coerce_bool(x: Any, default: bool = True) -> bool:
    if pd.isna(x):
        return default
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    return default


def _parse_date_value(x: Any) -> pd.Timestamp | pd.NaT:
    if pd.isna(x) or x == "":
        return pd.NaT
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(None)
    return pd.Timestamp(ts).normalize()


def _parse_date_series(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    if isinstance(out.dtype, pd.DatetimeTZDtype):
        out = out.dt.tz_convert(None)
    return out.dt.normalize()


def _normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return " ".join(str(x).strip().upper().split())


def _series_or_default(df: pd.DataFrame, col: str, default: Any) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _stable_hash(parts: Sequence[Any]) -> str:
    payload = "|".join("<NA>" if pd.isna(x) else str(x) for x in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_table(df: pd.DataFrame, path: Path, allow_csv_fallback: bool = True, compression: str = "snappy") -> Path:
    _ensure_parent(path)
    try:
        df.to_parquet(path, index=False, compression=compression)
        return path
    except Exception as exc:
        if not allow_csv_fallback:
            raise RuntimeError(
                f"Failed to write parquet to {path}. Install pyarrow or fastparquet. Original error: {exc}"
            ) from exc
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def write_json(obj: Mapping[str, Any], path: Path) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        if suffix == ".json":
            with p.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                return pd.DataFrame(raw)
            if isinstance(raw, dict):
                # SEC-style dictionary keyed by row id.
                if all(isinstance(v, Mapping) for v in raw.values()):
                    return pd.DataFrame(list(raw.values()))
                if "data" in raw and isinstance(raw["data"], list):
                    return pd.DataFrame(raw["data"])
                return pd.DataFrame([raw])
        return pd.read_json(p, lines=True)
    raise ValueError(f"Unsupported file extension for table input: {p.suffix}")


def normalize_columns(df: pd.DataFrame, aliases: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    out = df.copy()
    cols = {str(c): str(c) for c in out.columns}
    lower_map = {str(c).lower(): str(c) for c in out.columns}
    for canonical, variants in aliases.items():
        if canonical in out.columns:
            continue
        picked: Optional[str] = None
        for v in variants:
            if v in cols:
                picked = v
                break
            if v.lower() in lower_map:
                picked = lower_map[v.lower()]
                break
        if picked is not None:
            out = out.rename(columns={picked: canonical})
    return out


def _source_payload_hash(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()[:16]


# -----------------------------------------------------------------------------
# Input preparation
# -----------------------------------------------------------------------------
def _default_effective_from(cfg: Mapping[str, Any]) -> pd.Timestamp:
    return pd.Timestamp(cfg["identity"].get("default_effective_from", "1900-01-01")).normalize()


def _source_base_confidence(source_name: str, cfg: Mapping[str, Any]) -> float:
    base = cfg.get("sources", {}).get("base_confidence", {})
    return float(base.get(source_name, 0.50))


def _source_priority(source_name: str, cfg: Mapping[str, Any]) -> int:
    pr = cfg.get("sources", {}).get("priority", {})
    return int(pr.get(source_name, 0))


def _status_for_source(source_name: str, cfg: Mapping[str, Any]) -> str:
    status_map = cfg.get("resolution", {}).get("status_by_source", {})
    return str(status_map.get(source_name, "preferred"))


def _prepare_raw_input(
    df: pd.DataFrame,
    source_name: str,
    cfg: Mapping[str, Any],
    failures: List[Dict[str, Any]],
) -> pd.DataFrame:
    out = normalize_columns(df, COMMON_ALIASES).copy()
    required = ["symbol", "cik"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required identity columns: {missing}")

    now = _utc_now()
    out["symbol"] = out["symbol"].map(_to_symbol)
    out["cik"] = out["cik"].map(_to_cik)
    out["issuer_name"] = _series_or_default(out, "issuer_name", "").map(_normalize_text)
    out["exchange"] = _series_or_default(out, "exchange", "").map(_normalize_text)
    share_blank = cfg["identity"].get("share_class_blank_value", "COMMON")
    out["share_class"] = _series_or_default(out, "share_class", share_blank).map(lambda x: _normalize_text(x) or share_blank)
    out["effective_from"] = _parse_date_series(_series_or_default(out, "effective_from", pd.NaT))
    out["effective_to"] = _parse_date_series(_series_or_default(out, "effective_to", pd.NaT))
    default_from = _default_effective_from(cfg)
    out["effective_from"] = out["effective_from"].fillna(default_from)
    if cfg["identity"].get("input_effective_to_is_inclusive", False):
        mask = out["effective_to"].notna()
        out.loc[mask, "effective_to"] = out.loc[mask, "effective_to"] + pd.Timedelta(days=1)
    out["is_active"] = _series_or_default(out, "is_active", True).map(lambda x: _coerce_bool(x, default=True))
    out["ingest_ts_utc"] = pd.to_datetime(_series_or_default(out, "ingest_ts_utc", now), errors="coerce")
    if isinstance(out["ingest_ts_utc"].dtype, pd.DatetimeTZDtype):
        out["ingest_ts_utc"] = out["ingest_ts_utc"].dt.tz_convert(None)
    out["ingest_ts_utc"] = out["ingest_ts_utc"].fillna(now)

    explicit_conf = pd.to_numeric(_series_or_default(out, "confidence_score", np.nan), errors="coerce")
    base_conf = _source_base_confidence(source_name, cfg)
    out["heuristic_flag"] = _series_or_default(out, "heuristic_flag", False).map(lambda x: _coerce_bool(x, default=False))
    out["confidence_score"] = explicit_conf.fillna(base_conf)
    out.loc[out["heuristic_flag"], "confidence_score"] = np.minimum(
        out.loc[out["heuristic_flag"], "confidence_score"],
        max(base_conf - 0.10, 0.0),
    )
    out["confidence_score"] = out["confidence_score"].clip(lower=0.0, upper=1.0)
    out["source"] = source_name
    out["source_priority"] = _source_priority(source_name, cfg)
    out["resolution_status"] = _series_or_default(out, "resolution_status", _status_for_source(source_name, cfg)).fillna(
        _status_for_source(source_name, cfg)
    )
    out["mapping_version"] = _series_or_default(out, "mapping_version", cfg.get("ticker_cik_config_version", "1.0.0")).fillna(
        cfg.get("ticker_cik_config_version", "1.0.0")
    )

    invalid_rows = out[(out["symbol"] == "") | (out["cik"] == "")].copy()
    if not invalid_rows.empty:
        for _, row in invalid_rows.iterrows():
            failures.append(
                {
                    "conflict_class": "row_level_normalization_issue",
                    "conflict_status": "rejected",
                    "source": source_name,
                    "symbol": row.get("symbol", ""),
                    "cik": row.get("cik", ""),
                    "effective_from": row.get("effective_from"),
                    "effective_to": row.get("effective_to"),
                    "details": "Blank canonical symbol or CIK after normalization.",
                    "severity_rank": CONFLICT_PRECEDENCE["row_level_normalization_issue"],
                }
            )
        out = out[(out["symbol"] != "") & (out["cik"] != "")].copy()

    bad_window = out[(out["effective_to"].notna()) & (out["effective_to"] <= out["effective_from"])]
    if not bad_window.empty:
        for _, row in bad_window.iterrows():
            failures.append(
                {
                    "conflict_class": "schema_or_config_failure",
                    "conflict_status": "rejected",
                    "source": source_name,
                    "symbol": row["symbol"],
                    "cik": row["cik"],
                    "effective_from": row["effective_from"],
                    "effective_to": row["effective_to"],
                    "details": "effective_to must be strictly greater than effective_from for non-open intervals.",
                    "severity_rank": CONFLICT_PRECEDENCE["schema_or_config_failure"],
                }
            )
        out = out[~((out["effective_to"].notna()) & (out["effective_to"] <= out["effective_from"]))].copy()

    out["evidence_hash"] = [
        _stable_hash(
            [
                row["source"],
                row["symbol"],
                row["cik"],
                row["issuer_name"],
                row["exchange"],
                row["share_class"],
                row["effective_from"],
                row["effective_to"],
            ]
        )
        for _, row in out.iterrows()
    ]
    out = out.reset_index(drop=True)
    out["_row_id"] = np.arange(len(out), dtype=np.int64)
    return out


def load_candidate_sources(
    sec_source_path: str,
    internal_master_path: Optional[str],
    cfg: Mapping[str, Any],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    sec_df = _prepare_raw_input(read_table(sec_source_path), "official_SEC_mapping", cfg, failures)
    frames = [sec_df]
    source_meta: Dict[str, Any] = {
        "official_SEC_mapping": {
            "path": sec_source_path,
            "n_input": int(len(sec_df)),
            "payload_hash": _source_payload_hash(sec_source_path),
        }
    }
    if internal_master_path:
        int_df = _prepare_raw_input(read_table(internal_master_path), "trusted_internal_master", cfg, failures)
        frames.append(int_df)
        source_meta["trusted_internal_master"] = {
            "path": internal_master_path,
            "n_input": int(len(int_df)),
            "payload_hash": _source_payload_hash(internal_master_path),
        }
    elif not cfg["identity"].get("allow_empty_internal_master", True):
        raise ValueError("internal_master_path is required by configuration.")

    candidates = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    return candidates, failures, source_meta


# -----------------------------------------------------------------------------
# Scoring and arbitration
# -----------------------------------------------------------------------------
def attach_corroboration_and_metadata_score(candidates: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    if candidates.empty:
        out = candidates.copy()
        out["corroboration_count"] = 0
        out["metadata_consistency"] = 0.0
        out["lineage_count"] = 0
        return out

    out = candidates.copy()
    key_counts = out.groupby(["symbol", "cik"], dropna=False)["source"].nunique().reset_index(name="corroboration_count")
    out = out.merge(key_counts, on=["symbol", "cik"], how="left")
    out["corroboration_count"] = out["corroboration_count"].fillna(1).astype(int)

    lineage_counts = out.groupby(["symbol", "cik"], dropna=False)["evidence_hash"].nunique().reset_index(name="lineage_count")
    out = out.merge(lineage_counts, on=["symbol", "cik"], how="left")
    out["lineage_count"] = out["lineage_count"].fillna(1).astype(int)

    w = cfg["resolution"].get("metadata_weights", {})
    corroboration_norm = (out["corroboration_count"].clip(lower=1).astype(float) - 1.0).clip(lower=0.0) / 2.0
    out["metadata_consistency"] = (
        w.get("issuer_name", 0.35) * (out["issuer_name"] != "").astype(float)
        + w.get("exchange", 0.15) * (out["exchange"] != "").astype(float)
        + w.get("share_class", 0.15) * (out["share_class"] != "").astype(float)
        + w.get("cross_source_corroboration", 0.35) * corroboration_norm.clip(upper=1.0)
    )
    out["metadata_consistency"] = out["metadata_consistency"].clip(lower=0.0, upper=1.0)
    return out


def _candidate_active_mask(group: pd.DataFrame, start: pd.Timestamp) -> pd.Series:
    return (group["effective_from"] <= start) & (group["effective_to"].isna() | (group["effective_to"] > start)) & group["is_active"]


def _best_rows_for_segment(active: pd.DataFrame, previous_cik: Optional[str], cfg: Mapping[str, Any]) -> pd.DataFrame:
    if active.empty:
        return active
    out = active.copy()
    if cfg["identity"].get("prefer_temporal_continuity", True) and previous_cik:
        out["temporal_consistency"] = (out["cik"] == previous_cik).astype(int)
    else:
        out["temporal_consistency"] = 0
    out["source_priority"] = pd.to_numeric(out["source_priority"], errors="coerce").fillna(0).astype(int)
    out["confidence_score"] = pd.to_numeric(out["confidence_score"], errors="coerce").fillna(0.0)
    out["metadata_consistency"] = pd.to_numeric(out["metadata_consistency"], errors="coerce").fillna(0.0)
    out["_ingest_int"] = pd.to_datetime(out["ingest_ts_utc"], errors="coerce").astype("int64")
    out = out.sort_values(
        ["source_priority", "confidence_score", "temporal_consistency", "metadata_consistency", "_ingest_int"],
        ascending=[False, False, False, False, False],
        kind="mergesort",
    )
    top = out.iloc[0]
    tol = float(cfg["resolution"].get("tie_tolerance", 1e-12))
    mask = (
        (out["source_priority"] == int(top["source_priority"]))
        & (np.abs(out["confidence_score"] - float(top["confidence_score"])) <= tol)
        & (out["temporal_consistency"] == int(top["temporal_consistency"]))
        & (np.abs(out["metadata_consistency"] - float(top["metadata_consistency"])) <= tol)
        & (out["_ingest_int"] == int(top["_ingest_int"]))
    )
    return out.loc[mask].copy()


def _boundaries_for_symbol(group: pd.DataFrame) -> List[pd.Timestamp]:
    bounds: set[pd.Timestamp] = set()
    for start in group["effective_from"].dropna().tolist():
        bounds.add(pd.Timestamp(start).normalize())
    for end in group["effective_to"].dropna().tolist():
        bounds.add(pd.Timestamp(end).normalize())
    return sorted(bounds)


def resolve_symbol_history(
    symbol: str,
    group: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []
    if group.empty:
        return rows, conflicts

    boundaries = _boundaries_for_symbol(group)
    if not boundaries:
        return rows, conflicts

    prev_cik: Optional[str] = None
    prev_end: Optional[pd.Timestamp] = None

    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else pd.NaT
        active = group.loc[_candidate_active_mask(group, start)].copy()
        if active.empty:
            prev_cik = None
            prev_end = end if pd.notna(end) else None
            continue

        best = _best_rows_for_segment(active, prev_cik if prev_end == start else None, cfg)
        active_unique_cik = int(active["cik"].nunique())
        if len(best) > 1:
            winner_ciks = best["cik"].nunique()
            conflict_class = "unresolved_identity_conflict" if winner_ciks > 1 else "source_conflict"
            conflicts.append(
                {
                    "symbol": symbol,
                    "cik": pd.NA,
                    "effective_from": start,
                    "effective_to": end,
                    "conflict_class": conflict_class,
                    "conflict_status": "unresolved",
                    "source": ",".join(sorted(best["source"].astype(str).unique().tolist())),
                    "details": f"Multiple equally ranked active candidates remain for symbol={symbol}.",
                    "n_candidates": int(len(best)),
                    "n_active_cik": int(active_unique_cik),
                    "severity_rank": CONFLICT_PRECEDENCE.get(conflict_class, 70),
                    "candidate_evidence_hashes": ",".join(best["evidence_hash"].astype(str).tolist()),
                }
            )
            prev_cik = None
            prev_end = end if pd.notna(end) else None
            continue

        winner = best.iloc[0]
        if active_unique_cik > 1:
            losing = active[active["cik"] != winner["cik"]].copy()
            if not losing.empty:
                conflicts.append(
                    {
                        "symbol": symbol,
                        "cik": winner["cik"],
                        "effective_from": start,
                        "effective_to": end,
                        "conflict_class": "source_conflict",
                        "conflict_status": "resolved_by_precedence",
                        "source": ",".join(sorted(active["source"].astype(str).unique().tolist())),
                        "details": f"Concurrent distinct CIK candidates arbitrated deterministically; winner={winner['cik']}.",
                        "n_candidates": int(len(active)),
                        "n_active_cik": int(active_unique_cik),
                        "severity_rank": 60,
                        "candidate_evidence_hashes": ",".join(active["evidence_hash"].astype(str).tolist()),
                    }
                )

        rows.append(
            {
                "symbol": symbol,
                "cik": winner["cik"],
                "source": winner["source"],
                "source_priority": int(winner["source_priority"]),
                "confidence_score": float(winner["confidence_score"]),
                "metadata_consistency": float(winner["metadata_consistency"]),
                "temporal_consistency": int(winner["temporal_consistency"]),
                "effective_from": start,
                "effective_to": end,
                "is_active": True,
                "resolution_status": str(winner["resolution_status"]),
                "heuristic_flag": bool(winner["heuristic_flag"]),
                "issuer_name": winner.get("issuer_name", ""),
                "exchange": winner.get("exchange", ""),
                "share_class": winner.get("share_class", cfg["identity"].get("share_class_blank_value", "COMMON")),
                "evidence_hash": winner.get("evidence_hash", ""),
                "mapping_version": winner.get("mapping_version", cfg.get("ticker_cik_config_version", "1.0.0")),
                "lineage_count": int(winner.get("lineage_count", 1)),
                "corroboration_count": int(winner.get("corroboration_count", 1)),
                "previous_symbol_cik": prev_cik,
            }
        )
        prev_cik = str(winner["cik"])
        prev_end = end if pd.notna(end) else None

    return rows, conflicts


def merge_adjacent_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    sort_cols = ["symbol", "effective_from", "cik", "source_priority", "confidence_score"]
    out = df.sort_values(sort_cols).reset_index(drop=True).copy()
    keep_rows: List[Dict[str, Any]] = []
    prev: Optional[MutableMapping[str, Any]] = None
    same_cols = [
        "symbol",
        "cik",
        "source",
        "source_priority",
        "resolution_status",
        "heuristic_flag",
        "issuer_name",
        "exchange",
        "share_class",
        "evidence_hash",
        "mapping_version",
    ]
    for _, row in out.iterrows():
        row_dict = row.to_dict()
        if prev is None:
            prev = row_dict
            continue
        contiguous = pd.isna(prev["effective_to"]) or (prev["effective_to"] == row_dict["effective_from"])
        same_payload = all(prev[c] == row_dict[c] for c in same_cols)
        score_close = math.isclose(float(prev["confidence_score"]), float(row_dict["confidence_score"]), rel_tol=0.0, abs_tol=1e-12) and math.isclose(
            float(prev["metadata_consistency"]), float(row_dict["metadata_consistency"]), rel_tol=0.0, abs_tol=1e-12
        )
        if contiguous and same_payload and score_close:
            prev["effective_to"] = row_dict["effective_to"]
            prev["temporal_consistency"] = max(int(prev["temporal_consistency"]), int(row_dict["temporal_consistency"]))
            prev["lineage_count"] = max(int(prev["lineage_count"]), int(row_dict["lineage_count"]))
            prev["corroboration_count"] = max(int(prev["corroboration_count"]), int(row_dict["corroboration_count"]))
        else:
            keep_rows.append(dict(prev))
            prev = row_dict
    if prev is not None:
        keep_rows.append(dict(prev))
    merged = pd.DataFrame(keep_rows)
    merged["_history_row_id"] = np.arange(len(merged), dtype=np.int64)
    return merged


def harmonize_ticker_changes(history: pd.DataFrame, conflicts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if history.empty:
        return history, conflicts
    hist = history.sort_values(["cik", "effective_from", "symbol"]).copy().reset_index(drop=True)
    extra_conflicts: List[Dict[str, Any]] = []
    for cik, idx in hist.groupby("cik").groups.items():
        pos = list(idx)
        if len(pos) < 2:
            continue
        sub = hist.loc[pos].sort_values(["effective_from", "symbol"]).reset_index()
        for i in range(len(sub) - 1):
            a = sub.iloc[i]
            b = sub.iloc[i + 1]
            if a["symbol"] == b["symbol"]:
                continue
            same_class = str(a.get("share_class", "")) == str(b.get("share_class", ""))
            same_issuer = str(a.get("issuer_name", "")) == str(b.get("issuer_name", ""))
            a_end = a["effective_to"]
            b_start = b["effective_from"]
            overlap = pd.isna(a_end) or (pd.Timestamp(a_end) > pd.Timestamp(b_start))
            if same_class and same_issuer and overlap:
                hist.loc[a["index"], "effective_to"] = pd.Timestamp(b_start)
                extra_conflicts.append(
                    {
                        "symbol": str(a["symbol"]),
                        "cik": str(cik),
                        "effective_from": a["effective_from"],
                        "effective_to": b_start,
                        "conflict_class": "ticker_change",
                        "conflict_status": "resolved_by_historification",
                        "source": f"{a['source']},{b['source']}",
                        "details": f"Closed prior symbol {a['symbol']} at start of successor symbol {b['symbol']} for same CIK.",
                        "n_candidates": 2,
                        "n_active_cik": 1,
                        "severity_rank": 20,
                        "candidate_evidence_hashes": f"{a['evidence_hash']},{b['evidence_hash']}",
                    }
                )
    hist = hist[(hist["effective_to"].isna()) | (hist["effective_to"] > hist["effective_from"])].copy()
    hist = merge_adjacent_history(hist.drop(columns=[c for c in ["_history_row_id"] if c in hist.columns]))
    if extra_conflicts:
        conflicts = pd.concat([conflicts, pd.DataFrame(extra_conflicts)], ignore_index=True, sort=False) if not conflicts.empty else pd.DataFrame(extra_conflicts)
    return hist, conflicts


def build_history(candidates: pd.DataFrame, cfg: Mapping[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    candidates = attach_corroboration_and_metadata_score(candidates, cfg)
    history_rows: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []
    for symbol, group in candidates.groupby("symbol", sort=True):
        rows, cfs = resolve_symbol_history(symbol, group.copy(), cfg)
        history_rows.extend(rows)
        conflicts.extend(cfs)
    history = merge_adjacent_history(pd.DataFrame(history_rows))
    conflicts_df = pd.DataFrame(conflicts)
    if history.empty:
        return history, conflicts_df

    # Additional integrity checks on current history itself.
    overlap_conflicts: List[Dict[str, Any]] = []
    for symbol, group in history.groupby("symbol", sort=False):
        g = group.sort_values("effective_from").reset_index(drop=True)
        for i in range(len(g) - 1):
            a = g.iloc[i]
            b = g.iloc[i + 1]
            a_end = a["effective_to"]
            if pd.notna(a_end) and a_end > b["effective_from"]:
                overlap_conflicts.append(
                    {
                        "symbol": symbol,
                        "cik": pd.NA,
                        "effective_from": b["effective_from"],
                        "effective_to": a_end,
                        "conflict_class": "temporal_overlap_conflict",
                        "conflict_status": "unresolved",
                        "source": f"{a['source']},{b['source']}",
                        "details": "Resolved history contains overlapping active intervals for the same symbol.",
                        "n_candidates": 2,
                        "n_active_cik": int(pd.Series([a["cik"], b["cik"]]).nunique()),
                        "severity_rank": CONFLICT_PRECEDENCE["temporal_overlap_conflict"],
                        "candidate_evidence_hashes": f"{a['evidence_hash']},{b['evidence_hash']}",
                    }
                )
    if overlap_conflicts:
        conflicts_df = pd.concat([conflicts_df, pd.DataFrame(overlap_conflicts)], ignore_index=True, sort=False)
    history, conflicts_df = harmonize_ticker_changes(history, conflicts_df)
    return history, conflicts_df


# -----------------------------------------------------------------------------
# Current snapshot and PIT helpers
# -----------------------------------------------------------------------------
def active_asof(history: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    d = pd.Timestamp(asof).normalize()
    mask = (history["effective_from"] <= d) & (history["effective_to"].isna() | (history["effective_to"] > d))
    return history.loc[mask].copy()


def build_current(history: pd.DataFrame, conflicts: pd.DataFrame, asof: pd.Timestamp, cfg: Mapping[str, Any]) -> pd.DataFrame:
    current = active_asof(history, asof)
    if current.empty:
        return current
    min_conf = float(cfg["resolution"].get("min_confidence_for_current", 0.50))
    current = current[current["confidence_score"] >= min_conf].copy()
    if cfg["resolution"].get("current_requires_non_heuristic", False):
        current = current[~current["heuristic_flag"]].copy()

    if not conflicts.empty:
        unresolved = conflicts[conflicts["conflict_status"] == "unresolved"].copy()
        if not unresolved.empty:
            overlap_symbols: set[str] = set()
            d = pd.Timestamp(asof).normalize()
            for _, row in unresolved.iterrows():
                start = row.get("effective_from")
                end = row.get("effective_to")
                start_ok = pd.notna(start) and pd.Timestamp(start).normalize() <= d
                end_ok = pd.isna(end) or pd.Timestamp(end).normalize() > d
                if start_ok and end_ok and str(row.get("symbol", "")):
                    overlap_symbols.add(str(row["symbol"]))
            if overlap_symbols:
                current = current[~current["symbol"].isin(sorted(overlap_symbols))].copy()

    allow_multi = cfg["identity"].get("allow_multi_symbol_per_cik_when_share_class_distinct", True)
    if not allow_multi and not current.empty:
        dup_cik = current.groupby("cik")["symbol"].nunique().reset_index(name="n_symbol")
        bad = dup_cik[dup_cik["n_symbol"] > 1]["cik"]
        if not bad.empty:
            current = current[~current["cik"].isin(bad)].copy()
    else:
        if not current.empty:
            amb = current.groupby("cik").agg(n_symbol=("symbol", "nunique"), n_class=("share_class", "nunique")).reset_index()
            bad = amb[(amb["n_symbol"] > 1) & (amb["n_class"] <= 1)]["cik"]
            if not bad.empty:
                current = current[~current["cik"].isin(bad)].copy()

    current = current.sort_values(["symbol", "source_priority", "confidence_score"], ascending=[True, False, False])
    current = current.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)
    return current


def resolve_pit_identity(history: pd.DataFrame, symbol: str, asof: Any) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    sym = _to_symbol(symbol)
    d = pd.Timestamp(asof).normalize()
    sub = history[history["symbol"] == sym].copy()
    if sub.empty:
        return sub
    mask = (sub["effective_from"] <= d) & (sub["effective_to"].isna() | (sub["effective_to"] > d))
    return sub.loc[mask].copy()


# -----------------------------------------------------------------------------
# Metrics and artifacts
# -----------------------------------------------------------------------------
def _count_ticker_changes(history: pd.DataFrame) -> int:
    if history.empty:
        return 0
    changes = 0
    for cik, group in history.sort_values(["cik", "effective_from", "symbol"]).groupby("cik"):
        symbols = group["symbol"].tolist()
        if not symbols:
            continue
        prev = symbols[0]
        for sym in symbols[1:]:
            if sym != prev:
                changes += 1
                prev = sym
    return int(changes)


def compute_metrics(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    current: pd.DataFrame,
    conflicts: pd.DataFrame,
    internal_master_path: Optional[str],
) -> pd.DataFrame:
    internal_symbols = None
    if internal_master_path and Path(internal_master_path).exists():
        try:
            internal_symbols = set(_prepare_raw_input(read_table(internal_master_path), "trusted_internal_master", DEFAULT_CONFIG, []).get("symbol", pd.Series(dtype=object)).tolist())
        except Exception:
            internal_symbols = None
    union_symbols = set(candidates["symbol"].tolist()) if not candidates.empty else set()
    target_symbols = internal_symbols if internal_symbols else union_symbols
    current_symbol_set = set(current["symbol"].tolist()) if not current.empty else set()
    unresolved = conflicts[conflicts["conflict_status"] == "unresolved"] if not conflicts.empty else pd.DataFrame()

    rows: List[Dict[str, Any]] = [
        {"metric_name": "n_input_rows", "metric_value": int(len(candidates))},
        {"metric_name": "n_unique_input_symbols", "metric_value": int(candidates["symbol"].nunique() if not candidates.empty else 0)},
        {"metric_name": "n_unique_input_cik", "metric_value": int(candidates["cik"].nunique() if not candidates.empty else 0)},
        {"metric_name": "n_history_rows", "metric_value": int(len(history))},
        {"metric_name": "n_current_rows", "metric_value": int(len(current))},
        {"metric_name": "n_current_symbols", "metric_value": int(len(current_symbol_set))},
        {"metric_name": "n_current_cik", "metric_value": int(current["cik"].nunique() if not current.empty else 0)},
        {"metric_name": "n_conflicts_total", "metric_value": int(len(conflicts))},
        {"metric_name": "n_conflicts_unresolved", "metric_value": int(len(unresolved))},
        {"metric_name": "n_ticker_changes_detected", "metric_value": _count_ticker_changes(history)},
        {
            "metric_name": "coverage_over_target_symbols",
            "metric_value": float(len(current_symbol_set) / len(target_symbols)) if target_symbols else np.nan,
        },
    ]

    if not conflicts.empty:
        by_class = conflicts.groupby("conflict_class").size().reset_index(name="n")
        for _, row in by_class.iterrows():
            rows.append({"metric_name": f"conflict_class::{row['conflict_class']}", "metric_value": int(row["n"])})

    if not current.empty:
        for stat in ["mean", "min", "median", "max"]:
            val = getattr(current["confidence_score"], stat)()
            rows.append({"metric_name": f"current_confidence::{stat}", "metric_value": float(val)})
        by_source = current.groupby("source").size().reset_index(name="n")
        for _, row in by_source.iterrows():
            rows.append({"metric_name": f"current_source::{row['source']}", "metric_value": int(row['n'])})

    return pd.DataFrame(rows)


def build_manifest(
    run_id: str,
    asof: pd.Timestamp,
    cfg: Mapping[str, Any],
    source_meta: Mapping[str, Any],
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    current: pd.DataFrame,
    conflicts: pd.DataFrame,
    started_at: pd.Timestamp,
    finished_at: pd.Timestamp,
) -> Dict[str, Any]:
    by_class = conflicts.groupby(["conflict_class", "conflict_status"]).size().reset_index(name="n") if not conflicts.empty else pd.DataFrame()
    conflicts_summary: Dict[str, Any] = {}
    if not by_class.empty:
        for _, row in by_class.iterrows():
            key = f"{row['conflict_class']}::{row['conflict_status']}"
            conflicts_summary[key] = int(row["n"])
    manifest = {
        "run_id": run_id,
        "logical_asof_date": str(pd.Timestamp(asof).normalize().date()),
        "config_version": cfg.get("ticker_cik_config_version", "1.0.0"),
        "config_hash": _config_hash(cfg),
        "sources": source_meta,
        "n_input_rows": int(len(candidates)),
        "n_resolved_history_rows": int(len(history)),
        "n_current_rows": int(len(current)),
        "n_unique_history_symbols": int(history["symbol"].nunique() if not history.empty else 0),
        "n_unique_current_symbols": int(current["symbol"].nunique() if not current.empty else 0),
        "n_unique_current_cik": int(current["cik"].nunique() if not current.empty else 0),
        "conflicts": conflicts_summary,
        "n_historical_changes_detected": _count_ticker_changes(history),
        "started_at_utc": str(started_at),
        "finished_at_utc": str(finished_at),
        "interval_convention": {
            "effective_from": "inclusive",
            "effective_to": "exclusive",
            "pit_rule": "effective_from <= d < effective_to, or open-ended when effective_to is null",
        },
    }
    return manifest


def validate_run(candidates: pd.DataFrame, history: pd.DataFrame, conflicts: pd.DataFrame, cfg: Mapping[str, Any]) -> None:
    if candidates.empty:
        raise ValueError("No candidate mappings available after normalization.")
    hard = conflicts[conflicts["conflict_class"] == "schema_or_config_failure"] if not conflicts.empty else pd.DataFrame()
    if not hard.empty and cfg["identity"].get("strict_abort_on_schema_failure", True):
        raise ValueError("Aborting due to schema/config failures in ticker_cik input contract.")
    unresolved = conflicts[conflicts["conflict_status"] == "unresolved"] if not conflicts.empty else pd.DataFrame()
    if not unresolved.empty and cfg["identity"].get("strict_abort_on_unresolved_conflict", False):
        raise ValueError("Aborting due to unresolved identity conflicts.")
    if not history.empty:
        dup = history.duplicated(subset=["symbol", "effective_from", "effective_to"], keep=False)
        if dup.any():
            raise ValueError("History contains duplicate symbol interval rows after arbitration.")


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------
def run_ticker_cik(
    sec_source_path: str,
    internal_master_path: Optional[str] = None,
    config_path: Optional[str] = None,
    run_id: Optional[str] = None,
    asof: Optional[str] = None,
) -> Dict[str, Any]:
    started_at = _utc_now()
    cfg = load_config(config_path)
    if run_id is None:
        run_id = f"ticker_cik_{started_at.strftime('%Y%m%d_%H%M%S')}"
    asof_ts = pd.Timestamp(asof).normalize() if asof else _utc_now().normalize()

    candidates, prep_failures, source_meta = load_candidate_sources(sec_source_path, internal_master_path, cfg)
    history, resolution_conflicts = build_history(candidates, cfg)

    conflicts = pd.concat(
        [pd.DataFrame(prep_failures), resolution_conflicts],
        ignore_index=True,
        sort=False,
    ) if (prep_failures or not resolution_conflicts.empty) else pd.DataFrame(
        columns=[
            "symbol", "cik", "effective_from", "effective_to", "conflict_class", "conflict_status",
            "source", "details", "n_candidates", "n_active_cik", "severity_rank", "candidate_evidence_hashes"
        ]
    )

    if not history.empty:
        history["current_candidate_flag"] = False
    current = build_current(history, conflicts, asof_ts, cfg)
    if not current.empty:
        current["current_candidate_flag"] = True

    validate_run(candidates, history, conflicts, cfg)
    metrics = compute_metrics(candidates, history, current, conflicts, internal_master_path)
    finished_at = _utc_now()
    manifest = build_manifest(run_id, asof_ts, cfg, source_meta, candidates, history, current, conflicts, started_at, finished_at)

    output_root = Path(cfg["storage"].get("output_root", "data/edgar/mappings"))
    allow_csv_fallback = bool(cfg["storage"].get("allow_csv_fallback", True))
    compression = str(cfg["storage"].get("compression", "snappy"))

    history_path = write_table(history, output_root / "ticker_cik_history.parquet", allow_csv_fallback, compression)
    current_path = write_table(current, output_root / "ticker_cik_current.parquet", allow_csv_fallback, compression)
    conflicts_path = write_table(
        conflicts.sort_values(["severity_rank", "symbol", "effective_from"], ascending=[False, True, True]),
        output_root / f"ticker_cik_conflicts_{run_id}.parquet",
        allow_csv_fallback,
        compression,
    )
    metrics_path = write_table(metrics, output_root / f"ticker_cik_metrics_{run_id}.parquet", allow_csv_fallback, compression)
    manifest_path = output_root / f"ticker_cik_manifest_{run_id}.json"
    write_json(manifest, manifest_path)

    return {
        "history": history,
        "current": current,
        "conflicts": conflicts,
        "metrics": metrics,
        "manifest": manifest,
        "artifacts": {
            "history_path": str(history_path),
            "current_path": str(current_path),
            "conflicts_path": str(conflicts_path),
            "metrics_path": str(metrics_path),
            "manifest_path": str(manifest_path),
        },
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a historified, PIT-safe ticker↔CIK identity master.")
    parser.add_argument("--source-path", "--sec-source-path", dest="sec_source_path", required=True, help="Official SEC ticker/CIK source path.")
    parser.add_argument("--internal-master-path", default=None, help="Optional internal symbol master path.")
    parser.add_argument("--config-path", default=None, help="Optional YAML/JSON config path.")
    parser.add_argument("--run-id", default=None, help="Run identifier.")
    parser.add_argument("--asof", default=None, help="Logical as-of date, e.g. 2026-03-05.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_ticker_cik(
        sec_source_path=args.sec_source_path,
        internal_master_path=args.internal_master_path,
        config_path=args.config_path,
        run_id=args.run_id,
        asof=args.asof,
    )
    manifest = result["manifest"]
    payload = {
        "run_id": manifest["run_id"],
        "logical_asof_date": manifest["logical_asof_date"],
        "n_resolved_history_rows": manifest["n_resolved_history_rows"],
        "n_current_rows": manifest["n_current_rows"],
        "conflicts": manifest["conflicts"],
        "artifacts": result["artifacts"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
