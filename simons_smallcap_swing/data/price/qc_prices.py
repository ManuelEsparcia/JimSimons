from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# -----------------------------------------------------------------------------
# Exceptions and enums
# -----------------------------------------------------------------------------


class QCPricesError(RuntimeError):
    """Raised when a hard QC contract or IO invariant fails."""


class Severity(str, Enum):
    INFO = "INFO"
    WARN_SYMBOL = "WARN_SYMBOL"
    FAIL_ROW = "FAIL_ROW"
    FAIL_STRUCTURAL = "FAIL_STRUCTURAL"


class Gate(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class FailureCode(str, Enum):
    DATASET_NOT_FOUND = "DATASET_NOT_FOUND"
    DATASET_UNREADABLE = "DATASET_UNREADABLE"
    CALENDAR_NOT_FOUND = "CALENDAR_NOT_FOUND"
    CONFIG_INVALID = "CONFIG_INVALID"
    MISSING_REQUIRED_COLUMN = "MISSING_REQUIRED_COLUMN"
    INVALID_DATE_PARSE = "INVALID_DATE_PARSE"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    DUPLICATE_PK = "DUPLICATE_PK"
    EMPTY_DATASET = "EMPTY_DATASET"
    INCOMPATIBLE_CALENDAR = "INCOMPATIBLE_CALENDAR"
    OHLC_INVALID = "OHLC_INVALID"
    NONPOSITIVE_PRICE = "NONPOSITIVE_PRICE"
    NEGATIVE_VOLUME = "NEGATIVE_VOLUME"
    PENNY_PRICE = "PENNY_PRICE"
    MISSING_SESSIONS = "MISSING_SESSIONS"
    BAD_COVERAGE = "BAD_COVERAGE"
    EXTREME_RETURN_PLAUSIBLE = "EXTREME_RETURN_PLAUSIBLE"
    EXTREME_RETURN_SUSPECT = "EXTREME_RETURN_SUSPECT"
    RAW_ADJUSTED_INCONSISTENT = "RAW_ADJUSTED_INCONSISTENT"
    ADJUSTED_DATASET_MISSING = "ADJUSTED_DATASET_MISSING"
    ADJUSTED_DATASET_INCOMPLETE = "ADJUSTED_DATASET_INCOMPLETE"
    OUTSIDE_CALENDAR = "OUTSIDE_CALENDAR"


SEVERITY_RANK = {
    Severity.INFO.value: 0,
    Severity.WARN_SYMBOL.value: 1,
    Severity.FAIL_ROW.value: 2,
    Severity.FAIL_STRUCTURAL.value: 3,
}

RAW_REQUIRED_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume", "source_provider"]
ADJUSTED_BASE_COLUMNS = ["symbol", "date"]
ADJUSTED_OPTIONAL_COLUMNS = ["close_adj_split", "close_adj_total", "ret_1d_adj_split", "ret_1d_adj_total"]

FINDING_COLUMNS = ["symbol", "date", "check_name", "severity", "observed_value", "threshold", "message", "failure_code"]


def concat_findings(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        cur = frame.copy()
        for col in FINDING_COLUMNS:
            if col not in cur.columns:
                cur[col] = pd.NA
        for row in cur[FINDING_COLUMNS].to_dict(orient="records"):
            records.append(row)
    if not records:
        return pd.DataFrame(columns=FINDING_COLUMNS)
    return pd.DataFrame.from_records(records, columns=FINDING_COLUMNS)




# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class QCConfig:
    ohlc_invalid_fail_threshold: float = 0.001
    extreme_return_threshold: float = 0.40
    symbol_bad_coverage_threshold: float = 0.10
    pct_symbols_bad_coverage_fail_threshold: float = 0.10
    pct_fail_row_fail_threshold: float = 0.005
    penny_price_threshold: float = 1.0
    require_adjusted_consistency_check: bool = False
    raw_adjusted_sign_tolerance: float = 1e-10
    raw_adjusted_mag_ratio_threshold: float = 5.0
    raw_adjusted_factor_jump_threshold: float = 0.05
    pct_raw_adjusted_inconsistent_fail_threshold: float = 0.01
    symbol_extreme_return_warn_threshold: float = 0.05
    volume_plausible_zero_is_ok: bool = True
    allow_nonpositive_prices: bool = False
    min_symbol_rows_for_coverage: int = 2
    timezone: str = "UTC"
    code_version: str = "qc_prices_v1"


@dataclass
class RunArtifacts:
    row_level: pd.DataFrame
    symbol_level: pd.DataFrame
    failures: pd.DataFrame
    summary: Dict[str, Any]
    manifest: Dict[str, Any]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")



def sha256_of_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)



def config_hash(config: QCConfig) -> str:
    payload = dataclasses.asdict(config)
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()



def git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None



def ensure_parquet_support() -> None:
    try:
        import pyarrow  # type: ignore  # noqa: F401

        return
    except Exception:
        pass
    try:
        import fastparquet  # type: ignore  # noqa: F401

        return
    except Exception:
        pass
    raise QCPricesError(
        "Parquet support is required but neither 'pyarrow' nor 'fastparquet' is available in the environment."
    )



def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_parquet_support()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)



def write_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)



def load_yaml_or_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise QCPricesError(f"Config path does not exist: {path}")
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise QCPricesError("PyYAML is not installed, cannot load YAML config.")
        data = yaml.safe_load(text) or {}
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise QCPricesError(f"Unsupported config suffix: {suffix}")
    if not isinstance(data, dict):
        raise QCPricesError("QC config must deserialize to a dictionary.")
    return data



def load_config(path: Path) -> QCConfig:
    raw = load_yaml_or_json(path)
    allowed = {f.name for f in dataclasses.fields(QCConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise QCPricesError(f"Unknown QC config keys: {unknown}")
    cfg = QCConfig(**raw)
    if cfg.extreme_return_threshold <= 0:
        raise QCPricesError("extreme_return_threshold must be > 0")
    if not 0 <= cfg.symbol_bad_coverage_threshold <= 1:
        raise QCPricesError("symbol_bad_coverage_threshold must be in [0,1]")
    if not 0 <= cfg.pct_symbols_bad_coverage_fail_threshold <= 1:
        raise QCPricesError("pct_symbols_bad_coverage_fail_threshold must be in [0,1]")
    if not 0 <= cfg.pct_fail_row_fail_threshold <= 1:
        raise QCPricesError("pct_fail_row_fail_threshold must be in [0,1]")
    if not 0 <= cfg.pct_raw_adjusted_inconsistent_fail_threshold <= 1:
        raise QCPricesError("pct_raw_adjusted_inconsistent_fail_threshold must be in [0,1]")
    if cfg.penny_price_threshold <= 0:
        raise QCPricesError("penny_price_threshold must be > 0")
    return cfg



def infer_date_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None



def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise QCPricesError(f"Dataset path not found: {path}")
    suffix = path.suffix.lower()
    if path.is_dir():
        ensure_parquet_support()
        return pd.read_parquet(path)
    if suffix == ".parquet":
        ensure_parquet_support()
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise QCPricesError(f"Unsupported dataset format: {path}")



def normalize_date_column(df: pd.DataFrame, column: str, target: str = "date") -> pd.DataFrame:
    out = df.copy()
    parsed = pd.to_datetime(out[column], utc=False, errors="coerce")
    if parsed.isna().any():
        bad = int(parsed.isna().sum())
        raise QCPricesError(f"Failed to parse {bad} values from date column '{column}'")
    out[target] = parsed.dt.normalize()
    if column != target:
        out = out.drop(columns=[column])
    return out



def coerce_float(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out



def pick_max_severity(values: Iterable[str]) -> Optional[str]:
    vals = [v for v in values if isinstance(v, str) and v]
    if not vals:
        return None
    return max(vals, key=lambda x: SEVERITY_RANK.get(x, -1))



def build_finding_rows(
    base: pd.DataFrame,
    mask: pd.Series,
    *,
    check_name: str,
    severity: str,
    observed_value: Optional[pd.Series | Any],
    threshold: Any,
    message: str,
    code: str,
) -> pd.DataFrame:
    if mask is None or len(mask) == 0 or not mask.any():
        return pd.DataFrame(columns=["symbol", "date", "check_name", "severity", "observed_value", "threshold", "message", "failure_code"])
    flagged = base.loc[mask, ["symbol", "date"]].copy()
    flagged["check_name"] = check_name
    flagged["severity"] = severity
    if isinstance(observed_value, pd.Series):
        flagged["observed_value"] = observed_value.loc[mask].astype(object).values
    else:
        flagged["observed_value"] = observed_value
    flagged["threshold"] = threshold
    flagged["message"] = message
    flagged["failure_code"] = code
    return flagged


# -----------------------------------------------------------------------------
# Structural loading and validation
# -----------------------------------------------------------------------------


def read_calendar(calendar_path: Path) -> pd.DataFrame:
    cal = load_dataframe(calendar_path)
    date_col = infer_date_column(cal, ["date", "trade_date", "session_date"])
    if date_col is None:
        raise QCPricesError("Calendar dataset must contain one of: date, trade_date, session_date")
    cal = normalize_date_column(cal, date_col, target="date")
    if "is_session" in cal.columns:
        cal = cal.loc[cal["is_session"].fillna(True).astype(bool)].copy()
    cal = cal[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
    if cal.empty:
        raise QCPricesError("Calendar dataset is empty after filtering valid sessions")
    return cal



def structural_failure_df(
    *,
    check_name: str,
    severity: str,
    message: str,
    code: str,
    symbol: Optional[str] = None,
    date: Optional[pd.Timestamp] = None,
    observed_value: Any = None,
    threshold: Any = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "date": date,
                "check_name": check_name,
                "severity": severity,
                "observed_value": observed_value,
                "threshold": threshold,
                "message": message,
                "failure_code": code,
            }
        ]
    )



def validate_raw_schema(raw: pd.DataFrame) -> List[pd.DataFrame]:
    findings: List[pd.DataFrame] = []
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in raw.columns]
    for col in missing:
        findings.append(
            structural_failure_df(
                check_name="schema_required_column",
                severity=Severity.FAIL_STRUCTURAL.value,
                message=f"Required raw prices column is missing: {col}",
                code=FailureCode.MISSING_REQUIRED_COLUMN.value,
                observed_value=col,
            )
        )
    if raw.empty:
        findings.append(
            structural_failure_df(
                check_name="raw_dataset_empty",
                severity=Severity.FAIL_STRUCTURAL.value,
                message="Raw prices dataset is empty.",
                code=FailureCode.EMPTY_DATASET.value,
            )
        )
    if findings:
        return findings

    dup = raw.duplicated(subset=["symbol", "date"], keep=False)
    if dup.any():
        findings.append(
            build_finding_rows(
                raw,
                dup,
                check_name="duplicate_pk",
                severity=Severity.FAIL_STRUCTURAL.value,
                observed_value=None,
                threshold="unique(symbol,date)",
                message="Duplicate logical primary key (symbol,date) detected in raw dataset.",
                code=FailureCode.DUPLICATE_PK.value,
            )
        )
    return findings



def validate_adjusted_schema(adj: pd.DataFrame, require_check: bool) -> List[pd.DataFrame]:
    findings: List[pd.DataFrame] = []
    missing = [c for c in ADJUSTED_BASE_COLUMNS if c not in adj.columns]
    for col in missing:
        findings.append(
            structural_failure_df(
                check_name="adjusted_required_column",
                severity=Severity.FAIL_STRUCTURAL.value,
                message=f"Required adjusted prices column is missing: {col}",
                code=FailureCode.MISSING_REQUIRED_COLUMN.value,
                observed_value=col,
            )
        )
    if findings:
        return findings

    has_any_price = any(c in adj.columns for c in ["close_adj_split", "close_adj_total"])
    if require_check and not has_any_price:
        findings.append(
            structural_failure_df(
                check_name="adjusted_incomplete",
                severity=Severity.FAIL_STRUCTURAL.value,
                message="Adjusted dataset is present but lacks close_adj_split/close_adj_total required for consistency checks.",
                code=FailureCode.ADJUSTED_DATASET_INCOMPLETE.value,
            )
        )

    dup = adj.duplicated(subset=["symbol", "date"], keep=False)
    if dup.any():
        findings.append(
            build_finding_rows(
                adj,
                dup,
                check_name="adjusted_duplicate_pk",
                severity=Severity.FAIL_STRUCTURAL.value,
                observed_value=None,
                threshold="unique(symbol,date)",
                message="Duplicate logical primary key (symbol,date) detected in adjusted dataset.",
                code=FailureCode.DUPLICATE_PK.value,
            )
        )
    return findings



def load_inputs(
    prices_raw_path: Path,
    prices_adjusted_path: Optional[Path],
    calendar_path: Path,
    config: QCConfig,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, List[pd.DataFrame]]:
    findings: List[pd.DataFrame] = []

    try:
        raw = load_dataframe(prices_raw_path)
    except FileNotFoundError:
        findings.append(
            structural_failure_df(
                check_name="raw_dataset_not_found",
                severity=Severity.FAIL_STRUCTURAL.value,
                message=f"Raw dataset not found: {prices_raw_path}",
                code=FailureCode.DATASET_NOT_FOUND.value,
            )
        )
        return pd.DataFrame(), None, pd.DataFrame(), findings
    except Exception as exc:
        findings.append(
            structural_failure_df(
                check_name="raw_dataset_unreadable",
                severity=Severity.FAIL_STRUCTURAL.value,
                message=f"Failed to read raw dataset: {exc}",
                code=FailureCode.DATASET_UNREADABLE.value,
            )
        )
        return pd.DataFrame(), None, pd.DataFrame(), findings

    try:
        cal = read_calendar(calendar_path)
    except Exception as exc:
        findings.append(
            structural_failure_df(
                check_name="calendar_invalid",
                severity=Severity.FAIL_STRUCTURAL.value,
                message=f"Failed to read calendar: {exc}",
                code=FailureCode.INCOMPATIBLE_CALENDAR.value,
            )
        )
        return pd.DataFrame(), None, pd.DataFrame(), findings

    adj: Optional[pd.DataFrame] = None
    if prices_adjusted_path is not None:
        try:
            adj = load_dataframe(prices_adjusted_path)
        except Exception as exc:
            sev = Severity.FAIL_STRUCTURAL.value if config.require_adjusted_consistency_check else Severity.INFO.value
            findings.append(
                structural_failure_df(
                    check_name="adjusted_dataset_missing",
                    severity=sev,
                    message=f"Adjusted dataset could not be read: {exc}",
                    code=FailureCode.ADJUSTED_DATASET_MISSING.value,
                )
            )
            adj = None
    elif config.require_adjusted_consistency_check:
        findings.append(
            structural_failure_df(
                check_name="adjusted_dataset_missing",
                severity=Severity.FAIL_STRUCTURAL.value,
                message="Adjusted dataset is required by config but no path was provided.",
                code=FailureCode.ADJUSTED_DATASET_MISSING.value,
            )
        )
    else:
        findings.append(
            structural_failure_df(
                check_name="adjusted_dataset_missing",
                severity=Severity.INFO.value,
                message="Adjusted dataset not provided; raw-vs-adjusted block will be skipped.",
                code=FailureCode.ADJUSTED_DATASET_MISSING.value,
            )
        )

    raw_date_col = infer_date_column(raw, ["date", "trade_date", "session_date"])
    if raw_date_col is None:
        findings.append(
            structural_failure_df(
                check_name="raw_missing_date_column",
                severity=Severity.FAIL_STRUCTURAL.value,
                message="Raw dataset must contain one of: date, trade_date, session_date",
                code=FailureCode.MISSING_REQUIRED_COLUMN.value,
                observed_value="date/trade_date/session_date",
            )
        )
    else:
        try:
            raw = normalize_date_column(raw, raw_date_col, target="date")
        except Exception as exc:
            findings.append(
                structural_failure_df(
                    check_name="raw_invalid_date_parse",
                    severity=Severity.FAIL_STRUCTURAL.value,
                    message=str(exc),
                    code=FailureCode.INVALID_DATE_PARSE.value,
                )
            )

    if adj is not None:
        adj_date_col = infer_date_column(adj, ["date", "trade_date", "session_date"])
        if adj_date_col is None:
            findings.append(
                structural_failure_df(
                    check_name="adjusted_missing_date_column",
                    severity=Severity.FAIL_STRUCTURAL.value,
                    message="Adjusted dataset must contain one of: date, trade_date, session_date",
                    code=FailureCode.MISSING_REQUIRED_COLUMN.value,
                )
            )
        else:
            try:
                adj = normalize_date_column(adj, adj_date_col, target="date")
            except Exception as exc:
                findings.append(
                    structural_failure_df(
                        check_name="adjusted_invalid_date_parse",
                        severity=Severity.FAIL_STRUCTURAL.value,
                        message=str(exc),
                        code=FailureCode.INVALID_DATE_PARSE.value,
                    )
                )

    if "date" in raw.columns:
        raw = coerce_float(raw, ["open", "high", "low", "close", "volume"])
        findings.extend(validate_raw_schema(raw))

    if adj is not None and "date" in adj.columns:
        adj = coerce_float(adj, [c for c in ADJUSTED_OPTIONAL_COLUMNS if c in adj.columns])
        findings.extend(validate_adjusted_schema(adj, config.require_adjusted_consistency_check))

    return raw, adj, cal, findings


# -----------------------------------------------------------------------------
# QC checks
# -----------------------------------------------------------------------------


def prepare_raw(raw: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    cal_dates = pd.DatetimeIndex(calendar["date"].unique()).sort_values()
    out["date_in_calendar"] = out["date"].isin(cal_dates)
    out["ret_1d_raw"] = out.groupby("symbol", observed=True)["close"].pct_change()
    out["volume_filled"] = out["volume"].fillna(0.0)
    out["symbol_row_number"] = out.groupby("symbol", observed=True).cumcount() + 1
    return out



def check_intrabar_geometry(raw: pd.DataFrame, config: QCConfig) -> List[pd.DataFrame]:
    findings: List[pd.DataFrame] = []
    high_expected = raw[["open", "close", "low"]].max(axis=1)
    low_expected = raw[["open", "close", "high"]].min(axis=1)
    mask_ohlc = (raw["high"] < high_expected) | (raw["low"] > low_expected)
    findings.append(
        build_finding_rows(
            raw,
            mask_ohlc,
            check_name="ohlc_intrabar_geometry",
            severity=Severity.FAIL_ROW.value,
            observed_value=raw["high"].astype(str) + "|" + raw["low"].astype(str),
            threshold="high>=max(open,close,low) and low<=min(open,close,high)",
            message="Intrabar OHLC geometry is impossible.",
            code=FailureCode.OHLC_INVALID.value,
        )
    )

    if config.allow_nonpositive_prices:
        mask_nonpositive = pd.Series(False, index=raw.index)
    else:
        mask_nonpositive = (raw[["open", "high", "low", "close"]] <= 0).any(axis=1)
    findings.append(
        build_finding_rows(
            raw,
            mask_nonpositive,
            check_name="nonpositive_price",
            severity=Severity.FAIL_ROW.value,
            observed_value=raw[["open", "high", "low", "close"]].astype(str).agg("|".join, axis=1),
            threshold="> 0",
            message="One or more OHLC prices are non-positive.",
            code=FailureCode.NONPOSITIVE_PRICE.value,
        )
    )

    mask_negative_volume = raw["volume"] < 0
    findings.append(
        build_finding_rows(
            raw,
            mask_negative_volume,
            check_name="negative_volume",
            severity=Severity.FAIL_ROW.value,
            observed_value=raw["volume"],
            threshold=">= 0",
            message="Volume is negative.",
            code=FailureCode.NEGATIVE_VOLUME.value,
        )
    )

    mask_penny = raw["close"] < float(config.penny_price_threshold)
    findings.append(
        build_finding_rows(
            raw,
            mask_penny,
            check_name="penny_price",
            severity=Severity.INFO.value,
            observed_value=raw["close"],
            threshold=config.penny_price_threshold,
            message="Close price is below configured penny-price threshold.",
            code=FailureCode.PENNY_PRICE.value,
        )
    )

    mask_outside_calendar = ~raw["date_in_calendar"]
    findings.append(
        build_finding_rows(
            raw,
            mask_outside_calendar,
            check_name="date_outside_calendar",
            severity=Severity.FAIL_ROW.value,
            observed_value=raw["date"].astype(str),
            threshold="official_calendar",
            message="Observed price row falls outside the official trading calendar.",
            code=FailureCode.OUTSIDE_CALENDAR.value,
        )
    )
    return findings



def check_temporal_coverage(raw: pd.DataFrame, calendar: pd.DataFrame, config: QCConfig) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    findings: List[pd.DataFrame] = []
    cal_dates = pd.DatetimeIndex(calendar["date"].unique()).sort_values()
    symbol_rows: List[Dict[str, Any]] = []
    for symbol, sdf in raw.groupby("symbol", observed=True, sort=False):
        sdf = sdf.sort_values("date")
        obs_dates = pd.DatetimeIndex(pd.Series(sdf["date"].unique()).sort_values())
        start = obs_dates.min()
        end = obs_dates.max()
        expected = cal_dates[(cal_dates >= start) & (cal_dates <= end)]
        n_obs = int(len(obs_dates))
        n_exp = int(len(expected))
        missing_dates = expected.difference(obs_dates)
        pct_missing = float((n_exp - n_obs) / n_exp) if n_exp > 0 else 0.0
        bad_coverage = (n_exp >= config.min_symbol_rows_for_coverage) and (pct_missing > config.symbol_bad_coverage_threshold)
        symbol_rows.append(
            {
                "symbol": symbol,
                "start_date": start,
                "end_date": end,
                "n_rows": int(len(sdf)),
                "n_unique_dates": n_obs,
                "n_expected_sessions": n_exp,
                "n_missing_sessions": int(len(missing_dates)),
                "pct_missing_sessions_symbol": pct_missing,
                "coverage_ok": not bad_coverage,
                "missing_dates_sample": [d.strftime("%Y-%m-%d") for d in missing_dates[:10]],
            }
        )
        if len(missing_dates) > 0:
            findings.append(
                pd.DataFrame(
                    [
                        {
                            "symbol": symbol,
                            "date": pd.NaT,
                            "check_name": "missing_sessions",
                            "severity": Severity.WARN_SYMBOL.value if bad_coverage else Severity.INFO.value,
                            "observed_value": int(len(missing_dates)),
                            "threshold": config.symbol_bad_coverage_threshold,
                            "message": f"Missing {len(missing_dates)} expected sessions between {start.date()} and {end.date()}.",
                            "failure_code": FailureCode.MISSING_SESSIONS.value,
                        }
                    ]
                )
            )
        if bad_coverage:
            findings.append(
                pd.DataFrame(
                    [
                        {
                            "symbol": symbol,
                            "date": pd.NaT,
                            "check_name": "bad_symbol_coverage",
                            "severity": Severity.WARN_SYMBOL.value,
                            "observed_value": pct_missing,
                            "threshold": config.symbol_bad_coverage_threshold,
                            "message": "Symbol has poor calendar coverage over its observed life window.",
                            "failure_code": FailureCode.BAD_COVERAGE.value,
                        }
                    ]
                )
            )
    symbol_cov = pd.DataFrame(symbol_rows)
    return symbol_cov, findings



def infer_adjusted_event_candidates(adj_joined: pd.DataFrame, factor_col: str, threshold: float) -> pd.Series:
    if factor_col not in adj_joined.columns:
        return pd.Series(False, index=adj_joined.index)
    factor = adj_joined[factor_col].replace([np.inf, -np.inf], np.nan)
    prev = factor.groupby(adj_joined["symbol"], observed=True).shift(1)
    ratio = factor / prev
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    jumps = (ratio - 1.0).abs() > threshold
    return jumps.fillna(False)



def check_extreme_returns_and_adjusted(
    raw: pd.DataFrame,
    adj: Optional[pd.DataFrame],
    config: QCConfig,
    existing_row_fail_mask: pd.Series,
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    findings: List[pd.DataFrame] = []
    enriched = raw.copy()
    extreme_mask = enriched["ret_1d_raw"].abs() > float(config.extreme_return_threshold)

    adjusted_flags = pd.Series(False, index=enriched.index)
    comparable = pd.Series(False, index=enriched.index)

    if adj is not None:
        adj_small = adj.copy()
        if "ret_1d_adj_split" not in adj_small.columns and "close_adj_split" in adj_small.columns:
            adj_small = adj_small.sort_values(["symbol", "date"])
            adj_small["ret_1d_adj_split"] = adj_small.groupby("symbol", observed=True)["close_adj_split"].pct_change()
        if "ret_1d_adj_total" not in adj_small.columns and "close_adj_total" in adj_small.columns:
            adj_small = adj_small.sort_values(["symbol", "date"])
            adj_small["ret_1d_adj_total"] = adj_small.groupby("symbol", observed=True)["close_adj_total"].pct_change()
        merge_cols = [c for c in ["symbol", "date", "close_adj_split", "close_adj_total", "ret_1d_adj_split", "ret_1d_adj_total"] if c in adj_small.columns]
        merged = enriched.merge(adj_small[merge_cols], on=["symbol", "date"], how="left", validate="1:1")
        if "close_adj_split" in merged.columns:
            merged["factor_split"] = merged["close_adj_split"] / merged["close"]
            merged["split_event_candidate"] = infer_adjusted_event_candidates(
                merged, "factor_split", config.raw_adjusted_factor_jump_threshold
            )
        else:
            merged["split_event_candidate"] = False
        if "close_adj_total" in merged.columns:
            merged["factor_total"] = merged["close_adj_total"] / merged["close"]
            merged["total_event_candidate"] = infer_adjusted_event_candidates(
                merged, "factor_total", config.raw_adjusted_factor_jump_threshold
            )
        else:
            merged["total_event_candidate"] = False

        if "ret_1d_adj_split" in merged.columns:
            tol = float(config.raw_adjusted_sign_tolerance)
            raw_sign = np.sign(merged["ret_1d_raw"].where(merged["ret_1d_raw"].abs() > tol, 0.0))
            adj_sign = np.sign(merged["ret_1d_adj_split"].where(merged["ret_1d_adj_split"].abs() > tol, 0.0))
            same_sign_violation = (
                merged["ret_1d_raw"].notna()
                & merged["ret_1d_adj_split"].notna()
                & (raw_sign != adj_sign)
                & ~merged["split_event_candidate"].fillna(False)
            )
            mag_ratio = (
                (merged["ret_1d_raw"].abs() + tol) / (merged["ret_1d_adj_split"].abs() + tol)
            ).replace([np.inf, -np.inf], np.nan)
            mag_violation = (
                merged["ret_1d_raw"].notna()
                & merged["ret_1d_adj_split"].notna()
                & ~merged["split_event_candidate"].fillna(False)
                & ((mag_ratio > config.raw_adjusted_mag_ratio_threshold) | (mag_ratio < 1.0 / config.raw_adjusted_mag_ratio_threshold))
            )
            adjusted_flags = (same_sign_violation | mag_violation).fillna(False)
            comparable = merged["ret_1d_raw"].notna() & merged["ret_1d_adj_split"].notna() & ~merged["split_event_candidate"].fillna(False)
            enriched = merged
        else:
            enriched = merged

    plausible_extreme = extreme_mask & ~existing_row_fail_mask & ~adjusted_flags
    suspect_extreme = extreme_mask & (existing_row_fail_mask | adjusted_flags)

    findings.append(
        build_finding_rows(
            enriched,
            plausible_extreme,
            check_name="extreme_return_contextual",
            severity=Severity.INFO.value,
            observed_value=enriched["ret_1d_raw"],
            threshold=config.extreme_return_threshold,
            message="Extreme raw return detected but context remains economically plausible.",
            code=FailureCode.EXTREME_RETURN_PLAUSIBLE.value,
        )
    )
    findings.append(
        build_finding_rows(
            enriched,
            adjusted_flags,
            check_name="raw_adjusted_inconsistent",
            severity=Severity.FAIL_ROW.value,
            observed_value=enriched.get("ret_1d_adj_split", pd.Series(np.nan, index=enriched.index)),
            threshold="same sign / magnitude unless corporate-action candidate",
            message="Raw and adjusted returns are materially inconsistent outside inferred corporate-action dates.",
            code=FailureCode.RAW_ADJUSTED_INCONSISTENT.value,
        )
    )
    findings.append(
        build_finding_rows(
            enriched,
            suspect_extreme & ~adjusted_flags,
            check_name="extreme_return_suspect",
            severity=Severity.FAIL_ROW.value,
            observed_value=enriched["ret_1d_raw"],
            threshold=config.extreme_return_threshold,
            message="Extreme return is accompanied by other hard row failures, so it is treated as data corruption.",
            code=FailureCode.EXTREME_RETURN_SUSPECT.value,
        )
    )

    enriched["extreme_return_flag"] = extreme_mask
    enriched["raw_adjusted_inconsistent_flag"] = adjusted_flags
    enriched["raw_adjusted_comparable"] = comparable
    return findings, enriched


# -----------------------------------------------------------------------------
# Aggregation and gate logic
# -----------------------------------------------------------------------------


def aggregate_symbol_level(
    raw_enriched: pd.DataFrame,
    symbol_cov: pd.DataFrame,
    row_level: pd.DataFrame,
) -> pd.DataFrame:
    if raw_enriched.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "n_rows",
                "pct_missing_sessions_symbol",
                "pct_extreme_returns_symbol",
                "n_ohlc_violations",
                "n_nonpositive_prices",
                "n_raw_adjusted_flags",
                "severity_max",
            ]
        )

    base = raw_enriched.groupby("symbol", observed=True).agg(
        n_rows=("symbol", "size"),
        n_extreme_returns=("extreme_return_flag", "sum"),
        n_raw_adjusted_flags=("raw_adjusted_inconsistent_flag", "sum"),
    )
    base["pct_extreme_returns_symbol"] = base["n_extreme_returns"] / base["n_rows"].replace(0, np.nan)

    ohlc_counts = (
        row_level.loc[row_level["failure_code"] == FailureCode.OHLC_INVALID.value]
        .groupby("symbol", observed=True)
        .size()
        .rename("n_ohlc_violations")
    )
    nonpos_counts = (
        row_level.loc[row_level["failure_code"] == FailureCode.NONPOSITIVE_PRICE.value]
        .groupby("symbol", observed=True)
        .size()
        .rename("n_nonpositive_prices")
    )
    severity = row_level.groupby("symbol", observed=True)["severity"].agg(lambda s: pick_max_severity(s)).rename("severity_max")

    out = base.join(ohlc_counts, how="left").join(nonpos_counts, how="left").join(severity, how="left")
    out = out.reset_index()
    if not symbol_cov.empty:
        out = out.merge(symbol_cov[["symbol", "pct_missing_sessions_symbol"]], on="symbol", how="left")
    else:
        out["pct_missing_sessions_symbol"] = np.nan
    out["n_ohlc_violations"] = out["n_ohlc_violations"].fillna(0).astype(int)
    out["n_nonpositive_prices"] = out["n_nonpositive_prices"].fillna(0).astype(int)
    out["n_raw_adjusted_flags"] = out["n_raw_adjusted_flags"].fillna(0).astype(int)
    return out[
        [
            "symbol",
            "n_rows",
            "pct_missing_sessions_symbol",
            "pct_extreme_returns_symbol",
            "n_ohlc_violations",
            "n_nonpositive_prices",
            "n_raw_adjusted_flags",
            "severity_max",
        ]
    ].sort_values(["severity_max", "symbol"], ascending=[False, True], na_position="last")



def determine_gate(
    summary: MutableMapping[str, Any],
    structural_fail_count: int,
    config: QCConfig,
) -> Gate:
    if structural_fail_count > 0:
        return Gate.FAIL
    if summary["pct_fail_row"] > config.pct_fail_row_fail_threshold:
        return Gate.FAIL
    if summary["pct_symbols_bad_coverage"] > config.pct_symbols_bad_coverage_fail_threshold:
        return Gate.FAIL
    if summary["pct_raw_adjusted_inconsistent"] > config.pct_raw_adjusted_inconsistent_fail_threshold:
        return Gate.FAIL
    if summary["warn_count"] > 0 or summary["info_count"] > 0:
        return Gate.WARN
    return Gate.PASS


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def run_qc_prices(
    *,
    prices_raw_path: Path,
    prices_adjusted_path: Optional[Path],
    calendar_path: Path,
    config_path: Path,
    output_dir: Path,
    run_id: str,
    as_of_ts_utc: str,
) -> RunArtifacts:
    started = time.perf_counter()
    config = load_config(config_path)
    config_sha = config_hash(config)

    raw, adj, calendar, structural_findings = load_inputs(prices_raw_path, prices_adjusted_path, calendar_path, config)
    row_frames: List[pd.DataFrame] = [f for f in structural_findings if not f.empty]

    structural_fail_count = int(
        sum((f["severity"] == Severity.FAIL_STRUCTURAL.value).sum() for f in row_frames if "severity" in f.columns)
    )

    raw_enriched = pd.DataFrame()
    symbol_cov = pd.DataFrame()

    if structural_fail_count == 0:
        raw_enriched = prepare_raw(raw, calendar)
        row_frames.extend([f for f in check_intrabar_geometry(raw_enriched, config) if not f.empty])
        current_row_flags = concat_findings(row_frames)
        existing_row_fail_mask = pd.Series(False, index=raw_enriched.index)
        if not current_row_flags.empty:
            hard = current_row_flags.loc[current_row_flags["severity"] == Severity.FAIL_ROW.value, ["symbol", "date"]].drop_duplicates()
            if not hard.empty:
                existing_row_fail_mask = pd.MultiIndex.from_frame(raw_enriched[["symbol", "date"]]).isin(pd.MultiIndex.from_frame(hard))
                existing_row_fail_mask = pd.Series(existing_row_fail_mask, index=raw_enriched.index)

        symbol_cov, coverage_findings = check_temporal_coverage(raw_enriched, calendar, config)
        row_frames.extend([f for f in coverage_findings if not f.empty])

        extreme_findings, raw_enriched = check_extreme_returns_and_adjusted(raw_enriched, adj, config, existing_row_fail_mask)
        row_frames.extend([f for f in extreme_findings if not f.empty])

        # Promote repeated extremes under poor coverage to WARN_SYMBOL.
        if not symbol_cov.empty:
            symbol_extreme = (
                raw_enriched.groupby("symbol", observed=True)["extreme_return_flag"].mean().rename("pct_extreme_returns_symbol")
            )
            merged_warn = symbol_cov.merge(symbol_extreme.reset_index(), on="symbol", how="left")
            warn_symbols = merged_warn.loc[
                (merged_warn["pct_extreme_returns_symbol"].fillna(0) > config.symbol_extreme_return_warn_threshold)
                | (~merged_warn["coverage_ok"].fillna(True))
            ]
            if not warn_symbols.empty:
                row_frames.append(
                    pd.DataFrame(
                        {
                            "symbol": warn_symbols["symbol"],
                            "date": pd.NaT,
                            "check_name": "symbol_quality_degraded",
                            "severity": Severity.WARN_SYMBOL.value,
                            "observed_value": warn_symbols["pct_extreme_returns_symbol"].fillna(0),
                            "threshold": config.symbol_extreme_return_warn_threshold,
                            "message": "Symbol exhibits repeated extreme returns and/or poor calendar coverage.",
                            "failure_code": FailureCode.BAD_COVERAGE.value,
                        }
                    )
                )

    row_level = concat_findings(row_frames)
    if not row_level.empty and "date" in row_level.columns:
        row_level["date"] = pd.to_datetime(row_level["date"], errors="coerce")
        row_level = row_level.sort_values(["severity", "symbol", "date", "check_name"], ascending=[False, True, True, True], na_position="last")
        row_level = row_level.reset_index(drop=True)

    symbol_level = aggregate_symbol_level(raw_enriched, symbol_cov, row_level)

    n_rows = int(len(raw)) if not raw.empty else 0
    n_symbols = int(raw["symbol"].nunique()) if (not raw.empty and "symbol" in raw.columns) else 0
    fail_row_count = int((row_level["severity"] == Severity.FAIL_ROW.value).sum()) if not row_level.empty else 0
    warn_count = int((row_level["severity"] == Severity.WARN_SYMBOL.value).sum()) if not row_level.empty else 0
    info_count = int((row_level["severity"] == Severity.INFO.value).sum()) if not row_level.empty else 0
    fail_count = fail_row_count + structural_fail_count

    pct_rows_ohlc_invalid = 0.0
    pct_extreme_returns = 0.0
    pct_symbols_bad_coverage = 0.0
    pct_raw_adjusted_inconsistent = 0.0
    pct_fail_row = 0.0
    if n_rows > 0:
        pct_rows_ohlc_invalid = float((row_level["failure_code"] == FailureCode.OHLC_INVALID.value).sum() / n_rows)
        pct_fail_row = float(fail_row_count / n_rows)
    if not raw_enriched.empty:
        returns_denom = int(raw_enriched["ret_1d_raw"].notna().sum())
        if returns_denom > 0:
            pct_extreme_returns = float(raw_enriched["extreme_return_flag"].sum() / returns_denom)
        comparable = int(raw_enriched["raw_adjusted_comparable"].sum()) if "raw_adjusted_comparable" in raw_enriched.columns else 0
        if comparable > 0:
            pct_raw_adjusted_inconsistent = float(raw_enriched["raw_adjusted_inconsistent_flag"].sum() / comparable)
    if n_symbols > 0 and not symbol_cov.empty:
        pct_symbols_bad_coverage = float((~symbol_cov["coverage_ok"]).sum() / n_symbols)

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "as_of_ts_utc": as_of_ts_utc,
        "n_rows": n_rows,
        "n_symbols": n_symbols,
        "pct_rows_ohlc_invalid": pct_rows_ohlc_invalid,
        "pct_extreme_returns": pct_extreme_returns,
        "pct_symbols_bad_coverage": pct_symbols_bad_coverage,
        "pct_raw_adjusted_inconsistent": pct_raw_adjusted_inconsistent,
        "pct_fail_row": pct_fail_row,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "info_count": info_count,
        "structural_fail_count": structural_fail_count,
    }
    summary["gate"] = determine_gate(summary, structural_fail_count, config).value

    failures = row_level.loc[row_level["severity"].isin([Severity.FAIL_STRUCTURAL.value, Severity.FAIL_ROW.value])].copy()

    duration_sec = round(time.perf_counter() - started, 6)
    manifest = {
        "run_id": run_id,
        "as_of_ts_utc": as_of_ts_utc,
        "config_hash": config_sha,
        "prices_raw_snapshot": str(prices_raw_path),
        "prices_raw_sha256": sha256_of_file(prices_raw_path) if prices_raw_path.is_file() else None,
        "prices_adjusted_snapshot": str(prices_adjusted_path) if prices_adjusted_path is not None else None,
        "prices_adjusted_sha256": sha256_of_file(prices_adjusted_path) if prices_adjusted_path and prices_adjusted_path.is_file() else None,
        "calendar_snapshot": str(calendar_path),
        "calendar_sha256": sha256_of_file(calendar_path) if calendar_path.is_file() else None,
        "config_snapshot": str(config_path),
        "config_sha256": sha256_of_file(config_path) if config_path.is_file() else None,
        "code_version": config.code_version,
        "git_commit": git_commit(),
        "processing_duration_sec": duration_sec,
        "start_date": str(raw["date"].min().date()) if (not raw.empty and "date" in raw.columns) else None,
        "end_date": str(raw["date"].max().date()) if (not raw.empty and "date" in raw.columns) else None,
        "n_rows": n_rows,
        "n_symbols": n_symbols,
        "gate": summary["gate"],
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "generated_at_utc": utc_now_iso(),
        "outputs": {
            "qc_summary": str(output_dir / "qc_summary.json"),
            "qc_symbol_level": str(output_dir / "qc_symbol_level.parquet"),
            "qc_row_level": str(output_dir / "qc_row_level.parquet"),
            "qc_failures": str(output_dir / "qc_failures.parquet"),
            "manifest": str(output_dir / "manifest.json"),
        },
    }

    return RunArtifacts(
        row_level=row_level,
        symbol_level=symbol_level,
        failures=failures,
        summary=summary,
        manifest=manifest,
    )



def persist_artifacts(artifacts: RunArtifacts, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifacts.summary, output_dir / "qc_summary.json")
    write_parquet(artifacts.symbol_level, output_dir / "qc_symbol_level.parquet")
    write_parquet(artifacts.row_level, output_dir / "qc_row_level.parquet")
    write_parquet(artifacts.failures, output_dir / "qc_failures.parquet")
    write_json(artifacts.manifest, output_dir / "manifest.json")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quality-control daily price series for structural, temporal and economic anomalies.")
    p.add_argument("--prices-raw-path", required=True, type=Path)
    p.add_argument("--prices-adjusted-path", required=False, type=Path, default=None)
    p.add_argument("--calendar-path", required=True, type=Path)
    p.add_argument("--config-path", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--run-id", required=True)
    p.add_argument("--as-of-ts-utc", required=True)
    p.add_argument("--log-level", default="INFO")
    return p



def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        artifacts = run_qc_prices(
            prices_raw_path=args.prices_raw_path,
            prices_adjusted_path=args.prices_adjusted_path,
            calendar_path=args.calendar_path,
            config_path=args.config_path,
            output_dir=args.output_dir,
            run_id=args.run_id,
            as_of_ts_utc=args.as_of_ts_utc,
        )
        persist_artifacts(artifacts, args.output_dir)
        logging.info("qc_prices finished with gate=%s", artifacts.summary["gate"])
        return 0 if artifacts.summary["gate"] != Gate.FAIL.value else 2
    except QCPricesError as exc:
        logging.error("qc_prices failed: %s", exc)
        return 2
    except Exception as exc:  # pragma: no cover
        logging.exception("Unexpected error in qc_prices: %s", exc)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
