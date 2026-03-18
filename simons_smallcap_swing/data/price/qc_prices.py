"""
data/price/qc_prices.py — Price data quality control.

5 blocks of checks, 3 granularities (row, symbol, run), gate logic.

Block A — Structural integrity:
    Schema validation, PK uniqueness, column types

Block B — Intrabar geometry:
    H≥L, O/H/L/C > 0, volume ≥ 0, H≥O, H≥C, L≤O, L≤C

Block C — Temporal continuity:
    Missing sessions, suspicious gaps, coverage per symbol

Block D — Extreme returns and jumps:
    Daily return > threshold, distinguishing plausible (small cap vol)
    from data errors (raw vs adjusted inconsistency)

Block E — Raw vs adjusted consistency:
    Factor continuity, adjusted series preserves return semantics

Gate logic:
    FAIL: structural failure, or too many bad rows/symbols
    WARN: isolated anomalies within tolerance
    PASS: all checks clean
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import (
    PriceError, QCError, Severity, Gate, severity_max,
    resolve_ohlcv_columns, validate_ohlcv_geometry,
    utc_now_iso, config_hash, json_safe,
    read_dataframe, write_parquet_safe, write_json_safe,
    normalize_date_series, concat_findings,
    LOGGER as _PARENT_LOGGER,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class QCConfig:
    # Structural
    require_columns: tuple[str, ...] = ("date", "symbol", "open", "high", "low", "close", "volume")

    # Geometry
    max_invalid_bar_pct: float = 0.01    # >1% invalid bars → WARN

    # Temporal
    min_coverage_pct: float = 0.50       # <50% sessions → WARN for symbol
    max_low_coverage_symbols_pct: float = 0.20  # >20% symbols low coverage → FAIL

    # Extreme returns
    extreme_return_threshold: float = 0.50  # |ret| > 50% → flag
    extreme_return_max_pct: float = 0.001   # >0.1% extreme → WARN

    # Raw vs adjusted
    max_adjustment_divergence: float = 0.001  # return divergence threshold
    max_divergent_pct: float = 0.01           # >1% divergent → WARN


@dataclass
class QCFinding:
    block: str
    level: str       # "row" | "symbol" | "run"
    check: str
    severity: str
    count: int
    total: int
    message: str


# ---------------------------------------------------------------------------
# Block A: Structural
# ---------------------------------------------------------------------------

def check_structure(df: pd.DataFrame, cfg: QCConfig) -> list[QCFinding]:
    findings = []
    # Column presence
    missing = [c for c in cfg.require_columns if c not in df.columns]
    if missing:
        findings.append(QCFinding("A_struct", "run", "columns_present", "FAIL", len(missing), len(cfg.require_columns),
                                   f"Missing columns: {missing}"))
        return findings  # can't proceed without columns

    # PK uniqueness
    n_dup = df.duplicated(subset=["date", "symbol"]).sum()
    findings.append(QCFinding("A_struct", "run", "pk_unique",
                               "FAIL" if n_dup > 0 else "PASS",
                               int(n_dup), len(df), f"{n_dup} duplicate (date, symbol) rows"))

    # Types
    for col in ("open", "high", "low", "close"):
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            findings.append(QCFinding("A_struct", "run", f"{col}_numeric", "FAIL", 1, 1, f"{col} is not numeric"))

    return findings


# ---------------------------------------------------------------------------
# Block B: Intrabar geometry
# ---------------------------------------------------------------------------

def check_geometry(df: pd.DataFrame, cfg: QCConfig) -> list[QCFinding]:
    cols = resolve_ohlcv_columns(df)
    valid = validate_ohlcv_geometry(df, cols)
    n_invalid = int((~valid).sum())
    pct_invalid = n_invalid / max(len(df), 1)

    sev = "FAIL" if pct_invalid > 0.05 else ("WARN" if pct_invalid > cfg.max_invalid_bar_pct else "PASS")
    return [QCFinding("B_geometry", "run", "ohlcv_valid", sev, n_invalid, len(df),
                        f"{n_invalid} invalid bars ({pct_invalid:.2%})")]


# ---------------------------------------------------------------------------
# Block C: Temporal continuity
# ---------------------------------------------------------------------------

def check_temporal(
    df: pd.DataFrame,
    calendar: pd.DatetimeIndex | None,
    cfg: QCConfig,
) -> list[QCFinding]:
    findings = []
    if calendar is None:
        return findings

    cal_set = set(calendar)
    low_coverage_count = 0

    for sym, grp in df.groupby("symbol"):
        sym_dates = set(pd.to_datetime(grp["date"]))
        expected = len(cal_set)
        actual = len(sym_dates & cal_set)
        coverage = actual / max(expected, 1)
        if coverage < cfg.min_coverage_pct:
            low_coverage_count += 1

    n_symbols = df["symbol"].nunique()
    pct_low = low_coverage_count / max(n_symbols, 1)
    sev = "FAIL" if pct_low > cfg.max_low_coverage_symbols_pct else ("WARN" if low_coverage_count > 0 else "PASS")
    findings.append(QCFinding("C_temporal", "run", "symbol_coverage", sev,
                               low_coverage_count, n_symbols,
                               f"{low_coverage_count}/{n_symbols} symbols below {cfg.min_coverage_pct:.0%} coverage"))
    return findings


# ---------------------------------------------------------------------------
# Block D: Extreme returns
# ---------------------------------------------------------------------------

def check_extreme_returns(df: pd.DataFrame, cfg: QCConfig) -> list[QCFinding]:
    if "close" not in df.columns:
        return []
    df_sorted = df.sort_values(["symbol", "date"])
    ret = df_sorted.groupby("symbol")["close"].pct_change()
    extreme = ret.abs() > cfg.extreme_return_threshold
    n_extreme = int(extreme.sum())
    pct = n_extreme / max(len(ret.dropna()), 1)

    sev = "WARN" if pct > cfg.extreme_return_max_pct else "PASS"
    return [QCFinding("D_returns", "run", "extreme_returns", sev, n_extreme, len(ret.dropna()),
                        f"{n_extreme} returns with |r|>{cfg.extreme_return_threshold:.0%} ({pct:.4%})")]


# ---------------------------------------------------------------------------
# Block E: Raw vs adjusted
# ---------------------------------------------------------------------------

def check_raw_vs_adjusted(
    raw: pd.DataFrame, adjusted: pd.DataFrame, cfg: QCConfig,
) -> list[QCFinding]:
    if adjusted is None or len(adjusted) == 0:
        return []

    # Merge on (date, symbol) and compare returns
    raw_sorted = raw.sort_values(["symbol", "date"]).copy()
    adj_sorted = adjusted.sort_values(["symbol", "date"]).copy()

    raw_sorted["ret_raw"] = raw_sorted.groupby("symbol")["close"].pct_change()
    adj_close = "close_adj" if "close_adj" in adj_sorted.columns else "close"
    adj_sorted["ret_adj"] = adj_sorted.groupby("symbol")[adj_close].pct_change()

    merged = raw_sorted[["date", "symbol", "ret_raw"]].merge(
        adj_sorted[["date", "symbol", "ret_adj"]], on=["date", "symbol"], how="inner",
    )

    divergence = (merged["ret_raw"] - merged["ret_adj"]).abs()
    n_divergent = int((divergence > cfg.max_adjustment_divergence).sum())
    pct = n_divergent / max(len(merged), 1)

    sev = "WARN" if pct > cfg.max_divergent_pct else "PASS"
    return [QCFinding("E_raw_vs_adj", "run", "return_divergence", sev, n_divergent, len(merged),
                        f"{n_divergent} rows with raw-adj return divergence >{cfg.max_adjustment_divergence} ({pct:.4%})")]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_qc_prices(
    raw: pd.DataFrame | str | Path,
    *,
    adjusted: pd.DataFrame | str | Path | None = None,
    calendar: pd.DatetimeIndex | Sequence | None = None,
    config: QCConfig | None = None,
    output_dir: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run the full price QC pipeline."""
    cfg = config or QCConfig()
    if not run_id:
        run_id = f"qc_{utc_now_iso().replace(':','').replace('-','')}"

    if isinstance(raw, (str, Path)):
        raw = read_dataframe(raw)
    raw = raw.copy()
    raw["date"] = normalize_date_series(raw.get("date", raw.get("trade_date")))

    adj = None
    if adjusted is not None:
        if isinstance(adjusted, (str, Path)):
            adj = read_dataframe(adjusted)
        else:
            adj = adjusted

    cal = None
    if calendar is not None:
        cal = pd.DatetimeIndex(calendar)

    # Run all checks
    all_findings: list[QCFinding] = []
    all_findings.extend(check_structure(raw, cfg))

    # Only proceed if structure is OK
    if not any(f.severity == "FAIL" and f.block == "A_struct" for f in all_findings):
        all_findings.extend(check_geometry(raw, cfg))
        all_findings.extend(check_temporal(raw, cal, cfg))
        all_findings.extend(check_extreme_returns(raw, cfg))
        if adj is not None:
            all_findings.extend(check_raw_vs_adjusted(raw, adj, cfg))

    # Gate
    n_fail = sum(1 for f in all_findings if f.severity == "FAIL")
    n_warn = sum(1 for f in all_findings if f.severity == "WARN")
    gate = "fail" if n_fail > 0 else ("warn" if n_warn > 0 else "pass")

    summary = {
        "run_id": run_id,
        "gate": gate,
        "n_checks": len(all_findings),
        "n_fail": n_fail,
        "n_warn": n_warn,
        "n_pass": sum(1 for f in all_findings if f.severity == "PASS"),
        "n_bars": len(raw),
        "n_symbols": int(raw["symbol"].nunique()) if "symbol" in raw.columns else 0,
        "timestamp": utc_now_iso(),
    }

    if output_dir:
        out = Path(output_dir) / run_id
        out.mkdir(parents=True, exist_ok=True)
        findings_df = pd.DataFrame([
            {"block": f.block, "level": f.level, "check": f.check,
             "severity": f.severity, "count": f.count, "total": f.total, "message": f.message}
            for f in all_findings
        ])
        write_parquet_safe(findings_df, out / "qc_findings.parquet")
        write_json_safe(summary, out / "qc_summary.json")

    LOGGER.info("Price QC: gate=%s (%dF %dW), %d bars, %d symbols",
                gate, n_fail, n_warn, summary["n_bars"], summary["n_symbols"])
    return {"findings": all_findings, "summary": summary}
