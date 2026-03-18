"""
ops/data_health.py - Contract-first data quality governance and gating.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetContract:
    name: str
    dataset_type: str
    path_or_table: str | None = None
    criticality: float = 1.0
    expected_schema: Mapping[str, Any] = field(default_factory=dict)
    freshness_sla_hours: float = 48.0
    required_checks: tuple[str, ...] = (
        "availability",
        "schema",
        "freshness",
        "completeness",
        "temporal_consistency",
        "plausibility",
    )
    gating_policy: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataHealthConfig:
    dataset_warn_threshold: float = 0.80
    dataset_fail_threshold: float = 0.55
    global_warn_threshold: float = 0.82
    default_min_coverage: float = 0.95
    blocking_criticality: float = 0.80
    max_abs_return_warn: float = 0.40
    max_abs_return_fail: float = 0.70
    robust_z_warn: float = 10.0
    robust_z_fail: float = 14.0
    outlier_rate_warn: float = 0.05
    outlier_rate_fail: float = 0.12
    insufficient_sample_min_rows: int = 30
    drift_shift_warn: float = 0.50
    drift_shift_fail: float = 1.00


@dataclass(frozen=True)
class CheckResult:
    dataset_name: str
    check_name: str
    check_type: str
    status: str
    observed_value: Any
    threshold: Any | None
    severity: str
    hard_fail: bool
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetHealthResult:
    dataset_name: str
    dataset_type: str
    criticality: float
    status: str
    score: float | None
    hard_fail: bool
    checks: tuple[CheckResult, ...]
    observed_at: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataHealthRunResult:
    run_id: str
    execution_date: str
    global_gate: str
    global_score: float | None
    run_status: str
    datasets: tuple[DatasetHealthResult, ...]
    overrides_applied: tuple[Mapping[str, Any], ...]
    started_at: str
    ended_at: str


def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_id_with_prefix(prefix: str) -> str:
    from datetime import datetime, timezone

    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if np.isnan(v) or np.isinf(v) else v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat() if pd.notna(value) else None
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def config_hash(cfg: Mapping[str, Any], n: int = 24) -> str:
    import hashlib
    import json

    blob = json.dumps(json_safe(dict(cfg)), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:n]


def _to_df(obj: pd.DataFrame | str | Path | None) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, (str, Path)):
        p = Path(obj)
        if not p.exists():
            return pd.DataFrame()
        if p.suffix.lower() == ".parquet":
            return pd.read_parquet(p)
        if p.suffix.lower() in {".json", ".jsonl"}:
            return pd.read_json(p)
        return pd.read_csv(p)
    return obj.copy()


def _parse_maybe_json(x: Any) -> Any:
    import json

    if not isinstance(x, str):
        return x
    txt = x.strip()
    if not txt:
        return {}
    if (txt.startswith("{") and txt.endswith("}")) or (txt.startswith("[") and txt.endswith("]")):
        try:
            return json.loads(txt)
        except Exception:
            return x
    return x


def _normalize_contracts(contracts: Sequence[Any] | pd.DataFrame | str | Path) -> list[DatasetContract]:
    raw: list[dict[str, Any]] = []

    if isinstance(contracts, (str, Path)):
        p = Path(contracts)
        if p.suffix.lower() == ".parquet":
            cdf = pd.read_parquet(p)
        elif p.suffix.lower() in {".json", ".jsonl"}:
            cdf = pd.read_json(p)
        else:
            cdf = pd.read_csv(p)
        raw = cdf.to_dict(orient="records")
    elif isinstance(contracts, pd.DataFrame):
        raw = contracts.to_dict(orient="records")
    else:
        for c in contracts:
            if isinstance(c, DatasetContract):
                raw.append(asdict(c))
            elif isinstance(c, Mapping):
                raw.append(dict(c))
            else:
                raise TypeError("contracts must be DatasetContract, mapping, dataframe, or path")

    out: list[DatasetContract] = []
    for row in raw:
        expected_schema = _parse_maybe_json(row.get("expected_schema", {}))
        gating_policy = _parse_maybe_json(row.get("gating_policy", {}))
        required_checks = row.get("required_checks", DatasetContract.required_checks)
        if isinstance(required_checks, str):
            rc = _parse_maybe_json(required_checks)
            if isinstance(rc, list):
                required_checks = tuple(str(x) for x in rc)
            else:
                required_checks = tuple(x.strip() for x in str(required_checks).split(",") if x.strip())
        out.append(
            DatasetContract(
                name=str(row["name"]),
                dataset_type=str(row.get("dataset_type", "generic")),
                path_or_table=(None if pd.isna(row.get("path_or_table")) else str(row.get("path_or_table"))),
                criticality=float(row.get("criticality", 1.0)),
                expected_schema=(expected_schema if isinstance(expected_schema, Mapping) else {}),
                freshness_sla_hours=float(row.get("freshness_sla_hours", 48.0)),
                required_checks=tuple(str(x) for x in required_checks),
                gating_policy=(gating_policy if isinstance(gating_policy, Mapping) else {}),
            )
        )

    names = [c.name for c in out]
    if len(names) != len(set(names)):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate dataset contract names: {dup}")
    return out


def _normalize_overrides(overrides: Sequence[Mapping[str, Any]] | pd.DataFrame | str | Path | None) -> list[dict[str, Any]]:
    if overrides is None:
        return []
    if isinstance(overrides, (str, Path)):
        p = Path(overrides)
        if p.suffix.lower() == ".parquet":
            odf = pd.read_parquet(p)
        elif p.suffix.lower() in {".json", ".jsonl"}:
            odf = pd.read_json(p)
        else:
            odf = pd.read_csv(p)
        rows = odf.to_dict(orient="records")
    elif isinstance(overrides, pd.DataFrame):
        rows = overrides.to_dict(orient="records")
    else:
        rows = [dict(x) for x in overrides]

    out = []
    for r in rows:
        rr = dict(r)
        for k in ["created_at", "expires_at"]:
            if k in rr and rr[k] not in (None, "", np.nan):
                rr[k] = pd.Timestamp(rr[k])
            else:
                rr[k] = None
        out.append(rr)
    return out


def _status_severity(status: str) -> str:
    s = status.upper()
    if s in {"FAIL", "ERROR"}:
        return "high"
    if s in {"WARN", "INSUFFICIENT_SAMPLE"}:
        return "warning"
    return "info"


def _mk_check(
    dataset_name: str,
    check_name: str,
    check_type: str,
    status: str,
    observed_value: Any,
    threshold: Any | None,
    hard_fail: bool,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> CheckResult:
    return CheckResult(
        dataset_name=dataset_name,
        check_name=check_name,
        check_type=check_type,
        status=status.upper(),
        observed_value=json_safe(observed_value),
        threshold=json_safe(threshold),
        severity=_status_severity(status),
        hard_fail=bool(hard_fail),
        message=str(message),
        details=(details or {}),
    )


def _check_availability(df: pd.DataFrame, contract: DatasetContract) -> CheckResult:
    if df is None or len(df) == 0:
        return _mk_check(
            contract.name,
            "availability",
            "availability",
            "FAIL",
            0,
            1,
            True,
            "dataset missing or empty",
            {"rows": 0},
        )
    return _mk_check(
        contract.name,
        "availability",
        "availability",
        "PASS",
        len(df),
        1,
        False,
        "dataset readable",
        {"rows": int(len(df))},
    )


def _check_schema(df: pd.DataFrame, contract: DatasetContract) -> CheckResult:
    schema = dict(contract.expected_schema)
    required_cols = list(schema.get("required_columns", []))
    dtypes = dict(schema.get("dtypes", {}))
    unique_keys = list(schema.get("unique_keys", []))

    if len(required_cols) == 0 and len(dtypes) == 0 and len(unique_keys) == 0:
        return _mk_check(
            contract.name,
            "schema",
            "structural_consistency",
            "PASS",
            1.0,
            None,
            False,
            "no strict schema constraints",
            {},
        )

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return _mk_check(
            contract.name,
            "schema",
            "structural_consistency",
            "FAIL",
            missing,
            required_cols,
            True,
            "missing required columns",
            {"missing_columns": missing},
        )

    dtype_mismatch: list[dict[str, str]] = []
    for c, exp in dtypes.items():
        if c not in df.columns:
            dtype_mismatch.append({"column": c, "expected": str(exp), "observed": "missing"})
            continue
        obs = str(df[c].dtype)
        if str(exp).lower() not in obs.lower():
            dtype_mismatch.append({"column": c, "expected": str(exp), "observed": obs})

    if dtype_mismatch:
        return _mk_check(
            contract.name,
            "schema_dtypes",
            "structural_consistency",
            "WARN",
            len(dtype_mismatch),
            0,
            False,
            "dtype mismatches detected",
            {"dtype_mismatch": dtype_mismatch},
        )

    if unique_keys:
        key_cols = [c for c in unique_keys if c in df.columns]
        if len(key_cols) == len(unique_keys):
            dup = int(df.duplicated(subset=key_cols, keep=False).sum())
            if dup > 0:
                return _mk_check(
                    contract.name,
                    "schema_unique_keys",
                    "structural_consistency",
                    "FAIL",
                    dup,
                    0,
                    True,
                    "duplicate keys detected",
                    {"unique_keys": key_cols, "duplicate_rows": dup},
                )

    return _mk_check(
        contract.name,
        "schema",
        "structural_consistency",
        "PASS",
        1.0,
        1.0,
        False,
        "schema valid",
        {},
    )


def _check_freshness(df: pd.DataFrame, contract: DatasetContract, execution_ts: pd.Timestamp) -> CheckResult:
    ts_col = str(contract.expected_schema.get("timestamp_col", "date"))
    if ts_col not in df.columns:
        return _mk_check(
            contract.name,
            "freshness",
            "freshness",
            "WARN",
            None,
            contract.freshness_sla_hours,
            False,
            f"timestamp column not found: {ts_col}",
            {"timestamp_col": ts_col},
        )

    ts = pd.to_datetime(df[ts_col], errors="coerce")
    if ts.notna().sum() == 0:
        return _mk_check(
            contract.name,
            "freshness",
            "freshness",
            "FAIL",
            None,
            contract.freshness_sla_hours,
            bool(contract.gating_policy.get("freshness_hard_fail", False)),
            "timestamp values not parseable",
            {"timestamp_col": ts_col},
        )

    last_update = pd.Timestamp(ts.max())
    delta_h = float((execution_ts - last_update).total_seconds() / 3600.0)
    sla = float(contract.freshness_sla_hours)
    fail_mult = float(contract.gating_policy.get("freshness_fail_multiplier", 1.5))
    hard = bool(contract.gating_policy.get("freshness_hard_fail", False))

    if delta_h > sla * fail_mult:
        return _mk_check(
            contract.name,
            "freshness",
            "freshness",
            "FAIL",
            delta_h,
            sla,
            hard,
            "freshness SLA severely breached",
            {"last_update": str(last_update), "delta_hours": delta_h},
        )
    if delta_h > sla:
        return _mk_check(
            contract.name,
            "freshness",
            "freshness",
            "WARN",
            delta_h,
            sla,
            False,
            "freshness SLA breached",
            {"last_update": str(last_update), "delta_hours": delta_h},
        )

    return _mk_check(
        contract.name,
        "freshness",
        "freshness",
        "PASS",
        delta_h,
        sla,
        False,
        "freshness within SLA",
        {"last_update": str(last_update), "delta_hours": delta_h},
    )


def _check_completeness(df: pd.DataFrame, contract: DatasetContract, cfg: DataHealthConfig) -> CheckResult:
    gp = dict(contract.gating_policy)
    min_cov = float(gp.get("min_coverage", cfg.default_min_coverage))
    exp_rows = gp.get("expected_rows")

    if exp_rows is None:
        # If contract does not specify expected rows, use current rows as neutral baseline.
        exp_rows = int(len(df))

    exp_rows = max(int(exp_rows), 1)
    coverage = float(len(df) / exp_rows)

    hard = bool(gp.get("completeness_hard_fail", False))
    fail_mult = float(gp.get("completeness_fail_multiplier", 0.80))

    if coverage < min_cov * fail_mult:
        return _mk_check(
            contract.name,
            "completeness",
            "completeness",
            "FAIL",
            coverage,
            min_cov,
            hard,
            "coverage materially below minimum",
            {"observed_rows": int(len(df)), "expected_rows": exp_rows},
        )

    if coverage < min_cov:
        return _mk_check(
            contract.name,
            "completeness",
            "completeness",
            "WARN",
            coverage,
            min_cov,
            False,
            "coverage below minimum",
            {"observed_rows": int(len(df)), "expected_rows": exp_rows},
        )

    return _mk_check(
        contract.name,
        "completeness",
        "completeness",
        "PASS",
        coverage,
        min_cov,
        False,
        "coverage acceptable",
        {"observed_rows": int(len(df)), "expected_rows": exp_rows},
    )

def _check_temporal_consistency(df: pd.DataFrame, contract: DatasetContract, execution_ts: pd.Timestamp) -> CheckResult:
    ts_col = str(contract.expected_schema.get("timestamp_col", "date"))
    if ts_col not in df.columns:
        return _mk_check(
            contract.name,
            "temporal_consistency",
            "temporal_consistency",
            "SKIPPED",
            None,
            None,
            False,
            "timestamp column unavailable",
            {"timestamp_col": ts_col},
        )

    ts = pd.to_datetime(df[ts_col], errors="coerce")
    parse_ok = int(ts.notna().sum())
    if parse_ok == 0:
        return _mk_check(
            contract.name,
            "temporal_consistency",
            "temporal_consistency",
            "FAIL",
            0,
            1,
            True,
            "all timestamps invalid",
            {"timestamp_col": ts_col},
        )

    future_count = int((ts > execution_ts).sum())
    monotonic = bool(ts.dropna().is_monotonic_increasing)

    key_cols = list(contract.expected_schema.get("time_unique_keys", []))
    dup_count = 0
    if key_cols:
        cols = [c for c in key_cols if c in df.columns]
        if len(cols) == len(key_cols):
            dup_count = int(df.duplicated(subset=cols + [ts_col], keep=False).sum())

    if future_count > 0:
        return _mk_check(
            contract.name,
            "temporal_consistency",
            "temporal_consistency",
            "FAIL",
            future_count,
            0,
            True,
            "future timestamps detected",
            {"future_count": future_count, "timestamp_col": ts_col},
        )

    if dup_count > 0 or (not monotonic):
        return _mk_check(
            contract.name,
            "temporal_consistency",
            "temporal_consistency",
            "WARN",
            {"duplicates": dup_count, "monotonic": monotonic},
            {"duplicates": 0, "monotonic": True},
            False,
            "temporal consistency degraded",
            {"duplicates": dup_count, "monotonic": monotonic},
        )

    return _mk_check(
        contract.name,
        "temporal_consistency",
        "temporal_consistency",
        "PASS",
        1.0,
        1.0,
        False,
        "temporal consistency ok",
        {"parseable_timestamps": parse_ok},
    )


def _robust_zscore(x: pd.Series) -> pd.Series:
    vals = pd.to_numeric(x, errors="coerce")
    med = float(np.nanmedian(vals.values)) if np.isfinite(vals.values).any() else 0.0
    mad = float(np.nanmedian(np.abs(vals.values - med))) if np.isfinite(vals.values).any() else 0.0
    scale = 1.4826 * mad if mad > 0 else 1.0
    return np.abs((vals - med) / scale)


def _check_plausibility(df: pd.DataFrame, contract: DatasetContract, cfg: DataHealthConfig) -> CheckResult:
    dset_type = contract.dataset_type.lower()
    gp = dict(contract.gating_policy)

    if dset_type in {"price", "prices"}:
        price_col = str(gp.get("price_col", contract.expected_schema.get("price_col", "close")))
        if price_col in df.columns:
            px = pd.to_numeric(df[price_col], errors="coerce")
            bad = int((px <= 0).sum())
            if bad > 0:
                return _mk_check(
                    contract.name,
                    "plausibility_price_positive",
                    "plausibility",
                    "FAIL",
                    bad,
                    0,
                    True,
                    "non-positive prices detected",
                    {"price_col": price_col, "bad_rows": bad},
                )

    # Generic return-range check if return-like columns exist.
    ret_cols = [c for c in df.columns if str(c).lower() in {"return", "ret", "r", "return_1d"}]
    if ret_cols:
        s = pd.to_numeric(df[ret_cols[0]], errors="coerce")
        max_abs = float(np.nanmax(np.abs(s.values))) if s.notna().any() else 0.0
        if max_abs > cfg.max_abs_return_fail:
            return _mk_check(
                contract.name,
                "plausibility_return_range",
                "plausibility",
                "FAIL",
                max_abs,
                cfg.max_abs_return_fail,
                False,
                "return range exceeds fail threshold",
                {"column": ret_cols[0]},
            )
        if max_abs > cfg.max_abs_return_warn:
            return _mk_check(
                contract.name,
                "plausibility_return_range",
                "plausibility",
                "WARN",
                max_abs,
                cfg.max_abs_return_warn,
                False,
                "return range exceeds warn threshold",
                {"column": ret_cols[0]},
            )

    # Generic robust outlier-rate check.
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    ignored = {"id", "symbol", "asset", "volume", "turnover"}
    numeric_cols = [c for c in numeric_cols if str(c).lower() not in ignored][:20]

    if len(numeric_cols) == 0:
        return _mk_check(
            contract.name,
            "plausibility",
            "plausibility",
            "SKIPPED",
            None,
            None,
            False,
            "no numeric columns for plausibility",
            {},
        )

    outlier_rates: dict[str, float] = {}
    for c in numeric_cols:
        z = _robust_zscore(df[c])
        valid = z.notna().sum()
        if valid == 0:
            continue
        outlier_rates[str(c)] = float((z > cfg.robust_z_warn).sum() / max(valid, 1))

    if len(outlier_rates) == 0:
        return _mk_check(
            contract.name,
            "plausibility",
            "plausibility",
            "WARN",
            None,
            None,
            False,
            "insufficient numeric sample for plausibility",
            {},
        )

    worst_col, worst_rate = max(outlier_rates.items(), key=lambda kv: kv[1])

    if worst_rate > cfg.outlier_rate_fail:
        return _mk_check(
            contract.name,
            "plausibility_outlier_rate",
            "plausibility",
            "FAIL",
            worst_rate,
            cfg.outlier_rate_fail,
            False,
            "robust outlier rate too high",
            {"worst_column": worst_col, "outlier_rates": outlier_rates},
        )

    if worst_rate > cfg.outlier_rate_warn:
        return _mk_check(
            contract.name,
            "plausibility_outlier_rate",
            "plausibility",
            "WARN",
            worst_rate,
            cfg.outlier_rate_warn,
            False,
            "robust outlier rate elevated",
            {"worst_column": worst_col, "outlier_rates": outlier_rates},
        )

    return _mk_check(
        contract.name,
        "plausibility",
        "plausibility",
        "PASS",
        worst_rate,
        cfg.outlier_rate_warn,
        False,
        "plausibility checks passed",
        {"worst_column": worst_col, "outlier_rates": outlier_rates},
    )


def _check_drift_mean_shift(df_ref: pd.DataFrame | None, df_cur: pd.DataFrame, contract: DatasetContract, cfg: DataHealthConfig) -> CheckResult:
    if df_ref is None or len(df_ref) == 0:
        return _mk_check(
            contract.name,
            "drift_mean_shift",
            "drift",
            "NOT_APPLICABLE",
            None,
            None,
            False,
            "reference dataset unavailable",
            {},
        )

    nums_ref = [c for c in df_ref.columns if pd.api.types.is_numeric_dtype(df_ref[c])]
    nums_cur = [c for c in df_cur.columns if pd.api.types.is_numeric_dtype(df_cur[c])]
    common = sorted(set(nums_ref).intersection(nums_cur))[:30]

    if len(common) == 0:
        return _mk_check(
            contract.name,
            "drift_mean_shift",
            "drift",
            "SKIPPED",
            None,
            None,
            False,
            "no common numeric columns",
            {},
        )

    shifts = {}
    for c in common:
        x_ref = pd.to_numeric(df_ref[c], errors="coerce")
        x_cur = pd.to_numeric(df_cur[c], errors="coerce")
        if x_ref.notna().sum() < cfg.insufficient_sample_min_rows or x_cur.notna().sum() < cfg.insufficient_sample_min_rows:
            continue
        std_ref = float(np.nanstd(x_ref.values))
        if std_ref <= 1e-12:
            continue
        shift = abs(float(np.nanmean(x_cur.values) - np.nanmean(x_ref.values))) / max(std_ref, 1e-12)
        shifts[str(c)] = shift

    if len(shifts) == 0:
        return _mk_check(
            contract.name,
            "drift_mean_shift",
            "drift",
            "INSUFFICIENT_SAMPLE",
            None,
            cfg.drift_shift_warn,
            False,
            "insufficient sample for drift",
            {},
        )

    worst_col, worst_shift = max(shifts.items(), key=lambda kv: kv[1])

    if worst_shift > cfg.drift_shift_fail:
        return _mk_check(
            contract.name,
            "drift_mean_shift",
            "drift",
            "FAIL",
            worst_shift,
            cfg.drift_shift_fail,
            False,
            "mean shift above fail threshold",
            {"worst_column": worst_col, "shifts": shifts},
        )
    if worst_shift > cfg.drift_shift_warn:
        return _mk_check(
            contract.name,
            "drift_mean_shift",
            "drift",
            "WARN",
            worst_shift,
            cfg.drift_shift_warn,
            False,
            "mean shift above warn threshold",
            {"worst_column": worst_col, "shifts": shifts},
        )

    return _mk_check(
        contract.name,
        "drift_mean_shift",
        "drift",
        "PASS",
        worst_shift,
        cfg.drift_shift_warn,
        False,
        "mean shift within tolerance",
        {"worst_column": worst_col, "shifts": shifts},
    )


def _status_to_score(status: str) -> float | None:
    m = {
        "PASS": 1.0,
        "WARN": 0.5,
        "FAIL": 0.0,
        "ERROR": 0.0,
        "INSUFFICIENT_SAMPLE": 0.25,
        "NOT_APPLICABLE": None,
        "SKIPPED": None,
    }
    return m.get(status.upper(), 0.0)


def _compute_dataset_score(checks: Sequence[CheckResult], contract: DatasetContract) -> float | None:
    weights_cfg = dict(contract.gating_policy.get("check_weights", {}))
    vals = []
    wts = []
    for c in checks:
        score = _status_to_score(c.status)
        if score is None:
            continue
        w = float(weights_cfg.get(c.check_name, 1.0))
        vals.append(float(score))
        wts.append(max(w, 0.0))

    if len(vals) == 0:
        return None
    w_arr = np.asarray(wts, dtype=float)
    if np.sum(w_arr) <= 0:
        w_arr = np.ones_like(w_arr)
    v_arr = np.asarray(vals, dtype=float)
    return float(np.average(v_arr, weights=w_arr))


def _derive_dataset_gate(checks: Sequence[CheckResult], score: float | None, contract: DatasetContract, cfg: DataHealthConfig) -> tuple[str, bool]:
    sts = [c.status.upper() for c in checks]
    hard_fail = any((c.status.upper() in {"FAIL", "ERROR"}) and c.hard_fail for c in checks)

    if "ERROR" in sts:
        return "ERROR", True
    if hard_fail:
        return "FAIL", True

    if score is None:
        return "ERROR", True

    gp = dict(contract.gating_policy)
    warn_th = float(gp.get("warn_threshold", cfg.dataset_warn_threshold))
    fail_th = float(gp.get("fail_threshold", cfg.dataset_fail_threshold))

    if score < fail_th:
        return "FAIL", False

    if "FAIL" in sts and bool(gp.get("fail_on_any_fail", False)):
        return "FAIL", False

    if ("FAIL" in sts) or ("WARN" in sts) or ("INSUFFICIENT_SAMPLE" in sts) or (score < warn_th):
        return "WARN", False

    if all(s in {"SKIPPED", "NOT_APPLICABLE"} for s in sts):
        return "SKIPPED", False

    return "PASS", False


def _is_override_active(ov: Mapping[str, Any], execution_ts: pd.Timestamp) -> bool:
    created = ov.get("created_at")
    expires = ov.get("expires_at")
    if created is not None and execution_ts < pd.Timestamp(created):
        return False
    if expires is not None and execution_ts > pd.Timestamp(expires):
        return False
    return True


def _apply_overrides(
    dataset_results: Sequence[DatasetHealthResult],
    overrides: Sequence[Mapping[str, Any]],
    *,
    execution_ts: pd.Timestamp,
    cfg: DataHealthConfig,
) -> tuple[list[DatasetHealthResult], list[dict[str, Any]]]:
    applied: list[dict[str, Any]] = []

    # Index overrides by (dataset_name, check_name)
    idx: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for ov in overrides:
        dname = str(ov.get("dataset_name", "")).strip()
        cname = str(ov.get("check_name", "")).strip()
        if not dname or not cname:
            continue
        idx.setdefault((dname, cname), []).append(ov)

    out: list[DatasetHealthResult] = []
    for ds in dataset_results:
        new_checks: list[CheckResult] = []
        for ch in ds.checks:
            candidates = idx.get((ds.dataset_name, ch.check_name), [])
            chosen = None
            for ov in candidates:
                if _is_override_active(ov, execution_ts):
                    chosen = ov
                    break
            if chosen is None:
                new_checks.append(ch)
                continue

            new_status = str(chosen.get("overridden_status", ch.status)).upper()
            actor = str(chosen.get("actor", "unknown"))
            reason = str(chosen.get("reason", "override"))
            applied.append(
                {
                    "dataset_name": ds.dataset_name,
                    "check_name": ch.check_name,
                    "original_status": ch.status,
                    "overridden_status": new_status,
                    "reason": reason,
                    "actor": actor,
                    "created_at": chosen.get("created_at"),
                    "expires_at": chosen.get("expires_at"),
                }
            )
            new_checks.append(
                CheckResult(
                    dataset_name=ch.dataset_name,
                    check_name=ch.check_name,
                    check_type=ch.check_type,
                    status=new_status,
                    observed_value=ch.observed_value,
                    threshold=ch.threshold,
                    severity=_status_severity(new_status),
                    hard_fail=ch.hard_fail,
                    message=f"{ch.message} | override by {actor}: {reason}",
                    details={**dict(ch.details), "override_applied": True},
                )
            )

        # Recompute score/status after overrides.
        contract_stub = DatasetContract(
            name=ds.dataset_name,
            dataset_type=ds.dataset_type,
            criticality=ds.criticality,
            expected_schema={},
            freshness_sla_hours=48,
            required_checks=tuple(c.check_name for c in new_checks),
            gating_policy=ds.details.get("gating_policy", {}),
        )
        new_score = _compute_dataset_score(new_checks, contract_stub)
        new_status, hard_fail = _derive_dataset_gate(new_checks, new_score, contract_stub, cfg)

        out.append(
            DatasetHealthResult(
                dataset_name=ds.dataset_name,
                dataset_type=ds.dataset_type,
                criticality=ds.criticality,
                status=new_status,
                score=new_score,
                hard_fail=hard_fail,
                checks=tuple(new_checks),
                observed_at=ds.observed_at,
                details=ds.details,
            )
        )

    return out, applied


def _compute_global_score(results: Sequence[DatasetHealthResult]) -> float | None:
    vals = []
    wts = []
    for r in results:
        if r.score is None:
            continue
        if r.status.upper() in {"SKIPPED", "NOT_APPLICABLE"}:
            continue
        vals.append(float(r.score))
        wts.append(max(float(r.criticality), 0.0))

    if len(vals) == 0:
        return None
    w = np.asarray(wts, dtype=float)
    if np.sum(w) <= 0:
        w = np.ones_like(w)
    return float(np.average(np.asarray(vals, dtype=float), weights=w))


def _derive_global_gate(results: Sequence[DatasetHealthResult], cfg: DataHealthConfig) -> str:
    blocking_fail = False
    any_warn = False

    for r in results:
        st = r.status.upper()
        blocking = bool(r.details.get("blocking", False)) or (float(r.criticality) >= cfg.blocking_criticality)
        if st in {"FAIL", "ERROR"} and blocking:
            blocking_fail = True
        elif st in {"WARN", "FAIL", "ERROR", "INSUFFICIENT_SAMPLE"}:
            any_warn = True

    if blocking_fail:
        return "FAIL"
    if any_warn:
        return "WARN"
    return "PASS"


def _save_df(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return str(path)
    except Exception:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return str(csv_path)


def _save_json(payload: Mapping[str, Any], path: Path) -> str:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(dict(payload)), sort_keys=True, indent=2, default=str), encoding="utf-8")
    return str(path)


def persist_data_health_outputs(
    check_results: pd.DataFrame,
    dataset_results: pd.DataFrame,
    run_result: Mapping[str, Any],
    events: pd.DataFrame,
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "check_results": _save_df(check_results, out / "check_results.parquet"),
        "dataset_results": _save_df(dataset_results, out / "dataset_results.parquet"),
        "run_result": _save_json(run_result, out / "run_result.json"),
        "events": _save_df(events, out / "events.parquet"),
    }

def run_data_health(
    contracts: Sequence[Any] | pd.DataFrame | str | Path,
    *,
    datasets: Mapping[str, pd.DataFrame | str | Path] | None = None,
    overrides: Sequence[Mapping[str, Any]] | pd.DataFrame | str | Path | None = None,
    config: DataHealthConfig | None = None,
    execution_date: str | pd.Timestamp | None = None,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or DataHealthConfig()
    rid = run_id or run_id_with_prefix("datahl")
    exec_ts = pd.Timestamp(execution_date) if execution_date is not None else pd.Timestamp.utcnow()

    started_at = utc_now_iso()
    contracts_norm = _normalize_contracts(contracts)
    datasets_map = dict(datasets or {})

    if len(contracts_norm) == 0:
        raise ValueError("no dataset contracts provided")

    ds_results: list[DatasetHealthResult] = []

    for contract in contracts_norm:
        if not bool(contract.gating_policy.get("enabled", True)):
            ds_results.append(
                DatasetHealthResult(
                    dataset_name=contract.name,
                    dataset_type=contract.dataset_type,
                    criticality=float(contract.criticality),
                    status="SKIPPED",
                    score=None,
                    hard_fail=False,
                    checks=tuple(),
                    observed_at=utc_now_iso(),
                    details={"reason": "disabled", "gating_policy": dict(contract.gating_policy)},
                )
            )
            continue

        try:
            data_obj = datasets_map.get(contract.name, contract.path_or_table)
            df = _to_df(data_obj)

            check_results: list[CheckResult] = []
            check_results.append(_check_availability(df, contract))

            if len(df) > 0:
                if "schema" in contract.required_checks:
                    check_results.append(_check_schema(df, contract))
                if "freshness" in contract.required_checks:
                    check_results.append(_check_freshness(df, contract, exec_ts))
                if "completeness" in contract.required_checks:
                    check_results.append(_check_completeness(df, contract, cfg))
                if "temporal_consistency" in contract.required_checks:
                    check_results.append(_check_temporal_consistency(df, contract, exec_ts))
                if "plausibility" in contract.required_checks:
                    check_results.append(_check_plausibility(df, contract, cfg))
                if "drift" in contract.required_checks:
                    ref_key = f"{contract.name}__reference"
                    ref_df = _to_df(datasets_map.get(ref_key)) if ref_key in datasets_map else None
                    check_results.append(_check_drift_mean_shift(ref_df, df, contract, cfg))

            score = _compute_dataset_score(check_results, contract)
            status, hard_fail = _derive_dataset_gate(check_results, score, contract, cfg)

            ds_results.append(
                DatasetHealthResult(
                    dataset_name=contract.name,
                    dataset_type=contract.dataset_type,
                    criticality=float(contract.criticality),
                    status=status,
                    score=score,
                    hard_fail=hard_fail,
                    checks=tuple(check_results),
                    observed_at=utc_now_iso(),
                    details={
                        "rows": int(len(df)),
                        "required_checks": list(contract.required_checks),
                        "gating_policy": dict(contract.gating_policy),
                    },
                )
            )

        except Exception as exc:
            ds_results.append(
                DatasetHealthResult(
                    dataset_name=contract.name,
                    dataset_type=contract.dataset_type,
                    criticality=float(contract.criticality),
                    status="ERROR",
                    score=None,
                    hard_fail=True,
                    checks=tuple(
                        [
                            _mk_check(
                                contract.name,
                                "evaluation",
                                "runtime",
                                "ERROR",
                                None,
                                None,
                                True,
                                f"evaluation error: {type(exc).__name__}",
                                {"error_message": str(exc)},
                            )
                        ]
                    ),
                    observed_at=utc_now_iso(),
                    details={"error_type": type(exc).__name__, "error_message": str(exc)},
                )
            )

    overrides_norm = _normalize_overrides(overrides)
    ds_final, overrides_applied = _apply_overrides(ds_results, overrides_norm, execution_ts=exec_ts, cfg=cfg)

    global_score = _compute_global_score(ds_final)
    global_gate = _derive_global_gate(ds_final, cfg)

    run_status = "SUCCESS"
    if global_gate == "FAIL":
        run_status = "FAILED"
    elif any(r.status.upper() in {"ERROR", "SKIPPED"} for r in ds_final):
        run_status = "PARTIAL"

    ended_at = utc_now_iso()

    run_result = DataHealthRunResult(
        run_id=rid,
        execution_date=str(exec_ts.date()),
        global_gate=global_gate,
        global_score=global_score,
        run_status=run_status,
        datasets=tuple(ds_final),
        overrides_applied=tuple(overrides_applied),
        started_at=started_at,
        ended_at=ended_at,
    )

    # Flatten outputs.
    check_rows: list[dict[str, Any]] = []
    ds_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for ds in ds_final:
        ds_rows.append(
            {
                "dataset_name": ds.dataset_name,
                "dataset_type": ds.dataset_type,
                "criticality": ds.criticality,
                "status": ds.status,
                "score": ds.score,
                "hard_fail": ds.hard_fail,
                "observed_at": ds.observed_at,
                "details": json_safe(ds.details),
                "run_id": rid,
            }
        )

        event_rows.append(
            {
                "event_type": f"data_health_dataset_{ds.status.lower()}",
                "source": "data_health",
                "run_id": rid,
                "dataset_name": ds.dataset_name,
                "severity": _status_severity(ds.status),
                "timestamp": ds.observed_at,
                "payload": {
                    "status": ds.status,
                    "score": ds.score,
                    "hard_fail": ds.hard_fail,
                    "criticality": ds.criticality,
                },
            }
        )

        for ch in ds.checks:
            check_rows.append(
                {
                    "dataset_name": ch.dataset_name,
                    "check_name": ch.check_name,
                    "check_type": ch.check_type,
                    "status": ch.status,
                    "observed_value": json_safe(ch.observed_value),
                    "threshold": json_safe(ch.threshold),
                    "severity": ch.severity,
                    "hard_fail": ch.hard_fail,
                    "message": ch.message,
                    "details": json_safe(ch.details),
                    "run_id": rid,
                }
            )

    event_rows.append(
        {
            "event_type": "data_health_run_finished",
            "source": "data_health",
            "run_id": rid,
            "dataset_name": None,
            "severity": ("high" if global_gate == "FAIL" else ("warning" if global_gate == "WARN" else "info")),
            "timestamp": ended_at,
            "payload": {
                "global_gate": global_gate,
                "global_score": global_score,
                "run_status": run_status,
                "num_datasets": len(ds_final),
                "overrides_applied": len(overrides_applied),
            },
        }
    )

    check_df = pd.DataFrame(check_rows)
    ds_df = pd.DataFrame(ds_rows)
    events_df = pd.DataFrame(event_rows)

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_data_health_outputs(
            check_df,
            ds_df,
            {
                "run_id": run_result.run_id,
                "execution_date": run_result.execution_date,
                "global_gate": run_result.global_gate,
                "global_score": run_result.global_score,
                "run_status": run_result.run_status,
                "started_at": run_result.started_at,
                "ended_at": run_result.ended_at,
                "overrides_applied": json_safe(run_result.overrides_applied),
                "config_hash": config_hash(cfg.__dict__, n=24),
                "datasets": json_safe([asdict(d) for d in run_result.datasets]),
            },
            events_df,
            output_dir=output_dir,
        )

    return {
        "run_result": run_result,
        "check_results": check_df,
        "dataset_results": ds_df,
        "events": events_df,
        "artifacts": artifacts,
    }


__all__ = [
    "CheckResult",
    "DataHealthConfig",
    "DataHealthRunResult",
    "DatasetContract",
    "DatasetHealthResult",
    "persist_data_health_outputs",
    "run_data_health",
]
