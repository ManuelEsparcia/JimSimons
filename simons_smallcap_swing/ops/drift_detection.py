"""
ops/drift_detection.py - Statistical drift supervision with governed decisions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftDetectionConfig:
    min_samples_feature_ref: int = 200
    min_samples_feature_cur: int = 100
    min_samples_prediction_ref: int = 200
    min_samples_prediction_cur: int = 100
    min_samples_performance_ref: int = 120
    min_samples_performance_cur: int = 60
    psi_warn: float = 0.20
    psi_fail: float = 0.35
    ks_warn: float = 0.12
    ks_fail: float = 0.20
    missing_shift_warn: float = 0.05
    missing_shift_fail: float = 0.10
    ece_delta_warn: float = 0.02
    ece_delta_fail: float = 0.05
    brier_delta_warn: float = 0.01
    brier_delta_fail: float = 0.03
    ic_drop_warn: float = 0.02
    ic_drop_fail: float = 0.05
    hit_rate_drop_warn: float = 0.03
    hit_rate_drop_fail: float = 0.07
    spread_drop_warn: float = 0.002
    spread_drop_fail: float = 0.005
    score_col_candidates: tuple[str, ...] = ("score", "prob", "prediction", "yhat")
    label_col_candidates: tuple[str, ...] = ("label", "y", "target")
    signal_col_candidates: tuple[str, ...] = ("signal", "score", "alpha")
    target_col_candidates: tuple[str, ...] = ("future_return", "ret_fwd", "target", "y")
    structural_feature_fail: bool = False
    fail_on_feature_fail: bool = False
    fail_on_prediction_fail: bool = False
    fail_on_performance_fail: bool = True


@dataclass(frozen=True)
class DriftMetricResult:
    name: str
    dimension: str
    entity: str
    value: float | None
    threshold_warn: float | None
    threshold_fail: float | None
    sample_size_ref: int
    sample_size_cur: int
    status: str
    pvalue: float | None
    power: float | None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DriftRunResult:
    run_id: str
    execution_date: str
    reference_window: Mapping[str, Any]
    current_window: Mapping[str, Any]
    feature_status: str
    prediction_status: str
    performance_status: str
    global_status: str
    action: str
    metrics: tuple[DriftMetricResult, ...]
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


def _infer_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_numeric(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def _stable_psi(x_ref: np.ndarray, x_cur: np.ndarray, n_bins: int = 10) -> float:
    xr = np.asarray(x_ref, dtype=float)
    xc = np.asarray(x_cur, dtype=float)
    xr = xr[np.isfinite(xr)]
    xc = xc[np.isfinite(xc)]
    if xr.size == 0 or xc.size == 0:
        return float("nan")

    q = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.quantile(xr, q)
    bins = np.unique(bins)
    if bins.size < 3:
        return 0.0

    ref_hist, _ = np.histogram(xr, bins=bins)
    cur_hist, _ = np.histogram(xc, bins=bins)

    p_ref = ref_hist / max(ref_hist.sum(), 1)
    p_cur = cur_hist / max(cur_hist.sum(), 1)

    eps = 1e-12
    return float(np.sum((p_cur - p_ref) * np.log((p_cur + eps) / (p_ref + eps))))


def _ks_stat(x_ref: np.ndarray, x_cur: np.ndarray) -> tuple[float, float | None]:
    xr = np.asarray(x_ref, dtype=float)
    xc = np.asarray(x_cur, dtype=float)
    xr = xr[np.isfinite(xr)]
    xc = xc[np.isfinite(xc)]
    n1 = xr.size
    n2 = xc.size
    if n1 == 0 or n2 == 0:
        return float("nan"), None

    xr = np.sort(xr)
    xc = np.sort(xc)
    data = np.sort(np.concatenate([xr, xc]))
    cdf1 = np.searchsorted(xr, data, side="right") / n1
    cdf2 = np.searchsorted(xc, data, side="right") / n2
    ks = float(np.max(np.abs(cdf1 - cdf2)))

    # Asymptotic p-value approximation.
    en = np.sqrt(n1 * n2 / max(n1 + n2, 1))
    p = 2.0 * np.exp(-2.0 * (en * ks) ** 2)
    p = float(np.clip(p, 0.0, 1.0))
    return ks, p


def _expected_calibration_error(prob: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    p = np.asarray(prob, dtype=float)
    t = np.asarray(y, dtype=float)
    m = np.isfinite(p) & np.isfinite(t)
    p = p[m]
    t = t[m]
    if p.size == 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (p >= bins[i]) & (p < bins[i + 1])
        else:
            mask = (p >= bins[i]) & (p <= bins[i + 1])
        if not np.any(mask):
            continue
        acc = float(np.mean(t[mask]))
        conf = float(np.mean(p[mask]))
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def _brier(prob: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(prob, dtype=float)
    t = np.asarray(y, dtype=float)
    m = np.isfinite(p) & np.isfinite(t)
    if not np.any(m):
        return float("nan")
    return float(np.mean((p[m] - t[m]) ** 2))


def _information_coefficient(signal: pd.Series, target: pd.Series) -> float:
    a = _safe_numeric(signal)
    b = _safe_numeric(target)
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def _top_bottom_spread(signal: pd.Series, target: pd.Series, q: float = 0.2) -> float:
    a = _safe_numeric(signal)
    b = _safe_numeric(target)
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 10:
        return float("nan")
    lo = pair.iloc[:, 0].quantile(q)
    hi = pair.iloc[:, 0].quantile(1.0 - q)
    top = pair[pair.iloc[:, 0] >= hi].iloc[:, 1]
    bot = pair[pair.iloc[:, 0] <= lo].iloc[:, 1]
    if len(top) == 0 or len(bot) == 0:
        return float("nan")
    return float(top.mean() - bot.mean())


def _status_high_is_bad(value: float | None, warn: float, fail: float) -> str:
    if value is None or not np.isfinite(value):
        return "insufficient_sample"
    if value > fail:
        return "fail"
    if value > warn:
        return "warn"
    return "pass"


def _status_low_is_bad(value: float | None, warn_drop: float, fail_drop: float) -> str:
    # Here value represents drop (positive is bad), e.g. -(delta_ic).
    return _status_high_is_bad(value, warn_drop, fail_drop)


def _mk_metric(
    *,
    name: str,
    dimension: str,
    entity: str,
    value: float | None,
    threshold_warn: float | None,
    threshold_fail: float | None,
    sample_size_ref: int,
    sample_size_cur: int,
    status: str,
    pvalue: float | None = None,
    power: float | None = None,
    details: Mapping[str, Any] | None = None,
) -> DriftMetricResult:
    return DriftMetricResult(
        name=name,
        dimension=dimension,
        entity=entity,
        value=(None if value is None else float(value)),
        threshold_warn=(None if threshold_warn is None else float(threshold_warn)),
        threshold_fail=(None if threshold_fail is None else float(threshold_fail)),
        sample_size_ref=int(sample_size_ref),
        sample_size_cur=int(sample_size_cur),
        status=str(status).lower(),
        pvalue=(None if pvalue is None else float(pvalue)),
        power=(None if power is None else float(power)),
        details=(details or {}),
    )


def _feature_drift_suite(df_ref: pd.DataFrame, df_cur: pd.DataFrame, cfg: DriftDetectionConfig) -> list[DriftMetricResult]:
    metrics: list[DriftMetricResult] = []

    meta_cols = {"date", "timestamp", "symbol", "asset", "asset_id", "id"}
    num_ref = [c for c in df_ref.columns if pd.api.types.is_numeric_dtype(df_ref[c]) and str(c).lower() not in meta_cols]
    num_cur = [c for c in df_cur.columns if pd.api.types.is_numeric_dtype(df_cur[c]) and str(c).lower() not in meta_cols]

    common = sorted(set(num_ref).intersection(num_cur))
    new_feats = sorted(set(num_cur) - set(num_ref))
    gone_feats = sorted(set(num_ref) - set(num_cur))

    if new_feats:
        st = "fail" if cfg.structural_feature_fail else "warn"
        metrics.append(
            _mk_metric(
                name="feature_structural_new",
                dimension="feature",
                entity="__global__",
                value=float(len(new_feats)),
                threshold_warn=0.0,
                threshold_fail=0.0,
                sample_size_ref=int(len(df_ref)),
                sample_size_cur=int(len(df_cur)),
                status=st,
                details={"new_features": new_feats},
            )
        )

    if gone_feats:
        st = "fail" if cfg.structural_feature_fail else "warn"
        metrics.append(
            _mk_metric(
                name="feature_structural_missing",
                dimension="feature",
                entity="__global__",
                value=float(len(gone_feats)),
                threshold_warn=0.0,
                threshold_fail=0.0,
                sample_size_ref=int(len(df_ref)),
                sample_size_cur=int(len(df_cur)),
                status=st,
                details={"missing_features": gone_feats},
            )
        )

    for c in common:
        xr = _safe_numeric(df_ref[c]).dropna().values
        xc = _safe_numeric(df_cur[c]).dropna().values
        n_ref = int(xr.size)
        n_cur = int(xc.size)

        miss_ref = float(df_ref[c].isna().mean())
        miss_cur = float(df_cur[c].isna().mean())
        miss_shift = abs(miss_cur - miss_ref)
        miss_status = _status_high_is_bad(miss_shift, cfg.missing_shift_warn, cfg.missing_shift_fail)
        metrics.append(
            _mk_metric(
                name="feature_missing_shift",
                dimension="feature",
                entity=str(c),
                value=miss_shift,
                threshold_warn=cfg.missing_shift_warn,
                threshold_fail=cfg.missing_shift_fail,
                sample_size_ref=n_ref,
                sample_size_cur=n_cur,
                status=miss_status,
                details={"missing_ref": miss_ref, "missing_cur": miss_cur},
            )
        )

        if n_ref < cfg.min_samples_feature_ref or n_cur < cfg.min_samples_feature_cur:
            metrics.append(
                _mk_metric(
                    name="feature_psi",
                    dimension="feature",
                    entity=str(c),
                    value=None,
                    threshold_warn=cfg.psi_warn,
                    threshold_fail=cfg.psi_fail,
                    sample_size_ref=n_ref,
                    sample_size_cur=n_cur,
                    status="insufficient_sample",
                    details={"reason": "sample_below_min"},
                )
            )
            metrics.append(
                _mk_metric(
                    name="feature_ks",
                    dimension="feature",
                    entity=str(c),
                    value=None,
                    threshold_warn=cfg.ks_warn,
                    threshold_fail=cfg.ks_fail,
                    sample_size_ref=n_ref,
                    sample_size_cur=n_cur,
                    status="insufficient_sample",
                    details={"reason": "sample_below_min"},
                )
            )
            continue

        psi = _stable_psi(xr, xc)
        psi_status = _status_high_is_bad(psi, cfg.psi_warn, cfg.psi_fail)
        metrics.append(
            _mk_metric(
                name="feature_psi",
                dimension="feature",
                entity=str(c),
                value=psi,
                threshold_warn=cfg.psi_warn,
                threshold_fail=cfg.psi_fail,
                sample_size_ref=n_ref,
                sample_size_cur=n_cur,
                status=psi_status,
            )
        )

        ks, pval = _ks_stat(xr, xc)
        ks_status = _status_high_is_bad(ks, cfg.ks_warn, cfg.ks_fail)
        metrics.append(
            _mk_metric(
                name="feature_ks",
                dimension="feature",
                entity=str(c),
                value=ks,
                threshold_warn=cfg.ks_warn,
                threshold_fail=cfg.ks_fail,
                sample_size_ref=n_ref,
                sample_size_cur=n_cur,
                status=ks_status,
                pvalue=pval,
            )
        )

    if len(metrics) == 0:
        metrics.append(
            _mk_metric(
                name="feature_not_applicable",
                dimension="feature",
                entity="__global__",
                value=None,
                threshold_warn=None,
                threshold_fail=None,
                sample_size_ref=int(len(df_ref)),
                sample_size_cur=int(len(df_cur)),
                status="not_applicable",
                details={"reason": "no_numeric_features"},
            )
        )

    return metrics

def _prediction_drift_suite(df_ref: pd.DataFrame, df_cur: pd.DataFrame, cfg: DriftDetectionConfig) -> list[DriftMetricResult]:
    metrics: list[DriftMetricResult] = []
    score_col_ref = _infer_col(df_ref, cfg.score_col_candidates)
    score_col_cur = _infer_col(df_cur, cfg.score_col_candidates)

    if score_col_ref is None or score_col_cur is None:
        return [
            _mk_metric(
                name="prediction_not_applicable",
                dimension="prediction",
                entity="__global__",
                value=None,
                threshold_warn=None,
                threshold_fail=None,
                sample_size_ref=int(len(df_ref)),
                sample_size_cur=int(len(df_cur)),
                status="not_applicable",
                details={"reason": "missing_score_column"},
            )
        ]

    xr = _safe_numeric(df_ref[score_col_ref]).dropna().values
    xc = _safe_numeric(df_cur[score_col_cur]).dropna().values
    n_ref = int(xr.size)
    n_cur = int(xc.size)

    if n_ref < cfg.min_samples_prediction_ref or n_cur < cfg.min_samples_prediction_cur:
        metrics.append(
            _mk_metric(
                name="prediction_score_distribution",
                dimension="prediction",
                entity="__global__",
                value=None,
                threshold_warn=None,
                threshold_fail=None,
                sample_size_ref=n_ref,
                sample_size_cur=n_cur,
                status="insufficient_sample",
                details={"reason": "sample_below_min"},
            )
        )
    else:
        psi = _stable_psi(xr, xc)
        ks, pval = _ks_stat(xr, xc)
        metrics.append(
            _mk_metric(
                name="prediction_psi",
                dimension="prediction",
                entity="__global__",
                value=psi,
                threshold_warn=cfg.psi_warn,
                threshold_fail=cfg.psi_fail,
                sample_size_ref=n_ref,
                sample_size_cur=n_cur,
                status=_status_high_is_bad(psi, cfg.psi_warn, cfg.psi_fail),
            )
        )
        metrics.append(
            _mk_metric(
                name="prediction_ks",
                dimension="prediction",
                entity="__global__",
                value=ks,
                threshold_warn=cfg.ks_warn,
                threshold_fail=cfg.ks_fail,
                sample_size_ref=n_ref,
                sample_size_cur=n_cur,
                status=_status_high_is_bad(ks, cfg.ks_warn, cfg.ks_fail),
                pvalue=pval,
            )
        )

    lbl_ref = _infer_col(df_ref, cfg.label_col_candidates)
    lbl_cur = _infer_col(df_cur, cfg.label_col_candidates)

    if lbl_ref is None or lbl_cur is None:
        metrics.append(
            _mk_metric(
                name="prediction_calibration",
                dimension="prediction",
                entity="__global__",
                value=None,
                threshold_warn=None,
                threshold_fail=None,
                sample_size_ref=n_ref,
                sample_size_cur=n_cur,
                status="not_applicable",
                details={"reason": "labels_not_available"},
            )
        )
        return metrics

    y_ref = _safe_numeric(df_ref[lbl_ref]).values
    y_cur = _safe_numeric(df_cur[lbl_cur]).values

    ece_ref = _expected_calibration_error(xr, y_ref)
    ece_cur = _expected_calibration_error(xc, y_cur)
    d_ece = abs(ece_cur - ece_ref) if np.isfinite(ece_ref) and np.isfinite(ece_cur) else None

    brier_ref = _brier(xr, y_ref)
    brier_cur = _brier(xc, y_cur)
    d_brier = (brier_cur - brier_ref) if np.isfinite(brier_ref) and np.isfinite(brier_cur) else None

    metrics.append(
        _mk_metric(
            name="prediction_ece_delta",
            dimension="prediction",
            entity="__global__",
            value=d_ece,
            threshold_warn=cfg.ece_delta_warn,
            threshold_fail=cfg.ece_delta_fail,
            sample_size_ref=n_ref,
            sample_size_cur=n_cur,
            status=_status_high_is_bad(d_ece, cfg.ece_delta_warn, cfg.ece_delta_fail),
            details={"ece_ref": ece_ref, "ece_cur": ece_cur},
        )
    )

    metrics.append(
        _mk_metric(
            name="prediction_brier_delta",
            dimension="prediction",
            entity="__global__",
            value=d_brier,
            threshold_warn=cfg.brier_delta_warn,
            threshold_fail=cfg.brier_delta_fail,
            sample_size_ref=n_ref,
            sample_size_cur=n_cur,
            status=_status_high_is_bad(d_brier, cfg.brier_delta_warn, cfg.brier_delta_fail),
            details={"brier_ref": brier_ref, "brier_cur": brier_cur},
        )
    )

    return metrics


def _performance_drift_suite(df_ref: pd.DataFrame, df_cur: pd.DataFrame, cfg: DriftDetectionConfig) -> list[DriftMetricResult]:
    sig_ref = _infer_col(df_ref, cfg.signal_col_candidates)
    sig_cur = _infer_col(df_cur, cfg.signal_col_candidates)
    tgt_ref = _infer_col(df_ref, cfg.target_col_candidates)
    tgt_cur = _infer_col(df_cur, cfg.target_col_candidates)

    if sig_ref is None or sig_cur is None or tgt_ref is None or tgt_cur is None:
        return [
            _mk_metric(
                name="performance_not_applicable",
                dimension="performance",
                entity="__global__",
                value=None,
                threshold_warn=None,
                threshold_fail=None,
                sample_size_ref=int(len(df_ref)),
                sample_size_cur=int(len(df_cur)),
                status="not_applicable",
                details={"reason": "missing_signal_or_target"},
            )
        ]

    n_ref = int(pd.concat([_safe_numeric(df_ref[sig_ref]), _safe_numeric(df_ref[tgt_ref])], axis=1).dropna().shape[0])
    n_cur = int(pd.concat([_safe_numeric(df_cur[sig_cur]), _safe_numeric(df_cur[tgt_cur])], axis=1).dropna().shape[0])

    if n_ref < cfg.min_samples_performance_ref or n_cur < cfg.min_samples_performance_cur:
        return [
            _mk_metric(
                name="performance_insufficient_sample",
                dimension="performance",
                entity="__global__",
                value=None,
                threshold_warn=None,
                threshold_fail=None,
                sample_size_ref=n_ref,
                sample_size_cur=n_cur,
                status="insufficient_sample",
                details={"reason": "sample_below_min"},
            )
        ]

    ic_ref = _information_coefficient(df_ref[sig_ref], df_ref[tgt_ref])
    ic_cur = _information_coefficient(df_cur[sig_cur], df_cur[tgt_cur])
    ic_drop = (ic_ref - ic_cur) if np.isfinite(ic_ref) and np.isfinite(ic_cur) else None

    pair_ref = pd.concat([_safe_numeric(df_ref[sig_ref]), _safe_numeric(df_ref[tgt_ref])], axis=1).dropna()
    pair_cur = pd.concat([_safe_numeric(df_cur[sig_cur]), _safe_numeric(df_cur[tgt_cur])], axis=1).dropna()
    hr_ref = float((np.sign(pair_ref.iloc[:, 0]) == np.sign(pair_ref.iloc[:, 1])).mean()) if len(pair_ref) else float("nan")
    hr_cur = float((np.sign(pair_cur.iloc[:, 0]) == np.sign(pair_cur.iloc[:, 1])).mean()) if len(pair_cur) else float("nan")
    hr_drop = (hr_ref - hr_cur) if np.isfinite(hr_ref) and np.isfinite(hr_cur) else None

    sp_ref = _top_bottom_spread(df_ref[sig_ref], df_ref[tgt_ref])
    sp_cur = _top_bottom_spread(df_cur[sig_cur], df_cur[tgt_cur])
    sp_drop = (sp_ref - sp_cur) if np.isfinite(sp_ref) and np.isfinite(sp_cur) else None

    return [
        _mk_metric(
            name="performance_ic_drop",
            dimension="performance",
            entity="__global__",
            value=ic_drop,
            threshold_warn=cfg.ic_drop_warn,
            threshold_fail=cfg.ic_drop_fail,
            sample_size_ref=n_ref,
            sample_size_cur=n_cur,
            status=_status_low_is_bad(ic_drop, cfg.ic_drop_warn, cfg.ic_drop_fail),
            details={"ic_ref": ic_ref, "ic_cur": ic_cur},
        ),
        _mk_metric(
            name="performance_hit_rate_drop",
            dimension="performance",
            entity="__global__",
            value=hr_drop,
            threshold_warn=cfg.hit_rate_drop_warn,
            threshold_fail=cfg.hit_rate_drop_fail,
            sample_size_ref=n_ref,
            sample_size_cur=n_cur,
            status=_status_low_is_bad(hr_drop, cfg.hit_rate_drop_warn, cfg.hit_rate_drop_fail),
            details={"hit_rate_ref": hr_ref, "hit_rate_cur": hr_cur},
        ),
        _mk_metric(
            name="performance_spread_drop",
            dimension="performance",
            entity="__global__",
            value=sp_drop,
            threshold_warn=cfg.spread_drop_warn,
            threshold_fail=cfg.spread_drop_fail,
            sample_size_ref=n_ref,
            sample_size_cur=n_cur,
            status=_status_low_is_bad(sp_drop, cfg.spread_drop_warn, cfg.spread_drop_fail),
            details={"spread_ref": sp_ref, "spread_cur": sp_cur},
        ),
    ]


def _classify_dimension(metrics: list[DriftMetricResult], dimension: str) -> str:
    m = [x.status for x in metrics if x.dimension == dimension]
    if len(m) == 0:
        return "not_applicable"
    if any(s == "fail" for s in m):
        return "fail"
    if any(s == "warn" for s in m):
        return "warn"
    if all(s in {"not_applicable"} for s in m):
        return "not_applicable"
    if any(s == "insufficient_sample" for s in m):
        # If there are pass + insufficient, keep a warning posture.
        if any(s == "pass" for s in m):
            return "warn"
        return "insufficient_sample"
    if any(s == "pass" for s in m):
        return "pass"
    return "not_applicable"


def _derive_global_decision(
    feature_status: str,
    prediction_status: str,
    performance_status: str,
    cfg: DriftDetectionConfig,
) -> tuple[str, str]:
    if performance_status == "fail" and cfg.fail_on_performance_fail:
        return "FAIL", "investigate_model_and_freeze_promotion"
    if feature_status == "fail" and cfg.fail_on_feature_fail:
        return "FAIL", "review_feature_pipeline"
    if prediction_status == "fail" and cfg.fail_on_prediction_fail:
        return "FAIL", "review_prediction_calibration"

    if "fail" in {feature_status, prediction_status, performance_status}:
        return "WARN", "review_drift_signals"

    if any(s in {"warn", "insufficient_sample"} for s in {feature_status, prediction_status, performance_status}):
        if "insufficient_sample" in {feature_status, prediction_status, performance_status}:
            return "WARN", "collect_more_data_and_review"
        return "WARN", "monitor_and_review"

    return "PASS", "no_action"


def _save_df(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return str(path)
    except Exception:
        p = path.with_suffix(".csv")
        df.to_csv(p, index=False)
        return str(p)


def _save_json(payload: Mapping[str, Any], path: Path) -> str:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(dict(payload)), sort_keys=True, indent=2, default=str), encoding="utf-8")
    return str(path)


def persist_drift_outputs(
    metrics: pd.DataFrame,
    summary: Mapping[str, Any],
    events: pd.DataFrame,
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "metrics": _save_df(metrics, out / "metrics.parquet"),
        "summary": _save_json(summary, out / "summary.json"),
        "events": _save_df(events, out / "events.parquet"),
    }


def run_drift_detection(
    *,
    features_ref: pd.DataFrame | str | Path | None = None,
    features_cur: pd.DataFrame | str | Path | None = None,
    predictions_ref: pd.DataFrame | str | Path | None = None,
    predictions_cur: pd.DataFrame | str | Path | None = None,
    performance_ref: pd.DataFrame | str | Path | None = None,
    performance_cur: pd.DataFrame | str | Path | None = None,
    config: DriftDetectionConfig | None = None,
    execution_date: str | pd.Timestamp | None = None,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or DriftDetectionConfig()
    rid = run_id or run_id_with_prefix("drift")
    exec_ts = pd.Timestamp(execution_date) if execution_date is not None else pd.Timestamp.utcnow()

    started_at = utc_now_iso()

    fr = _to_df(features_ref)
    fc = _to_df(features_cur)
    pr = _to_df(predictions_ref)
    pc = _to_df(predictions_cur)
    rr = _to_df(performance_ref)
    rc = _to_df(performance_cur)

    metrics: list[DriftMetricResult] = []

    if len(fr) and len(fc):
        metrics.extend(_feature_drift_suite(fr, fc, cfg))
    else:
        metrics.append(
            _mk_metric(
                name="feature_not_applicable",
                dimension="feature",
                entity="__global__",
                value=None,
                threshold_warn=None,
                threshold_fail=None,
                sample_size_ref=int(len(fr)),
                sample_size_cur=int(len(fc)),
                status="not_applicable",
                details={"reason": "feature_windows_missing"},
            )
        )

    if len(pr) and len(pc):
        metrics.extend(_prediction_drift_suite(pr, pc, cfg))
    else:
        metrics.append(
            _mk_metric(
                name="prediction_not_applicable",
                dimension="prediction",
                entity="__global__",
                value=None,
                threshold_warn=None,
                threshold_fail=None,
                sample_size_ref=int(len(pr)),
                sample_size_cur=int(len(pc)),
                status="not_applicable",
                details={"reason": "prediction_windows_missing"},
            )
        )

    if len(rr) and len(rc):
        metrics.extend(_performance_drift_suite(rr, rc, cfg))
    else:
        metrics.append(
            _mk_metric(
                name="performance_not_applicable",
                dimension="performance",
                entity="__global__",
                value=None,
                threshold_warn=None,
                threshold_fail=None,
                sample_size_ref=int(len(rr)),
                sample_size_cur=int(len(rc)),
                status="not_applicable",
                details={"reason": "performance_windows_missing"},
            )
        )

    feat_status = _classify_dimension(metrics, "feature")
    pred_status = _classify_dimension(metrics, "prediction")
    perf_status = _classify_dimension(metrics, "performance")

    global_status, action = _derive_global_decision(feat_status, pred_status, perf_status, cfg)

    ended_at = utc_now_iso()
    run_result = DriftRunResult(
        run_id=rid,
        execution_date=str(exec_ts.date()),
        reference_window={
            "features": int(len(fr)),
            "predictions": int(len(pr)),
            "performance": int(len(rr)),
        },
        current_window={
            "features": int(len(fc)),
            "predictions": int(len(pc)),
            "performance": int(len(rc)),
        },
        feature_status=feat_status,
        prediction_status=pred_status,
        performance_status=perf_status,
        global_status=global_status,
        action=action,
        metrics=tuple(metrics),
        started_at=started_at,
        ended_at=ended_at,
    )

    metric_df = pd.DataFrame([asdict(m) for m in metrics])
    metric_df["run_id"] = rid

    events_df = pd.DataFrame(
        [
            {
                "event_type": f"drift_feature_{feat_status}",
                "source": "drift_detection",
                "run_id": rid,
                "severity": ("high" if feat_status == "fail" else ("warning" if feat_status in {"warn", "insufficient_sample"} else "info")),
                "timestamp": ended_at,
                "payload": {"status": feat_status},
            },
            {
                "event_type": f"drift_prediction_{pred_status}",
                "source": "drift_detection",
                "run_id": rid,
                "severity": ("high" if pred_status == "fail" else ("warning" if pred_status in {"warn", "insufficient_sample"} else "info")),
                "timestamp": ended_at,
                "payload": {"status": pred_status},
            },
            {
                "event_type": f"drift_performance_{perf_status}",
                "source": "drift_detection",
                "run_id": rid,
                "severity": ("high" if perf_status == "fail" else ("warning" if perf_status in {"warn", "insufficient_sample"} else "info")),
                "timestamp": ended_at,
                "payload": {"status": perf_status},
            },
            {
                "event_type": "drift_run_finished",
                "source": "drift_detection",
                "run_id": rid,
                "severity": ("high" if global_status == "FAIL" else ("warning" if global_status == "WARN" else "info")),
                "timestamp": ended_at,
                "payload": {
                    "global_status": global_status,
                    "action": action,
                },
            },
        ]
    )

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_drift_outputs(
            metric_df,
            {
                "run_id": run_result.run_id,
                "execution_date": run_result.execution_date,
                "reference_window": run_result.reference_window,
                "current_window": run_result.current_window,
                "feature_status": run_result.feature_status,
                "prediction_status": run_result.prediction_status,
                "performance_status": run_result.performance_status,
                "global_status": run_result.global_status,
                "action": run_result.action,
                "started_at": run_result.started_at,
                "ended_at": run_result.ended_at,
                "config_hash": config_hash(cfg.__dict__, n=24),
                "num_metrics": int(len(metrics)),
            },
            events_df,
            output_dir=output_dir,
        )

    return {
        "run_result": run_result,
        "metrics": metric_df,
        "events": events_df,
        "artifacts": artifacts,
    }


__all__ = [
    "DriftDetectionConfig",
    "DriftMetricResult",
    "DriftRunResult",
    "persist_drift_outputs",
    "run_drift_detection",
]
