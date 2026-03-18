"""
models/regimes/hmm.py - Hidden Markov regime inference.

Fits Gaussian HMMs on PIT regime features, selects number of states with
predictive and governance constraints, aligns state semantics across runs,
and emits filtered probabilities for downstream live gating.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


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


def _write_parquet_safe(df: pd.DataFrame, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(p, index=False)
        return str(p)
    except Exception:
        alt = p.with_suffix(".csv")
        df.to_csv(alt, index=False)
        return str(alt)


def _write_json_safe(payload: Mapping[str, Any], path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(json_safe(dict(payload)), sort_keys=True, indent=2, default=str),
        encoding="utf-8",
    )
    return str(p)


@dataclass(frozen=True)
class HMMConfig:
    k_candidates: tuple[int, ...] = (2, 3, 4, 5)
    covariance_type: str = "diag"  # diag | full
    train_window_mode: str = "rolling"  # rolling | expanding
    train_window_length: int = 756
    retrain_frequency: int = 21
    n_restarts: int = 5
    max_em_iter: int = 200
    tol_loglik: float = 1e-4
    cov_reg_lambda: float = 1e-5

    min_state_occupancy: float = 0.05
    min_median_duration: float = 3.0
    max_mean_entropy: float = 1.30

    selection_policy: str = "predictive_plus_bic"
    alignment_policy: str = "hungarian"
    acceptance_policy: str = "strict"

    random_seed: int = 42
    policy_version: str = "v1"
    run_id: str = ""


def load_regime_feature_matrix(regime_features: pd.DataFrame | str | Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = _to_df(regime_features)
    if len(df) == 0:
        raise ValueError("regime_features is empty")
    if "date" not in df.columns:
        raise ValueError("regime_features must include date column")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    drop_cols = {"date", "run_id", "low_confidence_flag_t"}
    feature_cols = [c for c in out.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(out[c])]
    if not feature_cols:
        raise ValueError("no numeric regime features available for HMM")

    X = out[feature_cols].to_numpy(dtype=float)
    # Causal imputation only uses past values through ffill, then global median fallback.
    Xdf = pd.DataFrame(X, columns=feature_cols)
    Xdf = Xdf.ffill()
    for c in feature_cols:
        med = float(np.nanmedian(Xdf[c].to_numpy(dtype=float))) if np.isfinite(Xdf[c].to_numpy(dtype=float)).any() else 0.0
        Xdf[c] = Xdf[c].fillna(med)
    X = Xdf.to_numpy(dtype=float)

    return out[["date"]].copy(), X, feature_cols


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    amax = np.nanmax(a, axis=axis, keepdims=True)
    shifted = a - amax
    s = np.nansum(np.exp(shifted), axis=axis, keepdims=True)
    out = amax + np.log(np.maximum(s, 1e-300))
    if axis is None:
        return np.asarray(out).reshape(())
    return np.squeeze(out, axis=axis)


def _gaussian_logpdf_diag(X: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    var = np.maximum(var, 1e-12)
    d = X.shape[1]
    diff = X - mean
    term = np.sum((diff * diff) / var, axis=1)
    ldet = np.sum(np.log(var))
    return -0.5 * (d * math.log(2 * math.pi) + ldet + term)


def _gaussian_logpdf_full(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    cov = cov + 1e-9 * np.eye(cov.shape[0])
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        cov = cov + 1e-6 * np.eye(cov.shape[0])
        sign, logdet = np.linalg.slogdet(cov)
    inv = np.linalg.pinv(cov)
    diff = X - mean
    q = np.einsum("ij,jk,ik->i", diff, inv, diff)
    d = X.shape[1]
    return -0.5 * (d * math.log(2 * math.pi) + logdet + q)


def _emission_log_probs(
    X: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    covariance_type: str,
) -> np.ndarray:
    T = X.shape[0]
    K = means.shape[0]
    out = np.zeros((T, K), dtype=float)
    for k in range(K):
        if covariance_type == "full":
            out[:, k] = _gaussian_logpdf_full(X, means[k], covs[k])
        else:
            out[:, k] = _gaussian_logpdf_diag(X, means[k], covs[k])
    return out


def _forward_filter(pi: np.ndarray, A: np.ndarray, log_emission: np.ndarray) -> tuple[np.ndarray, float]:
    T, K = log_emission.shape
    log_pi = np.log(np.maximum(pi, 1e-300))
    log_A = np.log(np.maximum(A, 1e-300))

    log_alpha = np.zeros((T, K), dtype=float)
    log_alpha[0] = log_pi + log_emission[0]

    for t in range(1, T):
        temp = log_alpha[t - 1][:, None] + log_A
        log_alpha[t] = log_emission[t] + _logsumexp(temp, axis=0)

    filtered = np.exp(log_alpha - _logsumexp(log_alpha, axis=1)[:, None])
    loglik = float(_logsumexp(log_alpha[-1]))
    return filtered, loglik


def _forward_backward_smoothing(pi: np.ndarray, A: np.ndarray, log_emission: np.ndarray) -> np.ndarray:
    T, K = log_emission.shape
    log_pi = np.log(np.maximum(pi, 1e-300))
    log_A = np.log(np.maximum(A, 1e-300))

    log_alpha = np.zeros((T, K), dtype=float)
    log_beta = np.zeros((T, K), dtype=float)

    log_alpha[0] = log_pi + log_emission[0]
    for t in range(1, T):
        temp = log_alpha[t - 1][:, None] + log_A
        log_alpha[t] = log_emission[t] + _logsumexp(temp, axis=0)

    log_beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        temp = log_A + log_emission[t + 1][None, :] + log_beta[t + 1][None, :]
        log_beta[t] = _logsumexp(temp, axis=1)

    log_gamma = log_alpha + log_beta
    log_gamma = log_gamma - _logsumexp(log_gamma, axis=1)[:, None]
    return np.exp(log_gamma)


def _viterbi_path(pi: np.ndarray, A: np.ndarray, log_emission: np.ndarray) -> np.ndarray:
    T, K = log_emission.shape
    log_pi = np.log(np.maximum(pi, 1e-300))
    log_A = np.log(np.maximum(A, 1e-300))

    dp = np.zeros((T, K), dtype=float)
    ptr = np.zeros((T, K), dtype=int)

    dp[0] = log_pi + log_emission[0]
    for t in range(1, T):
        for j in range(K):
            vals = dp[t - 1] + log_A[:, j]
            ptr[t, j] = int(np.argmax(vals))
            dp[t, j] = vals[ptr[t, j]] + log_emission[t, j]

    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = ptr[t + 1, path[t + 1]]
    return path

def _fit_candidate_hmm(
    X_train: np.ndarray,
    X_all: np.ndarray,
    K: int,
    cfg: HMMConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    # Preferred implementation: hmmlearn GaussianHMM.
    try:
        from hmmlearn.hmm import GaussianHMM

        model = GaussianHMM(
            n_components=int(K),
            covariance_type=str(cfg.covariance_type),
            n_iter=int(cfg.max_em_iter),
            tol=float(cfg.tol_loglik),
            random_state=int(seed),
        )
        model.fit(X_train)

        pi = np.asarray(model.startprob_, dtype=float)
        A = np.asarray(model.transmat_, dtype=float)
        means = np.asarray(model.means_, dtype=float)
        covars = np.asarray(model.covars_, dtype=float)
        if cfg.covariance_type == "diag" and covars.ndim == 3:
            covars = np.array([np.diag(c) for c in covars], dtype=float)

        log_em = _emission_log_probs(X_all, means, covars, cfg.covariance_type)
        filtered, loglik_all = _forward_filter(pi, A, log_em)
        try:
            smoothed = np.asarray(model.predict_proba(X_all), dtype=float)
        except Exception:
            smoothed = _forward_backward_smoothing(pi, A, log_em)
        viterbi = _viterbi_path(pi, A, log_em)

        monitor = getattr(model, "monitor_", None)
        converged = bool(getattr(monitor, "converged", True)) if monitor is not None else True
        n_iter = int(getattr(monitor, "iter", np.nan)) if monitor is not None else np.nan
        history = list(getattr(monitor, "history", [])) if monitor is not None else []

        loglik_train = float(model.score(X_train)) * max(len(X_train), 1)

        return {
            "backend": "hmmlearn",
            "pi": pi,
            "A": A,
            "means": means,
            "covars": covars,
            "filtered": filtered,
            "smoothed": smoothed,
            "viterbi": viterbi,
            "loglik_train": loglik_train,
            "loglik_all": float(loglik_all),
            "converged": converged,
            "n_iter": int(n_iter) if np.isfinite(n_iter) else None,
            "loglik_history": [float(x) for x in history],
        }

    except Exception:
        # Fallback: GMM approximation with estimated transition matrix.
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(
            n_components=int(K),
            covariance_type="full" if cfg.covariance_type == "full" else "diag",
            random_state=int(seed),
            reg_covar=float(cfg.cov_reg_lambda),
            max_iter=int(cfg.max_em_iter),
            tol=float(cfg.tol_loglik),
        )
        gmm.fit(X_train)

        means = np.asarray(gmm.means_, dtype=float)
        covars = np.asarray(gmm.covariances_, dtype=float)
        if cfg.covariance_type == "diag" and covars.ndim == 3:
            covars = np.array([np.diag(c) for c in covars], dtype=float)

        post_train = np.asarray(gmm.predict_proba(X_train), dtype=float)
        states_train = post_train.argmax(axis=1)

        pi = np.zeros(K, dtype=float)
        pi[states_train[0]] = 1.0
        pi = np.clip(pi + 1e-4, 1e-8, None)
        pi = pi / pi.sum()

        A = np.ones((K, K), dtype=float) * 1e-4
        for a, b in zip(states_train[:-1], states_train[1:]):
            A[int(a), int(b)] += 1.0
        A = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-12)

        log_em = _emission_log_probs(X_all, means, covars, cfg.covariance_type)
        filtered, loglik_all = _forward_filter(pi, A, log_em)
        smoothed = _forward_backward_smoothing(pi, A, log_em)
        viterbi = _viterbi_path(pi, A, log_em)

        loglik_train = float(gmm.score(X_train)) * max(len(X_train), 1)

        return {
            "backend": "gmm_fallback",
            "pi": pi,
            "A": A,
            "means": means,
            "covars": covars,
            "filtered": filtered,
            "smoothed": smoothed,
            "viterbi": viterbi,
            "loglik_train": loglik_train,
            "loglik_all": float(loglik_all),
            "converged": bool(getattr(gmm, "converged_", True)),
            "n_iter": int(getattr(gmm, "n_iter_", np.nan)) if hasattr(gmm, "n_iter_") else None,
            "loglik_history": [],
        }


def _durations_by_state(states: np.ndarray, K: int) -> dict[int, list[int]]:
    out = {k: [] for k in range(K)}
    if len(states) == 0:
        return out

    curr = int(states[0])
    run = 1
    for s in states[1:]:
        s = int(s)
        if s == curr:
            run += 1
        else:
            out[curr].append(run)
            curr = s
            run = 1
    out[curr].append(run)
    return out


def _candidate_diagnostics(
    fit: Mapping[str, Any],
    K: int,
    cfg: HMMConfig,
    n_obs: int,
    n_features: int,
) -> dict[str, Any]:
    filtered = np.asarray(fit["filtered"], dtype=float)
    dominant = filtered.argmax(axis=1)

    occupancy = {int(k): float((dominant == k).mean()) for k in range(K)}
    entropy = -np.sum(filtered * np.log(np.maximum(filtered, 1e-12)), axis=1)
    mean_entropy = float(np.mean(entropy))

    dur = _durations_by_state(dominant, K)
    median_duration = {int(k): float(np.median(v)) if len(v) else 0.0 for k, v in dur.items()}
    mean_duration = {int(k): float(np.mean(v)) if len(v) else 0.0 for k, v in dur.items()}

    A = np.asarray(fit["A"], dtype=float)
    expected_duration = {int(k): float(1.0 / max(1e-6, 1.0 - A[k, k])) for k in range(K)}
    duration_gap = {
        int(k): float(abs(mean_duration[k] - expected_duration[k]) / max(expected_duration[k], 1e-6))
        for k in range(K)
    }
    duration_misspec_flag = bool(any(v > 0.50 for v in duration_gap.values()))

    # Approximate BIC.
    if cfg.covariance_type == "full":
        cov_params = K * n_features * (n_features + 1) / 2
    else:
        cov_params = K * n_features
    n_params = (K - 1) + K * (K - 1) + K * n_features + cov_params
    loglik_train = float(fit["loglik_train"])
    bic = float(-2.0 * loglik_train + n_params * math.log(max(n_obs, 2)))

    hard_fail_reasons = []
    if min(occupancy.values()) < cfg.min_state_occupancy:
        hard_fail_reasons.append("low_state_occupancy")
    if min(median_duration.values()) < cfg.min_median_duration:
        hard_fail_reasons.append("low_median_duration")
    if mean_entropy > cfg.max_mean_entropy:
        hard_fail_reasons.append("high_entropy")

    return {
        "occupancy": occupancy,
        "mean_entropy": mean_entropy,
        "median_duration": median_duration,
        "mean_duration": mean_duration,
        "expected_duration": expected_duration,
        "duration_gap": duration_gap,
        "duration_misspec_flag": duration_misspec_flag,
        "bic": bic,
        "hard_fail_reasons": hard_fail_reasons,
        "eligible": len(hard_fail_reasons) == 0,
    }


def _alignment_cost_matrix(old_params: Mapping[str, Any], new_params: Mapping[str, Any]) -> np.ndarray:
    old_mu = np.asarray(old_params["means"], dtype=float)
    new_mu = np.asarray(new_params["means"], dtype=float)
    old_A = np.asarray(old_params["A"], dtype=float)
    new_A = np.asarray(new_params["A"], dtype=float)

    K_old = old_mu.shape[0]
    K_new = new_mu.shape[0]
    C = np.zeros((K_old, K_new), dtype=float)

    for i in range(K_old):
        for j in range(K_new):
            d_mu = float(np.linalg.norm(old_mu[i] - new_mu[j]))
            d_A = float(np.linalg.norm(old_A[i] - new_A[j])) if i < old_A.shape[0] and j < new_A.shape[0] else 0.0
            C[i, j] = d_mu + 0.5 * d_A
    return C


def _align_states(old_params: Mapping[str, Any] | None, fit: dict[str, Any]) -> tuple[dict[int, int], float, bool]:
    if old_params is None:
        K = fit["means"].shape[0]
        return {k: k for k in range(K)}, 0.0, True

    C = _alignment_cost_matrix(old_params, fit)
    try:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(C)
        mapping = {int(old): int(new) for old, new in zip(r, c)}
        cost = float(C[r, c].mean()) if len(r) else 0.0
    except Exception:
        # Greedy fallback.
        mapping = {}
        used = set()
        for i in range(C.shape[0]):
            j = int(np.argmin(C[i]))
            if j in used:
                continue
            mapping[i] = j
            used.add(j)
        cost = float(np.mean([C[i, j] for i, j in mapping.items()])) if mapping else float("inf")

    unstable = bool(cost > 2.0)
    return mapping, cost, not unstable

def persist_hmm_outputs(
    state_probs_filtered: pd.DataFrame,
    state_probs_smoothed: pd.DataFrame,
    state_sequence: pd.DataFrame,
    hmm_params: Mapping[str, Any],
    hmm_summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    return {
        "state_probs_filtered": _write_parquet_safe(state_probs_filtered, out / "state_probs_filtered.parquet"),
        "state_probs_smoothed": _write_parquet_safe(state_probs_smoothed, out / "state_probs_smoothed.parquet"),
        "state_sequence": _write_parquet_safe(state_sequence, out / "state_sequence.parquet"),
        "hmm_params": _write_json_safe(dict(hmm_params), out / "hmm_params.json"),
        "hmm_summary": _write_json_safe(dict(hmm_summary), out / "hmm_summary.json"),
        "manifest": _write_json_safe(dict(manifest), out / "manifest.json"),
    }


def run_hmm(
    regime_features: pd.DataFrame | str | Path,
    *,
    config: HMMConfig | None = None,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    data_snapshot_id: str = "unknown",
    previous_hmm_params: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    cfg = config or HMMConfig()
    rid = run_id or cfg.run_id or run_id_with_prefix("hmm")

    date_df, X_all, feature_cols = load_regime_feature_matrix(regime_features)
    n_total, n_features = X_all.shape

    if cfg.train_window_mode.lower() == "rolling" and n_total > cfg.train_window_length:
        X_train = X_all[-cfg.train_window_length :]
    else:
        X_train = X_all

    old_params_obj: Mapping[str, Any] | None = None
    if isinstance(previous_hmm_params, Mapping):
        old_params_obj = previous_hmm_params
    elif isinstance(previous_hmm_params, (str, Path)):
        p_prev = Path(previous_hmm_params)
        if p_prev.exists():
            try:
                old_params_obj = json.loads(p_prev.read_text(encoding="utf-8"))
            except Exception:
                old_raw = _to_df(p_prev)
                if len(old_raw):
                    old_params_obj = dict(old_raw.iloc[0].to_dict())

    candidates = []
    for K in cfg.k_candidates:
        best_fit = None
        best_score = -np.inf

        for r in range(cfg.n_restarts):
            seed = int(cfg.random_seed + 1000 * K + r)
            fit = _fit_candidate_hmm(X_train, X_all, int(K), cfg, seed=seed)
            diag = _candidate_diagnostics(fit, int(K), cfg, n_obs=len(X_train), n_features=n_features)

            # Predictive score prioritizes per-observation train log-score.
            pred_score = float(fit["loglik_train"] / max(len(X_train), 1))
            bic_norm = float(diag["bic"] / max(len(X_train), 1))
            deg_pen = float(len(diag["hard_fail_reasons"]))
            select_score = pred_score - 0.05 * bic_norm - 0.50 * deg_pen

            if select_score > best_score:
                best_score = select_score
                best_fit = {**fit, **diag, "selection_score": select_score, "K": int(K), "restart": int(r)}

        if best_fit is not None:
            candidates.append(best_fit)

    if not candidates:
        raise ValueError("HMM fitting failed for all K candidates")

    eligible = [c for c in candidates if c["eligible"]]
    chosen = max(eligible if eligible else candidates, key=lambda d: d["selection_score"])

    mapping, align_cost, aligned_flag = _align_states(old_params_obj, chosen)

    K = int(chosen["K"])
    filtered = np.asarray(chosen["filtered"], dtype=float)
    smoothed = np.asarray(chosen["smoothed"], dtype=float)
    viterbi = np.asarray(chosen["viterbi"], dtype=int)

    dominant_f = filtered.argmax(axis=1)
    dominant_s = smoothed.argmax(axis=1)
    conf = filtered.max(axis=1)
    ent = -np.sum(filtered * np.log(np.maximum(filtered, 1e-12)), axis=1)

    state_probs_filtered = date_df.copy()
    for k in range(K):
        state_probs_filtered[f"p_state_{k}_filtered"] = filtered[:, k]
    state_probs_filtered["dominant_state_filtered"] = dominant_f
    state_probs_filtered["state_confidence"] = conf
    state_probs_filtered["state_entropy"] = ent
    state_probs_filtered["run_id"] = rid

    state_probs_smoothed = date_df.copy()
    for k in range(K):
        state_probs_smoothed[f"p_state_{k}_smoothed"] = smoothed[:, k]
    state_probs_smoothed["dominant_state_smoothed"] = dominant_s
    state_probs_smoothed["state_confidence"] = smoothed.max(axis=1)
    state_probs_smoothed["state_entropy"] = -np.sum(smoothed * np.log(np.maximum(smoothed, 1e-12)), axis=1)
    state_probs_smoothed["run_id"] = rid

    state_sequence = date_df.copy()
    state_sequence["viterbi_state"] = viterbi
    state_sequence["dominant_state_filtered"] = dominant_f
    state_sequence["dominant_state_smoothed"] = dominant_s
    state_sequence["run_id"] = rid

    structural_break_flag = bool((not aligned_flag) or align_cost > 2.0)
    model_accepted = bool(
        chosen["converged"]
        and chosen["eligible"]
        and aligned_flag
        and (chosen["mean_entropy"] <= cfg.max_mean_entropy)
    )

    rejection_reason = ""
    if not model_accepted:
        reasons = []
        if not chosen["converged"]:
            reasons.append("not_converged")
        reasons.extend(chosen["hard_fail_reasons"])
        if not aligned_flag:
            reasons.append("unstable_mapping")
        rejection_reason = ",".join(sorted(set(reasons)))

    hmm_params = {
        "run_id": rid,
        "K": K,
        "pi": np.asarray(chosen["pi"], dtype=float).tolist(),
        "A": np.asarray(chosen["A"], dtype=float).tolist(),
        "mu": np.asarray(chosen["means"], dtype=float).tolist(),
        "Sigma": np.asarray(chosen["covars"], dtype=float).tolist(),
        "covariance_type": cfg.covariance_type,
        "lambda_reg": cfg.cov_reg_lambda,
        "loglik_final": float(chosen["loglik_train"]),
        "n_iter": chosen["n_iter"],
        "converged_flag": bool(chosen["converged"]),
        "backend": chosen["backend"],
    }

    hmm_summary = {
        "run_id": rid,
        "K_selected": K,
        "predictive_score": float(chosen["loglik_train"] / max(len(X_train), 1)),
        "BIC": float(chosen["bic"]),
        "occupancy_by_state": chosen["occupancy"],
        "median_duration_by_state": chosen["median_duration"],
        "mean_entropy": float(chosen["mean_entropy"]),
        "alignment_cost_mean": float(align_cost),
        "duration_misspec_flag": bool(chosen["duration_misspec_flag"]),
        "structural_break_flag": structural_break_flag,
        "model_accepted": model_accepted,
        "rejection_reason": rejection_reason,
        "config_hash": config_hash(asdict(cfg), n=24),
        "mapping_old_to_new": mapping,
        "selection_table": [
            {
                "K": int(c["K"]),
                "selection_score": float(c["selection_score"]),
                "eligible": bool(c["eligible"]),
                "mean_entropy": float(c["mean_entropy"]),
                "bic": float(c["bic"]),
            }
            for c in candidates
        ],
    }

    manifest = {
        "run_id": rid,
        "timestamp": utc_now_iso(),
        "input_features": feature_cols,
        "n_dates": int(n_total),
        "n_features": int(n_features),
        "train_window_mode": cfg.train_window_mode,
        "train_window_length": cfg.train_window_length,
        "retrain_frequency": cfg.retrain_frequency,
        "no_smoothing_in_live_assertion": True,
        "alignment_policy": cfg.alignment_policy,
        "acceptance_policy": cfg.acceptance_policy,
        "policy_version": cfg.policy_version,
        "data_snapshot_id": data_snapshot_id,
        "runtime_sec": float(time.perf_counter() - t0),
    }

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_hmm_outputs(
            state_probs_filtered,
            state_probs_smoothed,
            state_sequence,
            hmm_params,
            hmm_summary,
            manifest,
            output_dir=output_dir,
        )

    return {
        "state_probs_filtered": state_probs_filtered,
        "state_probs_smoothed": state_probs_smoothed,
        "state_sequence": state_sequence,
        "hmm_params": hmm_params,
        "hmm_summary": hmm_summary,
        "manifest": manifest,
        "artifacts": artifacts,
    }


__all__ = [
    "HMMConfig",
    "load_regime_feature_matrix",
    "persist_hmm_outputs",
    "run_hmm",
]
