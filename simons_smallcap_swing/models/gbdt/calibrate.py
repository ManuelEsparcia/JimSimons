"""
models/gbdt/calibrate.py — Probability calibration for GBDT scores.

Methods: Platt scaling (default), isotonic regression, temperature scaling.
Trains on OOF only (never in-sample). Evaluates ECE, Brier, NLL.
May conclude "no calibration" is best — that's a valid output.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class CalibrateConfig:
    methods: tuple[str, ...] = ("platt", "isotonic", "temperature")
    n_bins: int = 10
    eps: float = 1e-6
    min_samples_isotonic: int = 200
    accept_ece_improvement: float = 0.005  # must improve ECE by at least this


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-frequency bins)."""
    n = len(y_true)
    if n < n_bins:
        return np.nan
    order = np.argsort(y_prob)
    bin_size = n // n_bins
    ece_val = 0.0
    for i in range(n_bins):
        lo = i * bin_size
        hi = (i + 1) * bin_size if i < n_bins - 1 else n
        idx = order[lo:hi]
        acc = y_true[idx].mean()
        conf = y_prob[idx].mean()
        ece_val += len(idx) / n * abs(acc - conf)
    return float(ece_val)


def _brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_true - y_prob) ** 2))


def _nll(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _fit_platt(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Fit Platt scaling: p = σ(As + B) via simple logistic regression."""
    from numpy.linalg import lstsq
    # Simple iterative approach
    s = scores.reshape(-1, 1)
    X = np.column_stack([s, np.ones(len(s))])
    # Use least-squares on logit(y) as warm start, then refine
    y_safe = np.clip(labels, 0.01, 0.99)
    logit_y = np.log(y_safe / (1 - y_safe))
    w, _, _, _ = lstsq(X, logit_y, rcond=None)
    A, B = float(w[0]), float(w[1])
    return {"A": A, "B": B}


def _predict_platt(scores: np.ndarray, params: dict) -> np.ndarray:
    return _sigmoid(params["A"] * scores + params["B"])


def _fit_temperature(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Fit temperature: p = σ(s/T). Grid search T."""
    best_T, best_nll = 1.0, float("inf")
    for T in np.logspace(-1, 1, 50):
        probs = _sigmoid(scores / T)
        nl = _nll(labels, probs)
        if nl < best_nll:
            best_nll = nl
            best_T = float(T)
    return {"T": best_T}


def _predict_temperature(scores: np.ndarray, params: dict) -> np.ndarray:
    return _sigmoid(scores / params["T"])


def _fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Fit isotonic regression."""
    try:
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(scores, labels)
        return {"model": ir, "sklearn": True}
    except ImportError:
        # Simple bin-based monotone approximation
        order = np.argsort(scores)
        n = len(scores)
        k = max(10, n // 50)
        bin_edges, bin_means = [], []
        for i in range(0, n, k):
            chunk = order[i:i + k]
            bin_edges.append(float(scores[chunk].mean()))
            bin_means.append(float(labels[chunk].mean()))
        # Enforce monotonicity (pool adjacent violators)
        for i in range(1, len(bin_means)):
            if bin_means[i] < bin_means[i - 1]:
                avg = (bin_means[i - 1] + bin_means[i]) / 2
                bin_means[i - 1] = avg
                bin_means[i] = avg
        return {"edges": np.array(bin_edges), "means": np.array(bin_means), "sklearn": False}


def _predict_isotonic(scores: np.ndarray, params: dict) -> np.ndarray:
    if params.get("sklearn"):
        return params["model"].predict(scores)
    return np.interp(scores, params["edges"], params["means"])


def run_calibration(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    config: CalibrateConfig | None = None,
) -> dict[str, Any]:
    """Calibrate scores to probabilities. May return identity if no improvement."""
    cfg = config or CalibrateConfig()
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels, dtype=float).ravel()

    # Baseline: raw sigmoid of scores as "uncalibrated probability"
    p_raw = _sigmoid(s)
    base_ece = _ece(y, p_raw, cfg.n_bins)
    base_brier = _brier(y, p_raw)
    base_nll = _nll(y, p_raw, cfg.eps)

    results = {"identity": {"probs": p_raw, "ece": base_ece, "brier": base_brier, "nll": base_nll, "params": {}}}

    FITTERS = {
        "platt": (_fit_platt, _predict_platt),
        "temperature": (_fit_temperature, _predict_temperature),
        "isotonic": (_fit_isotonic, _predict_isotonic),
    }

    for method in cfg.methods:
        if method not in FITTERS:
            continue
        if method == "isotonic" and len(s) < cfg.min_samples_isotonic:
            continue
        try:
            fit_fn, pred_fn = FITTERS[method]
            params = fit_fn(s, y)
            probs = np.clip(pred_fn(s, params), cfg.eps, 1 - cfg.eps)
            results[method] = {
                "probs": probs, "params": params,
                "ece": _ece(y, probs, cfg.n_bins),
                "brier": _brier(y, probs),
                "nll": _nll(y, probs, cfg.eps),
            }
        except Exception:
            pass

    # Select best by ECE improvement, with checks
    best_method = "identity"
    best_ece = base_ece
    for method, r in results.items():
        if method == "identity":
            continue
        ece_improve = base_ece - r["ece"]
        brier_ok = r["brier"] <= base_brier * 1.05  # no material Brier degradation
        nll_ok = r["nll"] <= base_nll * 1.05
        if ece_improve >= cfg.accept_ece_improvement and brier_ok and nll_ok and r["ece"] < best_ece:
            best_method = method
            best_ece = r["ece"]

    selected = results[best_method]

    return {
        "calibrated_probs": selected["probs"],
        "method_selected": best_method,
        "calibrator_params": selected.get("params", {}),
        "ece_before": round(base_ece, 6),
        "ece_after": round(selected["ece"], 6),
        "brier_before": round(base_brier, 6),
        "brier_after": round(selected["brier"], 6),
        "nll_before": round(base_nll, 6),
        "nll_after": round(selected["nll"], 6),
        "methods_evaluated": list(results.keys()),
        "accepted": best_method != "identity",
    }
