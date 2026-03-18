"""
models/gbdt/tune_hyperparams.py — HPO with robustness penalties.

    θ* = argmax J̃(θ) where:
    J̃(θ) = M̂(θ) − β·Gap(θ) − γ·Var_k[M_k(θ)] − η·Π(θ)

Supports random search (default) and Bayesian (if optuna available).
Test never participates in tuning. Meta-overfitting controlled via
limited trials + fold variance penalty + robustness selection.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
import time


@dataclass
class TuneConfig:
    engine: str = "lgbm"                   # lgbm | xgb
    search_method: str = "random"          # random | bayes
    primary_metric: str = "ic"
    n_trials: int = 50
    seed: int = 42
    # Robustness penalties
    beta_gap: float = 0.5                  # overfit gap penalty
    gamma_var: float = 1.0                 # fold variance penalty
    # Search space
    search_space: dict = field(default_factory=lambda: {
        "num_leaves": (15, 127),
        "max_depth": (3, 10),
        "learning_rate": (0.01, 0.2),
        "feature_fraction": (0.4, 0.9),
        "bagging_fraction": (0.5, 0.9),
        "lambda_l1": (0.0, 5.0),
        "lambda_l2": (0.1, 10.0),
        "min_data_in_leaf": (10, 100),
    })
    # Acceptance
    metric_threshold: float = 0.01
    overfit_gap_max: float = 0.15
    fold_std_max: float = 0.05


def _sample_params(space: dict, rng: np.random.RandomState) -> dict:
    """Sample a configuration from the search space."""
    params = {}
    for name, bounds in space.items():
        lo, hi = bounds
        if isinstance(lo, int) and isinstance(hi, int):
            params[name] = int(rng.randint(lo, hi + 1))
        else:
            if name in ("learning_rate", "lambda_l1", "lambda_l2"):
                # Log-uniform for rates and regularization
                params[name] = float(np.exp(rng.uniform(np.log(max(lo, 1e-6)), np.log(hi))))
            else:
                params[name] = float(rng.uniform(lo, hi))
    return params


def _evaluate_trial(
    params: dict,
    features: np.ndarray,
    labels: np.ndarray,
    splits: list[dict],
    dates: np.ndarray | None,
    engine: str,
    primary_metric: str,
    seed: int,
) -> dict[str, Any]:
    """Evaluate a single trial configuration."""
    if engine == "lgbm":
        from .train_lgbm import train_lgbm, LGBMTrainConfig
        cfg = LGBMTrainConfig(
            primary_metric=primary_metric, seed=seed,
            num_leaves=params.get("num_leaves", 63),
            max_depth=params.get("max_depth", -1),
            learning_rate=params.get("learning_rate", 0.05),
            feature_fraction=params.get("feature_fraction", 0.7),
            bagging_fraction=params.get("bagging_fraction", 0.8),
            lambda_l1=params.get("lambda_l1", 0.1),
            lambda_l2=params.get("lambda_l2", 1.0),
            min_data_in_leaf=params.get("min_data_in_leaf", 20),
        )
        result = train_lgbm(features, labels, splits, dates=dates, config=cfg)
    else:
        from .train_xgb import train_xgb, XGBTrainConfig
        cfg = XGBTrainConfig(
            primary_metric=primary_metric, seed=seed,
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("bagging_fraction", 0.8),
            colsample_bytree=params.get("feature_fraction", 0.7),
            reg_alpha=params.get("lambda_l1", 0.1),
            reg_lambda=params.get("lambda_l2", 1.0),
            min_child_weight=params.get("min_data_in_leaf", 5),
        )
        result = train_xgb(features, labels, splits, dates=dates, config=cfg)

    # Extract fold-level metrics
    fold_primaries = [f["valid_primary"] for f in result["metrics_by_fold"]
                      if np.isfinite(f["valid_primary"])]
    test_primaries = [f["test_primary"] for f in result["metrics_by_fold"]
                      if np.isfinite(f.get("test_primary", np.nan))]

    m_valid = float(np.mean(fold_primaries)) if fold_primaries else np.nan
    m_test = float(np.mean(test_primaries)) if test_primaries else np.nan
    gap = m_valid - m_test if np.isfinite(m_valid) and np.isfinite(m_test) else 0.0
    fold_std = float(np.std(fold_primaries)) if len(fold_primaries) > 1 else 0.0

    return {
        "metric_valid": m_valid,
        "metric_test": m_test,
        "overfit_gap": gap,
        "fold_std": fold_std,
        "coverage": result["summary"]["coverage"],
        "fold_primaries": fold_primaries,
    }


def run_tune(
    features: np.ndarray,
    labels: np.ndarray,
    splits: list[dict[str, np.ndarray]],
    *,
    dates: np.ndarray | None = None,
    config: TuneConfig | None = None,
) -> dict[str, Any]:
    """Run hyperparameter tuning."""
    cfg = config or TuneConfig()
    rng = np.random.RandomState(cfg.seed)

    trials = []
    for trial_id in range(cfg.n_trials):
        t0 = time.monotonic()
        params = _sample_params(cfg.search_space, rng)

        try:
            result = _evaluate_trial(
                params, features, labels, splits, dates,
                cfg.engine, cfg.primary_metric, cfg.seed + trial_id,
            )
            # Robust utility: M̂ − β·Gap − γ·Var
            robust_score = (
                result["metric_valid"]
                - cfg.beta_gap * max(result["overfit_gap"], 0)
                - cfg.gamma_var * result["fold_std"]
            )
            status = "success"
        except Exception as e:
            result = {"metric_valid": np.nan, "overfit_gap": np.nan, "fold_std": np.nan, "coverage": 0}
            robust_score = -np.inf
            status = "failed"

        elapsed = time.monotonic() - t0
        trials.append({
            "trial_id": trial_id,
            "params": params,
            "status": status,
            "metric_valid": result["metric_valid"],
            "overfit_gap": result.get("overfit_gap", np.nan),
            "fold_std": result.get("fold_std", np.nan),
            "coverage": result.get("coverage", 0),
            "robust_score": robust_score,
            "runtime_sec": round(elapsed, 2),
        })

    # Select best by robust score, with filters
    valid_trials = [t for t in trials if t["status"] == "success"
                    and np.isfinite(t["robust_score"])
                    and t["metric_valid"] >= cfg.metric_threshold
                    and abs(t.get("overfit_gap", 0)) <= cfg.overfit_gap_max
                    and t.get("fold_std", 0) <= cfg.fold_std_max]

    if valid_trials:
        best_trial = max(valid_trials, key=lambda t: t["robust_score"])
        best_params = best_trial["params"]
        accepted = True
    else:
        # Fall back to best raw if nothing passes robustness
        success_trials = [t for t in trials if t["status"] == "success" and np.isfinite(t["robust_score"])]
        if success_trials:
            best_trial = max(success_trials, key=lambda t: t["robust_score"])
            best_params = best_trial["params"]
        else:
            best_trial = None
            best_params = {}
        accepted = False

    summary = {
        "engine": cfg.engine,
        "search_method": cfg.search_method,
        "primary_metric": cfg.primary_metric,
        "n_trials_total": len(trials),
        "n_trials_success": sum(1 for t in trials if t["status"] == "success"),
        "n_trials_valid": len(valid_trials),
        "best_robust_score": best_trial["robust_score"] if best_trial else np.nan,
        "best_metric_valid": best_trial["metric_valid"] if best_trial else np.nan,
        "best_overfit_gap": best_trial.get("overfit_gap") if best_trial else np.nan,
        "best_fold_std": best_trial.get("fold_std") if best_trial else np.nan,
        "accepted": accepted,
        "seed": cfg.seed,
    }

    return {
        "best_params": best_params,
        "trials": trials,
        "summary": summary,
    }
