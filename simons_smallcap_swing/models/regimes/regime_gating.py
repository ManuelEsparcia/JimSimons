"""
models/regimes/regime_gating.py - Operational policy layer from inferred regimes.

Consumes filtered HMM probabilities and maps them into stable, causal gating
actions over models, ensemble weights, risk caps, and trading permissions.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict, field
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
class RegimeGatingConfig:
    confidence_floor: float = 0.45
    confidence_full: float = 0.75
    confidence_min_transition: float = 0.55
    hysteresis_window: int = 3
    transition_margin: float = 0.10
    min_regime_duration: int = 2

    uncertain_policy: str = "neutral"  # neutral | keep_previous
    degraded_mode_policy: str = "neutral_gating_policy"

    kappa_params_by_regime: Mapping[str, Any] = field(
        default_factory=lambda: {
            "neutral": {
                "gross": {"a": 0.70, "b": 0.10, "c": 0.00, "min": 0.40, "max": 0.95},
                "net": {"a": 0.70, "b": 0.10, "c": 0.00, "min": 0.40, "max": 0.95},
                "turnover": {"a": 0.80, "b": 0.10, "c": 0.00, "min": 0.50, "max": 1.00},
                "short": {"a": 0.70, "b": 0.10, "c": 0.00, "min": 0.40, "max": 0.95},
            }
        }
    )
    model_gating_policy: Mapping[str, Any] = field(default_factory=dict)
    weight_gating_policy: Mapping[str, Any] = field(default_factory=dict)
    trade_gating_policy: Mapping[str, Any] = field(default_factory=dict)

    emergency_transition_policy: bool = True
    policy_version: str = "v1"
    run_id: str = ""


def load_regime_states(regime_states: pd.DataFrame | str | Path) -> tuple[pd.DataFrame, list[str]]:
    df = _to_df(regime_states)
    if len(df) == 0:
        raise ValueError("regime_states input is empty")

    out = df.copy()
    if "date" not in out.columns:
        raise ValueError("regime_states must include date")

    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    p_cols = [c for c in out.columns if c.startswith("p_state_") and c.endswith("_filtered")]
    if not p_cols:
        raise ValueError("regime_states requires p_state_k_filtered columns")

    probs = out[p_cols].to_numpy(dtype=float)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    row_sum = probs.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    probs = probs / row_sum
    out[p_cols] = probs

    if "dominant_state_filtered" not in out.columns:
        out["dominant_state_filtered"] = probs.argmax(axis=1)
    if "state_confidence" not in out.columns:
        out["state_confidence"] = probs.max(axis=1)
    if "state_entropy" not in out.columns:
        out["state_entropy"] = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
    if "model_accepted" not in out.columns:
        out["model_accepted"] = True

    out["dominant_state_filtered"] = pd.to_numeric(out["dominant_state_filtered"], errors="coerce").fillna(0).astype(int)
    out["state_confidence"] = pd.to_numeric(out["state_confidence"], errors="coerce").fillna(0.0)
    out["state_entropy"] = pd.to_numeric(out["state_entropy"], errors="coerce").fillna(np.nan)
    out["model_accepted"] = out["model_accepted"].astype(bool)

    return out, p_cols


def load_model_predictions(model_predictions: pd.DataFrame | str | Path) -> pd.DataFrame:
    df = _to_df(model_predictions)
    if len(df) == 0:
        raise ValueError("model_predictions input is empty")

    out = df.copy()
    rename = {}
    if "asset" in out.columns and "symbol" not in out.columns:
        rename["asset"] = "symbol"
    if "prediction" in out.columns and "raw_score" not in out.columns:
        rename["prediction"] = "raw_score"
    if "score" in out.columns and "raw_score" not in out.columns:
        rename["score"] = "raw_score"
    if "model" in out.columns and "model_id" not in out.columns:
        rename["model"] = "model_id"
    out = out.rename(columns=rename)

    for col in ["date", "symbol", "model_id", "raw_score"]:
        if col not in out.columns:
            raise ValueError(f"model_predictions missing required column: {col}")

    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)
    out["model_id"] = out["model_id"].astype(str)
    out["raw_score"] = pd.to_numeric(out["raw_score"], errors="coerce")

    if "raw_model_weight" in out.columns:
        out["raw_model_weight"] = pd.to_numeric(out["raw_model_weight"], errors="coerce")

    out = out.sort_values(["date", "symbol", "model_id"]).drop_duplicates(["date", "symbol", "model_id"], keep="last")
    return out.reset_index(drop=True)


def _regime_name(state_id: int) -> str:
    return f"state_{int(state_id)}"


def _gamma(conf: float, cfg: RegimeGatingConfig) -> float:
    den = max(cfg.confidence_full - cfg.confidence_floor, 1e-8)
    return float(np.clip((conf - cfg.confidence_floor) / den, 0.0, 1.0))


def resolve_operational_regime(
    regime_df: pd.DataFrame,
    p_cols: list[str],
    cfg: RegimeGatingConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    prev_regime = None
    prev_duration = 0
    candidate_streak = 0
    prev_belief = None

    lam_b = min(1.0, max(0.05, 1.0 / max(cfg.hysteresis_window, 1)))

    for _, r in regime_df.iterrows():
        probs = np.array([float(r[c]) for c in p_cols], dtype=float)
        probs = np.nan_to_num(probs, nan=0.0)
        probs = probs / max(probs.sum(), 1e-12)

        if prev_belief is None:
            belief = probs.copy()
        else:
            belief = (1.0 - lam_b) * prev_belief + lam_b * probs
            belief = belief / max(belief.sum(), 1e-12)

        dom = int(np.argmax(probs))
        conf = float(np.max(probs))
        g = _gamma(conf, cfg)
        model_ok = bool(r["model_accepted"])

        prev_prob = 0.0
        if prev_regime is not None and prev_regime.startswith("state_"):
            prev_idx = int(prev_regime.split("_")[-1])
            if 0 <= prev_idx < len(probs):
                prev_prob = float(probs[prev_idx])

        if (not model_ok) or conf < cfg.confidence_floor:
            if cfg.uncertain_policy == "keep_previous" and prev_regime is not None:
                candidate = prev_regime
                reason = "uncertain_keep_previous"
            else:
                candidate = "neutral"
                reason = "uncertain_neutral"
            g = min(g, 0.5)
        else:
            candidate = _regime_name(dom)
            reason = "dominant_state"

        if prev_regime is None:
            new_regime = candidate
            switch = True
            switch_reason = "init"
            candidate_streak = 1
            duration = 1
        elif candidate == prev_regime:
            new_regime = prev_regime
            switch = False
            switch_reason = "stay"
            candidate_streak += 1
            duration = prev_duration + 1
        else:
            margin = float(probs[dom] - prev_prob)
            allow = bool(
                conf >= cfg.confidence_min_transition
                and (
                    margin >= cfg.transition_margin
                    or candidate_streak >= cfg.hysteresis_window
                    or prev_duration >= cfg.min_regime_duration
                )
            )
            if cfg.emergency_transition_policy and candidate == "state_0" and conf >= max(cfg.confidence_full, 0.80):
                allow = True

            if allow:
                new_regime = candidate
                switch = True
                switch_reason = f"transition_allowed({reason})"
                candidate_streak = 1
                duration = 1
            else:
                new_regime = prev_regime
                switch = False
                switch_reason = "hysteresis_block"
                candidate_streak += 1
                duration = prev_duration + 1

        degraded_mode = bool((not model_ok) or (conf < cfg.confidence_floor))

        row = {
            "date": pd.Timestamp(r["date"]),
            "dominant_state_filtered": dom,
            "confidence": conf,
            "entropy": float(r["state_entropy"]),
            "gamma": g,
            "previous_regime": prev_regime if prev_regime is not None else "",
            "regime_operational": new_regime,
            "switch_flag": bool(switch),
            "switch_reason": switch_reason,
            "duration_current_regime": int(duration),
            "model_accepted": model_ok,
            "degraded_mode_flag": degraded_mode,
            "fallback_flag": degraded_mode,
            "belief_state": int(np.argmax(belief)),
        }
        for i, c in enumerate(p_cols):
            row[c] = float(probs[i])
            row[c.replace("_filtered", "_belief")] = float(belief[i])
        rows.append(row)

        prev_regime = new_regime
        prev_duration = duration
        prev_belief = belief

    return pd.DataFrame(rows)

def _model_policy_for(model_id: str, cfg: RegimeGatingConfig) -> dict[str, Any]:
    policy = cfg.model_gating_policy.get(model_id)
    if policy is None:
        policy = cfg.model_gating_policy.get("default", {})
    return dict(policy) if isinstance(policy, Mapping) else {}


def _model_multiplier(model_id: str, regime: str, gamma: float, cfg: RegimeGatingConfig) -> float:
    policy = _model_policy_for(model_id, cfg)
    allowed = policy.get("allowed_regimes")
    mult = policy.get("multipliers", {})

    target = 1.0
    if isinstance(allowed, (list, tuple, set)) and len(allowed):
        if regime not in set(str(x) for x in allowed):
            target = 0.0
    if isinstance(mult, Mapping):
        target = float(mult.get(regime, mult.get("default", target)))

    target = float(np.clip(target, 0.0, 1.5))
    return float((1.0 - gamma) * 1.0 + gamma * target)


def _base_weights(pred_df: pd.DataFrame, cfg: RegimeGatingConfig) -> dict[str, float]:
    models = sorted(pred_df["model_id"].unique().tolist())

    from_cfg = cfg.weight_gating_policy.get("base_weights", {}) if isinstance(cfg.weight_gating_policy, Mapping) else {}
    if isinstance(from_cfg, Mapping) and len(from_cfg):
        w = {m: float(from_cfg.get(m, 0.0)) for m in models}
    elif "raw_model_weight" in pred_df.columns and pred_df["raw_model_weight"].notna().any():
        w = (
            pred_df.groupby("model_id")["raw_model_weight"].median().reindex(models).fillna(0.0).astype(float).to_dict()
        )
    else:
        w = {m: 1.0 for m in models}

    s = float(sum(max(v, 0.0) for v in w.values()))
    if s <= 0:
        return {m: 1.0 / len(models) for m in models}
    return {m: max(v, 0.0) / s for m, v in w.items()}


def _regime_weight_multiplier(model_id: str, regime: str, cfg: RegimeGatingConfig) -> float:
    mult = cfg.weight_gating_policy.get("regime_multipliers", {}) if isinstance(cfg.weight_gating_policy, Mapping) else {}
    if isinstance(mult, Mapping):
        regime_map = mult.get(regime, {})
        if isinstance(regime_map, Mapping):
            return float(regime_map.get(model_id, regime_map.get("default", 1.0)))
    return 1.0


def apply_model_and_weight_gating(
    pred_df: pd.DataFrame,
    gating_state: pd.DataFrame,
    cfg: RegimeGatingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pred_df.copy()
    g = gating_state[["date", "regime_operational", "confidence", "gamma", "degraded_mode_flag"]].copy()
    pred = pred.merge(g, on="date", how="left")

    pred["regime_operational"] = pred["regime_operational"].fillna("neutral")
    pred["confidence"] = pd.to_numeric(pred["confidence"], errors="coerce").fillna(0.0)
    pred["gamma"] = pd.to_numeric(pred["gamma"], errors="coerce").fillna(0.0)
    pred["degraded_mode_flag"] = pred["degraded_mode_flag"].fillna(True).astype(bool)

    # Model-level attenuation.
    pred["model_attenuation"] = pred.apply(
        lambda r: _model_multiplier(str(r["model_id"]), str(r["regime_operational"]), float(r["gamma"]), cfg),
        axis=1,
    )
    pred["model_mask"] = pred["model_attenuation"] > 0
    pred["gated_score"] = pred["raw_score"] * pred["model_attenuation"]

    # Date-level ensemble weight gating.
    w0 = _base_weights(pred, cfg)

    decision_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    for d, grp in pred.groupby("date", sort=True):
        regime = str(grp["regime_operational"].iloc[0])
        gamma = float(grp["gamma"].iloc[0])

        model_multipliers = {
            m: _regime_weight_multiplier(m, regime, cfg) for m in w0.keys()
        }
        w_raw = {
            m: float(w0[m] * max(model_multipliers[m], 0.0))
            for m in w0.keys()
        }

        # Apply hard model mask at date level.
        masked_models = set(grp.loc[~grp["model_mask"], "model_id"].astype(str).tolist())
        for m in masked_models:
            w_raw[m] = 0.0

        s = float(sum(w_raw.values()))
        if s <= 0:
            w_eff = {m: 0.0 for m in w_raw}
        else:
            w_eff = {m: v / s for m, v in w_raw.items()}

        decision_rows.append(
            {
                "date": d,
                "regime_operational": regime,
                "gamma": gamma,
                "model_mask_by_model": json.dumps({m: bool(w_eff[m] > 0) for m in sorted(w_eff)}, sort_keys=True),
            }
        )

        for m, w in w_eff.items():
            weight_rows.append({"date": d, "model_id": m, "weight_eff": float(w)})

    weight_df = pd.DataFrame(weight_rows)
    decision_df = pd.DataFrame(decision_rows)

    pred = pred.merge(weight_df, on=["date", "model_id"], how="left")
    pred["weight_eff"] = pd.to_numeric(pred["weight_eff"], errors="coerce").fillna(0.0)

    # Ensemble score per date/symbol after gating.
    agg = (
        pred.groupby(["date", "symbol"], sort=True)
        .apply(
            lambda g: float(
                np.nansum(
                    pd.to_numeric(g["gated_score"], errors="coerce")
                    * pd.to_numeric(g["weight_eff"], errors="coerce")
                )
            )
        )
        .rename("ensemble_score_gated")
        .reset_index()
    )
    pred = pred.merge(agg, on=["date", "symbol"], how="left")

    return pred, decision_df


def _kappa(cap_name: str, regime: str, conf: float, duration: float, cfg: RegimeGatingConfig) -> float:
    reg = cfg.kappa_params_by_regime.get(regime)
    if reg is None:
        reg = cfg.kappa_params_by_regime.get("neutral", {})
    pars = reg.get(cap_name, {"a": 1.0, "b": 0.0, "c": 0.0, "min": 0.2, "max": 1.2})

    a = float(pars.get("a", 1.0))
    b = float(pars.get("b", 0.0))
    c = float(pars.get("c", 0.0))
    lo = float(pars.get("min", 0.2))
    hi = float(pars.get("max", 1.2))

    k = a + b * conf + c * duration
    return float(np.clip(k, lo, hi))


def build_risk_gating_caps(
    gating_state: pd.DataFrame,
    cfg: RegimeGatingConfig,
    risk_limits: pd.DataFrame | str | Path | None = None,
) -> pd.DataFrame:
    rl = _to_df(risk_limits)
    base = gating_state[["date", "regime_operational", "confidence", "gamma", "duration_current_regime"]].copy()

    if len(rl):
        x = rl.copy()
        x["date"] = pd.to_datetime(x["date"])
        rename = {
            "gross_cap": "base_gross_cap",
            "net_cap": "base_net_cap",
            "turnover_cap": "base_turnover_cap",
            "short_cap": "base_short_cap",
        }
        x = x.rename(columns=rename)
        for c, dflt in [
            ("base_gross_cap", 1.0),
            ("base_net_cap", 1.0),
            ("base_turnover_cap", 1.0),
            ("base_short_cap", 1.0),
        ]:
            if c not in x.columns:
                x[c] = dflt
        base = base.merge(x[["date", "base_gross_cap", "base_net_cap", "base_turnover_cap", "base_short_cap"]], on="date", how="left")
    else:
        base["base_gross_cap"] = 1.0
        base["base_net_cap"] = 1.0
        base["base_turnover_cap"] = 1.0
        base["base_short_cap"] = 1.0

    for c in ["base_gross_cap", "base_net_cap", "base_turnover_cap", "base_short_cap"]:
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(1.0)

    base["gross_cap_eff"] = base["base_gross_cap"] * base.apply(
        lambda r: _kappa("gross", str(r["regime_operational"]), float(r["confidence"]), float(r["duration_current_regime"]), cfg), axis=1
    )
    base["net_cap_eff"] = base["base_net_cap"] * base.apply(
        lambda r: _kappa("net", str(r["regime_operational"]), float(r["confidence"]), float(r["duration_current_regime"]), cfg), axis=1
    )
    base["turnover_cap_eff"] = base["base_turnover_cap"] * base.apply(
        lambda r: _kappa("turnover", str(r["regime_operational"]), float(r["confidence"]), float(r["duration_current_regime"]), cfg), axis=1
    )
    base["short_cap_eff"] = base["base_short_cap"] * base.apply(
        lambda r: _kappa("short", str(r["regime_operational"]), float(r["confidence"]), float(r["duration_current_regime"]), cfg), axis=1
    )

    return base[[
        "date",
        "regime_operational",
        "confidence",
        "gamma",
        "gross_cap_eff",
        "net_cap_eff",
        "turnover_cap_eff",
        "short_cap_eff",
    ]]

def build_trade_gating_flags(gating_state: pd.DataFrame, cfg: RegimeGatingConfig) -> pd.DataFrame:
    rows = []
    for _, r in gating_state.iterrows():
        regime = str(r["regime_operational"])
        pol = cfg.trade_gating_policy.get(regime, cfg.trade_gating_policy.get("default", {})) if isinstance(cfg.trade_gating_policy, Mapping) else {}
        allow_new_longs = bool(pol.get("allow_new_longs", True))
        allow_new_shorts = bool(pol.get("allow_new_shorts", regime not in {"neutral", "state_0"}))
        max_new_positions_scale = float(pol.get("max_new_positions_scale", 1.0))
        rows.append(
            {
                "date": pd.Timestamp(r["date"]),
                "regime_operational": regime,
                "allow_new_longs": allow_new_longs,
                "allow_new_shorts": allow_new_shorts,
                "max_new_positions_scale": max_new_positions_scale,
            }
        )
    return pd.DataFrame(rows)


def build_gating_summary(
    gating_state: pd.DataFrame,
    gated_scores: pd.DataFrame,
    risk_caps: pd.DataFrame,
    cfg: RegimeGatingConfig,
) -> dict[str, Any]:
    n_days = max(len(gating_state), 1)
    n_switches = int(pd.to_numeric(gating_state["switch_flag"], errors="coerce").fillna(0).sum())

    date_min = pd.to_datetime(gating_state["date"]).min()
    date_max = pd.to_datetime(gating_state["date"]).max()
    years = max((date_max - date_min).days / 365.25, 1e-6)

    by_regime = gating_state["regime_operational"].value_counts(normalize=True).to_dict()
    avg_gamma = float(pd.to_numeric(gating_state["gamma"], errors="coerce").mean())

    atten_by_model = (
        gated_scores.groupby("model_id")["model_attenuation"].mean().astype(float).to_dict()
        if len(gated_scores)
        else {}
    )

    # Optional realized-return diagnostics if available.
    delta_sharpe = np.nan
    delta_calmar = np.nan
    delta_mdd = np.nan
    apr = np.nan

    if "y_true" in gated_scores.columns:
        tmp = gated_scores.groupby(["date", "symbol"], as_index=False).agg(
            score_gated=("ensemble_score_gated", "last"),
            y_true=("y_true", "last"),
            regime=("regime_operational", "last"),
        )
        tmp = tmp.dropna(subset=["score_gated", "y_true"])
        if len(tmp):
            # Use score*return proxy as strategy PnL approximation for gating diagnostics.
            tmp["pnl_gated"] = tmp["score_gated"] * tmp["y_true"]
            pnl_daily = tmp.groupby("date")["pnl_gated"].mean()
            mu = float(pnl_daily.mean())
            sd = float(pnl_daily.std(ddof=0))
            sharpe = mu / max(sd, 1e-12)
            delta_sharpe = sharpe

            csum = pnl_daily.cumsum()
            dd = csum - csum.cummax()
            mdd = float(dd.min()) if len(dd) else np.nan
            delta_mdd = mdd
            delta_calmar = mu / max(abs(mdd), 1e-12) if np.isfinite(mdd) else np.nan

            good = tmp[~tmp["regime"].isin(["neutral", "state_0"])]["pnl_gated"]
            base = tmp["pnl_gated"]
            if len(good) and abs(base.mean()) > 1e-12:
                apr = float(good.mean() / (base.mean() + 1e-12))

    accepted = bool(
        (n_switches / years) <= 120.0
        and float((gating_state["confidence"] < cfg.confidence_floor).mean()) <= 0.50
    )

    return {
        "policy_version": cfg.policy_version,
        "pct_days_each_regime": {str(k): float(v) for k, v in by_regime.items()},
        "pct_days_low_confidence": float((gating_state["confidence"] < cfg.confidence_floor).mean()),
        "avg_state_duration": float(pd.to_numeric(gating_state["duration_current_regime"], errors="coerce").mean()),
        "n_switches": n_switches,
        "switches_per_year": float(n_switches / years),
        "avg_gamma": avg_gamma,
        "avg_model_attenuation_by_model": atten_by_model,
        "avg_gross_cap_eff": float(pd.to_numeric(risk_caps["gross_cap_eff"], errors="coerce").mean()),
        "avg_turnover_cap_eff": float(pd.to_numeric(risk_caps["turnover_cap_eff"], errors="coerce").mean()),
        "pct_trades_blocked": float((~gating_state.get("allow_new_longs", pd.Series([True]*n_days))).mean()) if "allow_new_longs" in gating_state.columns else 0.0,
        "APR": apr,
        "delta_sharpe_vs_base": delta_sharpe,
        "delta_calmar_vs_base": delta_calmar,
        "delta_mdd_vs_base": delta_mdd,
        "degraded_mode_days": int(pd.to_numeric(gating_state["degraded_mode_flag"], errors="coerce").fillna(0).sum()),
        "accepted_flag": accepted,
        "rejection_reason": "" if accepted else "switches_or_low_confidence_too_high",
    }


def persist_gating_outputs(
    gated_scores: pd.DataFrame,
    gating_decisions: pd.DataFrame,
    risk_gating_caps: pd.DataFrame,
    gating_state: pd.DataFrame,
    gating_summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    return {
        "gated_scores": _write_parquet_safe(gated_scores, out / "gated_scores.parquet"),
        "gating_decisions": _write_parquet_safe(gating_decisions, out / "gating_decisions.parquet"),
        "risk_gating_caps": _write_parquet_safe(risk_gating_caps, out / "risk_gating_caps.parquet"),
        "gating_state": _write_parquet_safe(gating_state, out / "gating_state.parquet"),
        "gating_summary": _write_json_safe(dict(gating_summary), out / "gating_summary.json"),
        "manifest": _write_json_safe(dict(manifest), out / "manifest.json"),
    }


def run_regime_gating(
    regime_states: pd.DataFrame | str | Path,
    model_predictions: pd.DataFrame | str | Path,
    *,
    config: RegimeGatingConfig | None = None,
    risk_limits: pd.DataFrame | str | Path | None = None,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    data_snapshot_id: str = "unknown",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    cfg = config or RegimeGatingConfig()
    rid = run_id or cfg.run_id or run_id_with_prefix("gating")

    reg_df, p_cols = load_regime_states(regime_states)
    pred_df = load_model_predictions(model_predictions)

    gating_state = resolve_operational_regime(reg_df, p_cols, cfg)
    trade_flags = build_trade_gating_flags(gating_state, cfg)
    gating_state = gating_state.merge(trade_flags, on=["date", "regime_operational"], how="left")

    gated_scores, decision_masks = apply_model_and_weight_gating(pred_df, gating_state, cfg)
    risk_caps = build_risk_gating_caps(gating_state, cfg, risk_limits=risk_limits)

    gating_decisions = gating_state[[
        "date",
        "regime_operational",
        "dominant_state_filtered",
        "confidence",
        "entropy",
        "degraded_mode_flag",
        "fallback_flag",
        "switch_flag",
        "switch_reason",
        "duration_current_regime",
        "allow_new_longs",
        "allow_new_shorts",
        "max_new_positions_scale",
    ]].copy()
    gating_decisions = gating_decisions.merge(decision_masks, on=["date", "regime_operational"], how="left")
    gating_decisions["run_id"] = rid

    gated_scores_out = gated_scores[[
        "date",
        "symbol",
        "model_id",
        "raw_score",
        "gated_score",
        "model_attenuation",
        "regime_operational",
        "confidence",
        "gamma",
        "weight_eff",
        "ensemble_score_gated",
    ]].copy()
    gated_scores_out["run_id"] = rid

    risk_caps_out = risk_caps.copy()
    risk_caps_out["run_id"] = rid

    gating_state_out = gating_state.copy()
    gating_state_out["run_id"] = rid

    summary = build_gating_summary(gating_state_out, gated_scores, risk_caps_out, cfg)
    summary = {"run_id": rid, **summary}

    manifest = {
        "run_id": rid,
        "timestamp": utc_now_iso(),
        "config_hash": config_hash(asdict(cfg), n=24),
        "namespace_assertion": "filtered_only",
        "policy_version": cfg.policy_version,
        "degraded_mode_policy": cfg.degraded_mode_policy,
        "hysteresis_policy": {
            "window": cfg.hysteresis_window,
            "transition_margin": cfg.transition_margin,
            "min_regime_duration": cfg.min_regime_duration,
        },
        "data_snapshot_id": data_snapshot_id,
        "runtime_sec": float(time.perf_counter() - t0),
    }

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_gating_outputs(
            gated_scores_out,
            gating_decisions,
            risk_caps_out,
            gating_state_out,
            summary,
            manifest,
            output_dir=output_dir,
        )

    return {
        "gated_scores": gated_scores_out,
        "gating_decisions": gating_decisions,
        "risk_gating_caps": risk_caps_out,
        "gating_state": gating_state_out,
        "gating_summary": summary,
        "manifest": manifest,
        "artifacts": artifacts,
    }


__all__ = [
    "RegimeGatingConfig",
    "load_regime_states",
    "load_model_predictions",
    "resolve_operational_regime",
    "apply_model_and_weight_gating",
    "build_risk_gating_caps",
    "build_trade_gating_flags",
    "persist_gating_outputs",
    "run_regime_gating",
]
