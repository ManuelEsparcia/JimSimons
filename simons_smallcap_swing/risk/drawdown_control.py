
"""
risk/drawdown_control.py - Reactive drawdown survival state machine.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DrawdownControlConfig:
    warning_down: float = -0.08
    warning_up: float = -0.06
    derisk_down: float = -0.12
    derisk_up: float = -0.10
    kill_down: float = -0.15
    kill_up: float = -0.12
    cooldown_days: int = 10
    derisk_multiplier: float = 0.50
    reentry_threshold: float = -0.05
    reentry_schedule: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    min_reentry_days_per_step: int = 5
    nav_jump_guard: float = 0.25
    nav_floor_policy: str = "terminal"  # terminal | clamp


def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_id_with_prefix(prefix: str) -> str:
    from datetime import datetime, timezone

    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def config_hash(cfg: Mapping[str, Any], n: int = 24) -> str:
    import hashlib
    import json

    blob = json.dumps(dict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:n]


def _to_nav_df(nav_series: pd.DataFrame | pd.Series | str | Path) -> pd.DataFrame:
    if isinstance(nav_series, (str, Path)):
        p = Path(nav_series)
        if p.suffix.lower() == ".parquet":
            nav_series = pd.read_parquet(p)
        else:
            nav_series = pd.read_csv(p)

    if isinstance(nav_series, pd.Series):
        df = nav_series.rename("nav").to_frame().reset_index().rename(columns={"index": "date"})
    else:
        df = nav_series.copy()

    if "date" not in df.columns:
        raise ValueError("nav series needs date column")
    if "nav" not in df.columns:
        for c in ["NAV", "equity", "value", "portfolio_value"]:
            if c in df.columns:
                df = df.rename(columns={c: "nav"})
                break
    if "nav" not in df.columns:
        raise ValueError("nav series needs nav column")

    df["date"] = pd.to_datetime(df["date"])
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    return df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)


def _state_multiplier(state: str, cfg: DrawdownControlConfig, reentry_step: int | None) -> float:
    if state in {"kill", "cooldown"}:
        return 0.0
    if state == "derisk":
        return float(cfg.derisk_multiplier)
    if state == "reentry":
        if reentry_step is None or reentry_step < 0:
            return float(cfg.reentry_schedule[0])
        i = min(reentry_step, len(cfg.reentry_schedule) - 1)
        return float(cfg.reentry_schedule[i])
    return 1.0


def _compute_episode_stats(series: pd.DataFrame) -> tuple[float, float]:
    dd = pd.to_numeric(series["dd"], errors="coerce").fillna(0.0)

    in_dd = dd < 0
    lengths: list[int] = []
    recoveries: list[int] = []

    cur = 0
    for i, flag in enumerate(in_dd.values):
        if flag:
            cur += 1
        else:
            if cur > 0:
                lengths.append(cur)
                recoveries.append(cur)
                cur = 0
    if cur > 0:
        lengths.append(cur)

    avg_rec = float(np.mean(recoveries)) if recoveries else float("nan")
    worst_len = float(max(lengths)) if lengths else 0.0
    return avg_rec, worst_len


def persist_drawdown_outputs(
    series: pd.DataFrame,
    actions: pd.DataFrame,
    summary: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    series_path = out / "series.parquet"
    actions_path = out / "actions.parquet"
    summary_path = out / "summary.json"

    try:
        series.to_parquet(series_path, index=False)
        sfile = str(series_path)
    except Exception:
        series.to_csv(series_path.with_suffix(".csv"), index=False)
        sfile = str(series_path.with_suffix(".csv"))

    try:
        actions.to_parquet(actions_path, index=False)
        afile = str(actions_path)
    except Exception:
        actions.to_csv(actions_path.with_suffix(".csv"), index=False)
        afile = str(actions_path.with_suffix(".csv"))

    import json

    summary_path.write_text(json.dumps(dict(summary), sort_keys=True, indent=2, default=str), encoding="utf-8")

    return {
        "series": sfile,
        "actions": afile,
        "summary": str(summary_path),
    }

def run_drawdown_control(
    nav_series: pd.DataFrame | pd.Series | str | Path,
    *,
    config: DrawdownControlConfig | None = None,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or DrawdownControlConfig()
    rid = run_id or run_id_with_prefix("ddctrl")

    df = _to_nav_df(nav_series)
    if len(df) == 0:
        raise ValueError("empty NAV series")

    records: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    state = "normal"
    cooldown_remaining = 0
    reentry_step: int | None = None
    reentry_days = 0

    hwm = -np.inf
    prev_nav = np.nan
    fallback_triggered = False
    status = "ok"
    reentry_failures = 0

    for _, row in df.iterrows():
        date = pd.Timestamp(row["date"])
        nav = float(row["nav"]) if pd.notna(row["nav"]) else np.nan

        anomaly = False
        if np.isfinite(prev_nav) and prev_nav > 0:
            jump = abs(nav / prev_nav - 1.0) if np.isfinite(nav) else np.inf
            anomaly = bool(jump > cfg.nav_jump_guard)

        if (not np.isfinite(nav)) or nav <= 0:
            fallback_triggered = True
            status = "inconsistent_nav"
            if cfg.nav_floor_policy == "terminal":
                state = "kill"
                nav = max(nav, 0.0) if np.isfinite(nav) else 0.0
                anomaly = True
            else:
                nav = max(float(nav) if np.isfinite(nav) else 0.0, 1e-8)

        hwm = max(hwm, nav)
        dd = float((nav - hwm) / hwm) if hwm > 0 else 0.0
        dd = float(np.clip(dd, -1.0, 0.0))

        prev_state = state
        prev_multiplier = _state_multiplier(state, cfg, reentry_step)

        reason = ""
        event_type = ""

        # Highest precedence: kill trigger
        if dd <= cfg.kill_down:
            if state != "kill":
                event_type = "enter_kill"
                reason = "dd<=kill_down"
            state = "kill"
            cooldown_remaining = int(cfg.cooldown_days)
            reentry_step = None
            reentry_days = 0

        elif state == "kill":
            # Transition from kill into cooldown when drawdown improves above kill_up.
            if dd > cfg.kill_up:
                state = "cooldown"
                event_type = "enter_cooldown"
                reason = "dd>kill_up"

        elif state == "cooldown":
            cooldown_remaining = max(cooldown_remaining - 1, 0)
            if cooldown_remaining == 0 and dd > cfg.reentry_threshold and dd > cfg.kill_up:
                state = "reentry"
                reentry_step = 0
                reentry_days = 0
                event_type = "enter_reentry"
                reason = "cooldown_complete"

        elif state == "reentry":
            if dd <= cfg.kill_down:
                state = "kill"
                cooldown_remaining = int(cfg.cooldown_days)
                reentry_step = None
                reentry_days = 0
                reentry_failures += 1
                event_type = "reentry_to_kill"
                reason = "dd<=kill_down"
            elif dd <= cfg.derisk_down:
                state = "derisk"
                reentry_step = None
                reentry_days = 0
                reentry_failures += 1
                event_type = "reentry_to_derisk"
                reason = "dd<=derisk_down"
            elif dd <= cfg.warning_down:
                state = "warning"
                reentry_step = None
                reentry_days = 0
                event_type = "reentry_to_warning"
                reason = "dd<=warning_down"
            else:
                reentry_days += 1
                if reentry_step is not None and reentry_days >= cfg.min_reentry_days_per_step:
                    if reentry_step < len(cfg.reentry_schedule) - 1:
                        reentry_step += 1
                        reentry_days = 0
                        event_type = "reentry_step_up"
                        reason = f"step={reentry_step}"
                    else:
                        state = "normal"
                        reentry_step = None
                        reentry_days = 0
                        event_type = "reentry_complete"
                        reason = "schedule_complete"

        else:
            # Normal/warning/derisk hysteresis logic.
            if dd <= cfg.derisk_down:
                if state != "derisk":
                    event_type = "enter_derisk"
                    reason = "dd<=derisk_down"
                state = "derisk"
            elif dd <= cfg.warning_down:
                if state != "warning":
                    event_type = "enter_warning"
                    reason = "dd<=warning_down"
                state = "warning"
            else:
                # exits by hysteresis thresholds
                if state == "derisk":
                    if dd > cfg.derisk_up:
                        if dd <= cfg.warning_down:
                            state = "warning"
                            event_type = "derisk_to_warning"
                            reason = "dd>derisk_up"
                        elif dd > cfg.warning_up:
                            state = "normal"
                            event_type = "derisk_to_normal"
                            reason = "dd>warning_up"
                elif state == "warning" and dd > cfg.warning_up:
                    state = "normal"
                    event_type = "warning_to_normal"
                    reason = "dd>warning_up"
                elif state == "normal":
                    state = "normal"

        multiplier = _state_multiplier(state, cfg, reentry_step)
        entry_allowed = bool(state not in {"kill", "cooldown", "derisk"})
        exit_forced = bool(state == "kill")

        if state != prev_state or abs(multiplier - prev_multiplier) > 1e-12:
            actions.append(
                {
                    "date": date,
                    "event_type": event_type or "state_change",
                    "from_state": prev_state,
                    "to_state": state,
                    "dd": dd,
                    "multiplier_before": prev_multiplier,
                    "multiplier_after": multiplier,
                    "reason": reason or "transition",
                }
            )

        records.append(
            {
                "date": date,
                "nav": nav,
                "hwm": hwm,
                "dd": dd,
                "state": state,
                "multiplier": multiplier,
                "entry_allowed": entry_allowed,
                "exit_forced": exit_forced,
                "cooldown_remaining": int(cooldown_remaining),
                "reentry_step": (int(reentry_step) + 1) if reentry_step is not None else np.nan,
                "anomaly_flag": bool(anomaly),
                "run_id": rid,
            }
        )

        prev_nav = nav

    series = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    actions_df = pd.DataFrame(actions).sort_values("date").reset_index(drop=True) if actions else pd.DataFrame(
        columns=["date", "event_type", "from_state", "to_state", "dd", "multiplier_before", "multiplier_after", "reason"]
    )

    max_dd = float(pd.to_numeric(series["dd"], errors="coerce").min()) if len(series) else float("nan")
    avg_rec, worst_episode = _compute_episode_stats(series)

    state_counts = series["state"].value_counts()
    num_warning_events = int((actions_df["to_state"] == "warning").sum()) if len(actions_df) else 0
    num_derisk_events = int((actions_df["to_state"] == "derisk").sum()) if len(actions_df) else 0
    num_kill_events = int((actions_df["to_state"] == "kill").sum()) if len(actions_df) else 0

    if status == "inconsistent_nav" and cfg.nav_floor_policy == "terminal":
        status = "terminal"

    summary = {
        "run_id": rid,
        "max_drawdown": max_dd,
        "days_in_warning": int(state_counts.get("warning", 0)),
        "days_in_derisk": int(state_counts.get("derisk", 0)),
        "days_in_kill": int(state_counts.get("kill", 0)),
        "days_in_cooldown": int(state_counts.get("cooldown", 0)),
        "days_in_reentry": int(state_counts.get("reentry", 0)),
        "num_warning_events": num_warning_events,
        "num_derisk_events": num_derisk_events,
        "num_kill_events": num_kill_events,
        "avg_recovery_time": avg_rec,
        "worst_episode_length": worst_episode,
        "percent_time_entries_blocked": float((~series["entry_allowed"]).mean()) if len(series) else float("nan"),
        "num_reentry_failures": int(reentry_failures),
        "parameters": cfg.__dict__,
        "fallback_triggered": bool(fallback_triggered),
        "config_hash": config_hash(cfg.__dict__, n=24),
        "status": status,
        "execution_timestamp": utc_now_iso(),
    }

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_drawdown_outputs(series, actions_df, summary, output_dir=output_dir)

    return {
        "series": series,
        "actions": actions_df,
        "summary": summary,
        "artifacts": artifacts,
    }


__all__ = [
    "DrawdownControlConfig",
    "persist_drawdown_outputs",
    "run_drawdown_control",
]
