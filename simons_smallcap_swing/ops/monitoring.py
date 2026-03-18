"""
ops/monitoring.py - Unified operational monitoring and alert governance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonitoringConfig:
    suppression_window_seconds: int = 1800
    escalation_recurrence_threshold: int = 5
    escalation_open_seconds: int = 3600
    rate_limit_window_seconds: int = 300
    channels: tuple[str, ...] = ("storage", "console")
    channel_min_severity: Mapping[str, str] = field(
        default_factory=lambda: {
            "storage": "info",
            "console": "info",
            "slack": "warning",
            "email": "high",
            "pager": "critical",
        }
    )
    rate_limit_budget: Mapping[str, int] = field(
        default_factory=lambda: {
            "info": 80,
            "warning": 50,
            "high": 30,
            "critical": 500,
        }
    )


@dataclass(frozen=True)
class CanonicalEvent:
    event_id: str
    event_type: str
    source: str
    source_run_id: str | None
    source_step: str | None
    entity_type: str | None
    entity_id: str | None
    severity_hint: str | None
    timestamp: str
    status: str | None
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlertRecord:
    alert_id: str
    dedupe_key: str
    alert_type: str
    source: str
    source_run_id: str | None
    severity: str
    status: str
    opened_at: str
    updated_at: str
    closed_at: str | None
    recurrence_count: int
    action_hint: str | None
    payload: Mapping[str, Any] = field(default_factory=dict)
    channels_tried: tuple[str, ...] = tuple()


@dataclass(frozen=True)
class MonitoringSnapshot:
    run_id: str
    execution_date: str
    total_events: int
    total_alerts_open: int
    total_alerts_dispatched: int
    total_alerts_suppressed: int
    by_severity: Mapping[str, int]
    by_type: Mapping[str, int]
    sla_breaches: tuple[Mapping[str, Any], ...]
    created_at: str


SEVERITY_ORDER = {"info": 0, "warning": 1, "high": 2, "critical": 3}


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


def _to_df(obj: pd.DataFrame | str | Path | Sequence[Mapping[str, Any]] | None) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, (str, Path)):
        p = Path(obj)
        if not p.exists():
            return pd.DataFrame()
        if p.suffix.lower() == ".parquet":
            return pd.read_parquet(p)
        if p.suffix.lower() in {".json", ".jsonl"}:
            return pd.read_json(p)
        return pd.read_csv(p)
    return pd.DataFrame([dict(x) for x in obj])


def _hash_text(text: str, n: int = 24) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def _infer_event_id(event_type: str, source: str, timestamp: str, payload: Mapping[str, Any]) -> str:
    import json

    blob = f"{event_type}|{source}|{timestamp}|{json.dumps(json_safe(payload), sort_keys=True, default=str)}"
    return _hash_text(blob, n=24)


def _normalize_events(events: pd.DataFrame | Sequence[Mapping[str, Any]] | str | Path | None) -> list[CanonicalEvent]:
    df = _to_df(events)
    if len(df) == 0:
        return []

    out: list[CanonicalEvent] = []
    for row in df.to_dict(orient="records"):
        et = str(row.get("event_type", "unknown_event"))
        src = str(row.get("source", "unknown"))
        ts = row.get("timestamp", utc_now_iso())
        ts = str(pd.Timestamp(ts)) if ts is not None else utc_now_iso()

        payload = row.get("payload", {})
        if isinstance(payload, str):
            import json

            try:
                payload = json.loads(payload)
            except Exception:
                payload = {"raw_payload": payload}

        ev_id = str(row.get("event_id", "")).strip() or _infer_event_id(et, src, ts, payload)

        out.append(
            CanonicalEvent(
                event_id=ev_id,
                event_type=et,
                source=src,
                source_run_id=(None if row.get("source_run_id") in {None, "", np.nan} else str(row.get("source_run_id"))),
                source_step=(None if row.get("source_step") in {None, "", np.nan} else str(row.get("source_step"))),
                entity_type=(None if row.get("entity_type") in {None, "", np.nan} else str(row.get("entity_type"))),
                entity_id=(None if row.get("entity_id") in {None, "", np.nan} else str(row.get("entity_id"))),
                severity_hint=(None if row.get("severity_hint") in {None, "", np.nan} else str(row.get("severity_hint")).lower()),
                timestamp=ts,
                status=(None if row.get("status") in {None, "", np.nan} else str(row.get("status"))),
                payload=(payload if isinstance(payload, Mapping) else {}),
            )
        )

    return sorted(out, key=lambda e: (e.timestamp, e.event_id))


def _classify_alert_type(event: CanonicalEvent) -> str:
    et = event.event_type.lower()
    src = event.source.lower()

    if any(k in et for k in ["run_", "step_", "preflight", "retry", "pipeline"]) or src == "daily_pipeline":
        return "pipeline"
    if any(k in et for k in ["data_health", "schema", "freshness", "completeness", "dataset"]) or src == "data_health":
        return "data"
    if any(k in et for k in ["drift", "prediction", "model", "calibration"]) or src == "drift_detection":
        return "model"
    if any(k in et for k in ["risk", "drawdown", "exposure", "capacity", "breach"]):
        return "risk"
    if any(k in et for k in ["validation", "invariant", "mismatch"]):
        return "validation"
    if any(k in et for k in ["dispatch", "infra", "storage", "credential", "timeout"]):
        return "infra"
    return "ops"


def _derive_severity(event: CanonicalEvent, alert_type: str) -> str:
    if event.severity_hint in SEVERITY_ORDER:
        return str(event.severity_hint)

    et = event.event_type.lower()
    if any(k in et for k in ["critical", "sev-1", "failed", "fail", "aborted", "breach"]):
        if alert_type in {"risk", "pipeline", "infra"}:
            return "critical"
        return "high"
    if any(k in et for k in ["warn", "warning", "retry", "drift"]):
        return "warning"
    return "info"


def _derive_action_hint(alert_type: str, severity: str) -> str:
    if severity == "critical":
        if alert_type == "pipeline":
            return "pause_pipeline_and_investigate"
        if alert_type == "data":
            return "block_downstream_and_validate_data"
        if alert_type == "model":
            return "freeze_model_changes_and_review"
        if alert_type == "risk":
            return "reduce_risk_and_investigate"
        return "escalate_immediately"
    if severity == "high":
        return "investigate_soon"
    if severity == "warning":
        return "monitor_and_triage"
    return "log_only"


def _root_cause_signature(event: CanonicalEvent) -> str:
    keys = ["error_type", "error_message", "reason", "check_name", "metric_name", "exception"]
    parts = [event.event_type, event.source, str(event.entity_id)]
    for k in keys:
        if k in event.payload:
            parts.append(f"{k}={event.payload[k]}")
    if len(parts) <= 3:
        parts.append(str(event.payload))
    return _hash_text("|".join(parts), n=24)


def _build_dedupe_key(alert_type: str, source: str, entity_id: str | None, root_hash: str) -> str:
    return _hash_text(f"{alert_type}|{source}|{entity_id or '-'}|{root_hash}", n=24)


def _severity_max(a: str, b: str) -> str:
    return a if SEVERITY_ORDER.get(a, 0) >= SEVERITY_ORDER.get(b, 0) else b


def _severity_meets(severity: str, minimum: str) -> bool:
    return SEVERITY_ORDER.get(severity, 0) >= SEVERITY_ORDER.get(minimum, 0)


class RateLimiter:
    def __init__(self, budgets: Mapping[str, int], window_seconds: int):
        self.budgets = {str(k): int(v) for k, v in budgets.items()}
        self.window_seconds = max(int(window_seconds), 1)
        self.counts: dict[tuple[int, str, str], int] = {}

    def allow(self, severity: str, channel: str, timestamp: pd.Timestamp) -> bool:
        budget = int(self.budgets.get(severity, 50))
        bucket = int(timestamp.timestamp() // self.window_seconds)
        key = (bucket, severity, channel)
        used = self.counts.get(key, 0)
        if used >= budget:
            return False
        self.counts[key] = used + 1
        return True


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

def persist_monitoring_outputs(
    canonical_events: pd.DataFrame,
    alerts: pd.DataFrame,
    dispatch_results: pd.DataFrame,
    snapshot: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "canonical_events": _save_df(canonical_events, out / "canonical_events.parquet"),
        "alerts": _save_df(alerts, out / "alerts.parquet"),
        "dispatch_results": _save_df(dispatch_results, out / "dispatch_results.parquet"),
        "snapshot": _save_json(snapshot, out / "snapshot.json"),
    }


def run_monitoring(
    events: pd.DataFrame | Sequence[Mapping[str, Any]] | str | Path,
    *,
    existing_alerts: pd.DataFrame | Sequence[Mapping[str, Any]] | str | Path | None = None,
    config: MonitoringConfig | None = None,
    run_id: str | None = None,
    execution_date: str | pd.Timestamp | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or MonitoringConfig()
    rid = run_id or run_id_with_prefix("mon")
    exec_date = str((pd.Timestamp(execution_date) if execution_date is not None else pd.Timestamp.utcnow()).date())

    canonical = _normalize_events(events)
    now_iso = utc_now_iso()

    existing_df = _to_df(existing_alerts)
    existing_by_id: dict[str, dict[str, Any]] = {}
    if len(existing_df):
        for r in existing_df.to_dict(orient="records"):
            if "alert_id" in r:
                existing_by_id[str(r["alert_id"])] = dict(r)

    active_alerts = [a for a in existing_by_id.values() if str(a.get("status", "")).lower() in {"open", "acknowledged", "suppressed"}]

    alerts_out: list[dict[str, Any]] = []
    dispatch_rows: list[dict[str, Any]] = []
    suppressed_count = 0
    dispatched_count = 0

    limiter = RateLimiter(cfg.rate_limit_budget, cfg.rate_limit_window_seconds)

    for ev in canonical:
        alert_type = _classify_alert_type(ev)
        sev = _derive_severity(ev, alert_type)
        action = _derive_action_hint(alert_type, sev)
        root_hash = _root_cause_signature(ev)
        dkey = _build_dedupe_key(alert_type, ev.source, ev.entity_id, root_hash)

        ev_ts = pd.Timestamp(ev.timestamp)

        # Find dedupe match among active alerts by dedupe_key.
        match = None
        for a in active_alerts:
            if str(a.get("dedupe_key")) == dkey:
                match = a
                break

        if match is not None:
            last_seen = pd.Timestamp(match.get("updated_at", match.get("opened_at", now_iso)))
            delta_s = float((ev_ts - last_seen).total_seconds())
            suppress = (delta_s <= cfg.suppression_window_seconds) and (
                SEVERITY_ORDER.get(sev, 0) <= SEVERITY_ORDER.get(str(match.get("severity", "info")), 0)
            )

            if suppress:
                suppressed_count += 1
                match["recurrence_count"] = int(match.get("recurrence_count", 1)) + 1
                match["updated_at"] = str(ev_ts)
                match["status"] = "suppressed"
                match_payload = dict(match.get("payload", {}))
                match_payload["last_suppressed_event_id"] = ev.event_id
                match_payload["last_suppressed_at"] = str(ev_ts)
                match["payload"] = match_payload
                alerts_out.append(match)
                continue

            # Re-open or update existing alert.
            match["recurrence_count"] = int(match.get("recurrence_count", 1)) + 1
            match["updated_at"] = str(ev_ts)
            match["status"] = "open"
            match["severity"] = _severity_max(str(match.get("severity", "info")), sev)
            match["action_hint"] = action
            payload = dict(match.get("payload", {}))
            payload["latest_event_id"] = ev.event_id
            payload["latest_event_type"] = ev.event_type
            payload["latest_source"] = ev.source
            payload["latest_payload"] = json_safe(ev.payload)
            match["payload"] = payload
            alert = match
        else:
            alert_id = _hash_text(f"{dkey}|{ev.event_id}|{ev.timestamp}", n=24)
            alert = {
                "alert_id": alert_id,
                "dedupe_key": dkey,
                "alert_type": alert_type,
                "source": ev.source,
                "source_run_id": ev.source_run_id,
                "severity": sev,
                "status": "open",
                "opened_at": str(ev_ts),
                "updated_at": str(ev_ts),
                "closed_at": None,
                "recurrence_count": 1,
                "action_hint": action,
                "payload": {
                    "event_id": ev.event_id,
                    "event_type": ev.event_type,
                    "entity_type": ev.entity_type,
                    "entity_id": ev.entity_id,
                    "status": ev.status,
                    "payload": json_safe(ev.payload),
                },
                "channels_tried": [],
            }
            active_alerts.append(alert)

        # Escalation by recurrence/open duration.
        open_duration_s = float((ev_ts - pd.Timestamp(alert.get("opened_at", ev_ts))).total_seconds())
        if int(alert.get("recurrence_count", 1)) >= cfg.escalation_recurrence_threshold or open_duration_s >= cfg.escalation_open_seconds:
            if SEVERITY_ORDER.get(str(alert.get("severity", "info")), 0) < SEVERITY_ORDER["critical"]:
                # One-step escalation.
                inv = {v: k for k, v in SEVERITY_ORDER.items()}
                cur = SEVERITY_ORDER.get(str(alert.get("severity", "info")), 0)
                alert["severity"] = inv.get(min(cur + 1, 3), "critical")

        # Dispatch loop, decoupled from alert persistence.
        channels_tried = []
        for ch in cfg.channels:
            min_sev = str(cfg.channel_min_severity.get(ch, "info"))
            if not _severity_meets(str(alert["severity"]), min_sev):
                dispatch_rows.append(
                    {
                        "alert_id": alert["alert_id"],
                        "channel": ch,
                        "status": "skipped_min_severity",
                        "severity": alert["severity"],
                        "timestamp": now_iso,
                        "reason": f"requires>={min_sev}",
                    }
                )
                continue

            if not limiter.allow(str(alert["severity"]), ch, ev_ts):
                dispatch_rows.append(
                    {
                        "alert_id": alert["alert_id"],
                        "channel": ch,
                        "status": "skipped_rate_limited",
                        "severity": alert["severity"],
                        "timestamp": now_iso,
                        "reason": "rate_limit",
                    }
                )
                continue

            # Local dispatch simulation.
            channels_tried.append(ch)
            dispatched_count += 1
            dispatch_rows.append(
                {
                    "alert_id": alert["alert_id"],
                    "channel": ch,
                    "status": "sent",
                    "severity": alert["severity"],
                    "timestamp": now_iso,
                    "reason": "ok",
                }
            )

        alert["channels_tried"] = sorted(set(list(alert.get("channels_tried", [])) + channels_tried))
        alerts_out.append(alert)

    # Merge touched alerts with untouched existing alerts for full state snapshot.
    touched_ids = {str(a["alert_id"]) for a in alerts_out}
    merged_alerts = [a for a in existing_by_id.values() if str(a.get("alert_id")) not in touched_ids] + alerts_out

    canonical_df = pd.DataFrame([asdict(e) for e in canonical])
    alerts_df = pd.DataFrame(merged_alerts)
    dispatch_df = pd.DataFrame(dispatch_rows)

    if len(alerts_df) == 0:
        alerts_df = pd.DataFrame(
            columns=[
                "alert_id",
                "dedupe_key",
                "alert_type",
                "source",
                "source_run_id",
                "severity",
                "status",
                "opened_at",
                "updated_at",
                "closed_at",
                "recurrence_count",
                "action_hint",
                "payload",
                "channels_tried",
            ]
        )

    if len(dispatch_df) == 0:
        dispatch_df = pd.DataFrame(columns=["alert_id", "channel", "status", "severity", "timestamp", "reason"])

    by_sev = alerts_df["severity"].value_counts().to_dict() if len(alerts_df) else {}
    by_type = alerts_df["alert_type"].value_counts().to_dict() if len(alerts_df) else {}
    open_count = int(alerts_df[alerts_df["status"].isin(["open", "acknowledged", "suppressed"])] .shape[0]) if len(alerts_df) else 0

    sla_breaches: list[dict[str, Any]] = []
    if len(canonical_df):
        for _, row in canonical_df.iterrows():
            payload = row.get("payload", {})
            if isinstance(payload, Mapping) and ("sla_breach" in payload or "sla_gap_seconds" in payload):
                sla_breaches.append(
                    {
                        "event_id": row.get("event_id"),
                        "event_type": row.get("event_type"),
                        "source": row.get("source"),
                        "payload": json_safe(payload),
                    }
                )

    snapshot = MonitoringSnapshot(
        run_id=rid,
        execution_date=exec_date,
        total_events=int(len(canonical_df)),
        total_alerts_open=open_count,
        total_alerts_dispatched=int(dispatched_count),
        total_alerts_suppressed=int(suppressed_count),
        by_severity={str(k): int(v) for k, v in by_sev.items()},
        by_type={str(k): int(v) for k, v in by_type.items()},
        sla_breaches=tuple(sla_breaches),
        created_at=utc_now_iso(),
    )

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = persist_monitoring_outputs(
            canonical_df,
            alerts_df,
            dispatch_df,
            {
                "run_id": snapshot.run_id,
                "execution_date": snapshot.execution_date,
                "total_events": snapshot.total_events,
                "total_alerts_open": snapshot.total_alerts_open,
                "total_alerts_dispatched": snapshot.total_alerts_dispatched,
                "total_alerts_suppressed": snapshot.total_alerts_suppressed,
                "by_severity": snapshot.by_severity,
                "by_type": snapshot.by_type,
                "sla_breaches": json_safe(snapshot.sla_breaches),
                "created_at": snapshot.created_at,
                "config_hash": config_hash(cfg.__dict__, n=24),
            },
            output_dir=output_dir,
        )

    return {
        "canonical_events": canonical_df,
        "alerts": alerts_df,
        "dispatch_results": dispatch_df,
        "snapshot": snapshot,
        "artifacts": artifacts,
    }


def events_from_data_health(result: Mapping[str, Any]) -> pd.DataFrame:
    events = result.get("events")
    return _to_df(events)


def events_from_drift(result: Mapping[str, Any]) -> pd.DataFrame:
    events = result.get("events")
    return _to_df(events)


def events_from_pipeline(result: Mapping[str, Any]) -> pd.DataFrame:
    events = result.get("events")
    return _to_df(events)


__all__ = [
    "AlertRecord",
    "CanonicalEvent",
    "MonitoringConfig",
    "MonitoringSnapshot",
    "events_from_data_health",
    "events_from_drift",
    "events_from_pipeline",
    "persist_monitoring_outputs",
    "run_monitoring",
]
