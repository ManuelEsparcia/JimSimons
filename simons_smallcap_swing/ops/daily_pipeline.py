"""
ops/daily_pipeline.py - Deterministic and resumable daily DAG orchestration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PipelineStep:
    name: str
    dependencies: tuple[str, ...] = tuple()
    critical: bool = True
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class PipelineConfig:
    run_mode: str = "daily"
    fail_fast: bool = True
    max_retries: int = 3
    max_parallel_steps: int = 1
    retry_base_seconds: float = 5.0
    retry_backoff_factor: float = 2.0
    retry_max_seconds: float = 300.0
    timeout_seconds_default: int = 1800
    resume: bool = True
    sleep_on_retry: bool = False
    preflight_required_paths: tuple[str, ...] = tuple()


@dataclass(frozen=True)
class PipelineContext:
    run_id: str
    run_key: str
    execution_date: str
    config: PipelineConfig
    code_version: str
    config_version: str


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    run_key: str
    execution_date: str
    run_mode: str
    status: str
    created_at: str
    started_at: str | None
    ended_at: str | None
    config_version: str
    code_version: str
    fail_fast: bool


@dataclass(frozen=True)
class StepRecord:
    run_id: str
    step_name: str
    step_key: str
    status: str
    attempt: int
    started_at: str | None
    ended_at: str | None
    input_fingerprint: str
    output_fingerprint: str | None
    error_type: str | None
    error_message: str | None
    critical: bool
    reused: bool = False
    skipped_reason: str | None = None
    duration_seconds: float | None = None
    output_payload: Mapping[str, Any] | None = None


class TransientStepError(Exception):
    """Explicit marker for recoverable step errors."""


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


def _hash_text(text: str, n: int = 24) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def config_hash(cfg: Mapping[str, Any], n: int = 24) -> str:
    import json

    return _hash_text(json.dumps(json_safe(dict(cfg)), sort_keys=True, default=str), n=n)


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


def _load_config_file(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}

    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")

    if p.suffix.lower() in {".json", ".jsonl"}:
        import json

        return dict(json.loads(p.read_text(encoding="utf-8")))

    if p.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            return dict(data or {})
        except Exception:
            # Minimal fallback parser for flat key:value YAML.
            data: dict[str, Any] = {}
            for line in p.read_text(encoding="utf-8").splitlines():
                t = line.strip()
                if not t or t.startswith("#") or ":" not in t:
                    continue
                k, v = t.split(":", 1)
                data[k.strip()] = v.strip().strip("\"'")
            return data

    # CSV fallback
    df = pd.read_csv(p)
    if {"key", "value"}.issubset(df.columns):
        return {str(r["key"]): r["value"] for _, r in df.iterrows()}
    return {}


def _deep_merge(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(dict(out[k]), v)
        else:
            out[k] = v
    return out


def backoff_delay(attempt: int, base_s: float = 5.0, gamma: float = 2.0, max_s: float = 300.0) -> float:
    return float(min(max_s, base_s * (gamma ** attempt)))


def _default_steps() -> list[PipelineStep]:
    return [
        PipelineStep(name="data_health", dependencies=tuple(), critical=True),
        PipelineStep(name="drift_detection", dependencies=("data_health",), critical=False),
        PipelineStep(name="monitoring", dependencies=("data_health", "drift_detection"), critical=False),
    ]


def _normalize_steps(steps: Sequence[PipelineStep | Mapping[str, Any]] | None) -> list[PipelineStep]:
    if steps is None:
        return _default_steps()

    out: list[PipelineStep] = []
    for s in steps:
        if isinstance(s, PipelineStep):
            out.append(s)
            continue
        if not isinstance(s, Mapping):
            raise TypeError("steps must contain PipelineStep or mapping entries")
        deps = s.get("dependencies", tuple())
        if isinstance(deps, str):
            deps = tuple(x.strip() for x in deps.split(",") if x.strip())
        out.append(
            PipelineStep(
                name=str(s["name"]),
                dependencies=tuple(str(x) for x in deps),
                critical=bool(s.get("critical", True)),
                timeout_seconds=(None if s.get("timeout_seconds") is None else int(s.get("timeout_seconds"))),
            )
        )

    names = [s.name for s in out]
    if len(names) != len(set(names)):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate step names: {dup}")

    return out


def _validate_dependencies(step_map: Mapping[str, PipelineStep]) -> None:
    missing = []
    for s in step_map.values():
        for d in s.dependencies:
            if d not in step_map:
                missing.append((s.name, d))
    if missing:
        raise ValueError(f"invalid dependencies: {missing}")


def _dependency_closure(selected: Sequence[str], step_map: Mapping[str, PipelineStep]) -> set[str]:
    need = set(selected)
    changed = True
    while changed:
        changed = False
        cur = list(need)
        for s in cur:
            for dep in step_map[s].dependencies:
                if dep not in need:
                    need.add(dep)
                    changed = True
    return need


def _topological_sort(step_map: Mapping[str, PipelineStep]) -> list[str]:
    indeg = {k: 0 for k in step_map}
    children: dict[str, list[str]] = {k: [] for k in step_map}

    for s in step_map.values():
        for d in s.dependencies:
            indeg[s.name] += 1
            children[d].append(s.name)

    q = sorted([k for k, v in indeg.items() if v == 0])
    out = []
    while q:
        n = q.pop(0)
        out.append(n)
        for ch in sorted(children[n]):
            indeg[ch] -= 1
            if indeg[ch] == 0:
                q.append(ch)

    if len(out) != len(step_map):
        raise ValueError("DAG validation failed: cycle detected")
    return out


def _fingerprint_obj(obj: Any) -> str:
    import json

    if obj is None:
        return _hash_text("none", n=24)
    if isinstance(obj, pd.DataFrame):
        sample = obj.head(50).copy()
        blob = f"df|{obj.shape}|{','.join(map(str, obj.columns))}|{sample.to_csv(index=False)}"
        return _hash_text(blob, n=24)
    if isinstance(obj, pd.Series):
        blob = f"series|{obj.shape}|{obj.head(100).to_string()}"
        return _hash_text(blob, n=24)
    if isinstance(obj, Mapping):
        blob = json.dumps(json_safe(obj), sort_keys=True, default=str)
        return _hash_text(blob, n=24)
    if isinstance(obj, (list, tuple, set, frozenset)):
        blob = json.dumps(json_safe(list(obj)), sort_keys=True, default=str)
        return _hash_text(blob, n=24)
    return _hash_text(str(obj), n=24)


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, TransientStepError)):
        return True
    txt = str(exc).lower()
    transient_tokens = ["timeout", "tempor", "rate limit", "connection", "lock", "busy", "retry"]
    return any(t in txt for t in transient_tokens)


def _default_runner(_: PipelineContext, __: Mapping[str, Any]) -> dict[str, Any]:
    return {"status": "ok", "artifacts": [], "message": "noop"}


def _execute_runner(
    runner: Callable[..., Any],
    ctx: PipelineContext,
    dep_outputs: Mapping[str, Any],
    timeout_seconds: int,
) -> Any:
    from concurrent.futures import ThreadPoolExecutor

    def _invoke() -> Any:
        try:
            return runner(ctx, dep_outputs)
        except TypeError:
            try:
                return runner(ctx)
            except TypeError:
                return runner()

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_invoke)
        return fut.result(timeout=timeout_seconds)


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


def _load_json(path: Path) -> dict[str, Any]:
    import json

    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))

def persist_pipeline_outputs(
    run_record: RunRecord,
    step_records: pd.DataFrame,
    events: pd.DataFrame,
    summary: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "run_record": _save_json(asdict(run_record), out / "run_record.json"),
        "step_records": _save_df(step_records, out / "step_records.parquet"),
        "events": _save_df(events, out / "events.parquet"),
        "summary": _save_json(summary, out / "summary.json"),
    }


def run_daily_pipeline(
    *,
    config_path: str | Path | None,
    execution_date: str,
    run_id: str | None = None,
    run_mode: str | None = None,
    fail_fast: bool | None = None,
    max_retries: int | None = None,
    resume: bool | None = None,
    selected_steps: Sequence[str] | None = None,
    steps: Sequence[PipelineStep | Mapping[str, Any]] | None = None,
    step_runners: Mapping[str, Callable[..., Any]] | None = None,
    output_dir: str | Path | None = None,
    code_version: str | None = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg_raw = _load_config_file(config_path)

    # Accept nested config containers.
    if "ops" in cfg_raw and isinstance(cfg_raw["ops"], Mapping):
        cfg_raw = dict(cfg_raw["ops"])
    if "daily_pipeline" in cfg_raw and isinstance(cfg_raw["daily_pipeline"], Mapping):
        cfg_raw = dict(cfg_raw["daily_pipeline"])

    if config_overrides:
        cfg_raw = _deep_merge(cfg_raw, dict(config_overrides))

    cfg = PipelineConfig(
        run_mode=str(run_mode or cfg_raw.get("run_mode", "daily")),
        fail_fast=bool(cfg_raw.get("fail_fast", True) if fail_fast is None else fail_fast),
        max_retries=int(cfg_raw.get("max_retries", 3) if max_retries is None else max_retries),
        max_parallel_steps=int(cfg_raw.get("max_parallel_steps", 1)),
        retry_base_seconds=float(cfg_raw.get("retry_base_seconds", 5.0)),
        retry_backoff_factor=float(cfg_raw.get("retry_backoff_factor", 2.0)),
        retry_max_seconds=float(cfg_raw.get("retry_max_seconds", 300.0)),
        timeout_seconds_default=int(cfg_raw.get("timeout_seconds_default", 1800)),
        resume=bool(cfg_raw.get("resume", True) if resume is None else resume),
        sleep_on_retry=bool(cfg_raw.get("sleep_on_retry", False)),
        preflight_required_paths=tuple(str(x) for x in cfg_raw.get("preflight_required_paths", [])),
    )

    rid = run_id or run_id_with_prefix("pipe")
    code_ver = str(code_version or cfg_raw.get("code_version", "unknown"))
    config_ver = config_hash(cfg.__dict__, n=24)
    run_key = _hash_text(f"{execution_date}|{cfg.run_mode}|{config_ver}|{code_ver}", n=24)
    ctx = PipelineContext(
        run_id=rid,
        run_key=run_key,
        execution_date=str(pd.Timestamp(execution_date).date()),
        config=cfg,
        code_version=code_ver,
        config_version=config_ver,
    )

    steps_norm = _normalize_steps(steps)
    step_map = {s.name: s for s in steps_norm}
    _validate_dependencies(step_map)

    selected = list(selected_steps) if selected_steps else list(step_map.keys())
    unknown_selected = [s for s in selected if s not in step_map]
    if unknown_selected:
        raise ValueError(f"unknown selected_steps: {unknown_selected}")

    plan_set = _dependency_closure(selected, step_map)
    plan_map = {k: v for k, v in step_map.items() if k in plan_set}
    topo = _topological_sort(plan_map)

    out_root = Path(output_dir) if output_dir is not None else (Path.cwd() / "artifacts" / "ops" / "daily_pipeline")
    run_dir = out_root / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    lock_dir = out_root / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = lock_dir / f"{run_key}.lock"

    events: list[dict[str, Any]] = []

    run_record = RunRecord(
        run_id=rid,
        run_key=run_key,
        execution_date=ctx.execution_date,
        run_mode=cfg.run_mode,
        status="created",
        created_at=utc_now_iso(),
        started_at=None,
        ended_at=None,
        config_version=config_ver,
        code_version=code_ver,
        fail_fast=cfg.fail_fast,
    )

    step_records: dict[str, StepRecord] = {}
    dep_outputs: dict[str, Any] = {}

    step_records_path = run_dir / "step_records.parquet"
    events_path = run_dir / "events.parquet"
    run_record_path = run_dir / "run_record.json"

    # Resume if requested and compatible.
    if cfg.resume and run_record_path.exists():
        old_rr = _load_json(run_record_path)
        if old_rr and str(old_rr.get("run_key")) == run_key:
            old_steps_df = _to_df(step_records_path)
            for r in old_steps_df.to_dict(orient="records"):
                sr = StepRecord(
                    run_id=str(r.get("run_id", rid)),
                    step_name=str(r["step_name"]),
                    step_key=str(r.get("step_key", "")),
                    status=str(r.get("status", "pending")),
                    attempt=int(r.get("attempt", 0)),
                    started_at=(None if pd.isna(r.get("started_at")) else str(r.get("started_at"))),
                    ended_at=(None if pd.isna(r.get("ended_at")) else str(r.get("ended_at"))),
                    input_fingerprint=str(r.get("input_fingerprint", "")),
                    output_fingerprint=(None if pd.isna(r.get("output_fingerprint")) else str(r.get("output_fingerprint"))),
                    error_type=(None if pd.isna(r.get("error_type")) else str(r.get("error_type"))),
                    error_message=(None if pd.isna(r.get("error_message")) else str(r.get("error_message"))),
                    critical=bool(r.get("critical", True)),
                    reused=bool(r.get("reused", False)),
                    skipped_reason=(None if pd.isna(r.get("skipped_reason")) else str(r.get("skipped_reason"))),
                    duration_seconds=(None if pd.isna(r.get("duration_seconds")) else float(r.get("duration_seconds"))),
                    output_payload=(r.get("output_payload") if isinstance(r.get("output_payload"), Mapping) else None),
                )
                step_records[sr.step_name] = sr
                if sr.output_payload is not None and sr.status == "success":
                    dep_outputs[sr.step_name] = sr.output_payload

            old_events_df = _to_df(events_path)
            if len(old_events_df):
                events.extend(old_events_df.to_dict(orient="records"))

            events.append(
                {
                    "event_type": "resume_reconciled",
                    "run_id": rid,
                    "step_name": None,
                    "severity": "info",
                    "timestamp": utc_now_iso(),
                    "payload": {"reused_records": int(len(step_records))},
                }
            )
        elif old_rr:
            events.append(
                {
                    "event_type": "resume_incompatible",
                    "run_id": rid,
                    "step_name": None,
                    "severity": "warning",
                    "timestamp": utc_now_iso(),
                    "payload": {"reason": "run_key_changed", "old_run_key": old_rr.get("run_key"), "new_run_key": run_key},
                }
            )

    # Acquire lock to prevent concurrent writers on same run_key.
    lock_acquired = False
    try:
        import os

        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        lock_acquired = True
    except FileExistsError:
        run_record = RunRecord(**{**asdict(run_record), "status": "aborted", "ended_at": utc_now_iso()})
        events.append(
            {
                "event_type": "preflight_failed",
                "run_id": rid,
                "step_name": None,
                "severity": "high",
                "timestamp": utc_now_iso(),
                "payload": {"reason": "run_key_locked", "run_key": run_key},
            }
        )
        step_df = pd.DataFrame([asdict(v) for v in step_records.values()])
        events_df = pd.DataFrame(events)
        summary = {
            "run_id": rid,
            "status": run_record.status,
            "plan": topo,
            "num_steps": len(topo),
            "num_success": int(sum(1 for s in step_records.values() if s.status == "success")),
            "num_failed": int(sum(1 for s in step_records.values() if s.status == "failed")),
            "num_blocked": int(sum(1 for s in step_records.values() if s.status == "blocked")),
            "num_reused": int(sum(1 for s in step_records.values() if s.reused)),
        }
        artifacts = persist_pipeline_outputs(run_record, step_df, events_df, summary, output_dir=run_dir)
        return {
            "run_record": run_record,
            "step_records": step_df,
            "events": events_df,
            "summary": summary,
            "artifacts": artifacts,
        }

    # Preflight checks.
    preflight_ok = True
    preflight_errors: list[str] = []
    if len(topo) == 0:
        preflight_ok = False
        preflight_errors.append("empty_execution_plan")
    if not cfg.run_mode:
        preflight_ok = False
        preflight_errors.append("run_mode_missing")

    for p in cfg.preflight_required_paths:
        if not Path(p).exists():
            preflight_ok = False
            preflight_errors.append(f"required_path_missing:{p}")

    if not preflight_ok:
        run_record = RunRecord(**{**asdict(run_record), "status": "aborted", "ended_at": utc_now_iso()})
        events.append(
            {
                "event_type": "preflight_failed",
                "run_id": rid,
                "step_name": None,
                "severity": "high",
                "timestamp": utc_now_iso(),
                "payload": {"errors": preflight_errors},
            }
        )
        step_df = pd.DataFrame([asdict(v) for v in step_records.values()])
        events_df = pd.DataFrame(events)
        summary = {
            "run_id": rid,
            "status": run_record.status,
            "plan": topo,
            "preflight_errors": preflight_errors,
        }
        artifacts = persist_pipeline_outputs(run_record, step_df, events_df, summary, output_dir=run_dir)
        return {
            "run_record": run_record,
            "step_records": step_df,
            "events": events_df,
            "summary": summary,
            "artifacts": artifacts,
        }

    run_record = RunRecord(**{**asdict(run_record), "status": "running", "started_at": utc_now_iso()})
    events.append(
        {
            "event_type": "run_started",
            "run_id": rid,
            "step_name": None,
            "severity": "info",
            "timestamp": utc_now_iso(),
            "payload": {"plan": topo, "run_key": run_key},
        }
    )

    runners = dict(step_runners or {})

    def persist_state() -> None:
        step_df_local = pd.DataFrame([asdict(v) for v in step_records.values()])
        events_df_local = pd.DataFrame(events)
        _save_json(asdict(run_record), run_record_path)
        _save_df(step_df_local, step_records_path)
        _save_df(events_df_local, events_path)

    persist_state()

    terminated_early = False

    for sname in topo:
        step = plan_map[sname]
        dep_states = [step_records.get(d).status if d in step_records else None for d in step.dependencies]

        if any(st in {"failed", "blocked"} for st in dep_states if st is not None):
            rec = StepRecord(
                run_id=rid,
                step_name=sname,
                step_key="",
                status="blocked",
                attempt=0,
                started_at=None,
                ended_at=utc_now_iso(),
                input_fingerprint="",
                output_fingerprint=None,
                error_type=None,
                error_message=None,
                critical=step.critical,
                reused=False,
                skipped_reason="upstream_failed",
            )
            step_records[sname] = rec
            events.append(
                {
                    "event_type": "step_blocked",
                    "run_id": rid,
                    "step_name": sname,
                    "severity": "warning",
                    "timestamp": utc_now_iso(),
                    "payload": {"dependencies": list(step.dependencies), "dep_states": dep_states},
                }
            )
            persist_state()
            continue

        upstream_fp = "|".join(
            [f"{d}:{step_records[d].output_fingerprint or 'none'}" for d in sorted(step.dependencies) if d in step_records]
        )
        if not upstream_fp:
            upstream_fp = "root"

        step_key = _hash_text(f"{run_key}|{sname}|{upstream_fp}|{config_ver}", n=24)
        input_fp = _hash_text(f"{step_key}|input", n=24)

        old = step_records.get(sname)
        if old is not None and old.status == "success" and old.step_key == step_key:
            step_records[sname] = StepRecord(
                **{
                    **asdict(old),
                    "reused": True,
                }
            )
            if old.output_payload is not None:
                dep_outputs[sname] = old.output_payload
            events.append(
                {
                    "event_type": "step_succeeded",
                    "run_id": rid,
                    "step_name": sname,
                    "severity": "info",
                    "timestamp": utc_now_iso(),
                    "payload": {"reused": True, "step_key": step_key},
                }
            )
            persist_state()
            continue

        timeout_s = int(step.timeout_seconds or cfg.timeout_seconds_default)
        runner = runners.get(sname, _default_runner)
        attempt_start = int(old.attempt + 1) if old is not None and old.status in {"failed", "running", "pending"} else 0

        success = False
        last_exc: Exception | None = None
        output_payload: Mapping[str, Any] | None = None
        output_fp: str | None = None

        for attempt in range(attempt_start, cfg.max_retries + 1):
            step_start = utc_now_iso()
            step_records[sname] = StepRecord(
                run_id=rid,
                step_name=sname,
                step_key=step_key,
                status="running",
                attempt=attempt,
                started_at=step_start,
                ended_at=None,
                input_fingerprint=input_fp,
                output_fingerprint=None,
                error_type=None,
                error_message=None,
                critical=step.critical,
                reused=False,
            )
            events.append(
                {
                    "event_type": "step_started",
                    "run_id": rid,
                    "step_name": sname,
                    "severity": "info",
                    "timestamp": step_start,
                    "payload": {"attempt": attempt, "timeout_seconds": timeout_s},
                }
            )
            persist_state()

            try:
                out = _execute_runner(runner, ctx, dep_outputs, timeout_s)
                if isinstance(out, Mapping):
                    output_payload = json_safe(out)
                else:
                    output_payload = {"value": json_safe(out)}
                output_fp = _fingerprint_obj(output_payload)
                success = True

                step_end = utc_now_iso()
                dur = float((pd.Timestamp(step_end) - pd.Timestamp(step_start)).total_seconds())
                step_records[sname] = StepRecord(
                    run_id=rid,
                    step_name=sname,
                    step_key=step_key,
                    status="success",
                    attempt=attempt,
                    started_at=step_start,
                    ended_at=step_end,
                    input_fingerprint=input_fp,
                    output_fingerprint=output_fp,
                    error_type=None,
                    error_message=None,
                    critical=step.critical,
                    reused=False,
                    duration_seconds=dur,
                    output_payload=output_payload,
                )
                dep_outputs[sname] = output_payload
                events.append(
                    {
                        "event_type": "step_succeeded",
                        "run_id": rid,
                        "step_name": sname,
                        "severity": "info",
                        "timestamp": step_end,
                        "payload": {"attempt": attempt, "output_fingerprint": output_fp},
                    }
                )
                persist_state()
                break

            except Exception as exc:
                last_exc = exc
                step_end = utc_now_iso()
                dur = float((pd.Timestamp(step_end) - pd.Timestamp(step_start)).total_seconds())
                transient = _is_transient_error(exc)
                err_type = type(exc).__name__
                err_msg = str(exc)

                step_records[sname] = StepRecord(
                    run_id=rid,
                    step_name=sname,
                    step_key=step_key,
                    status="failed",
                    attempt=attempt,
                    started_at=step_start,
                    ended_at=step_end,
                    input_fingerprint=input_fp,
                    output_fingerprint=None,
                    error_type=err_type,
                    error_message=err_msg,
                    critical=step.critical,
                    reused=False,
                    duration_seconds=dur,
                )

                events.append(
                    {
                        "event_type": "step_failed",
                        "run_id": rid,
                        "step_name": sname,
                        "severity": ("high" if step.critical else "warning"),
                        "timestamp": step_end,
                        "payload": {
                            "attempt": attempt,
                            "error_type": err_type,
                            "error_message": err_msg,
                            "transient": transient,
                        },
                    }
                )

                if transient and attempt < cfg.max_retries:
                    delay = backoff_delay(
                        attempt,
                        cfg.retry_base_seconds,
                        cfg.retry_backoff_factor,
                        cfg.retry_max_seconds,
                    )
                    events.append(
                        {
                            "event_type": "retry_scheduled",
                            "run_id": rid,
                            "step_name": sname,
                            "severity": "warning",
                            "timestamp": utc_now_iso(),
                            "payload": {
                                "attempt": attempt,
                                "next_attempt": attempt + 1,
                                "delay_seconds": delay,
                            },
                        }
                    )
                    persist_state()
                    if cfg.sleep_on_retry and delay > 0:
                        import time

                        time.sleep(min(delay, 2.0))
                    continue

                persist_state()
                break

        if not success:
            if cfg.fail_fast and step.critical:
                terminated_early = True
                events.append(
                    {
                        "event_type": "run_terminated_fail_fast",
                        "run_id": rid,
                        "step_name": sname,
                        "severity": "high",
                        "timestamp": utc_now_iso(),
                        "payload": {
                            "critical_step": sname,
                            "error_type": type(last_exc).__name__ if last_exc is not None else None,
                        },
                    }
                )
                break

    # Block any untouched steps after early termination.
    if terminated_early:
        for sname in topo:
            if sname not in step_records:
                s = plan_map[sname]
                step_records[sname] = StepRecord(
                    run_id=rid,
                    step_name=sname,
                    step_key="",
                    status="blocked",
                    attempt=0,
                    started_at=None,
                    ended_at=utc_now_iso(),
                    input_fingerprint="",
                    output_fingerprint=None,
                    error_type=None,
                    error_message=None,
                    critical=s.critical,
                    skipped_reason="fail_fast_terminated",
                )

    # Reconcile final status.
    statuses = {k: v.status for k, v in step_records.items() if k in topo}
    any_failed_critical = any(
        step_map[k].critical and statuses.get(k) == "failed" for k in topo if k in step_map
    )
    any_failed = any(statuses.get(k) == "failed" for k in topo)
    any_blocked = any(statuses.get(k) == "blocked" for k in topo)

    final_status = "success"
    if any_failed_critical or (cfg.fail_fast and any_failed):
        final_status = "failed"
    elif any_failed or any_blocked:
        final_status = "partial"
    elif set(plan_map.keys()) != set(step_map.keys()):
        final_status = "partial"

    run_record = RunRecord(**{**asdict(run_record), "status": final_status, "ended_at": utc_now_iso()})

    events.append(
        {
            "event_type": "run_finished",
            "run_id": rid,
            "step_name": None,
            "severity": ("high" if final_status == "failed" else ("warning" if final_status == "partial" else "info")),
            "timestamp": utc_now_iso(),
            "payload": {
                "status": final_status,
                "num_steps": len(topo),
                "num_success": int(sum(1 for s in step_records.values() if s.status == "success")),
                "num_failed": int(sum(1 for s in step_records.values() if s.status == "failed")),
                "num_blocked": int(sum(1 for s in step_records.values() if s.status == "blocked")),
                "num_reused": int(sum(1 for s in step_records.values() if s.reused)),
            },
        }
    )

    step_df = pd.DataFrame([asdict(v) for v in step_records.values()]).sort_values(["step_name", "attempt"]).reset_index(drop=True)
    events_df = pd.DataFrame(events)

    summary = {
        "run_id": rid,
        "run_key": run_key,
        "execution_date": ctx.execution_date,
        "status": final_status,
        "selected_steps": sorted(selected),
        "planned_steps": topo,
        "num_steps": int(len(topo)),
        "num_success": int(sum(1 for s in step_records.values() if s.status == "success")),
        "num_failed": int(sum(1 for s in step_records.values() if s.status == "failed")),
        "num_blocked": int(sum(1 for s in step_records.values() if s.status == "blocked")),
        "num_reused": int(sum(1 for s in step_records.values() if s.reused)),
        "config_hash": config_hash(cfg.__dict__, n=24),
    }

    artifacts = persist_pipeline_outputs(run_record, step_df, events_df, summary, output_dir=run_dir)

    # Release lock.
    if lock_acquired and lock_file.exists():
        try:
            lock_file.unlink()
        except Exception:
            pass

    return {
        "run_record": run_record,
        "step_records": step_df,
        "events": events_df,
        "summary": summary,
        "artifacts": artifacts,
    }


def main(config_path: str, execution_date: str, run_id: str | None = None) -> None:
    run_daily_pipeline(config_path=config_path, execution_date=execution_date, run_id=run_id)


__all__ = [
    "PipelineConfig",
    "PipelineContext",
    "PipelineStep",
    "RunRecord",
    "StepRecord",
    "TransientStepError",
    "backoff_delay",
    "main",
    "persist_pipeline_outputs",
    "run_daily_pipeline",
]
