"""
simons_core/logging.py — Structured observability.

    get_logger(name, run_context)  → StructuredLogger with bound context
    with_step(logger, step)        → context manager emitting start/complete/fail
    serialize_exception(exc)       → dict with type, message, traceback

Event schema:  ts_utc · level · logger · event · run_id · step · payload
Levels: DEBUG < INFO < WARNING < ERROR < CRITICAL
No side effects at import time. JSON-serializable output.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import time
import traceback
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)


class LogEvent(dict):
    """A structured log event (dict subclass for convenience)."""
    pass


def serialize_exception(exc: Exception) -> dict[str, Any]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "repr": repr(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


class StructuredLogger:
    """JSON structured logger with context propagation."""

    def __init__(self, name: str, base_fields: dict[str, Any] | None = None):
        self.name = name
        self.base_fields: dict[str, Any] = dict(base_fields or {})

    def bind(self, **fields) -> StructuredLogger:
        merged = {**self.base_fields, **fields}
        return StructuredLogger(self.name, merged)

    def _emit(self, level: str, event: str, **payload) -> LogEvent:
        record = LogEvent({
            "ts_utc": _utc_now_iso(), "level": level, "logger": self.name,
            "event": event, **self.base_fields,
            "payload": _jsonable(payload) if payload else {},
        })
        try:
            print(json.dumps(record, ensure_ascii=False, default=str))
        except Exception:
            pass
        return record

    def debug(self, event: str, **kw) -> LogEvent:   return self._emit("DEBUG", event, **kw)
    def info(self, event: str, **kw) -> LogEvent:    return self._emit("INFO", event, **kw)
    def warning(self, event: str, **kw) -> LogEvent: return self._emit("WARNING", event, **kw)
    def error(self, event: str, **kw) -> LogEvent:   return self._emit("ERROR", event, **kw)

    def exception(self, event: str, exc: Exception | None = None, **kw) -> LogEvent:
        if exc is not None:
            kw["exception"] = serialize_exception(exc)
        return self._emit("ERROR", event, **kw)


def get_logger(name: str, run_context: Any = None) -> StructuredLogger:
    """Build a logger, optionally binding RunContext fields."""
    base: dict[str, Any] = {}
    if run_context is not None:
        for f in ("run_id", "asof_date", "pipeline_version", "market"):
            if hasattr(run_context, f):
                val = getattr(run_context, f)
                base[f] = str(val) if hasattr(val, "isoformat") else val
    return StructuredLogger(name, base)


@contextmanager
def with_step(logger: StructuredLogger, step: str, **fields):
    """Instrument a pipeline step: start → complete/fail with duration."""
    sl = logger.bind(step=step, **fields)
    t0 = time.perf_counter()
    sl.info("step_started")
    try:
        yield sl
    except Exception as exc:
        dt = round(1000 * (time.perf_counter() - t0), 2)
        sl.exception("step_failed", exc=exc, duration_ms=dt)
        raise
    else:
        dt = round(1000 * (time.perf_counter() - t0), 2)
        sl.info("step_completed", duration_ms=dt)
