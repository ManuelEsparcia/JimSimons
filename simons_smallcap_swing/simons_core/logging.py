from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import logging
import time
from typing import Any

from .interfaces import RunContext


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return value


def serialize_exception(exc: Exception) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


@dataclass(frozen=True)
class LogEvent:
    event: str
    payload: dict[str, Any] = field(default_factory=dict)
    level: str = "INFO"


class StructuredLogger:
    """Tiny structured logger wrapper for MVP observability."""

    def __init__(self, name: str, base_fields: dict[str, Any] | None = None):
        self.name = name
        self.base_fields = dict(base_fields or {})
        self._logger = logging.getLogger(name)
        if not self._logger.handlers and not logging.getLogger().handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
            self._logger.propagate = False
        self._logger.setLevel(logging.INFO)

    def bind(self, **fields: Any) -> "StructuredLogger":
        merged = dict(self.base_fields)
        merged.update(fields)
        return StructuredLogger(self.name, merged)

    def _emit(self, level: int, event: str, **payload: Any) -> None:
        record = {
            "ts_utc": _utc_now_iso(),
            "level": logging.getLevelName(level),
            "logger": self.name,
            "event": event,
            **self.base_fields,
            **payload,
        }
        self._logger.log(level, json.dumps(record, default=_to_jsonable, sort_keys=True))

    def debug(self, event: str, **payload: Any) -> None:
        self._emit(logging.DEBUG, event, **payload)

    def info(self, event: str, **payload: Any) -> None:
        self._emit(logging.INFO, event, **payload)

    def warning(self, event: str, **payload: Any) -> None:
        self._emit(logging.WARNING, event, **payload)

    def error(self, event: str, **payload: Any) -> None:
        self._emit(logging.ERROR, event, **payload)

    def exception(
        self,
        event: str,
        exc: Exception | None = None,
        **payload: Any,
    ) -> None:
        if exc is not None:
            payload["exception"] = serialize_exception(exc)
        self._emit(logging.ERROR, event, **payload)


def get_logger(name: str, run_context: RunContext | None = None) -> StructuredLogger:
    base: dict[str, Any] = {}
    if run_context is not None:
        base["run_id"] = run_context.run_id
        base["asof_date"] = run_context.asof_date
        base["pipeline_version"] = run_context.pipeline_version
        base["market"] = run_context.market
    return StructuredLogger(name, base_fields=base)


def bind_context(logger: StructuredLogger, **fields: Any) -> StructuredLogger:
    return logger.bind(**fields)


@contextmanager
def with_step(logger: StructuredLogger, step: str, **fields: Any):
    step_logger = logger.bind(step=step, **fields)
    t0 = time.perf_counter()
    step_logger.info("step_started")
    try:
        yield step_logger
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        step_logger.exception("step_failed", exc=exc, duration_ms=elapsed_ms)
        raise
    else:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        step_logger.info("step_completed", duration_ms=elapsed_ms)


__all__ = [
    "LogEvent",
    "StructuredLogger",
    "bind_context",
    "get_logger",
    "serialize_exception",
    "with_step",
]
