
from __future__ import annotations

"""
simons_core.logging
===================

Structured, contract-first observability for the simons_smallcap_swing stack.

Design goals
------------
- Stable event envelope with explicit schema version.
- Cheap, predictable JSON emission.
- Context propagation without boilerplate.
- Safe payload serialization and redaction.
- Transport/backend decoupling through sinks.
- Ergonomic step instrumentation with duration and failure capture.

This module intentionally does *not* implement business metrics, alerting logic,
or backend-specific observability workflows. It provides structured facts about
execution that downstream tooling can consume.
"""

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal
import io
import json
import os
from pathlib import Path
import sys
import time as _time
import traceback
from types import TracebackType
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Protocol, Sequence
from uuid import UUID


__all__ = [
    "DEFAULT_SCHEMA_VERSION",
    "DEFAULT_REDACTED_KEYS",
    "LoggerError",
    "LogSink",
    "MemorySink",
    "MultiSink",
    "StructuredLogger",
    "StreamJsonSink",
    "StdoutJsonSink",
    "bind_context",
    "get_logger",
    "serialize_exception",
    "with_step",
]


DEFAULT_SCHEMA_VERSION = "1.0.0"
DEFAULT_REDACTED_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "bearer_token",
        "client_secret",
        "cookie",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "secret_key",
        "session_token",
        "token",
    }
)

_RESERVED_ENVELOPE_FIELDS = frozenset(
    {
        "ts_utc",
        "level",
        "logger",
        "event",
        "payload",
        "exception",
        "schema_version",
    }
)

_CONTEXTUAL_EVENT_FIELDS = frozenset(
    {
        "run_id",
        "step",
        "module",
        "asof_date",
        "pipeline_version",
        "market",
        "trace_id",
        "span_id",
        "error_code",
    }
)

_LEVEL_TO_NUM = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


class LoggerError(RuntimeError):
    """Raised when the logger cannot safely emit a structured record."""


class LogSink(Protocol):
    """Transport/backend interface for emitted log records."""

    def write(self, record: Mapping[str, Any]) -> None:
        """Persist or forward a structured record."""


class StreamJsonSink:
    """
    Write one JSON record per line to a text stream.

    Parameters
    ----------
    stream:
        Target text stream. Defaults to ``sys.stdout``.
    ensure_ascii:
        Passed through to ``json.dumps``.
    flush:
        Whether to flush the stream after every write.
    sort_keys:
        Whether to serialize keys in deterministic sorted order.
    """

    def __init__(
        self,
        stream: io.TextIOBase | None = None,
        *,
        ensure_ascii: bool = False,
        flush: bool = True,
        sort_keys: bool = True,
    ) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._ensure_ascii = ensure_ascii
        self._flush = flush
        self._sort_keys = sort_keys

    def write(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(
            record,
            ensure_ascii=self._ensure_ascii,
            separators=(",", ":"),
            sort_keys=self._sort_keys,
        )
        self._stream.write(line + os.linesep)
        if self._flush:
            self._stream.flush()


class StdoutJsonSink(StreamJsonSink):
    """Convenience sink that emits newline-delimited JSON to stdout."""

    def __init__(self) -> None:
        super().__init__(stream=sys.stdout, ensure_ascii=False, flush=True, sort_keys=True)


class MemorySink:
    """
    In-memory sink useful for tests and local inspection.

    Notes
    -----
    Records are stored exactly as emitted by the logger (already sanitized and
    JSON-safe by construction).
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write(self, record: Mapping[str, Any]) -> None:
        self.records.append(dict(record))

    def clear(self) -> None:
        self.records.clear()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)


class MultiSink:
    """Fan out a record to multiple sinks."""

    def __init__(self, sinks: Sequence[LogSink]) -> None:
        if not sinks:
            raise ValueError("MultiSink requires at least one sink.")
        self._sinks = tuple(sinks)

    @property
    def sinks(self) -> tuple[LogSink, ...]:
        return self._sinks

    def write(self, record: Mapping[str, Any]) -> None:
        errors: list[str] = []
        for sink in self._sinks:
            try:
                sink.write(record)
            except Exception as exc:  # pragma: no cover - defensive path
                errors.append(f"{type(sink).__name__}: {exc!r}")

        if errors:
            raise LoggerError("One or more sinks failed during write: " + "; ".join(errors))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _coerce_error_code(exc: BaseException) -> str:
    error_code = getattr(exc, "error_code", None)
    if isinstance(error_code, str) and error_code.strip():
        return error_code.strip().upper()
    return exc.__class__.__name__.upper()


def _normalize_level(level: str) -> str:
    normalized = level.upper()
    if normalized not in _LEVEL_TO_NUM:
        raise ValueError(
            f"Unsupported log level {level!r}. Expected one of {tuple(_LEVEL_TO_NUM)}."
        )
    return normalized


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _maybe_numpy_scalar(value: Any) -> Any:
    # Avoid importing numpy. Support the common scalar/zero-dim object protocol.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            extracted = item()
        except Exception:
            return value
        else:
            if _is_json_scalar(extracted):
                return extracted
    return value


def _truncate_text(text: str, *, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def _to_jsonable(
    value: Any,
    *,
    max_text_chars: int | None = 10_000,
    max_collection_items: int | None = 1_000,
    fallback: Callable[[Any], Any] | None = None,
) -> Any:
    """
    Convert a Python object into a JSON-safe structure.

    Conversion is intentionally conservative and stable:
    - scalars remain scalars
    - dataclasses become dictionaries
    - mappings become ``dict[str, Any]``
    - sequences become lists
    - common temporal/path/UUID/Decimal types become strings or numbers
    - unknown objects fall back to ``repr`` unless a custom fallback is given
    """
    value = _maybe_numpy_scalar(value)

    if _is_json_scalar(value):
        if isinstance(value, str):
            return _truncate_text(value, max_chars=max_text_chars)
        return value

    if is_dataclass(value):
        return _to_jsonable(
            asdict(value),
            max_text_chars=max_text_chars,
            max_collection_items=max_collection_items,
            fallback=fallback,
        )

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, (datetime, date, time)):
        if isinstance(value, datetime) and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()

    if isinstance(value, (UUID, Path)):
        return str(value)

    if isinstance(value, BaseException):
        return serialize_exception(
            value,
            include_traceback=True,
            max_text_chars=max_text_chars,
        )

    if isinstance(value, Mapping):
        items = list(value.items())
        if max_collection_items is not None:
            items = items[:max_collection_items]
        return {
            str(k): _to_jsonable(
                v,
                max_text_chars=max_text_chars,
                max_collection_items=max_collection_items,
                fallback=fallback,
            )
            for k, v in items
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        seq = list(value)
        if max_collection_items is not None:
            seq = seq[:max_collection_items]
        return [
            _to_jsonable(
                item,
                max_text_chars=max_text_chars,
                max_collection_items=max_collection_items,
                fallback=fallback,
            )
            for item in seq
        ]

    if fallback is not None:
        converted = fallback(value)
        if converted is value:
            return _truncate_text(repr(value), max_chars=max_text_chars)
        return _to_jsonable(
            converted,
            max_text_chars=max_text_chars,
            max_collection_items=max_collection_items,
            fallback=None,
        )

    return _truncate_text(repr(value), max_chars=max_text_chars)


def _sanitize_value(
    value: Any,
    *,
    redacted_keys: frozenset[str],
    max_text_chars: int | None,
    max_collection_items: int | None,
) -> Any:
    """
    Recursively sanitize a JSON-safe structure.

    Keys matching the redaction policy are replaced with ``"[REDACTED]"``.
    """
    jsonable = _to_jsonable(
        value,
        max_text_chars=max_text_chars,
        max_collection_items=max_collection_items,
    )

    if isinstance(jsonable, Mapping):
        sanitized: dict[str, Any] = {}
        for key, subvalue in jsonable.items():
            if str(key).lower() in redacted_keys:
                sanitized[str(key)] = "[REDACTED]"
            else:
                sanitized[str(key)] = _sanitize_value(
                    subvalue,
                    redacted_keys=redacted_keys,
                    max_text_chars=max_text_chars,
                    max_collection_items=max_collection_items,
                )
        return sanitized

    if isinstance(jsonable, list):
        return [
            _sanitize_value(
                item,
                redacted_keys=redacted_keys,
                max_text_chars=max_text_chars,
                max_collection_items=max_collection_items,
            )
            for item in jsonable
        ]

    return jsonable


def _validate_bound_fields(fields: Mapping[str, Any]) -> None:
    illegal = [key for key in fields if key in _RESERVED_ENVELOPE_FIELDS]
    if illegal:
        raise ValueError(
            f"Cannot bind reserved envelope fields {tuple(sorted(illegal))}. "
            "Use standard logger methods instead."
        )


def serialize_exception(
    exc: BaseException,
    *,
    include_traceback: bool = True,
    max_text_chars: int | None = 20_000,
    chain: bool = True,
) -> dict[str, Any]:
    """
    Serialize an exception into a stable JSON-safe payload.

    Parameters
    ----------
    exc:
        Exception instance to serialize.
    include_traceback:
        Whether to include traceback text.
    max_text_chars:
        Truncation limit applied to message, repr and traceback.
    chain:
        Whether to serialize ``__cause__`` / ``__context__`` recursively.
    """
    data: dict[str, Any] = {
        "type": exc.__class__.__name__,
        "message": _truncate_text(str(exc), max_chars=max_text_chars),
        "repr": _truncate_text(repr(exc), max_chars=max_text_chars),
    }

    if include_traceback:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        data["traceback"] = _truncate_text(tb, max_chars=max_text_chars)

    caused_by: BaseException | None = exc.__cause__ or (exc.__context__ if not exc.__suppress_context__ else None)
    if chain and caused_by is not None and caused_by is not exc:
        data["caused_by"] = serialize_exception(
            caused_by,
            include_traceback=include_traceback,
            max_text_chars=max_text_chars,
            chain=False,
        )
    else:
        data["caused_by"] = None

    return data


class StructuredLogger:
    """
    Structured logger with immutable bound context and interchangeable sinks.

    The logger emits records with a canonical envelope:

    - ts_utc
    - level
    - logger
    - event
    - schema_version
    - optional contextual fields (run_id, step, ...)
    - payload (JSON-safe mapping)
    - exception (optional)
    """

    def __init__(
        self,
        name: str,
        *,
        sink: LogSink | None = None,
        base_fields: Mapping[str, Any] | None = None,
        min_level: str = "INFO",
        schema_version: str = DEFAULT_SCHEMA_VERSION,
        redacted_keys: Iterable[str] | None = None,
        max_text_chars: int | None = 10_000,
        max_collection_items: int | None = 1_000,
        require_run_id: bool = False,
        on_emit_error: str = "raise",
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Logger name must be a non-empty string.")

        _validate_bound_fields(base_fields or {})
        self._name = name.strip()
        self._sink = sink if sink is not None else StdoutJsonSink()
        self._base_fields = dict(base_fields or {})
        self._min_level = _normalize_level(min_level)
        self._schema_version = schema_version
        self._redacted_keys = frozenset(str(k).lower() for k in (redacted_keys or DEFAULT_REDACTED_KEYS))
        self._max_text_chars = max_text_chars
        self._max_collection_items = max_collection_items
        self._require_run_id = bool(require_run_id)
        self._on_emit_error = on_emit_error.lower()
        if self._on_emit_error not in {"raise", "stderr"}:
            raise ValueError("on_emit_error must be either 'raise' or 'stderr'.")

    @property
    def name(self) -> str:
        return self._name

    @property
    def base_fields(self) -> dict[str, Any]:
        return dict(self._base_fields)

    @property
    def min_level(self) -> str:
        return self._min_level

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def sink(self) -> LogSink:
        return self._sink

    @property
    def require_run_id(self) -> bool:
        return self._require_run_id

    def bind(self, **fields: Any) -> "StructuredLogger":
        """
        Return a new logger with additional bound context.

        Bound fields are merged immutably; explicit new fields overwrite previous
        values except for reserved envelope fields, which are rejected.
        """
        _validate_bound_fields(fields)
        merged = {**self._base_fields, **fields}
        return StructuredLogger(
            self._name,
            sink=self._sink,
            base_fields=merged,
            min_level=self._min_level,
            schema_version=self._schema_version,
            redacted_keys=self._redacted_keys,
            max_text_chars=self._max_text_chars,
            max_collection_items=self._max_collection_items,
            require_run_id=self._require_run_id,
            on_emit_error=self._on_emit_error,
        )

    def child(self, suffix: str, **fields: Any) -> "StructuredLogger":
        """
        Create a child logger with dotted naming and optional context.

        Example
        -------
        ``logger.child("feature_build", module="features")``
        """
        suffix = str(suffix).strip()
        if not suffix:
            raise ValueError("suffix must be a non-empty string")
        return StructuredLogger(
            f"{self._name}.{suffix}",
            sink=self._sink,
            base_fields={**self._base_fields, **fields},
            min_level=self._min_level,
            schema_version=self._schema_version,
            redacted_keys=self._redacted_keys,
            max_text_chars=self._max_text_chars,
            max_collection_items=self._max_collection_items,
            require_run_id=self._require_run_id,
            on_emit_error=self._on_emit_error,
        )

    def is_enabled_for(self, level: str) -> bool:
        return _LEVEL_TO_NUM[_normalize_level(level)] >= _LEVEL_TO_NUM[self._min_level]

    def debug(self, event: str, **payload: Any) -> None:
        self._emit("DEBUG", event, payload=payload)

    def info(self, event: str, **payload: Any) -> None:
        self._emit("INFO", event, payload=payload)

    def warning(self, event: str, **payload: Any) -> None:
        self._emit("WARNING", event, payload=payload)

    def error(self, event: str, **payload: Any) -> None:
        self._emit("ERROR", event, payload=payload)

    def critical(self, event: str, **payload: Any) -> None:
        self._emit("CRITICAL", event, payload=payload)

    def exception(
        self,
        event: str,
        *,
        exc: BaseException | None = None,
        error_code: str | None = None,
        include_traceback: bool = True,
        **payload: Any,
    ) -> None:
        if exc is None:
            exc = sys.exc_info()[1]
        exception_payload = (
            serialize_exception(
                exc,
                include_traceback=include_traceback,
                max_text_chars=self._max_text_chars,
            )
            if exc is not None
            else None
        )
        resolved_error_code = error_code
        if resolved_error_code is None and exc is not None:
            resolved_error_code = _coerce_error_code(exc)

        self._emit(
            "ERROR",
            event,
            payload=payload,
            exception=exception_payload,
            error_code=resolved_error_code,
        )

    def log(self, level: str, event: str, **payload: Any) -> None:
        self._emit(level, event, payload=payload)

    def _emit(
        self,
        level: str,
        event: str,
        *,
        payload: Mapping[str, Any] | None = None,
        exception: Mapping[str, Any] | None = None,
        error_code: str | None = None,
    ) -> None:
        normalized_level = _normalize_level(level)
        if not self.is_enabled_for(normalized_level):
            return

        event_name = str(event).strip()
        if not event_name:
            raise ValueError("event must be a non-empty string")

        if self._require_run_id and "run_id" not in self._base_fields:
            raise LoggerError(
                f"Logger {self._name!r} requires run_id but no run_id is bound."
            )

        try:
            record = self._build_record(
                level=normalized_level,
                event=event_name,
                payload=payload or {},
                exception=exception,
                error_code=error_code,
            )
            self._sink.write(record)
        except Exception as exc:
            if self._on_emit_error == "stderr":
                sys.stderr.write(
                    f"[logging-fallback] failed to emit record for logger={self._name!r} "
                    f"event={event_name!r}: {exc!r}{os.linesep}"
                )
                sys.stderr.flush()
            else:
                raise LoggerError(
                    f"Failed to emit structured log record logger={self._name!r} "
                    f"event={event_name!r}"
                ) from exc

    def _build_record(
        self,
        *,
        level: str,
        event: str,
        payload: Mapping[str, Any],
        exception: Mapping[str, Any] | None,
        error_code: str | None,
    ) -> dict[str, Any]:
        sanitized_base = _sanitize_value(
            self._base_fields,
            redacted_keys=self._redacted_keys,
            max_text_chars=self._max_text_chars,
            max_collection_items=self._max_collection_items,
        )
        if not isinstance(sanitized_base, dict):
            raise LoggerError("Base fields could not be sanitized into a mapping.")

        sanitized_payload = _sanitize_value(
            payload,
            redacted_keys=self._redacted_keys,
            max_text_chars=self._max_text_chars,
            max_collection_items=self._max_collection_items,
        )
        if not isinstance(sanitized_payload, dict):
            raise LoggerError("Payload could not be sanitized into a mapping.")

        record: dict[str, Any] = {
            "ts_utc": _utc_now_iso(),
            "level": level,
            "logger": self._name,
            "event": event,
            "schema_version": self._schema_version,
            **sanitized_base,
            "payload": sanitized_payload,
        }

        if error_code is not None:
            record["error_code"] = str(error_code)

        if exception is not None:
            record["exception"] = _sanitize_value(
                exception,
                redacted_keys=self._redacted_keys,
                max_text_chars=self._max_text_chars,
                max_collection_items=self._max_collection_items,
            )

        return record


def bind_context(logger: StructuredLogger, **fields: Any) -> StructuredLogger:
    """Functional alias for ``StructuredLogger.bind``."""
    return logger.bind(**fields)


def get_logger(
    name: str,
    run_context: Any | None = None,
    *,
    sink: LogSink | None = None,
    min_level: str = "INFO",
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    base_fields: Mapping[str, Any] | None = None,
    require_run_id: bool = False,
    redacted_keys: Iterable[str] | None = None,
    on_emit_error: str = "raise",
) -> StructuredLogger:
    """
    Construct a logger and optionally seed it from a run context object.

    The ``run_context`` object is treated duck-typed: if present, the following
    attributes are copied when available:

    - run_id
    - asof_date
    - pipeline_version
    - market
    - module
    - trace_id
    - span_id
    """
    resolved_base: dict[str, Any] = dict(base_fields or {})
    if run_context is not None:
        for field in (
            "run_id",
            "asof_date",
            "pipeline_version",
            "market",
            "module",
            "trace_id",
            "span_id",
        ):
            if hasattr(run_context, field):
                value = getattr(run_context, field)
                if value is not None:
                    resolved_base[field] = value

    return StructuredLogger(
        name,
        sink=sink,
        base_fields=resolved_base,
        min_level=min_level,
        schema_version=schema_version,
        require_run_id=require_run_id,
        redacted_keys=redacted_keys,
        on_emit_error=on_emit_error,
    )


@contextmanager
def with_step(
    logger: StructuredLogger,
    step: str,
    *,
    error_code: str | None = None,
    **fields: Any,
) -> Iterator[StructuredLogger]:
    """
    Instrument a pipeline step with start / completion / failure events.

    Emitted events
    --------------
    - ``step_started``
    - ``step_completed`` with ``duration_ms``
    - ``step_failed`` with ``duration_ms`` and serialized exception

    Parameters
    ----------
    logger:
        Base logger.
    step:
        Step identifier to bind.
    error_code:
        Optional fallback error code used if the raised exception does not
        provide one.
    **fields:
        Additional contextual fields bound for the duration of the step.
    """
    step_name = str(step).strip()
    if not step_name:
        raise ValueError("step must be a non-empty string")

    step_logger = logger.bind(step=step_name, **fields)
    t0 = _time.perf_counter()
    step_logger.info("step_started")
    try:
        yield step_logger
    except Exception as exc:
        duration_ms = round(1000.0 * (_time.perf_counter() - t0), 3)
        step_logger.exception(
            "step_failed",
            exc=exc,
            error_code=error_code or _coerce_error_code(exc),
            duration_ms=duration_ms,
        )
        raise
    else:
        duration_ms = round(1000.0 * (_time.perf_counter() - t0), 3)
        step_logger.info("step_completed", duration_ms=duration_ms)
