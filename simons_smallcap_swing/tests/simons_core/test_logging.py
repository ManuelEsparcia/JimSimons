from __future__ import annotations

import io
import json
from pathlib import Path
import re

import pandas as pd
import pytest


pytestmark = [pytest.mark.simons_core, pytest.mark.logging]


class FailingSink:
    def write(self, record):
        raise RuntimeError("sink blew up")


class EchoObject:
    def __repr__(self) -> str:
        return "EchoObject(repr)"


class IdentityFallbackLoggerObject:
    pass


def _first_record(memory_sink):
    assert len(memory_sink.records) >= 1
    return memory_sink.records[0]


def _json_lines(stream: io.StringIO) -> list[dict]:
    text = stream.getvalue().strip()
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines()]


def test_default_constants_present(logging_mod):
    assert logging_mod.DEFAULT_SCHEMA_VERSION == "1.0.0"
    assert isinstance(logging_mod.DEFAULT_REDACTED_KEYS, frozenset)
    assert "token" in logging_mod.DEFAULT_REDACTED_KEYS
    assert "password" in logging_mod.DEFAULT_REDACTED_KEYS


def test_memory_sink_roundtrip_and_clear(logging_mod):
    sink = logging_mod.MemorySink()
    sink.write({"a": 1})
    sink.write({"b": 2})
    assert len(sink) == 2
    assert list(iter(sink)) == [{"a": 1}, {"b": 2}]
    sink.clear()
    assert len(sink) == 0
    assert sink.records == []


def test_stream_json_sink_writes_valid_json_line(logging_mod, json_stream):
    sink = logging_mod.StreamJsonSink(stream=json_stream, ensure_ascii=False, flush=True, sort_keys=True)
    sink.write({"b": 2, "a": "á"})
    lines = json_stream.getvalue().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"a": "á", "b": 2}
    assert lines[0].startswith('{"a"')


def test_multi_sink_requires_at_least_one_sink(logging_mod):
    with pytest.raises(ValueError, match="at least one sink"):
        logging_mod.MultiSink([])


def test_multi_sink_fans_out_to_all_sinks(logging_mod):
    left = logging_mod.MemorySink()
    right = logging_mod.MemorySink()
    sink = logging_mod.MultiSink([left, right])
    record = {"event": "hello", "payload": {"x": 1}}
    sink.write(record)
    assert left.records == [record]
    assert right.records == [record]


def test_multi_sink_wraps_sink_errors(logging_mod):
    good = logging_mod.MemorySink()
    sink = logging_mod.MultiSink([good, FailingSink()])
    with pytest.raises(logging_mod.LoggerError, match="One or more sinks failed"):
        sink.write({"event": "hello"})
    assert len(good.records) == 1


def test_get_logger_basic_emit_builds_canonical_envelope(base_logger, memory_sink, logging_mod):
    base_logger.info("unit_event", score=0.7, symbols=["AAA", "BBB"])
    record = _first_record(memory_sink)
    assert set(["ts_utc", "level", "logger", "event", "schema_version", "module", "payload"]).issubset(record)
    assert record["level"] == "INFO"
    assert record["logger"] == "simons_core.tests"
    assert record["event"] == "unit_event"
    assert record["schema_version"] == logging_mod.DEFAULT_SCHEMA_VERSION
    assert record["module"] == "tests"
    assert record["payload"] == {"score": 0.7, "symbols": ["AAA", "BBB"]}
    assert record["ts_utc"].endswith("Z")


def test_logger_level_filtering(base_logger, memory_sink):
    base_logger.info("visible")
    logger = base_logger.bind(extra="x")
    logger = logger.__class__(
        logger.name,
        sink=logger.sink,
        base_fields=logger.base_fields,
        min_level="ERROR",
        schema_version=logger.schema_version,
        require_run_id=False,
        on_emit_error="raise",
    )
    logger.info("hidden")
    logger.error("shown")
    events = [r["event"] for r in memory_sink.records]
    assert events == ["visible", "shown"]


def test_is_enabled_for_respects_threshold(base_logger):
    assert base_logger.is_enabled_for("DEBUG") is True
    warn_logger = base_logger.__class__(
        base_logger.name,
        sink=base_logger.sink,
        base_fields=base_logger.base_fields,
        min_level="WARNING",
        schema_version=base_logger.schema_version,
    )
    assert warn_logger.is_enabled_for("INFO") is False
    assert warn_logger.is_enabled_for("ERROR") is True


def test_bind_context_adds_top_level_fields(base_logger, memory_sink, logging_mod):
    logger = logging_mod.bind_context(base_logger, run_id="run_123", asof_date="2024-01-05")
    logger.info("evt", value=1)
    record = _first_record(memory_sink)
    assert record["run_id"] == "run_123"
    assert record["asof_date"] == "2024-01-05"
    assert record["payload"] == {"value": 1}


def test_bind_rejects_reserved_envelope_fields(base_logger):
    with pytest.raises(ValueError, match="reserved envelope fields"):
        base_logger.bind(level="INFO")


def test_child_logger_uses_dotted_name_and_inherits_context(base_logger, memory_sink):
    child = base_logger.bind(run_id="run_1").child("features", trace_id="t-1")
    child.info("built", rows=10)
    record = _first_record(memory_sink)
    assert record["logger"] == "simons_core.tests.features"
    assert record["run_id"] == "run_1"
    assert record["trace_id"] == "t-1"
    assert record["payload"] == {"rows": 10}


def test_child_rejects_empty_suffix(base_logger):
    with pytest.raises(ValueError, match="non-empty string"):
        base_logger.child("   ")


def test_get_logger_from_run_context_binds_fields(contextual_logger, memory_sink):
    contextual_logger.info("ctx_evt", ok=True)
    record = _first_record(memory_sink)
    assert record["run_id"] == "run_unit_test_001"
    assert str(record["asof_date"]).startswith("2024-01-05")
    assert record["pipeline_version"] == "1.0.0"
    assert record["market"] == "US_EQ"


def test_require_run_id_enforced(logging_mod, memory_sink):
    logger = logging_mod.get_logger(
        "simons_core.tests",
        sink=memory_sink,
        min_level="DEBUG",
        require_run_id=True,
        on_emit_error="raise",
    )
    with pytest.raises(logging_mod.LoggerError, match="requires run_id"):
        logger.info("evt")


def test_empty_event_name_rejected(base_logger):
    with pytest.raises(ValueError, match="non-empty string"):
        base_logger.info("   ")


def test_invalid_level_and_emit_policy_rejected(logging_mod):
    with pytest.raises(ValueError, match="Unsupported log level"):
        logging_mod.StructuredLogger("x", min_level="NOPE")
    with pytest.raises(ValueError, match="on_emit_error"):
        logging_mod.StructuredLogger("x", on_emit_error="ignore")


def test_logger_name_must_be_non_empty(logging_mod):
    with pytest.raises(ValueError, match="non-empty string"):
        logging_mod.StructuredLogger("   ")


def test_exception_serialization_includes_traceback_and_chain(logging_mod):
    try:
        try:
            raise ValueError("root cause")
        except ValueError as exc:
            raise RuntimeError("outer failure") from exc
    except RuntimeError as exc:
        payload = logging_mod.serialize_exception(exc, include_traceback=True)
    assert payload["type"] == "RuntimeError"
    assert "outer failure" in payload["message"]
    assert "traceback" in payload
    assert payload["caused_by"] is not None
    assert payload["caused_by"]["type"] == "ValueError"


def test_exception_logging_uses_error_code_from_exception(base_logger, memory_sink, dummy_exception):
    base_logger.exception("boom_evt", exc=dummy_exception, stage="fit")
    record = _first_record(memory_sink)
    assert record["level"] == "ERROR"
    assert record["event"] == "boom_evt"
    assert record["error_code"] == "UNIT_TEST_BOOM"
    assert record["payload"] == {"stage": "fit"}
    assert record["exception"]["type"] == "RuntimeError"
    assert "unit-test-boom" in record["exception"]["message"]


def test_exception_logging_accepts_explicit_error_code_override(base_logger, memory_sink, dummy_exception):
    base_logger.exception("boom_evt", exc=dummy_exception, error_code="OVERRIDE", stage="fit")
    record = _first_record(memory_sink)
    assert record["error_code"] == "OVERRIDE"


def test_sensitive_payload_is_redacted(base_logger, memory_sink, sensitive_payload):
    base_logger.info("secrets_evt", **sensitive_payload)
    payload = _first_record(memory_sink)["payload"]
    assert payload["token"] == "[REDACTED]"
    assert payload["api_key"] == "[REDACTED]"
    assert payload["nested"]["password"] == "[REDACTED]"
    assert payload["nested"]["safe"] == 7
    assert payload["notes"] == "visible"


def test_serialization_of_common_python_objects(base_logger, memory_sink, serializable_payload):
    base_logger.info("serialize_evt", **serializable_payload)
    payload = _first_record(memory_sink)["payload"]
    assert payload["path"] == "models/dummy.pkl"
    assert payload["timestamp"].startswith("2024-01-05T12:00:00")
    assert payload["date"] == "2024-01-05"
    assert payload["items"] == [1, 2, 3]
    assert payload["number"] == 1.23


def test_dataclass_payload_serializes_to_mapping(base_logger, memory_sink, dummy_exception_payload):
    base_logger.info("dataclass_evt", payload_obj=dummy_exception_payload)
    payload = _first_record(memory_sink)["payload"]
    assert payload["payload_obj"] == {"code": 7, "text": "boom"}


def test_unknown_object_falls_back_to_repr(logging_mod, memory_sink):
    logger = logging_mod.StructuredLogger("x", sink=memory_sink, min_level="DEBUG")
    logger.info("repr_evt", obj=EchoObject())
    payload = _first_record(memory_sink)["payload"]
    assert payload["obj"] == "EchoObject(repr)"


def test_long_text_is_truncated(logging_mod, memory_sink):
    logger = logging_mod.StructuredLogger(
        "x",
        sink=memory_sink,
        min_level="DEBUG",
        max_text_chars=12,
    )
    logger.info("truncate_evt", msg="abcdefghijklmnopqrstuvwxyz")
    payload = _first_record(memory_sink)["payload"]
    assert payload["msg"] == "abcdefghi..."
    assert len(payload["msg"]) == 12


def test_large_collection_is_truncated(logging_mod, memory_sink):
    logger = logging_mod.StructuredLogger(
        "x",
        sink=memory_sink,
        min_level="DEBUG",
        max_collection_items=3,
    )
    logger.info("coll_evt", values=[1, 2, 3, 4, 5], mapping={"a": 1, "b": 2, "c": 3, "d": 4})
    payload = _first_record(memory_sink)["payload"]
    assert payload["values"] == [1, 2, 3]
    assert list(payload["mapping"].keys()) == ["a", "b", "c"]


def test_base_fields_are_sanitized_and_redacted(logging_mod, memory_sink):
    logger = logging_mod.StructuredLogger(
        "x",
        sink=memory_sink,
        min_level="DEBUG",
        base_fields={"module": "tests", "token": "SECRET", "nested": {"password": "bad"}},
    )
    logger.info("evt", ok=True)
    record = _first_record(memory_sink)
    assert record["module"] == "tests"
    assert record["token"] == "[REDACTED]"
    assert record["nested"]["password"] == "[REDACTED]"


def test_stream_sink_roundtrip_with_logger(logging_mod, json_stream):
    logger = logging_mod.get_logger(
        "simons_core.tests",
        sink=logging_mod.StreamJsonSink(stream=json_stream, ensure_ascii=False, flush=True, sort_keys=True),
        min_level="DEBUG",
        base_fields={"module": "tests"},
    )
    logger.info("stream_evt", a=1)
    logger.error("stream_err", b=2)
    lines = _json_lines(json_stream)
    assert [rec["event"] for rec in lines] == ["stream_evt", "stream_err"]
    assert lines[0]["payload"] == {"a": 1}
    assert lines[1]["level"] == "ERROR"


def test_on_emit_error_raise_wraps_sink_failure(logging_mod):
    logger = logging_mod.get_logger(
        "simons_core.tests",
        sink=FailingSink(),
        min_level="DEBUG",
        on_emit_error="raise",
    )
    with pytest.raises(logging_mod.LoggerError, match="Failed to emit structured log record"):
        logger.info("evt", x=1)


def test_on_emit_error_stderr_falls_back(logging_mod, capsys):
    logger = logging_mod.get_logger(
        "simons_core.tests",
        sink=FailingSink(),
        min_level="DEBUG",
        on_emit_error="stderr",
    )
    logger.info("evt", x=1)
    captured = capsys.readouterr()
    assert "logging-fallback" in captured.err
    assert "evt" in captured.err


def test_with_step_emits_started_and_completed(base_logger, memory_sink, logging_mod):
    with logging_mod.with_step(base_logger.bind(run_id="run_1"), "feature_build", module="features") as step_logger:
        step_logger.info("inner_evt", rows=10)

    events = [r["event"] for r in memory_sink.records]
    assert events == ["step_started", "inner_evt", "step_completed"]
    started, inner, completed = memory_sink.records
    assert started["step"] == "feature_build"
    assert inner["step"] == "feature_build"
    assert started["module"] == "features"
    assert inner["payload"] == {"rows": 10}
    assert isinstance(completed["payload"]["duration_ms"], float)
    assert completed["payload"]["duration_ms"] >= 0.0


def test_with_step_failure_emits_step_failed_and_reraises(base_logger, memory_sink, logging_mod):
    with pytest.raises(RuntimeError, match="kaboom"):
        with logging_mod.with_step(base_logger.bind(run_id="run_1"), "fit_model"):
            raise RuntimeError("kaboom")

    events = [r["event"] for r in memory_sink.records]
    assert events == ["step_started", "step_failed"]
    failed = memory_sink.records[-1]
    assert failed["level"] == "ERROR"
    assert failed["step"] == "fit_model"
    assert failed["error_code"] == "RUNTIMEERROR"
    assert failed["exception"]["type"] == "RuntimeError"
    assert isinstance(failed["payload"]["duration_ms"], float)


def test_with_step_explicit_error_code_overrides(base_logger, memory_sink, logging_mod):
    with pytest.raises(ValueError):
        with logging_mod.with_step(base_logger.bind(run_id="run_1"), "fit_model", error_code="CUSTOM_FAIL"):
            raise ValueError("bad")
    failed = memory_sink.records[-1]
    assert failed["error_code"] == "CUSTOM_FAIL"


def test_with_step_rejects_empty_step(base_logger, logging_mod):
    with pytest.raises(ValueError, match="non-empty string"):
        with logging_mod.with_step(base_logger, "   "):
            pass


def test_log_method_accepts_dynamic_level(base_logger, memory_sink):
    base_logger.log("WARNING", "warn_evt", score=0.2)
    record = _first_record(memory_sink)
    assert record["level"] == "WARNING"
    assert record["event"] == "warn_evt"


def test_bind_returns_new_logger_without_mutating_original(base_logger, memory_sink):
    child = base_logger.bind(run_id="run_2")
    base_logger.info("base_evt")
    child.info("child_evt")
    first, second = memory_sink.records
    assert "run_id" not in first
    assert second["run_id"] == "run_2"


def test_base_fields_property_returns_copy(base_logger):
    fields = base_logger.base_fields
    fields["module"] = "mutated"
    assert base_logger.base_fields["module"] == "tests"


def test_exception_payload_without_traceback_option(base_logger, memory_sink, dummy_exception):
    base_logger.exception("boom_evt", exc=dummy_exception, include_traceback=False)
    record = _first_record(memory_sink)
    assert "traceback" not in record["exception"]
    assert record["exception"]["type"] == "RuntimeError"


def test_contextual_event_fields_live_at_top_level(contextual_logger, memory_sink):
    contextual_logger.bind(trace_id="trace-1", span_id="span-1").info("evt", value=3)
    record = _first_record(memory_sink)
    assert record["trace_id"] == "trace-1"
    assert record["span_id"] == "span-1"
    assert record["payload"] == {"value": 3}


def test_timestamp_format_is_iso_utc(base_logger, memory_sink):
    base_logger.info("evt")
    ts = _first_record(memory_sink)["ts_utc"]
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$", ts)


def test_payload_exception_object_is_serialized(base_logger, memory_sink):
    try:
        raise KeyError("missing")
    except KeyError as exc:
        base_logger.info("payload_exc_evt", err=exc)
    payload = _first_record(memory_sink)["payload"]
    assert payload["err"]["type"] == "KeyError"
    assert "missing" in payload["err"]["message"]


def test_record_is_json_serializable(base_logger, memory_sink, serializable_payload, sensitive_payload):
    base_logger.info("json_evt", **serializable_payload, **sensitive_payload)
    record = _first_record(memory_sink)
    encoded = json.dumps(record, ensure_ascii=False)
    decoded = json.loads(encoded)
    assert decoded["event"] == "json_evt"
    assert decoded["payload"]["token"] == "[REDACTED]"


def test_structured_logger_repr_fallback_for_non_jsonable_with_custom_fallback_disabled(logging_mod, memory_sink):
    logger = logging_mod.StructuredLogger("x", sink=memory_sink, min_level="DEBUG")
    logger.info("obj_evt", obj=IdentityFallbackLoggerObject())
    payload = _first_record(memory_sink)["payload"]
    assert "IdentityFallbackLoggerObject" in payload["obj"]


def test_sort_keys_on_stream_sink_produces_deterministic_key_order(logging_mod):
    stream = io.StringIO()
    sink = logging_mod.StreamJsonSink(stream=stream, sort_keys=True)
    sink.write({"z": 1, "a": 2, "m": 3})
    line = stream.getvalue().strip()
    assert line.startswith('{"a":2,"m":3,"z":1}')
