from __future__ import annotations

import json

import pandas as pd
import pytest


pytestmark = [pytest.mark.simons_core, pytest.mark.integration]


def _build_labels_for_features(interfaces_mod, features, *, name: str = "label"):
    values = pd.Series(
        [0.05, -0.02, 0.04][: len(features.features)],
        index=features.features.index,
        name=name,
        dtype=float,
    )
    return interfaces_mod.Labels(
        asof_date=features.asof_date,
        values=values,
        name=name,
        metadata={"source": "integration"},
    )


def _make_logger(logging_mod, memory_sink, context):
    return logging_mod.get_logger(
        "simons_core.integration",
        run_context=context,
        sink=memory_sink,
        min_level="DEBUG",
        base_fields={"suite": "integration"},
        require_run_id=True,
        on_emit_error="raise",
    )


def test_end_to_end_core_pipeline_happy_path(
    schemas_mod,
    interfaces_mod,
    logging_mod,
    market_calendar,
    memory_sink,
    run_context,
    canonical_prices_frame,
    dummy_provider,
    dummy_feature_builder,
    dummy_trainer,
    dummy_portfolio_engine,
    dummy_validator,
):
    session_label = market_calendar.align_to_session("2024-07-03 10:15", mode="session_label")
    context = run_context.child(run_id="run_integration_happy", asof_date=session_label)
    logger = _make_logger(logging_mod, memory_sink, context).bind(
        calendar_session=str(session_label.date()),
        market=context.market,
    )

    with logging_mod.with_step(logger, "core_pipeline", stage="integration") as step_logger:
        schemas_mod.assert_schema(canonical_prices_frame, "prices_adjusted")
        step_logger.info("schema_validated", schema_ref="prices_adjusted", n_rows=len(canonical_prices_frame))

        raw = dummy_provider.fetch(context.asof_date, context)
        interfaces_mod.ensure_no_future_data(
            raw.prices,
            asof_date=context.asof_date,
            timestamp_column=raw.timestamp_column,
        )

        features = dummy_feature_builder.build(raw, context)
        labels = _build_labels_for_features(interfaces_mod, features)
        interfaces_mod.validate_trainer_inputs(features, labels, context)

        model = dummy_trainer.fit(features, labels, context)
        preds = dummy_trainer.predict(features, model, context)
        interfaces_mod.validate_prediction_output(preds, features=features)

        decision = dummy_portfolio_engine.rebalance(preds, context)
        interfaces_mod.validate_portfolio_output(decision, context=context)

        gate = dummy_validator.validate(model, context)
        interfaces_mod.validate_gate_output(gate)

        step_logger.info(
            "pipeline_artifacts",
            model_ref=model.model_ref,
            n_features=int(features.features.shape[1]),
            n_predictions=int(len(preds.predictions)),
            gate_passed=gate.passed,
        )

    events = [r["event"] for r in memory_sink.records]
    assert events == [
        "step_started",
        "schema_validated",
        "pipeline_artifacts",
        "step_completed",
    ]

    started, validated, artifacts, completed = memory_sink.records
    assert started["run_id"] == "run_integration_happy"
    assert started["step"] == "core_pipeline"
    assert started["suite"] == "integration"
    assert validated["payload"]["schema_ref"] == "prices_adjusted"
    assert validated["payload"]["n_rows"] == len(canonical_prices_frame)
    assert artifacts["payload"]["gate_passed"] is True
    assert artifacts["payload"]["n_predictions"] == 3
    assert isinstance(completed["payload"]["duration_ms"], float)
    assert completed["payload"]["duration_ms"] >= 0.0


def test_calendar_driven_asof_propagates_consistently_through_interfaces(
    interfaces_mod,
    market_calendar,
    run_context,
    dummy_provider,
    dummy_feature_builder,
):
    aligned_label = market_calendar.align_to_session("2024-07-05 07:45", mode="session_label")
    context = run_context.child(run_id="run_calendar_alignment", asof_date=aligned_label)

    raw = dummy_provider.fetch(context.asof_date, context)
    features = dummy_feature_builder.build(raw, context)
    labels = _build_labels_for_features(interfaces_mod, features, name="fwd_ret")

    common_asof = interfaces_mod.ensure_same_asof_date(raw, features, labels)
    assert common_asof == context.asof_date
    assert raw.asof_date == context.asof_date
    assert features.asof_date == context.asof_date
    assert labels.asof_date == context.asof_date


def test_schema_failure_inside_with_step_emits_failed_event_and_reraises(
    schemas_mod,
    logging_mod,
    market_calendar,
    memory_sink,
    run_context,
    canonical_prices_frame,
):
    session_label = market_calendar.align_to_session("2024-07-03 10:15", mode="session_label")
    context = run_context.child(run_id="run_schema_failure", asof_date=session_label)
    logger = _make_logger(logging_mod, memory_sink, context)

    bad = canonical_prices_frame.copy()
    bad.loc[0, "high"] = bad.loc[0, "low"] - 0.5

    with pytest.raises(schemas_mod.SchemaValidationError):
        with logging_mod.with_step(logger, "schema_gate", stage="integration"):
            schemas_mod.assert_schema(bad, "prices_adjusted")

    events = [r["event"] for r in memory_sink.records]
    assert events == ["step_started", "step_failed"]

    failed = memory_sink.records[-1]
    assert failed["level"] == "ERROR"
    assert failed["step"] == "schema_gate"
    assert failed["error_code"] == "SCHEMAVALIDATIONERROR"
    assert failed["exception"]["type"] == "SchemaValidationError"
    assert isinstance(failed["payload"]["duration_ms"], float)


def test_future_data_relative_to_calendar_aligned_context_is_rejected(
    interfaces_mod,
    logging_mod,
    market_calendar,
    memory_sink,
    run_context,
    dummy_provider,
):
    session_label = market_calendar.align_to_session("2024-07-03 10:15", mode="session_label")
    context = run_context.child(run_id="run_future_data", asof_date=session_label)
    logger = _make_logger(logging_mod, memory_sink, context)

    raw = dummy_provider.fetch(context.asof_date, context)
    leaked = raw.prices.copy()
    leaked.loc[len(leaked) - 1, raw.timestamp_column] = pd.Timestamp("2024-08-01")

    with pytest.raises(interfaces_mod.DataLeakageError):
        with logging_mod.with_step(logger, "anti_lookahead_check", stage="integration"):
            interfaces_mod.ensure_no_future_data(
                leaked,
                asof_date=context.asof_date,
                timestamp_column=raw.timestamp_column,
            )

    assert [r["event"] for r in memory_sink.records] == ["step_started", "step_failed"]
    failed = memory_sink.records[-1]
    assert failed["error_code"] == "DATALEAKAGEERROR"
    assert failed["exception"]["type"] == "DataLeakageError"


def test_logging_event_can_carry_calendar_and_schema_metadata_together(
    schemas_mod,
    logging_mod,
    market_calendar,
    stream_sink,
    json_stream,
    run_context,
    canonical_prices_frame,
):
    session_label = market_calendar.align_to_session("2024-07-03 10:15", mode="session_label")
    session_type = market_calendar.session_type("2024-07-03").value
    prev_close = market_calendar.align_to_session("2024-07-03 10:15", mode="prev_close")
    context = run_context.child(run_id="run_structured_event", asof_date=session_label)

    logger = logging_mod.get_logger(
        "simons_core.integration",
        run_context=context,
        sink=stream_sink,
        min_level="DEBUG",
        base_fields={"suite": "integration"},
        require_run_id=True,
        on_emit_error="raise",
    )

    result = schemas_mod.validate_schema(canonical_prices_frame, "prices_adjusted")
    assert result.passed is True

    logger.info(
        "calendar_schema_evt",
        schema_ref=result.schema_name,
        schema_version=result.schema_version,
        calendar_session=str(session_label.date()),
        session_type=session_type,
        prev_close=str(prev_close),
        n_rows=len(canonical_prices_frame),
    )

    lines = [line for line in json_stream.getvalue().splitlines() if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["event"] == "calendar_schema_evt"
    assert record["run_id"] == "run_structured_event"
    assert record["suite"] == "integration"
    assert record["payload"]["schema_ref"] == "prices_adjusted"
    assert record["payload"]["session_type"] in {"regular", "early_close", "half_day", "special"}
    assert record["payload"]["n_rows"] == len(canonical_prices_frame)


def test_builtin_schema_registry_supports_minimal_cross_module_flow(
    schemas_mod,
    interfaces_mod,
    canonical_features_frame,
    run_context,
):
    schema = schemas_mod.get_schema("features")
    result = schemas_mod.validate_schema(canonical_features_frame, schema)
    assert result.passed is True

    feature_cols = tuple(col for col in canonical_features_frame.columns if col.startswith("feature_"))
    features = interfaces_mod.FeatureMatrix(
        asof_date=run_context.asof_date,
        features=canonical_features_frame.set_index(
            canonical_features_frame["symbol"].astype(str)
            + "_"
            + canonical_features_frame["date"].dt.strftime("%Y-%m-%d")
        )[list(feature_cols)],
        feature_names=feature_cols,
        metadata={"schema_ref": schema.name, "schema_version": schema.version},
    )

    assert features.metadata["schema_ref"] == "features"
    assert len(features.feature_names) >= 1
    assert all(name.startswith("feature_") for name in features.feature_names)
