from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path as _Path

import pandas as pd
import pytest


pytestmark = [pytest.mark.simons_core, pytest.mark.interfaces]


def test_module_constants_are_non_empty_strings(interfaces_mod):
    assert isinstance(interfaces_mod.DEFAULT_PIPELINE_VERSION, str)
    assert interfaces_mod.DEFAULT_PIPELINE_VERSION
    assert isinstance(interfaces_mod.DEFAULT_MARKET, str)
    assert interfaces_mod.DEFAULT_MARKET
    assert isinstance(interfaces_mod.CONTRACT_VERSION, str)
    assert interfaces_mod.CONTRACT_VERSION


def test_run_context_normalizes_and_freezes(asof_date, interfaces_mod):
    ctx = interfaces_mod.RunContext(
        run_id="run_001",
        asof_date=str(asof_date.date()),
        seed=123,
        config_hash="cfg_abc",
        metadata={"owner": "quant"},
    )
    assert isinstance(ctx.asof_date, pd.Timestamp)
    assert ctx.asof_date == asof_date
    assert dict(ctx.metadata) == {"owner": "quant"}
    with pytest.raises(TypeError):
        ctx.metadata["owner"] = "other"  # type: ignore[index]


def test_run_context_is_frozen(run_context):
    with pytest.raises(FrozenInstanceError):
        run_context.run_id = "mutated"  # type: ignore[misc]


def test_run_context_child_preserves_lineage_and_defaults(run_context):
    child = run_context.child(run_id="run_child_001")
    assert child.run_id == "run_child_001"
    assert child.parent_run_id == run_context.run_id
    assert child.asof_date == run_context.asof_date
    assert child.seed == run_context.seed
    assert child.config_hash == run_context.config_hash
    assert child.pipeline_version == run_context.pipeline_version
    assert dict(child.metadata) == dict(run_context.metadata)


def test_run_context_child_can_override_asof_and_seed(run_context):
    child = run_context.child(run_id="run_child_002", asof_date="2024-01-04", seed=999)
    assert child.asof_date == pd.Timestamp("2024-01-04")
    assert child.seed == 999


@pytest.mark.parametrize(
    ("payload", "exc_type", "message_part"),
    [
        (dict(run_id="", asof_date="2024-01-05", seed=1, config_hash="cfg"), "ReproducibilityError", "run_id"),
        (dict(run_id="r", asof_date="2024-01-05", seed="bad", config_hash="cfg"), "ReproducibilityError", "seed"),
        (dict(run_id="r", asof_date="2024-01-05", seed=1, config_hash=""), "ReproducibilityError", "config_hash"),
        (
            dict(run_id="r", asof_date="2024-01-05", seed=1, config_hash="cfg", pipeline_version=""),
            "ReproducibilityError",
            "pipeline_version",
        ),
    ],
)
def test_run_context_invalid_reproducibility_raises(interfaces_mod, payload, exc_type, message_part):
    exc = getattr(interfaces_mod, exc_type)
    with pytest.raises(exc, match=message_part):
        interfaces_mod.RunContext(**payload)


def test_run_context_to_dict_is_json_friendly(run_context):
    payload = run_context.to_dict()
    assert payload["run_id"] == run_context.run_id
    assert payload["asof_date"] == run_context.asof_date.isoformat()
    assert payload["seed"] == run_context.seed
    assert payload["config_hash"] == run_context.config_hash


def test_gate_result_normalizes_and_freezes(gate_result):
    assert gate_result.passed is True
    assert gate_result.score == 0.91
    assert gate_result.threshold == 0.8
    assert gate_result.reasons == ("all_good",)
    assert gate_result.metrics["sharpe"] == 1.6
    with pytest.raises(TypeError):
        gate_result.metrics["sharpe"] = 99.0  # type: ignore[index]
    with pytest.raises(TypeError):
        gate_result.metadata["validator"] = "other"  # type: ignore[index]


def test_gate_result_from_score_sets_passed_and_keeps_payload(interfaces_mod):
    result = interfaces_mod.GateResult.from_score(
        0.75,
        threshold=0.7,
        reasons=["good"],
        metrics={"alpha": 1.2},
        metadata={"tag": "x"},
    )
    assert result.passed is True
    assert result.score == 0.75
    assert result.threshold == 0.7
    assert result.reasons == ("good",)
    assert result.metrics["alpha"] == 1.2
    assert result.metadata["tag"] == "x"


@pytest.mark.parametrize(
    ("kwargs", "message_part"),
    [
        (dict(passed=True, score=1.1, threshold=0.8), "score"),
        (dict(passed=True, score=0.9, threshold=1.1), "threshold"),
        (dict(passed=False, score=0.9, threshold=0.8), "inconsistent"),
        (dict(passed=True, score=0.9, threshold=0.8, metrics={"bad": "x"}), "metrics"),
    ],
)
def test_gate_result_invalid_semantics_raise(interfaces_mod, kwargs, message_part):
    with pytest.raises((interfaces_mod.GateSemanticsError, interfaces_mod.SchemaViolation), match=message_part):
        interfaces_mod.GateResult(**kwargs)


def test_price_data_success_and_n_rows(price_data):
    assert price_data.n_rows == 3
    assert price_data.symbols == ("AAA",)
    assert dict(price_data.metadata) == {"source": "fixture"}


def test_price_data_non_dataframe_raises(interfaces_mod, asof_date):
    with pytest.raises(interfaces_mod.SchemaViolation, match="DataFrame"):
        interfaces_mod.PriceData(asof_date=asof_date, prices=[1, 2, 3])  # type: ignore[arg-type]


def test_price_data_empty_raises(interfaces_mod, asof_date):
    with pytest.raises(interfaces_mod.SchemaViolation, match="must not be empty"):
        interfaces_mod.PriceData(asof_date=asof_date, prices=pd.DataFrame())


def test_price_data_future_timestamp_raises(interfaces_mod, asof_date, raw_price_frame):
    bad = raw_price_frame.copy()
    bad.loc[len(bad)] = [pd.Timestamp("2024-01-06"), "AAA", 10.5, 123]
    with pytest.raises(interfaces_mod.DataLeakageError, match="beyond asof_date"):
        interfaces_mod.PriceData(asof_date=asof_date, prices=bad, timestamp_column="date")


def test_feature_matrix_success(feature_matrix, assert_frame_equal):
    assert feature_matrix.shape == (3, 2)
    assert feature_matrix.feature_names == ("feature_mom_2d", "feature_volume_z")
    assert_frame_equal(feature_matrix.feature_frame(), feature_matrix.features)


def test_feature_matrix_selective_feature_frame(interfaces_mod, asof_date, features_frame, assert_frame_equal):
    fm = interfaces_mod.FeatureMatrix(
        asof_date=asof_date,
        features=features_frame.assign(non_feature=["x", "y", "z"]),
        feature_names=("feature_mom_2d", "feature_volume_z"),
    )
    expected = features_frame[["feature_mom_2d", "feature_volume_z"]]
    assert_frame_equal(fm.feature_frame(), expected)


def test_feature_matrix_missing_declared_feature_name_raises(interfaces_mod, asof_date, features_frame):
    with pytest.raises(interfaces_mod.SchemaViolation, match="absent from features"):
        interfaces_mod.FeatureMatrix(
            asof_date=asof_date,
            features=features_frame,
            feature_names=("feature_mom_2d", "missing_feature"),
        )


def test_feature_matrix_non_numeric_declared_feature_raises(interfaces_mod, asof_date, features_frame):
    bad = features_frame.copy()
    bad["feature_volume_z"] = ["a", "b", "c"]
    with pytest.raises(interfaces_mod.SchemaViolation, match="non-numeric"):
        interfaces_mod.FeatureMatrix(
            asof_date=asof_date,
            features=bad,
            feature_names=("feature_mom_2d", "feature_volume_z"),
        )


def test_labels_success(labels_obj):
    assert labels_obj.name == "label"
    assert labels_obj.values.name == "label"
    assert labels_obj.asof_date == pd.Timestamp("2024-01-05")


def test_labels_non_numeric_raise(interfaces_mod, asof_date, labels_series):
    bad = labels_series.astype(str)
    with pytest.raises(interfaces_mod.SchemaViolation, match="numeric"):
        interfaces_mod.Labels(asof_date=asof_date, values=bad, name="label")


def test_model_artifact_normalizes_metrics_and_path(model_artifact):
    assert model_artifact.model_ref == "dummy.linear.v1"
    assert isinstance(model_artifact.artifact_path_obj, _Path)
    assert model_artifact.artifact_path_obj == _Path("models/dummy_linear_v1.pkl")
    assert model_artifact.metrics["train_rmse"] == 0.12
    assert model_artifact.created_at_utc.tzinfo is None


def test_model_artifact_invalid_inputs_raise(interfaces_mod):
    with pytest.raises(interfaces_mod.SchemaViolation, match="model_ref"):
        interfaces_mod.ModelArtifact(model_ref="", artifact_path="a.pkl")
    with pytest.raises(interfaces_mod.SchemaViolation, match="artifact_path"):
        interfaces_mod.ModelArtifact(model_ref="m", artifact_path="")
    with pytest.raises(interfaces_mod.SchemaViolation, match="metrics"):
        interfaces_mod.ModelArtifact(model_ref="m", artifact_path="a.pkl", metrics={"rmse": "bad"})


def test_prediction_frame_success(prediction_frame):
    assert prediction_frame.score_columns == ("score",)
    assert prediction_frame.predictions.shape == (3, 1)


def test_prediction_frame_missing_score_column_raises(interfaces_mod, asof_date, predictions_frame):
    with pytest.raises(interfaces_mod.SchemaViolation, match="absent from predictions"):
        interfaces_mod.PredictionFrame(
            asof_date=asof_date,
            predictions=predictions_frame,
            score_columns=("score", "missing"),
        )


def test_prediction_frame_non_numeric_score_raises(interfaces_mod, asof_date, predictions_frame):
    bad = predictions_frame.copy()
    bad["score"] = ["x", "y", "z"]
    with pytest.raises(interfaces_mod.SchemaViolation, match="non-numeric"):
        interfaces_mod.PredictionFrame(
            asof_date=asof_date,
            predictions=bad,
            score_columns=("score",),
        )


def test_portfolio_decision_success_and_exposures(portfolio_decision):
    assert portfolio_decision.timestamp == pd.Timestamp("2024-01-05")
    assert portfolio_decision.turnover_l1 == 0.15
    assert portfolio_decision.notional_exposure == 1.0
    assert portfolio_decision.gross_exposure == pytest.approx(1.0)
    assert portfolio_decision.net_exposure == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kwargs", "exc_name", "message_part"),
    [
        (dict(weights=[]), "SchemaViolation", "Series"),
        (dict(weights=pd.Series(dtype=float)), "FeasibilityError", "must not be empty"),
        (dict(weights=pd.Series([0.1, None], index=["A", "B"])), "FeasibilityError", "NaN"),
        (dict(weights=pd.Series(["x", "y"], index=["A", "B"])), "SchemaViolation", "numeric"),
        (dict(turnover_l1=-0.1), "FeasibilityError", "turnover_l1"),
        (dict(notional_exposure=-1.0), "FeasibilityError", "notional_exposure"),
    ],
)
def test_portfolio_decision_invalid_cases_raise(interfaces_mod, asof_date, weights_series, kwargs, exc_name, message_part):
    payload = dict(
        timestamp=asof_date,
        weights=weights_series,
        turnover_l1=0.15,
        notional_exposure=1.0,
    )
    payload.update(kwargs)
    exc = getattr(interfaces_mod, exc_name)
    with pytest.raises(exc, match=message_part):
        interfaces_mod.PortfolioDecision(**payload)


def test_ensure_no_future_data_passes_for_valid_frame(interfaces_mod, raw_price_frame, asof_date):
    interfaces_mod.ensure_no_future_data(raw_price_frame, asof_date=asof_date, timestamp_column="date")


def test_ensure_no_future_data_ignores_missing_timestamp_column(interfaces_mod):
    interfaces_mod.ensure_no_future_data(pd.DataFrame({"x": [1, 2]}), asof_date="2024-01-05", timestamp_column="date")


def test_ensure_no_future_data_invalid_timestamp_column_raises(interfaces_mod):
    frame = pd.DataFrame({"date": ["not-a-date"], "x": [1]})
    with pytest.raises(interfaces_mod.SchemaViolation, match="non-coercible"):
        interfaces_mod.ensure_no_future_data(frame, asof_date="2024-01-05", timestamp_column="date")


def test_ensure_index_aligned_passes_and_fails(interfaces_mod):
    idx = pd.Index(["a", "b", "c"])
    interfaces_mod.ensure_index_aligned(idx, idx.copy(), label="stuff")
    with pytest.raises(interfaces_mod.AlignmentError, match="Misaligned stuff"):
        interfaces_mod.ensure_index_aligned(idx, pd.Index(["a", "c", "b"]), label="stuff")


def test_ensure_same_asof_date_passes_and_returns_timestamp(interfaces_mod, feature_matrix, labels_obj):
    ts = interfaces_mod.ensure_same_asof_date(feature_matrix, labels_obj)
    assert ts == pd.Timestamp("2024-01-05")


def test_ensure_same_asof_date_fails_on_missing_attribute(interfaces_mod, feature_matrix):
    with pytest.raises(interfaces_mod.AlignmentError, match="no asof_date"):
        interfaces_mod.ensure_same_asof_date(feature_matrix, object())


def test_ensure_same_asof_date_fails_on_mismatch(interfaces_mod, feature_matrix, labels_obj):
    other = interfaces_mod.Labels(
        asof_date="2024-01-04",
        values=labels_obj.values,
        name="label",
    )
    with pytest.raises(interfaces_mod.AlignmentError, match="same asof_date"):
        interfaces_mod.ensure_same_asof_date(feature_matrix, other)


def test_ensure_single_effective_timestamp_passes_and_fails(interfaces_mod):
    good = pd.DataFrame({"date": pd.to_datetime(["2024-01-05", "2024-01-05"]), "score": [0.1, 0.2]})
    ts = interfaces_mod.ensure_single_effective_timestamp(good, timestamp_column="date")
    assert ts == pd.Timestamp("2024-01-05")

    bad = pd.DataFrame({"date": pd.to_datetime(["2024-01-05", "2024-01-06"]), "score": [0.1, 0.2]})
    with pytest.raises(interfaces_mod.AlignmentError, match="single effective timestamp"):
        interfaces_mod.ensure_single_effective_timestamp(bad, timestamp_column="date")


def test_ensure_single_effective_timestamp_invalid_cases_raise(interfaces_mod):
    with pytest.raises(interfaces_mod.SchemaViolation, match="DataFrame"):
        interfaces_mod.ensure_single_effective_timestamp([1, 2, 3])  # type: ignore[arg-type]
    with pytest.raises(interfaces_mod.SchemaViolation, match="timestamp column"):
        interfaces_mod.ensure_single_effective_timestamp(pd.DataFrame({"x": [1]}), timestamp_column="date")
    with pytest.raises(interfaces_mod.SchemaViolation, match="invalid timestamps"):
        interfaces_mod.ensure_single_effective_timestamp(
            pd.DataFrame({"date": ["bad", "bad"], "x": [1, 2]}), timestamp_column="date"
        )


def test_ensure_numeric_frame_and_series_pass_and_fail(interfaces_mod):
    interfaces_mod.ensure_numeric_frame(pd.DataFrame({"a": [1.0], "b": [2]}), label="frame")
    interfaces_mod.ensure_numeric_series(pd.Series([1.0, 2.0]), label="series")

    with pytest.raises(interfaces_mod.SchemaViolation, match="DataFrame"):
        interfaces_mod.ensure_numeric_frame([1, 2, 3], label="frame")  # type: ignore[arg-type]
    with pytest.raises(interfaces_mod.SchemaViolation, match="non-numeric"):
        interfaces_mod.ensure_numeric_frame(pd.DataFrame({"a": [1.0], "b": ["x"]}), label="frame")
    with pytest.raises(interfaces_mod.SchemaViolation, match="Series"):
        interfaces_mod.ensure_numeric_series([1, 2, 3], label="series")  # type: ignore[arg-type]
    with pytest.raises(interfaces_mod.SchemaViolation, match="numeric"):
        interfaces_mod.ensure_numeric_series(pd.Series(["x", "y"]), label="series")


def test_contract_stage_version_check(dummy_provider, interfaces_mod):
    dummy_provider.assert_compatible_version()
    dummy_provider.contract_version = "0.0.1"
    with pytest.raises(interfaces_mod.ContractVersionError, match="contract_version"):
        dummy_provider.assert_compatible_version()


def test_abstract_stages_cannot_be_instantiated(interfaces_mod):
    for cls in (
        interfaces_mod.DataProvider,
        interfaces_mod.FeatureBuilder,
        interfaces_mod.ModelTrainer,
        interfaces_mod.PortfolioEngine,
        interfaces_mod.Validator,
    ):
        with pytest.raises(TypeError):
            cls()  # type: ignore[abstract]


def test_dummy_provider_and_validator_output_contracts(dummy_provider, dummy_validator, run_context, interfaces_mod):
    data = dummy_provider.fetch(run_context.asof_date, run_context)
    interfaces_mod.validate_provider_output(data, requested_asof_date=run_context.asof_date)

    gate = dummy_validator.validate(data, run_context)
    interfaces_mod.validate_gate_output(gate)
    assert gate.passed is True


def test_dummy_feature_builder_output_contract(dummy_provider, dummy_feature_builder, run_context, interfaces_mod):
    data = dummy_provider.fetch(run_context.asof_date, run_context)
    features = dummy_feature_builder.build(data, run_context)
    interfaces_mod.validate_feature_builder_output(features, source=data)
    assert features.shape == (3, 2)


def test_validate_trainer_inputs_passes(dummy_provider, dummy_feature_builder, labels_obj, run_context, interfaces_mod):
    data = dummy_provider.fetch(run_context.asof_date, run_context)
    features = dummy_feature_builder.build(data, run_context)
    interfaces_mod.validate_trainer_inputs(features, labels_obj, run_context)


def test_validate_trainer_inputs_fails_on_index_mismatch(feature_matrix, labels_obj, run_context, interfaces_mod):
    bad_labels = interfaces_mod.Labels(
        asof_date=labels_obj.asof_date,
        values=pd.Series([0.1, 0.2, 0.3], index=pd.Index(["x", "y", "z"], name="row_id"), name="label"),
        name="label",
    )
    with pytest.raises(interfaces_mod.AlignmentError, match="features and labels"):
        interfaces_mod.validate_trainer_inputs(feature_matrix, bad_labels, run_context)


def test_validate_trainer_inputs_fails_on_future_input(feature_matrix, labels_obj, interfaces_mod):
    context = interfaces_mod.RunContext(
        run_id="r",
        asof_date="2024-01-04",
        seed=1,
        config_hash="cfg",
    )
    with pytest.raises(interfaces_mod.DataLeakageError, match="exceed RunContext.asof_date"):
        interfaces_mod.validate_trainer_inputs(feature_matrix, labels_obj, context)


def test_dummy_trainer_fit_and_predict_contracts(
    dummy_provider,
    dummy_feature_builder,
    dummy_trainer,
    labels_obj,
    run_context,
    interfaces_mod,
):
    data = dummy_provider.fetch(run_context.asof_date, run_context)
    features = dummy_feature_builder.build(data, run_context)

    model = dummy_trainer.fit(features, labels_obj, run_context)
    assert model.model_ref == "dummy.linear.v1"

    preds = dummy_trainer.predict(features, model, run_context)
    interfaces_mod.validate_prediction_output(preds, features=features)
    assert preds.score_columns == ("score",)


def test_validate_prediction_output_fails_on_index_mismatch(feature_matrix, prediction_frame, interfaces_mod):
    bad = interfaces_mod.PredictionFrame(
        asof_date=prediction_frame.asof_date,
        predictions=prediction_frame.predictions.rename(index=lambda x: f"bad_{x}"),
        score_columns=prediction_frame.score_columns,
    )
    with pytest.raises(interfaces_mod.AlignmentError, match="predictions and features"):
        interfaces_mod.validate_prediction_output(bad, features=feature_matrix)


def test_dummy_portfolio_engine_output_contract(dummy_portfolio_engine, prediction_frame, run_context, interfaces_mod):
    decision = dummy_portfolio_engine.rebalance(prediction_frame, run_context)
    interfaces_mod.validate_portfolio_output(decision, context=run_context)
    assert decision.gross_exposure == pytest.approx(1.0)


def test_validate_portfolio_output_fails_on_future_timestamp(portfolio_decision, interfaces_mod):
    context = interfaces_mod.RunContext(
        run_id="r",
        asof_date="2024-01-04",
        seed=1,
        config_hash="cfg",
    )
    with pytest.raises(interfaces_mod.DataLeakageError, match="exceeds RunContext.asof_date"):
        interfaces_mod.validate_portfolio_output(portfolio_decision, context=context)


def test_validate_gate_output_type_check(interfaces_mod):
    interfaces_mod.validate_gate_output(interfaces_mod.GateResult.from_score(0.9))
    with pytest.raises(interfaces_mod.SchemaViolation, match="GateResult"):
        interfaces_mod.validate_gate_output({"passed": True})  # type: ignore[arg-type]
