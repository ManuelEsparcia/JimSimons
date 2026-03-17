from __future__ import annotations

import pandas as pd
import pytest


pytestmark = [pytest.mark.simons_core, pytest.mark.schemas]


def _make_duplicate_column_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [[pd.Timestamp("2024-01-02"), "AAA", 1.0, 1.1]],
        columns=["date", "symbol", "value", "value"],
    )


def _make_labels_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "symbol": ["AAA", "BBB", "CCC"],
            "future_ret_5d": [0.02, -0.01, 0.03],
            "binary_swing": [1, 0, 1],
        }
    )


def _make_predictions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "symbol": ["AAA", "BBB", "CCC"],
            "score": [0.7, 0.2, 0.6],
            "prob_long": [0.8, 0.3, 0.55],
            "confidence": [0.9, 0.4, 0.7],
        }
    )


def test_column_pattern_spec_compiles_regex(schemas_mod):
    spec = schemas_mod.ColumnPatternSpec(pattern=r"feature_[A-Za-z0-9_]+", dtype="numeric")
    regex = spec.compiled()
    assert regex.fullmatch("feature_mom_5d") is not None
    assert regex.fullmatch("foo_feature") is None



def test_data_schema_properties_and_get_column(simple_schema):
    assert simple_schema.column_names == ("date", "symbol", "value", "flag")
    assert simple_schema.required_columns == ("date", "symbol", "value")
    value_spec = simple_schema.get_column("value")
    assert value_spec is not None
    assert value_spec.dtype == "float"
    assert simple_schema.get_column("missing") is None



def test_validation_result_by_code_and_summary_for_success(schemas_mod, simple_schema, valid_simple_frame):
    result = schemas_mod.validate_schema(valid_simple_frame, simple_schema)
    assert result.passed is True
    assert result.by_code(schemas_mod.MISSING_COLUMN) == ()
    summary = result.summary()
    assert "Validation passed" in summary
    assert simple_schema.name in summary



def test_validate_schema_simple_success(schemas_mod, simple_schema, valid_simple_frame):
    result = schemas_mod.validate_schema(valid_simple_frame, simple_schema)
    assert result.passed is True
    assert result.issues == ()
    assert result.schema_name == simple_schema.name
    assert result.schema_version == simple_schema.version



def test_validate_schema_accepts_schema_name_from_registry(schemas_mod, canonical_prices_frame):
    result = schemas_mod.validate_schema(canonical_prices_frame, "prices_adjusted")
    assert result.passed is True
    assert result.schema_name == "prices_adjusted"



def test_validate_schema_missing_required_column(schemas_mod, simple_schema, invalid_simple_frame_missing_column):
    result = schemas_mod.validate_schema(invalid_simple_frame_missing_column, simple_schema)
    assert result.passed is False
    issues = result.by_code(schemas_mod.MISSING_COLUMN)
    assert issues
    assert any(issue.column == "value" for issue in issues)



def test_validate_schema_unexpected_column_when_not_allowed(schemas_mod, simple_schema, valid_simple_frame):
    df = valid_simple_frame.assign(extra_col=[1, 2, 3])
    result = schemas_mod.validate_schema(df, simple_schema)
    assert result.passed is False
    issues = result.by_code(schemas_mod.UNEXPECTED_COLUMN)
    assert len(issues) == 1
    assert issues[0].column == "extra_col"



def test_validate_schema_allows_extra_columns_when_enabled(schemas_mod, valid_simple_frame):
    schema = schemas_mod.DataSchema(
        name="test.extra",
        version="v1",
        columns=(
            schemas_mod.ColumnSpec("date", "datetime", nullable=False),
            schemas_mod.ColumnSpec("symbol", "string", nullable=False),
        ),
        primary_key=("date", "symbol"),
        allow_extra_columns=True,
    )
    df = valid_simple_frame.assign(extra_col=[1, 2, 3])
    result = schemas_mod.validate_schema(df, schema)
    assert result.passed is True



def test_validate_schema_duplicate_column_name_detected(schemas_mod, simple_schema):
    result = schemas_mod.validate_schema(_make_duplicate_column_frame(), simple_schema)
    assert result.passed is False
    issues = result.by_code(schemas_mod.DUPLICATE_COLUMN_NAME)
    assert issues
    assert issues[0].column == "value"



def test_validate_schema_empty_dataset_forbidden(schemas_mod, simple_schema):
    empty = pd.DataFrame(columns=["date", "symbol", "value", "flag"])
    result = schemas_mod.validate_schema(empty, simple_schema)
    assert result.passed is False
    issues = result.by_code(schemas_mod.EMPTY_DATASET)
    assert len(issues) == 1



def test_validate_schema_invalid_dtype_detected(schemas_mod, simple_schema, valid_simple_frame):
    df = valid_simple_frame.copy()
    df["value"] = ["1.0", "2.0", "3.0"]
    result = schemas_mod.validate_schema(df, simple_schema)
    assert result.passed is False
    issues = result.by_code(schemas_mod.INVALID_DTYPE)
    assert any(issue.column == "value" for issue in issues)



def test_validate_schema_null_in_non_nullable_detected(schemas_mod, simple_schema, valid_simple_frame):
    df = valid_simple_frame.copy()
    df.loc[1, "value"] = None
    result = schemas_mod.validate_schema(df, simple_schema)
    assert result.passed is False
    issues = result.by_code(schemas_mod.NULL_IN_NONNULLABLE)
    assert any(issue.column == "value" for issue in issues)



def test_validate_schema_min_value_domain_violation(schemas_mod, simple_schema, valid_simple_frame):
    df = valid_simple_frame.copy()
    df.loc[2, "value"] = -0.01
    result = schemas_mod.validate_schema(df, simple_schema)
    assert result.passed is False
    issues = result.by_code(schemas_mod.DOMAIN_VIOLATION)
    assert any(issue.column == "value" for issue in issues)



def test_validate_schema_allowed_values_violation(schemas_mod):
    schema = schemas_mod.DataSchema(
        name="test.allowed_values",
        version="v1",
        columns=(
            schemas_mod.ColumnSpec("state", "string", nullable=False, allowed_values=("ok", "bad")),
        ),
        allow_extra_columns=False,
    )
    df = pd.DataFrame({"state": ["ok", "weird"]})
    result = schemas_mod.validate_schema(df, schema)
    assert result.passed is False
    issues = result.by_code(schemas_mod.DOMAIN_VIOLATION)
    assert len(issues) == 1
    assert issues[0].column == "state"



def test_pattern_schema_success(schemas_mod, pattern_schema, valid_pattern_frame):
    result = schemas_mod.validate_schema(valid_pattern_frame, pattern_schema)
    assert result.passed is True



def test_pattern_schema_min_count_enforced(schemas_mod, pattern_schema):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["AAA", "BBB"],
            "feature_only_one": [0.1, 0.2],
        }
    )
    result = schemas_mod.validate_schema(df, pattern_schema)
    assert result.passed is False
    issues = result.by_code(schemas_mod.MISSING_COLUMN)
    assert issues
    assert any("Expected at least 2 column(s) matching pattern" in issue.message for issue in issues)



def test_pattern_schema_invalid_dtype_detected(schemas_mod, pattern_schema, valid_pattern_frame):
    df = valid_pattern_frame.copy()
    df["feature_vol_20d"] = ["bad", "bad", "bad"]
    result = schemas_mod.validate_schema(df, pattern_schema)
    assert result.passed is False
    issues = result.by_code(schemas_mod.INVALID_DTYPE)
    assert any(issue.column == "feature_vol_20d" for issue in issues)



def test_primary_key_duplicate_detected(schemas_mod, simple_schema, invalid_simple_frame_duplicate_pk):
    result = schemas_mod.validate_schema(invalid_simple_frame_duplicate_pk, simple_schema)
    assert result.passed is False
    issues = result.by_code(schemas_mod.PRIMARY_KEY_VIOLATION)
    assert len(issues) == 1



def test_non_monotonic_date_detected(schemas_mod, simple_schema, valid_simple_frame):
    df = valid_simple_frame.iloc[[1, 0, 2]].reset_index(drop=True)
    result = schemas_mod.validate_schema(df, simple_schema)
    assert result.passed is False
    issues = result.by_code(schemas_mod.NON_MONOTONIC_DATE)
    assert len(issues) == 1
    assert issues[0].column == "date"



def test_future_timestamp_violation_detected(schemas_mod, simple_schema, invalid_simple_frame_future_date, asof_date):
    result = schemas_mod.validate_schema(invalid_simple_frame_future_date, simple_schema, asof_date=asof_date)
    assert result.passed is False
    issues = result.by_code(schemas_mod.FUTURE_TIMESTAMP_VIOLATION)
    assert len(issues) == 1
    assert issues[0].column == "date"



def test_custom_validator_runs_and_accumulates_issues(schemas_mod, valid_simple_frame):
    def custom_validator(df: pd.DataFrame, schema) -> list:
        return [schemas_mod.ValidationIssue(code="CUSTOM", message=f"validator for {schema.name}")]

    schema = schemas_mod.DataSchema(
        name="test.custom_validator",
        version="v1",
        columns=(
            schemas_mod.ColumnSpec("date", "datetime", nullable=False),
            schemas_mod.ColumnSpec("symbol", "string", nullable=False),
            schemas_mod.ColumnSpec("value", "float", nullable=False),
        ),
        validators=(custom_validator,),
    )
    result = schemas_mod.validate_schema(valid_simple_frame[["date", "symbol", "value"]], schema)
    assert result.passed is False
    assert result.by_code("CUSTOM")



def test_price_adjusted_schema_success(schemas_mod, canonical_prices_frame):
    result = schemas_mod.validate_schema(canonical_prices_frame, schemas_mod.PRICE_ADJUSTED_SCHEMA)
    assert result.passed is True



def test_price_adjusted_schema_high_less_than_low_fails(schemas_mod, invalid_prices_frame_high_lt_low):
    result = schemas_mod.validate_schema(invalid_prices_frame_high_lt_low, schemas_mod.PRICE_ADJUSTED_SCHEMA)
    assert result.passed is False
    issues = result.by_code(schemas_mod.DOMAIN_VIOLATION)
    assert any("high < low" in issue.message for issue in issues)



def test_price_adjusted_schema_non_positive_ohlc_and_negative_volume_fail(schemas_mod, canonical_prices_frame):
    df = canonical_prices_frame.copy()
    df.loc[0, "open"] = 0.0
    df.loc[1, "volume"] = -1
    result = schemas_mod.validate_schema(df, schemas_mod.PRICE_ADJUSTED_SCHEMA)
    assert result.passed is False
    issues = result.by_code(schemas_mod.DOMAIN_VIOLATION)
    assert any(issue.column == "open" for issue in issues)
    assert any(issue.column == "volume" for issue in issues)



def test_features_schema_success(schemas_mod, canonical_features_frame):
    result = schemas_mod.validate_schema(canonical_features_frame, schemas_mod.FEATURES_SCHEMA)
    assert result.passed is True



def test_features_schema_requires_at_least_one_feature_column(schemas_mod):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["AAA", "BBB"],
        }
    )
    result = schemas_mod.validate_schema(df, schemas_mod.FEATURES_SCHEMA)
    assert result.passed is False
    issues = result.by_code(schemas_mod.MISSING_COLUMN)
    assert issues
    assert any("must contain at least one column matching" in issue.message for issue in issues)



def test_labels_schema_success(schemas_mod):
    result = schemas_mod.validate_schema(_make_labels_frame(), schemas_mod.LABELS_SCHEMA)
    assert result.passed is True



def test_labels_schema_detects_domain_violation(schemas_mod):
    df = _make_labels_frame()
    df.loc[1, "future_ret_5d"] = 1.5
    result = schemas_mod.validate_schema(df, schemas_mod.LABELS_SCHEMA)
    assert result.passed is False
    issues = result.by_code(schemas_mod.DOMAIN_VIOLATION)
    assert any(issue.column == "future_ret_5d" for issue in issues)



def test_predictions_schema_success(schemas_mod):
    result = schemas_mod.validate_schema(_make_predictions_frame(), schemas_mod.PREDICTIONS_SCHEMA)
    assert result.passed is True



def test_predictions_schema_detects_probability_range_violation(schemas_mod):
    df = _make_predictions_frame()
    df.loc[0, "prob_long"] = 1.2
    df.loc[2, "confidence"] = -0.1
    result = schemas_mod.validate_schema(df, schemas_mod.PREDICTIONS_SCHEMA)
    assert result.passed is False
    issues = result.by_code(schemas_mod.DOMAIN_VIOLATION)
    assert any(issue.column == "prob_long" for issue in issues)
    assert any(issue.column == "confidence" for issue in issues)



def test_get_schema_and_list_schemas(schemas_mod):
    names = schemas_mod.list_schemas()
    assert names == sorted(names)
    assert {"prices_adjusted", "features", "labels", "predictions"}.issubset(set(names))
    schema = schemas_mod.get_schema("features")
    assert schema.name == "features"



def test_get_schema_unknown_raises_key_error(schemas_mod):
    with pytest.raises(KeyError, match="Unknown schema"):
        schemas_mod.get_schema("definitely_missing")



def test_format_validation_issues_renders_deterministically(schemas_mod, simple_schema, invalid_simple_frame_missing_column):
    result = schemas_mod.validate_schema(invalid_simple_frame_missing_column, simple_schema)
    rendered = schemas_mod.format_validation_issues(result)
    assert schemas_mod.MISSING_COLUMN in rendered
    assert "[value]" in rendered



def test_assert_schema_raises_with_formatted_message(schemas_mod, simple_schema, invalid_simple_frame_missing_column):
    with pytest.raises(schemas_mod.SchemaValidationError, match="MISSING_COLUMN"):
        schemas_mod.assert_schema(invalid_simple_frame_missing_column, simple_schema)



def test_assert_schema_passes_silently_on_valid_input(schemas_mod, simple_schema, valid_simple_frame):
    schemas_mod.assert_schema(valid_simple_frame, simple_schema)
