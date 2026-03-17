from __future__ import annotations

"""
Shared pytest fixtures for ``tests/simons_core``.

This file centralizes reusable fixtures for the four foundational modules in
``simons_core``:

- ``schemas.py``
- ``interfaces.py``
- ``calendar.py``
- ``logging.py``

Design goals
------------
- Support both likely import layouts:
    1. ``simons_core.*``
    2. ``simons_smallcap_swing.simons_core.*``
  with a final plain-module fallback for local development.
- Provide small, deterministic, hand-verifiable datasets.
- Expose concrete dummy implementations for the abstract contracts in
  ``interfaces.py`` so interface tests can focus on contract behavior rather
  than boilerplate.
- Keep fixtures explicit, isolated and composable.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import io
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------


def _import_first(*candidates: str):
    """Import the first module path that resolves."""
    last_exc: BaseException | None = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except BaseException as exc:  # pragma: no cover - only hit on missing layouts
            last_exc = exc
    joined = ", ".join(candidates)
    raise ImportError(f"Could not import any of: {joined}") from last_exc


def _try_import_first(*candidates: str):
    """Best-effort import. Returns ``None`` if nothing resolves."""
    for name in candidates:
        try:
            return importlib.import_module(name)
        except BaseException:
            continue
    return None


@pytest.fixture(scope="session")
def schemas_mod():
    return _import_first(
        "simons_core.schemas",
        "simons_smallcap_swing.simons_core.schemas",
        "schemas",
    )


@pytest.fixture(scope="session")
def interfaces_mod():
    return _import_first(
        "simons_core.interfaces",
        "simons_smallcap_swing.simons_core.interfaces",
        "interfaces",
    )


@pytest.fixture(scope="session")
def logging_mod():
    return _import_first(
        "simons_core.logging",
        "simons_smallcap_swing.simons_core.logging",
        "logging",
    )


@pytest.fixture(scope="session")
def calendar_mod():
    module = _try_import_first(
        "simons_core.calendar",
        "simons_smallcap_swing.simons_core.calendar",
        "calendar",
    )
    if module is None:
        pytest.skip(
            "calendar module unavailable or optional dependency "
            "'exchange_calendars' is not installed in this runtime"
        )
    return module


# ---------------------------------------------------------------------------
# Pytest configuration / generic helpers
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "simons_core: tests for top-level core modules")
    config.addinivalue_line("markers", "schemas: tests for simons_core.schemas")
    config.addinivalue_line("markers", "interfaces: tests for simons_core.interfaces")
    config.addinivalue_line("markers", "calendar: tests for simons_core.calendar")
    config.addinivalue_line("markers", "logging: tests for simons_core.logging")
    config.addinivalue_line("markers", "optional_dependency: tests requiring optional extras")
    config.addinivalue_line("markers", "slow: slower integration / runtime tests")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Best-effort repository root."""
    here = Path(__file__).resolve()
    for root in [here.parent, *here.parents]:
        if (root / "tests").exists() or (root / "pyproject.toml").exists() or (root / ".git").exists():
            return root
    return here.parent


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260315)


@pytest.fixture(scope="session")
def assert_frame_equal() -> Callable[..., None]:
    """Thin wrapper around :func:`pandas.testing.assert_frame_equal`."""

    def _assert(left: pd.DataFrame, right: pd.DataFrame, **kwargs: Any) -> None:
        defaults = {"check_like": False, "check_dtype": True}
        defaults.update(kwargs)
        pd.testing.assert_frame_equal(left, right, **defaults)

    return _assert


@pytest.fixture(scope="session")
def assert_series_equal() -> Callable[..., None]:
    """Thin wrapper around :func:`pandas.testing.assert_series_equal`."""

    def _assert(left: pd.Series, right: pd.Series, **kwargs: Any) -> None:
        defaults = {"check_names": True, "check_dtype": True}
        defaults.update(kwargs)
        pd.testing.assert_series_equal(left, right, **defaults)

    return _assert


# ---------------------------------------------------------------------------
# Canonical dates / timestamps
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def asof_date() -> pd.Timestamp:
    return pd.Timestamp("2024-01-05")


@pytest.fixture(scope="session")
def asof_date_str() -> str:
    return "2024-01-05"


@pytest.fixture(scope="session")
def utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@pytest.fixture(scope="session")
def business_index() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02", periods=5, freq="B")


@pytest.fixture(scope="session")
def market_timestamp_open_ny() -> pd.Timestamp:
    return pd.Timestamp("2024-07-03 09:30:00", tz="America/New_York")


@pytest.fixture(scope="session")
def market_timestamp_mid_ny() -> pd.Timestamp:
    return pd.Timestamp("2024-07-03 12:15:00", tz="America/New_York")


@pytest.fixture(scope="session")
def market_timestamp_after_close_ny() -> pd.Timestamp:
    return pd.Timestamp("2024-07-03 17:30:00", tz="America/New_York")


# ---------------------------------------------------------------------------
# schemas.py fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def simple_schema(schemas_mod):
    return schemas_mod.DataSchema(
        name="test.simple",
        version="v1",
        columns=(
            schemas_mod.ColumnSpec("date", "datetime", nullable=False, required=True),
            schemas_mod.ColumnSpec("symbol", "string", nullable=False, required=True),
            schemas_mod.ColumnSpec("value", "float", nullable=False, required=True, min_value=0.0),
            schemas_mod.ColumnSpec("flag", "bool", nullable=True, required=False),
        ),
        primary_key=("date", "symbol"),
        allow_extra_columns=False,
        date_column="date",
        require_monotonic_date=True,
        allow_empty=False,
        description="Tiny canonical schema for unit tests.",
    )


@pytest.fixture(scope="session")
def pattern_schema(schemas_mod):
    return schemas_mod.DataSchema(
        name="test.patterns",
        version="v1",
        columns=(
            schemas_mod.ColumnSpec("date", "datetime", nullable=False, required=True),
            schemas_mod.ColumnSpec("symbol", "string", nullable=False, required=True),
        ),
        column_patterns=(
            schemas_mod.ColumnPatternSpec(
                pattern=r"feature_[A-Za-z0-9_]+",
                dtype="numeric",
                nullable=False,
                min_count=2,
            ),
        ),
        primary_key=("date", "symbol"),
        allow_extra_columns=False,
        date_column="date",
        require_monotonic_date=False,
        allow_empty=False,
        description="Schema with dynamic feature_* columns.",
    )


@pytest.fixture(scope="session")
def valid_simple_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "symbol": ["AAA", "BBB", "CCC"],
            "value": [1.0, 2.5, 3.25],
            "flag": [True, False, None],
        }
    )


@pytest.fixture(scope="session")
def valid_pattern_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "symbol": ["AAA", "BBB", "CCC"],
            "feature_mom_5d": [0.10, -0.05, 0.02],
            "feature_vol_20d": [1.2, 0.8, 0.9],
        }
    )


@pytest.fixture(scope="session")
def invalid_simple_frame_missing_column() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["AAA", "BBB"],
            # value column intentionally missing
        }
    )


@pytest.fixture(scope="session")
def invalid_simple_frame_duplicate_pk() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "symbol": ["AAA", "AAA"],
            "value": [1.0, 1.1],
            "flag": [True, False],
        }
    )


@pytest.fixture(scope="session")
def invalid_simple_frame_future_date() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-02-01"]),
            "symbol": ["AAA", "BBB"],
            "value": [1.0, 2.0],
            "flag": [True, False],
        }
    )


@pytest.fixture(scope="session")
def canonical_prices_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                ]
            ),
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "open": [10.0, 20.0, 10.5, 19.8],
            "high": [10.8, 20.5, 10.9, 20.1],
            "low": [9.8, 19.7, 10.3, 19.5],
            "close": [10.5, 20.2, 10.7, 19.9],
            "volume": [1_000_000, 2_000_000, 1_100_000, 2_100_000],
        }
    )


@pytest.fixture(scope="session")
def invalid_prices_frame_high_lt_low(canonical_prices_frame: pd.DataFrame) -> pd.DataFrame:
    df = canonical_prices_frame.copy()
    df.loc[0, "high"] = 9.0
    return df


@pytest.fixture(scope="session")
def canonical_features_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "symbol": ["AAA", "BBB", "AAA"],
            "feature_mom_5d": [0.11, -0.03, 0.08],
            "feature_vol_20d": [1.2, 0.9, 1.1],
        }
    )


@pytest.fixture(scope="session")
def canonical_labels_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "symbol": ["AAA", "BBB", "AAA"],
            "horizon_days": [5, 5, 5],
            "label": [0.02, -0.01, 0.03],
        }
    )


@pytest.fixture(scope="session")
def canonical_predictions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "symbol": ["AAA", "BBB", "AAA"],
            "score": [0.7, 0.2, 0.6],
        }
    )


# ---------------------------------------------------------------------------
# interfaces.py fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def run_context(interfaces_mod, asof_date: pd.Timestamp):
    return interfaces_mod.RunContext(
        run_id="run_unit_test_001",
        asof_date=asof_date,
        seed=7,
        config_hash="cfg_hash_abc123",
        pipeline_version="1.0.0",
        market="US_EQ",
        metadata={"suite": "simons_core"},
    )


@pytest.fixture(scope="session")
def raw_price_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
            "symbol": ["AAA", "AAA", "AAA"],
            "close": [10.0, 10.2, 10.4],
            "volume": [1_000_000, 1_100_000, 1_050_000],
        }
    )


@pytest.fixture(scope="session")
def features_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_mom_2d": [0.02, 0.01, 0.03],
            "feature_volume_z": [0.1, 0.2, -0.1],
        },
        index=pd.Index(["AAA_2024-01-03", "AAA_2024-01-04", "AAA_2024-01-05"], name="row_id"),
    )


@pytest.fixture(scope="session")
def labels_series() -> pd.Series:
    return pd.Series(
        [0.05, -0.02, 0.04],
        index=pd.Index(["AAA_2024-01-03", "AAA_2024-01-04", "AAA_2024-01-05"], name="row_id"),
        name="label",
    )


@pytest.fixture(scope="session")
def predictions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "score": [0.8, 0.1, 0.7],
        },
        index=pd.Index(["AAA_2024-01-03", "AAA_2024-01-04", "AAA_2024-01-05"], name="row_id"),
    )


@pytest.fixture(scope="session")
def weights_series() -> pd.Series:
    return pd.Series(
        [0.4, 0.35, 0.25],
        index=pd.Index(["AAA", "BBB", "CCC"], name="symbol"),
        name="weight",
    )


@pytest.fixture(scope="session")
def price_data(interfaces_mod, asof_date: pd.Timestamp, raw_price_frame: pd.DataFrame):
    return interfaces_mod.PriceData(
        asof_date=asof_date,
        prices=raw_price_frame,
        symbols=("AAA",),
        timestamp_column="date",
        metadata={"source": "fixture"},
    )


@pytest.fixture(scope="session")
def feature_matrix(interfaces_mod, asof_date: pd.Timestamp, features_frame: pd.DataFrame):
    return interfaces_mod.FeatureMatrix(
        asof_date=asof_date,
        features=features_frame,
        feature_names=("feature_mom_2d", "feature_volume_z"),
        metadata={"builder": "dummy"},
    )


@pytest.fixture(scope="session")
def labels_obj(interfaces_mod, asof_date: pd.Timestamp, labels_series: pd.Series):
    return interfaces_mod.Labels(
        asof_date=asof_date,
        values=labels_series,
        name="label",
        metadata={"horizon_days": 5},
    )


@pytest.fixture(scope="session")
def model_artifact(interfaces_mod):
    return interfaces_mod.ModelArtifact(
        model_ref="dummy.linear.v1",
        artifact_path="models/dummy_linear_v1.pkl",
        params={"alpha": 0.1},
        metrics={"train_rmse": 0.12},
        metadata={"trainer": "DummyTrainer"},
    )


@pytest.fixture(scope="session")
def prediction_frame(interfaces_mod, asof_date: pd.Timestamp, predictions_frame: pd.DataFrame):
    return interfaces_mod.PredictionFrame(
        asof_date=asof_date,
        predictions=predictions_frame,
        score_columns=("score",),
        metadata={"model_ref": "dummy.linear.v1"},
    )


@pytest.fixture(scope="session")
def portfolio_decision(interfaces_mod, asof_date: pd.Timestamp, weights_series: pd.Series):
    return interfaces_mod.PortfolioDecision(
        timestamp=asof_date,
        weights=weights_series,
        turnover_l1=0.15,
        notional_exposure=1.0,
        constraints_active=("gross<=1.0",),
        metadata={"engine": "DummyPortfolioEngine"},
    )


@pytest.fixture(scope="session")
def gate_result(interfaces_mod):
    return interfaces_mod.GateResult(
        passed=True,
        score=0.91,
        threshold=0.8,
        reasons=("all_good",),
        metrics={"sharpe": 1.6, "mdd": 0.08},
        metadata={"validator": "DummyValidator"},
    )


@pytest.fixture(scope="session")
def dummy_provider_cls(interfaces_mod):
    class DummyProvider(interfaces_mod.DataProvider):
        contract_version = interfaces_mod.CONTRACT_VERSION

        def fetch(self, asof_date: pd.Timestamp, context):
            prices = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
                    "symbol": ["AAA", "AAA", "AAA"],
                    "close": [10.0, 10.2, 10.4],
                    "volume": [1_000_000, 1_100_000, 1_050_000],
                }
            )
            return interfaces_mod.PriceData(
                asof_date=asof_date,
                prices=prices,
                symbols=("AAA",),
                timestamp_column="date",
                metadata={"provider": "dummy"},
            )

    return DummyProvider


@pytest.fixture(scope="session")
def dummy_feature_builder_cls(interfaces_mod):
    class DummyFeatureBuilder(interfaces_mod.FeatureBuilder):
        contract_version = interfaces_mod.CONTRACT_VERSION

        def build(self, raw_data, context):
            idx = pd.Index(
                [f"AAA_{d.date().isoformat()}" for d in raw_data.prices["date"]],
                name="row_id",
            )
            frame = pd.DataFrame(
                {
                    "feature_mom_2d": [0.02, 0.01, 0.03],
                    "feature_volume_z": [0.1, 0.2, -0.1],
                },
                index=idx,
            )
            return interfaces_mod.FeatureMatrix(
                asof_date=raw_data.asof_date,
                features=frame,
                feature_names=tuple(frame.columns),
                metadata={"builder": "dummy"},
            )

    return DummyFeatureBuilder


@pytest.fixture(scope="session")
def dummy_trainer_cls(interfaces_mod):
    class DummyTrainer(interfaces_mod.ModelTrainer):
        contract_version = interfaces_mod.CONTRACT_VERSION

        def fit(self, features, labels, context):
            return interfaces_mod.ModelArtifact(
                model_ref="dummy.linear.v1",
                artifact_path="models/dummy_linear_v1.pkl",
                params={"alpha": 0.1},
                metrics={"train_rmse": 0.12},
                metadata={"trainer": "dummy"},
            )

        def predict(self, features, model, context):
            frame = pd.DataFrame(
                {"score": np.linspace(0.2, 0.8, num=len(features.features))},
                index=features.features.index,
            )
            return interfaces_mod.PredictionFrame(
                asof_date=features.asof_date,
                predictions=frame,
                score_columns=("score",),
                metadata={"model_ref": model.model_ref},
            )

    return DummyTrainer


@pytest.fixture(scope="session")
def dummy_portfolio_engine_cls(interfaces_mod):
    class DummyPortfolioEngine(interfaces_mod.PortfolioEngine):
        contract_version = interfaces_mod.CONTRACT_VERSION

        def rebalance(self, signals, context):
            raw = signals.predictions["score"].astype(float)
            denom = float(raw.abs().sum())
            weights = raw / denom if denom > 0 else raw
            weights.name = "weight"
            return interfaces_mod.PortfolioDecision(
                timestamp=signals.asof_date,
                weights=weights,
                turnover_l1=float(weights.abs().sum()),
                notional_exposure=float(weights.abs().sum()),
                constraints_active=("unit_test_constraint",),
                metadata={"engine": "dummy"},
            )

    return DummyPortfolioEngine


@pytest.fixture(scope="session")
def dummy_validator_cls(interfaces_mod):
    class DummyValidator(interfaces_mod.Validator):
        contract_version = interfaces_mod.CONTRACT_VERSION

        def validate(self, artifact, context):
            return interfaces_mod.GateResult(
                passed=True,
                score=0.95,
                threshold=0.8,
                reasons=("ok",),
                metrics={"score": 0.95},
                metadata={"artifact_type": type(artifact).__name__},
            )

    return DummyValidator


@pytest.fixture
def dummy_provider(dummy_provider_cls):
    return dummy_provider_cls()


@pytest.fixture
def dummy_feature_builder(dummy_feature_builder_cls):
    return dummy_feature_builder_cls()


@pytest.fixture
def dummy_trainer(dummy_trainer_cls):
    return dummy_trainer_cls()


@pytest.fixture
def dummy_portfolio_engine(dummy_portfolio_engine_cls):
    return dummy_portfolio_engine_cls()


@pytest.fixture
def dummy_validator(dummy_validator_cls):
    return dummy_validator_cls()


# ---------------------------------------------------------------------------
# calendar.py fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def calendar_date_range() -> tuple[str, str]:
    # Includes an early-close period around US Independence Day.
    return ("2024-07-01", "2024-07-10")


@pytest.fixture
def market_calendar(calendar_mod, calendar_date_range: tuple[str, str]):
    start_date, end_date = calendar_date_range
    return calendar_mod.load_market_calendar(
        market="US_EQ",
        start_date=start_date,
        end_date=end_date,
    )


# ---------------------------------------------------------------------------
# logging.py fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_sink(logging_mod):
    return logging_mod.MemorySink()


@pytest.fixture
def json_stream() -> io.StringIO:
    return io.StringIO()


@pytest.fixture
def stream_sink(logging_mod, json_stream: io.StringIO):
    return logging_mod.StreamJsonSink(stream=json_stream, ensure_ascii=False, flush=True, sort_keys=True)


@pytest.fixture
def base_logger(logging_mod, memory_sink):
    return logging_mod.get_logger(
        "simons_core.tests",
        sink=memory_sink,
        min_level="DEBUG",
        base_fields={"module": "tests"},
        require_run_id=False,
        on_emit_error="raise",
    )


@pytest.fixture
def contextual_logger(logging_mod, memory_sink, run_context):
    return logging_mod.get_logger(
        "simons_core.tests",
        run_context=run_context,
        sink=memory_sink,
        min_level="DEBUG",
        base_fields={"module": "tests"},
        require_run_id=True,
        on_emit_error="raise",
    )


@pytest.fixture(scope="session")
def sensitive_payload() -> dict[str, Any]:
    return {
        "token": "SECRET_TOKEN_123",
        "nested": {
            "password": "super-secret",
            "safe": 7,
        },
        "api_key": "ABC123",
        "notes": "visible",
    }


@pytest.fixture(scope="session")
def serializable_payload() -> dict[str, Any]:
    return {
        "path": Path("models/dummy.pkl"),
        "timestamp": pd.Timestamp("2024-01-05 12:00:00"),
        "date": pd.Timestamp("2024-01-05").date(),
        "number": 1.23,
        "items": [1, 2, 3],
    }


@dataclass
class DummyExceptionPayload:
    code: int
    text: str


@pytest.fixture(scope="session")
def dummy_exception() -> RuntimeError:
    exc = RuntimeError("unit-test-boom")
    setattr(exc, "error_code", "UNIT_TEST_BOOM")
    return exc


@pytest.fixture(scope="session")
def dummy_exception_payload() -> DummyExceptionPayload:
    return DummyExceptionPayload(code=7, text="boom")
