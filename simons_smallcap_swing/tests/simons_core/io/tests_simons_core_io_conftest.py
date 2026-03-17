from __future__ import annotations

"""
Shared pytest fixtures for ``tests/simons_core/io``.

Design goals
------------
- Provide deterministic, isolated filesystem fixtures for cache / parquet /
  path tests.
- Support both likely import layouts:
    1. ``simons_core.io.*``
    2. ``simons_smallcap_swing.simons_core.io.*``
  and a final plain-module fallback for local development.
- Expose a lightweight fake schema registry so ``parquet_store`` tests can
  exercise write/read/validate flows without depending on the real
  ``simons_core.schemas`` implementation.
- Keep fixtures small, explicit and hand-verifiable.
"""

from dataclasses import dataclass
import importlib
import tempfile
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping

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


@pytest.fixture(scope="session")
def cache_mod():
    return _import_first(
        "simons_core.io.cache",
        "simons_smallcap_swing.simons_core.io.cache",
        "cache",
    )


@pytest.fixture(scope="session")
def parquet_store_mod():
    return _import_first(
        "simons_core.io.parquet_store",
        "simons_smallcap_swing.simons_core.io.parquet_store",
        "parquet_store",
    )


@pytest.fixture(scope="session")
def paths_mod():
    return _import_first(
        "simons_core.io.paths",
        "simons_smallcap_swing.simons_core.io.paths",
        "paths",
    )


# ---------------------------------------------------------------------------
# Pytest configuration / generic helpers
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "io: tests for simons_core.io modules")
    config.addinivalue_line(
        "markers",
        "parquet: tests requiring pandas parquet read/write support",
    )
    config.addinivalue_line(
        "markers",
        "filesystem: tests that touch the filesystem and temporary directories",
    )
    config.addinivalue_line("markers", "slow: slower IO / integrity / corruption tests")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """
    Best-effort repository root for path-based checks.

    Walk upward from this file location until a plausible root is found.
    """
    here = Path(__file__).resolve()
    candidates = [here.parent, *here.parents]
    for root in candidates:
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
def parquet_engine_available() -> bool:
    """
    Capability probe for pandas parquet support in this runtime.

    We use a real write/read roundtrip on a tiny temporary file instead of
    only checking attribute presence, because some runtimes expose the methods
    but lack a usable engine backend.
    """
    frame = pd.DataFrame({"x": [1], "y": ["a"]})
    with tempfile.TemporaryDirectory(prefix="simons_io_parquet_probe_") as tmp:
        path = Path(tmp) / "probe.parquet"
        try:
            frame.to_parquet(path, index=False)
            observed = pd.read_parquet(path)
        except Exception:
            return False
    return observed.equals(frame)


@pytest.fixture
def require_parquet(parquet_engine_available: bool) -> None:
    if not parquet_engine_available:
        pytest.skip("Parquet engine unavailable in this runtime")


# ---------------------------------------------------------------------------
# Filesystem roots / environment roots
# ---------------------------------------------------------------------------


@pytest.fixture
def io_root(tmp_path: Path) -> Path:
    """
    Isolated root for a single IO test.

    Shape:
        io_root/
            cache/
            datasets/
            env_local/
            env_test/
    """
    root = tmp_path / "simons_core_io"
    root.mkdir(parents=True, exist_ok=True)
    (root / "cache").mkdir()
    (root / "datasets").mkdir()
    (root / "env_local").mkdir()
    (root / "env_test").mkdir()
    return root


@pytest.fixture
def datasets_root(io_root: Path) -> Path:
    path = io_root / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def cache_root(io_root: Path) -> Path:
    path = io_root / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def env_roots_file(io_root: Path) -> dict[str, str]:
    """
    Deterministic roots for ``paths.py``.

    ``local`` and ``test`` use real file URIs so path tests can exercise
    confinement and directory creation. ``dev`` and ``prod`` keep canonical
    S3-style roots because some tests will want to verify backend parsing
    without touching a real filesystem path.
    """
    return {
        "local": (io_root / "env_local").resolve().as_uri(),
        "test": (io_root / "env_test").resolve().as_uri(),
        "dev": "s3://simons-dev",
        "prod": "s3://simons-prod",
    }


@pytest.fixture
def allowed_local_roots(datasets_root: Path) -> tuple[Path, ...]:
    return (datasets_root.resolve(),)


# ---------------------------------------------------------------------------
# Cache fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_local_cache(cache_mod, cache_root: Path):
    """
    Factory returning a fresh :class:`LocalCache` rooted under ``cache_root``.

    The default byte budget is intentionally small enough that eviction tests
    can be written without creating huge blobs.
    """

    def _make(
        name: str = "default",
        *,
        max_bytes: int = 2 * 1024 * 1024,
        validate_checksum: bool = True,
        quarantine_corrupt: bool = True,
        durability_fsync_dir: bool = False,
    ):
        root = cache_root / name
        root.mkdir(parents=True, exist_ok=True)
        return cache_mod.LocalCache(
            root,
            max_bytes=max_bytes,
            validate_checksum=validate_checksum,
            quarantine_corrupt=quarantine_corrupt,
            durability_fsync_dir=durability_fsync_dir,
        )

    return _make


@pytest.fixture
def local_cache(make_local_cache):
    return make_local_cache()


@pytest.fixture(scope="session")
def sample_cache_namespace() -> str:
    return "features"


@pytest.fixture(scope="session")
def sample_cache_fn_id() -> str:
    return "build_features.daily_momentum"


@pytest.fixture(scope="session")
def sample_cache_params() -> dict[str, Any]:
    return {
        "window": 20,
        "universe": "smallcap",
        "winsorize": True,
        "groups": ["sector", "exchange"],
    }


@pytest.fixture(scope="session")
def sample_cache_context() -> dict[str, Any]:
    return {
        "asof_date": "2024-03-15",
        "schema_version": "v1",
        "feature_set": "baseline",
    }


@pytest.fixture
def sample_cache_key(cache_mod, sample_cache_namespace: str, sample_cache_fn_id: str,
                     sample_cache_params: Mapping[str, Any], sample_cache_context: Mapping[str, Any]) -> str:
    return cache_mod.cache_key(
        namespace=sample_cache_namespace,
        fn_id=sample_cache_fn_id,
        params=sample_cache_params,
        context=sample_cache_context,
    )


@pytest.fixture(scope="session")
def sample_cache_value_dict() -> dict[str, Any]:
    return {
        "alpha": 0.0123,
        "beta": -0.4,
        "n_obs": 128,
        "tags": ["research", "baseline"],
    }


@pytest.fixture(scope="session")
def sample_cache_value_bytes() -> bytes:
    return b"simons-cache-payload"


@pytest.fixture(scope="session")
def sample_cache_value_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "symbol": ["AAA", "AAA", "BBB"],
            "score": [0.10, 0.15, -0.05],
        }
    )


@pytest.fixture(scope="session")
def sample_cache_metadata() -> dict[str, Any]:
    return {
        "producer": "research.pipeline",
        "tags": ["research", "daily", "smallcap"],
        "user_metadata": {"owner": "quant", "priority": "high"},
        "producer_runtime_ms": 42.5,
    }


# ---------------------------------------------------------------------------
# Paths fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_descriptor() -> dict[str, Any]:
    return {
        "artifact_class": "data",
        "domain": "prices",
        "entity": "adjusted",
        "date": "2024-01-03",
        "version": "v1",
        "partition": "exchange=nasdaq",
    }


@pytest.fixture
def sample_data_locator(paths_mod, env_roots_file: Mapping[str, str]):
    return paths_mod.build_locator(
        artifact_class="data",
        env="local",
        domain="prices",
        entity="adjusted",
        version="v1",
        date="2024-01-03",
        partition="exchange=nasdaq",
        artifact_name="part-000.parquet",
        env_roots=env_roots_file,
    )


@pytest.fixture
def sample_cache_locator(paths_mod, env_roots_file: Mapping[str, str]):
    digest = paths_mod.canonical_descriptor_digest(
        {"namespace": "features", "fn_id": "build_features", "window": 20}
    )
    return paths_mod.build_locator(
        artifact_class="cache",
        env="local",
        domain="features",
        entity="daily_momentum",
        version="v1",
        digest=digest,
        artifact_name="payload.cache",
        env_roots=env_roots_file,
    )


@pytest.fixture
def sample_tmp_locator(paths_mod, env_roots_file: Mapping[str, str]):
    return paths_mod.build_tmp_locator(
        env="local",
        scope="research",
        run_id="run_test_case_001",
        artifact_name="scratch.json",
        date="2024-01-03",
        env_roots=env_roots_file,
    )


# ---------------------------------------------------------------------------
# Parquet fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def schema_ref_prices() -> str:
    return "prices.adjusted"


@pytest.fixture(scope="session")
def prices_frame_unsorted() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-03",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-02",
                    "2024-01-04",
                    "2024-01-04",
                ]
            ),
            "symbol": ["BBB", "AAA", "AAA", "BBB", "AAA", "BBB"],
            "open": [20.0, 10.0, 10.5, 19.5, 10.8, 20.2],
            "high": [21.0, 10.8, 11.0, 20.0, 11.2, 20.6],
            "low": [19.5, 9.8, 10.2, 19.0, 10.6, 19.9],
            "close": [20.7, 10.5, 10.9, 19.8, 11.0, 20.1],
            "volume": [2_000_000, 1_000_000, 1_200_000, 2_100_000, 1_250_000, 2_050_000],
            "exchange": ["NYSE", "NASDAQ", "NASDAQ", "NYSE", "NASDAQ", "NYSE"],
            "sector": ["Industrials", "Tech", "Tech", "Industrials", "Tech", "Industrials"],
        }
    )


@pytest.fixture(scope="session")
def prices_frame_sorted_expected(prices_frame_unsorted: pd.DataFrame) -> pd.DataFrame:
    return prices_frame_unsorted.sort_values(["date", "symbol"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def partition_cols() -> tuple[str, ...]:
    return ("exchange",)


@pytest.fixture(scope="session")
def projection_columns() -> tuple[str, ...]:
    return ("date", "symbol", "close")


@pytest.fixture(scope="session")
def simple_filters() -> list[tuple[str, str, Any]]:
    return [("exchange", "==", "NASDAQ")]


@pytest.fixture(scope="session")
def stored_schema_meta() -> dict[str, Any]:
    return {
        "schema_ref": "prices.adjusted",
        "schema_version": "v1",
        "columns": [
            {"name": "date", "dtype": "datetime64[ns]", "nullable": False, "required": True},
            {"name": "symbol", "dtype": "object", "nullable": False, "required": True},
            {"name": "close", "dtype": "float64", "nullable": False, "required": True},
        ],
        "primary_key": ["date", "symbol"],
        "allow_extra_columns": False,
    }


@pytest.fixture(scope="session")
def widened_expected_schema_meta() -> dict[str, Any]:
    return {
        "schema_ref": "prices.adjusted",
        "schema_version": "v2",
        "columns": [
            {"name": "date", "dtype": "datetime64[ns]", "nullable": False, "required": True},
            {"name": "symbol", "dtype": "string", "nullable": False, "required": True},
            {"name": "close", "dtype": "float64", "nullable": False, "required": True},
        ],
        "primary_key": ["date", "symbol"],
        "allow_extra_columns": False,
    }


@pytest.fixture(scope="session")
def breaking_expected_schema_meta() -> dict[str, Any]:
    return {
        "schema_ref": "prices.adjusted",
        "schema_version": "v2",
        "columns": [
            {"name": "date", "dtype": "datetime64[ns]", "nullable": False, "required": True},
            {"name": "symbol", "dtype": "int64", "nullable": False, "required": True},
            {"name": "close", "dtype": "float64", "nullable": False, "required": True},
            {"name": "volume", "dtype": "int64", "nullable": False, "required": True},
        ],
        "primary_key": ["date", "symbol"],
        "allow_extra_columns": False,
    }


# ---------------------------------------------------------------------------
# Fake schema registry for parquet_store tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeColumnSpec:
    name: str
    dtype: str
    nullable: bool = False
    required: bool = True


@dataclass(frozen=True)
class _FakeSchema:
    name: str
    version: str
    columns: tuple[_FakeColumnSpec, ...]
    primary_key: tuple[str, ...] = ()
    allow_extra_columns: bool = False


@dataclass(frozen=True)
class _FakeValidationResult:
    passed: bool
    issues: tuple[str, ...] = ()


class _FakeSchemaValidationError(ValueError):
    pass


@pytest.fixture
def install_fake_schema_registry(monkeypatch: pytest.MonkeyPatch, parquet_store_mod):
    """
    Patch ``parquet_store`` to use a tiny in-memory schema registry.

    This keeps IO tests independent from the real schema module while still
    exercising the write/read/validate contract at a meaningful level.
    """

    def _install(
        *,
        schema_ref: str = "prices.adjusted",
        columns: Iterable[_FakeColumnSpec] | None = None,
        primary_key: tuple[str, ...] = ("date", "symbol"),
        allow_extra_columns: bool = False,
    ) -> _FakeSchema:
        schema_columns = tuple(
            columns
            if columns is not None
            else (
                _FakeColumnSpec("date", "datetime64[ns]"),
                _FakeColumnSpec("symbol", "object"),
                _FakeColumnSpec("open", "float64"),
                _FakeColumnSpec("high", "float64"),
                _FakeColumnSpec("low", "float64"),
                _FakeColumnSpec("close", "float64"),
                _FakeColumnSpec("volume", "int64"),
                _FakeColumnSpec("exchange", "object"),
                _FakeColumnSpec("sector", "object"),
            )
        )
        schema = _FakeSchema(
            name=schema_ref,
            version="v1",
            columns=schema_columns,
            primary_key=primary_key,
            allow_extra_columns=allow_extra_columns,
        )
        registry = {schema_ref: schema, schema.name: schema}

        def _get_schema(name: str) -> _FakeSchema:
            try:
                return registry[name]
            except KeyError as exc:  # pragma: no cover - defensive
                raise KeyError(f"Unknown fake schema ref: {name}") from exc

        def _validate_schema(df: pd.DataFrame, ref: str):
            sch = _get_schema(ref)
            issues: list[str] = []

            expected_names = {col.name for col in sch.columns}
            required_names = {col.name for col in sch.columns if col.required}
            actual_names = {str(col) for col in df.columns}

            missing = sorted(required_names - actual_names)
            if missing:
                issues.append(f"Missing required columns: {missing}")

            if not sch.allow_extra_columns:
                extra = sorted(actual_names - expected_names)
                if extra:
                    issues.append(f"Unexpected columns: {extra}")

            for col in sch.columns:
                if col.name not in df.columns:
                    continue
                series = df[col.name]
                if not col.nullable and bool(series.isna().any()):
                    issues.append(f"Column {col.name} contains nulls")

            if sch.primary_key and all(col in df.columns for col in sch.primary_key):
                if bool(df.duplicated(subset=list(sch.primary_key)).any()):
                    issues.append(f"Primary key not unique: {sch.primary_key!r}")

            return _FakeValidationResult(passed=not issues, issues=tuple(issues))

        def _format_validation_issues(result: _FakeValidationResult) -> str:
            return "; ".join(result.issues)

        def _assert_schema(df: pd.DataFrame, ref: str) -> None:
            result = _validate_schema(df, ref)
            if not result.passed:
                raise _FakeSchemaValidationError(_format_validation_issues(result))

        monkeypatch.setattr(parquet_store_mod, "DataSchema", _FakeSchema, raising=False)
        monkeypatch.setattr(parquet_store_mod, "CoreSchemaValidationError", _FakeSchemaValidationError, raising=False)
        monkeypatch.setattr(parquet_store_mod, "get_schema", _get_schema, raising=False)
        monkeypatch.setattr(parquet_store_mod, "validate_schema", _validate_schema, raising=False)
        monkeypatch.setattr(parquet_store_mod, "format_validation_issues", _format_validation_issues, raising=False)
        monkeypatch.setattr(parquet_store_mod, "assert_schema", _assert_schema, raising=False)
        return schema

    return _install
