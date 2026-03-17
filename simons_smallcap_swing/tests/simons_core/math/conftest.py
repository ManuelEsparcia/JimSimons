from __future__ import annotations

"""
Shared pytest fixtures for ``tests/simons_core/math``.

Design goals
------------
- Provide deterministic, low-noise numeric fixtures for robust/stats/
  optimization tests.
- Keep fixtures small and hand-verifiable.
- Avoid binding tests to a single repository import layout by supporting the two
  most likely package roots:
    1. ``simons_core.*``
    2. ``simons_smallcap_swing.simons_core.*``
- Expose a tiny set of helpers commonly needed across the math suite.
"""

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
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
        except BaseException as exc:  # pragma: no cover - only exercised on bad layouts
            last_exc = exc
    joined = ", ".join(candidates)
    raise ImportError(f"Could not import any of: {joined}") from last_exc


@pytest.fixture(scope="session")
def robust_mod():
    return _import_first(
        "simons_core.math.robust",
        "simons_smallcap_swing.simons_core.math.robust",
    )


@pytest.fixture(scope="session")
def stats_mod():
    return _import_first(
        "simons_core.math.stats",
        "simons_smallcap_swing.simons_core.math.stats",
    )


@pytest.fixture(scope="session")
def optimization_mod():
    return _import_first(
        "simons_core.math.optimization",
        "simons_smallcap_swing.simons_core.math.optimization",
    )


@pytest.fixture(scope="session")
def cvxpy_available(optimization_mod) -> bool:
    try:
        optimization_mod.available_solvers()
        optimization_mod._require_cvxpy()  # intentional capability probe for solver tests
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Pytest configuration / helpers
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "math: tests for simons_core.math modules")
    config.addinivalue_line(
        "markers",
        "solver: tests requiring cvxpy and at least one available convex solver backend",
    )
    config.addinivalue_line("markers", "slow: numerically heavier tests")


@pytest.fixture(scope="session")
def assert_allclose() -> Callable[..., None]:
    """Thin wrapper around ``numpy.testing.assert_allclose`` for consistency."""

    def _assert(
        actual: Any,
        expected: Any,
        *,
        rtol: float = 1e-12,
        atol: float = 1e-12,
        equal_nan: bool = True,
    ) -> None:
        np.testing.assert_allclose(
            np.asarray(actual, dtype=float),
            np.asarray(expected, dtype=float),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )

    return _assert


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """
    Best-effort repository root for path-based checks.

    Walks upward from this file location until a plausible root is found.
    This is intentionally conservative and only used by tests that need a real
    filesystem anchor.
    """
    here = Path(__file__).resolve()
    candidates = [here.parent, *here.parents]
    for root in candidates:
        if (root / "tests").exists() or (root / "pyproject.toml").exists() or (root / ".git").exists():
            return root
    return here.parent


# ---------------------------------------------------------------------------
# Randomness / base arrays
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260315)


@pytest.fixture(scope="session")
def clean_vector() -> np.ndarray:
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)


@pytest.fixture(scope="session")
def clean_even_vector() -> np.ndarray:
    return np.array([1.0, 2.0, 3.0, 4.0], dtype=float)


@pytest.fixture(scope="session")
def vector_with_nonfinite() -> np.ndarray:
    return np.array([1.0, np.nan, 2.0, np.inf, -np.inf, 3.0], dtype=float)


@pytest.fixture(scope="session")
def all_nonfinite_vector() -> np.ndarray:
    return np.array([np.nan, np.inf, -np.inf], dtype=float)


@pytest.fixture(scope="session")
def constant_vector() -> np.ndarray:
    return np.array([7.5, 7.5, 7.5, 7.5, 7.5], dtype=float)


@pytest.fixture(scope="session")
def signed_residuals() -> np.ndarray:
    return np.array([-10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0], dtype=float)


@pytest.fixture(scope="session")
def outlier_vector() -> np.ndarray:
    return np.array([10.0, 10.2, 10.1, 9.9, 10.0, 100.0], dtype=float)


@pytest.fixture(scope="session")
def tiny_vector() -> np.ndarray:
    return np.array([2.0, 100.0, 3.0], dtype=float)


@pytest.fixture(scope="session")
def symmetric_vector() -> np.ndarray:
    return np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=float)


# ---------------------------------------------------------------------------
# Pandas / time-series fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def business_index() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02", periods=8, freq="B", tz=None)


@pytest.fixture(scope="session")
def return_series(business_index: pd.DatetimeIndex) -> pd.Series:
    values = np.array([0.010, -0.020, 0.015, 0.000, 0.005, -0.010, 0.020, 0.010], dtype=float)
    return pd.Series(values, index=business_index, name="returns")


@pytest.fixture(scope="session")
def return_series_with_nan(business_index: pd.DatetimeIndex) -> pd.Series:
    values = np.array([0.010, np.nan, 0.015, -0.005, np.nan, 0.020, -0.010, 0.005], dtype=float)
    return pd.Series(values, index=business_index, name="returns_nan")


@pytest.fixture(scope="session")
def monotone_gain_returns() -> np.ndarray:
    return np.array([0.01, 0.01, 0.01, 0.01], dtype=float)


@pytest.fixture(scope="session")
def drawdown_returns() -> np.ndarray:
    # Equity path: 1.0 -> 1.1 -> 0.99 -> 1.0395 -> 0.93555
    return np.array([0.10, -0.10, 0.05, -0.10], dtype=float)


@pytest.fixture(scope="session")
def benchmark_scalar() -> float:
    return 0.0


@pytest.fixture(scope="session")
def benchmark_series(business_index: pd.DatetimeIndex) -> pd.Series:
    values = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], dtype=float)
    return pd.Series(values, index=business_index, name="rf")


@pytest.fixture(scope="session")
def pearson_pair_perfect_pos() -> tuple[np.ndarray, np.ndarray]:
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    y = np.array([2.0, 4.0, 6.0, 8.0], dtype=float)
    return x, y


@pytest.fixture(scope="session")
def pearson_pair_perfect_neg() -> tuple[np.ndarray, np.ndarray]:
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    y = np.array([8.0, 6.0, 4.0, 2.0], dtype=float)
    return x, y


@pytest.fixture(scope="session")
def spearman_pair_with_ties() -> tuple[np.ndarray, np.ndarray]:
    x = np.array([1.0, 1.0, 2.0, 3.0, 3.0], dtype=float)
    y = np.array([10.0, 10.0, 20.0, 30.0, 40.0], dtype=float)
    return x, y


# ---------------------------------------------------------------------------
# Optimization fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QPFixture:
    Q: np.ndarray
    c: np.ndarray
    C: np.ndarray | None = None
    d: np.ndarray | None = None
    E: np.ndarray | None = None
    f: np.ndarray | None = None
    lb: np.ndarray | None = None
    ub: np.ndarray | None = None
    w_prev: np.ndarray | None = None
    turnover_penalty: float = 0.0


@pytest.fixture(scope="session")
def simplex_input() -> np.ndarray:
    return np.array([0.2, -0.1, 2.0], dtype=float)


@pytest.fixture(scope="session")
def already_simplex_vector() -> np.ndarray:
    return np.array([0.2, 0.3, 0.5], dtype=float)


@pytest.fixture(scope="session")
def box_bounds() -> tuple[np.ndarray, np.ndarray]:
    lb = np.array([-0.5, 0.0, 0.1], dtype=float)
    ub = np.array([0.5, 0.7, 0.9], dtype=float)
    return lb, ub


@pytest.fixture(scope="session")
def psd_matrix() -> np.ndarray:
    a = np.array(
        [
            [1.0, 2.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return a.T @ a


@pytest.fixture(scope="session")
def indefinite_matrix() -> np.ndarray:
    return np.array(
        [
            [1.0, 2.0],
            [2.0, -1.0],
        ],
        dtype=float,
    )


@pytest.fixture(scope="session")
def qp_unconstrained_identity() -> QPFixture:
    return QPFixture(
        Q=np.eye(2, dtype=float),
        c=np.array([-1.0, -2.0], dtype=float),
    )


@pytest.fixture(scope="session")
def qp_box_only() -> QPFixture:
    return QPFixture(
        Q=np.eye(2, dtype=float),
        c=np.array([-0.8, 0.3], dtype=float),
        lb=np.array([0.0, -0.5], dtype=float),
        ub=np.array([0.6, 0.5], dtype=float),
    )


@pytest.fixture(scope="session")
def qp_sum_to_one_box() -> QPFixture:
    return QPFixture(
        Q=np.eye(3, dtype=float),
        c=np.array([-0.4, -0.2, -0.1], dtype=float),
        E=np.array([[1.0, 1.0, 1.0]], dtype=float),
        f=np.array([1.0], dtype=float),
        lb=np.zeros(3, dtype=float),
        ub=np.ones(3, dtype=float),
        w_prev=np.array([1 / 3, 1 / 3, 1 / 3], dtype=float),
        turnover_penalty=0.01,
    )


@pytest.fixture(scope="session")
def qp_with_ineq() -> QPFixture:
    # x1 + x2 <= 1.0
    return QPFixture(
        Q=np.eye(2, dtype=float),
        c=np.array([-1.0, -1.0], dtype=float),
        C=np.array([[1.0, 1.0]], dtype=float),
        d=np.array([1.0], dtype=float),
        lb=np.array([0.0, 0.0], dtype=float),
        ub=np.array([1.0, 1.0], dtype=float),
    )


@pytest.fixture(scope="session")
def qp_incompatible_shapes() -> dict[str, np.ndarray]:
    return {
        "Q": np.eye(3, dtype=float),
        "c": np.array([1.0, 2.0], dtype=float),
    }


@pytest.fixture(scope="session")
def warm_start_good() -> np.ndarray:
    return np.array([0.4, 0.6], dtype=float)


@pytest.fixture(scope="session")
def warm_start_bad() -> np.ndarray:
    return np.array([np.nan, 0.5], dtype=float)


# ---------------------------------------------------------------------------
# Temp / artifact fixtures used by some math smoke tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_json_path(tmp_path: Path) -> Path:
    return tmp_path / "artifact.json"


@pytest.fixture()
def tmp_cli_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path
