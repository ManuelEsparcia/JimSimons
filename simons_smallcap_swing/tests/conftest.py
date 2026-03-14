from __future__ import annotations

import sys
from pathlib import Path
import shutil

import pandas as pd
import pytest

# Ensure `simons_core` is importable when running pytest from repo root.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from simons_core.interfaces import RunContext
from simons_core.schemas import ColumnSpec, DataSchema


@pytest.fixture
def tmp_workspace() -> dict[str, Path]:
    # Use deterministic in-repo workspace because system temp is permission-restricted here.
    root = APP_ROOT / "artifacts" / "_pytest_mvp"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    data = root / "data"
    configs = root / "configs"
    artifacts = root / "artifacts"
    for path in (data, configs, artifacts):
        path.mkdir(parents=True, exist_ok=True)
    workspace = {"root": root, "data": data, "configs": configs, "artifacts": artifacts}
    try:
        yield workspace
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def pit_schema() -> DataSchema:
    return DataSchema(
        name="pit_observation_test",
        version="1.0.0",
        columns=(
            ColumnSpec("symbol", "string", nullable=False),
            ColumnSpec("asof", "datetime64[ns, UTC]", nullable=False),
            ColumnSpec("acceptance_ts", "datetime64[ns, UTC]", nullable=False),
            ColumnSpec("value", "float64", nullable=True),
        ),
        primary_key=("symbol", "asof"),
        allow_extra_columns=True,
    )


@pytest.fixture
def pit_valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "asof": pd.to_datetime(
                ["2026-01-09T16:00:00Z", "2026-01-09T16:00:00Z"], utc=True
            ),
            "acceptance_ts": pd.to_datetime(
                ["2026-01-09T15:00:00Z", "2026-01-09T16:00:00Z"], utc=True
            ),
            "value": [1.0, 2.0],
        }
    )


@pytest.fixture
def pit_invalid_df(pit_valid_df: pd.DataFrame) -> pd.DataFrame:
    bad = pit_valid_df.copy()
    bad.loc[1, "acceptance_ts"] = pd.Timestamp("2026-01-10T10:00:00Z")
    return bad


@pytest.fixture
def sample_calendar_path(tmp_workspace: dict[str, Path]) -> Path:
    dates = pd.bdate_range("2026-01-05", "2026-01-16", freq="B")
    frame = pd.DataFrame({"date": dates, "is_session": True})
    calendar_path = tmp_workspace["data"] / "trading_calendar.parquet"
    frame.to_parquet(calendar_path, index=False)
    return calendar_path


@pytest.fixture
def run_context() -> RunContext:
    return RunContext(
        run_id="test_run_001",
        asof_date=pd.Timestamp("2026-01-09T16:00:00Z"),
        seed=42,
        config_hash="cfg_abc123",
    )
