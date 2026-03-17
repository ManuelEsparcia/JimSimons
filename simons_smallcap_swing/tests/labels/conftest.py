from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# Repo import plumbing
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


LABEL_MODULE_CANDIDATES: dict[str, list[str]] = {
    "purged_splits": [
        "simons_smallcap_swing.labels.purged_splits",
        "labels.purged_splits",
        "purged_splits",
    ],
    "build_labels": [
        "simons_smallcap_swing.labels.build_labels",
        "labels.build_labels",
        "build_labels",
    ],
    "neutralized_targets": [
        "simons_smallcap_swing.labels.neutralized_targets",
        "labels.neutralized_targets",
        "neutralized_targets",
    ],
    "event_labels": [
        "simons_smallcap_swing.labels.event_labels",
        "labels.event_labels",
        "event_labels",
    ],
    "label_qc": [
        "simons_smallcap_swing.labels.label_qc",
        "labels.label_qc",
        "label_qc",
    ],
}


def import_from_candidates(candidates: list[str]):
    """Import the first available module from a list of candidates.

    This mirrors the fallback style used throughout the labels test suite so the
    same tests can run whether the code lives under the package namespace,
    directly under ``labels/`` or as a flat module during local experimentation.
    """
    last_exc: Exception | None = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:  # pragma: no cover - fallback path
            last_exc = exc
    if last_exc is None:  # pragma: no cover - defensive branch
        raise ModuleNotFoundError("No import candidates were supplied.")
    raise last_exc


# -----------------------------------------------------------------------------
# Generic filesystem / serialization helpers
# -----------------------------------------------------------------------------

def write_csv(df: pd.DataFrame, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)



def write_json(payload: dict[str, Any], path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return str(path)



def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))



def ensure_columns(frame: pd.DataFrame, required: set[str] | list[str]) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise AssertionError(f"Missing required columns: {sorted(missing)}")


# -----------------------------------------------------------------------------
# Common fixtures: repo paths, module loaders, temp dirs
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def labels_module_loader() -> Callable[[str], Any]:
    def _loader(module_key: str):
        if module_key not in LABEL_MODULE_CANDIDATES:
            raise KeyError(f"Unknown labels module key: {module_key}")
        return import_from_candidates(LABEL_MODULE_CANDIDATES[module_key])

    return _loader


@pytest.fixture(scope="session")
def purged_splits_module(labels_module_loader: Callable[[str], Any]):
    return labels_module_loader("purged_splits")


@pytest.fixture(scope="session")
def build_labels_module(labels_module_loader: Callable[[str], Any]):
    return labels_module_loader("build_labels")


@pytest.fixture(scope="session")
def neutralized_targets_module(labels_module_loader: Callable[[str], Any]):
    return labels_module_loader("neutralized_targets")


@pytest.fixture(scope="session")
def event_labels_module(labels_module_loader: Callable[[str], Any]):
    return labels_module_loader("event_labels")


@pytest.fixture(scope="session")
def label_qc_module(labels_module_loader: Callable[[str], Any]):
    return labels_module_loader("label_qc")


@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    return out


# -----------------------------------------------------------------------------
# Generic panel builders
# -----------------------------------------------------------------------------
@pytest.fixture()
def make_business_calendar() -> Callable[..., pd.DataFrame]:
    def _make(start: str = "2025-01-02", periods: int = 20) -> pd.DataFrame:
        return pd.DataFrame({"date": pd.bdate_range(start, periods=periods)})

    return _make


@pytest.fixture()
def make_symbol_list() -> Callable[..., list[str]]:
    def _make(n: int = 10, prefix: str = "S") -> list[str]:
        return [f"{prefix}{i:03d}" for i in range(n)]

    return _make


@pytest.fixture()
def make_universe_panel() -> Callable[..., pd.DataFrame]:
    def _make(
        dates: list[pd.Timestamp] | pd.DatetimeIndex,
        symbols: list[str],
        *,
        eligibility_col: str = "is_eligible",
        value: bool = True,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for date in pd.to_datetime(list(dates)):
            for symbol in symbols:
                rows.append({"date": date, "symbol": symbol, eligibility_col: value})
        return pd.DataFrame(rows)

    return _make


@pytest.fixture()
def make_cost_panel() -> Callable[..., pd.DataFrame]:
    def _make(
        dates: list[pd.Timestamp] | pd.DatetimeIndex,
        symbols: list[str],
        *,
        entry_cost: float = 0.0005,
        exit_cost: float = 0.0005,
        carry_cost_per_day: float = 0.0,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for date in pd.to_datetime(list(dates)):
            for symbol in symbols:
                rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "entry_cost": entry_cost,
                        "exit_cost": exit_cost,
                        "carry_cost_per_day": carry_cost_per_day,
                    }
                )
        return pd.DataFrame(rows)

    return _make


@pytest.fixture()
def make_feature_index_panel() -> Callable[..., pd.DataFrame]:
    def _make(
        dates: list[pd.Timestamp] | pd.DatetimeIndex,
        symbols: list[str],
        *,
        include_all: bool = True,
        with_sample_index: bool = False,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        idx = 0
        for date in pd.to_datetime(list(dates)):
            for symbol in symbols:
                row: dict[str, Any] = {
                    "date": date,
                    "symbol": symbol,
                    "feature_timestamp": date,
                    "feature_valid_flag": True,
                    "inclusion_flag": include_all,
                }
                if with_sample_index:
                    row["sample_index"] = idx
                rows.append(row)
                idx += 1
        return pd.DataFrame(rows)

    return _make


@pytest.fixture()
def make_price_panel() -> Callable[..., pd.DataFrame]:
    def _make(
        *,
        start: str = "2024-01-02",
        n_dates: int = 30,
        n_symbols: int = 10,
        symbol_prefix: str = "S",
        start_price: float = 10.0,
        base_step: float = 0.10,
        drift: float = 0.0005,
        adj_open_col: str = "adj_open",
        adj_close_col: str = "adj_close",
        adj_vwap_col: str = "adj_vwap",
    ) -> pd.DataFrame:
        dates = pd.bdate_range(start, periods=n_dates)
        symbols = [f"{symbol_prefix}{i:03d}" for i in range(n_symbols)]

        rows: list[dict[str, Any]] = []
        for s_idx, symbol in enumerate(symbols):
            close = start_price + base_step * s_idx
            for d_idx, date in enumerate(dates):
                daily_ret = drift + 0.00008 * s_idx + 0.00003 * ((d_idx % 5) - 2)
                close *= 1.0 + daily_ret
                open_px = close * (1.0 - 0.00035)
                vwap_px = (2.0 * open_px + close) / 3.0
                rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        adj_open_col: float(open_px),
                        adj_close_col: float(close),
                        adj_vwap_col: float(vwap_px),
                    }
                )
        return pd.DataFrame(rows)

    return _make


# -----------------------------------------------------------------------------
# Labels / build_labels synthetic bundle
# -----------------------------------------------------------------------------
@pytest.fixture()
def make_build_labels_bundle() -> Callable[..., dict[str, pd.DataFrame]]:
    def _make(
        *,
        n_dates: int = 36,
        n_symbols: int = 30,
        start: str = "2024-01-02",
        horizons: tuple[int, ...] = (5, 10),
    ) -> dict[str, pd.DataFrame]:
        dates = pd.bdate_range(start, periods=n_dates)
        symbols = [f"S{i:03d}" for i in range(n_symbols)]

        price_rows: list[dict[str, Any]] = []
        universe_rows: list[dict[str, Any]] = []
        exposure_rows: list[dict[str, Any]] = []
        cost_rows: list[dict[str, Any]] = []
        calendar_df = pd.DataFrame({"date": dates})

        for sym_idx, symbol in enumerate(symbols):
            beta = 0.60 + 0.03 * sym_idx
            market_cap = 100_000_000.0 + 2_000_000.0 * sym_idx
            liquidity = 1_000_000.0 + 25_000.0 * sym_idx
            sector = ("tech", "health", "industrial")[sym_idx % 3]

            close = 10.0 + 0.15 * sym_idx
            for t_idx, date in enumerate(dates):
                daily_ret = -0.0012 + 0.00012 * sym_idx + 0.00004 * ((t_idx % 4) - 1.5)
                close *= 1.0 + daily_ret
                open_px = close * (1.0 - 0.00035)
                vwap_px = (2.0 * open_px + close) / 3.0

                price_rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "adj_open": float(open_px),
                        "adj_close": float(close),
                        "adj_vwap": float(vwap_px),
                    }
                )
                universe_rows.append({"date": date, "symbol": symbol, "is_eligible": True})
                exposure_rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "sector": sector,
                        "market_beta": beta,
                        "market_cap": market_cap,
                        "liquidity": liquidity,
                    }
                )
                cost_rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "entry_cost": 0.0005 + 0.00001 * (sym_idx % 5),
                        "exit_cost": 0.0006 + 0.00001 * (sym_idx % 7),
                        "carry_cost_per_day": 0.0,
                    }
                )

        config_payload = {
            "run_id": "unit_build_labels",
            "label_set_name": "unit_test_labels",
            "horizons": list(horizons),
            "decision_lag": 1,
            "entry_price": "close",
            "exit_price": "close",
            "return_type": "simple",
            "target_family": "fwd_ret",
            "target_variant": "net",
            "classification": {
                "enabled": True,
                "scheme": "cross_sectional_quantiles",
                "quantiles": [0.3, 0.7],
                "emit_binary": True,
                "emit_ternary": True,
            },
            "neutralization": {
                "enabled": True,
                "method": "weighted_ridge",
                "factor_cols": ["market_beta", "market_cap", "liquidity"],
                "group_col": "sector",
                "weight_col": None,
            },
            "missing_cost_policy": "strict_invalidate",
            "persist_partitioned_output": True,
        }

        return {
            "prices": pd.DataFrame(price_rows),
            "universe": pd.DataFrame(universe_rows),
            "factor_exposures": pd.DataFrame(exposure_rows),
            "execution_costs": pd.DataFrame(cost_rows),
            "calendar": calendar_df,
            "config": config_payload,
        }

    return _make


# -----------------------------------------------------------------------------
# neutralized_targets synthetic bundle
# -----------------------------------------------------------------------------
@pytest.fixture()
def make_neutralization_bundle() -> Callable[..., tuple[pd.DataFrame, pd.DataFrame]]:
    def _make(
        *,
        n_dates: int = 3,
        n_symbols: int = 30,
        horizons: tuple[int, ...] = (5, 10),
        seed: int = 12345,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2024-01-02", periods=n_dates)
        symbols = [f"S{i:03d}" for i in range(n_symbols)]
        sectors = ("tech", "health", "industrial", "energy", "utilities")
        sector_effects = {
            "tech": 0.030,
            "health": -0.015,
            "industrial": 0.000,
            "energy": 0.020,
            "utilities": -0.010,
        }

        label_rows: list[dict[str, Any]] = []
        exposure_rows: list[dict[str, Any]] = []

        for date_idx, date in enumerate(dates):
            for sym_idx, symbol in enumerate(symbols):
                sector = sectors[sym_idx % len(sectors)]
                beta = float(rng.normal(1.0 + 0.03 * date_idx, 0.18))
                log_mktcap = float(10.0 + rng.normal(0.0, 0.85))
                liq_log = float(12.0 + rng.normal(0.0, 0.60))
                exposure_rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "sector": sector,
                        "market_beta": beta,
                        "market_cap": float(np.exp(log_mktcap)),
                        "liquidity": float(np.expm1(liq_log)),
                        "quality_weight": 1.0 + 0.05 * (sym_idx % 7),
                    }
                )

                base = (
                    0.18 * beta
                    + 0.05 * log_mktcap
                    + 0.035 * liq_log
                    + sector_effects[sector]
                    + float(rng.normal(0.0, 0.01))
                )
                row: dict[str, Any] = {"date": date, "symbol": symbol, "label_valid_flag": True}
                for horizon in horizons:
                    row[f"y_fwd_ret_net_{horizon}d"] = float(base * (1.0 + 0.01 * horizon) + rng.normal(0.0, 0.008))
                label_rows.append(row)

        return pd.DataFrame(label_rows), pd.DataFrame(exposure_rows)

    return _make


@pytest.fixture()
def make_long_labels_from_wide() -> Callable[..., pd.DataFrame]:
    def _make(labels_wide: pd.DataFrame, horizons: tuple[int, ...] = (5, 10)) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for _, rec in labels_wide.iterrows():
            for horizon in horizons:
                rows.append(
                    {
                        "date": rec["date"],
                        "symbol": rec["symbol"],
                        "horizon": horizon,
                        "label_valid_flag": rec["label_valid_flag"],
                        "y_fwd_ret_net": rec[f"y_fwd_ret_net_{horizon}d"],
                    }
                )
        return pd.DataFrame(rows)

    return _make


# -----------------------------------------------------------------------------
# event_labels synthetic bundle
# -----------------------------------------------------------------------------
@pytest.fixture()
def make_event_bundle() -> Callable[[], dict[str, pd.DataFrame]]:
    def _make() -> dict[str, pd.DataFrame]:
        base_dates = pd.bdate_range("2026-01-05", periods=15)

        symbols = {"AAA": 10.0, "BBB": 20.0, "CCC": 30.0}
        price_rows: list[dict[str, Any]] = []
        for symbol, start_open in symbols.items():
            step = {"AAA": 0.2, "BBB": 0.3, "CCC": 0.4}[symbol]
            for idx, date in enumerate(base_dates):
                open_px = start_open + step * idx
                close_px = open_px * (1.0 + 0.01 + 0.001 * idx)
                price_rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "adj_open": round(open_px, 6),
                        "adj_close": round(close_px, 6),
                    }
                )

        universe = make_universe_panel.__wrapped__()  # type: ignore[attr-defined]
        costs = make_cost_panel.__wrapped__()  # type: ignore[attr-defined]

        benchmark_rows: list[dict[str, Any]] = []
        benchmark_spec = {
            (pd.Timestamp("2026-01-05"), 1): {"market": 0.005, "sector": {"Tech": 0.004}},
            (pd.Timestamp("2026-01-05"), 3): {"market": 0.012, "sector": {"Tech": 0.006}},
            (pd.Timestamp("2026-01-06"), 1): {"market": 0.006, "sector": {"Health": 0.002}},
            (pd.Timestamp("2026-01-06"), 3): {"market": 0.014, "sector": {"Health": 0.005}},
        }
        for (date, horizon), payload in benchmark_spec.items():
            benchmark_rows.append(
                {
                    "date": date,
                    "horizon_days": horizon,
                    "benchmark_type": "market",
                    "benchmark_return": payload["market"],
                }
            )
            for sector, value in payload["sector"].items():
                benchmark_rows.append(
                    {
                        "date": date,
                        "horizon_days": horizon,
                        "benchmark_type": "sector",
                        "sector": sector,
                        "benchmark_return": value,
                    }
                )

        events = pd.DataFrame(
            [
                {
                    "event_id": "earn_pre",
                    "symbol": "AAA",
                    "event_type": "earnings",
                    "event_source": "newswire",
                    "event_timestamp_known": "2026-01-05 07:00:00",
                    "timing_hint": "pre_market",
                },
                {
                    "event_id": "news_after",
                    "symbol": "BBB",
                    "event_type": "corp_news",
                    "event_source": "newswire",
                    "event_timestamp_known": "2026-01-05 16:30:00",
                    "timing_hint": "after_close",
                },
                {
                    "event_id": "gap_open",
                    "symbol": "CCC",
                    "event_type": "open_gap",
                    "event_source": "scanner",
                    "event_timestamp_known": "2026-01-06 09:20:00",
                    "timing_hint": "at_open",
                },
            ]
        )

        config = {
            "event_to_horizons_map": {
                "EARNINGS": [1, 3],
                "CORP_NEWS": [1, 3],
                "OPEN_GAP": [1, 2],
                "TECH_BREAK": [3],
                "VOL_SHOCK": [1],
                "MICROSTRUCTURE_SHOCK": [1],
            },
            "entry_exit_convention": "open_to_open",
            "timestamp_policy": "strict",
            "overlap_policy": "first_event_wins_with_cooldown",
            "cooldown_policy": "max_horizon",
            "min_family_sample": 1,
            "classification_policy": {
                "enabled": True,
                "top_quantile": 0.75,
                "bottom_quantile": 0.25,
                "within_family": True,
                "emit_binary": True,
                "emit_ternary": True,
            },
            "abnormal_mode": "auto",
            "benchmark_mode_default_corporate": "sector_market_adjusted",
            "benchmark_mode_default_technical": "none",
            "missing_cost_policy": "strict_invalidate",
            "net_of_costs": True,
            "policy_version": "unit-test",
            "taxonomy_version": "unit-test",
        }

        return {
            "prices": pd.DataFrame(price_rows),
            "universe": universe(base_dates, ["AAA", "BBB", "CCC"]),
            "costs": costs(base_dates, ["AAA", "BBB", "CCC"], entry_cost=0.0005, exit_cost=0.0005),
            "sectors": pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"], "sector": ["Tech", "Health", "Industrial"]}),
            "benchmarks": pd.DataFrame(benchmark_rows),
            "events": events,
            "calendar": pd.DataFrame({"date": base_dates}),
            "config": config,
        }

    return _make


# -----------------------------------------------------------------------------
# label_qc synthetic bundles
# -----------------------------------------------------------------------------
@pytest.fixture()
def make_qc_continuous_bundle() -> Callable[..., dict[str, pd.DataFrame]]:
    def _make(
        *,
        n_dates: int = 20,
        n_symbols: int = 15,
        horizon: int = 5,
        start: str = "2025-01-02",
    ) -> dict[str, pd.DataFrame]:
        dates = pd.bdate_range(start, periods=n_dates)
        symbols = [f"S{i:03d}" for i in range(n_symbols)]
        feature_rows: list[dict[str, Any]] = []
        label_rows: list[dict[str, Any]] = []

        last_valid_idx = n_dates - horizon - 1
        for d_idx, date in enumerate(dates):
            for s_idx, symbol in enumerate(symbols):
                feature_rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "feature_timestamp": date,
                        "feature_valid_flag": True,
                    }
                )

                valid = d_idx <= last_valid_idx
                exclusion_reason = None if valid else "INCOMPLETE_FORWARD_WINDOW"
                y_value = (0.0009 * s_idx) + (0.00015 * (d_idx % 4)) if valid else np.nan
                label_rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "run_id": "upstream_build_labels",
                        f"y_fwd_ret_net_{horizon}d": y_value,
                        f"label_valid_flag_{horizon}d": valid,
                        f"label_exclusion_reason_{horizon}d": exclusion_reason,
                        f"event_start_{horizon}d": date + pd.tseries.offsets.BDay(1),
                        f"event_end_{horizon}d": date + pd.tseries.offsets.BDay(horizon),
                        "feature_timestamp": date,
                    }
                )

        return {"features": pd.DataFrame(feature_rows), "labels": pd.DataFrame(label_rows)}

    return _make


@pytest.fixture()
def make_qc_discrete_bundle() -> Callable[..., dict[str, pd.DataFrame]]:
    def _make(
        *,
        n_dates: int = 20,
        n_symbols: int = 15,
        horizon: int = 5,
        start: str = "2025-02-03",
        minority_every: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        dates = pd.bdate_range(start, periods=n_dates)
        symbols = [f"D{i:03d}" for i in range(n_symbols)]
        feature_rows: list[dict[str, Any]] = []
        label_rows: list[dict[str, Any]] = []

        last_valid_idx = n_dates - horizon - 1
        for d_idx, date in enumerate(dates):
            for s_idx, symbol in enumerate(symbols):
                feature_rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "feature_timestamp": date,
                        "feature_valid_flag": True,
                    }
                )
                valid = d_idx <= last_valid_idx
                exclusion_reason = None if valid else "INCOMPLETE_FORWARD_WINDOW"
                if valid:
                    value = 1
                    if minority_every is not None and ((d_idx * n_symbols + s_idx) % minority_every == 0):
                        value = 0
                else:
                    value = np.nan
                label_rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        f"y_cls_{horizon}d": value,
                        f"label_valid_flag_{horizon}d": valid,
                        f"label_exclusion_reason_{horizon}d": exclusion_reason,
                        f"event_start_{horizon}d": date + pd.tseries.offsets.BDay(1),
                        f"event_end_{horizon}d": date + pd.tseries.offsets.BDay(horizon),
                        "feature_timestamp": date,
                    }
                )

        return {"features": pd.DataFrame(feature_rows), "labels": pd.DataFrame(label_rows)}

    return _make


# -----------------------------------------------------------------------------
# Convenience fixture: synthetic CSV writer bundle for multi-file tests
# -----------------------------------------------------------------------------
@pytest.fixture()
def write_bundle() -> Callable[[dict[str, Any], Path], dict[str, str]]:
    def _write(bundle: dict[str, Any], base_dir: Path) -> dict[str, str]:
        base_dir.mkdir(parents=True, exist_ok=True)
        out: dict[str, str] = {}
        for key, value in bundle.items():
            if isinstance(value, pd.DataFrame):
                out[key] = write_csv(value, base_dir / f"{key}.csv")
            elif isinstance(value, dict):
                out[key] = write_json(value, base_dir / f"{key}.json")
            else:
                raise TypeError(f"Unsupported bundle value for key={key!r}: {type(value)}")
        return out

    return _write
