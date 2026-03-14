from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

# Allow direct script execution: `python simons_smallcap_swing/features/build_features.py`
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simons_core.io.parquet_store import read_parquet, write_parquet
from simons_core.io.paths import data_dir
from simons_core.logging import get_logger
from simons_core.schemas import ColumnSpec, DataSchema, assert_schema


MODULE_VERSION = "features_mvp_v1"
DEFAULT_DECISION_LAG = 1

ADJUSTED_INPUT_SCHEMA = DataSchema(
    name="features_adjusted_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("close_adj", "number", nullable=False),
        ColumnSpec("volume_adj", "number", nullable=False),
    ),
    primary_key=("date", "instrument_id"),
    allow_extra_columns=True,
)

UNIVERSE_INPUT_SCHEMA = DataSchema(
    name="features_universe_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("is_eligible", "bool", nullable=False),
    ),
    primary_key=("date", "instrument_id"),
    allow_extra_columns=True,
)

MARKET_INPUT_SCHEMA = DataSchema(
    name="features_market_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("breadth_up", "number", nullable=False),
        ColumnSpec("equal_weight_return", "number", nullable=False),
        ColumnSpec("cross_sectional_vol", "number", nullable=False),
        ColumnSpec("coverage_ratio", "number", nullable=False),
    ),
    primary_key=("date",),
    allow_extra_columns=True,
)

FUNDAMENTALS_INPUT_SCHEMA = DataSchema(
    name="features_fundamentals_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("asof_date", "datetime64", nullable=False),
        ColumnSpec("metric_name", "string", nullable=False),
        ColumnSpec("metric_value", "number", nullable=False),
    ),
    primary_key=(),
    allow_extra_columns=True,
)

CALENDAR_INPUT_SCHEMA = DataSchema(
    name="features_calendar_input_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("is_session", "bool", nullable=False),
    ),
    primary_key=("date",),
    allow_extra_columns=True,
)

FEATURES_OUTPUT_SCHEMA = DataSchema(
    name="features_matrix_mvp",
    version="1.0.0",
    columns=(
        ColumnSpec("date", "datetime64", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("ticker", "string", nullable=False),
        ColumnSpec("ret_1d_lag1", "float64", nullable=True),
        ColumnSpec("ret_5d_lag1", "float64", nullable=True),
        ColumnSpec("ret_20d_lag1", "float64", nullable=True),
        ColumnSpec("momentum_20d_excl_1d", "float64", nullable=True),
        ColumnSpec("vol_5d", "float64", nullable=True),
        ColumnSpec("vol_20d", "float64", nullable=True),
        ColumnSpec("abs_ret_1d_lag1", "float64", nullable=True),
        ColumnSpec("log_volume_lag1", "float64", nullable=True),
        ColumnSpec("turnover_proxy_lag1", "float64", nullable=True),
        ColumnSpec("log_dollar_volume_lag1", "float64", nullable=True),
        ColumnSpec("mkt_breadth_up_lag1", "float64", nullable=True),
        ColumnSpec("mkt_equal_weight_return_lag1", "float64", nullable=True),
        ColumnSpec("mkt_cross_sectional_vol_lag1", "float64", nullable=True),
        ColumnSpec("mkt_coverage_ratio_lag1", "float64", nullable=True),
        ColumnSpec("log_total_assets", "float64", nullable=True),
        ColumnSpec("shares_outstanding", "float64", nullable=True),
        ColumnSpec("revenue_scale_proxy", "float64", nullable=True),
        ColumnSpec("net_income_scale_proxy", "float64", nullable=True),
    ),
    primary_key=("date", "instrument_id"),
    allow_extra_columns=True,
)


METRIC_ALIASES: dict[str, set[str]] = {
    "revenue": {
        "revenue",
        "revenues",
        "salesrevenuenet",
        "revenuefromcontractwithcustomerexcludingassessedtax",
    },
    "net_income": {
        "netincome",
        "netincomeloss",
        "profitloss",
    },
    "total_assets": {
        "assets",
        "totalassets",
    },
    "shares_outstanding": {
        "sharesoutstanding",
        "commonstocksharesoutstanding",
        "entitycommonstocksharesoutstanding",
    },
}


@dataclass(frozen=True)
class BuildFeaturesResult:
    features_path: Path
    summary_path: Path
    row_count: int
    n_instruments: int
    n_features: int
    feature_names: tuple[str, ...]
    config_hash: str


def _normalize_date(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Column '{column}' contains invalid dates.")
    return parsed.dt.normalize()


def _normalize_metric_token(value: object) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _canonical_metric_name(value: object) -> str | None:
    token = _normalize_metric_token(value)
    for canonical, aliases in METRIC_ALIASES.items():
        if token in aliases:
            return canonical
    return None


def _config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_inputs(
    *,
    adjusted_prices_path: str | Path | None,
    universe_history_path: str | Path | None,
    market_proxies_path: str | Path | None,
    fundamentals_pit_path: str | Path | None,
    trading_calendar_path: str | Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path, Path, Path]:
    base_data = data_dir()
    adjusted_source = (
        Path(adjusted_prices_path).expanduser().resolve()
        if adjusted_prices_path
        else base_data / "price" / "adjusted_prices.parquet"
    )
    universe_source = (
        Path(universe_history_path).expanduser().resolve()
        if universe_history_path
        else base_data / "universe" / "universe_history.parquet"
    )
    market_source = (
        Path(market_proxies_path).expanduser().resolve()
        if market_proxies_path
        else base_data / "price" / "market_proxies.parquet"
    )
    fundamentals_source = (
        Path(fundamentals_pit_path).expanduser().resolve()
        if fundamentals_pit_path
        else base_data / "edgar" / "fundamentals_pit.parquet"
    )
    calendar_source = (
        Path(trading_calendar_path).expanduser().resolve()
        if trading_calendar_path
        else base_data / "reference" / "trading_calendar.parquet"
    )

    adjusted = read_parquet(adjusted_source)
    universe = read_parquet(universe_source)
    market = read_parquet(market_source)
    fundamentals = read_parquet(fundamentals_source)
    calendar = read_parquet(calendar_source)
    return (
        adjusted,
        universe,
        market,
        fundamentals,
        calendar,
        adjusted_source,
        universe_source,
        market_source,
        fundamentals_source,
        calendar_source,
    )


def _validate_and_prepare_inputs(
    *,
    adjusted: pd.DataFrame,
    universe: pd.DataFrame,
    market: pd.DataFrame,
    fundamentals: pd.DataFrame,
    calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    assert_schema(adjusted, ADJUSTED_INPUT_SCHEMA)
    assert_schema(universe, UNIVERSE_INPUT_SCHEMA)
    assert_schema(market, MARKET_INPUT_SCHEMA)
    assert_schema(fundamentals, FUNDAMENTALS_INPUT_SCHEMA)
    assert_schema(calendar, CALENDAR_INPUT_SCHEMA)

    adjusted = adjusted.copy()
    universe = universe.copy()
    market = market.copy()
    fundamentals = fundamentals.copy()
    calendar = calendar.copy()

    adjusted["date"] = _normalize_date(adjusted["date"], column="date")
    adjusted["instrument_id"] = adjusted["instrument_id"].astype(str)
    adjusted["ticker"] = adjusted["ticker"].astype(str).str.upper().str.strip()
    adjusted["close_adj"] = pd.to_numeric(adjusted["close_adj"], errors="coerce")
    adjusted["volume_adj"] = pd.to_numeric(adjusted["volume_adj"], errors="coerce")
    if adjusted[["close_adj", "volume_adj"]].isna().any().any():
        raise ValueError("adjusted_prices contains non-numeric close_adj/volume_adj values.")
    if (adjusted["close_adj"] <= 0).any():
        raise ValueError("adjusted_prices contains non-positive close_adj values.")
    if (adjusted["volume_adj"] < 0).any():
        raise ValueError("adjusted_prices contains negative volume_adj values.")
    if adjusted.duplicated(["date", "instrument_id"]).any():
        raise ValueError("adjusted_prices has duplicate (date, instrument_id) rows.")

    universe["date"] = _normalize_date(universe["date"], column="date")
    universe["instrument_id"] = universe["instrument_id"].astype(str)
    universe["ticker"] = universe["ticker"].astype(str).str.upper().str.strip()
    universe["is_eligible"] = universe["is_eligible"].astype(bool)
    if universe.duplicated(["date", "instrument_id"]).any():
        raise ValueError("universe_history has duplicate (date, instrument_id) rows.")

    market["date"] = _normalize_date(market["date"], column="date")
    for col in (
        "breadth_up",
        "equal_weight_return",
        "cross_sectional_vol",
        "coverage_ratio",
    ):
        market[col] = pd.to_numeric(market[col], errors="coerce")
        if market[col].isna().any():
            raise ValueError(f"market_proxies contains invalid numeric values in '{col}'.")
    if market.duplicated(["date"]).any():
        raise ValueError("market_proxies has duplicate date rows.")

    fundamentals["instrument_id"] = fundamentals["instrument_id"].astype(str)
    fundamentals["metric_name"] = fundamentals["metric_name"].astype(str)
    fundamentals["metric_value"] = pd.to_numeric(fundamentals["metric_value"], errors="coerce")
    if fundamentals["metric_value"].isna().any():
        raise ValueError("fundamentals_pit contains non-numeric metric_value values.")

    asof_ts = pd.to_datetime(fundamentals["asof_date"], utc=True, errors="coerce")
    if asof_ts.isna().any():
        raise ValueError("fundamentals_pit contains invalid asof_date values.")
    fundamentals["asof_date"] = asof_ts.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()

    if "acceptance_ts" in fundamentals.columns:
        acceptance_ts = pd.to_datetime(fundamentals["acceptance_ts"], utc=True, errors="coerce")
        if acceptance_ts.isna().any():
            raise ValueError("fundamentals_pit contains invalid acceptance_ts values.")
        if (acceptance_ts > asof_ts).any():
            raise ValueError("fundamentals_pit violates PIT visibility: acceptance_ts > asof_date.")
        fundamentals["acceptance_ts"] = acceptance_ts
    if "filing_date" in fundamentals.columns:
        fundamentals["filing_date"] = pd.to_datetime(
            fundamentals["filing_date"], errors="coerce"
        ).dt.normalize()

    calendar["date"] = _normalize_date(calendar["date"], column="date")
    calendar["is_session"] = calendar["is_session"].astype(bool)
    if calendar.duplicated(["date"]).any():
        raise ValueError("trading_calendar has duplicate date rows.")

    sessions = pd.DatetimeIndex(sorted(calendar.loc[calendar["is_session"], "date"].unique()))
    if sessions.empty:
        raise ValueError("trading_calendar has no active sessions.")
    valid_sessions = set(sessions.tolist())

    for label, frame in (
        ("adjusted_prices", adjusted),
        ("universe_history", universe),
        ("market_proxies", market),
    ):
        invalid = frame.loc[~frame["date"].isin(valid_sessions), "date"]
        if not invalid.empty:
            sample = sorted({str(pd.Timestamp(item).date()) for item in invalid.head(10).tolist()})
            raise ValueError(f"{label} contains dates outside trading calendar sessions. Sample: {sample}")

    fundamentals["metric_name_canonical"] = fundamentals["metric_name"].map(_canonical_metric_name)
    fundamentals = fundamentals[fundamentals["metric_name_canonical"].notna()].copy()
    if fundamentals.empty:
        raise ValueError("fundamentals_pit has no rows matching canonical metrics for MVP.")

    sort_cols = ["instrument_id", "metric_name_canonical", "asof_date"]
    if "acceptance_ts" in fundamentals.columns:
        sort_cols.append("acceptance_ts")
    if "filing_date" in fundamentals.columns:
        sort_cols.append("filing_date")
    if "source_record_id" in fundamentals.columns:
        sort_cols.append("source_record_id")
    fundamentals = fundamentals.sort_values(sort_cols).drop_duplicates(
        ["instrument_id", "metric_name_canonical", "asof_date"], keep="last"
    )

    return adjusted, universe, market, fundamentals, sessions


def _build_decision_reference_dates(
    frame: pd.DataFrame,
    *,
    sessions: pd.DatetimeIndex,
    decision_lag: int,
) -> pd.DataFrame:
    if decision_lag < 1:
        raise ValueError("decision_lag must be >= 1.")
    session_df = pd.DataFrame({"date": sessions, "session_pos": range(len(sessions))})
    out = frame.merge(session_df, on="date", how="left")
    if out["session_pos"].isna().any():
        raise ValueError("Failed to map feature rows to trading calendar session positions.")
    out["session_pos"] = out["session_pos"].astype(int)
    out["decision_pos"] = out["session_pos"] - int(decision_lag)
    out["decision_ref_date"] = pd.NaT
    valid = out["decision_pos"] >= 0
    if valid.any():
        session_values = pd.to_datetime(sessions.to_numpy())
        out.loc[valid, "decision_ref_date"] = pd.to_datetime(
            session_values[out.loc[valid, "decision_pos"].to_numpy()]
        )
    return out


def _compute_price_features(frame: pd.DataFrame, *, decision_lag: int) -> pd.DataFrame:
    out = frame.sort_values(["instrument_id", "date"]).copy()
    grouped = out.groupby("instrument_id", sort=False)

    out["ret_1d_lag1"] = grouped["close_adj"].transform(lambda s: s.pct_change().shift(decision_lag))
    out["ret_5d_lag1"] = grouped["close_adj"].transform(lambda s: s.pct_change(5).shift(decision_lag))
    out["ret_20d_lag1"] = grouped["close_adj"].transform(lambda s: s.pct_change(20).shift(decision_lag))
    out["momentum_20d_excl_1d"] = out["ret_20d_lag1"] - out["ret_1d_lag1"]

    out["vol_5d"] = grouped["close_adj"].transform(
        lambda s: s.pct_change().rolling(5, min_periods=5).std(ddof=1).shift(decision_lag)
    )
    out["vol_20d"] = grouped["close_adj"].transform(
        lambda s: s.pct_change().rolling(20, min_periods=20).std(ddof=1).shift(decision_lag)
    )
    out["abs_ret_1d_lag1"] = out["ret_1d_lag1"].abs()

    out["log_volume_lag1"] = grouped["volume_adj"].transform(lambda s: np.log1p(s.shift(decision_lag)))
    out["dollar_volume_raw"] = out["close_adj"] * out["volume_adj"]
    out["log_dollar_volume_lag1"] = grouped["dollar_volume_raw"].transform(
        lambda s: np.log1p(s.shift(decision_lag))
    )

    def _turnover_lagged(volume: pd.Series) -> pd.Series:
        denom = volume.rolling(20, min_periods=5).median().replace(0, np.nan)
        return (volume / denom).shift(decision_lag)

    out["turnover_proxy_lag1"] = grouped["volume_adj"].transform(_turnover_lagged)
    out = out.drop(columns=["dollar_volume_raw"])
    return out


def _merge_market_context(frame: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    market_context = market[
        [
            "date",
            "breadth_up",
            "equal_weight_return",
            "cross_sectional_vol",
            "coverage_ratio",
        ]
    ].rename(
        columns={
            "date": "decision_ref_date",
            "breadth_up": "mkt_breadth_up_lag1",
            "equal_weight_return": "mkt_equal_weight_return_lag1",
            "cross_sectional_vol": "mkt_cross_sectional_vol_lag1",
            "coverage_ratio": "mkt_coverage_ratio_lag1",
        }
    )
    out = frame.merge(market_context, on="decision_ref_date", how="left")
    return out


def _attach_metric_asof(
    *,
    frame: pd.DataFrame,
    fundamentals: pd.DataFrame,
    metric_name: str,
) -> pd.Series:
    metric = fundamentals[fundamentals["metric_name_canonical"] == metric_name][
        ["instrument_id", "asof_date", "metric_value"]
    ].copy()
    if metric.empty:
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    left = frame[["instrument_id", "decision_ref_date"]].copy()
    left["__row_id"] = left.index
    valid_left = left[left["decision_ref_date"].notna()].copy()
    if valid_left.empty:
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    # pandas.merge_asof requires global monotonic order on the "on" keys.
    valid_left = valid_left.sort_values(["decision_ref_date", "instrument_id"])
    metric = metric.sort_values(["asof_date", "instrument_id"])
    merged = pd.merge_asof(
        valid_left,
        metric,
        by="instrument_id",
        left_on="decision_ref_date",
        right_on="asof_date",
        direction="backward",
        allow_exact_matches=True,
    )

    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    mapped = merged.set_index("__row_id")["metric_value"]
    out.loc[mapped.index.to_numpy()] = mapped.astype(float).to_numpy()
    return out


def _apply_fundamental_features(frame: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    revenue = _attach_metric_asof(frame=out, fundamentals=fundamentals, metric_name="revenue")
    net_income = _attach_metric_asof(frame=out, fundamentals=fundamentals, metric_name="net_income")
    total_assets = _attach_metric_asof(frame=out, fundamentals=fundamentals, metric_name="total_assets")
    shares = _attach_metric_asof(frame=out, fundamentals=fundamentals, metric_name="shares_outstanding")

    out["revenue_scale_proxy"] = np.sign(revenue) * np.log1p(np.abs(revenue))
    out["net_income_scale_proxy"] = np.sign(net_income) * np.log1p(np.abs(net_income))
    out["log_total_assets"] = np.where(total_assets > 0, np.log(total_assets), np.nan)
    out["shares_outstanding"] = shares
    return out


def build_features(
    *,
    adjusted_prices_path: str | Path | None = None,
    universe_history_path: str | Path | None = None,
    market_proxies_path: str | Path | None = None,
    fundamentals_pit_path: str | Path | None = None,
    trading_calendar_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    decision_lag: int = DEFAULT_DECISION_LAG,
    run_id: str = MODULE_VERSION,
) -> BuildFeaturesResult:
    logger = get_logger("features.build_features")
    lag = int(decision_lag)
    if lag != 1:
        raise ValueError(
            "build_features MVP currently supports decision_lag=1 only to keep explicit naming (_lag1)."
        )

    (
        adjusted,
        universe,
        market,
        fundamentals,
        _calendar,
        adjusted_source,
        universe_source,
        market_source,
        fundamentals_source,
        calendar_source,
    ) = _load_inputs(
        adjusted_prices_path=adjusted_prices_path,
        universe_history_path=universe_history_path,
        market_proxies_path=market_proxies_path,
        fundamentals_pit_path=fundamentals_pit_path,
        trading_calendar_path=trading_calendar_path,
    )

    adjusted, universe, market, fundamentals, sessions = _validate_and_prepare_inputs(
        adjusted=adjusted,
        universe=universe,
        market=market,
        fundamentals=fundamentals,
        calendar=_calendar,
    )

    eligible = universe[universe["is_eligible"]].copy()
    if eligible.empty:
        raise ValueError("universe_history has no eligible rows (is_eligible=True).")

    base = eligible[["date", "instrument_id", "ticker"]].merge(
        adjusted[["date", "instrument_id", "ticker", "close_adj", "volume_adj"]],
        on=["date", "instrument_id"],
        how="left",
        suffixes=("_universe", "_price"),
    )

    ticker_conflict = (
        base["ticker_price"].notna()
        & (base["ticker_universe"].astype(str) != base["ticker_price"].astype(str))
    )
    if ticker_conflict.any():
        sample = (
            base.loc[ticker_conflict, ["date", "instrument_id", "ticker_universe", "ticker_price"]]
            .head(10)
            .to_string(index=False)
        )
        raise ValueError(
            "Ticker mismatch between universe_history and adjusted_prices for same (date, instrument_id). "
            f"Sample:\n{sample}"
        )

    base["ticker"] = base["ticker_universe"].astype(str)
    base = base.drop(columns=["ticker_universe", "ticker_price"])
    base = base[base["close_adj"].notna() & base["volume_adj"].notna()].copy()
    if base.empty:
        raise ValueError("No eligible rows with valid adjusted prices were found.")
    base["close_adj"] = pd.to_numeric(base["close_adj"], errors="coerce")
    base["volume_adj"] = pd.to_numeric(base["volume_adj"], errors="coerce")

    base = _build_decision_reference_dates(base, sessions=sessions, decision_lag=lag)
    base = _compute_price_features(base, decision_lag=lag)
    base = _merge_market_context(base, market)
    base = _apply_fundamental_features(base, fundamentals)

    feature_columns = [
        "ret_1d_lag1",
        "ret_5d_lag1",
        "ret_20d_lag1",
        "momentum_20d_excl_1d",
        "vol_5d",
        "vol_20d",
        "abs_ret_1d_lag1",
        "log_volume_lag1",
        "turnover_proxy_lag1",
        "log_dollar_volume_lag1",
        "mkt_breadth_up_lag1",
        "mkt_equal_weight_return_lag1",
        "mkt_cross_sectional_vol_lag1",
        "mkt_coverage_ratio_lag1",
        "log_total_assets",
        "shares_outstanding",
        "revenue_scale_proxy",
        "net_income_scale_proxy",
    ]
    for col in feature_columns:
        base[col] = pd.to_numeric(base[col], errors="coerce").astype(float)

    output = base[
        ["date", "instrument_id", "ticker", *feature_columns]
    ].sort_values(["date", "instrument_id"]).reset_index(drop=True)

    if output.duplicated(["date", "instrument_id"]).any():
        raise ValueError("features_matrix contains duplicate (date, instrument_id) rows.")
    assert_schema(output, FEATURES_OUTPUT_SCHEMA)

    config_hash = _config_hash(
        {
            "version": MODULE_VERSION,
            "decision_lag": lag,
            "features": feature_columns,
            "input_paths": {
                "adjusted_prices": str(adjusted_source),
                "universe_history": str(universe_source),
                "market_proxies": str(market_source),
                "fundamentals_pit": str(fundamentals_source),
                "trading_calendar": str(calendar_source),
            },
            "fundamental_metric_aliases": {
                key: sorted(value) for key, value in METRIC_ALIASES.items()
            },
        }
    )
    built_ts_utc = datetime.now(UTC).isoformat()
    output["run_id"] = run_id
    output["config_hash"] = config_hash
    output["built_ts_utc"] = built_ts_utc

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else (data_dir() / "features")
    target_dir.mkdir(parents=True, exist_ok=True)
    features_path = write_parquet(
        output,
        target_dir / "features_matrix.parquet",
        schema_name=FEATURES_OUTPUT_SCHEMA.name,
        run_id=run_id,
    )

    pct_missing = {
        col: float(output[col].isna().mean()) for col in feature_columns
    }
    summary_payload = {
        "created_at_utc": built_ts_utc,
        "run_id": run_id,
        "config_hash": config_hash,
        "decision_lag_days": lag,
        "n_rows": int(len(output)),
        "n_instruments": int(output["instrument_id"].nunique()),
        "start_date": str(pd.Timestamp(output["date"].min()).date()),
        "end_date": str(pd.Timestamp(output["date"].max()).date()),
        "n_features": int(len(feature_columns)),
        "feature_names": feature_columns,
        "pct_missing_by_feature": pct_missing,
        "input_paths": {
            "adjusted_prices": str(adjusted_source),
            "universe_history": str(universe_source),
            "market_proxies": str(market_source),
            "fundamentals_pit": str(fundamentals_source),
            "trading_calendar": str(calendar_source),
        },
        "output_path": str(features_path),
    }
    summary_path = target_dir / "features_matrix.summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger.info(
        "features_matrix_built",
        run_id=run_id,
        row_count=int(len(output)),
        n_instruments=int(output["instrument_id"].nunique()),
        n_features=int(len(feature_columns)),
        output_path=str(features_path),
    )

    return BuildFeaturesResult(
        features_path=features_path,
        summary_path=summary_path,
        row_count=int(len(output)),
        n_instruments=int(output["instrument_id"].nunique()),
        n_features=int(len(feature_columns)),
        feature_names=tuple(feature_columns),
        config_hash=config_hash,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build MVP PIT-safe feature matrix.")
    parser.add_argument("--adjusted-prices-path", type=str, default=None)
    parser.add_argument("--universe-history-path", type=str, default=None)
    parser.add_argument("--market-proxies-path", type=str, default=None)
    parser.add_argument("--fundamentals-pit-path", type=str, default=None)
    parser.add_argument("--trading-calendar-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--decision-lag", type=int, default=DEFAULT_DECISION_LAG)
    parser.add_argument("--run-id", type=str, default=MODULE_VERSION)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = build_features(
        adjusted_prices_path=args.adjusted_prices_path,
        universe_history_path=args.universe_history_path,
        market_proxies_path=args.market_proxies_path,
        fundamentals_pit_path=args.fundamentals_pit_path,
        trading_calendar_path=args.trading_calendar_path,
        output_dir=args.output_dir,
        decision_lag=args.decision_lag,
        run_id=args.run_id,
    )
    print("Features matrix built:")
    print(f"- path: {result.features_path}")
    print(f"- summary: {result.summary_path}")
    print(f"- rows: {result.row_count}")
    print(f"- n_instruments: {result.n_instruments}")
    print(f"- n_features: {result.n_features}")
    print(f"- features: {list(result.feature_names)}")


if __name__ == "__main__":
    main()
