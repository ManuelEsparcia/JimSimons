from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# Import adapters
# -----------------------------------------------------------------------------


def _load_module(module_basename: str):
    module_names = [
        f"simons_smallcap_swing.data.price.{module_basename}",
        f"data.price.{module_basename}",
        module_basename,
    ]
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue

    here = Path(__file__).resolve()
    candidate_paths = [
        here.parents[3] / "simons_smallcap_swing" / "data" / "price" / f"{module_basename}.py",
        here.parents[2] / "data" / "price" / f"{module_basename}.py",
        here.parents[1] / f"{module_basename}.py",
        Path(f"/mnt/data/{module_basename}.py"),
    ]
    for path in candidate_paths:
        if path.exists():
            spec = importlib.util.spec_from_file_location(f"{module_basename}_under_test", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module

    raise ModuleNotFoundError(f"Could not import {module_basename}.py")


@pytest.fixture(scope="session")
def fetch_prices_module():
    return _load_module("fetch_prices")


@pytest.fixture(scope="session")
def adjust_prices_module():
    return _load_module("adjust_prices")


@pytest.fixture(scope="session")
def qc_prices_module():
    return _load_module("qc_prices")


@pytest.fixture(scope="session")
def market_proxies_module():
    return _load_module("market_proxies")


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path



def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path



def _load_csvish(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)



def _universe_history() -> pd.DataFrame:
    rows = []
    for d in ["2024-01-02", "2024-01-03", "2024-01-04"]:
        for symbol in ["AAA", "BBB"]:
            rows.append({"date": d, "symbol": symbol, "is_eligible": True, "run_id": "u_snap_001"})
    return pd.DataFrame(rows)



def _calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "is_session": [True, True, True],
        }
    )



def _provider_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)



def _base_provider_payloads(*, revised_last_close: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
    aaa_last_close = 55.0 if revised_last_close is None else float(revised_last_close)
    return {
        "primary": {
            "AAA": _provider_frame(
                [
                    {"date": "2024-01-02", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
                    {"date": "2024-01-03", "open": 50.0, "high": 51.0, "low": 49.0, "close": 50.0, "volume": 2200.0},
                    {"date": "2024-01-04", "open": aaa_last_close - 1.0, "high": aaa_last_close + 1.0, "low": aaa_last_close - 2.0, "close": aaa_last_close, "volume": 2100.0},
                ]
            ),
            "BBB": _provider_frame(
                [
                    {"date": "2024-01-02", "open": 20.0, "high": 20.5, "low": 19.5, "close": 20.0, "volume": 1500.0},
                    {"date": "2024-01-03", "open": 22.0, "high": 22.5, "low": 21.5, "close": 22.0, "volume": 1600.0},
                    {"date": "2024-01-04", "open": 23.0, "high": 23.5, "low": 22.5, "close": 23.0, "volume": 1700.0},
                ]
            ),
        }
    }



def _corporate_actions_split() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "event_type": "split",
                "effective_date": "2024-01-03",
                "announcement_ts": "2024-01-02T21:00:00Z",
                "effective_ts_utc": "2024-01-03T00:00:00Z",
                "split_ratio": 0.5,
                "source_provider": "corp_actions_primary",
                "provider_priority": 0,
                "event_id": "AAA_SPLIT_20240103",
            }
        ]
    )



def _raw_for_adjust(raw: pd.DataFrame) -> pd.DataFrame:
    cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume"]
    out = raw[cols].copy()
    out = out.rename(columns={"trade_date": "date"})
    out["date"] = pd.to_datetime(out["date"])
    return out



def _adjusted_for_market(adjusted: pd.DataFrame) -> pd.DataFrame:
    if "close_adj_total" in adjusted.columns:
        close_col = "close_adj_total"
        ret_col = "ret_1d_adj_total"
        high_col = "high_adj_total"
        low_col = "low_adj_total"
    else:
        close_col = "close_adj_split"
        ret_col = "ret_1d_adj_split"
        high_col = "high_adj_split"
        low_col = "low_adj_split"

    out = pd.DataFrame(
        {
            "symbol": adjusted["symbol"],
            "date": adjusted["date"],
            "close_adj": adjusted[close_col],
            "ret_1d_adj": adjusted[ret_col],
            "volume_adj": adjusted["volume_adj"],
            "high_adj": adjusted[high_col],
            "low_adj": adjusted[low_col],
        }
    )
    out["dollar_volume"] = out["close_adj"] * out["volume_adj"]
    return out



def _default_fetch_config(module, output_root: Path, provider_aliases: Iterable[str]):
    return module.FetchPricesConfig(
        retry=module.RetryPolicy(max_retries=1, base_delay_sec=0.0, alpha=1.0, cap_delay_sec=0.0, jitter_low=1.0, jitter_high=1.0),
        rate_limit=module.RateLimitPolicy(min_interval_sec=0.0),
        validation=module.ValidationPolicy(),
        persistence=module.PersistencePolicy(output_root=str(output_root), canonical_relpath="canonical/canonical_prices.parquet"),
        fetch=module.FetchPolicy(provider_priority=tuple(provider_aliases), frequency="1d", mode="full_refresh"),
        normalization=module.NormalizationPolicy(),
    )



def _patch_fetch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    module,
    tmp_path: Path,
    provider_payloads: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    class _Client:
        def __init__(self, alias: str, payloads: Mapping[str, Any]):
            self.provider_alias = alias
            self.provider_name = alias.upper()
            self.dataset_id = f"dataset_{alias}"
            self._payloads: MutableMapping[str, list[Any]] = {}
            for symbol, item in payloads.items():
                if isinstance(item, list):
                    self._payloads[str(symbol)] = list(item)
                else:
                    self._payloads[str(symbol)] = [item]

        def fetch_daily_bars(self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp, frequency: str) -> pd.DataFrame:
            items = self._payloads.get(symbol)
            if not items:
                raise module.PermanentProviderError(f"No payload configured for symbol={symbol}")
            item = items.pop(0)
            if isinstance(item, Exception):
                raise item
            return item.copy()

    output_root = tmp_path / "fetch_runtime"
    cfg = _default_fetch_config(module, output_root=output_root, provider_aliases=provider_payloads.keys())
    providers = {
        alias: module.ProviderSpec(alias=alias, kind="python", name=alias.upper(), params={}, dataset_id=f"ds_{alias}", timezone=None, enabled=True)
        for alias in provider_payloads.keys()
    }
    raw_cfg = {"providers": {alias: {"kind": "python", "enabled": True} for alias in provider_payloads.keys()}}
    clients = {alias: _Client(alias, payloads) for alias, payloads in provider_payloads.items()}

    monkeypatch.setattr(module, "load_config", lambda config_path, provider_config_path: (cfg, providers, raw_cfg, "fetch_cfg_hash"))
    monkeypatch.setattr(module, "build_provider_client", lambda spec: clients[spec.alias])
    monkeypatch.setattr(module, "backoff_sleep", lambda *args, **kwargs: None)

    def _write_parquet_deduped(path: Path, frame: pd.DataFrame, dedupe_keys=None, overwrite: bool = False, compression: str = "snappy") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    def _load_existing_canonical(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=module.REQUIRED_CANONICAL_COLUMNS)
        df = pd.read_csv(path)
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    monkeypatch.setattr(module, "write_parquet_deduped", _write_parquet_deduped)
    monkeypatch.setattr(module, "load_existing_canonical", _load_existing_canonical)
    monkeypatch.setattr(module, "write_json", lambda path, payload: _write_json(path, payload))
    monkeypatch.setattr(module, "require_parquet_engine", lambda: None)

    def _df_to_parquet(self, path, index: bool = False, compression: str | None = None, **kwargs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_csv(path, index=index)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _df_to_parquet, raising=False)
    return {"output_root": output_root, "config": cfg, "providers": providers}



def _patch_qc_persistence(monkeypatch: pytest.MonkeyPatch, module) -> None:
    def _write_parquet(path: Path, df: pd.DataFrame, *, index: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=index)

    monkeypatch.setattr(module, "write_parquet", _write_parquet)



def _patch_market_persistence(monkeypatch: pytest.MonkeyPatch, module) -> None:
    def _persist_outputs(df, warn_df, summary, validation, manifest, output_dir, run_id):
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        payload = df.copy()
        payload["date"] = pd.to_datetime(payload["date"]).dt.strftime("%Y-%m-%d")
        for d, day in payload.groupby("date", sort=True):
            part_dir = outdir / f"date={d}"
            part_dir.mkdir(parents=True, exist_ok=True)
            day.to_csv(part_dir / "part-00000.parquet", index=False)

        warn_df.to_csv(outdir / f"validation_rows_{run_id}.parquet", index=False)
        _write_json(outdir / f"summary_{run_id}.json", summary)
        _write_json(outdir / f"validation_{run_id}.json", validation)
        _write_json(outdir / f"manifest_{run_id}.json", manifest)

    monkeypatch.setattr(module, "persist_outputs", _persist_outputs)



def _write_qc_config(path: Path, *, require_adjusted: bool = True) -> Path:
    payload = {
        "require_adjusted_consistency_check": bool(require_adjusted),
        "pct_fail_row_fail_threshold": 0.01,
        "pct_symbols_bad_coverage_fail_threshold": 0.5,
        "symbol_bad_coverage_threshold": 0.5,
        "extreme_return_threshold": 0.6,
        "raw_adjusted_factor_jump_threshold": 0.10,
        "raw_adjusted_mag_ratio_threshold": 5.0,
        "code_version": "qc_prices_v1",
    }
    return _write_json(path, payload)



def _write_market_config(path: Path, output_dir: Path) -> Path:
    payload = {
        "return_windows": [1, 5, 20],
        "realized_vol_window": 20,
        "turnover_ref_window": 20,
        "winsor_lower": 0.05,
        "winsor_upper": 0.95,
        "coverage": {"n_min": 1, "ratio_min": 0.50, "warn_margin_n": 0, "warn_margin_ratio": 0.0},
        "exploratory": {"enabled": []},
        "smoothing": {"enabled": False},
        "zscore": {"enabled": False},
        "output_dir": str(output_dir),
    }
    return _write_json(path, payload)



def _run_market_stage(module, monkeypatch: pytest.MonkeyPatch, prices_path: Path, universe_path: Path, config_path: Path, run_id: str) -> Dict[str, Any]:
    _patch_market_persistence(monkeypatch, module)
    return module.run_market_proxies(
        prices_path=prices_path,
        universe_path=universe_path,
        config_path=config_path,
        run_id=run_id,
        asof_ts_utc="2026-03-15T12:00:00Z",
    )



def _run_price_stack(
    *,
    fetch_prices_module,
    adjust_prices_module,
    qc_prices_module,
    market_proxies_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider_payloads: Mapping[str, Mapping[str, Any]],
    corporate_actions: Optional[pd.DataFrame] = None,
    qc_require_adjusted: bool = True,
    qc_raw_override: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    universe_path = _write_csv(_universe_history(), tmp_path / "universe_history.csv")
    calendar_path = _write_csv(_calendar(), tmp_path / "market_calendar.csv")
    provider_cfg_path = _write_json(tmp_path / "provider_config.json", {"providers": list(provider_payloads.keys())})

    fetch_patch = _patch_fetch_runtime(monkeypatch, fetch_prices_module, tmp_path, provider_payloads)
    fetch_artifacts = fetch_prices_module.run_fetch_prices(
        universe_snapshot_path=universe_path,
        calendar_path=calendar_path,
        provider_config_path=provider_cfg_path,
        start_date="2024-01-02",
        end_date="2024-01-04",
        frequency="1d",
        fetch_mode="full_refresh",
        run_id="price_fetch_run",
        asof_ts_utc="2026-03-15T12:00:00Z",
        config_path=None,
    )

    raw_for_adjust = _raw_for_adjust(fetch_artifacts.raw)
    corp_actions = corporate_actions.copy() if corporate_actions is not None else pd.DataFrame()

    adjust_cfg = adjust_prices_module.AdjustPricesConfig(
        reverse_split=adjust_prices_module.ReverseSplitPolicy(extreme_threshold=20.0),
        conflict=adjust_prices_module.ConflictPolicy(provider_priority=("corp_actions_primary",), same_day_precedence=("split", "cash_dividend", "special_cash_dividend")),
        adjustment=adjust_prices_module.AdjustmentPolicy(mode="dual_output", allow_empty_corporate_actions=True),
        output=adjust_prices_module.OutputPolicy(),
    )
    adjust_artifacts = adjust_prices_module.adjust_prices(
        raw_for_adjust,
        corp_actions,
        run_id="price_adjust_run",
        asof_ts_utc="2026-03-15T12:00:00Z",
        start_date="2024-01-02",
        end_date="2024-01-04",
        config=adjust_cfg,
    )

    qc_raw_df = qc_raw_override.copy() if qc_raw_override is not None else fetch_artifacts.raw.copy()
    raw_path = _write_csv(qc_raw_df, tmp_path / "prices_raw.csv")
    adjusted_path = _write_csv(adjust_artifacts.adjusted, tmp_path / "prices_adjusted.csv")
    qc_config_path = _write_qc_config(tmp_path / "qc_config.json", require_adjusted=qc_require_adjusted)
    qc_output_dir = tmp_path / "qc_output"
    _patch_qc_persistence(monkeypatch, qc_prices_module)
    qc_artifacts = qc_prices_module.run_qc_prices(
        prices_raw_path=raw_path,
        prices_adjusted_path=adjusted_path,
        calendar_path=calendar_path,
        config_path=qc_config_path,
        output_dir=qc_output_dir,
        run_id="price_qc_run",
        as_of_ts_utc="2026-03-15T12:00:00Z",
    )

    market_prices = _adjusted_for_market(adjust_artifacts.adjusted)
    market_prices_path = _write_csv(market_prices, tmp_path / "market_prices_adjusted.csv")
    market_output_dir = tmp_path / "market_output"
    market_config_path = _write_market_config(tmp_path / "market_config.json", market_output_dir)
    market_result = _run_market_stage(
        market_proxies_module,
        monkeypatch,
        prices_path=market_prices_path,
        universe_path=universe_path,
        config_path=market_config_path,
        run_id="market_proxy_run",
    )

    return {
        "fetch": fetch_artifacts,
        "adjust": adjust_artifacts,
        "qc": qc_artifacts,
        "market": market_result,
        "paths": {
            "universe": universe_path,
            "calendar": calendar_path,
            "provider_cfg": provider_cfg_path,
            "raw": raw_path,
            "adjusted": adjusted_path,
            "market_prices": market_prices_path,
            "qc_config": qc_config_path,
            "market_config": market_config_path,
            "fetch_output_root": fetch_patch["output_root"],
        },
    }



def _require_qc_pass_then_run_market(
    *,
    qc_artifacts,
    market_proxies_module,
    monkeypatch: pytest.MonkeyPatch,
    prices_path: Path,
    universe_path: Path,
    market_config_path: Path,
    run_id: str,
):
    if str(qc_artifacts.summary.get("gate", "")).lower() != "pass":
        raise RuntimeError(f"QC gate must be pass before market proxies. Got={qc_artifacts.summary.get('gate')}")
    return _run_market_stage(
        market_proxies_module,
        monkeypatch,
        prices_path=prices_path,
        universe_path=universe_path,
        config_path=market_config_path,
        run_id=run_id,
    )


# -----------------------------------------------------------------------------
# Integration tests
# -----------------------------------------------------------------------------


def test_fetch_adjust_qc_market_happy_path(
    fetch_prices_module,
    adjust_prices_module,
    qc_prices_module,
    market_proxies_module,
    monkeypatch,
    tmp_path,
):
    result = _run_price_stack(
        fetch_prices_module=fetch_prices_module,
        adjust_prices_module=adjust_prices_module,
        qc_prices_module=qc_prices_module,
        market_proxies_module=market_proxies_module,
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        provider_payloads=_base_provider_payloads(),
        corporate_actions=_corporate_actions_split(),
        qc_require_adjusted=True,
    )

    fetch_art = result["fetch"]
    adjust_art = result["adjust"]
    qc_art = result["qc"]
    market = result["market"]

    assert len(fetch_art.raw) == 6
    assert not adjust_art.adjusted.empty
    assert str(qc_art.summary["gate"]).lower() == "pass"
    assert market["validation"]["gate_status"] in {"PASS", "WARN"}
    assert len(market["data"]) == 3
    assert {"market_ret_1d", "breadth_pct_up", "cross_section_dispersion"}.issubset(set(market["data"].columns))



def test_split_event_flows_through_adjust_and_qc_without_false_fail(
    fetch_prices_module,
    adjust_prices_module,
    qc_prices_module,
    market_proxies_module,
    monkeypatch,
    tmp_path,
):
    result = _run_price_stack(
        fetch_prices_module=fetch_prices_module,
        adjust_prices_module=adjust_prices_module,
        qc_prices_module=qc_prices_module,
        market_proxies_module=market_proxies_module,
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        provider_payloads=_base_provider_payloads(),
        corporate_actions=_corporate_actions_split(),
        qc_require_adjusted=True,
    )

    adjusted = result["adjust"].adjusted.sort_values(["symbol", "date"]).reset_index(drop=True)
    aaa = adjusted.loc[adjusted["symbol"] == "AAA"].reset_index(drop=True)
    qc_failures = result["qc"].failures

    assert aaa.loc[0, "close_adj_split"] == pytest.approx(50.0)
    assert aaa.loc[1, "close_adj_split"] == pytest.approx(50.0)
    assert aaa.loc[0, "volume_adj"] == pytest.approx(2000.0)

    if not qc_failures.empty:
        inconsistent = qc_failures.loc[
            (qc_failures["symbol"] == "AAA")
            & (qc_failures["failure_code"].astype(str).str.contains("RAW_ADJUSTED", case=False, na=False))
        ]
        assert inconsistent.empty
    assert str(result["qc"].summary["gate"]).lower() == "pass"



def test_fetch_revision_is_detected_and_downstream_uses_updated_snapshot(
    fetch_prices_module,
    adjust_prices_module,
    monkeypatch,
    tmp_path,
):
    universe_path = _write_csv(_universe_history(), tmp_path / "universe_history.csv")
    calendar_path = _write_csv(_calendar(), tmp_path / "market_calendar.csv")
    provider_cfg_path = _write_json(tmp_path / "provider_config.json", {"providers": ["primary"]})

    # First fetch run.
    _patch_fetch_runtime(monkeypatch, fetch_prices_module, tmp_path, _base_provider_payloads())
    first = fetch_prices_module.run_fetch_prices(
        universe_snapshot_path=universe_path,
        calendar_path=calendar_path,
        provider_config_path=provider_cfg_path,
        start_date="2024-01-02",
        end_date="2024-01-04",
        frequency="1d",
        fetch_mode="full_refresh",
        run_id="rev_fetch_1",
        asof_ts_utc="2026-03-15T12:00:00Z",
        config_path=None,
    )
    assert first.revisions.empty

    # Second fetch run with revised AAA close on the last day.
    monkeypatch.undo()
    _patch_fetch_runtime(monkeypatch, fetch_prices_module, tmp_path, _base_provider_payloads(revised_last_close=60.0))
    second = fetch_prices_module.run_fetch_prices(
        universe_snapshot_path=universe_path,
        calendar_path=calendar_path,
        provider_config_path=provider_cfg_path,
        start_date="2024-01-02",
        end_date="2024-01-04",
        frequency="1d",
        fetch_mode="reconcile",
        run_id="rev_fetch_2",
        asof_ts_utc="2026-03-15T12:00:00Z",
        config_path=None,
    )

    assert not second.revisions.empty
    revised_close = second.canonical.loc[
        (second.canonical["symbol"] == "AAA") & (pd.to_datetime(second.canonical["trade_date"]) == pd.Timestamp("2024-01-04")),
        "close",
    ].iloc[-1]
    assert revised_close == pytest.approx(60.0)

    adjust_cfg = adjust_prices_module.AdjustPricesConfig(
        reverse_split=adjust_prices_module.ReverseSplitPolicy(extreme_threshold=20.0),
        conflict=adjust_prices_module.ConflictPolicy(provider_priority=("corp_actions_primary",), same_day_precedence=("split", "cash_dividend", "special_cash_dividend")),
        adjustment=adjust_prices_module.AdjustmentPolicy(mode="dual_output", allow_empty_corporate_actions=True),
        output=adjust_prices_module.OutputPolicy(),
    )
    adjust_artifacts = adjust_prices_module.adjust_prices(
        _raw_for_adjust(second.raw),
        pd.DataFrame(),
        run_id="rev_adjust",
        asof_ts_utc="2026-03-15T12:00:00Z",
        start_date="2024-01-02",
        end_date="2024-01-04",
        config=adjust_cfg,
    )
    close_raw = adjust_artifacts.adjusted.loc[
        (adjust_artifacts.adjusted["symbol"] == "AAA") & (adjust_artifacts.adjusted["date"] == pd.Timestamp("2024-01-04")),
        "close_raw",
    ].iloc[0]
    assert close_raw == pytest.approx(60.0)



def test_qc_fail_gate_blocks_market_proxy_stage_when_required(
    fetch_prices_module,
    adjust_prices_module,
    qc_prices_module,
    market_proxies_module,
    monkeypatch,
    tmp_path,
):
    bad_raw = None
    clean_result = _run_price_stack(
        fetch_prices_module=fetch_prices_module,
        adjust_prices_module=adjust_prices_module,
        qc_prices_module=qc_prices_module,
        market_proxies_module=market_proxies_module,
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        provider_payloads=_base_provider_payloads(),
        corporate_actions=_corporate_actions_split(),
        qc_require_adjusted=True,
    )
    bad_raw = clean_result["fetch"].raw.copy()
    bad_idx = bad_raw.index[(bad_raw["symbol"] == "BBB") & (pd.to_datetime(bad_raw["trade_date"]) == pd.Timestamp("2024-01-03"))][0]
    bad_raw.loc[bad_idx, "low"] = 30.0  # impossible: low > high/open/close

    monkeypatch.undo()
    result = _run_price_stack(
        fetch_prices_module=fetch_prices_module,
        adjust_prices_module=adjust_prices_module,
        qc_prices_module=qc_prices_module,
        market_proxies_module=market_proxies_module,
        monkeypatch=monkeypatch,
        tmp_path=tmp_path / "bad_gate_case",
        provider_payloads=_base_provider_payloads(),
        corporate_actions=_corporate_actions_split(),
        qc_require_adjusted=True,
        qc_raw_override=bad_raw,
    )

    assert str(result["qc"].summary["gate"]).lower() == "fail"

    with pytest.raises(RuntimeError, match="QC gate must be pass"):
        _require_qc_pass_then_run_market(
            qc_artifacts=result["qc"],
            market_proxies_module=market_proxies_module,
            monkeypatch=monkeypatch,
            prices_path=result["paths"]["market_prices"],
            universe_path=result["paths"]["universe"],
            market_config_path=result["paths"]["market_config"],
            run_id="blocked_market_stage",
        )
