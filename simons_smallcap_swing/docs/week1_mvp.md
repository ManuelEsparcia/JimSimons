# Week 1 MVP Consolidation

## Scope Covered
Week 1 closes a reproducible MVP data backbone with PIT-safe contracts:
- `simons_core` base contracts, schema checks, calendar, paths, parquet I/O, logging
- Reference data build (`data/reference`)
- Universe build + QC (`data/universe`)
- Price pipeline (raw + adjusted passthrough) + QC (`data/price`)
- EDGAR identity + fundamentals PIT + QC (`data/edgar`)

No advanced business logic was introduced beyond the MVP contracts and QC gates.

## Unified Runner
Single orchestration entrypoint:
- `run_week1_mvp.py`

It executes, in this exact order:
1. Build reference data
2. Build universe
3. Run universe QC
4. Build raw prices
5. Build adjusted prices
6. Run price QC
7. Build ticker/CIK map
8. Build fundamentals PIT
9. Run EDGAR QC

### Run Command
```bash
cd simons_smallcap_swing
python run_week1_mvp.py --run-prefix week1_final
```

Optional:
```bash
python run_week1_mvp.py --run-prefix week1_final --data-root data
```

The runner writes a final manifest:
- `data/week1_mvp_manifest_<run-prefix>.json`

## Smoke Test
End-to-end smoke test:
- `tests/test_week1_end_to_end.py`

Run:
```bash
cd simons_smallcap_swing
python -m pytest -q tests/test_week1_end_to_end.py -p no:cacheprovider
```

## Minimal Test Suite for Week 1
```bash
cd simons_smallcap_swing
python -m pytest -q \
  tests/test_data_pit.py \
  tests/test_reference_data.py \
  tests/test_universe_pit.py \
  tests/test_price_pipeline.py \
  tests/test_edgar_pit.py \
  tests/test_week1_end_to_end.py \
  -p no:cacheprovider
```

## Key Artifacts Produced
- Reference:
  - `data/reference/trading_calendar.parquet`
  - `data/reference/ticker_history_map.parquet`
  - `data/reference/symbols_metadata.parquet`
  - `data/reference/sector_industry_map.parquet`
- Universe:
  - `data/universe/universe_history.parquet`
  - `data/universe/universe_current.parquet`
  - `data/universe/qc/<run_id>/...`
- Price:
  - `data/price/raw_prices.parquet`
  - `data/price/adjusted_prices.parquet`
  - `data/price/qc/<run_id>/...`
- EDGAR:
  - `data/edgar/ticker_cik_map.parquet`
  - `data/edgar/fundamentals_events.parquet`
  - `data/edgar/fundamentals_pit.parquet`
  - `data/edgar/qc/<run_id>/...`

## Known MVP Limits
- Reference/universe/price/fundamentals datasets are deterministic MVP datasets, not full production feeds.
- `adjust_prices.py` is explicit passthrough (`passthrough_mvp`), no corporate-action adjustment engine yet.
- EDGAR layer is PIT-safe but intentionally synthetic and compact.
- No integration yet with features/models/portfolio/backtest production workflows.

## Next Phase (Week 2+)
- Replace synthetic data sources with validated real ingestion slices.
- Expand corporate-actions logic and adjusted-price semantics.
- Increase EDGAR filing coverage and canonicalization depth.
- Promote QC thresholds from MVP defaults to research-grade gates.
