# Week 2 Data Upgrade Consolidation

## Scope Covered
Week 2 consolidates data-layer upgrades on top of Week 1 baseline:
- `fetch_prices.py` v2 (local-first with synthetic fallback)
- `corporate_actions.py` canonical MVP
- `adjust_prices.py` v2 (`split_only`)
- `qc_prices.py` v2 (stronger structural/economic checks)
- `survivorship.py` MVP (PIT vs naive comparison)
- `market_proxies.py` MVP
- `fetch_submissions.py` MVP (local-first/cacheable)
- `fetch_companyfacts.py` MVP (local-first/cacheable)
- `point_in_time.py` v2 (companyfacts/submissions-backed PIT)
- `edgar_qc.py` v2

## What Improved vs Week 1
- Prices moved from purely synthetic-only baseline to a local-first ingestion architecture.
- Corporate actions became a canonical artifact used by `adjust_prices` split-only logic.
- Price QC now validates split consistency, coverage, and stronger diagnostics.
- EDGAR pipeline now supports raw submissions/companyfacts ingestion with cache reuse.
- Fundamentals PIT v2 uses real/semi-real raw EDGAR inputs when available.
- Survivorship and market context artifacts are now generated explicitly.

## Unified Runner (Week 2)
Entrypoint:
- `run_week2_data_upgrade.py`

Execution order:
1. fetch prices v2
2. build corporate actions
3. adjust prices v2 split-only
4. run price QC v2
5. run survivorship analysis
6. build market proxies
7. fetch submissions
8. fetch companyfacts
9. build fundamentals PIT v2
10. run EDGAR QC v2

### Prerequisite
Week 2 runner expects Week 1 baseline artifacts in the selected `data_root`:
- `reference/trading_calendar.parquet`
- `reference/ticker_history_map.parquet`
- `universe/universe_history.parquet`
- `universe/universe_current.parquet`
- `edgar/ticker_cik_map.parquet`

If they are missing, run:
```bash
cd simons_smallcap_swing
python run_week1_mvp.py --run-prefix week1_baseline
```

### Run Command
```bash
cd simons_smallcap_swing
python run_week2_data_upgrade.py \
  --run-prefix week2_final \
  --submissions-ingestion-mode local_file \
  --submissions-local-source data/edgar/source/submissions \
  --companyfacts-ingestion-mode local_file \
  --companyfacts-local-source <path_to_local_companyfacts_source> \
  --force-rebuild-edgar
```

Optional flags:
- `--data-root <path>`
- `--price-ingestion-mode {auto,local_file,synthetic_fallback,provider_stub}`
- `--allow-edgar-remote` (optional, not required for tests)
- `--allow-fail-gates`
- `--enforce-survivorship-gate`

Final manifest:
- `data/week2_data_upgrade_manifest_<run-prefix>.json`

## Smoke Test (Week 2)
Test file:
- `tests/test_week2_end_to_end.py`

Run:
```bash
cd simons_smallcap_swing
python -m pytest -q tests/test_week2_end_to_end.py -p no:cacheprovider
```

## Key Artifacts Produced
- Universe:
  - `data/universe/corporate_actions.parquet`
  - `data/universe/audit/<run_id>/survivorship_*.{parquet,json}`
- Price:
  - `data/price/raw_prices.parquet`
  - `data/price/adjusted_prices.parquet`
  - `data/price/qc/<run_id>/...`
  - `data/price/market_proxies.parquet`
- EDGAR:
  - `data/edgar/submissions_raw.parquet`
  - `data/edgar/companyfacts_raw.parquet`
  - `data/edgar/fundamentals_pit.parquet`
  - `data/edgar/qc/<run_id>/...`

## Current MVP Limits
- No full `parse_xbrl`/semantic arbitration pipeline yet.
- Companyfacts/submissions remain local-first MVP, not full SEC historical ingestion.
- Price adjustments are `split_only` (no dividends/total-return engine yet).
- Survivorship and market proxies are diagnostic MVP layers (not regime modeling).

## Next Phase (After Week 2)
- EDGAR canonicalization depth (`parse_xbrl`/filing-quality flags).
- Corporate actions and price adjustment expansion beyond split-only.
- Data-layer hardening for multi-source reconciliation and stricter production gates.
