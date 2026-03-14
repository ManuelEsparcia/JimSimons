# Week 5: Signals + Paper Portfolio (MVP)

## Scope
Week 5 consolidates the signal-to-selection layer with three modules:

1. `signals/build_signals.py`
2. `signals/decile_analysis.py`
3. `signals/paper_portfolio.py`
4. `run_week5_signals_paper.py` (orchestrator)

## Improvements vs Week 4
- Adds a reproducible bridge from model predictions to cross-sectional scores.
- Adds bucket/decile diagnostics with top-minus-bottom spread.
- Adds simple equal-weight paper portfolios (`long_only_top`, `long_short_top_bottom`).
- Keeps PIT/no-leakage semantics by consuming OOS predictions and forward labels.

## Week 5 Runner
Run the consolidated Week 5 pipeline:

```bash
python simons_smallcap_swing/run_week5_signals_paper.py --run-prefix week5_signals_paper
```

Useful flags:

- `--data-root <path>`: run on alternate data root.
- `--predictions-path <path>`: explicit predictions artifact.
- `--universe-history-path <path>` / `--labels-path <path>`.
- `--model-name` (default `ridge_baseline`) and `--label-name` (default `fwd_ret_5d`).
- `--split-name`, `--horizon-days`.
- `--split-roles valid,test`.
- `--n-buckets`, `--top-buckets`, `--bottom-buckets`.
- `--portfolio-modes long_only_top,long_short_top_bottom`.
- `--disable-universe-filter` (default filter is enabled).

## Tests (minimum)

```bash
python -m pytest simons_smallcap_swing/tests/test_week5_end_to_end.py -q
python -m pytest simons_smallcap_swing/tests/test_build_signals_mvp.py simons_smallcap_swing/tests/test_decile_analysis_mvp.py simons_smallcap_swing/tests/test_paper_portfolio_mvp.py simons_smallcap_swing/tests/test_week5_end_to_end.py -q
```

## Key Artifacts
- `data/signals/signals_daily.parquet`
- `data/signals/signals_daily.summary.json`
- `data/signals/decile_daily.parquet`
- `data/signals/decile_summary.parquet`
- `data/signals/decile_analysis_summary.json`
- `data/signals/paper_portfolio_daily.parquet`
- `data/signals/paper_portfolio_positions.parquet`
- `data/signals/paper_portfolio_summary.json`
- `data/week5_signals_paper_manifest_<run_prefix>.json`

## Current MVP Limits
- No costs/slippage/borrow assumptions.
- No turnover model or rebalancing frictions.
- No optimizer/risk-model constraints.
- No full portfolio backtest in this stage.

## Next Phase
- Extend to top-N selection variants and richer portfolio diagnostics.
- Add turnover/cost layers before full strategy backtesting.
- Compare signals/portfolios across models in a homogeneous evaluation layer.
