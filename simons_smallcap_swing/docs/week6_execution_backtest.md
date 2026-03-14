# Week 6: Portfolio Construction + Execution + Costs + Backtest + Diagnostics (MVP)

## Scope
Week 6 consolidates the full execution chain built on top of Week 5 signals:

1. `portfolio/construct_portfolio.py`
2. `execution/assumptions.py`
3. `execution/cost_model.py`
4. `backtest/engine.py`
5. `backtest/diagnostics.py`
6. `run_week6_execution_backtest.py` (orchestrator)

## Improvements vs Week 5
- Moves from paper portfolio evaluation to explicit holdings/rebalance artifacts.
- Makes execution timing explicit (`signal_date`, `execution_date`, `cost_date`).
- Applies simple, auditable costs to rebalance activity.
- Produces daily gross/net backtest outputs with explicit cost accounting.
- Adds diagnostics summary by `portfolio_mode` with cost drag and drawdown view.

## Week 6 Runner
Run the consolidated Week 6 pipeline:

```bash
python simons_smallcap_swing/run_week6_execution_backtest.py --run-prefix week6_execution_backtest
```

Useful flags:

- `--data-root <path>`
- `--signals-path <path>`, `--universe-history-path <path>`
- `--trading-calendar-path <path>`, `--adjusted-prices-path <path>`
- `--model-name` (default `ridge_baseline`), `--label-name` (default `fwd_ret_5d`)
- `--split-name`, `--horizon-days`
- `--split-roles valid,test`
- `--portfolio-modes long_only_top_n,long_short_top_bottom_n`
- `--top-n`, `--bottom-n`
- `--execution-delay-sessions` (default `1`)
- `--fill-assumption` (default `full_fill`)
- `--cost-timing` (default `apply_on_execution_date`)
- `--cost-bps-per-turnover`, `--entry-bps`, `--exit-bps`

## Tests (minimum)

```bash
python -m pytest simons_smallcap_swing/tests/test_week6_end_to_end.py -q
python -m pytest simons_smallcap_swing/tests/test_construct_portfolio_mvp.py simons_smallcap_swing/tests/test_execution_assumptions_mvp.py simons_smallcap_swing/tests/test_cost_model_execution_timing_mvp.py simons_smallcap_swing/tests/test_backtest_engine_execution_timing_mvp.py simons_smallcap_swing/tests/test_backtest_diagnostics_mvp.py simons_smallcap_swing/tests/test_week6_end_to_end.py -q
```

## Key Artifacts
- `data/portfolio/portfolio_holdings.parquet`
- `data/portfolio/portfolio_rebalance.parquet`
- `data/portfolio/portfolio_summary.json`
- `data/execution/execution_holdings.parquet`
- `data/execution/execution_rebalance.parquet`
- `data/execution/execution_assumptions_summary.json`
- `data/execution/costs_positions.parquet`
- `data/execution/costs_daily.parquet`
- `data/execution/costs_summary.json`
- `data/backtest/backtest_daily.parquet`
- `data/backtest/backtest_contributions.parquet`
- `data/backtest/backtest_summary.json`
- `data/backtest/backtest_diagnostics_daily.parquet`
- `data/backtest/backtest_diagnostics_by_mode.parquet`
- `data/backtest/backtest_diagnostics_summary.json`
- `data/week6_execution_backtest_manifest_<run_prefix>.json`

## Current MVP Limits
- No slippage/impact model upgrade in this sprint.
- No borrow enhancement in this sprint.
- No optimizer/risk model portfolio construction.
- No visual tearsheet or advanced statistical diagnostics.

## Next Phase
- Add richer execution realism (slippage/impact extensions).
- Add more robust backtest diagnostics/reporting layer.
- Prepare consolidation closeout for Week 6 and handoff to Week 7.

