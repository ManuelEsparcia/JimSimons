# Week 5 Results Snapshot

Generated from current artifacts to create a reproducible checkpoint before Week 6.

## Holdout Baselines vs Dummy
- Regression (`fwd_ret_5d`):
  - primary metric: `mse` (lower is better)
  - ridge valid/test: 0.0003359327886724925 / 0.0002998037724843384
  - dummy valid/test: 0.00018778261188234373 / 0.000119761499506244
  - winner valid/test: model_b / model_b
- Classification (`fwd_dir_up_5d`):
  - primary metric: `log_loss` (lower is better)
  - logistic valid/test: 1.1244412793440988 / 1.0015414871810622
  - dummy valid/test: 0.7848800582635733 / 0.6978861680260375
  - winner valid/test: model_b / model_b

## CV Comparison (Week 4)
- Regression: status=comparable, metric=mse, mean_delta=4.69912263640446e-05, winner=model_b
- Classification: status=comparable, metric=log_loss, mean_delta=0.11822010572425579, winner=model_b

## Signal Deciles
- label: `fwd_ret_5d`
- n_dates: 22
- mean top-bottom spread: 0.0014238264906728576
- positive spread rate: 0.5909090909090909
- monotonicity score: 0.02468808224617116

## Paper Portfolio Gross Returns
- `long_only_top`: mean=0.0013522261301768964, median=-0.0010681349514373162, std=0.01336456741216384, positive_rate=0.45454545454545453, n_dates=22
- `long_short_top_bottom`: mean=0.0014238264906728576, median=0.003194129335434559, std=0.016392701450718736, positive_rate=0.5909090909090909, n_dates=22

## Files
- JSON snapshot: `simons_smallcap_swing/data/week5_results_snapshot.json`
- Sources:
  - `simons_smallcap_swing/data/models/artifacts/baseline_benchmark_summary.json`
  - `simons_smallcap_swing/data/models/artifacts/cv_model_comparison_summary.json`
  - `simons_smallcap_swing/data/signals/decile_analysis_summary.json`
  - `simons_smallcap_swing/data/signals/paper_portfolio_summary.json`
