# Week 7: Strong Validation / Anti-Overfitting (MVP Consolidation)

## Scope
Week 7 consolidates the validation stack introduced after Week 6 execution/backtest:

1. `validation/leakage_audit.py`
2. `validation/validation_suite.py`
3. `validation/pbo_cscv.py`
4. `validation/multiple_testing.py`
5. `run_week7_validation.py` (orchestrator)

## Improvements vs Week 6
- Adds formal leakage auditing for labels/features/dataset/splits.
- Adds unified status roll-up across leakage, CV robustness, signal quality and portfolio/backtest sanity.
- Adds MVP CSCV/PBO fragility estimate by task (regression/classification).
- Adds MVP multiple-testing control (Bonferroni + BH) with explicit heuristic mode when p-values are missing.
- Standardizes Week 7 outputs into auditable summary + findings artifacts.

## Week 7 Runner
Run the consolidated Week 7 flow:

```bash
python simons_smallcap_swing/run_week7_validation.py --run-prefix week7_validation
```

Useful flags:

- `--data-root <path>`
- `--labels-path`, `--features-path`, `--model-dataset-path`, `--trading-calendar-path`
- `--purged-splits-path`, `--purged-cv-folds-path`, `--fundamentals-pit-path`
- `--backtest-diagnostics-summary-path`
- `--cv-model-comparison-summary-path`
- `--decile-analysis-summary-path`
- `--paper-portfolio-summary-path`
- `--ridge-cv-path`, `--dummy-regressor-cv-path`, `--logistic-cv-path`, `--dummy-classifier-cv-path`
- `--candidate-tests-path`
- `--max-partitions` (default `64`)
- `--seed` (default `42`)
- `--alpha` (default `0.05`)
- `--output-dir <path>` (defaults to `data/validation`)

## Tests (minimum)

```bash
python -m pytest simons_smallcap_swing/tests/test_week7_end_to_end.py -q
python -m pytest simons_smallcap_swing/tests/test_leakage_audit_mvp.py simons_smallcap_swing/tests/test_validation_suite_mvp.py simons_smallcap_swing/tests/test_pbo_cscv_mvp.py simons_smallcap_swing/tests/test_multiple_testing_mvp.py simons_smallcap_swing/tests/test_week7_end_to_end.py -q
```

## Key Artifacts
- `data/validation/leakage_audit_findings.parquet`
- `data/validation/leakage_audit_metrics.parquet`
- `data/validation/leakage_audit_summary.json`
- `data/validation/validation_suite_findings.parquet`
- `data/validation/validation_suite_metrics.parquet`
- `data/validation/validation_suite_summary.json`
- `data/validation/pbo_cscv_results.parquet`
- `data/validation/pbo_cscv_partitions.parquet`
- `data/validation/pbo_cscv_summary.json`
- `data/validation/multiple_testing_results.parquet`
- `data/validation/multiple_testing_metrics.parquet`
- `data/validation/multiple_testing_summary.json`
- `data/week7_validation_manifest_<run_prefix>.json`

## Current MVP Limits
- No formal White Reality Check / SPA.
- No full promotion-gate automation.
- No advanced bootstrap inference or rich visualization.
- Multiple-testing layer is intentionally conservative when raw p-values are unavailable.

## Next Phase
- Tighten promotion gates on top of Week 7 summaries.
- Add stronger inferential layer (e.g., RC/SPA-style extensions).
- Extend validation reports for candidate approval/rejection workflows.
