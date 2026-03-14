# Week 4: Purged Multi-Fold CV Validation (MVP)

## Scope
Week 4 consolidates temporal validation with purged multi-fold CV for baseline models:

1. `labels/purged_cv.py`
2. `models/baselines/cross_validated_baselines.py`
3. `models/baselines/cv_model_comparison.py`
4. `run_week4_cv_validation.py` (orchestrator)

## Improvements vs Week 3
- Moves from single holdout purged split to multi-fold purged CV.
- Adds homogeneous fold-by-fold baseline evaluation for:
  - `ridge_cv` vs `dummy_regressor_cv`
  - `logistic_cv` vs `dummy_classifier_cv`
- Adds robust CV comparison artifacts with explicit comparability checks.
- Keeps PIT/no-leakage discipline with embargo and dropped-role exclusion.

## Week 4 Runner
Run the consolidated Week 4 pipeline:

```bash
python simons_smallcap_swing/run_week4_cv_validation.py --run-prefix week4_cv_mvp
```

Optional flags:

- `--data-root <path>`: run on an alternate data root.
- `--model-dataset-path <path>`: override default `data/datasets/model_dataset.parquet`.
- `--n-folds <int>`: purged CV folds (default `5`).
- `--embargo-sessions <int>`: embargo after valid block (default `1`).
- `--regression-label-name <name>` / `--classification-label-name <name>`.
- `--horizon-days <int>`: label horizon filter for CV baselines.
- `--ridge-alphas <csv>` / `--logistic-cs <csv>`.
- `--dummy-regressor-strategy mean|median`.
- `--dummy-classifier-strategy prior|majority`.
- `--fail-on-invalid-fold`: fail-fast if any fold has no train/valid.
- `--comparison-strict-missing`: fail if comparison inputs are missing.

## Tests (minimum)

```bash
python -m pytest simons_smallcap_swing/tests/test_week4_end_to_end.py -q
python -m pytest simons_smallcap_swing/tests/test_purged_cv_mvp.py simons_smallcap_swing/tests/test_cross_validated_baselines_mvp.py simons_smallcap_swing/tests/test_cv_model_comparison_mvp.py simons_smallcap_swing/tests/test_week4_end_to_end.py -q
```

## Key Artifacts
- `data/labels/purged_cv_folds.parquet`
- `data/labels/purged_cv_folds.summary.json`
- `data/models/artifacts/ridge_cv/cv_baseline_fold_metrics.parquet`
- `data/models/artifacts/logistic_cv/cv_baseline_fold_metrics.parquet`
- `data/models/artifacts/dummy_regressor_cv/cv_baseline_fold_metrics.parquet`
- `data/models/artifacts/dummy_classifier_cv/cv_baseline_fold_metrics.parquet`
- `data/models/artifacts/cv_model_comparison_summary.json`
- `data/models/artifacts/cv_model_comparison_table.parquet`
- `data/week4_cv_validation_manifest_<run_prefix>.json`

## Current MVP Limits
- No statistical significance testing on fold deltas.
- No nested CV / CPCV.
- No advanced model families beyond baseline + dummy.
- No direct backtest/portfolio coupling in this stage.

## Next Phase
- Consolidated Week 4 closeout is ready for:
  - richer CV model ranking/selection policy,
  - optional significance layer on fold deltas,
  - controlled integration into downstream strategy validation.
