# Week 3: Modeling Baselines (MVP)

## Scope
Week 3 closes the first end-to-end modeling baseline flow on top of Weeks 1-2 data artifacts:

1. `labels/build_labels.py`
2. `features/build_features.py`
3. `labels/purged_splits.py`
4. `datasets/build_model_dataset.py`
5. `models/baselines/train_ridge.py`
6. `models/baselines/train_logistic.py`
7. `models/baselines/train_dummy_baselines.py`
8. `models/baselines/run_baseline_benchmarks.py`

## Improvements vs Weeks 1-2
- Adds PIT-safe forward labels and feature matrix for modeling.
- Adds purge/embargo temporal splitting for leakage control.
- Adds trainable baseline models (ridge, logistic) and dummy baselines.
- Adds homogeneous baseline benchmark suite by task:
  - regression: ridge vs dummy_regressor
  - classification: logistic vs dummy_classifier

## Week 3 Runner
Use the Week 3 consolidator:

```bash
python simons_smallcap_swing/run_week3_modeling_baselines.py --run-prefix week3_mvp
```

Optional flags:

- `--data-root <path>`: run on an alternate data root.
- `--disable-binary-direction-labels`: skip generation of `fwd_dir_up_*`.
- `--allow-missing-binary-label`: do not fail if classification path cannot be built.
- `--benchmark-strict-missing`: fail if benchmark artifacts are missing.

## Tests (minimum)

```bash
python -m pytest simons_smallcap_swing/tests/test_week3_end_to_end.py -q
python -m pytest simons_smallcap_swing/tests/test_train_ridge_baseline_mvp.py simons_smallcap_swing/tests/test_train_logistic_baseline_mvp.py simons_smallcap_swing/tests/test_train_dummy_baselines_mvp.py simons_smallcap_swing/tests/test_model_comparison_mvp.py simons_smallcap_swing/tests/test_run_baseline_benchmarks_mvp.py simons_smallcap_swing/tests/test_week3_end_to_end.py -q
```

## Key Artifacts
- `data/labels/labels_forward.parquet`
- `data/features/features_matrix.parquet`
- `data/labels/purged_splits.parquet`
- `data/datasets/regression/model_dataset.parquet`
- `data/datasets/classification/model_dataset.parquet`
- `data/models/artifacts/ridge_baseline_metrics.json`
- `data/models/artifacts/logistic_baseline_metrics.json`
- `data/models/artifacts/dummy_regressor_metrics.json`
- `data/models/artifacts/dummy_classifier_metrics.json`
- `data/models/artifacts/baseline_benchmark_summary.json`
- `data/week3_modeling_baselines_manifest_<run_prefix>.json`

## Current MVP Limits
- Baselines only (no advanced model families).
- No multi-fold purged CV orchestration yet.
- No trading/backtest integration in this stage.
- Benchmark suite compares only homogeneous tasks (no cross-task ranking).

## Next Phase
- Week 3 close-out runner/smoke already in place.
- Next natural step: consolidated model evaluation/ranking layer on top of baseline artifacts, then controlled integration into validation/backtest layers.
