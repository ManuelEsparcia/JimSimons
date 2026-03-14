# Week 8 Research Refinement

## Scope
Week 8 closes the first research loop around the H20 hypothesis (`fwd_ret_20d`, ridge baseline) without adding richer models.

The weekly closure orchestrates and consolidates:
- `feature_ablation`
- `label_horizon_ablation`
- `edge_decision_report`
- `improve_best_candidate`
- `refine_h20_features`
- `refine_h20_target`
- `h20_regime_diagnostics`
- `h20_regime_conditioned_refinement`

## Main outcome
- Global H20 edge remains fragile.
- Regime-conditioned evidence shows potential niches, but robustness is still limited.
- Week 8 recommendation is kept conservative: `improve_features_or_labels`.
- `should_try_richer_model_now` remains explicitly gated and typically `false` unless strict conditions are met.

## Runner
Run:

```bash
python simons_smallcap_swing/run_week8_research_refinement.py \
  --run-prefix week8_research_refinement
```

Optional useful overrides:
- `--data-root`
- `--model-dataset-path`
- `--h20-model-dataset-path`
- `--labels-forward-path`
- `--features-matrix-path`
- `--purged-cv-folds-path`
- `--validation-suite-summary-path`

## Week 8 manifest
Runner writes:

- `data/week8_research_refinement_manifest_<run_prefix>.json`

The manifest includes:
- execution statuses by step
- key output artifact paths
- `week8_final_recommendation`
- `global_h20_status`
- `regime_conditioned_h20_status`
- `should_try_richer_model_now`
- best global and regime-conditioned candidates (if available)

## Smoke test
Run:

```bash
python -m pytest simons_smallcap_swing/tests/test_week8_end_to_end.py -q
```

The smoke test validates:
- end-to-end runner execution on synthetic deterministic fixtures
- final manifest generation and consumable recommendation fields
- no erroneous promotion to richer model
- compatibility with prior-week core artifacts (row-count stability checks)

## Current limits (MVP)
- No new model class is introduced.
- No additional backtest/execution complexity is introduced in this closure.
- Regime niches are diagnostic evidence, not a production promotion gate.

## Recommended next phase
- Continue targeted feature/label refinement only in regimes with repeatable signal.
- Strengthen fold-level robustness requirements before any richer-model trial.
