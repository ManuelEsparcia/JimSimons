# Edge Decision Report (MVP)

- `overall_research_status`: **WARN**
- `recommendation_next_step`: **improve_features_or_labels**
- `recommendation_rationale`: Some candidates beat dummy but fail promotion gates; improve labels/features before richer models.
- `best_candidate_overall`: `regression|all_features|fwd_ret_20d|h20|ridge_cv`

## Credibility Context
- validation_suite: `WARN` (leakage: `WARN`)
- pbo_cscv: `PASS`
- multiple_testing: `WARN`
- missing_inputs: none

## Candidate Counts
- total_candidates: 6
- promoted_candidates: 0
- candidates_beating_dummy: 1
- candidates_flagged_fragile: 6
