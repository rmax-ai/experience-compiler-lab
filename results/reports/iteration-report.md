# Iteration report — onboarding

Seed: 7
Iterations run: 2

## Iteration 1

Training: 13 / 30 success
Observed patterns:
(none)

Candidate candidate-04:
2 lines modified
Validation:
v1 20.0% -> v2 20.0%
Regressions:
0
Decision:
REJECT


## Iteration 2

Training: 13 / 30 success
Observed patterns:
(none)

Candidate candidate-05:
2 lines modified
Validation:
v1 20.0% -> v2 20.0%
Regressions:
0
Decision:
REJECT


## Provenance chain

- Iteration 1: failures [run_036, run_037, run_038, run_040, run_042, run_045, run_046, run_047, run_048, run_049, run_051, run_052, run_053, run_054, run_056, run_057, run_062] -> evidence [] -> patch candidate-04 -> eval 0.20 -> 0.20 -> REJECTED
- Iteration 2: failures [run_086, run_087, run_088, run_090, run_092, run_095, run_096, run_097, run_098, run_099, run_101, run_102, run_103, run_104, run_106, run_107, run_112] -> evidence [] -> patch candidate-05 -> eval 0.20 -> 0.20 -> REJECTED


---
The held-out test set was not used anywhere in this evolution run.