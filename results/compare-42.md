# M5 compare

Seed: 42

| config | dev | validation | heldout | tool_calls | tokens | cost | skill_version | decisions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | - | - | 0.00% | 0.00 | 0 | 0.000000 | 0 |  |
| trace2skill | 100.00% | 20.00% | 0.00% | 0.00 | 0 | 0.000000 | 1 | rejected |
| memory | 100.00% | - | 0.00% | 0.00 | 0 | 0.000000 | 1 |  |
| compiler | 100.00% | 20.00% | 0.00% | 0.00 | 0 | 0.000000 | 1 | rejected |

## Hypotheses

- H1: compare Trace2Skill with Compiler to test whether structured knowledge improves proposals.
- H2: compare Compiler with Baseline on held-out success to test whether evolved skills generalize.
- H3: compare Memory with Compiler to test whether keeping knowledge out of executor context yields more reusable skills.

This run shows deterministic outcomes for these configurations and seed. It cannot establish statistical significance, model transfer, or generalization beyond this simulator and scenario set.
