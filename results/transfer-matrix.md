# M6 cross-model transfer matrix

## Transfer matrix

Held-out success rate per cell (rows = skill source, columns = executor model).

| skill source \ executor | fake |
| --- | --- |
| fake | 0.00% |

## Per-cell cost

| skill_source | executor_model | cost_usd |
| --- | --- | --- |
| fake | fake | 0.018830 |

Grand total cost: 0.018830 USD

## Reproducibility

- seed: 42
- iterations: 1
- dataset git version: f99aeb5a30e7051cbc2e08355a38e45d1ac365ee
- models: fake

Cell seed = seed + source_index * 100 + executor_index; per-cell costs
include the source model's phase-1 training runs and the cell's
held-out execution traces.
