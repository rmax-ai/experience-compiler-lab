# Getting Started

## Install

```bash
git clone https://github.com/rmax-ai/experience-compiler-lab.git
cd experience-compiler-lab
uv sync
source .venv/bin/activate  # puts `exp` on PATH
exp version
```

## A full evolution pass (after M4)

```bash
exp run train                      # 1. execute dev tasks → experience/runs/
exp mine                           # 2. extract evidence → knowledge/
exp propose onboarding             # 3. candidate patch + provenance (candidate-01)
exp eval candidate-01              # 4. candidate vs baseline on validation
exp promote candidate-01           # 5. policy decision → history
exp evolve --iterations 10         # 1–5 repeated
exp compare                        # M5: four baseline configurations
exp matrix --models fake           # M6: cross-model transfer matrix
exp report                         # latest Markdown iteration report
```

## Reading the results

1. `results/proposals/` — every candidate with its decision, forever.
2. `results/reports/` — per-iteration Markdown (SPEC.md §17 shape).
3. `knowledge/index.yaml` — evidence inventory with run links.
4. `experience/runs/<run_id>.jsonl` — the raw trace; `exp inspect run_001`.

The provenance chain to verify in any report:
**failure → evidence → hypothesis → patch → evaluation → decision**.
