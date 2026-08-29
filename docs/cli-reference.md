# CLI Reference

Console script: `exp` (Typer). Thin dispatch over `src/cli.py`.

```
exp run train [--split dev|validation|test] [--iterations N]
exp mine [--since-run <id>]
exp propose <workflow>
exp eval <candidate-id>
exp promote <candidate-id> [--allow-regressions N]
exp evolve --iterations N
exp compare [--configs baseline,trace2skill,memory,compiler]
exp matrix --skill-source '*' --executor-model '*'
exp inspect <run-id|pattern-id|proposal-id>
exp report [--iteration N]
```

Planned flags (stable across milestones):
- `--seed INT` — override the run seed
- `--model NAME` — execution model override (matrix experiments)
- `--skill-version INT` — pin a skill version for a run
- `--dry-run` — proposer/eval without persisting candidates

Outputs land in `results/` and `experience/runs/`; nothing prints to stdout
except status lines — reports are files (Markdown/CSV), not terminal art.

See docs/data-formats.md for all artifacts. Full `exp <cmd> --help` after M4.
