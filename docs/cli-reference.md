# CLI Reference

Console script: `exp` (Typer). Thin dispatch over `src/cli.py`.

```
exp version
exp run <train|validation|test> [--limit N] [--seed INT] [--model NAME] [--workflow NAME]
exp mine [--since-run <id>] [--limit N] [--model NAME]
exp propose <workflow> [--model NAME] [--traces N]
exp eval <candidate-id> [--model NAME] [--seed INT] [--allowed-regressions N]
exp promote <candidate-id> [--model NAME] [--seed INT] [--allowed-regressions N]
exp evolve --iterations N [--dev-limit N] [--model NAME] [--seed INT] [--workflow NAME]
exp compare [--configs baseline,trace2skill,memory,compiler] [--iterations N]
            [--dev-limit N] [--model NAME] [--seed INT] [--workflow NAME]
exp matrix --models A,B [--iterations N] [--seed INT] [--dev-limit N] [--workflow NAME]
exp inspect <run_<id>|pattern-id|candidate-id>
exp report [--path FILE]
```

With no `EXP_LLM_API_KEY`, every command defaults to `--model fake` — the
deterministic scripted model (no network, no cost). Real runs set
`EXP_LLM_BASE_URL` + `EXP_LLM_API_KEY` (see DECISIONS.md D-09).

Still planned (not yet implemented):
- `--skill-version INT` — pin a skill version for a run
- `--dry-run` — proposer/eval without persisting candidates

Outputs land in `results/` and `experience/runs/`; nothing prints to stdout
except status lines — reports are files (Markdown/CSV), not terminal art.

See docs/data-formats.md for all artifacts. Every command ships full
`exp <cmd> --help` (M0–M6 complete).
