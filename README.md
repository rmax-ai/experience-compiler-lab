# Experience Compiler Lab

**Can agent experience be compiled into validated procedural knowledge?**

A small research harness that turns agent execution traces into persistent
evidence, extracts reusable patterns, proposes skill patches, validates them
independently, and promotes only improvements.

```
execution traces → evidence → persistent knowledge → candidate patch
                 → independent evaluation → promotion / rejection
```

Three stores, one invariant:
`experience/` (immutable observations) · `knowledge/` (append-only
interpretations) · `skills/` (mutable, reversible, provenance-tracked).

## Research questions

1. Can repeated agent failures/successes be compiled into useful procedural
   knowledge?
2. Does persistent structured knowledge beat raw recent traces as skill-proposal
   input?
3. Can candidate changes be validated without contaminating evaluation?
4. Do evolved skills transfer across models, or encode model-specific
   compensation?

## Status

PoC under active development. See ROADMAP.md and the GitHub board.

## Quickstart

```bash
uv sync
source .venv/bin/activate  # puts `exp` on PATH

exp version
exp run train              # execute dev tasks against the active skill
exp mine                   # extract evidence into knowledge/
exp propose onboarding     # generate a candidate patch (first id: candidate-01)
exp eval candidate-01      # evaluate candidate vs baseline on validation set
exp promote candidate-01   # apply promotion policy
exp evolve --iterations 10
exp compare                # baseline configurations (M5)
exp matrix --models fake   # cross-model transfer matrix (M6)
exp inspect run_001        # inspect a single run
exp report                 # latest iteration report
```

Without `EXP_LLM_API_KEY` set, every command runs deterministically against
the scripted fake model (no network, no cost).

## Architecture

See ARCHITECTURE.md for the component diagram and data flows,
docs/data-formats.md for the contracts (traces, knowledge records, skills,
scenarios), docs/DECISIONS.md for rationale.

## Documentation index

| Doc | Purpose |
|-----|---------|
| SPEC.md | ground-truth specification (verbatim kickoff) |
| ARCHITECTURE.md | components, stores, data flow, trade-offs |
| DECISIONS.md | decision log with rejections |
| ROADMAP.md | milestones v0.1.0 → v0.2.0 |
| AGENTS.md | conventions for coding agents |
| docs/data-formats.md | schema contracts |
| docs/cli-reference.md | CLI commands |
| docs/getting-started.md | walkthrough |
| PYTHON_DEVELOPMENT.md, PYTHON_ARCHITECTURE.md | Python idioms & layout |

## License

MIT — see LICENSE.
