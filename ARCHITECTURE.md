# ARCHITECTURE.md — experience-compiler-lab

## Problem statement

Can repeated agent failures and successes be compiled automatically into
validated procedural knowledge (skills) that measurably improves unseen-task
performance — more reliably than trace-only or memory-only approaches?
(SPEC.md §1)

## Design goals

1. **Observability** — every improvement must be traceable: which failures
   generated which knowledge, which knowledge changed which skill, which
   validation evidence justified promotion.
2. **Isolation** — learning-time knowledge never contaminates execution-time
   context (H3).
3. **Determinism** — in-process world, fixed seeds, deterministic graders.
4. **Reproducibility** — every experiment is a software build with a manifest
   of git versions.

## Component architecture

```
                    ┌─────────────────────┐
                    │   Task Dataset      │  datasets/*.jsonl
                    └─────────┬───────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ Execution Agent │  src/agent  (skills + task + tools ONLY)
                     └────────┬────────┘
                              │
                              ▼  immutable trajectories
                     ┌─────────────────┐
                     │ Evidence Miner  │  src/knowledge/miner.py
                     └────────┬────────┘
                              ▼
                       Knowledge Base   knowledge/ (append-only, index.yaml)
                              │
                              ▼
                     ┌─────────────────┐
                     │ Skill Proposer  │  src/skills/proposer.py (patches)
                     └────────┬────────┘
                              ▼  candidate patch + PURPOSE.yaml
                     ┌─────────────────┐
                     │ Eval Harness    │  src/evals  (deterministic graders)
                     └───────┬─────────┘
                         pass │ fail
                              ▼
                        promotion ledger  (proposal history, permanent)
```

## The three stores

| Store | Contents | Mutability | Location |
|-------|----------|-----------|----------|
| experience | raw run artifacts (traces, actions, outcomes) | append-only | `experience/runs/` |
| knowledge | structured interpretations with evidence links | append-only (supersede, never overwrite) | `knowledge/` + `knowledge/index.yaml` |
| skills | deployed procedural knowledge | mutable, reversible, patched with provenance | `skills/<workflow>/SKILL.md` + `PURPOSE.yaml` |

## Execution-time vs learning-time context

| Consumer | Receives |
|----------|----------|
| Execution agent | system instructions, active skills, task, tool interfaces |
| Evidence Miner | batches of trajectories (historical) |
| Skill Proposer | active skill, structured knowledge, proposal history, relevant traces |

## Data flow

1. `exp run train` — executor runs dev tasks against current skill; each run
   produces an immutable `Trace` (schema: docs/data-formats.md).
2. `exp mine` — Evidence Miner extracts candidate patterns; deterministic
   merger links them to supporting/counterexample run IDs; store updates
   `knowledge/` append-only.
3. `exp propose onboarding` — Skill Proposer emits a diff-style patch + a
   PURPOSE.yaml with `derived_from` evidence refs.
4. `exp eval candidate-N` — eval harness runs candidate vs baseline skill on
   the SAME validation set (never the dev set that generated the patch).
5. `exp promote candidate-N` — promotion policy (`candidate.success_rate >
   baseline.success_rate and regressions <= allowed`); decision recorded in
   proposal history; rejected proposals stay in history forever.

## Key design decisions

- **Tiny custom agent loop**, not ADK/LangGraph — experimental control over
  context construction is the point of the study (SPEC.md §18).
- **Minimal LLM adapter** (httpx + provider config) instead of LiteLLM —
  fewer moving parts, explicit cost capture per call.
- **Deterministic graders first** — final-state invariants, not LLM judgments
  (SPEC.md §15).
- **SQLite for indexes, JSONL for raw runs** — greppable, git-friendly,
  zero-ops.

## Trade-offs

| Choice | For | Against |
|--------|-----|---------|
| Synthetic world vs real SaaS APIs | perfect determinism, inspectable skills | results may not transfer to messy real tooling |
| YAML/Markdown artifacts vs DB | provenance readable in diffs | no concurrent multi-user access |
| Single model first | tight experimental control | cross-model effects deferred to M6 |
| No validation on dev set | candidate eval is uncontaminated | smaller eval pool per iteration |

## Risks

- Evolution loops that overfit the 10-task validation set (mitigate: fixed
  held-out test set touched only for final reporting).
- LLM-generated patches that are unparsable diffs (mitigate: strict patch
  grammar, deterministic apply/verify).
- Cost of repeated eval runs (mitigate: cost capture per run; dev-set size
  capped).
