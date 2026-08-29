# PYTHON_ARCHITECTURE.md

Module boundaries and dependency direction.

## Package layout (flat packages under src/)

| Package | Responsibility | Depends on |
|---------|----------------|------------|
| `world` | deterministic enterprise world: `api.py` (tools), `state.py` (World model), fixtures | (none) |
| `traces` | `schema.py` (Trace/Action/Outcome/Metrics), `store.py` (JSONL + SQLite index) | world (for state snapshots) |
| `agent` | `executor.py` (tiny loop), `context.py` (execution-time context ONLY) | world, traces |
| `knowledge` | `miner.py` (evidence extraction), `schema.py` (KnowledgeRecord), `store.py` | traces |
| `skills` | `proposer.py` (patch gen), `loader.py`, `patch.py` (apply/verify) | knowledge |
| `evals` | `runner.py`, `graders.py` (deterministic), `metrics.py` | world, traces, skills |
| `experiments` | `evolution_loop.py`, baselines, matrix | everything above |
| `cli` | thin Typer dispatch | experiments, evals, knowledge, skills |

## Rules

1. **`agent` never imports `knowledge`.** This is the H3 mechanism enforced
   in code: the execution agent's context is assembled from skills + task +
   tool interfaces only.
2. **Learning components read, never write, `experience/`.** Raw traces are
   immutable after a run completes.
3. **`evals` is the only package that renders a decision.** Promotion policy
   lives in `evals/runner.py` or `experiments`; the proposer never decides.
4. **No imports from `tests/` or across top-level packages via relative
   paths** — absolute imports only.

## Store ownership

| Path | Owner (writer) |
|------|----------------|
| `experience/runs/` | `traces.store` (append-only after run close) |
| `knowledge/patterns/`, `knowledge/index.yaml` | `knowledge.store` (append-only) |
| `skills/**` | `skills.patch` (only via apply/rollback with PURPOSE.yaml) |
| `results/` | `experiments` (reports, matrices) |

## Context assembly (the experiment's core)

```python
# executor context — ONLY these inputs
context = ExecutionContext(
    system=SYSTEM_INSTRUCTIONS,      # fixed, versioned in git
    skill=loader.load("onboarding"), # active skill markdown
    task=scenario.description,
    tools=world.tool_specs(),        # name + JSON schema, no implementation hints
)
```

The Evidence Miner and Skill Proposer get `knowledge/` + historical traces
instead. If a code path lets knowledge reach the executor, that's a bug
against H3 — test it explicitly.
