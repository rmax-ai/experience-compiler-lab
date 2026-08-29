# AGENTS.md — experience-compiler-lab

**Project DNA.** This is a *research harness*, not a production agent. The
artifact that matters is the evolution record: failure → evidence → hypothesis
→ procedural modification → evaluation → accepted knowledge. Provenance beats
absolute scores.

## 1. Code Organisation

```
src/
  agent/        execution agent (executor.py, context.py) — tiny custom loop
  world/        deterministic enterprise world (api.py, state.py, fixtures/)
  traces/       trace schema + store (schema.py, store.py)
  knowledge/    evidence miner + structured knowledge store
  skills/       skill proposer, loader, patch application
  evals/        eval runner, deterministic graders, metrics
  experiments/  evolution loop + baseline/matrix experiments
experience/     IMMUTABLE run artifacts (append-only)
knowledge/      interpretations (append-only), patterns/, index.yaml
skills/         deployed procedural knowledge (mutable, reversible)
datasets/       train.jsonl, validation.jsonl, test.jsonl
results/        reports, transfer matrix
```

- Single responsibility per module; keep the CLI (`src/cli.py`) a thin dispatch
  layer over `agent`, `evals`, `experiments`.
- Absolute imports from the top-level packages (`from world.api import World`).
- Dependency direction: `experiments` → everything; `agent` → `world` only
  (execution-time context). Learning components (`knowledge`, `skills`) read
  traces but never mutate them.

## 2. The Three-Store Invariant (non-negotiable)

- `experience/` — immutable observations. Append-only. Never edited, never
  deleted.
- `knowledge/` — accumulated interpretations. Append-only records; a record
  may be superseded via `supersedes`, never overwritten.
- `skills/` — currently deployed procedural knowledge. Mutable and reversible;
  every change goes through a patch with PURPOSE.yaml provenance.

## 3. Execution-Time vs Learning-Time Separation (non-negotiable)

The execution agent receives ONLY: system instructions, active skills, task,
tool interfaces. The knowledge base goes to the Evidence Miner and Skill
Proposer only. This is the mechanism H3 exists to test — do not "helpfully"
leak knowledge into executor context.

## 4. Language Conventions

- Python ≥3.12, `uv` for env management. Pydantic v2 models for all schemas
  (use `model_validate_json`, set `extra` policy explicitly).
- Typer for CLI (`exp`). Pydantic + Typer: parse args into models at the
  boundary, keep models plain inside.
- Use `datetime.now(timezone.utc)`; never `datetime.utcnow()`.
- Type hints everywhere; `list[X] | None` style, no `Optional`.
- Ruff (line-length 100, rules E F I UP B SIM). Format before commit.

## 5. Determinism Rules

- The world is in-process and deterministic. Reset it before every task.
- Fixed seeds for dataset generation and any sampling. Record seed in the run
  manifest.
- Every experiment emits a manifest (SPEC.md §19) with git versions of
  dataset/skill/knowledge/environment.
- Deterministic graders are the primary oracle. LLM graders, if ever added,
  are secondary and must warn, not fail (stochastic).

## 6. Testing

- pytest. One test file per module. Fixtures: fresh `World` per test.
- Deterministic invariants only in assertions. Never assert on LLM output
  quality in CI; assert structure, warn on semantics.
- Mock the LLM adapter in unit tests (`FakeModel` with scripted responses);
  live API calls only behind env-gated integration tests.
- Verification gate before commit: `uv sync && uv run pytest -q && uvx ruff
  check src/ tests/`. Never chain `pytest | tail && git commit` — pipes mask
  exit codes.

## 7. Dependencies

- Minimal. SQLite (stdlib) + JSONL for stores, PyYAML for artifacts, Jinja2
  for reports, Pydantic, Typer, httpx for the LLM adapter.
- DO NOT add vector DBs, embeddings, K8s, MCP, distributed exec, or knowledge
  graphs before M5. They add surface, not evidence.
- New dep requires a DECISIONS.md entry.

## 8. Data Formats

Traces, knowledge records, skills, PURPOSE.yaml, and scenario JSONL are
contracts shared by all components — defined in `docs/data-formats.md`.
Changing a format is a breaking change: bump the version field and update the
doc first.

## 9. Hygiene

- Never copy real machine values (Telegram IDs, chat IDs, API keys, prices)
  into fixtures, datasets, or docs — synthetic enterprise data only. A
  pre-commit hook (`.githooks/`) blocks known leak patterns.
- Secrets via `.envrc` + direnv; never in code, prompts, or committed files.
- LLM API costs must be recorded per run in the trace metrics (token counts,
  estimated cost) — cost evidence in the store, not just logs.

## 10. Git / Workflow

- HTTPS remotes only (no SSH keys on this box). `gh` credential helper.
- One commit per story, push after verification. If a push is rejected,
  `git pull --rebase` then push.
- Commit messages: `<area>: <imperative summary>`.

## 11. References

- SPEC.md — ground-truth specification (cite sections in commits)
- docs/ARCHITECTURE.md, docs/DECISIONS.md, docs/ROADMAP.md (root-level)
- docs/data-formats.md, docs/cli-reference.md
- PYTHON_DEVELOPMENT.md, PYTHON_ARCHITECTURE.md
