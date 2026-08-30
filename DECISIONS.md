# DECISIONS.md — experience-compiler-lab

Log of non-obvious decisions. Add a row whenever a new choice is made.

## D-01 — Custom agent loop, no ADK/LangGraph
**Chosen:** minimal custom executor loop (~200 LOC) calling the LLM adapter.
**Rejected:** Google ADK, LangGraph. Frameworks introduce context-construction
semantics the experiment must control precisely. SPEC.md §18: "study
experience → knowledge → skill evolution, not framework semantics."

## D-02 — Minimal internal LLM adapter, no LiteLLM
**Chosen:** httpx-based adapter with provider config (OpenAI-compatible
endpoints), explicit token/cost capture per call into trace metrics.
**Rejected:** LiteLLM — fine, but the adapter needs per-call cost accounting
and zero-magic context assembly; a 100-line adapter gives both.

## D-03 — Python ≥3.12 (spec said 3.13)
Spec §18 says Python 3.13. Repo declares `requires-python = ">=3.12"`: the
lab box runs 3.12, the code targets no 3.13-only features, and uv can run
either. Relaxing the floor keeps CI options open; nothing depends on 3.13.

## D-04 — Literal spec directory layout
`src/` contains flat top-level packages (`agent`, `world`, `traces`,
`knowledge`, `skills`, `evals`, `experiments`) per SPEC.md §4, plus `src/cli.py`
as the thin Typer entrypoint (`exp`). Hatchling `packages` list pins them.
Slight packaging awkwardness accepted for spec fidelity.

## D-05 — `exp` console script name
Spec §16 uses `exp ...` verbatim. `exp` is generic but scoped to the project
venv. Kept as specified.

## D-06 — Public repo, MIT license
rmax-ai org default (public, MIT). Reversible via `gh repo edit --visibility`.
Flagged in the Phase 0 boundary report per fp workflow.

## D-07 — Dataset split: 50 scenarios = 30 dev / 10 validation / 10 held-out
Spec §10 requires development / validation / held-out separation; §22 asks
for 30–50 scenarios. Chosen: 50 total, deterministic graders per scenario.
Held-out test set is touched only for final reports (never during evolution).

## D-08 — Stores: SQLite for indexes, JSONL for raw runs
`experience/runs/*.jsonl` are the immutable raw artifacts (greppable, git-
friendly). SQLite (`knowledge/` index + run index) accelerates lookups only;
the JSONL/YAML files are the source of truth, the DB is a rebuildable cache.

## D-09 — Default execution model: DeepSeek V4 Flash (BYOK) via env config
Default provider for PoC runs; configurable via env (`EXP_LLM_PROVIDER`,
`EXP_LLM_MODEL`, `EXP_LLM_API_KEY`). Cross-model matrix (M6) switches the
same adapter. No keys committed; `.envrc` + direnv.

## D-10 — Phase 1 (research) skipped
The kickoff spec is a structured PRD with pinned technology choices (§18) —
the PRD is the authoritative reference. Greenfield-structured-PRD rule from
the fp workflow.

## D-11 — Promotion rule is score-vector aware, single-score deferred
Accept = candidate beats baseline on success rate AND regressions ≤ allowed
(default 0 on the validation set). Score vector (tool calls, tokens, latency,
recovery rate) is recorded per candidate from day one, but never collapsed
into one number for decisions before M5 analysis.

## D-12 — experience/runs/ is gitignored at runtime
`experience/runs/*.jsonl` + `experience/index.db` are runtime artifacts —
dev runs with FakeModel must not pollute the research record. Real experiment
runs (M5+) are committed selectively through `results/` manifests; the git
short-hash in a manifest pins the environment, not the raw runs.

## D-13 — M6 matrix CLI: `--models` list instead of SPEC §16 globs
**Chosen:** `exp matrix --models A,B [--iterations N] [--seed INT]`. SPEC §16
sketched `--skill-source '*' --executor-model '*'`. The research question is
square-matrix transfer: every training model also executes. One comma-list
flag keeps the CLI consistent with `compare`/`evolve`; the asymmetric glob
form can be added if a later experiment needs it.

## D-14 — H3 guard scoped to the execution path
The source-grep H3 test originally scanned all of `src/experiments/` for
`knowledge` imports. The learning-time evolution loop (§11) legitimately
drives the miner/proposer — the only components allowed to see the knowledge
base (AGENTS.md §3). Guard now covers `src/agent/` plus the runtime
orchestration modules (runner, proposal_store, promote, report);
`evolution.py` and `matrix.py` are exempt. Execution-agent isolation is
unchanged.

## D-15 — runner extension: `persist=False` + `memory_notes`
`run_tasks` gained two keyword arguments (both default to prior behavior):
`persist` (skip TraceStore append for held-out/matrix runs that must not
pollute the evidence store) and `memory_notes` (inject the M5 memory-config
notes via build_context). Compare and matrix rely on both.